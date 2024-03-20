#pragma once
#include <alaya/index/bucket/ivf.h>
#include <alaya/index/bucket/multitable.h>
#include <alaya/utils/metric_type.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/thread/detail/thread_group.hpp>
#include <boost/thread/thread.hpp>
// #include <boost/thread/thread_group.hpp>
#include <ctime>
#include <map>
#include <vector>

#include "alaya/index/bucket/bucket.h"

using boost::lexical_cast;
using boost::split;

extern int THREADS_COUNT;

namespace alaya {
// /**
//  * \struct MultiIndex incorporates all data structures we need to make search
//  */
// template <typename IDType, typename DataType>
// struct MultiIndex {
//   std::vector<Record> multiindex_;
//   Multitable<int> cell_edges_;  ///< Table with index cell edges in array
// };
int GetCellCount(int subspace_cnt, int bucket_num);

template <typename IDType, typename DataType>
struct InvertedMultiIndex : Bucket<IDType, DataType> {
  InvertedMultiIndex<IDType, DataType>(const int bucket_num, MetricType metric, const int data_dim,
                                       const int subspace_cnt, const int clustering_iter)
      : bucket_num_(bucket_num),
        metric_type_(metric),
        data_dim_(data_dim),
        clustering_iter_(clustering_iter),
        subspace_cnt_(subspace_cnt) {
    int cell_cnt = GetCellCount(subspace_cnt, bucket_num);
    id_buckets_.resize(cell_cnt);
    buckets_.resize(cell_cnt);
    cell_data_cnt_.resize(cell_cnt);
    assert(data_dim_ % subspace_cnt_ == 0 || !"please change the subspace number.");
  }
  typedef std::vector<DataType> Centroids;

  int data_dim_;              // data dimensions
  IDType data_num_;           // data numbers
  MetricType metric_type_;    // metric type
  DataType* data_ = nullptr;  // data ptr
  int clustering_iter_;       // max kmeans iterations
  int subspace_cnt_;          // subspace count, usually set to 2
  int bucket_num_;  // bucket num in each subspace, actually, except for bucket_num_ means cluster
                    // num in each subspace, all the other buckets(id_buckets_, buckets_) means
                    // cells

  // store data ids in each cell, equals to a two-dimension-version of multiindex
  // since it has two dimensions, there is no need to store offsets in cell_edges
  std::vector<std::vector<IDType>> id_buckets_;
  std::vector<std::vector<DataType>> buckets_;  // store data points in each cell

  // std::vector<int> subspace_dim_; // vector to store each subspace dimension

  // equals to coarse_vocabs, stores centroids in each subspace
  /** structure is:
  --------------------------------------
         |- float float float float .....   centroid0
  shard1 |- float float float float .....   centroid1
         |- float float float float .....   centroid2   共有bucket_num_个
        -------------------------------
         |- float float float float .....   centroid0
  shard2 |- float float float float .....   centroid1
         |- float float float float .....   centroid2   共有bucket_num_个
  --------------------------------------
   */
  std::vector<std::vector<Centroids>> subspace_ivf_centroids_;

  // equals to transposed_coarse_quantizations, stores nearest subspace centroid ids for each point
  /* structure is:
  --------------------------------------
  shard1 |- int int int int int .....   共有data_num_个
        -------------------------------
  shard2 |- int int int int int .....   共有data_num_个
  --------------------------------------
  */
  std::vector<std::vector<int>> nearest_subspace_centroid_ids_;
  // equals to point_in_cells_count_, stores data counts in each cell
  // since dimensions of each subspace are predefined by bucket_num_ and subspace_cnt_,
  // no need to use std::vector<int> dimensions in class Multitable
  std::vector<int> cell_data_cnt_;

  boost::mutex cell_counts_mutex_;

  void BuildIndex(IDType data_num, const DataType* kData) override {
    data_num_ = data_num;
    data_ = kData;
    int subspace_dim = data_dim_ / subspace_cnt_;

    GetNearestCentroidIdList(subspace_dim);

    FillIndex();
  }

  void FillIndex() {
    std::cout << "indexing started..." << std::endl;
    std::vector<int> cell_data_written_cnt(cell_data_cnt_.size());
    int thread_data_count = data_num_ / THREADS_COUNT;
    boost::thread_group threads;
    for (int thread_id = 0; thread_id < THREADS_COUNT; ++thread_id) {
      IDType start_pid = thread_data_count * thread_id;
      threads.create_thread(boost::bind(&FillIndexForSubset, this, data_, start_pid,
                                        thread_data_count, cell_data_written_cnt));
    }
  }

  void FillIndexForSubset(const DataType* kData, int start_pid, int subset_size,
                          std::vector<int>& cell_data_written_cnt) {
    std::vector<int> nearest_ids(subspace_cnt_);
    IDType pid = 0;
    for (int data_number = 0; data_number < subset_size; ++data_number) {
      if (data_number % 10000 == 0) {
        std::cout << "Filling multiindex, point # " << start_pid + data_number << std::endl;
        pid = start_pid + data_number;
        for (int i = 0; i < subspace_cnt_; ++i) {
          nearest_ids[i] = nearest_subspace_centroid_ids_[i][pid];
        }
        int global_index = GetGlobalCellIndex(nearest_ids);
        cell_counts_mutex_.lock();
        ++cell_data_written_cnt[global_index];
        id_buckets_[global_index].emplace_back(pid);
        // 加数据
        cell_counts_mutex_.unlock();
      }
    }
// copy data
#pragma omp parallel for
    for (int i = 0; i < buckets_.size(); ++i) {
      buckets_[i].resize(cell_data_written_cnt[i] * data_dim_);
      DataType* single_data_ptr = buckets_[i].data();
      for (int j = 0; j < id_buckets_[i].size(); ++j) {
        DataType* v_begin = data_ + id_buckets_[i][j] * data_dim_;
        DataType* v_end = v_begin + data_dim_;
        std::copy(v_begin, v_end, single_data_ptr);
        single_data_ptr += data_dim_;
      }
    }
  }
  /**
   * @brief Get the nearest subspace centroid ids for each point, structured as
   * nearest_subspace_centroid_ids_. Meanwhile get the data count for each cell in cell_data_cnt_,
   * get the centroids in subspaces in subspace_ivf_centroids_.
   *
   * @param subspace_dim
   */
  void GetNearestCentroidIdList(int subspace_dim) {
    // 得到每个子空间的中心点，结构和 coarse_vocabs 相同
    utils::kmeans(data_, data_num_, data_dim_, subspace_dim, subspace_ivf_centroids_, bucket_num_);
    nearest_subspace_centroid_ids_.resize(subspace_cnt_);
    for (int i = 0; i < subspace_cnt_; ++i) {
      nearest_subspace_centroid_ids_[i].resize(data_num_);
    }

    std::cout << "Memory for coarse quantizations allocated" << std::endl;
    boost::thread_group index_threads;
    int thread_data_count = data_num_ / THREADS_COUNT;
    for (int thread_id = 0; thread_id < THREADS_COUNT; ++thread_id) {
      IDType start_pid = thread_data_count * thread_id;

      index_threads.create_thread(boost::bind(&GetNearestCentroidIdListForSubset, this, data_,
                                              start_pid, thread_data_count));
    }
    // transposed_coarse_quantizations 中存储的是  两个维度中的  每个节点 对应的最近 centroid 的索引
    index_threads.join_all();
  }

  void GetNearestCentroidIdListForSubset(const DataType* kData, int start_pid, int subset_size) {
    // nearest subspace centroid ids for each point
    std::vector<int> nearest_ids(subspace_cnt_);  // coarse_quantization
    // for each points in the subset
    for (int data_number = 0; data_number < subset_size; ++data_number) {
      if (data_number % 10000 == 0) {
        std::cout << "Getting coarse quantization, point # " << start_pid + data_number
                  << std::endl;
        // std::vector<DataType> current_data(data_dim_);

        // get the beginning ptr of each point
        DataType* v_begin = data_ + (data_number + start_pid) * data_dim_;
        DataType* v_end = v_begin + data_dim_;
        // std::copy(v_begin, v_end, current_data.data());

        // get dimensions of each subspace
        int subspace_dim = data_dim_ / subspace_cnt_;
        // for each subspace
        for (int subspace_index = 0; subspace_index < subspace_cnt_; ++subspace_index) {
          // get the beginning dimension of this point in this subspace
          int begin_dim = subspace_index * subspace_dim;
          int end_dim = begin_dim + subspace_dim;

          // get the subvector beginning addr of this point in this subspace
          DataType* subv_begin = v_begin + begin_dim;
          DataType* subv_end = subv_begin + subspace_dim;

          std::vector<DataType> current_subdata(subspace_dim);
          std::copy(subv_begin, subv_end, current_subdata.data());

          int nearest = GetNearestCentroidId(current_subdata,
                                             subspace_ivf_centroids_[subspace_index], subspace_dim);

          nearest_subspace_centroid_ids_[subspace_index][start_pid + data_number] = nearest;
          nearest_ids[subspace_index] = nearest;
        }
        int global_index = GetGlobalCellIndex(nearest_ids);

        cell_counts_mutex_.lock();
        ++(cell_data_cnt_[global_index]);
        cell_counts_mutex_.unlock();
      }
    }
  }

  int GetNearestCentroidId(const std::vector<DataType>& kCurrentData,
                           const std::vector<Centroids>& kSubspaceIvfCentroid, int subspace_dim) {
    int nearest = 0;
    auto distFunc = GetDistFunc<DataType, false>(metric_type_);
    assert(distFunc != nullptr || !"metric_type invalid!");
    DataType min_dist = distFunc(kCurrentData.data(), kSubspaceIvfCentroid[0].data(), subspace_dim);
    for (IDType pid = 1; pid < kSubspaceIvfCentroid.size(); ++pid) {
      DataType current_dist = 0;
      current_dist = distFunc(kCurrentData.data(), kSubspaceIvfCentroid[pid].data(), subspace_dim);
      if (current_dist < min_dist) {
        min_dist = current_dist;
        nearest = pid;
      }
    }
    return nearest;
  }
  int GetGlobalCellIndex(std::vector<int> nearest_ids) {
    int global_index = 0;
    int subtable_cap = cell_data_cnt_.size();
    for (int i = 0; i < subspace_cnt_; ++i) {
      subtable_cap = subtable_cap / bucket_num_;
      global_index += subtable_cap * nearest_ids[i];
    }
    return global_index;
  }

  /**
   * @brief Index file structure
   * metadata:                 | data_num_ | data_dim_ | subspace_cnt_ | bucket_num | cell_num |
   * centroids in subspace0:   | centroid0 in subspace0 | centroid1 in subspace0 | ... |
   * centroids in subspace1:   | centroid0 in subspace1 | centroid1 in subspace1 | ... |
   * buckets:
   * | bucket0 size_ |  point ids in 0 | data in 0
   * | bucket1 size_ |  point ids in 1 | data in 1
   *
   * @param kPath
   */

  void Save(const char* kPath) override {
    std::ofstream output(kPath, std::ios::binary);
    if (!output.is_open()) {
      throw std::runtime_error("Cannot open file");
    }
    int cell_cnt = GetCellCount(subspace_cnt_, bucket_num_);
    printf(
        "[Report] - cluster number: %zu in each subspace, total cell num: %zu, starting to save "
        "index...\n",
        static_cast<size_t>(bucket_num_), static_cast<size_t>(cell_cnt));

    output.write((char*)&data_num_, sizeof(IDType));
    output.write((char*)&data_dim_, sizeof(int));
    output.write((char*)&subspace_cnt_, sizeof(int));
    // write bucket_num_
    output.write((char*)&bucket_num_, sizeof(int));

    output.write((char*)&cell_cnt, sizeof(int));
    int subspace_dim = data_dim_ / subspace_cnt_;
    // write all centroids
    for (int i = 0; i < subspace_cnt_; ++i) {
      for (int j = 0; j < bucket_num_; ++j) {
        output.write((char*)subspace_ivf_centroids_[i][j].data(), subspace_dim * sizeof(DataType));
      }
    }
    // write each cell size and id_buckets and data
    for (int i = 0; i < cell_cnt; ++i) {
      uint32_t each_bucket_size = static_cast<uint32_t>(id_buckets_[i].size());
      output.write((char*)&each_bucket_size, sizeof(uint32_t));
      output.write((char*)id_buckets_[i].data(), sizeof(IDType) * each_bucket_size);
      output.write((char*)buckets_[i].data(), sizeof(DataType) * each_bucket_size * data_dim_);
    }
    printf("[Report] - saving index complete!\n");

    output.close();
  }

  void Load(const char* kPath) override {
    std::ifstream input(kPath, std::ios::binary);
    if (!input.is_open()) {
      throw std::runtime_error("Cannot open file");
    }
    int cell_cnt = 0;
    input.read((char*)&data_num_, sizeof(IDType));
    input.read((char*)&data_dim_, sizeof(int));
    input.read((char*)&subspace_cnt_, sizeof(int));
    input.read((char*)&bucket_num_, sizeof(int));
    input.read((char*)&cell_cnt, sizeof(int));
    assert(cell_cnt == GetCellCount(subspace_cnt_, bucket_num_) || !"cell count mismatch");
    int subspace_dim = data_dim_ / subspace_cnt_;
    printf(
        "[Report] - data_num = %zu, data_dim = %zu, subspace_cnt = %zu, bucket_num = %zu, cell_cnt "
        "= %zu, starting to read index...\n",
        static_cast<size_t>(data_num_), static_cast<size_t>(data_dim_),
        static_cast<size_t>(subspace_cnt_), static_cast<size_t>(cell_cnt),
        static_cast<size_t>(bucket_num_));

    // read all centroids
    subspace_ivf_centroids_.resize(subspace_cnt_);
    for (int i = 0; i < subspace_cnt_; ++i) {
      subspace_ivf_centroids_[i].resize(bucket_num_);
      for (int j = 0; j < bucket_num_; ++j) {
        subspace_ivf_centroids_[i][j].resize(subspace_dim);
        input.read((char*)subspace_ivf_centroids_[i][j].data(), subspace_dim * sizeof(DataType));
      }
    }

    // read each cell size and id_buckets and data
    id_buckets_.resize(cell_cnt);
    buckets_.resize(cell_cnt);
    for (int i = 0; i < cell_cnt; ++i) {
      uint32_t each_bucket_size = 0;
      input.read((char*)&each_bucket_size, sizeof(uint32_t));
      id_buckets_[i].resize(each_bucket_size);
      input.read((char*)id_buckets_[i].data(), sizeof(IDType) * each_bucket_size);
      buckets_[i].resize(each_bucket_size * data_dim_);
      input.read((char*)buckets_[i].data(), sizeof(DataType) * each_bucket_size * data_dim_);
    }

    printf("[Report] - reading index complete!\n");

    input.close();
  }
};

int GetCellCount(int subspace_cnt, int bucket_num) {
  int cell_num = 1;
  for (int i = 0; i < subspace_cnt; ++i) {
    cell_num = cell_num * bucket_num;
  }
  return cell_num;
}

}  // namespace alaya