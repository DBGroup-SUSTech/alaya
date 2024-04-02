#pragma once
#include <alaya/index/bucket/ivf.h>
#include <alaya/utils/metric_type.h>
#include <faiss/Clustering.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/thread.hpp>
#include <boost/thread/detail/thread_group.hpp>
#include <boost/thread/thread.hpp>
// #include <boost/thread/thread_group.hpp>
#include <omp.h>

#include <ctime>
#include <fstream>
#include <map>
#include <vector>

#include "../../utils/memory.h"
#include "alaya/index/bucket/bucket.h"

using boost::lexical_cast;
using boost::split;

extern int THREADS_COUNT = 40;
static omp_lock_t lock;

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

template <typename DataType, typename IDType>
struct InvertedMultiIndex : Index<DataType, IDType> {
  InvertedMultiIndex(const int bucket_num, MetricType metric, const int vec_dim,
                     const int subspace_cnt)
      : Index<DataType, IDType>(vec_dim, metric),
        bucket_num_(bucket_num),
        subspace_cnt_(subspace_cnt) {
    int cell_cnt = GetCellCount(subspace_cnt, bucket_num);
    id_buckets_.resize(cell_cnt);
    data_buckets_.resize(cell_cnt);
    cell_data_cnt_ = (int*)Alloc64B(sizeof(int) * cell_cnt);
    assert(this->vec_dim_ % subspace_cnt_ == 0 || !"please change the subspace number.");
  }

  ~InvertedMultiIndex() {
    // for (int i = 0; i < id_buckets_.size(); ++i) {
    //   id_buckets_[i].clear();
    // }
    // id_buckets_.clear();
    // for (int i = 0; i < data_buckets_.size(); ++i) {
    //   data_buckets_[i].clear();
    // }
    // data_buckets_.clear();
    delete[] cell_data_cnt_;
  }

  const DataType* data_ = nullptr;  // data ptr
  int subspace_cnt_;                // subspace count, usually set to 2
  unsigned int clustering_iter_;    // max kmeans iterations
  int bucket_num_;  // bucket num in each subspace, actually, except for bucket_num_ means cluster
                    // num in each subspace, all the other buckets(id_buckets_, buckets_) means
                    // cells

  // store data ids in each cell, equals to a two-dimension-version of multiindex
  // since it has two dimensions, there is no need to store offsets in cell_edges
  std::vector<IDType*> id_buckets_;
  std::vector<DataType*> data_buckets_;  // store data points in each cell

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
  std::vector<DataType*> subspace_ivf_centroids_;

  // equals to transposed_coarse_quantizations, stores nearest subspace centroid ids for each point
  /* structure is:
  --------------------------------------
  shard1 |- int int int int int .....   共有vec_num_个
        -------------------------------
  shard2 |- int int int int int .....   共有vec_num_个
  --------------------------------------
  */
  std::vector<int*> nearest_subspace_centroid_ids_;
  // equals to point_in_cells_count_, stores data counts in each cell
  // since dimensions of each subspace are predefined by bucket_num_ and subspace_cnt_,
  // no need to use std::vector<int> dimensions in class Multitable
  int* cell_data_cnt_;

  void BuildIndex(IDType vec_num, const DataType* kData) override {
    this->vec_num_ = vec_num;
    data_ = kData;
    int subspace_dim = this->vec_dim_ / subspace_cnt_;

    GetNearestCentroidIdList(subspace_dim);

    FillIndex();
    int count = 0;
    for (int i = 0; i < 9; i++) {
      printf("cell_data_cnt = %d\n", cell_data_cnt_[i]);
      count += cell_data_cnt_[i];
    }
    std::cout << count << std::endl;
  }

  void FillIndex() {
    std::cout << "indexing started..." << std::endl;
    std::vector<int> cell_data_written_cnt(cell_data_cnt_.size());
    int thread_data_count = this->vec_num_ / THREADS_COUNT;

    //     omp_init_lock(&lock);
    // #pragma omp parallel for
    for (int thread_id = 0; thread_id < THREADS_COUNT; ++thread_id) {
      IDType start_pid = thread_data_count * thread_id;
      FillIndexForSubset(data_, start_pid, thread_data_count, cell_data_written_cnt);
      /*
      boost::thread_group threads;
      for (int thread_id = 0; thread_id < THREADS_COUNT; ++thread_id) {
        IDType start_pid = thread_data_count * thread_id;
        threads.create_thread(boost::bind(&InvertedMultiIndex<IDType, DataType>::FillIndexForSubset,
                                          this, data_, start_pid, thread_data_count,
                                          cell_data_written_cnt));
                                          */
    }
  }

  void FillIndexForSubset(const DataType* kData, int start_pid, int subset_size,
                          std::vector<int>& cell_data_written_cnt) {
    std::vector<int> nearest_ids(subspace_cnt_);
    IDType pid = 0;
    for (int vec_number = 0; vec_number < subset_size; ++vec_number) {
      pid = start_pid + vec_number;
      for (int i = 0; i < subspace_cnt_; ++i) {
        nearest_ids[i] = nearest_subspace_centroid_ids_[i][pid];
      }
      int global_index = GetGlobalCellIndex(nearest_ids);
      // cell_counts_mutex_.lock();
      // omp_set_lock(&lock);
      ++cell_data_written_cnt[global_index];
      id_buckets_[global_index].emplace_back(pid);
      // omp_unset_lock(&lock);
      // 加数据
      // cell_counts_mutex_.unlock();
    }
    // copy data
    // #pragma omp parallel for
    for (int i = 0; i < data_buckets_.size(); ++i) {
      buckets_[i].resize(cell_data_written_cnt[i] * vec_dim_);
      DataType* single_data_ptr = buckets_[i].data();
      for (int j = 0; j < id_buckets_[i].size(); ++j) {
        const DataType* v_begin = data_ + id_buckets_[i][j] * vec_dim_;
        const DataType* v_end = v_begin + vec_dim_;
        std::copy(v_begin, v_end, single_data_ptr);
        single_data_ptr += vec_dim_;
      }
    }
    // omp_destroy_lock(&lock);
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
    kmeans<DataType, IDType>(data_, vec_num_, vec_dim_, subspace_dim, subspace_ivf_centroids_,
                             bucket_num_, false, clustering_iter_);

    nearest_subspace_centroid_ids_.resize(subspace_cnt_);
    for (int i = 0; i < subspace_cnt_; ++i) {
      nearest_subspace_centroid_ids_[i].resize(vec_num_);
    }

    std::cout << "Memory for coarse quantizations allocated" << std::endl;

    //     omp_init_lock(&lock);

    // #pragma omp parallel for
    for (int thread_id = 0; thread_id < THREADS_COUNT; ++thread_id) {
      int thread_data_count = vec_num_ / THREADS_COUNT;
      IDType start_pid = thread_data_count * thread_id;
      GetNearestCentroidIdListForSubset(data_, start_pid, thread_data_count);
    }
    //~ DEBUG
    // for (int i = 0; i < subspace_cnt_; ++i) {
    //   for (int j = 0; j < nearest_subspace_centroid_ids_[i].size(); ++j) {
    //     printf("nearest_subspace_centroid_ids_[%d][%d] = %d\n", i, j,
    //            nearest_subspace_centroid_ids_[i][j]);
    //   }
    // }

    /*
    boost::thread_group index_threads;
    int thread_data_count = vec_num_ / THREADS_COUNT;
    for (int thread_id = 0; thread_id < THREADS_COUNT; ++thread_id) {
      IDType start_pid = thread_data_count * thread_id;
      index_threads.create_thread(
          boost::bind(&InvertedMultiIndex<IDType, DataType>::GetNearestCentroidIdListForSubset,
                      this, data_, start_pid, thread_data_count));
    }
    printf("after thread split\n");
    // transposed_coarse_quantizations 中存储的是  两个维度中的  每个节点 对应的最近 centroid 的索引
    index_threads.join_all();
    printf("after join\n");
    */
  }

  void GetNearestCentroidIdListForSubset(const DataType* kData, int start_pid, int subset_size) {
    // nearest subspace centroid ids for each point
    std::vector<int> nearest_ids(subspace_cnt_);  // coarse_quantization
    // for each points in the subset
    for (int vec_number = 0; vec_number < subset_size; ++vec_number) {
      if (vec_number % 10000 == 0)
        std::cout << "Getting coarse quantization, point # " << start_pid + vec_number << std::endl;
      // std::vector<DataType> current_data(vec_dim_);

      // get the beginning ptr of each point
      const DataType* v_begin = data_ + (vec_number + start_pid) * vec_dim_;
      const DataType* v_end = v_begin + vec_dim_;
      // std::copy(v_begin, v_end, current_data.data());

      // get dimensions of each subspace
      int subspace_dim = vec_dim_ / subspace_cnt_;

      // for each subspace
      for (int subspace_index = 0; subspace_index < subspace_cnt_; ++subspace_index) {
        // get the beginning dimension of this point in this subspace
        int begin_dim = subspace_index * subspace_dim;
        int end_dim = begin_dim + subspace_dim;

        // get the subvector beginning addr of this point in this subspace
        const DataType* subv_begin = v_begin + begin_dim;
        const DataType* subv_end = subv_begin + subspace_dim;
        // actually begin addr = data_ + vec_number * vec_dim_ + subspace_index* subspace_dim
        std::vector<DataType> current_subdata(subspace_dim);
        std::copy(subv_begin, subv_end, current_subdata.data());

        int nearest = GetNearestCentroidId(current_subdata, subspace_ivf_centroids_[subspace_index],
                                           subspace_dim);
        nearest_subspace_centroid_ids_[subspace_index][start_pid + vec_number] = nearest;
        nearest_ids[subspace_index] = nearest;
      }
      int global_index = GetGlobalCellIndex(nearest_ids);

      // cell_counts_mutex_.lock();
      // omp_set_lock(&lock);
      ++(cell_data_cnt_[global_index]);
      // omp_unset_lock(&lock);

      // cell_counts_mutex_.unlock();
    }
  }

  int GetNearestCentroidId(const std::vector<DataType>& kCurrentData,
                           const std::vector<Centroids>& kSubspaceIvfCentroid, int subspace_dim) {
    int nearest = 0;
    // TODO FIX THIS
    // auto distFunc = GetDistFunc<DataType, false>(metric_type_);
    // assert(distFunc != nullptr || !"metric_type invalid!");
    // DataType min_dist = distFunc(kCurrentData.data(), kSubspaceIvfCentroid[0].data(),
    // subspace_dim);
    DataType min_dist =
        L2Sqr<DataType>(kCurrentData.data(), kSubspaceIvfCentroid[0].data(), subspace_dim);
    // printf("min_dist = %f\n", min_dist);
    // DataType dist = L2Sqr<DataType>(data_ + i * vec_dim_, centroids_[j].data(), vec_dim_);
    for (IDType pid = 1; pid < kSubspaceIvfCentroid.size(); ++pid) {
      DataType current_dist = 0;
      // current_dist = distFunc(kCurrentData.data(), kSubspaceIvfCentroid[pid].data(),
      // subspace_dim);
      current_dist =
          L2Sqr<DataType>(kCurrentData.data(), kSubspaceIvfCentroid[pid].data(), subspace_dim);
      // printf("current_dist = %f\n", current_dist);
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
   * metadata:                 | vec_num_ | vec_dim_ | subspace_cnt_ | bucket_num | cell_num |
   * centroids in subspace0:   | centroid0 in subspace0 | centroid1 in subspace0 | ... |
   * centroids in subspace1:   | centroid0 in subspace1 | centroid1 in subspace1 | ... |
   * buckets:
   * | bucket0 size_ |  point ids in 0 | data in 0
   * | bucket1 size_ |  point ids in 1 | data in 1
   *
   * @param kPath
   */

  void Save(const char* kPath) const override {
    std::ofstream output(kPath, std::ios::binary);
    if (!output.is_open()) {
      throw std::runtime_error("Cannot open file");
    }
    int cell_cnt = GetCellCount(subspace_cnt_, bucket_num_);
    printf(
        "[Report] - cluster number: %zu in each subspace, total cell num: %zu, starting to save "
        "index...\n",
        static_cast<size_t>(bucket_num_), static_cast<size_t>(cell_cnt));

    output.write((char*)&vec_num_, sizeof(IDType));
    output.write((char*)&vec_dim_, sizeof(int));
    output.write((char*)&subspace_cnt_, sizeof(int));
    // write bucket_num_
    output.write((char*)&bucket_num_, sizeof(int));

    output.write((char*)&cell_cnt, sizeof(int));
    int subspace_dim = vec_dim_ / subspace_cnt_;
    // write all centroids
    for (int i = 0; i < subspace_cnt_; ++i) {
      for (int j = 0; j < bucket_num_; ++j) {
        output.write((char*)subspace_ivf_centroids_[i][j].data(), subspace_dim * sizeof(DataType));
      }
    }
    printf("[Report] - saving centroids complete!\n");
    // write each cell size and id_buckets and data
    for (int i = 0; i < cell_cnt; ++i) {
      int each_bucket_size = 0;
      output.write((char*)&each_bucket_size, sizeof(uint32_t));
      output.write((char*)id_buckets_[i].data(), sizeof(IDType) * each_bucket_size);
      output.write((char*)buckets_[i].data(), sizeof(DataType) * each_bucket_size * vec_dim_);
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
    input.read((char*)&vec_num_, sizeof(IDType));
    input.read((char*)&vec_dim_, sizeof(int));
    input.read((char*)&subspace_cnt_, sizeof(int));
    input.read((char*)&bucket_num_, sizeof(int));
    input.read((char*)&cell_cnt, sizeof(int));
    assert(cell_cnt == GetCellCount(subspace_cnt_, bucket_num_) || !"cell count mismatch");
    int subspace_dim = vec_dim_ / subspace_cnt_;
    printf(
        "[Report] - vec_num = %zu, vec_dim = %zu, subspace_cnt = %zu, bucket_num = %zu, cell_cnt "
        "= %zu, starting to read index...\n",
        static_cast<size_t>(vec_num_), static_cast<size_t>(vec_dim_),
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
      buckets_[i].resize(each_bucket_size * vec_dim_);
      input.read((char*)buckets_[i].data(), sizeof(DataType) * each_bucket_size * vec_dim_);
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