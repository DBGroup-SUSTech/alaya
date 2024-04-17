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
#include <fmt/core.h>
#include <omp.h>

#include <ctime>
#include <fstream>
#include <map>
#include <vector>

#include "../../utils/kmeans.h"
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
//   Multitable<int> cell_edges_;  ///< Table with index cell edges in array
//   std::vector<Record> multiindex_;
// };

template <typename DataType, typename IDType = uint64_t>
struct IMI : Index<DataType, IDType> {
  int cell_cnt_;
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
  DistFunc<DataType, DataType, DataType> dist_func_;
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

  IMI(const int vec_dim, MetricType metric, const int bucket_num, const int subspace_cnt)
      : Index<DataType, IDType>(vec_dim, metric),
        bucket_num_(bucket_num),
        subspace_cnt_(subspace_cnt),
        dist_func_(GetDistFunc<DataType, false>(metric)) {
    cell_cnt_ = GetCellCount(subspace_cnt, bucket_num);
    subspace_ivf_centroids_.resize(subspace_cnt);
    nearest_subspace_centroid_ids_.resize(subspace_cnt);
    // for(int i = 0; i < subspace_cnt_; i++) {
    //   nearest_subspace_centroid_ids_[i] = (int*)Alloc64B(sizeof(int) * cell_cnt);
    // }
    id_buckets_.resize(cell_cnt_);
    data_buckets_.resize(cell_cnt_);
    cell_data_cnt_ = (int*)Alloc64B(sizeof(int) * cell_cnt_);
    assert(this->vec_dim_ % subspace_cnt_ == 0 || !"please change the subspace number.");
  }

  ~IMI() {
    fmt::println("destroy imi");
    for (int i = 0; i < id_buckets_.size(); ++i) {
      delete[] id_buckets_[i];
      delete[] data_buckets_[i];
    }
    for (int i = 0; i < subspace_cnt_; ++i) {
      delete[] subspace_ivf_centroids_[i];
      delete[] nearest_subspace_centroid_ids_[i];
    }
    delete[] cell_data_cnt_;
  }

  void BuildIndex(IDType vec_num, const DataType* kData) override {
    this->vec_num_ = vec_num;
    data_ = kData;
    int subspace_dim = this->vec_dim_ / subspace_cnt_;
    // get nearest centroid id list for each point, and get the data count for each cell in
    // cell_data_cnt_, get the centroids in subspaces in subspace_ivf_centroids_
    GetNearestCentroidIdList(subspace_dim);

    std::vector<std::vector<IDType>> id_buckets_vec(id_buckets_.size());
    FillIndex(id_buckets_vec);
  }

  void FillIndex(std::vector<std::vector<IDType>>& id_buckets_vec) {
    std::cout << "indexing started..." << std::endl;
    std::vector<int> cell_data_written_cnt(id_buckets_.size());
    int thread_data_count = this->vec_num_ / THREADS_COUNT;

    //     omp_init_lock(&lock);
    // #pragma omp parallel for
    for (int thread_id = 0; thread_id < THREADS_COUNT; ++thread_id) {
      IDType start_pid = thread_data_count * thread_id;
      FillIndexForSubset(data_, start_pid, thread_data_count, cell_data_written_cnt,
                         id_buckets_vec);
    }

    for (int i = 0; i < id_buckets_.size(); ++i) {
      cell_data_cnt_[i] = id_buckets_vec[i].size();
      id_buckets_[i] = (IDType*)Alloc64B(sizeof(IDType) * cell_data_cnt_[i]);
      std::memcpy(id_buckets_[i], id_buckets_vec[i].data(), sizeof(IDType) * cell_data_cnt_[i]);
    }

    for (int i = 0; i < id_buckets_.size(); ++i) {
      data_buckets_[i] = (DataType*)Alloc64B(sizeof(DataType) * this->vec_dim_ * cell_data_cnt_[i]);
      DataType* single_data_ptr = data_buckets_[i];
      for (int j = 0; j < cell_data_cnt_[i]; ++j) {
        const DataType* v_begin = data_ + id_buckets_[i][j] * this->vec_dim_;
        const DataType* v_end = v_begin + this->vec_dim_;
        std::copy(v_begin, v_end, single_data_ptr);
        single_data_ptr += this->vec_dim_;
      }
    }
  }

  void FillIndexForSubset(const DataType* kData, int start_pid, int subset_size,
                          std::vector<int>& cell_data_written_cnt,
                          std::vector<std::vector<IDType>>& id_buckets) {
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
      id_buckets[global_index].emplace_back(pid);
      // omp_unset_lock(&lock);
      // 加数据
      // cell_counts_mutex_.unlock();
    }
    // copy data
    // #pragma omp parallel for

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
    DataType* sub_data = (DataType*)Alloc64B(sizeof(DataType) * this->vec_num_ * subspace_dim);
    // 得到每个子空间的中心点，结构和 coarse_vocabs 相同
    for (int subspace_id = 0; subspace_id < subspace_cnt_; ++subspace_id) {
      // 为每个子空间的中心点分配内存
      subspace_ivf_centroids_[subspace_id] =
          (DataType*)Alloc64B(sizeof(DataType) * bucket_num_ * subspace_dim);
      // 为每个子空间的最近中心点分配内存
      nearest_subspace_centroid_ids_[subspace_id] = (int*)Alloc64B(sizeof(int) * this->vec_num_);

      for (int i = 0; i < this->vec_num_; ++i) {
        std::memcpy(sub_data + i * subspace_dim,
                    data_ + i * this->vec_dim_ + subspace_id * subspace_dim,
                    subspace_dim * sizeof(DataType));
      }
      std::vector<DataType> centroids;
      centroids =
          faiss_kmeans(sub_data, this->vec_num_, subspace_dim, bucket_num_, this->metric_type_);
      std::memcpy(subspace_ivf_centroids_[subspace_id], centroids.data(),
                  sizeof(DataType) * bucket_num_ * subspace_dim);
    }
    delete[] sub_data;

    std::cout << "Memory for coarse quantizations allocated" << std::endl;

    //     omp_init_lock(&lock);

    // #pragma omp parallel for
    for (int thread_id = 0; thread_id < THREADS_COUNT; ++thread_id) {
      int thread_data_count = this->vec_num_ / THREADS_COUNT;
      IDType start_pid = thread_data_count * thread_id;
      GetNearestCentroidIdListForSubset(data_, start_pid, thread_data_count);
    }
  }

  void GetNearestCentroidIdListForSubset(const DataType* kData, int start_pid, int subset_size) {
    // nearest subspace centroid ids for each point
    std::vector<int> nearest_ids(subspace_cnt_);  // coarse_quantization
    // for each points in the subset
    for (int vec_number = 0; vec_number < subset_size; ++vec_number) {
      // get the beginning ptr of each point
      const DataType* v_begin = data_ + (vec_number + start_pid) * this->vec_dim_;
      const DataType* v_end = v_begin + this->vec_dim_;
      // std::copy(v_begin, v_end, current_data.data());

      // get dimensions of each subspace
      int subspace_dim = this->vec_dim_ / subspace_cnt_;

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
                           const DataType* kSubspaceIvfCentroid, int subspace_dim) {
    int nearest = 0;
    DataType min_dist = dist_func_(kCurrentData.data(), kSubspaceIvfCentroid, subspace_dim);
    for (IDType pid = 1; pid < bucket_num_; ++pid) {
      DataType current_dist = 0;
      current_dist =
          dist_func_(kCurrentData.data(), kSubspaceIvfCentroid + pid * subspace_dim, subspace_dim);
      if (current_dist < min_dist) {
        min_dist = current_dist;
        nearest = pid;
      }
    }
    return nearest;
  }
  int GetGlobalCellIndex(std::vector<int> nearest_ids) const {
    int global_index = 0;
    int subtable_cap = id_buckets_.size();
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
    printf(
        "[Report] - cluster number: %zu in each subspace, total cell num: %zu, starting to save "
        "index...\n",
        static_cast<size_t>(bucket_num_), static_cast<size_t>(cell_cnt_));

    output.write((char*)&this->vec_num_, sizeof(IDType));
    output.write((char*)&this->vec_dim_, sizeof(int));
    output.write((char*)&subspace_cnt_, sizeof(int));
    // write bucket_num_
    output.write((char*)&bucket_num_, sizeof(int));

    output.write((char*)&cell_cnt_, sizeof(int));
    int subspace_dim = this->vec_dim_ / subspace_cnt_;
    // write all centroids
    for (int i = 0; i < subspace_cnt_; ++i) {
      output.write((char*)subspace_ivf_centroids_[i],
                   bucket_num_ * subspace_dim * sizeof(DataType));
    }
    printf("[Report] - saving centroids complete!\n");
    // write each cell size and id_buckets and data
    for (int i = 0; i < cell_cnt_; ++i) {
      int each_bucket_size = cell_data_cnt_[i];
      output.write((char*)&each_bucket_size, sizeof(int));
      output.write((char*)id_buckets_[i], sizeof(IDType) * each_bucket_size);
      output.write((char*)data_buckets_[i], sizeof(DataType) * each_bucket_size * this->vec_dim_);
    }
    printf("[Report] - saving index complete!\n");

    output.close();
  }

  void Load(const char* kPath) override {
    std::ifstream input(kPath, std::ios::binary);
    if (!input.is_open()) {
      throw std::runtime_error("Cannot open file");
    }
    input.read((char*)&this->vec_num_, sizeof(IDType));
    input.read((char*)&this->vec_dim_, sizeof(int));
    input.read((char*)&subspace_cnt_, sizeof(int));
    input.read((char*)&bucket_num_, sizeof(int));
    input.read((char*)&cell_cnt_, sizeof(int));
    assert(cell_cnt_ == GetCellCount(subspace_cnt_, bucket_num_) || !"cell count mismatch");
    int subspace_dim = this->vec_dim_ / subspace_cnt_;
    printf(
        "[Report] - vec_num = %zu, vec_dim = %zu, subspace_cnt = %zu, bucket_num = %zu, cell_cnt = "
        "%zu, starting to read index...\n",
        static_cast<size_t>(this->vec_num_), static_cast<size_t>(this->vec_dim_),
        static_cast<size_t>(subspace_cnt_), static_cast<size_t>(cell_cnt_),
        static_cast<size_t>(bucket_num_));

    // read all centroids
    subspace_ivf_centroids_.resize(subspace_cnt_);
    for (int i = 0; i < subspace_cnt_; ++i) {
      subspace_ivf_centroids_[i] =
          (DataType*)Alloc64B(sizeof(DataType) * bucket_num_ * subspace_dim);
      input.read((char*)subspace_ivf_centroids_[i], bucket_num_ * subspace_dim * sizeof(DataType));
    }

    // read each cell size and id_buckets and data
    id_buckets_.resize(cell_cnt_);
    data_buckets_.resize(cell_cnt_);
    cell_data_cnt_ = (int*)Alloc64B(sizeof(int) * cell_cnt_);

    for (int i = 0; i < cell_cnt_; ++i) {
      input.read((char*)cell_data_cnt_ + i * sizeof(int), sizeof(int));
      id_buckets_[i] = (IDType*)Alloc64B(sizeof(IDType) * cell_data_cnt_[i]);
      input.read((char*)id_buckets_[i], sizeof(IDType) * cell_data_cnt_[i]);
      data_buckets_[i] = (DataType*)Alloc64B(sizeof(DataType) * cell_data_cnt_[i] * this->vec_dim_);
      input.read((char*)data_buckets_[i], sizeof(DataType) * cell_data_cnt_[i] * this->vec_dim_);
    }

    printf("[Report] - reading index complete!\n");

    input.close();
  }

  int GetCellCount(int subspace_cnt, int bucket_num) {
    int cell_num = 1;
    for (int i = 0; i < subspace_cnt; ++i) {
      cell_num = cell_num * bucket_num;
    }
    return cell_num;
  }

  template <MetricType metric>
  struct Computer {
    constexpr static auto dist_func_ = GetDistFunc<DataType, true>(metric);
    std::vector<int*> order_;
    std::vector<DataType*> centroids_dist_;
    const IMI& kImi_;

    Computer(const IMI& imi, const DataType* kQuery) : kImi_(imi) {
      order_.resize(kImi_.subspace_cnt_);
      centroids_dist_.resize(kImi_.subspace_cnt_);
      int subspace_dim = kImi_.vec_dim_ / kImi_.subspace_cnt_;
      for (int i = 0; i < kImi_.subspace_cnt_; ++i) {
        order_[i] = (int*)Alloc64B(sizeof(int) * kImi_.bucket_num_);
        centroids_dist_[i] = (DataType*)Alloc64B(sizeof(DataType) * kImi_.bucket_num_);
        int start_dim = i * subspace_dim;
        int final_dim = start_dim + subspace_dim;
        for (int j = 0; j < kImi_.bucket_num_; ++j) {
          centroids_dist_[i][j] =
              dist_func_(kQuery + start_dim, kImi_.subspace_ivf_centroids_[i] + j * subspace_dim,
                         subspace_dim);
          order_[i][j] = j;
        }
        std::sort(order_[i], order_[i] + kImi_.bucket_num_,
                  [centroids_dist = centroids_dist_[i]](int a, int b) {
                    return centroids_dist[a] < centroids_dist[b];
                  });
        std::sort(centroids_dist_[i], centroids_dist_[i] + kImi_.bucket_num_);
      }
    }

    ~Computer() {
      if (order_.size() > 0) {
        for (int i = 0; i < order_.size(); i++) {
          delete[] order_[i];
          delete[] centroids_dist_[i];
        }
      }
    }
  };
  template <MetricType metric>
  auto GetComputer(const DataType* kQuery) const {
    return Computer<metric>(*this, kQuery);
  }
};

}  // namespace alaya