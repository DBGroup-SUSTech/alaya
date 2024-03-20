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
  int bucket_num_;            // bucket_num_ in each subspace;
  DataType* data_ = nullptr;  // data ptr
  int clustering_iter_;       // max kmeans iterations
  int subspace_cnt_;          // subspace count, usually set to 2

  // store data ids in each cell, equals to a two-dimension-version of multiindex
  // since it has two dimensions, there is no need to store offsets in cell_edges
  std::vector<std::vector<IDType>> id_buckets_;
  std::vector<std::vector<DataType>> buckets_;  // store data points in each cell

  // std::vector<int> subspace_dim_; // vector to store each subspace dimension

  // equals to coarse_vocabs, stores centroids in each subspace
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

  void FillIndex() {}

  void FillIndexForSubset() {}
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
    // 每个线程拿到的点的个数
    int thread_data_count = data_num_ / THREADS_COUNT;
    for (int thread_id = 0; thread_id < THREADS_COUNT; ++thread_id) {
      // 开始id
      IDType start_pid = thread_data_count * thread_id;
      // 函数计算某个子集的量化
      // 函数参数分别为：this、文件名、开始节点id、节点个数、传引用的coarse_vocabs和transposed_coarse_quantizations
      index_threads.create_thread(
          boost::bind(&GetNearestCentroidIdsForSubset, this, data_, start_pid, thread_data_count));
    }
    // transposed_coarse_quantizations 中存储的是  两个维度中的  每个节点 对应的最近 centroid 的索引
    index_threads.join_all();
  }

  void GetNearestCentroidIdsForSubset(const DataType* kData, int start_pid, int subset_size) {
    std::vector<int> nearest_ids(subspace_cnt_);  // coarse_quantization
    for (int data_number = 0; data_number < subset_size; ++data_number) {
      if (data_number % 10000 == 0) {
        std::cout << "Getting coarse quantization, point # " << start_pid + data_number
                  << std::endl;
        // std::vector<DataType> current_data(data_dim_);
        DataType* v_begin = data_ + (data_number + start_pid) * data_dim_;
        DataType* v_end = v_begin + data_dim_;
        // std::copy(v_begin, v_end, current_data.data());

        int subspace_dim = data_dim_ / subspace_cnt_;

        for (int subspace_index = 0; subspace_index < subspace_cnt_; ++subspace_index) {
          int begin_dim = subspace_index * subspace_dim;
          int end_dim = begin_dim + subspace_dim;

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
  int GetGlobalCellIndex(std::vector<int> nearest_ids);
};

template <typename IDType, typename DataType>
int InvertedMultiIndex<IDType, DataType>::GetGlobalCellIndex(std::vector<int> nearest_ids) {
  int global_index = 0;
  int subtable_cap = cell_data_cnt_.size();
  for (int i = 0; i < subspace_cnt_; ++i) {
    subtable_cap = subtable_cap / bucket_num_;
    global_index += subtable_cap * nearest_ids[i];
  }
  return global_index;
}

int GetCellCount(int subspace_cnt, int bucket_num) {
  int cell_num = 1;
  for (int i = 0; i < subspace_cnt; ++i) {
    cell_num = cell_num * bucket_num;
  }
  return cell_num;
}

}  // namespace alaya