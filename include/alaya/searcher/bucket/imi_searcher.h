#pragma once

#include <alaya/index/bucket/imi.h>
#include <alaya/searcher/searcher.h>
#include <alaya/utils/pool.h>

#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "ordered_list_merger.h"

namespace alaya {
template <MetricType metric, typename DataType = float, typename IDType = uint64_t>
struct ImiSearcher : Searcher<metric, IMI<DataType, IDType>, DataType> {
  int nprobe_;
  DistFunc<DataType, DataType, DataType> dist_func_;
  const DataType* query_;

  ImiSearcher(const IMI<DataType, IDType>* index)
      : Searcher<metric, IMI<DataType, IDType>, DataType>(index),
        dist_func_(GetDistFunc<DataType, false>(this->index_->metric_type_)) {}

  ~ImiSearcher() { printf("ImiSearcher destructor\n"); };

  void SetNprobe(int n) { nprobe_ = n; }

  // 这里是从merger_ 中获取最小距离的cell，类似于bfs过程。
  bool TraverseNextIMICell(const DataType* kQuery, OrderedListMerger<DataType, IDType>& merger,
                           LinearPool<DataType>& pool) const {
    std::vector<int> cell_inner_indices;
    // 从堆中取出下一个应该取出的cell，坐标放在cell_inner_indices中，并且上下左右探索这个cell的邻居，插入到heap中
    if (!merger.GetNextMergedItemIndices(&cell_inner_indices)) {
      return false;
    }
    // 得到cell在ordered list中的真实坐标
    std::vector<int> cell_coordinates(cell_inner_indices.size());
    for (int list_index = 0; list_index < merger.order_ptr->size(); ++list_index) {
      cell_coordinates[list_index] =
          merger.order_ptr->at(list_index)[cell_inner_indices[list_index]];
    }
    int global_index = this->index_->GetGlobalCellIndex(cell_coordinates);

    IDType* id_pool = this->index_->id_buckets_.at(global_index);
    DataType* data_pool = this->index_->data_buckets_.at(global_index);
    DataType distance = 0;
    DataType* data_point = nullptr;

    for (int cell_data_index = 0; cell_data_index < this->index_->cell_data_cnt_[global_index];
         cell_data_index++) {
      data_point = data_pool + cell_data_index * this->index_->vec_dim_;
      distance = dist_func_(kQuery, data_point, this->index_->vec_dim_);
      pool.Insert(id_pool[cell_data_index], distance);
    }
    return true;
  }

  void GetNearestNeighbours(const DataType* kQuery, int k,
                            OrderedListMerger<DataType, IDType>& merger,
                            LinearPool<DataType>& pool) const {
    assert(this->index_->subspace_cnt_ > 0 || !"Subspace count must be greater than 0.");

    int found_neighbour_cnt = 0;
    bool traverse_next_cell = true;
    int cells_visited = 0;
    // if the number of traversed cells is less than nprobe, then continue to traverse
    while (cells_visited < nprobe_) {
      traverse_next_cell = TraverseNextIMICell(kQuery, merger, pool);
      cells_visited += 1;
    }
  }

  void Search(const DataType* kQuery, int64_t query_dim, int64_t k, DataType* distances,
              int64_t* labels) const override {
    assert(query_dim == this->index_->vec_dim_ &&
           "Query dimension must be equal to data dimension.");
    LinearPool<DataType> pool(k);
    // manage the traverse of search table for each query
    OrderedListMerger<DataType, IDType> merger;
    auto computer = this->index_->template GetComputer<metric>(kQuery);

    merger.SetLists(computer.order_, computer.centroids_dist_, this->index_->bucket_num_);
    // get the nearest neighbours of each query
    GetNearestNeighbours(kQuery, k, merger, pool);
    for (int i = 0; i < k; i++) {
      distances[i] = pool.pool_[i].dis_;
      labels[i] = pool.pool_[i].id_;
    }
  }

  void BatchSearch(int64_t query_num, int64_t query_dim, const DataType* kQueries, int64_t k,
                   DataType* distances, int64_t* result_ids
                   // const SearchParameters* search_params = nullptr
  ) const override {
#pragma omp parallel for schedule(dynamic)
    for (std::size_t q = 0; q < query_num; ++q) {
      Search(kQueries + q * query_dim, query_dim, k, distances + q * k, result_ids + q * k);
    }
  };
};
}  // namespace alaya
