#pragma once

#include <alaya/index/bucket/imi.h>
#include <alaya/searcher/searcher.h>
#include <alaya/utils/pool.h>

#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "ordered_list_merger.h"

namespace alaya {
template <typename DataType, typename IDType,
          typename IndexType = InvertedMultiIndex<DataType, IDType>>
struct IMISearcher : Searcher<IndexType, DataType> {
  using ClusterId = int;
  using Distance = DataType;
  using MergedItemIndices = std::vector<int>;
  using NearestSubspaceCentroids = std::vector<std::pair<Distance, ClusterId>>;
  using Coord = DataType;
  using Centroids = std::vector<Coord>;

  DistFunc<DataType, DataType, DataType> dist_func_;
  std::vector<IDType*> order_;
  std::vector<DataType*> centroids_dist_;

  const DataType* query_;
  ResultPool<IDType, DataType>* res_;
  // int64_t query_num_;
  mutable OrderedListMerger merger_;
  IMISearcher(const IndexType* index)
      : Searcher<IndexType, DataType>(index, nullptr),
        dist_func_(GetDistFunc<DataType, false>(this->index_->metric_type_)) {}
  ~IMISearcher() {
    delete res_;
    printf("IMISearcher destructor\n");
  };
  // setup
  void SetIndex(const IndexType& index) { this->index_ = std::make_unique<IndexType>(index); }

  void print_subspaces_short_lists(std::vector<NearestSubspaceCentroids>* subspaces_short_lists) {
    for (int index_subspaces_short_lists = 0;
         index_subspaces_short_lists < subspaces_short_lists->size();
         index_subspaces_short_lists++) {
      printf("subspace : %d \n", index_subspaces_short_lists);
      for (int index_cluster = 0;
           index_cluster < subspaces_short_lists->at(index_subspaces_short_lists).size();
           index_cluster++) {
        printf("distance: %f ; id %d \n",
               subspaces_short_lists->at(index_subspaces_short_lists).at(index_cluster).first,
               subspaces_short_lists->at(index_subspaces_short_lists).at(index_cluster).second);
      }
    }
  }

  inline float cal_distance(const DataType* query, Centroids centroids, int dimension) {
    Distance dis = 0;
    dis = L2Sqr<DataType>(query, centroids.data(), dimension);
    return dis;
  }
  // 这个方法是计算出每一个子空间中聚类中心到query点的距离，在这个方法里面已经做好排序了；返回的是一个vector，vector.size()对应的是子空间的个数：2（通常）。
  // 使用nth_element优化排序的过程
  void GetNearestSubspacesCentroids(std::vector<NearestSubspaceCentroids>* subspaces_short_lists,
                                    int subspace_centroids_cnt) {
    printf("subspace_cnt: %lu \n", index_->subspace_ivf_centroids_.size());
    printf("subspace_cluster_cnt: %lu \n", index_->subspace_ivf_centroids_[0].size());
    int cnt = 0;
    printf("subspace_centroids_cnt: %d\n", subspace_centroids_cnt);
    // printf("find centroids entering");
    subspaces_short_lists->resize(index_->subspace_cnt_);  // init()
    int subspace_dimensions = query_dim_ / index_->subspace_cnt_;
    Distance distance = 0;
    // 对于每个子空间
    for (int subspace_indexer = 0; subspace_indexer < index_->subspace_cnt_; subspace_indexer++) {
      int start_dim = subspace_indexer * subspace_dimensions;
      int final_dim = query_dim_;
      if (final_dim > start_dim + subspace_dimensions) {
        final_dim = start_dim + subspace_dimensions;
      }
      subspaces_short_lists->at(subspace_indexer)
          .resize(index_->subspace_ivf_centroids_[subspace_indexer].size());
      // 计算距离，为了排序做准备
      for (ClusterId cluster_index = 0;
           cluster_index < index_->subspace_ivf_centroids_[subspace_indexer].size();
           cluster_index++) {
        const DataType* sub_query = query_ + start_dim;

        // const DataType* sub_query = query_ + start_dim;
        distance = cal_distance(sub_query,
                                index_->subspace_ivf_centroids_[subspace_indexer][cluster_index],
                                final_dim - start_dim);
        subspaces_short_lists->at(subspace_indexer)[cluster_index] =
            std::make_pair(distance, cluster_index);
      }
      // sort()

      // 只排序 subspaces_short_lists->at(subspace_indexer) 的前 subspace_centroids_cnt 个元素
      // 也就是搜索的数量
      std::nth_element(subspaces_short_lists->at(subspace_indexer).begin(),
                       subspaces_short_lists->at(subspace_indexer).begin() + subspace_centroids_cnt,
                       subspaces_short_lists->at(subspace_indexer).end());
      subspaces_short_lists->at(subspace_indexer).resize(subspace_centroids_cnt);
      std::sort(subspaces_short_lists->at(subspace_indexer).begin(),
                subspaces_short_lists->at(subspace_indexer).end());
    }  // 每一列都是按照升序排序的
  }

  // 这里是从merger_ 中获取最小距离的cell，类似于bfs过程。
  bool TraverseNextMultiIndexCell() {
    // std::vector<int> cell_coordinates;  是这个cell的坐标，通常大小为2
    MergedItemIndices cell_inner_indices;
    printf("print heap_: \n");
    merger_.print_heap();
    // merger.lists_ptr 就是查询向量在每个维度下按照距离排序过后的中心点id 和 距离  first 是距离
    // second 是id
    if (!merger_.GetNextMergedItemIndices(&cell_inner_indices)) {
      printf("kong\n");
      return false;
    }
    // 应该也是一个二维的东西
    std::vector<int> cell_coordinates(cell_inner_indices.size());
    // 每个子空间 去 get cell id
    for (int list_index = 0; list_index < merger_.lists_ptr->size(); ++list_index) {
      cell_coordinates[list_index] =
          merger_.lists_ptr->at(list_index)[cell_inner_indices[list_index]].second;
    }  // 定位坐标的，cluster的坐标，
    printf("get global index\n");
    // 拿到global index  计算得到结果
    int global_index = index_->GetGlobalCellIndex(cell_coordinates);

    printf("global_index: %d\n", global_index);

    std::vector<IDType> id_pool = index_->id_buckets_.at(global_index);
    std::vector<DataType> data_pool = index_->buckets_.at(global_index);
    DataType distance = 0;
    const float* ptr = nullptr;
    // printf("begin add result\n ");

    printf("id_pool.size()  :%lu \n", id_pool.size());
    printf("data_pool.size()  :%lu \n", data_pool.size());
    for (int index_id_pool = 0; index_id_pool < id_pool.size(); index_id_pool++) {
      ptr = &data_pool.at(index_id_pool * index_->data_dim_);
      distance = L2Sqr<DataType>(query_, ptr, query_dim_);
      res_->Insert(id_pool.at(index_id_pool), distance);
    }

    // printf("finish add result\n ");
    return true;
    // return false;
  }
  void GetNearestNeighbours(const DataType* query, int k) {
    // printf("begin GetNearestNeighbours\n");
    assert(this->index_->subspace_cnt_ > 0);
    std::vector<IDType*> order(this->index_->subspace_cnt_);
    std::vector<DataType*> centroids_dist(this->index_->subspace_cnt_);

    // printf("begin GetNearestSubspacesCentroids\n");
    GetNearestSubspacesCentroids(
        &subspaces_short_lists,
        6);  // 在这里调用方法计算
             // query到每一个子空间的距离，subspaces_short_lists是一个vector，size对应着子空间的个数
             // 通常是2.
    // printf("subspaces_short_lists.size():  %lu\n ",subspaces_short_lists.size());
    printf("print short_list\n");
    print_subspaces_short_lists(&subspaces_short_lists);
    merger_.setLists(subspaces_short_lists);
    int found_neighbour_cnt = 0;
    bool traverse_next_cell = true;
    int cells_visited = 0;
    // while(index_->found_neghbours_count_ < k && traverse_next_cell) {
    //     traverse_next_cell = TraverseNextMultiIndexCell();
    //     cells_visited += 1;
    // }
    // printf("enter begin while loop\n");
    while (traverse_next_cell) {
      traverse_next_cell = TraverseNextMultiIndexCell();
      cells_visited += 1;
    }
  }

  void Search(int64_t query_num, int64_t query_dim, const DataType* query, int64_t k,
              DataType* distances, int64_t* labels) {
    // printf("begin search imi\n");
    res_ = new ResultPool<IDType, DataType>(index_->data_num_, 2 * k, k);
    assert(query_dim == index_->data_dim_ && "Query dimension must be equal to data dimension.");
    distances_ = distances;
    GetNearestNeighbours(query, k);
  }
};
}  // namespace alaya
