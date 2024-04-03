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

  std::vector<int*> order_;
  std::vector<DataType*> centroids_dist_;

  const DataType* query_;
  ResultPool<IDType, DataType>* res_;
  // int64_t query_num_;
  mutable OrderedListMerger<DataType, IDType> merger_;
  IMISearcher(const IndexType* index)
      : Searcher<IndexType, DataType>(index, nullptr),
        dist_func_(GetDistFunc<DataType, false>(this->index_->metric_type_)) {}
  ~IMISearcher() {
    delete res_;
    for (int i = 0; i < this->index_->subspace_cnt_; i++) {
      // delete[] order_[i];
      // Free(centroids_dist_[i]);
    }
    printf("IMISearcher destructor\n");
  };

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
  void GetNearestSubspacesCentroids(const DataType* kQuery, std::vector<int*>& order,
                                    std::vector<DataType*>& centroids_dist,
                                    int subspace_centroids_cnt) {
    int cnt = 0;
    printf("subspace_centroids_cnt: %d\n", subspace_centroids_cnt);
    // printf("find centroids entering");
    order.resize(this->index_->subspace_cnt_);  // init()
    centroids_dist.resize(this->index_->subspace_cnt_);
    int subspace_dimensions = this->index_->vec_dim_ / this->index_->subspace_cnt_;
    DataType dist = 0;
    fmt::println("before subspace");
    // 对于每个子空间
    for (int subspace_indexer = 0; subspace_indexer < this->index_->subspace_cnt_;
         subspace_indexer++) {
      fmt::println("subsapce {}", subspace_indexer);
      int start_dim = subspace_indexer * subspace_dimensions;
      int final_dim = this->index_->vec_dim_;
      if (final_dim > start_dim + subspace_dimensions) {
        final_dim = start_dim + subspace_dimensions;
      }
      order[subspace_indexer] = (int*)Alloc64B(sizeof(int) * this->index_->bucket_num_);
      centroids_dist[subspace_indexer] =
          (DataType*)Alloc64B(sizeof(DataType) * this->index_->bucket_num_);

      fmt::println("entering cluster");
      // 计算距离，为了排序做准备
      for (int cluster_index = 0; cluster_index < this->index_->bucket_num_; cluster_index++) {
        const DataType* sub_query = kQuery + start_dim;
        // const DataType* sub_query = query_ + start_dim;
        dist = dist_func_(sub_query,
                          this->index_->subspace_ivf_centroids_[subspace_indexer] +
                              cluster_index * subspace_dimensions,
                          final_dim - start_dim);
        order[subspace_indexer][cluster_index] = cluster_index;
        centroids_dist[subspace_indexer][cluster_index] = dist;
      }
      fmt::println("before sorting");
      // sort()
      DataType* centroids_dist_tmp = centroids_dist[subspace_indexer];
      std::sort(order[subspace_indexer], order[subspace_indexer] + this->index_->bucket_num_,
                [centroids_dist_tmp](int a, int b) {
                  return centroids_dist_tmp[a] < centroids_dist_tmp[b];
                });
      std::sort(centroids_dist[subspace_indexer],
                centroids_dist[subspace_indexer] + this->index_->bucket_num_);  // 从小到大排序
      std::cout << "done calc orderlist." << std::endl;

      // for (int i = 0; i < this->index_->bucket_num_; i++) {
      //   std::cout << "order: " << order[subspace_indexer][i]
      //             << " dist: " << centroids_dist[subspace_indexer][i] << std::endl;
      // }
      // 只排序 subspaces_short_lists->at(subspace_indexer) 的前 subspace_centroids_cnt 个元素
      // 也就是搜索的数量
      // std::nth_element(subspaces_short_lists->at(subspace_indexer).begin(),
      //                  subspaces_short_lists->at(subspace_indexer).begin() +
      //                  subspace_centroids_cnt,
      //                  subspaces_short_lists->at(subspace_indexer).end());
      // subspaces_short_lists->at(subspace_indexer).resize(subspace_centroids_cnt);
      // std::sort(subspaces_short_lists->at(subspace_indexer).begin(),
      //           subspaces_short_lists->at(subspace_indexer).end());
    }  // 每一列都是按照升序排序的
  }

  // 这里是从merger_ 中获取最小距离的cell，类似于bfs过程。
  bool TraverseNextMultiIndexCell(const DataType* kQuery) {
    // std::vector<int> cell_coordinates;  是这个cell的坐标，通常大小为2
    std::vector<int> cell_inner_indices;
    printf("print heap_: \n");
    merger_.print_heap();
    // merger.lists_ptr 就是查询向量在每个维度下按照距离排序过后的中心点id 和 距离  first 是距离
    // second 是id
    if (!merger_.GetNextMergedItemIndices(&cell_inner_indices)) {
      printf("kong\n");
      return false;
    }
    std::cout << "00000000000000000000000000000" << std::endl;
    for (int i = 0; i < cell_inner_indices.size(); ++i) {
      std::cout << cell_inner_indices[i] << "  ";
    }
    // 应该也是一个二维的东西
    std::vector<int> cell_coordinates(cell_inner_indices.size());
    // 得到两个维度的坐标，放在cell_coordinates中
    for (int list_index = 0; list_index < merger_.order_ptr->size(); ++list_index) {
      cell_coordinates[list_index] =
          merger_.order_ptr->at(list_index)[cell_inner_indices[list_index]];
    }  // 定位坐标的，cluster的坐标，
    printf("get global index\n");
    // 拿到global index  计算得到结果
    int global_index = this->index_->GetGlobalCellIndex(cell_coordinates);

    printf("global_index: %d\n", global_index);

    IDType* id_pool = this->index_->id_buckets_.at(global_index);
    DataType* data_pool = this->index_->data_buckets_.at(global_index);
    DataType distance = 0;
    // printf("begin add result\n ");
    DataType* data_point = nullptr;
    std::cout << "cell data cnt: " << this->index_->cell_data_cnt_[global_index] << std::endl;
    for (int cell_data_index = 0; cell_data_index < this->index_->cell_data_cnt_[global_index];
         cell_data_index++) {
      data_point = data_pool + cell_data_index * this->index_->vec_dim_;
      distance = dist_func_(kQuery, data_point, this->index_->vec_dim_);
      // std::cout << "adding .... " << std::endl;
      // std::cout << id_pool[cell_data_index] << "  " << distance << std::endl;
      res_->Insert(id_pool[cell_data_index], distance);
    }

    // printf("finish add result\n ");
    return true;
    // return false;
  }
  void GetNearestNeighbours(const DataType* kQuery, int k) {
    // printf("begin GetNearestNeighbours\n");
    assert(this->index_->subspace_cnt_ > 0);
    std::vector<int*> order(this->index_->subspace_cnt_);
    std::vector<DataType*> centroids_dist(this->index_->subspace_cnt_);

    // printf("begin GetNearestSubspacesCentroids\n");
    GetNearestSubspacesCentroids(
        kQuery, order, centroids_dist,
        6);  // 在这里调用方法计算
             // query到每一个子空间的距离，subspaces_short_lists是一个vector，size对应着子空间的个数
             // 通常是2.
    // printf("subspaces_short_lists.size():  %lu\n ",subspaces_short_lists.size());
    // printf("print short_list\n");
    // print_subspaces_short_lists(&subspaces_short_lists);

    for (int i = 0; i < this->index_->subspace_cnt_; ++i) {
      for (int j = 0; j < this->index_->bucket_num_; ++j) {
        printf("order[%d][%d] = %d, centroids_dist[%d][%d] = %f\n", i, j, order[i][j], i, j,
               centroids_dist[i][j]);
      }
    }
    merger_.setLists(order, centroids_dist, this->index_->bucket_num_);
    int found_neighbour_cnt = 0;
    bool traverse_next_cell = true;
    int cells_visited = 0;
    // while(index_->found_neghbours_count_ < k && traverse_next_cell) {
    //     traverse_next_cell = TraverseNextMultiIndexCell();
    //     cells_visited += 1;
    // }
    // printf("enter begin while loop\n");
    while (cells_visited <= 0) {
      traverse_next_cell = TraverseNextMultiIndexCell(kQuery);
      cells_visited += 1;
    }

    for (int i = 0; i < this->index_->subspace_cnt_; ++i) {
      delete[] order[i];
      delete[] centroids_dist[i];
    }
  }

  void Search(int64_t query_num, int64_t query_dim, const DataType* query, int64_t k,
              DataType* distances, int64_t* labels) override {
    // printf("begin search imi\n");
    res_ = new ResultPool<IDType, DataType>(this->index_->vec_num_, 2 * k, k);
    assert(query_dim == this->index_->vec_dim_ &&
           "Query dimension must be equal to data dimension.");

    GetNearestNeighbours(query, k);
  }
};
}  // namespace alaya
