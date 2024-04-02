#pragma once

#include <alaya/index/bucket/ivf_origin.h>
#include <alaya/searcher/searcher.h>
#include <alaya/utils/distance.h>
#include <alaya/utils/memory.h>
#include <alaya/utils/pool.h>

#include <cstdint>
#include <cstdio>
#include <memory>

namespace alaya {

template <typename DataType, typename IDType, typename IndexType = InvertedList<DataType, IDType>>
struct InvertedListSearcher : Searcher<IndexType, DataType> {
  using OrderPair = std::vector<std::pair<int, DataType>>;
  // int64_t k_;

  DistFunc<DataType, DataType, DataType> dist_func_;

  // int64_t query_num_;

  explicit InvertedListSearcher(const IndexType* index)
      : Searcher<IndexType, DataType>(index, nullptr),
        dist_func_(GetDistFunc<DataType, false>(this->index_->metric_type_)) {}

  ~InvertedListSearcher() { printf("InvertedListSearcher destructor\n"); };

  // main search function
  void Search(int64_t query_num, int64_t query_dim, const DataType* query, int64_t k,
              DataType* distances, IDType* labels) const override {
    printf("begin search\n");
    // init result pool
    ResultPool<DataType, IDType>* res =
        new ResultPool<DataType, IDType>(this->index_->vec_num_, 2 * k, k);
    OrderPair order_list = OrderPair(this->index_->bucket_num_);
    printf("sizeof order_list = %lu\n", order_list.size());
    assert(query_dim == this->index_->vec_dim_ &&
           "Query dimension must be equal to data dimension.");

    // init search ordered list
    InitSearcher(query, order_list);
    std::cout << "after init" << std::endl;

    int ordered_bucket_id = 0;
    for (int i = 0; i < this->index_->bucket_num_; ++i) {
      ordered_bucket_id = order_list[i].first;

      printf("ordered_bucket_id = %d\n", ordered_bucket_id);
      const DataType* data_point = this->index_->buckets_[ordered_bucket_id].data();

      PrefetchL1(data_point);
      auto& id_list = this->index_->id_buckets_[ordered_bucket_id];
      DataType dist = 0;
      for (int j = 0; j < id_list.size(); ++j) {
        dist = dist_func_(query, data_point + j * query_dim, query_dim);
        res->Insert(id_list[j], dist);
      }
    }
    printf("after inserting ids\n");

    for (int i = 0; i < k; ++i) {
      distances[i] = res->result_.pool_[i].dis_;
      labels[i] = res->result_.pool_[i].id_;

      std::cout << res->result_.pool_[i].id_ << " " << res->result_.pool_[i].dis_ << std::endl;
    }

    printf("Search finished\n");

    delete res;
  }
  void InitSearcher(const DataType* query, OrderPair& order_list) const {
    PrefetchL1(this->index_->centroids_);
    InitOrderList(query, order_list);
    std::sort(order_list.begin(), order_list.end(),
              [](const std::pair<int, DataType>& a, const std::pair<int, DataType>& b) {
                return a.second < b.second;
              });
  }

  void InitOrderList(const DataType* query, OrderPair& order_list) const {
    for (int i = 0; i < this->index_->bucket_num_; i++) {
      order_list[i].first = i;
      order_list[i].second = dist_func_(
          query, this->index_->centroids_ + i * this->index_->vec_dim_, this->index_->vec_dim_);
      std::cout << "id = " << order_list[i].first << "  dist = " << order_list[i].second
                << std::endl;
    }
  }
};
}  // namespace alaya