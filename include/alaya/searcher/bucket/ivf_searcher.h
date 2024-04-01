#pragma once

#include <alaya/index/bucket/ivf.h>
#include <alaya/searcher/searcher.h>
#include <alaya/utils/distance.h>
#include <alaya/utils/memory.h>
#include <alaya/utils/pool.h>

#include <cstdint>
#include <cstdio>
#include <memory>

namespace alaya {

template <typename IDType, typename DataType>
struct InvertedListSearcher {
  typedef InvertedList<IDType, DataType> IndexType;
  std::unique_ptr<IndexType> index_ = nullptr;
  int64_t query_dim_;
  const DataType* query_;
  int64_t k_;
  DataType* distances_ = nullptr;
  ResultPool<IDType, DataType>* res_;
  DistFunc<DataType, DataType, DataType> dist_func_;
  // int64_t query_num_;

  InvertedListSearcher() = default;
  ~InvertedListSearcher() {
    delete res_;
    printf("InvertedListSearcher destructor\n");
  };
  // setup index for search
  void SetIndex(const IndexType& index) { this->index_ = std::make_unique<IndexType>(index); }

  // main search function
  void Search(int64_t query_num, int64_t query_dim, const DataType* query, int64_t k,
              DataType* distances, int64_t* labels) {
    dist_func_ = GetDistFunc<DataType, false>(index_->metric_type_);
    printf("begin search\n");
    // init result pool
    res_ = new ResultPool<IDType, DataType>(index_->vec_num_, 2 * k, k);
    assert(query_dim == index_->vec_dim_ && "Query dimension must be equal to data dimension.");
    query_dim_ = query_dim;
    query_ = query;
    k_ = k;
    distances_ = distances;

    // init search ordered list
    InitSearcher(query);

    int ordered_bucket_id = 0;
    for (int i = 0; i < index_->bucket_num_; ++i) {
      ordered_bucket_id = index_->order_list_[i].first;
      printf("ordered_bucket_id = %d\n", ordered_bucket_id);

      DataType* data_point = index_->buckets_[ordered_bucket_id].data();
      PrefetchL1(data_point);
      auto& id_list = index_->id_buckets_[ordered_bucket_id];
      DataType dist = 0;
      for (int j = 0; j < id_list.size(); ++j) {
        dist = dist_func_(query_, data_point + j * query_dim_, query_dim_);
        res_->Insert(id_list[j], dist);
      }
    }
    printf("after inserting ids\n");

    for (int i = 0; i < k; ++i) {
      distances_[i] = res_->result_.pool_[i].dis_;
      printf("%d %f\n", res_->result_.pool_[i].id_, res_->result_.pool_[i].dis_);
      // std::cout << res.result_.pool_[i].id_ << " " << res.result_.pool_[i].dis_ << std::endl;
    }

    printf("Search finished\n");
  }
  void InitSearcher(const DataType* query) {
    query_ = query;
    PrefetchL1(index_->centroids_);
    InitOrderList();
    std::sort(index_->order_list_.begin(), index_->order_list_.end(),
              [](const std::pair<int, DataType>& a, const std::pair<int, DataType>& b) {
                return a.second < b.second;
              });
  }

  void InitOrderList() {
    for (int i = 0; i < index_->bucket_num_; i++) {
      index_->order_list_[i].first = i;
      index_->order_list_[i].second =
          dist_func_(query_, index_->centroids_ + i * index_->vec_dim_, index_->vec_dim_);
    }
  }
};
}  // namespace alaya