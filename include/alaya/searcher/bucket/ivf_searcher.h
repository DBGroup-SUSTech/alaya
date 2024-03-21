#pragma once

#include <alaya/index/bucket/ivf.h>
#include <alaya/searcher/searcher.h>
#include <alaya/utils/distance.h>
#include <alaya/utils/heap.h>
#include <alaya/utils/memory.h>

#include <cstdint>

namespace alaya {

template <typename IDType, typename DataType>
struct InvertedListSearcher : Searcher<InvertedList<IDType, DataType>, DataType> {
  typedef InvertedList<IDType, DataType> IndexType;
  std::unique_ptr<IndexType> index_ = nullptr;
  //   int64_t query_num_;
  int64_t query_dim_;
  DataType* query_;
  int64_t k_;
  DataType* distances_;

  InvertedListSearcher() = default;
  ~InvertedListSearcher() = default;

  void SetIndex(const IndexType& index) override {
    this->index_ = std::make_unique<IndexType>(index);
  }

  void Search(int64_t query_dim, DataType* query, int64_t k, DataType* distances,
              int64_t* labels) override {
    // query_num_ = query_num;
    ResultPool<IDType, DataType> res(index_->data_num_, 2 * k, k);
    assert(query_dim == index_->data_dim_ && "Query dimension must be equal to data dimension.");
    query_dim_ = query_dim;
    query_ = query;
    k_ = k;
    InitSearcher(query);
    int ordered_bucket_id = 0;
    for (int i = 0; i < index_->bucket_num_; ++i) {
      ordered_bucket_id = index_->order_list_[i].first;
      DataType* data_point = index_->buckets_[ordered_bucket_id].data();
      PrefetchL1(data_point);
      auto& id_list = index_->id_buckets_[ordered_bucket_id];
      auto DistFunc = GetDistanceFunc(index_->metric_type_);
      for (int j = 0; j < id_list.size(); ++j) {
        DataType dist = DistFunc(query_, data_point + j * query_dim_, query_dim_);
        res.Insert(id_list[j], dist);
      }
    }

    for (int i = 0; i < k; ++i) {
      distances_[i] = res.result_.pool_[i].dis;
      std::cout << res.result_.pool_[i].id << " " << res.result_.pool_[i].dis << std::endl;
    }
  }
  void InitSearcher(DataType* query) {
    query_ = query;
    PrefetchL1(index_->centroids_data_);
    InitOrderList();
    std::sort(index_->order_list_.begin(), index_->order_list_.end(),
              [](const std::pair<int, DataType>& a, const std::pair<int, DataType>& b) {
                return a.second < b.second;
              });
  }

  void InitOrderList() {
    for (int i = 0; i < index_->bucket_num_; i++) {
      index_->order_list_[i].first = i;
      auto DistFunc = GetDistanceFunc(index_->metric_type_);
      index_->order_list_[i].second =
          DistFunc(query_, index_->centroids_data_ + i * index_->data_dim_, index_->data_dim_,
                   index_->metric_type_);
    }
  }
};
}  // namespace alaya