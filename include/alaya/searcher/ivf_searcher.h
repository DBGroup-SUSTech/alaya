#pragma once

#include "../index/bucket/ivf.h"
#include "../utils/metric_type.h"
#include "../utils/pool.h"
#include "fmt/core.h"
#include "searcher.h"

namespace alaya {
template <MetricType metric, typename DataType = float, typename IDType = uint64_t>
struct IvfSearcher : Searcher<metric, IVF<DataType, IDType>, DataType> {
  int nprobe_;
  explicit IvfSearcher(const IVF<DataType, IDType>* index)
      : Searcher<metric, IVF<DataType, IDType>, DataType>(index) {}

  ~IvfSearcher() = default;

  void Optimize(int num_threads = 0) override {
    // TODO Determining prefetching parameters through sampling experiments
  }

  // void SetNprobe(int n) override {}
  void SetNprobe(int n) { nprobe_ = n; }

  void Search(const DataType* query, int64_t k, DataType* distance,
              int64_t* result_id) const override {
    auto computer = this->index_->template GetComputer<metric>(query);

    LinearPool<DataType> pool(k);
    for (std::size_t i = 0; i < nprobe_; ++i) {
      auto bid = computer.order_[i];
      auto bucket_size = this->index_->GetBucketSize(bid);
      auto ids = (IDType*)(((char*)this->index_->id_buckets_[bid]) + 4);

      for (std::size_t j = 0; j < bucket_size; ++j) {
        auto dist = computer(bid, j, query);
        pool.Insert(ids[j], dist);
      }
    }
    for (std::size_t i = 0; i < k; ++i) {
      distance[i] = pool.pool_[i].dis_;
      result_id[i] = pool.pool_[i].id_;
    }
  }  // Search

  void BatchSearch(int64_t query_num, int64_t query_dim, const DataType* queries, int64_t k,
                   DataType* distances, int64_t* result_ids
                   // const SearchParameters* search_params = nullptr
  ) const override {
#pragma omp parallel for schedule(dynamic)
    for (std::size_t q = 0; q < query_num; ++q) {
      Search(queries + q * query_dim, k, distances + q * k, result_ids + q * k);
    }
  }  // BatchSearch
};

}  // namespace alaya