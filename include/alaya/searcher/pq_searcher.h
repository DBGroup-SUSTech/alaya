#pragma once

#include <cstdint>
#include <memory>

#include "../index/quantizer/product_quantizer.h"
#include "alaya/utils/distance.h"
#include "alaya/utils/pool.h"
#include "searcher.h"

namespace alaya {

template <unsigned CodeBits = 8, typename DataType = float, typename IDType = int64_t,
          typename IndexType = ProductQuantizer<CodeBits, DataType, IDType>>
struct PQSearcher : Searcher<IndexType, IndexType, DataType> {
  // PQSearcher() = default;

  explicit PQSearcher(const IndexType* index) : Searcher<IndexType, DataType>(index, nullptr) {}

  ~PQSearcher() = default;

  // void SetIndex(const IndexType& index) override { this->index_ = std::move(index); }

  void Optimize(int num_threads = 0) override {
    // TODO Determining prefetching parameters through sampling experiments
  }

  void SetEf(int ef) override {}

  void Search(int64_t query_num, int64_t query_dim, const DataType* queries, int64_t k,
              DataType* distances, int64_t* labels) override {
    for (auto qid = 0; qid < query_num; ++qid) {
      this->index_->InitCodeDist(queries + qid * query_dim);
      LinearPool<DataType, int64_t> pool(k);
      SearchImpl<LinearPool<DataType, int64_t>, true>(pool);
      for (auto i=0; i<k; ++i) {
        distances[qid * k + i] = pool.distances_[i];
        labels[qid * k + i] = pool.labels_[i];
      }
    }
  }

  template <typename PoolType, bool IsLookup>
  void SearchImpl(PoolType& pool, const DataType* kQuery) const {
    auto vec_num = this->index_->vec_num_;
    if constexpr (IsLookup) {
      for (auto vid = 0; vid < vec_num; ++vid) {
        auto dist = this->index_(vid);
        pool.Insert(dist, vid);
      }
    } else {
      if (this->index_->encode_vecs_ == nullptr) {
        this->index_->Encode();
      }
      for (auto vid = 0; vid < vec_num; ++vid) {
        auto dist = this->index_(vid, kQuery);
        pool.Insert(dist, vid);
      }
    }
  }
};

}  // namespace alaya