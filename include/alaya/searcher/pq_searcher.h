#pragma once

#include <memory>

#include "../index/quantizer/product_quantizer.h"
#include "alaya/utils/distance.h"
#include "searcher.h"

namespace alaya {

template <unsigned CodeBits = 8, typename IDType = int64_t, typename DataType = float,
          typename IndexType = ProductQuantizer<CodeBits, IDType, DataType>>
struct PQSearcher : Searcher<IndexType, DataType> {
  PQSearcher() = default;
  virtual ~PQSearcher() = default;

  void SetIndex(const IndexType& index) override { this->index_ = std::make_unique(index); }

  void Optimize(int num_threads = 0) override {
    // TODO Determining prefetching parameters through sampling experiments
  }

  void SetEf(int ef) override {}

  void InitSearch(const DataType* kQuery) {
    auto book_num = this->index_->book_num_;
    auto book_size = this->index_->book_size_;
    auto sub_dim = this->index_->sub_dimensions_;
    auto sub_start = this->index_->sub_vec_start_;
    auto codebook = this->index_->codebook_;
    auto code_dist = this->index_->code_dist_;
    for (unsigned i = 0; i < book_num; i += 2) {
      Sgemv(kQuery + sub_start[i], codebook[i].data(), code_dist + i * book_size, sub_dim[i],
            book_size);
      // VecMatMul(kQuery + sub_start[i], codebook[i].data(), code_dist + i * book_size, sub_dim[i],
      //           book_size);
    }
  }
};

}  // namespace alaya