#pragma once

#include "../../utils/memory.h"
#include "../index.h"
#include "quantizer.h"

namespace alaya {

template <unsigned CodeBits = 8, typename IDType = int64_t, typename DataType = float>
struct ProductQuantizer : Quantizer<CodeBits, IDType, DataType> {
  ProductQuantizer() = default;

  ProductQuantizer(int vec_dim, IDType vec_num, MetricType metric, unsigned book_num)
      : Quantizer<CodeBits, IDType, DataType>(vec_dim, vec_num, metric) {
    this->book_num_ = book_num;
    
  }

  DataType* Decode(IDType data_id) override { return nullptr; }

  DataType* GetCodeWord(IDType data_id) override { return nullptr; }

  void BuildIndex(IDType vec_num, const DataType* kVecData) override{};

  void Save(const char* kFilePath) const override{};

  void Load(const char* kFilePath) override{};

  ~ProductQuantizer() override{};
};

}  // namespace alaya