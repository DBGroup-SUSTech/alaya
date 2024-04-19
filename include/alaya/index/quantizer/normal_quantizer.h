//
// Created by weijian on 4/1/24.
//

#pragma once

#include "quantizer.h"
#include "../../utils/metric_type.h"
#include "../../utils/memory.h"

namespace alaya {

template <unsigned CodeBits = 8, typename DataType = float, typename IDType = int64_t>
struct NormalQuantizer : Quantizer<CodeBits, DataType, IDType> {
  DataType* data = nullptr;
  DistFunc<DataType, DataType, DataType> dist_func;
  DataType* query = nullptr;

//  NormalQuantizer() = default;

  NormalQuantizer(int vec_dim, MetricType metric) : Quantizer<CodeBits, DataType, IDType>(vec_dim, kAlgin16, metric) {
    dist_func = GetDistFunc<DataType, false>(this->metric_type_);
  }

  void BuildIndex(IDType vec_num, const DataType* kVecData) override {
    this->vec_num_ = vec_num;
    this->data = const_cast<DataType*>(kVecData);
  }

//  void set_query(DataType* q) {
//    query = q;
//  }
//
//  DataType operator()(IDType vec_id) const override{
//    return dist_func(query, data + vec_id * this->vec_dim_, this->vec_dim_);
//  }

  DataType* GetCodeWord(IDType data_id) override {
    return data + data_id * this->vec_dim_;
  }

  struct Computer {
    DistFunc<DataType, DataType, DataType> dist_func;
    DataType* q;
    const NormalQuantizer& quant;
    Computer(const NormalQuantizer& quant, const DataType* query)
        : quant(quant), q((DataType*)Alloc64B(quant.vec_dim_)) {

    }
    ~Computer() {free(q);}
    DataType operator()(IDType vec_id) const {
//      return
    }
  };

  // todo: to be defined
  void Save(const char* kFilePath) const override {

  }

  void Load(const char* kFilePath) override {

  }

};

}  // namespace alaya


