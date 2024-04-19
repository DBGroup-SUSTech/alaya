#pragma once

#include <atomic>
#include <random>
#include <stack>
#include <cstring>
#include <algorithm>

#include "graph.h"
#include "nsglib/nsg.hpp"
#include "alaya/utils/memory.h"
#include "alaya/utils/metric_type.h"


namespace alaya {

template <typename IDType, typename DataType>
struct NSG : public Graph<IDType, DataType> {
  int R, L;
  std::string metric;
  NSG(int dim, std::string_view metric, int R = 32, int L = 200)
      : Graph<IDType, DataType>(dim, metric, R), R(R), L(L), metric(metric) {

  }

  // todo: 建图时不要死板将数据集转成float，应该改一下nsglib来支持其他类型的训练数据
  void BuildIndex(IDType vec_num, const DataType* kVecData) override {
    this->vec_num_ = vec_num;
    size_t len = this->vec_num_ * this->edge_num_ * sizeof(IDType);
    this->linklist_ = (IDType*)Alloc2M(len);

    float* data = nullptr;
    std::unique_ptr<float[]> data_buf = nullptr;
    if constexpr (std::is_same_v<DataType, float>) {
      data = const_cast<DataType* >(kVecData);
    } else {
      data_buf = std::make_unique<float[]>(this->vec_num_ * this->vec_dim_);
      data = data_buf.get();
      std::transform(kVecData, kVecData + this->vec_num_ * this->vec_dim_, data,
                     [](const DataType& val) { return static_cast<float>(val); });
    }

    glass::NSG builder(this->vec_dim_, this->metric, this->R, this->L);
    builder.Build(data, vec_num);

    // todo: glass::NSG暂不支持模板，id默认为int
    auto& graph = builder.final_graph;
    this->eps_.assign(graph.eps.size(), 0);
    std::transform(graph.eps.begin(),graph.eps.end(),this->eps_.begin(),[](int val){return static_cast<IDType>(val);});
//    this->edge_num_ = R;
    std::memcpy(this->linklist_, graph.data, len);

  }

  // todo: to be defined
  void Save(const char* kFilePath) const override {

  }

  void Load(const char* kFilePath) override {

  }

};

} // namespace alaya