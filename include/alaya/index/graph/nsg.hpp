#pragma once

#include <atomic>
#include <random>
#include <stack>
#include <cstring>

#include "graph.h"
#include "nsglib/nsg.hpp"
#include "alaya/utils/memory.h"


namespace alaya {

template <typename IDType, typename DataType>
struct NSG : public Graph<IDType, DataType> {
  int R, L;
  NSG(int dim, IDType num, std::string_view metric, int R = 32, int L = 200)
      : Graph<IDType, DataType>(dim, num, metric), R(R), L(L) {

  }

  void BuildIndex(IDType vec_num, const DataType* kVecData) {
    this->vec_num_ = vec_num;
    // todo: 封装内存对齐分配函数
    size_t len = this->vec_num_ * this->edge_num_ * sizeof(IDType);
    this->linklist_ = Alloc2M(len);

    glass::NSG builder(this->vec_dim_, "L2", this->R, this->L);
    builder.Build(kVecData, vec_num);

    auto& graph = builder.final_graph;
    this->eps_ = graph.eps;
    this->edge_num_ = R;
    std::memcpy(this->linklist_, graph.data, len);

  }

};

} // namespace alaya