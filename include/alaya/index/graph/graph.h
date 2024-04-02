#pragma once

#include <vector>
#include <memory>

#include "upper_layer.h"
#include "alaya/index/index.h"
#include "alaya/utils/metric_type.h"

namespace alaya {

template <typename IDType, typename DataType>
struct Graph : Index<DataType, IDType> {
  int edge_num_;

  IDType *linklist_ = nullptr;

  std::unique_ptr<UpperLayer<IDType>> upper_layer_ = nullptr;

  std::vector<IDType> eps_;

  Graph() = default;
  Graph(int dim, std::string_view metric, int edge_num)
      : Index<DataType, IDType>(dim, metric), edge_num_(edge_num) {}

  const IDType *edges(int u) const { return linklist_ + edge_num_ * u; }

  IDType *edges(int u) { return linklist_ + edge_num_ * u; }

  IDType at(int i, int j) const { return linklist_[i * edge_num_ + j]; }

  IDType &at(int i, int j) { return linklist_[i * edge_num_ + j]; }

  // todo: 预期能改善性能，需要提供预取函数
//  void prefetch(int u, int lines) const {
//    mem_prefetch((char *)edges(u), lines);
//  }

};

} // namespace alaya