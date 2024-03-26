#pragma once

#include <vector>

#include "upper_layer.h"
#include "../index.h"
#include "../../utils/metric_type.h"

namespace alaya {

template <typename IDType, typename DataType>
struct Graph : Index<IDType, DataType> {
  int edge_num_;

  IDType *linklist_ = nullptr;

  std::unique_ptr<UpperLayer<IDType>> upper_layer_ = nullptr;

  std::vector<IDType> eps_;

  Graph() = default;

};

} // namespace alaya