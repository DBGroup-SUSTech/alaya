#pragma once

#include "../index.h"
#include "../MetricType.h"
#include "upper_layer.h"
#include <vector>

namespace azalea {

template <typename IDType, typename DistType>
struct Graph : Index<IDType, DistType> {
  int edge_num_;

  IDType *linklist_ = nullptr;

  std::unique_ptr<UpperLayer<IDType>> initializer = nullptr;

  std::vector<int> eps;

  Graph() = default;

  // constexpr static auto dist_func = select_distance_func<metric_type_>();
};

} // namespace azalea