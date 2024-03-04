#pragma once

#include <unordered_map>
#include "distance.h"

namespace azalea {


enum class MetricType {
  L2,
  IP,
  COS,
};

template<MetricType metric>
constexpr auto select_distance_func() {
  if constexpr (metric == MetricType::L2) {
    return l2_dist;
  } else if constexpr (metric == MetricType::IP) {
    return ip_dist;
  } else if constexpr (metric == MetricType::COS) {
    return cos_dist;
  }
}


} // namespace azalea