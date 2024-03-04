#pragma once

#include "MetricType.h"
#include "index.h"

namespace azalea {

template <typename IDType, typename DistType>
Index<IDType, DistType>* index_factory(
  int dim,
  const char* description,
  MetricType metric
);

} // namespace azalea