#pragma once

#include "../utils/metric_type.h"
#include "index.h"

namespace alaya {

/**
 * @brief 
 * 
 * @tparam IDType 
 * @tparam DataType 
 * @param dim 
 * @param description 
 * @param metric 
 * @return Index<IDType, DataType>* 
 */
template <typename IDType, typename DataType>
Index<IDType, DataType>* index_factory(
  int dim,
  const char* description,
  MetricType metric
);

} // namespace alaya