#pragma once

#include "quant.h"

namespace azalea {
template <typename IDType, typename CodeType, typename DistType>
struct PQ : Quantizer<IDType, CodeType, DistType> {
  PQ() = default;
};

} // namespace azaela