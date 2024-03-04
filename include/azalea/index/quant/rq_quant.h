#pragma once

#include "quant.h"

namespace azalea {

template <typename IDType, typename CodeType, typename DistType>
struct RQ : Quantizer<IDType, CodeType, DistType> {
  RQ() = default;
};

} // namespace azalea