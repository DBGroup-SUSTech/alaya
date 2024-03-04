#pragma once

#include "quant.h"

namespace azalea {

template <typename IDType, typename CodeType, typename DistType>
struct SQ : Quantizer<IDType, CodeType, DistType> {
  SQ() = default;
};

} // namespace azalea