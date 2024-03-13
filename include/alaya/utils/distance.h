#pragma once

#include <cmath>

#include "avx2_distance.h"
#include "avx512_distance.h"
#include "platform_macros.h"
#include "sse_distance.h"
// #if defined(USE_AVX) || defined(USE_AVX512F)
// #include <immintrin.h>
// #endif

#include "metric_type.h"

namespace alaya {

template <typename T1, typename T2, typename U, typename... Params>
using DistFunc = U (*)(const T1*, const T2*, int, Params...);

UNROLL_BEGIN
template <typename DataType>
ALWAYS_INLINE DataType NaiveL2Sqr(const DataType* x, const DataType* y, int d) {
  DataType dist = 0;
  for (int i = 0; i < d; ++i) {
    DataType diff = x[i] - y[i];
    dist += diff * diff;
  }
  return dist;
}
UNROLL_END

UNROLL_BEGIN
template <typename DataType>
ALWAYS_INLINE DataType NaiveIp(const DataType* x, const DataType* y, int d) {
  DataType dist = 0;
  for (int i = 0; i < d; ++i) {
    dist += x[i] * y[i];
  }
  return dist;
}
UNROLL_END

UNROLL_BEGIN
template <typename DataType>
ALWAYS_INLINE DataType NaiveGetNorm(const DataType* x, int d) {
  DataType norm = 0;
  for (int i = 0; i < d; ++i) {
    norm += x[i] * x[i];
  }
  return std::sqrt(norm);
}
UNROLL_END

UNROLL_BEGIN
template <typename DataType>
ALWAYS_INLINE DataType NaiveCos(const DataType* x, const DataType* y, int d) {
  DataType norm_x = NaiveGetNorm(x, d);
  DataType norm_y = NaiveGetNorm(x, d);
  return NaiveIp(x, y, d) / (norm_x * norm_y);
}
UNROLL_END

template <typename DataType>
ALWAYS_INLINE DataType InnerProduct(const DataType* x, const DataType* y, int d) {
  if constexpr (std::is_same<DataType, float>::value) {
#if defined(USE_AVX512F)
    return InnerProductFloatAVX512(x, y, d);
#elif defined(USE_AVX)
    return InnerProductFloatAVX(x, y, d);
#else
    return NaiveIp(x, y, d);
#endif
  } else {
    return NaiveIp(x, y, d);
  }
}

template <typename DataType>
auto GetDistFunc(MetricType metric) {
  if (metric == MetricType::L2) {
    return NaiveL2Sqr;
  } else if (metric == MetricType::IP) {
    return InnerProduct;
  } else {
    return NaiveCos;
  }
}

}  // namespace alaya