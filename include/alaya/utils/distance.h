#pragma once

#include <bits/floatn-common.h>

#include <cmath>
#include <type_traits>

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
DataType NaiveCos(const DataType* x, const DataType* y, int d) {
  DataType norm_x = NaiveGetNorm(x, d);
  DataType norm_y = NaiveGetNorm(x, d);
  return NaiveIp(x, y, d) / (norm_x * norm_y);
}
UNROLL_END

UNROLL_BEGIN
template <typename DataType>
void AddAssign(const DataType* kSrc, DataType* des, int dim) {
#pragma omp simd
  for (size_t i = 0; i < dim; ++i) {
    des[i] += kSrc[i];
  }
}
UNROLL_END

ALWAYS_INLINE float L2SqrFloat(const float* kX, const float* kY, int dim) {
#if defined(USE_AVX512F)
  return L2SqrFloatAVX512(kX, kY, dim);
#elif defined(USE_AVX)
  return L2SqrFloatAVX(kX, kY, dim);
#elif defined(USE_SSE)
  return L2SqrFloatSSE(kX, kY, dim);
#else
  return NaiveL2Sqr(kX, kY, dim);
#endif
  __builtin_unreachable();
}

ALWAYS_INLINE float AlignL2SqrFloat(const float* kX, const float* kY, int dim) {
#if defined(USE_AVX512F)
  return AlignL2SqrFloatAVX512(kX, kY, dim);
#elif defined(USE_AVX)
  return AlignL2SqrFloatAVX(kX, kY, dim);
#elif defined(USE_SSE)
  return AlignL2SqrFloatSSE(kX, kY, dim);
#else
  return NaiveL2Sqr(kX, kY, dim);
#endif
  __builtin_unreachable();
}

ALWAYS_INLINE float InnerProductFloat(const float* kX, const float* kY, int dim) {
#if defined(USE_AVX512F)
  return InnerProductFloatAVX512(kX, kY, dim);
#elif defined(USE_AVX)
  return InnerProductFloatAVX(kX, kY, dim);
#elif defined(USE_SSE)
  return InnerProductFloatSSE(kX, kY, dim);
#else
  return NaiveIp(kX, kY, dim);
#endif
  __builtin_unreachable();
}

ALWAYS_INLINE float AlignInnerProductFloat(const float* kX, const float* kY, int dim) {
#if defined(USE_AVX512F)
  return AlignInnerProductFloatAVX512(kX, kY, dim);
#elif defined(USE_AVX)
  return AlignInnerProductFloatAVX(kX, kY, dim);
#elif defined(USE_SSE)
  return AlignInnerProductFloatSSE(kX, kY, dim);
#else
  return NaiveIp(kX, kY, dim);
#endif
  __builtin_unreachable();
}

ALWAYS_INLINE float NormSqrFloat(const float* kX, int dim) {
#if defined(USE_AVX512F)
  return NormSqrFloatAVX512(kX, dim);
#elif defined(USE_AVX)
  return NormSqrFloatAVX(kX, dim);
#elif defined(USE_SSE)
  return NormSqrFloatSSE(kX, dim);
#else
  return NaiveGetNorm(kX, dim);
#endif
  __builtin_unreachable();
}

ALWAYS_INLINE float NormSqrTFloat(const float* kX, int dim) {
#if defined(USE_AVX512F)
  return NormSqrTFloatAVX512(kX, dim);
#elif defined(USE_AVX)
  return NormSqrTFloatAVX(kX, dim);
#elif defined(USE_SSE)
  return NormSqrTFloatSSE(kX, dim);
#else
  return NaiveGetNorm(kX, dim);
#endif
  __builtin_unreachable();
}

template <typename DataType>
DataType InnerProduct(const DataType* kX, const DataType* kY, int dim) {
  if constexpr (std::is_same<DataType, float>::value) {
    return InnerProductFloat(kX, kY, dim);
  } else {
    return NaiveIp(kX, kY, dim);
  }
}

template <typename DataType>
DataType L2Sqr(const DataType* kX, const DataType* kY, int dim) {
  if constexpr (std::is_same<DataType, float>::value) {
    return L2SqrFloat(kX, kY, dim);
  } else {
    return NaiveL2Sqr(kX, kY, dim);
  }
}

template <typename DataType>
DataType AlignL2Sqr(const DataType* kX, const DataType* kY, int dim) {
  if constexpr (std::is_same<DataType, float>::value) {
    return AlignL2SqrFloat(kX, kY, dim);
  } else {
    return NaiveL2Sqr(kX, kY, dim);
  }
}

template <typename DataType, bool IsAlign>
auto GetDistFunc(MetricType metric) {
  if constexpr (IsAlign == false) {
    if (metric == MetricType::L2) {
      return L2Sqr;
    } else if (metric == MetricType::IP) {
      return InnerProduct;
    } else {
      return NaiveCos;
    }
  } else {
    if (metric == MetricType::L2) {
      return AlignL2Sqr;
    } else if (metric == MetricType::IP) {
      return AlignInnerProductFloat;
    } else {
      return NaiveCos;
    }
  }
}

}  // namespace alaya