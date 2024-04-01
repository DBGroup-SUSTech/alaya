#pragma once

#include <cmath>
#include <type_traits>

#include "avx2_distance.h"
#include "avx512_distance.h"
#include "metric_type.h"
#include "platform_macros.h"
#include "sse_distance.h"

#ifdef USE_MKL
#include "mkl.h"
#elif USE_BLAS
#include <cblas.h>
#endif

namespace alaya {

/**
 * Represents a function pointer type for calculating distance between two
 * vectors.
 *
 * @tparam T1 The type of elements in the first vector.
 * @tparam T2 The type of elements in the second vector.
 * @tparam U The return type of the distance calculation function.
 * @tparam Params Additional parameters for the distance calculation function.
 *
 * @param ptr1 Pointer to the first array.
 * @param ptr2 Pointer to the second array.
 * @param size The size of the arrays.
 * @param params Additional parameters for the distance calculation function.
 *
 * @return The calculated distance between the two arrays.
 */
template <typename T1, typename T2, typename U, typename... Params>
using DistFunc = U (*)(const T1*, const T2*, int, Params...);

/**
 * Calculates the squared L2 distance between two arrays.
 *
 * @tparam DataType The type of elements in the arrays.
 * @param x Pointer to the first array.
 * @param y Pointer to the second array.
 * @param d The size of the arrays.
 * @return The squared L2 distance between the two arrays.
 */
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

/**
 * Calculates the inner product of two arrays using a naive approach.
 *
 * @param x Pointer to the first array.
 * @param y Pointer to the second array.
 * @param d The size of the arrays.
 * @return The inner product of the two arrays.
 */
UNROLL_BEGIN
template <typename DataType>
ALWAYS_INLINE inline DataType NaiveIp(const DataType* x, const DataType* y, int d) {
  DataType dist = 0;
  for (int i = 0; i < d; ++i) {
    dist += x[i] * y[i];
  }
  return dist;
}
UNROLL_END

UNROLL_BEGIN
template <typename DataType>
ALWAYS_INLINE DataType NaiveGetSqrNorm(const DataType* kX, int d) {
  DataType norm = 0;
  for (int i = 0; i < d; ++i) {
    norm += kX[i] * kX[i];
  }
  return norm;
}
UNROLL_END

UNROLL_BEGIN
template <typename DataType>
ALWAYS_INLINE DataType NaiveGetNorm(const DataType* kX, int d) {
  return std::sqrt(NaiveGetSqrNorm(kX, d));
}
UNROLL_END

UNROLL_BEGIN
template <typename DataType>
DataType NaiveCos(const DataType* kX, const DataType* kY, int d) {
  DataType norm_x = NaiveGetNorm(kX, d);
  DataType norm_y = NaiveGetNorm(kX, d);
  return NaiveIp(kX, kY, d) / (norm_x * norm_y);
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

ALWAYS_INLINE
inline float L2SqrFloat(const float* kX, const float* kY, int dim) {
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

ALWAYS_INLINE
inline float AlignL2SqrFloat(const float* kX, const float* kY, int dim) {
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

ALWAYS_INLINE
inline float InnerProductFloat(const float* kX, const float* kY, int dim) {
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

ALWAYS_INLINE
inline float AlignInnerProductFloat(const float* kX, const float* kY, int dim) {
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

ALWAYS_INLINE
inline float NormSqrFloat(const float* kX, int dim) {
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

ALWAYS_INLINE
inline float NormFloat(const float* kX, int dim) {
#if defined(USE_AVX512F)
  return NormFloatAVX512(kX, dim);
#elif defined(USE_AVX)
  return NormFloatAVX(kX, dim);
#elif defined(USE_SSE)
  return NormFloatSSE(kX, dim);
#else
  return NaiveGetNorm(kX, dim);
#endif
  __builtin_unreachable();
}

template <typename DataType>
DataType GetNorm(const DataType* kX, int dim) {
  if constexpr (std::is_same<DataType, float>::value) {
    return NormFloat(kX, dim);
  } else {
    return NaiveGetNorm(kX, dim);
  }
}

template <typename DataType>
DataType GetSqrNorm(const DataType* kX, int dim) {
  if constexpr (std::is_same<DataType, float>::value) {
    return NormSqrFloat(kX, dim);
  } else {
    return NaiveGetSqrNorm(kX, dim);
  }
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
DataType AlignInnerProduct(const DataType* kX, const DataType* kY, int dim) {
  if constexpr (std::is_same<DataType, float>::value) {
    return AlignInnerProductFloat(kX, kY, dim);
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

/**
 * Calculates the squared L2 distance of two vectors aligned in 16 dimensions.
 *
 * @param kX Pointer to the first array of data.
 * @param kY Pointer to the second array of data.
 * @param dim The dimension of the arrays.
 * @return The squared L2 distance between the two arrays.
 */
template <typename DataType>
DataType AlignL2Sqr(const DataType* kX, const DataType* kY, int dim) {
  if constexpr (std::is_same<DataType, float>::value) {
    return AlignL2SqrFloat(kX, kY, dim);
  } else {
    return NaiveL2Sqr(kX, kY, dim);
  }
}

/**
 * @brief Returns the distance function based on the specified metric.
 *
 * This function returns a distance function based on the specified metric.
 *
 * @param metric The metric type used to determine the distance function.
 * @return The distance function corresponding to the specified metric.
 */
template <typename DataType, bool IsAlign>
DistFunc<DataType, DataType, DataType> GetDistFunc(MetricType metric) {
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

inline void Sgemv(const float* kVec, const float* kMat, float* res, const int kDim,
                  const int kNum) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, kNum, kDim, 1.0, kMat, kDim, kVec, 1, 0.0, res, 1);
}

inline void Sgemm(const float* mat1, const int dim1, const int n1, const float* mat2,
                  const int dim2, const int n2, float* res) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n1, n2, dim1, 1.0, mat1, dim1, mat2, dim2,
              0.0, res, n2);
}

inline void VecMatMul(const float* kVec, const float* kMat, float* res, const int kDim,
                      const int kNum) {
  // TODO Impl SIMD Vec Mat Mul
}

}  // namespace alaya