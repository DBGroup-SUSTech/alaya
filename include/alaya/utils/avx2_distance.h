#pragma once

#include <immintrin.h>

#include <cmath>
#include <cstdint>

#include "platform_macros.h"
#include "sse_distance.h"

#if defined(USE_AVX)

namespace alaya {

/**
 * Calculates the sum of all elements in a 256-bit AVX2 register containing 8
 * single-precision floating-point values.
 *
 * @param x The AVX2 register containing the values to be summed.
 * @return The sum of all elements in the AVX2 register.
 */
inline float ReduceAddF32x8(__m256 x) {
  __m128 h128 = _mm256_extractf128_ps(x, 1);
  __m128 l128 = _mm256_castps256_ps128(x);
  __m128 sum128 = _mm_add_ps(h128, l128);
  return ReduceAddF32x4(sum128);
}

/**
 * Calculates the sum of 8 32-bit integer in the given 256-bit integer vector.
 *
 * @param x The 256-bit integer vector (__m256i) to reduce.
 * @return The sum of all elements in the vector as a 32-bit integer.
 */
inline int32_t ReduceAddI32x8(__m256i x) {
  __m128i h128 = _mm256_extractf128_si256(x, 1);
  __m128i l128 = _mm256_castsi256_si128(x);
  __m128i sum128 = _mm_add_epi32(h128, l128);
  return ReduceAddI32x4(sum128);
}

/**
 * Calculates the sum of 32 8-bit integer in the 256-bit integer vector.
 *
 * @param x The 256-bit integer vector (__m256i) to be reduced.
 * @return The sum of all elements in the vector as a 32-bit integer.
 */
inline int32_t ReduceAddI8x32(__m256i x) {
  __m128i h128 = _mm256_extractf128_si256(x, 1);
  __m128i l128 = _mm256_castsi256_si128(x);
  return ReduceAddI8x16(h128) + ReduceAddI8x16(l128);
}

/**
 * Calculates the inner product of two float arrays using AVX instructions.
 *
 * @param kX Pointer to the first float array.
 * @param kY Pointer to the second float array.
 * @param dim The dimension of the arrays.
 * @return The inner product of the two arrays.
 */
UNROLL_BEGIN
inline float InnerProductFloatAVX(const float* kX, const float* kY, int dim) {
  __m256 sum256 = _mm256_setzero_ps();

  while (dim >= 8) {
    __m256 x256 = _mm256_loadu_ps(kX);
    kX += 8;
    __m256 y256 = _mm256_loadu_ps(kY);
    kY += 8;
    sum256 = _mm256_fmadd_ps(x256, y256, sum256);
    dim -= 8;
  }
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                             _mm256_extractf128_ps(sum256, 1));
  __m128 x128, y128;

  if (dim >= 4) {
    x128 = _mm_loadu_ps(kX);
    kX += 4;
    y128 = _mm_loadu_ps(kY);
    kY += 4;
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
    dim -= 4;
  }

  if (dim > 0) {
    x128 = MaskedReadFloat(dim, kX);
    y128 = MaskedReadFloat(dim, kY);
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
  }
  return ReduceAddF32x4(sum128);
}
UNROLL_END

/**
 * Calculates the inner product of two aligned float arrays using AVX
 * instructions.
 *
 * @param kX Pointer to the first float array.
 * @param kY Pointer to the second float array.
 * @param dim The dimension of the arrays.
 * @return The inner product of the two aligned arrays.
 */
UNROLL_BEGIN
inline float AlignInnerProductFloatAVX(const float* kX, const float* kY,
                                       int dim) {
  __m256 sum256 = _mm256_setzero_ps();

  const float* kEnd = kX + dim;

  while (kX < kEnd) {
    __m256 x256 = _mm256_loadu_ps(kX);
    __m256 y256 = _mm256_loadu_ps(kY);
    sum256 = _mm256_fmadd_ps(x256, y256, sum256);
    kX += 8;
    kY += 8;
  }
  return ReduceAddF32x8(sum256);
}
UNROLL_END

/**
 * Calculates the squared L2 distance between two float arrays using AVX
 * instructions.
 *
 * @param kX Pointer to the first float array.
 * @param kY Pointer to the second float array.
 * @param dim The dimension of the arrays.
 * @return The squared L2 distance between the two arrays.
 */
UNROLL_BEGIN
inline float L2SqrFloatAVX(const float* kX, const float* kY, int dim) {
  __m256 sum256 = _mm256_setzero_ps();

  while (dim >= 8) {
    __m256 mx256 = _mm256_loadu_ps(kX);
    kX += 8;
    __m256 my256 = _mm256_loadu_ps(kY);
    kY += 8;
    __m256 diff256 = _mm256_sub_ps(mx256, my256);
    sum256 = _mm256_fmadd_ps(diff256, diff256, sum256);
    dim -= 8;
  }
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                             _mm256_extractf128_ps(sum256, 1));
  __m128 mx128, my128, diff128;

  if (dim >= 4) {
    mx128 = _mm_loadu_ps(kX);
    kX += 4;
    my128 = _mm_loadu_ps(kY);
    kY += 4;
    diff128 = _mm_sub_ps(mx128, my128);
    sum128 = _mm_fmadd_ps(diff128, diff128, sum128);
    dim -= 4;
  }

  if (dim > 0) {
    mx128 = MaskedReadFloat(dim, kX);
    my128 = MaskedReadFloat(dim, kY);
    diff128 = _mm_sub_ps(mx128, my128);
    sum128 = _mm_fmadd_ps(diff128, diff128, sum128);
  }
  return ReduceAddF32x4(sum128);
}
UNROLL_END

/**
 * Calculates the squared L2 distance between two align float arrays using AVX
 * instructions.
 *
 * @param kX Pointer to the first float array.
 * @param kY Pointer to the second float array.
 * @param dim The dimension of the arrays.
 * @return The squared L2 distance between the two align float arrays.
 */
UNROLL_BEGIN
inline float AlignL2SqrFloatAVX(const float* kX, const float* kY, int dim) {
  __m256 sum256 = _mm256_setzero_ps();

  const float* kEnd = kX + dim;

  while (kX < kEnd) {
    __m256 mx256 = _mm256_loadu_ps(kX);
    __m256 my256 = _mm256_loadu_ps(kY);
    __m256 diff256 = _mm256_sub_ps(mx256, my256);
    sum256 = _mm256_fmadd_ps(diff256, diff256, sum256);
    kX += 8;
    kY += 8;
  }
  return ReduceAddF32x8(sum256);
}
UNROLL_END

UNROLL_BEGIN
inline float NormSqrFloatAVX(const float* kX, int dim) {
  __m256 sum256 = _mm256_setzero_ps();
  __m256 x256;

  while (dim >= 8) {
    x256 = _mm256_loadu_ps(kX);
    kX += 8;
    sum256 = _mm256_fmadd_ps(x256, x256, sum256);
    dim -= 8;
  }
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                             _mm256_extractf128_ps(sum256, 1));
  __m128 x128;

  if (dim >= 4) {
    x128 = _mm_loadu_ps(kX);
    kX += 4;
    sum128 = _mm_fmadd_ps(x128, x128, sum128);
    dim -= 4;
  }

  if (dim > 0) {
    x128 = MaskedReadFloat(dim, kX);
    sum128 = _mm_fmadd_ps(x128, x128, sum128);
  }
  return ReduceAddF32x4(sum128);
}
UNROLL_END

inline float NormFloatAVX(const float* kX, int dim) {
  return std::sqrt(NormSqrFloatAVX(kX, dim));
}

UNROLL_BEGIN
inline float AlignNormSqrFloatAVX(const float* pV, int dim) {
  __m256 sum256 = _mm256_setzero_ps();
  __m256 x256;

  const float* pEnd = pV + dim;
  while (pV < pEnd) {
    x256 = _mm256_loadu_ps(pV);
    sum256 = _mm256_fmadd_ps(x256, x256, sum256);
    pV += 4;
  }
  return ReduceAddF32x8(sum256);
}
UNROLL_END

inline float AlignNormFloatAVX(const float* pV, int dim) {
  return std::sqrt(NormSqrFloatAVX(pV, dim));
}

}  // namespace alaya
#endif  // defined(USE_AVX)