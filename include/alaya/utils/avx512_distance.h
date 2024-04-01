#pragma once

#include "alaya/utils/sse_distance.h"
#include "platform_macros.h"

#if defined(USE_AVX512F)

#include <immintrin.h>

#include "avx2_distance.h"

namespace alaya {

inline float ReduceAddF32x16(__m512 x) {
  __m256 h256 = _mm512_extractf32x8_ps(x, 1);
  __m256 l256 = _mm512_castps512_ps256(x);
  __m256 sum256 = _mm256_add_ps(h256, l256);
  return ReduceAddF32x8(sum256);
}

/**
 * Calculates the inner product of two float arrays using AVX512 instructions.
 *
 * @param kX Pointer to the first float array.
 * @param kY Pointer to the second float array.
 * @param dim The dimension of the arrays.
 * @return The inner product of the two arrays.
 */
UNROLL_BEGIN
inline float InnerProductFloatAVX512(const float* kX, const float* kY,
                                     int dim) {
  __m512 x512, y512, diff512;
  __m512 sum512 = _mm512_setzero_ps();

  while (dim >= 16) {
    x512 = _mm512_loadu_ps(kX);
    kX += 16;
    y512 = _mm512_loadu_ps(kY);
    kY += 16;
    sum512 = _mm512_fmadd_ps(x512, y512, sum512);
    dim -= 16;
  }
  __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512),
                                _mm512_extractf32x8_ps(sum512, 1));

  if (dim >= 8) {
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

UNROLL_BEGIN
inline float AlignInnerProductFloatAVX512(const float* kX, const float* kY,
                                          int dim) {
  __m512 sum512 = _mm512_setzero_ps();

  const float* kEnd = kX + dim;

  while (kX < kEnd) {
    __m512 x512 = _mm512_loadu_ps(kX);
    kX += 16;
    __m512 y512 = _mm512_loadu_ps(kY);
    kY += 16;
    sum512 = _mm512_fmadd_ps(x512, y512, sum512);
  }
  return _mm512_reduce_add_ps(sum512);
}
UNROLL_END

UNROLL_BEGIN
inline float L2SqrFloatAVX512(const float* kX, const float* kY, int dim) {
  __m512 mx512, my512, diff512;
  __m512 sum512 = _mm512_setzero_ps();

  while (dim >= 16) {
    mx512 = _mm512_loadu_ps(kX);
    kX += 16;
    my512 = _mm512_loadu_ps(kY);
    kY += 16;
    diff512 = _mm512_sub_ps(mx512, my512);
    sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
    dim -= 16;
  }
  __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512),
                                _mm512_extractf32x8_ps(sum512, 1));

  if (dim >= 8) {
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

UNROLL_BEGIN
inline float AlignL2SqrFloatAVX512(const float* kX, const float* kY, int dim) {
  __m512 mx512, my512, diff512;
  __m512 sum512 = _mm512_setzero_ps();

  const float* kEnd = kX + dim;

  while (kX < kEnd) {
    mx512 = _mm512_loadu_ps(kX);
    kX += 16;
    my512 = _mm512_loadu_ps(kY);
    kY += 16;
    diff512 = _mm512_sub_ps(mx512, my512);
    sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
  }
  return _mm512_reduce_add_ps(sum512);
}
UNROLL_END

UNROLL_BEGIN
inline float NormSqrFloatAVX512(const float* kX, int dim) {
  __m512 sum512 = _mm512_setzero_ps();
  __m512 x512;

  while (dim >= 16) {
    x512 = _mm512_loadu_ps(kX);
    kX += 16;
    sum512 = _mm512_fmadd_ps(x512, x512, sum512);
    dim -= 16;
  }
  __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512),
                                _mm512_extractf32x8_ps(sum512, 1));
  __m256 x256;

  if (dim >= 8) {
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

inline float NormFloatAVX512(const float* kX, int dim) {
  return std::sqrt(NormSqrFloatAVX512(kX, dim));
}

UNROLL_BEGIN
inline float AlignNormSqrFloatAVX512(const float* pV, int dim) {
  __m512 sum512 = _mm512_setzero_ps();
  __m512 x512;

  const float* pEnd = pV + dim;
  while (pV < pEnd) {
    x512 = _mm512_loadu_ps(pV);
    sum512 = _mm512_fmadd_ps(x512, x512, sum512);
    pV += 16;
  }
  return _mm512_reduce_add_ps(sum512);
}
UNROLL_END

inline float AlignNormFloatAVX512(const float* pV, int dim) {
  return std::sqrt(AlignNormSqrFloatAVX512(pV, dim));
}

}  // namespace alaya

#endif