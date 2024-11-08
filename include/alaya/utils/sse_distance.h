#pragma once

#include "platform_macros.h"

#if defined(USE_SSE)
#include <x86intrin.h>

#include <cmath>

namespace alaya {

/**
 * Reads a masked float value from the given data array using SSE instructions.
 *
 * @param dim The size of the data array.
 * @param data Pointer to the data array.
 * @return The masked float value read from the data array.
 */
inline __m128 MaskedReadFloat(const std::size_t dim, const float* data) {
  assert(0 < dim && dim < 4);
  ALIGNED(16) float buf[4] = {0, 0, 0, 0};
  switch (dim) {
    case 3:
      buf[2] = data[2];
    case 2:
      buf[1] = data[1];
    case 1:
      buf[0] = data[0];
  }
  return _mm_load_ps(buf);
}

/**
 * Reads an integer value from the given data array using a mask.
 *
 * @param dim The size of the data array.
 * @param data Pointer to the array of integers.
 * @return The integer value read from the data array.
 */
inline __m128i MaskedReadInt(const std::size_t dim, const int* data) {
  assert(0 < dim && dim < 4);
  ALIGNED(16) int buf[4] = {0, 0, 0, 0};
  switch (dim) {
    case 3:
      buf[2] = data[2];
    case 2:
      buf[1] = data[1];
    case 1:
      buf[0] = data[0];
  }
  return _mm_load_si128((__m128i*)buf);
}

// Adapted from
// https://stackoverflow.com/questions/60108658/fastest-method-to-calculate-sum-of-all-packed-32-bit-integers-using-avx512-or-av

/**
 * Calculates the sum of the four float in the given __m128 vector.
 *
 * @param x The input __m128 vector.
 * @return The sum of the four float in the vector.
 */
inline float ReduceAddF32x4(__m128 x) {
  // __m128 shuf = _mm_movehdup_ps(x);  // broadcast elements 3,1 to 2,0
  // __m128 sums = _mm_add_ps(x, shuf);
  // shuf = _mm_movehl_ps(shuf, sums);  // high half -> low half
  // sums = _mm_add_ss(sums, shuf);
  // return _mm_cvtss_f32(sums);

  __m128 h64 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
  __m128 sum64 = _mm_add_ps(h64, x);
  __m128 h32 = _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(0, 1, 2, 3));
  __m128 sum32 = _mm_add_ps(sum64, h32);
  return _mm_cvtss_f32(sum32);
}

/**
 * Calculates the sum of the four 32-bit integers in the given __m128i vector.
 *
 * @param x The input __m128i vector containing four 32-bit integers.
 * @return The sum of the four 32-bit integers.
 */
inline int32_t ReduceAddI32x4(__m128i x) {
  // __m128i hi64 = _mm_unpackhi_epi64(x, x);
  __m128i hi64 = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
  __m128i sum64 = _mm_add_epi32(hi64, x);
  __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(0, 1, 2, 3));
  __m128i sum32 = _mm_add_epi32(sum64, hi32);
  return _mm_cvtsi128_si32(sum32);
}

/**
 * Calculates the sum of all elements in a 128-bit vector of unsigned 8-bit
 * integers.
 *
 * @param x The input vector of unsigned 8-bit integers.
 * @return The sum of all elements in the input vector.
 */
inline int32_t ReduceAddI8x16(__m128i x) {
  __m128i vsum = _mm_sad_epu8(x, _mm_setzero_si128());
  return _mm_cvtsi128_si32(vsum) + _mm_extract_epi16(vsum, 4);
}

// inline int32_t ReduceAddI16x8(__m128i x) {
//   __m128i hi64 = _mm_unpackhi_epi64(x, x);
//   __m128i sum64 = _mm_add_epi16(hi64, x);
//   __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(0, 1, 2, 3));
//   __m128i sum32 = _mm_add_epi16(sum64, hi32);
//   __m128i hi16 = _mm_shufflelo_epi16(sum32, _MM_SHUFFLE(1, 0, 3, 2));
//   __m128i sum16 = _mm_add_epi16(sum32, hi16);
//   return _mm_cvtsi128_si32(sum16);
// }

// inline int32_t ReduceAddI8x16(__m128i x) {

// }

/**
 * Calculates the squared L2 distance between two float arrays using SSE
 * instructions.
 *
 * @param kX Pointer to the first float array.
 * @param kY Pointer to the second float array.
 * @param dim The dimension of the arrays.
 * @return The squared L2 distance between the two arrays.
 */
inline float L2SqrFloatSSE(const float* kX, const float* kY, int dim) {
  __m128 sum128 = _mm_setzero_ps();
  __m128 mx128, my128, diff128;

  while (dim >= 4) {
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

inline float AlignL2SqrFloatSSE(const float* kX, const float* kY, int dim) {
  __m128 sum128 = _mm_setzero_ps();

  const float* kEnd = kX + dim;

  while (kX < kEnd) {
    __m128 x128 = _mm_loadu_ps(kX);
    kX += 4;
    __m128 y128 = _mm_loadu_ps(kY);
    kY += 4;
    __m128 diff128 = _mm_sub_ps(x128, y128);
    sum128 = _mm_fmadd_ps(diff128, diff128, sum128);
  }
  return ReduceAddF32x4(sum128);
}

/**
 * Calculates the inner product of two float arrays using SSE instructions.
 *
 * @param kX The first float array.
 * @param kY The second float array.
 * @param dim The dimension of the arrays.
 * @return The inner product of the two arrays.
 */
inline float InnerProductFloatSSE(const float* kX, const float* kY, int dim) {
  __m128 sum128 = _mm_setzero_ps();
  __m128 x128, y128;

  while (dim >= 4) {
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

inline float AlignInnerProductFloatSSE(const float* kX, const float* kY, int dim) {
  __m128 sum128 = _mm_setzero_ps();

  const float* kEnd = kX + dim;

  while (kX < kEnd) {
    __m128 x128 = _mm_loadu_ps(kX);
    kX += 4;
    __m128 y128 = _mm_loadu_ps(kY);
    kY += 4;
    sum128 = _mm_fmadd_ps(x128, y128, sum128);
  }
  return ReduceAddF32x4(sum128);
}

/**
 * Calculates the squared Euclidean norm of a float array using SSE
 * instructions.
 *
 * @param pV Pointer to the float array.
 * @param dim The dimension of the array.
 * @return The squared Euclidean norm of the array.
 */
inline float NormSqrFloatSSE(const float* pV, int dim) {
  __m128 res128 = _mm_setzero_ps();
  __m128 x128;

  while (dim >= 4) {
    x128 = _mm_loadu_ps(pV);
    pV += 4;
    res128 = _mm_fmadd_ps(x128, x128, res128);
    dim -= 4;
  }

  if (dim > 0) {
    x128 = MaskedReadFloat(dim, pV);
    res128 = _mm_fmadd_ps(x128, x128, res128);
  }
  return ReduceAddF32x4(res128);
}

/**
 * Calculates the squared Euclidean norm of a float array using SSE
 * instructions.
 *
 * @param pV Pointer to the float array.
 * @param dim The dimension of the array.
 * @return The squared Euclidean norm of the array.
 */
inline float NormFloatSSE(const float* pV, int dim) { return std::sqrt(NormSqrFloatSSE(pV, dim)); }

/**
 * Calculates the squared Euclidean norm of a align float array using SSE
 * instructions.
 *
 * @param pV Pointer to the float array.
 * @param dim The dimension of the array.
 * @return The squared Euclidean norm of the array.
 */
inline float AlignNormSqrFloatSSE(const float* pV, int dim) {
  __m128 res128 = _mm_setzero_ps();
  __m128 x128;

  const float* pEnd = pV + dim;
  while (pV < pEnd) {
    x128 = _mm_loadu_ps(pV);
    res128 = _mm_fmadd_ps(x128, x128, res128);
    pV += 4;
  }
  return ReduceAddF32x4(res128);
}

inline float AlignNormFloatSSE(const float* pV, int dim) {
  return std::sqrt(AlignNormSqrFloatSSE(pV, dim));
}

}  // namespace alaya
#endif  // defined(USE_SSE)
