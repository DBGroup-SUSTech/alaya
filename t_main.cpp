//#include <alaya/index/quantizer/product_quantizer.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <xmmintrin.h>

#include <cstdint>
#include <iostream>

#include <algorithm>
#include <memory>
#include <cmath>

//inline int32_t ReduceAddI16x8(__m128i x) {
//  __m128i hi64 = _mm_unpackhi_epi64(x, x);
//  __m128i sum64 = _mm_add_epi16(hi64, x);
//  __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(0, 1, 2, 3));
//  __m128i sum32 = _mm_add_epi16(sum64, hi32);
//  __m128i hi16 = _mm_shufflelo_epi16(sum32, _MM_SHUFFLE(1, 0, 3, 2));
//  __m128i sum16 = _mm_add_epi16(sum32, hi16);
//  return _mm_cvtsi128_si32(sum16);
//}
//
//inline uint32_t _mm_sum_epu8(const __m128i v) {
//  __m128i vsum = _mm_sad_epu8(v, _mm_setzero_si128());
//  return _mm_cvtsi128_si32(vsum) + _mm_extract_epi16(vsum, 4);
//}
//
//inline uint32_t ReduceAddI8x32(const __m256i x) {
//  __m256i vsum = _mm256_sad_epu8(x, _mm256_setzero_si256());
//
//  // __m128i lo = _mm256_castsi256_si128(x);
//  // __m128i hi = _mm256_extractf128_si256(x, 1);
//  // return _mm_sum_epu8(lo) + _mm_sum_epu8(hi);
//  return 0;
//}

int main() {
//  auto pq = alaya::ProductQuantizer<8, int64_t, float>(10, 100, alaya::MetricType::L2, 10);
//  std::cout << "pq book size: " << pq.book_size_ << std::endl;
//  std::cout << "pq vec dim: " << pq.vec_dim_ << std::endl;
//  std::cout << "pq vec num: " << pq.vec_num_ << std::endl;
//
//  __m128 x128 = _mm_set_ps(0, 1, 2, 3);
//  float* x = (float*)&x128;
//  for (int i = 0; i < 4; i++) {
//    std::cout << x[i] << " ";
//  }
//  std::cout << std::endl;
//  __m128 y128 = _mm_shuffle_ps(x128, x128, _MM_SHUFFLE(1, 0, 3, 2));
//  float* y = (float*)&y128;
//  for (int i = 0; i < 4; i++) {
//    std::cout << y[i] << " ";
//  }
//  std::cout << std::endl;
//
//  __m128i i128 =
//      _mm_set_epi8(122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137);
//
//  __m128i ii128 = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
//                               255, 255);  // 0x0f0e0d0c0b0a09080706050403020100
//
//  std::cout << _mm_sum_epu8(i128) << std::endl;
//
//  std::cout << _mm_sum_epu8(ii128) << std::endl;
//
//  __m256i i256 = _mm256_set_epi8(122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
//                                 135, 136, 137, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
//                                 132, 133, 134, 135, 136, 137);
//  __m256i sum256 = _mm256_sad_epu8(i256, _mm256_setzero_si256());
//
//  uint64_t* si = (uint64_t*)&sum256;
//
//  for (int i = 0; i < 4; ++i) {
//    std::cout << (int)si[i] << " ";
//  }
//  std::cout << std::endl;
//
//  __m128i sum128 = _mm_sad_epu8(i128, _mm_setzero_si128());
//  uint64_t* sii = (uint64_t*)&sum128;
//
//  for (int i = 0; i < 2; ++i) {
//    std::cout << (int)sii[i] << " ";
//  }
//  std::cout << std::endl;
//
//  std::cout << _mm_cvtsi128_si32(sum128) << std::endl;

  // std::cout << ReduceAddI16x8(i128) << std::endl;

  // __m128i isum128 = _mm_add_epi8(i128, i128);

  // uint8_t* ii = (uint8_t*)&i128;
  // for (int i = 0; i < 16; ++i) {
  //   std::cout << (int)ii[i] << " ";
  // }
  // std::cout << std::endl;

  // uint16_t* si = (uint16_t*)&isum128;
  // for (int i = 0; i < 8; ++i) {
  //   std::cout << (int)isum128[i] << " ";
  // }
  // std::cout << std::endl;

  unsigned d_num=1, d_dim=128;
  uint8_t *data, *query;
  data = new uint8_t [128]{};
  for(int i=0;i<128;i++){
    data[i] = (uint8_t)(rand()/256);
  }

  std::unique_ptr<float[]> buf = std::make_unique<float[]>(d_num * d_dim);
  float* data_uint8_t = buf.get();
  std::transform(data, data + d_num * d_dim, data_uint8_t,
                 [](uint8_t val) { return static_cast<float>(val); });

  for(int i=0;i<d_dim;i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;

  for(int i=0;i<d_dim;i++) {
    std::cout << static_cast<float>(data[i]) << " ";
  }
  std::cout << std::endl;

  for(int i=0;i<d_dim;i++) {
    std::cout << data_uint8_t[i] << " ";
  }

  delete data;

  return 0;
}