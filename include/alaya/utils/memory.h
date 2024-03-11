#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#if defined(__SSE2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace alaya {

#define ALWAYS_INLINE __attribute__((always_inline))

ALWAYS_INLINE inline void prefetch_L1(const void* address) {
#if defined(__SSE2__)
  _mm_prefetch((const char*)address, _MM_HINT_T0);
#else
  __builtin_prefetch(address, 0, 3);
#endif
}

ALWAYS_INLINE inline void prefetch_L2(const void* address) {
#if defined(__SSE2__)
  _mm_prefetch((const char*)address, _MM_HINT_T1);
#else
  __builtin_prefetch(address, 0, 2);
#endif
}

ALWAYS_INLINE inline void prefetch_L3(const void* address) {
#if defined(__SSE2__)
  _mm_prefetch((const char*)address, _MM_HINT_T2);
#else
  __builtin_prefetch(address, 0, 1);
#endif
}

inline void mem_prefetch(char* ptr, const int num_lines) {
  switch (num_lines) {
    default:
      [[fallthrough]];
    case 28:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 27:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 26:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 25:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 24:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 23:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 22:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 21:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 20:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 19:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 18:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 17:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 16:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 15:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 14:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 13:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 12:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 11:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 10:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 9:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 8:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 7:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 6:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 5:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 4:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 3:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 2:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 1:
      prefetch_L1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 0:
      break;
  }
}

/**
 * @brief Allocates a block of memory of the specified size in bytes.
 *
 * This function allocates a block of memory of the specified size in bytes.
 * The allocated memory is aligned to a 2MB boundary.
 *
 * @param num_bytes The size of the memory block to allocate, in bytes.
 * @return A pointer to the allocated memory block, or nullptr if the allocation fails.
 */
inline void* Alloc2M(std::size_t num_bytes) {
  std::size_t len = (num_bytes + (1 << 21) - 1) >> 21 << 21;
  auto ptr = std::aligned_alloc(1 << 21, len);
  std::memset(ptr, 0, len);
  return ptr;
}

/**
 * @brief Allocates memory of size `num_bytes` aligned to 64 bytes.
 *
 * @param num_bytes The number of bytes to allocate.
 * @return A pointer to the allocated memory.
 */
inline void* Alloc64B(std::size_t num_bytes) {
  std::size_t len = (num_bytes + (1 << 6) - 1) >> 6 << 6;
  auto ptr = std::aligned_alloc(64, len);
  std::memset(ptr, 0, len);
  return ptr;
}

}  // namespace alaya