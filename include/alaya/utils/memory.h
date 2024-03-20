#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstring>
#include <stdio.h>
#include <sys/mman.h>

#include "platform_macros.h"

#if defined(__SSE2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace alaya {

ALWAYS_INLINE
inline void PrefetchL1(const void* address) {
#if defined(__SSE2__)
  _mm_prefetch((const char*)address, _MM_HINT_T0);
#else
  __builtin_prefetch(address, 0, 3);
#endif
}

ALWAYS_INLINE
inline void PrefetchL2(const void* address) {
#if defined(__SSE2__)
  _mm_prefetch((const char*)address, _MM_HINT_T1);
#else
  __builtin_prefetch(address, 0, 2);
#endif
}

ALWAYS_INLINE
inline void PrefetchL3(const void* address) {
#if defined(__SSE2__)
  _mm_prefetch((const char*)address, _MM_HINT_T2);
#else
  __builtin_prefetch(address, 0, 1);
#endif
}

ALWAYS_INLINE
inline void MemPrefetch(char* ptr, const int num_lines) {
  switch (num_lines) {
    default:
      [[fallthrough]];
    case 28:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 27:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 26:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 25:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 24:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 23:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 22:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 21:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 20:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 19:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 18:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 17:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 16:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 15:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 14:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 13:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 12:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 11:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 10:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 9:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 7:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 6:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 5:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 4:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 3:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 2:
      PrefetchL1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 1:
      PrefetchL1(ptr);
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
ALWAYS_INLINE
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
ALWAYS_INLINE
inline void* Alloc64B(std::size_t num_bytes) {
  std::size_t len = (num_bytes + (1 << 6) - 1) >> 6 << 6;
  auto ptr = std::aligned_alloc(64, len);
  std::memset(ptr, 0, len);
  return ptr;
}

}  // namespace alaya

namespace glass {

template <typename T> struct align_alloc {
  T *ptr = nullptr;
  using value_type = T;
  T *allocate(int n) {
    if (n <= 1 << 14) {
      int sz = (n * sizeof(T) + 63) >> 6 << 6;
      return ptr = (T *)std::aligned_alloc(64, sz);
    }
    int sz = (n * sizeof(T) + (1 << 21) - 1) >> 21 << 21;
    ptr = (T *)std::aligned_alloc(1 << 21, sz);
    madvise(ptr, sz, MADV_HUGEPAGE);
    return ptr;
  }
  void deallocate(T *, int) { free(ptr); }
  template <typename U> struct rebind {
    typedef align_alloc<U> other;
  };
  bool operator!=(const align_alloc &rhs) { return ptr != rhs.ptr; }
};

inline void *alloc2M(size_t nbytes) {
  size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
  auto p = std::aligned_alloc(1 << 21, len);
  std::memset(p, 0, len);
  return p;
}

inline void *alloc64B(size_t nbytes) {
  size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
  auto p = std::aligned_alloc(1 << 6, len);
  std::memset(p, 0, len);
  return p;
}

} // namespace glass