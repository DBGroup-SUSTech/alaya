#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>

// Adapted from https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswlib.h
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512F
#endif  // __AVX512F__
#endif  // __AVX__
#endif  // __SSE__

#if defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE [[gnu::always_inline]]
#elif defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#endif

#if defined(__GNUC__)
#define UNROLL_BEGIN          \
  _Pragma("GCC push_options") \
      _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define UNROLL_END _Pragma("GCC pop_options")
#else
#define UNROLL_BEGIN
#define UNROLL_END
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#define ALIGNED(x) __attribute__((aligned(x)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif  // defined(__GNUC__)