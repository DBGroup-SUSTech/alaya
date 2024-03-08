#pragma once

#include <type_traits>
#include <utility>
#include <concepts>

#include "alaya_assert.h"

namespace alaya {

template <unsigned N>
auto dependent_func() {
    if constexpr (N <= 8) {
        return std::declval<uint8_t>();
    } else if constexpr (N <= 16) {
        return std::declval<uint16_t>();
    } else if constexpr (N <= 32) {
        return std::declval<uint32_t>();
    } else {
        return std::declval<uint64_t>();
    }
}

template <unsigned N>
struct dependent_type {
    using type = decltype(dependent_func<N>());
};

template <unsigned N>
using dependent_type_t = typename dependent_type<N>::type;

constexpr unsigned GetMaxIntegral(unsigned BitsNum) {
  return (1 << BitsNum) - 1;
}

// template <unsigned N>
// struct A {
//     using T = dependent_type_t<N>;
// };



}// namespace alaya