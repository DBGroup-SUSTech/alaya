#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "alaya_assert.h"

namespace alaya {

/**
 * @brief Returns a dependent type based on the value of template parameter N.
 * If N is less than or equal to 8, returns uint8_t.
 * If N is less than or equal to 16, returns uint16_t.
 * If N is less than or equal to 32, returns uint32_t.
 * Otherwise, returns uint64_t.
 *
 * @tparam N The value used to determine the return type.
 * @return The dependent type based on the value of N.
 */
template <unsigned N>
auto DependentFunc() {
  if constexpr (N <= 8) {
    // return std::declval<uint8_t>();
    return static_cast<uint8_t>(0);
  } else if constexpr (N <= 16) {
    // return std::declval<uint16_t>();
    return static_cast<uint16_t>(0);
  } else if constexpr (N <= 32) {
    // return std::declval<uint32_t>();
    return static_cast<uint32_t>(0);
  } else {
    // return std::declval<uint64_t>();
    return static_cast<uint64_t>(0);
  }
}

/**
 * @brief A template struct that provides a dependent type based on the result of a dependent
 * function.
 *
 * @tparam N The value used to invoke the dependent function.
 */
template <unsigned N>
struct DependentType {
  using type = decltype(DependentFunc<N>());
};

/**
 * @brief A type alias template that represents the type of dependent bits.
 *
 * This template alias is used to define the type of dependent bits based on the value of N.
 * It relies on the `DependentType` template to determine the type.
 *
 * @tparam N The number of dependent bits.
 */
template <unsigned N>
using DependentBitsType = typename DependentType<N>::type;

/**
 * @brief Calculates the maximum value that can be represented by an unsigned integral type with a
 * given number of bits.
 *
 * @param BitsNum The number of bits in the unsigned integral type.
 * @return The maximum value that can be represented by the unsigned integral type.
 */
constexpr unsigned GetMaxIntegral(unsigned BitsNum) { return (1 << BitsNum); }

}  // namespace alaya