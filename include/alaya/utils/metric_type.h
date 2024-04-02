#pragma once

#include <array>
#include <string>
#include <string_view>
#include <tuple>

namespace alaya {

enum class MetricType {
  L2,
  IP,
  COS,
};

struct MetricMap {
  static constexpr std::array<std::tuple<std::string_view, MetricType>, 3> kStaticMap = {
      std::make_tuple("L2", MetricType::L2),
      std::make_tuple("IP", MetricType::IP),
      std::make_tuple("COS", MetricType::COS),
  };

  // static consteval MetricType operator[](std::string_view str) {
  //   for (const auto& [key, val] : kStaticMap) {
  //     if (key == str) {
  //       return val;
  //     }
  //   }
  //   __builtin_unreachable();
  // }

  static constexpr MetricType operator[](std::string_view str) {
    for (const auto& [key, val] : kStaticMap) {
      if (key == str) {
        return val;
      }
    }
    __builtin_unreachable();
  }

};

inline constexpr MetricMap kMetricMap{};

// static_assert(kMetricMap["L2"] == MetricType::L2);

// inline constexpr static std::unordered_map<std::string, MetricType> kMetricMap = {
//     {"L2", MetricType::L2},
//     {"IP", MetricType::IP},
//     {"COS", MetricType::COS},
// };

}  // namespace alaya