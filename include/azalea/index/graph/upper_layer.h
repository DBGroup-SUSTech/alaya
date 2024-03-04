#pragma once

#include <vector>
namespace azalea {

template<typename IDType>
struct UpperLayer {
  int ep_;
  std::vector<int> levels_;
  std::vector<std::vector<IDType>> list_;

  UpperLayer() = default;
};

} // namespace azalea