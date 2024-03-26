#pragma once

#include <vector>
namespace alaya {

template<typename IDType>
struct UpperLayer {
  int ep_;
  std::vector<IDType> levels_;
  std::vector<std::vector<IDType>> list_;

  UpperLayer() = default;
};

} // namespace alaya