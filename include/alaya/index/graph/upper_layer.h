#pragma once

#include <vector>
namespace alaya {

template<typename IDType>
struct UpperLayer {
  int nb, M;
  int ep_;
  std::vector<IDType> levels_;
  std::vector<std::vector<IDType>> list_;

  UpperLayer() = default;
  UpperLayer(int vec_num, int M):levels_(vec_num), list_(vec_num), nb(vec_num), M(M) {}

  IDType at(int level, int u, int i) const {
    return list_[u][(level - 1) * M + i];
  }

  IDType &at(int level, int u, int i) { return list_[u][(level - 1) * M + i]; }

  const IDType *edges(int level, int u) const {
    return list_[u].data() + (level - 1) * M;
  }

  IDType *edges(int level, int u) { return list_[u].data() + (level - 1) * M; }
};

} // namespace alaya