#pragma once

#include "graph.hpp"

namespace glass {

struct Builder {
  virtual void Build(float *data, int nb) = 0;
  virtual std::unique_ptr<Graph<int>> GetGraph() = 0;
  virtual ~Builder() = default;
};

} // namespace glass