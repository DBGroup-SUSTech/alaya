#pragma once


#include <memory>
#include "index.h
#include "graph/graph.h"
#include "quant/quant.h"
#include "bucket/bucket.h"

namespace azalea {

template <typename IndexType, typename IDType, typename CodeType, typename DistType>
struct Searcher {
  std::unique_ptr<Graph<IDType, DistType>> graph_ = nullptr;
  std::unique_ptr<Quantizer<IDType, CodeType, DistType>> quant_ = nullptr;
  std::unique_ptr<> buckets_ = nullptr;

};

} // namespace azalea