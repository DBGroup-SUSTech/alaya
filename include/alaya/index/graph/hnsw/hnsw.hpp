#pragma once

#include <chrono>
#include <memory>

#include "alaya/index/graph/builder.hpp"
#include "alaya/index/graph/hnswlib/hnswalg.h"
#include "alaya/index/graph/hnswlib/hnswlib.h"
#include "alaya/index/graph/hnswlib/space_ip.h"
#include "alaya/index/graph/hnswlib/space_l2.h"
#include "HNSWInitializer.hpp"
#include "alaya/index/graph/graph.hpp"
#include "alaya/utils/common.hpp"
#include "alaya/index/index.h"

namespace glass {

struct HNSW : public Builder {
  int nb, dim;
  int M, efConstruction;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw = nullptr;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space = nullptr;

  std::unique_ptr<Graph<int>> final_graph;

  HNSW(int dim, const std::string &metric, int R = 32, int L = 200)
      : dim(dim), M(R / 2), efConstruction(L) {
    auto m = metric_map[metric];
    if (m == Metric::L2) {
      space = std::make_unique<hnswlib::L2Space>(dim);
    } else if (m == Metric::IP) {
      space = std::make_unique<hnswlib::InnerProductSpace>(dim);
    } else {
      printf("Unsupported metric type\n");
    }
  }

  void Build(float *data, int N) override {
    nb = N;
    hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), N, M,
                                                             efConstruction);
    std::atomic<int> cnt{0};
    auto st = std::chrono::high_resolution_clock::now();
    hnsw->addPoint(data, 0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < nb; ++i) {
      hnsw->addPoint(data + i * dim, i);
      int cur = cnt += 1;
      if (cur % 10000 == 0) {
        printf("HNSW building progress: [%d/%d]\n", cur, nb);
      }
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    printf("HNSW building cost: %.2lfs\n", ela);
    final_graph->init(nb, 2 * M);
#pragma omp parallel for
    for (int i = 0; i < nb; ++i) {
      int *edges = (int *)hnsw->get_linklist0(i);
      for (int j = 1; j <= edges[0]; ++j) {
        final_graph->at(i, j - 1) = edges[j];
      }
    }
    auto initializer = std::make_unique<HNSWInitializer>(nb, M);
    initializer->ep = hnsw->enterpoint_node_;
    for (int i = 0; i < nb; ++i) {
      int level = hnsw->element_levels_[i];
      initializer->levels[i] = level;
      if (level > 0) {
        initializer->lists[i].assign(level * M, -1);
        for (int j = 1; j <= level; ++j) {
          int *edges = (int *)hnsw->get_linklist(i, j);
          for (int k = 1; k <= edges[0]; ++k) {
            initializer->at(j, i, k - 1) = edges[k];
          }
        }
      }
    }
    final_graph->initializer = std::move(initializer);
  }

  std::unique_ptr<Graph<int>> GetGraph() override {
    auto res = std::move(final_graph);
    final_graph = nullptr;
    return res;
  }
};
} // namespace glass

namespace alaya {

template <typename Quantizer, typename IDType = int64_t, typename DataType = float>
struct HNSWimp : public Index<IDType,DataType> {
  std::unique_ptr<glass::Graph<IDType>> graph = std::make_unique<glass::Graph<IDType>>();
  std::unique_ptr<Quantizer> quant = std::make_unique<Quantizer>();
  std::unique_ptr<glass::HNSW> builder = nullptr;

  explicit HNSWimp(int dim, const std::string& metric, int R = 32, int L = 200):Index<IDType, DataType>(dim, 0, metric), builder(std::make_unique<glass::HNSW>(dim,metric,R,L)){};

  void BuildIndex(IDType vec_num, const DataType* kVecData) override {
    this->vec_num_ = vec_num;
    builder->Build(kVecData, vec_num);
    graph = builder->GetGraph();

    quant = std::make_unique<Quantizer>(this->vec_dim_,this->vec_num_,this->metric_type_, 0);
    quant->BuildIndex(this->vec_num_, kVecData);
  }

  void Save(const char* kFilePath) const override {
    graph->save(std::string(kFilePath));
    quant->Save(kFilePath);
  }

  void Load(const char* kFilePath) override {
    graph->load(std::string(kFilePath));
    quant->Load(kFilePath);
  }

};

}  // namespace alaya
