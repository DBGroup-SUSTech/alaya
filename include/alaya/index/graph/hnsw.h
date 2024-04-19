#pragma once

#include "alaya/index/graph/graph.h"
#include "alaya/index/graph/upper_layer.h"
#include "alaya/utils/metric_type.h"
#include "alaya/index/graph/hnswlib/hnswlib.h"
#include "alaya/index/graph/hnswlib/space_ip.h"
#include "alaya/index/graph/hnswlib/space_l2.h"
#include "alaya/index/graph/hnswlib/hnswalg.h"
#include "alaya/utils/memory.h"
#include <chrono>
#include <memory>
#include <sys/mman.h>
#include <cstring>
#include <iostream>


namespace alaya {

template <typename IDType, typename DataType>
struct HNSW : public Graph<IDType, DataType> {
  int M, efConstruction;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw = nullptr;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space = nullptr;

  HNSW(int dim, std::string_view metric, int R = 32, int L = 200)
      : Graph<IDType, DataType>(dim, metric, R), M(R / 2), efConstruction(L) {
    if (this->metric_type_ == MetricType::L2) {
      space = std::make_unique<hnswlib::L2Space>(dim);
    } else if (this->metric_type_ == MetricType::IP) {
      space = std::make_unique<hnswlib::InnerProductSpace>(dim);
    } else {
      printf("Unsupported metric type\n");
    }

  }

  void BuildIndex(IDType vec_num, const DataType* kVecData) override {
    this->vec_num_ = vec_num;
    size_t len = this->vec_num_ * this->edge_num_ * sizeof(IDType);
    this->linklist_ = (IDType*)Alloc2M(len);

    float* data = nullptr;
    std::unique_ptr<float[]> data_buf = nullptr;
    if constexpr (std::is_same_v<DataType, float>) {
      data = const_cast<DataType* >(kVecData);
    } else {
      data_buf = std::make_unique<float[]>(this->vec_num_ * this->vec_dim_);
      data = data_buf.get();
      std::transform(kVecData, kVecData + this->vec_num_ * this->vec_dim_, data,
                     [](const DataType& val) { return static_cast<float>(val); });
    }

    int nb = this->vec_num_;
    hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), nb, M,
                                                             efConstruction);
    std::atomic<int> cnt{0};
    auto st = std::chrono::high_resolution_clock::now();
    hnsw->addPoint(data, 0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < nb; ++i) {
      hnsw->addPoint(data + i * this->vec_dim_, i);
      int cur = cnt += 1;
      if (cur % 10000 == 0) {
        printf("HNSW building progress: [%d/%d]\n", cur, nb);
      }
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    printf("HNSW building cost: %.2lfs\n", ela);
//    final_graph.init(nb, 2 * M);
#pragma omp parallel for
    for (int i = 0; i < nb; ++i) {
      int *edges = (int *)hnsw->get_linklist0(i);
      for (int j = 1; j <= edges[0]; ++j) {
//        final_graph.at(i, j - 1) = edges[j];
          this->linklist_[i * this->edge_num_ + (j - 1)] = edges[j];
      }
    }

    this->upper_layer_ = std::make_unique<UpperLayer<IDType>>(nb, M);
    this->upper_layer_->ep_ = hnsw->enterpoint_node_;
    for (int i = 0; i < nb; ++i) {
      int level = hnsw->element_levels_[i];
      this->upper_layer_->levels_[i] = level;
      if (level > 0) {
        this->upper_layer_->list_[i].assign(level * M, -1);
        for (int j = 1; j <= level; ++j) {
          int *edges = (int *)hnsw->get_linklist(i, j);
          for (int k = 1; k <= edges[0]; ++k) {
//            this->upper_layer_->at(j, i, k - 1) = edges[k];
            this->upper_layer_->at(j, i, k - 1) = edges[k];
          }
        }
      }
    }
//    final_graph.initializer = std::move(initializer);
  }

  // todo: to be defined
  void Save(const char* kFilePath) const override {

  }

  void Load(const char* kFilePath) override {

  }



};

}  // namespace alaya