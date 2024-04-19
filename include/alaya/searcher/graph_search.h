#pragma once

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "searcher.h"
#include "../utils/pool.h"



namespace alaya {

template <typename IndexType, typename QuantizerType, typename IDType, typename DataType>
struct GraphSearch : Searcher<IndexType, QuantizerType, DataType> {
  int ef = 32;

  explicit GraphSearch(const IndexType* index, const QuantizerType* quantizer, int ef = 32)
      : Searcher<IndexType, QuantizerType, DataType>(index, quantizer), ef(ef) {}

  ~GraphSearch() = default;

  // todo:
  void Optimize(int num_threads = 0) override {}

//  void SetEf(int ef) { this->ef = ef; }



  void Search(int64_t query_num, int64_t query_dim, const DataType* queries, int64_t k,
                   DataType* distances, int64_t* labels) const {
    static auto init_nsg_eps = [](auto& graph, auto& computer, auto& pool) {
      for (auto ep : graph.eps_) {
        pool.insert(ep, computer(ep));
      }
    };

    static auto init_hnsw_eps = [](auto& graph, auto& computer, auto& pool) {
      auto& sub_graph = graph->upper_layer;
      int u = sub_graph->ep;
      auto cur_dist = computer(u);
      for (int level = sub_graph->levels[u]; level > 0; --level) {
        bool changed = true;
        const IDType* list = sub_graph->edges(level, u);
        for (int i = 0; i < graph.edges_nums_; i++) {
          IDType v = list[i];
          auto dist = computer(v);
          if (dist < cur_dist) {
            cur_dist = dist;
            u = v;
            changed = true;
          }
        }
      }
      pool.insert(u, cur_dist);
      //    pool.vis.set(u);
    };
    assert(this->index_->vec_dim_ == query_dim);
    for(int i = 0; i < query_num; i++) {
      LinearPool<DataType, IDType> pool(std::max(k, (int64_t)ef));
      auto init_eps = this->index_->upper_layer_ == nullptr ? init_nsg_eps : init_hnsw_eps;
      init_eps(*(this->index_), *(this->quantizer_), pool);
      SearchImpl(pool);
      for(int j = 0; j < k; j++) {
        distances[i * k + j] = pool.pool_[j]->dis_;
        labels[i * k + j] = pool.pool_[j]->id_;
      }
    }
  }

  // todo: prefetch
  void SearchImpl(LinearPool<DataType, IDType>& pool, const DataType* query) const {
    auto& computer = *(this->quantizer_);
    auto& graph = *(this->index_);
    while (pool.has_next()) {
      auto u = pool.pop();
      //      graph.prefetch(u, graph_po);
      //      for (int i = 0; i < po; ++i) {
      //        int to = graph.at(u, i);
      //        computer.prefetch(to, pl);
      //      }
      for (int i = 0; i < graph.K; ++i) {
        int v = graph.at(u, i);
        if (v == -1) {
          break;
        }
        //        if (i + po < graph.K && graph.at(u, i + po) != -1) {
        //          int to = graph.at(u, i + po);
        //          computer.prefetch(to, pl);
        //        }
        if (pool.vis.get(v)) {
          continue;
        }
        pool.vis.set(v);
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
      }
    }
  }
};

} // namespace alaya