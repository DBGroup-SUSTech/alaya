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

#include "alaya/index/graph/graph.hpp"
#include "alaya/utils/common.hpp"
#include "alaya/utils/neighbor.hpp"
#include "alaya/utils/utils.hpp"
#include "alaya/index/quantizer/quantizer.h"
#include "alaya/searcher/searcher.h"

namespace alaya {

// todo: The distance calculation should be provided by the quantizer class
template <typename Quantizer, typename DataType = float>
struct Computer {
  std::unique_ptr<DataType[]> q;
  const std::unique_ptr<Quantizer>& quant;
  Computer(const std::unique_ptr<Quantizer>& quant, const DataType *query)
  :quant(quant){
    std::memcpy(q.get(),query,quant->vec_dim_);
  }
  DataType operator()(int u)const {
    return dist_func(q.get(),quant->Decode(u),quant->vec_dim_);
  }
  void prefetch(int u, int lines) const {
    mem_prefetch(quant->Decode(u), lines);
  }
};

template <typename Pool>
void reorder(const Pool &pool, const float *, int *dst, int k){
  for (int i = 0; i < k; ++i) {
    dst[i] = pool.id(i);
  }
}

template <typename Quantizer, typename IndexType, typename DataType = float>
struct GraphSearcher: public Searcher<IndexType, DataType> {
//    glass::Graph<IndexType>& graph;
//    Quantizer& quant;
    int nb;
    int d;

    // Search parameters
    int ef = 32;

    // Memory prefetch parameters
    int po = 1;
    int pl = 1;

    // Optimization parameters
    constexpr static int kOptimizePoints = 1000;
    constexpr static int kTryPos = 10;
    constexpr static int kTryPls = 5;
    constexpr static int kTryK = 10;
    int sample_points_num;
    std::vector<float> optimize_queries;
    const int graph_po;

//    GraphSearcher(const IndexType& index):graph(index.builder.GetGraph),quant(index.quant) {
//      d = index.vec_dim_;
//    }

    void SetIndex(const IndexType& index) override {
      this->index_ = std::make_unique<IndexType>(std::move(index));
    }

    void SetEf(int ef) override {
      this->ef = ef;
    }

    void Optimize(int num_threads = 0) override {
      sample_points_num = std::min(kOptimizePoints, nb - 1);
      std::vector<int> sample_points(sample_points_num);
      std::mt19937 rng;
      glass::GenRandom(rng, sample_points.data(), sample_points_num, nb);
      optimize_queries.resize(sample_points_num * d);
      for (int i = 0; i < sample_points_num; ++i) {
        // todo: free memory
        memcpy(optimize_queries.data() + i * d, this->index_.Decode(i),
               d * sizeof(float));
      }

      if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
      }
      std::vector<int> try_pos(std::min(kTryPos, this->index_->graph.K));
      std::vector<int> try_pls(
          std::min(kTryPls, (int)upper_div(this->index_->quant->code_size, 64)));
      std::iota(try_pos.begin(), try_pos.end(), 1);
      std::iota(try_pls.begin(), try_pls.end(), 1);
      std::vector<int> dummy_dst(kTryK);
      printf("=============Start optimization=============\n");
      { // warmup
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (int i = 0; i < sample_points_num; ++i) {
          Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
        }
      }

      float min_ela = std::numeric_limits<float>::max();
      int best_po = 0, best_pl = 0;
      for (auto try_po : try_pos) {
        for (auto try_pl : try_pls) {
          this->po = try_po;
          this->pl = try_pl;
          auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
          for (int i = 0; i < sample_points_num; ++i) {
            Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
          }

          auto ed = std::chrono::high_resolution_clock::now();
          auto ela = std::chrono::duration<double>(ed - st).count();
          if (ela < min_ela) {
            min_ela = ela;
            best_po = try_po;
            best_pl = try_pl;
          }
        }
      }
      this->po = 1;
      this->pl = 1;
      auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
      for (int i = 0; i < sample_points_num; ++i) {
        Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
      }
      auto ed = std::chrono::high_resolution_clock::now();
      float baseline_ela = std::chrono::duration<double>(ed - st).count();
      printf("settint best po = %d, best pl = %d\n"
          "gaining %.2f%% performance improvement\n============="
          "Done optimization=============\n",
          best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1));
      this->po = best_po;
      this->pl = best_pl;
    }

    void Search(const float *q, int k, int *dst) const override {
      glass::searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type>
          pool(nb, std::max(k, ef), k);
      Computer<Quantizer, DataType> computer(this->index_->quant, q);
      this->index_->graph->initialize_search(pool, computer);
      SearchImpl(pool, computer);
      reorder(pool, q, dst, k);
    }

    template <typename Pool, typename Computer>
    void SearchImpl(Pool &pool, const Computer &computer) const {
      auto& graph = this->index_->graph;
      auto& quant = this->index_->quant;
      while (pool.has_next()) {
        auto u = pool.pop();
        graph.prefetch(u, graph_po);
        for (int i = 0; i < po; ++i) {
          int to = graph.at(u, i);
          computer.prefetch(to, pl);
        }
        for (int i = 0; i < graph.K; ++i) {
          int v = graph.at(u, i);
          if (v == -1) {
            break;
          }
          if (i + po < graph.K && graph.at(u, i + po) != -1) {
            int to = graph.at(u, i + po);
            computer.prefetch(to, pl);
          }
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


}  // namespace alaya

