//
// Created by weijian on 3/20/24.
//
#include "alaya/index/graph/nsg/nsg.hpp"
#include "alaya/index/quantizer/fp32_quant.hpp"
#include "alaya/searcher/graph/searcher.hpp"
#include <iostream>
#include <memory>

std::unique_ptr<float[]> fill_array_with_random_numbers(size_t n, size_t d) {
  auto dataset = std::make_unique<float[]>(n * d);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < n * d; ++i) {
    dataset[i] = dist(gen);
  }
  return dataset;
}

int main() {
  std::cout << "hello alaya" << std::endl;
  int n = 1000, d = 128;
  auto dataset = fill_array_with_random_numbers(n, d);
  int R = 32, L = 100;
  std::string metric = "L2";
  alaya::NSG<glass::FP32Quantizer<glass::Metric::L2>, int, float> index(d,metric,R,L);
  index.BuildIndex(n, dataset.get());
  alaya::GraphSearcher<glass::FP32Quantizer<glass::Metric::L2>, decltype(index), float> searcher;



}