//
// Created by weijian on 3/20/24.
//
#include <iostream>
#include <memory>

#include "alaya/index/graph/nsg/nsg.hpp"
#include "alaya/searcher/graph_searcher.hpp"

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


}