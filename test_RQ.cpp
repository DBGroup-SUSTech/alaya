//
// Created by yujun on 3/15/24.
//

#include <cassert>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <alaya/utils/heap.h>
#include <alaya/index/quantizer/residual_quantizer.h>
// #include "../include/index/index.h"


void test_RQ() {
  alaya::ResidualQuantizer RQ(1, 1, alaya::MetricType::L2, 1, 4);
  puts("!");
}
int main() {
  test_RQ();
  return 0;
}