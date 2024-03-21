#pragma once
#include <alaya/index/bucket/ivf.h>

void test() {
  alaya::InvertedList<int, float>* ivf =
      new alaya::InvertedList<int, float>(10, alaya::MetricType::L2, 128, 10);
  ivf->centroids_data_;
}