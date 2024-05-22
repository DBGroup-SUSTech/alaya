#include <alaya/index/quantizer/product_quantizer.h>
#include <alaya/utils/io_utils.h>
#include <alaya/utils/metric_type.h>
#include <gtest/gtest.h>

#include <iostream>
#include <algorithm>
#include <iterator>

#include "alaya/utils/random_utils.h"
#include "gtest/gtest.h"

#include "alaya/index/graph/hnsw.h"
#include "alaya/index/graph/nsg.h"
#include "alaya/index/quantizer/normal_quantizer.h"
#include "alaya/searcher/graph_search.h"

TEST(GraphTest, BuildHNSW_float) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::AlignLoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);

  std::string metric_l2 = "L2";
  alaya::HNSW<unsigned, float> index(d_dim, metric_l2);

  index.BuildIndex(d_dim, data);

  delete[] data;
}

// todo: 建图时只使用float进行距离计算，需要提供将其他数据类型转成float的函数
//TEST(GraphTest, BuildHNSW_uint8_t) {
//  std::string netflix_path = "/dataset/netflix";
//  unsigned d_num, d_dim, q_num, q_dim;
//  uint8_t *data, *query;
//  alaya::AlignLoadVecsDataset<uint8_t >(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);
//
//  std::string metric_l2 = "L2";
//  alaya::HNSW<unsigned, uint8_t> index(d_dim, metric_l2);
//
//  index.BuildIndex(d_dim, data);
//
//  delete[] data;
//}

TEST(GraphTest, BuildNSG) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::AlignLoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);

  std::string metric_l2 = "L2";
  alaya::NSG<unsigned, float> index(d_dim, metric_l2);
  index.BuildIndex(d_dim, data);

  delete[] data;
}

//TEST(GraphTest, BuildNSG_uint8_t) {
//  std::string sift_path = "";
//  unsigned d_num, d_dim, q_num, q_dim;
//  uint8_t *data, *query;
//  alaya::AlignLoadVecsDataset<uint8_t >(sift_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);
//
//
//  std::string metric_l2 = "L2";
//  alaya::NSG<unsigned, uint8_t > index(d_dim, metric_l2);
//  index.BuildIndex(d_dim, data);
//
//  delete[] data;
//}

TEST(GraphTest, SearchHNSW) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::AlignLoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);

  // todo: index的模板列表为DataType IDType， 而NSG的模板列表为IDType DataType，后续应该统一下
  std::string metric_l2 = "L2";
  std::unique_ptr<alaya::Index<float, unsigned >> index = std::make_unique<alaya::NSG<unsigned, float>>(d_dim, metric_l2);
  index->BuildIndex(d_dim, data);

  std::unique_ptr<alaya::Quantizer<8, float, unsigned>> quant = std::make_unique<alaya::NormalQuantizer<8, float, unsigned>>(d_dim, alaya::MetricType::L2);
  quant->BuildIndex(d_num, data);

  alaya::GraphSearch<alaya::Index<float, unsigned >, alaya::Quantizer<8, float, unsigned>, unsigned, float> searcher(index.get(), quant.get());




}