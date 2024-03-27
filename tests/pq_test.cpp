#include <alaya/index/quantizer/product_quantizer.h>
#include <alaya/utils/io_utils.h>
#include <alaya/utils/metric_type.h>
#include <gtest/gtest.h>

TEST(ProductQuantizerTest, Constructor) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);

  alaya::ProductQuantizer<8> pq(d_dim, d_num, alaya::MetricType::L2, 8);
}