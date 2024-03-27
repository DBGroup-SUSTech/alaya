#include <alaya/index/quantizer/product_quantizer.h>
#include <alaya/utils/io_utils.h>
#include <alaya/utils/metric_type.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>

#include "gtest/gtest.h"

TEST(ProductQuantizerTest, Constructor) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);

  alaya::ProductQuantizer<8> pq(d_dim, d_num, alaya::MetricType::L2, 8);

  pq.BuildIndex(d_num, data);

  EXPECT_EQ(8, pq.book_num_);
  EXPECT_EQ(8, pq.sub_dimensions_.size());
  unsigned sub_dim[] = {37, 37, 37, 37, 37, 37, 37, 41};
  // EXPECT_EQ(sub_dim, pq.sub_dimensions_.data());
  EXPECT_TRUE(
      std::equal(pq.sub_dimensions_.begin(), pq.sub_dimensions_.end(), std::begin(sub_dim)));

  EXPECT_EQ(8, pq.sub_vec_start_.size());
  unsigned sub_start[] = {0, 37, 74, 111, 148, 185, 222, 259};
  EXPECT_TRUE(
      std::equal(pq.sub_vec_start_.begin(), pq.sub_vec_start_.end(), std::begin(sub_start)));

  EXPECT_EQ(256, pq.kBookSize_);
  EXPECT_EQ(8 * 256, pq.dist_line_size_);
}
