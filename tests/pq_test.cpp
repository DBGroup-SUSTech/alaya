#include <alaya/index/quantizer/product_quantizer.h>
#include <alaya/utils/io_utils.h>
#include <alaya/utils/metric_type.h>
#include <alaya/utils/random_utils.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>

TEST(ProductQuantizerTest, Constructor) {
  unsigned d_num = 1000;
  unsigned d_dim = 300;

  alaya::ProductQuantizer<8> pq(d_dim, alaya::MetricType::L2, 8);

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

  EXPECT_EQ(8, pq.codebook_.size());
}

TEST(ProductQuantizerTest, BuildIndex) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  // alaya::LoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);
  alaya::AlignLoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);

  alaya::ProductQuantizer<8> pq(d_dim, alaya::MetricType::L2, 8);

  pq.BuildIndex(d_num, data);

  pq.Encode();

  int vec_num = 10;
  for (auto i = 0; i < vec_num; ++i) {
    float err = 0;
    auto vid = alaya::GenRandInt(0, d_num - 1);
    for (auto d = 0; d < d_dim; ++d) {
      err = std::fabs(data[vid * d_dim + d] - pq.encode_vecs_[vid * d_dim + d]);
    }
    fmt::println("vid: {}, err: {}", vid, err);
    for (auto d = 0; d < d_dim; ++d) {
      fmt::print("dim: {}, data: {:.2f}, code: {:.2f} ", d, data[vid * d_dim + d],
                 pq.encode_vecs_[vid * d_dim + d]);
    }
    fmt::println("\n");
  }

  delete[] data;
}