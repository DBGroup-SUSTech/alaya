#include <alaya/utils/distance.h>
#include <alaya/utils/io_utils.h>
#include <alaya/utils/memory.h>
#include <alaya/utils/metric_type.h>
#include <alaya/utils/random_utils.h>
#include <fmt/core.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

constexpr float kDeltaFloat = 1e-3;

TEST(InnerProductTest, Netflix) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query,
                                q_num, q_dim);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * q_dim;
    float* d_data = data + d_id * d_dim;

    float naive_res = alaya::NaiveIp(q_data, d_data, d_dim);
    float simd_res = alaya::InnerProduct(q_data, d_data, d_dim);
    // fmt::println("naive: {}, simd: {}", naive_res, simd_res);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  delete[] data;
  delete[] query;
}

TEST(InnerProductTest, AlignNetflix) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::AlignLoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim,
                                     query, q_num, q_dim);
  unsigned align_dim = alaya::DoAlign(d_dim, 16);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * align_dim;
    float* d_data = data + d_id * align_dim;

    float naive_res = alaya::NaiveIp(q_data, d_data, d_dim);
    float simd_res = alaya::AlignInnerProduct(q_data, d_data, d_dim);
    // fmt::println("naive: {}, simd: {}", naive_res, simd_res);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  std::free(data);
  std::free(query);
}

TEST(InnerProductTest, Sift1m) {
  std::string sift1m_path = "/dataset/sift1m";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(sift1m_path.c_str(), data, d_num, d_dim, query,
                                q_num, q_dim);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * q_dim;
    float* d_data = data + d_id * d_dim;

    float naive_res = alaya::NaiveIp(q_data, d_data, d_dim);
    float simd_res = alaya::InnerProduct(q_data, d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  delete[] data;
  delete[] query;
}

TEST(InnerProductTest, AlignSift1m) {
  std::string sift1m_path = "/dataset/sift1m";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::AlignLoadVecsDataset<float>(sift1m_path.c_str(), data, d_num, d_dim,
                                     query, q_num, q_dim);
  unsigned align_dim = alaya::DoAlign(d_dim, 16);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * align_dim;
    float* d_data = data + d_id * align_dim;

    float naive_res = alaya::NaiveIp(q_data, d_data, d_dim);
    float simd_res = alaya::AlignInnerProductFloat(q_data, d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  std::free(data);
  std::free(query);
}

TEST(InnerProductTest, Deep1m) {
  std::string deep1m_path = "/dataset/deep1m";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(deep1m_path.c_str(), data, d_num, d_dim, query,
                                q_num, q_dim);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * q_dim;
    float* d_data = data + d_id * d_dim;

    float naive_res = alaya::NaiveIp(q_data, d_data, d_dim);
    float simd_res = alaya::InnerProduct(q_data, d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  delete[] data;
  delete[] query;
}

TEST(InnerProductTest, AlignDeep1m) {
  std::string deep1m_path = "/dataset/deep1m";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::AlignLoadVecsDataset<float>(deep1m_path.c_str(), data, d_num, d_dim,
                                     query, q_num, q_dim);
  unsigned align_dim = alaya::DoAlign(d_dim, 16);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * align_dim;
    float* d_data = data + d_id * align_dim;

    float naive_res = alaya::NaiveIp(q_data, d_data, d_dim);
    float simd_res = alaya::AlignInnerProduct(q_data, d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  std::free(data);
  std::free(query);
}

TEST(InnerProductTest, Gist) {
  std::string gist_path = "/dataset/gist";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(gist_path.c_str(), data, d_num, d_dim, query,
                                q_num, q_dim);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * q_dim;
    float* d_data = data + d_id * d_dim;

    float naive_res = alaya::NaiveIp(q_data, d_data, d_dim);
    float simd_res = alaya::InnerProduct(q_data, d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  delete[] data;
  delete[] query;
}

TEST(InnerProductTest, AlignGist) {
  std::string gist_path = "/dataset/gist";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::AlignLoadVecsDataset<float>(gist_path.c_str(), data, d_num, d_dim,
                                     query, q_num, q_dim);
  unsigned align_dim = alaya::DoAlign(d_dim, 16);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * align_dim;
    float* d_data = data + d_id * align_dim;

    float naive_res = alaya::NaiveIp(q_data, d_data, d_dim);
    float simd_res = alaya::AlignInnerProduct(q_data, d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  std::free(data);
  std::free(query);
}

TEST(L2SqrTest, Netflix) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query,
                                q_num, q_dim);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * q_dim;
    float* d_data = data + d_id * d_dim;

    float naive_res = alaya::NaiveL2Sqr(q_data, d_data, d_dim);
    float simd_res = alaya::L2SqrFloat(q_data, d_data, d_dim);
    // fmt::println("naive: {}, simd: {}", naive_res, simd_res);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  delete[] data;
  delete[] query;
}

TEST(L2SqrTest, AlignNetflix) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::AlignLoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim,
                                     query, q_num, q_dim);
  unsigned align_dim = alaya::DoAlign(d_dim, 16);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * align_dim;
    float* d_data = data + d_id * align_dim;

    float naive_res = alaya::NaiveL2Sqr(q_data, d_data, d_dim);
    float simd_res = alaya::AlignL2Sqr(q_data, d_data, d_dim);
    // fmt::println("naive: {}, simd: {}", naive_res, simd_res);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  std::free(data);
  std::free(query);
}

TEST(L2SqrTest, Sift1m) {
  std::string sift1m_path = "/dataset/sift1m";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(sift1m_path.c_str(), data, d_num, d_dim, query,
                                q_num, q_dim);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * q_dim;
    float* d_data = data + d_id * d_dim;

    float naive_res = alaya::NaiveL2Sqr(q_data, d_data, d_dim);
    float simd_res = alaya::L2SqrFloat(q_data, d_data, q_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  delete[] data;
  delete[] query;
}

TEST(L2SqrTest, AlignSift1m) {
  std::string sift1m_path = "/dataset/sift1m";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::AlignLoadVecsDataset<float>(sift1m_path.c_str(), data, d_num, d_dim,
                                     query, q_num, q_dim);
  unsigned align_dim = alaya::DoAlign(d_dim, 16);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * align_dim;
    float* d_data = data + d_id * align_dim;

    float naive_res = alaya::NaiveL2Sqr(q_data, d_data, d_dim);
    float simd_res = alaya::AlignL2SqrFloat(q_data, d_data, q_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  std::free(data);
  std::free(query);
}

TEST(L2SqrTest, Deep1m) {
  std::string deep_path = "/dataset/deep1m";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(deep_path.c_str(), data, d_num, d_dim, query,
                                q_num, q_dim);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * q_dim;
    float* d_data = data + d_id * d_dim;

    float naive_res = alaya::NaiveL2Sqr(q_data, d_data, d_dim);
    float simd_res = alaya::L2SqrFloat(q_data, d_data, q_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  delete[] data;
  delete[] query;
}

TEST(L2SqrTest, AlignDeep1m) {
  std::string deep_path = "/dataset/deep1m";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(deep_path.c_str(), data, d_num, d_dim, query,
                                q_num, q_dim);
  unsigned align_dim = alaya::DoAlign(d_dim, 16);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * align_dim;
    float* d_data = data + d_id * align_dim;

    float naive_res = alaya::NaiveL2Sqr(q_data, d_data, d_dim);
    float simd_res = alaya::AlignL2SqrFloat(q_data, d_data, q_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  delete[] data;
  delete[] query;
}

TEST(L2SqrTest, Gist) {
  std::string gist_path = "/dataset/gist";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(gist_path.c_str(), data, d_num, d_dim, query,
                                q_num, q_dim);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * q_dim;
    float* d_data = data + d_id * d_dim;

    float naive_res = alaya::NaiveL2Sqr(q_data, d_data, d_dim);
    float simd_res = alaya::L2SqrFloat(q_data, d_data, q_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  delete[] data;
  delete[] query;
}

TEST(L2SqrTest, AlignGist) {
  std::string gist_path = "/dataset/gist";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(gist_path.c_str(), data, d_num, d_dim, query,
                                q_num, q_dim);
  unsigned align_dim = alaya::DoAlign(d_dim, 16);

  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int q_id = alaya::GenRandInt(0, q_num - 1);
    int d_id = alaya::GenRandInt(0, d_num - 1);
    // fmt::println("Query Id: {}, Data Id: {}", q_id, d_id);
    float* q_data = query + q_id * align_dim;
    float* d_data = data + d_id * align_dim;

    float naive_res = alaya::NaiveL2Sqr(q_data, d_data, d_dim);
    float simd_res = alaya::L2SqrFloat(q_data, d_data, q_dim);
    EXPECT_TRUE(std::fabs(naive_res - simd_res) < kDeltaFloat);
  }
  delete[] data;
  delete[] query;
}

TEST(GetDistFuncTest, FuncTest) {
  float x[] = {1.0, 2.0, 3.0};
  float y[] = {4.0, 5.0, 6.0};

  auto l2_dist = alaya::GetDistFunc<float, false>(alaya::MetricType::L2);
  float result = l2_dist(x, y, 3);
  EXPECT_FLOAT_EQ(result, 27.0);

  auto ip_dist = alaya::GetDistFunc<float, false>(alaya::MetricType::IP);
  result = ip_dist(x, y, 3);
  EXPECT_FLOAT_EQ(result, 32.0);

  float align_x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0,
                     0,   0,   0,   0,   0,   0,   0,   0};
  float align_y[] = {2.0, 8.0, 4.0, 2.0, 1.0, 3.0, 9.0, 0,
                     0,   0,   0,   0,   0,   0,   0,   0};

  auto align_l2 = alaya::GetDistFunc<float, true>(alaya::MetricType::L2);
  result = align_l2(align_x, align_y, 3);
  EXPECT_FLOAT_EQ(result, 71.0);

  auto align_ip = alaya::GetDistFunc<float, true>(alaya::MetricType::IP);
  result = align_ip(align_x, align_y, 3);
  EXPECT_FLOAT_EQ(result, 124);
}

TEST(NormTest, Netflix) {
  std::string netflix_path = "/dataset/netflix/netflix_base.fvecs";
  unsigned d_num, d_dim;
  float* data = alaya::LoadVecs<float>(netflix_path.c_str(), d_num, d_dim);
  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int d_id = alaya::GenRandInt(0, d_num - 1);

    float* d_data = data + d_id * d_dim;

    float naive_sqr_res = alaya::NaiveGetSqrNorm(d_data, d_dim);
    float simd_sqr_res = alaya::GetSqrNorm(d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_sqr_res - simd_sqr_res) < kDeltaFloat);
    float naive_norm = alaya::NaiveGetNorm(d_data, d_dim);
    float simd_norm = alaya::GetNorm(d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_norm - simd_norm) < kDeltaFloat);
    EXPECT_TRUE(std::fabs(std::sqrt(simd_sqr_res) - simd_norm) < kDeltaFloat);
  }
}

TEST(NormTest, Sift1m) {
  std::string sift1m_path = "/dataset/netflix/netflix_base.fvecs";
  unsigned d_num, d_dim;
  float* data = alaya::LoadVecs<float>(sift1m_path.c_str(), d_num, d_dim);
  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int d_id = alaya::GenRandInt(0, d_num - 1);

    float* d_data = data + d_id * d_dim;

    float naive_sqr_res = alaya::NaiveGetSqrNorm(d_data, d_dim);
    float simd_sqr_res = alaya::GetSqrNorm(d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_sqr_res - simd_sqr_res) < kDeltaFloat);
    float naive_norm = alaya::NaiveGetNorm(d_data, d_dim);
    float simd_norm = alaya::GetNorm(d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_norm - simd_norm) < kDeltaFloat);
    EXPECT_TRUE(std::fabs(std::sqrt(simd_sqr_res) - simd_norm) < kDeltaFloat);
  }
}

TEST(NormTest, Deep1m) {
  std::string deep1m_path = "/dataset/deep1m/deep1m_base.fvecs";
  unsigned d_num, d_dim;
  float* data = alaya::LoadVecs<float>(deep1m_path.c_str(), d_num, d_dim);
  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int d_id = alaya::GenRandInt(0, d_num - 1);

    float* d_data = data + d_id * d_dim;

    float naive_sqr_res = alaya::NaiveGetSqrNorm(d_data, d_dim);
    float simd_sqr_res = alaya::GetSqrNorm(d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_sqr_res - simd_sqr_res) < kDeltaFloat);
    float naive_norm = alaya::NaiveGetNorm(d_data, d_dim);
    float simd_norm = alaya::GetNorm(d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_norm - simd_norm) < kDeltaFloat);
    EXPECT_TRUE(std::fabs(std::sqrt(simd_sqr_res) - simd_norm) < kDeltaFloat);
  }
}

TEST(NormTest, Gist) {
  std::string gist_path = "/dataset/gist/gist_base.fvecs";
  unsigned d_num, d_dim;
  float* data = alaya::LoadVecs<float>(gist_path.c_str(), d_num, d_dim);
  int num_test = 100;
  for (int i = 0; i < num_test; ++i) {
    int d_id = alaya::GenRandInt(0, d_num - 1);

    float* d_data = data + d_id * d_dim;

    float naive_sqr_res = alaya::NaiveGetSqrNorm(d_data, d_dim);
    float simd_sqr_res = alaya::GetSqrNorm(d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_sqr_res - simd_sqr_res) < kDeltaFloat);
    float naive_norm = alaya::NaiveGetNorm(d_data, d_dim);
    float simd_norm = alaya::GetNorm(d_data, d_dim);
    EXPECT_TRUE(std::fabs(naive_norm - simd_norm) < kDeltaFloat);
    EXPECT_TRUE(std::fabs(std::sqrt(simd_sqr_res) - simd_norm) < kDeltaFloat);
  }
}