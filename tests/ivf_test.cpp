#include <alaya/index/bucket/ivf.h>
#include <alaya/searcher/ivf_searcher.h>
#include <alaya/utils/io_utils.h>
#include <alaya/utils/metric_type.h>
#include <gtest/gtest.h>

#include <vector>

#include "alaya/utils/distance.h"
#include "alaya/utils/random_utils.h"
#include "fmt/core.h"

constexpr float kDeltaFloat = 1e-3;

TEST(IvfTest, Constructor) {
  alaya::IVF<float> ivf(300, alaya::MetricType::L2, 1000);

  EXPECT_EQ(300, ivf.vec_dim_);
  EXPECT_EQ(304, ivf.align_dim_);
  EXPECT_EQ(1000, ivf.bucket_num_);
}

TEST(IvfTest, BuildIndex) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);

  unsigned bucket_num = 100;
  alaya::IVF<float> ivf(d_dim, alaya::MetricType::L2, bucket_num);

  ivf.BuildIndex(d_num, data);

  EXPECT_EQ(300, ivf.vec_dim_);
  EXPECT_EQ(304, ivf.align_dim_);
  EXPECT_EQ(bucket_num, ivf.bucket_num_);
  EXPECT_EQ(bucket_num, ivf.data_buckets_.size());
  EXPECT_EQ(bucket_num, ivf.id_buckets_.size());

  unsigned sum_bucket = 0;
  for (auto i = 0; i < bucket_num; ++i) {
    unsigned bucket_size = ivf.GetBucketSize(i);
    // fmt::println("bucket: {}, bucket_size: {}", i, bucket_size);
    sum_bucket += bucket_size;
  }

  EXPECT_EQ(d_num, sum_bucket);

  std::vector<bool> visited(d_num, false);

  unsigned count = 0;
  for (auto i = 0; i < bucket_num; ++i) {
    unsigned bucket_size = ivf.GetBucketSize(i);
    for (auto offset = 0; offset < bucket_size; ++offset) {
      auto id = ivf.GetDataId(i, offset);
      visited[id] = true;
      ++count;
    }
  }

  EXPECT_EQ(count, d_num);

  for (auto i = 0; i < d_num; ++i) {
    EXPECT_TRUE(visited[i]);
  }

  for (auto bid = 0; bid < bucket_num; ++bid) {
    unsigned bucket_size = ivf.GetBucketSize(bid);
    for (auto j = 0; j < bucket_size; ++j) {
      auto d_id = ivf.GetDataId(bid, j);
      auto vbdata = ivf.data_buckets_[bid] + j * ivf.align_dim_;
      auto vdata = data + d_id * d_dim;
      for (auto k = 0; k < d_dim; ++k) {
        EXPECT_TRUE(std::abs(vbdata[k] - vdata[k]) < kDeltaFloat);
      }
    }
  }

  // for (auto i = 0; i < 10; ++i) {
  //   auto bucket_id = alaya::GenRandInt(0, bucket_num - 1);
  //   unsigned bucket_size = *((unsigned*)ivf.id_buckets_[bucket_id]);
  //   fmt::println("bucket id:{}, bucket size: {}", bucket_id, bucket_size);
  //   std::vector<float> vec(d_dim, 0);
  //   for (auto j = 0; j < bucket_size; ++j) {
  //     alaya::AddAssign(ivf.data_buckets_[bucket_id] + j * ivf.align_dim_, vec.data(), d_dim);
  //   }
  //   for (auto j = 0; j < d_dim; ++j) {
  //     vec[j] /= bucket_size;
  //     fmt::print("dim {}, avg_vec {}: centriods {}\n", j, vec[j],
  //                ivf.centroids_[bucket_id * ivf.align_dim_ + j]);
  //     EXPECT_TRUE(std::abs(vec[j] - ivf.centroids_[bucket_id * ivf.align_dim_ + j]) <
  //     kDeltaFloat);
  //   }
  // }
}

TEST(IvfTest, Search) {
  std::string netflix_path = "/dataset/netflix";
  unsigned d_num, d_dim, q_num, q_dim;
  // float *data, *query;
  // alaya::LoadVecsDataset(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);
  // alaya::AlignLoadVecsDataset(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);
  float* data = alaya::LoadVecs<float>(fmt::format("{}/netflix_base.fvecs", netflix_path).c_str(),
                                       d_num, d_dim);
  float* query = alaya::AlignLoadVecs<float>(
      fmt::format("{}/netflix_query.fvecs", netflix_path).c_str(), q_num, q_dim);
  fmt::println("d_num: {}, d_dim: {}, q_num: {}, q_dim: {}", d_num, d_dim, q_num, q_dim);

  // unsigned ad_num, ad_dim;
  // float* a_data = alaya::AlignLoadVecs<float>(
  //     fmt::format("{}/netflix_base.fvecs", netflix_path).c_str(), ad_num, ad_dim);

  unsigned bucket_num = 100;
  alaya::IVF<float> ivf(d_dim, alaya::MetricType::L2, bucket_num);

  ivf.BuildIndex(d_num, data);

  alaya::IvfSearcher<alaya::MetricType::L2, float> searcher(&ivf);

  unsigned k = 10;
  float* distance = new float[k];
  int64_t* result_id = new int64_t[k];

  searcher.SetNprobe(10);

  searcher.Search(query, q_dim, k, distance, result_id);

  for (auto i = 0; i < k; ++i) {
    fmt::println("distance: {}, result_id: {}", distance[i], result_id[i]);
  }

  delete[] distance;
  delete[] result_id;
}