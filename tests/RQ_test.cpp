#include <gtest/gtest.h>
#include <alaya/index/quantizer/residual_quantizer.h>
#include <cstdint>
#include <alaya/utils/distance.h>
#include <alaya/utils/io_utils.h>
#include <alaya/utils/memory.h>
#include <alaya/utils/metric_type.h>
#include <alaya/utils/random_utils.h>
#include <fmt/core.h>
#include <gtest/gtest.h>
namespace {
    TEST(ResidualQuantizer, DefaultConstructor) {
        int vec_dim = 10;
        int vec_num = 100;
        int book_num = 1000;
        int level = 4;
        alaya::ResidualQuantizer<8, int64_t, float> rq(vec_dim, vec_num, alaya::MetricType::L2, book_num, level);
        EXPECT_EQ(rq.vec_dim_, vec_dim);
        EXPECT_EQ(rq.vec_num_, vec_num);
        EXPECT_EQ(rq.book_num_, book_num);
        EXPECT_EQ(rq.level_, level);
    }
    TEST(ResidualQuantizer, BuildIndex) {
        std::string netflix_path = "/dataset/netflix";
        unsigned d_num, d_dim, q_num, q_dim;
        float *data, *query;
        alaya::AlignLoadVecsDataset<float>(netflix_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);
        unsigned align_dim = alaya::DoAlign(d_dim, 16);
        alaya::ResidualQuantizer<8, int64_t, float> rq(d_dim, d_num, alaya::MetricType::L2, 256, 4);
        rq.BuildIndex(d_num, data);
        delete[] data;
        delete[] query;
    }
}