#include <alaya/utils/distance.h>
#include <gtest/gtest.h>

TEST(InnerProductTest, Float) {
  float x[] = {1.0, 2.0, 3.0};
  float y[] = {4.0, 5.0, 6.0};
  float result = alaya::NaiveIp(x, y, 3);
  EXPECT_FLOAT_EQ(result, 32.0);
  float simd_res = alaya::InnerProduct(x, y, 3);
  EXPECT_FLOAT_EQ(simd_res, 32.0);
}
