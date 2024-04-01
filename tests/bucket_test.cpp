#include <gtest/gtest.h>

#include <fstream>

#include "alaya/index/bucket/ivf.h"
#include "alaya/utils/metric_type.h"

template <typename IDType, typename DataType>
inline void load_bin(const std::string& file_path, std::vector<DataType>& data, int& data_dim,
                     int& data_num) {
  std::ifstream in(file_path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Error: cannot open file " << file_path << std::endl;
    exit(1);
  }
  in.read(reinterpret_cast<char*>(&data_num), sizeof(int));
  in.read(reinterpret_cast<char*>(&data_dim), sizeof(int));
  data.resize(data_dim * data_num);
  in.read(reinterpret_cast<char*>(data.data()), sizeof(DataType) * data_dim * data_num);
  in.close();
}

TEST(BucketTest, Constructor) {
  std::vector<float> data;
  int data_dim, data_num;
  load_bin<int, float>("/home/dongjiang/datasets/siftsmall/siftsmall_base.fbin", data, data_dim,
                       data_num);
  EXPECT_EQ(data_dim, 128);
  EXPECT_EQ(data_num, 10000);

  alaya::MetricType metric = alaya::MetricType::L2;
  alaya::InvertedList<int, float> ivfindex(10, metric, data_dim);
  ivfindex.BuildIndex(data_num, data.data());
}