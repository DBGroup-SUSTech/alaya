
#include <alaya/searcher/bucket/ivf_searcher.h>
#include <gtest/gtest.h>

#include <fstream>

#include "alaya/index/bucket/ivf.h"
#include "alaya/utils/metric_type.h"

template <typename DataType, typename IDType>
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
  load_bin<float, int64_t>("/home/dongjiang/datasets/siftsmall/siftsmall_base.fbin", data, data_dim,
                           data_num);
  EXPECT_EQ(data_dim, 128);
  EXPECT_EQ(data_num, 10000);

  alaya::MetricType metric = alaya::MetricType::L2;

  alaya::InvertedList<float, int64_t> ivfindex(10, metric, data_dim);
  ivfindex.BuildIndex(data_num, data.data());

  alaya::InvertedListSearcher<float, int64_t> ivfsearcher(&ivfindex);

  int query_dim, query_num;
  std::vector<float> query;
  load_bin<float, int64_t>("/home/dongjiang/datasets/siftsmall/siftsmall_query.fbin", query,
                           query_dim, query_num);
  int k = 100;
  float* dist = new float[100];
  int64_t* ids = new int64_t[100];
  ivfsearcher.Search(query_num, query_dim, query.data(), k, dist, ids);

  std::ifstream gt_reader("/home/dongjiang/datasets/siftsmall/siftsmall_groundtruth.ivecs",
                          std::ios::binary);
  if (!gt_reader.is_open()) {
    std::cerr << "Error: cannot open file " << std::endl;
    exit(1);
  }
  printf("1\n");
  int* gt = new int[100];
  int nothing = 0;
  for (int i = 0; i < 1; ++i) {
    gt_reader.read(reinterpret_cast<char*>(&nothing), sizeof(int));
    gt_reader.read(reinterpret_cast<char*>(gt), sizeof(int) * 100);
    // for (int j = 0; j < 101; ++j) {
    //   printf("%d ", gt[j]);
    // }
    // printf("\n");
  }
  printf("1\n");
  std::cout << std::endl;
  int count = 0;
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      if (gt[i] == ids[j]) {
        count++;
        break;
      }
    }
  }
  std::cout << count << std::endl;
  delete[] dist;
  delete[] ids;
}
