
#include <alaya/index/bucket/imi.h>
#include <alaya/searcher/bucket/imi_searcher.h>
#include <gtest/gtest.h>

#include <fstream>

#include "alaya/index/bucket/ivf_origin.h"
#include "alaya/utils/io_utils.h"
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

// TEST(BucketTest, Constructor) {
//   std::vector<float> data;
//   int data_dim, data_num;
//   load_bin<float, int64_t>("/home/dongjiang/datasets/siftsmall/siftsmall_base.fbin", data,
//   data_dim,
//                            data_num);
//   EXPECT_EQ(data_dim, 128);
//   EXPECT_EQ(data_num, 10000);

//   alaya::MetricType metric = alaya::MetricType::L2;

//   alaya::InvertedList<float, int64_t> ivfindex(10, metric, data_dim);
//   ivfindex.BuildIndex(data_num, data.data());

//   alaya::InvertedListSearcher<float, int64_t> ivfsearcher(&ivfindex);

//   int query_dim, query_num;
//   std::vector<float> query;
//   load_bin<float, int64_t>("/home/dongjiang/datasets/siftsmall/siftsmall_query.fbin", query,
//                            query_dim, query_num);
//   int k = 100;
//   float* dist = new float[100];
//   int64_t* ids = new int64_t[100];
//   ivfsearcher.Search(query_num, query_dim, query.data(), k, dist, ids);

//   std::ifstream gt_reader("/home/dongjiang/datasets/siftsmall/siftsmall_groundtruth.ivecs",
//                           std::ios::binary);
//   if (!gt_reader.is_open()) {
//     std::cerr << "Error: cannot open file " << std::endl;
//     exit(1);
//   }
//   printf("1\n");
//   int* gt = new int[100];
//   int nothing = 0;
//   for (int i = 0; i < 1; ++i) {
//     gt_reader.read(reinterpret_cast<char*>(&nothing), sizeof(int));
//     gt_reader.read(reinterpret_cast<char*>(gt), sizeof(int) * 100);
//     // for (int j = 0; j < 101; ++j) {
//     //   printf("%d ", gt[j]);
//     // }
//     // printf("\n");
//   }
//   printf("1\n");
//   std::cout << std::endl;
//   int count = 0;
//   for (int i = 0; i < k; ++i) {
//     for (int j = 0; j < k; ++j) {
//       if (gt[i] == ids[j]) {
//         count++;
//         break;
//       }
//     }
//   }
//   std::cout << count << std::endl;
//   delete[] dist;
//   delete[] ids;
// }

// void BruteForceSearch(float* query, int query_dim, int k,
//                       alaya::ImiSearcher<float, int64_t>& imiindex,
//                       alaya::ResultPool<float, int64_t>& res) {
//   printf("into BFS\n");
//   for (int i = 0; i < imiindex.id_buckets_.size(); ++i) {
//     // printf("into first for loop\n");
//     float* data_point = imiindex.data_buckets_[i];
//     for (int j = 0; j < imiindex.cell_data_cnt_[i]; ++j) {
//       float dist = alaya::L2Sqr<float>(query, data_point + j * query_dim, query_dim);
//       res.Insert(imiindex.id_buckets_[i][j], dist);
//       // printf("done inserting...i = %d, j = %d, %d, %f\n", i, j, imiindex.id_buckets_[i][j],
//       // dist);
//     }
//   }
//   printf("done BFS\n");
// }

TEST(IMITest, BuildIndex) {
  std::string siftsmall_path = "/home/dongjiang/datasets/siftsmall";
  unsigned d_num, d_dim, q_num, q_dim;
  float *data, *query;
  alaya::LoadVecsDataset(siftsmall_path.c_str(), data, d_num, d_dim, query, q_num, q_dim);

  unsigned bucket_num = 10;

  EXPECT_EQ(d_dim, 128);
  EXPECT_EQ(d_num, 10000);

  alaya::IMI<float> imi(d_dim, alaya::MetricType::L2, bucket_num, 2);
  imi.BuildIndex(d_num, data);

  EXPECT_EQ(128, imi.vec_dim_);
  // EXPECT_EQ(304, imi.align_dim_);
  EXPECT_EQ(bucket_num, imi.bucket_num_);
  EXPECT_EQ(100, imi.data_buckets_.size());
  EXPECT_EQ(100, imi.id_buckets_.size());

  unsigned sum_bucket = 0;
  for (int i = 0; i < imi.cell_cnt_; ++i) {
    unsigned cell_size = imi.cell_data_cnt_[i];
    std::cout << cell_size << ' ';
    sum_bucket += cell_size;
  }

  std::cout << std::endl;

  EXPECT_EQ(d_num, sum_bucket);

  // alaya::IMI<float> imiindex_save(d_dim, alaya::MetricType::L2, bucket_num, 2);
  imi.Save("/home/dongjiang/datasets/siftsmall/imiindex_test_v2.bin");

  alaya::IMI<float> imiindex_save(d_dim, alaya::MetricType::L2, bucket_num, 2);
  imiindex_save.Load("/home/dongjiang/datasets/siftsmall/imiindex_test_v2.bin");

  int k = 100;

  float* distances = new float[100 * k];
  int64_t* labels = new int64_t[100 * k];
  alaya::ImiSearcher<alaya::MetricType::L2> imisearcher(&imiindex_save);
  imisearcher.SetNprobe(7);
  imisearcher.BatchSearch(q_num, q_dim, query, k, distances, labels);

  std::ifstream gt_reader("/home/dongjiang/datasets/siftsmall/siftsmall_groundtruth.ivecs",
                          std::ios::binary);
  if (!gt_reader.is_open()) {
    std::cerr << "Error: cannot open file " << std::endl;
    exit(1);
  }
  int* gt = new int[100 * k];
  int nothing = 0;
  for (int i = 0; i < k; ++i) {
    gt_reader.read(reinterpret_cast<char*>(&nothing), sizeof(int));
    gt_reader.read(reinterpret_cast<char*>(gt + i * 100), sizeof(int) * 100);
  }
  printf("1\n");
  std::cout << std::endl;

  for (int q = 0; q < q_num; q++) {
    int count = 0;
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < k; ++j) {
        if (gt[i + q * k] == labels[j + q * k]) {
          count++;
          break;
        }
      }
    }
    std::cout << "-----------------------------------begin BFS--------------------------------"
              << std::endl;
    std::cout << "count is ";
    std::cout << count << std::endl;
  }

  delete[] distances;
  delete[] labels;
}