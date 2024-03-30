#include <alaya/searcher/bucket/ivf_searcher.h>
#include <fmt/format.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/alaya/utils/heap.h"
#include "alaya/index/bucket/imi.h"
#include "alaya/index/bucket/ivf.h"
#include "alaya/utils/heap.h"
#include "alaya/utils/metric_type.h"
#include "fmt/core.h"
// TODO  delete

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

void BruteForceSearch(float* query, int query_dim, int k,
                      alaya::InvertedMultiIndex<int, float>& imiindex,
                      alaya::ResultPool<int, float>& res) {
  printf("into BFS\n");
  for (int i = 0; i < imiindex.buckets_.size(); ++i) {
    printf("into first for loop\n");
    float* data_point = imiindex.buckets_[i].data();
    for (int j = 0; j < imiindex.id_buckets_[i].size(); ++j) {
      float dist = alaya::L2Sqr<float>(query, data_point + j * query_dim, query_dim);
      res.Insert(imiindex.id_buckets_[i][j], dist);
      printf("done inserting...i = %d, j = %d, %d, %f\n", i, j, imiindex.id_buckets_[i][j], dist);
    }
  }
  printf("done BFS\n");
}
int main() {
  std::vector<float> data;
  int data_dim, data_num;
  load_bin<int, float>("/home/dongjiang/datasets/siftsmall/siftsmall_base.fbin", data, data_dim,
                       data_num);
  std::cout << data_dim << " " << data_num << std::endl;

  alaya::MetricType metric = alaya::MetricType::L2;
  alaya::InvertedMultiIndex<int, float> imiindex(3, metric, data_dim, 2, 10);
  imiindex.BuildIndex(data_num, data.data());

  int query_dim, query_num;
  std::vector<float> query;
  load_bin<int, float>("/home/dongjiang/datasets/siftsmall/siftsmall_query.fbin", query, query_dim,
                       query_num);
  printf("query_num = %d, query_dim = %d\n", query_num, query_dim);
  int k = 100;
  alaya::ResultPool<int, float> res(imiindex.data_num_, 2 * k, k);
  BruteForceSearch(query.data(), query_dim, k, imiindex, res);
  printf("after BFS\n");

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
      if (gt[i] == res.result_.pool_[j].id_) {
        count++;
        break;
      }
    }
  }
  std::cout << "count is ";
  std::cout << count << std::endl;
  // imiindex.Save("/home/dongjiang/datasets/siftsmall/imiindex_test.bin");
  // // alaya::InvertedList<int, float> loadtest(10, metric, data_dim, 10);
  // // loadtest.BuildIndex(data_num, data.data());
  // alaya::InvertedMultiIndex<int, float> imiindex_load(3, metric, data_dim, 2, 10);
  // imiindex_load.Load("/home/dongjiang/datasets/siftsmall/imiindex_test.bin");
  // printf("imi_index.bucket_num_ = %d, data_num = %d, data_dim = %d\n", imiindex_load.bucket_num_,
  //        imiindex_load.data_num_, imiindex_load.data_dim_);
}

//= IVF Test
/*
void xxx() {
  alaya::InvertedList<int, float> loadtest(10, metric, data_dim, 10);
  loadtest.BuildIndex(data_num, data.data());
  printf("before save\n");
  loadtest.Save("/home/dongjiang/datasets/siftsmall/ivfindex_test.bin");
  printf("after save\n");

  alaya::InvertedList<int, float> ivfindex(10, metric, data_dim, 10);
  ivfindex.Load("/home/dongjiang/datasets/siftsmall/ivfindex_test.bin");
  printf("ivf_index.bucket_num_ = %d, data_num = %d, data_dim = %d\n", ivfindex.bucket_num_,
         ivfindex.data_num_, ivfindex.data_dim_);

  alaya::InvertedListSearcher<int, float> ivfsearcher;
  int query_dim, query_num;
  std::vector<float> query;
  load_bin<int, float>("/home/dongjiang/datasets/siftsmall/siftsmall_query.fbin", query, query_dim,
                       query_num);

  printf("query_num = %d, query_dim = %d\n", query_num, query_dim);
  ivfsearcher.SetIndex(ivfindex);
  printf("1\n");
  int k = 100;
  float* distance = new float[k];
  ivfsearcher.Search(query_num, query_dim, query.data(), k, distance, nullptr);
  printf("1\n");
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
      if (gt[i] == ivfsearcher.res_->result_.pool_[j].id_) {
        count++;
        break;
      }
    }
  }
  std::cout << count << std::endl;
  delete[] distance;
}
*/

// for (int i = 0; i < ivfindex.bucket_num_; ++i) {
//   printf("bucket %d: ", i);
//   for (int j = 0; j < ivfindex.centroids_[i].size(); ++j) {
//     printf("%f ", ivfindex.centroids_[i][j]);
//   }
//   printf("\n");
// }
// int bucket_sum = 0;
// for (int i = 0; i < ivfindex.bucket_num_; ++i) {
//   printf("bucket %d: -------  bucket size is %ld\n", i, ivfindex.id_buckets_[i].size());
//   bucket_sum += ivfindex.id_buckets_[i].size();
// }
// std::cout << bucket_sum << std::endl;