#pragma once

#include <alaya/index/bucket/bucket.h>
#include <alaya/utils/distance.h>
#include <alaya/utils/memory.h>
#include <alaya/utils/metric_type.h>

#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace alaya {

template <typename IDType, typename DataType>
struct InvertedList : Bucket<IDType, DataType> {
  int data_dim_;                          // vector dimension
  IDType data_num_;                       // total number of indexed vectors
  MetricType metric_type_;                // type of metric this index uses for index building
  int bucket_num_;                        // cluster number
  DataType* data_ = nullptr;              // input vector base data
  int clustering_iter_;                   // max kmeans iterations
  std::vector<int> nearest_centroid_id_;  // nearest_centroid_id_
  DataType centroids_data_ = nullptr;     // centroid data ptr

  std::vector<std::vector<float>>
      centroids_;  // array for centroids, each centroids data is stored in one dimension
  std::vector<std::vector<DataType>>
      buckets_;  // array for buckets, vector data are stored in each bucket
  std::vector<std::vector<IDType>>
      id_buckets_;  // array for buckets, vector ids are stored in each bucket
  // array for query to order the list of centroids based on distance
  std::vector<std::pair<int, DataType>> order_list_;

  std::unordered_map<IDType, IDType>
      id_maps_;  // unsure to use.  mapping for original id to local id is stored

  /**
   * @brief Construct a new InvertedList<ID Type,  Data Type> object
   *
   * @param bucket_num  cluster number
   * @param metric  type of metric this index uses for index building
   * @param data_dim  vector dimension
   * @param clustering_iter  max kmeans iterations
   */
  InvertedList<IDType, DataType>(const int bucket_num, MetricType metric, const int data_dim,
                                 const int clustering_iter)
      : bucket_num_(bucket_num),
        metric_type_(metric),
        data_dim_(data_dim),
        clustering_iter_(clustering_iter) {}

  ~InvertedList() {}

  void GetNearestCentroidIds() {
#pragma omp parallel for
    for (IDType i = 0; i < data_num_; ++i) {
      DataType min_dist = std::numeric_limits<int>::max();
      for (int j = 0; j < bucket_num_; ++j) {
        auto distFunc = GetDistFunc<DataType, false>(metric_type_);
        assert(distFunc != nullptr || !"metric_type invalid!");
        DataType dist = distFunc(data_ + i * data_dim_, centroids_[j].data(), data_dim_);
        if (dist < min_dist) {
          min_dist = dist;
          nearest_centroid_id_[i] = j;
        }
      }
    }
  }
  void FillIndex() {
    // fill id bucket with bucket num of each vector
    id_buckets_.resize(bucket_num_);
    for (IDType i = 0; i < data_num_; ++i) {
      id_buckets_[nearest_centroid_id_[i]].emplace_back(i);
    }
    // fill vector bucket with bucket num of each vector
    buckets_.resize(bucket_num_);

#pragma omp parallel for
    // get each bucket
    for (IDType i = 0; i < bucket_num_; ++i) {
      buckets_[i].resize(id_buckets_[i].size() * data_dim_);
      // get the beginning ptr of each bucket
      DataType* single_data_ptr = buckets_[i].data();
      // copy all data according to the id in the id_bucket from data_ to buckets[i]
      for (int j = 0; j < id_buckets_[i].size(); ++j) {
        DataType* v_begin = data_ + id_buckets_[i][j] * data_dim_;
        DataType* v_end = v_begin + data_dim_;
        // copy data from v_begin to v_end to single_data_ptr aka buckets_[i].data()
        std::copy(v_begin, v_end, single_data_ptr);
        single_data_ptr += data_dim_;
      }
    }
  }

  /**
   * @brief Based on the data_num and the data array pointer to build an ivf index
   *
   * @param data_num  number of data items of input data
   * @param data_ptr  data array pointer
   */
  void BuildIndex(IDType data_num, const DataType* data_ptr) override {
    data_num_ = data_num;
    data_ = data_ptr;
    nearest_centroid_id_.resize(data_num);
    centroids_.resize(0);

    // compute the centroids
    utils::kmeans(data_, data_num_, data_dim_, centroids_, bucket_num_, clustering_iter_);

    printf("[Report] - kmeans over\n");

    printf("[Report] - cluster number is: %zu\n", static_cast<size_t>(bucket_num_));
    assert(bucket_num_ == centroids_.size() || !"cluster number do not match!");

    GetNearestCentroidIds();

    FillIndex();

    centroids_data_ = Alloc64B(bucket_num_ * data_dim_ * sizeof(DataType));
#pragma omp parallel for
    for (size_t i = 0; i < bucket_num_; ++i) {
      std::copy(centroids_[i].begin(), centroids_[i].end(), centroids_data_ + i * data_dim_);
    }

    order_list_.resize(bucket_num_);
    printf("[Report] - build data bucket complete!\n");
  }

  /**
   * @brief Build an index of the input data, and replace the fake ids with actual ids in data_ids
   *
   * @param data_num  number of data items of input data
   * @param data_ids  actual ids of the input data
   * @param data_ptr  data array pointer
   */
  void BuildIndexWithIds(IDType data_num, const IDType* data_ids,
                         const DataType* data_ptr) override {
    data_num_ = data_num;
    data_ = data_ptr;
    nearest_centroid_id_.resize(data_num);
    centroids_.resize(0);

// init id_map_, original id to local id is stored
#pragma omp parallel for
    for (int i = 0; i < data_num_; ++i) {
      id_maps_[data_ids + i] = i;
    }

    // compute the centroids
    utils::kmeans(data_, data_num_, data_dim_, centroids_, bucket_num_, clustering_iter_);

    printf("[Report] - kmeans over\n");
    printf("[Report] - cluster number is: %zu\n", static_cast<size_t>(bucket_num_));
    assert(bucket_num_ == centroids_.size() || !"cluster number do not match!");

    GetNearestCentroidIds();

    FillIndex();

    centroids_data_ = Alloc64B(bucket_num_ * data_dim_ * sizeof(DataType));

// replace ids with actual ids
#pragma omp parallel for
    for (int i = 0; i < bucket_num_; ++i) {
      for (int j = 0; j < id_buckets_[i].size(); ++j) {
        IDType fake_id = id_buckets_[i][j];
        id_buckets_[i][j] = *(data_ids + fake_id);
      }
      std::copy(centroids_[i].begin(), centroids_[i].end(), centroids_data_ + i * data_dim_);
    }

    order_list_.resize(bucket_num_);
    printf("[Report] - build data bucket complete!\n");

    // vec_ids 到最后进行一个替换即可
  }

  // index file structure:
  // metadata:   | data_num_ | data_dim_ | bucket_num_ |
  // centroids:  | centroid0 | centroid1 | centroid2 | ...
  // buckets:
  // | bucket0 size_ |  point ids in 0 | data in 0
  // | bucket1 size_ |  point ids in 1 | data in 1
  // ...

  /**
   * @brief Save the built index to file
   *
   * @param kPath  file path
   */
  void Save(const char* kPath) const override {
    std::ofstream output(kPath, std::ios::binary);
    if (!output.is_open()) {
      throw std::runtime_error("Cannot open file");
    }

    printf("[Report] - cluster number: %zu, starting to save index...\n",
           static_cast<size_t>(bucket_num_));
    output.write((char*)&data_num_, sizeof(IDType));
    output.write((char*)&data_dim_, sizeof(int));
    // write bucket_num_
    output.write((char*)&bucket_num_, sizeof(int));
    // write all centroids
    for (int i = 0; i < bucket_num_; ++i) {
      output.write((char*)centroids_[i].data(), data_dim_ * sizeof(DataType));
    }
    // write each bucket size and id_buckets
    for (int i = 0; i < bucket_num_; i++) {
      uint32_t each_bucket_size = static_cast<uint32_t>(id_buckets_[i].size());
      output.write((char*)&each_bucket_size, sizeof(uint32_t));
      output.write((char*)id_buckets_[i].data(), sizeof(IDType) * each_bucket_size);
      output.write((char*)buckets_[i].data(), sizeof(DataType) * each_bucket_size * data_dim_);
    }
    printf("[Report] - saving index complete!\n");

    output.close();
  }

  /**
   * @brief Load the built index from file
   *
   * @param kPath  file path
   */
  void Load(const char* kPath) override {
    std::ifstream input(kPath, std::ios::binary);
    if (!input.is_open()) {
      throw std::runtime_error("Cannot open file");
    }
    input.read((char*)&data_num_, sizeof(IDType));
    input.read((char*)&data_dim_, sizeof(int));
    input.read((char*)&bucket_num_, sizeof(int));

    printf(
        "[Report] - data_num = %zu, data_dim = %zu, bucket_num = %zu, starting to read index...\n",
        static_cast<size_t>(data_num_), static_cast<size_t>(data_dim_),
        static_cast<size_t>(bucket_num_));

    centroids_.resize(bucket_num_);
    id_buckets_.resize(bucket_num_);
    buckets_.resize(bucket_num_);
    order_list_.resize(bucket_num_);

    for (int i = 0; i < bucket_num_; ++i) {
      input.read((char*)centroids_[i].data(), data_dim_ * sizeof(DataType));
    }

    for (int i = 0; i < bucket_num_; ++i) {
      uint32_t each_bucket_size = 0;
      input.read((char*)&each_bucket_size, sizeof(uint32_t));
      input.read((char*)id_buckets_[i].data(), sizeof(IDType) * each_bucket_size);
      input.read((char*)buckets_[i].data(), sizeof(DataType) * each_bucket_size * data_dim_);
    }

    centroids_data_ = Alloc64B(bucket_num_ * data_dim_ * sizeof(DataType));
#pragma omp parallel for
    for (size_t i = 0; i < bucket_num_; ++i) {
      std::copy(centroids_[i].begin(), centroids_[i].end(), centroids_data_ + i * data_dim_);
    }

    printf("[Report] - reading index complete!\n");

    input.close();
  }
};

}  // namespace alaya
