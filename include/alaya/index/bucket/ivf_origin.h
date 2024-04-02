#pragma once
#include <alaya/index/bucket/bucket.h>
#include <alaya/utils/distance.h>
#include <alaya/utils/memory.h>
#include <alaya/utils/metric_type.h>
#include <faiss/Clustering.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace alaya {

template <typename DataType = float, typename IDType = int64_t>
struct InvertedList : Bucket<DataType, IDType> {
  const DataType* data_ = nullptr;        // input vector base data
  unsigned int clustering_iter_;          // max kmeans iterations
  std::vector<int> nearest_centroid_id_;  // nearest_centroid_id_
  // DataType* centroids_data_ = nullptr;    // centroid data ptr

  DataType* centroids_ = nullptr;  // array for centroids
  DistFunc<DataType, DataType, DataType> dist_func_;

  InvertedList() = default;

  /**
   * @brief Construct a new InvertedList<ID Type,  Data Type> object
   *
   * @param bucket_num  cluster number
   * @param metric  type of metric this index uses for index building
   * @param vec_dim  vector dimension
   * @param clustering_iter  max kmeans iterations
   */
  explicit InvertedList(const int bucket_num, MetricType metric, const int vec_dim)
      : Bucket<DataType, IDType>(bucket_num, metric, vec_dim),
        dist_func_(GetDistFunc<DataType, false>(metric)) {
    this->centroids_ = (DataType*)Alloc64B(sizeof(DataType) * this->bucket_num_ * this->vec_dim_);
  }

  ~InvertedList() {}

  void GetNearestCentroidIds() {
#pragma omp parallel for
    for (IDType i = 0; i < this->vec_num_; ++i) {
      DataType min_dist = std::numeric_limits<DataType>::max();
      for (int j = 0; j < this->bucket_num_; ++j) {
        // auto distFunc = GetDistFunc<DataType, false>(metric_type_);
        // assert(distFunc != nullptr || !"metric_type invalid!");
        // DataType dist = distFunc(data_ + i * vec_dim_, centroids_[j].data(), vec_dim_);
        // DataType dist = L2Sqr<DataType>(data_ + i * vec_dim_, centroids_[j].data(), vec_dim_);
        DataType dist =
            dist_func_(data_ + i * this->vec_dim_, centroids_ + j * this->vec_dim_, this->vec_dim_);
        if (dist < min_dist) {
          min_dist = dist;
          nearest_centroid_id_[i] = j;
        }
      }
    }
  }
  void FillIndex() {
    // fill id bucket with bucket num of each vector
    this->id_buckets_.resize(this->bucket_num_);
    for (IDType i = 0; i < this->vec_num_; ++i) {
      this->id_buckets_[nearest_centroid_id_[i]].emplace_back(i);
    }
    // fill vector bucket with bucket num of each vector
    this->buckets_.resize(this->bucket_num_);

#pragma omp parallel for
    // get each bucket
    for (IDType i = 0; i < this->bucket_num_; ++i) {
      this->buckets_[i].resize(this->id_buckets_[i].size() * this->vec_dim_);
      // get the beginning ptr of each bucket
      DataType* single_data_ptr = this->buckets_[i].data();
      // copy all data according to the id in the id_bucket from data_ to buckets[i]
      for (int j = 0; j < this->id_buckets_[i].size(); ++j) {
        const DataType* v_begin = data_ + this->id_buckets_[i][j] * this->vec_dim_;
        const DataType* v_end = v_begin + this->vec_dim_;
        // copy data from v_begin to v_end to single_data_ptr aka buckets_[i].data()
        std::copy(v_begin, v_end, single_data_ptr);
        single_data_ptr += this->vec_dim_;
      }
    }
  }

  void BuildIndex(IDType vec_num, const DataType* data_ptr) override {
    this->vec_num_ = vec_num;
    data_ = data_ptr;
    nearest_centroid_id_.resize(vec_num);

    // compute the centroids
    auto kmeans_err = faiss::kmeans_clustering(this->vec_dim_, this->vec_num_, this->bucket_num_,
                                               data_, (float*)centroids_);
    // kmeans<DataType, IDType>(data_, vec_num_, vec_dim_, centroids_, bucket_num_, false,
    //                          clustering_iter_);
    printf("[Report] - kmeans over\n");

    printf("[Report] - cluster number is: %zu\n", static_cast<size_t>(this->bucket_num_));

    GetNearestCentroidIds();

    FillIndex();
    // this->order_list_.resize(this->bucket_num_);
    printf("[Report] - build data bucket complete!\n");
  }

  void BuildIndexWithIds(IDType vec_num, const IDType* data_ids,
                         const DataType* data_ptr) override {
    this->vec_num_ = vec_num;
    data_ = data_ptr;
    nearest_centroid_id_.resize(vec_num);

    // init id_map_, original id to local id is stored
    // #pragma omp parallel for
    //         for (int i = 0; i < vec_num_; ++i)
    //         {
    //             id_maps_[data_ids + i] = i;
    //         }

    // compute the centroids
    auto kmeans_err = faiss::kmeans_clustering(this->vec_dim_, this->vec_num_, this->bucket_num_,
                                               data_, centroids_);

    printf("[Report] - kmeans over\n");
    printf("[Report] - cluster number is: %zu\n", static_cast<size_t>(this->bucket_num_));

    GetNearestCentroidIds();

    FillIndex();

// replace ids with actual ids
#pragma omp parallel for
    for (int i = 0; i < this->bucket_num_; ++i) {
      for (int j = 0; j < this->id_buckets_[i].size(); ++j) {
        IDType fake_id = this->id_buckets_[i][j];
        const IDType* real_id = data_ids + fake_id;
        this->id_buckets_[i][j] = *real_id;
      }
    }

    printf("[Report] - build data bucket complete!\n");

    // vec_ids 到最后进行一个替换即可
  }

  // index file structure:
  // metadata:   | vec_num_ | vec_dim_ | bucket_num_ |
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
           static_cast<size_t>(this->bucket_num_));
    output.write((char*)&this->vec_num_, sizeof(IDType));
    output.write((char*)&this->vec_dim_, sizeof(int));
    // write bucket_num_
    output.write((char*)&this->bucket_num_, sizeof(int));
    // write all centroids
    output.write((char*)centroids_, this->vec_dim_ * this->bucket_num_ * sizeof(DataType));

    // write each bucket size and id_buckets
    for (int i = 0; i < this->bucket_num_; i++) {
      int each_bucket_size = this->id_buckets_[i].size();
      output.write((char*)&each_bucket_size, sizeof(int));
      output.write((char*)this->id_buckets_[i].data(), sizeof(IDType) * each_bucket_size);
      output.write((char*)this->buckets_[i].data(),
                   sizeof(DataType) * each_bucket_size * this->vec_dim_);
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
    input.read((char*)&this->vec_num_, sizeof(IDType));
    input.read((char*)&this->vec_dim_, sizeof(int));
    input.read((char*)&this->bucket_num_, sizeof(int));

    printf("[Report] - vec_num = %zu, vec_dim = %zu, bucket_num = %zu, starting to read index...\n",
           static_cast<size_t>(this->vec_num_), static_cast<size_t>(this->vec_dim_),
           static_cast<size_t>(this->bucket_num_));
    centroids_ = (DataType*)Alloc64B(this->bucket_num_ * this->vec_dim_ * sizeof(DataType));
    this->id_buckets_.resize(this->bucket_num_);
    this->buckets_.resize(this->bucket_num_);
    // this->order_list_.resize(this->bucket_num_);

    input.read((char*)centroids_, this->bucket_num_ * this->vec_dim_ * sizeof(DataType));

    for (int i = 0; i < this->bucket_num_; ++i) {
      int each_bucket_size = 0;
      input.read((char*)&each_bucket_size, sizeof(int));
      printf("each bucket size = %d\n", each_bucket_size);
      this->id_buckets_[i].resize(each_bucket_size);
      this->buckets_[i].resize(each_bucket_size * this->vec_dim_);
      input.read((char*)this->id_buckets_[i].data(), sizeof(IDType) * each_bucket_size);
      input.read((char*)this->buckets_[i].data(),
                 sizeof(DataType) * each_bucket_size * this->vec_dim_);
    }
    printf("[Report] - reading index complete!\n");

    input.close();
  }
};

}  // namespace alaya