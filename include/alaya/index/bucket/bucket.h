#pragma once

#include <map>
#include <vector>

#include "../index.h"

namespace alaya {

template <typename IDType, typename DataType>
struct Bucket : Index<IDType, DataType> {
  int bucket_num_;  // cluster number  or  cell number

  std::vector<std::vector<DataType>> buckets_;
  std::vector<std::vector<IDType>> id_buckets_;
  std::vector<std::pair<int, DataType>> order_list_;
  std::map<IDType, IDType>
      id_maps_;  // unsure to use.  mapping for original id to local id is stored

  //     std::vector<std::vector<DataType>>
  //     buckets_;  // array for buckets, vector data are stored in each bucket
  // std::vector<std::vector<IDType>>
  //     id_buckets_;  // array for buckets, vector ids are stored in each bucket
  // // array for query to order the list of centroids based on distance
  // std::vector<std::pair<int, DataType>> order_list_;

  explicit Bucket(const int bucket_num, MetricType metric, const int vec_dim)
      : Index<IDType, DataType>(vec_dim, metric), bucket_num_(bucket_num) {}

  ~Bucket() {}

  virtual void BuildIndex(IDType vec_dim, const DataType* kVecData) = 0;
  virtual void BuildIndexWithIds(IDType vec_dim, const IDType* vec_ids, const DataType* data_ptr) {}

  virtual void Save(const char* k_path) const = 0;
  virtual void Load(const char* k_path) = 0;
};

}  // namespace alaya