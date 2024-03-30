#pragma once

#include <map>
#include <vector>

#include "../index.h"

namespace alaya {

template <typename IDType, typename DataType>
struct Bucket : Index<IDType, DataType> {
  int data_dim_;            // vector dimension
  IDType data_num_;         // total number of indexed vectors
  MetricType metric_type_;  // type of metric this index uses for index building
  int bucket_num_;          // cluster number  or  cell number

  std::vector<std::vector<DataType>> buckets_;
  std::vector<std::vector<IDType>> id_buckets_;
  std::vector<std::pair<int, DataType>> order_list_;
  std::map<IDType, IDType>
      id_maps_;  // unsure to use.  mapping for original id to local id is stored
  explicit Bucket() {}

  explicit Bucket(const int bucket_num, MetricType metric, const int data_dim)
      : bucket_num_(bucket_num), metric_type_(metric), data_dim_(data_dim) {}

  ~Bucket() {}

  virtual void BuildIndex(IDType data_num, const DataType* data_ptr) = 0;
  virtual void BuildIndexWithIds(IDType data_num, const IDType* data_ids,
                                 const DataType* data_ptr) {}

  virtual void Save(const char* k_path) const = 0;
  virtual void Load(const char* k_path) = 0;
};

}  // namespace alaya