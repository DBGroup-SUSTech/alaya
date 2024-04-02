#pragma once

#include <fmt/core.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "../../utils/distance.h"
#include "../../utils/kmeans.h"
#include "../../utils/memory.h"
#include "../index.h"

namespace alaya {
template <typename DataType = float, typename IDType = int64_t>
struct IVF : Index<DataType, IDType> {
  int bucket_num_;
  DistFunc<DataType, DataType, DataType> dist_func_;
  // Save
  std::vector<DataType*> buckets_;
  // Save
  std::vector<IDType*> id_buckets_;
  // Save
  std::vector<unsigned> buckets_size_;
  // Save
  DataType* centroids_;

  IDType* order_;
  DataType* centroids_dist_;

  IVF() = default;

  IVF(int dim, MetricType metric, int bucket_num)
      : Index<DataType, IDType>(dim, kAlgin16, metric),
        bucket_num_(bucket_num),
        dist_func_(GetDistFunc<DataType, false>(metric)) {
    order_ = (IDType*)Alloc64B(sizeof(IDType) * bucket_num_);
    centroids_dist_ = (DataType*)Alloc64B(sizeof(DataType) * bucket_num_);
  }

  void BuildIndex(IDType vec_num, const DataType* kVecData) override {
    this->vec_num_ = vec_num;
    buckets_.resize(bucket_num_);
    id_buckets_.resize(bucket_num_);
    buckets_size_.resize(bucket_num_);

    std::vector<std::vector<IDType>> ids;
    std::vector<float> centroids;
    if (this->metric_type_ == MetricType::COS) {
      DataType* norm_vec = (DataType*)Alloc2M(sizeof(DataType) * vec_num * this->vec_dim_);
      std::memcpy(norm_vec, kVecData, vec_num * this->vec_dim_ * sizeof(DataType));
      for (auto vid = 0; vid < vec_num; vid++) {
        auto norm = GetNorm(norm_vec + vid * this->vec_dim_, this->vec_dim_);
        for (auto dim = 0; dim < this->vec_dim_; dim++) {
          norm_vec[vid * this->vec_dim_ + dim] /= norm;
        }
      }
      centroids = kmeans(norm_vec, vec_num, this->vec_dim_, this->bucket_num_, this->metric_type_);

      ids = Assign(norm_vec, vec_num, this->vec_dim_, centroids.data(), this->bucket_num_,
                   this->metric_type_);
    } else {
      auto centroids =
          kmeans(kVecData, vec_num, this->vec_dim_, this->bucket_num_, this->metric_type_);
      ids = Assign(kVecData, vec_num, this->vec_dim_, this->bucket_num_, this->metric_type_);
    }

    centroids_ = (DataType*)Alloc64B(sizeof(DataType) * this->bucket_num_ * this->align_dim_);
    for (auto bid = 0; bid < this->bucket_num_; ++bid) {
      std::memcpy(centroids_[bid * this->align_dim_], centroids.data()[bid * this->vec_dim_],
                  this->vec_dim_);
      buckets_[bid] = (DataType*)Alloc64B(sizeof(DataType) * ids[bid].size() * this->vec_dim_);
      id_buckets_[bid] = (IDType*)Alloc64B(sizeof(IDType) * ids[bid].size() + 4);
      buckets_size_[bid] = ids[bid].size();
      unsigned bucket_size = ids[bid].size();
      std::memcpy(id_buckets_[bid], &bucket_size, 4);
      auto id_bucket = (IDType*)((char*)id_buckets_[bid] + 4);
      for (auto vid = 0; vid < ids[bid].size(); ++vid) {
        std::memcpy(buckets_[bid] + vid * this->vec_dim_, kVecData + ids[bid][vid] * this->vec_dim_,
                    this->vec_dim_ * sizeof(DataType));
        id_bucket[vid] = ids[bid][vid];
      }
    }
  }

  void InitOrder(const DataType* query) {
    auto dist_func = GetDistFunc<DataType, true>(this->metric_type_);
    for (auto bid = 0; bid < this->bucket_num_; ++bid) {
      centroids_dist_[bid] =
          dist_func(query, centroids_ + bid * this->align_dim_, this->align_dim_);
      order_[bid] = bid;
    }
    std::sort(order_, order_ + this->bucket_num_,
              [this](IDType a, IDType b) { return centroids_dist_[a] < centroids_dist_[b]; });
  }

  void Save(const char* kFilePath) const override {
    fmt::println("Save IVF index to file: {}", kFilePath);
  }

  void Load(const char* kFilePath) override {}

  ~IVF() {
    for (auto bid = 0; bid < this->bucket_num_; ++bid) {
      std::free(buckets_[bid]);
      std::free(id_buckets_[bid]);
    }
  }
};

}  // namespace alaya