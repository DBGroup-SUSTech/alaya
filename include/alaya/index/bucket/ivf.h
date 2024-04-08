#pragma once

#include <faiss/Clustering.h>
#include <fmt/core.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include "../../utils/distance.h"
#include "../../utils/io_utils.h"
#include "../../utils/kmeans.h"
#include "../../utils/memory.h"
#include "../../utils/platform_macros.h"
#include "../index.h"

namespace alaya {
template <typename DataType = float, typename IDType = uint64_t>
struct IVF : Index<DataType, IDType> {
  int bucket_num_;
  DistFunc<DataType, DataType, DataType> dist_func_;
  std::vector<DataType*> data_buckets_;  // data_buckets_
  std::vector<IDType*> id_buckets_;      //
  DataType* centroids_ = nullptr;        // bucket_num * dim

  IVF() = default;

  IVF(int dim, MetricType metric, int bucket_num)
      : Index<DataType, IDType>(dim, kAlgin16, metric),
        bucket_num_(bucket_num),
        dist_func_(GetDistFunc<DataType, false>(metric)),
        data_buckets_(bucket_num, nullptr),
        id_buckets_(bucket_num, nullptr) {}

  void BuildIndex(IDType vec_num, const DataType* kVecData) override {
    this->vec_num_ = vec_num;
    // data_buckets_.resize(bucket_num_);
    // id_buckets_.resize(bucket_num_);

    std::vector<std::vector<IDType>> ids;
    std::vector<float> centroids;
    if (this->metric_type_ == MetricType::COS) {
      float* norm_vec = (float*)Alloc2M(sizeof(DataType) * vec_num * this->vec_dim_);
      // std::memcpy(norm_vec, kVecData, vec_num * this->vec_dim_ * sizeof(DataType));
      for (auto vid = 0; vid < vec_num; vid++) {
        auto norm = GetNorm(norm_vec + vid * this->vec_dim_, this->vec_dim_);
        for (auto dim = 0; dim < this->vec_dim_; dim++) {
          norm_vec[vid * this->vec_dim_ + dim] = kVecData[vid * this->vec_dim_ + dim] / norm;
        }
      }
      centroids = faiss_kmeans((const float*)norm_vec, vec_num, this->vec_dim_, this->bucket_num_,
                               this->metric_type_);
      // fmt::println("centriods size: {}", centroids.size());
      ids = Assign(norm_vec, vec_num, this->vec_dim_, centroids.data(), this->bucket_num_,
                   this->metric_type_);

    } else {
      centroids = faiss_kmeans((const float*)kVecData, vec_num, this->vec_dim_, this->bucket_num_,
                               this->metric_type_);
      // fmt::println("centriods size: {}", centroids.size());
      ids = Assign(kVecData, vec_num, this->vec_dim_, centroids.data(), this->bucket_num_,
                   this->metric_type_);
    }

    // fmt::println("out cenriods size: {}", centroids.size());
    // fmt::println("Assign over");

    centroids_ = (DataType*)Alloc64B(sizeof(DataType) * this->bucket_num_ * this->align_dim_);
    for (auto bid = 0; bid < this->bucket_num_; ++bid) {
      std::memcpy(centroids_ + bid * this->align_dim_, centroids.data() + bid * this->vec_dim_,
                  this->vec_dim_ * sizeof(float));

      unsigned bucket_size = ids[bid].size();
      data_buckets_[bid] = (DataType*)Alloc64B(sizeof(DataType) * bucket_size * this->align_dim_);
      id_buckets_[bid] = (IDType*)Alloc64B(sizeof(IDType) * bucket_size + 4);
      // buckets_size_[bid] = ids[bid].size();
      std::memcpy(id_buckets_[bid], &bucket_size, 4);
      // fmt::println("bid: {}, size: {}, bucket_size: {}", bid, bucket_size,
      //              *((unsigned*)id_buckets_[bid]));
      auto id_bucket = (IDType*)((char*)id_buckets_[bid] + 4);
      for (auto vid = 0; vid < ids[bid].size(); ++vid) {
        std::memcpy(data_buckets_[bid] + vid * this->align_dim_,
                    kVecData + ids[bid][vid] * this->vec_dim_, this->vec_dim_ * sizeof(DataType));
        id_bucket[vid] = ids[bid][vid];
      }
    }
  }

  ALWAYS_INLINE
  IDType GetDataId(int bucket_id, int offset) const {
    return ((IDType*)((char*)id_buckets_[bucket_id] + 4))[offset];
  }

  ALWAYS_INLINE
  unsigned GetBucketSize(int bucket_id) const { return *((unsigned*)id_buckets_[bucket_id]); }

  void Save(const char* kFilePath) const override {
    fmt::println("Save IVF index to file: {}", kFilePath);
    std::ofstream out(kFilePath, std::ios::binary);
    if (!out.is_open()) {
      fmt::println("open file error");
      exit(-1);
    }
    WriteBinary(out, this->bucket_num_);
    out.write((char*)centroids_, sizeof(DataType) * this->bucket_num_ * this->align_dim_);
    for (auto bid = 0; bid < this->bucket_num_; ++bid) {
      unsigned bucket_size;
      std::memcpy(&bucket_size, id_buckets_[bid], 4);
      WriteBinary(out, bucket_size);
      out.write((char*)(id_buckets_[bid] + 4), sizeof(IDType) * bucket_size);
      out.write((char*)data_buckets_[bid], sizeof(DataType) * bucket_size * this->vec_dim_);
    }
    out.close();
  }

  void Load(const char* kFilePath) override {
    fmt::println("Load IVF index from file: {}", kFilePath);
    std::ifstream in(kFilePath, std::ios::binary);
    if (!in.is_open()) {
      fmt::print("Failed to open file: {}", kFilePath);
      exit(-1);
    }
    ReadBinary(in, this->bucket_num_);
    centroids_ = (DataType*)Alloc64B(sizeof(DataType) * this->bucket_num_ * this->align_dim_);
    in.read((char*)centroids_, sizeof(DataType) * this->bucket_num_ * this->align_dim_);
    data_buckets_.resize(this->bucket_num_);
    id_buckets_.resize(this->bucket_num_);
    for (auto bid = 0; bid < this->bucket_num_; ++bid) {
      unsigned bucket_size;
      ReadBinary(in, bucket_size);
      id_buckets_[bid] = (IDType*)Alloc64B(sizeof(IDType) * bucket_size + 4);
      data_buckets_[bid] = (DataType*)Alloc64B(sizeof(DataType) * bucket_size * this->vec_dim_);
      std::memcpy(id_buckets_[bid], &bucket_size, 4);
      in.read((char*)(id_buckets_[bid] + 4), sizeof(IDType) * bucket_size);
      in.read((char*)data_buckets_[bid], sizeof(DataType) * bucket_size * this->vec_dim_);
    }
    in.close();
  }

  ~IVF() {
    if (centroids_ != nullptr) std::free(centroids_);
    for (auto bid = 0; bid < this->bucket_num_; ++bid) {
      if (data_buckets_[bid] != nullptr) std::free(data_buckets_[bid]);
      if (id_buckets_[bid] != nullptr) std::free(id_buckets_[bid]);
    }
  }

  template <MetricType metric>
  struct Computer {
    constexpr static auto dist_func_ = GetDistFunc<DataType, true>(metric);
    DistFunc<DataType, DataType, DataType> get_dist_;
    int* order_ = nullptr;
    DataType* centroids_dist_ = nullptr;
    const IVF& kIvf_;

    Computer(const IVF& ivf, const DataType* kQuery) : kIvf_(ivf) {
      order_ = (int*)Alloc64B(sizeof(int) * kIvf_.bucket_num_);
      centroids_dist_ = (DataType*)Alloc64B(sizeof(DataType) * kIvf_.bucket_num_);
      for (auto bid = 0; bid < kIvf_.bucket_num_; ++bid) {
        order_[bid] = bid;
        centroids_dist_[bid] =
            dist_func_(kQuery, kIvf_.centroids_ + bid * kIvf_.align_dim_, kIvf_.align_dim_);
      }
      std::sort(order_, order_ + kIvf_.bucket_num_,
                [this](IDType a, IDType b) { return centroids_dist_[a] < centroids_dist_[b]; });
      // fmt::println("\n\n");
      // for (auto i = 0; i < kIvf_.bucket_num_; ++i) {
      //   fmt::println("i:{}, order:{}, dist: {}", i, order_[i], centroids_dist_[order_[i]]);
      // }
    }

    ALWAYS_INLINE
    DataType operator()(std::size_t bid, std::size_t offset, const DataType* kQuery) const {
      return dist_func_(kQuery, kIvf_.data_buckets_[bid] + offset * kIvf_.align_dim_,
                        kIvf_.align_dim_);
    }

    ~Computer() {
      if (order_ != nullptr) {
        std::free(order_);
      }
      if (centroids_dist_ != nullptr) {
        std::free(centroids_dist_);
      }
    }
  };  // struct Computer

  template <MetricType metric>
  auto GetComputer(const DataType* kQuery) const {
    return Computer<metric>(*this, kQuery);
  }
};

}  // namespace alaya