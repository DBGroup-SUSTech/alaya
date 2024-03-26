#pragma once

#include <alaya/searcher/searcher.h>
#include <alaya/utils/heap.h>
#include <alaya/utils/metric_type.h>
#include "quantizer.h"
#include <alaya/utils/kmeans.h>
#include <sys/wait.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace alaya {

template <unsigned CodeBits = 8, typename IDType = int64_t, typename DataType = float>
struct ScalarQuantizer : Quantizer<CodeBits, IDType, DataType> {
  using CodeType = DependentBitsType<CodeBits>;
  static constexpr auto book_size_ = GetMaxIntegral(CodeBits);
  ScalarQuantizer() = default;
  DataType* min_;
  float* delta_;
  int ivf_num_;
  std::vector<std::vector<IDType>> buckets_;
  ScalarQuantizer(int vec_dim, IDType vec_num, MetricType metric, int clusters, int ivf_num)
      : Quantizer<CodeBits, IDType, DataType>(vec_dim, vec_num, metric), 
        buckets_(clusters, std::vector<IDType>()), ivf_num_(ivf_num) {
    this->vec_dim_ = vec_dim;
    this->book_num_ = clusters;
    this->metric_type_ = metric;
    this->codes_ = (CodeType*)Alloc64B(vec_num * vec_dim * sizeof(CodeType));
    this->codebook_ = (CodeType*)Alloc64B(clusters * vec_dim * sizeof(CodeType));
    this->min_ = (DataType*)Alloc64B(vec_dim * sizeof(DataType));
    this->delta_ = (float*)Alloc64B(vec_dim * sizeof(float));
  }

  DataType DecodeX(int x, CodeType value) {
    return min_[x] + delta_[x] * value;
  }

  DataType* Decode(IDType data_id) { 
    DataType* res = new DataType[this->vec_dim_];
    for (int i = 0; i < this->vec_dim_; i++) {
      res[i] = DecodeX(i, this->codebook_[data_id * this->vec_dim_ + i]);
    }
    return res; 
  }

  DataType* GetCodeWord(IDType data_id) {
    DataType* res = new DataType[this->vec_dim_];
    for (int i = 0; i < this->vec_dim_; i++) {
      res[i] = this->codes_[data_id * this->vec_dim_ + i];
    }
    return res;
  }

  void BuildIndex(IDType vec_num, const DataType* kVecData) {
    for (int i = 0; i < this->vec_dim_; i++) {
      DataType u = std::numeric_limits<DataType>::lowest();
      DataType l = std::numeric_limits<DataType>::max();
      for (int j = 0; j < this->vec_num_; j++) {
        u = std::max(u, kVecData[i * this->vec_dim_ + j]);
        l = std::min(l, kVecData[i * this->vec_dim_ + j]);
      }
      float delta = (float)(u - l) / (1 << CodeBits);
      for (int j = 0; j < this->vec_num_; j++) {
        this->codes_[i * this->vec_dim_ + j] = (uint8_t)((kVecData[i * this->vec_dim_ + j] - l) / delta + 0.5);
      }
      min_[i] = l;
      delta_[i] = delta;
    }
    std::vector<std::vector<DataType>> centroids(this->book_num_, std::vector<CodeType>());
    kmeans(kVecData, vec_num, this->vec_dim_, centroids, this->book_num_, 20);
    for (int i = 0; i < this->book_num_; i++) {
      for (int j = 0; j < this->vec_dim_; j++) {
        this->codebook_[i * this->vec_dim_ + j] = centroids[i][j];
      }
    }
    for (IDType i = 0; i < this->vec_num_; i++) {
      size_t id = -1;
      DataType min_dist = 0;
      for (size_t j = 0; j < centroids.size(); j++) {
        DataType dist = GetDistFunc<DataType, false>(this->metric_type_)(kVecData + i * this->vec_dim_, centroids[j].data(), this->vec_dim_);
        if (id == -1 || min_dist > dist) {
          id = j;
          min_dist = dist;
        }
      }
      buckets_[id].emplace_back(i);
    }
  }

  void Save(const char* kFilePath) const {}

  void Load(const char* kFilePath) {}

  ~ScalarQuantizer()  {
    free(this->codes_);
    free(this->codebook_);
    free(this->min_);
    free(this->delta_);
  }
};

template <typename IndexType, typename DataType = float>
struct SQSearcher : Searcher<IndexType, DataType> {
  void SetIndex(const IndexType& index) {
    this->index_ = std::make_unique(index);
  }
  void Optimize(int num_threads = 0) {}
  void SetEf(int ef) {}
  void Search(
    int64_t query_num,
    int64_t query_dim,
    const DataType* queries,
    int64_t k,
    DataType* distances,
    int64_t* labels
    // const SearchParameters* search_params = nullptr
  ) const {
    const int64_t vec_num = this->index_->vec_num_;
    const unsigned &book_num = this->index_->book_num_;
    const int &level = this->index_->level;
    const DataType* codes = this->index_->codes_;
    const DataType* codebook = this->index_->codebook_;

    for (int64_t qid = 0; qid < query_num; qid++) {
      DataType* q = queries + qid * query_dim;
      DataType* ans_dist = distances + qid * k;
      int64_t* ans_label = labels + qid * k;
      MaxHeap<unsigned, float> centroids(this->index_->ivf_num_);
      for (unsigned i = 0; i < book_num; i++) {
        centroids.Push(i, GetDistFunc<DataType, false>(this->metric_type_)(this->index_->codebook + i * query_dim, q, query_dim));
      }
      LinearPool<int64_t, DataType> pool(k);
      for (int64_t i = 0; i < centroids.Size(); i++) {
        int id = centroids.pool_[i].id_;
        for (auto j : this->index_->buckets_[id]) {
          auto decode_vector = this->index_->Decode(j);
          pool.Insert(j, GetDistFunc<DataType, false>(this->metric_type_)(decode_vector, q, query_dim));
          delete decode_vector;
        }
      }
      for (int64_t i = 0; i < k; i++) {
        ans_dist[i] = pool.pool_[i].dis_;
        ans_dist[i] = pool.pool_[i].id_;
      }
    }
  }
};

}  // namespace alaya