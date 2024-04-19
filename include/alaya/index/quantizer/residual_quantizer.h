#pragma once

// #include <alaya/searcher/searcher.h>
#include <alaya/utils/kmeans.h>
#include <alaya/utils/metric_type.h>
#include <alaya/utils/pool.h>
#include <sys/wait.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../searcher/searcher.h"
#include "quantizer.h"

namespace alaya {

template <unsigned CodeBits = 8, typename IDType = int64_t, typename DataType = float>
struct ResidualQuantizer : Quantizer<CodeBits, IDType, DataType> {
  using CodeType = DependentBitsType<CodeBits>;
  static constexpr auto book_size_ = GetMaxIntegral(CodeBits);
  ResidualQuantizer() = default;
  DataType* vec_l2sqr_;
  int level_;
  ResidualQuantizer(int vec_dim, IDType vec_num, MetricType metric, unsigned book_num, int level)
      : Quantizer<CodeBits, IDType, DataType>(vec_dim, vec_num, metric) {
    this->vec_dim_ = vec_dim;
    this->book_num_ = book_num;
    this->codes_ = (CodeType*)Alloc64B(vec_num * level * sizeof(CodeType));
    this->codebook_ = (DataType*)Alloc64B(book_num * vec_dim * sizeof(DataType));
    this->code_dist_ = (DataType*)Alloc64B(book_num * vec_dim * sizeof(DataType));
    if (metric == MetricType::L2) {
      this->vec_l2sqr_ = (DataType*)Alloc64B(vec_num * sizeof(DataType));
    }
    this->level_ = level;
  }

  DataType* Decode(IDType data_id) {
    DataType* res = new DataType[this->vec_dim_ * level_];
    for (int i = 0; i < level_; i++) {
      int start = this->vec_dim_ * this->codes_[data_id * level_ + i];
      for (int j = 0; j < this->vec_dim_; j++) {
        res[i * this->vec_dim_ + j] = this->codebook_[start + j];
      }
    }
    return res;
  }

  DataType* GetCodeWord(IDType data_id) {
    DataType* res = new DataType[level_];
    for (int i = 0; i < level_; i++) {
      res[i] = this->codes_[data_id * level_ + i];
    }
    return res;
  }

  void BuildIndex(IDType vec_num, const DataType* kVecData) {
    DataType* vec_data = (DataType*)malloc(vec_num * this->vec_dim_ * sizeof(DataType));
    memcpy(vec_data, kVecData, vec_num * this->vec_dim_ * sizeof(DataType));
    if (this->metric_type_ == MetricType::L2) {
      for (IDType i = 0; i < vec_num; i++) {
        this->vec_l2sqr_[i] = NormSqrFloat(kVecData + this->vec_dim_ * i, this->vec_dim_);
      }
    }
    if (this->metric_type_ == MetricType::COS) {
      for (auto i = 0; i < vec_num; i++) {
        float norm = NormSqrTFloat(vec_data + this->vec_dim_ * i, this->vec_dim_);
        for (int j = 0; j < this->vec_dim_; j++) {
          vec_data[this->vec_dim_ * i + j] /= norm;
        }
      }
    }
    for (int lev = 0; lev < level_; lev++) {
      unsigned int cluster_num = this->book_num_ / level_;
      std::vector<std::vector<float>> centroids(cluster_num, std::vector<float>());
      kmeans(vec_data, vec_num, this->vec_dim_, centroids, cluster_num, 20);
      for (IDType i = 0; i < vec_num; i++) {
        size_t id = -1;
        float min_dist = 0;
        for (size_t j = 0; j < centroids.size(); j++) {
          float dist =
              L2Sqr<float>(vec_data + i * this->vec_dim_, centroids[j].data(), this->vec_dim_);
          if (id == -1 || min_dist > dist) {
            id = j;
            min_dist = dist;
          }
        }
        this->codes_[lev + i * level_] = id;
        for (size_t j = 0; j < this->vec_dim_; j++) {
          vec_data[i * this->vec_dim_ + j] -= centroids[id][j];
        }
      }
      for (size_t st = lev * centroids.size() * this->vec_dim_, i = 0; i < centroids.size(); i++) {
        for (size_t j = 0; j < centroids[i].size(); j++) {
          this->codebook_[st + i * centroids.size() + j] = centroids[i][j];
        }
      }
    }
    free(vec_data);
  }

  void Save(const char* kFilePath) const {}

  void Load(const char* kFilePath) {}

  ~ResidualQuantizer() {
    free(this->codes_);
    free(this->codebook_);
    free(this->code_dist_);
  }
};

template <typename IndexType, typename DataType = float>
struct RQSearcher : Searcher<IndexType, DataType> {
  void SetIndex(const IndexType& index) { this->index_ = std::make_unique(index); }
  void Optimize(int num_threads = 0) {}
  void SetEf(int ef) {}
  void Search(int64_t query_num, int64_t query_dim, const DataType* queries, int64_t k,
              DataType* distances, int64_t* labels
              // const SearchParameters* search_params = nullptr
  ) const override {
    const int64_t vec_num = this->index_->vec_num_;
    const unsigned& book_num = this->index_->book_num_;
    const int& level = this->index_->level;
    const DataType* codes = this->index_->codes_;
    const DataType* codebook = this->index_->codebook_;
    const DataType* code_dist = this->index_->code_dist_;

    for (int64_t qid = 0; qid < query_num; qid++) {
      DataType* q = queries + qid * query_dim;
      DataType* ans_dist = distances + qid * k;
      int64_t* ans_label = labels + qid * k;
      for (unsigned i = 0; i < book_num; i++) {
        code_dist[i] = InnerProduct<DataType>(codebook[i], q, query_dim);
      }
      LinearPool<int64_t, DataType> pool(k);
      for (int64_t i = 0; i < vec_num; i++) {
        DataType appr_dist = 0;
        if (this->index_->memetric_type_ == MetricType::L2) {
          appr_dist = this->index_->vec_l2sqr_[i] + InnerProduct<DataType>(q, q, query_dim);
        }
        for (int j = 0; j < level; j++) {
          appr_dist += code_dist[codes[i * level + j]];
        }
        if (this->index_->memetric_type_ == MetricType::COS) {
          appr_dist /= NormSqrTFloat(q, query_dim);
        }
        pool.Insert(i, appr_dist);
      }
      for (int64_t i = 0; i < k; i++) {
        ans_dist[i] = pool.pool_[i].dis_;
        ans_label[i] = pool.pool_[i].id_;
      }
    }
  }
};
}  // namespace alaya