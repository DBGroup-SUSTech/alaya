#pragma once

//#include <cuda_runtime_api.h>
#include <faiss/Clustering.h>
#include <fmt/core.h>

#include <cstddef>
#include <fstream>
#include <limits>
#include <vector>

#include "../../utils/distance.h"
#include "../../utils/io_utils.h"
// #include "../../utils/kmeans.h"
#include "../../utils/memory.h"
#include "../index.h"
#include "quantizer.h"

namespace alaya {

template <unsigned CodeBits = 8, typename IDType = int64_t, typename DataType = float>
struct ProductQuantizer : Quantizer<CodeBits, IDType, DataType> {
  using CodeType = DependentBitsType<CodeBits>;
  static constexpr auto kBookSize_ = GetMaxIntegral(CodeBits);
  std::size_t dist_line_size_;
  DistFunc<DataType, DataType, DataType> dist_func_;
  std::vector<unsigned> sub_dimensions_;
  std::vector<unsigned> sub_vec_start_;

  // std::vector<std::size_t> sub_code_start_;

  ProductQuantizer() = default;

  ProductQuantizer(int vec_dim, IDType vec_num, MetricType metric, unsigned book_num)
      : Quantizer<CodeBits, IDType, DataType>(vec_dim, vec_num, metric),
        sub_dimensions_(book_num, (unsigned)vec_dim / book_num),
        sub_vec_start_(book_num, 0),
        dist_func_(GetDistFunc<DataType, false>(metric)) {
    this->book_num_ = book_num;  // 4, 8, 16 ...

    dist_line_size_ = book_num * kBookSize_;

    unsigned reminder = this->vec_dim_ % book_num;
    if (reminder) {
      sub_dimensions_[book_num - 1] += reminder;
    }

    // codes_ = vec_num * book_num_
    this->codes_ = (CodeType*)Alloc64B(sizeof(CodeType) * vec_num * book_num);
    // code_dist_ = kBookSize_ * book_num
    this->code_dist_ = (DataType*)Alloc64B(sizeof(DataType) * kBookSize_ * book_num);
    // Each codebook has sub_dimensions_ * kBookSize_ elements
    this->codebook_.resize(book_num);
    for (std::size_t i = 0; i < book_num; i++) {
      this->codebook_[i] = (DataType*)Alloc64B(sizeof(DataType) * sub_dimensions_[i] * kBookSize_);
    }
    for (unsigned i = 1; i < book_num; ++i) {
      sub_vec_start_[i] = sub_vec_start_[i - 1] + sub_dimensions_[i - 1];
    }
  }

  void BuildIndex(IDType vec_num, const DataType* kVecData) override {
    std::vector<std::vector<DataType>> sub_vec(this->book_num_);

    for (auto i = 0; i < this->book_num_; ++i) {
      sub_vec[i].resize(vec_num * sub_dimensions_[i]);
    }

    for (auto i = 0; i < vec_num; ++i) {
      for (auto j = 0; j < this->book_num_; ++j) {
        std::memcpy(sub_vec[j].data() + i * sub_dimensions_[j],
                    kVecData + i * this->vec_dim_ + sub_vec_start_[j],
                    sub_dimensions_[j] * sizeof(DataType));
      }
    }

    for (auto i = 0; i < this->book_num_; ++i) {
      // kmeans(sub_vec[i].data(), vec_num, sub_dimensions_[i], centroids[i], kBookSize_, 20);
      auto quan_err = faiss::kmeans_clustering(sub_dimensions_[i], vec_num, kBookSize_,
                                               sub_vec[i].data(), (float*)this->codebook_[i]);
      fmt::println("book:{}, quan_err: {}", i, quan_err);
    }

    // Init codes
    for (auto i = 0; i < this->book_num_; i++) {
      for (auto j = 0; j < vec_num; j++) {
        float min_dist = std::numeric_limits<float>::max();
        for (auto k = 0; k < kBookSize_; k++) {
          float dist = dist_func_(sub_vec[i].data() + j * sub_dimensions_[i],
                                  this->codebook_[i] + k * sub_dimensions_[i], sub_dimensions_[i]);
          if (dist < min_dist) {
            this->codes_[j * this->book_num_ + i] = k;
            min_dist = dist;
          }
        }
      }
    }  // Init codes
  }    // BuildIndex

  void SetDistFunc(MetricType metric) { dist_func_ = GetDistFunc<DataType, false>(metric); }

  DataType operator()(IDType vec_id) const override {
    DataType res = 0;
    CodeType* codes = this->codes_ + vec_id * this->book_num_;
    for (auto i = 0; i < this->book_num_; ++i) {
      PrefetchL1(this->code_dist_ + i * dist_line_size_ + codes[i]);
    }
    for (auto i = 0; i < this->book_num_; ++i) {
      res += this->code_dist_[i * dist_line_size_ + codes[i]];
    }
    return res;
  }

  DataType* Decode(IDType data_id) override { return nullptr; }

  DataType* GetCodeWord(IDType data_id) override {
    DataType* vec = new DataType[this->vec_dim_];
    for (auto i = 0; i < this->book_num_; ++i) {
      std::memcpy(
          vec + i * sub_dimensions_[i],
          this->codebook_[i] + this->codes_[data_id * this->book_num_ + i] * sub_dimensions_[i],
          sub_dimensions_[i] * sizeof(DataType));
    }
    return vec;
  }

  void Save(const char* kFilePath) const override {
    fmt::println("Save PQ index to {}", kFilePath);
    std::ofstream out(kFilePath, std::ios::binary);
    if (!out.is_open()) {
      fmt::println("open file error");
      exit(-1);
    }
    unsigned book_num = this->book_num_;
    WriteBinary(out, book_num);
    unsigned book_size = this->kBookSize_;
    WriteBinary(out, book_size);
    for (unsigned i = 0; i < book_num; ++i) {
      out.write((char*)this->codebook_[i], sizeof(DataType) * sub_dimensions_[i] * kBookSize_);
    }
    out.write((char*)this->codes_, sizeof(CodeType) * this->vec_num_ * book_num);
    out.close();
  }  // Save

  void Load(const char* kFilePath) override {
    fmt::println("Load PQ index from {}", kFilePath);
    std::ifstream in(kFilePath, std::ios::binary);
    if (!in.is_open()) {
      fmt::println("open file error");
      exit(-1);
    }
    unsigned book_num;
    ReadBinary(in, book_num);
    this->book_num_ = book_num;

    unsigned book_size;
    ReadBinary(in, book_size);
    if (this->kBookSize_ != book_size) {
      fmt::println("book_size not match, inference size: {}, file size: {}", this->kBookSize_,
                   book_size);
      exit(-1);
    }

    this->sub_dimensions_.resize(book_num);
    this->codebook_.resize(book_num);
    for (auto i = 0; i < book_num; ++i) {
      this->sub_dimensions_[i] = this->vec_dim_ / book_num;
      this->codebook_[i] = (DataType*)Alloc64B(sizeof(DataType) * sub_dimensions_[i] * kBookSize_);
    }
    unsigned reminder = this->vec_dim_ % book_num;
    if (reminder) {
      sub_dimensions_[book_num - 1] += reminder;
    }
    this->codes_ = (CodeType*)Alloc64B(sizeof(CodeType) * this->vec_num_ * this->book_num_);
    for (auto i = 0; i < book_num; ++i) {
      in.read((char*)this->codebook_[i], sizeof(DataType) * sub_dimensions_[i] * kBookSize_);
    }
    in.read((char*)this->codes_, sizeof(CodeType) * this->vec_num_ * this->book_num_);
    in.close();
  }  // Load

  ~ProductQuantizer() override {
    if (this->codes_) {
      std::free(this->codes_);
    }
    if (this->code_dist_) {
      std::free(this->code_dist_);
    }
    if (this->codebook_.size()) {
      for (auto& cb : this->codebook_) {
        if (cb) {
          std::free(cb);
        }
      }
    }
  }  // ~ProductQuantizer
};

}  // namespace alaya