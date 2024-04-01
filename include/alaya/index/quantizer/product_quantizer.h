#pragma once

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
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

template <unsigned CodeBits = 8, typename DataType = float, typename IDType = int64_t>
struct ProductQuantizer : Quantizer<CodeBits, DataType, IDType> {
  using CodeType = DependentBitsType<CodeBits>;
  constexpr static auto kBookSize_ = GetMaxIntegral(CodeBits);
  // const unsigned kBookSize_ = GetMaxIntegral(CodeBits);
  // int algin_dim_;
  std::size_t dist_line_size_;
  DistFunc<DataType, DataType, DataType> dist_func_;
  std::vector<unsigned> sub_dimensions_;
  std::vector<unsigned> sub_vec_start_;
  DataType* encode_vecs_ = nullptr;

  ProductQuantizer() = default;

  ProductQuantizer(int vec_dim, MetricType metric, unsigned book_num)
      : Quantizer<CodeBits, DataType, IDType>(vec_dim, kAlgin16, metric),
        sub_dimensions_(book_num, (unsigned)vec_dim / book_num),
        sub_vec_start_(book_num, 0),
        dist_func_(GetDistFunc<DataType, false>(metric)) {
    this->book_num_ = book_num;  // 4, 8, 16 ...

    dist_line_size_ = book_num * kBookSize_;

    unsigned reminder = this->vec_dim_ % book_num;
    if (reminder) {
      sub_dimensions_[book_num - 1] += reminder;
    }

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

  // ProductQuantizer(ProductQuantizer&& other) noexcept {}

  void BuildIndex(IDType vec_num, const DataType* kVecData) override {
    this->vec_num_ = vec_num;
    // codes_ = vec_num * book_num_
    this->codes_ = (CodeType*)Alloc64B(sizeof(CodeType) * vec_num * this->book_num_);

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

  void InitCodeDist(const DataType* kQuery) {
    auto book_size = this->book_size_;
    auto sub_dim = this->sub_dimensions_;
    auto sub_start = this->sub_vec_start_;
    auto codebook = this->codebook_;
    auto code_dist = this->code_dist_;
    for (auto i = 0; i < this->book_num; ++i) {
      Sgemv(kQuery + sub_start[i], codebook[i].data(), code_dist + i * book_size, sub_dim[i],
            book_size);
      // VecMatMul(kQuery + sub_start[i], codebook[i].data(), code_dist + i * book_size, sub_dim[i],
      //           book_size);
    }
  }

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

  DataType operator()(IDType vec_id, const DataType* kQuery) const {
    return dist_func_(kQuery, encode_vecs_ + vec_id * this->algin_dim_, this->algin_dim_);
  }

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

  void Encode() override {
    if (encode_vecs_) return;
    encode_vecs_ = (DataType*)Alloc2M(sizeof(DataType) * this->vec_num_ * this->align_dim_);
#pragma omp parallel for schedule(static)
    for (auto i = 0; i < this->vec_num_; ++i) {
      for (auto j = 0; j < this->book_num_; ++j) {
        std::memcpy(encode_vecs_ + i * this->align_dim_ + sub_vec_start_[j],
                    this->codebook_[j] + this->codes_[i * this->book_num_ + j] * sub_dimensions_[j],
                    sub_dimensions_[j] * sizeof(DataType));
      }
      // DataType* vec = GetCodeWord(i);
      // std::memcpy(encode_vecs_ + i * algin_dim_, vec, this->vec_dim_ * sizeof(DataType));
      // delete[] vec;
    }
  }

  DataType* Decode(IDType data_id) override { return nullptr; }

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