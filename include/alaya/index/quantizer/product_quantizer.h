#pragma once

#include <fmt/core.h>

#include <cstddef>
#include <fstream>
#include <limits>
#include <vector>

#include "../../utils/distance.h"
#include "../../utils/io_utils.h"
#include "../../utils/kmeans.h"
#include "../../utils/memory.h"
#include "../index.h"
#include "quantizer.h"

namespace alaya {

template <unsigned CodeBits = 8, typename IDType = int64_t, typename DataType = float>
struct ProductQuantizer : Quantizer<CodeBits, IDType, DataType> {
  using CodeType = DependentBitsType<CodeBits>;
  static constexpr auto book_size_ = GetMaxIntegral(CodeBits);
  DistFunc<DataType, DataType, DataType> build_dist_func_;
  std::vector<unsigned> sub_dimensions_;
  std::vector<unsigned> sub_vec_start_;
  // std::vector<std::size_t> sub_code_start_;

  ProductQuantizer() = default;

  ProductQuantizer(int vec_dim, IDType vec_num, MetricType metric, unsigned book_num)
      : Quantizer<CodeBits, IDType, DataType>(vec_dim, vec_num, metric),
        sub_dimensions_(book_num, (unsigned)vec_dim / book_num),
        sub_vec_start_(book_num, 0),
        build_dist_func_(GetDistFunc<DataType, false>(metric)) {
    this->book_num_ = book_num;  // 4, 8, 16 ...

    unsigned reminder = this->vec_dim_ % book_num;
    if (reminder) {
      sub_dimensions_[book_num - 1] += reminder;
    }

    // codes_ = vec_num * book_num_
    this->codes_ = (CodeType*)Alloc64B(sizeof(CodeType) * vec_num * book_num);
    // code_dist_ = book_size_ * book_num
    this->code_dist_ = (DataType*)Alloc64B(sizeof(DataType) * book_size_ * book_num);
    // Each codebook has sub_dimensions_ * book_size_ elements
    this->codebook_.resize(book_num);
    for (std::size_t i = 0; i < book_num; i++) {
      this->codebook_[i] = (DataType*)Alloc64B(sizeof(DataType) * sub_dimensions_[i] * book_size_);
    }
    for (unsigned i = 1; i < book_num; ++i) {
      sub_vec_start_[i] = sub_vec_start_[i - 1] + sub_dimensions_[i - 1];
    }
  }

  void BuildIndex(IDType vec_num, const DataType* kVecData) override {
    std::vector<std::vector<DataType>> sub_vec(this->book_num_);
    std::vector<std::vector<std::vector<float>>> centroids(
        this->book_num_, std::vector<std::vector<float>>(book_size_));
    // std::vector<unsigned> start_local(this->book_num_);

    // for (unsigned i = 0, dim_start = 0; i < this->book_num_; ++i) {
    for (unsigned i = 0; i < this->book_num_; ++i) {
      sub_vec[i].resize(vec_num * sub_dimensions_[i]);
      // start_local[i] = dim_start;
      // dim_start += sub_dimensions_[i];
    }

    for (int i = 0; i < vec_num; ++i) {
      for (int j = 0; j < this->book_num_; ++j) {
        std::memcpy(sub_vec[j].data() + i * sub_dimensions_[j],
                    kVecData + i * this->vec_dim_ + sub_vec_start_[j],
                    sub_dimensions_[j] * sizeof(DataType));
      }
    }

    for (unsigned i = 0; i < this->book_num_; ++i) {
      kmeans(sub_vec[i].data(), vec_num, sub_dimensions_[i], centroids[i], book_size_, 20);
    }

    // Init codebook
    for (unsigned i = 0; i < this->book_num_; ++i) {
      for (unsigned j = 0; j < book_size_; ++j) {
        std::memcpy(this->codebook_[i] + j * sub_dimensions_[i], centroids[i][j].data(),
                    sub_dimensions_[i] * sizeof(DataType));
      }
    }

    // Init codes
    for (unsigned i = 0; i < vec_num; ++i) {            // Each Vector
      for (unsigned j = 0; j < this->book_num_; ++j) {  // Each Codebook
        float min_dist = std::numeric_limits<float>::max();
        for (unsigned k = 0; k < book_size_; ++k) {
          float dist = build_dist_func_(sub_vec[j].data() + i * sub_dimensions_[j],
                                        centroids[j][k].data(), sub_dimensions_[j]);
          if (dist < min_dist) {
            // codes = vec_num * book_num
            this->codes_[i * this->book_num_ + j] = k;
            min_dist = dist;
          }
        }
      }
    }  // Init codes
  }    // BuildIndex

  DataType* Decode(IDType data_id) override {
    // DataType* vec = new DataType[this->vec_dim_];
    // for (unsigned i = 0; i < this->book_num_; ++i) {
    //   std::memcpy(
    //       vec + i * sub_dimensions_[i],
    //       this->codebook_[i] +
    //           this->codes_[data_id * this->book_num_ + i] *
    //           sub_dimensions_[i],
    //       sub_dimensions_[i] * sizeof(DataType));
    // }
    return nullptr;
  }

  DataType* GetCodeWord(IDType data_id) override {
    DataType* vec = new DataType[this->vec_dim_];
    for (unsigned i = 0; i < this->book_num_; ++i) {
      std::memcpy(
          vec + i * sub_dimensions_[i],
          this->codebook_[i] + this->codes_[data_id * this->book_num_ + i] * sub_dimensions_[i],
          sub_dimensions_[i] * sizeof(DataType));
    }
    return nullptr;
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
    unsigned book_size = this->book_size_;
    WriteBinary(out, book_size);
    for (unsigned i = 0; i < book_num; ++i) {
      out.write((char*)this->codebook_[i], sizeof(DataType) * sub_dimensions_[i] * book_size_);
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
    if (this->book_size_ != book_size) {
      fmt::println("book_size not match, inference size: {}, file size: {}", this->book_size_,
                   book_size);
      exit(-1);
    }

    this->sub_dimensions_.resize(book_num);
    this->codebook_.resize(book_num);
    for (unsigned i = 0; i < book_num; ++i) {
      this->sub_dimensions_[i] = this->vec_dim_ / book_num;
      this->codebook_[i] = (DataType*)Alloc64B(sizeof(DataType) * sub_dimensions_[i] * book_size_);
    }
    unsigned reminder = this->vec_dim_ % book_num;
    if (reminder) {
      sub_dimensions_[book_num - 1] += reminder;
    }
    this->codes_ = (CodeType*)Alloc64B(sizeof(CodeType) * this->vec_num_ * this->book_num_);
    for (unsigned i = 0; i < book_num; ++i) {
      in.read((char*)this->codebook_[i], sizeof(DataType) * sub_dimensions_[i] * book_size_);
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