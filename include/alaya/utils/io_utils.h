#pragma once

#include <fmt/core.h>

#include <filesystem>
#include <fstream>

#include "memory.h"

namespace alaya {
/**
 * @brief
 *
 * @tparam T
 * @param out
 * @param podRef
 */
template <typename T>
void WriteBinary(std::ostream& out, const T& podRef) {
  out.write((char*)&podRef, sizeof(T));
}

template <typename T>
void ReadBinary(std::istream& in, T& podRef) {
  in.read((char*)&podRef, sizeof(T));
}

template <typename DataType>
DataType* LoadVecs(const char* kFileName, unsigned& num, unsigned& dim) {
  fmt::println("Load data from {}", kFileName);
  std::ifstream in(kFileName, std::ios::binary);
  if (!in.is_open()) {
    fmt::println("open file error");
    exit(-1);
  }

  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  std::size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  fmt::println("Data number: {}, data dimension: {}", num, dim);
  DataType* data = new DataType[num * dim * sizeof(float)];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
  return data;
}  // LoadVecs

template <typename DataType>
void LoadVecsDataset(const char* kDatasetPath, DataType*& data, unsigned& d_num,
                     unsigned& d_dim, DataType*& query, unsigned& q_num,
                     unsigned& q_dim) {
  std::filesystem::path dataset_path(kDatasetPath);
  std::string dataset_name = dataset_path.filename().string();
  std::string data_path =
      fmt::format("{}/{}_base.fvecs", kDatasetPath, dataset_name);
  std::string query_path =
      fmt::format("{}/{}_query.fvecs", kDatasetPath, dataset_name);

  data = LoadVecs<DataType>(data_path.c_str(), d_num, d_dim);
  query = LoadVecs<DataType>(query_path.c_str(), q_num, q_dim);
}  // LoadDataset

template <typename DataType>
DataType* AlignLoadVecs(const char* kFileName, unsigned& num, unsigned& dim) {
  fmt::println("Load data from {}", kFileName);
  std::ifstream in(kFileName, std::ios::binary);
  if (!in.is_open()) {
    fmt::println("open file error");
    exit(-1);
  }

  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  unsigned align_dim = DoAlign(dim, 16);
  std::ios::pos_type ss = in.tellg();
  std::size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  fmt::println("Data number: {}, data dimension: {}, align dim: {}", num, dim,
               align_dim);
  DataType* data =
      (DataType*)Alloc64B(std::size_t(num) * align_dim * sizeof(float));

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * align_dim), dim * 4);
  }
  in.close();
  return data;
}  // AlignLoadVecs

template <typename DataType>
void AlignLoadVecsDataset(const char* kDatasetPath, DataType*& data,
                          unsigned& d_num, unsigned& d_dim, DataType*& query,
                          unsigned& q_num, unsigned& q_dim) {
  std::filesystem::path dataset_path(kDatasetPath);
  std::string dataset_name = dataset_path.filename().string();
  std::string data_path =
      fmt::format("{}/{}_base.fvecs", kDatasetPath, dataset_name);
  std::string query_path =
      fmt::format("{}/{}_query.fvecs", kDatasetPath, dataset_name);

  data = AlignLoadVecs<DataType>(data_path.c_str(), d_num, d_dim);
  query = AlignLoadVecs<DataType>(query_path.c_str(), q_num, q_dim);
}  // LoadDataset

}  // namespace alaya