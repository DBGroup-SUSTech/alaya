#pragma once

#include <cstdint>
#include "../index.h"

namespace azalea {

template <typename IDType, typename CodeType, typename DistType>
struct Quantizer : Index<IDType, DistType> {
  DistType* v_data_;
  int book_num_, book_size_;
  uint16_t bits_;
  CodeType* codes_;
  DistType* codebook_;
  DistType* code_dist_;

  void init_search(DistType* query);

  DistType compute(int data_id);

};

} // namespace azalea