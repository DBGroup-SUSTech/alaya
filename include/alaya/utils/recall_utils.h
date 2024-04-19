#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace alaya {

template <typename DataType = float>
float CalRecallByVal(const DataType* kResVal, const uint32_t kNum, const uint32_t kK,
                     const DataType* kGtVal, const uint32_t kGtLine) {
  float exact_num = 0;
  constexpr float kErrBound = 1e-6;
  for (auto n = 0; n < kNum; ++n) {
    std::vector<bool> visted(kK, false);
    for (uint32_t i = 0; i < kK; ++i) {
      for (uint32_t j = 0; j < kK; ++j) {
        if (!visted[j] && std::fabs(kResVal[n * kK + i] - kGtVal[n * kGtLine + j]) < kErrBound) {
          exact_num++;
          visted[j] = true;
          break;
        }
      }
    }
  }
  return exact_num / (kNum * kK);
}

template <typename IDType = uint32_t>
float CalRecallById(const IDType* kResVal, const uint32_t kNum, const uint32_t kK,
                    const IDType* kGtVal, const uint32_t kGtLine) {
  float exact_num = 0;
  for (auto n = 0; n < kNum; ++n) {
    std::vector<bool> visted(kK, false);
    for (uint32_t i = 0; i < kK; ++i) {
      for (uint32_t j = 0; j < kK; ++j) {
        if (!visted[j] && kResVal[n * kK + i] == kGtVal[n * kGtLine + j]) {
          exact_num++;
          visted[j] = true;
          break;
        }
      }
    }
  }
  return exact_num / (kNum * kK);
}
}  // namespace alaya