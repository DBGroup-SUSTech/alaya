#pragma once

#include <cstdint>

#include "../../utils/distance.h"
#include "../../utils/kmeans.h"
#include "../../utils/memory.h"
#include "../../utils/type_utils.h"
#include "../index.h"

namespace alaya {

/**
 * @brief
 *
 * @tparam IDType
 * @tparam CodeType
 * @tparam DataType
 */
template <unsigned CodeBits = 8, typename DataType = float, typename IDType = int64_t>
struct Quantizer : Index<DataType, IDType> {
  using CodeType = DependentBitsType<CodeBits>;
  static constexpr auto book_size_ = GetMaxIntegral(CodeBits);
  unsigned book_num_;                ///<
  CodeType* codes_ = nullptr;        ///< Line id for codebook
  std::vector<DataType*> codebook_;  ///<
  DataType* code_dist_ = nullptr;    ///<

  Quantizer() = default;

  Quantizer(int vec_dim, MetricType metric) : Index<DataType, IDType>(vec_dim, metric) {}

  Quantizer(int vec_dim, int align_num, MetricType metric)
      : Index<DataType, IDType>(vec_dim, align_num, metric) {}

  /**
   * @brief
   *
   * @tparam Pool
   * @param pool
   * @param query
   * @param data
   * @param k
   * @param distances
   * @param labels
   */
  template <typename Pool>
  void Reorder(const Pool& pool, const DataType* query, const DataType* data, int64_t k,
               DataType* distances, int64_t* labels){};

  virtual void Encode() {}

  /**
   * @brief
   *
   * @param data_id
   * @return DataType*
   */
  virtual DataType* Decode(IDType data_id) { return nullptr; };

//  /**
//   * @brief Override the () operator, with the input parameter being the vector ID from the dataset,
//   * and the return value being the approximate distance obtained by looking up the Distance Table.
//   *
//   * @param vec_id The vector ID for looking up the approximate distnce.
//   * @return DataType
//   */
//  virtual DataType operator()(IDType vec_id) const = 0;

  // 考虑到quantizer需要并发处理多个查询，对于每个查询，返回一个结构体
  // 结构体提供“DataType operator()(IDType vec_id)”接口进行距离计算
  auto get_computer(const DataType* query) const = 0;

  /**
   * @brief
   *
   * @param data_id
   * @return DataType*
   */
  virtual DataType* GetCodeWord(IDType data_id) = 0;

  virtual ~Quantizer() override{};
};

}  // namespace alaya