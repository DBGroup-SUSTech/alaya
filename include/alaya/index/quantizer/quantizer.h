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
template <unsigned CodeBits = 8, typename IDType = int64_t, typename DataType = float>
struct Quantizer : Index<IDType, DataType> {
  using CodeType = DependentBitsType<CodeBits>;
  static constexpr auto book_size_ = GetMaxIntegral(CodeBits);
  unsigned book_num_;                ///<
  CodeType* codes_ = nullptr;        ///< Line id for codebook
  std::vector<DataType*> codebook_;  ///<
  DataType* code_dist_ = nullptr;    ///<

  Quantizer() = default;

  Quantizer(int vec_dim, IDType vec_num, MetricType metric)
      : Index<IDType, DataType>(vec_dim, vec_num, metric) {}

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

  /**
   * @brief
   *
   * @param data_id
   * @return DataType*
   */
  virtual DataType* Decode(IDType data_id) { return nullptr; };

  /**
   * @brief Override the () operator, with the input parameter being the vector ID from the dataset,
   * and the return value being the approximate distance obtained by looking up the Distance Table.
   *
   * @param vec_id The vector ID for looking up the approximate distnce.
   * @return DataType
   */
  virtual DataType operator()(IDType vec_id) const = 0;

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