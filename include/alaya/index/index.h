#pragma once

#include <cstdint>
#include <string_view>

#include "../utils/memory.h"
#include "../utils/metric_type.h"

namespace alaya {

/**
 * @brief Abstract structure for an index, supports building index for vectors.
 *
 * @tparam IDType The data type for storing IDs is determined by the number of
 vectors that need to be indexed, with the default type being int64_t.
 * @tparam DataType The data type for vector data, with the default being float.
 */
template <typename DataType = float, typename IDType = int64_t>
struct Index {
  int vec_dim_;             ///< Vector dimension
  int align_dim_ = 0;       ///< aligned Vector dimension
  IDType vec_num_;          ///< The total number of indexed vectors
  MetricType metric_type_;  ///< The type of metric this index uses for index building

  Index() = default;
  /**
   * @brief Construct a new Index object
   *
   * @param dim Dimension of vectors.
   * @param metric_type The type of metric this index uses for index building
   */
  Index(int dim, std::string_view metric) : vec_dim_(dim), metric_type_(kMetricMap[metric]) {}

  Index(int dim, MetricType metric) : vec_dim_(dim), metric_type_(metric) {}

  /**
   * @brief Construct a new Index object
   *
   * @param dim Dimension of vectors.
   * @param align_num Align the vector dimensions to an integer multiple of `align_num`.
   * @param metric_type The type of metric this index uses for index building
   */
  Index(int dim, int align_num, std::string_view metric)
      : vec_dim_(dim), align_dim_(DoAlign(dim, align_num)), metric_type_(kMetricMap[metric]) {}

  Index(int dim, int align_num, MetricType metric)
      : vec_dim_(dim), align_dim_(DoAlign(dim, align_num)), metric_type_(metric) {}

  // /**
  //  * @brief Construct a new Index object
  //  *
  //  * @param dim Dimension of vectors.
  //  * @param num Number of vectors to build index.
  //  * @param metric_type The type of metric this index uses for index building
  //  */
  // explicit Index(int dim, IDType num, MetricType metric)
  //     : vec_dim_(dim), vec_num_(num), metric_type_(metric) {}

  /**
   * @brief Destructor, derived classes need to implement their own object
   * destruction process.
   *
   */
  virtual ~Index(){};

  /**
   * @brief Build index for vectors.
   *
   * @param vec_num  Number of vectors.
   * @param kVecData Pointer to the vectors.
   */
  virtual void BuildIndex(IDType vec_num, const DataType* kVecData) = 0;

  /**
   * @brief Build index for vectors with ids.
   * Not all index need to provide an implementation.
   *
   * @param vec_num  Number of vectors.
   * @param kVecIds  Pointer to the ids of vectors.
   * @param kVecData Pointer to the vectors.
   */
  virtual void BuildIndexWithIds(IDType vec_num, const IDType* kVecIds, const DataType* kVecData){};

  /**
   * @brief Save the index to a file on disk.
   *
   * @param kFilePath File path.
   */
  virtual void Save(const char* kFilePath) const = 0;

  /**
   * @brief Load the index from a file.
   *
   * @param kFilePath File path.
   */
  virtual void Load(const char* kFilePath) = 0;
};

}  // namespace alaya