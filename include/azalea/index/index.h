#pragma once

#include <cstdint>
#include <cstdio>
#include "MetricType.h"

namespace azalea {

/**
 * @brief Specify more comprehensive search parameters
 * 
 */
struct SearchParameters {

};

/**
 * @brief Abstract structure for an index, supports building index for vectors.
 * 
 * @tparam IDType The data type for storing IDs is determined by the number of vectors 
           that need to be indexed, with the default type being int64_t.
 * @tparam DataType The data type for vector data, with the default being float.
 */
template < typename DataType = float, typename IDType = int64_t>
struct Index {
  int dim_;                ///< vector dimension
  IDType num_;             ///< The total number of indexed vectors
  MetricType metric_type_; ///< The type of metric this index uses for index building

  /**
   * @brief Construct a new Index object
   * 
   * @param dim Dimension of vectors.
   * @param num Number of vectors to build index.
   * @param metric_type The type of metric this index uses for index building
   */
  explicit
  Index(int dim, IDType num, MetricType metric)
    : dim_(dim),
      num_(num),
      metric_type_(metric) {}

  virtual ~Index();

  /**
   * @brief Build index for vectors.
   * 
   * @param vec_num  Number of vectors.
   * @param kVecData Pointer to the vectors.
   */
  virtual void BuildIndex(IDType vec_num, const DataType* kVecData) = 0;

  /**
   * @brief Build index for vectors with ids.
   * 
   * @param vec_num  Number of vectors.
   * @param kVecIds  Pointer to the ids of vectors.
   * @param kVecData Pointer to the vectors.
   */
  virtual void BuildIndexWithIds(IDType vec_num, const IDType* kVecIds, const DataType* kVecData);

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


  // virtual void add(IDType n, const DistType* vec) = 0;

  // DistType operator() (int data_id) const {
  //   return 0;
  // }

  // template<typename... Types>
  // void search(
  //   IDType n,
  //   const DistType* q,
  //   IDType k,
  //   DistType* dist,
  //   IDType* idx,
  //   Types... args) const {
  //     do_search(n, q, k, dist, idx, args...);
  // }

  // virtual void do_search() const = 0;
};

} // namespace azalea