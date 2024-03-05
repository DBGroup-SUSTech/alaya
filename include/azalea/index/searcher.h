#pragma once


#include <memory>
#include "index.h"
#include "index.h
#include "graph/graph.h"
#include "quant/quant.h"
#include "bucket/bucket.h"

namespace azalea {


/**
 * @brief 
 * 
 * @tparam IndexType 
 * @tparam DataType 
 */
template <typename IndexType, typename DataType = float>
struct Searcher {

  std::unique_ptr<IndexType> index_ = nullptr;

  /**
   * @brief Set the index for the searcher
   * 
   * @param index is the index to be set
   */
  virtual void SetIndex(const IndexType& index) = 0;

  /**
   * @brief Pre-optimization process before search
   *        e.g. Determine the number of rows to prefetch during graph retrieval, etc.
   *        It is not supported by all indexes.
   * 
   * @param num_threads Available threads in the current system 
   */
  virtual void Optimize(int num_threads = 0);

  /**
   * @brief Set ef value
   *        It is not supported by all indexes.
   * 
   * @param ef is the number of candidate vertices to be visited by the algorithm.
   */
  virtual void SetEf(int ef);

  /**
   * @brief Search query_num query vectors on the index
   *        return at most k distances and labels.
   * 
   * @param query_num      Number of query vectors
   * @param query_dim      Dimension of query vectors
   * @param queries        Input queries, size query_num * dimension
   * @param k              Number of search result
   * @param distances      Output distances, size query_num * k
   * @param labels         Output labels, size query_num * k
  //  * @param search_params  SearchParameter pointer, defaut is nullptr
   */
  virtual void Search(
    int64_t query_num,
    int64_t query_dim,
    const DataType* queries,
    int64_t k,
    DataType* distances,
    int64_t* labels
    // const SearchParameters* search_params = nullptr
  ) const = 0;

  /**
   * @brief Destroy the Searcher object
   * 
   */
  virtual ~Searcher() = default;

};

// template <typename IndexType, typename IDType, typename CodeType, typename DistType>
// struct Searcher {
//   std::unique_ptr<Graph<IDType, DistType>> graph_ = nullptr;
//   std::unique_ptr<Quantizer<IDType, CodeType, DistType>> quant_ = nullptr;
//   std::unique_ptr<> buckets_ = nullptr;

// };

} // namespace azalea