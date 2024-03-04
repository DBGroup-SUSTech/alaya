#pragma once

#include <cstdint>
#include <cstdio>
#include "MetricType.h"

namespace azalea {

// using id_t = uint64_t;
template <typename IDType, typename DistType>
struct Index {
  int dim_;
  IDType num_;

  MetricType metric_type_;

  Index(int dim, IDType num, MetricType metric_type)
      : dim_(dim),
        num_(num),
        metric_type_(metric_type) {}

  virtual ~Index();

  virtual void add(IDType n, const DistType* vec) = 0;

  virtual void train(IDType n, const DistType* x) = 0;

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