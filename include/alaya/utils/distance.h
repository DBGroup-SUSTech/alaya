#pragma once

namespace azalea {

template <typename T1, typename T2, typename U, typename... Params>
using Dist = U (*)(const T1 *, const T2 *, int, Params...);


template <typename DistType>
inline DistType l2_dist(
  const DistType *x,
  const DistType *y,
  int d) {}

template <typename DistType>
inline DistType ip_dist(
  const DistType *x,
  const DistType *y,
  int d) {}

template <typename DistType>
inline DistType cos_dist(
  const DistType *x,
  const DistType *y,
  int d) {}

} // namespace azalea