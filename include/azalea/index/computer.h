#pragma once

#include "index.h"

namespace azaela {

template <typename IndexType, typename DistType>
struct Computer {
  const IndexType &index;
  DistType* query;

  Computer(const IndexType &index, DistType* query)
    : index(index), query(query) {}

  DistType operator() (int data_id) const {
    return index(data_id);
  }

};

} // namespace azalea