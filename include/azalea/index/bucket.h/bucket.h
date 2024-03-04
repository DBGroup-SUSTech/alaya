#pragma once 

#include <vector>
#include "../index.h"

namespace azalea {

template <typename IDType, typename DistType, typename ItemType>
struct Bucket : Index<IDType, DistType> {
  int bucket_num_;
  std::vector<std::vector<ItemType>> buckets_;
  std::vector<std::vector<IDType>> id_buckets_;

  std::vector<std::pair<int, DistType>> order_list_;

  void init_search(DistType* query);
};

} // namespace azalea