#pragma once 

#include <vector>
#include "../index.h"

namespace alaya {

template <typename IDType, typename DataType, typename ItemType>
struct Bucket : Index<IDType, DataType> {
  int bucket_num_;
  std::vector<std::vector<ItemType>> buckets_;
  std::vector<std::vector<IDType>> id_buckets_;

  std::vector<std::pair<int, DataType>> order_list_;

};

} // namespace alaya