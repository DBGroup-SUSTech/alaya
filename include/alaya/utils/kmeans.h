#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include "distance.h"
#include "random_utils.h"

namespace alaya {

void kmeans(const float* kData, const std::size_t kDataNum, const std::size_t kDataDim,
            std::vector<std::vector<float>>& centroids, unsigned& cluster_num,
            const unsigned kKMeansIter);

}  // namespace alaya