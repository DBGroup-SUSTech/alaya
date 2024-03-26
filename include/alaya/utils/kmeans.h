#pragma once

#include <fmt/core.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace alaya {

void kmeans(const float* kData, const std::size_t kDataNum, const std::size_t kDataDim,
            std::vector<std::vector<float>>& centroids, unsigned& cluster_num,
            const unsigned kKMeansIter);

template <typename DataType, typename IDType>
void kmeans(const DataType* kData, const IDType kDataNum, const int kDataDim,
            std::vector<std::vector<DataType>>& centroids, const IDType kClusterNum,
            bool init_centroids_norm, const unsigned int kKMeansIter);

}  // namespace alaya