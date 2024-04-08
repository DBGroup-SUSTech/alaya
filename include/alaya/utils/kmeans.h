#pragma once

#include <fmt/core.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

#include "distance.h"
#include "metric_type.h"

namespace alaya {

template <typename DataType, typename IDType>
std::vector<std::vector<IDType>> Assign(const DataType* kVecData, const IDType kVecNum,
                                        const int kVecDim, const DataType* kCentroid,
                                        const int kCentroidNum, MetricType metric) {
  DistFunc<DataType, DataType, DataType> dist_func;
  if (metric == MetricType::IP) {
    dist_func = GetDistFunc<DataType, false>(MetricType::IP);
  } else {
    dist_func = GetDistFunc<DataType, false>(MetricType::L2);
  }
  std::vector<std::vector<IDType>> ids(kCentroidNum);
  std::vector<IDType> cid4data(kVecNum);
#pragma parallel for
  for (auto vid = 0; vid < kVecNum; ++vid) {
    float min_dist = std::numeric_limits<float>::max();
    for (auto cid = 0; cid < kCentroidNum; ++cid) {
      float dist = dist_func(kVecData + vid * kVecDim, kCentroid + cid * kVecDim, kVecDim);
      if (dist < min_dist) {
        cid4data[vid] = cid;
        min_dist = dist;
      }
    }
  }

  for (auto vid = 0; vid < kVecNum; ++vid) {
    ids[cid4data[vid]].emplace_back(vid);
  }

  return ids;
}

std::vector<float> faiss_kmeans(const float* kData, const std::size_t kDataNum,
                                const std::size_t kDataDim, unsigned cluster_num,
                                MetricType metric);

void kmeans(const float* kData, const std::size_t kDataNum, const std::size_t kDataDim,
            std::vector<std::vector<float>>& centroids, unsigned cluster_num,
            const unsigned kKMeansIter);

// template <typename DataType, typename IDType = unsigned>
// void InitCentroidsByNorm(const DataType* kData, const IDType kDataNum, const IDType kDataDim,
//                          std::vector<std::vector<DataType>>& centroids,
//                          const unsigned kClusterNum) {
//   std::vector<std::pair<DataType, unsigned>> norm_id(kDataNum);
// #pragma omp parallel for
//   for (std::size_t i = 0; i < kDataNum; ++i) {
//     norm_id[i].first = GetSqrNorm(kData + i * kDataDim, kDataDim);
//     norm_id[i].second = i;
//   }
//   std::sort(norm_id.begin(), norm_id.end());

//   std::size_t interval = (std::size_t)kDataNum / kClusterNum;
// #pragma omp parallel for
//   for (std::size_t i = 0; i < kClusterNum; ++i) {
//     centroids[i].assign(kData + norm_id[i * interval].second * kDataDim,
//                         kData + norm_id[i * interval].second * kDataDim + kDataDim);
//   }
// }

// template <typename DataType, typename IDType = unsigned>
// void kmeans(const DataType* kData, const IDType kDataNum, const int kDataDim,
//             std::vector<std::vector<DataType>>& centroids, const IDType kClusterNum,
//             bool init_centroids_norm, const unsigned int kKMeansIter) {
//   unsigned interval = kDataNum / kClusterNum;
//   std::vector<std::vector<unsigned>> id_buckets(kClusterNum);
//   // std::vector<std::vector<DataType>> temp_centroids(kClusterNum);
//   std::vector<IDType> id_cid(kDataNum);

//   if (init_centroids_norm) {
//     InitCentroidsByNorm(kData, kDataNum, kDataDim, centroids, kClusterNum);
//   } else {
//     std::size_t interval = (std::size_t)kDataNum / kClusterNum;
//     for (std::size_t i = 0; i < kClusterNum; ++i) {
//       auto begin = kData + i * interval * kDataDim;
//       centroids[i].assign(begin, begin + kDataDim);
//     }
//   }

//   for (unsigned iter = 0; iter < kKMeansIter; ++iter) {
// #pragma omp parallel for
//     for (std::size_t i = 0; i < kDataNum; ++i) {
//       DataType min_dist = std::numeric_limits<DataType>::max();
//       for (std::size_t j = 0; j < kClusterNum; ++j) {
//         DataType dist = L2Sqr<DataType>(kData + i * kDataDim, centroids[j].data(), kDataDim);
//         if (dist < min_dist) {
//           id_cid[i] = j;
//           min_dist = dist;
//         }
//       }
//     }  // for assign vector to cluster

// #pragma omp parallel for
//     for (std::size_t i = 0; i < kClusterNum; ++i) {
//       id_buckets[i].clear();
//     }

//     for (std::size_t i = 0; i < kDataNum; ++i) {
//       id_buckets[id_cid[i]].push_back(i);
//     }

//     for (std::size_t i = 0; i < kClusterNum; ++i) {
//       centroids[i].assign(kData + id_buckets[i][0] * kDataDim,
//                           kData + id_buckets[i][0] * kDataDim + kDataDim);
//       for (std::size_t j = 1; j < id_buckets[i].size(); ++j) {
//         AddAssign<DataType>(kData + id_buckets[i][j] * kDataDim, centroids[i].data(), kDataDim);
//       }
//       for (std::size_t j = 0; j < kDataDim; ++j) {
//         centroids[i][j] /= id_buckets[i].size();
//       }
//     }

//     std::vector<float> errs(kClusterNum, 0);
//     float avg_err = 0;
// #pragma omp parallel for reduction(+ : avg_err)
//     for (std::size_t i = 0; i < kClusterNum; ++i) {
//       for (std::size_t j = 0; j < id_buckets[i].size(); ++j) {
//         errs[i] +=
//             L2Sqr<DataType>(kData + id_buckets[i][j] * kDataDim, centroids[i].data(), kDataDim);
//       }
//       errs[i] /= id_buckets[i].size();
//       avg_err += errs[i];
//     }
//     std::sort(errs.begin(), errs.end());
//     fmt::println("Iter: {}, Avg Err: {}, Min Err: {}, Max Err:{}", iter, avg_err / kClusterNum,
//                  errs[0], errs[kClusterNum - 1]);
//   }  // for iter
//   // TODO Split Lage Cluster
// }

}  // namespace alaya