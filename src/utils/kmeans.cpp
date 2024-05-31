#include <alaya/utils/distance.h>
#include <alaya/utils/kmeans.h>
#include <alaya/utils/metric_type.h>
#include <alaya/utils/random_utils.h>
// #include <faiss/Clustering.h>
// #include <faiss/IndexFlat.h>
#include <fmt/core.h>

#include <limits>
#include <vector>

namespace alaya {

std::vector<float> faiss_kmeans(const float* kData, const std::size_t kDataNum,
                                const std::size_t kDataDim, unsigned cluster_num,
                                MetricType metric) {
  faiss::Clustering cluster(kDataDim, cluster_num);
  cluster.verbose = true;
  if (metric == MetricType::IP) {
    faiss::IndexFlatIP ip_index(kDataDim);
    // ip_index.add(kDataNum, kData);
    cluster.train(kDataNum, kData, ip_index);
  } else if (metric == MetricType::L2 || metric == MetricType::COS) {
    faiss::IndexFlatL2 l2_index(kDataDim);
    // l2_index.add(kDataNum, kData);
    cluster.train(kDataNum, kData, l2_index);
  } else {
    throw std::runtime_error("Unsupported metric type");
  }

  fmt::println("cluster size: {} = {}(cluster_num ) * {}(dim)",
               cluster.centroids.size(), cluster_num, kDataDim);

  std::vector<float> res = cluster.centroids;

  return res;
}

void kmeans(const float* kData, const std::size_t kDataNum, const std::size_t kDataDim,
            std::vector<std::vector<float>>& centroids, unsigned cluster_num,
            const unsigned int kKMeansIter) {
  std::vector<std::vector<float>> train(kDataNum, std::vector<float>(kDataDim));
#pragma omp parallel for
  for (size_t i = 0; i < kDataNum; ++i) {
    const float* v_begin = kData + i * kDataDim;
    train[i].assign(v_begin, v_begin + kDataDim);
  }

  unsigned interval = kDataNum / cluster_num;
  std::vector<std::vector<unsigned>> t_ivf(cluster_num);
  std::vector<std::vector<float>> t_centroid(cluster_num);
  std::vector<unsigned> t_id(kDataNum);

  std::vector<std::pair<float, unsigned>> norm_id(kDataNum);
#pragma omp parallel for
  for (size_t i = 0; i < kDataNum; ++i) {
    norm_id[i].first = GetSqrNorm<float>(train[i].data(), kDataDim);
    norm_id[i].second = i;
  }
  std::sort(norm_id.begin(), norm_id.end());

  float avg_norm = 0;
#pragma omp parallel for reduction(+ : avg_norm)
  for (size_t i = 0; i < kDataNum; ++i) {
    avg_norm += norm_id[i].first;
  }

  fmt::println("Avg Norm: {}, Min Norm: {}, Max Norm: {}", avg_norm / kDataNum, norm_id[0].first,
               norm_id[kDataNum - 1].first);

#pragma omp parallel for
  for (size_t i = 0; i < cluster_num; ++i) {
    t_centroid[i].assign(train[norm_id[i * interval].second].begin(),
                         train[norm_id[i * interval].second].end());
  }

  std::vector<bool> centroid_empty(cluster_num, false);
  float g_err = std::numeric_limits<float>::max();
  unsigned iter = kKMeansIter;
  while (iter) {
#pragma omp parallel for
    for (size_t i = 0; i < kDataNum; ++i) {
      float min_dist = std::numeric_limits<float>::max();
      for (size_t j = 0; j < cluster_num; ++j) {
        if (centroid_empty[j]) continue;
        float dist = L2Sqr<float>(train[i].data(), t_centroid[j].data(), kDataDim);
        if (dist < min_dist) {
          t_id[i] = j;
          min_dist = dist;
        }
      }
    }

#pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i) {
      t_ivf[i].clear();
    }

    for (size_t i = 0; i < kDataNum; ++i) {
      t_ivf[t_id[i]].push_back(i);
    }

    std::vector<std::pair<unsigned, unsigned>> bucket_size(cluster_num);
#pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i) {
      bucket_size[i].first = t_ivf[i].size();
      bucket_size[i].second = i;
    }
    std::sort(bucket_size.begin(), bucket_size.end(),
              [](const std::pair<unsigned, unsigned>& a, const std::pair<unsigned, unsigned>& b) {
                return a.first > b.first;
              });

    std::cout << "max bucket size: " << bucket_size[0].first << std::endl;
    std::cout << "min bucket size: " << bucket_size[cluster_num - 1].first << std::endl;

    unsigned avg_num = kDataNum / cluster_num / 2;

    unsigned c_num = 0;
    unsigned b_id = 0;
    // #pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i) {
      if ((iter == kKMeansIter && t_ivf[i].size() <= 1) ||
          (iter < kKMeansIter && t_ivf[i].size() <= 2)) {
        int r_id = GenRandInt(0, bucket_size[b_id].first - 1);
        t_centroid[i] = train[t_ivf[bucket_size[b_id].second][r_id]];
        ++b_id;
      } else {
        t_centroid[i] = train[t_ivf[i][0]];
        for (size_t j = 1; j < t_ivf[i].size(); ++j) {
          AddAssign<float>(train[t_ivf[i][j]].data(), t_centroid[i].data(), kDataDim);
        }
        for (size_t j = 0; j < kDataDim; ++j) {
          t_centroid[i][j] /= t_ivf[i].size();
        }
        ++c_num;
      }
    }

    std::vector<float> err_clusters(cluster_num, 0);
#pragma omp parallel for
    for (size_t i = 0; i < cluster_num; ++i) {
      if (t_ivf[i].size()) {
        float err = 0;
        for (size_t j = 0; j < t_ivf[i].size(); ++j) {
          err += L2Sqr<float>(t_centroid[i].data(), train[t_ivf[i][j]].data(), kDataDim);
        }
        err_clusters[i] = err / t_ivf[i].size();
      }
    }
    std::sort(err_clusters.begin(), err_clusters.end());

    float avg_err = 0;
    for (size_t i = 0; i < cluster_num; ++i) {
      if (t_ivf[i].size()) {
        avg_err += err_clusters[i];
      }
    }
    avg_err /= c_num;
    std::cout << "Iter: " << iter-- << std::endl;

    // fmt::println("Avg Err: {}, Min Err: {}, Max Err: {}", avg_err,
    // err_clusters[0],
    //              err_clusters[c_num - 1]);

    // if (avg_err < g_err) {
    //     g_err = avg_err;
    // } else {
    //     break;
    // }
  }  // while(iter)

  for (size_t i = 0; i < cluster_num; ++i) {
    if (t_ivf[i].size() > 1) {
      centroids.emplace_back(t_centroid[i]);
    }
  }
}
}  // namespace alaya