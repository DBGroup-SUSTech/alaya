// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "Basic/Console/console_V3.hpp"
#include "common_includes.h"
#include <boost/program_options.hpp>
#include <cstdint>
#include <cstdlib>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"

#include <boost/coroutine/all.hpp>
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#include "MessageQueue/message_queue.hpp"
#include "main.hpp"

#define WARMUP false

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results)
{
    diskann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(8) << percentiles[s] << "%";
    }
    diskann::cout << std::endl;
    diskann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(9) << results[s];
    }
    diskann::cout << std::endl;
}

template <typename T, typename LabelT = uint32_t>
int search_disk_index_mpi(diskann::Metric &metric, const std::string &index_path_prefix,
                          const std::string &result_output_prefix, uint32_t **&result_ids, float **&result_dists,
                          T *query, const size_t query_num, const size_t query_dim, const std::string gt_file,
                          const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                          const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                          const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                          const std::vector<std::string> &query_filters, const bool use_reorder_data = false)
{
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t gt_num, gt_dim;
    size_t query_aligned_dim = ROUND_UP(query_dim, 8);

    bool filtered_search = false;


    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";
    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif
    std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndex<T, LabelT>(reader, metric));

    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());


    if (res != 0)
    {
        return res;
    }

    std::vector<uint32_t> node_list;
    diskann::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    // if (num_nodes_to_cache > 0)
    //     _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
    //     num_threads, node_list);
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();


    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP)
    {
        if (file_exists(warmup_query_file))
        {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        }
        else
        {
            warmup_num = (std::min)((uint32_t)150000, (uint32_t)15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++)
            {
                for (uint32_t d = 0; d < warmup_dim; d++)
                {
                    warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                }
            }
        }
        diskann::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)warmup_num; i++)
        {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        diskann::cout << "..done" << std::endl;
    }
    //=
    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "CPU (s)";
    if (calc_recall_flag)
    {
        diskann::cout << std::setw(16) << recall_string << std::endl;
    }
    else
        diskann::cout << std::endl;
    diskann::cout << "==============================================================="
                     "======================================================="
                  << std::endl;
    //=
    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    double best_recall = 0.0;

    result_ids = new uint32_t *[Lvec.size()];
    result_dists = new float *[Lvec.size()];

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        if (beamwidth <= 0)
        {
            diskann::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        }
        else
            optimized_beamwidth = beamwidth;

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new diskann::QueryStats[query_num];

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            if (!filtered_search)
            {
                // query这个地方取的是一段地址内的数据。也就是一个query，对齐过后的query
                // query_result_ids_64.data() 是这个数组的首地址，而且这个数据是 uint32_t 类型
                _pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, L,
                                                 query_result_ids_64.data() + (i * recall_at),
                                                 query_result_dists[test_id].data() + (i * recall_at),
                                                 optimized_beamwidth, use_reorder_data, stats + i);
            }
            else
            {
                LabelT label_for_search;
                if (query_filters.size() == 1)
                { // one label for all queries
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[0]);
                }
                else
                { // one label for each query
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[i]);
                }
                _pFlashIndex->cached_beam_search(
                    query + (i * query_aligned_dim), recall_at, L, query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, true, label_for_search,
                    use_reorder_data, stats + i);
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        // query_result_ids_64 是本轮L产生的结果，然后将结果转移到query_result_ids[test_id]中去
        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
                                                   query_num, recall_at);

        //=
        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });
        if (test_id == Lvec.size() - 1)
            for (int ids_index = 0; ids_index < recall_at; ++ids_index)
                printf("before sending, in function.. result_ids in Lvec %d, is: %d\n", test_id,
                         query_result_ids[test_id][ids_index]);
        std::cout << std::endl;

        //= ==============================================计算recall============================================= =//
        // double recall = 0;
        // if (calc_recall_flag)
        // {
        //     //* 计算recall函数，返回值为recall，对每个L都计算一次。
        //     //* 传入参数为：query_num 查询个数、gt_ids, gt_dists, gt_dim,
        //     //* query_result_ids[test_id]这个是一维数组首地址，recall_at 就是K，所以参数传递没有问题。
        //     recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
        //                                        query_result_ids[test_id].data(), recall_at, recall_at);
        //     // 在L个结果中，选出最佳recall
        //     best_recall = std::max(recall, best_recall);
        //     if constexpr (DEBUG)
        //     {
        //         Msg_warn("recall and best_recall is: %f, %f", recall, best_recall);
        //         for (int j = 0; j < query_result_ids[test_id].size(); ++j)
        //         {
        //             Msg_major("query_result_ids[%d][%d] = %d", test_id, j, query_result_ids[test_id][j]);
        //         }
        //         // Msg_major("query_result_ids[%zu][]")
        //     }
        // }

        // = 修改返回结果指针的指向

        result_ids[test_id] = new uint32_t[recall_at * query_num];
        memcpy(result_ids[test_id], query_result_ids[test_id].data(), recall_at * query_num * sizeof(uint32_t));

        result_dists[test_id] = new float[recall_at * query_num];
        memcpy(result_dists[test_id], query_result_dists[test_id].data(), recall_at * query_num * sizeof(float));

        diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_cpuus;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << 0 << std::endl;
        }
        else
            diskann::cout << std::endl;
        delete[] stats;
    } // end of for Lvec.size()

    //= ==============================================存储结果到文件============================================= =//
    diskann::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
            continue;

        std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    }

    diskann::aligned_free(query);
    if (warmup != nullptr)
        diskann::aligned_free(warmup);

    // if constexpr (DEBUG)
    // {
    //     Msg_warn("result_num and result_dim is: %zu, %zu...", Lvec.size(), recall_at * query_num);
    //     for (int test_id = 0; test_id < Lvec.size(); test_id++)
    //     {
    //         for (int j = 0; j < recall_at * query_num; ++j)
    //         {
    //             Msg_major("after pointer reverted: result_ids[%d][%d] = %d", test_id, j, result_ids[test_id][j]);
    //         }
    //     }
    // }
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

template <typename T, typename LabelT = uint32_t>
int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false)
{
    diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        diskann::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // load query bin
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndex<T, LabelT>(reader, metric));

    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

    if (res != 0)
    {
        return res;
    }

    std::vector<uint32_t> node_list;
    diskann::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    // if (num_nodes_to_cache > 0)
    //     _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
    //     num_threads, node_list);
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP)
    {
        if (file_exists(warmup_query_file))
        {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        }
        else
        {
            warmup_num = (std::min)((uint32_t)150000, (uint32_t)15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++)
            {
                for (uint32_t d = 0; d < warmup_dim; d++)
                {
                    warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                }
            }
        }
        diskann::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)warmup_num; i++)
        {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        diskann::cout << "..done" << std::endl;
    }

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "CPU (s)";
    if (calc_recall_flag)
    {
        diskann::cout << std::setw(16) << recall_string << std::endl;
    }
    else
        diskann::cout << std::endl;
    diskann::cout << "==============================================================="
                     "======================================================="
                  << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        if (beamwidth <= 0)
        {
            diskann::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        }
        else
            optimized_beamwidth = beamwidth;

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new diskann::QueryStats[query_num];

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            if (!filtered_search)
            {
                // query这个地方取的是一段地址内的数据。也就是一个query，对齐过后的query
                // query_result_ids_64.data() 是这个数组的首地址，而且这个数据是 uint32_t 类型
                _pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, L,
                                                 query_result_ids_64.data() + (i * recall_at),
                                                 query_result_dists[test_id].data() + (i * recall_at),
                                                 optimized_beamwidth, use_reorder_data, stats + i);
            }
            else
            {
                LabelT label_for_search;
                if (query_filters.size() == 1)
                { // one label for all queries
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[0]);
                }
                else
                { // one label for each query
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[i]);
                }
                _pFlashIndex->cached_beam_search(
                    query + (i * query_aligned_dim), recall_at, L, query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, true, label_for_search,
                    use_reorder_data, stats + i);
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
                                                   query_num, recall_at);

        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        double recall = 0;
        if (calc_recall_flag)
        {
            recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, recall_at);
            best_recall = std::max(recall, best_recall);
        }

        diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_cpuus;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << recall << std::endl;
        }
        else
            diskann::cout << std::endl;
        delete[] stats;
    } // end of for Lvec.size()

    diskann::cout << "Done searching. Now saving results " << std::endl;
    // uint64_t test_id = 0;
    // for (auto L : Lvec)
    // {
    //     if (L < recall_at)
    //         continue;

    //     std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    //     diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

    //     cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
    //     diskann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    // }

    diskann::aligned_free(query);
    if (warmup != nullptr)
        diskann::aligned_free(warmup);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    // = 初始化MPI
    Env::initEnv(argc, argv);

    std::string data_type, dist_fn, index_path_prefix, result_path_prefix, query_file, gt_file, filter_label,
        label_type, query_filters_file;
    uint32_t num_threads, K, W, num_nodes_to_cache, search_io_limit;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    float fail_if_recall_below = 0.0f;

    po::options_description desc{
        program_options_utils::make_program_description("search_disk_index", "Searches on-disk DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path_prefix)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                                       program_options_utils::BEAMWIDTH);
        optional_configs.add_options()("num_nodes_to_cache", po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                                       program_options_utils::NUMBER_OF_NODES_TO_CACHE);
        optional_configs.add_options()(
            "search_io_limit",
            po::value<uint32_t>(&search_io_limit)->default_value(std::numeric_limits<uint32_t>::max()),
            "Max #IOs for search.  Default value: uint32::max()");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("use_reorder_data", po::bool_switch()->default_value(false),
                                       "Include full precision data in the index. Use only in "
                                       "conjuction with compressed data on SSD.  Default value: false");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (vm["use_reorder_data"].as<bool>())
            use_reorder_data = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) && (metric == diskann::Metric::INNER_PRODUCT))
    {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    if (use_reorder_data && data_type != std::string("float"))
    {
        std::cout << "Error: Reorder data for reordering currently only "
                     "supported for float data type."
                  << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "")
    {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    std::vector<std::string> query_filters;
    if (filter_label != "")
    {
        query_filters.push_back(filter_label);
    }
    else if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

    // = server0加载query file，gt file

    //*======================================================================================================================*/
    //*
    //*                                                     server0
    //*
    //*======================================================================================================================*/
    if (serverId() == 0)
    {
        float *query = nullptr;

        size_t query_num, query_dim, query_aligned_dim;
        uint32_t *gt_ids = nullptr;
        float *gt_dists = nullptr;
        size_t gt_num, gt_dim;
        diskann::load_aligned_bin<float>(query_file, query, query_num, query_dim, query_aligned_dim);
        // query_num = 1;
        uint32_t query_buffer_capacity = (query_num * query_dim * sizeof(data_type) + 2 * sizeof(size_t)) * 2;
        uint32_t result_buffer_capacity = (K * Lvec.size() * query_num * sizeof(data_type) + 2 * sizeof(size_t)) * 1.2;

        uint32_t beamwidth = W;
        uint32_t optimized_beamwidth = 2;

        // filter一直是空的，可以暂时先不管。
        bool filtered_search = false;
        if (!query_filters.empty())
        {
            filtered_search = true;
            if (query_filters.size() != 1 && query_filters.size() != query_num)
            {
                std::cout << "Error. Mismatch in number of queries and size of query "
                             "filters file"
                          << std::endl;
                return -1; // To return -1 or some other error handling?
            }
        }

        bool calc_recall_flag = false;
        if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
        {
            diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
            if (gt_num != query_num)
            {
                diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
            }
            calc_recall_flag = true;
        }
        auto s = std::chrono::high_resolution_clock::now();
        Message::MessageQueue<char> *msgq0 =
            new Message::MessageQueue<char>(query_buffer_capacity, result_buffer_capacity);
        msgq0->sendMessage<float>(1, query, query_num, query_dim, QUERY);
        msgq0->sendMessage<float>(2, query, query_num, query_dim, QUERY);
        msgq0->sendMessage<float>(3, query, query_num, query_dim, QUERY);
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> send_time = e - s;

        uint32_t **result_ids2 = nullptr;
        float **result_dists2 = nullptr;
        size_t result_num2, result_dim2 = 0;
        auto before_recv3 = std::chrono::high_resolution_clock::now();

        msgq0->recvMessage<uint32_t>(3, result_ids2, result_num2, result_dim2, IDX);
        msgq0->recvMessage<float>(3, result_dists2, result_num2, result_dim2, DIST);

        auto after_recv3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> recv3_time = after_recv3 - before_recv3;

        uint32_t **result_ids1 = nullptr;
        float **result_dists1 = nullptr;
        size_t result_num1, result_dim1 = 0;

        msgq0->recvMessage<uint32_t>(2, result_ids1, result_num1, result_dim1, IDX);
        msgq0->recvMessage<float>(2, result_dists1, result_num1, result_dim1, DIST);
        auto after_recv2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> recv2_time = after_recv2 - after_recv3;

        uint32_t **result_ids0 = nullptr;
        float **result_dists0 = nullptr;
        size_t result_num0, result_dim0 = 0;

        msgq0->recvMessage<uint32_t>(1, result_ids0, result_num0, result_dim0, IDX);
        msgq0->recvMessage<float>(1, result_dists0, result_num0, result_dim0, DIST);

        // if constexpr (DEBUG)
        // {
        //     Msg_warn("result_num and result_dim is: %zu, %zu...", Lvec.size(), K * query_num);
        //     for (int test_id = 0; test_id < Lvec.size(); test_id++)
        //     {
        //         for (int j = 0; j < K * query_num; ++j)
        //         {
        //             Msg_major("after recv before vector in pro 1: result_ids[%d][%d] = %d", test_id, j,
        //                       result_ids0[test_id][j]);
        //         }
        //     }
        // }

        // auto after_recv1 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> recv1_time = after_recv1 - after_recv2;
        // Msg_warn("----------------------recv 3 time is: %f", recv1_time.count());

        // assert(result_num0 == result_num1);
        // assert(result_num1 == result_num2);

        //= result merge
        std::vector<std::vector<uint32_t>> res_id(result_num0);
        std::vector<std::vector<float>> res_dist(result_num0);

        // int lastLvec = Lvec.size() - 1;
        merge_to_vector1<uint32_t, float, uint32_t, float>(res_id, res_dist, result_ids0, result_dists0, result_num0,
                                                           result_dim0);

        // for (int ids_index = 0; ids_index < K; ++ids_index)
        // {
        //     Msg_info("after merge result_ids in Lvec %d, is: %d, and its dist is: %f", lastLvec,
        //              res_id[lastLvec][ids_index], );
        // }
        merge_to_vector2<uint32_t, float, uint32_t, float>(res_id, res_dist, result_ids1, result_dists1, result_num0,
                                                           result_dim0);
        merge_to_vector3<uint32_t, float, uint32_t, float>(res_id, res_dist, result_ids2, result_dists2, result_num0,
                                                           result_dim0);

        sort_vector<uint32_t, float>(res_id, res_dist, K, query_num);

        //= begin Lvec.size() loop----------------------------------------------
        double best_recall = 0.0;
        for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
        {
            if (test_id == Lvec.size() - 1)
                for (int ids_index = 0; ids_index < K; ++ids_index)
                    printf("after sort result_ids in Lvec %d, is: %d\n", test_id, res_id[test_id][ids_index]);
            std::cout << std::endl;
            double recall = 0;
            if (calc_recall_flag)
            {
                //* 计算recall函数，返回值为recall，对每个L都计算一次。
                //* 传入参数为：query_num 查询个数、gt_ids, gt_dists, gt_dim,
                //* query_result_ids[test_id]这个是一维数组首地址，recall_at 就是K，所以参数传递没有问题。
                recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                   res_id[test_id].data(), K, K);
                // 在L个结果中，选出最佳recall
                best_recall = std::max(recall, best_recall);
            }
            if (calc_recall_flag)
            {
                diskann::cout << std::setw(16) << recall << std::endl;
            }
            else
                diskann::cout << std::endl;
        } // end of Lvec.size()
        auto dis_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_dis_time = dis_time - s;
        std::cout << "----------------------total single time is: " << total_dis_time.count() << std::endl;
    } // end of serverId()=0

    //*======================================================================================================================*/
    //*
    //*                                                     server1
    //*
    //*======================================================================================================================*/

    else
    {
        // try
        // {
        uint32_t query_buffer_capacity = (1000 * 960 * sizeof(data_type) + 2 * sizeof(size_t)) * 2;
        uint32_t result_buffer_capacity = (K * Lvec.size() * 1000 * sizeof(data_type) + 2 * sizeof(size_t)) * 1.2;

        Message::MessageQueue<char> *msgq1 =
            new Message::MessageQueue<char>(result_buffer_capacity, query_buffer_capacity);
        size_t query_num, query_dim = 0;
        float *recv_query = nullptr;
        uint32_t **result_ids = nullptr;
        float **result_dists = nullptr;

        msgq1->recvMessage(0, recv_query, query_num, query_dim, QUERY);

        if (!query_filters.empty() && label_type == "ushort")
        {
            if (data_type == std::string("float"))
            {
                return search_disk_index<float, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data);
            }

            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("float"))
            {
                search_disk_index_mpi<float>(metric, index_path_prefix, result_path_prefix, result_ids, result_dists,
                                             recv_query, query_num, query_dim, gt_file, num_threads, K, W,
                                             num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below,
                                             query_filters, use_reorder_data);
                // std::cout << "----------------------total single time is: " << diff.count() << std::endl;


                // return search_disk_index<float>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                //                                 num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                //                                 fail_if_recall_below, query_filters, use_reorder_data);
                msgq1->sendMessage<uint32_t>(0, result_ids, Lvec.size(), K * query_num, IDX);
                msgq1->sendMessage<float>(0, result_dists, Lvec.size(), K * query_num, DIST);
            }
            else if (data_type == std::string("int8"))
                search_disk_index<int8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                          fail_if_recall_below, query_filters, use_reorder_data);
            else if (data_type == std::string("uint8"))
                search_disk_index<uint8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                           num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                           fail_if_recall_below, query_filters, use_reorder_data);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                Env::endEnv();
                return -1;
            }
        }
    } // end of serverId()=1
    Env::endEnv();
    // catch (const std::exception &e)
    // {
    //     std::cout << std::string(e.what()) << std::endl;
    //     diskann::cerr << "Index search failed." << std::endl;
    //     return -1;
    // }
}