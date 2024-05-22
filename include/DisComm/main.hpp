#include "mpi_env.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#define MACHINE_NUM 3

namespace Env {
void initEnv(int argc, char* argv[])
{
    //> MPI
    global_mpiEnv().mpi_env_init(&argc, &argv);
    //> LOG
    // global_logFile().set_log_file();
}

void endEnv()
{
    mpi_barrier();
    // Msg_node("hello, from CodeTest/mpi_test!");
    mpi_env_finalize();
}
} // namespace Env

template <typename T1, typename T2, typename T3, typename T4>
void merge_to_vector1(std::vector<std::vector<T1>>& res_id, std::vector<std::vector<T2>>& res_dist, T3** ids,
    T4** dists, int num, int dim)
{
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < dim; j++) {
            res_id[i].push_back(ids[i][j]);
            res_dist[i].push_back(dists[i][j]);
        }
    }
}

template <typename T1, typename T2, typename T3, typename T4>
void merge_to_vector2(std::vector<std::vector<T1>>& res_id, std::vector<std::vector<T2>>& res_dist, T3** ids,
    T4** dists, int num, int dim)
{
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < dim; j++) {
            res_id[i].push_back(ids[i][j] + 333334);
            res_dist[i].push_back(dists[i][j]);
        }
    }
}

template <typename T1, typename T2, typename T3, typename T4>
void merge_to_vector3(std::vector<std::vector<T1>>& res_id, std::vector<std::vector<T2>>& res_dist, T3** ids,
    T4** dists, int num, int dim)
{
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < dim; j++) {
            res_id[i].push_back(ids[i][j] + 333334 + 333333);
            res_dist[i].push_back(dists[i][j]);
        }
    }
}
// template <typename T1, typename T2> bool CustomSort(const std::pair<T1, T2> &a, const std::pair<T1, T2> &b) { return
// a.second < b.second; }

template <typename T1, typename T2>
void sort_vector(std::vector<std::vector<T1>>& res_id, std::vector<std::vector<T2>>& res_dist, uint32_t recall_at,
    uint32_t query_num)
{

    for (size_t i = 0; i < res_id.size(); ++i) {
        auto comparator = [](const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
            return a.second < b.second; // 按照 pair 的第二个元素进行排序
        };

        // 在这里初始化一个数组用来merge几个机器之间的数据
        for (size_t query_id = 0; query_id < query_num; ++query_id) {

            std::vector<std::pair<T1, T2>> pairs;
            for (size_t mach_id = 0; mach_id < MACHINE_NUM; ++mach_id) {

                size_t begin_index = mach_id * recall_at * query_num + query_id * recall_at;
                for (size_t k = 0; k < recall_at; ++k) {
                    size_t index = begin_index + k;
                    pairs.emplace_back(res_id[i][index], res_dist[i][index]);
                }
            }

            std::sort(pairs.begin(), pairs.end(), comparator);

            for (size_t j = 0; j < recall_at; ++j) {
                res_id[i][query_id * recall_at + j] = pairs[j].first;
                res_dist[i][query_id * recall_at + j] = pairs[j].second;
            }
        }

        res_id[i].resize(recall_at * query_num);
        res_dist[i].resize(recall_at * query_num);
        // 对 vector 中的元素进行排序
    }
}

