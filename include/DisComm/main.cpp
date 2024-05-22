#include "main.hpp"
#include "mpi_env.hpp"
#include "message_queue.hpp"
#include <iostream>
#include "../alaya/utils/metric_type.h"
#include "../alaya/utils/io_utils.h"
#include <alaya/index/bucket/ivf.h>
#include <alaya/searcher/ivf_searcher.h>


int main(int argc, char** argv)
{
    Env::initEnv(argc, argv);
    std::cout << "Hello, from DisComm!\n";
    if (serverId() == 0) {
        std::string query_path = "/dataset/netflix/netflix_query.fvecs";
        unsigned d_num, d_dim, q_num, q_dim, K=10;
        float *data, *query;
        query = alaya::AlignLoadVecs<float>(query_path.c_str(), q_num, q_dim);
        uint32_t *gt_ids = nullptr;
        float *gt_dists = nullptr;
        size_t gt_num, gt_dim;
        // q_num = 1;
        uint32_t query_buffer_capacity = (q_num * q_dim * sizeof(float) + 2 * sizeof(size_t)) * 2;
        uint32_t result_buffer_capacity = (K * 30 * q_num * sizeof(float) + 2 * sizeof(size_t)) * 1.2;

        auto s = std::chrono::high_resolution_clock::now();
        Message::MessageQueue<char> *msgq0 =
            new Message::MessageQueue<char>(query_buffer_capacity, result_buffer_capacity);
        msgq0->sendMessage<float>(1, query, q_num, q_dim, QUERY);
        msgq0->sendMessage<float>(2, query, q_num, q_dim, QUERY);
        msgq0->sendMessage<float>(3, query, q_num, q_dim, QUERY);
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> send_time = e - s;

        uint32_t **result_id2 = nullptr;
        float **distance2 = nullptr;
        size_t result_num2, result_dim2 = 0;
        auto before_recv3 = std::chrono::high_resolution_clock::now();

        msgq0->recvMessage<uint32_t>(3, result_id2, result_num2, result_dim2, IDX);
        msgq0->recvMessage<float>(3, distance2, result_num2, result_dim2, DIST);

        auto after_recv3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> recv3_time = after_recv3 - before_recv3;

        uint32_t **result_id1 = nullptr;
        float **distance1 = nullptr;
        size_t result_num1, result_dim1 = 0;

        msgq0->recvMessage<uint32_t>(2, result_id1, result_num1, result_dim1, IDX);
        msgq0->recvMessage<float>(2, distance1, result_num1, result_dim1, DIST);
        auto after_recv2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> recv2_time = after_recv2 - after_recv3;

        uint32_t **result_id0 = nullptr;
        float **distance0 = nullptr;
        size_t result_num0, result_dim0 = 0;

        msgq0->recvMessage<uint32_t>(1, result_id0, result_num0, result_dim0, IDX);
        msgq0->recvMessage<float>(1, distance0, result_num0, result_dim0, DIST);

        std::vector<std::vector<uint32_t>> res_id(result_num0);
        std::vector<std::vector<float>> res_dist(result_num0);


        merge_to_vector1<uint32_t, float, uint32_t, float>(res_id, res_dist, result_id0, distance0, result_num0,
                                                           result_dim0);
        merge_to_vector2<uint32_t, float, uint32_t, float>(res_id, res_dist, result_id1, distance1, result_num0,
                                                           result_dim0);
        merge_to_vector3<uint32_t, float, uint32_t, float>(res_id, res_dist, result_id2, distance2, result_num0,
                                                           result_dim0);

        sort_vector<uint32_t, float>(res_id, res_dist, K, q_num);

        //= begin Lvec.size() loop----------------------------------------------
        double best_recall = 0.0;
        for (uint32_t test_id = 0; test_id < 30; test_id++)
        {
            if (test_id == 30 - 1)
                for (int ids_index = 0; ids_index < K; ++ids_index)
                    printf("after sort result_id in Lvec %d, is: %d\n", test_id, res_id[test_id][ids_index]);
            std::cout << std::endl;
        //     double recall = 0;
        //     if (calc_recall_flag)
        //     {
        //         //* 计算recall函数，返回值为recall，对每个L都计算一次。
        //         //* 传入参数为：q_num 查询个数、gt_ids, gt_dists, gt_dim,
        //         //* query_result_id[test_id]这个是一维数组首地址，recall_at 就是K，所以参数传递没有问题。
        //         recall = diskann::calculate_recall((uint32_t)q_num, gt_ids, gt_dists, (uint32_t)gt_dim,
        //                                            res_id[test_id].data(), K, K);
        //         // 在L个结果中，选出最佳recall
        //         best_recall = std::max(recall, best_recall);
        //     }
        //     if (calc_recall_flag)
        //     {
        //         diskann::cout << std::setw(16) << recall << std::endl;
        //     }
        //     else
        //         diskann::cout << std::endl;
        } // end of Lvec.size()
        auto dis_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_dis_time = dis_time - s;
        std::cout << "----------------------total single time is: " << total_dis_time.count() << std::endl;
    } // end of serverId()=0

    else
    {
        // // try
        // // {
        // unsigned K = 10;
        // uint32_t query_buffer_capacity = (1000 * 960 * sizeof(float) + 2 * sizeof(size_t)) * 2;
        // uint32_t result_buffer_capacity = (K * 30 * 1000 * sizeof(float) + 2 * sizeof(size_t)) * 1.2;

        // Message::MessageQueue<char> *msgq1 =
        //     new Message::MessageQueue<char>(result_buffer_capacity, query_buffer_capacity);
        // size_t query_num, query_dim = 0;
        // float* recv_query = nullptr;
        // float* distance = new float[K];
        // int64_t* result_id = new int64_t[K];

        // msgq1->recvMessage(0, recv_query, query_num, query_dim, QUERY);

        // std::string netflix_path = "/dataset/netflix";
        // unsigned d_num, d_dim, q_num, q_dim;
        // float* data = alaya::LoadVecs<float>(fmt::format("{}/netflix_base.fvecs", netflix_path).c_str(),
        //                                     d_num, d_dim);
        // float* query = alaya::AlignLoadVecs<float>(
        //     fmt::format("{}/netflix_query.fvecs", netflix_path).c_str(), q_num, q_dim);
        // fmt::println("d_num: {}, d_dim: {}, q_num: {}, q_dim: {}", d_num, d_dim, q_num, q_dim);
        // unsigned bucket_num = 100;
        // alaya::IVF<float> ivf(d_dim, alaya::MetricType::L2, bucket_num);

        // ivf.BuildIndex(d_num, data);

        // alaya::IvfSearcher<alaya::MetricType::L2, float> searcher(&ivf);

        // searcher.SetNprobe(10);

        // searcher.Search(query, q_dim, K, distance, result_id);

        // for (auto i = 0; i < K; ++i) {
        //     fmt::println("distance: {}, result_id: {}", distance[i], result_id[i]);
        // }

        // msgq1->sendMessage<int64_t>(0, &result_id, 30, K * query_num, IDX);
        // msgq1->sendMessage<float>(0, &distance, 30, K * query_num, DIST);
        // delete[] distance;
        // delete[] result_id;

    } // end of serverId()=1

    Env::endEnv();
}
