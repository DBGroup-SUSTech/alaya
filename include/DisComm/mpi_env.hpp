#pragma once

#include <cstdint>
#include <iostream>

#include <assert.h>
#include <mpi.h>

static int serverId_static = 0;
static int serverNum_static = 0;

#define serverId_static serverId()
#define serverNum_static serverNum()

class MPIEnv
{

  private:
    int serverId_;
    int serverNum_;
    MPI_Comm comm_;
    int namelen_;
    char serverName_[MPI_MAX_PROCESSOR_NAME];

  public:
    const int &serverId;
    const int &serverNum; // 集群中的节点数
    const char *serverName;
    const int &namelen;
    MPI_Comm &comm;

    // 构造函数
    MPIEnv() : serverId(serverId_), serverNum(serverNum_), serverName(serverName_), namelen(namelen_), comm(comm_)
    {
    }

    // 初始化
    void mpi_env_init(int *argc, char **argv[])
    {
        int is_inited = 0;
        MPI_Initialized(&is_inited);
        if (is_inited)
            return;
        MPI_Init(argc, argv);
        comm_ = MPI_COMM_WORLD;
        MPI_Comm_rank(MPI_COMM_WORLD, &serverId_);
        MPI_Comm_size(MPI_COMM_WORLD, &serverNum_);
        MPI_Get_processor_name(serverName_, &namelen_);

        // check that every node should have only one MPI process.
        MPI_Comm _tmp_comm;
        int num = 0;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &_tmp_comm);
        MPI_Comm_size(_tmp_comm, &num);
        // if (num != 1)
        // {
        //     printf("Every node should have only one MPI process\n");
        //     assert(0);
        // }

        // std::cout << "serverId = " << serverId_ << ", serverNum = " << serverNum_ << ",procname" << serverName_ <<
        // std::endl;
    }

    void mpi_env_init()
    {
        int is_inited = 0;
        MPI_Initialized(&is_inited);
        if (is_inited)
            return;
        MPI_Init(0, NULL);
        comm_ = MPI_COMM_WORLD;
        MPI_Comm_rank(MPI_COMM_WORLD, &serverId_);
        MPI_Comm_size(MPI_COMM_WORLD, &serverNum_);
        MPI_Get_processor_name(serverName_, &namelen_);

        // serverId_static = serverId_;
        // serverNum_static = serverNum_;

        // check that every node should have only one MPI process.
        MPI_Comm _tmp_comm;
        int num = 0;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &_tmp_comm);
        MPI_Comm_size(_tmp_comm, &num);
        // if (num != 1)
        // {
        //     printf("Every node should have only one MPI process\n");
        //     assert(0);
        // }

        // std::cout << "serverId = " << serverId_ << ", serverNum = " << serverNum_ << ",procname = " << serverName_ <<
        // std::endl;
    }

    void mpi_env_finalize()
    {
        int is_finalized = 0;
        MPI_Finalized(&is_finalized);
        if (is_finalized)
            return;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }

}; // end of class MPIEnv

/* ===================================================================================================
 *
 *                                        【定义公共变量】
 *
 *===================================================================================================*/
inline MPIEnv &global_mpiEnv()
{
    static MPIEnv mpi_env;
    return mpi_env;
}

inline int serverNum()
{
    return global_mpiEnv().serverNum;
}

inline int serverId()
{
    return global_mpiEnv().serverId;
}

inline std::string serverName()
{
    return std::string(global_mpiEnv().serverName, global_mpiEnv().namelen);
}

inline void mpi_barrier()
{
    MPI_Barrier(global_mpiEnv().comm);
}

inline MPI_Comm get_mpi_comm()
{
    return global_mpiEnv().comm;
}

inline void mpi_env_finalize()
{
    global_mpiEnv().mpi_env_finalize();
}

const constexpr bool showMPIDataType = false;
template <typename T> MPI_Datatype MPIDataType()
{

    if constexpr (std::is_same<T, char>::value)
    {
        if constexpr (showMPIDataType)
            printf("MPI_CHAR\n");
        return MPI_CHAR;
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        if constexpr (showMPIDataType)
            printf("MPI_UNSIGNED_CHAR\n");
        return MPI_UNSIGNED_CHAR;
    }
    else if constexpr (std::is_same<T, int>::value)
    {
        if constexpr (showMPIDataType)
            printf("MPI_INT\n");
        return MPI_INT;
    }
    else if constexpr (std::is_same<T, unsigned>::value)
    {
        if constexpr (showMPIDataType)
            printf("MPI_UNSIGNED\n");
        return MPI_UNSIGNED;
    }
    else if constexpr (std::is_same<T, long>::value)
    {
        if constexpr (showMPIDataType)
            printf("MPI_LONG\n");
        return MPI_LONG;
    }
    else if constexpr (std::is_same<T, unsigned long>::value)
    {
        if constexpr (showMPIDataType)
            printf("MPI_UNSIGNED_LONG\n");
        return MPI_UNSIGNED_LONG;
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        if constexpr (showMPIDataType)
            printf("MPI_FLOAT\n");
        return MPI_FLOAT;
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        if constexpr (showMPIDataType)
            printf("MPI_DOUBLE\n");
        return MPI_DOUBLE;
    }
    else if constexpr (std::is_same<T, uint64_t>::value)
    {
        if constexpr (showMPIDataType)
            printf("MPI_DOUBLE\n");
        return MPI_UNSIGNED_LONG_LONG;
    }
    else
    {
        printf("type not supported\n");
        exit(-1);
    }
}

enum MPIMessageType
{
    QPINFO
};

/*****************************************************************************
 * 这里全部转为char在网络中传播
 * 此外，T必须是可序列化的
 *****************************************************************************/
template <typename T>
void MPISend(int destNode, T *address, size_t len, MPIMessageType messageType = MPIMessageType::QPINFO)
{
    MPI_Send(address, sizeof(T) * len, MPIDataType<char>(), destNode, messageType, get_mpi_comm());
}

template <typename T>
void MPIRecv(int sourceNode, T *address, size_t len, MPI_Status *status,
             MPIMessageType messageType = MPIMessageType::QPINFO)
{
    MPI_Recv(address, sizeof(T) * len, MPIDataType<char>(), sourceNode, messageType, get_mpi_comm(), status);
}

template <typename T> void MPISend_RDMA(int destNode, T *address, size_t len, int tag)
{
    MPI_Send(address, sizeof(T) * len, MPIDataType<char>(), destNode, tag, get_mpi_comm());
}

template <typename T> void MPIRecv_RDMA(int sourceNode, T *address, size_t len, MPI_Status *status, int tag)
{
    MPI_Recv(address, sizeof(T) * len, MPIDataType<char>(), sourceNode, tag, get_mpi_comm(), status);
}
