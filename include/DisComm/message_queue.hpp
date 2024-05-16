
#pragma once
#include "mpi_env.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mpi.h>

#define DEBUG 0

#define QUERY 0
#define IDX 1
#define DIST 2

namespace Message {

template <typename MsgType = char>
class MessageQueue {
private:
    const int _machineId = serverId();
    const int _machineNum = serverNum();

public:
    //
    size_t mach_capacity_of_sendBuf; // max capacity of machine_sendBuffer: MsgType
    MsgType** machine_sendBuffer = nullptr; // [machineNum][msgNum * sizeof(MsgType)]
    size_t* mach_sendCnt = nullptr; // msgCnt of current machine

    //^ Recv
    size_t mach_capacity_of_recvBuf; // max capacity of machine_recvBuffer: MsgType
    MsgType** machine_recvBuffer = nullptr; // [machineNum][msgNum * sizeof(MsgType)]
    size_t* mach_recvCnt = nullptr; // msgCnt of current machine

public:
    MessageQueue(size_t mach_capacity_of_sendBuf_, size_t mach_capacity_of_recvBuf_)
        : mach_capacity_of_sendBuf(mach_capacity_of_sendBuf_)
        , mach_capacity_of_recvBuf(mach_capacity_of_recvBuf_)
    {
        if (mach_capacity_of_sendBuf == 0) {
            assert(false || "mach_capacity_of_sendBuf = 0");
        }

        if (mach_capacity_of_recvBuf == 0) {
            mach_capacity_of_recvBuf = mach_capacity_of_sendBuf;
        }
        if constexpr (DEBUG) {
            printf("size of send and recv buffer is: %ld, %ld\n", mach_capacity_of_sendBuf, mach_capacity_of_recvBuf);
        }
        //^ 初始化Machine (send 和 recv)
        machine_sendBuffer = new MsgType*[_machineNum];
        machine_recvBuffer = new MsgType*[_machineNum];

        for (int machineId = 0; machineId < _machineNum; machineId++) {
            // machine_sendBuffer[machineId] = new MsgType[mach_capacity_of_sendBuf * _msgItemLen];
            machine_sendBuffer[machineId] = new MsgType[mach_capacity_of_sendBuf];
            machine_recvBuffer[machineId] = new MsgType[mach_capacity_of_recvBuf];
        }

        mach_sendCnt = new size_t[_machineNum];
        mach_recvCnt = new size_t[_machineNum];

        memset(mach_sendCnt, 0, _machineNum * sizeof(size_t));
        memset(mach_recvCnt, 0, _machineNum * sizeof(size_t));
    }

    ~MessageQueue()
    {
        if (machine_sendBuffer != nullptr) {
            for (int machineId = 0; machineId < _machineNum; machineId++) {
                delete[] machine_sendBuffer[machineId];
            }
            delete[] machine_sendBuffer;
        }

        if (machine_recvBuffer != nullptr) {
            for (int machineId = 0; machineId < _machineNum; machineId++) {
                delete[] machine_recvBuffer[machineId];
            }
            delete[] machine_recvBuffer;
        }
    }

public:
    // void sendRecvMessage(int machineId)
    // {
    //     if (machineId == _machineId)
    //     {
    //         printf("machineId == broker! \n");
    //         return;
    //     }

    //     MPI_Status recv_status;
    //     MPI_Sendrecv(machine_sendBuffer[machineId], mach_sendCnt[machineId] * sizeof(MsgType), MPIDataType<char>(),
    //     machineId, 0,
    //                  machine_recvBuffer[machineId], mach_recvCnt[machineId] * sizeof(MsgType), MPIDataType<char>(),
    //                  machineId, 0, get_mpi_comm(), &recv_status);
    //     int return_recvCnt;
    //     MPI_Get_count(&recv_status, MPIDataType<char>(), &return_recvCnt); // 返回消息数量
    //     mach_recvCnt[machineId] = return_recvCnt / sizeof(MsgType);
    // }
    template <typename SendType>
    void sendMessage(int machineId, SendType* msg, size_t num, size_t dim, int tag)
    {
        if (machineId == _machineId) {
            printf("machineId == broker! \n");
            return;
        }
        memcpy(machine_sendBuffer[machineId], &num, sizeof(size_t));
        memcpy(machine_sendBuffer[machineId] + sizeof(size_t), &dim, sizeof(size_t));

        // memcpy是字节数，所以要 * sizeof(SendType)
        memcpy(machine_sendBuffer[machineId] + 2 * sizeof(size_t), msg, num * dim * sizeof(SendType));

        mach_sendCnt[machineId] = 2 * sizeof(size_t) + dim * num * sizeof(SendType);
        MPI_Send(machine_sendBuffer[machineId], mach_sendCnt[machineId], MPIDataType<MsgType>(), machineId, tag,
            get_mpi_comm());
    }

    //= sendMessage 重载  发送二维数组     num = L,  dim = K * query_num
    template <typename SendType>
    void sendMessage(int machineId, SendType** msg, size_t num, size_t dim, int tag)
    {
        if (machineId == _machineId) {
            printf("machineId == broker! \n");
            return;
        }
        memcpy(machine_sendBuffer[machineId], &num, sizeof(size_t));
        memcpy(machine_sendBuffer[machineId] + sizeof(size_t), &dim, sizeof(size_t));
        for (int i = 0; i < num; ++i) {
            memcpy(machine_sendBuffer[machineId] + 2 * sizeof(size_t) + i * dim * sizeof(SendType), msg[i],
                sizeof(SendType) * dim);
        }
        mach_sendCnt[machineId] = 2 * sizeof(size_t) + num * dim * sizeof(SendType);
        MPI_Send(machine_sendBuffer[machineId], mach_sendCnt[machineId], MPIDataType<MsgType>(), machineId, tag,
            get_mpi_comm());
    }

    template <typename RecvType>
    void recvMessage(int machineId, RecvType*& msg, size_t& num, size_t& dim, int tag)
    {
        if (machineId == _machineId) {
            printf("machineId == broker! \n");
            return;
        }
        MPI_Status recv_status;
        MPI_Recv(machine_recvBuffer[machineId], mach_capacity_of_recvBuf, MPIDataType<MsgType>(), machineId, tag,
            get_mpi_comm(), &recv_status);
        int return_recvCnt;
        MPI_Get_count(&recv_status, MPIDataType<MsgType>(), &return_recvCnt); // 返回消息数量

        memcpy(&num, machine_recvBuffer[machineId], sizeof(size_t));
        memcpy(&dim, machine_recvBuffer[machineId] + sizeof(size_t), sizeof(size_t));

        assert((return_recvCnt == num * dim * sizeof(RecvType) + 2 * sizeof(size_t)) || "recv cnt is wrong");
        mach_recvCnt[machineId] = return_recvCnt;
        msg = new RecvType[num * dim];
        memcpy(msg, machine_recvBuffer[machineId] + 2 * sizeof(size_t), num * dim * sizeof(RecvType));
    }

    //= recvMessage 重载  接收并返回二维数组   num = L,  dim = query_num * K
    template <typename RecvType>
    void recvMessage(int machineId, RecvType**& msg, size_t& num, size_t& dim, int tag)
    {
        if (machineId == _machineId) {
            printf("machineId == broker! \n");
            return;
        }
        MPI_Status recv_status;
        MPI_Recv(machine_recvBuffer[machineId], mach_capacity_of_recvBuf, MPIDataType<MsgType>(), machineId, tag,
            get_mpi_comm(), &recv_status);
        int return_recvCnt;
        MPI_Get_count(&recv_status, MPIDataType<MsgType>(), &return_recvCnt); // 返回消息数量

        memcpy(&num, machine_recvBuffer[machineId], sizeof(size_t));
        memcpy(&dim, machine_recvBuffer[machineId] + sizeof(size_t), sizeof(size_t));

        assert((return_recvCnt == num * dim * sizeof(RecvType) + 2 * sizeof(size_t)) || "recv cnt is wrong");
        mach_recvCnt[machineId] = return_recvCnt;

        msg = new RecvType*[num];
        for (int i = 0; i < num; ++i) {
            msg[i] = new RecvType[dim];
            memcpy(msg[i], machine_recvBuffer[machineId] + i * dim * sizeof(RecvType) + 2 * sizeof(size_t),
                dim * sizeof(RecvType));
        }
    }
    void reset()
    {
        for (int machineId = 0; machineId < _machineNum; ++machineId) {
            memset(machine_sendBuffer[machineId], 0, mach_capacity_of_sendBuf * sizeof(MsgType));
            memset(machine_recvBuffer[machineId], 0, mach_capacity_of_recvBuf * sizeof(MsgType));
        }
        memset(mach_sendCnt, 0, _machineNum * sizeof(size_t));
        memset(mach_recvCnt, 0, _machineNum * sizeof(size_t));
    }
    MsgType* getMachine_RecvMsgData(int machineId)
    {
        return (MsgType*)machine_recvBuffer[machineId];
    }

    size_t getMachine_RecvMsgCount(int machineId)
    {
        return mach_recvCnt[machineId];
    }
}; // class MessageQueue

} // namespace Message