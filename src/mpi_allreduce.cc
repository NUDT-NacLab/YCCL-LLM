/*
 * @Author: pengjintaoHPC 1272813056@qq.com
 * @Date: 2022-07-19 20:47:01
 * @LastEditors: pengjintaoHPC 1272813056@qq.com
 * @LastEditTime: 2022-07-20 15:29:37
 * @FilePath: \yhccl\yhccl_allreduce_pjt\mpi_allreduce.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <vector>
#include <stdio.h>
#include "yhccl_contexts.h"
#include "yhccl_allreduce.h"
#include "yhccl_bcast.h"
#include "yhccl_options.h"
#include "yhccl_reduce.h"
#include "yhccl_allgather.h"
#include "./BIO_Bcast/BIO.h"

#define PJT_MPI_MIDWARE

#ifdef PJT_MPI_MIDWARE

#include "mpi.h"
std::vector<int> pjt_msg_szcount;
static int comm_world_procn;
static pjtccl_contexts ccl_ctx;
static int context_inited;

extern "C" int MPI_Init(int *argc, char ***argv)
{
    int my_rank, err;
    err = PMPI_Init(argc, argv);
    PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // PMPI_Comm_size(MPI_COMM_WORLD, &comm_world_procn);
    // ccl_ctx.init(MPI_COMM_WORLD);
    // context_inited = 1;
    // PMPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
        printf("[YCCL_LLM] - MPI_Init: OK.\n", my_rank);
    return err;
}

extern "C" int MPI_Allreduce(
    const void *sendbuf,
    void *recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm)
{
    if (comm == MPI_COMM_WORLD && (op == MPI_SUM) && (datatype == MPI_DOUBLE))
    {
        // printf("count=%d\n", count);
        if (context_inited++ == 0)
        {
            // 初始化
            ccl_ctx.init(comm);
            ccl_ctx._ctxp->_opt.dynamical_tune = true;
            ccl_ctx._ctxp->_opt.mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
            ccl_ctx._ctxp->_opt.intra_node_reduce_type = MIXED;
            ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
            ccl_ctx._ctxp->_opt.inter_node_algorithm = 2;
            ccl_ctx._ctxp->_opt.pjt_inner_cpy = 1;
            ccl_ctx._ctxp->_opt.using_non_temporal = 1;
            // ccl_ctx._ctxp->_opt.core_per_numa = comm_world_procn / 2;
            // ccl_ctx._ctxp->_opt.numa_n = 2;
            if (yhccl_contexts::_ctx->intra_node_rank == 0)
            {
                puts("================MPI_Allreduce pjt============================");
            }
            PMPI_Barrier(MPI_COMM_WORLD);
        }
        // 调用all-reduce
        //  if (ccl_ctx.global_rank == 0)
        //  {
        //      puts("pjt");
        //  }
        //  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnuma.so /mnt/d/新工作空间/工作空间/mpi-yhccl/yhccl-build/build/lib/libyhccl.so"
        yhccl_allreduce(sendbuf, recvbuf, count, datatype, op, 0);
        //  PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }
    else
    {
        PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }
    int elem_sz = -1;
    MPI_Type_size(datatype, &elem_sz);
    pjt_msg_szcount.push_back(count * elem_sz);
    // printf("[yizimi %d]: finished MPI_Allreduce\n", yhccl_contexts::_ctx->intra_node_rank);
    return MPI_SUCCESS;
}

// zq's bcast : BIO
extern "C" int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int rank;
    PMPI_Comm_rank(comm, &rank);
    // printf("[yizimi %d]: start MPI_Bcast\n", rank);
    int ret = BIO_Bcast(buffer, count, datatype, root, comm);
    // printf("[yizimi %d]: start MPI_Bcast\n", rank);
    return ret;
}

// pjt's bcast
// extern "C" int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
// {
//     // return PMPI_Bcast(buffer, count, datatype, root, comm);
//     if (comm == MPI_COMM_WORLD)
//     {
//         if (context_inited++ == 0)
//         {
//             puts("MPI_Bcast pjt");
//             ccl_ctx.init(comm);
//             PMPI_Barrier(MPI_COMM_WORLD);
//         }
//         // printf("yhccl_contexts::_ctx->inter_node_procn=%d\n", yhccl_contexts::_ctx->inter_node_procn);
//         if (yhccl_contexts::_ctx->inter_node_procn == 1)
//         {
//             return yhccl_intra_node_bcast_pjt(buffer, count, datatype, root, comm);
//         }
//         else
//         {
//             printf("错误，yhccl_contexts::_ctx->inter_node_procn == %d\n", yhccl_contexts::_ctx->inter_node_procn);
//         }
//     }
//     return PMPI_Bcast(buffer, count, datatype, root, comm);
// }

// extern "C" int MPI_Reduce(
//     const void *send_data,
//     void *recv_data,
//     int count,
//     MPI_Datatype datatype,
//     MPI_Op op,
//     int root,
//     MPI_Comm comm)
// {
//     if (comm == MPI_COMM_WORLD)
//     {
//         if (context_inited++ == 0)
//         {
//             // puts("MPI_Reduce pjt");
//             ccl_ctx.init(comm);
//             PMPI_Barrier(MPI_COMM_WORLD);
//         }
//         if (yhccl_contexts::_ctx->inter_node_procn == 1)
//         {
//             int re = yhccl_intra_node_reduce_pjt(send_data, recv_data, count, datatype, op, root, comm);
//             if (re != -1)
//                 return re;
//         }
//         else
//         {
//             printf("REDUCE 错误，yhccl_contexts::_ctx->inter_node_procn == %d\n", yhccl_contexts::_ctx->inter_node_procn);
//         }
//     }
//     return PMPI_Reduce(send_data, recv_data, count, datatype, op, root, comm);
// }

// extern "C" int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
//                              void *recvbuf, int recvcount, MPI_Datatype recvtype,
//                              MPI_Comm comm)
// {
//     if (comm == MPI_COMM_WORLD)
//     {
//         if (context_inited++ == 0)
//         {
//             // puts("MPI_Allgather pjt");
//             ccl_ctx.init(comm);
//             PMPI_Barrier(MPI_COMM_WORLD);
//         }
//         if (yhccl_contexts::_ctx->inter_node_procn == 1)
//         {
//             int re = yhccl_intra_node_allgather_pjt(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
//             if (re != -1)
//                 return re;
//         }
//         else
//         {
//             printf("Allgather 错误，yhccl_contexts::_ctx->inter_node_procn == %d\n", yhccl_contexts::_ctx->inter_node_procn);
//         }
//     }
//     return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
// }

extern "C" int MPI_Finalize()
{
    extern int bio_shm_fd;
    extern volatile char *bio_fin_f;
    extern int bio_buff_size;
    extern int if_bio_bcast;
    int my_rank, err;

    PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // if (my_rank == 0) {
    // printf("[yizimi_test](rk %d): bio finalize start\n", my_rank);
    // }
    if (if_bio_bcast)
    {
        if (bio_fin_f)
            munmap((void *)bio_fin_f, bio_buff_size);
        if (bio_shm_fd)
            close(bio_shm_fd);
    }
    // if (my_rank == 0) {
    // printf("[yizimi_test](rk %d): bio finalize OK\n", my_rank);
    // }
    // if (my_rank == 0) {
    // printf("[yizimi_test](rk %d): pjt fmcc-rt finalize start\n", my_rank);
    // }
    // if (my_rank == 0)
    // {
    //     for (auto sz : pjt_msg_szcount)
    //         printf("%d\n", sz);
    // }
    PMPI_Barrier(MPI_COMM_WORLD);

    extern int if_pjt_yhccl_content;
    extern int pjt_fd;
    extern long long pjt_memory_size;
    extern void *pjt_larger_msg_allreduce_shareM;

    if (if_pjt_yhccl_content)
    {
        if (pjt_larger_msg_allreduce_shareM)
            munmap(pjt_larger_msg_allreduce_shareM, pjt_memory_size);
        if (pjt_fd)
            close(pjt_fd);
    }

    // if (my_rank == 0) {
    // printf("[yizimi_test](rk %d): pjt fmcc-rt finalize ok\n", my_rank);
    // }
    return PMPI_Finalize();
}

// extern "C" int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
//     int rank;
//     PMPI_Comm_rank(comm, &rank);
//     // if (rank == 0)
//     //     printf("[yizimi test]: reduce start\n");
//     int ret = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);

//     // if (rank == 0)
//     //     printf("[yizimi test]: reduce finnish\n");
//     return ret;
// }

#endif