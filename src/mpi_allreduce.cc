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
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cstring>
#include <map>
#include <iomanip> // 用于格式化输出
#include <string>


#define YIZIMI_TESTTOOL

#ifdef PJT_MPI_MIDWARE
#include "mpi.h"
static std::vector<int> pjt_msg_szcount;
static int comm_world_procn;
static pjtccl_contexts ccl_ctx;
static int context_inited;


const int MAX_2_MSGSIZE = 40;
static int allreduce_total_count;
static double allreduce_total_time;
static int allreduce_msg_count[MAX_2_MSGSIZE];
static double allreduce_msg_time[MAX_2_MSGSIZE];
static int unaligned_count;
static double allreduce_unalign_time;
static double allreduce_sz_3072;
static double allreduce_sz_3M;
static double allreduce_sz_30M;


int get_mpi_type_size(MPI_Datatype datatype) {
    int size;
    MPI_Type_size(datatype, &size);
    return size;
}

struct msg_info {
    int count;
    MPI_Datatype datatype;
    MPI_Op op;
    int mpi_primitive;
    msg_info (int _count, MPI_Datatype _datatype, MPI_Op _op, int _mpi_primitive)
            : count(_count), datatype(_datatype), op(_op), mpi_primitive(_mpi_primitive) {}
    
    std::string primitive_to_string() {
        switch (mpi_primitive) {
            case 0: return "Send";
            case 1: return "Recv";
            case 2: return "Reduce";
            case 3: return "Barrier";
            case 4: return "Bcast";
            case 5: return "Reduce Scatter";
            case 6: return "All Gather";
            case 7: return "Gather";
            case 8: return "All Reduce";
            case 9: return "All to All";
            // 其他操作的case...
            default: return "Unknown";
        }
    }
    // 重载小于运算符，用于map的键比较
    bool operator<(const msg_info& other) const {
        if (mpi_primitive != other.mpi_primitive) {
            return mpi_primitive < other.mpi_primitive;
        }
        if (datatype != other.datatype) {
            // MPI_Datatype的比较可能需要转换为整数类型，因为MPI_Datatype是MPI的类型定义
            return static_cast<int>(datatype) < static_cast<int>(other.datatype);
        }
        return count < other.count;
    }
};

std::map<msg_info, int> message_counts;
std::map<msg_info, double> message_latencies;

void Msg_Count(msg_info info) {
    ++message_counts[info];
} 

void Msg_Latency(msg_info info, double latency) {
    message_latencies[info] += latency;
} 

std::string type_to_string(MPI_Datatype datatype) {
    switch (datatype) {
        case MPI_INT: return "MPI_INT";
        case MPI_FLOAT: return "MPI_FLOAT";
        case MPI_DOUBLE: return "MPI_DOUBLE";
        case MPI_LONG: return "MPI_LONG";
        case MPI_SHORT: return "MPI_SHORT";
        case MPI_UNSIGNED: return "MPI_UNSIGNED";
        case MPI_UNSIGNED_LONG: return "MPI_UNSIGNED_LONG";
        case MPI_UNSIGNED_SHORT: return "MPI_UNSIGNED_SHORT";
        // 其他数据类型的case...
        default: return "Unknown";
    }
}

std::string op_to_string(MPI_Op op) {
    switch (op) {
        case MPI_MAX: return "MPI_MAX";
        case MPI_MIN: return "MPI_MIN";
        case MPI_SUM: return "MPI_SUM";
        case MPI_PROD: return "MPI_PROD";
        case MPI_LAND: return "MPI_LAND";
        case MPI_BAND: return "MPI_BAND";
        case MPI_LOR: return "MPI_LOR";
        case MPI_BOR: return "MPI_BOR";
        case MPI_LXOR: return "MPI_LXOR";
        case MPI_BXOR: return "MPI_BXOR";
        case MPI_MAXLOC: return "MPI_MAXLOC";
        case MPI_MINLOC: return "MPI_MINLOC";
        case MPI_REPLACE: return "MPI_REPLACE";
        default: return "Unknown MPI_Op";
    }
}


extern "C" int if_2_N(int count) {
    int i = 0;
    for (i = 0; i < MAX_2_MSGSIZE; ++i)
        if (!((1LL * count) ^ (1LL << i)))
            return i;
    return -1;
} 

int MPI_Init(int *argc, char ***argv)
{
    int my_rank, err;
    err = PMPI_Init(argc, argv);
    PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &comm_world_procn);
    // ccl_ctx.init(MPI_COMM_WORLD);
    // context_inited = 1;
    PMPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
        printf(" ========== NACLab YCCL-LLM ==========.\n", my_rank);
    if (context_inited++ == 0)
    {
        //初始化
        ccl_ctx.init(MPI_COMM_WORLD);
        // ccl_ctx._ctxp->_opt.dynamical_tune = true;
        // ccl_ctx._ctxp->_opt.mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
        // ccl_ctx._ctxp->_opt.intra_node_reduce_type = MIXED;
        // ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
        // ccl_ctx._ctxp->_opt.inter_node_algorithm = 6;
        // ccl_ctx._ctxp->_opt.pjt_inner_cpy = 1;
        // ccl_ctx._ctxp->_opt.using_non_temporal = 1;
        
        ccl_ctx._ctxp->_opt.intra_node_sync_type = Atomic_as_sync;
        ccl_ctx._ctxp->_opt.intra_node_reduce_byte_unit =(1 << 13);
        ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
        ccl_ctx._ctxp->_opt.intra_node_bcast_type = CacheEfficientBcast;
        // if (count * 4 < 2097152)
        //     ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 8;
        // else
        // ccl_ctx._ctxp->_opt.intra_node_reduce_byte_unit = (count * sizeof(float)) / (ccl_ctx._ctxp->intra_node_procn * ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio);
        // int pipeline_stage=
        ccl_ctx._ctxp->_opt.dynamical_tune = true;
        ccl_ctx._ctxp->_opt.mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
        ccl_ctx._ctxp->_opt.intra_node_reduce_type = MemoryEfficient;
        ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
        ccl_ctx._ctxp->_opt.inter_node_algorithm = 6;
        ccl_ctx._ctxp->_opt.pjt_inner_cpy = 1;
        ccl_ctx._ctxp->_opt.using_non_temporal = 1;
        ccl_ctx._ctxp->_opt.core_per_numa = 8;
        ccl_ctx._ctxp->_opt.numa_n = 8;
        ccl_ctx._ctxp->_opt.qp_vp_count = 4;
        ccl_ctx._ctxp->_opt.MLHA_submitted_max = 4;
        // ccl_ctx._ctxp->_opt.
        // ccl_ctx._ctxp->_opt.core_per_numa = comm_world_procn / 2;
        // ccl_ctx._ctxp->_opt.numa_n = 2;

        // if(yhccl_contexts::_ctx->intra_node_rank == 0)
        // {
        //     puts("================MPI_Allreduce pjt============================");
        // }
        // puts("yhccl");
        PMPI_Barrier(MPI_COMM_WORLD);
    }
#ifdef YIZIMI_TESTTOOL
    if (0) 
    {
        allreduce_total_count = 0;
        allreduce_total_time = 0.0;
        memset(allreduce_msg_count, 0, sizeof(allreduce_msg_count));
        memset(allreduce_msg_time, 0, sizeof(allreduce_msg_time));
    }
    {
        message_counts.clear();
        message_latencies.clear();
    }
#endif
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
    // if (comm == MPI_COMM_WORLD)
    int flag;
    PMPI_Comm_compare(MPI_COMM_WORLD, comm, &flag);
    // if (1)
    if (flag != MPI_UNEQUAL && count >= 512)
    {
        // printf("count=%d\n", count);
        if (context_inited++ == 0)
        {
            //初始化
            ccl_ctx.init(MPI_COMM_WORLD);
            ccl_ctx._ctxp->_opt.intra_node_sync_type = Atomic_as_sync;
            ccl_ctx._ctxp->_opt.intra_node_reduce_byte_unit =(1 << 13);
            ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
            ccl_ctx._ctxp->_opt.intra_node_bcast_type = CacheEfficientBcast;
            // if (count * 4 < 2097152)
            //     ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 8;
            // else
            // ccl_ctx._ctxp->_opt.intra_node_reduce_byte_unit = (count * sizeof(float)) / (ccl_ctx._ctxp->intra_node_procn * ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio);
            // int pipeline_stage=
            ccl_ctx._ctxp->_opt.dynamical_tune = true;
            ccl_ctx._ctxp->_opt.mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
            ccl_ctx._ctxp->_opt.intra_node_reduce_type = MemoryEfficient;
            ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
            ccl_ctx._ctxp->_opt.inter_node_algorithm = 6;
            ccl_ctx._ctxp->_opt.pjt_inner_cpy = 1;
            ccl_ctx._ctxp->_opt.using_non_temporal = 1;
            ccl_ctx._ctxp->_opt.core_per_numa = 8;
            ccl_ctx._ctxp->_opt.numa_n = 8;
            ccl_ctx._ctxp->_opt.qp_vp_count = 4;
            ccl_ctx._ctxp->_opt.MLHA_submitted_max = 4;
            // ccl_ctx._ctxp->_opt.
            // ccl_ctx._ctxp->_opt.core_per_numa = comm_world_procn / 2;
            // ccl_ctx._ctxp->_opt.numa_n = 2;

            // if(yhccl_contexts::_ctx->intra_node_rank == 0)
            // {
            //     puts("================MPI_Allreduce pjt============================");
            // }
            // puts("yhccl");
            PMPI_Barrier(MPI_COMM_WORLD);
        }
        if (count * 4 >= 67108864)
            ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = (1 << 11);
        else if (count * 4 >= 16777216)
            ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = (1 << 10);
        else if (count * 4 >= 8388608)
            ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = (1 << 8);
        else if (count * 4 >= 2097152)
            ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = (1 << 7);
        else 
            ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = (1 << 7);
        //调用all-reduce
        // if (ccl_ctx.global_rank == 0)
        // {
        //     puts("pjt");
        // }
        // export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnuma.so /mnt/d/新工作空间/工作空间/mpi-yhccl/yhccl-build/build/lib/libyhccl.so"
#ifdef YIZIMI_TESTTOOL
        double start_time, end_time;
        // printf("[yizimi_yhccl_test]: allreduce: count=%d datatype=%d op=%d\n", count, datatype, op);
        start_time = MPI_Wtime();
#endif
        yhccl_allreduce(sendbuf, recvbuf, count, datatype, op, 0);
#ifdef YIZIMI_TESTTOOL
        end_time = MPI_Wtime();
        Msg_Count(msg_info(count, datatype, op, 8));
        Msg_Latency(msg_info(count, datatype, op, 8), end_time - start_time);
#endif
        // PMPI_Barrier(MPI_COMM_WORLD);
        //  PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }
    else
    {
#ifdef YIZIMI_TESTTOOL
        double start_time, end_time;
        // if (!allreduce_total_count)
        //     printf("!!! MPI_COMM_WORLD=%X\n", MPI_COMM_WORLD);
        // printf("[yizimi_yhccl_test]: allreduce[naive]: count=%d datatype=%d op=%d comm=%X\n", count, datatype, op, comm);
        start_time = MPI_Wtime();
#endif 
        PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
#ifdef YIZIMI_TESTTOOL
        end_time = MPI_Wtime();
        Msg_Count(msg_info(count, datatype, op, 8));
        Msg_Latency(msg_info(count, datatype, op, 8), end_time - start_time);
#endif 
        // PMPI_Barrier(MPI_COMM_WORLD);
    }
    int elem_sz = -1;
    PMPI_Type_size(datatype, &elem_sz);
    pjt_msg_szcount.push_back(count * elem_sz);
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
    int my_rank, err, procn;
    int namelen;
    char host_name[64], name[64];
    PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &procn);
    PMPI_Get_processor_name(host_name, &namelen);
    // if (my_rank == 0) {
    // printf("[yizimi_test](rk %d): bio finalize start\n", my_rank);
    // }
    PMPI_Barrier(MPI_COMM_WORLD);
    if (if_bio_bcast)
    {
        if (bio_fin_f)
            munmap((void *)bio_fin_f, bio_buff_size);
        if (bio_shm_fd)
            close(bio_shm_fd);

        sprintf(name, "rm /dev/shm/YCCL-bio-daiy-%s -rf", host_name);
        system(name);
    }
    // // if (my_rank == 0) {
    // // printf("[yizimi_test](rk %d): bio finalize OK\n", my_rank);
    // // }
    // // if (my_rank == 0) {
    // // printf("[yizimi_test](rk %d): pjt fmcc-rt finalize start\n", my_rank);
    // // }
    // // if (my_rank == 0)
    // // {
    // //     for (auto sz : pjt_msg_szcount)
    // //         printf("%d\n", sz);
    // // }
    // PMPI_Barrier(MPI_COMM_WORLD);

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
        
        sprintf(name, "rm /dev/shm/YCCL-daiy-%s -rf", host_name);
        system(name);
    }

    PMPI_Barrier(MPI_COMM_WORLD);
#ifdef YIZIMI_TESTTOOL
    if (0) 
    { 
        // allreduce test output
        PMPI_Barrier(MPI_COMM_WORLD);
        double AR_total_time, AR_msg_time[MAX_2_MSGSIZE], AR_unalign_time;
        PMPI_Reduce((void*)(&allreduce_total_time), (void*)(&AR_total_time), 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        PMPI_Reduce((void*)allreduce_msg_time, (void*)AR_msg_time, MAX_2_MSGSIZE, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        PMPI_Reduce((void*)(&allreduce_unalign_time), (void*)(&AR_unalign_time), 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if (my_rank == 0) {
            printf("==== [yizimi_testtool]: Allreduce test ====\n");
            printf("total count : %d\n", allreduce_total_count);
            printf("total time  : %lf\n", AR_total_time);
            printf("unalign time: %lf\n", AR_unalign_time);
            printf("unalign count:%d\n", unaligned_count);
            printf("msg size\tmsg count\tmsg total time(us)\tmsg avg time(us)\n");
            for (int i = 0; i < MAX_2_MSGSIZE; ++i) {
                printf("%d\t\t%d\t\t%lf\t\t%lf\n", (1 << i), allreduce_msg_count[i], AR_msg_time[i] * 1000000.00, 
                    (allreduce_msg_count[i] == 0) ? 0.00 : (AR_msg_time[i] * 1000000.00 / allreduce_msg_count[i]));
            }
            printf("==== [yizimi_testtool]: Over ====\n");
        }
    }
    {
        std::map<msg_info, double> global_avg_latency;
        std::map<msg_info, double> global_min_latency;
        std::map<msg_info, double> global_max_latency;
        double Total_latency;
        for (auto &kv: message_latencies) {
            if (kv.second < 1e-8) continue;
            // std::cout << "Count: " << kv.first.first << ", datatype: " << type_to_string(kv.first.second) << "\n";
            double now_latency = kv.second, total_latency = 0.0;
            PMPI_Reduce(&now_latency, &total_latency, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            global_avg_latency[kv.first] = total_latency / procn;
            PMPI_Reduce(&now_latency, &total_latency, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            global_max_latency[kv.first] = total_latency;
            PMPI_Reduce(&now_latency, &total_latency, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            global_min_latency[kv.first] = total_latency;
        }
        
        if (my_rank == 0) {
            std::cout << "Message Statistics:" << std::endl;
            // if (0)
            {
                // 设置表格标题
                std::cout << std::left << std::setw(10) << "Size"
                        << std::setw(15) << "Count"
                        << std::setw(15) << "Operation"
                        << std::setw(15) << "Datatype"
                        << std::setw(10) << "Messages"
                        << std::setw(15) << "Avg Latency(s)"
                        << std::setw(19) << "Min Latency(s)"
                        << std::setw(19) << "Max Latency(s)" << std::endl;

                // 设置表格分隔线
                std::cout << std::string(80, '-') << std::endl;

                // 输出表格内容
                for (const auto& count_pair : message_counts) {
                    // if (count_pair.first.count <= 32768)
                    //     continue;
                    std::cout << std::setw(10) << get_mpi_type_size(count_pair.first.datatype) * count_pair.first.count
                            << std::setw(15) << count_pair.first.count
                            << std::setw(15) << count_pair.first.primitive_to_string()
                            << std::setw(15) << type_to_string(count_pair.first.datatype)
                            << std::setw(10) << count_pair.second
                            << std::setw(15) << global_avg_latency[count_pair.first]
                            << std::setw(19) << global_min_latency[count_pair.first]
                            << std::setw(19) << global_max_latency[count_pair.first] << std::endl;
                    Total_latency += global_avg_latency[count_pair.first];
                }
                std::cout << "Total latency: " << Total_latency << "\n";
            }
        }
    }
#endif
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