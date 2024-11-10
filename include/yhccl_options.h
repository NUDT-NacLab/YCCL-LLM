/*
 * @Author: pengjintaoHPC 1272813056@qq.com
 * @Date: 2022-06-11 15:08:48
 * @LastEditors: pengjintaoHPC 1272813056@qq.com
 * @LastEditTime: 2022-07-20 17:01:56
 * @FilePath: \yhccl\yhccl_allreduce_pjt\yhccl_options.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef YHCCL_OPTIONS_H
#define YHCCL_OPTIONS_H
#include <vector>
#include <cmath>

enum m_leader_options
{
    DPML = 0,
    PIPELINED_DPML,
    MEMORY_BANDWIDTH_EFFICIENT,
    INTEL_RG,
    RING_AR,
    R_ALL_REDUCE
};
enum intra_reduce_types
{
    CacheEfficient = 0,
    MemoryEfficient,
    MIXED,
    REDUCE_BCAST,
    REDUCE_SCATTER
};
enum intra_node_sync
{
    MPIBarrier_as_sync = 1,
    Atomic_as_sync
};
enum intra_node_bcast
{
    MEMCPY = 1,
    CacheEfficientBcast
};
enum inter_node_allreduce
{
    MPIALLREDUCE = 1,
    THREAD_MPIALLREDUCE_AUTO
};
class allreduce_option
{
public:
    int intra_node_reduce_byte_unit = (1 << 12);
    int inter_node_slice_ct_ratio = 2;
    int intra_node_proc_reduce_bcast_unit = (1 << 14);
    int intra_reduce_slice_slice_size = (1 << 10);
    int inter_node_slice_num = 3;
    int qp_vp_count = 64;
    int open_inter_node_communication = 2;
    int open_intra_node_communication = 1;
    bool overlapping_inter_node_with_intra_node = true;
    int inter_node_algorithm = 6; //(MPI_Iallreduce),(hierarchy allreduce)
    bool dynamical_tune = true;
    int pp_zni = 32; //性能：该数值非常影响小消息的性能
    int pp_chip = 64;
    int pp_node = -1;
    int numa_n;
    int core_per_numa = 2;
    int using_non_temporal = 1;
    int allreduce_tree_K = 2;
    int MLHA_submitted_max = 4;
    // int inter_node_leader_n = 4;
    double intra_node_reduce_thoughput = 4.0;
    m_leader_options mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
    intra_reduce_types intra_node_reduce_type = MIXED;
    // MemoryEfficient;
    intra_node_sync intra_node_sync_type = Atomic_as_sync;
    intra_node_bcast intra_node_bcast_type = CacheEfficientBcast;
    inter_node_allreduce inter_node_allreduce_type = MPIALLREDUCE;
    int bcast_overlap_type = 1;
    int barrier_type = 1;
    int pjt_inner_cpy = 1;

    int NT_boundary_msg_sz;
    // MPIALLREDUCE;
    // THREAD_MPIALLREDUCE_AUTO;
};


typedef struct All_reduce_MCC_search_A_B_C {
    double e;
    // unsigned short A
    int A;
    int B;
} All_reduce_mcc_search_A_B_C;


class bcast_option
{
    public:
    int intra_bcast_slice_size = (1<<22);
    int using_non_temporal_memory_access = 1;
    int using_numa_feature = 1;
};

class reduce_option
{
    public:
        intra_reduce_types intra_node_reduce_type = MIXED;
        int intra_reduce_slice_size = (1 << 20);
        int using_non_temporal_memory_access = 1;
        int using_numa_feature = 1;
};

class allgather_option
{
    public:
        int intra_slice_size = (1 << 18);
        int using_non_temporal_memory_access = 1;
        int using_numa_feature = 1;
};

class MCC_option
{
    public:
    //ms 开销
    double TH3_1B_to_1G_MCC4[31]={0.008565925,0.008267455,0.008565925,0.008267455,0.008328531,0.008274262,0.00828213,0.008310898,0.008395804,0.009445391,0.009523551,0.009895244,0.010372201,0.011330632,0.012721907,0.015736401,0.022786433,0.031330964,0.03900744,0.07835215,0.159973704,0.319759164,0.63214387,1.204324584,2.646722116,5.149018908,10.33859062,20.03704525,41.09678286,84.17838379,164.9535266};
    double BSCC_T6_1B_to_1G_MCC4[31]={0.00362437,0.00362437,0.003624373,0.003662811,0.00361583,0.003648545,0.003892849,0.003988952,0.003892001,0.003799671,0.00396541,0.003954378,0.004203693,0.00474677,0.005952737,0.0080739,0.013704418,0.02176132,0.034758227,0.058646187,0.096999187,0.187873422,0.36519959,0.699666451,1.398272976,2.761129623,5.571251869,11.11208327,21.94913074,43.86911782,87.72513343};
    double BSCC_T6_1B_to_512M_MCC1[35]={1.516666667,1.523333333,1.53,1.506666667,1.71,1.71,1.68,1.746666667,1.823333333,1.88,2.063333333,2.38,2.97,3.73,6.676666667,9.193333333,16.43666667,22.76,33.13,56.67333333,101.31,193.79,391.21,755.22,1515.936667,2999.706667,6033.316667,11947.73,23804.72667,47879.91333,95759.82667,191519.6533,383039.3067,766078.6133,1532157.227};
    double TH2_1B_to_256M_MCC1[29]={4.6,4.6,4.55,4.43,4.42,4.44,4.88,4.91,4.97,5.02,5.08,5.28,5.74,6.83,8.63,15.07,17.84,23.28,34.22,56.2,100.37,188.67,364.26,716.04,1420.76,2834.9,5664.58,11323.63,22712.3};
   
    double estimation(unsigned long long s,int ppn)
    {
        estimation_table = TH3_1B_to_1G_MCC4;
        // if (ppn=4)
        {
            // int sleftindex=
            unsigned long long s1=s;
            int indexp=-1;
            while(s1 >0){
                s1=(s1>>1);
                indexp++;
            }
            int indexq=indexp+1;
            unsigned long long left = (1 << indexp);
            unsigned long long right = (1 << indexq);
            if (s == left)
            {
                return estimation_table[indexp];
            }else
                return estimation_table[indexp] +
                       (estimation_table[indexq] - estimation_table[indexp]) * (s - left) / (right - left);
        }
        return -1.0;
    }
    double * estimation_table;
    double MCC_all_reduce_model(int s,int ppn,int A,int B, int k){
        double a = (A-1)*estimation(1+s/A,ppn);
        double b = estimation(2*s*(k-1)/(A*k),ppn)*(log2(B));
        return a+b;

    }
    double MCC_all_reduce_model_two_level(int s, int ppn, int A, int B, int C, int k)
    {
        double a = 2.0 * (A - 1) * estimation(1 + s / A, ppn);
        double b = 1.2* 2.0 * (B - 1) * estimation(1 + s / (A * B), ppn);
        // double c = estimation(2 * s * (k - 1) / (A * B * k), ppn) * (log2(C));
        double c = 1.2 * estimation(1.0 + 2.0 * s * (k - 1.0) / (A * B), ppn) * (log2(C));
        ;
        return a+b+c;
    }
};
#endif
