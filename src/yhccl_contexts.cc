#define _GNU_SOURCE /* See feature_test_macros(7) */

#include <sched.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <ctime>
using namespace std;
// #include "glex.h"
#include "yhccl_contexts.h"
#include "yhccl_allreduce.h"
#include "yhccl_options.h"
#include "yhccl_communicator.h"

#ifdef NUMA
#include <numa.h>
#include <numaif.h>
#endif

std::mutex yhccl_contexts::init_mtx;
yhccl_contexts *yhccl_contexts::_ctx = 0;
bool yhccl_contexts::am_i_init = false;

void pjtccl_contexts::init(MPI_Comm comm)
{
    _ctxp = new yhccl_contexts();
    _ctxp->init(comm);
}
void pjtccl_contexts::destroy()
{
    _ctxp->destroy();
}

void yhccl_contexts::init_large_msg_allreduce_buffer(int intra_node_rank, int intra_procn, int inter_node_rank)
{
#ifdef NUMA
    {
        // int numa_n = 1 + numa_max_node(); // ffprintf(stderr,stderr,"tid=%d new_mask=%08X was_mask=%08X\n", tid, *(unsigned int *)(&new_mask), *(unsigned int *)(&was_mask));
        // _opt.core_per_numa = processor_per_node / numa_n;
        // if (processor_per_node % numa_n != 0)
        // {
        //     printf("numa数量和核心数量对应不上\n");
        // }
        // int my_numa_id = intra_node_rank / _opt.core_per_numa;
        // numa_run_on_node(my_numa_id);
        // numa_set_localalloc();
        // printf("core_per_numa =%d my_numa_id = %d\n", _opt.core_per_numa, my_numa_id);
        // struct bitmask *bmask = numa_get_interleave_mask();
        // extern struct bitmask *numa_all_nodes_ptr;

        // for (int i = 0; i < intra_node_procn; i++)
        // {
        //     if (intra_node_rank == i)
        //     {
        //         MPI_Barrier(Comm_intra_node);
        //         printf("%d ", intra_node_rank);
        //         for (int i = 0; i < numa_n; i++)
        //         {
        //             printf("%d", numa_bitmask_isbitset(numa_all_nodes_ptr, i));
        //         }
        //         puts("");
        //     }
        // }
        // _opt.numa_n = numa_n;
    // _opt.core_per_numa = intra_node_procn;
    // _opt.numa_n = 1;
        // _opt.numa_n = 3;
        // _opt.core_per_numa = 2;
        
        _opt.core_per_numa = intra_node_procn/2;
        _opt.numa_n = 2;
    }
#else
    _opt.core_per_numa = intra_node_procn/2;
    _opt.numa_n = 2;

    //手动测试numa的情形
    // _opt.core_per_numa = 8;
    // _opt.numa_n = 1;
#endif

    MPI_Barrier(Comm_intra_node);
    char name[100];
    // if (_opt.pp_node == -1)
    //     sprintf(name, "1-%s", host_name);
    // else
    //     sprintf(name, "1-%s-%d", host_name, inter_node_rank);
    MPI_Barrier(Comm_intra_node);
    sprintf(name, "rm /dev/shm/yizimi-test653-%s -rf", host_name);
    system(name);
    MPI_Barrier(Comm_intra_node);
    temp_buf = malloc(large_msg_allreduce_buff_sz + 8 * inter_node_procn);
    if (_opt.pp_node == -1)
        sprintf(name, "yizimi-test653-%s", host_name);
    else
        sprintf(name, "yizimi-test653-%s-%d", host_name, inter_node_rank);
    //节点内规约缓冲区+2个节点间rdma通信缓冲区+ allreduce 规约用的flags
    // long long sz1 = large_msg_allreduce_buff_sz * (intra_procn + 2UL) + global_procn * 8;
    long long sz1 = traditional_shm_allreduce_buff_sz * intra_procn + large_msg_allreduce_buff_sz * (2UL) + intra_node_procn * 64UL;
    long long memory_sz = sz1 + sizeof(unsigned long long) * (1UL << 20);
    int fd = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if(fd==-1)
    {
            fprintf(stderr, "error shm_open fd=%d errno=%d\n", fd, errno);
            exit(0);
    }
    if (intra_node_rank == 0)
    {
        int re = ftruncate64(fd, memory_sz);
        if (re != 0)
        {
            fprintf(stderr, "error ftruncate64 re=%d errno=%d\n", re, errno);
            exit(0);
        };
    }
    MPI_Barrier(Comm_intra_node);
    // if(intra_node_rank == 0)
    //     printf("%s,", host_name);
    // MPI_Barrier(Comm_intra_node);
    larger_msg_allreduce_shareM = mmap(NULL, memory_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    larger_msg_allreduce_my_sendbuf = larger_msg_allreduce_shareM + (long long)intra_node_rank * traditional_shm_allreduce_buff_sz;
    larger_msg_allreduce_result_start_0 = larger_msg_allreduce_shareM + (long long)intra_procn * traditional_shm_allreduce_buff_sz;
    larger_msg_allreduce_result_start_1 = larger_msg_allreduce_result_start_0 + large_msg_allreduce_sendbuff_sz;

    MPI_Barrier(Comm_intra_node);
    //分配节点内同步flag
    intra_node_flags = new void *[intra_node_procn];
    void *p = larger_msg_allreduce_result_start_1 + large_msg_allreduce_sendbuff_sz;
    for (int i = 0; i < intra_node_procn; i++)
    {
        intra_node_flags[i] = p + 64 * i;
        *(char *)(intra_node_flags[i]) = 'P';
        // if (i == 1)
        //     ffprintf(stderr,stderr,"rank1 flag=%c\n", *(volatile char *)(intra_node_flags[i]));
    }
    if (intra_node_rank == 0)
    {
        // memset(p, 0, intra_node_procn * 64);
        memset(larger_msg_allreduce_result_start_0, 0, large_msg_allreduce_buff_sz);
        memset(larger_msg_allreduce_result_start_1, 0, large_msg_allreduce_buff_sz);
    }
    MPI_Barrier(Comm_intra_node);
    // if(global_rank==0)
    // {
    //     puts("==========144============");
    // }
    // MPI_Barrier(Comm_intra_node);
    allreduce_flags = (volatile unsigned long long *)(larger_msg_allreduce_shareM + sz1);
    // __sync_lock_test_and_set(allreduce_flags + intra_node_rank, (long long)intra_node_procn);
    memset(allreduce_flags, 0, sizeof(unsigned long long) * (1UL << 20));
    for (int i = 0; i < intra_procn; i++)
    {
        neigbbor_buffers[i] = (volatile int *)(larger_msg_allreduce_shareM + (long long)traditional_shm_allreduce_buff_sz * i);
    }
    MPI_Barrier(Comm_intra_node);
}


void defineMPIStruct(MPI_Datatype *tstype)
{
    const int count = 3;
    int blocklens[count];
    MPI_Datatype types[count];
    MPI_Aint disps[count];

    {
        types[0] = MPI_DOUBLE;
        blocklens[0] = 1;
        types[1] = MPI_INT;
        blocklens[1] = 1;
        types[2] = MPI_INT;
        blocklens[2] = 1;
    }
    disps[0] = 0;
    disps[1] = sizeof(double);
    disps[2] = sizeof(double)+sizeof(int);

    MPI_Type_create_struct(count, blocklens, disps, types, tstype);
    MPI_Type_commit(tstype);
}
void min_struct(All_reduce_MCC_search_A_B_C *in, All_reduce_MCC_search_A_B_C *inout, int *len, MPI_Datatype *type){
    for (int i=0; i < *len; i++)
        inout[i]  = (inout[i].e < in[i].e) ? inout[i]:in[i];
}
extern MPI_Datatype MCC_search_type;
extern MPI_Op MPI_search_MIN;


void yhccl_contexts::init(MPI_Comm comm)
{

    // puts("init 151");
    Comm_global = comm;
    MPI_Comm_size(Comm_global, &global_procn);
    MPI_Comm_rank(Comm_global, &global_rank);
    // MPI_Barrier(Comm_global);
    // if (global_rank == 0)
    //     puts("init 151");
    // fflush(stdout);
    MPI_Barrier(Comm_global);
    // if (global_rank == 0)
    //     puts("init 152");
    // fflush(stdout);
    MPI_Barrier(Comm_global);
    // ffush(stdout);
    // std::lock_guard<std::mutex> lck(init_mtx, std::adopt_lock);
    yhccl_contexts::_ctx = this;
    //这一函数每个进程只能执行一次，可在任意一个通信子上做初始化
    if (yhccl_contexts::am_i_init == true)
    {
        std::cout << "初始化错误，每个进程只能在一个通信子上初始化yhccl" << std::endl;
        fflush(stdout);
    }
    else
    {
        yhccl_contexts::am_i_init = true;
    }


    MPI_Barrier(Comm_global);
    // if(global_rank == 0)
    // {
    //     printf(" yhccl_contexts::init rank=%d\n",global_rank);
    //     fflush(stdout);
    // }
    MPI_Barrier(Comm_global);
    //第一步进行节点内节点间通信子划分：
    //注意节点内
    int namelen;
    MPI_Get_processor_name(host_name, &namelen);
    int bytes = global_procn * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
    int color = 0;
    //决定同一个节点内的进程color，方便分割节点通信子
    {
        char *host_names = (char *)malloc(bytes);
#ifdef PJT_MPI
        // puts("错误，使用openmpi时请调用其它模块的allgather");
        PMPI_Allgather(host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                       host_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, Comm_global);
#else
        PMPI_Allgather(host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                       host_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, Comm_global);

#endif
        char *prev = host_names;
        for (int i = 1; i <= global_rank; i++)
        {
            char *p = host_names + i * MPI_MAX_PROCESSOR_NAME;
            if (strcmp(p, prev) != 0)
                color++;
            prev = p;
        }
        free(host_names);
    }
    //节点内通信子
    if (_opt.pp_node != -1)
    {
        color = global_rank / _opt.pp_node;
        if (global_rank == 0)
        {
            fprintf(stderr, "PJT：测试模式：ppn = %d\n", _opt.pp_node);
        }
    }
    MPI_Comm_split(Comm_global, color, global_rank, &Comm_intra_node);
    MPI_Comm_size(Comm_intra_node, &intra_node_procn);
    MPI_Comm_rank(Comm_intra_node, &intra_node_rank);

    //进程绑定
    if(0)
    {
        cpu_set_t new_mask;
        cpu_set_t was_mask;
        CPU_ZERO(&new_mask);
        int rank = yhccl_contexts::_ctx->intra_node_rank;
        int procn = yhccl_contexts::_ctx->intra_node_procn;
        int bind_dest =  rank;
        // if(rank >= procn/2)
        //     bind_dest = 32 + rank - (procn/2);
        
        // CPU_SET(yhccl_contexts::_ctx->intra_node_rank, &new_mask);

        // int start=0;
        // if(rank >= procn/2){
        //     start = 1;
        // }
        //交叉分布的机器
        // bind_dest=start + 2 * (rank % (procn >> 1));
        // printf("rank %d bind to %d\n",rank,bind_dest);
        CPU_SET(bind_dest, &new_mask);
        pthread_t thread;
        thread = pthread_self();
        if (pthread_setaffinity_np(thread, sizeof(new_mask), &new_mask) != 0)
        {
            fprintf(stderr, "Error: pthread_setaffinity_np(%d, sizeof(new_mask), &new_mask)\n", rank);
        }

        // ffprintf(stderr, stderr, "tid=%d new_mask=%08X was_mask=%08X\n", tid, *(unsigned int *)(&new_mask), *(unsigned int *)(&was_mask));
    }
    // cout << global_rank << " " << intra_node_rank << " " << color << endl;

    //节点间通信子
    color = intra_node_rank;
    MPI_Comm_split(comm, color, global_rank, &Comm_inter_node);
    MPI_Comm_size(Comm_inter_node, &inter_node_procn);
    MPI_Comm_rank(Comm_inter_node, &inter_node_rank);

    // if (intra_node_rank == 0)
    // if (0)
    {
        //板内通信子
        color = inter_node_rank / (_opt.pp_zni);
        MPI_Comm_split(Comm_inter_node, color, inter_node_rank, &Comm_intra_zni);
        MPI_Comm_rank(Comm_intra_zni, &intra_zni_rank);
        MPI_Comm_size(Comm_intra_zni, &intra_zni_procn);

        //划分芯片内通信子
        color = (inter_node_rank % _opt.pp_zni) * 1e6 + (inter_node_rank / (_opt.pp_chip * _opt.pp_zni));
        MPI_Comm_split(Comm_inter_node, color, inter_node_rank, &Comm_intra_chip);
        MPI_Comm_rank(Comm_intra_chip, &intra_chip_rank);
        MPI_Comm_size(Comm_intra_chip, &intra_chip_procn);

        //划分芯片间通信子
        color = inter_node_rank % (_opt.pp_chip * _opt.pp_zni);
        MPI_Comm_split(Comm_inter_node, color, inter_node_rank, &Comm_inter_chip);
        MPI_Comm_rank(Comm_inter_chip, &inter_chip_rank);
        MPI_Comm_size(Comm_inter_chip, &inter_chip_procn);
    }
    MPI_Barrier(Comm_global);
    // for (int i = 0; i < inter_node_procn; i++)
    // {
    //     MPI_Barrier(Comm_global);
    //     if (inter_node_rank == i)
    //         fprintf(stderr, "global rank=%d intra_node_rank=%d, intra_zni_rank=%d,intra_chip_rank=%d,inter_chip_rank=%d \n",
    //                 global_rank, intra_node_rank, intra_zni_rank, intra_chip_rank, inter_chip_rank);
    //     fflush(stdout);
    // }
    // MPI_Barrier(Comm_global);
    // exit(0);
    //接下来初始化节点内共享内存
    //设置节点间通信子的leader数量。
    // if (0)

    MPI_Barrier(Comm_global);
    // if(global_rank == 0)
    // {
    //     printf(" yhccl_contexts::init before init_large_msg_allreduce_buffer rank=%d\n",global_rank);
    //     fflush(stdout);
    // }
    MPI_Barrier(Comm_global);
    pjt_leadern = min(pjt_leadern, intra_node_procn);
    processor_per_node = sysconf(_SC_NPROCESSORS_ONLN);
    init_large_msg_allreduce_buffer(intra_node_rank, intra_node_procn, inter_node_rank);
    MPI_Barrier(Comm_global);

    // if (intra_node_rank == 0)
    {
#ifdef GLEX_RDMA
        //接下来初始化RDMA
        // RDMA的端口数量
        _rdma_infoV.init(this);
        if (global_rank == 0)
            puts("finish rdma init");

#endif
    }

    // if (intra_node_procn < processor_per_node / 2)
    // using_multi_thread_communication = true;
    // else
    //     using_multi_thread_communication = false;
    // if (using_multi_thread_communication)
    // yhccl_communicator::start();
    init_allreduce_algorithm();
    // puts("init check");

    {
        //初始化all-reduce节点间搜索使用的通信
        defineMPIStruct(&MCC_search_type);
        MPI_Op_create(min_struct, 1, &MPI_search_MIN);
    }
    MPI_Barrier(Comm_global);
    // if(global_rank == 0)
    // {
    //     printf(" yhccl_contexts::init rank=%d finish\n",global_rank);
    //     fflush(stdout);
    // }
    MPI_Barrier(Comm_global);
}

void yhccl_contexts::destroy()
{
    MPI_Type_free(&MCC_search_type);
    // MPI_barrier()
    destroy_allreduce_algorithm();
    //销毁注册的内存
#ifdef GLEX_RDMA
    _rdma_infoV.free();
#endif
    free(temp_buf);
    // A::start();
    // if (using_multi_thread_communication)
    //     yhccl_communicator::destroy(0);
    // delete allreduce_flags;
}

// #define PJT_AVX_ASSEMBLY_MEMCPY

template <typename T>
void yhccl_sum_op(const void *invec, void *inoutvec, int *len,
                  MPI_Datatype *datatype = NULL)
{
    T *in = (T *)invec;
    T *inout = (T *)inoutvec;
    for (int i = 0; i < *len; i++)
    {
        inout[i] += in[i];
    }
}

template <typename T>
void yhccl_max_op(const void *invec, void *inoutvec, int *len,
                  MPI_Datatype *datatype = NULL)
{
    T *in = (T *)invec;
    T *inout = (T *)inoutvec;
    // #pragma omp simd
    for (int i = 0; i < *len; i++)
    {
        inout[i] = std::max(inout[i], in[i]);
    }
}

template <typename T>
void yhccl_min_op(const void *invec, void *inoutvec, int *len,
                  MPI_Datatype *datatype = NULL)
{
    T *in = (T *)invec;
    T *inout = (T *)inoutvec;
    // #pragma omp simd
    for (int i = 0; i < *len; i++)
    {
        inout[i] = std::min(inout[i], in[i]);
    }
}
// #undef PJT_AVX_ASSEMBLY_MEMCPY
#ifdef PJT_AVX_ASSEMBLY_MEMCPY
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <stdio.h>



inline void float_sum_calc_line_1_nt(const void *s, void *t)
{
    //一个cache line 为64字节或者512位
    __m128 s1, s2, s3, s4;
    __m128 t1, t2, t3, t4;
    // s1=_mm_load_ps((const float *)s);
    // t1=_mm_load_ps((const float *)t);
    // s2=_mm_load_ps((const float *)(s+16));
    // t2=_mm_load_ps((const float *)(t+16));
    // s3=_mm_load_ps((const float *)(s+32));
    // t3=_mm_load_ps((const float *)(t+32));
    // s4=_mm_load_ps((const float *)(s+48));
    // t4=_mm_load_ps((const float *)(t+48));

    // t1=_mm_add_ps(s1,t1);
    // t2=_mm_add_ps(s2,t2);
    // t3=_mm_add_ps(s3,t3);
    // t4=_mm_add_ps(s4,t4);

    // _mm_store_ps(( float *)t,t1);
    // _mm_store_ps(( float *)(t+16),t2);
    // _mm_store_ps(( float *)(t+32),t3);
    // _mm_store_ps(( float *)(t+48),t4);
    // __asm __volatile__(
    //     "movntdqa (%8), %0\n\t"   //装载源
    //     "movaps  (%9), %1\n\t"    //装在t
    //     "movntdqa 16(%8), %2\n\t" //装载源
    //     "movaps  16(%9), %3\n\t"  //装在t
    //     "movntdqa 32(%8), %4\n\t" //装载源
    //     "movaps  32(%9), %5\n\t"  //装在t
    //     "movntdqa 48(%8), %6\n\t" //装载源
    //     "movaps  48(%9), %7\n\t"  //装在t
    //     "addps %0,%1\n\t"         //浮点加法
    //     "addps %2,%3\n\t"         //浮点加法
    //     "addps %4,%5\n\t"         //浮点加法
    //     "addps %6,%7\n\t"         //浮点加法
    //     "movaps %1,(%9)\n\t"      //返回存储
    //     "movaps %3,16(%9)\n\t"    //返回存储
    //     "movaps %5,32(%9)\n\t"    //返回存储
    //     "movaps %7,48(%9)\n\t"    //返回存储
    //     : "=x"(s1), "=x"(t1), "=x"(s2), "=x"(t2), "=x"(s3), "=x"(t3), "=x"(s4), "=x"(t4)
    //     : "r"(s), "r"(t)
    //     : "memory");
    __asm __volatile__(
        "movaps (%8), %0\n\t"    //装载源
        "movaps  (%9), %1\n\t"   //装在t
        "movaps 16(%8), %2\n\t"  //装载源
        "movaps  16(%9), %3\n\t" //装在t
        "movaps 32(%8), %4\n\t"  //装载源
        "movaps  32(%9), %5\n\t" //装在t
        "movaps 48(%8), %6\n\t"  //装载源
        "movaps  48(%9), %7\n\t" //装在t
        "addps %0,%1\n\t"        //浮点加法
        "addps %2,%3\n\t"        //浮点加法
        "addps %4,%5\n\t"        //浮点加法
        "addps %6,%7\n\t"        //浮点加法
        "movaps %1,(%9)\n\t"     //返回存储
        "movaps %3,16(%9)\n\t"   //返回存储
        "movaps %5,32(%9)\n\t"   //返回存储
        "movaps %7,48(%9)\n\t"   //返回存储
        : "=x"(s1), "=x"(t1), "=x"(s2), "=x"(t2), "=x"(s3), "=x"(t3), "=x"(s4), "=x"(t4)
        : "r"(s), "r"(t)
        : "memory");

    // __asm __volatile__(
    // "movaps (%8), %0\n\t"   //装载源
    // "movntdqa  (%9), %1\n\t"    //装在t
    // "movaps 16(%8), %2\n\t" //装载源
    // "movntdqa  16(%9), %3\n\t"  //装在t
    // "movaps 32(%8), %4\n\t" //装载源
    // "movntdqa  32(%9), %5\n\t"  //装在t
    // "movaps 48(%8), %6\n\t" //装载源
    // "movntdqa  48(%9), %7\n\t"  //装在t
    // "addps %0,%1\n\t"         //浮点加法
    // "addps %2,%3\n\t"         //浮点加法
    // "addps %4,%5\n\t"         //浮点加法
    // "addps %6,%7\n\t"         //浮点加法
    // "movntps %1,(%9)\n\t"      //返回存储
    // "movntps %3,16(%9)\n\t"    //返回存储
    // "movntps %5,32(%9)\n\t"    //返回存储
    // "movntps %7,48(%9)\n\t"    //返回存储
    // : "=x"(s1), "=x"(t1), "=x"(s2), "=x"(t2), "=x"(s3), "=x"(t3), "=x"(s4), "=x"(t4)
    // : "r"(s), "r"(t)
    // : "memory");
}
inline void float_sum_calc_line_1_nt_new(const void *s, void *t)
{
    //一个cache line 为64字节或者512位
    __m128 s1, s2, s3, s4;
    __m128 t1, t2, t3, t4;
    __asm __volatile__(
        "movntdqa (%8), %0\n\t"    //装载源
        "movaps  (%9), %1\n\t"   //装在t
        "movntdqa 16(%8), %2\n\t"  //装载源
        "movaps  16(%9), %3\n\t" //装在t
        "movntdqa 32(%8), %4\n\t"  //装载源
        "movaps  32(%9), %5\n\t" //装在t
        "movntdqa 48(%8), %6\n\t"  //装载源
        "movaps  48(%9), %7\n\t" //装在t
        "addps %0,%1\n\t"        //浮点加法
        "addps %2,%3\n\t"        //浮点加法
        "addps %4,%5\n\t"        //浮点加法
        "addps %6,%7\n\t"        //浮点加法
        "movaps %1,(%9)\n\t"     //返回存储
        "movaps %3,16(%9)\n\t"   //返回存储
        "movaps %5,32(%9)\n\t"   //返回存储
        "movaps %7,48(%9)\n\t"   //返回存储
        : "=x"(s1), "=x"(t1), "=x"(s2), "=x"(t2), "=x"(s3), "=x"(t3), "=x"(s4), "=x"(t4)
        : "r"(s), "r"(t)
        : "memory");
}

inline void float_sum_calc_line_1(const void *s, void *t)
{
    //一个cache line 为64字节或者512位
    __m128 s1, s2, s3, s4;
    __m128 t1, t2, t3, t4;
    s1 = _mm_load_ps((const float *)s);
    t1 = _mm_load_ps((const float *)t);
    s2 = _mm_load_ps((const float *)(s + 16));
    t2 = _mm_load_ps((const float *)(t + 16));
    s3 = _mm_load_ps((const float *)(s + 32));
    t3 = _mm_load_ps((const float *)(t + 32));
    s4 = _mm_load_ps((const float *)(s + 48));
    t4 = _mm_load_ps((const float *)(t + 48));

    t1 = _mm_add_ps(s1, t1);
    t2 = _mm_add_ps(s2, t2);
    t3 = _mm_add_ps(s3, t3);
    t4 = _mm_add_ps(s4, t4);

    _mm_store_ps((float *)t, t1);
    _mm_store_ps((float *)(t + 16), t2);
    _mm_store_ps((float *)(t + 32), t3);
    _mm_store_ps((float *)(t + 48), t4);
    // __asm __volatile__(
    //     "movaps (%8), %0\n\t"    //装载源
    //     "movaps  (%9), %1\n\t"   //装在t
    //     "movaps 16(%8), %2\n\t"  //装载源
    //     "movaps  16(%9), %3\n\t" //装在t
    //     "movaps 32(%8), %4\n\t"  //装载源
    //     "movaps  32(%9), %5\n\t" //装在t
    //     "movaps 48(%8), %6\n\t"  //装载源
    //     "movaps  48(%9), %7\n\t" //装在t
    //     "addps %0,%1\n\t"        //浮点加法
    //     "addps %2,%3\n\t"        //浮点加法
    //     "addps %4,%5\n\t"        //浮点加法
    //     "addps %6,%7\n\t"        //浮点加法
    //     "movaps %1,(%9)\n\t"     //返回存储
    //     "movaps %3,16(%9)\n\t"   //返回存储
    //     "movaps %5,32(%9)\n\t"   //返回存储
    //     "movaps %7,48(%9)\n\t"   //返回存储
    //     : "=x"(s1), "=x"(t1), "=x"(s2), "=x"(t2), "=x"(s3), "=x"(t3), "=x"(s4), "=x"(t4)
    //     : "r"(s), "r"(t)
    //     : "memory");
}

void yhccl_sum_float_op_nt(const void *invec, void *inoutvec, int *len, MPI_Datatype *datatype = NULL)
{
    // puts("410")
    // memory_fence();
    // dest和source必须按128位/16 Byte对齐
    size_t a = ((size_t)invec & 0xF);
    size_t b = ((size_t)inoutvec & 0xF);
    int elem_sz = 4;
    // if (a == 0 && b == 0)
    if (*len >= 32)
    {
        size_t sz = (*len) * elem_sz;
        void *end_addr = inoutvec + sz;
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, inoutvec);
        while ((size_t)((size_t)inoutvec & 0x3F) != 0 && inoutvec < end_addr)
        {
            *(float *)(inoutvec) += *(const float *)invec;
            invec += elem_sz;
            inoutvec += elem_sz;
        }

        // cache对齐部分的加法
        while (inoutvec + 64 < end_addr - 8192)
        {
            // float_sum_calc_line(invec, inoutvec);
            _mm_prefetch((const char *)(invec + 8192), _MM_HINT_NTA);
            _mm_prefetch((const char *)(inoutvec + 8192), _MM_HINT_T0);
            
            // _mm_prefetch((const char *)(invec + 8192), _MM_HINT_NTA);
            // _mm_prefetch((const char *)(inoutvec + 8192), _MM_HINT_NTA);
            float_sum_calc_line_1_nt(invec, inoutvec);
            // float_sum_calc_line_2(invec, inoutvec);

            inoutvec += 64;
            invec += 64;
        }
        while (inoutvec + 64 < end_addr)
        {
            // float_sum_calc_line(invec, inoutvec);
            float_sum_calc_line_1_nt(invec, inoutvec);
            // float_sum_calc_line_2(invec, inoutvec);

            inoutvec += 64;
            invec += 64;
        }
        while (inoutvec < end_addr)
        {
            *(float *)(inoutvec) += *(const float *)invec;
            invec += elem_sz;
            inoutvec += elem_sz;
        }
        memory_fence();
    }
    else
    {
        // puts("yhccl_sum_float_op_nt 建议对缓冲区进行内存对齐");
        // exit(0);
        float *in = (float *)invec;
        float *inout = (float *)inoutvec;
        for (int i = 0; i < *len; i++)
        {
            inout[i] += in[i];
        }
    }
}

void yhccl_sum_float_op_prefetch_T2(const void *invec, void *inoutvec, int *len, MPI_Datatype *datatype = NULL)
{
    // puts("410")
    // memory_fence();
    // dest和source必须按128位/16 Byte对齐
    size_t a = ((size_t)invec & 0xF);
    size_t b = ((size_t)inoutvec & 0xF);
    int elem_sz = 4;
    // if (a == 0 && b == 0)
    if (*len >= 32)
    {
        size_t sz = (*len) * elem_sz;
        void *end_addr = inoutvec + sz;
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, inoutvec);
        while ((size_t)((size_t)inoutvec & 0x3F) != 0 && inoutvec < end_addr)
        {
            *(float *)(inoutvec) += *(const float *)invec;
            invec += elem_sz;
            inoutvec += elem_sz;
        }

        // cache对齐部分的加法
        while (inoutvec + 64 < end_addr - 8192)
        {
            // float_sum_calc_line(invec, inoutvec);
            _mm_prefetch((const char *)(invec + 8192), _MM_HINT_T2);
            _mm_prefetch((const char *)(inoutvec + 8192), _MM_HINT_T2);
            
            // _mm_prefetch((const char *)(invec + 8192), _MM_HINT_NTA);
            // _mm_prefetch((const char *)(inoutvec + 8192), _MM_HINT_NTA);
            float_sum_calc_line_1_nt(invec, inoutvec);
            // float_sum_calc_line_2(invec, inoutvec);

            inoutvec += 64;
            invec += 64;
        }
        while (inoutvec + 64 < end_addr)
        {
            // float_sum_calc_line(invec, inoutvec);
            float_sum_calc_line_1_nt(invec, inoutvec);
            // float_sum_calc_line_2(invec, inoutvec);

            inoutvec += 64;
            invec += 64;
        }
        while (inoutvec < end_addr)
        {
            *(float *)(inoutvec) += *(const float *)invec;
            invec += elem_sz;
            inoutvec += elem_sz;
        }
        memory_fence();
    }
    else
    {
        // puts("yhccl_sum_float_op_prefetch_T2 建议对缓冲区进行内存对齐");
        // exit(0);
        float *in = (float *)invec;
        float *inout = (float *)inoutvec;
        for (int i = 0; i < *len; i++)
        {
            inout[i] += in[i];
        }
    }
}
void yhccl_sum_float_op(const void *invec, void *inoutvec, int *len, MPI_Datatype *datatype = NULL)
{
    // puts("410")
    // memory_fence();
    // dest和source必须按128位/16 Byte对齐
    size_t a = ((size_t)invec & 0xF);
    size_t b = ((size_t)inoutvec & 0xF);
    int elem_sz = 4;
    // if (a == 0 && b == 0)
    if (*len >= 32)
    {
        size_t sz = (*len) * elem_sz;
        void *end_addr = inoutvec + sz;
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, inoutvec);
        while ((size_t)((size_t)inoutvec & 0x3F) != 0 && inoutvec < end_addr)
        {
            *(float *)(inoutvec) += *(const float *)invec;
            invec += elem_sz;
            inoutvec += elem_sz;
        }

        // cache对齐部分的加法
        while (inoutvec + 64 < end_addr - 1024)
        {
            // float_sum_calc_line(invec, inoutvec);
            // _mm_prefetch(inoutvec + 1024, _MM_HINT_T0);
            float_sum_calc_line_1(invec, inoutvec);
            // float_sum_calc_line_2(invec, inoutvec);

            inoutvec += 64;
            invec += 64;
        }
        while (inoutvec + 64 < end_addr)
        {
            // float_sum_calc_line(invec, inoutvec);
            float_sum_calc_line_1(invec, inoutvec);
            // float_sum_calc_line_2(invec, inoutvec);

            inoutvec += 64;
            invec += 64;
        }
        while (inoutvec < end_addr)
        {
            *(float *)(inoutvec) += *(const float *)invec;
            invec += elem_sz;
            inoutvec += elem_sz;
        }
        memory_fence();
    }
    else
    {
        // puts("建议对缓冲区进行内存对齐");
        // exit(0);
        float *in = (float *)invec;
        float *inout = (float *)inoutvec;
        for (int i = 0; i < *len; i++)
        {
            inout[i] += in[i];
        }
    }
}
inline void float_sum_calc_line_A_plus_B_eq_C(const void *a, const void *b,void *c)
{
    __m128 s1, s2, s3, s4;
    __m128 t1, t2, t3, t4;
    // s1 = _mm_load_ps((const float *)a);
    // t1 = _mm_load_ps((const float *)b);
    // s2 = _mm_load_ps((const float *)(a + 16));
    // t2 = _mm_load_ps((const float *)(b + 16));
    // s3 = _mm_load_ps((const float *)(a + 32));
    // t3 = _mm_load_ps((const float *)(b + 32));
    // s4 = _mm_load_ps((const float *)(a + 48));
    // t4 = _mm_load_ps((const float *)(b + 48));

    // t1 = _mm_add_ps(s1, t1);
    // t2 = _mm_add_ps(s2, t2);
    // t3 = _mm_add_ps(s3, t3);
    // t4 = _mm_add_ps(s4, t4);

    // _mm_stream_ps((float *)c, t1);
    // _mm_stream_ps((float *)(c + 16), t2);
    // _mm_stream_ps((float *)(c + 32), t3);
    // _mm_stream_ps((float *)(c + 48), t4);

    __asm __volatile__(
        "movntdqa (%8), %0\n\t"    //装载源
        "movaps  (%9), %1\n\t"   //装在t
        "movntdqa 16(%8), %2\n\t"  //装载源
        "movaps  16(%9), %3\n\t" //装在t
        "movntdqa 32(%8), %4\n\t"  //装载源
        "movaps  32(%9), %5\n\t" //装在t
        "movntdqa 48(%8), %6\n\t"  //装载源
        "movaps  48(%9), %7\n\t" //装在t
        "addps %0,%1\n\t"        //浮点加法
        "addps %2,%3\n\t"        //浮点加法
        "addps %4,%5\n\t"        //浮点加法
        "addps %6,%7\n\t"        //浮点加法
        "movntps %1,(%10)\n\t"   //返回存储
        "movntps %3,16(%10)\n\t"   //返回存储
        "movntps %5,32(%10)\n\t"   //返回存储
        "movntps %7,48(%10)\n\t"   //返回存储
        : "=x"(s1), "=x"(t1), "=x"(s2), "=x"(t2), "=x"(s3), "=x"(t3), "=x"(s4), "=x"(t4)
        : "r"(a), "r"(b), "r"(c)
        : "memory");
}


void yhccl_sum_float_op_A_plus_B_eq_C(const void *a, const void *b,const void *c, int *len)
{
    // puts("410")
    // memory_fence();
    // dest和source必须按128位/16 Byte对齐
    int elem_sz = 4;
    // if (a == 0 && b == 0)
    if (*len >= 128 && ((size_t)((size_t)c & 0xF) ==  (size_t)((size_t)a & 0xF)))
    {
        size_t sz = (*len) * elem_sz;
        void *end_addr = c + sz;
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, inoutvec);
        while ((size_t)((size_t)c & 0x3F) != 0 && c < end_addr)
        {
            *(float *)(c) = *(const float *)a + *(const float *)b;
            c += elem_sz;
            a += elem_sz;
            b += elem_sz;
        }

        // cache对齐部分的加法
        while (c + 64 < end_addr - 1024)
        {
            // float_sum_calc_line(invec, inoutvec);
            // _mm_prefetch(c + 1024, _MM_HINT_T0);
            float_sum_calc_line_A_plus_B_eq_C(a, b, c);
            c += 64;
            a += 64;
            b += 64;
        }
        while (c + 64 < end_addr)
        {
            // float_sum_calc_line(invec, inoutvec);
            float_sum_calc_line_A_plus_B_eq_C(a, b, c);
            c += 64;
            a += 64;
            b += 64;
        }
        while (c < end_addr)
        {
            *(float *)(c) = *(const float *)a + *(const float *)b;
            c += elem_sz;
            a += elem_sz;
            b += elem_sz;
        }
        memory_fence();
    }
    else
    {
        puts("708");
        exit(0);
        for (int i = 0; i < *len; i++)
        {
            *(float *)(c) = *(const float *)a + *(const float *)b;
            c += elem_sz;
            a += elem_sz;
            b += elem_sz;
        }
    }
}

#else
void yhccl_sum_float_op_A_plus_B_eq_C(const void *a, const void *b,const void *c, int *len)
{
    int elem_sz = sizeof(float);
    for (int i = 0; i < *len; i++)
    {
        *(float *)(c) = *(const float *)a + *(const float *)b;
        c += elem_sz;
        a += elem_sz;
        b += elem_sz;
        }
}

#endif

yhccl_op operation_switch(MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op reducefp)
{
    yhccl_op reduce_op = 0;
    if (reducefp != 0)
        return reducefp;
    if (mpi_op == MPI_SUM)
    {
        if (mpitype == MPI_UINT8_T)
            reduce_op = yhccl_sum_op<uint8_t>;
        else if (mpitype == MPI_INT8_T)
            reduce_op = yhccl_sum_op<int8_t>;
        else if (mpitype == MPI_UINT16_T)
            reduce_op = yhccl_sum_op<uint16_t>;
        else if (mpitype == MPI_INT16_T)
            reduce_op = yhccl_sum_op<int16_t>;
        else if (mpitype == MPI_UINT32_T)
            reduce_op = yhccl_sum_op<uint32_t>;
        else if (mpitype == MPI_INT32_T)
            reduce_op = yhccl_sum_op<int32_t>;
        else if (mpitype == MPI_UINT64_T)
            reduce_op = yhccl_sum_op<uint64_t>;
        else if (mpitype == MPI_INT64_T)
            reduce_op = yhccl_sum_op<int64_t>;
        else if (mpitype == MPI_FLOAT)
        {

#ifdef PJT_AVX_ASSEMBLY_MEMCPY
            // reduce_op = yhccl_sum_float_op_nt;
            // if (yhccl_contexts::_ctx->_opt.using_non_temporal == 0)
            //     // reduce_op = yhccl_sum_op<float>;
            //     reduce_op = yhccl_sum_float_op_prefetch_T2;
            // else{
            //     reduce_op = yhccl_sum_float_op_nt;
            // }
                reduce_op = yhccl_sum_float_op_prefetch_T2;
            // else
            // reduce_op = yhccl_sum_float_op;
#else
            reduce_op = yhccl_sum_op<float>;
#endif
        }
        else if (mpitype == MPI_DOUBLE)
            reduce_op = yhccl_sum_op<double>;
        else if (mpitype == MPI_C_BOOL)
            reduce_op = yhccl_sum_op<bool>;
    }
    else if (mpi_op == MPI_MAX)
    {
        if (mpitype == MPI_UINT8_T)
            reduce_op = yhccl_max_op<uint8_t>;
        else if (mpitype == MPI_INT8_T)
            reduce_op = yhccl_max_op<int8_t>;
        else if (mpitype == MPI_UINT16_T)
            reduce_op = yhccl_max_op<uint16_t>;
        else if (mpitype == MPI_INT16_T)
            reduce_op = yhccl_max_op<int16_t>;
        else if (mpitype == MPI_UINT32_T)
            reduce_op = yhccl_max_op<uint32_t>;
        else if (mpitype == MPI_INT32_T)
            reduce_op = yhccl_max_op<int32_t>;
        else if (mpitype == MPI_UINT64_T)
            reduce_op = yhccl_max_op<uint64_t>;
        else if (mpitype == MPI_INT64_T)
            reduce_op = yhccl_max_op<int64_t>;
        else if (mpitype == MPI_FLOAT)
            reduce_op = yhccl_max_op<float>;
        else if (mpitype == MPI_DOUBLE)
            reduce_op = yhccl_max_op<double>;
        else if (mpitype == MPI_C_BOOL)
            reduce_op = yhccl_max_op<bool>;
    }
    else if (mpi_op == MPI_MIN)
    {
        if (mpitype == MPI_UINT8_T)
            reduce_op = yhccl_min_op<uint8_t>;
        else if (mpitype == MPI_INT8_T)
            reduce_op = yhccl_min_op<int8_t>;
        else if (mpitype == MPI_UINT16_T)
            reduce_op = yhccl_min_op<uint16_t>;
        else if (mpitype == MPI_INT16_T)
            reduce_op = yhccl_min_op<int16_t>;
        else if (mpitype == MPI_UINT32_T)
            reduce_op = yhccl_min_op<uint32_t>;
        else if (mpitype == MPI_INT32_T)
            reduce_op = yhccl_min_op<int32_t>;
        else if (mpitype == MPI_UINT64_T)
            reduce_op = yhccl_min_op<uint64_t>;
        else if (mpitype == MPI_INT64_T)
            reduce_op = yhccl_min_op<int64_t>;
        else if (mpitype == MPI_FLOAT)
            reduce_op = yhccl_min_op<float>;
        else if (mpitype == MPI_DOUBLE)
            reduce_op = yhccl_min_op<double>;
        else if (mpitype == MPI_C_BOOL)
            reduce_op = yhccl_min_op<bool>;
    }
    else
    {
        return 0;
        if (reducefp != 0)
        {
            //默认类型自定义op
            reduce_op = reducefp;
        }
        else
        {
            // std::cout << "目前支持的allreduce操作只有自定义操作，或者sum，min，max。有其他需求请联系1272813056@qq.com 彭晋韬" << std::endl;
        }
    }
    // unsigned long long swichv = mpi_op;
    // unsigned long long mpi_sum_int = MPI_SUM;
    // unsigned long long mpi_max_int = MPI_MAX;
    // unsigned long long mpi_min_int = MPI_MIN;
    // switch (swichv)
    // {
    // case mpi_sum_int:
    //     if (reducefp != 0)
    //         return reducefp;
    //     switch (mpitype)
    //     {
    //     case MPI_UINT8_T:
    //         reduce_op = yhccl_sum_op<uint8_t>;
    //         break;
    //     case MPI_INT8_T:
    //         reduce_op = yhccl_sum_op<int8_t>;
    //         break;
    //     case MPI_UINT16_T:
    //         reduce_op = yhccl_sum_op<uint16_t>;
    //         break;
    //     case MPI_INT16_T:
    //         reduce_op = yhccl_sum_op<int16_t>;
    //         break;
    //     case MPI_UINT32_T:
    //         reduce_op = yhccl_sum_op<uint32_t>;
    //         break;
    //     case MPI_INT32_T:
    //         reduce_op = yhccl_sum_op<int32_t>;
    //         break;
    //     case MPI_UINT64_T:
    //         reduce_op = yhccl_sum_op<uint64_t>;
    //         break;
    //     case MPI_INT64_T:
    //         reduce_op = yhccl_sum_op<int64_t>;
    //         break;
    //     case MPI_FLOAT:
    //         reduce_op = yhccl_sum_op<float>;
    //         break;
    //     case MPI_DOUBLE:
    //         reduce_op = yhccl_sum_op<double>;
    //     case MPI_C_BOOL:
    //         reduce_op = yhccl_sum_op<bool>;
    //     default:
    //         break;
    //     };
    //     break;
    // case mpi_max_int:
    //     if (reducefp != 0)
    //         return reducefp;
    //     switch (mpitype)
    //     {
    //     case MPI_UINT8_T:
    //         reduce_op = yhccl_max_op<uint8_t>;
    //         break;
    //     case MPI_INT8_T:
    //         reduce_op = yhccl_max_op<int8_t>;
    //         break;
    //     case MPI_UINT16_T:
    //         reduce_op = yhccl_max_op<uint16_t>;
    //         break;
    //     case MPI_INT16_T:
    //         reduce_op = yhccl_max_op<int16_t>;
    //         break;
    //     case MPI_UINT32_T:
    //         reduce_op = yhccl_max_op<uint32_t>;
    //         break;
    //     case MPI_INT32_T:
    //         reduce_op = yhccl_max_op<int32_t>;
    //         break;
    //     case MPI_UINT64_T:
    //         reduce_op = yhccl_max_op<uint64_t>;
    //         break;
    //     case MPI_INT64_T:
    //         reduce_op = yhccl_max_op<int64_t>;
    //         break;
    //     case MPI_FLOAT:
    //         reduce_op = yhccl_max_op<float>;
    //         break;
    //     case MPI_DOUBLE:
    //         reduce_op = yhccl_max_op<double>;
    //     case MPI_C_BOOL:
    //         reduce_op = yhccl_max_op<bool>;
    //     default:
    //         break;
    //     };
    //     break;
    // case mpi_min_int:
    //     if (reducefp != 0)
    //         return reducefp;
    //     switch (mpitype)
    //     {
    //     case MPI_UINT8_T:
    //         reduce_op = yhccl_min_op<uint8_t>;
    //         break;
    //     case MPI_INT8_T:
    //         reduce_op = yhccl_min_op<int8_t>;
    //         break;
    //     case MPI_UINT16_T:
    //         reduce_op = yhccl_min_op<uint16_t>;
    //         break;
    //     case MPI_INT16_T:
    //         reduce_op = yhccl_min_op<int16_t>;
    //         break;
    //     case MPI_UINT32_T:
    //         reduce_op = yhccl_min_op<uint32_t>;
    //         break;
    //     case MPI_INT32_T:
    //         reduce_op = yhccl_min_op<int32_t>;
    //         break;
    //     case MPI_UINT64_T:
    //         reduce_op = yhccl_min_op<uint64_t>;
    //         break;
    //     case MPI_INT64_T:
    //         reduce_op = yhccl_min_op<int64_t>;
    //         break;
    //     case MPI_FLOAT:
    //         reduce_op = yhccl_min_op<float>;
    //         break;
    //     case MPI_DOUBLE:
    //         reduce_op = yhccl_min_op<double>;
    //     case MPI_C_BOOL:
    //         reduce_op = yhccl_min_op<bool>;
    //     default:
    //         break;
    //     };
    //     break;
    // default:
    //     if (reducefp != 0)
    //     {
    //         //默认类型自定义op
    //         reduce_op = reducefp;
    //     }
    //     else
    //     {
    //         std::cout << "目前支持的allreduce操作只有自定义操作，或者sum，min，max。有其他需求请联系1272813056@qq.com 彭晋韬" << std::endl;
    //     }
    //     break;
    // };
    // if (reduce_op == 0)
    // {
    //     std::cout << "检查到不支持的操作类型" << std::endl;
    //     fflush(stdout);
    //     exit(1);
    // }
    return reduce_op;
}
