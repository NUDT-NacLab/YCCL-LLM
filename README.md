# YCCL_LLM

YCCL_LLM是一款基于PMPI层实现的、面向高性能计算通信与大模型通信的MPI通信库。



## 安装说明

```
1. 安装 MPI 发行版，保证可以兼容 gcc 11.2.0 版本
2. 安装 gcc 11.2.0 或以上版本（YCCL中的allreduce采用FMCC-RT算法实现使用C++20 所支持的协程实现）
3. 下载 YCCL_LLM 库
4. cd YCCL_LLM
5. cmake .
6. make -j 8
```

## 使用示范

以 intel_mpi 编译 osu_micro_benchmark 测试为例：

1. 配置好intel mpi，推荐使用intel mpi 2022.02；

2. 下载osu_micro_benchmark，进行源码编译，使用intel mpi编译：

   ```
   ./configure --prefix=/path/to/install/benchmark CC=/path/to/mpiicc CXX=/path/to/mpiicpc
   ```

3. 按照上一章方法编译 YCCL_LLM 库；

4. 编写sbatch：

   ```
   #!/bin/bash
   
   # 此处通过
   export LD_PROLOAD=/path/to/YCCL_LLM/build/lib/libyccl_llm.so
   
   mpirun -n 64 /path/to/install/benchmark/libexec/osu-micro-benchmarks/mpi/osu_allreduce -m 32:268435456 >> test.out
   
   mpirun -n 64 /path/to/install/benchmark/libexec/osu-micro-benchmarks/mpi/osu_bcast -m 32:268435456 >> test.out
   ```

   可以根据实际想要测试方案和集群环境改写sbatch；

5. 使用集群进行测试运行即可。

 ## 参考文献

