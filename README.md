# YCCL_LLM

YCCL_LLM是一款基于PMPI层实现的、面向高性能计算通信与大模型通信的MPI通信库

## 安装说明

- 1. 安装 OpenMPI 4.1.1 或以上版本
- 2. 安装 gcc 11.2.0 或以上版本（YCCL中的allreduce采用FMCC-RT算法，使用C++20 所支持的协程实现）
- 3. 下载 YCCL_LLM 库
- 4. cd YCCL_LLM
- 5. cmake .
- 6. make -j 8

## 使用说明

- 1. export LD_PROLOAD=/path/to/YCCL_LLM/build/lib/libyccl_llm.so
- 2. 使用mpirun/mpiexec等程序运行即可

 
