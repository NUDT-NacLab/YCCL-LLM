#!/bin/bash

source /public21/home/sc94715/yizimi/start.sh

yizimi_mpi
# intelmpi
yizimi_openmpi_osu_benchmark


# openmpi test 

# module unload mpi
# module load mpi/4.1.1-para
# export OMP_NUM_THREADS=1 
export LD_PRELOAD=

echo "============ hello ============" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/openmpitest.yizimi
mpiexec -n 64 osu_hello -m 32:268435456 -i 10  >> /public21/home/sc94715/yizimi/YCCL_LLM/test/openmpitest.yizimi

echo "" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/openmpitest.yizimi
echo "" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/openmpitest.yizimi

echo "============ allreduce ============" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/openmpitest.yizimi
mpiexec -n 64 osu_allreduce -m 32:268435456 -i 10  >> /public21/home/sc94715/yizimi/YCCL_LLM/test/openmpitest.yizimi

echo "" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/openmpitest.yizimi
echo "" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/openmpitest.yizimi

echo "============ bcast ============" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/openmpitest.yizimi
mpiexec -n 64 osu_bcast -m 32:268435456 -i 10  >> /public21/home/sc94715/yizimi/YCCL_LLM/test/openmpitest.yizimi


# YCCL_LLM test 

export LD_PRELOAD=/public21/home/sc94715/yizimi/YCCL_LLM/build/lib/libyccl_llm.so

echo "============ hello ============" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/yccl_llmtest.yizimi
mpiexec -n 64 osu_hello -m 32:268435456 -i 10  >> /public21/home/sc94715/yizimi/YCCL_LLM/test/yccl_llmtest.yizimi

echo "" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/yccl_llmtest.yizimi
echo "" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/yccl_llmtest.yizimi

echo "============ allreduce ============" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/yccl_llmtest.yizimi
mpiexec -n 64 osu_allreduce -m 32:268435456 -i 10 >> /public21/home/sc94715/yizimi/YCCL_LLM/test/yccl_llmtest.yizimi

echo "" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/yccl_llmtest.yizimi
echo "" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/yccl_llmtest.yizimi

echo "============ bcast ============" >> /public21/home/sc94715/yizimi/YCCL_LLM/test/yccl_llmtest.yizimi
mpiexec -n 64 osu_bcast -m 32:268435456 -i 10  >> /public21/home/sc94715/yizimi/YCCL_LLM/test/yccl_llmtest.yizimi
