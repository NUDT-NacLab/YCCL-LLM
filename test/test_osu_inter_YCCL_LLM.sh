#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64

source /public21/home/sc94715/yizimi/start.sh

yizimi_mpi
# intelmpi
yizimi_openmpi_osu_benchmark

# mpicxx ./allreduce_bcast.cc -o ./allreduce_bcast
export LD_PRELOAD=/public21/home/sc94715/yizimi/YCCL_LLM/build/lib/libyccl_llm.so 
mpiexec -N 2 -npernode 64 osu_benchmark