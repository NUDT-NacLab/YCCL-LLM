#!/bin/bash

LD_PRELOAD=/public1/home/scfa1117/YCCL-LLM/YCCL_LLM/build/lib/libyccl_llm.so mpiexec -n 32 ./test_YCCL-LLM
