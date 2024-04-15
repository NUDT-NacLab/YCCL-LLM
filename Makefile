# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yizimi/YCCL-LLM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yizimi/YCCL-LLM

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/yizimi/YCCL-LLM/CMakeFiles /home/yizimi/YCCL-LLM//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/yizimi/YCCL-LLM/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named yccl_llm

# Build rule for target.
yccl_llm: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 yccl_llm
.PHONY : yccl_llm

# fast build rule for target.
yccl_llm/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/build
.PHONY : yccl_llm/fast

#=============================================================================
# Target rules for targets named yccl_llm_static

# Build rule for target.
yccl_llm_static: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 yccl_llm_static
.PHONY : yccl_llm_static

# fast build rule for target.
yccl_llm_static/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/build
.PHONY : yccl_llm_static/fast

BIO_Bcast/BIO.o: BIO_Bcast/BIO.cc.o
.PHONY : BIO_Bcast/BIO.o

# target to build an object file
BIO_Bcast/BIO.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/BIO_Bcast/BIO.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/BIO_Bcast/BIO.cc.o
.PHONY : BIO_Bcast/BIO.cc.o

BIO_Bcast/BIO.i: BIO_Bcast/BIO.cc.i
.PHONY : BIO_Bcast/BIO.i

# target to preprocess a source file
BIO_Bcast/BIO.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/BIO_Bcast/BIO.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/BIO_Bcast/BIO.cc.i
.PHONY : BIO_Bcast/BIO.cc.i

BIO_Bcast/BIO.s: BIO_Bcast/BIO.cc.s
.PHONY : BIO_Bcast/BIO.s

# target to generate assembly for a file
BIO_Bcast/BIO.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/BIO_Bcast/BIO.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/BIO_Bcast/BIO.cc.s
.PHONY : BIO_Bcast/BIO.cc.s

PJT_allreduce_algorithms/PJT_X86_AVX_operations.o: PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.o
.PHONY : PJT_allreduce_algorithms/PJT_X86_AVX_operations.o

# target to build an object file
PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.o
.PHONY : PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.o

PJT_allreduce_algorithms/PJT_X86_AVX_operations.i: PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.i
.PHONY : PJT_allreduce_algorithms/PJT_X86_AVX_operations.i

# target to preprocess a source file
PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.i
.PHONY : PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.i

PJT_allreduce_algorithms/PJT_X86_AVX_operations.s: PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.s
.PHONY : PJT_allreduce_algorithms/PJT_X86_AVX_operations.s

# target to generate assembly for a file
PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.s
.PHONY : PJT_allreduce_algorithms/PJT_X86_AVX_operations.c.s

PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.o: PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.o
.PHONY : PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.o

# target to build an object file
PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.o
.PHONY : PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.o

PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.i: PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.i
.PHONY : PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.i

# target to preprocess a source file
PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.i
.PHONY : PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.i

PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.s: PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.s
.PHONY : PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.s

# target to generate assembly for a file
PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.s
.PHONY : PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc.s

PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.o: PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.o
.PHONY : PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.o

# target to build an object file
PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.o
.PHONY : PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.o

PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.i: PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.i
.PHONY : PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.i

# target to preprocess a source file
PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.i
.PHONY : PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.i

PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.s: PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.s
.PHONY : PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.s

# target to generate assembly for a file
PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.s
.PHONY : PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.cc.s

src/Rdma_contexts.o: src/Rdma_contexts.cc.o
.PHONY : src/Rdma_contexts.o

# target to build an object file
src/Rdma_contexts.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/Rdma_contexts.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/Rdma_contexts.cc.o
.PHONY : src/Rdma_contexts.cc.o

src/Rdma_contexts.i: src/Rdma_contexts.cc.i
.PHONY : src/Rdma_contexts.i

# target to preprocess a source file
src/Rdma_contexts.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/Rdma_contexts.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/Rdma_contexts.cc.i
.PHONY : src/Rdma_contexts.cc.i

src/Rdma_contexts.s: src/Rdma_contexts.cc.s
.PHONY : src/Rdma_contexts.s

# target to generate assembly for a file
src/Rdma_contexts.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/Rdma_contexts.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/Rdma_contexts.cc.s
.PHONY : src/Rdma_contexts.cc.s

src/mpi_allreduce.o: src/mpi_allreduce.cc.o
.PHONY : src/mpi_allreduce.o

# target to build an object file
src/mpi_allreduce.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/mpi_allreduce.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/mpi_allreduce.cc.o
.PHONY : src/mpi_allreduce.cc.o

src/mpi_allreduce.i: src/mpi_allreduce.cc.i
.PHONY : src/mpi_allreduce.i

# target to preprocess a source file
src/mpi_allreduce.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/mpi_allreduce.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/mpi_allreduce.cc.i
.PHONY : src/mpi_allreduce.cc.i

src/mpi_allreduce.s: src/mpi_allreduce.cc.s
.PHONY : src/mpi_allreduce.s

# target to generate assembly for a file
src/mpi_allreduce.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/mpi_allreduce.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/mpi_allreduce.cc.s
.PHONY : src/mpi_allreduce.cc.s

src/yhccl_allgather.o: src/yhccl_allgather.cc.o
.PHONY : src/yhccl_allgather.o

# target to build an object file
src/yhccl_allgather.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_allgather.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_allgather.cc.o
.PHONY : src/yhccl_allgather.cc.o

src/yhccl_allgather.i: src/yhccl_allgather.cc.i
.PHONY : src/yhccl_allgather.i

# target to preprocess a source file
src/yhccl_allgather.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_allgather.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_allgather.cc.i
.PHONY : src/yhccl_allgather.cc.i

src/yhccl_allgather.s: src/yhccl_allgather.cc.s
.PHONY : src/yhccl_allgather.s

# target to generate assembly for a file
src/yhccl_allgather.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_allgather.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_allgather.cc.s
.PHONY : src/yhccl_allgather.cc.s

src/yhccl_allreduce.o: src/yhccl_allreduce.cc.o
.PHONY : src/yhccl_allreduce.o

# target to build an object file
src/yhccl_allreduce.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_allreduce.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_allreduce.cc.o
.PHONY : src/yhccl_allreduce.cc.o

src/yhccl_allreduce.i: src/yhccl_allreduce.cc.i
.PHONY : src/yhccl_allreduce.i

# target to preprocess a source file
src/yhccl_allreduce.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_allreduce.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_allreduce.cc.i
.PHONY : src/yhccl_allreduce.cc.i

src/yhccl_allreduce.s: src/yhccl_allreduce.cc.s
.PHONY : src/yhccl_allreduce.s

# target to generate assembly for a file
src/yhccl_allreduce.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_allreduce.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_allreduce.cc.s
.PHONY : src/yhccl_allreduce.cc.s

src/yhccl_barrier.o: src/yhccl_barrier.cc.o
.PHONY : src/yhccl_barrier.o

# target to build an object file
src/yhccl_barrier.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_barrier.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_barrier.cc.o
.PHONY : src/yhccl_barrier.cc.o

src/yhccl_barrier.i: src/yhccl_barrier.cc.i
.PHONY : src/yhccl_barrier.i

# target to preprocess a source file
src/yhccl_barrier.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_barrier.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_barrier.cc.i
.PHONY : src/yhccl_barrier.cc.i

src/yhccl_barrier.s: src/yhccl_barrier.cc.s
.PHONY : src/yhccl_barrier.s

# target to generate assembly for a file
src/yhccl_barrier.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_barrier.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_barrier.cc.s
.PHONY : src/yhccl_barrier.cc.s

src/yhccl_bcast.o: src/yhccl_bcast.cc.o
.PHONY : src/yhccl_bcast.o

# target to build an object file
src/yhccl_bcast.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_bcast.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_bcast.cc.o
.PHONY : src/yhccl_bcast.cc.o

src/yhccl_bcast.i: src/yhccl_bcast.cc.i
.PHONY : src/yhccl_bcast.i

# target to preprocess a source file
src/yhccl_bcast.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_bcast.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_bcast.cc.i
.PHONY : src/yhccl_bcast.cc.i

src/yhccl_bcast.s: src/yhccl_bcast.cc.s
.PHONY : src/yhccl_bcast.s

# target to generate assembly for a file
src/yhccl_bcast.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_bcast.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_bcast.cc.s
.PHONY : src/yhccl_bcast.cc.s

src/yhccl_communicator.o: src/yhccl_communicator.cc.o
.PHONY : src/yhccl_communicator.o

# target to build an object file
src/yhccl_communicator.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_communicator.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_communicator.cc.o
.PHONY : src/yhccl_communicator.cc.o

src/yhccl_communicator.i: src/yhccl_communicator.cc.i
.PHONY : src/yhccl_communicator.i

# target to preprocess a source file
src/yhccl_communicator.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_communicator.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_communicator.cc.i
.PHONY : src/yhccl_communicator.cc.i

src/yhccl_communicator.s: src/yhccl_communicator.cc.s
.PHONY : src/yhccl_communicator.s

# target to generate assembly for a file
src/yhccl_communicator.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_communicator.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_communicator.cc.s
.PHONY : src/yhccl_communicator.cc.s

src/yhccl_contexts.o: src/yhccl_contexts.cc.o
.PHONY : src/yhccl_contexts.o

# target to build an object file
src/yhccl_contexts.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_contexts.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_contexts.cc.o
.PHONY : src/yhccl_contexts.cc.o

src/yhccl_contexts.i: src/yhccl_contexts.cc.i
.PHONY : src/yhccl_contexts.i

# target to preprocess a source file
src/yhccl_contexts.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_contexts.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_contexts.cc.i
.PHONY : src/yhccl_contexts.cc.i

src/yhccl_contexts.s: src/yhccl_contexts.cc.s
.PHONY : src/yhccl_contexts.s

# target to generate assembly for a file
src/yhccl_contexts.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_contexts.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_contexts.cc.s
.PHONY : src/yhccl_contexts.cc.s

src/yhccl_reduce.o: src/yhccl_reduce.cc.o
.PHONY : src/yhccl_reduce.o

# target to build an object file
src/yhccl_reduce.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_reduce.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_reduce.cc.o
.PHONY : src/yhccl_reduce.cc.o

src/yhccl_reduce.i: src/yhccl_reduce.cc.i
.PHONY : src/yhccl_reduce.i

# target to preprocess a source file
src/yhccl_reduce.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_reduce.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_reduce.cc.i
.PHONY : src/yhccl_reduce.cc.i

src/yhccl_reduce.s: src/yhccl_reduce.cc.s
.PHONY : src/yhccl_reduce.s

# target to generate assembly for a file
src/yhccl_reduce.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_reduce.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_reduce.cc.s
.PHONY : src/yhccl_reduce.cc.s

src/yhccl_reduce_scatter.o: src/yhccl_reduce_scatter.cc.o
.PHONY : src/yhccl_reduce_scatter.o

# target to build an object file
src/yhccl_reduce_scatter.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_reduce_scatter.cc.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_reduce_scatter.cc.o
.PHONY : src/yhccl_reduce_scatter.cc.o

src/yhccl_reduce_scatter.i: src/yhccl_reduce_scatter.cc.i
.PHONY : src/yhccl_reduce_scatter.i

# target to preprocess a source file
src/yhccl_reduce_scatter.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_reduce_scatter.cc.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_reduce_scatter.cc.i
.PHONY : src/yhccl_reduce_scatter.cc.i

src/yhccl_reduce_scatter.s: src/yhccl_reduce_scatter.cc.s
.PHONY : src/yhccl_reduce_scatter.s

# target to generate assembly for a file
src/yhccl_reduce_scatter.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm.dir/build.make CMakeFiles/yccl_llm.dir/src/yhccl_reduce_scatter.cc.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/yccl_llm_static.dir/build.make CMakeFiles/yccl_llm_static.dir/src/yhccl_reduce_scatter.cc.s
.PHONY : src/yhccl_reduce_scatter.cc.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... yccl_llm"
	@echo "... yccl_llm_static"
	@echo "... BIO_Bcast/BIO.o"
	@echo "... BIO_Bcast/BIO.i"
	@echo "... BIO_Bcast/BIO.s"
	@echo "... PJT_allreduce_algorithms/PJT_X86_AVX_operations.o"
	@echo "... PJT_allreduce_algorithms/PJT_X86_AVX_operations.i"
	@echo "... PJT_allreduce_algorithms/PJT_X86_AVX_operations.s"
	@echo "... PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.o"
	@echo "... PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.i"
	@echo "... PJT_allreduce_algorithms/pjt_hierarchy_reduce_scatter.s"
	@echo "... PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.o"
	@echo "... PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.i"
	@echo "... PJT_allreduce_algorithms/pjt_memory_bandidth_efficient.s"
	@echo "... src/Rdma_contexts.o"
	@echo "... src/Rdma_contexts.i"
	@echo "... src/Rdma_contexts.s"
	@echo "... src/mpi_allreduce.o"
	@echo "... src/mpi_allreduce.i"
	@echo "... src/mpi_allreduce.s"
	@echo "... src/yhccl_allgather.o"
	@echo "... src/yhccl_allgather.i"
	@echo "... src/yhccl_allgather.s"
	@echo "... src/yhccl_allreduce.o"
	@echo "... src/yhccl_allreduce.i"
	@echo "... src/yhccl_allreduce.s"
	@echo "... src/yhccl_barrier.o"
	@echo "... src/yhccl_barrier.i"
	@echo "... src/yhccl_barrier.s"
	@echo "... src/yhccl_bcast.o"
	@echo "... src/yhccl_bcast.i"
	@echo "... src/yhccl_bcast.s"
	@echo "... src/yhccl_communicator.o"
	@echo "... src/yhccl_communicator.i"
	@echo "... src/yhccl_communicator.s"
	@echo "... src/yhccl_contexts.o"
	@echo "... src/yhccl_contexts.i"
	@echo "... src/yhccl_contexts.s"
	@echo "... src/yhccl_reduce.o"
	@echo "... src/yhccl_reduce.i"
	@echo "... src/yhccl_reduce.s"
	@echo "... src/yhccl_reduce_scatter.o"
	@echo "... src/yhccl_reduce_scatter.i"
	@echo "... src/yhccl_reduce_scatter.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
