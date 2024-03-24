#include <iostream>
#include <mpi.h>

#define ARRAY_SIZE 100000

// LD_PRELOAD="/public1/home/scfa1117/YCCL-LLM/yhccl_test/libbio.so:/public1/home/scfa1117/YCCL-LLM/yhccl_test/libyhccl.so"

int main(int argc, char **argv)
{
    int rank, procn;
    int i;
    double data[ARRAY_SIZE];
    double x[ARRAY_SIZE];
    double y[ARRAY_SIZE];
    double sum = 0;
    double global_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procn);

    // Initialize data array
    if (rank == 0) 
    for (i = 0; i < ARRAY_SIZE; i++)
    {
        data[i] = i;
    }
    
    for (i = 0; i < ARRAY_SIZE; ++i)
    {
        x[i] = i * 1.0;
    }

    // Perform multiple MPI_Bcast and MPI_Allreduce operations
    // for (i = 0; i < 10; i++)
    // {
    MPI_Bcast(data, ARRAY_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Allreduce(x, y, ARRAY_SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // global_sum += sum;
    // }

    // Check the correctness of the operations
    if (rank == 3)
    {
        // double expected_sum = 0;
        bool flag = 1;
        for (i = 0; i < ARRAY_SIZE; i++)
        {
            if (abs(y[i] - i * procn) > 0.000001) { flag = 0; printf("allreduce %d: %lf %lf\n", i, y[i], i * procn); break; }
            if (abs(data[i] - i) > 0.00001) { flag = 0; printf("bcast %d: %lf %lf\n", i, data[i], i); break; }
        }

        if (flag)
        {
            std::cout << "MPI_Allreduce operations are correct." << std::endl;
        }
        else
        {
            std::cout << "MPI_Allreduce operations are incorrect." << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}