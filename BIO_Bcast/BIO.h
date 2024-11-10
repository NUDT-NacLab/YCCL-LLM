#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <fcntl.h> /* For O_* constants */
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */

#include <arpa/inet.h>
#include <netdb.h>
#include <string.h>

#include <assert.h>

#include <unistd.h>

#include <math.h>
#include <iostream>
#include <map>

#include <chrono>
#include <string>

#include "mpi.h"



void updata_children_and_parent();
int BIO_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
