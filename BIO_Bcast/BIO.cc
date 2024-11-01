#include "BIO.h"

using std::cout;
using std::endl;
using std::map;
using std::vector;

using std::string;
using std::to_string;

int bio_shm_fd = 0;                       //共享内存的文件描述符
volatile char* bio_fin_f = nullptr;       //标志位首地址,存放 intra_size 个 char
int bio_buff_size = (1L << 31) - 1;       //共享内存的大小

vector<int> leader_rank_coll;  //节点间leader的rank集合
int intra_rank = 0;            //节点内的rank
int intra_size = 0;            //节点内的size

volatile char* shm_head_addr = NULL;  //共享内存的首地址
volatile char* shm_data_addr = NULL;  //存放用户数据的首地址

int init_f = 0;                 //判断是否是第一次进入,第一次进入需要进行耗时初始化
vector<int> children;           //本进程的所有子进程rank集合
int parent = -1;                //本进程的父进程rank
int degree = -1, size, m_rank;  //叉度、全局size,rank
MPI_Status status_t;

MPI_Aint type_size;

const int cache_line_size = 64;

int get_parent(int intra_process_size, int degree, int rank) {
    if (rank < intra_process_size || rank % intra_process_size != 0)
        return -1;

    int q = rank;
    rank /= intra_process_size;
    int t = 0, previous_layer_end = 0, m = 1;

    while (t < rank) {
        previous_layer_end = t;
        t += (degree * m);
        m *= (degree + 1);
    }

    int n = rank;
    while ((n - 1) % degree != 0) {
        --n;
    }

    if (rank * intra_process_size == q)
        return (rank - previous_layer_end - 1) / degree * intra_process_size + (rank - n);
    else
        return -1;
}

vector<int> get_children(int intra_process_size, int degree, int rank, int size) {
    if (rank % intra_process_size >= degree)
        return {};

    int t = 0, m = 1, q = rank, p = size;
    rank /= intra_process_size;
    size /= intra_process_size;
    // ++size;
    while (t < rank) {
        t += (degree * m);  //本rank所在层的最后一个rank
        m *= (degree + 1);
    }

    vector<int> v;
    while (t < size) {
        int g = t + rank * degree + 1;
        for (int i = 0; i < degree; ++i) {
            if (g < size) {
                v.push_back(g++);
            } else {
                break;
            }
        }
        t += (degree * m);  //下一层
        m *= (degree + 1);
    }

    q = q - rank * intra_process_size;
    vector<int> ans;
    for (int i = q; i < v.size(); i += degree) {
        int t = v[i] * intra_process_size;
        if (t >= p)
            break;
        ans.push_back(t);
    }

    return ans;
}

void shared_mem() {
    //根据IP获取节点间 节点内的情况
    char my_ip[16] = {0};
    char str_host_name[100];
    gethostname(str_host_name, 100);

    inet_ntop(AF_INET, gethostbyname(str_host_name)->h_addr_list[0], my_ip, 16);
    char* all_ip = (char*)calloc(size, 16);
    PMPI_Allgather(my_ip, 16, MPI_CHAR, all_ip, 16, MPI_CHAR, MPI_COMM_WORLD);

    intra_rank = 0;
    intra_size = 0;

    char* tp = all_ip;
    for (int i = 0; i < size; ++i, tp += 16) {
        if (strcmp(tp, my_ip) == 0) {
            ++intra_size;
            if (i < m_rank) {
                ++intra_rank;
            }
        }
        if (i != 0) {
            if (strcmp(tp, tp - 16) == 0) {
                continue;
            }
        }
        leader_rank_coll.push_back(i);
    }

    //创建共享内存
    char host_name[64], name[64];
    int namelen;
    
    MPI_Get_processor_name(host_name, &namelen);
    sprintf(name, "rm /dev/shm/bio-test653-%s -rf", host_name);
    sprintf(name, "bio-test653-%s", host_name);
    // "bcast_ppppp_optim_adapt"
    bio_shm_fd = shm_open(name, O_CREAT | O_RDWR | O_TRUNC, 0777);
    assert(bio_shm_fd >= 0);

    if (ftruncate(bio_shm_fd, bio_buff_size) == -1) {
        close(bio_shm_fd);
        bio_shm_fd = -1;
        assert(0);
    }

    shm_head_addr = (char*)mmap(NULL, bio_buff_size, PROT_READ | PROT_WRITE, MAP_SHARED, bio_shm_fd, 0);
    if (shm_head_addr == MAP_FAILED) {
        close(bio_shm_fd);
        bio_shm_fd = -1;
        assert(0);
    }

    bio_fin_f = shm_head_addr;

    shm_data_addr = shm_head_addr + (intra_size * cache_line_size);
    free(all_ip);

    PMPI_Barrier(MPI_COMM_WORLD);
}

static volatile char cccc = 0;
void while_wait() __attribute__((optimize("O0")));
void while_wait() {
    while (bio_fin_f[0] == cccc)
        ;

    if (cccc + 1 < 0)
        cccc = 0;
    else
        ++cccc;
}

void set_fin_my_to1() __attribute__((optimize("O0")));
void set_fin_my_to1() {
    if (intra_rank)
        bio_fin_f[intra_rank * cache_line_size] = 1;
    else {
        if (cccc + 1 < 0)
            cccc = 0;
        else
            ++cccc;

        bio_fin_f[0] = cccc;
    }
}

int is_all_1() __attribute__((optimize("O0")));
int is_all_1() {
    for (int i = 1; i < intra_size; i++) {
        if (bio_fin_f[i * cache_line_size] == 0)
            return 0;
    }
    return 1;
}

void set_all_0() __attribute__((optimize("O0")));
void set_all_0() {
    // ((volatile int*)bio_fin_f)[0] = ++ccc;
    for (int i = 1; i < intra_size; i++) {
        bio_fin_f[i * cache_line_size] = 0;
    }
}

struct statistical_data {
    int count = 0;
    int degree = 1;
    vector<int> v;
    double avg_time = 0;
};

int is_degree_fin = 0;
map<int, vector<statistical_data>> degree_map;
map<int, int> size_degree_map;

map<int, int> parent_map;
map<int, vector<int>> children_map;

int degree_max_test_count = 50;
int if_bio_bcast = 0;

void updata_children_and_parent() {
    if (parent_map.find(degree) == parent_map.end()) {
        parent = get_parent(intra_size, degree, m_rank);
        parent_map[degree] = parent;
    } else {
        parent = parent_map[degree];
    }

    if (children_map.find(degree) == children_map.end()) {
        children = get_children(intra_size, degree, m_rank, size);
        children_map[degree] = children;
    } else {
        children = children_map[degree];
    }
}

int is_need_count_time = 0;
double start_time = 0, end_time = 0;

int BIO_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    int c_size = -1;
    PMPI_Comm_size(comm, &c_size);
    
    if (if_bio_bcast == 0)
        if_bio_bcast = 1;

    if (!init_f) {
        PMPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
        PMPI_Comm_size(MPI_COMM_WORLD, &size);
        // if (root == m_rank) {
        //     puts(" =========== BIO Bcast ===========");
        // }
    }
    // if (m_rank == 0) {
    //     static int ccc = 0;
    //     cout << ++ccc << endl;
    // }

    // cout << "rank:" << m_rank << "  enter  " << ssss << endl;

    // int ladder_size = (int)(log(mpi_Datetype_size(datatype) * count) / log(2));
    // PMPI_Type_extent(datatype, &type_size);
    MPI_Aint lb;
    MPI_Type_get_extent(datatype, &lb, &type_size);
    // cout << "rank:" << rank << "  " << ssss << "  type_size:" << type_size << "  count:" << count << endl;
    int ladder_size = (int)(log(type_size * count) / log(2));
    int last_degree = degree;

    if (!init_f) {
        init_f = 1;
        shared_mem();
    }

    if (c_size != size || root != 0 || bio_buff_size < type_size * count + intra_size * cache_line_size) {
        return PMPI_Bcast(buffer, count, datatype, root, comm);
    }

    if (size_degree_map.find(ladder_size) == size_degree_map.end()) {
        is_need_count_time = 1;

        if (degree_map.find(ladder_size) == degree_map.end()) {
            statistical_data d;
            d.avg_time = 0;
            d.count = 0;
            d.degree = 1;

            degree_map[ladder_size].push_back(d);
        }

        //未完成最佳叉度寻找
        vector<statistical_data>& v = degree_map[ladder_size];

        statistical_data& data = v.back();

        if (data.count < degree_max_test_count) {
            //每个统计degree_max_test_count次，count从0开始计数
            degree = data.degree;
        } else {
            //此时最后一个叉度统计完毕
            if (degree == intra_size) {
                vector<statistical_data>& v = degree_map[ladder_size];
                double* temp = (double*)calloc(sizeof(double), v.size());
                double* recv = (double*)calloc(sizeof(double), v.size());
                for (int i = 0; i < v.size(); i++) {
                    temp[i] = v[i].avg_time;
                    // if (m_rank == 0) {
                    //         printf("%.3f ", temp[i]);
                    // }
                }
                // if (m_rank == 1)
                //     printf("\n");

                PMPI_Allreduce(temp, recv, v.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                double minv = recv[0];
                int min_pos = 0;
                for (int i = 1; i < v.size(); i++) {
                    if (recv[i] < minv) {
                        minv = recv[i];
                        min_pos = i;
                    }
                }

                degree = (int)pow(2, min_pos);
                if (degree > intra_size) {
                    degree = intra_size;
                }

                size_degree_map[ladder_size] = degree;

                is_need_count_time = 0;
                // if (m_rank == 0) {
                //     for (int i = 0; i < v.size(); i++) {
                //         printf("%.3f ",recv[i]);
                //     }
                //     cout << "  best pos is:" << min_pos << " degree:" << degree << endl;
                // }

                free(temp);
                free(recv);

            } else {
                degree *= 2;
                if (degree > intra_size) {
                    degree = intra_size;
                }

                statistical_data d;
                d.avg_time = 0;
                d.count = 0;
                d.degree = degree;

                degree_map[ladder_size].push_back(d);
            }
        }
    } else {
        // cout << "rank:" << rank << "  " << ssss << "  ladder_size:" << ladder_size << " 此类消息还yijing完成计时" << endl;
        // cout << "is_need_count_time = 0" << endl;
        is_need_count_time = 0;
        degree = size_degree_map[ladder_size];
    }

    // degree发生改变,更新parent和children
    if (last_degree != degree) {
        updata_children_and_parent();
    }

    if (m_rank == 0) {
        if (is_need_count_time == 1) {
            start_time = PMPI_Wtime();
        }

        if (intra_size == 1) {
            for (auto x : children) {
                PMPI_Send(buffer, count, datatype, x, 99, MPI_COMM_WORLD);
            }

        } else {
            memcpy((void*)shm_data_addr, buffer, count * type_size);
            set_fin_my_to1();

            for (auto x : children) {
                PMPI_Send((void*)shm_data_addr, count, datatype, x, 99, MPI_COMM_WORLD);
            }

            while (is_all_1() == 0)
                ;

            set_all_0();
        }

        if (is_need_count_time == 1) {
            end_time = PMPI_Wtime();

            vector<statistical_data>& v = degree_map[ladder_size];
            statistical_data& data = v.back();
            data.v.push_back((end_time - start_time) * 1e6);

            ++data.count;

            if (data.count == degree_max_test_count) {
                double t = 0;
                for (int i = 0; i < data.v.size(); i++) {
                    t += data.v[i];
                }

                data.avg_time = t / data.v.size();
            }
        }
    } else {
        if (is_need_count_time == 1) {
            start_time = PMPI_Wtime();
        }

        if (intra_size == 1) {
            // 1ppn
            PMPI_Recv(buffer, count, datatype, parent, 99, MPI_COMM_WORLD, &status_t);

            if (children.size() != 0) {
                for (auto x : children) {
                    PMPI_Send(buffer, count, datatype, x, 99, MPI_COMM_WORLD);
                }
            }
        } else {
            if (parent != -1) {
                PMPI_Recv((void*)shm_data_addr, count, datatype, parent, 99, MPI_COMM_WORLD, &status_t);
                set_fin_my_to1();
            }

            if (intra_rank)
                while_wait();

            if (children.size() != 0) {
                for (auto x : children) {
                    PMPI_Send((void*)shm_data_addr, count, datatype, x, 99, MPI_COMM_WORLD);
                }
            }

            memcpy(buffer, (void*)shm_data_addr, count * type_size);

            if (intra_rank)
                set_fin_my_to1();

            if (parent != -1) {
                while (is_all_1() == 0)
                    ;
                set_all_0();
            }
        }

        if (is_need_count_time == 1) {
            end_time = PMPI_Wtime();

            vector<statistical_data>& v = degree_map[ladder_size];
            statistical_data& data = v.back();
            data.v.push_back((end_time - start_time) * 1e6);

            ++data.count;

            if (data.count == degree_max_test_count) {
                double t = 0;
                for (int i = 0; i < data.v.size(); i++) {
                    t += data.v[i];
                }

                data.avg_time = t / data.v.size();
            }
        }
    }

    return 0;
}

// int MPI_Finalize(void) {
//     if (bio_fin_f)
//         munmap((void*)bio_fin_f, bio_buff_size);
//     if (bio_shm_fd)
//         close(bio_shm_fd);
//     return PMPI_Finalize();
// }