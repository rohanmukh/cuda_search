//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_GPU_MANAGER_H
#define CUDA_CODE_SEARCH_GPU_MANAGER_H

#include <vector>
#include <string>
#include "gpu_ops.h"

class gpu_manager {
    long batch_size;
    int dimension;
    int num_devices;
    std::vector<gpu_ops*> list_of_users;
    double* result_vector;

public:
    gpu_manager(long, int);
//    void add_user(int);
    long compute_batch_offset(int);
    void copy_database_to_device(double*, double*, double*);
    void add_query(double* , double* );
    double *get_result();
    void search();
    void _free();
};


#endif //CUDA_CODE_SEARCH_GPU_MANAGER_H
