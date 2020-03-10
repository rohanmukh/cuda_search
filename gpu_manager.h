//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_GPU_MANAGER_H
#define CUDA_CODE_SEARCH_GPU_MANAGER_H

#include <vector>
#include <string>
#include <tuple>
#include <cassert>
#include <chrono>
#include <iostream>
#include <algorithm>    // std::partial_sort
#include "single_gpu_manager.h"
#include "cuda_utils.h"
#include "utils.h"


class gpu_manager {
    long device_num_batches, batch_size;
    int dimension;
    int num_devices;
    std::vector<single_gpu_manager*> list_of_users;
    float* result_vector;

public:
    gpu_manager(int, int, long, int);
    std::vector<std::tuple<int, int, float>> top_k(int k=10);
    long compute_device_num_batch_offset(int);
    void copy_database_to_device(float**, float**, float**);
    void add_query(float* , float* );
    float *get_result();
    void search();
    void _free();
};


#endif //CUDA_CODE_SEARCH_GPU_MANAGER_H
