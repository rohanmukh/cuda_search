//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_GPU_MANAGER_H
#define CUDA_CODE_SEARCH_GPU_MANAGER_H

#include <vector>
#include "gpu_ops.h"

class gpu_manager {
    std::vector<gpu_ops*> list_of_users;

public:
    gpu_manager(){
    }

    void add_user(int, int, int, int);
    static int get_device_count();
    void copy_data(double*, double*);
    float compute_and_store(double*);
    void _free();
};


#endif //CUDA_CODE_SEARCH_GPU_MANAGER_H
