//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_GPU_MANAGER_H
#define CUDA_CODE_SEARCH_GPU_MANAGER_H

#include "gpu_ops.h"

class gpu_manager {
    gpu_ops *gpu_user;

public:
    gpu_manager(int device_id, int matRowSize, int matColSize, int vlength){
        gpu_user = new gpu_ops(matRowSize, matColSize, vlength);
        gpu_user->set_device(0);
    }

    static int get_device_count();
    void copy_data(double*, double*);
    float compute_and_store(double*);
    void _free();
};


#endif //CUDA_CODE_SEARCH_GPU_MANAGER_H
