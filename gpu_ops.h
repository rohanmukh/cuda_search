//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_GPU_OPS_H
#define CUDA_CODE_SEARCH_GPU_OPS_H

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string>
#include "cuda_utils.h"
#include "utils.h"

class gpu_ops {
private:

    // size of the data
    int matRowSize, matColSize, vlength;

    // data
    double *device_Mat,*device_Vect, *device_ResVect;

    // device
    cudaDeviceProp deviceProp;

    // timers
    cudaEvent_t start, stop; float elapsedTime;


public:
    gpu_ops(int, int, int);

    //data
    void allocate_memory();
    void copy_to_device(double*, double*);
    void copy_to_host(double*);
    void _free();

    // device
    void launch_kernel();
    void set_device(int, std::string);

    //timers
    void start_event();
    float stop_event();
};


#endif //CUDA_CODE_SEARCH_GPU_OPS_H
