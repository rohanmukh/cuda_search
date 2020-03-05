//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_SINGLE_GPU_MANAGER_H
#define CUDA_CODE_SEARCH_SINGLE_GPU_MANAGER_H

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string>
#include "cuda_utils.h"
#include "utils.h"

class single_gpu_manager {
private:

    // size of the data
    long device_num_batches, batch_size, device_data_size;
    int dimension;

    // data
    double *device_database_B, *device_database_A, *device_database_probY;
    double *device_input_B, *device_input_A;
    double *device_result_vector;

    // device
    int device_id;
    cudaDeviceProp deviceProp;

    // timers
    cudaEvent_t start, stop; float elapsedTime;


public:
    single_gpu_manager(int, long, long, int);

    //data
    void allocate_memory();
    void copy_data_to_device(long, double*, double*, double*);
    void copy_input_to_device(double*, double*);
    void copy_result_to_host(double*);
    void _free();

    // device
    void launch_kernel();
    void set_device(int, const std::string&);
    void get_device_property();

    //timers
    void start_event();
    double stop_event();
};


#endif //CUDA_CODE_SEARCH_SINGLE_GPU_MANAGER_H
