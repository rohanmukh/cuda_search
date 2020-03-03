//
// Created by rm38 on 3/2/20.
//

#include "gpu_manager.h"
#include <cuda_runtime_api.h>

/*Get the number of GPU devices present on the host */
int gpu_manager::get_device_count() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

void gpu_manager::copy_data(double* host_Mat, double* host_Vect) {
    gpu_user->allocate_memory();
    gpu_user->copy_to_device(host_Mat, host_Vect);
}

float gpu_manager::compute_and_store(double* host_ResVect) {
    gpu_user->start_event();
    gpu_user->launch_kernel();
    gpu_user->copy_to_host(host_ResVect);
    float time_sec = gpu_user->stop_event();
    return time_sec;
}

void gpu_manager::_free() {
    gpu_user->_free();
}