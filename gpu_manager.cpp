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

void gpu_manager::add_user(int device_id, int matRowSize, int matColSize, int vlength){
    gpu_ops *gpu_user = new gpu_ops(matRowSize, matColSize, vlength);
    gpu_user->set_device(0);
    list_of_users.push_back(gpu_user);
}

void gpu_manager::copy_data(double* host_Mat, double* host_Vect) {
    list_of_users.at(0)->allocate_memory();
    list_of_users.at(0)->copy_to_device(host_Mat, host_Vect);
}

float gpu_manager::compute_and_store(double* host_ResVect) {
    list_of_users.at(0)->start_event();
    list_of_users.at(0)->launch_kernel();
    list_of_users.at(0)->copy_to_host(host_ResVect);
    float time_sec = list_of_users.at(0)->stop_event();
    return time_sec;
}

void gpu_manager::_free() {
    list_of_users.at(0)->_free();
}