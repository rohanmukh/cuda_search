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

gpu_manager::gpu_manager(){
    int count = get_device_count();
    for(int i=0; i<count; i++)
        list_of_users.push_back(NULL);
}



void gpu_manager::add_user(int device_id, int matRowSize, int matColSize, int vlength){
    gpu_ops *gpu_user = new gpu_ops(matRowSize, matColSize, vlength);
    list_of_users.at(device_id) = gpu_user;
}

void gpu_manager::copy_data(double* host_Mat, double* host_Vect) {
    for(int i=0; i<list_of_users.size(); i++){
        list_of_users.at(i)->set_device(i,"Data Copy");
        list_of_users.at(i)->allocate_memory();
        list_of_users.at(i)->copy_to_device(host_Mat, host_Vect);
    }
}

float gpu_manager::compute_and_store(double* host_ResVect) {
    float time_sec = 0.;
    for(int i=0; i<list_of_users.size(); i++) {
        list_of_users.at(i)->set_device(i, "Compute and Store");
        list_of_users.at(i)->start_event();
        list_of_users.at(i)->launch_kernel();
        list_of_users.at(i)->copy_to_host(host_ResVect);
        time_sec = std::max(time_sec, list_of_users.at(i)->stop_event());
    }
    return time_sec;
}

void gpu_manager::_free() {
    for(int i=0; i<list_of_users.size(); i++) {
        list_of_users.at(i)->set_device(i, "Free Memory");
        list_of_users.at(i)->_free();
    }
}
