//
// Created by rm38 on 3/2/20.
//

#include "gpu_manager.h"
#include <cuda_runtime_api.h>

/*Get the number of GPU devices present on the host */
gpu_manager::gpu_manager(int batch_size, int dimension){
    this->batch_size = batch_size;
    this->dimension = dimension;
    cudaGetDeviceCount(&this->num_devices);

    batch_matRowSize = batch_size / this->num_devices;
    // ASSERT TODO

    for(int i=0; i<this->num_devices; i++)
        this->list_of_users.push_back(new gpu_ops(i, batch_matRowSize, dimension));
}


void gpu_manager::add_user(int device_id){
    list_of_users.at(device_id) = new gpu_ops(device_id, batch_size, dimension);
}

void gpu_manager::copy_data(double* host_Mat, double* host_Vect) {
    int offset = 0; 
    for(int i=0; i<list_of_users.size(); i++){
        list_of_users.at(i)->set_device(i,"Data Copy");
        list_of_users.at(i)->allocate_memory();
        offset += i*batch_matRowSize;
        list_of_users.at(i)->copy_to_device(host_Mat + offset, host_Vect);
    }
}

float gpu_manager::compute_and_store(double* host_ResVect) {
    float time_sec = 0.;
    int offset = 0;
    for(int i=0; i<list_of_users.size(); i++) {
        list_of_users.at(i)->set_device(i, "Compute and Store");
        list_of_users.at(i)->start_event();
        list_of_users.at(i)->launch_kernel();
        offset += i*batch_matRowSize;
        list_of_users.at(i)->copy_to_host(host_ResVect + offset);
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
