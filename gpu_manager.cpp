//
// Created by rm38 on 3/2/20.
//

#include "gpu_manager.h"
#include "cuda_utils.h"
#include <cassert>
#include <iostream>

/*Get the number of GPU devices present on the host */
gpu_manager::gpu_manager(int batch_size, int dimension){
    this->batch_size = batch_size;
    this->dimension = dimension;

    // get number of devices

    this->num_devices = get_DeviceCount();

    assert(batch_size % this->num_devices == 0);
    this->batch_matRowSize = batch_size / this->num_devices;

    for(int i=0; i<this->num_devices; i++)
        this->list_of_users.push_back(new gpu_ops(i, batch_matRowSize, dimension));
}


void gpu_manager::add_user(int device_id){
    list_of_users.at(device_id) = new gpu_ops(device_id, batch_size, dimension);
}

void gpu_manager::copy_data(double* host_Mat, double* host_Vect) {
    std::cout << "Copying data for all GPUs" << std::endl;
    for(int i=0; i<list_of_users.size(); i++){
        list_of_users.at(i)->set_device(i,"Data Copy");
        list_of_users.at(i)->allocate_memory();
        int offset = compute_batch_offset(i);
        list_of_users.at(i)->copy_to_device(host_Mat + offset, host_Vect);
    }
}

float gpu_manager::compute_and_store(double* host_ResVect) {
    float time_sec = 0.;
    std::cout << "Computing data for all GPUs" << std::endl;
    for(int i=0; i<list_of_users.size(); i++) {
        list_of_users.at(i)->set_device(i, "Compute and Store");
        list_of_users.at(i)->start_event();
        list_of_users.at(i)->launch_kernel();
        int offset = compute_batch_offset(i);
        list_of_users.at(i)->copy_to_host(host_ResVect + offset);
        time_sec = std::max(time_sec, list_of_users.at(i)->stop_event());
    }
    return time_sec;
}

int gpu_manager::compute_batch_offset(int batch_id){
    return batch_id * batch_matRowSize;
}

void gpu_manager::_free() {
    for(int i=0; i<list_of_users.size(); i++) {
        list_of_users.at(i)->set_device(i, "Free Memory");
        list_of_users.at(i)->_free();
    }
}
