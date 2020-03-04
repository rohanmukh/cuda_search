//
// Created by rm38 on 3/2/20.
//

#include "gpu_manager.h"
#include "cuda_utils.h"
#include "utils.h"
#include <cassert>
#include <iostream>

/*Get the number of GPU devices present on the host */
gpu_manager::gpu_manager(long data_size, int dimension){

    std::cout << "Initializing GPU Manager" << std::endl;
    this->dimension = dimension;

    // get number of devices
    this->num_devices = get_DeviceCount();

    assert(data_size % this->num_devices == 0);
    this->batch_size = data_size / this->num_devices;

    std::cout << "========================Initializing data for all GPUs======================================\n" << std::endl;
    for(int i=0; i<this->num_devices; i++)
        this->list_of_users.push_back(new gpu_ops(i, batch_size, dimension));

    result_vector = (double*)malloc(data_size * sizeof(double)); // new double[data_size];
    if(result_vector == nullptr)
        mem_error("result_vector", "vectmatmul", data_size, "double");

}


//void gpu_manager::add_user(int device_id){
//    list_of_users.at(device_id) = new gpu_ops(device_id, batch_size, dimension);
//}

void gpu_manager::copy_database_to_device(double* host_database_B, double* host_database_A, double* host_database_prob_Y) {
    std::cout << "================================Copying database to all GPUs======================================\n" << std::endl;
    for(int i=0; i<list_of_users.size(); i++){
        long offset = compute_batch_offset(i);
        list_of_users.at(i)->set_device(i,"Database Copy");
        list_of_users.at(i)->copy_data_to_device(host_database_B + offset * dimension, host_database_A+offset, host_database_prob_Y+offset);
    }
}


void gpu_manager::add_query(double* host_input_B, double* host_input_A) {
    std::cout << "=================================Copying query to all GPUs======================================\n" << std::endl;
    for(int i=0; i<list_of_users.size(); i++){
        list_of_users.at(i)->set_device(i,"Input Copy");
        list_of_users.at(i)->copy_input_to_device(host_input_B, host_input_A);
    }
}


void gpu_manager::search() {
    double time_sec = 0.;
    std::cout << "================================Computing data for all GPUs======================================\n" << std::endl;
    for(int i=0; i<list_of_users.size(); i++) {
        list_of_users.at(i)->set_device(i, "Compute and Store");
        list_of_users.at(i)->start_event();
        list_of_users.at(i)->launch_kernel();
        long offset = compute_batch_offset(i);
        list_of_users.at(i)->copy_result_to_host(result_vector + offset);
        time_sec = std::max(time_sec, list_of_users.at(i)->stop_event());
    }
    std::cout << "===========================================GPU Calculation========================================\n" << std::endl;
    double gflops = calculate_gflops(time_sec, batch_size * dimension);
    print_on_screen("GPU Search", time_sec, gflops, batch_size * dimension, 1);

}


long gpu_manager::compute_batch_offset(int batch_id){
    return batch_id * batch_size;
}

void gpu_manager::_free() {
    std::cout << "===================================Freeing data for all GPUs======================================\n" << std::endl;
    for(int i=0; i<list_of_users.size(); i++) {
        list_of_users.at(i)->set_device(i, "Free Memory");
        list_of_users.at(i)->_free();
    }
    std::cout << std::endl;
}

double *gpu_manager::get_result(){
    return result_vector;
}