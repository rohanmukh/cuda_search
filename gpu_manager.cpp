//
// Created by rm38 on 3/2/20.
//

#include "gpu_manager.h"
#include "cuda_utils.h"
#include "utils.h"
#include <cassert>
#include <iostream>

/*Get the number of GPU devices present on the host */
gpu_manager::gpu_manager(int num_batches, long batch_size, int dimension){

    std::cout << "Initializing GPU Manager" << std::endl;
    this->dimension = dimension;
    this->batch_size = batch_size;
    // get number of devices
    this->num_devices = get_DeviceCount();

    assert(num_batches % this->num_devices == 0);
    this->device_num_batches = num_batches / this->num_devices;

    std::cout << "========================Initializing data for all GPUs======================================\n" << std::endl;
    for(int i=0; i<this->num_devices; i++)
        this->list_of_users.push_back(new single_gpu_manager(i, device_num_batches, batch_size, dimension));

    result_vector = (double*)malloc(device_num_batches * batch_size * sizeof(double)); // new double[data_size];
    if(result_vector == nullptr)
        mem_error("result_vector", "vectmatmul", device_num_batches * batch_size, "double");

}


//void gpu_manager::add_user(int device_id){
//    list_of_users.at(device_id) = new single_gpu_manager(device_id, device_num_batches, dimension);
//}

void gpu_manager::copy_database_to_device(double** host_database_B, double** host_database_A, double** host_database_prob_Y) {
    std::cout << "================================Copying database to all GPUs======================================\n" << std::endl;
    for(int i=0; i<list_of_users.size(); i++){
        long offset = compute_device_num_batch_offset(i);
        list_of_users.at(i)->set_device(i,"Database Copy");
        for (int j=0; j<device_num_batches; j++){
            list_of_users.at(i)->copy_data_to_device(j*batch_size , host_database_B[offset+j], host_database_A[offset+j], host_database_prob_Y[offset+j]);
        }
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
        long offset = compute_device_num_batch_offset(i);
        list_of_users.at(i)->copy_result_to_host(result_vector + offset);
        time_sec = std::max(time_sec, list_of_users.at(i)->stop_event());
    }
    std::cout << "===========================================GPU Calculation========================================\n" << std::endl;
    double gflops = calculate_gflops(time_sec, device_num_batches * dimension);
    print_on_screen("GPU Search", time_sec, gflops, device_num_batches * dimension, 1);

}


long gpu_manager::compute_device_num_batch_offset(int device_id){
    return device_id * device_num_batches;
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
