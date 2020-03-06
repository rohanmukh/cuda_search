//
// Created by rm38 on 3/2/20.
//

#include "single_gpu_manager.h"
#include <cstdio>
#include <iostream>
#include "matrix_vector_op.cu"

/*function to launch kernel*/
void single_gpu_manager::launch_kernel() {
    // Launching kernel..........

    /* threads_per_block, blocks_per_grid  */
    int max=BLOCKSIZE*BLOCKSIZE;
    long BlocksPerGrid= device_data_size / max + 1;
    dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
    if(device_data_size % max == 0)BlocksPerGrid--;
    dim3 dimGrid(1,BlocksPerGrid);
    check_block_grid_dim(deviceProp,dimBlock,dimGrid);

    MatVectMultiplication<<<dimGrid,dimBlock>>>(device_database_A, device_database_B, device_database_probY,
            device_input_A, device_input_B, device_data_size,dimension,device_result_vector);

}


void single_gpu_manager::get_device_property() {
    int device;
    // Current Device Detection
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp,device);
    // printf("Using device %d: %s \n", device, deviceProp.name);

}

void single_gpu_manager::set_device(int _device_id, const std::string& message="") {
    // Device Selection, Device 1: Tesla C1060
    // std::cout << "Setting Device :: " << _device_id << " for " << message << std::endl;
    cudaSetDevice(_device_id);
}


void single_gpu_manager::allocate_memory() {
    //allocating memory on GPU
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_database_B, device_data_size * dimension * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_database_A, device_data_size * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_database_probY, device_data_size * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_input_B, dimension * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_input_A, sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_result_vector, device_data_size * sizeof(double)));
}


void single_gpu_manager::copy_data_to_device(long offset, float* host_database_B, float* host_database_A, float* host_database_probY) {
    //moving data from CPU to GPU
    CUDA_SAFE_CALL(cudaMemcpy((void*)(device_database_B + offset*dimension), (void*)host_database_B, batch_size * dimension * sizeof(double) , cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)(device_database_A + offset), (void*)host_database_A, batch_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)(device_database_probY + offset), (void*)host_database_probY, batch_size * sizeof(double), cudaMemcpyHostToDevice));
}

void single_gpu_manager::copy_input_to_device(float* host_input_B, float* host_input_A) {
    //moving data from CPU to GPU
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_input_B, (void*)host_input_B, dimension * sizeof(double) , cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_input_A, (void*)host_input_A, sizeof(double), cudaMemcpyHostToDevice));
}

void single_gpu_manager::copy_result_to_host(float *host_ResVect) {
    //retriving result from device
    CUDA_SAFE_CALL(cudaMemcpy((void*)host_ResVect, (void*)device_result_vector, device_data_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void single_gpu_manager::_free() {
    /*free the memory from GPU */
    CUDA_SAFE_CALL(cudaFree(device_database_B));
    CUDA_SAFE_CALL(cudaFree(device_database_A));
    CUDA_SAFE_CALL(cudaFree(device_database_probY));
    CUDA_SAFE_CALL(cudaFree(device_input_B));
    CUDA_SAFE_CALL(cudaFree(device_input_A));
    CUDA_SAFE_CALL(cudaFree(device_result_vector));
}


void single_gpu_manager::start_event() {
    //event creation...
    CUDA_SAFE_CALL(cudaEventCreate (&start));
    CUDA_SAFE_CALL(cudaEventRecord (start, 0));
    CUDA_SAFE_CALL(cudaEventCreate (&stop));
}

double single_gpu_manager::stop_event() {
    CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize (stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime ( &elapsedTime, start, stop));
    double Tsec= 1.0e-3*elapsedTime; // time in seconds
    return Tsec;
}

single_gpu_manager::single_gpu_manager(int device_id, long device_num_batches, long batch_size, int dimension) {
    this->device_id = device_id;

    this->device_num_batches = device_num_batches;
    this->batch_size = batch_size;
    this->device_data_size = device_num_batches*batch_size;

    this->dimension = dimension;
    set_device(device_id, "Initialization");
    get_device_property();
    allocate_memory();
}
