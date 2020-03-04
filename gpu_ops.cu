//
// Created by rm38 on 3/2/20.
//

#include "gpu_ops.h"
#include <cstdio>
#include <iostream>
#include "matrix_vector_op.cu"

/*function to launch kernel*/
void gpu_ops::launch_kernel() {
    // Launching kernel..........

    /* threads_per_block, blocks_per_grid  */
    int max=BLOCKSIZE*BLOCKSIZE;
    long BlocksPerGrid= batch_size / max + 1;
    dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
    if(batch_size % max == 0)BlocksPerGrid--;
    dim3 dimGrid(1,BlocksPerGrid);
    check_block_grid_dim(deviceProp,dimBlock,dimGrid);

    MatVectMultiplication<<<dimGrid,dimBlock>>>(device_database_A, device_database_B, device_database_probY,
            device_input_A, device_input_B, batch_size,dimension,device_ResDistance);

}


void gpu_ops::get_device_property() {
    int device;
    // Current Device Detection
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp,device);
    printf("Using device %d: %s \n", device, deviceProp.name);

}

void gpu_ops::set_device(int _device_id, const std::string& message="") {
    // Device Selection, Device 1: Tesla C1060
    std::cout << "Setting Device :: " << _device_id << " for " << message << std::endl;
    cudaSetDevice(_device_id);
}


void gpu_ops::allocate_memory() {
    //allocating memory on GPU
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_database_B, batch_size * dimension * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_database_A, batch_size * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_database_probY, batch_size * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_input_B, dimension * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_input_A, sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&device_ResDistance, batch_size * sizeof(double)));
}


void gpu_ops::copy_data_to_device(double* host_database_B, double* host_database_A, double* host_database_probY) {
    //moving data from CPU to GPU
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_database_B, (void*)host_database_B, batch_size * dimension * sizeof(double) , cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_database_A, (void*)host_database_A, batch_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_database_probY, (void*)host_database_probY, batch_size * sizeof(double), cudaMemcpyHostToDevice));
}

void gpu_ops::copy_input_to_device(double* host_input_B, double* host_input_A) {
    //moving data from CPU to GPU
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_input_B, (void*)host_input_B, dimension * sizeof(double) , cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_input_A, (void*)host_input_A, sizeof(double), cudaMemcpyHostToDevice));
}

void gpu_ops::copy_result_to_host(double *host_ResVect) {
    //retriving result from device
    CUDA_SAFE_CALL(cudaMemcpy((void*)host_ResVect, (void*)device_ResDistance, batch_size * sizeof(double), cudaMemcpyDeviceToHost));
}

void gpu_ops::_free() {
    /*free the memory from GPU */
    CUDA_SAFE_CALL(cudaFree(device_database_B));
    CUDA_SAFE_CALL(cudaFree(device_database_A));
    CUDA_SAFE_CALL(cudaFree(device_input_B));
    CUDA_SAFE_CALL(cudaFree(device_input_A));
    CUDA_SAFE_CALL(cudaFree(device_ResDistance));
}


void gpu_ops::start_event() {
    //event creation...
    CUDA_SAFE_CALL(cudaEventCreate (&start));
    CUDA_SAFE_CALL(cudaEventRecord (start, 0));
    CUDA_SAFE_CALL(cudaEventCreate (&stop));
}

float gpu_ops::stop_event() {
    CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize (stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime ( &elapsedTime, start, stop));
    float Tsec= 1.0e-3*elapsedTime; // time in seconds
    return Tsec;
}

gpu_ops::gpu_ops(int device_id, long batch_size, int dimension) {
    this->device_id = device_id;
    this->batch_size = batch_size;
    this->dimension = dimension;
    set_device(device_id, "Initialization");
    get_device_property();
    allocate_memory();
}

