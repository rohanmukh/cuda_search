//
// Created by rm38 on 3/2/20.
//

#include "gpu_ops.h"
#include <cstdio>

#define BLOCKSIZE 16


/////////////////////////////////////////////////////////////////////////////////////////
//
// MatVect : this kernel will perform actual MatrixVector Multiplication
//
/////////////////////////////////////////////////////////////////////////////////////////
__global__
void MatVectMultiplication(const double *_device_Mat, const double *_device_Vect, int _matRowSize, int _vlength, double *_device_ResVect)
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int tindex=tidx+gridDim.x*BLOCKSIZE*tidy;


    if(tindex < _matRowSize)
    {
        int i;int m= tindex * _vlength;
        _device_ResVect[tindex]=0.00;
        for(i=0; i < _vlength; i++)
            _device_ResVect[tindex]+= _device_Mat[m + i] * _device_Vect[i];
    }

    __syncthreads();

}//end of MatVect device function



/*function to launch kernel*/
void gpu_ops::launch_kernel() {
    // Launching kernel..........

    /* threads_per_block, blocks_per_grid  */
    int max=BLOCKSIZE*BLOCKSIZE;
    int BlocksPerGrid=matRowSize/max+1;
    dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
    if(matRowSize%max==0)BlocksPerGrid--;
    dim3 dimGrid(1,BlocksPerGrid);
    check_block_grid_dim(deviceProp,dimBlock,dimGrid);

    MatVectMultiplication<<<dimGrid,dimBlock>>>(device_Mat,device_Vect,matRowSize,vlength,device_ResVect);

}


void gpu_ops::get_device_property() {

    int device;
    // Current Device Detection
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp,device);
    printf("Using device %d: %s \n", device, deviceProp.name);

}

void gpu_ops::set_device(int device_id, std::string message="") {
    // Device Selection, Device 1: Tesla C1060
    cudaSetDevice(device_id);
}


void gpu_ops::allocate_memory() {
    //allocating memory on GPU
    CUDA_SAFE_CALL(cudaMalloc( (void**)&device_Mat, matRowSize*matColSize* sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&device_Vect, vlength* sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&device_ResVect, matRowSize* sizeof(double)));
}


void gpu_ops::copy_to_device(double* host_Mat, double* host_Vect) {
    //moving data from CPU to GPU
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_Mat, (void*)host_Mat, matRowSize*matColSize*sizeof(double) ,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_Vect, (void*)host_Vect,vlength*sizeof(double),cudaMemcpyHostToDevice));
}



void gpu_ops::copy_to_host(double *host_ResVect) {
    //retriving result from device
    CUDA_SAFE_CALL(cudaMemcpy((void*)host_ResVect, (void*)device_ResVect,matRowSize*sizeof(double),cudaMemcpyDeviceToHost));

}

void gpu_ops::_free() {
    /*free the memory from GPU */
    CUDA_SAFE_CALL(cudaFree(device_Mat));
    CUDA_SAFE_CALL(cudaFree(device_Vect));
    CUDA_SAFE_CALL(cudaFree(device_ResVect));
    printf("mem freed\n");
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

gpu_ops::gpu_ops(int device_id, int matRowSize, int matColSize, int vlength) {
    this->device_id = device_id;
    this->matRowSize = matRowSize;
    this->matColSize = matColSize;
    this->vlength = vlength;
    set_device(device_id);
    get_device_property();
}

