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



gpu_ops::gpu_ops(int matRowSize, int matColSize, double *host_Mat,
                         double *host_Vect, int vlength, int size) {
    this->matRowSize = matRowSize;
    this->matColSize = matColSize;
    this->host_Mat = host_Mat;
    this->host_Vect = host_Vect;
    this->vlength = vlength;
    this->size = size;
}


void gpu_ops::allocate_memory() {
    //allocating memory on GPU
    CUDA_SAFE_CALL(cudaMalloc( (void**)&device_Mat, matRowSize*matColSize* sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&device_Vect, vlength* sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&device_ResVect, matRowSize* sizeof(double)));
}


void gpu_ops::copy_to_device() {
    //moving data from CPU to GPU
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_Mat, (void*)host_Mat, matRowSize*matColSize*sizeof(double) ,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_Vect, (void*)host_Vect,vlength*sizeof(double),cudaMemcpyHostToDevice));
}

void gpu_ops::launch_kernel(cudaDeviceProp deviceProp) {
    // Launching kernell..........

    /* threads_per_block, blocks_per_grid  */
    int max=BLOCKSIZE*BLOCKSIZE;
    int BlocksPerGrid=matRowSize/max+1;
    dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
    if(matRowSize%max==0)BlocksPerGrid--;
    dim3 dimGrid(1,BlocksPerGrid);
    check_block_grid_dim(deviceProp,dimBlock,dimGrid);

    MatVectMultiplication<<<dimGrid,dimBlock>>>(device_Mat,device_Vect,matRowSize,vlength,device_ResVect);

}


void gpu_ops::get_data_to_host(double *host_ResVect) {
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
