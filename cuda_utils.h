//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_CUDA_UTILS_H
#define CUDA_CODE_SEARCH_CUDA_UTILS_H


/*Check for safe return of all calls to the device */
void CUDA_SAFE_CALL(cudaError_t call);

/*funtion to check blocks per grid and threads per block*/
void check_block_grid_dim(cudaDeviceProp devProp,dim3 blockDim,dim3 gridDim);

/*Get the number of GPU devices present on the host */
int get_DeviceCount();


#endif //CUDA_CODE_SEARCH_CUDA_UTILS_H
