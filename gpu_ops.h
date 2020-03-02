//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_GPU_OPS_H
#define CUDA_CODE_SEARCH_GPU_OPS_H

#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_utils.h"
#include "utils.h"

class gpu_ops {
private:
    int matRowSize, matColSize, vlength, size;
    double *host_Mat, *host_Vect;
    double *device_Mat,*device_Vect,*device_ResVect;
    /*function to launch kernel*/

public:
    gpu_ops(int, int, double*, double*, int, int);
//    void MatVectMultiplication(const double *_device_Mat, const double *_device_Vect, int _matRowSize, int _vlength, double *_device_ResVect);

    void allocate_memory();
    void copy_to_device();
    void launch_kernel(cudaDeviceProp);
    void get_data_to_host(double*);
    void _free();
};


#endif //CUDA_CODE_SEARCH_GPU_OPS_H
