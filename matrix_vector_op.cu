//
// Created by rm38 on 3/2/20.
//
#include <cuda.h>
#include <cmath>
#define BLOCKSIZE 16

/////////////////////////////////////////////////////////////////////////////////////////
//
// MatVect : this kernel will perform actual MatrixVector Multiplication
//
/////////////////////////////////////////////////////////////////////////////////////////

__global__
void MatVectMultiplication(const float *device_database_A, const float *device_database_B, const float *device_database_probY,
        const float *device_input_A, const float *device_input_B,
        int batch_size, int dimension, float *_device_ResVect)
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int tindex=tidx+gridDim.x*BLOCKSIZE*tidy;

    if(tindex < batch_size) {
        int m = tindex * dimension;
        _device_ResVect[tindex] = 0.00;
        for (int i = 0; i < dimension; i++) {
            _device_ResVect[tindex] += pow(device_input_B[i], 2) / (4 * device_input_A[0]); // additive ab1 1st item
            _device_ResVect[tindex] +=
                    pow(device_database_B[m + i], 2) / (4 * device_database_A[tindex]); // additive ab2 1st item
            _device_ResVect[tindex] -= pow(device_database_B[m + i] + device_input_B[i], 2) /
                                       (4 * (device_database_A[tindex] + device_input_A[0])); // subtractive ab_star 1st item
        }
        _device_ResVect[tindex] += 0.5 * dimension * log(-1 * (device_input_A[0]) / M_PI); // additive ab1 2nd item
        _device_ResVect[tindex] += 0.5 * dimension * log(-1 * (device_database_A[tindex]) / M_PI); // additive ab2_2nd item
        _device_ResVect[tindex] -= 0.5 * dimension * log(-1 * (device_database_A[tindex] + device_input_A[0]) /
                                                         M_PI); // subtractive ab_star 2nd item
        _device_ResVect[tindex] -= 0.5 * dimension * log(2 * M_PI); // subtractive cons
        _device_ResVect[tindex] += device_database_probY[tindex];
    }
    __syncthreads();

}//end of MatVect device function

