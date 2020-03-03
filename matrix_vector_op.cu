//
// Created by rm38 on 3/2/20.
//
#include <cuda.h>
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

