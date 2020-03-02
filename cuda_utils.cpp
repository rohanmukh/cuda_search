//
// Created by rm38 on 3/2/20.
//

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include "cuda_utils.h"

/*Check for safe return of all calls to the device */
void CUDA_SAFE_CALL(cudaError_t call)
{
    cudaError_t ret = call;
    //printf("RETURN FROM THE CUDA CALL:%d\t:",ret);
    switch(ret)
    {
        case cudaSuccess:
            //              printf("Success\n");
            break;
            /*      case cudaErrorInvalidValue:
                                    {
                                    printf("ERROR: InvalidValue:%i.\n",__LINE__);
                                    exit(-1);
                                    break;
                                    }
                    case cudaErrorInvalidDevicePointer:
                                    {
                                    printf("ERROR:Invalid Device pointeri:%i.\n",__LINE__);
                                    exit(-1);
                                    break;
                                    }
                    case cudaErrorInvalidMemcpyDirection:
                                    {
                                    printf("ERROR:Invalid memcpy direction:%i.\n",__LINE__);
                                    exit(-1);
                                    break;
                                    }                       */
        default:
        {
            printf(" ERROR at line :%i.%d' ' %s\n",__LINE__,ret,cudaGetErrorString(ret));
            exit(-1);
            break;
        }
    }
}

/*free memory*/
void dfree(double * arr[],int len)
{
    for(int i=0;i<len;i++)

    printf("mem freed\n");
}

/*funtion to check blocks per grid and threads per block*/
void check_block_grid_dim(cudaDeviceProp devProp,dim3 blockDim,dim3 gridDim)
{

    if( blockDim.x >= devProp.maxThreadsDim[0] || blockDim.y >= devProp.maxThreadsDim[1] || blockDim.z >= devProp.maxThreadsDim[2] )
    {
        printf("\nBlock Dimensions exceed the maximum limits:%d * %d * %d \n",devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
        exit(-1);
    }

    if( gridDim.x >= devProp.maxGridSize[0] || gridDim.y >= devProp.maxGridSize[1] || gridDim.z >= devProp.maxGridSize[2] )
    {
        printf("\nGrid Dimensions exceed the maximum limits:%d * %d * %d \n",devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
        exit(-1);
    }
}

/*Get the number of GPU devices present on the host */
int get_DeviceCount()
{
    int count;
    cudaGetDeviceCount(&count);
    return count;
}
