/*****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example     :  cuda-matrix-vector-multiplication.cu

  Objective   : Write CUDA program to compute Matrix-Vector multiplication.

  Input       : None

  Output      : Execution time in seconds , Gflops achieved

  Created     : August-2013

  E-mail      : hpcfte@cdac.in

****************************************************************************/


#include<cstdio>
#include<cuda.h>
#include "serial_code.h"
#include "utils.h"
#include "cuda_utils.h"

#define BLOCKSIZE 16
#define SIZE 1024

cudaDeviceProp deviceProp;


double *device_Mat,*device_Vect,*device_ResVect;
int     vlength, matRowSize , matColSize;
int     size = SIZE;



/////////////////////////////////////////////////////////////////////////////////////////
//
// MatVect : this kernel will perform actual MatrixVector Multiplication
//
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void MatVectMultiplication(double *_device_Mat, double *_device_Vect, int _matRowSize, int _vlength, double *_device_ResVect)
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
void launch_Kernel_MatVectMul()
{
/*          threads_per_block, blocks_per_grid  */
    int max=BLOCKSIZE*BLOCKSIZE;
    int BlocksPerGrid=matRowSize/max+1;
    dim3 dimBlock(BLOCKSIZE,BLOCKSIZE);
    if(matRowSize%max==0)BlocksPerGrid--;
    dim3 dimGrid(1,BlocksPerGrid);
    check_block_grid_dim(deviceProp,dimBlock,dimGrid);

    MatVectMultiplication<<<dimGrid,dimBlock>>>(device_Mat,device_Vect,matRowSize,vlength,device_ResVect);

}


/*main function*/
int main()
{
    // Vector length , Matrix Row and Col sizes..............
    vlength = matColSize = SIZE;
    matRowSize = SIZE;

    //  printf("this programs does computation of square matrix only\n");
    float elapsedTime;
    cudaEvent_t start,stop;

    int device_Count=get_DeviceCount();
    printf("\n\nNumber of Devices : %d\n\n", device_Count);

    // Device Selection, Device 1: Tesla C1060
    cudaSetDevice(0);

    int device;
    // Current Device Detection
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp,device);
    printf("Using device %d: %s \n", device, deviceProp.name);



    /*allocating the memory for each matrix */
    double *host_Mat =new double[matRowSize*matColSize];
    double *host_Vect = new double[vlength];
    double *host_ResVect = new double[matRowSize];


    // ---------------checking host memory  for error..............................
    if(host_Mat==NULL)
        mem_error("host_Mat","vectmatmul",matRowSize*matColSize,"double");

    if(host_Vect==NULL)
        mem_error("host_Vect","vectmatmul",vlength,"double");

    if(host_ResVect==NULL)
        mem_error("host_ResVect","vectmatmul",matRowSize,"double");

    //--------------Initializing the input arrays..............
    fill_with_random_doubles(host_Mat, matRowSize * matColSize);
    fill_with_random_doubles(host_Vect, vlength);

    /* allocate memory for GPU events
       start = (cudaEvent_t) malloc (sizeof(cudaEvent_t));
       stop = (cudaEvent_t) malloc (sizeof(cudaEvent_t));
       if(start==NULL)
               mem_error("start","vectvectmul",1,"cudaEvent_t");
       if(stop==NULL)
               mem_error("stop","vectvectmul",1,"cudaEvent_t");*/

    //event creation...
    CUDA_SAFE_CALL(cudaEventCreate (&start));
    CUDA_SAFE_CALL(cudaEventCreate (&stop));

    //allocating memory on GPU
    CUDA_SAFE_CALL(cudaMalloc( (void**)&device_Mat, matRowSize*matColSize* sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&device_Vect, vlength* sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&device_ResVect, matRowSize* sizeof(double)));

    //moving data from CPU to GPU
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_Mat, (void*)host_Mat, matRowSize*matColSize*sizeof(double) ,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void*)device_Vect, (void*)host_Vect,vlength*sizeof(double),cudaMemcpyHostToDevice));

    // Launching kernell..........
    CUDA_SAFE_CALL(cudaEventRecord (start, 0));

    launch_Kernel_MatVectMul();

    CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize (stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime ( &elapsedTime, start, stop));


    // calling funtion for measuring Gflops & printing the result on screen
    float Tsec= 1.0e-3*elapsedTime;
    print_on_screen("MAT VECT MULTIPLICATION",Tsec,calculate_gflops(Tsec, size),size,1);


    //retriving result from device
    CUDA_SAFE_CALL(cudaMemcpy((void*)host_ResVect, (void*)device_ResVect,matRowSize*sizeof(double),cudaMemcpyDeviceToHost));

    // CPU calculation..and checking error deviation....
    serial_code *cpu_user = new serial_code(matRowSize, matColSize, host_Mat, host_Vect, vlength, size);
    cpu_user->CPU_MatVect();
    relative_error(cpu_user->get_result(), host_ResVect, size);
    printf("\n ----------------------------------------------------------------------\n");

    /*free the memory from GPU */
    CUDA_SAFE_CALL(cudaFree(device_Mat));
    CUDA_SAFE_CALL(cudaFree(device_Vect));
    CUDA_SAFE_CALL(cudaFree(device_ResVect));
    printf("mem freed\n");

    //free host memory----------
    free(host_Mat);
    free(host_Vect);
    free(host_ResVect);
    cpu_user->_free();

    return 0;
}// end of main

