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
#include "gpu_ops.h"

#define SIZE 1024

cudaDeviceProp deviceProp;

int vlength, matRowSize , matColSize;
int size = SIZE;

/*main function*/
int main()
{
    // Vector length , Matrix Row and Col sizes..............
    vlength = matColSize = SIZE;
    matRowSize = SIZE;


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


    // CUDA ops can start
    //  printf("this programs does computation of square matrix only\n");
    float elapsedTime;
    cudaEvent_t stop;

    int device_Count=get_DeviceCount();
    printf("\n\nNumber of Devices : %d\n\n", device_Count);

    // Device Selection, Device 1: Tesla C1060
    cudaSetDevice(0);

    int device;
    // Current Device Detection
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp,device);
    printf("Using device %d: %s \n", device, deviceProp.name);



    //event creation...
    cudaEvent_t start;
    CUDA_SAFE_CALL(cudaEventCreate (&start));
    CUDA_SAFE_CALL(cudaEventRecord (start, 0));
    CUDA_SAFE_CALL(cudaEventCreate (&stop));

    gpu_ops *gpu_user = new gpu_ops(matRowSize, matColSize, host_Mat, host_Vect, vlength, size);
    gpu_user->allocate_memory();
    gpu_user->copy_to_device();
    gpu_user->launch_kernel(deviceProp);
    gpu_user->get_data_to_host(host_ResVect);

    CUDA_SAFE_CALL(cudaEventRecord (stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize (stop));
    CUDA_SAFE_CALL(cudaEventElapsedTime ( &elapsedTime, start, stop));


    // calling funtion for measuring Gflops & printing the result on screen
    float Tsec= 1.0e-3*elapsedTime;
    print_on_screen("MAT VECT MULTIPLICATION",Tsec,calculate_gflops(Tsec, size),size,1);


    // CPU calculation..and checking error deviation....
    serial_code *cpu_user = new serial_code(matRowSize, matColSize, host_Mat, host_Vect, vlength, size);
    cpu_user->CPU_MatVect();
    relative_error(cpu_user->get_result(), host_ResVect, size);
    printf("\n ----------------------------------------------------------------------\n");

    //free host memory----------
    free(host_Mat);
    free(host_Vect);
    free(host_ResVect);
    gpu_user->_free();
    cpu_user->_free();

    return 0;
}// end of main

