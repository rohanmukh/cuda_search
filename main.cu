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
#include "host_ops.h"

#define SIZE 1024


int vlength, matRowSize , matColSize;
int size = SIZE;

/*main function*/
int main()
{
    // Vector length , Matrix Row and Col sizes..............
    vlength = matColSize = SIZE;
    matRowSize = SIZE;

    host_ops *host_system = new host_ops(matRowSize, matColSize, vlength);
    host_system->fill_with_random_data();

    int device_Count=get_DeviceCount();
    printf("\n\nNumber of Devices : %d\n\n", device_Count);



    gpu_ops *gpu_user = new gpu_ops(matRowSize, matColSize, vlength);
    gpu_user->set_device(0);
    gpu_user->start_event();
    gpu_user->allocate_memory();
    gpu_user->copy_to_device(host_system->host_Mat, host_system->host_Vect);
    gpu_user->launch_kernel();
    gpu_user->copy_to_host(host_system->host_ResVect);
    float time_sec = gpu_user->stop_event();



    // calling funtion for measuring Gflops & printing the result on screen
    print_on_screen("MAT VECT MULTIPLICATION",time_sec, calculate_gflops(time_sec, size),size,1);


    // CPU calculation..and checking error deviation....
    serial_code *cpu_user = new serial_code(matRowSize, matColSize, host_system->host_Mat, host_system->host_Vect, vlength, size);
    cpu_user->CPU_MatVectMult();
    relative_error(cpu_user->get_result(), host_system->host_ResVect, size);
    printf("\n ----------------------------------------------------------------------\n");

    host_system->_free();
    gpu_user->_free();
    cpu_user->_free();

    return 0;
}// end of main

