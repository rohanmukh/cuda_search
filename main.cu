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
#include "gpu_manager.h"
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


    gpu_manager* manager = new gpu_manager(0, matRowSize, matColSize, vlength);
    manager->copy_data(host_system->host_Mat, host_system->host_Vect);
    float time_sec = manager->compute_and_store(host_system->host_ResVect);


    // calling funtion for measuring Gflops & printing the result on screen
    print_on_screen("MAT VECT MULTIPLICATION",time_sec, calculate_gflops(time_sec, size),size,1);


    // CPU calculation..and checking error deviation....
    serial_code *cpu_user = new serial_code(matRowSize, matColSize, host_system->host_Mat, host_system->host_Vect, vlength, size);
    cpu_user->CPU_MatVectMult();
    relative_error(cpu_user->get_result(), host_system->host_ResVect, size);
    printf("\n ----------------------------------------------------------------------\n");

    host_system->_free();
    manager->_free();
    cpu_user->_free();

    return 0;
}// end of main

