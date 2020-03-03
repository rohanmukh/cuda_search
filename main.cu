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
#include "serial_code.h"
#include "utils.h"
#include "gpu_manager.h"
#include "host_ops.h"

#define SIZE 1024


int data_size , dimension;
int size = SIZE;

/*main function*/
int main()
{
    // Vector length , Matrix Row and Col sizes..............
    dimension = SIZE;
    data_size = SIZE * 1;

    host_ops *host_system = new host_ops(data_size, dimension);
    host_system->fill_with_random_data();


    gpu_manager* manager = new gpu_manager(data_size, dimension);
    manager->copy_data(host_system->host_Mat, host_system->host_Vect);
    float time_sec = manager->compute_and_store(host_system->host_ResVect);


    // calling funtion for measuring Gflops & printing the result on screen
    print_on_screen("MAT VECT MULTIPLICATION",time_sec, calculate_gflops(time_sec, size),size,1);


    // CPU calculation..and checking error deviation....
    serial_code *cpu_user = new serial_code(data_size, dimension, host_system->host_Mat, host_system->host_Vect, size);
    cpu_user->CPU_MatVectMult();
    relative_error(cpu_user->get_result(), host_system->host_ResVect, size);
    printf("\n ----------------------------------------------------------------------\n");

    host_system->_free();
    manager->_free();
    cpu_user->_free();

    return 0;
}// end of main

