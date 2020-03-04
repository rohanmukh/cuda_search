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
#include <iostream>
#include "serial_code.h"
#include "utils.h"
#include "gpu_manager.h"
#include "host_ops.h"

#define DATA_SIZE 1024
#define DIMENSION 256


/*main function*/
int main()
{
    // Vector length , Matrix Row and Col sizes..............
    int dimension = DIMENSION;
    long data_size = DATA_SIZE;

    std::cout << "Initializing Host" << std::endl;
    auto *host_system = new host_ops(data_size, dimension);
    host_system->fill_database();
    host_system->fill_input_query();

    std::cout << "Initializing GPU Manager" << std::endl;
    auto* manager = new gpu_manager(data_size, dimension);
    manager->copy_data_to_database(host_system->host_database_B, host_system->host_database_A, host_system->host_database_prob_Y);
    manager->copy_input_to_device(host_system->host_input_B, host_system->host_input_A);
    float time_sec = manager->compute_and_store(host_system->host_ResVect);


    // calling funtion for measuring Gflops & printing the result on screen
    double gfops = calculate_gflops(time_sec, data_size*dimension);
    print_on_screen("MAT VECT MULTIPLICATION",time_sec, gfops, data_size*dimension,1);

    printf("\n ----------------------------------------------------------------------\n");
    // CPU calculation..and checking error deviation....
    std::cout << "===========================CPU Calculation==================================" << std::endl;
    auto *cpu_user = new serial_code(
            data_size, dimension, host_system->host_database_B,
            host_system->host_database_A, host_system->host_database_prob_Y,
            host_system->host_input_B, host_system->host_input_A
            );
    double elapsed_time = cpu_user->CPU_MatVectMult();
    std::cout << "Time elapsed :: " << elapsed_time << std::endl;

    std::cout << "===========================Relative Error==================================" << std::endl;
    relative_error(cpu_user->get_result(), host_system->host_ResVect, data_size);

    // Free Memory
    host_system->_free();
    manager->_free();
    cpu_user->_free();

    return 0;
}// end of main

