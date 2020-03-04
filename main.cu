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
#include "cpu_manager.h"
#include "utils.h"
#include "gpu_manager.h"
#include "host_database.h"
#include "query_holder.h"


#define DATA_SIZE 1024
#define DIMENSION 256


/*main function*/
int main()
{
    // Vector length , Matrix Row and Col sizes..............
    int dimension = DIMENSION;
    long data_size = DATA_SIZE;

    std::cout << "Initializing Host" << std::endl;
    auto *host_db = new host_database(data_size, dimension);
    host_db->fill_database();

    auto *query = new query_holder(dimension);
    query->fill_input_query();

    std::cout << "Initializing GPU Manager" << std::endl;
    auto* gpu_user = new gpu_manager(data_size, dimension);
    gpu_user->copy_database_to_device(host_db->host_database_B, host_db->host_database_A,
                                      host_db->host_database_prob_Y
                                      );
    gpu_user->copy_input_to_device(query->host_input_B, query->host_input_A);
    float gpu_time = gpu_user->compute_and_store(host_db->host_ResVect);


    // calling funtion for measuring Gflops & printing the result on screen
    calculate_gflops(gpu_time, data_size * dimension);

    // CPU calculation..and checking error deviation....
    auto *cpu_user = new cpu_manager(
            data_size, dimension, host_db->host_database_B,
            host_db->host_database_A, host_db->host_database_prob_Y,
            query->host_input_B, query->host_input_A
            );
    cpu_user->search();

    relative_error(cpu_user->get_result(), host_db->host_ResVect, data_size);

    // Free Memory
    host_db->_free();
    gpu_user->_free();
    cpu_user->_free();

    return 0;
}// end of main

