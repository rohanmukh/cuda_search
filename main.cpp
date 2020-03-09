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


#include <iostream>
#include "cpu_manager.h"
#include "utils.h"
#include "gpu_manager.h"
#include "host_database.h"
#include "query_holder.h"
#include "database_reader.h"

#define DIMENSION 256
#define DATA_SIZE 100000
#define NUM_JSONS 1
#define NUM_THREADS 1

/*main function*/
int main()
{
    // Vector length , Matrix Row and Col sizes..............
    int dimension = DIMENSION;
//    long data_size = DATA_SIZE;

//    auto *host_db = new host_database(DATA_SIZE, dimension);
//    host_db->fill_database();

    auto* host_db = new database_reader(NUM_THREADS, DATA_SIZE, DIMENSION);
    host_db->read(NUM_JSONS);
    host_db->reorganize();


    auto *query = new query_holder(dimension);
    query->fill_input_query();


    std::cout << host_db->num_batches  << std::endl ;
    auto* gpu_user = new gpu_manager(host_db->num_batches, host_db->batch_size, dimension);
    gpu_user->copy_database_to_device(host_db->host_database_B, host_db->host_database_A,
                                      host_db->host_database_prob_Y
                                      );

    gpu_user->add_query(query->host_query_B, query->host_query_A);
    gpu_user->search();
    //gpu_user->top_k();

    auto *cpu_user = new cpu_manager(
            host_db->num_batches, host_db->batch_size, dimension, host_db->host_database_B,
            host_db->host_database_A, host_db->host_database_prob_Y
            );
    cpu_user->add_query(query->host_query_B, query->host_query_A);
    cpu_user->search();

    relative_error(cpu_user->get_result(), gpu_user->get_result(), DATA_SIZE);

    // Free Memory
    host_db->_free();
    query->_free();
    gpu_user->_free();
    cpu_user->_free();

//    return 0;
}// end of main
