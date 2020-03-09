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
#include <tuple>

#define DIMENSION 256

#define NUM_BATCHES 8
#define DATA_SIZE_PER_BATCH 100000

#define NUM_JSONS 8

/*main function*/
int main()
{
    // Vector length , Matrix Row and Col sizes..............
    int dimension = DIMENSION;

//    auto *host_db = new host_database(NUM_BATCHES * DATA_SIZE_PER_BATCH, dimension);
//    host_db->fill_database();

    auto* host_db = new database_reader(NUM_BATCHES, DATA_SIZE_PER_BATCH, DIMENSION);
    host_db->read(NUM_JSONS);
    host_db->reorganize();


    auto *query = new query_holder(dimension);
    query->fill_input_query();


    auto* gpu_user = new gpu_manager(host_db->num_batches, host_db->batch_size, dimension);
    gpu_user->copy_database_to_device(host_db->host_database_B, host_db->host_database_A,
                                      host_db->host_database_prob_Y
                                      );

    gpu_user->add_query(query->host_query_B, query->host_query_A);
    gpu_user->search();
    std::vector<std::tuple<int, int>> top_prog_ids =  gpu_user->top_k();
    for(std::tuple<int,int> prog_id : top_prog_ids){
       int batch_id = std::get<0>(prog_id);
       int batch_prog_id = std::get<1>(prog_id); 
       Program* p = host_db->get_program(batch_id, batch_prog_id);      
       std::cout << batch_id << " " << batch_prog_id << std::endl;
       std::cout << p->get_body() << std::endl;
       
    }

    auto *cpu_user = new cpu_manager(
            host_db->num_batches, host_db->batch_size, dimension, host_db->host_database_B,
            host_db->host_database_A, host_db->host_database_prob_Y
            );
    cpu_user->add_query(query->host_query_B, query->host_query_A);
    cpu_user->search();

    relative_error(cpu_user->get_result(), gpu_user->get_result(), DATA_SIZE_PER_BATCH);

    // Free Memory
    host_db->_free();
    query->_free();
    gpu_user->_free();
    cpu_user->_free();

//    return 0;
}// end of main
