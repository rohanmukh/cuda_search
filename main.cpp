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
#include "utils.h"
#include "codec.h"
#include "server.h"
#include "query_holder.h"
#include <tuple>

#define DIMENSION 256

#define NUM_BATCHES 8
#define DATA_SIZE_PER_BATCH 100000

#define NUM_JSONS 8

/*main function*/
int main()
{

    auto *system = new codec( NUM_BATCHES,  DATA_SIZE_PER_BATCH, DIMENSION, NUM_JSONS);
    auto *query = new query_holder(DIMENSION);
    auto *_server =  new server();

    std::cout << "Waiting for client signal" << std::endl;
    while (true){
        std::cout << "Filling with random query" << std::endl;
        query->fill_input_query();

        _server->unblock_and_run();
        system->search(query->host_query_B, query->host_query_A);
        system->verify(query->host_query_B, query->host_query_A);
    }

    // Free Memory
    query->_free();
    system->_free();

//    return 0;
}// end of main
