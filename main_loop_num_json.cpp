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
#include <cmath>

#define DIMENSION 256

#define DATA_SIZE_PER_BATCH 100000
#define NUM_JSONS 256
#define MAX_DEVICES 16

std::string todo_log_path = "/home/ubuntu/cuda_search/encoder_op.json";
/*main function*/
int main()
{

    auto *system = new codec( DATA_SIZE_PER_BATCH, DIMENSION, NUM_JSONS);
    system->set_gpu_user(MAX_DEVICES);

    auto *query = new query_holder(DIMENSION);
    std::vector<int> num_json_array{256, 128, 64, 32, 16};

    for(int i=0;i<num_json_array.size();i++){
        int num_jsons = num_json_array.at(i);
        system->shrink_data(num_jsons, MAX_DEVICES);
        std::cout << "/* Running on NUM JSON :: " <<  num_jsons << "*/"<< std::endl;
        query->read_input_json(todo_log_path);
        system->search(query->host_query_B, query->host_query_A);
        //system->verify(query->host_query_B, query->host_query_A);
    }

    // Free Memory
    query->_free();
    system->_free();

//    return 0;
}// end of main
