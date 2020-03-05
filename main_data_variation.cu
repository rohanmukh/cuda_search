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
#include <vector>
#include "cpu_manager.h"
#include "utils.h"
#include "gpu_manager.h"
#include "host_database.h"
#include "query_holder.h"


#define DATA_SIZE 1000000
#define DIMENSION 256


/*main function*/
int main()
{
    int dimension = DIMENSION;

    std::vector<int> multiplier{ 1, 2, 4, 8, 16, 26, 32};
    for(int m : multiplier){
        long data_size = DATA_SIZE * m;
        std::cout << "/* Running on Data Size :: " << data_size << "*/"<< std::endl;
        auto *host_db = new host_database(data_size, dimension);
        host_db->fill_database();

        auto *query = new query_holder(dimension);
        query->fill_input_query();

        auto* gpu_user = new gpu_manager(host_db->num_batches, host_db->batch_size, dimension);
        gpu_user->copy_database_to_device(host_db->host_database_B, host_db->host_database_A,
                                          host_db->host_database_prob_Y
                                          );

        gpu_user->add_query(query->host_query_B, query->host_query_A);
        gpu_user->search();
        gpu_user->top_k();
        host_db->_free();
        query->_free();
        gpu_user->_free();
    }


    return 0;
}// end of main
