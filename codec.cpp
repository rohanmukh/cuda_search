//
// Created by ROHAN MUKHERJEE on 3/9/20.
//

#include "codec.h"


codec::codec( int data_size_per_batch, int dimension, int num_jsons){

    this->data_size_per_batch = data_size_per_batch;

    host_db = new database_reader(num_jsons, data_size_per_batch, dimension);
    host_db->read(num_jsons);
    host_db->reorganize();


    //TODO: change num_batches -> is same as num_jsons
    gpu_user = new gpu_manager(host_db->num_batches, host_db->batch_size, dimension);
    gpu_user->copy_database_to_device(host_db->host_database_B, host_db->host_database_A,
                                      host_db->host_database_prob_Y
    );

    cpu_user = new cpu_manager(
            host_db->num_batches, host_db->batch_size, dimension, host_db->host_database_B,
            host_db->host_database_A, host_db->host_database_prob_Y
    );
    //    auto *host_db = new host_database(NUM_BATCHES * DATA_SIZE_PER_BATCH, DIMENSION);
    //    host_db->fill_database();

}



void codec::search(float *host_query_B, float *host_query_A){
    gpu_user->add_query(host_query_B, host_query_A);
    gpu_user->search();
    std::vector<std::tuple<int, int>> top_prog_ids =  gpu_user->top_k();
    for(std::tuple<int,int> prog_id : top_prog_ids){
        int batch_id = std::get<0>(prog_id);
        int batch_prog_id = std::get<1>(prog_id);
        Program* p = host_db->get_program(batch_id, batch_prog_id);
        std::cout << batch_id << " " << batch_prog_id << std::endl;
        std::cout << p->get_body() << std::endl;
    }
    return;
}


void codec::verify(float *host_query_B, float *host_query_A){
    cpu_user->add_query(host_query_B, host_query_A);
    cpu_user->search();
    relative_error(cpu_user->get_result(), gpu_user->get_result(), data_size_per_batch);
}


void codec::_free(){
    host_db->_free();
    gpu_user->_free();
    cpu_user->_free();
}
