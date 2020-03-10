//
// Created by ROHAN MUKHERJEE on 3/9/20.
//

#include "codec.h"

codec::codec( int data_size_per_batch, int dimension, int num_jsons, int max_devices){

    this->data_size_per_batch = data_size_per_batch;

    host_db = new database_reader(num_jsons, data_size_per_batch, dimension);
    host_db->read(num_jsons);
    host_db->reorganize();


    //TODO: change num_batches -> is same as num_jsons
    gpu_user = new gpu_manager(max_devices, host_db->num_batches, host_db->batch_size, dimension);
    gpu_user->copy_database_to_device(host_db->host_database_B, host_db->host_database_A,
                                      host_db->host_database_prob_Y
    );

    //cpu_user = new cpu_manager(
    //        host_db->num_batches, host_db->batch_size, dimension, host_db->host_database_B,
    //        host_db->host_database_A, host_db->host_database_prob_Y
    //);
    //    auto *host_db = new host_database(NUM_BATCHES * DATA_SIZE_PER_BATCH, DIMENSION);
    //    host_db->fill_database();

}



void codec::search(float *host_query_B, float *host_query_A){
    gpu_user->add_query(host_query_B, host_query_A);
    gpu_user->search();
    std::vector<std::tuple<int, int, float>> top_prog_ids =  gpu_user->top_k();
    int i=0;

    Json::Value op_program_head;
    Json::Value op_prog_array(Json::arrayValue);
 
    for(std::tuple<int,int, float> prog_id : top_prog_ids){
        int batch_id = std::get<0>(prog_id);
        int batch_prog_id = std::get<1>(prog_id);
        float prob = std::get<2>(prog_id);
        Program* p = host_db->get_program(batch_id, batch_prog_id);
        i++;
        
        Json::Value op_prog;
        op_prog["Rank"] = i;
        op_prog["Probability"] = prob;
        op_prog["Body"] = p->get_body();
        op_prog_array.append(op_prog); 
      
        //std::cout << " Rank :: " << i << std::endl;
        //std::cout << " Probability :: " << prob << std::endl;
        //std::cout << p->get_body() << std::endl;
    }
    op_program_head["top_programs"]=op_prog_array;
    dump_json(op_program_head);

    return;
}

void codec::dump_json(Json::Value event){
    std::ofstream file_id;
    file_id.open("top_programs.json");

    Json::StyledWriter styledWriter;
    file_id << styledWriter.write(event);

    file_id.close();
}

void codec::verify(float *host_query_B, float *host_query_A){
    cpu_user->add_query(host_query_B, host_query_A);
    cpu_user->search();
    relative_error(cpu_user->get_result(), gpu_user->get_result(), data_size_per_batch);
}


void codec::_free(){
    host_db->_free();
    gpu_user->_free();
    //cpu_user->_free();
}
