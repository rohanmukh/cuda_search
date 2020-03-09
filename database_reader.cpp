//
// Created by rm38 on 3/5/20.
//

#include "database_reader.h"

database_reader::database_reader(int num_batches, int data_size_per_batch, int dimension){
    this->num_batches = num_batches;
    for (int i=0;i<num_batches;i++){
        ProgramBatch *batch1 = new ProgramBatch(data_size_per_batch, dimension);
        list_of_batches.push_back(batch1);
    }
}


void database_reader::read(int num_jsons) {
    #pragma omp parallel for
    for (int j=0;j<num_jsons;j++) {
        int thread_id = j % num_batches;
        auto batch = list_of_batches.at(thread_id);
        std::string file_name = "/home/ubuntu/DATABASE/Program_output_" + std::to_string(j+1) + ".json";
        batch->read_single_database_json(file_name);
        std::cout << "JSON " << j << " has " << batch->num_programs << " elements." << std::endl;
    }
}


void database_reader::reorganize() {

    num_batches = list_of_batches.size();
    batch_size = list_of_batches.at(0)->num_programs;
    host_database_B = (float**)malloc(num_batches * sizeof(float*)); 
    host_database_A = (float**)malloc(num_batches * sizeof(float*)); 
    host_database_prob_Y = (float**)malloc(num_batches * sizeof(float*));

    for(int i=0; i<num_batches;i++){
        auto batch = list_of_batches.at(i);
        host_database_B[i] = batch->json_database_B;
        host_database_A[i] = batch->json_database_A;
        host_database_prob_Y[i] = batch->json_database_prob_Y;
    }
    
    return;
}

void database_reader::_free(){
    int i=0;
    for(ProgramBatch* batch : list_of_batches){
        free(host_database_B[i]);
        free(host_database_A[i]);
        free(host_database_prob_Y[i]);
        i++;
    }
    free(host_database_B);
    free(host_database_A);
    free(host_database_prob_Y);
}
