//
// Created by rm38 on 3/5/20.
//

#include "database_reader.h"

database_reader::database_reader(int num_threads, int data_size, int dimension){
    this->num_threads = num_threads;
    for (int i=0;i<num_threads;i++){
        auto batch1 = ProgramBatch(data_size, dimension);
        list_of_batches.push_back(batch1);
    }
}


void database_reader::read(int num_jsons) {
    #pragma omp parallel for
    for (int j=0;j<num_jsons;j++) {
        int thread_id = j % num_threads;
        auto batch = list_of_batches.at(thread_id);
        std::string file_name = "Program_output_" + std::to_string(j+1) + ".json";
        batch.read_single_database_json(file_name);
        std::cout << "JSON " << j << " has " << batch.num_programs << " elements." << std::endl;
    }
}


void database_reader::get_as_double_pointer() {

    int num_batches = list_of_batches.size();
    host_database_B = (float**)malloc(num_batches * sizeof(float*)); //new float[data_size * dimension];
    host_database_A = (float**)malloc(num_batches * sizeof(float*)); //new float[dimension];
    host_database_prob_Y = (float**)malloc(num_batches * sizeof(float*)); //new float[dimension];

    int i = 0;
    for(ProgramBatch& batch : list_of_batches){
        host_database_B[i] = batch.json_database_B;
        host_database_A[i] = batch.json_database_A;
        host_database_prob_Y[i] = batch.json_database_prob_Y;
    }
    return;
}