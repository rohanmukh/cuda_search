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