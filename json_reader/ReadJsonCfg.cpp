//
// Created by rm38 on 3/5/20.
//
#include <iostream>
#include <json/value.h>
#include <jsoncpp/json/json.h>
#include <vector>
#include "Program.h"
#include "ProgramBatch.h"

#define DIMENSION 256
#define DATA_SIZE 1000000
# define NUM_JSONS 1
#define NUM_THREADS 32

int main()
{
    int num_threads = NUM_THREADS;
    std::vector<ProgramBatch> list_of_batches;
    for (int i=0;i<num_threads;i++){
        auto batch1 = ProgramBatch(DATA_SIZE, DIMENSION);
        list_of_batches.push_back(batch1);
    }
    int offset = 0;
    for (int j=0;j<NUM_JSONS;j++) {
        int thread_id = j % num_threads;
        auto batch = list_of_batches.at(thread_id);
        std::string file_name = "Program_output_" + std::to_string(j) + ".json";
        batch.read_single_database_json(file_name);
        std::cout << "JSON " << j << " has " << batch.num_programs << " elements." std::endl;
    }

}

