//
// Created by rm38 on 3/5/20.
//

#ifndef CUDA_CODE_SEARCH_DATABASE_READER_H
#define CUDA_CODE_SEARCH_DATABASE_READER_H
#include <iostream>
#include <json/value.h>
#include <jsoncpp/json/json.h>
#include <vector>
#include "Program.h"
#include "ProgramBatch.h"

class database_reader {
    int num_threads;
    std::vector<ProgramBatch> list_of_batches;
    float **host_database_B, **host_database_A, **host_database_prob_Y;


public:
    database_reader(int,int,int);
    void read(int);
    void get_as_double_pointer();
};


#endif //CUDA_CODE_SEARCH_DATABASE_READER_H
