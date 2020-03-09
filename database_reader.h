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
    std::vector<ProgramBatch*> list_of_batches;

public:
    int num_threads;
    float **host_database_B, **host_database_A, **host_database_prob_Y;

public:
    int num_batches, batch_size;

public:
    database_reader(int,int,int);
    void read(int);
    void reorganize();
    void _free();
};


#endif //CUDA_CODE_SEARCH_DATABASE_READER_H
