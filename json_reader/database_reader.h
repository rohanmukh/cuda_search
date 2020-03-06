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


public:
    database_reader(int,int,int);
    void read(int);

};


#endif //CUDA_CODE_SEARCH_DATABASE_READER_H
