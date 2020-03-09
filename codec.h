//
// Created by ROHAN MUKHERJEE on 3/9/20.
//

#ifndef CUDA_SEARCH_CODEC_H
#define CUDA_SEARCH_CODEC_H
#include "gpu_manager.h"
#include "cpu_manager.h"
//#include "host_database.h"
#include "database_reader.h"

class codec {
    database_reader *host_db;
    gpu_manager *gpu_user;
    cpu_manager *cpu_user;
    int data_size_per_batch;
public:
    codec(int data_size_per_batch, int dimension, int num_jsons);
    void search(float*, float*);
    void verify(float*, float*);
    void _free();

};


#endif //CUDA_SEARCH_CODEC_H
