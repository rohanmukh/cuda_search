//
// Created by ROHAN MUKHERJEE on 3/9/20.
//

#ifndef CUDA_SEARCH_CODEC_H
#define CUDA_SEARCH_CODEC_H
#include "gpu_manager.h"
#include "cpu_manager.h"
#include "host_database.h"
#include "database_reader.h"

#include <fstream>
#include <iostream>
#include<json/writer.h>

class codec {
//    database_reader *host_db;
    host_database *host_db;
    gpu_manager *gpu_user;
    cpu_manager *cpu_user;
    int data_size_per_batch, dimension;
public:
    codec(int data_size_per_batch, int dimension, int num_jsons);
    void set_gpu_user(int);
    void search(float*, float*);
    void verify(float*, float*);
    void shrink_data(int max_jsons, int num_devices);
    void _free();
    void dump_json(std::vector<std::tuple<int, int, float>> );
};


#endif //CUDA_SEARCH_CODEC_H
