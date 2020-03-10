//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_HOST_DATABASE_H
#define CUDA_CODE_SEARCH_HOST_DATABASE_H
#include "Program.h"

class host_database {
    long data_size;
    int dimension;

public:
    int num_batches; int batch_size;

public:
    float **host_database_B, **host_database_A, **host_database_prob_Y;

public:
    host_database(int, int, int);
    void allocate();
    void fill_database();
    void _free();
    Program* get_program(int,int);

};


#endif //CUDA_CODE_SEARCH_HOST_DATABASE_H
