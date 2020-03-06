//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_HOST_DATABASE_H
#define CUDA_CODE_SEARCH_HOST_DATABASE_H


class host_database {
    long data_size;
    int dimension;

public:
    int num_batches; long batch_size;

public:
    double **host_database_B, **host_database_A, **host_database_prob_Y;

public:
    host_database(long, int);
    void allocate();
    void fill_database();
    void _free();

};


#endif //CUDA_CODE_SEARCH_HOST_DATABASE_H
