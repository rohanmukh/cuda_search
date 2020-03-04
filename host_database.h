//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_HOST_DATABASE_H
#define CUDA_CODE_SEARCH_HOST_DATABASE_H


class host_database {
    long batch_size;
    int dimension;

public:
    double *host_database_B, *host_database_A, *host_database_prob_Y;
    double *host_ResVect;

public:
    host_database(long, int);
    void fill_database();
    void _free();
    double* get_host_vector();
    double* get_host_matrix();
    double* get_result_vector();

};


#endif //CUDA_CODE_SEARCH_HOST_DATABASE_H
