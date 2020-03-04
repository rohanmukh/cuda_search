//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_HOST_OPS_H
#define CUDA_CODE_SEARCH_HOST_OPS_H


class host_ops {
    long batch_size;
    int dimension;

public:
    double *host_database_B, *host_database_A, *host_input_B, *host_input_A, *host_ResVect;

public:
    host_ops(long, int);
    void fill_database();
    void fill_input_query();
    void _free();
    double* get_host_vector();
    double* get_host_matrix();
    double* get_result_vector();

};


#endif //CUDA_CODE_SEARCH_HOST_OPS_H
