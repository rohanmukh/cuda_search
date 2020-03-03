//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_HOST_OPS_H
#define CUDA_CODE_SEARCH_HOST_OPS_H


class host_ops {
    long batch_size;
    int dimension;

public:
    double *host_Mat, *host_Vect, *host_ResVect;

public:
    host_ops(long, int);
    void fill_with_random_data();
    void _free();
    double* get_host_vector();
    double* get_host_matrix();
    double* get_result_vector();

};


#endif //CUDA_CODE_SEARCH_HOST_OPS_H
