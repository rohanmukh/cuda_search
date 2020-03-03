//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_HOST_OPS_H
#define CUDA_CODE_SEARCH_HOST_OPS_H


class host_ops {
    int batch_size, dimension;

public:
    double *host_Mat, *host_Vect, *host_ResVect;

public:
    host_ops(int, int);
    void fill_with_random_data();
    void _free();
    double* get_host_vector();
    double* get_host_matrix();
    double* get_result_vector();

};


#endif //CUDA_CODE_SEARCH_HOST_OPS_H
