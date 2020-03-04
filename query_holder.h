//
// Created by rm38 on 3/4/20.
//

#ifndef CUDA_CODE_SEARCH_QUERY_HOLDER_H
#define CUDA_CODE_SEARCH_QUERY_HOLDER_H


class query_holder {
    int dimension;

public:

    double *host_input_B, *host_input_A;

public:
    query_holder(int);

    void fill_input_query();
    void _free();

};


#endif //CUDA_CODE_SEARCH_QUERY_HOLDER_H
