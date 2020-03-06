//
// Created by rm38 on 3/4/20.
//

#ifndef CUDA_CODE_SEARCH_QUERY_HOLDER_H
#define CUDA_CODE_SEARCH_QUERY_HOLDER_H


class query_holder {
    int dimension;

public:

    float *host_query_B, *host_query_A;

public:
    explicit query_holder(int);

    void fill_input_query();
    void _free();

};


#endif //CUDA_CODE_SEARCH_QUERY_HOLDER_H
