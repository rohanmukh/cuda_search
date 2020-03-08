//
// Created by rm38 on 3/4/20.
//

#include "query_holder.h"
#include "utils.h"

void query_holder::fill_input_query() {
    //--------------Initializing the input arrays..............
    fill_with_random_doubles(host_query_B, dimension);
    fill_with_constant(host_query_A, 1, -0.5);
}

void query_holder::_free() {
    //free host memory----------
    free(host_query_B);
    free(host_query_A);
}

query_holder::query_holder(int dimension) {

    this->dimension = dimension;

    /*allocating the memory for each matrix */

    host_query_B = (float*)malloc(dimension * sizeof(float)); //new float[device_num_batches * dimension];
    host_query_A = (float*)malloc(sizeof(float)); //new float[dimension];


    // ---------------checking host memory  for error..............................

    if(host_query_B == nullptr)
        mem_error("host_query_B", "vectmatmul", dimension, "float");

    if(host_query_A == nullptr)
        mem_error("host_query_A", "vectmatmul", dimension, "float");

}