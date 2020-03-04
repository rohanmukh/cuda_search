//
// Created by rm38 on 3/4/20.
//

#include "query_holder.h"
#include "utils.h"

void query_holder::fill_input_query() {
    //--------------Initializing the input arrays..............
    fill_with_random_doubles(host_input_B, dimension);
    fill_with_constant(host_input_A, 1, -0.5);
}

void query_holder::_free() {
    //free host memory----------
    free(host_input_B);
    free(host_input_A);
}

query_holder::query_holder(int dimension) {

    this->dimension = dimension;

    /*allocating the memory for each matrix */

    host_input_B = (double*)malloc( dimension * sizeof(double)); //new double[batch_size * dimension];
    host_input_A = (double*)malloc(sizeof(double)); //new double[dimension];


    // ---------------checking host memory  for error..............................

    if(host_input_B == nullptr)
        mem_error("host_input_B", "vectmatmul", dimension, "double");

    if(host_input_A == nullptr)
        mem_error("host_input_A", "vectmatmul", dimension, "double");

}