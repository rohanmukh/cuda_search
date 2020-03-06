//
// Created by rm38 on 3/2/20.
//

#include "host_database.h"
#include "utils.h"
#include <cstdlib>
#include <iostream>

host_database::host_database(long data_size, int dimension) {

    this->dimension = dimension;
    this->data_size = data_size;

    this->num_batches = 64;
    this->batch_size = this->data_size/this->num_batches;

    host_database_B = (float**)malloc(num_batches * sizeof(float*)); //new float[data_size * dimension];
    host_database_A = (float**)malloc(num_batches * sizeof(float*)); //new float[dimension];
    host_database_prob_Y = (float**)malloc(num_batches * sizeof(float*)); //new float[dimension];

}


void host_database::allocate() {
    /*allocating the memory for each matrix */
    for (int i=0;i<this->num_batches; i++){
        host_database_B[i] = (float*)malloc(batch_size * dimension * sizeof(float)); //new float[data_size * dimension];
        host_database_A[i] = (float*)malloc(batch_size * sizeof(float)); //new float[dimension];
        host_database_prob_Y[i] = (float*)malloc(batch_size * sizeof(float)); //new float[dimension];
    }

    // ---------------checking host memory  for error..............................
    if(host_database_B == nullptr)
        mem_error("host_database_B", "vectmatmul", data_size * dimension, "float");

    if(host_database_A == nullptr)
        mem_error("host_database_A", "vectmatmul", data_size, "float");

    if(host_database_prob_Y == nullptr)
        mem_error("host_database_prob_Y", "vectmatmul", data_size, "float");

}


void host_database::fill_database() {
    //--------------Initializing the input arrays..............
    allocate();
    #pragma omp parallel for
    for (int i=0;i<this->num_batches; i++) {
        fill_with_random_doubles(host_database_B[i], batch_size * dimension);
        fill_with_constant(host_database_A[i], batch_size, -0.5);
        fill_with_random_doubles(host_database_prob_Y[i], batch_size);
    }
}

void host_database::_free() {
    //free host memory----------
    for (int i=0;i<this->num_batches; i++) {
        free(host_database_B[i]);
        free(host_database_A[i]);
        free(host_database_prob_Y[i]);
    }
    free(host_database_B);
    free(host_database_A);
    free(host_database_prob_Y);
}
