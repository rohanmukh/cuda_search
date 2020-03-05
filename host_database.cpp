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

    host_database_B = (double**)malloc(num_batches * sizeof(double*)); //new double[data_size * dimension];
    host_database_A = (double**)malloc(num_batches * sizeof(double*)); //new double[dimension];
    host_database_prob_Y = (double**)malloc(num_batches * sizeof(double*)); //new double[dimension];

    /*allocating the memory for each matrix */
    for (int i=0;i<this->num_batches; i++){
        host_database_B[i] = (double*)malloc(batch_size * dimension * sizeof(double)); //new double[data_size * dimension];
        host_database_A[i] = (double*)malloc(batch_size * sizeof(double)); //new double[dimension];
        host_database_prob_Y[i] = (double*)malloc(batch_size * sizeof(double)); //new double[dimension];
    }

    // ---------------checking host memory  for error..............................
    if(host_database_B == nullptr)
        mem_error("host_database_B", "vectmatmul", data_size * dimension, "double");

    if(host_database_A == nullptr)
        mem_error("host_database_A", "vectmatmul", data_size, "double");

    if(host_database_prob_Y == nullptr)
        mem_error("host_database_prob_Y", "vectmatmul", data_size, "double");

}

void host_database::fill_database() {
    //--------------Initializing the input arrays..............
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
