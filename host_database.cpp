//
// Created by rm38 on 3/2/20.
//

#include "host_database.h"
#include "utils.h"
#include <cstdlib>

host_database::host_database(long batch_size, int dimension) {

    this->dimension = dimension;
    this->batch_size = batch_size;

    /*allocating the memory for each matrix */

    host_database_B = (double*)malloc(batch_size * dimension * sizeof(double)); //new double[batch_size * dimension];
    host_database_A = (double*)malloc(batch_size * sizeof(double)); //new double[dimension];
    host_database_prob_Y = (double*)malloc(batch_size * sizeof(double)); //new double[dimension];


    // ---------------checking host memory  for error..............................
    if(host_database_B == nullptr)
        mem_error("host_database_B", "vectmatmul", batch_size * dimension, "double");

    if(host_database_A == nullptr)
        mem_error("host_database_A", "vectmatmul", batch_size, "double");

    if(host_database_prob_Y == nullptr)
        mem_error("host_database_prob_Y", "vectmatmul", batch_size, "double");


}

void host_database::fill_database() {
    //--------------Initializing the input arrays..............
    fill_with_random_doubles(host_database_B, batch_size * dimension);
    fill_with_constant(host_database_A, batch_size, -0.5);
    fill_with_random_doubles(host_database_prob_Y, batch_size);
}


void host_database::_free() {
    //free host memory----------
    free(host_database_B);
    free(host_database_A);
    free(host_database_prob_Y);
}
