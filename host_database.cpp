//
// Created by rm38 on 3/2/20.
//

#include "host_database.h"
#include "utils.h"
#include <cstdlib>

host_database::host_database(long data_size, int dimension) {

    this->dimension = dimension;
    this->data_size = data_size;

    /*allocating the memory for each matrix */
    host_database_B = (double*)malloc(data_size * dimension * sizeof(double)); //new double[data_size * dimension];
    host_database_A = (double*)malloc(data_size * sizeof(double)); //new double[dimension];
    host_database_prob_Y = (double*)malloc(data_size * sizeof(double)); //new double[dimension];

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
    fill_with_random_doubles(host_database_B, data_size * dimension);
    fill_with_constant(host_database_A, data_size, -0.5);
    fill_with_random_doubles(host_database_prob_Y, data_size);
}

void host_database::_free() {
    //free host memory----------
    free(host_database_B);
    free(host_database_A);
    free(host_database_prob_Y);
}
