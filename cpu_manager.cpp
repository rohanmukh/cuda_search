//
// Created by rm38 on 3/2/20.
//

#include "cpu_manager.h"
#include <cuda.h>
#include "utils.h"
#include <cmath>
#include <chrono>
#include <iostream>

cpu_manager::cpu_manager( long num_batches, long batch_size, int dimension, float **host_database_B,
                         float **host_database_A, float **host_database_probY) {
    this->batch_size = batch_size;
    this->dimension = dimension;
    this->num_batches = num_batches;

    this->host_database_B = host_database_B;
    this->host_database_A = host_database_A;
    this->host_database_probY = host_database_probY;


    this->result_vector = (float *) malloc(num_batches * batch_size *  sizeof(float));
    if (result_vector == nullptr)
        mem_error("result_vector", "vectmatmul", batch_size, "float");
}


void cpu_manager::add_query( float *_host_input_B, float* _host_input_A) {
    this->host_input_B = _host_input_B;
    this->host_input_A = _host_input_A;
}


/*sequential function for mat vect multiplication*/\
void cpu_manager::search() {
    std::cout << "===========================CPU Calculation==================================" << std::endl;
    auto start = std::chrono::steady_clock::now();
    for (int batch_id=0; batch_id<num_batches; batch_id++){
        for (int k = 0; k < batch_size; k++) {
            int offset = k * dimension;
            result_vector[batch_id*batch_size + k] = 0.00;
            for (int i = 0; i < dimension; i++) {
                result_vector[batch_id*batch_size + k] += pow(host_input_B[i], 2) / (4 * host_input_A[0]); // additive ab1 1st item
                result_vector[batch_id*batch_size + k] +=
                        pow(host_database_B[batch_id][offset + i], 2) / (4 * host_database_A[batch_id][k]); // additive ab2 1st item
                result_vector[batch_id*batch_size + k] -= pow(host_database_B[batch_id][offset + i] + host_input_B[i], 2) /
                                    (4 * (host_database_A[batch_id][k] + host_input_A[0])); // subtractive ab_star 1st item
            }
            result_vector[batch_id*batch_size + k] += 0.5 * dimension * log(-1 * (host_input_A[0]) / M_PI); // additive ab1 2nd item
            result_vector[batch_id*batch_size + k] += 0.5 * dimension * log(-1 * (host_database_A[batch_id][k]) / M_PI); // additive ab2_2nd item
            result_vector[batch_id*batch_size + k] -= 0.5 * dimension * log(-1 * (host_database_A[batch_id][k] + host_input_A[0]) /
                                                      M_PI); // subtractive ab_star 2nd item
            result_vector[batch_id*batch_size + k] -= 0.5 * dimension * log(2 * M_PI); // subtractive cons
            result_vector[batch_id*batch_size + k] += host_database_probY[batch_id][k];
        }
    }
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1e-9;
    double gflops = calculate_gflops(time, batch_size * dimension);
    print_on_screen("CPU Search", time, gflops, batch_size * dimension, 1);
}

float *cpu_manager::get_result() {
    return this->result_vector;
}


void cpu_manager::_free(){
    free(this->result_vector);
}

