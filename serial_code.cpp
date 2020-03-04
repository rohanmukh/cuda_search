//
// Created by rm38 on 3/2/20.
//

#include "serial_code.h"
#include <cuda.h>
#include "utils.h"
#include <cmath>
#include <chrono>
#include <iostream>

serial_code::serial_code(long batch_size, int dimension, double *host_database_B,
                         double *host_database_A, double *host_database_probY, double *host_input_B, double* host_input_A) {
    this->batch_size = batch_size;
    this->dimension = dimension;
    this->host_database_B = host_database_B;
    this->host_database_A = host_database_A;
    this->host_input_B = host_input_B;
    this->host_input_A = host_input_A;
    this->host_database_probY = host_database_probY;
    this->cpu_ResVect = (double *) malloc(batch_size * sizeof(double));
    if (cpu_ResVect == nullptr)
        mem_error("cpu_ResVect", "vectmatmul", batch_size, "double");
}

/*sequential function for mat vect multiplication*/\
double serial_code::CPU_MatVectMult() {
    auto start = std::chrono::steady_clock::now();
    for (int k = 0; k < batch_size; k++) {
        int offset = k * dimension;
        cpu_ResVect[k] = 0.00;
        for (int i = 0; i < dimension; i++) {
            cpu_ResVect[k] += pow(host_input_B[i], 2) / (4 * host_input_A[0]); // additive ab1 1st item
            cpu_ResVect[k] +=
                    pow(host_database_B[offset + i], 2) / (4 * host_database_A[k]); // additive ab2 1st item
            cpu_ResVect[k] -= pow(host_database_B[offset + i] + host_input_B[i], 2) /
                              (4 * (host_database_A[k] + host_input_A[0])); // subtractive ab_star 1st item
        }
        cpu_ResVect[k] += 0.5 * dimension * log(-1 * (host_input_A[0]) / M_PI); // additive ab1 2nd item
        cpu_ResVect[k] += 0.5 * dimension * log(-1 * (host_database_A[k]) / M_PI); // additive ab2_2nd item
        cpu_ResVect[k] -= 0.5 * dimension * log(-1 * (host_database_A[k] + host_input_A[0]) /
                                                M_PI); // subtractive ab_star 2nd item
        cpu_ResVect[k] -= 0.5 * dimension * log(2 * M_PI); // subtractive cons
        cpu_ResVect[k] += host_database_probY[k];
    }
    auto stop = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    return 1.0e-9 * time;
}

double *serial_code::get_result() {
    return this->cpu_ResVect;
}


void serial_code::_free(){
    free(this->cpu_ResVect);
}

