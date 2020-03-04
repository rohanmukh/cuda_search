//
// Created by rm38 on 3/2/20.
//

#include "serial_code.h"
#include <cuda.h>
#include "utils.h"
#include <cmath>

serial_code::serial_code(long batch_size, int dimension, double *host_database_B,
                         double *host_database_A, double *host_input_B, double* host_input_A) {
    this->batch_size = batch_size;
    this->dimension = dimension;
    this->host_database_B = host_database_B;
    this->host_database_A = host_database_A;
    this->host_input_B = host_input_B;
    this->host_input_A = host_input_A;
}

/*sequential function for mat vect multiplication*/\
void serial_code::CPU_MatVectMult() {
    cpu_ResVect = (double *) malloc(batch_size * sizeof(double));
    if (cpu_ResVect == NULL)
        mem_error("cpu_ResVect", "vectmatmul", batch_size, "double");

    for (int k = 0; k < batch_size; k++) {
        int offset = k * dimension;
        cpu_ResVect[k] = 0.00;
        for (int i = 0; i < dimension; i++) {
            cpu_ResVect[k] += pow(host_input_B[i], 2) / (4 * host_input_A[0]); // additive ab1 1st item
            cpu_ResVect[k] +=
                    pow(host_database_B[offset + i], 2) / (4 * host_database_A[offset]); // additive ab2 1st item
            cpu_ResVect[k] -= pow(host_database_B[offset + i] + host_input_B[i], 2) /
                              (4 * (host_database_A[offset] + host_input_A[0])); // subtractive ab_star 1st item
        }
        cpu_ResVect[k] += 0.5 * dimension * log(-1 * (host_input_A[0]) / M_PI); // additive ab1 2nd item
        cpu_ResVect[k] += 0.5 * dimension * log(-1 * (host_database_A[offset]) / M_PI); // additive ab2_2nd item
        cpu_ResVect[k] -= 0.5 * dimension * log(-1 * (host_database_A[offset] + host_input_A[0]) /
                                                M_PI); // subtractive ab_star 2nd item
        cpu_ResVect[k] -= 0.5 * dimension * log(2 * M_PI); // subtractive cons
        cpu_ResVect[k] += 0.; // TODO ProbY
    }
}

double *serial_code::get_result() {
    return this->cpu_ResVect;
}


void serial_code::_free(){
    free(this->cpu_ResVect);
}

