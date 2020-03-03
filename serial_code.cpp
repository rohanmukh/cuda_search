//
// Created by rm38 on 3/2/20.
//

#include "serial_code.h"
#include <cuda.h>
#include "utils.h"


serial_code::serial_code(long batch_size, int dimension, double *host_Mat,
                         double *host_Vect) {
    this->batch_size = batch_size;
    this->dimension = dimension;
    this->host_Mat = host_Mat;
    this->host_Vect = host_Vect;
}

/*sequential function for mat vect multiplication*/\
void serial_code::CPU_MatVectMult() {
    cpu_ResVect = (double *) malloc(batch_size * sizeof(double));
    if (cpu_ResVect == NULL)
        mem_error("cpu_ResVect", "vectmatmul", batch_size, "double");

    int i, j;
    for (i = 0; i < batch_size; i++) {
        cpu_ResVect[i] = 0;
        for (j = 0; j < dimension; j++)
            cpu_ResVect[i] += host_Mat[i * dimension + j] * host_Vect[j];
    }
}

double *serial_code::get_result() {
    return this->cpu_ResVect;
}


void serial_code::_free(){
    free(this->cpu_ResVect);
}

