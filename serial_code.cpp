//
// Created by rm38 on 3/2/20.
//

#include "serial_code.h"
#include <cuda.h>
#include "utils.h"


serial_code::serial_code(int matRowSize, int matColSize, double *host_Mat,
                         double *host_Vect, int vlength, int size) {
    this->matRowSize = matRowSize;
    this->matColSize = matColSize;
    this->host_Mat = host_Mat;
    this->host_Vect = host_Vect;
    this->vlength = vlength;
    this->size = size;
}

/*sequential function for mat vect multiplication*/\
void serial_code::CPU_MatVectMult() {
    cpu_ResVect = (double *) malloc(matRowSize * sizeof(double));
    if (cpu_ResVect == NULL)
        mem_error("cpu_ResVect", "vectmatmul", size, "double");

    int i, j;
    for (i = 0; i < matRowSize; i++) {
        cpu_ResVect[i] = 0;
        for (j = 0; j < matColSize; j++)
            cpu_ResVect[i] += host_Mat[i * vlength + j] * host_Vect[j];
    }
}

double *serial_code::get_result() {
    return this->cpu_ResVect;
}


void serial_code::_free(){
    free(this->cpu_ResVect);
}

