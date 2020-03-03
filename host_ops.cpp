//
// Created by rm38 on 3/2/20.
//

#include "host_ops.h"
#include "utils.h"
#include <cstdlib>

host_ops::host_ops(long batch_size, int dimension) {

    this->dimension = dimension;
    this->batch_size = batch_size;

    /*allocating the memory for each matrix */
    host_Mat =new double[batch_size * dimension];
    host_Vect = new double[dimension];
    host_ResVect = new double[batch_size];

    // ---------------checking host memory  for error..............................
    if(host_Mat==NULL)
        mem_error("host_Mat", "vectmatmul", batch_size * dimension, "double");

    if(host_Vect==NULL)
        mem_error("host_Vect", "vectmatmul", batch_size, "double");

    if(host_ResVect==NULL)
        mem_error("host_ResVect", "vectmatmul", batch_size, "double");

}

void host_ops::fill_with_random_data() {
    //--------------Initializing the input arrays..............
    fill_with_random_doubles(host_Mat, batch_size * dimension);
    fill_with_random_doubles(host_Vect, dimension);
}

double *host_ops::get_result_vector(){
    return host_ResVect;
}

double *host_ops::get_host_vector() {
    return host_Vect;
}

double *host_ops::get_host_matrix() {
    return host_Mat;
}

void host_ops::_free() {
    //free host memory----------
    free(host_Mat);
    free(host_Vect);
    free(host_ResVect);
}
