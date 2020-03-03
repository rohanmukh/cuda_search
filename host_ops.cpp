//
// Created by rm38 on 3/2/20.
//

#include "host_ops.h"
#include "utils.h"
#include <cstdlib>

host_ops::host_ops(int matRowSize, int matColSize, int vlength) {

    this->matColSize = matColSize;
    this->matRowSize = matRowSize;
    this->vlength = vlength;

    /*allocating the memory for each matrix */
    host_Mat =new double[matRowSize*matColSize];
    host_Vect = new double[vlength];
    host_ResVect = new double[matRowSize];


    // ---------------checking host memory  for error..............................
    if(host_Mat==NULL)
        mem_error("host_Mat","vectmatmul",matRowSize*matColSize,"double");

    if(host_Vect==NULL)
        mem_error("host_Vect","vectmatmul",vlength,"double");

    if(host_ResVect==NULL)
        mem_error("host_ResVect","vectmatmul",matRowSize,"double");

}

void host_ops::fill_with_random_data() {
    //--------------Initializing the input arrays..............
    fill_with_random_doubles(host_Mat, matRowSize * matColSize);
    fill_with_random_doubles(host_Vect, vlength);
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
