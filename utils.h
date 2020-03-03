//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_UTILS_H
#define CUDA_CODE_SEARCH_UTILS_H
#include <string>

/*mem error*/
void mem_error(std::string arrayname, std::string benchmark, int len, std::string type);
void print_on_screen(std::string program_name,float tsec,double gflops,int size,int flag);
void fill_with_random_doubles(double* vec, int size);
void relative_error(double* dRes, double* hRes, int size);
double calculate_gflops(float &Tsec, int size);

#endif //CUDA_CODE_SEARCH_UTILS_H
