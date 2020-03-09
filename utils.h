//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_UTILS_H
#define CUDA_CODE_SEARCH_UTILS_H
#include <string>

/*mem error*/
void mem_error(const std::string& arrayname, const std::string& benchmark, long len, const std::string& type);
void print_on_screen(const std::string& program_name,double tsec,double gflops,long size,int flag);
void fill_with_random_doubles(float* vec, long size);
void fill_with_constant(float* vec, long size, float constant);
void relative_error(float* dRes, float* hRes, long size);
double calculate_gflops(double &Tsec, long size);

#endif //CUDA_CODE_SEARCH_UTILS_H
