//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_UTILS_H
#define CUDA_CODE_SEARCH_UTILS_H


/*mem error*/
void mem_error(char *arrayname, char *benchmark, int len, char *type);
void print_on_screen(char * program_name,float tsec,double gflops,int size,int flag)
;
#endif //CUDA_CODE_SEARCH_UTILS_H
