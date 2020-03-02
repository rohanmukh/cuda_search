//
// Created by rm38 on 3/2/20.
//

#include "utils.h"
#include <cstdio>
#include <cstdlib>

/*mem error*/
void mem_error(char *arrayname, char *benchmark, int len, char *type)
{
    printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
    exit(-1);
}

/*prints the result in screen*/
void print_on_screen(char * program_name,float tsec,double gflops,int size,int flag)//flag=1 if gflops has been calculated else flag =0
{
    printf("\n---------------%s----------------\n",program_name);
    printf("\tSIZE\t TIME_SEC\t Gflops\n");
    if(flag==1)
        printf("\t%d\t%f\t%lf\t",size,tsec,gflops);
    else
        printf("\t%d\t%lf\t%lf\t",size,"---","---");

}