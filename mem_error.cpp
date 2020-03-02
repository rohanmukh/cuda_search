//
// Created by rm38 on 3/2/20.
//

#include "mem_error.h"
#include <cstdio>
#include <cstdlib>

/*mem error*/
void mem_error(char *arrayname, char *benchmark, int len, char *type)
{
    printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname, benchmark, len, type);
    exit(-1);
}