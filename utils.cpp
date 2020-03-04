//
// Created by rm38 on 3/2/20.
//

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include "utils.h"
#define EPS 1.0e-15

/*mem error*/
void mem_error(const std::string& arrayname, const std::string& benchmark, long len, const std::string& type)
{
    printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %ld number of %s elements\n",arrayname.c_str(), benchmark.c_str(), len, type.c_str());
    exit(-1);
}

/*prints the result in screen*/
void print_on_screen(const std::string& program_name,float tsec,double gflops,long size, int flag)//flag=1 if gflops has been calculated else flag =0
{
    printf("\n---------------%s----------------\n",program_name.c_str());
    printf("\tSIZE\t TIME_SEC\t Gflops\n");
    if(flag==1)
        printf("\t%ld\t%f\t%lf\t",size,tsec,gflops);
    else
        printf("\t%ld\t%s\t%s\t",size,"---","---");
    printf("\n ----------------------------------------------------------------------\n");
}

/*calculate Gflops*/
double calculate_gflops(float &Tsec, long size)
{
    double gflops=(1.0e-9 * (( 2.0 * size )/Tsec));
    print_on_screen("MAT VECT MULTIPLICATION", Tsec, gflops, size, 1);
    return gflops;
}


/*Fill in the vector with double precision values */
void fill_with_random_doubles(double* vec, long size)
{
    for(int ind=0;ind<size;ind++)
        vec[ind]=drand48();
}

/*Fill in the vector with double precision values */
void fill_with_constant(double* vec, long size, double constant)
{
    for(int ind=0;ind<size;ind++)
        vec[ind]=constant;
}

/* function to calculate relative error*/
void relative_error(double* dRes, double* hRes, long size)
{
    double relativeError,errorNorm=0.0;
    int flag=0;
    int i;

    for( i = 0; i < size; ++i) {
        if (fabs(hRes[i]) > fabs(dRes[i]))
            relativeError = fabs((hRes[i] - dRes[i]) / hRes[i]);
        else
            relativeError = fabs((dRes[i] - hRes[i]) / dRes[i]);

        if (relativeError > EPS && relativeError != 0.0e+00 )
        {
            if(errorNorm < relativeError)
            {
                errorNorm = relativeError;
                flag=1;
            }
        }
    }
    std::cout << "===========================Relative Error==================================" << std::endl;
    if( flag == 1){
        printf("\n ----------------------------------------------------------------------\n");
        printf(" \n--------------------Results verification : Failed------------------------");
        printf(" \n------------------Considered machine precision : %e-------------------", EPS);
        printf(" \n--------------------Relative Error : %e------------------------------\n", errorNorm);
        printf("\n ----------------------------------------------------------------------\n");
    }
    else{
        printf("\n ----------------------------------------------------------------------");
        printf("\n ------------------Results verification : Success--------------------------");
        printf("\n ----------------------------------------------------------------------\n\n");

    }

}
