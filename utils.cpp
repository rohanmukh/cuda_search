//
// Created by rm38 on 3/2/20.
//

#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define EPS 1.0e-15


/*mem error*/
void mem_error(const std::string& arrayname, const std::string& benchmark, long len, const std::string& type)
{
    printf("\nMemory not sufficient to allocate for array %s\n\tBenchmark : %s  \n\tMemory requested = %d number of %s elements\n",arrayname.c_str(), benchmark.c_str(), len, type.c_str());
    exit(-1);
}

/*prints the result in screen*/
void print_on_screen(const std::string& program_name,float tsec,double gflops,int size, int flag)//flag=1 if gflops has been calculated else flag =0
{
    printf("\n---------------%s----------------\n",program_name.c_str());
    printf("\tSIZE\t TIME_SEC\t Gflops\n");
    if(flag==1)
        printf("\t%d\t%f\t%lf\t",size,tsec,gflops);
    else
        printf("\t%d\t%s\t%s\t",size,"---","---");

}

/*calculate Gflops*/
double calculate_gflops(float &Tsec, int size)
{
    double gflops=(1.0e-9 * (( 2.0 * size )/Tsec));
    return gflops;
}


/*Fill in the vector with double precision values */
void fill_with_random_doubles(double* vec, long size)
{
    int ind;
    for(ind=0;ind<size;ind++)
        vec[ind]=drand48();
}

/* function to calculate relative error*/
void relative_error(double* dRes, double* hRes, int size)
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
