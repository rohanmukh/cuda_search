//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_SERIAL_CODE_H
#define CUDA_CODE_SEARCH_SERIAL_CODE_H


class serial_code {
    private:
        int matRowSize, matColSize, vlength, size;
        double *host_Mat, *host_Vect;
        double *cpu_ResVect;


    public:
    serial_code(int, int, double*, double*, int, int);

    void CPU_MatVect();
    double *get_result();
    void _free();
};


#endif //CUDA_CODE_SEARCH_SERIAL_CODE_H
