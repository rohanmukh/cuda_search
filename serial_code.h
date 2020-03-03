//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_SERIAL_CODE_H
#define CUDA_CODE_SEARCH_SERIAL_CODE_H


class serial_code {
    private:
        int batch_size, dimension, size;
        double *host_Mat, *host_Vect;
        double *cpu_ResVect;


    public:
    serial_code(int, int, double*, double*, int);

    void CPU_MatVectMult();
    double *get_result();
    void _free();
};


#endif //CUDA_CODE_SEARCH_SERIAL_CODE_H
