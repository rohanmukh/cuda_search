//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_SERIAL_CODE_H
#define CUDA_CODE_SEARCH_SERIAL_CODE_H


class serial_code {
    private:
        long batch_size;
        int dimension;
        double *host_database_B, *host_database_A, *host_database_probY;
        double *host_input_B, *host_input_A;
        double *cpu_ResVect;


    public:
    serial_code(long, int, double*, double*, double*, double*, double*);

    void CPU_MatVectMult();
    double *get_result();
    void _free();
};


#endif //CUDA_CODE_SEARCH_SERIAL_CODE_H
