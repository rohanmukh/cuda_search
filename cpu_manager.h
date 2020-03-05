//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_CPU_MANAGER_H
#define CUDA_CODE_SEARCH_CPU_MANAGER_H


class cpu_manager {
    private:
        long batch_size, num_batches;
        int dimension;
        double **host_database_B, **host_database_A, **host_database_probY;
        double *host_input_B, *host_input_A;
        double *result_vector;


    public:
    cpu_manager(long, long, int, double**, double**, double**);
    void add_query(double*, double*);
    void search();
    double *get_result();
    void _free();
};


#endif //CUDA_CODE_SEARCH_CPU_MANAGER_H
