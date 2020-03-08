//
// Created by rm38 on 3/2/20.
//

#ifndef CUDA_CODE_SEARCH_CPU_MANAGER_H
#define CUDA_CODE_SEARCH_CPU_MANAGER_H


class cpu_manager {
    private:
        long batch_size, num_batches;
        int dimension;
        float **host_database_B, **host_database_A, **host_database_probY;
        float *host_input_B, *host_input_A;
        float *result_vector;


    public:
    cpu_manager(long, long, int, float**, float**, float**);
    void add_query(float*, float*);
    void search();
    float *get_result();
    void _free();
};


#endif //CUDA_CODE_SEARCH_CPU_MANAGER_H
