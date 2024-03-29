//
// Created by rm38 on 3/5/20.
//

#ifndef CUDA_CODE_SEARCH_PROGRAMBATCH_H
#define CUDA_CODE_SEARCH_PROGRAMBATCH_H

#include <string>
#include <json/value.h>
#include <jsoncpp/json/json.h>
#include <fstream>

#include "Program.h"

class ProgramBatch {
public:
    int dimension;
    int num_programs;
    std::vector<Program*> list_of_programs;

public:
    float *json_database_B, *json_database_A, *json_database_prob_Y;

public:
    ProgramBatch(int, int);

    void read_single_database_json(const std::string);
    Program* get_program(int);

};


#endif //CUDA_CODE_SEARCH_PROGRAMBATCH_H
