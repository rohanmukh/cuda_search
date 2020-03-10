//
// Created by rm38 on 3/5/20.
//

#ifndef CUDA_CODE_SEARCH_PROGRAM_H
#define CUDA_CODE_SEARCH_PROGRAM_H

#include <string>
#include <iostream>
#include <json/value.h>

class Program {
    std::string body;
    int dimension;
//    float a2; float *b2_array;

public:
    Program(int dimension, Json::Value &program_json, float*, float*, float*);
    Program(int dimension); //fake program
    std::string get_body();
};


#endif //CUDA_CODE_SEARCH_PROGRAM_H
