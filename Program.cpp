//
// Created by rm38 on 3/5/20.
//

#include "Program.h"


Program::Program(int dimension, Json::Value &program_json, float* A, float* B_vec, float* PY) {
    this->body = program_json["body"].asString();
    A[0] = std::stof(program_json["a2"].asString());
    PY[0] = std::stof(program_json["ProbY"].asString());
    
    int d = 0;
    for (auto b: program_json["b2"])
        B_vec[d++] = std::stof(b.asString());
}


Program::Program(int dimension){ //, Json::Value &program_json, float* A, float* B_vec, float* PY) {
    this->body = "__fake__" ;
    A[0] = -0.5;
    PY[0] = -100.;

    for (int d=0;d<dimension;d++)
        B_vec[d] = -100.;
}

std::string Program::get_body(){
    return body;
}
