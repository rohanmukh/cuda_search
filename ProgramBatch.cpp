//
// Created by rm38 on 3/5/20.
//

#include "ProgramBatch.h"

ProgramBatch::ProgramBatch(int data_size_per_batch, int dimension){
    this->dimension = dimension;
    this->num_programs = 0;
    json_database_B = (float*)malloc(data_size_per_batch * dimension * sizeof(float));
    json_database_A = (float*)malloc(data_size_per_batch * sizeof(float));
    json_database_prob_Y = (float*)malloc(data_size_per_batch * sizeof(float)); 
}

void ProgramBatch::read_single_database_json(const std::string file_name){
    Json::Value single_db;
    std::ifstream cfgfile(file_name);
    cfgfile >> single_db;

    for (Json::Value &program: single_db["programs"]){
        auto temp = new Program(this->dimension, program, json_database_A+num_programs, json_database_B + num_programs*dimension, json_database_prob_Y+num_programs );
        list_of_programs.push_back(temp);
        num_programs++;
    }
}


Program* ProgramBatch::get_program(int id){
    return list_of_programs.at(id);
}
