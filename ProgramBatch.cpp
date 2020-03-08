//
// Created by rm38 on 3/5/20.
//

#include "ProgramBatch.h"

ProgramBatch::ProgramBatch(int data_size, int dimension){
    this->dimension = dimension;
    this->num_programs = 0;
    json_database_B = (float*)malloc(data_size * dimension * sizeof(float)); //new float[data_size * dimension];
    json_database_A = (float*)malloc(data_size * sizeof(float)); //new float[dimension];
    json_database_prob_Y = (float*)malloc(data_size * sizeof(float)); //new float[dimension];
}

void ProgramBatch::read_single_database_json(const std::string file_name){
    Json::Value single_db;
    std::ifstream cfgfile(file_name);
    cfgfile >> single_db;

    std::vector<Program*> list_of_programs;
    for (Json::Value &program: single_db["programs"]){
        auto temp = new Program(this->dimension, program, json_database_A+num_programs, json_database_B + num_programs*dimension, json_database_prob_Y+num_programs );
        list_of_programs.push_back(temp);
        num_programs++;
    }
}
