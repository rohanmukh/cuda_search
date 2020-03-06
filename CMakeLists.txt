cmake_minimum_required(VERSION 3.15)
project(cuda_code_search)

set(CMAKE_CXX_STANDARD 14)

add_executable(cuda_code_search main.cu cpu_manager.cpp cpu_manager.h utils.cpp utils.h cuda_utils.cu cuda_utils.h single_gpu_manager.cu single_gpu_manager.h host_database.cpp host_database.h gpu_manager.cpp gpu_manager.h query_holder.cpp query_holder.h json_reader/json_reader.cpp json_reader/Program.cpp json_reader/Program.h json_reader/ProgramBatch.cpp json_reader/ProgramBatch.h)