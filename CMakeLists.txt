cmake_minimum_required(VERSION 3.15)
project(cuda_code_search)

set(CMAKE_CXX_STANDARD 14)

add_executable(cuda_code_search main.cu serial_code.cpp serial_code.h utils.cpp utils.h cuda_utils.cpp cuda_utils.h)