cmake_minimum_required(VERSION 3.5)
project(cuda_search)

set(CMAKE_CXX_STANDARD 14)

add_executable(cuda_search main.cpp cpu_manager.cpp cpu_manager.h utils.cpp utils.h cuda_utils.cu cuda_utils.h single_gpu_manager.cu single_gpu_manager.h host_database.cpp host_database.h gpu_manager.cpp gpu_manager.h query_holder.cpp query_holder.h json_reader/json_reader.cpp Program.cpp Program.h ProgramBatch.cpp ProgramBatch.h database_reader.cpp database_reader.h codec.cpp codec.h server.cpp server.h client.cpp client.h)
target_link_libraries(${PROJECT_NAME} jsoncpp)
