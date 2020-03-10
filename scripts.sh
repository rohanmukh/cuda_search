nvcc -O3 -std=c++11 -o codec_server  -L/usr/local/lib -I. -I/usr/include/jsoncpp database_reader.cpp query_holder.cpp utils.cpp gpu_manager.cpp cuda_utils.cu host_database.cpp ProgramBatch.cpp Program.cpp single_gpu_manager.cu cpu_manager.cpp main.cpp server.cpp codec.cpp  -ljsoncpp -Xcompiler -fopenmp


nvcc -O3 -std=c++11 -o codec_server_loopy  -L/usr/local/lib -I. -I/usr/include/jsoncpp database_reader.cpp query_holder.cpp utils.cpp gpu_manager.cpp cuda_utils.cu host_database.cpp ProgramBatch.cpp Program.cpp single_gpu_manager.cu cpu_manager.cpp main_loop_device.cpp server.cpp codec.cpp  -ljsoncpp -Xcompiler -fopenmp


g++ -O3 -std=c++11 -o demo_client client.cpp
