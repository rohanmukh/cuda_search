//
// Created by ROHAN MUKHERJEE on 3/9/20.
//

#ifndef CUDA_SEARCH_CLIENT_H
#define CUDA_SEARCH_CLIENT_H
// Client side C/C++ program to demonstrate Socket programming
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>
#define PORT 8080

class client {
    int sock = 0, valread;
    struct sockaddr_in serv_addr;
    char buffer[1024] = {0};

public:
    client();
    int send_query();
};


#endif //CUDA_SEARCH_CLIENT_H
