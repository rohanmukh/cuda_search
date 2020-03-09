//
// Created by ROHAN MUKHERJEE on 3/9/20.
//

#ifndef CUDA_SEARCH_SERVER_H
#define CUDA_SEARCH_SERVER_H
#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#define PORT 8080

class server {
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};
    char *hello = "Hello from server";

public:
    server();

    void unblock_and_run();
};


#endif //CUDA_SEARCH_SERVER_H
