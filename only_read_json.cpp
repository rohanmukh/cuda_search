//
// Created by rm38 on 3/5/20.
//

#include "database_reader.h"
#define DIMENSION 256
#define DATA_SIZE 1000000
# define NUM_JSONS 1
#define NUM_THREADS 32

int main()
{
    auto* db_read = new database_reader(NUM_THREADS, DATA_SIZE, DIMENSION);
    db_read->read(NUM_JSONS);
    db_read->reorganize();

}

