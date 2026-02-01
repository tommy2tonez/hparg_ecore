#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

#include "map_test.h"
#include "kvfeed_test.h"
#include "keyfeed_test.h"
#include <type_traits>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include "cyclic_queue_test.h"
#include "fsys_test.h"
#include "kernel_map_test.h"

//we have hopes guys
//I'll be back on the project when there is a seed investment 
//around 1-2 months
//client is asking us to build a virtual net (on top of the cloud infrastructure) to transfer data from A - B, essentially to not stress the infrastructure yet still crunching cuda + host operations 
//we built the socket fine, at least the memory and their effects are compromised and the results are there reliably if the capacity is under some number
//this is gonna be bigger than we dreamt of 

int main(){

    keyfeed_test::run();
    kvfeed_test::run();
    pow2_cyclic_queue_test::run();    
    fileio_test::run();
    map_test::run();
    kernel_map_test::run();
}