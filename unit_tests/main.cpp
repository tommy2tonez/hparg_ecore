#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

// #include "map_test.h"
#include "kvfeed_test.h"
#include "keyfeed_test.h"
#include <type_traits>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include "cyclic_queue_test.h"

int main(){

    keyfeed_test::run();    
    kvfeed_test::run();
    pow2_cyclic_queue_test::run();
    // run_map_test();
}