#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

// #include "map_test.h"
#include "kvfeed_test.h"
#include "keyfeed_test.h"
#include <type_traits>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

int main(){

    keyfeed_test::run_feed_test();    
    kvfeed_test::run_kv_feed_test();
    // run_map_test();
}