#define DEBUG_MODE_FLAG true

#include "stdx.h"

int main(){

    //alright - let's fix undefined behavior today
    //assume we live in a perfect world - such world is __attribute(flattent__ )int (main)
    //then the memory_fence(std::memory_order_seq_cst) transaction works perfectly - no issues

    //but we are not living in a perfect world
    //such that the mem_fence is consume - that it is effective for only inlinable functions - 
    //other functions that are not-inlinable - compiled separately or (no_inline) - are out of protection
    //those functions have to do std::signal_thread_fence(std::memory_order_seq_cst) transaction or you risk undefined behavior
}