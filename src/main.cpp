#define DEBUG_MODE_FLAG false

// #include "network_tileops_host_poly.h"
#include <vector>
#include <iostream>
#include <chrono>
// #include "network_uma_tlb.h"
#include <chrono>

int main(){

    //alright - I hope that we could get the f(g(x)) -> x ring to practice this week or next week - running on CPUs only
    //the art is really not programming in C++, but is to program in CUDA, dispatch 256 tile - with an overhead of ~= 3 host flops/ dispatch 
    //in order to do so, there are tons of alien bare-metal optimizations like static inline and friends - because it would be very very bad if let's say you are wasting 30 flops to dispatch 256 flops to CUDA - and your cuda device is running 1 << 16 faster than your host device
    //fatten the tile size defeats the purpose of having this entire thing in the first place - so...
    //256 tile size is a MUST - for optimization purposes and uniform context distribution purposes
    //as long as we keep the 100.000Hz of forward scanner - I think things would be fine
    //as for backprop - we want to implement a more elegant solution than the one we have right now - but the idea remains the same - you want to schedule the tiles in an efficient way
}