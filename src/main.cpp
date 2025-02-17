//
#include <random>
#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <functional>
#include <chrono>
#include <utility>

int main(){

    //assume a == 1

    //what happened in the case of branch prediction is this
    //when we do code path analysis - we sectorize the things that the if goes into and the things the if does not go into
    //when we do branch pipeline optimization - we want the distribution to be very skewed - either we go into the branch or we don't
    //(1) let's look at the jmp statement of val == 0 or val == 1 - we either jump into the val == 0 block or val == 1, which distribution is uniform - such halts the system - which is bad  

    //(2): always hit the branch - cost is 1

    //(3): 50% hits the branch - cost == (1)

    //so what really happened there?
    //what happened is that if you flatten all the branches - and make it one big jump statement - slot 0 mean if 1 true if 2 false, slot 1 means if 1 true if 2 true
    //the hardware would use a nice heuristics to precompute the most likely gonna happen slot - if things work out fine - cpu continues running, if things do not work out fine - cpu throws away the computation and now really considers to go into the correct branch 

    //so we, developers, must increase the skewness of the branches - consider 95% betting on slot 0 - and 5% betting on slot 1 - this means you have a good branching pipeline

    //what happened if you don't have a good branching pipeline? we have a solution for it - its called virtualization of dispatch payload + radix partitioning (this only works in certain cases where you set up for batch dispatches)
    //thing is we mostly don't use the trick to avoid branching but rather for many other reasons: (1): instruction cache fetch, (2): hardware branching pipeline memorization (if your code goes everywhere - the number of slots would be too many to be remembered) 
    //this is probably the biggest optimizable that we low-life must first consider 
    //the second is word size operation vs non-word size operation (size_t is faster than uint8_t if L1 fit)
    //I'm too busy thinking about the branches to remember my Dad's lottery - apparently

    //2030 is the year of no poor man - mark my words - we'll see how things work out with our interstealler plan
    //we'll be pushing Yottabyte/s by then (well this is the Chinese dream 1 billion device, each 10 PB flops/s)

    size_t a        = {};
    std::cin >> a;

    const size_t SZ = size_t{1} << 27;
    auto vec        = std::vector<size_t>(SZ); 
    auto vec1       = std::vector<size_t>(SZ);
    auto vec2       = std::vector<size_t>(SZ);
    size_t total    = {};

    std::generate(vec.begin(), vec.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{1}));
    std::generate(vec1.begin(), vec1.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{2}));
    std::generate(vec2.begin(), vec2.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{3}));

    auto now        = std::chrono::high_resolution_clock::now();

    for (size_t i = 0u; i < SZ; ++i){
        size_t val  = vec[i] & a;

        if (val == 0){
            total += vec1[i];
        } else if (val == 1){
            total += vec1[i];
        } else{
            std::unreachable();
        }
    }

    for (size_t i = 0u; i < SZ; ++i){
        size_t val  = vec[i] & a;

        // total += (val == 1 || val == 2);

        if (val == 0 || val == 1){
            total += vec1[i];
        }
    }

    for (size_t i = 0u; i < SZ; ++i){
        size_t val  = vec[i] & a;

        // total += (val == 1 || val == 2);

        if (val == 1 || val == 2){
            total += vec1[i];
        }
    }


    auto then       = std::chrono::high_resolution_clock::now();
    auto lapsed     = std::chrono::duration_cast<std::chrono::milliseconds>(then - now).count();

    std::cout << total << "<>" << lapsed << "<ms>" << std::endl;
}