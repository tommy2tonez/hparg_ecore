#include <iostream>
#include "dg_map_variants.h"
#include "dense_hash_map/dense_hash_map.hpp"
#include <random>
#include <utility>
#include <algorithm>
#include <chrono>
#include <vector>
#include <functional>
#include <unordered_map>
#include "test_map.h"

struct NullKeyGen{

    constexpr auto operator()() -> std::pair<size_t,size_t>{
        return {std::numeric_limits<size_t>::max(), {}};
    }
};

int main(){

    using namespace std::chrono;

    //alright guys
    //optimization techniques for neural network is actually very simple
    //you want a small model - and you want to fit that small model on a small dataset
    //it's path problems
    //and you want to scale the small model
    //its always worked that way
    //people often think they have to have big data to do ML - its actually the entire opposite - you want to start small - test your theories - and scale it (on the algorithmic scale)
    //let's crack the HTTPs session asymmetric encryption within 2 months
    //it's gonna be a bumpy road - but we'll get there - we could actually be rich
    //it seems easy - neural networks - approx + paths + scale + etc. - but few people could actually crack this

    //but if you follow all my lectures - you probably understand every aspect of C++ - some of them aren't necessarily good practices - like the decision to NOEXCEPT allocations to compromise resource leaks - this is actually a good practice - because we want our program to be accurate - and finite allocation pool is an easy problem to solve
    //or decomposition by inheritance
    //or reinterpret_cast by using polymorphic laundering (this is defined - according to the std wordings - but not to some compilers)
    //or fastest concurrent radix sort by unbranching
    //or batch polymorphic dispatch by moving up one polymorphic level and use std::vector<void *> approach (each polymorphic dispatch is on avg 40ns - this is more expensive than a std::mutex acquisition)
    //or const prop by using built-in std::vector<> access - not build your own "container" or your own "iterator" or "pairs" or "tuples" - these are strongly discouraged - they are parts of the compilers - people spent 10-20 years working on optimize every bit of those guys - and they do some VERY ALIEN optis that your customized containers/ iterators can't do
    //or not trying to unbranch/unswitch by building your own lookup tables - these are parts of the compiler's heuristic - and it's better to leave the compiler to do its job - as long as you don't have "holes" in your dispatch code
    //or to guard dispatch codes - by adding DEBUG_MODE_FLAG at default: or else clause - we want to do std::unreachable() in production
    //or std::atomic_thread_fence(std::mmeory_order_seq_cst) - it is a virtue now - because hardwares are moving in that direction - people are done with bad C programs
    //or std::atomic_signal_thread_fence(std::memory_order_seq_cst) for static inline class (or global inline) access - this is a mandatory - if you dont want to risk undefined behaviors - or memory_transaction guard whenever you initialize those variables
    //or not trying to come up with your own "SIMD" - compiler already did that - and they did a very good job at that - all you need to do is to provide environmental arguments - like const *, __restrict__, assume_aligned<SIMD_ALIGNMENT_SZ>, etc. 
    //or to template MEMREGION_SIZE - to do division and modulo - you don't bitshift, & (MEMREGION_SZ - 1) because compiler might not do those operations AT ALL by simply doing an offset read (says uint64_t mod uint32_t - 1)
    //or polymorphic allocations by radixing their purposes - I will talk about this in depth

    //we'll write our own compiler - soon - simply by training f(g(x)) -> x
    //we'll talk about that later
}