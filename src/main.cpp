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
}