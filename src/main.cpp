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

struct PairNullValueGen{

    constexpr auto operator()() -> std::pair<size_t, size_t>{
        return std::make_pair(std::numeric_limits<size_t>::max(), size_t{});
    }
};

int main(){

    using namespace std::chrono;

    //alright guys - we were supposed to work on 5 cpu flops/ tile_dispatch (init + ping/pong + gpu/cpu dispatch, etc.) last week but I guess it's gonna be the next week task - this time we'll get it done cleanly
    //unordered_map should be good (functionality-wise + language-wise + cache-wise + branch-prediction-pipeline-wise - implementation is little messy but we don't care about that for now) - if we demote size_type from uint64_t -> uint32_t or uint16_t and use fastinsert_map - with a slight modification
    //the fastinsert map with correct usage of clear is probably the fastest map out there (it leverages stack approach to increase locality of temporal lookups) 
}
