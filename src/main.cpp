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

    //this implies that perfect hashing (indexing entries) by using key_internalizer is required
    //with the right implementation - and reduce virtual_addr to 24 bits per table - we are expecting to see a 1.5 billion lookups per second
    //with a right concurrent approach - we are expecting to see at least 5 billions entry lookups per second
    //this might change the hash_table industry forever - but take my words for granted - it's the direction I happened to explore - not the direction I wanted to explore
    //hopefully someone would continue my hash_table research - I'm back to the neural network for now
    //thing is the unordered_map is only useful in cuda_environment - where we don't want the branching - we want perfect hashing of the hash_table - and dispatch 1 billion concurrent hash_tables to cuda

    //this map is also useful in probably 10-15 years - when we reach granual cache access implementation from hardware
    //right now - cache is discrete, L1, L2, L3, L4, etc.
    //there will be a time - where there is no number - just distances of memory accesses
    //so our formula will be of use then - but for now, this is only usable in cuda env - and it requires a specialized implementation of internalizer at node level

    const size_t SZ = size_t{1} << 26;
    dg::map_variants::unordered_unstable_fast_map<uint32_t, uint32_t, NullKeyGen, uint32_t> map{};
    std::vector<uint8_t> buf(SZ);

    std::iota(buf.begin(), buf.end(), 0u);
    size_t total{};
    
    for (size_t e: buf){
        map[e] += 1;
    }

    std::shuffle(buf.begin(), buf.end(), std::mt19937{});

    auto now = high_resolution_clock::now();

    for (size_t e: buf){
        total += map.at(e);
    }

    auto then = high_resolution_clock::now();
    std::cout << total << "<>" << duration_cast<milliseconds>(then - now).count() << std::endl;
}
