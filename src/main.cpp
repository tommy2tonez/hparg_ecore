#define DEBUG_MODE_FLAG true
#define  STRONG_MEMORY_ORDERING_FLAG true

// #include <stdint.h>
// #include <stdlib.h>
// #include <type_traits>
// #include <utility>
// #include "network_kernel_mailbox_impl1.h"
// #include <expected>
#include <iostream>
// #include "network_producer_consumer.h"
// #include "network_producer_consumer.h"
#include "network_datastructure.h"
// #include <bit>
// #include <climits>
#include <chrono>
#include "dense_hash_map/dense_hash_map.hpp"

int main(){

    const size_t SZ = size_t{1} << 25;

    dg::network_datastructure::unordered_map_variants::unordered_node_map<uint32_t, uint32_t, uint32_t> map{};

    auto then = std::chrono::high_resolution_clock::now();

    for (size_t i = 0u; i < SZ; ++i){
        map[i] = i;
    }

    // map.clear();

    for (size_t i = 0u; i < SZ; ++i){
        map.erase(i);
    }

    for (size_t i = 0u; i < SZ; ++i){
        map[i] = i;
    }

    auto now = std::chrono::high_resolution_clock::now();
    auto lapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();

    std::cout << map.size() << "<map_sz>" << lapsed << "<ms>" << std::endl;
}