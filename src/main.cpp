#include "dg_map_variants.h"
#include <random>
#include <utility>
#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <functional>

int main(){

    using namespace std::chrono;

    const size_t SZ = size_t{1} << 28;
    std::vector<uint8_t> buf(SZ);
    std::generate(buf.begin(), buf.end(), std::bind(std::uniform_int_distribution<uint8_t>(), std::mt19937()));
    dg::map_variants::unordered_unstable_map<uint8_t, size_t> map_container{};
    map_container.reserve(512);

    for (size_t i = 0u; i < 256; ++i){
        // map_container[i] = 0u;
        map_container[i] = 0u;
    }

    auto now = high_resolution_clock::now(); 
    for (uint8_t c: buf){
        map_container[c] += 1;
        // map_container.at(c) += 1;
        // map_container[c] += 1;
        // auto iterator = map_container.find(c);
        // iterator->second += 1;
        // iterator->second += 1;
        // iterator->second += 1;
    }
    auto then = high_resolution_clock::now();

    std::cout << duration_cast<milliseconds>(then - now).count() << "<ms>" << map_container[0u] << std::endl; 
    // map_container.insert({1, 1});
}
