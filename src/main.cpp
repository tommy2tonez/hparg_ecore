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

    const size_t SZ = size_t{1} << 30;
    dg::map_variants::unordered_unstable_fast_map<size_t, size_t, NullKeyGen, uint32_t, robin_hood::hash<size_t>> map{};
    // std::vector<size_t> map(256);
    std::vector<uint8_t> buf(SZ);
    std::vector<size_t> table1(256);
    std::vector<size_t> table2(256);
    std::generate(buf.begin(), buf.end(), std::bind(std::uniform_int_distribution<uint32_t>{}, std::mt19937_64{}));

    std::iota(table1.begin(), table1.end(), 0u);
    // for (size_t i = 0u; i < 256; ++i){
    //     map.insert({i, i});
    // }

    // std::cout << map.at(0) << std::endl;
    // for (size_t i = 0u; i < 256; ++i){
        // std::cout << i << "<>" << map.at(i) << "<>" << map.find(i)->second << std::endl;
    // }
    // map.reserve(SZ * 2);
    // for (uint32_t e: buf){
    //     // map[e] += 1;
    //     map[e] += 1;
    // }
    
    // std::shuffle(buf.begin(), buf.end(), std::mt19937{});
    auto now = high_resolution_clock::now();
    
    for (uint32_t e: buf){
        // map.at(e) += 1;
        table2[e] += 1;
    }

    auto then = high_resolution_clock::now();

    // // for (size_t i = 0u; i < 64; ++i){
    // //     std::cout << map.find(i)->first << "<>" << map.find(i)->second << std::endl;
    // // }

    std::cout << table2[0] << "<>" << duration_cast<milliseconds>(then - now).count() << std::endl;

}
