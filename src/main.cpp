#include <iostream>
#include "dg_map_variants.h"
#include "dense_hash_map/dense_hash_map.hpp"
#include <random>
#include <utility>
#include <algorithm>
#include <chrono>
#include <vector>
#include <functional>

struct PairNullValueGen{

    constexpr auto operator()() -> std::pair<uint32_t, size_t>{
        return std::make_pair(std::numeric_limits<uint32_t>::max(), size_t{});
    }
};

template <class T>
auto to_const_reference(T& value) -> const T&{
    
    return value;
}

int main(){

    using namespace std::chrono;

    const size_t SZ = size_t{1} << 22;
    std::vector<uint32_t> buf(SZ);
    std::generate(buf.begin(), buf.end(), std::mt19937());
    // jg::dense_hash_map<uint32_t, uint32_t> map_container{};
    // std::vector<size_t> map_container(512);

    dg::map_variants::unordered_unstable_fast_map<uint32_t, uint32_t, PairNullValueGen> map_container;

    auto now = high_resolution_clock::now();     

    for (uint32_t c: buf){
        map_container[c] = 1u;
    }
    // std::shuffle(buf.begin(), buf.end(), std::mt19937{});

    // for (uint32_t c: buf){
    //     total += map_container.at(c);
    // }
    auto then = high_resolution_clock::now();

    std::cout << duration_cast<milliseconds>(then - now).count() << "<ms>" << "<>" << map_container.size()  << std::endl; 
    // map_container.insert({1, 1});
}
