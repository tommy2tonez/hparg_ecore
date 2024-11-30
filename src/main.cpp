#include <iostream>
#include "dg_map_variants.h"
// #include "dense_hash_map/dense_hash_map.hpp"
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

    //it's hard to beat my brother benchmark - he stored two way pointers - but its very very crucial to have the internal allocation as std::vector<std::pair<Key, Value>> because it's vector, and the iterators have full fledged compiler supports 
    //I could store the other way pointer in another vector - fine 
    //but I dont know if the cache is worth it - whatever, let's store that in another vector and let the bnech speaks

    using namespace std::chrono;

    const size_t SZ = size_t{1} << 22;
    std::vector<uint32_t> buf(SZ);
    std::generate(buf.begin(), buf.end(), std::mt19937());
    dg::map_variants::unordered_unstable_fastinsert_map<uint32_t, uint32_t, PairNullValueGen> map_container{};
    map_container.reserve(SZ * 4);
    // std::vector<size_t> map_container(512);

    auto now = high_resolution_clock::now(); 
    for (uint32_t c: buf){
        map_container[c] = 1u;
    }
    
    auto then = high_resolution_clock::now();

    std::cout << duration_cast<milliseconds>(then - now).count() << "<ms>" << map_container.capacity() << "<>" << map_container.size() << std::endl; 
    // map_container.insert({1, 1});
}
