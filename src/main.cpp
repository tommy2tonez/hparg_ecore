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

struct PairNullValueGen{

    constexpr auto operator()() -> std::pair<size_t, size_t>{
        return std::make_pair(std::numeric_limits<size_t>::max(), size_t{});
    }
};

int main(){

    //let's move on from the benchmark
    //this micro benchmark is meaningless in a macro situation
    //the only thing that matters in a macro situation is cache access - and locality of memory access
    //I've benched this map for 2 years now - truth is things that appear faster in the benchmark is actually slower in applications - like in dg_heap
    //we don't really care about branch pipeline - which is the current erase problem - quicksort vs block_quicksort for example
    //as long as we minimize the cache access - we move on from the problem - and call it a day  

    using namespace std::chrono;

    const size_t SZ = size_t{1} << 26;

    std::vector<size_t> buf(SZ);
    std::generate(buf.begin(), buf.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937_64{}));
    dg::map_variants::unordered_unstable_fast_map<size_t, size_t, PairNullValueGen, uint32_t> map{};
    // jg::dense_hash_map<size_t, size_t> map{};
    // size_t total{};
    // robin_hood::unordered_map<size_t, size_t> map{};

    // map.size();
    map.reserve(SZ * 2);

    auto now = high_resolution_clock::now();

    for (size_t e: buf){
        map[e] = 0u;
        // auto [it, hint] = map.erase_find(e);
        // map.erase(it, hint);
    }

    auto then = high_resolution_clock::now();

    std::cout << "<>" << "<>" << duration_cast<milliseconds>(then - now).count() << "<ms>" << std::endl;
}
