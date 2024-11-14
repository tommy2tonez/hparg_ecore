// #include "stdx.h"

#define DEBUG_MODE_FLAG true

#include "dense_hash_map/dense_hash_map.hpp"
#include <random>
#include <utility>
#include <algorithm>
#include <functional>
#include <chrono>
#include <iostream>
#include "network_tile_member_access.h"

template <class Task>
auto timeit(Task task) noexcept -> std::chrono::milliseconds{

    using namespace std::chrono;
    auto now    = high_resolution_clock::now();
    task();
    auto then   = high_resolution_clock::now();

    return duration_cast<milliseconds>(then - now);
}

int main(){

    const size_t SZ = size_t{1} << 28;
    std::vector<char> arr(SZ);
    std::generate(arr.begin(), arr.end(), std::bind(std::uniform_int_distribution<char>{}, std::mt19937{}));
    jg::dense_hash_map<uint8_t, size_t> counter{};
    // std::vector<size_t> counter(256);

    //Father - forgive me - for my ignorance - this is actually god damn fast 

    for (size_t i = 0u; i < 256; ++i){
        counter[i] = 0;
    }

    auto task = [&]{
        for (auto c: arr){
            // counter[std::bit_cast<uint8_t>(c)] += 1;
            counter.find(std::bit_cast<uint8_t>(c))->second += 1;
        }
    };

    auto lapsed = timeit(task);
    std::cout << counter[0] << "<>" << lapsed.count() << std::endl;
}