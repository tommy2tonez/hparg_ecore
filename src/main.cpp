#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

// #include "network_memlock.h"
// #include "network_memlock_proxyspin.h"
#include <atomic>
#include <random>
#include <memory>
#include <functional>
#include <utility>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
#include "assert.h"
#include "stdx.h"
#include "sort_variants.h"

//let's strategize
//what do we have

//collapse the wall (cand reduction optimization)
//block quicksort (branching optimization)
//insertion iteration up to n flops sort
//find minimum pivot greater than (by leveraging the back index of the insertion sort)
//pivoting the remaining array + continue the quicksort

template <class Task>
auto timeit(Task task) -> size_t{

    auto then = std::chrono::high_resolution_clock::now();
    task();
    auto now = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
} 

int main(){

    const size_t SZ = size_t{1} << 23;

    std::vector<uint32_t> vec(SZ);
    std::generate(vec.begin(), vec.end(), std::bind(std::uniform_int_distribution<uint32_t>{}, std::mt19937{}));

    std::vector<uint32_t> vec2 = vec;
    std::vector<uint32_t> vec3 = vec;

    std::cout << "<insertion_sort_1>" << timeit([&]{std::sort(vec.data(), std::next(vec.data(), vec.size()));}) << "<ms>" << std::endl;
    std::cout << "<insertion_sort_2>" << timeit([&]{dg::sort_variants::quicksort::quicksort(vec2.data(), std::next(vec2.data(), vec2.size()));}) << "<ms>" << std::endl;

    // std::cout << "<insertion_sort_1>" << timeit([&]{std::sort(vec.data(), std::next(vec.data(), vec.size()));}) << "<ms>" << std::endl;
    // std::cout << "<insertion_sort_2>" << timeit([&]{dg::sort_variants::quicksort::quicksort(vec2.data(), std::next(vec2.data(), vec2.size()));}) << "<ms>" << std::endl;

    // for (size_t i = 0u; i < vec2.size(); ++i){
    //     if (vec[i] != vec2[i]){
    //         std::cout << i << "<>" << vec[i] << "<>" << vec2[i] << std::endl;
    //     }
    // }
    // for (uint32_t e: vec2){
        // std::cout << e << std::endl;
    // }
    // std::sort(vec3.begin(), vec3.end());

    assert(vec == vec2);
    // assert(vec2 == vec3);

    //our agenda today is to work on the frame + resolutor + quicksort (to improve the sorting speed of the heap allocator, we got a feedback about this insertion sort feature a while ago, roughly 1-2 months ago)

}