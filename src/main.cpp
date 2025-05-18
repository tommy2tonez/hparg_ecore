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

    // const size_t SZ = size_t{1} << 23;

    auto random_device = std::bind(std::uniform_int_distribution<uint32_t>(0, 2048), std::mt19937{}); 

    while (true){
        const size_t SZ = random_device();
        std::vector<uint32_t> vec(SZ);
        std::generate(vec.begin(), vec.end(), std::ref(random_device));
        std::vector<uint32_t> vec2 = vec;

        std::sort(vec.begin(), vec.end());
        dg::sort_variants::quicksort::quicksort(vec2.data(), std::next(vec2.data(), vec2.size()));

        if (vec != vec2){
            std::cout << "mayday" << std::endl;
            std::abort();
        }
    }
}