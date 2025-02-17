//
#include <random>
#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <functional>
#include <chrono>
#include <utility>
#include <array>
#include <cstring>

int main(){

    //alright fellas - we'll be back tomorrow 

    // size_t a        = {};
    // std::cin >> a;

    // const size_t SZ = size_t{1} << 27;
    // auto vec        = std::vector<size_t>(SZ); 
    // auto vec1       = std::vector<size_t>(SZ);
    // auto vec2       = std::vector<size_t>(SZ);
    // size_t total    = {};

    // std::generate(vec.begin(), vec.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{1}));
    // std::generate(vec1.begin(), vec1.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{2}));
    // std::generate(vec2.begin(), vec2.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{3}));

    // auto now        = std::chrono::high_resolution_clock::now();

    // for (size_t i = 0u; i < SZ; ++i){
    //     size_t val  = vec[i] & a;

    //     if (val == 0){
    //         total += vec1[i];
    //     } else if (val == 1){
    //         total += vec1[i];
    //     } else{
    //         std::unreachable();
    //     }
    // }

    // for (size_t i = 0u; i < SZ; ++i){
    //     size_t val  = vec[i] & a;

    //     // total += (val == 1 || val == 2);

    //     if (val == 0 || val == 1){
    //         total += vec1[i];
    //     }
    // }

    // for (size_t i = 0u; i < SZ; ++i){
    //     size_t val  = vec[i] & a;

    //     // total += (val == 1 || val == 2);

    //     if (val == 1 || val == 2){
    //         total += vec1[i];
    //     }
    // }


    // auto then       = std::chrono::high_resolution_clock::now();
    // auto lapsed     = std::chrono::duration_cast<std::chrono::milliseconds>(then - now).count();

    // std::cout << total << "<>" << lapsed << "<ms>" << std::endl;

    const size_t SZ     = size_t{1} << 27;
    const size_t ARR_SZ = 256;

    auto vec            = std::vector<size_t>(SZ);
    alignas(512) std::array<size_t, 256> tmp_arr{}; 
    alignas(512) std::array<size_t, 256> tmp1_arr{};

    std::generate(vec.begin(), vec.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{static_cast<size_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}));
    std::generate(tmp_arr.begin(), tmp_arr.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{static_cast<size_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}));
    std::generate(tmp1_arr.begin(), tmp1_arr.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{static_cast<size_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}));
    tmp1_arr = tmp_arr;

    size_t total    = 0u;
    auto now        = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0u; i < SZ - 256; ++i){
        size_t e    = vec[i];
        total       += std::memcmp(tmp_arr.data(), std::next(vec.data(), i), 256 * sizeof(size_t)); //the case of memcmp is just another beast - we dont really know why it's fast - except for it is
    }

    auto then       = std::chrono::high_resolution_clock::now();
    auto lapsed     = std::chrono::duration_cast<std::chrono::milliseconds>(then - now).count();

    std::cout << total << "<>" << lapsed << "<ms>" << std::endl;

    //well well - that was a bold statement boys - people have walked through this path and showed you the way of differential + string calibration + massive parallel synchronized engine
    //it's up to you to reach the Yottabyte scale
    //its not for a faint of heart to understand the code - it's complicated - because it is - yet once you have got the hinge of computer science - you would probably cut the thing like I do
    //at the heart of computer sciense is localtiy + affinity
}