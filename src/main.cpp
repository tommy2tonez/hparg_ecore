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

int main(){

    const size_t SZ = size_t{1} << 27;
    auto vec        = std::vector<size_t>(SZ); 
    auto vec1       = std::vector<size_t>(SZ);
    auto vec2       = std::vector<size_t>(SZ);
    size_t total    = {};

    std::generate(vec.begin(), vec.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{1}));
    std::generate(vec1.begin(), vec1.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{2}));
    std::generate(vec2.begin(), vec2.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{3}));

    auto now        = std::chrono::high_resolution_clock::now();

    for (size_t i = 0u; i < SZ; ++i){
        size_t val  = vec[i] % 2;
        size_t buf[2];
        buf[0]      = vec1[i];
        buf[1]      = vec2[i];
        total       += buf[val];
    }

    auto then       = std::chrono::high_resolution_clock::now();
    auto lapsed     = std::chrono::duration_cast<std::chrono::milliseconds>(then - now).count();

    std::cout << total << "<>" << lapsed << "<ms>" << std::endl;
}