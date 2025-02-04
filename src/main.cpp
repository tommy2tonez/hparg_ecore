//
#include <random>
#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <functional>
#include <chrono>

int main(){

    //correctly predicted branches aren't expensive - what expensive is the cache fetch + hardware pollution + kernel scheduling overheads
    //I'll try my best to do 1TB linear/host_core * s - this is doable

    const size_t SZ = size_t{1} << 30;
    auto vec        = std::vector<size_t>(SZ);
    size_t total    = {}; 
    std::generate(vec.begin(), vec.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{}));

    auto now        = std::chrono::high_resolution_clock::now();

    for (size_t i = 0u; i < SZ; ++i){
        if (vec[i] == vec[0]){
            continue;
        }

        if (vec[i] == vec[1]){
            continue;
        }

        if (vec[i] == vec[2]){
            continue;
        }

        if (vec[i] == vec[3]){
            continue;
        }

        if (vec[i] == vec[4]){
            continue;
        }

        if (vec[i] == vec[5]){
            continue;
        }

        if (vec[i] == vec[6]){
            continue;
        }

        if (vec[i] == vec[7]){
            continue;
        }
        
        if (vec[i] == vec[8]){
            continue;
        }

        total += vec[i];
    }

    auto then       = std::chrono::high_resolution_clock::now();
    auto lapsed     = std::chrono::duration_cast<std::chrono::milliseconds>(then - now).count();

    std::cout << total << "<>" << lapsed << std::endl; 
}