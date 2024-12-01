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
    
    //alright guys - I think our map is very good now
    //let's get back on track to crack the asymmetric encoding method - probably at least 2 months
}
