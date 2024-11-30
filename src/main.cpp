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


}
