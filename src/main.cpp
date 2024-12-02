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

    constexpr auto operator()() -> std::pair<size_t, size_t>{
        return std::make_pair(std::numeric_limits<size_t>::max(), size_t{});
    }
};

int main(){

}
