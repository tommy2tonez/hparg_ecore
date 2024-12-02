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

    //alright guys, just talked to our client - they want chaining instead of linear probing - like my brother
    //I proposed a different apporach of storing the key as part of the address_bucket - for trivial reflectable serialization
    //and use SIMD approach to do key cmp
    //somewhat a mixed of swiss table and our implementation - not chaining - because there is no SIMD in chaining
    //we want to minimize cache miss by not dereferencing the "intermediate" buckets  

}
