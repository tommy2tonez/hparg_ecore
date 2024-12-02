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
    //how do we do that precisely - okay - this is hard
    //we know that the hashing function - normally - for linear probing uniformly distribute the hash_range
    //everytime we read - we read 1 cache_line - so we want to reduce our hash_to_index_range by a factor of CACHE_LINE_SZ / size(address)
    //so we hash the key -> spit to index -> re-multiply to get the region -> read one cache line -> do SIMD directly on the read region to find the key -> extract address and return the bucket
    //with this approach - we, hopefully, can reduce our lookup time by a factor of 2 and make this, truly, the fastest flat_hash_table on Earth

}
