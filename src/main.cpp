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

    //unordered_map is not hard to build - but there are so many little details that make it hard to implement
    //from the branching [[likely]], [[unlikely]] - to compiler vectorization - to last mohicans - to maybe_rehash - to locality - to cache read of keys - to emplace requirements - to hetergenous lookup - to rehash on virtual_load_factor (1 - e^-insert_factor) - to not include the reverse address lookup (this is a design decision to reduce cache miss)
    //to using compiler_built_in_iterator - vector::iterator - because they support full fledged compiler supports - to pointer stability after swap, etc. - to atomic methods (rehash, insert, erase , etc) - to noexceptability of lookup and erase
    //it's just too many little requirements that made it hard to increment (implement) - but with the right engineering angle - we can solve it incrementally
    //I think this is a decent map (based on the idea of my brother - jg::dense_hash_map) that answers most of the usecases - the next step is to make this part of dg_buf and do cuda_dispatch - we will get back to this in the future
    //for now we move on from the topic
    //this should meet the minimum viable product

    //we get back on track and implement other components for now
}
