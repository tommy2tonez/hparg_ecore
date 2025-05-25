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
#include "network_compact_serializer.h"

template <class Task>
auto timeit(Task task) -> size_t{

    auto then = std::chrono::high_resolution_clock::now();
    task();
    auto now = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
} 

struct Foo{
    uint32_t x;
    std::optional<uint64_t> y;

    template <class Reflector>
    void dg_reflect(const Reflector& reflector) const{
        reflector(x, y);
    }

    template <class Reflector>
    void dg_reflect(const Reflector& reflector){
        reflector(x, y);
    }

    bool operator ==(const Foo& other) const noexcept{

        return std::tie(x, y) == std::tie(other.x, other.y);  
    }

    bool operator !=(const Foo& other) const noexcept{

        return std::tie(x, y) != std::tie(other.x, other.y);  
    }
};

struct Bar{
    std::vector<Foo> x;
    std::map<size_t, size_t> y; //its this guy
    std::vector<uint64_t> xy;
    std::set<size_t> xyz;
    std::unique_ptr<Foo> d;
    double f1;
    float f2;

    template <class Reflector>
    void dg_reflect(const Reflector& reflector) const{
        reflector(x, y, xy, xyz, d, f1, f2);
    }

    template <class Reflector>
    void dg_reflect(const Reflector& reflector){
        reflector(x, y, xy, xyz, d, f1, f2);
    }

    bool operator ==(const Bar& other) const noexcept{

        return std::tie(x, y, xy) == std::tie(other.x, other.y, other.xy);  
    }

    bool operator !=(const Bar& other) const noexcept{

        return std::tie(x, y, xy) != std::tie(other.x, other.y, other.xy);  
    }
};

int main(){

    //we have spent years working on the concepts
    //it is about the unit of the tile, flattened tree -> tile
    //multiple trees -> tile
    //training strategies:
    //directional backprop
    //single variable directional backprop
    //search + linearity hint (every tile1xtile2 in binary tree is self sufficient to approx)
    //its complicated
    //the most complicated of them all is synthetic data
    //we are training our product on inferior data (we need to generate NP search problems for AI to train on)
    //self-generated synthetic daat, etc.
    
    Bar bar{{Foo{1, std::nullopt},
             Foo{2, uint64_t{2}}},
            {{1, 2}, {2, 3}, {3, 4}},
            {1, 2, 3, 4, 5, 6},
            {},
            std::make_unique<Foo>(Foo{1, 2}),
            1,
            2};

    std::string buf     = dg::network_compact_serializer::dgstd_serialize<std::string>(bar);
    Bar deserialized    = dg::network_compact_serializer::dgstd_deserialize<Bar>(buf);
    
    if (deserialized.y != bar.y){
        std::cout << "mayday1" << std::endl;
        std::abort();
    }

    if (deserialized.xy != bar.xy){
        std::cout << "mayday2" << std::endl;
        std::abort();
    }

    if (bar != deserialized){
        std::cout << "mayday bar" << std::endl;
        std::abort();
    }

    // std::cout << (deserialized << std::endl;

    std::string buf2    = dg::network_compact_serializer::dgstd_serialize<std::string>(deserialized);

    if (buf != buf2){
        std::cout << buf.size() << std::endl;
        std::cout << buf2.size() << std::endl;
        std::cout << "mayday" << std::endl;
        std::abort();
    }

    // Bar buf2_deserialized = dg::network_compact_serializer::dgstd_deserialize<Bar>(buf2);

    // if (buf2_deserialized.y != bar.y){
        // std::cout << "mayday3" << std::endl;
        // std::abort();
    // }

    //careful boys
    //we've been researching this for longer than you have been alive
    //there exists an optimal intercourse operation for tile1 x tile2 (tile1 pair tile2, uint8_t -> uint16_t promotion) pair (taylor_coeef tile3 uint128_t)
    //we have not been able to do that yet, which is a subject to be researched
    //and there exists an optimal backprop algorithm (which does not involve differentiations, yet translatable to differentiations ???)
    //we usually would want to do that by using a set of crit tiles -> search -> backprop the "expected domain projection space"

    //it's already incredibly hard to get the engine running with no problems + not thrashing the cores memory orderings + friends
    //I guess it's yall jobs to research within a confined requirements

    //I've been exposed to destructive interferences of brain virtues, forcing a rewrite that is potentially dangerous to my writing ability
    //we would want to complete the project as soon as possible, probably 3-12 months
}
