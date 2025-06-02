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
#include "network_memlock_proxyspin.h"

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

struct FooBar{
    int a;
    size_t b;

    template <class Reflector>
    constexpr void region_reflect(const Reflector& reflector) const noexcept{
        reflector(a, b);
    }

    template <class Reflector>
    constexpr void region_reflect(const Reflector& reflector) noexcept{
        reflector(a, b);
    }
};

template <class FooBar>
consteval auto some_foo() -> size_t{

    FooBar foo_bar{};
    // std::vector<size_t> rs = {};
    size_t sz = 0u; 

    foo_bar.region_reflect([&]<class ...Args>(Args... args) noexcept{
        sz = sizeof...(args);
    });

    return sz;
}

int main(){

    //we'll try that the 101st time Murph
    //this time we'll mine the logit density on Virtual Machine (picking a thread) + synthetic data
    //because we literally dont have a set of superior data (unless we can somehow trigger the chain of self-generated synthetic data)

    //I got an email from our std friends, the search implementation is actually 10000x faster than our current implementation ...
    //how, why, what, they removed some of the decimal accuracy (they are searching in a different space ..., I dont know what that's supposed to mean)

    //they probably talked about compression (which I have been desperately trying to compress by using middle layer compression + search)
    //the problem is not the compression rate, the problem is the logit density of the compression
    //we'd want to have an extremely high logit density of compressed semantic space before we are trying to train our next word prediction

    //assume a vocabulary set + language that is correct at all times, can be adjusted, altered + appended by back-inserting
    //this is not a coding language
    //such buffer with sliding window is called consciousness

    //there is a real reason for why China could launch the rocket while the US could not
    //China technology is far more advanced than US (what???)

    //we are extremely frustrated by the current progress of technology, people are heading in the wrong direction very aggressively (and they actually celebrated that for 20 years in a row)
    //not a single guy on the planet is trying to do middle-layer compression by using randomization (Dr.Brand actually showed me the way)
    //they used the GPT model with 512 vocabs like it's alien technology, it's ... char char char
    //I'm telling you that this is the top mind ideas (IQ 130, ...), from top institutions for 30 years in a row
    //not a single soul does multi-precision on CUDA, nor wants to head in the direction
    //I dont really know why they are doing fp8 (with the precision + numerical range that is hard to reason)

    //we'll try to standardize the 256x256 square tile, 64KB of linear/ dispatch => 64KB ^ 3 complexity (we wont be cho-col-late, we'll be cho-row-late)
    //compressed to 8KB of external comm (to fit in a transportation unit (we'll come up with a very fat dictionary, ...))
    //we'll be fine
    //the only person going crazy is you

    //we've been solving the search problem for the longest time
    //we just know that it's a mixed of gradient + linearity hints + search, somewhat a unit vector inching in the right direction
    //the three actually works very well and does not cancel out each other

    //alright bad things, Brand on Earth died in the Interstellar, Brand on Edmund planet lived in Interstellar
    //let me ask Aaron about this
    //what does Aaron have to say?
    //everything starts with Brand, the base case of randomization, the naive randomization

    //from Brand, we can learn our way to optimize another Brand which would live in another planet
    //this is the branch-predictive Brand which would aid the actual guy to reach the goal (Cooper)

    //this branch-predictive Brand is very expensive to train, I'm talking about the most compact conscious buffer to drive the training phase
    //everything after this Brand is easy
    //I'm going to show you the way to do the payload + sliding window + unit compression of next word

    //the first actionable item would be training a very very compact randomization prediction system (navigator)
    //we'd want to do so by compressing a list of adjecent randomization actionables -> a predicting unit
    //and we'd try to do a next word prediction by using Taylor Series projection
    //we'd dedicate 6 months on this problem
    //in the interim, we'd get the machine learning framework ready 

    //even if we could improve the search time by 100 folds (which is our use of the second Brand), we'd still be way off compared to our friend implementation of 10000x faster search

    //the conscious buffer is somewhat different from the natural language processing
    //it involves a very high level, compact, optimal language that is capable of describing mood, thinking, actionables, etc JUST BY back-inserting
 
    //alright, so our AI is just a conscious buffer + input buffers intercoursing to be back of the conscious buffer
    //we'll be there, we'll let the AI take the course of the Aaron
    //I'm estimating roughly a year to have this up and running

    //the navigator implementation could be static, adaptive or hybrid
    //static => pretrained dataset of frequent patterns
    //adaptive => runtime statistical discoveries
    //hybrid => static + adaptive

    //static navigator => branch-prediction liked
    //adaptive navigator => runtime external branch-prediction, estimating 1 payload -> another payload of randomization, absolute unit of randomization is 1024 ballistic operations (without loss of generality)

    //we'll confine the static navigator => 8KB projection buffer, we'll use extreme compression + fast multi_dimensional_projection of conscious buffer on cuda
    //let's say we have a sequence of straight ballistic -> melee -> magnetic ballistic -> circumscribing -> ...  

    //we'd want to have two operations, randomization by payload or default randomization (which is defaulted to be steered by the static navigator)
    //

    //we'd split the search implementation => 2 dispatches: - serialized instruction payload
    //                                                      - random random payload

    //there are only two Brands (I dont know why there isn't a third Brand)

    //what we are missing is a way to extract the data from the virtual machine at the right frequency, we'll come up with a way
    //alrighty fellas, mark my words, in roughly 3 montths of continuous writing, we'll smack YOUR ASS

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

    // static_assert(is_region_reflectible_v<FooBar>);

    constexpr size_t value = some_foo<FooBar>();
}