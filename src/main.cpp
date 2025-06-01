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