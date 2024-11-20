// #define DEBUG_MODE_FLAG true 

// #include "network_tile_member_access.h"
// #include "network_tileops_host_static.h"

#include <stdint.h>
#include <stdlib.h>
#include <chrono>
#include <memory>
#include <vector>
#include <random>
#include <iostream>
#include <utility>
#include <algorithm>
#include <functional>
#include <array>

class FooInterface{

    public:

        virtual ~FooInterface() noexcept = default;
        virtual auto foo() noexcept -> size_t = 0;
};

template <size_t ID>
class FooImplementation: public virtual FooInterface{

    public:

        auto foo() noexcept -> size_t{

            return ID;
        }
};

template <size_t ID>
auto foo() noexcept -> size_t{

    return ID;
}


static inline std::array<size_t (*)() noexcept, 8> foo_dispatch{foo<0>, foo<1>, foo<2>, foo<3>, foo<4>, foo<5>, foo<6>, foo<7>};

int main(){

    using namespace std::chrono;

    constexpr size_t FOO_SZ = 8;
    std::vector<std::unique_ptr<FooInterface>> foo_ins_arr{};

    [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
        (
            [&]{
                foo_ins_arr.push_back(std::make_unique<FooImplementation<IDX>>());
            }(), ...
        );
    }(std::make_index_sequence<FOO_SZ>{});

    std::cout << foo_ins_arr.size() << std::endl;
    
    //this is going to be a mistake - but I think radix sort + tuple<> dispatch going to improve the perf significantly
    //so instead of letting the branch prediction does it job - we could take the matter into our hands by radix-sorting the dispatch codes - this is actually a hard task - let's put this into the backlog - and build an interface for this - to not tie our hands with the implementation yet allow room for future optimizations - and not making premature optimizations without actual benchmarks
    //the rawest form of virtualization is probably function pointers - this is heavily optimized - so let's stick with the approach for now
    //200 million function dispatchs/ second
    //assume 1024 bytes/ dispatch
    //an average of 200 overlapped GBs are touched
    //assume RAM rate of 20 GB/s - the locality rate has to be at least 90% to bottleneck the function call overhead  

    constexpr size_t RANDOM_SZ = size_t{1} << 30;
    std::vector<uint8_t> random_arr(RANDOM_SZ);
    std::generate(random_arr.begin(), random_arr.end(), std::bind(std::uniform_int_distribution<uint8_t>{0, FOO_SZ - 1}, std::mt19937{}));
    size_t total{};
    auto now = high_resolution_clock::now();

    for (uint8_t dispatch_code: random_arr){
        total += foo_dispatch[dispatch_code]();
    }
    
    auto then = high_resolution_clock::now();

    std::cout << total << "<>" << duration_cast<milliseconds>(then - now).count();
}