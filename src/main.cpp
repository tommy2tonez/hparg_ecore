#define DEBUG_MODE_FLAG true
#define STRONG_MEMORY_ORDERING_FLAG true

#include "network_memlock.h"
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
#include "network_producer_consumer.h"

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

class KeyValueTest: public virtual dg::network_producer_consumer::KVConsumerInterface<size_t, size_t>{

    public:

        size_t * total;

    public:

        void push(const size_t& idx, std::move_iterator<size_t *> arr, size_t sz) noexcept{

            *this->total += sz;
        }
};

int main(){

    //alright, is machine learning 10 folds better if we replace the + operation by the two sum operation and matrix multiplication row x column by advanced column dimensional reduction rule (2 sum, 3 sum, 4 sum)

    //essentially, when we are doing the row dimensional reduction, we are hitting the base case of multi variate projection
    //is there the best dimensional reduction rule that just involes the 2 sum 3 sum and 4 sum?

    //we have talked yesterday about how the finite conscious buffer (called the mind buffer) is only required for the sub-optimal case of projection (the human case)
    //our goal is to overcome that suboptimal and become optimal by projecting all the possible context without "confining" the context -> the finite buffer
    //the "soul" is the actionable, which we'd want to project the mind buffer to retrieve

    //we have talked about every possible way to do this, it all comes down to square 0, square 1, square 2, square 4, square 8 with fixed population size (dynamic datatype), this is the sufficient rule to approximate literally everything

    //we can only overcome that suboptimal if we "plug" in the suboptimal to the machine, essentially a "mind buffer" on every machine to communicate + drive this engine backward, we are responsible for the forward construction

    //as you could see, we have collected 6 infinity stones:

    //the reality   (virtual machine L1 + L2 + L3 cache data)
    //the mind      (finite storage buffer)
    //the soul      (finite storage buffer projection -> immediate actionables)
    //the power     (parallel computing)
    //the space     (deviation space of instrument, and the actual projection by using dense hash map)
    //the time      (Wanted, by Jolie)

    //we'll make the mystery gauntlet (we wont snap)

    //it's extremely hard to build component bridges, it requires a significant level of designing to actually singleton a component
    //because that's where we draw the bridges of instance connections

    size_t a                    = {};
    size_t b                    = {};
    auto [aa, bb]               = std::tie(a, b); 

    size_t SZ                   = size_t{1} << 28;
    size_t total                = 0u;
    auto internal_resolutor     = KeyValueTest{};
    internal_resolutor.total    = &total;

    auto handle                 = dg::network_producer_consumer::delvrsrv_kv_open_raiihandle(&internal_resolutor, 32768).value();

    auto random_vec             = std::vector<size_t>(SZ);
    std::generate(random_vec.begin(), random_vec.end(), std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937{}));

    auto task = [&]() noexcept{
        for (size_t value: random_vec){
            dg::network_producer_consumer::delvrsrv_kv_deliver(handle.get(), value & 127u, value);
        }
    };

    std::cout << timeit(task) << "<ms>" << total << std::endl;

    using lock_t = dg::network_memlock_impl1::Lock<size_t, std::integral_constant<size_t, 2>>;
    dg::network_memlock::recursive_lock_guard_many(lock_t{}, nullptr, nullptr);
}
