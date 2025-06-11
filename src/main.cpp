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

    //alright, we'll try to have a demo soon guys
    //remember, the driver is very important

    //well Son, I might not be coding within a year or two but I'll make sure my legacy lives on
    //until then, we have to thru the lectures
    //remember, the backprop driver is the most important driver in the history of mankind
    //we need to find that driver by using randomization (THERE IS NO OTHER WAY)

    //the request was done correctly by using a loop of synchronization
    //we thru the requests almost instantly, and push the promise to the synchronizable warehouse which then would be delivered to synchronization worker who would check for the result and do another requests

    //the synchronizable waiter is simply a worker waiting on an ascending order of timeout
    //we'd want that to be in an ascending order because we dont want to not notify people on time, which would "hinder the warehouse exhaustability" + "hinder the ontimeness of user-requests hooked responses"
    //we reckoned that a simple dg::network_exception::ExceptionHandlerInterface is sufficient, because people can actually build a synchronization protocol on top of that
    //says using a binary semaphore to block the response

    //we'd want that "detached" for various reasons, we dont want to be forcy for tasks that are not memory dangerous like cuda_asynchronous_device or host_asynchronous_device

    //as for other concerns, it should be the dg::network_rest to take care of other responsibility from the security + optimizeness of requests batching + to ontimeness of responses, to etc.
    //the dropbox responsibility is clearly cut, as for its only responsibility is to call network_rest efficiently and do proper handshakes by using auth2, and there isn't a better way to do so  

    //i was thinking about the completeness of the optimal network
    //we were writing this paper,
    //in short, can we prove:
    //given a neural network with a unit size of 1 bit and a neural network with a unit size of 1KB
    //if we have found the optimal solution for the 1 bit, can we convert it to the optimal solution for 1KB
    //this has a lot to do with input compression, such is we would want to maximize the density of input context (we are speaking in a language that is extremely dense)
    //as the logit density grew for the input context, the uniform responsibility converges
    //and as for the unit size, we always compress with the ratio of 2:1, 2 base operation for 1 upper operation, so there is not a problem of lost context

    //no, literally no one cares about how you bind your internal components or how your component is written
    //people care about if your component does the job
    //how to use your component
    //how to reset your component
    //and how to spawn your component

    //if you keep thinking in terms of that, a.k.a. client perspective, we call it software engineering
    //the engineering is the interfaces, the components, the responsibility of those components, not the low level implementation or your coding virtues

    //if we look at the dropbox responsibility, it's to call the network_rest to make requests + wait (we might or might not want to improve the latency)
    //if we look at the network_rest, it's to make a batch request + implement a Promise, this promise is not like other promises, it's called right after the return of the promise, and get() is simply a synchronization protocol
    //alright, why don't we use the Promise<etc.> again? because it would be languagely incorrect, this is precisely why I said human being is terrible terrible at designing things that they think are reusable

    //if we look at the kernel_map responsibility, it's to provide an interface of memory mapping, and the "border" of those memory maps are guaranteed by the allocator awaring of those boundaries
    //this kernel_map is differnt from other cudamap or etc. map, why? because it's just different, we can't implement a generic version for these guys because those "different" parameters are our clue to optimization, why would we want to polymorphic the solution there?
    //may or maybe the memory mapping will attempt to do "segmentation" mapping, we dont really care, the mapped memory must be IN BOUND

    //the C++ language is very beautiful if we keep using unique_ptr<> + shared_ptr<>
    //literally

    //the problem with C++ is too many people are being too serious about the constructors move + copy, operator move + copy
    //it's not that hard fellas

    //stick to the number one rule, for the datastructure + semantic containers, we'd want to use those move + copy, operator move + copy, std::move_iterator<> + etc
    //for other logic handler components, we'd want to use std::unique_ptr<> + std::shared_ptr<>, period, call a factory, get the component, and get on with it  

    //we are doing tile forward dispatches roughly 10000x faster just by staying on CPU, imagine that we use the power of GPU + let CPU do all the kernel things
    //this is thanks to our magical asynchronous device
    //we split the workorders into two categories: the RAM_HEAVY and the CPU_HEAVY

    //the CPU_HEAVY simply just loads everything into the cache and does linear-liked operations
    //the RAM_HEAVY is intentionally bottlenecked by the hyperthreading + same-core affinity, we are "sacrificing" the cores to save the RAM bus 
    //best yet, we just have 1 dedicated RAM worker to do the RAM_HEAVY, such would sacrifice the latency, we'll see about the options

    //we scale with cores almost linearly, because we don't miss cache, everything is within the confinement of the unit
    //unit is actually one of our proudest architect choice

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
