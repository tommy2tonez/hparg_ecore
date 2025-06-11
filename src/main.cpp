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
