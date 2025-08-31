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
#include "network_kernel_mailbox_impl1.h"
#include "network_trivial_serializer.h"
#include "network_std_container.h"
#include <iostream>
#include "assert.h"
#include "network_kernel_mailbox_impl1_x.h"

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
    std::variant<int, float> z;

    template <class Reflector>
    void dg_reflect(const Reflector& reflector) const{
        reflector(x, y, z);
    }

    template <class Reflector>
    void dg_reflect(const Reflector& reflector){
        reflector(x, y, z);
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

    auto sock = dg::network_kernel_mailbox_impl1_flash_streamx::spawn({});
    auto sock2 = dg::network_kernel_mailbox_impl1::spawn({});
}
