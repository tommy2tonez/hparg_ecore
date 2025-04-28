#define DEBUG_MODE_FLAG true
#define  STRONG_MEMORY_ORDERING_FLAG true

// #include <stdint.h>
// #include <stdlib.h>
// #include <type_traits>
// #include <utility>
// #include "network_kernel_mailbox_impl1.h"
// #include <expected>
#include <iostream>
// #include "network_producer_consumer.h"
// #include "network_producer_consumer.h"
// // #include "network_datastructure.h"
// // #include <bit>
// // #include <climits>
// #include <chrono>
// // #include "dense_hash_map/dense_hash_map.hpp"
// // #include <unordered_map>
// // #include "test_map.h"
// // #include "dg_dense_hash_map.h"
// // #include "network_kernel_mailbox_impl1_x.h"
// // #include <vector>
// // #include <type_traits>
// // #include "network_datastructure.h"
// // #include "network_fileio.h"
// // #include "network_fileio_chksum_x.h"
// #include "network_host_asynchronous.h"
// #include <stdlib.h>
// #include <stdint.h>
// #include <type_traits>
// #include "network_allocation.h"
#include <atomic>
#include "stdx.h"
#include <semaphore>

#include <cassert>
#include <cstddef>
#include <new>
 
struct Base
{
    virtual int transmogrify();
};
 
struct Derived : Base
{
    int transmogrify() override
    {
        new(this) Base;
        return 2;
    }
};
 
int Base::transmogrify()
{
    new(this) Derived;
    return 1;
}
 
static_assert(sizeof(Derived) == sizeof(Base));
 
int main()
{
    // Case 1: the new object failed to be transparently replaceable because
    // it is a base subobject but the old object is a complete object.
    stdx::volatile_container<Base> base(std::in_place_t{});
    int n = base->transmogrify();
    // int m = base.transmogrify(); // undefined behavior
    int m = base->transmogrify(); // OK
    assert(m + n == 3);
 
    // Case 2: access to a new object whose storage is provided
    // by a byte array through a pointer to the array.
    struct Y { int z; };
    alignas(Y) std::byte s[sizeof(Y)];
    Y* q = new(&s) Y{2};
    const int f = reinterpret_cast<Y*>(&s)->z; // Class member access is undefined
                                               // behavior: reinterpret_cast<Y*>(&s)
                                               // has value "pointer to s" and does
                                               // not point to a Y object
    const int g = q->z; // OK
    const int h = std::launder(reinterpret_cast<Y*>(&s))->z; // OK
 
    [](...){}(f, g, h); // evokes [[maybe_unused]] effect
}