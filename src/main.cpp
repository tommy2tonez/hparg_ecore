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
// #include "network_datastructure.h"
// #include <bit>
// #include <climits>
#include <chrono>
// #include "dense_hash_map/dense_hash_map.hpp"
#include <unordered_map>
// #include "test_map.h"
// #include "dg_dense_hash_map.h"
#include "network_kernel_mailbox_impl1_x.h"
#include <vector>
#include <type_traits>
#include "network_datastructure.h"

// template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
// constexpr auto ulog2(T val) noexcept -> T{

//     return static_cast<T>(sizeof(T) * CHAR_BIT - 1u) - static_cast<T>(std::countl_zero(val));
// }

// template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
// static constexpr auto ceil2(T val) noexcept -> T{

//     if (val < 2u) [[unlikely]]{
//         return 1u;
//     } else [[likely]]{
//         T uplog_value = ulog2(static_cast<T>(val - 1u)) + 1u;
//         return T{1u} << uplog_value;
//     }
// }

template <class Iter, class T, class = void>
struct is_const_iter: std::false_type{};

template <class Iter, class T>
struct is_const_iter<Iter, T, std::void_t<std::enable_if_t<std::is_same_v<decltype(*std::declval<Iter&>()), std::add_lvalue_reference_t<std::add_const_t<T>>>>>>: std::true_type{};

template <class Iter, class T>
static inline constexpr bool is_const_iter_v = is_const_iter<Iter, T>::value;

template <class Iter, class T, class = void>
struct is_normal_iter: std::false_type{};

template <class Iter, class T>
struct is_normal_iter<Iter, T, std::void_t<std::enable_if_t<std::is_same_v<decltype(*std::declval<Iter&>()), std::add_lvalue_reference_t<T>>>>>: std::true_type{};

template <class Iter, class T>
static inline constexpr bool is_normal_iter_v = is_normal_iter<Iter, T>::value;

int main(){

    // std::cout << queue[0];
    // auto vec = std::vector<size_t>();

    // static_assert(is_normal_iter_v<typename decltype(vec)::iterator, size_t>);

    //alright, we have talked to a team of people worked on naive words compression
    //its a bloom filter + sliding window compression to convert to an intermediate semantic space of size{1} << 1024
    //assume that we have this sentence and a sliding window of size 3
    //<assume that we>, <that we have>, <we have this>, <have this sentence>, <this sentence and>, <sentence and sliding> etc.
    //each phrase is mapped -> a bit representation in the output semantic space
    //this could be lossless compression if used with integrity hash being part of the compressed semantic space
    //the optimal compressed space should be the numerical range of all possible word representations (we are converging to the point)

    //we'll be back tomorrow, I have estimated at least 50MB of raw source code to get this core up and running
    //we have literally a shit ton of work to do
}