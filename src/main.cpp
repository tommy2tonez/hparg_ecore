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
// #include "network_kernel_mailbox_impl1_x.h"
#include <vector>
#include <type_traits>
#include "network_datastructure.h"
#include "network_fileio.h"
#include "network_fileio_chksum_x.h"

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

    const char * fp     = "/home/tommy2tonez/dg_projects/dg_polyobjects/src/test.txt";
    std::string msg     = "Hello World!"; 
    std::string other   = "            ";
    
    dg::network_fileio_chksum_x::dg_create_cbinary(fp, msg.size());
    dg::network_fileio_chksum_x::dg_write_binary_indirect(fp, msg.data(), msg.size());
    dg::network_fileio_chksum_x::dg_read_binary_indirect(fp, other.data(), other.size());

    std::cout << other;
}