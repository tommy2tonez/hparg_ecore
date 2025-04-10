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
#include "dense_hash_map/dense_hash_map.hpp"
#include <unordered_map>
#include "test_map.h"
// #include "dg_dense_hash_map.h"
#include "network_kernel_mailbox_impl1_x.h"

template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
constexpr auto ulog2(T val) noexcept -> T{

    return static_cast<T>(sizeof(T) * CHAR_BIT - 1u) - static_cast<T>(std::countl_zero(val));
}

template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
static constexpr auto ceil2(T val) noexcept -> T{

    if (val < 2u) [[unlikely]]{
        return 1u;
    } else [[likely]]{
        T uplog_value = ulog2(static_cast<T>(val - 1u)) + 1u;
        return T{1u} << uplog_value;
    }
}

int main(){

}