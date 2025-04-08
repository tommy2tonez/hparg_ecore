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
#include "network_datastructure.h"
// #include <bit>
// #include <climits>
#include <chrono>
#include "dense_hash_map/dense_hash_map.hpp"


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

    const size_t SZ = size_t{1} << 30;
    size_t total = 0u;

    dg::network_datastructure::unordered_map_variants::unordered_node_map<uint32_t, uint32_t> map{};

    auto then = std::chrono::high_resolution_clock::now();

    for (size_t i = 0u; i < SZ; ++i){
        total += ceil2(static_cast<uint32_t>(i));
        // std::cout << i << "<>" << ceil2(i) << std::endl;
    }

    auto now = std::chrono::high_resolution_clock::now();
    auto lapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();

    std::cout << total << "<total>" << lapsed << "<ms>" << std::endl;
}