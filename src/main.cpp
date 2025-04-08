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

int main(){

    dg::network_datastructure::unordered_map_variants::unordered_node_map<uint32_t, uint32_t> map{};
    std::swap(map, map);
    bool rs = map == map;
    bool rs1 = map != map;

    std::erase_if(map, [](const auto& kv_pair){return kv_pair.first == 0u;});
}