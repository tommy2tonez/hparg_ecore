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
// // #include "network_host_asynchronous.h"
// #include <stdlib.h>
// #include <stdint.h>
// #include <type_traits>
// #include "network_allocation.h"
#include <atomic>
// #include "stdx.h"
#include <semaphore>

int main(){

    std::atomic<size_t> int_value{};
    size_t rep{};

    int_value.compare_exchange_strong(rep, rep, std::memory_order_relaxed);
    //we have finally reached 7.0 MB of code, damn
}