#define DEBUG_MODE_FLAG true

// #include "stdx.h"
// #include <type_traits>
// #include "network_tileops_host_static.h"
// #include <iostream>
// #include <math.h>
// #include <utility>
// #include <functional>
// #include <algorithm> 
// #include "network_memlock_proxyspin.h"

// #include <chrono>
// #include <memory>
// #include <stdint.h>
// #include <stdlib.h>
// #include <vector>
// #include <random>
// #include <functional>
// #include <utility>
// #include <algorithm>
// #include <bit>
// #include <iostream>
// #include "network_uma_tlb_impl1.h"
// #include "network_uma.h"

#include <memory>
#include <atomic>
#include "stdx.h"
#include "network_memlock.h"
#include <type_traits>
#include "network_tile_member_access.h"

int main(){
    
    // using memlock_ins = dg::network_memlock_impl1::ReferenceLock<std::integral_constant<size_t, 0>, std::integral_constant<size_t, 1024>>;
    // memlock_ins::acquire_wait({});

    // using hardware_destructive_atomic_flag = std::atomic_flag alignas(64); 
    // std::unique_ptr<hardware_destructive_atomic_flag[]> arr{};
    // dg::network_uma::lockmap_safewait_many<1>({});
}