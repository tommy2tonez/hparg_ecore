#define DEBUG_MODE_FLAG true

#include "stdx.h"
#include "network_memlock.h"

// #include <stdio.h>
// #include <stdint.h>
// #include "network_fileio_unified_x.h"
// #include <iostream>
// #include "network_kernel_mailbox_impl1.h"
// #include "network_uma_tlb_impl1.h" 
#include <type_traits>

int main(){

    using memlock = dg::network_memlock_impl1::Lock<std::integral_constant<size_t, 0>, std::integral_constant<size_t, 1024>>;
    // static_assert(std::is_trivial_v<std::tuple<size_t>>);
    auto rs = dg::network_memlock::recursive_lock_guard_many(memlock{}, std::add_pointer_t<const void>{});
}