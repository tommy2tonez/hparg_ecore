#ifndef __NETWORK_POINTER_H__
#define __NETWORK_POINTER_H__

#include <stdint.h>
#include <stdlib.h>

namespace dg::network_pointer{

    using cufs_ptr_t    = uint64_t;
    using fsys_ptr_t    = uint64_t;

    static inline constexpr cufs_ptr_t NULL_CUFS_PTR    = cufs_ptr_t{};
    static inline constexpr fsys_ptr_t NULL_FSYS_PTR    = fsys_ptr_t{};
} 

#endif