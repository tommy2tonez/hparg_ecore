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

namespace dg{

    template <class T>
    struct ptr_info{
        using max_unsigned_t = uintptr_t;
    };

    template <class T>
    static inline constexpr bool is_ptr_v = true;

    template <class T1, class T>
    auto pointer_cast(T) noexcept -> T1{

    }
}
#endif