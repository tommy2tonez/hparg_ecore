#ifndef __NETWORK_POINTER_H__
#define __NETWORK_POINTER_H__

//define HEADER_CONTROL 0

#include <stdint.h>
#include <stdlib.h>

namespace dg::network_pointer{

    using cufs_ptr_t    = uint64_t;
    using fsys_ptr_t    = uint64_t;
    using uma_ptr_t     = uintptr_t;
    using vma_ptr_t     = uintptr_t;
    using device_id_t   = uint16_t; 

    static inline constexpr cufs_ptr_t NULL_CUFS_PTR        = cufs_ptr_t{};
    static inline constexpr fsys_ptr_t NULL_FSYS_PTR        = fsys_ptr_t{};
    static inline constexpr size_t MEMREGION_SZ             = size_t{1u};
    static inline constexpr size_t MEMORY_PLATFORM_SZ       = size_t{1};
    static inline constexpr size_t MAX_MEMORY_PLATFORM_SZ   = size_t{32};
} 

namespace dg{

    template <class T = void>
    struct ptr_info{
        using max_unsigned_t = uintptr_t;
    };

    template <class T>
    static inline constexpr bool is_ptr_v = true;

    template <class T1, class T>
    auto pointer_cast(T lhs) noexcept -> T1{

        return reinterpret_cast<T1>(lhs);
    }

    template <class T>
    struct ptr_limits{
        // static inline constexpr T null_value = T{};

        static consteval auto null_value() noexcept -> T{

            return T{};
        }
    };

}
#endif