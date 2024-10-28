#ifndef __NETWORK_MEMORY_UTILITY_H__
#define __NETWORK_MEMORY_UTILITY_H__

#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <type_traits>
#include <bit>
#include <assert.h>
#include <memory>
#include <limits.h>

namespace dg::memult{

    static inline constexpr size_t HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE   = size_t{1} << 6;
    static inline constexpr size_t HARDWARE_CONSTRUCTIVE_INTERFERENCE_SIZE  = size_t{1} << 6;
    static inline constexpr size_t SIMD_ALIGN_SIZE                          = size_t{1} << 5;
    static inline constexpr bool IS_STRONG_UB_PRUNE_ENABLED                 = true;

    constexpr auto is_pow2(size_t val) noexcept -> bool{ //type coercion - static_assert subset - 

        return val != 0u && (val & (val - 1)) == 0u; 
    }

    constexpr auto pow2(size_t bit_offset) noexcept -> size_t{

        return size_t{1} << bit_offset;
    }

    template <class PtrType>
    constexpr auto region(PtrType ptr, size_t memregion_sz) noexcept -> PtrType{

        return {}; //TODOs:
    }

    template <class PtrType>
    constexpr auto region_offset(PtrType ptr, size_t memregion_sz) noexcept -> size_t{

        return {}; //TODOs:
    }

    template <class T, std::enable_if_t<std::is_trivial_v<T>, bool> = true>
    inline auto start_lifetime_as(void * buf) noexcept -> T{

        return std::launder(static_cast<T *>(std::memmove(buf, buf, sizeof(T))));
    } 

    template <class T, std::enable_if_t<std::is_fundamental_v<T>, bool> = true> //UB-check for current implementation - forced to be is_fundamental_v only - 
    inline auto start_lifetime_as_array(void * arr, size_t n) noexcept -> T *{

        return static_cast<T *>(arr);
    }

    constexpr auto align(uintptr_t buf, size_t alignment_sz) noexcept -> uintptr_t{

        assert(is_pow2(alignment_sz));

        uintptr_t fwd       = alignment_sz - 1;
        uintptr_t bitmask   = ~fwd;

        return (buf + fwd) & bitmask;
    }

    template <size_t ALIGNMENT_SZ>
    constexpr auto align(uintptr_t buf, const std::integral_constant<size_t, ALIGNMENT_SZ>) noexcept -> uintptr_t{ 

        static_assert(is_pow2(ALIGNMENT_SZ));

        constexpr uintptr_t FWD     = ALIGNMENT_SZ - 1;
        constexpr uintptr_t BITMASK = ~FWD;

        return (buf + FWD) & BITMASK;
    } 

    inline auto align(void * buf, size_t alignment_sz) noexcept -> void *{

        return reinterpret_cast<void *>(align(reinterpret_cast<uintptr_t>(buf), alignment_sz));
    } 

    inline auto align(const void * buf, size_t alignment_sz) noexcept -> const void *{

        return reinterpret_cast<const void *>(align(reinterpret_cast<uintptr_t>(buf), alignment_sz));
    } 

    template <size_t ALIGNMENT_SZ>
    inline auto align(void * buf, const std::integral_constant<size_t, ALIGNMENT_SZ>) noexcept -> void *{

        return reinterpret_cast<void *>(align(reinterpret_cast<uintptr_t>(buf), std::integral_constant<size_t, ALIGNMENT_SZ>{}));
    }

    template <size_t ALIGNMENT_SZ>
    inline auto align(const void * buf, const std::integral_constant<size_t, ALIGNMENT_SZ>) noexcept -> const void *{

        return reinterpret_cast<const void *>(align(reinterpret_cast<uintptr_t>(buf), std::integral_constant<size_t, ALIGNMENT_SZ>{}));
    }

    template <class T>
    static consteval auto simd_align_val() noexcept -> size_t{
        
        return std::max(static_cast<size_t>(SIMD_ALIGN_SIZE), static_cast<size_t>(alignof(T)));
    }

    static consteval auto simd_align_val_max() noexcept -> size_t{

        return simd_align_val<std::max_align_t>();
    }

    //TODOs: 
    
    // template <class T>
    // auto badvance() noexcept -> T{

    // }

    // template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true> //
    // static constexpr auto advance_fwd(T ptr, size_t sz) noexcept -> T{

    //     return static_cast<T>(static_cast<size_t>(ptr) + sz);
    // }

    // template <class T, std::enable_if_t<std::is_pointer_v<T>, bool> = true> //
    // static constexpr auto advance_fwd(T ptr, size_t sz) noexcept -> T{

    //     return ptr + sz;
    // }

    // template <class PtrType, std::enable_if_t<dg::is_ptr_v<PtrType>, bool> = true>
    // static constexpr auto is_nullptr(PtrType ptr) noexcept -> bool{

    //     return ptr == dg::pointer_limits<PtrType>::null_value();
    // }

    // template <class PtrType, std::enable_if_t<dg::is_ptr_v<PtrType>, bool> = true>
    // static constexpr auto is_validptr(PtrType ptr) noexcept -> bool{

    //     return ptr != dg::pointer_limits<PtrType>::null_value();
    // }

    static constexpr auto ptrcmp_aligned(...) noexcept -> int{

        return {};
    }
}

#endif 