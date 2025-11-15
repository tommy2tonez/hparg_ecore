#ifndef __NETWORK_MEMORY_UTILITY_H__
#define __NETWORK_MEMORY_UTILITY_H__

//define HEADER_CONTROL 1

#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <type_traits>
#include <bit>
#include <assert.h>
#include <memory>
#include <limits.h>
#include "network_pointer.h"
#include "stdx.h"

namespace dg::memult{

    static inline constexpr size_t HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE   = size_t{1} << 6;
    static inline constexpr size_t HARDWARE_CONSTRUCTIVE_INTERFERENCE_SIZE  = size_t{1} << 6;
    static inline constexpr size_t SIMD_ALIGN_SIZE                          = size_t{1} << 5;
    static inline constexpr bool IS_STRONG_UB_PRUNE_ENABLED                 = true;

    constexpr auto is_pow2(size_t val) noexcept -> bool{ //type coercion - static_assert subset - 

        return val != 0u && (val & static_cast<size_t>(val - 1u)) == 0u; 
    }

    constexpr auto pow2(size_t bit_offset) noexcept -> size_t{

        return size_t{1} << bit_offset;
    }

    constexpr auto aligned_modulo(size_t lhs, size_t rhs) -> size_t 
    {
        assert(is_pow2(rhs));

        return lhs & (rhs - 1u);
    }

    constexpr auto aligned_round_down(size_t lhs, size_t rhs) -> size_t
    {
        return lhs - aligned_modulo(lhs, rhs);
    } 

    template <class ptr_t>
    constexpr auto is_region(ptr_t ptr, size_t memregion_sz) noexcept -> bool{

        uintptr_t uptr = dg::pointer_cast<uintptr_t>(ptr);

        return aligned_modulo(static_cast<size_t>(uptr), memregion_sz) == 0u;
    }

    template <class ptr_t>
    constexpr auto region(ptr_t ptr, size_t memregion_sz) noexcept -> ptr_t{

        return dg::pointer_cast<ptr_t>(reinterpret_cast<uintptr_t>(aligned_round_down(static_cast<size_t>(dg::pointer_cast<uintptr_t>(ptr)), memregion_sz)));
    }

    template <class ptr_t>
    constexpr auto region_offset(ptr_t ptr, size_t memregion_sz) noexcept -> size_t{

        return aligned_modulo(static_cast<size_t>(dg::pointer_cast<uintptr_t>(ptr)), memregion_sz);
    }

    template <class ptr_t>
    constexpr auto distance(ptr_t first, ptr_t last) noexcept -> intmax_t{

        return dg::pointer_cast<intptr_t>(last) - dg::pointer_cast<intptr_t>(first);
    }

    template <class ptr_t>
    constexpr auto next(ptr_t ptr, intmax_t dist) noexcept -> ptr_t{

        return dg::pointer_cast<ptr_t>(dg::pointer_cast<intmax_t>(ptr) + dist);
    }

    template <class ptr_t>
    constexpr auto advance(ptr_t ptr, intmax_t dist) noexcept -> ptr_t{ //change sematics -> byte_advance - this is too ambiguous - and potentially buggy

        return memult::next(ptr, dist);
    }
    
    // template <class T, std::enable_if_t<std::is_fundamental_v<T>, bool> = true> //UB-check for current implementation - forced to be is_fundamental_v only - this is DEFINED in C but not in C++ 
    
    template <class T>
    inline __attribute__((always_inline)) auto start_lifetime_as_array(void * arr, size_t n) noexcept -> T *{

        return stdx::launder_pointer<T>(arr);
    }

    constexpr auto align(uintptr_t buf, size_t alignment_sz) noexcept -> uintptr_t{

        assert(is_pow2(alignment_sz));

        uintptr_t fwd       = alignment_sz - 1u;
        uintptr_t bitmask   = ~fwd;

        return (buf + fwd) & bitmask;
    }

    template <size_t ALIGNMENT_SZ>
    constexpr auto internal_align(uintptr_t buf, const std::integral_constant<size_t, ALIGNMENT_SZ>) noexcept -> uintptr_t{ 

        static_assert(is_pow2(ALIGNMENT_SZ));

        constexpr uintptr_t FWD     = ALIGNMENT_SZ - 1u;
        constexpr uintptr_t BITMASK = ~FWD;

        return (buf + FWD) & BITMASK;
    } 

    template <class T, size_t ALIGNMENT_SZ>
    constexpr auto align(T ptr, const std::integral_constant<size_t, ALIGNMENT_SZ>) noexcept -> T{

        return dg::pointer_cast<T>(internal_align(dg::pointer_cast<uintptr_t>(ptr), std::integral_constant<size_t, ALIGNMENT_SZ>{}));
    }

    template <class T>
    constexpr auto align(T ptr, size_t alignment_sz) noexcept -> T{

        return dg::pointer_cast<T>(align(dg::pointer_cast<uintptr_t>(ptr), alignment_sz));
    }

    template <class T>
    static consteval auto simd_align_val() noexcept -> size_t{
        
        return std::max(static_cast<size_t>(SIMD_ALIGN_SIZE), static_cast<size_t>(alignof(T)));
    }

    static consteval auto simd_align_val_max() noexcept -> size_t{

        return simd_align_val<std::max_align_t>();
    }

    template <class ptr_t>
    static constexpr auto ptrcmp(ptr_t lhs, ptr_t rhs) noexcept -> int{

        uintptr_t lhs_uptr  = dg::pointer_cast<uintptr_t>(lhs);
        uintptr_t rhs_uptr  = dg::pointer_cast<uintptr_t>(rhs);

        if (*std::launder(&lhs_uptr) < *std::launder(&rhs_uptr))
        {
            return -1;
        }

        if (*std::launder(&lhs_uptr) > *std::launder(&rhs_uptr))
        {
            return 1;
        }

        return 0;
    }

    template <class ptr_t>
    static constexpr auto ptrcmp_less_equal(ptr_t lhs, ptr_t rhs) noexcept -> bool{

        uintptr_t lhs_uptr  = dg::pointer_cast<uintptr_t>(lhs);
        uintptr_t rhs_uptr  = dg::pointer_cast<uintptr_t>(rhs); 

        return *std::launder(&lhs_uptr) <= *std::launder(&rhs_uptr);
    }

    template <class ptr_t>
    static constexpr auto ptrcmp_equal(ptr_t lhs, ptr_t rhs) noexcept -> bool{

        uintptr_t lhs_uptr  = dg::pointer_cast<uintptr_t>(lhs);
        uintptr_t rhs_uptr  = dg::pointer_cast<uintptr_t>(rhs);  

        return *std::launder(&lhs_uptr) == *std::launder(&rhs_uptr);
    }

    template <class ptr_t>
    static constexpr auto ptrcmp_less(ptr_t lhs, ptr_t rhs) noexcept -> bool{

        uintptr_t lhs_uptr  = dg::pointer_cast<uintptr_t>(lhs);
        uintptr_t rhs_uptr  = dg::pointer_cast<uintptr_t>(rhs);

        return *std::launder(&lhs_uptr) < *std::launder(&rhs_uptr);
    }

    static inline constexpr auto ptrcmpless_lambda = []<class U, class T>(U lhs, T rhs) noexcept -> bool{
        return ptrcmp_less(lhs, rhs);
    };
}

#endif 