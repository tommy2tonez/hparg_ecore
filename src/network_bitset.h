#ifndef __NETWORK_BITSET_H__
#define __NETWORK_BITSET_H__

#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

namespace dg::network_unsigned_bitset{

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto true_toggle(size_t idx) noexcept -> T{

        return T{1} << idx;
    } 

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto false_mask(size_t idx) noexcept -> T{

        return ~T{} ^ true_toggle(idx);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr void add(T& data, size_t value) noexcept{

        constexpr size_t BIT_COUNT = sizeof(T) * CHAR_BIT;
        assert(value < BIT_COUNT);
        data |= true_toggle<T>(value);
    }

    template <class T, size_t VALUE, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr void add(T& data, const std::integral_constant<size_t, VALUE>) noexcept{

        constexpr size_t BIT_COUNT = sizeof(T) * CHAR_BIT;
        static_assert(VALUE < BIT_COUNT);
        data |= true_toggle<T>(VALUE);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr void erase(T& data, size_t value) noexcept{

        constexpr size_t BIT_COUNT = sizeof(T) * CHAR_BIT;
        assert(value < BIT_COUNT);
        data &= false_mask<T>(value);
    }    

    template <class T, size_t VALUE, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr void erase(T& data, const std::integral_constant<size_t, VALUE>) noexcept{

        constexpr size_t BIT_COUNT = sizeof(T) * CHAR_BIT;
        static_assert(VALUE < BIT_COUNT);
        data &= false_mask<T>(VALUE);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr void pop(T& data, size_t& rs) noexcept{
        
        rs      = std::countr_zero(data);
        data    &= data - 1;
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto empty(T data) noexcept -> bool{

        return data == T{};
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static consteval auto max(T) noexcept -> size_t{

        return static_cast<size_t>(sizeof(T)) * CHAR_BIT - 1;
    }
} 

#endif 