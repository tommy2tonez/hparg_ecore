#ifndef __NETWORK_XMATH_HOST_H__
#define __NETWORK_XMATH_HOST_H__

#include <stddef.h>
#include <stdint.h>
#include <stdfloat>
#include <math.h> 

namespace dg::network_xmath_host{
    
    template <class T>
    struct is_std_float: std::false_type{};

    template <>
    struct is_std_float<std::float16_t>: std::true_type{};

    template <>
    struct is_std_float<std::float32_t>: std::true_type{};

    template <>
    struct is_std_float<std::float64_t>: std::true_type{};

    template <>
    struct is_std_float<std::bfloat16_t>: std::true_type{};

    template <class T>
    static inline constexpr bool is_std_float_v = is_std_float<T>::value;
    
    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true> 
    inline auto sign(T value) noexcept -> T{
        
        return (value > 0) - (value < 0);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto exp(T value) noexcept -> T{

        return std::exp(value);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto log(T value) noexcept -> T{

        return std::log(value); //nan
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto abs(T value) noexcept -> T{

        return std::abs(value);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto cos(T value) noexcept -> T{

        return std::cos(value);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto acos(T value) noexcept -> T{

        return std::acos(value);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto sin(T value) noexcept -> T{

        return std::sin(value);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto asin(T value) noexcept -> T{

        return std::asin(value);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto tan(T value) noexcept -> T{

        return std::tan(value);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto atan(T value) noexcept -> T{

        return std::atan(value);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto sqrt(T value) noexcept -> T{

        return std::sqrt(value); //nan
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto invsqrt(T value) noexcept -> T{

        return 1 / std::sqrt(value); //nan
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto negative(T value) noexcept -> T{

        return -value;
    } 

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto add(T lhs, T rhs) noexcept -> T{

        return lhs + rhs; //sat
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto sub(T lhs, T rhs) noexcept -> T{

        return lhs - rhs; //sat
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto mul(T lhs, T rhs) noexcept -> T{

        return lhs * rhs;
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto div(T lhs, T rhs) noexcept -> T{

        return lhs / rhs;
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto pow(T lhs, T rhs) noexcept -> T{

        return std::pow(lhs, rhs);
    }

    template <class T, size_t RHS_VALUE, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto pow(T lhs, const std::integral_constant<size_t, RHS_VALUE>) noexcept -> T{

        return std::pow(lhs, RHS_VALUE);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto fma(T first, T second, T third){

        return first * second + third;
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto min(T lhs, T rhs) noexcept -> T{

        return std::min(lhs, rhs);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto max(T lhs, T rhs) noexcept -> T{

        return std::max(lhs, rhs);
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto eqcmp_mul(T lcmp, T rcmp, T val) -> T{

        return (lcmp == rcmp) * val;
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto bitwise_or(T lhs, T rhs) noexcept -> T{

        static_assert(std::numeric_limits<T>::has_quiet_NaN);
        return std::numeric_limits<T>::quiet_NaN();
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto bitwise_and(T lhs, T rhs) noexcept -> T{

        static_assert(std::numeric_limits<T>::has_quiet_NaN);
        return std::numeric_limits<T>::quiet_NaN();
    }

    template <class T, std::enable_if_t<is_std_float_v<T>, bool> = true>
    inline auto bitwise_xor(T lhs, T rhs) noexcept -> T{

        static_assert(std::numeric_limits<T>::has_quiet_Nan);
        return std::numeric_limits<T>::quiet_NaN();
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto sign(T value) noexcept -> T{
        
        return 1;
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto exp(T value) noexcept -> T{

        return std::exp(value);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto log(T value) noexcept -> T{

        return std::log(value); //nan
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto abs(T value) noexcept -> T{

        return std::abs(value);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto cos(T value) noexcept -> T{

        return std::cos(value);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto acos(T value) noexcept -> T{

        return std::acos(value);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto sin(T value) noexcept -> T{

        return std::sin(value);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto asin(T value) noexcept -> T{

        return std::asin(value);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto tan(T value) noexcept -> T{

        return std::tan(value);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto atan(T value) noexcept -> T{

        return std::atan(value);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto sqrt(T value) noexcept -> T{

        return std::sqrt(value); //nan
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto invsqrt(T value) noexcept -> T{

        return 1 / std::sqrt(value); //nan
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto negative(T value) noexcept -> T{

        return 0;
    } 

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto add(T lhs, T rhs) noexcept -> T{

        return lhs + rhs; //sat
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto sub(T lhs, T rhs) noexcept -> T{

        return lhs - rhs; //sat
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto mul(T lhs, T rhs) noexcept -> T{

        return lhs * rhs; //sat
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto div(T lhs, T rhs) noexcept -> T{

        return lhs / rhs; //inf
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto pow(T lhs, T rhs) noexcept -> T{

        return std::pow(lhs, rhs);
    }

    template <class T, size_t RHS_VALUE, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto pow(T lhs, const std::integral_constant<size_t, RHS_VALUE>) noexcept -> T{

        return std::pow(lhs, RHS_VALUE);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto fma(T first, T second, T third){

        return first * second + third; //sat
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto min(T lhs, T rhs) noexcept -> T{

        return std::min(lhs, rhs);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto max(T lhs, T rhs) noexcept -> T{

        return std::max(lhs, rhs);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto eqcmp_mul(T lcmp, T rcmp, T val) noexcept -> T{

        return ((lcmp ^ rcmp) == 0u) * val;
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto bitwise_or(T lhs, T rhs) noexcept -> T{

        return lhs | rhs;
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto bitwise_and(T lhs, T rhs) noexcept -> T{

        return lhs & rhs;
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    inline auto bitwise_xor(T lhs, T rhs) noexcept -> T{

        return lhs ^ rhs;
    }

}

#endif