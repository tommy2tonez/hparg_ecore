#ifndef __NETWORK_LANG_X_H__
#define __NETWORK_LANG_X_H__

#include <type_traits>

namespace dg::network_lang_x{

    template <class T, class T1, class = void>
    struct is_static_castable: std::false_type{};

    template <class T, class T1>
    struct is_static_castable<T, T1, std::void_t<decltype(static_cast<T>(std::declval<T1>()))>>: std::true_type{};

    template <class T, class T1, class = void>
    struct is_reinterpret_castable: std::false_type{};

    template <class T, class T1, class = void>
    struct is_reinterpret_castable<T, T1, std::void_t<decltype(reinterpret_cast<T>(std::declval<T1>()))>>: std::true_type{};

    template <class T, class T1>
    static inline constexpr bool is_static_castable_v       = is_static_castable<T, T1>::value;

    template <class T, class T1>
    static inline constexpr bool is_reinterpret_castable_v  = is_reinterpret_castable<T, T1>::value; 

    template <class T, class T1>
    static inline auto static_cast_nothrow(T1&& value) noexcept{

        static_assert(noexcept(static_cast<T>(std::forward<T1>(value))));
        return static_cast<T>(std::forward<T1>(value));
    }

    template <class T, class T1>
    static inline auto reinterpret_cast_nothrow(T1&& value) noexcept{

        static_assert(noexcept(reinterpret_cast<T>(std::forward<T1>(value))));
        return reinterpret_cast<T>(std::forward<T1>(value));
    }

    template <class T, class T1>
    static inline auto static_reinterpret_cast_nothrow(T1&& value) noexcept{

        if constexpr(is_static_castable_v<T, T1>){
            static_assert(noexcept(static_cast<T>(std::forward<T1>(value))));
            return static_cast<T>(std::forward<T1>(value));
        } else{
            static_assert(noexcept(reinterpret_cast<T>(std::forward<T1>(value))));
            return reinterpret_cast<T>(std::forward<T1>(value));
        }
    }
}

#endif