#ifndef __NETWORK_TYPE_TRAITS_X_H__
#define __NETWORK_TYPE_TRAITS_X_H__

#include <type_traits>
#include <tuple>
#include <stdint.h>
#include <stddef.h>
#include <utility>

namespace dg::network_type_traits_x{
    
    //good - std-compliant

    template <class ...Args>
    struct tags{};

    struct empty{}; 

    template <class T>
    struct base_type: std::enable_if<true, T>{};

    template <class T>
    struct base_type<const T>: base_type<T>{};

    template <class T>
    struct base_type<volatile T>: base_type<T>{};

    template <class T>
    struct base_type<T&>: base_type<T>{};

    template <class T>
    struct base_type<T&&>: base_type<T>{};

    template <class T>
    using base_type_t = typename base_type<T>::type;
    
    template <class T, class = void>
    struct is_base_type: std::false_type{}; 

    template <class T>
    struct is_base_type<T, std::void_t<std::enable_if_t<std::is_same_v<T, base_type_t<T>>>>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_base_type_v = is_base_type<T>::value;

    template <class T, class = void>
    struct is_tuple: std::false_type{};

    template <class T>
    struct is_tuple<T, std::void_t<decltype(std::tuple_size<T>::value)>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_tuple_v = is_tuple<T>::value;

    template <class T, class = void>
    struct is_stdprimitive_integer: std::false_type{};

    template <class T>
    struct is_stdprimitive_integer<T, std::void_t<std::enable_if_t<std::numeric_limits<T>::is_integer>>>: std::true_type{}; 

    template <class T>
    static inline constexpr bool is_stdprimitive_integer_v = is_stdprimitive_integer<T>::value; 

    template <class T>
    struct mono_reduction_type_helper{};

    template <class First, class Second, class ...Args>
    struct mono_reduction_type_helper<tags<First, Second, Args...>>: std::conditional_t<std::is_same_v<First, Second>, 
                                                                                        mono_reduction_type_helper<tags<Second, Args...>>,
                                                                                        empty>{}; //this is not sfinae - consider std::void_t<> - this is a bad hack

    template <class T>
    struct mono_reduction_type_helper<tags<T>>: std::enable_if<true, T>{};

    template <class ...Args>
    struct mono_reduction_type: mono_reduction_type_helper<tags<Args...>>{};

    template <class ...Args>
    using mono_reduction_type_t = typename mono_reduction_type<Args...>::type; 

    template <class Functor, class Tag, class = void>
    struct is_nothrow_invokable_helper: std::false_type{};

    template <class Functor, class ...Args>
    struct is_nothrow_invokable_helper<Functor, tags<Args...>, std::void_t<std::enable_if_t<noexcept(std::declval<Functor>()(std::declval<Args>()...))>>>: std::true_type{}; 

    template <class Functor, class ...Args>
    struct is_nothrow_invokable: is_nothrow_invokable_helper<Functor, tags<Args...>>{}; 

    template <class Functor, class ...Args>
    static inline constexpr bool is_nothrow_invokable_v = is_nothrow_invokable<Functor, Args...>::value; 
} 

#endif