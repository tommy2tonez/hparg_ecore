#ifndef __NETWORK_TYPE_TRAITS_X_H__
#define __NETWORK_TYPE_TRAITS_X_H__

#include <type_traits>
#include <tuple>
#include <stdint.h>
#include <stddef.h>

namespace dg::network_type_traits_x{
    
    struct immutable_resource_handle_tag{};
    
    template <class ...Args>
    struct tags{};

    struct std_tuple_tag{}; 
    struct std_pair_tag{};
    struct std_array_tag{};

    template <class T>
    struct base_type{
        using type = T;
    };

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
    struct is_tuple: std::false_type{};

    template <class T>
    struct is_tuple<T, std::void_t<decltype(std::tuple_size<T>::value)>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_tuple_v = is_tuple<T>::value;

    template <class T>
    struct container_type{};

    template <class ...Args>
    struct container_type<std::tuple<Args...>>{
        using type = std_tuple_tag;
    };

    template <class ...Args>
    struct container_type<std::pair<Args...>>{
        using type = std_pair_tag;
    };

    template <class T, size_t SZ>
    struct container_type<std::array<T, SZ>>{
        using type = std_array_tag;
    };

    template <class T>
    using container_type_t = container_type<T>::type;

    template <class T>
    struct mono_reduction_type_helper{};

    template <class First, class Second, class ...Args>
    struct mono_reduction_type_helper<tags<First, Second, Args...>>: std::enable_if_t<std::is_same_v<First, Second>, mono_reduction_type_helper<tags<Second, Args...>>>{};

    template <class T>
    struct mono_reduction_type_helper<tags<T>>{
        using type = T;
    };

    template <class ...Args>
    struct mono_reduction_type: mono_reduction_type_helper<tags<Args...>>{};

    template <class ...Args>
    using mono_reduction_type_t = typename mono_reduction_type<Args...>::type; 

    template <class T, class = void>
    struct is_immutable_resource_handle: std::false_type{}; 

    template <class T>
    struct is_immutable_resource_handle<T, std::void_t<decltype(static_cast<immutable_resource_handle_tag&>(std::declval<T&>()))>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_immutable_resource_handle_v = is_immutable_resource_handle<T>::value;

    template <class Functor, class Tup, class = void>
    struct is_nothrow_invokable_helper: std::false_type{};

    template <class Functor, class ...Args>
    struct is_nothrow_invokable_helper<Functor, tags<Args...>, std::enable_if_t<noexcept(std::declval<Functor>()(std::declval<Args>()...))>>: std::true_type{}; 

    template <class Functor, class ...Args>
    struct is_nothrow_invokable: is_nothrow_invokable_helper<Functor, tags<Args...>>{}; 

    template <class Functor, class ...Args>
    static inline constexpr bool is_nothrow_invokable_v = is_nothrow_invokable<Functor, Args...>::value; 

} 

#endif