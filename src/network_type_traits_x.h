#ifndef __NETWORK_TYPE_TRAITS_X_H__
#define __NETWORK_TYPE_TRAITS_X_H__

#include <type_traits>
#include <tuple>

namespace dg::network_type_traits_x{
    
    struct immutable_resource_handle_tag{};
    
    template <class ...Args>
    struct tags{};

    template <class T, class = void>
    struct is_immutable_resource_handle: std::false_type{}; 

    template <class T>
    struct is_immutable_resource_handle<T, std::void_t<decltype(static_cast<immutable_resource_handle_tag&>(std::declval<T&>()))>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_immutable_resource_handle_v = is_immutable_resource_handle<T>::value;

    template <class Functor, class Tup, class = void>
    struct is_nothrow_invokable_helper: std::false_type{};

    template <class Functor, class ...Args>
    struct is_nothrow_invokable_helper<Functor, std::tuple<Args...>, std::enable_if_t<noexcept(std::declval<Functor>()(std::declval<Args>()...))>>: std::true_type{}; 

    template <class Functor, class ...Args>
    struct is_nothrow_invokable: is_nothrow_invokable_helper<Functor, std::tuple<Args...>>{}; 

    template <class Functor, class ...Args>
    static inline constexpr bool is_nothrow_invokable_v = is_nothrow_invokable<Functor, Args...>::value; 

} 

#endif