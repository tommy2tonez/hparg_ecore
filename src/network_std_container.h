#ifndef __DG_NETWORK_STD_CONTAINER_H__
#define __DG_NETWORK_STD_CONTAINER_H__

#include <type_traits> 
#include "network_type_traits_x.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include "network_allocation.h"
#include "network_hash.h"

namespace dg::network_std_container{

    struct empty{};

    template <class T, std::enable_if_t<dg::network_trivial_serializer::is_serializable_v<T>, bool> = true>
    struct trivial_reflectable_hasher{

        constexpr auto operator()(const T& value) const noexcept -> size_t{

            constexpr size_t SZ         = dg::network_trivial_serializer::size(T{});  
            std::array<char, SZ> buf    = {};
            dg::network_trivial_serializer::serialize_into(buf.data(), value);

            return dg::network_hash::hash_bytes(buf.data(), std::integral_constant<size_t, SZ>{});
        }
    };

    template <class T, std::enable_if_t<dg::network_trivial_serializer::is_serializable_v<T>, bool> = true>
    struct trivial_reflectable_equalto{

        constexpr auto operator()(const T& lhs, const T& rhs) const noexcept -> bool{

            constexpr size_t SZ             = dg::network_trivial_serializer::size(T{}); 
            std::array<char, SZ> lhs_buf    = {};
            std::array<char, SZ> rhs_buf    = {};
            dg::network_trivial_serializer::serialize_into(lhs_buf.data(), lhs);
            dg::network_trivial_serializer::serialize_into(rhs_buf.data(), rhs);

            return lhs_buf == rhs_buf;
        }
    };

    template <class T, class = void>
    struct is_std_hashable: std::false_type{};

    template <class T>
    struct is_std_hashable<T, std::void_t<decltype(std::declval<std::hash<T>&>()(std::declval<const T&>()))>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_std_hashable_v = is_std_hashable<T>::value;

    template <class T, class = void>
    struct is_std_equalto_able: std::false_type{};

    template <class T>
    struct is_std_equalto_able<T, std::void_t<decltype(std::declval<const T&>() == std::declval<const T&>())>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_std_equalto_able_v = is_std_equalto_able<T>::value;

    template <class T, class = void>
    struct stdhasher_or_empty{};

    template <class T>
    struct stdhasher_or_empty<T, std::void_t<std::enable_if_t<is_std_hashable_v<T>>>>: std::hash<T>{};

    template <class T, class = void>
    struct trivial_reflectable_hasher_or_empty{};

    template <class T>
    struct trivial_reflectable_hasher_or_empty<T, std::void_t<trivial_reflectable_hasher<T>>>: trivial_reflectable_hasher<T>{};

    template <class T, class = void>
    struct std_equalto_or_empty{};

    template <class T>
    struct std_equalto_or_empty<T, std::void_t<std::enable_if_t<is_std_equalto_able_v<T>>>>: std::equal_to<T>{};

    template <class T, class = void>
    struct trivial_reflectable_equalto_or_empty{};

    template <class T>
    struct trivial_reflectable_equalto_or_empty<T, std::void_t<trivial_reflectable_equalto<T>>>: trivial_reflectable_equalto<T>{};

    template <class T>
    using hasher = std::conditional_t<is_std_hashable_v<T>,
                                      stdhasher_or_empty<T>,
                                      std::conditional_t<dg::network_trivial_serializer::is_serializable_v<T>,
                                                         trivial_reflectable_hasher_or_empty<T>,
                                                         empty>>;

    template <class T>
    using equal_to = std::conditional_t<is_std_equalto_able_v<T>, 
                                        std_equalto_or_empty<T>,
                                        std::conditional_t<dg::network_trivial_serializer::is_serializable_v<T>,
                                                           trivial_reflectable_equalto_or_empty<T>,
                                                           empty>>;

    template <class T>
    struct optional_type{
        using type = T;
    };

    template <>
    struct optional_type<empty>{};

    template <class T>
    using optional_type_t = typename optional_type<T>::type; 

    template <class T>
    using unordered_set = std::unordered_set<T, optional_type_t<hasher<T>>, optional_type_t<equal_to<T>>, dg::network_allocation::NoExceptAllocator<T>>;

    template <class Key, class Value>
    using unordered_map = std::unordered_map<Key, Value, optional_type_t<hasher<Key>>, optional_type_t<equal_to<Key>>, dg::network_allocation::NoExceptAllocator<std::pair<const Key, Value>>>;

    template <class T>
    using vector        = std::vector<T, dg::network_allocation::NoExceptAllocator<T>>;

    template <class T>
    using deque         = std::deque<T, dg::network_allocation::NoExceptAllocator<T>>;

    using string        = std::basic_string<char, std::char_traits<char>, dg::network_allocation::NoExceptAllocator<char>>;
}

#endif