#ifndef __DG_NETWORK_HASH_FACTORY__
#define __DG_NETWORK_HASH_FACTORY__

#include <functional>
#include "network_trivial_serializer.h"
#include "network_hash.h"

namespace dg::network_hash_factory
{
    template <class T>
    using std_hasher = std::hash<T>;

    template <class T, class = void>
    struct is_std_hashable: std::false_type{};

    template <class T>
    struct is_std_hashable<T, std::void_t<decltype(std::declval<std::hash<T>&>()(std::declval<const T&>()))>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_std_hashable_v = is_std_hashable<T>::value;

    template <class T, std::enable_if_t<dg::network_trivial_serializer::is_serializable_v<T>, bool> = true>
    struct trivial_reflectable_hasher{

        constexpr auto operator()(const T& value) const noexcept -> size_t{

            constexpr size_t SZ = dg::network_trivial_serializer::size(T{});  
            std::array<char, SZ> buf = {};
            dg::network_trivial_serializer::serialize_into(buf.data(), value);

            return dg::network_hash::hash_bytes(buf.data(), std::integral_constant<size_t, SZ>{});
        }
    };

    template <class T>
    static inline constexpr bool is_trivial_hashable_v = dg::network_trivial_serializer::is_serializable_v<T>;

    template <class = void>
    static inline constexpr bool FALSE_VAL = false;

    template <class T>
    struct type_container{
        using type = T;
    };

    template <>
    struct type_container<void>{};

    template <class T>
    auto internal_default_hasher_chooser(){

        if constexpr(is_std_hashable_v<T>){
            return type_container<std_hasher<T>>();
        } else if constexpr(is_trivial_hashable_v<T>){
            return type_container<trivial_reflectable_hasher<T>>();
        } else{
            return type_container<void>();
        }
    }

    template <class T>
    using default_hasher = typename decltype(internal_default_hasher_chooser<T>())::type;

    template <class T>
    using std_equal_to = std::equal_to<T>;

    template <class T, class = void>
    struct is_std_equal_to_able: std::false_type{};

    template <class T>
    struct is_std_equal_to_able<T, std::void_t<decltype(std::declval<const T&>() == std::declval<const T&>())>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_std_equal_to_able_v = is_std_equal_to_able<T>::value;

    template <class T, std::enable_if_t<dg::network_trivial_serializer::is_serializable_v<T>, bool> = true>
    struct trivial_reflectable_equal_to{

        constexpr auto operator()(const T& lhs, const T& rhs) const noexcept -> bool{

            return dg::network_trivial_serializer::reflectible_is_equal(lhs, rhs);
        }
    };

    template <class T>
    static inline constexpr bool is_trivial_equal_to_able_v = dg::network_trivial_serializer::is_serializable_v<T>;

    template <class T>
    auto internal_default_equal_to_chooser(){

        if constexpr(is_std_equal_to_able_v<T>){
            return type_container<std_equal_to<T>>();
        } else if constexpr(is_trivial_equal_to_able_v<T>){
            return type_container<trivial_reflectable_equal_to<T>>();
        } else{
            return type_container<void>();
        }
    }

    template <class T>
    using default_equal_to = typename decltype(internal_default_equal_to_chooser<T>())::type;
}

#endif