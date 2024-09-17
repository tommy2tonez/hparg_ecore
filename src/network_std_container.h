#ifndef __DG_NETWORK_HASHER_H__
#define __DG_NETWORK_HASHER_H__

#include <type_traits> 
#include "network_type_traits_x.h"
#include "network_trivial_serializer.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include "network_allocation.h"

namespace dg::network_std_container{

    static constexpr auto rotl64(uint64_t x, int8_t r) -> uint64_t{
    
        return (x << r) | (x >> (64 - r));
    }

    static constexpr auto fmix64(uint64_t k) -> uint64_t{
        
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccd;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53;
        k ^= k >> 33;

        return k;
    }

    static constexpr auto murmur_hash(const char * buf, size_t len, const uint32_t seed = 0xFF) -> uint64_t{
    
        const size_t nblocks = len / 16;

        uint64_t h1 = seed;
        uint64_t h2 = seed;

        const uint64_t c1 = 0x87c37b91114253d5;
        const uint64_t c2 = 0x4cf5ad432745937f;

        for(size_t i = 0; i < nblocks; i++)
        {   
            uint64_t k1{};
            uint64_t k2{};

            dg::network_trivial_serializer::deserialize_into(k1, buf + (i * 2 + 0) * sizeof(uint64_t));
            dg::network_trivial_serializer::deserialize_into(k2, buf + (i * 2 + 1) * sizeof(uint64_t));

            k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
            h1 = rotl64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;
            k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;
            h2 = rotl64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
        }

        const char * tail = buf + nblocks*16;

        uint64_t k1 = 0;
        uint64_t k2 = 0;

        switch(len & 15)
        {
            case 15: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[14])) << 48;
            case 14: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[13])) << 40;
            case 13: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[12])) << 32;
            case 12: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[11])) << 24;
            case 11: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[10])) << 16;
            case 10: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[9])) << 8;
            case  9: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[8])) << 0;
                    k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;

            case  8: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[7])) << 56;
            case  7: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[6])) << 48;
            case  6: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[5])) << 40;
            case  5: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[4])) << 32;
            case  4: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[3])) << 24;
            case  3: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[2])) << 16;
            case  2: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[1])) << 8;
            case  1: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[0])) << 0;
                    k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
        };

        h1 ^= static_cast<uint64_t>(len); 
        h2 ^= static_cast<uint64_t>(len);

        h1 += h2;
        h2 += h1;

        h1 = fmix64(h1);
        h2 = fmix64(h2);

        h1 += h2;
        h2 += h1;

        return h1;
    }

    template <size_t LEN, size_t SEED = 0xFF>
    static constexpr auto murmur_hash(const char * buf, const std::integral_constant<size_t, LEN>, const std::integral_constant<uint64_t, SEED> seed = std::integral_constant<size_t, SEED>{}) -> uint64_t{ //this should be compiler responsibility - yet reimplementation for now (because of compiler limitation)

        const size_t nblocks = LEN / 16;

        uint64_t h1 = seed;
        uint64_t h2 = seed;

        const uint64_t c1 = 0x87c37b91114253d5;
        const uint64_t c2 = 0x4cf5ad432745937f;

        for(size_t i = 0; i < nblocks; i++)
        {   
            uint64_t k1{};
            uint64_t k2{};

            dg::network_trivial_serializer::deserialize_into(k1, buf + (i * 2 + 0) * sizeof(uint64_t));
            dg::network_trivial_serializer::deserialize_into(k2, buf + (i * 2 + 1) * sizeof(uint64_t));

            k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
            h1 = rotl64(h1,27); h1 += h2; h1 = h1*5+0x52dce729;
            k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;
            h2 = rotl64(h2,31); h2 += h1; h2 = h2*5+0x38495ab5;
        }

        const char * tail = buf + nblocks*16;

        uint64_t k1 = 0;
        uint64_t k2 = 0;

        switch(LEN & 15)
        {
            case 15: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[14])) << 48;
            case 14: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[13])) << 40;
            case 13: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[12])) << 32;
            case 12: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[11])) << 24;
            case 11: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[10])) << 16;
            case 10: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[9])) << 8;
            case  9: k2 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[8])) << 0;
                    k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;

            case  8: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[7])) << 56;
            case  7: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[6])) << 48;
            case  6: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[5])) << 40;
            case  5: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[4])) << 32;
            case  4: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[3])) << 24;
            case  3: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[2])) << 16;
            case  2: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[1])) << 8;
            case  1: k1 ^= static_cast<uint64_t>(std::bit_cast<uint8_t>(tail[0])) << 0;
                    k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
        };

        h1 ^= static_cast<uint64_t>(LEN); 
        h2 ^= static_cast<uint64_t>(LEN);

        h1 += h2;
        h2 += h1;

        h1 = fmix64(h1);
        h2 = fmix64(h2);

        h1 += h2;
        h2 += h1;

        return h1;
    } 

    constexpr auto hash_bytes(const char * inp, size_t n) noexcept -> size_t{

        return murmur_hash(inp, n);
    }      

    template <size_t N>
    constexpr auto hash_bytes(const char * inp, const std::integral_constant<size_t, N> n) noexcept -> size_t{

        return murmur_hash(inp, n);
    }
    
    struct empty{};

    template <class T, std::enable_if_t<dg::network_trivial_serializer::is_serializable_v<T>, bool> = true>
    struct trivial_reflectable_hasher{

        constexpr auto operator()(const T& value) const noexcept -> size_t{

            constexpr size_t SZ         = dg::network_trivial_serializer::size(T{});  
            std::array<char, SZ> buf    = {};
            dg::network_trivial_serializer::serialize_into(buf.data(), value);

            return hash_bytes(buf.data(), std::integral_constant<size_t, SZ>{});
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
    using vector = std::vector<T, dg::network_allocation::NoExceptAllocator<T>>;

    using string = std::basic_string<char, std::char_traits<char>, dg::network_allocation::NoExceptAllocator<char>>;
}

#endif