#ifndef __DG_NETWORK_HASHER_H__
#define __DG_NETWORK_HASHER_H__

#include <type_traits> 
#include "network_type_traits_x.h"
#include "network_compact_serializer.h"
#include "network_trivial_serializer.h"

namespace dg::network_hasher{

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
            case 15: k2 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[14])) << 48;
            case 14: k2 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[13])) << 40;
            case 13: k2 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[12])) << 32;
            case 12: k2 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[11])) << 24;
            case 11: k2 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[10])) << 16;
            case 10: k2 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[9])) << 8;
            case  9: k2 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[8])) << 0;
                    k2 *= c2; k2  = rotl64(k2,33); k2 *= c1; h2 ^= k2;

            case  8: k1 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[7])) << 56;
            case  7: k1 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[6])) << 48;
            case  6: k1 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[5])) << 40;
            case  5: k1 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[4])) << 32;
            case  4: k1 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[3])) << 24;
            case  3: k1 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[2])) << 16;
            case  2: k1 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[1])) << 8;
            case  1: k1 ^= ((uint64_t)std::bit_cast<uint8_t>(tail[0])) << 0;
                    k1 *= c1; k1  = rotl64(k1,31); k1 *= c2; h1 ^= k1;
        };

        h1 ^= len; h2 ^= len;

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

    template <class T, std::enable_if_t<dg::network_compact_serializer::is_serializable_v<T>>>
    struct reflectable_hasher{

        auto operator()(const T& value) const noexcept -> size_t{ //I don't want to put noexcept here - global memory exhaustion is not recoverable (if global memory exhaustion happens then reservoir should be used - it is the last of the mohicans) 

            size_t sz = dg::network_compact_serializer::size(value); 
            std::vector<char> buf(sz);
            dg::network_compact_serializer::serialize_into(buf.get(), value);

            return hash_bytes(buf.get(), sz);
        }
    };

    template <class T, std::enable_if_t<dg::network_trivial_serializer::is_serializable_v<T>>>
    struct trivial_reflectable_hasher{

        constexpr auto operator()(const T& value) const noexcept -> size_t{

            constexpr size_t sz         = dg::network_trivial_serializer::size(value);  
            std::array<char, sz> buf    = {};
            dg::network_trivial_serializer::serialize_into(buf.get(), value);

            return hash_bytes(buf.get(), sz);
        }
    };

    template <class T, class = void>
    struct stdhasher_or_empty{};

    template <class T>
    struct stdhasher_or_empty<T, std::void_t<std::hash<T>>>: std::hash<T>{};

    template <class T, class = void>
    struct trivial_reflectable_hasher_or_empty{};

    template <class T>
    struct trivial_reflectable_hasher_or_empty<T, std::void_t<trivial_reflectable_hasher<T>>>: trivial_reflectable_hasher<T>{};

    template <class T, class = void>
    struct reflectable_hasher_or_empty{};

    template <class T>
    struct reflectable_hasher_or_empty<T, std::void_t<reflectable_hasher<T>>>: reflectable_hasher<T>{};

    template <class T>
    using hasher = std::conditional_t<dg::network_type_traits_x::is_std_hashable_v<T>,
                                      stdhasher_or_empty<T>,
                                      std::conditional_t<dg::network_compact_serializer::is_trivial_constexpr_serializable_v<T>,
                                                         trivial_reflectable_hasher_or_empty<T>,
                                                         std::conditional_t<dg::network_compact_serializer::is_serializable_v<T>,
                                                                            reflectable_hasher_or_empty<T>,
                                                                            dg::network_type_traits_x::empty>>>;  

} 

#endif