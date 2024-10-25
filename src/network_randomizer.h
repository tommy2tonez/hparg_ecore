#ifndef __NETWORK_RANDOMIZER_H__
#define __NETWORK_RANDOMIZER_H__

#include <stdlib.h>
#include <stdint.h>
#include <utility>
#include "network_memory_utility.h"
#include "network_concurrency.h" 
#include <limits.h>
#include <bit>
#include <random>
#include "stdx.h"

namespace dg::network_randomizer{

    struct BitRandomizer{
        
        private:

            struct RandomizationUnit{
                uint64_t value;
                size_t bit_precision;
                std::mt19937_64 randomizer;
            };

            static inline std::vector<RandomizationUnit> table = []{
                std::vector<RandomizationUnit> rs{};

                for (size_t i = 0u; i < dg::network_concurrency::THREAD_COUNT; ++i){
                    size_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count(); 
                    std::mt19937_64 randomizer{seed};
                    rs.push_back(RandomizationUnit{randomizer(), sizeof(uint64_t) * CHAR_BIT, randomizer});
                }

                return rs;
            }();

            static inline void re_randomize(RandomizationUnit& random_unit) noexcept{

                random_unit.value           = random_unit.randomizer();
                random_unit.bit_precision   = sizeof(uint64_t) * CHAR_BIT;
            }

        public: 

            template <T RANGE_SZ>
            static inline auto randomize_bit(const std::integral_constant<size_t, BIT_SIZE>) noexcept -> uint64_t{

                static_assert(BIT_SIZE != 0);
                static_assert(BIT_SIZE <= sizeof(uint64_t) * CHAR_BIT);

                RandomizationUnit& unit = table[dg::network_concurrency::this_thread_id()];

                if (unit.bit_precision < BIT_SIZE){
                    re_randomize(unit);
                }

                uint64_t ret_value  = unit.value & low<uint64_t>(std::integral_constant<size_t, BIT_SIZE>{});
                unit.bit_precision  -= BIT_SIZE;
                unit.value          >>= BIT_SIZE;;

                return ret_value;
            }
    };

    template <size_t RANGE_SZ>
    auto randomize_range(const std::integral_constant<size_t, RANGE_SZ>) noexcept -> size_t{

        static_assert(RANGE_SZ != 0u);

        constexpr size_t BIT_SIZE = to_bit_size(RANGE_SZ); 
        uint64_t rs = BitRandomizer::randomize_bit(std::integral_constant<size_t, BIT_SIZE>{});

        if constexpr(is_pow2(RANGE_SZ)){
            return rs;
        } else{
            return rs % RANGE_SZ;
        }
    }

    auto randomize_bool() noexcept -> bool{

        uint64_t rs = BitRandomizer::randomize_bit(std::integral_constant<size_t, 1u>{}); 
        return static_cast<bool>(rs);
    }

    template <class T, std::enable_if_t<std::numeric_limits<T>::is_integer, bool> = true>
    auto randomize_int() noexcept -> T{

        using unsigned_ver_t = std::conditional_t<sizeof(T) == sizeof(uint8_t), 
                                                  uint8_t, 
                                                  std::conditional_t<sizeof(T) == sizeof(uint16_t),
                                                  uint16_t,
                                                  std::conditional_t<sizeof(T) == sizeof(uint32_t)>,
                                                                     uint32_t,
                                                                     std::conditional_t<size_t(T) == sizeof(uint64_t),
                                                                                        uint64_t,
                                                                                        void>>>>;
        
        unsigned_ver_t rs = BitRandomizer::randomize_bit(std::integral_constant<size_t, sizeof(unsigned_ver_t) * CHAR_BIT>{});
        return std::bit_cast<T>(rs);
    }
} 

#endif