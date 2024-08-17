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

namespace dg::network_randomizer{

    //avoid timing attack - hard
    template <class T, class RandomDevice, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    struct UIntRangeRandomizer{

        private:

            struct RandomizationUnit{
                T value;
                size_t bit_precision;
                RandomDevice randomizer;
            };

            static inline RandomizationUnit * table{};

            template <T RANGE_SZ>
            static consteval auto bitcount_significant(std::integral_constant<T, RANGE_SZ>) noexcept -> T{

                static_assert(RANGE_SZ != 0);
                return static_cast<T>(sizeof(T) * CHAR_BIT) - std::countl_zero(RANGE_SZ); 
            }

            static inline void re_randomize(RandomizationUnit& random_unit) noexcept{

                static_assert(std::is_same_v<T, decltype(random_unit.randomizer())>);
                static_assert(noexcept(random_unit.randomizer()));

                random_unit.value           = random_unit.randomizer();
                random_unit.bit_precision   = sizeof(T) * CHAR_BIT;
            }

        public: 

            static void init() noexcept{

                table = {}; //
            }

            template <T RANGE_SZ>
            static inline auto randomize_range(const std::integral_constant<T, RANGE_SZ>) noexcept -> T{

                static_assert(RANGE_SZ != 0);
                static_assert(RANGE_SZ <= std::numeric_limits<T>::max());

                RandomizationUnit& unit = table[dg::network_concurrency::this_thread_id()];

                if (unit.bit_precision < bitcount_significant(std::integral_constant<T, RANGE_SZ>{})){
                    re_randomize(unit);
                    return randomize_range(std::integral_constant<T, RANGE_SZ>{});
                }

                T ret_value         = unit.value % RANGE_SZ;
                unit.bit_precision  -= bitcount_significant(std::integral_constant<T, RANGE_SZ>{});
                unit.value          /= RANGE_SZ;

                return ret_value;
            }
    };

    //randomizer is a very very complicated component to write correctly
    using uint_randomizer = UIntRandomizer; 

    template <size_t RANGE_SZ>
    inline auto randomize_range(const std::integral_constant<size_t, RANGE_SZ>) noexcept -> size_t{

        return uint_randomizer::randomize_range(std::integral_constant<size_t, RANGE_SZ>{}); 
    }

    inline auto randomize_bool() noexcept -> bool{

        return static_cast<bool>(randomize_range(std::integral_constant<size_t, 2>{});)
    }
} 

#endif