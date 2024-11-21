#ifndef __NETWORK_TILEOPS_STATIC_H__
#define __NETWORK_TILEOPS_STATIC_H__ 

#include <stdint.h>
#include <stdlib.h>
#include <type_traits> 
#include <math.h>
#include "network_xmath_host.h"
#include <limits.h>
#include <bit>
#include "network_tile_metadata.h"
#include <memory>
#include <stdfloat>

namespace dg::network_tileops_host_static::templated_ops{

    static constexpr auto pow2(size_t val) noexcept -> size_t{

        return size_t{1} << val;
    }

    static constexpr auto log2(size_t val) noexcept -> size_t{

        return static_cast<size_t>(sizeof(size_t) * CHAR_BIT - 1) - static_cast<size_t>(std::countl_zero(val));
    }
 
    static constexpr auto sqrt(size_t val) noexcept -> size_t{

        return templated_ops::pow2(templated_ops::log2(val) >> 1);
    }

    static constexpr auto is_pow2(size_t val) noexcept -> bool{

        return val != 0u && (val & (val - 1u)) == 0u;
    }
  
    template <class arithmetic_ops_t>
    struct coerced_x_math{

        static inline auto sign(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::sign(value);
        }

        static inline auto exp(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::exp(value);
        }

        static inline auto log(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::log(value);
        }

        static inline auto abs(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::abs(value);
        }

        static inline auto cos(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::cos(value);
        }

        static inline auto acos(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::acos(value);
        }

        static inline auto sin(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::sin(value);
        }

        static inline auto asin(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::asin(value);
        }

        static inline auto tan(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::tan(value);
        }

        static inline auto atan(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::atan(value);
        }

        static inline auto sqrt(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::sqrt(value);
        }

        static inline auto invsqrt(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::invsqrt(value);
        } 

        static inline auto negative(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::negative(value);
        } 

        static inline auto add(arithmetic_ops_t lhs, arithmetic_ops_t rhs) noexcept -> arithmetic_ops_t{

            return network_xmath_host::add(lhs, rhs);
        }

        static inline auto sub(arithmetic_ops_t lhs, arithmetic_ops_t rhs) noexcept -> arithmetic_ops_t{

            return network_xmath_host::sub(lhs, rhs);
        }

        static inline auto mul(arithmetic_ops_t lhs, arithmetic_ops_t rhs) noexcept -> arithmetic_ops_t{

            return network_xmath_host::mul(lhs, rhs);
        }

        static inline auto div(arithmetic_ops_t lhs, arithmetic_ops_t rhs) noexcept -> arithmetic_ops_t{

            return network_xmath_host::div(lhs, rhs);
        }

        static inline auto pow(arithmetic_ops_t lhs, arithmetic_ops_t rhs) noexcept -> arithmetic_ops_t{

            return network_xmath_host::pow(lhs, rhs);
        }

        template <size_t RHS_VALUE>
        static inline auto pow(arithmetic_ops_t lhs, const std::integral_constant<size_t, RHS_VALUE>) noexcept -> arithmetic_ops_t{

            return network_xmath_host::pow(lhs, std::integral_constant<size_t, RHS_VALUE>{});
        }

        static inline auto fma(arithmetic_ops_t first, arithmetic_ops_t second, arithmetic_ops_t third) noexcept -> arithmetic_ops_t{

            return network_xmath_host::fma(first, second, third);
        }

        static inline auto min(arithmetic_ops_t lhs, arithmetic_ops_t rhs) noexcept -> arithmetic_ops_t{

            return network_xmath_host::min(lhs, rhs);
        }

        static inline auto max(arithmetic_ops_t lhs, arithmetic_ops_t rhs) noexcept -> arithmetic_ops_t{

            return network_xmath_host::max(lhs, rhs);
        }

        static inline auto eqcmp_mul(arithmetic_ops_t lcmp, arithmetic_ops_t rcmp, arithmetic_ops_t val) noexcept -> arithmetic_ops_t{

            return network_xmath_host::eqcmp_mul(lcmp, rcmp, val);
        }

        static inline auto bitwise_or(arithmetic_ops_t lhs, arithmetic_ops_t rhs) noexcept -> arithmetic_ops_t{

            return network_xmath_host::bitwise_or(lhs, rhs);
        }

        static inline auto bitwise_and(arithmetic_ops_t lhs, arithmetic_ops_t rhs) noexcept -> arithmetic_ops_t{

            return network_xmath_host::bitwise_and(lhs, rhs);
        }

        static inline auto bitwise_xor(arithmetic_ops_t lhs, arithmetic_ops_t rhs) noexcept -> arithmetic_ops_t{

            return network_xmath_host::bitwise_xor(lhs, rhs);
        }
    };

    struct base_bitwise_gradient{

        static inline auto bitwise_and(uint8_t wrt, uint8_t other) noexcept -> uint8_t{

            return {}; //TODOs
        }

        static inline auto bitwise_and(uint16_t wrt, uint16_t other) noexcept -> uint16_t{

            return {}; //TODOs
        }

        static inline auto bitwise_and(uint32_t wrt, uint32_t other) noexcept -> uint32_t{

            return {}; //TODOs
        }

        static inline auto bitwise_and(uint64_t wrt, uint64_t other) noexcept -> uint64_t{

            return {}; //TODOs
        }

        static inline auto bitwise_and(std::float16_t wrt, std::float16_t other) noexcept -> std::float16_t{

            return std::numeric_limits<std::float16_t>::quiet_NaN();
        }

        static inline auto bitwise_and(std::bfloat16_t wrt, std::bfloat16_t other) noexcept -> std::bfloat16_t{

            return {};
            // return std::numeric_limits<std::bfloat16_t>::quiet_NaN();
        }

        static inline auto bitwise_and(std::float32_t wrt, std::float32_t other) noexcept -> std::float32_t{

            return std::numeric_limits<std::float32_t>::quiet_NaN();
        }

        static inline auto bitwise_and(std::float64_t wrt, std::float64_t other) noexcept -> std::float64_t{

            return std::numeric_limits<std::float64_t>::quiet_NaN();
        }

        static inline auto bitwise_or(uint8_t wrt, uint8_t other) noexcept -> uint8_t{

            return {}; //TODOs
        }

        static inline auto bitwise_or(uint16_t wrt, uint16_t other) noexcept -> uint16_t{

            return {}; //TODOs
        }

        static inline auto bitwise_or(uint32_t wrt, uint32_t other) noexcept -> uint32_t{

            return {}; //TODOs
        }

        static inline auto bitwise_or(uint64_t wrt, uint64_t other) noexcept -> uint64_t{

            return {}; //TODOs
        }

        static inline auto bitwise_or(std::float16_t wrt, std::float16_t other) noexcept -> std::float16_t{

            return std::numeric_limits<std::float16_t>::quiet_NaN();
        }

        static inline auto bitwise_or(std::bfloat16_t wrt, std::bfloat16_t other) noexcept -> std::bfloat16_t{

            return {};
            // return std::numeric_limits<std::bfloat16_t>::quiet_NaN();
        }

        static inline auto bitwise_or(std::float32_t wrt, std::float32_t other) noexcept -> std::float32_t{

            return std::numeric_limits<std::float32_t>::quiet_NaN();
        }

        static inline auto bitwise_or(std::float64_t wrt, std::float64_t other) noexcept -> std::float64_t{

            return std::numeric_limits<std::float64_t>::quiet_NaN();
        }

        static inline auto bitwise_xor(uint8_t wrt, uint8_t other) noexcept -> uint8_t{

            return {}; //TODOs
        }

        static inline auto bitwise_xor(uint16_t wrt, uint16_t other) noexcept -> uint16_t{

            return {}; //TODOs
        }

        static inline auto bitwise_xor(uint32_t wrt, uint32_t other) noexcept -> uint32_t{

            return {}; //TODOs
        }

        static inline auto bitwise_xor(uint64_t wrt, uint64_t other) noexcept -> uint64_t{

            return {}; //TODOs
        }

        static inline auto bitwise_xor(std::float16_t wrt, std::float16_t other) noexcept -> std::float16_t{

            return std::numeric_limits<std::float16_t>::quiet_NaN();
        }

        static inline auto bitwise_xor(std::bfloat16_t wrt, std::bfloat16_t other) noexcept -> std::bfloat16_t{

            return {};
            // return std::numeric_limits<std::bfloat16_t>::quiet_NaN();
        }

        static inline auto bitwise_xor(std::float32_t wrt, std::float32_t other) noexcept -> std::float32_t{

            return std::numeric_limits<std::float32_t>::quiet_NaN();
        }

        static inline auto bitwise_xor(std::float64_t wrt, std::float64_t other) noexcept -> std::float64_t{

            return std::numeric_limits<std::float64_t>::quiet_NaN();
        }
    };

    template  <class arithmetic_ops_t>
    struct coerced_bitwise_gradient{

        static inline auto bitwise_and(arithmetic_ops_t wrt, arithmetic_ops_t other) noexcept -> arithmetic_ops_t{

            return base_bitwise_gradient::bitwise_and(wrt, other);
        }

        static inline auto bitwise_or(arithmetic_ops_t wrt, arithmetic_ops_t other) noexcept -> arithmetic_ops_t{

            return base_bitwise_gradient::bitwise_or(wrt, other);
        }

        static inline auto bitwise_xor(arithmetic_ops_t wrt, arithmetic_ops_t other) noexcept -> arithmetic_ops_t{

            return base_bitwise_gradient::bitwise_xor(wrt, other);
        }
    };

    template <class dst_logit_value_t, class src_logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_mono_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>;

        static inline void exp(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::exp(src[i]);
            }
        } 

        static inline void log(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::log(src[i]);
            }
        }

        static inline void clone(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = static_cast<casting_ops_t>(src[i]);
            }
        } 

        static inline void negative(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::negative(src[i]);
            }
        }

        static inline void inverse(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::div(1, src[i]);
            }
        }

        static inline void abs(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::abs(src[i]);
            }    
        }

        static inline void cos(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::cos(src[i]);
            }
        }

        static inline void acos(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::acos(src[i]);
            }
        }

        static inline void sin(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::sin(src[i]);
            }
        }

        static inline void asin(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{
            
            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::asin(src[i]);
            }
        }

        static inline void tan(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::tan(src[i]);
            }
        }

        static inline void atan(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::atan(src[i]);
            }
        }

        static inline void transpose(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t i = 0u; i < BLK_SZ; ++i){
                for (size_t j = 0u; j < BLK_SZ; ++j){
                    dst[i * BLK_SZ + j] = static_cast<casting_ops_t>(src[j * BLK_SZ + i]);
                }
            }
        }
    };

    template <class dst_logit_value_t, class src_logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_uacm_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void max(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::max(dst[i], src[i]);
            }
        }

        static inline void min(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::min(dst[i], src[i]);
            }
        }

        static inline void sum(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src[i]);
            }
        }

        //avg is actually not uacm - but sum / constant - this is achievable by using uacm + pair
    };

    template <class dst_logit_value_t, class lhs_logit_value_t, class rhs_logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_pacm_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void accum_add(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::add(lhs[i], rhs[i]));
            }
        }

        static inline void accum_sub(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::sub(lhs[i], rhs[i]));
            }
        }

        static inline void accum_mul(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::mul(lhs[i], rhs[i]));
            }
        }

        static inline void accum_div(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(lhs[i], rhs[i]));
            }
        }

        static inline void accum_pow(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::pow(lhs[i], rhs[i]));
            }
        }
        
        static inline void accum_bitwise_or(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::bitwise_or(lhs[i], rhs[i]));
            }
        }

        static inline void accum_bitwise_and(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::bitwise_and(lhs[i], rhs[i]));
            }
        }

        static inline void accum_bitwise_xor(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::bitwise_xor(lhs[i], rhs[i]));
            }
        }

        static inline void accum_linear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t j = 0u; j < BLK_SZ; ++j){
                for (size_t i = 0u; i < BLK_SZ; ++i){
                    casting_ops_t dot_sum{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        dot_sum = x_math::add(dot_sum, x_math::mul(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], dot_sum);
                }
            }
        }

        static inline void accum_addnear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t j = 0u; j < BLK_SZ; ++j){
                for (size_t i = 0u; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::add(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], total);
                }
            }
        }

        static inline void accum_ornear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t j = 0u; j < BLK_SZ; ++j){
                for (size_t i = 0u; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::bitwise_or(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], total);
                }
            }
        }

        static inline void accum_andnear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t j = 0u; j < BLK_SZ; ++j){
                for (size_t i = 0u; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::bitwise_and(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], total);
                }
            }
        }

        static inline void accum_xornear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t j = 0u; j < BLK_SZ; ++j){
                for (size_t i = 0u; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::bitwise_xor(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], total);
                }
            }
        }
    };

    template <class dst_logit_value_t, class lhs_logit_value_t, class rhs_logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_pair_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void add(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{
            
            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(lhs[i], rhs[i]);
            }
        }

        static inline void sub(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::sub(lhs[i], rhs[i]);
            }
        }

        static inline void mul(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::mul(lhs[i], rhs[i]);
            }
        }

        static inline void div(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::div(lhs[i], rhs[i]);
            }
        }

        static inline void pow(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::pow(lhs[i], rhs[i]);
            }
        }

        static inline void bitwise_or(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::bitwise_or(lhs[i], rhs[i]);
            }
        }

        static inline void bitwise_and(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::bitwise_and(lhs[i], rhs[i]);
            }
        }

        static inline void bitwise_xor(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::bitwise_xor(lhs[i], rhs[i]);
            }
        }

        static inline void linear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t j = 0u; j < BLK_SZ; ++j){
                for (size_t i = 0u; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::mul(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = total;
                }
            }
        }

        static inline void addnear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t j = 0u; j < BLK_SZ; ++j){
                for (size_t i = 0u; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::add(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = total;
                }
            }
        }

        static inline void ornear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t j = 0u; j < BLK_SZ; ++j){
                for (size_t i = 0u; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::bitwise_or(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = total;
                }
            }
        }

        static inline void andnear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t j = 0u; j < BLK_SZ; ++j){
                for (size_t i = 0u; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::bitwise_and(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = total;
                }
            }
        }

        static inline void xornear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t j = 0u; j < BLK_SZ; ++j){
                for (size_t i = 0u; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::bitwise_xor(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = total;
                }
            }
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class src_grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_mono_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>;

        //de^x/ dx = e^x
        static inline void exp(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], x_math::exp(dst_logit[i]), dst[i]);
            }
        }

        //dln(x)/dx = 1/x
        static inline void log(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(src_grad[i], dst_logit[i]));
            }
        }

        //dx/dx = 1
        static inline void clone(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src_grad[i]);
            }
        }

        //d-x/dx = -1
        static inline void negative(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], src_grad[i]);
            }
        }

        //d1/x/dx = d-1/x^2
        //dst - grad/x^2
        static inline void inverse(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::div(src_grad[i], x_math::pow(dst_logit[i], std::integral_constant<size_t, 2>{})));
            }
        }
        
        //dabs(x)/dx = 1, x > 0
        //dabs(x)/dx = -1, x < 0
        //dabs(x)/dx = 0, x = 0
        static inline void abs(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], x_math::sign(dst_logit[i]), dst[i]);
            }
        }

        //dcos(x)/dx = -sin(x)
        static inline void cos(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::mul(src_grad[i], x_math::sin(dst_logit[i])));
            }
        }

        //d acos(x)/dx = -(1-x^2) ^ (-1/2)
        static inline void acos(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::div(src_grad[i], x_math::sqrt(x_math::sub(1, x_math::pow(dst_logit[i], std::integral_constant<size_t, 2>{})))));
            }
        } 

        //dsin(x)/dx = cos(x)
        static inline void sin(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], x_math::cos(dst_logit[i]), dst[i]);
            }
        }

        //dasin(x)/dx = (1-x^2) ^ (-1/2)
        static inline void asin(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(src_grad[i], x_math::sqrt(x_math::sub(1, x_math::pow(dst_logit[i], std::integral_constant<size_t, 2>{})))));
            }
        }

        //dtan(x)/dx = 1/cos^2(x)
        static inline void tan(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(src_grad[i], x_math::pow(x_math::cos(dst_logit[i]), std::integral_constant<size_t, 2>{})));
            }
        }

        //datan(x)/dx = (1+x^2) ^ -1
        static inline void atan(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(src_grad[i], x_math::add(x_math::pow(dst_logit[i], std::integral_constant<size_t, 2>{}), 1)));
            }
        }

        static inline void transpose(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ); 

            //adjecent write is faster than adjecent read
            for (size_t i = 0u; i < BLK_SZ; ++i){
                for (size_t j = 0u; j < BLK_SZ; ++j){
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], src_grad[j * BLK_SZ + i]);
                }
            }
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class src_logit_value_t, class src_grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_uacm_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void max(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const src_logit_value_t * src_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::eqcmp_mul(dst_logit[i], src_logit[i], src_grad[i]));
            }
        }

        static inline void min(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const src_logit_value_t * src_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::eqcmp_mul(dst_logit[i], src_logit[i], src_grad[i]));
            }
        }

        static inline void sum(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const src_logit_value_t *) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src_grad[i]);
            }
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class other_logit_value_t, class src_grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_pair_lhs_unaligned_ops{

        using x_math            = coerced_x_math<casting_ops_t>; 
        using bitwise_gradient  = coerced_bitwise_gradient<casting_ops_t>; 

        //d(a + b)/ da = 1
        static inline void add(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t *) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src_grad[i]);
            }
        }

        //d(a * b)/ da = b
        static inline void mul(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], other_logit[i], dst[i]);
            }
        }

        //d(a - b)/ da = 1
        static inline void sub(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t *) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src_grad[i]);
            }
        }

        //d(a / b)/ da = 1 / b
        static inline void div(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(src_grad[i], other_logit[i]));
            }
        }

        //d(a^b)/ da = b*a^(b-1)
        static inline void pow(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], x_math::mul(other_logit[i], x_math::pow(dst_logit[i], x_math::sub(other_logit[i], 1))), dst[i]);
            }
        }
        
        static inline void bitwise_and(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::mul(src_grad[i], bitwise_gradient::bitwise_and(dst_logit[i], other_logit[i])));
            }
        }

        static inline void bitwise_or(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::mul(src_grad[i], bitwise_gradient::bitwise_or(dst_logit[i], other_logit[i])));
            }
        }

        static inline void bitwise_xor(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::mul(src_grad[i], bitwise_gradient::bitwise_xor(dst_logit[i], other_logit[i])));
            }
        }

        static inline void andnear(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t i = 0u; i < BLK_SZ; ++i){
                for (size_t j = 0u; j < BLK_SZ; ++j){
                    casting_ops_t and_sum{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        and_sum = x_math::add(and_sum, x_math::mul(src_grad[i * BLK_SZ + z], bitwise_gradient::bitwise_and(dst_logit[j * BLK_SZ + z], other_logit[j * BLK_SZ + z])));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], and_sum);
                }
            }
        }

        static inline void ornear(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t i = 0u; i < BLK_SZ; ++i){
                for (size_t j = 0u; j < BLK_SZ; ++j){
                    casting_ops_t or_sum{};
                    for (size_t z = 0; z < BLK_SZ; ++z){
                        or_sum = x_math::add(or_sum, x_math::mul(src_grad[i * BLK_SZ + z], bitwise_gradient::bitwise_or(dst_logit[j * BLK_SZ + z], other_logit[j * BLK_SZ + z])));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], or_sum);
                }
            }
        }

        static inline void xornear(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t i = 0u; i < BLK_SZ; ++i){
                for (size_t j = 0u; j < BLK_SZ; ++j){
                    casting_ops_t xor_sum{};
                    for (size_t z = 0; z < BLK_SZ; ++z){
                        xor_sum = x_math::add(xor_sum, x_math::mul(src_grad[i * BLK_SZ + z], bitwise_gradient::bitwise_xor(dst_logit[j * BLK_SZ + z], other_logit[j * BLK_SZ + z])));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], xor_sum);
                }
            }
        }

        static inline void addnear(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t i = 0u; i < BLK_SZ; ++i){
                casting_ops_t row_sum{};
                
                for (size_t z = 0u; z < BLK_SZ; ++z){
                    row_sum = x_math::add(row_sum, src_grad[i * BLK_SZ + z]);
                }

                for (size_t j = 0u; j < BLK_SZ; ++j){
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], row_sum);
                }
            }
        }

        static inline void linear(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{
            
            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ); 

            for (size_t i = 0u; i < BLK_SZ; ++i){
                for (size_t j = 0u; j < BLK_SZ; ++j){
                    casting_ops_t dot_sum{}; 
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        dot_sum = x_math::add(dot_sum, x_math::mul(src_grad[i * BLK_SZ + z], other_logit[j * BLK_SZ + z]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], dot_sum);
                }
            }
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class other_logit_value_t, class src_grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_pair_rhs_unaligned_ops{

        using x_math            = coerced_x_math<casting_ops_t>;
        using bitwise_gradient  = coerced_bitwise_gradient<casting_ops_t>;

        static inline void add(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t *) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src_grad[i]);
            }
        }

        static inline void mul(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], other_logit[i], dst[i]);
            }
        }

        static inline void sub(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t *) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], src_grad[i]);
            }
        }

        static inline void div(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::mul(src_grad[i], x_math::div(other_logit[i], x_math::pow(dst_logit[i], std::integral_constant<size_t, 2>{})))); 
            }
        }

        static inline void pow(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], x_math::mul(x_math::pow(other_logit[i], dst_logit[i]), x_math::log(other_logit[i])), dst[i]);
            }
        }

        static inline void bitwise_and(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::mul(src_grad[i], bitwise_gradient::bitwise_and(dst_logit[i], other_logit[i])));
            }
        }

        static inline void bitwise_or(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::mul(src_grad[i], bitwise_gradient::bitwise_or(dst_logit[i], other_logit[i])));
            }
        }

        static inline void bitwise_xor(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0u; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::mul(src_grad[i], bitwise_gradient::bitwise_xor(dst_logit[i], other_logit[i])));
            }
        }

        static inline void andnear(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t i = 0u; i < BLK_SZ; ++i){
                for (size_t j = 0u; j < BLK_SZ; ++j){
                    casting_ops_t and_sum{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        and_sum = x_math::add(and_sum, x_math::mul(bitwise_gradient::bitwise_and(dst_logit[z * BLK_SZ + i], other_logit[z * BLK_SZ + i]), src_grad[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], and_sum);
                }
            }
        }

        static inline void ornear(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t i = 0u; i < BLK_SZ; ++i){
                for (size_t j = 0u; j < BLK_SZ; ++j){
                    casting_ops_t or_sum{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        or_sum = x_math::add(or_sum, x_math::mul(bitwise_gradient::bitwise_or(dst_logit[z * BLK_SZ + i], other_logit[z * BLK_SZ + i]), src_grad[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], or_sum);
                }
            }
        }

        static inline void xornear(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t i = 0u; i < BLK_SZ; ++i){
                for (size_t j = 0u; j < BLK_SZ; ++j){
                    casting_ops_t xor_sum{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        xor_sum = x_math::add(xor_sum, x_math::mul(bitwise_gradient::bitwise_xor(dst_logit[z * BLK_SZ + i], other_logit[z * BLK_SZ + i]), src_grad[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], xor_sum);
                }
            }
        }

        static inline void addnear(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t *) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ); 

            for (size_t j = 0u; j < BLK_SZ; ++j){
                casting_ops_t col_sum{};

                for (size_t z = 0u; z < BLK_SZ; ++z){
                    col_sum = x_math::add(col_sum, src_grad[z * BLK_SZ + j]);
                }

                for (size_t i = 0u; i < BLK_SZ; ++i){
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], col_sum);
                }
            }
        }

        static inline void linear(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t i = 0u; i < BLK_SZ; ++i){
                for (size_t j = 0u; j < BLK_SZ; ++j){
                    casting_ops_t dot_sum{};
                    for (size_t z = 0u; z < BLK_SZ; ++z){
                        dot_sum = x_math::add(dot_sum, x_math::mul(other_logit[z * BLK_SZ + i], src_grad[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], dot_sum);
                }
            }
        }
    };

    template <class dst_logit_value_t, class src_logit_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct fwd_mono_restrict_aligned_ops{

        using base = fwd_mono_unaligned_ops<dst_logit_value_t, src_logit_value_t, casting_ops_t, SZ>;

        static __attribute__((flatten)) void exp(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::exp(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        } 

        static __attribute__((flatten)) void log(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::log(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void clone(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::clone(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        } 

        static __attribute__((flatten)) void negative(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::negative(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void inverse(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::inverse(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void abs(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::abs(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void cos(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::cos(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void acos(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::acos(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void sin(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::sin(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void asin(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{
            
            base::asin(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void tan(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::tan(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void atan(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::atan(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void transpose(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::transpose(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }
    };

    template <class dst_logit_value_t, class src_logit_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct fwd_uacm_restrict_aligned_ops{

        using base = fwd_uacm_unaligned_ops<dst_logit_value_t, src_logit_value_t, casting_ops_t, SZ>;

        static __attribute__((flatten)) void max(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::max(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void min(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::min(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void sum(dst_logit_value_t * __restrict__ dst, const src_logit_value_t * __restrict__ src) noexcept{

            base::sum(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }
    };

    template <class dst_logit_value_t, class lhs_logit_value_t, class rhs_logit_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct fwd_pacm_restrict_aligned_ops{

        using base = fwd_pacm_unaligned_ops<dst_logit_value_t, lhs_logit_value_t, rhs_logit_value_t, casting_ops_t, SZ>;

        static __attribute__((flatten)) void accum_add(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_add(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_sub(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_sub(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_mul(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_mul(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_div(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_div(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_pow(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_pow(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_bitwise_or(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_bitwise_or(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_bitwise_and(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_bitwise_and(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_bitwise_xor(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_bitwise_xor(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_linear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{
            
            base::accum_linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_addnear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_addnear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_ornear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_ornear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_andnear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_andnear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void accum_xornear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::accum_xornear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }
    };

    template <class dst_logit_value_t, class lhs_logit_value_t, class rhs_logit_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct fwd_pair_restrict_aligned_ops{

        using base      = fwd_pair_unaligned_ops<dst_logit_value_t, lhs_logit_value_t, rhs_logit_value_t, casting_ops_t, SZ>;
        using x_math    = coerced_x_math<casting_ops_t>;

        static __attribute__((flatten)) void add(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::add(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void sub(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::sub(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void mul(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::mul(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void div(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::div(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void pow(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::pow(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void bitwise_or(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{
            
            base::bitwise_or(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void bitwise_and(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::bitwise_and(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void bitwise_xor(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::bitwise_xor(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void linear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void addnear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::addnear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void ornear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::ornear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void andnear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::andnear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void xornear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::xornear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class src_grad_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct bwd_mono_restrict_aligned_ops{

        using base = bwd_mono_unaligned_ops<dst_logit_value_t, dst_grad_value_t, src_grad_value_t, casting_ops_t, SZ>; 

        static __attribute__((flatten)) void exp(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::exp(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }

        static __attribute__((flatten)) void log(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::log(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }

        static __attribute__((flatten)) void clone(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{
            
            base::clone(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }

        static __attribute__((flatten)) void negative(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::negative(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }

        static __attribute__((flatten)) void inverse(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{
            
            base::inverse(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }

        static __attribute__((flatten)) void abs(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::abs(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }

        static __attribute__((flatten)) void cos(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::cos(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }

        static __attribute__((flatten)) void acos(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::acos(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        } 

        static __attribute__((flatten)) void sin(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::sin(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }

        static __attribute__((flatten)) void asin(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::asin(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }

        static __attribute__((flatten)) void tan(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::tan(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }

        static __attribute__((flatten)) void atan(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::atan(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }
    
        static __attribute__((flatten)) void transpose(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad) noexcept{

            base::transpose(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad));
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class src_logit_value_t, class src_grad_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct bwd_uacm_restrict_aligned_ops{

        using base = bwd_uacm_unaligned_ops<dst_logit_value_t, dst_grad_value_t, src_logit_value_t, src_grad_value_t, casting_ops_t, SZ>; 

        static __attribute__((flatten)) void max(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const src_logit_value_t * __restrict__ src_logit) noexcept{

            base::max(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(src_logit));
        }

        static __attribute__((flatten)) void min(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const src_logit_value_t * __restrict__ src_logit) noexcept{

            base::min(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(src_logit));
        }

        static __attribute__((flatten)) void sum(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const src_logit_value_t * __restrict__ src_logit) noexcept{

            base::sum(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(src_logit));
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class other_logit_value_t, class src_grad_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct bwd_pair_lhs_restrict_aligned_ops{

        using base = bwd_pair_lhs_unaligned_ops<dst_logit_value_t, dst_grad_value_t, other_logit_value_t, src_grad_value_t, casting_ops_t, SZ>; 

        static __attribute__((flatten)) void add(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::add(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void mul(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::mul(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void sub(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::sub(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void div(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::div(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void pow(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::pow(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void bitwise_and(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::bitwise_and(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void bitwise_or(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::bitwise_or(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void bitwise_xor(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::bitwise_xor(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void andnear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::andnear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void ornear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::ornear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void xornear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::xornear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void addnear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::andnear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void linear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class other_logit_value_t, class src_grad_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct bwd_pair_rhs_restrict_aligned_ops{

        using base = bwd_pair_rhs_unaligned_ops<dst_logit_value_t, dst_grad_value_t, other_logit_value_t, src_grad_value_t, casting_ops_t, SZ>;

        static __attribute__((flatten)) void add(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::add(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        } 

        static __attribute__((flatten)) void mul(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::mul(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void sub(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::sub(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void div(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::div(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void pow(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::pow(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void bitwise_and(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::bitwise_and(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void bitwise_or(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::bitwise_or(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void bitwise_xor(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::bitwise_xor(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void andnear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::andnear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void ornear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::ornear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void xornear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::xornear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }

        static __attribute__((flatten)) void addnear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::addnear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }
        
        static __attribute__((flatten)) void linear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
        }
    };
}

namespace dg::network_tileops_host_static{

    static inline constexpr size_t LOGIT_COUNT_PER_TILE             = dg::network_tile_metadata::LOGIT_COUNT_PER_TILE;
    static inline constexpr size_t ALIGNMENT_SZ                     = std::min(dg::network_tile_metadata::LOGIT_ALIGNMENT_SZ, dg::network_tile_metadata::GRAD_ALIGNMENT_SZ); 

    using tile_u8_t                                                 = dg::network_tile_metadata::host_u8_t;
    using tile_u16_t                                                = dg::network_tile_metadata::host_u16_t;
    using tile_u32_t                                                = dg::network_tile_metadata::host_u32_t;
    using tile_u64_t                                                = dg::network_tile_metadata::host_u64_t;
    using tile_f8_t                                                 = dg::network_tile_metadata::host_f8_t;
    using tile_f16_t                                                = dg::network_tile_metadata::host_f16_t; 

    using fwd_mono_ops_u8_u8_u8                             		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u8_u8_u16                            		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u8_u8_f16                            		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u16_u8_u8                            		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u16_u8_u16                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u16_u8_f16                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_f16_u8_u8                            		= templated_ops::fwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_f16_u8_u16                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_f16_u8_f16                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u8_u16_u8                            		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u8_u16_u16                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u8_u16_f16                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u16_u16_u8                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u16_u16_u16                          		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u16_u16_f16                          		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_f16_u16_u8                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_f16_u16_u16                          		= templated_ops::fwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_f16_u16_f16                          		= templated_ops::fwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u8_f16_u8                            		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u8_f16_u16                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u8_f16_f16                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u16_f16_u8                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u16_f16_u16                          		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_u16_f16_f16                          		= templated_ops::fwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_f16_f16_u8                           		= templated_ops::fwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_f16_f16_u16                          		= templated_ops::fwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_f16_f16_f16                          		= templated_ops::fwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u8_u8_u8                             		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u8_u8_u16                            		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u8_u8_f16                            		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u16_u8_u8                            		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u16_u8_u16                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u16_u8_f16                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_f16_u8_u8                            		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_f16_u8_u16                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_f16_u8_f16                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u8_u16_u8                            		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u8_u16_u16                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u8_u16_f16                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u16_u16_u8                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u16_u16_u16                          		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u16_u16_f16                          		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_f16_u16_u8                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_f16_u16_u16                          		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_f16_u16_f16                          		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u8_f16_u8                            		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u8_f16_u16                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u8_f16_f16                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u16_f16_u8                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u16_f16_u16                          		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_u16_f16_f16                          		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_f16_f16_u8                           		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_f16_f16_u16                          		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_f16_f16_f16                          		= templated_ops::fwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u8_u8_u8                          		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u8_u8_u16                         		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u8_u8_f16                         		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u8_u8_u8                         		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u8_u8_u16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u8_u8_f16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u8_u8_u8                         		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u8_u8_u16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u8_u8_f16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u8_u16_u8                         		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u8_u16_u16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u8_u16_f16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u8_u16_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u8_u16_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u8_u16_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u8_u16_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u8_u16_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u8_u16_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u8_f16_u8                         		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u8_f16_u16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u8_f16_f16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u8_f16_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u8_f16_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u8_f16_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u8_f16_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u8_f16_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u8_f16_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u16_u8_u8                         		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u16_u8_u16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u16_u8_f16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u16_u8_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u16_u8_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u16_u8_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u16_u8_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u16_u8_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u16_u8_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u16_u16_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u16_u16_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u16_u16_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u16_u16_u8                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u16_u16_u16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u16_u16_f16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u16_u16_u8                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u16_u16_u16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u16_u16_f16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u16_f16_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u16_f16_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_u16_f16_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u16_f16_u8                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u16_f16_u16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_u16_f16_f16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u16_f16_u8                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u16_f16_u16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_u16_f16_f16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_f16_u8_u8                         		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_f16_u8_u16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_f16_u8_f16                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_f16_u8_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_f16_u8_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_f16_u8_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_f16_u8_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_f16_u8_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_f16_u8_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_f16_u16_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_f16_u16_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_f16_u16_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_f16_u16_u8                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_f16_u16_u16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_f16_u16_f16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_f16_u16_u8                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_f16_u16_u16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_f16_u16_f16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_f16_f16_u8                        		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_f16_f16_u16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u8_f16_f16_f16                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_f16_f16_u8                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_f16_f16_u16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_u16_f16_f16_f16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_f16_f16_u8                       		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_f16_f16_u16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pacm_ops_f16_f16_f16_f16                      		= templated_ops::fwd_pacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u8_u8_u8                          		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u8_u8_u16                         		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u8_u8_f16                         		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u8_u8_u8                         		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u8_u8_u16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u8_u8_f16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u8_u8_u8                         		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u8_u8_u16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u8_u8_f16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u8_u16_u8                         		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u8_u16_u16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u8_u16_f16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u8_u16_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u8_u16_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u8_u16_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u8_u16_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u8_u16_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u8_u16_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u8_f16_u8                         		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u8_f16_u16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u8_f16_f16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u8_f16_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u8_f16_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u8_f16_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u8_f16_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u8_f16_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u8_f16_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u16_u8_u8                         		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u16_u8_u16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u16_u8_f16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u16_u8_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u16_u8_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u16_u8_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u16_u8_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u16_u8_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u16_u8_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u16_u16_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u16_u16_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u16_u16_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u16_u16_u8                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u16_u16_u16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u16_u16_f16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u16_u16_u8                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u16_u16_u16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u16_u16_f16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u16_f16_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u16_f16_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_u16_f16_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u16_f16_u8                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u16_f16_u16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_u16_f16_f16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u16_f16_u8                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u16_f16_u16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_u16_f16_f16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_f16_u8_u8                         		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_f16_u8_u16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_f16_u8_f16                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_f16_u8_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_f16_u8_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_f16_u8_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_f16_u8_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_f16_u8_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_f16_u8_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_f16_u16_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_f16_u16_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_f16_u16_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_f16_u16_u8                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_f16_u16_u16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_f16_u16_f16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_f16_u16_u8                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_f16_u16_u16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_f16_u16_f16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_f16_f16_u8                        		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_f16_f16_u16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u8_f16_f16_f16                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_f16_f16_u8                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_f16_f16_u16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_u16_f16_f16_f16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_f16_f16_u8                       		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_f16_f16_u16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_f16_f16_f16_f16                      		= templated_ops::fwd_pair_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u8_u8_u8                          		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u8_u8_u16                         		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u8_u8_f16                         		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u16_u8_u8                         		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u16_u8_u16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u16_u8_f16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_f16_u8_u8                         		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_f16_u8_u16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_f16_u8_f16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u8_u8_u8                         		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u8_u8_u16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u8_u8_f16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u16_u8_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u16_u8_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u16_u8_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_f16_u8_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_f16_u8_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_f16_u8_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u8_u8_u8                         		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u8_u8_u16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u8_u8_f16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u16_u8_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u16_u8_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u16_u8_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_f16_u8_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_f16_u8_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_f16_u8_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u8_u16_u8                         		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u8_u16_u16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u8_u16_f16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u16_u16_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u16_u16_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u16_u16_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_f16_u16_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_f16_u16_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_f16_u16_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u8_u16_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u8_u16_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u8_u16_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u16_u16_u8                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u16_u16_u16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u16_u16_f16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_f16_u16_u8                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_f16_u16_u16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_f16_u16_f16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u8_u16_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u8_u16_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u8_u16_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u16_u16_u8                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u16_u16_u16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u16_u16_f16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_f16_u16_u8                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_f16_u16_u16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_f16_u16_f16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u8_f16_u8                         		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u8_f16_u16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u8_f16_f16                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u16_f16_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u16_f16_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_u16_f16_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_f16_f16_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_f16_f16_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u8_f16_f16_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u8_f16_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u8_f16_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u8_f16_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u16_f16_u8                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u16_f16_u16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_u16_f16_f16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_f16_f16_u8                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_f16_f16_u16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_u16_f16_f16_f16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u8_f16_u8                        		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u8_f16_u16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u8_f16_f16                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u16_f16_u8                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u16_f16_u16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_u16_f16_f16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_f16_f16_u8                       		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_f16_f16_u16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_f16_f16_f16_f16                      		= templated_ops::bwd_mono_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u8_u8_u8                       		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u8_u8_u16                      		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u8_u8_f16                      		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u8_u8_u8                      		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u8_u8_u16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u8_u8_f16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u8_u8_u8                      		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u8_u8_u16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u8_u8_f16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u8_u8_u8                      		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u8_u8_u16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u8_u8_f16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u8_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u8_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u8_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u8_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u8_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u8_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u8_u8_u8                      		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u8_u8_u16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u8_u8_f16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u8_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u8_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u8_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u8_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u8_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u8_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u8_u16_u8                      		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u8_u16_u16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u8_u16_f16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u8_u16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u8_u16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u8_u16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u8_u16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u8_u16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u8_u16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u8_u16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u8_u16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u8_u16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u8_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u8_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u8_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u8_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u8_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u8_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u8_u16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u8_u16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u8_u16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u8_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u8_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u8_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u8_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u8_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u8_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u8_f16_u8                      		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u8_f16_u16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u8_f16_f16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u8_f16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u8_f16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u8_f16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u8_f16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u8_f16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u8_f16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u8_f16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u8_f16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u8_f16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u8_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u8_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u8_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u8_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u8_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u8_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u8_f16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u8_f16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u8_f16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u8_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u8_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u8_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u8_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u8_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u8_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u16_u8_u8                      		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u16_u8_u16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u16_u8_f16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u16_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u16_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u16_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u16_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u16_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u16_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u16_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u16_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u16_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u16_u8_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u16_u8_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u16_u8_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u16_u8_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u16_u8_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u16_u8_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u16_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u16_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u16_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u16_u8_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u16_u8_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u16_u8_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u16_u8_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u16_u8_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u16_u8_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u16_u16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u16_u16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u16_u16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u16_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u16_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u16_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u16_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u16_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u16_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u16_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u16_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u16_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u16_u16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u16_u16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u16_u16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u16_u16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u16_u16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u16_u16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u16_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u16_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u16_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u16_u16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u16_u16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u16_u16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u16_u16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u16_u16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u16_u16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u16_f16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u16_f16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_u16_f16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u16_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u16_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_u16_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u16_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u16_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_u16_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u16_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u16_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_u16_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u16_f16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u16_f16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_u16_f16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u16_f16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u16_f16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_u16_f16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u16_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u16_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_u16_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u16_f16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u16_f16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_u16_f16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u16_f16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u16_f16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_u16_f16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_f16_u8_u8                      		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_f16_u8_u16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_f16_u8_f16                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_f16_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_f16_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_f16_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_f16_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_f16_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_f16_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_f16_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_f16_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_f16_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_f16_u8_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_f16_u8_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_f16_u8_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_f16_u8_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_f16_u8_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_f16_u8_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_f16_u8_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_f16_u8_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_f16_u8_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_f16_u8_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_f16_u8_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_f16_u8_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_f16_u8_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_f16_u8_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_f16_u8_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_f16_u16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_f16_u16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_f16_u16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_f16_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_f16_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_f16_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_f16_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_f16_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_f16_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_f16_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_f16_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_f16_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_f16_u16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_f16_u16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_f16_u16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_f16_u16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_f16_u16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_f16_u16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_f16_u16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_f16_u16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_f16_u16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_f16_u16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_f16_u16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_f16_u16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_f16_u16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_f16_u16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_f16_u16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_f16_f16_u8                     		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_f16_f16_u16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u8_f16_f16_f16                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_f16_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_f16_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_u16_f16_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_f16_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_f16_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u8_f16_f16_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_f16_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_f16_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u8_f16_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_f16_f16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_f16_f16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_u16_f16_f16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_f16_f16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_f16_f16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_u16_f16_f16_f16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_f16_f16_u8                    		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_f16_f16_u16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u8_f16_f16_f16                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_f16_f16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_f16_f16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_u16_f16_f16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_f16_f16_u8                   		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_f16_f16_u16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_f16_f16_f16_f16_f16                  		= templated_ops::bwd_uacm_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u8_u8_u8                   		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u8_u8_u16                  		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u8_u8_f16                  		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u8_u8_u8                  		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u8_u8_u16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u8_u8_f16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u8_u8_u8                  		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u8_u8_u16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u8_u8_f16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u8_u8_u8                  		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u8_u8_u16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u8_u8_f16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u8_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u8_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u8_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u8_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u8_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u8_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u8_u8_u8                  		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u8_u8_u16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u8_u8_f16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u8_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u8_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u8_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u8_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u8_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u8_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u16_u8_u8                  		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u16_u8_u16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u16_u8_f16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u16_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u16_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u16_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u16_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u16_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u16_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u16_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u16_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u16_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u16_u8_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u16_u8_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u16_u8_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u16_u8_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u16_u8_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u16_u8_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u16_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u16_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u16_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u16_u8_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u16_u8_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u16_u8_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u16_u8_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u16_u8_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u16_u8_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_f16_u8_u8                  		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_f16_u8_u16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_f16_u8_f16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_f16_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_f16_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_f16_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_f16_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_f16_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_f16_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_f16_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_f16_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_f16_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_f16_u8_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_f16_u8_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_f16_u8_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_f16_u8_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_f16_u8_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_f16_u8_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_f16_u8_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_f16_u8_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_f16_u8_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_f16_u8_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_f16_u8_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_f16_u8_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_f16_u8_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_f16_u8_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_f16_u8_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u8_u16_u8                  		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u8_u16_u16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u8_u16_f16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u8_u16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u8_u16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u8_u16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u8_u16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u8_u16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u8_u16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u8_u16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u8_u16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u8_u16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u8_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u8_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u8_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u8_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u8_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u8_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u8_u16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u8_u16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u8_u16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u8_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u8_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u8_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u8_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u8_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u8_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u16_u16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u16_u16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u16_u16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u16_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u16_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u16_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u16_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u16_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u16_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u16_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u16_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u16_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u16_u16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u16_u16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u16_u16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u16_u16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u16_u16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u16_u16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u16_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u16_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u16_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u16_u16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u16_u16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u16_u16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u16_u16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u16_u16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u16_u16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_f16_u16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_f16_u16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_f16_u16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_f16_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_f16_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_f16_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_f16_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_f16_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_f16_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_f16_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_f16_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_f16_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_f16_u16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_f16_u16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_f16_u16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_f16_u16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_f16_u16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_f16_u16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_f16_u16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_f16_u16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_f16_u16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_f16_u16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_f16_u16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_f16_u16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_f16_u16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_f16_u16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_f16_u16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u8_f16_u8                  		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u8_f16_u16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u8_f16_f16                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u8_f16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u8_f16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u8_f16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u8_f16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u8_f16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u8_f16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u8_f16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u8_f16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u8_f16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u8_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u8_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u8_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u8_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u8_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u8_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u8_f16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u8_f16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u8_f16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u8_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u8_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u8_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u8_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u8_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u8_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u16_f16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u16_f16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_u16_f16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u16_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u16_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_u16_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u16_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u16_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_u16_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u16_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u16_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_u16_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u16_f16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u16_f16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_u16_f16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u16_f16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u16_f16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_u16_f16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u16_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u16_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_u16_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u16_f16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u16_f16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_u16_f16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u16_f16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u16_f16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_u16_f16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_f16_f16_u8                 		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_f16_f16_u16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u8_f16_f16_f16                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_f16_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_f16_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_u16_f16_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_f16_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_f16_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u8_f16_f16_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_f16_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_f16_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u8_f16_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_f16_f16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_f16_f16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_u16_f16_f16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_f16_f16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_f16_f16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_u16_f16_f16_f16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_f16_f16_u8                		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_f16_f16_u16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u8_f16_f16_f16               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_f16_f16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_f16_f16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_u16_f16_f16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_f16_f16_u8               		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_f16_f16_u16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_f16_f16_f16_f16_f16              		= templated_ops::bwd_pair_lhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u8_u8_u8                   		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u8_u8_u16                  		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u8_u8_f16                  		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u8_u8_u8                  		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u8_u8_u16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u8_u8_f16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u8_u8_u8                  		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u8_u8_u16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u8_u8_f16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u8_u8_u8                  		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u8_u8_u16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u8_u8_f16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u8_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u8_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u8_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u8_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u8_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u8_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u8_u8_u8                  		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u8_u8_u16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u8_u8_f16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u8_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u8_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u8_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u8_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u8_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u8_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u16_u8_u8                  		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u16_u8_u16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u16_u8_f16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u16_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u16_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u16_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u16_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u16_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u16_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u16_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u16_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u16_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u16_u8_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u16_u8_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u16_u8_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u16_u8_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u16_u8_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u16_u8_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u16_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u16_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u16_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u16_u8_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u16_u8_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u16_u8_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u16_u8_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u16_u8_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u16_u8_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_f16_u8_u8                  		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_f16_u8_u16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_f16_u8_f16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_f16_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_f16_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_f16_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_f16_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_f16_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_f16_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_f16_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_f16_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_f16_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_f16_u8_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_f16_u8_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_f16_u8_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_f16_u8_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_f16_u8_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_f16_u8_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_f16_u8_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_f16_u8_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_f16_u8_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_f16_u8_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_f16_u8_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_f16_u8_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_f16_u8_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_f16_u8_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_f16_u8_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u8_u16_u8                  		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u8_u16_u16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u8_u16_f16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u8_u16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u8_u16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u8_u16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u8_u16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u8_u16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u8_u16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u8_u16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u8_u16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u8_u16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u8_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u8_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u8_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u8_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u8_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u8_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u8_u16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u8_u16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u8_u16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u8_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u8_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u8_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u8_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u8_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u8_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u16_u16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u16_u16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u16_u16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u16_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u16_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u16_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u16_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u16_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u16_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u16_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u16_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u16_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u16_u16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u16_u16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u16_u16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u16_u16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u16_u16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u16_u16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u16_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u16_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u16_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u16_u16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u16_u16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u16_u16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u16_u16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u16_u16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u16_u16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_f16_u16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_f16_u16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_f16_u16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_f16_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_f16_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_f16_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_f16_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_f16_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_f16_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_f16_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_f16_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_f16_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_f16_u16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_f16_u16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_f16_u16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_f16_u16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_f16_u16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_f16_u16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_f16_u16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_f16_u16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_f16_u16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_f16_u16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_f16_u16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_f16_u16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_f16_u16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_f16_u16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_f16_u16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u8_f16_u8                  		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u8_f16_u16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u8_f16_f16                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u8_f16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u8_f16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u8_f16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u8_f16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u8_f16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u8_f16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u8_f16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u8_f16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u8_f16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u8_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u8_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u8_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u8_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u8_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u8_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u8_f16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u8_f16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u8_f16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u8_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u8_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u8_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u8_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u8_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u8_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u16_f16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u16_f16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_u16_f16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u16_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u16_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_u16_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u16_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u16_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_u16_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u16_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u16_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_u16_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u16_f16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u16_f16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_u16_f16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u16_f16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u16_f16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_u16_f16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u16_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u16_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_u16_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u16_f16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u16_f16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_u16_f16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u16_f16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u16_f16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_u16_f16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_f16_f16_u8                 		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_f16_f16_u16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u8_f16_f16_f16                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_f16_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_f16_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_u16_f16_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_f16_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_f16_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u8_f16_f16_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_f16_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_f16_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u8_f16_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_f16_f16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_f16_f16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_u16_f16_f16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_f16_f16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_f16_f16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_u16_f16_f16_f16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_f16_f16_u8                		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_f16_f16_u16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u8_f16_f16_f16               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u8_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_f16_f16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_f16_f16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_u16_f16_f16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_u16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_f16_f16_u8               		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u8_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_f16_f16_u16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_u16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_f16_f16_f16_f16_f16              		= templated_ops::bwd_pair_rhs_restrict_aligned_ops<tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, tile_f16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
};

#endif 