#ifndef __NETWORK_TILEOPS_STATIC_H__
#define __NETWORK_TILEOPS_STATIC_H__ 

#include <stdint.h>
#include <stdlib.h>
#include <type_traits> 
#include <math.h>
#include "network_xmath_host.h"
#include <limits.h>
#include <bit>

namespace dg::network_tileops_host_static::templated_ops{

    static constexpr auto pow2(size_t val) noexcept -> size_t{

        return size_t{1} << val;
    }

    static constexpr auto log2(size_t val) noexcept -> size_t{

        return static_cast<size_t>(sizeof(size_t) * CHAR_BIT - 1) - static_cast<size_t>(std::countl_zero(val));
    } 
 
    static constexpr auto sqrt(size_t val) noexcept -> size_t{

        return pow2(log2(val) >> 1);
    }

    static constexpr auto is_pow2(size_t val) noexcept -> bool{

        return val != 0u && (val & (val - 1)) == 0u;
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
    };

    template <class dst_logit_value_t, class src_logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_mono_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>;

        static inline void exp(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::exp(src[i]);
            }
        } 

        static inline void log(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::log(src[i]);
            }
        }

        static inline void clone(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = static_cast<casting_ops_t>(src[i]);
            }
        } 

        static inline void negative(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::negative(src[i]);
            }
        }

        static inline void inverse(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::div(1, src[i]);
            }
        }

        static inline void abs(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::abs(src[i]);
            }    
        }

        static inline void cos(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::cos(src[i]);
            }        
        }

        static inline void acos(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::acos(src[i]);
            }            
        }

        static inline void sin(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sin(src[i]);
            }        
        }

        static inline void asin(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{
            
            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::asin(src[i]);
            }      
        }

        static inline void tan(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::tan(src[i]);
            }      
        }

        static inline void atan(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::atan(src[i]);
            }          
        }

        static inline void transpose(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ);

            for (size_t i = 0; i < BLK_SZ; ++i){
                for (size_t j = 0; j < BLK_SZ; ++j){
                    dst[i * BLK_SZ + j] = static_cast<casting_ops_t>(src[j * BLK_SZ + i]);
                }
            }
        }
    };

    template <class dst_logit_value_t, class src_logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_uacm_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void max(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i]  = x_math::max(dst[i], src[i]);
            }
        }

        static inline void min(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i]  = x_math::min(dst[i], src[i]);
            }
        }

        static inline void sum(dst_logit_value_t * dst, const src_logit_value_t * src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src[i]);
            }
        }        
    };

    template <class dst_logit_value_t, class lhs_logit_value_t, class rhs_logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_pacm_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void add(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::add(lhs[i], rhs[i]));
            }
        }

        static inline void sub(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::sub(lhs[i], rhs[i]));
            }    
        }

        static inline void mul(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::mul(lhs[i], rhs[i]));
            }            
        }

        static inline void div(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(lhs[i], rhs[i]));
            }    
        }

        static inline void pow(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::pow(lhs[i], rhs[i]));
            }    
        }

        static inline void linear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ); 

            for (size_t j = 0; j < BLK_SZ; ++j){
                for (size_t i = 0; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::mul(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(total, dst[i * BLK_SZ + j]);
                }
            }
        }
    };

    template <class dst_logit_value_t, class lhs_logit_value_t, class rhs_logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_pair_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void add(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(lhs[i], rhs[i]);
            }
        }

        static inline void sub(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(lhs[i], rhs[i]);
            }    
        }

        static inline void mul(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::mul(lhs[i], rhs[i]);
            }            
        }

        static inline void div(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::div(lhs[i], rhs[i]);
            }    
        }

        static inline void pow(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::pow(lhs[i], rhs[i]);
            }    
        }

        static inline void linear(dst_logit_value_t * dst, const lhs_logit_value_t * lhs, const rhs_logit_value_t * rhs) noexcept{

            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ); 

            for (size_t j = 0; j < BLK_SZ; ++j){
                for (size_t i = 0; i < BLK_SZ; ++i){
                    casting_ops_t total{};
                    for (size_t z = 0; z < BLK_SZ; ++z){
                        total = x_math::add(total, x_math::mul(lhs[i * BLK_SZ + z], rhs[z * BLK_SZ + j]));
                    }
                    dst[i * BLK_SZ + j] = total;
                }
            }
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class src_grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_mono_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void exp(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], x_math::exp(dst_logit[i]), dst[i]);
            }
        }

        static inline void log(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(src_grad[i], dst_logit[i]));
            }
        }

        static inline void clone(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src_grad[i]);
            }
        }

        static inline void negative(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], src_grad[i]);
            }
        }

        static inline void inverse(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::div(src_grad[i], x_math::pow(dst_logit[i], std::integral_constant<size_t, 2>{})));
            }
        }

        static inline void abs(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], x_math::sign(dst_logit[i]), dst[i]);
            }
        }

        static inline void cos(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::mul(src_grad[i], x_math::sin(dst_logit[i])));
            }
        }

        static inline void acos(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::div(src_grad[i], x_math::sqrt(x_math::sub(1, x_math::pow(dst_logit[i], std::integral_constant<size_t, 2>{})))));
            }
        } 

        static inline void sin(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], x_math::cos(dst_logit[i]), dst[i]);
            }
        }

        static inline void asin(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(src_grad[i], x_math::sqrt(x_math::sub(1, x_math::pow(dst_logit[i], std::integral_constant<size_t, 2>{})))));
            }
        }

        static inline void tan(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(src_grad[i], x_math::pow(x_math::cos(dst_logit[i]), std::integral_constant<size_t, 2>{})));
            }
        }

        static inline void atan(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(src_grad[i], x_math::add(x_math::pow(dst_logit[i], std::integral_constant<size_t, 2>{}), 1)));
            }
        }

        static inline void transpose(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad) noexcept{
            
            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ); 

            for (size_t i = 0; i < BLK_SZ; ++i){
                for (size_t j = 0; j < BLK_SZ; ++j){
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], src_grad[j * BLK_SZ + i]);
                }
            }
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class src_logit_value_t, class src_grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_uacm_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void max(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const src_logit_value_t * src_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::eqcmp_mul(dst_logit[i], src_logit[i], src_grad[i]));
            }
        }

        static inline void min(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const src_logit_value_t * src_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::eqcmp_mul(dst_logit[i], src_logit[i], src_grad[i]));
            }
        }

        static inline void sum(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const src_logit_value_t *) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src_grad[i]);
            }
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class other_logit_value_t, class src_grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_pair_bdr_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void add(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t *) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src_grad[i]);
            }
        }

        static inline void mul(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t * other) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], other[i], dst[i]);
            }
        }

        static inline void linear(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t * other) noexcept{
            
            static_assert(templated_ops::is_pow2(SZ));
            constexpr size_t BLK_SZ = templated_ops::sqrt(SZ); 

            for (size_t i = 0; i < BLK_SZ; ++i){
                for (size_t j = 0; j < BLK_SZ; ++j){
                    casting_ops_t dot_sum{}; 
                    for (size_t z = 0; z < BLK_SZ; ++z){
                        dot_sum = x_math::add(dot_sum, x_math::mul(src_grad[i * BLK_SZ + z], other[j * BLK_SZ + z]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], dot_sum);
                }
            }
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class other_logit_value_t, class src_grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_pair_lhs_unaligned_ops: bwd_pair_bdr_unaligned_ops<dst_logit_value_t, dst_grad_value_t, other_logit_value_t, src_grad_value_t, casting_ops_t, SZ>{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void sub(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t *) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src_grad[i]);
            }
        }

        static inline void div(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(src_grad[i], other_logit[i]));
            }
        }

        static inline void pow(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], x_math::mul(other_logit[i], x_math::pow(dst_logit[i], x_math::sub(other_logit[i], 1))), dst[i]);
            }
        }
    };

    template <class dst_logit_value_t, class dst_grad_value_t, class other_logit_value_t, class src_grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_pair_rhs_unaligned_ops: bwd_pair_bdr_unaligned_ops<dst_logit_value_t, dst_grad_value_t, other_logit_value_t, src_grad_value_t, casting_ops_t, SZ>{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void sub(dst_grad_value_t * dst, const dst_logit_value_t *, const src_grad_value_t * src_grad, const other_logit_value_t *) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], src_grad[i]);
            }
        }

        static inline void div(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{
            
            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::mul(src_grad[i], x_math::div(other_logit[i], x_math::pow(dst_logit[i], std::integral_constant<size_t, 2>{})))); 
            }
        }

        static inline void pow(dst_grad_value_t * dst, const dst_logit_value_t * dst_logit, const src_grad_value_t * src_grad, const other_logit_value_t * other_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(src_grad[i], x_math::mul(x_math::pow(other_logit[i], dst_logit[i]), x_math::log(other_logit[i])), dst[i]);
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

        static __attribute__((flatten)) void linear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{
            
            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
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

        static __attribute__((flatten)) void linear(dst_logit_value_t * __restrict__ dst, const lhs_logit_value_t * __restrict__ lhs, const rhs_logit_value_t * __restrict__ rhs) noexcept{

            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
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

        static __attribute__((flatten)) void linear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
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

        static __attribute__((flatten)) void linear(dst_grad_value_t * __restrict__ dst, const dst_logit_value_t * __restrict__ dst_logit, const src_grad_value_t * __restrict__ src_grad, const other_logit_value_t * __restrict__ other_logit) noexcept{

            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(dst_logit), std::assume_aligned<ALIGNMENT_SZ>(src_grad), std::assume_aligned<ALIGNMENT_SZ>(other_logit));
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
    };
}

namespace dg::network_tileops_host_static{

    static inline constexpr size_t LOGIT_COUNT_PER_TILE     = size_t{1} << 16;
    static inline constexpr size_t ALIGNMENT_SZ             = size_t{1} << 10; 

    //
    
    // using nw_uint8_t                = uint8_t;
    // using nw_uint16_t               = uint16_t;
    // using nw_float8_t               = int8_t;
    // using nw_float16_t              = int16_t;

    // using fwd_mono_ops_uu_8_8       = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_uu_8_16      = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_uu_8_8       = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_uu_8_16      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_uu_8_8       = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_uu_8_16      = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_uu_8_8       = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_uu_8_16      = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    // using bwd_uacm_ops_uu_8_8       = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_uacm_ops_uu_8_16      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_uu_8_8   = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_uu_8_16  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    // using bwd_pair_rhs_ops_uu_8_8   = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_rhs_ops_uu_8_16  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_uu_16_8      = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_uu_16_16     = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_uu_16_8      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_uu_16_16     = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_uu_16_8      = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_uu_16_16     = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_uu_16_8      = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_uu_16_16     = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    // using bwd_uacm_ops_uu_16_8      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_uacm_ops_uu_16_16     = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_uu_16_8  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_uu_16_16 = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    // using bwd_pair_rhs_ops_uu_16_8  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_rhs_ops_uu_16_16 = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;

    // using fwd_mono_ops_uf_8_8       = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_uf_8_16      = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_uf_8_8       = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_uf_8_16      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_uf_8_8       = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_uf_8_16      = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_uf_8_8       = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_uf_8_16      = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    // using bwd_uacm_ops_uf_8_8       = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_uacm_ops_uf_8_16      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_uf_8_8   = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_uf_8_16  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    // using bwd_pair_rhs_ops_uf_8_8   = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_rhs_ops_uf_8_16  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_uf_16_8      = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_uf_16_16     = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_uf_16_8      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_uf_16_16     = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_uf_16_8      = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_uf_16_16     = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_uf_16_8      = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_uf_16_16     = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    // using bwd_uacm_ops_uf_16_8      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_uacm_ops_uf_16_16     = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_uf_16_8  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_uf_16_16 = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    // using bwd_pair_rhs_ops_uf_16_8  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_rhs_ops_uf_16_16 = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;

    // using fwd_mono_ops_fu_8_8       = templated_ops::fwd_mono_restrict_aligned_ops<nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_fu_8_16      = templated_ops::fwd_mono_restrict_aligned_ops<nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_fu_8_8       = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_fu_8_16      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_fu_8_8       = templated_ops::fwd_pair_restrict_aligned_ops<nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_fu_8_16      = templated_ops::fwd_pair_restrict_aligned_ops<nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_fu_8_8       = templated_ops::bwd_mono_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_fu_8_16      = templated_ops::bwd_mono_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    // using bwd_uacm_ops_fu_8_8       = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_uacm_ops_fu_8_16      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_fu_8_8   = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_fu_8_16  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    // using bwd_pair_rhs_ops_fu_8_8   = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_rhs_ops_fu_8_16  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_fu_16_8      = templated_ops::fwd_mono_restrict_aligned_ops<nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_fu_16_16     = templated_ops::fwd_mono_restrict_aligned_ops<nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_fu_16_8      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_fu_16_16     = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_fu_16_8      = templated_ops::fwd_pair_restrict_aligned_ops<nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_fu_16_16     = templated_ops::fwd_pair_restrict_aligned_ops<nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_fu_16_8      = templated_ops::bwd_mono_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_fu_16_16     = templated_ops::bwd_mono_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    // using bwd_uacm_ops_fu_16_8      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_uacm_ops_fu_16_16     = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_fu_16_8  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_fu_16_16 = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    // using bwd_pair_rhs_ops_fu_16_8  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_rhs_ops_fu_16_16 = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;

    // using fwd_mono_ops_ff_8_8       = templated_ops::fwd_mono_restrict_aligned_ops<nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_ff_8_16      = templated_ops::fwd_mono_restrict_aligned_ops<nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_ff_8_8       = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_ff_8_16      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_ff_8_8       = templated_ops::fwd_pair_restrict_aligned_ops<nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_ff_8_16      = templated_ops::fwd_pair_restrict_aligned_ops<nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_ff_8_8       = templated_ops::bwd_mono_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_ff_8_16      = templated_ops::bwd_mono_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    // using bwd_uacm_ops_ff_8_8       = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_uacm_ops_ff_8_16      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_ff_8_8   = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_ff_8_16  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    // using bwd_pair_rhs_ops_ff_8_8   = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_rhs_ops_ff_8_16  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_ff_16_8      = templated_ops::fwd_mono_restrict_aligned_ops<nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_mono_ops_ff_16_16     = templated_ops::fwd_mono_restrict_aligned_ops<nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_ff_16_8      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_uacm_ops_ff_16_16     = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_ff_16_8      = templated_ops::fwd_pair_restrict_aligned_ops<nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using fwd_pair_ops_ff_16_16     = templated_ops::fwd_pair_restrict_aligned_ops<nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_ff_16_8      = templated_ops::bwd_mono_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_mono_ops_ff_16_16     = templated_ops::bwd_mono_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    // using bwd_uacm_ops_ff_16_8      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_uacm_ops_ff_16_16     = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_ff_16_8  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_lhs_ops_ff_16_16 = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    // using bwd_pair_rhs_ops_ff_16_8  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    // using bwd_pair_rhs_ops_ff_16_16 = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
};

#endif 