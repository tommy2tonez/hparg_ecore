#ifndef __NETWORK_TILEOPS_STATIC_H__
#define __NETWORK_TILEOPS_STATIC_H__ 

#include <stdint.h>
#include <stdlib.h>
#include <type_traits> 
#include <math.h>
#include "network_xmath_host.h"

namespace dg::network_tileops_host_static::templated_ops{

    static constexpr auto pow2(size_t val) noexcept -> size_t{

        return size_t{1} << val;
    }

    static constexpr auto log2(size_t val) noexcept -> size_t{

        return std::countr_zero(val);
    } 
 
    static constexpr auto sqrt2(size_t val) noexcept -> size_t{

        return pow2(log2(val) >> 1);
    }

    template <class arithmetic_ops_t>
    struct coerced_x_math{

        static inline auto sign(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::sign(value);
        }

        static inline auto exp(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

            return network_xmath_host::exp(value);
        }

        static inline auto ln(arithmetic_ops_t value) noexcept -> arithmetic_ops_t{

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

    template <class logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_mono_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>;

        static inline void exp(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::exp(src[i]);
            }
        } 

        static inline void ln(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::ln(src[i]);
            }
        }

        static inline void clone(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = static_cast<casting_ops_t>(src[i]);
            }
        } 

        static inline void negative(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::negative(src[i]);
            }
        }

        static inline void inverse(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::div(1, src[i]);
            }
        }

        static inline void abs(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::abs(src[i]);
            }    
        }

        static inline void cos(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::cos(src[i]);
            }        
        }

        static inline void acos(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::acos(src[i]);
            }            
        }

        static inline void sin(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sin(src[i]);
            }        
        }

        static inline void asin(logit_value_t * const dst, const logit_value_t * const src) noexcept{
            
            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::asin(src[i]);
            }      
        }

        static inline void tan(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::tan(src[i]);
            }      
        }

        static inline void atan(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::atan(src[i]);
            }          
        }

        static inline void transpose(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            constexpr size_t BLK_SZ = sqrt2(SZ);

            for (size_t i = 0; i < BLK_SZ; ++i){
                for (size_t j = 0; j < BLK_SZ; ++j){
                    dst[i * BLK_SZ + j] = static_cast<casting_ops_t>(src[j * BLK_SZ + i]);
                }
            }
        }
    };

    template <class logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_uacm_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void max(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i]  = x_math::max(dst[i], src[i]);
            }
        }

        static inline void min(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i]  = x_math::min(dst[i], src[i]);
            }
        }

        static inline void sum(logit_value_t * const dst, const logit_value_t * const src) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], src[i]);
            }
        }        
    };

    template <class logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_pacm_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void add(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::add(lhs[i], rhs[i]));
            }
        }

        static inline void sub(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::sub(lhs[i], rhs[i]));
            }    
        }

        static inline void mul(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::mul(lhs[i], rhs[i]));
            }            
        }

        static inline void div(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(lhs[i], rhs[i]));
            }    
        }

        static inline void pow(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::pow(lhs[i], rhs[i]));
            }    
        }

        static inline void linear(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{
            
            constexpr size_t BLK_SZ = sqrt2(SZ); 

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

    template <class logit_value_t, class casting_ops_t, size_t SZ>
    struct fwd_pair_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void add(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(lhs[i], rhs[i]);
            }
        }

        static inline void sub(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(lhs[i], rhs[i]);
            }    
        }

        static inline void mul(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::mul(lhs[i], rhs[i]);
            }            
        }

        static inline void div(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::div(lhs[i], rhs[i]);
            }    
        }

        static inline void pow(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::pow(lhs[i], rhs[i]);
            }    
        }

        static inline void linear(logit_value_t * const dst, const logit_value_t * const lhs, const logit_value_t * const rhs) noexcept{
            
            constexpr size_t BLK_SZ = sqrt2(SZ); 

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

    template <class logit_value_t, class grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_mono_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void exp(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(rhs_grad[i], x_math::exp(lhs_logit[i]), dst[i]);
            }
        }

        static inline void ln(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(rhs_grad[i], lhs_logit[i]));
            }
        }

        static inline void clone(grad_value_t * const dst, const logit_value_t * const, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], rhs_grad[i]);
            }
        }

        static inline void negative(grad_value_t * const dst, const logit_value_t * const, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], rhs_grad[i]);
            }
        }

        static inline void inverse(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::div(rhs_grad[i], x_math::pow(lhs_logit[i], std::integral_constant<size_t, 2>{})));
            }
        }

        static inline void abs(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(rhs_grad[i], x_math::sign(lhs_logit[i]), dst[i]);
            }
        }

        static inline void cos(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::mul(rhs_grad[i], x_math::sin(lhs_logit[i])));
            }
        }

        static inline void acos(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::div(rhs_grad[i], x_math::sqrt(x_math::sub(1, x_math::pow(lhs_logit[i], std::integral_constant<size_t, 2>{})))));
            }
        } 

        static inline void sin(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(rhs_grad[i], x_math::cos(lhs_logit[i]), dst[i]);
            }
        }

        static inline void asin(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(rhs_grad[i], x_math::sqrt(x_math::sub(1, x_math::pow(lhs_logit[i], std::integral_constant<size_t, 2>{})))));
            }
        }

        static inline void tan(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(rhs_grad[i], x_math::pow(x_math::cos(lhs_logit[i]), std::integral_constant<size_t, 2>{})));
            }
        }

        static inline void atan(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(rhs_grad[i], x_math::add(x_math::pow(lhs_logit[i], std::integral_constant<size_t, 2>{}), 1)));
            }
        }

        static inline void transpose(grad_value_t * const dst, const logit_value_t * const, const grad_value_t * const rhs_grad) noexcept{
            
            constexpr size_t BLK_SZ = sqrt2(SZ);

            for (size_t i = 0; i < BLK_SZ; ++i){
                for (size_t j = 0; j < BLK_SZ; ++j){
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], rhs_grad[j * BLK_SZ + i]);
                }
            }
        }
    };

    template <class logit_value_t, class grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_uacm_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void max(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad, const logit_value_t * const rhs_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::eqcmp_mul(lhs_logit[i], rhs_logit[i], rhs_grad[i]));
            }
        }

        static inline void min(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad, const logit_value_t * const rhs_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::eqcmp_mul(lhs_logit[i], rhs_logit[i], rhs_grad[i]));
            }
        }

        static inline void sum(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad, const logit_value_t * const) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], rhs_grad[i]);
            }
        }
    };
    
    template <class logit_value_t, class grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_pair_bdr_unaligned_ops{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void add(grad_value_t * const dst, const logit_value_t * const, const grad_value_t * const rhs_grad, const logit_value_t * const) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], rhs_grad[i]);
            }
        }

        static inline void mul(grad_value_t * const dst, const logit_value_t * const, const grad_value_t * const rhs_grad, const logit_value_t * const other) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(rhs_grad[i], other[i], dst[i]);
            }
        }

        static inline void linear(grad_value_t * const dst, const logit_value_t * const, const grad_value_t * const rhs_grad, const logit_value_t * const other) noexcept{
            
            constexpr size_t BLK_SZ = sqrt2(SZ);

            for (size_t i = 0; i < BLK_SZ; ++i){
                for (size_t j = 0; j < BLK_SZ; ++j){
                    casting_ops_t dot_sum{}; 
                    for (size_t z = 0; z < BLK_SZ; ++z){
                        dot_sum = x_math::add(dot_sum, x_math::mul(rhs_grad[i * BLK_SZ + z], other[j * BLK_SZ + z]));
                    }
                    dst[i * BLK_SZ + j] = x_math::add(dst[i * BLK_SZ + j], dot_sum);
                }
            }
        }
    };

    template <class logit_value_t, class grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_pair_lhs_unaligned_ops: bwd_pair_bdr_unaligned_ops<logit_value_t, grad_value_t, casting_ops_t, SZ>{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void sub(grad_value_t * const dst, const logit_value_t * const, const grad_value_t * const rhs_grad, const logit_value_t * const) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], rhs_grad[i]);
            }
        }

        static inline void div(grad_value_t * const dst, const logit_value_t * const, const grad_value_t * const rhs_grad, const logit_value_t * const rhs_rhs_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::add(dst[i], x_math::div(rhs_grad[i], rhs_rhs_logit[i]));
            }
        }

        static inline void pow(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad, const logit_value_t * const rhs_rhs_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(rhs_grad[i], x_math::mul(rhs_rhs_logit[i], x_math::pow(lhs_logit[i], x_math::sub(rhs_rhs_logit[i], 1))), dst[i]);
            }
        }
    };

    template <class logit_value_t, class grad_value_t, class casting_ops_t, size_t SZ>
    struct bwd_pair_rhs_unaligned_ops: bwd_pair_bdr_unaligned_ops<logit_value_t, grad_value_t, casting_ops_t, SZ>{

        using x_math = coerced_x_math<casting_ops_t>; 

        static inline void sub(grad_value_t * const dst, const logit_value_t * const, const grad_value_t * const rhs_grad, const logit_value_t * const) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], rhs_grad[i]);
            }
        }

        static inline void div(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad, const logit_value_t * const rhs_lhs_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::sub(dst[i], x_math::mul(rhs_grad[i], x_math::div(rhs_lhs_logit[i], x_math::pow(lhs_logit[i], std::integral_constant<size_t, 2>{})))); 
            }
        }

        static inline void pow(grad_value_t * const dst, const logit_value_t * const lhs_logit, const grad_value_t * const rhs_grad, const logit_value_t * rhs_lhs_logit) noexcept{

            for (size_t i = 0; i < SZ; ++i){
                dst[i] = x_math::fma(rhs_grad[i], x_math::mul(x_math::pow(rhs_lhs_logit[i], lhs_logit[i]), x_math::log(rhs_lhs_logit[i])), dst[i]);
            }
        }
    };

    template <class logit_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct fwd_mono_restrict_aligned_ops{

        using base = fwd_mono_unaligned_ops<logit_value_t, casting_ops_t, SZ>;

        static __attribute__((flatten)) void exp(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::exp(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        } 

        static __attribute__((flatten)) void ln(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::ln(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void clone(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::clone(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        } 

        static __attribute__((flatten)) void negative(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::negative(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void inverse(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::inverse(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void abs(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::abs(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void cos(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::cos(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void acos(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::acos(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void sin(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::sin(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void asin(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{
            
            base::asin(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void tan(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::tan(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void atan(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::atan(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }
    };

    template <class logit_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct fwd_uacm_restrict_aligned_ops{

        using base = fwd_uacm_unaligned_ops<logit_value_t, casting_ops_t, SZ>;

        static __attribute__((flatten)) void max(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::max(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void min(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::min(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }

        static __attribute__((flatten)) void sum(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const src) noexcept{

            base::sum(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(src));
        }
    };

    template <class logit_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct fwd_pacm_restrict_aligned_ops{

        using base = fwd_pacm_unaligned_ops<logit_value_t, casting_ops_t, SZ>;

        static __attribute__((flatten)) void add(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::add(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void sub(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::sub(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void mul(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::mul(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void div(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::div(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void pow(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::pow(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void linear(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{
            
            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }
    };

    template <class logit_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct fwd_pair_restrict_aligned_ops{

        using base      = fwd_pair_unaligned_ops<logit_value_t, casting_ops_t, SZ>;
        using x_math    = coerced_x_math<casting_ops_t>;

        static __attribute__((flatten)) void add(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::add(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void sub(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::sub(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void mul(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::mul(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void div(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::div(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void pow(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::pow(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }

        static __attribute__((flatten)) void linear(logit_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs, const logit_value_t * __restrict__ const rhs) noexcept{

            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs), std::assume_aligned<ALIGNMENT_SZ>(rhs));
        }
    };

    template <class logit_value_t, class grad_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct bwd_mono_restrict_aligned_ops{

        using base = bwd_mono_unaligned_ops<logit_value_t, grad_value_t, casting_ops_t, SZ>; 

        static __attribute__((flatten)) void exp(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::exp(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }

        static __attribute__((flatten)) void ln(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::ln(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }

        static __attribute__((flatten)) void clone(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{
            
            base::clone(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }

        static __attribute__((flatten)) void negative(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::negative(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }

        static __attribute__((flatten)) void inverse(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{
            
            base::inverse(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }

        static __attribute__((flatten)) void abs(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::abs(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }

        static __attribute__((flatten)) void cos(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::cos(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }

        static __attribute__((flatten)) void acos(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::acos(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        } 

        static __attribute__((flatten)) void sin(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::sin(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }

        static __attribute__((flatten)) void asin(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::asin(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }

        static __attribute__((flatten)) void tan(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::tan(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }

        static __attribute__((flatten)) void atan(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::atan(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }
    
        static __attribute__((flatten)) void transpose(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad) noexcept{

            base::transpose(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad));
        }
    };

    template <class logit_value_t, class grad_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct bwd_uacm_restrict_aligned_ops{

        using base = bwd_uacm_unaligned_ops<logit_value_t, grad_value_t, casting_ops_t, SZ>; 

        static __attribute__((flatten)) void max(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_logit) noexcept{

            base::max(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_logit));
        }

        static __attribute__((flatten)) void min(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_logit) noexcept{

            base::min(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_logit));
        }

        static __attribute__((flatten)) void sum(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_logit) noexcept{

            base::sum(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_logit));
        }
    };

    template <class logit_value_t, class grad_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct bwd_pair_lhs_restrict_aligned_ops{

        using base = bwd_pair_lhs_unaligned_ops<logit_value_t, grad_value_t, casting_ops_t, SZ>; 

        static __attribute__((flatten)) void add(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_rhs_logit) noexcept{

            base::add(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_rhs_logit));
        }

        static __attribute__((flatten)) void mul(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_rhs_logit) noexcept{

            base::mul(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_rhs_logit));
        }

        static __attribute__((flatten)) void linear(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_rhs_logit) noexcept{

            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_rhs_logit));
        }

        static __attribute__((flatten)) void sub(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_rhs_logit) noexcept{

            base::sub(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_rhs_logit));
        }

        static __attribute__((flatten)) void div(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_rhs_logit) noexcept{

            base::div(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_rhs_logit));
        }

        static __attribute__((flatten)) void pow(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_rhs_logit) noexcept{

            base::pow(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_rhs_logit));
        }
    };

    template <class logit_value_t, class grad_value_t, class casting_ops_t, size_t ALIGNMENT_SZ, size_t SZ>
    struct bwd_pair_rhs_restrict_aligned_ops{

        using base = bwd_pair_rhs_unaligned_ops<logit_value_t, grad_value_t, casting_ops_t, SZ>;

        static __attribute__((flatten)) void add(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_lhs_logit) noexcept{

            base::add(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_lhs_logit));
        } 

        static __attribute__((flatten)) void mul(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_lhs_logit) noexcept{

            base::mul(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_lhs_logit));
        }

        static __attribute__((flatten)) void linear(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_rhs_logit) noexcept{

            base::linear(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_rhs_logit));
        }

        static __attribute__((flatten)) void sub(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_lhs_logit) noexcept{

            base::sub(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_lhs_logit));
        }

        static __attribute__((flatten)) void div(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_lhs_logit) noexcept{

            base::div(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_lhs_logit));
        }

        static __attribute__((flatten)) void pow(grad_value_t * __restrict__ const dst, const logit_value_t * __restrict__ const lhs_logit, const grad_value_t * __restrict__ const rhs_grad, const logit_value_t * __restrict__ const rhs_lhs_logit) noexcept{

            base::pow(std::assume_aligned<ALIGNMENT_SZ>(dst), std::assume_aligned<ALIGNMENT_SZ>(lhs_logit), std::assume_aligned<ALIGNMENT_SZ>(rhs_grad), std::assume_aligned<ALIGNMENT_SZ>(rhs_lhs_logit));
        }
    };

}

namespace dg::network_tileops_host_static{

    static inline constexpr size_t LOGIT_COUNT_PER_TILE     = size_t{1} << 16;
    static inline constexpr size_t ALIGNMENT_SZ             = size_t{1} << 10; 

    using nw_uint8_t                = uint8_t;
    using nw_uint16_t               = uint16_t;
    using nw_float8_t               = int8_t;
    using nw_float16_t              = int16_t;

    using fwd_mono_ops_uu_8_8       = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_uu_8_16      = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_uu_8_8       = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_uu_8_16      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_uu_8_8       = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_uu_8_16      = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_uu_8_8       = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_uu_8_16      = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    using bwd_uacm_ops_uu_8_8       = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_uu_8_16      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_uu_8_8   = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_uu_8_16  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    using bwd_pair_rhs_ops_uu_8_8   = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_uu_8_16  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_uu_16_8      = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_uu_16_16     = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_uu_16_8      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_uu_16_16     = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_uu_16_8      = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_uu_16_16     = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_uu_16_8      = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_uu_16_16     = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    using bwd_uacm_ops_uu_16_8      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_uu_16_16     = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_uu_16_8  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_uu_16_16 = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    using bwd_pair_rhs_ops_uu_16_8  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_uu_16_16 = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;

    using fwd_mono_ops_uf_8_8       = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_uf_8_16      = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_uf_8_8       = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_uf_8_16      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_uf_8_8       = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_uf_8_16      = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_uf_8_8       = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_uf_8_16      = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    using bwd_uacm_ops_uf_8_8       = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_uf_8_16      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_uf_8_8   = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_uf_8_16  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    using bwd_pair_rhs_ops_uf_8_8   = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_uf_8_16  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint8_t, nw_uint8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_uf_16_8      = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_uf_16_16     = templated_ops::fwd_mono_restrict_aligned_ops<nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_uf_16_8      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_uf_16_16     = templated_ops::fwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_uf_16_8      = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_uf_16_16     = templated_ops::fwd_pair_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_uf_16_8      = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_uf_16_16     = templated_ops::bwd_mono_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    using bwd_uacm_ops_uf_16_8      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_uf_16_16     = templated_ops::bwd_uacm_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_uf_16_8  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_uf_16_16 = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    using bwd_pair_rhs_ops_uf_16_8  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_uf_16_16 = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_uint16_t, nw_uint16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;

    using fwd_mono_ops_fu_8_8       = templated_ops::fwd_mono_restrict_aligned_ops<nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_fu_8_16      = templated_ops::fwd_mono_restrict_aligned_ops<nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_fu_8_8       = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_fu_8_16      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_fu_8_8       = templated_ops::fwd_pair_restrict_aligned_ops<nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_fu_8_16      = templated_ops::fwd_pair_restrict_aligned_ops<nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_fu_8_8       = templated_ops::bwd_mono_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_fu_8_16      = templated_ops::bwd_mono_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    using bwd_uacm_ops_fu_8_8       = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_fu_8_16      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_fu_8_8   = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_fu_8_16  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    using bwd_pair_rhs_ops_fu_8_8   = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_fu_8_16  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_fu_16_8      = templated_ops::fwd_mono_restrict_aligned_ops<nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_fu_16_16     = templated_ops::fwd_mono_restrict_aligned_ops<nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_fu_16_8      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_fu_16_16     = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_fu_16_8      = templated_ops::fwd_pair_restrict_aligned_ops<nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_fu_16_16     = templated_ops::fwd_pair_restrict_aligned_ops<nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_fu_16_8      = templated_ops::bwd_mono_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_fu_16_16     = templated_ops::bwd_mono_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    using bwd_uacm_ops_fu_16_8      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_fu_16_16     = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_fu_16_8  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_fu_16_16 = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    using bwd_pair_rhs_ops_fu_16_8  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_fu_16_16 = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_uint16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;

    using fwd_mono_ops_ff_8_8       = templated_ops::fwd_mono_restrict_aligned_ops<nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_ff_8_16      = templated_ops::fwd_mono_restrict_aligned_ops<nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_ff_8_8       = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_ff_8_16      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_ff_8_8       = templated_ops::fwd_pair_restrict_aligned_ops<nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_ff_8_16      = templated_ops::fwd_pair_restrict_aligned_ops<nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_ff_8_8       = templated_ops::bwd_mono_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_ff_8_16      = templated_ops::bwd_mono_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    using bwd_uacm_ops_ff_8_8       = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_ff_8_16      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_ff_8_8   = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_ff_8_16  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    using bwd_pair_rhs_ops_ff_8_8   = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_ff_8_16  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float8_t, nw_float8_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_ff_16_8      = templated_ops::fwd_mono_restrict_aligned_ops<nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_mono_ops_ff_16_16     = templated_ops::fwd_mono_restrict_aligned_ops<nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_ff_16_8      = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_uacm_ops_ff_16_16     = templated_ops::fwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_ff_16_8      = templated_ops::fwd_pair_restrict_aligned_ops<nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using fwd_pair_ops_ff_16_16     = templated_ops::fwd_pair_restrict_aligned_ops<nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_ff_16_8      = templated_ops::bwd_mono_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_mono_ops_ff_16_16     = templated_ops::bwd_mono_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;  
    using bwd_uacm_ops_ff_16_8      = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_uacm_ops_ff_16_16     = templated_ops::bwd_uacm_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_ff_16_8  = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_lhs_ops_ff_16_16 = templated_ops::bwd_pair_lhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>; 
    using bwd_pair_rhs_ops_ff_16_8  = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float8_t,  ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
    using bwd_pair_rhs_ops_ff_16_16 = templated_ops::bwd_pair_rhs_restrict_aligned_ops<nw_float16_t, nw_float16_t, nw_float16_t, ALIGNMENT_SZ, LOGIT_COUNT_PER_TILE>;
};

#endif 