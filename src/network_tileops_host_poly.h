#ifndef __NETWORK_TILEOPS_POLY_H__
#define __NETWORK_TILEOPS_POLY_H__

#include <stdint.h>
#include <stddef.h> 
#include <stdlib.h>
#include "network_tileops_static.h"
#include "network_memory_utility.h"
#include <limits.h>

namespace dg::network_tileops_host_poly::taxonomy{

    using ops_t     = uint8_t;
    using type_t    = uint8_t; 

    enum mono_option: ops_t{
        exp         = 0,
        ln          = 1,
        clone       = 2,
        negative    = 3,
        inverse     = 4,
        abs         = 5,
        cos         = 6,
        acos        = 7,
        sin         = 8,
        asin        = 9,
        tan         = 10,
        atan        = 11,
        transpose   = 12
    };

    enum pair_option: ops_t{
        add         = 0,
        sub         = 1,
        mul         = 2,
        div         = 3,
        pow         = 4,
        linear      = 5
    };

    enum uacm_option: ops_t{
        sum         = 0,
        max         = 1,
        min         = 2
    };

    enum type_option: type_t{
        uu_8_8      = 0,
        uu_8_16     = 1,
        uu_16_8     = 2,
        uu_16_16    = 3,
        uf_8_8      = 4,
        uf_8_16     = 5,
        uf_16_8     = 6,
        uf_16_16    = 7,
        fu_8_8      = 8,
        fu_8_16     = 9,
        fu_16_8     = 10,
        fu_16_16    = 11,
        ff_8_8      = 12,
        ff_8_16     = 13,
        ff_16_8     = 14,
        ff_16_16    = 15
    };
} 

namespace dg::network_tileops_host_poly::poly_ops{

    template <class T, class U, class OperationHandler>
    static void fwd_mono(T * dst, const U * src, const ops_t ops_opt, const OperationHandler) noexcept{

        using namespace taxonomy;

        switch (ops_opt){
            case exp:
                OperationHandler::exp(dst, src);
                break;
            case ln:
                OperationHandler::ln(dst, src);
                break;
            case clone:
                OperationHandler::clone(dst, src);
                break;
            case negative:
                OperationHandler::negative(dst, src);
                break;
            case inverse:
                OperationHandler::inverse(dst, src);
                break;
            case abs:
                OperationHandler::abs(dst, src);
                break;
            case cos:
                OperationHandler::cos(dst, src);
                break;
            case acos:
                OperationHandler::acos(dst, src);
                break;
            case sin:
                OperationHandler::sin(dst, src);
                break;
            case asin:
                OperationHandler::asin(dst, src);
                break;
            case tan:
                OperationHandler::tan(dst, src);
                break;
            case atan:
                OperationHandler::atan(dst, src);
                break;
            case transpose:
                OperationHandler::transpose(dst, src);
                break;
            default:
                std::abort();
                break;
        }
    }

    template <class T, class U, class OperationHandler>
    static void fwd_uacm(T * dst, const U * src, const ops_t ops_opt, const OperationHandler) noexcept{

        using namespace taxonomy;

        switch (ops_opt){
            case sum:
                OperationHandler::sum(dst, src);
                break;
            case max:
                OperationHandler::max(dst, src);
                break;
            case min:
                OperationHandler::min(dst, src);
                break;
            default:
                std::abort();
                break;
        }
    }

    template <class T, class U, class K, class OperationHandler>
    static void fwd_pair(T * dst, const U * lhs, const K * rhs, const ops_t ops_opt, const OperationHandler) noexcept{

        using namespace taxonomy;

        switch (ops_opt){
            case add:
                OperationHandler::add(dst, lhs, rhs);
                break;
            case mul:
                OperationHandler::mul(dst, lhs, rhs);
                break;
            case sub:
                OperationHandler::sub(dst, lhs, rhs);
                break;
            case div:
                OperationHandler::div(dst, lhs, rhs);
                break;
            case pow:
                OperationHandler::pow(dst, lhs, rhs);
                break;
            case linear:
                OperationHandler::linear(dst, lhs, rhs);
                break;
            default:
                std::abort();
                break;
        }
    }

    template <class T, class U, class K, class OperationHandler>
    static void bwd_mono(T * dst, const U * lhs_logit, const K * rhs_grad, const ops_t ops_opt, const OperationHandler) noexcept{

        using namespace taxonomy;

        switch (ops_opt){
            case exp:
                OperationHandler::exp(dst, lhs_logit, rhs_grad);
                break;
            case ln:
                OperationHandler::ln(dst, lhs_logit, rhs_grad);
                break;
            case clone:
                OperationHandler::clone(dst, lhs_logit, rhs_grad)
                break;
            case negative:
                OperationHandler::negative(dst, lhs_logit, rhs_grad);
                break;
            case inverse:
                OperationHandler::inverse(dst, lhs_logit, rhs_grad);
                break;
            case abs:
                OperationHandler::abs(dst, lhs_logit, rhs_grad);
                break;
            case cos:
                OperationHandler::cos(dst, lhs_logit, rhs_grad);
                break;
            case acos:
                OperationHandler::acos(dst, lhs_logit, rhs_grad);
                break;
            case sin:
                OperationHandler::sin(dst, lhs_logit, rhs_grad);
                break;
            case asin:
                OperationHandler::asin(dst, lhs_logit, rhs_grad);
                break;
            case tan:
                OperationHandler::tan(dst, lhs_logit, rhs_grad);
                break;
            case atan:
                OperationHandler::atan(dst, lhs_logit, rhs_grad);
                break;
            case transpose:
                OperationHandler::transpose(dst, lhs_logit, rhs_grad);
                break;
            default:
                std::abort();
                break;
        }       
    }

    template <class T, class U, class K, class G, class OperationHandler>
    static void bwd_pair(T * dst, const U * lhs_logit, const K * rhs_grad, const G * other_logit, const ops_t ops_opt, const OperationHandler) noexcept{

        using namespace taxonomy;

        switch (ops_opt){
            case add:
                OperationHandler::add(dst, lhs_logit, rhs_grad, other_logit);
                break;
            case mul:
                OperationHandler::mul(dst, lhs_logit, rhs_grad, other_logit);
                break;
            case sub:
                OperationHandler::sub(dst, lhs_logit, rhs_grad, other_logit);
                break;
            case div:
                OperationHandler::div(dst, lhs_logit, rhs_grad, other_logit);
                break;
            case pow:
                OperationHandler::pow(dst, lhs_logit, rhs_grad, other_logit);
                break;
            case linear:
                OperationHandler::linear(dst, lhs_logit, rhs_grad, other_logit);
                break;
            default:
                std::abort();
                break;
        }
    } 
} 

namespace dg::network_tileops_host_poly::poly_ops_type{

    static void fwd_mono(void * dst, const void * src, const ops_t ops_opt, const type_t type_opt) noexcept{

        using namespace taxonomy;
        using namespace dg::network_tileops_static;
        
        switch (type_opt){

            case uu_8_8:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uu_8_8{});
                break;
            case uu_8_16:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uu_8_16{});
                break;
            case uu_16_8:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uu_16_8{});
                break;
            case uu_16_16:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uu_16_16{})
                break;
            case uf_8_8:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uf_8_8{});
                break;
            case uf_8_16:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uf_8_16{});
                break;
            case uf_16_8:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uf_16_8{});
                break;
            case uf_16_16:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uf_16_16{});
                break;
            case fu_8_8:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_fu_8_8{});
                break;
            case fu_8_16:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_fu_8_16{});
                break;
            case fu_16_8:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_fu_16_8{});
                break;
            case fu_16_16:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_fu_16_16{});
                break;
            case ff_8_8:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_ff_8_8{});
                break;
            case ff_8_16:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_ff_8_16{});
                break;
            case ff_16_8:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_ff_16_8{});
                break;
            case ff_16_16:
                poly_ops::fwd_mono(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_ff_16_16{});
                break;
            default:
                std::abort();
                break;
        }
    }

    static void fwd_uacm(void * dst, const void * src, const ops_t ops_opt, const type_t type_opt) noexcept{

        using namespace taxonomy;
        using namespace dg::network_tileops_static;
        
        switch (type_opt){

            case uu_8_8:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uu_8_8{});
                break;
            case uu_8_16:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uu_8_16{});
                break;
            case uu_16_8:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uu_16_8{});
                break;
            case uu_16_16:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uu_16_16{})
                break;
            case uf_8_8:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uf_8_8{});
                break;
            case uf_8_16:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uf_8_16{});
                break;
            case uf_16_8:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uf_16_8{});
                break;
            case uf_16_16:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uf_16_16{});
                break;
            case fu_8_8:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_fu_8_8{});
                break;
            case fu_8_16:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_fu_8_16{});
                break;
            case fu_16_8:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_fu_16_8{});
                break;
            case fu_16_16:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_fu_16_16{});
                break;
            case ff_8_8:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_ff_8_8{});
                break;
            case ff_8_16:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_ff_8_16{});
                break;
            case ff_16_8:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_ff_16_8{});
                break;
            case ff_16_16:
                poly_ops::fwd_uacm(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_ff_16_16{});
                break;
            default:
                std::abort();
                break;
        }
    }

    static void fwd_pacm(void * dst, const void * lhs, const void * rhs, const ops_t ops_opt, const type_t type_opt) noexcept{

        using namespace taxonomy;
        using namespace dg::network_tileops_static;
        
        switch (type_opt){

            case uu_8_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uu_8_8{});
                break;
            case uu_8_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uu_8_16{});
                break;
            case uu_16_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uu_16_8{});
                break;
            case uu_16_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uu_16_16{})
                break;
            case uf_8_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uf_8_8{});
                break;
            case uf_8_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uf_8_16{});
                break;
            case uf_16_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uf_16_8{});
                break;
            case uf_16_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uf_16_16{});
                break;
            case fu_8_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_fu_8_8{});
                break;
            case fu_8_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_fu_8_16{});
                break;
            case fu_16_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_fu_16_8{});
                break;
            case fu_16_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_fu_16_16{});
                break;
            case ff_8_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_ff_8_8{});
                break;
            case ff_8_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_ff_8_16{});
                break;
            case ff_16_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_ff_16_8{});
                break;
            case ff_16_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_ff_16_16{});
                break;
            default:
                std::abort();
                break;
        }
    }

    static void fwd_pair(void * dst, const void * lhs, const void * rhs, const ops_t ops_opt, const type_t type_opt) noexcept{

        using namespace taxonomy;
        using namespace dg::network_tileops_static;
        
        switch (type_opt){

            case uu_8_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uu_8_8{});
                break;
            case uu_8_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uu_8_16{});
                break;
            case uu_16_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uu_16_8{});
                break;
            case uu_16_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uu_16_16{})
                break;
            case uf_8_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uf_8_8{});
                break;
            case uf_8_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uf_8_16{});
                break;
            case uf_16_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uf_16_8{});
                break;
            case uf_16_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uf_16_16{});
                break;
            case fu_8_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_fu_8_8{});
                break;
            case fu_8_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_fu_8_16{});
                break;
            case fu_16_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_fu_16_8{});
                break;
            case fu_16_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_fu_16_16{});
                break;
            case ff_8_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_ff_8_8{});
                break;
            case ff_8_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_ff_8_16{});
                break;
            case ff_16_8:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_ff_16_8{});
                break;
            case ff_16_16:
                poly_ops::fwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_ff_16_16{});
                break;
            default:
                std::abort();
                break;
        }
    }

    static void bwd_mono(void * dst, const void * lhs_logit, const void * rhs_grad, const ops_t ops_opt, const type_t type_opt) noexcept{

        using namespace taxonomy;
        using namespace dg::network_tileops_static;
        
        switch (type_opt){

            case uu_8_8:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uu_8_8{});
                break;
            case uu_8_16:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uu_8_16{});
                break;
            case uu_16_8:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uu_16_8{});
                break;
            case uu_16_16:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uu_16_16{})
                break;
            case uf_8_8:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uf_8_8{});
                break;
            case uf_8_16:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uf_8_16{});
                break;
            case uf_16_8:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uf_16_8{});
                break;
            case uf_16_16:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uf_16_16{});
                break;
            case fu_8_8:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_fu_8_8{});
                break;
            case fu_8_16:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_fu_8_16{});
                break;
            case fu_16_8:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_fu_16_8{});
                break;
            case fu_16_16:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_fu_16_16{});
                break;
            case ff_8_8:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_ff_8_8{});
                break;
            case ff_8_16:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_ff_8_16{});
                break;
            case ff_16_8:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_ff_16_8{});
                break;
            case ff_16_16:
                poly_ops::bwd_mono(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_ff_16_16{});
                break;
            default:
                std::abort();
                break;
        }
    }

    static void bwd_pair_rhs(void * dst, const void * lhs_logit, const void * rhs_grad, const void * rhs_lhs_logit, const ops_t ops_opt, const type_t type_opt) noexcept{

        using namespace taxonomy;
        using namespace dg::network_tileops_static;
        
        switch (type_opt){

            case uu_8_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uu_8_8{});
                break;
            case uu_8_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uu_8_16{});
                break;
            case uu_16_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uu_16_8{});
                break;
            case uu_16_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uu_16_16{})
                break;
            case uf_8_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uf_8_8{});
                break;
            case uf_8_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uf_8_16{});
                break;
            case uf_16_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uf_16_8{});
                break;
            case uf_16_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uf_16_16{});
                break;
            case fu_8_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_fu_8_8{});
                break;
            case fu_8_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_fu_8_16{});
                break;
            case fu_16_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_fu_16_8{});
                break;
            case fu_16_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_fu_16_16{});
                break;
            case ff_8_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_ff_8_8{});
                break;
            case ff_8_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_ff_8_16{});
                break;
            case ff_16_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_ff_16_8{});
                break;
            case ff_16_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_ff_16_16{});
                break;
            default:
                std::abort();
                break;
        }
    }

    static void bwd_pair_lhs(void * dst, const void * lhs_logit, const void * rhs_grad, const void * rhs_rhs_logit, const ops_t ops_opt, const type_t type_opt) noexcept{

        using namespace taxonomy;
        using namespace dg::network_tileops_static;
        
        switch (type_opt){

            case uu_8_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uu_8_8{});
                break;
            case uu_8_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uu_8_16{});
                break;
            case uu_16_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uu_16_8{});
                break;
            case uu_16_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uu_16_16{})
                break;
            case uf_8_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uf_8_8{});
                break;
            case uf_8_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uf_8_16{});
                break;
            case uf_16_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uf_16_8{});
                break;
            case uf_16_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_uint16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_uint16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uf_16_16{});
                break;
            case fu_8_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_fu_8_8{});
                break;
            case fu_8_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_fu_8_16{});
                break;
            case fu_16_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_fu_16_8{});
                break;
            case fu_16_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_fu_16_16{});
                break;
            case ff_8_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_ff_8_8{});
                break;
            case ff_8_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float8_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_ff_8_16{});
                break;
            case ff_16_8:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_ff_16_8{});
                break;
            case ff_16_16:
                poly_ops::bwd_pair(memory_utility::start_lifetime_as_array<nw_float16_t>(dst, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), memory_utility::start_lifetime_as_array<nw_float16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_ff_16_16{});
                break;
            default:
                std::abort();
                break;
        }
    } 
} 

namespace dg::network_tileops_host_poly{

    using namespace taxonomy;
    using dispatch_t = uint16_t; 

    constexpr auto dispatch_make(ops_t ops_opt, type_t type_opt) noexcept -> dispatch_t{

        static_assert(sizeof(ops_t) + sizeof(cast_t) <= sizeof(dispatch_t));
        return (static_cast<dispatch_t>(ops_opt) << (sizeof(cast_t) * CHAR_BIT)) | static_cast<dispatch_t>(type_opt);
    } 

    static constexpr auto dispatch_extract_ops(dispatch_t dispatch_opt) noexcept -> ops_t{

        return dispatch_opt >> (sizeof(cast_t) * CHAR_BIT);
    } 

    static constexpr auto dispatch_extract_type(dispatch_t dispatch_opt) noexcept -> cast_t{

        return dispatch_opt & low<dispatch_t>(std::integral_constant<size_t, sizeof(cast_t) * CHAR_BIT>{});
    }

    void fwd_mono(void * dst, const void * src, const dispatch_t dispatch_opt) noexcept{

        poly_ops_type::fwd_mono(dst, src, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    } 

    void fwd_uacm(void * dst, const void * src, const dispatch_t dispatch_opt) noexcept{
        
        poly_ops_type::fwd_uacm(dst, src, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }
    
    void fwd_pacm(void * dst, const void * lhs, const void * rhs, const dispatch_t dispatch_opt) noexcept{

        poly_ops_type::fwd_pacm(dst, lhs, rhs, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }

    void fwd_pair(void * dst, const void * lhs, const void * rhs, const dispatch_t dispatch_opt) noexcept{

        poly_ops_type::fwd_pair(dst, lhs, rhs, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }

    void bwd_mono(void * dst, const void * lhs_logit, const void * rhs_grad, const dispatch_t dispatch_opt) noexcept{

        poly_ops_type::bwd_mono(dst, lhs_logit, rhs_grad, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }

    void bwd_pair_rhs(void * dst, const void * lhs_logit, const void * rhs_grad, const void * rhs_lhs_logit, const dispatch_t dispatch_opt) noexcept{

        poly_ops_type::bwd_pair_rhs(dst, lhs_logit, rhs_grad, rhs_lhs_logit, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }

    void bwd_pair_lhs(void * dst, const void * lhs_logit, const void * rhs_grad, const void * rhs_rhs_logit, const dispatch_t dispatch_opt) noexcept{

        poly_ops_type::bwd_pair_lhs(dst, lhs_logit, rhs_grad, rhs_rhs_logit, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }
}

#endif