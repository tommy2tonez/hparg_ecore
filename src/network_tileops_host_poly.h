#ifndef __NETWORK_TILEOPS_POLY_H__
#define __NETWORK_TILEOPS_POLY_H__

#include <stdint.h>
#include <stddef.h> 
#include <stdlib.h>
#include "network_tileops_host_static.h"
#include "network_memult.h"
#include <limits.h>

namespace dg::network_tileops_host_poly::taxonomy{

    using ops_kind_t    = uint8_t;
    using tile_kind_t   = uint8_t; 

    enum enum_ops_kind: ops_kind_t{
        exp         = 0u,
        ln          = 1u,
        clone       = 2u,
        negative    = 3u,
        inverse     = 4u,
        abs         = 5u,
        cos         = 6u,
        acos        = 7u,
        sin         = 8u,
        asin        = 9u,
        tan         = 10u,
        atan        = 11u,
        transpose   = 12u,
        add         = 13u,
        sub         = 14u,
        mul         = 15u,
        div         = 16u,
        pow         = 17u,
        linear      = 18u,
        sum         = 19u,
        max         = 20u,
        min         = 21u
    };

    enum enum_tile_kind: tile_kind_t{
        uu_8_8      = 0u,
        uu_8_16     = 1u,
        uu_16_8     = 2u,
        uu_16_16    = 3u,
        uf_8_8      = 4u,
        uf_8_16     = 5u,
        uf_16_8     = 6u,
        uf_16_16    = 7u,
        fu_8_8      = 8u,
        fu_8_16     = 9u,
        fu_16_8     = 10u,
        fu_16_16    = 11u,
        ff_8_8      = 12u,
        ff_8_16     = 13u,
        ff_16_8     = 14u,
        ff_16_16    = 15u
    };
} 

namespace dg::network_tileops_host_poly::ops_dispatcher{

    template <class T, class U, class OperationHandler>
    static void fwd_mono(T *  dst, const U * src, ops_kind_t ops_kind, const OperationHandler) noexcept{

        using namespace taxonomy;

        switch (ops_kind){
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
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    }

    template <class T, class U, class OperationHandler>
    static void fwd_uacm(T * dst, const U * src, ops_kind_t ops_kind, const OperationHandler) noexcept{

        using namespace taxonomy;

        switch (ops_kind){
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
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    }

    template <class T, class U, class K, class OperationHandler>
    static void fwd_pair(T * dst, const U * lhs, const K * rhs, ops_kind_t ops_kind, const OperationHandler) noexcept{

        using namespace taxonomy;

        switch (ops_kind){
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
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    }

    template <class T, class U, class K, class OperationHandler>
    static void bwd_mono(T * dst, const U * lhs_logit, const K * rhs_grad, ops_kind_t ops_kind, const OperationHandler) noexcept{

        using namespace taxonomy;

        switch (ops_kind){
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
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;            
                } else{
                    std::unreachable();
                    break;
                }
        }       
    }

    template <class T, class U, class K, class G, class OperationHandler>
    static void bwd_pair(T * dst, const U * lhs_logit, const K * rhs_grad, const G * other_logit, ops_kind_t ops_kind, const OperationHandler) noexcept{

        using namespace taxonomy;

        switch (ops_kind){
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
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    }
}

namespace dg::network_tileops_host_poly::dispatcher{

    static void fwd_mono(void * __restrict__ dst, const void * __restrict__ src, ops_kind_t ops_kind, tile_kind_t tile_kind) noexcept{

        using namespace taxonomy;

        switch (tile_kind){
            case uu_8_8:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uu_8_8{});
                break;
            case uu_8_16:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uu_8_16{});
                break;
            case uu_16_8:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uu_16_8{});
                break;
            case uu_16_16:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uu_16_16{})
                break;
            case uf_8_8:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uf_8_8{});
                break;
            case uf_8_16:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uf_8_16{});
                break;
            case uf_16_8:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uf_16_8{});
                break;
            case uf_16_16:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_uf_16_16{});
                break;
            case fu_8_8:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_fu_8_8{});
                break;
            case fu_8_16:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_fu_8_16{});
                break;
            case fu_16_8:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_fu_16_8{});
                break;
            case fu_16_16:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_fu_16_16{});
                break;
            case ff_8_8:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_ff_8_8{});
                break;
            case ff_8_16:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_ff_8_16{});
                break;
            case ff_16_8:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_ff_16_8{});
                break;
            case ff_16_16:
                ops_dispatcher::fwd_mono(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_mono_ops_ff_16_16{});
                break;
            default:
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    }

    static void fwd_uacm(void * __restrict__ dst, const void * __restrict__ src, ops_kind_t ops_kind, tile_kind_t tile_kind) noexcept{

        using namespace taxonomy;
        
        switch (tile_kind){
            case uu_8_8:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uu_8_8{});
                break;
            case uu_8_16:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uu_8_16{});
                break;
            case uu_16_8:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uu_16_8{});
                break;
            case uu_16_16:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uu_16_16{})
                break;
            case uf_8_8:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uf_8_8{});
                break;
            case uf_8_16:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uf_8_16{});
                break;
            case uf_16_8:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uf_16_8{});
                break;
            case uf_16_16:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_uf_16_16{});
                break;
            case fu_8_8:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_fu_8_8{});
                break;
            case fu_8_16:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_fu_8_16{});
                break;
            case fu_16_8:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_fu_16_8{});
                break;
            case fu_16_16:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_fu_16_16{});
                break;
            case ff_8_8:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_ff_8_8{});
                break;
            case ff_8_16:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_ff_8_16{});
                break;
            case ff_16_8:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_ff_16_8{});
                break;
            case ff_16_16:
                ops_dispatcher::fwd_uacm(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(src, LOGIT_COUNT_PER_TILE), ops_opt, fwd_uacm_ops_ff_16_16{});
                break;
            default:
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    }

    static void fwd_pacm(void * __restrict__ dst, const void * __restrict__ lhs, const void * __restrict__ rhs, ops_kind_t ops_kind, tile_kind_t tile_kind) noexcept{

        using namespace taxonomy;
        
        switch (tile_kind){
            case uu_8_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uu_8_8{});
                break;
            case uu_8_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uu_8_16{});
                break;
            case uu_16_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uu_16_8{});
                break;
            case uu_16_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uu_16_16{})
                break;
            case uf_8_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uf_8_8{});
                break;
            case uf_8_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uf_8_16{});
                break;
            case uf_16_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uf_16_8{});
                break;
            case uf_16_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_uf_16_16{});
                break;
            case fu_8_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_fu_8_8{});
                break;
            case fu_8_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_fu_8_16{});
                break;
            case fu_16_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_fu_16_8{});
                break;
            case fu_16_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_fu_16_16{});
                break;
            case ff_8_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_ff_8_8{});
                break;
            case ff_8_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_ff_8_16{});
                break;
            case ff_16_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_ff_16_8{});
                break;
            case ff_16_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pacm_ops_ff_16_16{});
                break;
            default:
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    }

    static void fwd_pair(void * __restrict__ dst, const void * __restrict__ lhs, const void * __restrict__ rhs, ops_kind_t ops_kind, tile_kind_t tile_kind) noexcept{

        using namespace taxonomy;
        
        switch (tile_kind){
            case uu_8_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uu_8_8{});
                break;
            case uu_8_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uu_8_16{});
                break;
            case uu_16_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uu_16_8{});
                break;
            case uu_16_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uu_16_16{})
                break;
            case uf_8_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uf_8_8{});
                break;
            case uf_8_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uf_8_16{});
                break;
            case uf_16_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uf_16_8{});
                break;
            case uf_16_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_uf_16_16{});
                break;
            case fu_8_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_fu_8_8{});
                break;
            case fu_8_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_fu_8_16{});
                break;
            case fu_16_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_fu_16_8{});
                break;
            case fu_16_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_fu_16_16{});
                break;
            case ff_8_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_ff_8_8{});
                break;
            case ff_8_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_ff_8_16{});
                break;
            case ff_16_8:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_ff_16_8{});
                break;
            case ff_16_16:
                ops_dispatcher::fwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs, LOGIT_COUNT_PER_TILE), ops_opt, fwd_pair_ops_ff_16_16{});
                break;
            default:
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    }

    static void bwd_mono(void * __restrict__ dst, const void * __restrict__ lhs_logit, const void * __restrict__ rhs_grad, ops_kind_t ops_kind, tile_kind_t tile_kind) noexcept{

        using namespace taxonomy;
        
        switch (tile_kind){
            case uu_8_8:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uu_8_8{});
                break;
            case uu_8_16:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uu_8_16{});
                break;
            case uu_16_8:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uu_16_8{});
                break;
            case uu_16_16:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uu_16_16{})
                break;
            case uf_8_8:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uf_8_8{});
                break;
            case uf_8_16:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uf_8_16{});
                break;
            case uf_16_8:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uf_16_8{});
                break;
            case uf_16_16:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_uf_16_16{});
                break;
            case fu_8_8:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_fu_8_8{});
                break;
            case fu_8_16:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_fu_8_16{});
                break;
            case fu_16_8:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_fu_16_8{});
                break;
            case fu_16_16:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_fu_16_16{});
                break;
            case ff_8_8:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_ff_8_8{});
                break;
            case ff_8_16:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_ff_8_16{});
                break;
            case ff_16_8:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_ff_16_8{});
                break;
            case ff_16_16:
                ops_dispatcher::bwd_mono(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), ops_opt, bwd_mono_ops_ff_16_16{});
                break;
            default:
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    }

    static void bwd_pair_rhs(void * __restrict__ dst, const void * __restrict__ lhs_logit, const void * __restrict__ rhs_grad, const void * __restrict__ rhs_lhs_logit, ops_kind_t ops_kind, tile_kind_t tile_kind) noexcept{

        using namespace taxonomy;
        
        switch (tile_kind){
            case uu_8_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uu_8_8{});
                break;
            case uu_8_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uu_8_16{});
                break;
            case uu_16_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uu_16_8{});
                break;
            case uu_16_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uu_16_16{})
                break;
            case uf_8_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uf_8_8{});
                break;
            case uf_8_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uf_8_16{});
                break;
            case uf_16_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uf_16_8{});
                break;
            case uf_16_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_uf_16_16{});
                break;
            case fu_8_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_fu_8_8{});
                break;
            case fu_8_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_fu_8_16{});
                break;
            case fu_16_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_fu_16_8{});
                break;
            case fu_16_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_fu_16_16{});
                break;
            case ff_8_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_ff_8_8{});
                break;
            case ff_8_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_ff_8_16{});
                break;
            case ff_16_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_ff_16_8{});
                break;
            case ff_16_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_lhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_rhs_ops_ff_16_16{});
                break;
            default:
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    }

    static void bwd_pair_lhs(void * __restrict__ dst, const void * __restrict__ lhs_logit, const void * __restrict__ rhs_grad, const void * __restrict__ rhs_rhs_logit, ops_kind_t ops_kind, tile_kind_t tile_kind) noexcept{

        using namespace taxonomy;
        
        switch (tile_kind){
            case uu_8_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uu_8_8{});
                break;
            case uu_8_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uu_8_16{});
                break;
            case uu_16_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uu_16_8{});
                break;
            case uu_16_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uu_16_16{})
                break;
            case uf_8_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uf_8_8{});
                break;
            case uf_8_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uf_8_16{});
                break;
            case uf_16_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uf_16_8{});
                break;
            case uf_16_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<uint16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_uf_16_16{});
                break;
            case fu_8_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_fu_8_8{});
                break;
            case fu_8_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_fu_8_16{});
                break;
            case fu_16_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_fu_16_8{});
                break;
            case fu_16_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<uint16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_fu_16_16{});
                break;
            case ff_8_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_ff_8_8{});
                break;
            case ff_8_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float8_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_ff_8_16{});
                break;
            case ff_16_8:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float8_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_ff_16_8{});
                break;
            case ff_16_16:
                ops_dispatcher::bwd_pair(dg::memult::start_lifetime_as_array<std::float16_t>(dst, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(lhs_logit, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_grad, LOGIT_COUNT_PER_TILE), dg::memult::start_lifetime_as_array<std::float16_t>(rhs_rhs_logit, LOGIT_COUNT_PER_TILE), ops_opt, bwd_pair_lhs_ops_ff_16_16{});
                break;
            default:
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                    break;
                } else{
                    std::unreachable();
                    break;
                }
        }
    } 
}

namespace dg::network_tileops_host_poly{

    using namespace taxonomy;
    using dispatch_t = uint16_t; 

    constexpr auto dispatch_make(ops_kind_t ops_opt, tile_kind_t type_opt) noexcept -> dispatch_t{

        static_assert(sizeof(ops_kind_t) + sizeof(tile_kind_t) <= sizeof(dispatch_t));
        return (static_cast<dispatch_t>(ops_opt) << (sizeof(tile_kind_t) * CHAR_BIT)) | static_cast<dispatch_t>(type_opt);
    } 

    static constexpr auto dispatch_extract_ops(dispatch_t dispatch_opt) noexcept -> ops_kind_t{

        return dispatch_opt >> (sizeof(tile_kind_t) * CHAR_BIT);
    } 

    static constexpr auto dispatch_extract_type(dispatch_t dispatch_opt) noexcept -> cast_t{

        return dispatch_opt & low<dispatch_t>(std::integral_constant<size_t, sizeof(tile_kind_t) * CHAR_BIT>{});
    }

    extern void fwd_mono(void * __restrict__ dst, const void * __restrict__ src, dispatch_t dispatch_opt) noexcept{

        dispatcher::fwd_mono(dst, src, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    } 

    extern void fwd_uacm(void * __restrict__ dst, const void * __restrict__ src, dispatch_t dispatch_opt) noexcept{
        
        dispatcher::fwd_uacm(dst, src, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }
    
    extern void fwd_pacm(void * __restrict__ dst, const void * __restrict__ lhs, const void * __restrict__ rhs, dispatch_t dispatch_opt) noexcept{

        dispatcher::fwd_pacm(dst, lhs, rhs, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }

    extern void fwd_pair(void * __restrict__ dst, const void * __restrict__ lhs, const void * __restrict__ rhs, dispatch_t dispatch_opt) noexcept{

        dispatcher::fwd_pair(dst, lhs, rhs, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }

    extern void bwd_mono(void * __restrict__ dst, const void * __restrict__ lhs_logit, const void * __restrict__ rhs_grad, dispatch_t dispatch_opt) noexcept{

        dispatcher::bwd_mono(dst, lhs_logit, rhs_grad, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }

    extern void bwd_pair_rhs(void * __restrict__ dst, const void * __restrict__ lhs_logit, const void * __restrict__ rhs_grad, const void * __restrict__ rhs_lhs_logit, dispatch_t dispatch_opt) noexcept{

        dispatcher::bwd_pair_rhs(dst, lhs_logit, rhs_grad, rhs_lhs_logit, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }

    extern void bwd_pair_lhs(void * __restrict__ dst, const void * __restrict__ lhs_logit, const void * __restrict__ rhs_grad, const void * __restrict__ rhs_rhs_logit, dispatch_t dispatch_opt) noexcept{

        dispatcher::bwd_pair_lhs(dst, lhs_logit, rhs_grad, rhs_rhs_logit, dispatch_extract_ops(dispatch_opt), dispatch_extract_type(dispatch_opt));
    }
}

#endif