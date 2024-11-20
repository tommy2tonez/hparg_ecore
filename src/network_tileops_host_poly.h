#ifndef __NETWORK_TILEOPS_POLY_H__
#define __NETWORK_TILEOPS_POLY_H__

#include <stdint.h>
#include <stddef.h> 
#include <stdlib.h>
#include "network_tileops_host_static.h"
#include "network_memult.h"
#include <limits.h>

namespace dg::network_tileops_host_poly::enumeration_space{

    using ops_kind_t    = uint8_t;
    using tile_kind_t   = uint8_t; 
    using dispatch_t    = uint16_t;

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

namespace dg::network_tileops_host_poly::dispatch{

    using namespace dg::network_tileops_host_poly::enumeration_space;
    using dispatch_t = uint16_t; 

    constexpr auto make_dispatch_code(ops_kind_t ops_kind, tile_kind_t tile_kind) noexcept -> dispatch_t{

        static_assert(sizeof(ops_kind_t) + sizeof(tile_kind_t) <= sizeof(dispatch_t));
        return (static_cast<dispatch_t>(ops_kind) << (sizeof(tile_kind_t) * CHAR_BIT)) | static_cast<dispatch_t>(tile_kind);
    } 

    static constexpr auto extract_ops_kind(dispatch_t dispatch_code) noexcept -> ops_kind_t{

        return dispatch_code >> (sizeof(tile_kind_t) * CHAR_BIT);
    }

    static constexpr auto extract_tile_kind(dispatch_t dispatch_code) noexcept -> tile_kind_t{

        return stdx::low_bit<sizeof(tile_kind_t) * CHAR_BIT>(dispatch_code);
    }
}

namespace dg::network_tileops_host_poly{

    //today let's spend some time talking about theoretical physics - particularly photon
    //just assume 
    //for the same differential dimension (such dimension is where 1s this exactly == 1s that - s is not a second - but a render unit)
    //that our physic system is cyclic - every massful object is moving in circle
    //and every logit has a tight loop of infinite clones - and it is moving forward one clone at a time unit, we call this dstep/dt

    //when photon passes through a very small crack - a crack that splits two worlds - an information problem arises - can the tensor at the crack stores all the information of one world?
    //the problem is no, so compression is used. What compression? particularly loop compression - such that our loop of infinite clones are approaching 0 clone (what?) - and dx/dstep, wrt L'Hospital differential rule - is 1
    //so dstep/dt * dx/dstep = dx/dt = dstep/dt which is cyclic - so this is where the wave property comes from    

    using namespace dg::network_tileops_host_poly::enumeration_space;

    extern auto make_dispatch_code(ops_kind_t ops_kind, tile_kind_t tile_kind) noexcept -> dispatch_t{

        return dg::network_tileops_host_poly::dispatch::make_dispatch_code(ops_kind, tile_kind);
    }

    extern void fwd_mono(void * __restrict__ dst, const void * __restrict__ src, dispatch_t dispatch_code) noexcept{

        dg::network_tileops_host_poly::dispatch::fwd_mono_dispatch_table[dispatch_code](dst, src);
    } 

    extern void fwd_uacm(void * __restrict__ dst, const void * __restrict__ src, dispatch_t dispatch_code) noexcept{

        dg::network_tileops_host_poly::dispatch::fwd_uacm_dispatch_table[dispatch_code](dst, src);
    }
    
    extern void fwd_pacm(void * __restrict__ dst, const void * __restrict__ lhs, const void * __restrict__ rhs, dispatch_t dispatch_code) noexcept{

        dg::network_tileops_host_poly::dispatch::fwd_pacm_dispatch_table[dispatch_code](dst, lhs, rhs);
    }

    extern void fwd_pair(void * __restrict__ dst, const void * __restrict__ lhs, const void * __restrict__ rhs, dispatch_t dispatch_code) noexcept{

        dg::network_tileops_host_poly::dispatch::fwd_pair_dispatch_table[dispatch_code](dst, lhs, rhs);
    }

    extern void bwd_mono(void * __restrict__ dst, const void * __restrict__ dst_logit, const void * __restrict__ src_grad, dispatch_t dispatch_code) noexcept{

        dg::network_tileops_host_poly::dispatch::bwd_mono_dispatch_table[dispatch_code](dst, dst_logit, src_grad);
    }

    extern void bwd_pair_rhs(void * __restrict__ dst, const void * __restrict__ dst_logit, const void * __restrict__ src_grad, const void * __restrict__ other_logit, dispatch_t dispatch_code) noexcept{

        dg::network_tileops_host_poly::dispatch::bwd_pair_rhs_dispatch_table[dispatch_code](dst, dst_logit, src_grad, other_logit);
    }

    extern void bwd_pair_lhs(void * __restrict__ dst, const void * __restrict__ dst_logit, const void * __restrict__ src_grad, const void * __restrict__ other_logit, dispatch_t dispatch_code) noexcept{

        dg::network_tileops_host_poly::dispatch::bwd_pair_lhs_dispatch_table[dispatch_code](dst, dst_logit, src_grad, other_logit);
    }
}

#endif