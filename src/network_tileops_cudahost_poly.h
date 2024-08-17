#ifndef __NETWORK_TILEOPS_MULTI_DEVICE_POLY_H__
#define __NETWORK_TILEOPS_MULTI_DEVICE_POLY_H__

#include "network_tileops_cuda_poly.h"
#include "network_tileops_host_poly.h"

namespace dg::network_tileops_multidevice_poly::taxonomy{

    using device_opt_t  = uint8_t;
    using dispatch_t    = uint_t_of_byte<std::max(sizeof(network_tileops_cuda_poly::dispatch_t), sizeof(network_tileops_host_poly::dispatch_t)) + sizeof(device_opt_t)>;

    enum device_option{
        cpu     = 0,
        cuda    = 1
    };
} 

namespace dg::network_tileops_multidevice_poly{

    using namespace taxonomy;

    constexpr auto dispatch_cuda_make(dg::network_tileops_host_poly::ops_t, dg::network_tileops_host_poly::type_t) noexcept -> dispatch_t{

    }

    constexpr auto dispatch_host_make(dg::network_tileops_cuda_poly::ops_t, dg::network_tileops_cuda_poly::type_t) noexcept -> dispatch_t{

    }

    void fwd_mono(void * dst, const void * src, const dispatch_t dispatch_opt) noexcept{

    } 

    void fwd_uacm(void * dst, const void * src, const dispatch_t dispatch_opt) noexcept{
        
    }
    
    void fwd_pacm(void * dst, const void * lhs, const void * rhs, const dispatch_t dispatch_opt) noexcept{

    }

    void fwd_pair(void * dst, const void * lhs, const void * rhs, const dispatch_t dispatch_opt) noexcept{

    }

    void bwd_mono(void * dst, const void * lhs_logit, const void * rhs_grad, const dispatch_t dispatch_opt) noexcept{

    }

    void bwd_pair_rhs(void * dst, const void * lhs_logit, const void * rhs_grad, const void * rhs_lhs_logit, const dispatch_t dispatch_opt) noexcept{

    }

    void bwd_pair_lhs(void * dst, const void * lhs_logit, const void * rhs_grad, const void * rhs_rhs_logit, const dispatch_t dispatch_opt) noexcept{

    }
}

#endif
