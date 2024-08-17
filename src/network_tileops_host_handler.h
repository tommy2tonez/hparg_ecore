#ifndef __NETWORK_TILEOPS_H__
#define __NETWORK_TILEOPS_H__

#include "network_tileops.h"
// #include "network_tileops_16_32.h"
// #include "network_tileops_32_16.h"
// #include "network_tileops_32_32.h"
// #include "network_memregion_lock.h"
#include "network_memory_utility.h"
#include "network_function_concurrent_buffer.h"
#include "network_tileops_poly.h"

namespace dg::network_tileops_handler{

    using namespace dg::network_tileops_poly::taxonomy;
    using dispatch_t = poly_t;

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto bit_broadcast(T val) noexcept -> T{

        static_assert(-1 == ~0u);
        return ~(val - 1); //should [[assume val == 0 || val == 1]] then use if else - let compiler decide the lastest instruction
    }
 
    void restrict_forward_mono_transform(void ** dst, void ** dst_ver_ctrl, const void ** dst_rcu_lock, 
                                         const void ** src, void ** src_ver_ctrl, const void **  src_rcu_lock, 
                                         const dispatch_t * dispatch_option) noexcept{
        
    }

    void restrict_forward_pair_transform(void ** dst, void ** dst_ver_ctrl, const void ** dst_rcu_lock, 
                                         const void ** lhs, void ** lhs_ver_ctrl, const void ** lhs_rcu_lock, 
                                         const void ** rhs, void ** rhs_ver_ctrl, const void ** rhs_rcu_lock, 
                                         const dispatch_t * dispatch_option) noexcept{

    }

    void restrict_backward_mono_transform(void ** lhs_grad, const void ** lhs_logit, void ** lhs_ver_ctrl, const void ** lhs_rcu_lock,
                                          const void ** rhs_grad, void ** rhs_ver_ctrl, const void ** rhs_rcu_lock,
                                          const dispatch_t * dispatch_option) noexcept{

    }
    
    void restrict_backward_pair_transform_lhs(void ** lhs_grad, const void ** lhs_logit, void ** lhs_ver_ctrl, const void ** lhs_rcu_lock,
                                              const void ** rhs_grad, void ** rhs_ver_ctrl, const void ** rhs_rcu_lock, 
                                              const void ** rhs_rhs_logit, const void ** rhs_rhs_ver_ctrl, const void ** rhs_rhs_rcu_lock, 
                                              const dispatch_t * dispatch_option) noexcept{

    }

    void restrict_backward_pair_transform_rhs(void ** lhs_grad, const void ** lhs_logit, void ** lhs_ver_ctrl, const void ** lhs_rcu_lock,
                                              const void ** rhs_grad, void ** rhs_ver_ctrl, const void ** rhs_rcu_lock, 
                                              const void ** rhs_lhs_logit, void ** rhs_lhs_ver_ctrl, const void ** rhs_lhs_rcu_lock, 
                                              const dispatch_t * dispatch_option) noexcept{

    }

} 

#endif