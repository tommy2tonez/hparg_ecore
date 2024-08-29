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
#include "network_tile_member_getsetter.h" 

namespace dg::network_tileops_handler{

    //scenerios:
    //devices:
    //fsys-fsys
    //fsys-host
    //host-host
    //cuda-cuda
    
    //dispatch_t:
    //float8-float8
    //float8-float16
    //float16-float8
    //float16-float16

    //all commitables are dismissibles
    //dismissibles are logged as optional or returned as exception_t - or return exception_t then logged as optional
    //5 hrs 

    using namespace dg::network_tileops_poly::taxonomy;
    using dispatch_t = poly_t;

    void forward_mono(uma_ptr_t dst, uma_ptr_t src){
        
        using namespace dg::network_tile_member_global_getsetter;
        
        uma_ptr_t dst_lck_addr              = get_mono_rcu_addr(dst); //
        uma_ptr_t src_lck_addr              = get_rcu_addr(src); //
        auto lck_grd                        = dg::network_uma::memacquire_guard_many(dst_lck_addr, src_lck_addr); //
        operatable_id_t dst_operatable_id   = get_mono_operatable_id(dst); //
        operatable_id_t src_operatable_id   = get_operatable_id(src);

        if (dst_operatable_id != src_operatable_id){
            dg::network_exception::throw_exception(dg::network_exception::INCOMPATIBLE_OPERATABLE_ID);
        }

        uma_ptr_t dst_logit_umaptr                      = get_mono_logit_addr(dst);
        uma_ptr_t src_logit_umaptr                      = get_logit_addr(src);
        dispatch_control_t dispatch_control             = get_mono_dispatch_control(dst);
        auto [dst_vd_id, src_vd_id, dst_dispatch_id]    = dg::network_dispatch_control::decode_mono(dispatch_control);
        auto dst_logit_vmaptr_map                       = dg::network_uma::map_wait(dst_logit_umaptr, dst_vd_id);
        auto dst_logit_vmaptr_map_relgrd                = dg::network_uma::map_relguard(dst_logit_vmaptr_map); 
        auto src_logit_vmaptr_map                       = dg::network_uma::map_wait(src_logit_umaptr, src_vd_id);
        auto src_logit_vmaptr_map_relgrd                = dg::network_uma::map_relguard(src_logit_vmaptr_map);
        vma_ptr_t dst_logit_vmaptr                      = dg::network_uma::get_vma_ptr(dst_logit_vmaptr_map);
        vma_ptr_t src_logit_vmaptr                      = dg::network_uma::get_vma_const_ptr(src_logit_vmaptr_map);

        if (dg::network_virtual_device::is_fsys_ptr(dst_logit_vmaptr) && dg::network_virtual_device::is_fsys_ptr(src_logit_vmaptr)){
            return;
        }

        if (dg::network_virtual_device::is_fsys_ptr(dst_logit_vmaptr) && dg::network_virtual_device::is_host_ptr(src_logit_vmaptr)){
            return;
        }

        if (dg::network_virtual_device::is_host_ptr(dst_logit_vmaptr) && dg::network_virtual_device::is_fsys_ptr(src_logit_vmaptr)){
            return;
        }

        if (dg::network_virtual_device::is_host_ptr(dst_logit_vmaptr) && dg::network_virtual_device::is_host_ptr(src_logit_vmaptr)){
            return;
        }

        if (dg::network_virtual_device::is_cuda_ptr(dst_logit_vmaptr) && dg::network_virtual_device::is_cuda_ptr(src_logit_vmaptr)){
            return;
        }

        dg::network_exception::throw_exception(dg::network_exception::INVALID_TABLE_DISPATCH_CODE);
    }

    void forward_mono_nothrow(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void forward_pair(uma_ptr_t dst, uma_ptr_t lhs, uma_ptr_t rhs){

        using namespace dg::network_tile_member_global_getsetter;

        uma_ptr_t dst_lck_addr              = get_pair_rcu_addr(dst);
        uma_ptr_t lhs_lck_addr              = get_rcu_addr(lhs);
        uma_ptr_t rhs_lck_addr              = get_rcu_addr(rhs);
        auto lck_grd                        = dg::network_uma::memacquire_guard_many(dst_lck_addr, lhs_lck_addr, rhs_lck_addr); //
        operatable_id_t dst_operatable_id   = get_pair_operatable_id(dst);
        operatable_id_t lhs_operatable_id   = get_operatable_id(lhs);
        operatable_id_t rhs_operatable_id   = get_operatable_id(rhs);

        if (!dg::utility::is_same_value(dst_operatable_id, lhs_operatable_id, rhs_operatable_id)){
            dg::network_exception::throw_exception(dg::network_exception::INCOMPATIBLE_OPERATABLE_ID);
        }

        uma_ptr_t dst_logit_umaptr                                  = get_pair_logit_addr(dst);
        uma_ptr_t lhs_logit_umaptr                                  = get_logit_addr(lhs);
        uma_ptr_t rhs_logit_umaptr                                  = get_logit_addr(rhs);
        dispatch_control_t dispatch_control                         = get_pair_dispatch_control(dst);
        // auto [dst_vd_id, lhs_vd_id, rhs_vd_id, dst_dispatch_id]     = dg::network_dispatch_control::decode_pair(dispatch_control);


    }

    void forward_pair_nothrow(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void forward_uacm(uma_ptr_t dst, std::array<uma_ptr_t, UACM_COUNT> src){

    }

    void forward_uacm_nothrow(uma_ptr_t dst, std::array<uma_ptr_t, UACM_COUNT> src) noexcept{

    }

    void forward_pacm(uma_ptr_t dst, std::array<uma_ptr_t, PACM_COUNT> lhs, std::array<uma_ptr_t, PACM_COUNT> rhs){

    }

    void forward_pacm_nothrow(uma_ptr_t dst, std::array<uma_ptr_t, PACM_COUNT> lhs, std::array<uma_ptr_t, PACM_COUNT> rhs) noexcept{

    }

    void forward_crit(uma_ptr_t dst, uma_ptr_t src){

    }

    void forward_crit_nothrow(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void forward_msgrfwd(una_ptr_t dst, uma_ptr_t src){

    }

    void forward_msgrfwd_nothrow(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void forward_msgrbwd(uma_ptr_t dst, uma_ptr_t src){

    }

    void forward_msgrbwd_nothrow(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void backward_mono(uma_ptr_t dst, uma_ptr_t src){

    }
    
    void backward_mono_nothrow(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void backward_pair_lhs(uma_ptr_t dst, uma_ptr_t src, uma_ptr_t rhs){

    }

    void backward_pair_lhs_nothrow(uma_ptr_t dst, uma_ptr_t src, ){

    }

    void backward_pair_rhs(uma_ptr_t dst, uma_ptr_t src, uma_ptr_t lhs){

    }

    void backward_pair_rhs_nothrow(uma_ptr_t dst, uma_ptr_t src, uma_ptr_t lhs) noexcept{

    }

    void backward_uacm(uma_ptr_t dst, uma_ptr_t src){

    }

    void backward_uacm_nothrow(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void backward_pacm(uma_ptr_t dst, uma_ptr_t src){

    }

    void backward_pacm_nothrow(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }
} 

#endif