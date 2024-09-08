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
#include "network_memops_uma.h"

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
    
    //without loss of generality - I will sleep on this
    using namespace dg::network_tileops_poly::taxonomy;
    using dispatch_t = poly_t;

    auto forward_mono(uma_ptr_t dst, uma_ptr_t src) noexcept -> bool{
        
        //all_lock should be acquired once - for best practice

        using namespace dg::network_tile_member_getsetter; //namespace should only be used once - to avoid namespace collision -
        
        uma_ptr_t dst_lck_addr              = get_mono_rcu_addr_nothrow(dst);
        uma_ptr_t src_lck_addr              = get_rcu_addr_nothrow(src);
        auto dst_lck_grd                    = dg::network_memops_uma::memlock_guard_many(dst_lck_addr, src_lck_addr); //deadlock recipe
        operatable_id_t dst_operatable_id   = get_mono_operatable_id_nothrow(dst); //
        operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

        if (dst_operatable_id != src_operatable_id){
            return false;
        }

        uma_ptr_t dst_logit_umaptr                      = get_mono_logit_addr_nothrow(dst);
        uma_ptr_t src_logit_umaptr                      = get_logit_addr_nothrow(src);
        dispatch_control_t dispatch_control             = get_mono_dispatch_control_nothrow(dst);
        auto [dst_vd_id, src_vd_id, tileops_dp_id]      = dg::network_dispatch_control::decode_mono(dispatch_control);
        auto [dst_map_resource, src_map_resource]       = dg::network_uma::mapsafe_recursivewait_many<2u>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}}); //deadlock recipe 
        auto dst_logit_vmaptr                           = dg::network_uma::get_vma_ptr(dst_map_resource);
        auto src_logit_vmaptr                           = dg::network_uma::get_vma_ptr(src_map_resource);

        if (dg::network_virtual_device::is_cuda_ptr(dst_logit_vmaptr) && dg::network_virtual_device::is_cuda_ptr(src_logit_vmaptr)){
            auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
            auto [src_logit_cudaptr, src_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(src_logit_vmaptr);
            dg::network_tileops_cuda_poly::fwd_mono(dst_logit_cudaptr, dst_logit_cudaid, src_logit_cudaptr, src_logit_cudaid, tileops_dp_id);
            return true;
        }

        if (dg::network_virtual_device::is_ptr(dst_logit_vmaptr, FSYS_PTR_FLAG | HOST_PTR_FLAG) && dg::network_virtual_device::is_ptr(src_logit_vmaptr, FSYS_PTR_FLAG | HOST_PTR_FLAG)){
            auto dst_fsyshost_resource  = get_fsyshost_resource(dst_logit_vmaptr);
            auto src_fsyshost_resource  = get_fsyshost_resource(src_logit_vmaptr); 
            dg::network_tileops_host_poly::fwd_mono(get_cptr(dst_fsyshost_resource), get_cptr(src_fsyshost_resource), tileops_dp_id);
            return true;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
        return false;
    }

    auto forward_pair(uma_ptr_t dst, uma_ptr_t lhs, uma_ptr_t rhs) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter;

        uma_ptr_t dst_lck_addr              = get_pair_rcu_addr_nothrow(dst);
        uma_ptr_t lhs_lck_addr              = get_rcu_addr_nothrow(lhs);
        uma_ptr_t rhs_lck_addr              = get_rcu_addr_nothrow(rhs);
        auto lck_grd                        = dg::network_memops_uma::memlock_guard_many(dst_lck_addr, lhs_lck_addr, rhs_lck_addr); //
        operatable_id_t dst_operatable_id   = get_pair_operatable_id_nothrow(dst);
        operatable_id_t lhs_operatable_id   = get_operatable_id_nothrow(lhs);
        operatable_id_t rhs_operatable_id   = get_operatable_id_nothrow(rhs);

        if (!dg::network_genult::is_same_value(dst_operatable_id, lhs_operatable_id, rhs_operatable_id)){
            return false;
        }

        uma_ptr_t dst_logit_umaptr                                  = get_pair_logit_addr_nothrow(dst); //I'll fix the tabs later by running a syntax program - just to keep my sanity - I need the alignment - 
        uma_ptr_t lhs_logit_umaptr                                  = get_logit_addr_nothrow(lhs);
        uma_ptr_t rhs_logit_umaptr                                  = get_logit_addr_nothrow(rhs);
        dispatch_control_t dispatch_control                         = get_pair_dispatch_control_nothrow(dst);
        auto [dst_vd_id, lhs_vd_id, rhs_vd_id, tileops_dp_id]       = dg::network_dispatch_control::decode_pair(dispatch_control);
        auto [dst_map_resource, lhs_map_resource, rhs_map_resource] = dg::network_uma::mapsafe_recursivewait_many<3u>({{dst_logit_umaptr, dst_vd_id}, {lhs_logit_umaptr, lhs_vd_id}, {rhs_logit_umaptr, rhs_vd_id}});
        auto dst_logit_vmaptr                                       = dg::network_uma::get_vma_ptr(dst_map_resource);
        auto lhs_logit_vmaptr                                       = dg::network_uma::get_vma_ptr(lhs_map_resource);
        auto rhs_logit_vmaptr                                       = dg::network_uma::get_vma_ptr(rhs_map_resource); 

        if (dg::network_virtual_device::is_cuda_ptr(dst_logit_vmaptr) && dg::network_virtual_device::is_cuda_ptr(lhs_logit_vmaptr) && dg::network_virtual_device::is_cuda_ptr(rhs_logit_vmaptr)){
            auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
            auto [lhs_logit_cudaptr, lhs_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(lhs_logit_vmaptr);
            auto [rhs_logit_cudaptr, rhs_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(rhs_logit_vmaptr);
            dg::network_tileops_cuda_poly::fwd_pair(dst_logit_cudaptr, dst_logit_cudaid, lhs_logit_cudaptr, lhs_logit_cudaid, rhs_logit_cudaptr, rhs_logit_cudaid, tileops_dp_id);
            return true;
        }

        if (dg::network_virtual_device::is_ptr(dst_logit_vmaptr, FSYS_PTR_FLAG | HOST_PTR_FLAG) && dg::network_virtual_device::is_ptr(lhs_logit_vmaptr, FSYS_PTR_FLAG | HOST_PTR_FLAG) && dg::network_virtual_device::is_ptr(rhs_logit_vmaptr, FSYS_PTR_FLAG | HOST_PTR_FLAG)){
            auto dst_fsyshost_resource  = get_fsyshost_resource(dst_logit_vmaptr);
            auto lhs_fsyshost_resource  = get_fsyshost_resource(lhs_logit_vmaptr);
            auto rhs_fsyshost_resource  = get_fsyshost_resource(rhs_logit_vmaptr); 
            dg::network_tileops_host_poly::fwd_pair(get_cptr(dst_fsyshost_resource), get_cptr(lhs_fsyshost_resource), get_cptr(rhs_fsyshost_resource), tileops_dp_id);
            return true;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
        return false;
    }

    auto forward_uacm(uma_ptr_t dst, std::array<uma_ptr_t, UACM_COUNT> src) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter; 

        //very pythonic - yet it is impossible to write otherwise 

        auto addr       = dg::genult::tuple_join(std::make_tuple(dst), src); //
        auto lck_addr   = dg::genult::tuple_transform(addr, get_rcu_addr_nothrow); 
        auto lck_grd    = dg::genult::tuple_invoke(dg::network_memops_uma::memlock_guard_many, lck_addr);
        auto ops_id     = dg::genult::tuple_transform(addr, get_operatable_id_nothrow);

        if (!dg::network_genult::tuple_invoke(dg::network_genult::is_same_value, ops_id)){
            return false;
        }

        auto logit_umaaddr      = dg::genult::tuple_transform(addr, get_logit_addr_nothrow);
        auto dispatch_control   = get_uacm_dispatch_control_nothrow(dst);
        auto vd_id              = dg::network_dispatch_control::decode_uacm_vd_id(dispatch_control);
        auto tileops_dp_id      = dg::network_dispatch_control::decode_uacm_tileops_dp_id(dispatch_control); //weird - 
        auto map_resource_arg   = dg::network_genult::zip(logit_umaaddr, vd_id);
        auto map_resource       = dg::network_uma::mapsafe_recursivewait_many_nothrow(map_resource_arg);
        auto logit_vmaptr       = dg::network_genult::tuple_transform(map_resource, dg::network_uma::get_vma_ptr); 
        bool is_cuda_only       = dg::network_genult::tuple_reduce(dg::network_genult::tuple_transform(logit_vmaptr, dg::network_virtual_device::is_cuda_ptr), dg::network_genult::and<bool>{});
        bool is_hostfsys_only   = dg::network_genult::tuple_reduce(dg::network_genult::tuple_transform(logit_vmaptr, dg::network_genult::bind_back(dg::network_virtual_device::is_ptr, FSYS_PTR_FLAG | HOST_PTR_FLAG)), dg::network_genult::and<bool>{}); 

        if (is_cuda_only){
            auto cuda_resource  = dg::network_genult::tuple_transform(logit_vmaptr, dg::network_virtual_device::devirtualize_cuda_ptr);
            dg::network_tileops_cuda_poly::fwd_uacm(cuda_resource, tileops_dp_id); //
            return true;
        }

        if (is_hostfsys_only){
            auto fsyshost_resource  = dg::network_genult::tuple_transform(logit_vmaptr, get_fsyshost_resource);
            auto cptr_array         = dg::network_genult::tuple_transform(fsyshost_resource, get_cptr); 
            dg::network_tileops_host_poly::fwd_uacm(cptr_array, tileops_dp_id); //
            return true;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
        return false;
    }

    auto forward_pacm(uma_ptr_t dst, std::array<uma_ptr_t, PACM_COUNT> lhs, std::array<uma_ptr_t, PACM_COUNT> rhs) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter; 

        auto addr       = dg::network_genult::tuple_join(std::make_tuple(dst), lhs, rhs);
        auto lck_addr   = dg::network_genult::tuple_transform(addr, get_rcu_addr_nothrow);
        auto lck_grd    = dg::network_genult::tuple_invoke(dg::network_memops_uma::memlock_guard_many, lck_addr);
        auto ops_id     = dg::network_genult::tuple_transform(addr, get_operatable_id_nothrow);

        if (!dg::network_genult::tuple_invoke(dg::network_genult::is_same_value, ops_id)){
            return false;
        }

        auto logit_umaaddr      = dg::network_genult::tuple_transform(addr, get_logit_addr_nothrow);
        auto dispatch_control   = get_pacm_dispatch_control_nothrow(dst);
        auto vd_id              = dg::network_dispatch_control::decode_pacm_vd_id(dispatch_control);
        auto tileops_dp_id      = dg::network_dispatch_control::decode_pacm_tileops_dp_id(dispatch_control);
        auto map_resource_arg   = dg::network_genult::tuple_zip(logit_umaaddr, vd_id); //
        auto map_resource       = dg::network_uma::mapsafe_recursivewait_many_nothrow(map_resource_arg); //
        auto logit_vmaptr       = dg::network_genult::tuple_transform(map_resource, dg::network_uma::get_vma_ptr);
        bool is_cuda_only       = dg::network_genult::tuple_reduce(dg::network_genult::tuple_transform(logit_vmaptr, dg::network_virtual_device::is_cuda_ptr), dg::network_genult::and<>{});
        bool is_fsyshost_only   = dg::network_genult::tuple_reduce(dg::network_genult::tuple_transform(logit_vmaptr, dg::network_genult::bind_back(dg::network_virtual_device::is_ptr, FSYS_PTR_FLAG | HOST_PTR_FLAG)), dg::network_genult::and<>{});

        if (is_cuda_only){
            auto cuda_resource          = dg::network_genult::tuple_transform(logit_vmaptr, dg::network_virtual_device::devirtualize_cuda_ptr);
            auto [first, second, third] = dg::network_genult::tuple_peek_many(cuda_resource, std::integral_constant<size_t, 1>{}, std::integral_constant<size_t, PACM_COUNT>{}, std::integral_constant<size_t, PACM_COUNT>{});
            dg::network_tileops_cuda_poly::fwd_pacm(first, second, third, tileops_dp_id); //array flatten
            return true;
        }

        if (is_fsyshost_only){
            auto fsyshost_resource      = dg::network_genult::tuple_transform(logit_vmaptr, get_fsyshost_resource);
            auto cptr_array             = dg::network_genult::tuple_transform(fsyshost_resource, get_cptr);
            auto [first, second, third] = dg::network_genult::tuple_peek_many(cptr_array, std::integral_constant<size_t, 1>{}, std::integral_constant<size_t, PACM_COUNT>{}, std::integral_constant<size_t, PACM_COUNT>{});
            dg::network_tileops_host_poly::fwd_pacm(first, second, third, tileops_dp_id); //array flatten
            return true;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
        return false;
    }

    void forward_crit(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void forward_msgrfwd(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void forward_msgrbwd(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }
    
    void backward_mono(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void backward_pair_lhs(uma_ptr_t dst, uma_ptr_t src, ){

    }

    void backward_pair_rhs(uma_ptr_t dst, uma_ptr_t src, uma_ptr_t lhs) noexcept{

    }

    void backward_uacm(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }

    void backward_pacm(uma_ptr_t dst, uma_ptr_t src) noexcept{

    }
} 

#endif