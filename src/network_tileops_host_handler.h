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
#include "network_vmamap.h"

namespace dg::network_tileops_handler{

    using namespace dg::network_tileops_poly::taxonomy;
    using dispatch_t = poly_t;
    
    auto forward_mono(uma_ptr_t dst) noexcept -> bool{
        
        using namespace dg::network_tile_member_getsetter;
        
        uma_ptr_t dst_lck_addr  = get_mono_rculock_addr_nothrow(dst);
        uma_ptr_t src           = {}; 

        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(dst_lck_addr); 
            src = get_mono_src_nothrow(src);
        }

        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
        operatable_id_t dst_operatable_id   = get_mono_operatable_id_nothrow(dst);
        operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

        if (dst_operatable_id != src_operatable_id){
            return false;
        }

        uma_ptr_t dst_logit_umaptr                          = get_mono_logit_addr_nothrow(dst);
        uma_ptr_t src_logit_umaptr                          = get_logit_addr_nothrow(src);
        dispatch_control_t dispatch_control                 = get_mono_dispatch_control_nothrow(dst);
        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp]  = dg::network_dispatch_control::decode_mono(dispatch_control);
        auto [dst_map_resource, src_map_resource]           = dg::network_uma::mapsafe_recursivewait_many<2u>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}}); //weird - I mean this could combined with vmamap - yet I have yet wanted to complicate this further
        auto dst_logit_vmaptr                               = dg::network_uma::get_vma_ptr(dst_map_resource);
        auto src_logit_vmaptr                               = dg::network_uma::get_vma_ptr(src_map_resource); 
        auto dst_logit_vmamap                               = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
        auto src_logit_vmamap                               = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            dg::network_tileops_cuda_poly::fwd_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), tileops_dp);
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), tileops_dp);
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    auto forward_pair(uma_ptr_t dst) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter;

        uma_ptr_t dst_lck_addr  = get_pair_rculock_addr_nothrow(dst);
        uma_ptr_t lhs           = {};
        uma_ptr_t rhs           = {};

        //fine - refactor later
        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(dst_lck_addr);
            lhs = get_pair_lhs_nothrow(dst);
            rhs = get_pair_rhs_nothrow(dst);
        }

        uma_ptr_t lhs_lck_addr              = get_rculock_addr_nothrow(lhs);
        uma_ptr_t rhs_lck_addr              = get_rculock_addr_nothrow(rhs);
        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, lhs_lck_addr, rhs_lck_addr);
        operatable_id_t dst_operatable_id   = get_pair_operatable_id_nothrow(dst);
        operatable_id_t lhs_operatable_id   = get_operatable_id_nothrow(lhs);
        operatable_id_t rhs_operatable_id   = get_operatable_id_nothrow(rhs);

        if (!dg::network_genult::is_same_value(dst_operatable_id, lhs_operatable_id, rhs_operatable_id)){
            return false;
        }

        uma_ptr_t dst_logit_umaptr              = get_pair_logit_addr_nothrow(dst);
        uma_ptr_t lhs_logit_umaptr              = get_logit_addr_nothrow(lhs);
        uma_ptr_t rhs_logit_umaptr              = get_logit_addr_nothrow(rhs);
        dispatch_control_t dispatch_control     = get_pair_dispatch_control_nothrow(dst);
        auto [dst_vd_id, lhs_vd_id, rhs_vd_id, dp_device, tileops_dp]   = dg::network_dispatch_control::decode_pair(dispatch_control);
        auto [dst_map_resource, lhs_map_resource, rhs_map_resource]     = dg::network_uma::mapsafe_recursivewait_many<3u>({{dst_logit_umaptr, dst_vd_id}, {lhs_logit_umaptr, lhs_vd_id}, {rhs_logit_umaptr, rhs_vd_id}});
        auto dst_logit_vmaptr                   = dg::network_uma::get_vma_ptr(dst_map_resource);
        auto lhs_logit_vmaptr                   = dg::network_uma::get_vma_ptr(lhs_map_resource);
        auto rhs_logit_vmaptr                   = dg::network_uma::get_vma_ptr(rhs_map_resource); 
        auto dst_logit_vmamap                   = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
        auto lhs_logit_vmamap                   = dg::network_vmamap::mapsafe_nothrow(lhs_logit_vmaptr);
        auto rhs_logit_vmamap                   = dg::network_vmamap::mapsafe_nothrow(rhs_logit_vmaptr); 

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            dg::network_tileops_cuda_poly::fwd_pair(get_cuda_ptr(dst_logit_vmamap), get_cuda_ptr(lhs_logit_vmamap), get_cuda_ptr(rhs_logit_vmamap), tileops_dp_id);
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            dg::network_tileops_host_poly::fwd_pair(get_host_ptr(dst_logit_vmamap), get_host_ptr(lhs_logit_vmamap), get_host_ptr(rhs_logit_vmamap), tileops_dp_id);
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    auto forward_uacm(uma_ptr_t dst) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter; 

        std::array<uma_ptr_t, UACM_COUNT> src = {};

        //fine - refactor later
        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(get_uacm_rcu_add_nothrow(dst));
            src = get_uacm_src(dst);
        }

        auto addr       = dg::genult::tuple_join(std::make_tuple(dst), src);
        auto lck_addr   = dg::genult::tuple_transform(addr, get_rculock_addr_nothrow); 
        auto lck_grd    = dg::genult::tuple_invoke(dg::network_memops_uma::memlock_many_guard, lck_addr);
        auto ops_id     = dg::genult::tuple_transform(addr, get_operatable_id_nothrow);

        if (!dg::network_genult::tuple_invoke(dg::network_genult::is_same_value, ops_id)){
            return false;
        }

        auto logit_umaaddr      = dg::genult::tuple_transform(addr, get_logit_addr_nothrow);
        auto dispatch_control   = get_uacm_dispatch_control_nothrow(dst);
        auto vd_id              = dg::network_dispatch_control::decode_uacm_vd_id(dispatch_control);
        auto tileops_dp_id      = dg::network_dispatch_control::decode_uacm_tileops_dp_id(dispatch_control); //weird - 
        auto dp_device          = dg::network_dispatch_control::decode_uacm_dp_device(dispatch_control); 
        auto map_resource_arg   = dg::network_genult::tuple_zip(logit_umaaddr, vd_id);
        auto map_resource       = dg::network_uma::mapsafe_recursivewait_many_nothrow(map_resource_arg); //
        auto logit_vmaptr       = dg::network_genult::tuple_transform(map_resource, dg::network_uma::get_vma_ptr); 

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            auto cuda_resource  = dg::network_genult::tuple_transform(logit_vmaptr, dg::network_virtual_device::devirtualize_cuda_ptr);
            dg::network_tileops_cuda_poly::fwd_uacm(cuda_resource, tileops_dp_id); //
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto fsyshost_resource  = dg::network_genult::tuple_transform(logit_vmaptr, map_fsyshost);
            auto cptr_array         = dg::network_genult::tuple_transform(fsyshost_resource, get_cptr); 
            dg::network_tileops_host_poly::fwd_uacm(cptr_array, tileops_dp_id); //
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    auto forward_pacm(uma_ptr_t dst) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter; 

        std::array<uma_ptr_t, PACM_COUNT> lhs = {};
        std::array<uma_ptr_t, PACM_COUNT> rhs = {};

        //fine - refactor later
        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(get_pacm_rculock_addr(dst));
            lhs = get_pacm_lhs(dst);
            rhs = get_pacm_rhs(dst);
        }

        auto addr       = dg::network_genult::tuple_join(std::make_tuple(dst), lhs, rhs);
        auto lck_addr   = dg::network_genult::tuple_transform(addr, get_rculock_addr_nothrow);
        auto lck_grd    = dg::network_genult::tuple_invoke(dg::network_memops_uma::memlock_many_guard, lck_addr);
        auto ops_id     = dg::network_genult::tuple_transform(addr, get_operatable_id_nothrow);

        if (!dg::network_genult::tuple_invoke(dg::network_genult::is_same_value, ops_id)){
            return false;
        }

        auto logit_umaaddr      = dg::network_genult::tuple_transform(addr, get_logit_addr_nothrow);
        auto dispatch_control   = get_pacm_dispatch_control_nothrow(dst);
        auto vd_id              = dg::network_dispatch_control::decode_pacm_vd_id(dispatch_control);
        auto tileops_dp_id      = dg::network_dispatch_control::decode_pacm_tileops_dp_id(dispatch_control);
        auto dp_device          = dg::network_dispatch_control::decode_pacm_dp_device(dispatch_control);
        auto map_resource_arg   = dg::network_genult::tuple_zip(logit_umaaddr, vd_id); //
        auto map_resource       = dg::network_uma::mapsafe_recursivewait_many_nothrow(map_resource_arg); //
        auto logit_vmaptr       = dg::network_genult::tuple_transform(map_resource, dg::network_uma::get_vma_ptr);
        
        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            auto cuda_resource          = dg::network_genult::tuple_transform(logit_vmaptr, dg::network_virtual_device::devirtualize_cuda_ptr);
            auto [first, second, third] = dg::network_genult::tuple_peek_many(cuda_resource, std::integral_constant<size_t, 1>{}, std::integral_constant<size_t, PACM_COUNT>{}, std::integral_constant<size_t, PACM_COUNT>{});
            dg::network_tileops_cuda_poly::fwd_pacm(first, second, third, tileops_dp_id); //array flatten
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto fsyshost_resource      = dg::network_genult::tuple_transform(logit_vmaptr, map_fsyshost);
            auto cptr_array             = dg::network_genult::tuple_transform(fsyshost_resource, get_cptr);
            auto [first, second, third] = dg::network_genult::tuple_peek_many(cptr_array, std::integral_constant<size_t, 1>{}, std::integral_constant<size_t, PACM_COUNT>{}, std::integral_constant<size_t, PACM_COUNT>{});
            dg::network_tileops_host_poly::fwd_pacm(first, second, third, tileops_dp_id); //array flatten
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }    

        return false;
    }

    auto forward_crit(uma_ptr_t dst) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter; 
 
        uma_ptr_t dst_lck_addr  = get_crit_rculock_addr_nothrow(dst); 
        uma_ptr_t src           = {};

        //fine - refactor later
        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(dst_lck_addr);
            src = get_crit_src(dst);
        }

        uma_ptr_t src_lck_addr              = get_rculock_addr_nothrow(src);
        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
        operatable_id_t dst_operatable_id   = get_crit_operatable_id_nothrow(dst);
        operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

        if (dst_operatable_id != src_operatable_id){
            return false;
        }

        uma_ptr_t dst_logit_umaptr                  = get_crit_logit_addr_nothrow(dst);
        uma_ptr_t src_logit_umaptr                  = get_logit_addr_nothrow(src);
        dispatch_control_t dispatch_control         = get_crit_dispatch_control_nothrow(dst);
        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_id]  = dg::network_dispatch_control::decode_crit(dispatch_control);
        auto [dst_map_resource, src_map_resource]   = dg::network_uma::mapsafe_recursivewait_many_nothrow<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
        vma_ptr_t dst_logit_vmaptr                  = dg::network_uma::get_vma_ptr(dst_map_resource);
        vma_ptr_t src_logit_vmaptr                  = dg::network_uma::get_vma_ptr(src_map_resource);

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
            auto [src_logit_cudaptr, src_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(src_logit_vmaptr);
            dg::network_tileops_cuda_poly::fwd_clone(dst_logit_cudaptr, dst_logit_cudaid, src_logit_cudaptr, src_logit_cudaid, tileops_dp_id);
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto dst_logit_fsyshost_resource = map_fsyshost(dst_logit_vmaptr);
            auto src_logit_fsyshost_resource = map_fsyshost(src_logit_vmaptr);
            dg::network_tileops_host_poly::fwd_clone(dst_logit_fsyshost_resource, src_logit_fsyshost_resource, tileops_dp_id);
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    template <class T>
    struct TileTranslatorInterface{

        inline auto translate(uma_ptr_t ptr) noexcept -> std::optional<uma_ptr_t>{

            return static_cast<T *>(this)->translate(ptr);
        }
    };

    template <class T>
    auto forward_extn(uma_ptr_t dst, TileTranslatorInterface<T>& translator) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter;

        uam_ptr_t dst_lck_addr  = get_extn_rculock_addr_nothrow(dst);
        uma_ptr_t src           = {};

        //fine - refactor later
        {
            auto lck_grd        = dg::network_memops_uma::memlock_guard(dst_lck_addr);
            uma_ptr_t ext_src   = get_extn_src_nothrow(dst);
            std::optional<uma_ptr_t> int_src = translator.translate(ext_src);

            if (!int_src.has_value()){
                return false;
            }

            src = int_src.value();
        }

        uma_ptr_t src_lck_addr              = get_rculock_addr_nothrow(src);
        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
        operatable_id_t dst_operatable_id   = get_extn_operatable_id_nothrow(dst);
        operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

        if (dst_operatable_id != src_operatable_id){
            return false;
        }

        uma_ptr_t dst_logit_umaptr                  = get_extn_logit_addr_nothrow(dst);
        uma_ptr_t src_logit_umaptr                  = get_logit_addr_nothrow(src);
        dispatch_control_t dispatch_control         = get_extn_dispatch_control_nothrow(dst);
        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_extn(dispatch_control);    
        auto [dst_map_resource, src_map_resource]   = dg::network_uma::mapsafe_recursivewait_many_nothrow<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}}); //should be consistent - memlock_many_guard
        vma_ptr_t dst_logit_vmaptr                  = dg::network_uma::get_vma_ptr(dst_map_resource);
        vma_ptr_t src_logit_vmaptr                  = dg::network_uma::get_vma_ptr(src_map_resource);

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
            auto [src_logit_cudaptr, src_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(src_logit_vmaptr);
            dg::network_tileops_cuda_poly::fwd_clone(dst_logit_cudaptr, dst_logit_cudaid, src_logit_cudaptr, src_logit_cudaid, tileops_dp_id);
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto dst_map_fsyshost_resource = map_fsyshost(dst_logit_vmaptr); 
            auto src_map_fsyshost_resource = map_fsyshost(src_logit_vmaptr); 
            dg::network_tileops_host_poly::fwd_clone(dst_map_fsyshost_resource, src_map_fsyshost_resource, tileops_dp_id);
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    auto forward_msgrfwd(uma_ptr_t dst) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter;

        uma_ptr_t dst_lck_addr  = get_msgrfwd_rculock_addr_nothrow(dst);
        uma_ptr_t src           = {};

        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(dst_lck_addr);
            src = get_msgrfwd_src_nothrow(dst);
        }

        uma_ptr_t src_lck_addr              = get_rculock_addr_nothrow(src);
        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
        operatable_id_t dst_operatable_id   = get_msgrfwd_operatable_id_nothrow(dst);
        operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

        if (dst_operatable_id != src_operatable_id){
            return false;
        }

        uma_ptr_t dst_logit_umaptr                  = get_msgrfwd_logit_addr_nothrow(dst);
        uma_ptr_t src_logit_umaptr                  = get_logit_addr_nothrow(src);
        dispatch_control_t dispatch_control         = get_msgrfwd_dispatch_control_nothrow(dst);
        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_msgrfwd(dispatch_control);
        auto [dst_map_resource, src_map_resource]   = dg::network_uma::mapsafe_recursivewait_many_nothrow<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
        vma_ptr_t dst_logit_vmaptr                  = dg::network_uma::get_vma_ptr(dst_map_resource);
        vma_ptr_t src_logit_vmaptr                  = dg::network_uma::get_vma_ptr(src_map_resource);

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
            auto [src_logit_cudaptr, src_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(src_logit_vmaptr);
            dg::network_tileops_cuda_poly::fwd_clone(dst_logit_cudaptr, dst_logit_cudaid, src_logit_cudaptr, src_logit_cudaid, tileops_dp_id);
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto dst_map_fsyshost_resource  = map_fsyshost(dst_logit_vmaptr);
            auto src_map_fsyshost_resource  = map_fsyshost(src_logit_vmaptr);
            dg::network_tileops_host_poly::fwd_clone(dst_map_fsyshost_resource, src_map_fsyshost_resource, tileops_dp_id);
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        //single responsibility - forward only does tile_forward - the fwd in msgrfwd is another functor responsibility - either change function name semantic to remove ambiguity or keep the function name and document the decision 
        return false;
    }

    auto forward_msgrbwd(uma_ptr_t dst) noexcept -> bool{

        //good to repeat the code here - this is not refactorable - refactor this code increase coupling of component - not a good practice
        using namespace dg::network_tile_member_getsetter;

        uma_ptr_t dst_lck_addr  = get_msgrbwd_rculock_addr_nothrow(dst); 
        uma_ptr_t src           = {};

        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(dst_lck_addr);
            src = get_msgrbwd_src_nothrow(dst);
        }

        uma_ptr_t src_lck_addr              = get_rculock_addr_nothrow(src);
        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
        operatable_id_t dst_operatable_id   = get_msgrbwd_operatable_id_nothrow(dst);
        operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

        if (dst_operatable_id != src_operatable_id){
            return false;
        }

        uma_ptr_t dst_logit_umaptr                  = get_msgrbwd_logit_addr_nothrow(dst);
        uma_ptr_t src_logit_umaptr                  = get_logit_addr_nothrow(src);
        dispatch_control_t dispatch_control         = get_msgrbwd_dispatch_control_nothrow(dst);
        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_msgrbwd(dispatch_control);
        auto [dst_map_resource, src_map_resource]   = dg::network_uma::mapsafe_recursivewait_many_nothrow<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
        vma_ptr_t dst_logit_vmaptr                  = dg::network_uma::get_vma_ptr(dst_map_resource);
        vma_ptr_t src_logit_vmaptr                  = dg::network_uma::get_vma_ptr(src_map_resource);

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
            auto [src_logit_cudaptr, src_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(src_logit_vmaptr);
            dg::network_tileops_cuda_poly::fwd_clone(dst_logit_cudaptr, dst_logit_cudaid, src_logit_cudaptr, src_logit_cudaid, tileops_dp_id);
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto dst_map_fsyshost_resource  = map_fsyshost(dst_logit_vmaptr); //weird name 
            auto src_map_fsyshost_resource  = map_fsyshost(src_logit_vmaptr);
            dg::network_tileops_host_poly::fwd_clone(dst_map_fsyshost_resource, src_map_fsyshost_resource, tileops_dp_id);
            return true;
        }
        
        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    auto backward_mono(uma_ptr_t src) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter; 

        uam_ptr_t src_lck_addr  = get_mono_rculock_addr_nothrow(src);
        uma_ptr_t dst           = {};
        
        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
            dst = get_mono_src_nothrow(src);
        }
        
        auto dst_lck_addr                   = get_rculock_addr_nothrow(dst);
        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
        operatable_id_t dst_operatable_id   = get_operatable_id_nothrow(dst);
        operatable_id_t src_operatable_id   = get_mono_operatable_id_nothrow(src);

        if (dst_operatable_id != src_operatable_id){
            return false;
        }

        uma_ptr_t dst_grad_umaptr                   = get_grad_addr_nothrow(dst);
        uma_ptr_t dst_logit_umaptr                  = get_logit_addr_nothrow(dst);
        uma_ptr_t src_grad_umaptr                   = get_mono_grad_addr_nothrow(src);
        dispatch_control_t dispatch_control         = get_mono_dispatch_control_nothrow(src);
        auto [src_vd_id, dst_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_mono(dispatch_control);
        auto [dst_grad_map_resource, dst_logit_map_resource, src_grad_map_resource] = dg::network_uma::mapsafe_recursivewait_many_nothrow<3>({{dst_grad_umaptr, dst_vd_id}, {dst_logit_umaptr, dst_vd_id}, {src_grad_umaptr, src_vd_id}});
        vma_ptr_t dst_grad_vmaptr                   = dg::network_uma::get_vma_ptr(dst_grad_map_resource);
        vma_ptr_t dst_logit_vmaptr                  = dg::network_uma::get_vma_ptr(dst_logit_map_resource);
        vma_ptr_t src_grad_vmaptr                   = dg::network_uma::get_vma_ptr(src_grad_map_resource);

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            auto [dst_grad_cudaptr, dst_grad_cudaid]    = dg::network_virtual_device::devirtualize_cuda_ptr(dst_grad_vmaptr);
            auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
            auto [src_grad_cudaptr, src_grad_cudaid]    = dg::network_virtual_device::devirtualize_cuda_ptr(src_grad_vmaptr);
            dg::network_tileops_cuda_poly::bwdzr_mono(dst_grad_cudaptr, dst_grad_cudaid, dst_logit_cudaptr, dst_logit_cudaid, src_grad_cudaptr, src_grad_cudaid, tileops_dp_id);
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto dst_grad_fsyshost_resource     = map_fsyshost(dst_grad_vmaptr); //weird
            auto dst_logit_fsyshost_resource    = map_fsyshost(dst_logit_vmaptr); //weird 
            auto src_grad_fsyshost_resource     = map_fsyshost(src_grad_vmaptr); //weird
            dg::network_tileops_host_poly::bwdzr_mono(dst_grad_fsyshost_resource, dst_logit_fsyshost_resource, src_grad_fsyshost_resource, tileops_dp_id);
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            std::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    auto backward_pair(uma_ptr_t src) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter;

        uma_ptr_t src_lck_addr  = get_pair_rculock_addr_nothrow(src);
        uma_ptr_t lhs           = {};
        uma_ptr_t rhs           = {};

        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
            lhs = get_pair_lhs_nothrow(src);
            rhs = get_pair_rhs_nothrow(src);
        }
        
        //I'll fix the tabs later
        uma_ptr_t lhs_lck_addr              = get_rculock_addr_nothrow(lhs);
        uma_ptr_t rhs_lck_addr              = get_rculock_addr_nothrow(rhs);
        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(src_lck_addr, lhs_lck_addr, rhs_lck_addr);
        operatable_id_t src_operatable_id   = get_pair_operatable_id_nothrow(src);
        operatable_id_t lhs_operatable_id   = get_operatable_id_nothrow(lhs);
        operatable_id_t rhs_operatable_id   = get_operatable_id_nothrow(rhs);

        if (!dg::network_genult::is_same_value(src_operatable_id, lhs_operatable_id, rhs_operatable_id)){
            return false;
        }

        uma_ptr_t lhs_grad_umaptr           = get_grad_addr_nothrow(lhs);
        uma_ptr_t lhs_logit_umaptr          = get_logit_addr_nothrow(lhs);
        uma_ptr_t rhs_grad_umaptr           = get_grad_addr_nothrow(rhs);
        uma_ptr_t rhs_logit_umaptr          = get_logit_addr_nothrow(rhs);
        uma_ptr_t src_grad_umaptr           = get_pair_grad_addr_nothrow(src); //
        dispatch_control_t dispatch_control = get_pair_dispatch_control_nothrow(src);
        auto [src_vd_id, lhs_vd_id, rhs_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_pair(dispatch_control);
        auto [lhs_grad_map_resource, lhs_logit_map_resource, rhs_grad_map_resource, rhs_logit_map_resource, src_grad_map_resource] = dg::network_uma::mapsafe_recursivewait_many_nothrow<4>({{lhs_grad_umaptr, lhs_vd_id}, {lhs_logit_umaptr, lhs_vd_id}, {rhs_grad_umaptr, rhs_vd_id}, {rhs_logit_umaptr, rhs_vd_id}, {src_grad_umaptr, src_vd_id}}); //too lengthy - syntax program fix this
        
        vma_ptr_t lhs_grad_vmaptr           = dg::network_uma::get_vma_ptr(lhs_grad_map_resource);
        vma_ptr_t lhs_logit_vmaptr          = dg::network_uma::get_vma_ptr(lhs_logit_map_resource);
        vma_ptr_t rhs_grad_vmaptr           = dg::network_uma::get_vma_ptr(rhs_grad_map_resource);
        vma_ptr_t rhs_logit_vmaptr          = dg::network_uma::get_vma_ptr(rhs_logit_map_resource);
        vma_ptr_t src_grad_vmaptr           = dg::network_uma::get_vma_ptr(src_grad_map_resource);

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            auto [lhs_grad_cudaptr, lhs_grad_cudaid]    = dg::network_virtual_device::devirtualize_cuda_ptr(lhs_grad_vmaptr);
            auto [lhs_logit_cudaptr, lhs_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(lhs_logit_vmaptr);
            auto [rhs_grad_cudaptr, rhs_grad_cudaid]    = dg::network_virtual_device::devirtualize_cuda_ptr(rhs_grad_vmaptr);
            auto [rhs_logit_cudaptr, rhs_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(rhs_logit_vmaptr);
            auto [src_grad_cudaptr, src_grad_cudaid]    = dg::network_virtual_device::devirtualize_cuda_ptr(src_grad_vmaptr);

            dg::network_tileops_cuda_poly::bwdzr_pair(lhs_grad_cudaptr, lhs_grad_cudaid, lhs_logit_cudaptr, lhs_logit_cudaid, 
                                                      rhs_grad_cudaptr, rhs_grad_cudaid, rhs_logit_cudaptr, rhs_logit_cudaid, 
                                                      src_grad_cudaptr, src_grad_cudaid);
            return true;
        } 
        
        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto lhs_grad_fsyshost_resource     = map_fsyshost(lhs_grad_vmaptr);
            auto lhs_logit_fsyshost_resource    = map_fsyshost(lhs_logit_vmaptr);
            auto rhs_grad_fsyshost_resource     = map_fsyshost(rhs_grad_vmaptr);
            auto rhs_logit_fsyshost_resource    = map_fsyshost(rhs_logit_vmaptr);
            auto src_grad_fsyshost_resource     = map_fsyshost(src_grad_vmaptr); 

            dg::network_tileops_host_poly::bwdzr_pair(lhs_grad_fsyshost_resource, lhs_logit_fsyshost_resource,
                                                      rhs_grad_fsyshost_resource, rhs_logit_fsyshost_resource,
                                                      src_grad_fsyshost_resource); //weird
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    auto backward_uacm(uma_ptr_t src) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter;
        uma_ptr_t src_lck_addr                  = get_uacm_rculock_addr_nothrow(src);
        std::array<uma_ptr_t, UACM_COUNT> dst   = {}; 

        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
            dst = get_uacm_src_nothrow(src);
        }

        auto addr       = dg::network_genult::tuple_join(std::tuple(src), dst);
        auto lck_addr   = dg::network_genult::tuple_transform(addr, get_rculock_addr_nothrow);
        auto lck_grd    = dg::network_genult::tuple_invoke(dg::network_memops_uma::memlock_many_guard, lck_addr);
        auto ops_id     = dg::network_genult::tuple_transform(addr, get_operatable_id_nothrow);

        if (!dg::network_genult::tuple_invoke(is_same_value, ops_id)){
            return false;
        }

        auto logit_addr                     = dg::network_genult::tuple_transform(addr, get_logit_addr_nothrow); //
        auto grad_addr                      = dg::network_genult::tuple_transform(addr, get_grad_addr_nothrow); //
        dispatch_control_t dispatch_control = get_uacm_dispatch_control_nothrow(src); 
        auto vd_id                          = dg::network_dispatch_control::decode_uacm_vd_id(dispatch_control); 
        auto tileops_dp_id                  = dg::network_dispatch_control::decode_uacm_tileops_dp_id(dispatch_control); 
        auto zipped_logit_addr              = dg::network_genult::tuple_zip(logit_addr, vd_id);
        auto zipped_grad_addr               = dg::network_genult::tuple_zip(grad_addr, vd_id); 

    }

    auto backward_pacm(uma_ptr_t src) noexcept -> bool{

    }

    auto backward_crit(uma_ptr_t src) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter;

        uma_ptr_t src_lck_addr  = get_crit_rculock_addr_nothrow(src); 
        uma_ptr_t dst           = {};

        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
            dst = get_crit_src_nothrow(src);
        }

        uma_ptr_t dst_lck_addr              = get_rculock_addr_nothrow(dst); 
        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(src_lck_addr, dst_lck_addr);
        operatable_id_t src_operatable_id   = get_crit_operatable_id_nothrow(src);
        operatable_id_t dst_operatable_id   = get_operatable_id_nothrow(dst);

        if (src_operatable_id != dst_operatable_id){
            return false;
        }

        uma_ptr_t dst_logit_umaptr                  = get_logit_addr_nothrow(dst);
        uma_ptr_t dst_grad_umaptr                   = get_grad_addr_nothrow(dst);
        uma_ptr_t src_grad_umaptr                   = get_grad_addr_nothrow(src);
        dispatch_control_t dispatch_control         = get_crit_dispatch_control_nothrow(src);
        auto [src_vd_id, dst_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_crit(dispatch_control);
        auto [dst_logit_map_resource, dst_grad_map_resource, src_grad_map_resource] = dg::network_uma::mapsafe_recursivewait_many_nothrow<3>({{dst_logit_umaptr, dst_vd_id}, {dst_grad_umaptr, dst_vd_id}, {src_grad_umaptr, src_vd_id}});
        vma_ptr_t dst_grad_vmaptr                   = dg::network_uma::get_vma_ptr(dst_grad_map_resource);
        vma_ptr_t src_grad_vmaptr                   = dg::network_uma::get_vma_ptr(src_grad_map_resource);
        

        //cuda_id + cuda_handle_t are singleton managed by network_tileops_cuda_poly 
        //cross device computation or in device computation are runtime-checked by the controller + internal corruption if failed 

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            auto [dst_grad_cudaptr, dst_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(dst_grad_vmaptr);
            auto [src_grad_cudaptr, src_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(src_grad_vmaptr);
            dg::network_tileops_cuda_poly::bwdzr_crit(dst_grad_cudaptr, dst_grad_cudaid, src_grad_cudaptr, src_grad_cudaid, tileops_dp_id);
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto dst_grad_fsyshost_resource = map_fsyshost(dst_grad_vmaptr);
            auto src_grad_fsyshost_resource = get_fsyshost_reosurce(src_grad_vmaptr); 
            dg::network_tileops_host_poly::bwdzr_crit(dst_grad_fsyshost_resource, src_grad_fsyshost_resource, tileops_dp_id); //implicit conversion is not allowed here - fsyshost resource is vma_ptr_t wrapper - should only implicitly convert to vma_ptr_t - placeholder for now
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    auto backward_extn(uma_ptr_t src) noexcept -> bool{

    }

    auto backward_msgrfwd(uma_ptr_t src) noexcept -> bool{

        using namespace dg::network_tile_member_getsetter; 
        
        uma_ptr_t src_lck_addr  = get_msgrfwd_rculock_addr_nothrow(src);
        uma_ptr_t dst           = {};

        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
            dst = get_msgrfwd_src_nothrow(src);
        }

        uma_ptr_t dst_lck_addr              = get_rculock_addr_nothrow(dst);
        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(src_lck_addr, dst_lck_addr);
        operatable_id_t src_operatable_id   = get_msgrfwd_operatable_id_nothrow(src);
        operatable_id_t dst_operatable_id   = get_operatable_id_nothrow(dst);

        if (src_operatable_id != dst_operatable_id){
            return false;
        }

        uma_ptr_t dst_logit_umaptr                  = get_logit_addr_nothrow(dst);
        uma_ptr_t dst_grad_umaptr                   = get_grad_addr_nothrow(dst);
        uam_ptr_t src_grad_umaptr                   = get_msgrfwd_grad_addr_nothrow(src);
        dispatch_control_t dispatch_control         = get_msgrfwd_dispatch_control_nothrow(src);
        auto [src_vd_id, dst_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_msgrfwd(dispatch_control); //weird
        auto [dst_logit_map_resource, dst_grad_map_resource, src_grad_map_resource] = dg::network_uma::mapsafe_recursivewait_many_nothrow<3>({{dst_logit_umaptr, dst_vd_id}, {dst_grad_umaptr, dst_vd_id}, {src_grad_umaptr, src_vd_id}});
        vma_ptr_t dst_grad_vmaptr                   = dg::network_uma::get_vma_ptr(dst_grad_map_resource);
        vma_ptr_t src_grad_vmaptr                   = dg::network_uma::get_vma_ptr(src_grad_map_resource);

        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){ //uma_ptr_t is cuda_dispatchable - not dp_device - change semantic
            auto [dst_grad_cudaptr, dst_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(dst_grad_vmaptr);
            auto [src_grad_cudaptr, src_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(src_grad_vmaptr);
            dg::network_tileops_cuda_poly::bwdzr_clone(dst_grad_cudaptr, dst_grad_cudaid, src_grad_cudaptr, src_grad_cudaid, tileops_dp_id);
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto dst_grad_fsyshost_resource = map_fsyshost(dst_grad_vmaptr);
            auto src_grad_fsyshost_resource = map_fsyshost(src_grad_vmaptr); 
            dg::network_tileops_host_poly::bwdzr_clone(dst_grad_fsyshost_resource, src_grad_fsyshost_resource, tileops_dp_id);
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    auto backward_msgrbwd(uma_ptr_t src) noexcept -> bool{
        
        using namespace dg::network_tile_member_getsetter; 

        uma_ptr_t src_lck_addr  = get_msgrbwd_rculock_addr_nothrow(src);
        uma_ptr_t dst           = {};

        {
            auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
            dst = get_msgrbwd_src_nothrow(src);
        }

        uma_ptr_t dst_lck_addr              = get_rculock_addr_nothrow(dst);
        auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(src_lck_addr, dst_lck_addr);
        operatable_id_t src_operatable_id   = get_msgrbwd_operatable_id_nothrow(src);
        operatable_id_t dst_operatable_id   = get_operatable_id_nothrow(dst);

        if (src_operatable_id != dst_operatable_id){
            return false;
        }

        uma_ptr_t dst_logit_umaptr          = get_logit_addr_nothrow(dst);
        uma_ptr_t dst_grad_umaptr           = get_grad_addr_nothrow(dst);
        uma_ptr_t src_grad_umaptr           = get_msgrbwd_grad_addr_nothrow(src);
        dispatch_control_t dispatch_control = get_msgrbwd_dispatch_control_nothrow(src); 
        auto [src_vd_id, dst_vd_id, dp_device, tileops_dp_id] =  dg::network_dispatch_control::decode_msgrbwd(dispatch_control);
        auto [dst_logit_map_resource, dst_grad_map_resource, src_grad_map_resource] = dg::network_uma::mapsafe_recursivewait_many_nothrow<3>({{dst_logit_umaptr, dst_vd_id}, {dst_grad_umaptr, dst_vd_id}, {src_grad_umaptr, src_vd_id}});
        vma_ptr_t dst_grad_vmaptr           = dg::network_uma::get_vma_ptr(dst_grad_map_resource);
        vma_ptr_t src_grad_vmaptr           = dg::network_uma::get_vma_ptr(src_grad_map_resource);
        
        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
            auto [dst_grad_cudaptr, dst_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(dst_grad_vmaptr);
            auto [src_grad_cudaptr, src_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(src_grad_vmaptr);
            dg::network_tileops_cuda_poly::bwdzr_clone(dst_grad_cudaptr, dst_grad_cudaid, src_grad_cudaptr, src_grad_cudaid, tileops_dp_id);
            return true;
        }

        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
            auto dst_grad_fsyshost_resource = map_fsyshost(dst_grad_vmaptr);
            auto src_grad_fsyshost_resource = map_fsyshost(src_grad_vmaptr); 
            dg::network_tileops_host_poly::bwdzr_clone(dst_grad_fsyshost_resource, src_grad_fsyshost_resource, tileops_dp_id);
            return true;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return false;
    }

    //----
    void dispatch_fwdpong(uma_ptr_t dst) noexcept{

    }

    void dispatch_fwdmsgr(uma_ptr_t dst) noexcept{

    }

    void dispatch_bwdmsgr(uma_ptr_t dst) noexcept{

    }


} 

#endif