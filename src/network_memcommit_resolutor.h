#ifndef __EVENT_DISPATCHER_H__
#define __EVENT_DISPATCHER_H__

#include <stdint.h>
#include <stddef.h>
#include <network_addr_lookup.h>
#include "network_tile_member_getsetter.h"
#include "network_memcommit_factory.h"
#include "network_producer_consumer.h"

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

// namespace dg::network_tileops_handler{

//     using namespace dg::network_tileops_poly::taxonomy;
//     using dispatch_t = poly_t;
    
//     auto forward_mono(uma_ptr_t dst) noexcept -> bool{
        
//         using namespace dg::network_tile_member_getsetter;
        
//         uma_ptr_t dst_lck_addr  = get_mono_rculock_addr_nothrow(dst);
//         uma_ptr_t src           = {}; 

//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(dst_lck_addr); 
//             src = get_mono_src_nothrow(src);
//         }

//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
//         operatable_id_t dst_operatable_id   = get_mono_operatable_id_nothrow(dst);
//         operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

//         if (dst_operatable_id != src_operatable_id){
//             return false;
//         }

//         uma_ptr_t dst_logit_umaptr                          = get_mono_logit_addr_nothrow(dst);
//         uma_ptr_t src_logit_umaptr                          = get_logit_addr_nothrow(src);
//         dispatch_control_t dispatch_control                 = get_mono_dispatch_control_nothrow(dst);
//         auto [dst_vd_id, src_vd_id, dp_device, tileops_dp]  = dg::network_dispatch_control::decode_mono(dispatch_control);
//         auto [dst_map_resource, src_map_resource]           = dg::network_uma::mapsafe_recursivewait_many<2u>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}}); //weird - I mean this could combined with vmamap - yet I have yet wanted to complicate this further
//         auto dst_logit_vmaptr                               = dg::network_uma::get_vma_ptr(dst_map_resource);
//         auto src_logit_vmaptr                               = dg::network_uma::get_vma_ptr(src_map_resource); 
//         auto dst_logit_vmamap                               = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
//         auto src_logit_vmamap                               = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             dg::network_tileops_cuda_poly::fwd_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), tileops_dp);
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), tileops_dp);
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     auto forward_pair(uma_ptr_t dst) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter;

//         uma_ptr_t dst_lck_addr  = get_pair_rculock_addr_nothrow(dst);
//         uma_ptr_t lhs           = {};
//         uma_ptr_t rhs           = {};

//         //fine - refactor later
//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(dst_lck_addr);
//             lhs = get_pair_lhs_nothrow(dst);
//             rhs = get_pair_rhs_nothrow(dst);
//         }

//         uma_ptr_t lhs_lck_addr              = get_rculock_addr_nothrow(lhs);
//         uma_ptr_t rhs_lck_addr              = get_rculock_addr_nothrow(rhs);
//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, lhs_lck_addr, rhs_lck_addr);
//         operatable_id_t dst_operatable_id   = get_pair_operatable_id_nothrow(dst);
//         operatable_id_t lhs_operatable_id   = get_operatable_id_nothrow(lhs);
//         operatable_id_t rhs_operatable_id   = get_operatable_id_nothrow(rhs);

//         if (!dg::network_genult::is_same_value(dst_operatable_id, lhs_operatable_id, rhs_operatable_id)){
//             return false;
//         }

//         uma_ptr_t dst_logit_umaptr              = get_pair_logit_addr_nothrow(dst);
//         uma_ptr_t lhs_logit_umaptr              = get_logit_addr_nothrow(lhs);
//         uma_ptr_t rhs_logit_umaptr              = get_logit_addr_nothrow(rhs);
//         dispatch_control_t dispatch_control     = get_pair_dispatch_control_nothrow(dst);
//         auto [dst_vd_id, lhs_vd_id, rhs_vd_id, dp_device, tileops_dp]   = dg::network_dispatch_control::decode_pair(dispatch_control);
//         auto [dst_map_resource, lhs_map_resource, rhs_map_resource]     = dg::network_uma::mapsafe_recursivewait_many<3u>({{dst_logit_umaptr, dst_vd_id}, {lhs_logit_umaptr, lhs_vd_id}, {rhs_logit_umaptr, rhs_vd_id}});
//         auto dst_logit_vmaptr                   = dg::network_uma::get_vma_ptr(dst_map_resource);
//         auto lhs_logit_vmaptr                   = dg::network_uma::get_vma_ptr(lhs_map_resource);
//         auto rhs_logit_vmaptr                   = dg::network_uma::get_vma_ptr(rhs_map_resource); 
//         auto dst_logit_vmamap                   = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
//         auto lhs_logit_vmamap                   = dg::network_vmamap::mapsafe_nothrow(lhs_logit_vmaptr);
//         auto rhs_logit_vmamap                   = dg::network_vmamap::mapsafe_nothrow(rhs_logit_vmaptr); 

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             dg::network_tileops_cuda_poly::fwd_pair(get_cuda_ptr(dst_logit_vmamap), get_cuda_ptr(lhs_logit_vmamap), get_cuda_ptr(rhs_logit_vmamap), tileops_dp_id);
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             dg::network_tileops_host_poly::fwd_pair(get_host_ptr(dst_logit_vmamap), get_host_ptr(lhs_logit_vmamap), get_host_ptr(rhs_logit_vmamap), tileops_dp_id);
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     auto forward_uacm(uma_ptr_t dst) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter; 

//         std::array<uma_ptr_t, UACM_COUNT> src = {};

//         //fine - refactor later
//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(get_uacm_rcu_add_nothrow(dst));
//             src = get_uacm_src(dst);
//         }

//         auto addr       = dg::genult::tuple_join(std::make_tuple(dst), src);
//         auto lck_addr   = dg::genult::tuple_transform(addr, get_rculock_addr_nothrow); 
//         auto lck_grd    = dg::genult::tuple_invoke(dg::network_memops_uma::memlock_many_guard, lck_addr);
//         auto ops_id     = dg::genult::tuple_transform(addr, get_operatable_id_nothrow);

//         if (!dg::network_genult::tuple_invoke(dg::network_genult::is_same_value, ops_id)){
//             return false;
//         }

//         auto logit_umaaddr      = dg::genult::tuple_transform(addr, get_logit_addr_nothrow);
//         auto dispatch_control   = get_uacm_dispatch_control_nothrow(dst);
//         auto vd_id              = dg::network_dispatch_control::decode_uacm_vd_id(dispatch_control);
//         auto tileops_dp_id      = dg::network_dispatch_control::decode_uacm_tileops_dp_id(dispatch_control); //weird - 
//         auto dp_device          = dg::network_dispatch_control::decode_uacm_dp_device(dispatch_control); 
//         auto map_resource_arg   = dg::network_genult::tuple_zip(logit_umaaddr, vd_id);
//         auto map_resource       = dg::network_uma::mapsafe_recursivewait_many_nothrow(map_resource_arg); //
//         auto logit_vmaptr       = dg::network_genult::tuple_transform(map_resource, dg::network_uma::get_vma_ptr); 

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             auto cuda_resource  = dg::network_genult::tuple_transform(logit_vmaptr, dg::network_virtual_device::devirtualize_cuda_ptr);
//             dg::network_tileops_cuda_poly::fwd_uacm(cuda_resource, tileops_dp_id); //
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto fsyshost_resource  = dg::network_genult::tuple_transform(logit_vmaptr, map_fsyshost);
//             auto cptr_array         = dg::network_genult::tuple_transform(fsyshost_resource, get_cptr); 
//             dg::network_tileops_host_poly::fwd_uacm(cptr_array, tileops_dp_id); //
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     auto forward_pacm(uma_ptr_t dst) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter; 

//         std::array<uma_ptr_t, PACM_COUNT> lhs = {};
//         std::array<uma_ptr_t, PACM_COUNT> rhs = {};

//         //fine - refactor later
//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(get_pacm_rculock_addr(dst));
//             lhs = get_pacm_lhs(dst);
//             rhs = get_pacm_rhs(dst);
//         }

//         auto addr       = dg::network_genult::tuple_join(std::make_tuple(dst), lhs, rhs);
//         auto lck_addr   = dg::network_genult::tuple_transform(addr, get_rculock_addr_nothrow);
//         auto lck_grd    = dg::network_genult::tuple_invoke(dg::network_memops_uma::memlock_many_guard, lck_addr);
//         auto ops_id     = dg::network_genult::tuple_transform(addr, get_operatable_id_nothrow);

//         if (!dg::network_genult::tuple_invoke(dg::network_genult::is_same_value, ops_id)){
//             return false;
//         }

//         auto logit_umaaddr      = dg::network_genult::tuple_transform(addr, get_logit_addr_nothrow);
//         auto dispatch_control   = get_pacm_dispatch_control_nothrow(dst);
//         auto vd_id              = dg::network_dispatch_control::decode_pacm_vd_id(dispatch_control);
//         auto tileops_dp_id      = dg::network_dispatch_control::decode_pacm_tileops_dp_id(dispatch_control);
//         auto dp_device          = dg::network_dispatch_control::decode_pacm_dp_device(dispatch_control);
//         auto map_resource_arg   = dg::network_genult::tuple_zip(logit_umaaddr, vd_id); //
//         auto map_resource       = dg::network_uma::mapsafe_recursivewait_many_nothrow(map_resource_arg); //
//         auto logit_vmaptr       = dg::network_genult::tuple_transform(map_resource, dg::network_uma::get_vma_ptr);
        
//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             auto cuda_resource          = dg::network_genult::tuple_transform(logit_vmaptr, dg::network_virtual_device::devirtualize_cuda_ptr);
//             auto [first, second, third] = dg::network_genult::tuple_peek_many(cuda_resource, std::integral_constant<size_t, 1>{}, std::integral_constant<size_t, PACM_COUNT>{}, std::integral_constant<size_t, PACM_COUNT>{});
//             dg::network_tileops_cuda_poly::fwd_pacm(first, second, third, tileops_dp_id); //array flatten
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto fsyshost_resource      = dg::network_genult::tuple_transform(logit_vmaptr, map_fsyshost);
//             auto cptr_array             = dg::network_genult::tuple_transform(fsyshost_resource, get_cptr);
//             auto [first, second, third] = dg::network_genult::tuple_peek_many(cptr_array, std::integral_constant<size_t, 1>{}, std::integral_constant<size_t, PACM_COUNT>{}, std::integral_constant<size_t, PACM_COUNT>{});
//             dg::network_tileops_host_poly::fwd_pacm(first, second, third, tileops_dp_id); //array flatten
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }    

//         return false;
//     }

//     auto forward_crit(uma_ptr_t dst) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter; 
 
//         uma_ptr_t dst_lck_addr  = get_crit_rculock_addr_nothrow(dst); 
//         uma_ptr_t src           = {};

//         //fine - refactor later
//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(dst_lck_addr);
//             src = get_crit_src(dst);
//         }

//         uma_ptr_t src_lck_addr              = get_rculock_addr_nothrow(src);
//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
//         operatable_id_t dst_operatable_id   = get_crit_operatable_id_nothrow(dst);
//         operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

//         if (dst_operatable_id != src_operatable_id){
//             return false;
//         }

//         uma_ptr_t dst_logit_umaptr                  = get_crit_logit_addr_nothrow(dst);
//         uma_ptr_t src_logit_umaptr                  = get_logit_addr_nothrow(src);
//         dispatch_control_t dispatch_control         = get_crit_dispatch_control_nothrow(dst);
//         auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_id]  = dg::network_dispatch_control::decode_crit(dispatch_control);
//         auto [dst_map_resource, src_map_resource]   = dg::network_uma::mapsafe_recursivewait_many_nothrow<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
//         vma_ptr_t dst_logit_vmaptr                  = dg::network_uma::get_vma_ptr(dst_map_resource);
//         vma_ptr_t src_logit_vmaptr                  = dg::network_uma::get_vma_ptr(src_map_resource);

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
//             auto [src_logit_cudaptr, src_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(src_logit_vmaptr);
//             dg::network_tileops_cuda_poly::fwd_clone(dst_logit_cudaptr, dst_logit_cudaid, src_logit_cudaptr, src_logit_cudaid, tileops_dp_id);
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto dst_logit_fsyshost_resource = map_fsyshost(dst_logit_vmaptr);
//             auto src_logit_fsyshost_resource = map_fsyshost(src_logit_vmaptr);
//             dg::network_tileops_host_poly::fwd_clone(dst_logit_fsyshost_resource, src_logit_fsyshost_resource, tileops_dp_id);
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     template <class T>
//     struct TileTranslatorInterface{

//         inline auto translate(uma_ptr_t ptr) noexcept -> std::optional<uma_ptr_t>{

//             return static_cast<T *>(this)->translate(ptr);
//         }
//     };

//     template <class T>
//     auto forward_extn(uma_ptr_t dst, TileTranslatorInterface<T>& translator) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter;

//         uam_ptr_t dst_lck_addr  = get_extn_rculock_addr_nothrow(dst);
//         uma_ptr_t src           = {};

//         //fine - refactor later
//         {
//             auto lck_grd        = dg::network_memops_uma::memlock_guard(dst_lck_addr);
//             uma_ptr_t ext_src   = get_extn_src_nothrow(dst);
//             std::optional<uma_ptr_t> int_src = translator.translate(ext_src);

//             if (!int_src.has_value()){
//                 return false;
//             }

//             src = int_src.value();
//         }

//         uma_ptr_t src_lck_addr              = get_rculock_addr_nothrow(src);
//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
//         operatable_id_t dst_operatable_id   = get_extn_operatable_id_nothrow(dst);
//         operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

//         if (dst_operatable_id != src_operatable_id){
//             return false;
//         }

//         uma_ptr_t dst_logit_umaptr                  = get_extn_logit_addr_nothrow(dst);
//         uma_ptr_t src_logit_umaptr                  = get_logit_addr_nothrow(src);
//         dispatch_control_t dispatch_control         = get_extn_dispatch_control_nothrow(dst);
//         auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_extn(dispatch_control);    
//         auto [dst_map_resource, src_map_resource]   = dg::network_uma::mapsafe_recursivewait_many_nothrow<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}}); //should be consistent - memlock_many_guard
//         vma_ptr_t dst_logit_vmaptr                  = dg::network_uma::get_vma_ptr(dst_map_resource);
//         vma_ptr_t src_logit_vmaptr                  = dg::network_uma::get_vma_ptr(src_map_resource);

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
//             auto [src_logit_cudaptr, src_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(src_logit_vmaptr);
//             dg::network_tileops_cuda_poly::fwd_clone(dst_logit_cudaptr, dst_logit_cudaid, src_logit_cudaptr, src_logit_cudaid, tileops_dp_id);
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto dst_map_fsyshost_resource = map_fsyshost(dst_logit_vmaptr); 
//             auto src_map_fsyshost_resource = map_fsyshost(src_logit_vmaptr); 
//             dg::network_tileops_host_poly::fwd_clone(dst_map_fsyshost_resource, src_map_fsyshost_resource, tileops_dp_id);
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     auto forward_msgrfwd(uma_ptr_t dst) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter;

//         uma_ptr_t dst_lck_addr  = get_msgrfwd_rculock_addr_nothrow(dst);
//         uma_ptr_t src           = {};

//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(dst_lck_addr);
//             src = get_msgrfwd_src_nothrow(dst);
//         }

//         uma_ptr_t src_lck_addr              = get_rculock_addr_nothrow(src);
//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
//         operatable_id_t dst_operatable_id   = get_msgrfwd_operatable_id_nothrow(dst);
//         operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

//         if (dst_operatable_id != src_operatable_id){
//             return false;
//         }

//         uma_ptr_t dst_logit_umaptr                  = get_msgrfwd_logit_addr_nothrow(dst);
//         uma_ptr_t src_logit_umaptr                  = get_logit_addr_nothrow(src);
//         dispatch_control_t dispatch_control         = get_msgrfwd_dispatch_control_nothrow(dst);
//         auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_msgrfwd(dispatch_control);
//         auto [dst_map_resource, src_map_resource]   = dg::network_uma::mapsafe_recursivewait_many_nothrow<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
//         vma_ptr_t dst_logit_vmaptr                  = dg::network_uma::get_vma_ptr(dst_map_resource);
//         vma_ptr_t src_logit_vmaptr                  = dg::network_uma::get_vma_ptr(src_map_resource);

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
//             auto [src_logit_cudaptr, src_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(src_logit_vmaptr);
//             dg::network_tileops_cuda_poly::fwd_clone(dst_logit_cudaptr, dst_logit_cudaid, src_logit_cudaptr, src_logit_cudaid, tileops_dp_id);
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto dst_map_fsyshost_resource  = map_fsyshost(dst_logit_vmaptr);
//             auto src_map_fsyshost_resource  = map_fsyshost(src_logit_vmaptr);
//             dg::network_tileops_host_poly::fwd_clone(dst_map_fsyshost_resource, src_map_fsyshost_resource, tileops_dp_id);
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         //single responsibility - forward only does tile_forward - the fwd in msgrfwd is another functor responsibility - either change function name semantic to remove ambiguity or keep the function name and document the decision 
//         return false;
//     }

//     auto forward_msgrbwd(uma_ptr_t dst) noexcept -> bool{

//         //good to repeat the code here - this is not refactorable - refactor this code increase coupling of component - not a good practice
//         using namespace dg::network_tile_member_getsetter;

//         uma_ptr_t dst_lck_addr  = get_msgrbwd_rculock_addr_nothrow(dst); 
//         uma_ptr_t src           = {};

//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(dst_lck_addr);
//             src = get_msgrbwd_src_nothrow(dst);
//         }

//         uma_ptr_t src_lck_addr              = get_rculock_addr_nothrow(src);
//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
//         operatable_id_t dst_operatable_id   = get_msgrbwd_operatable_id_nothrow(dst);
//         operatable_id_t src_operatable_id   = get_operatable_id_nothrow(src);

//         if (dst_operatable_id != src_operatable_id){
//             return false;
//         }

//         uma_ptr_t dst_logit_umaptr                  = get_msgrbwd_logit_addr_nothrow(dst);
//         uma_ptr_t src_logit_umaptr                  = get_logit_addr_nothrow(src);
//         dispatch_control_t dispatch_control         = get_msgrbwd_dispatch_control_nothrow(dst);
//         auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_msgrbwd(dispatch_control);
//         auto [dst_map_resource, src_map_resource]   = dg::network_uma::mapsafe_recursivewait_many_nothrow<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
//         vma_ptr_t dst_logit_vmaptr                  = dg::network_uma::get_vma_ptr(dst_map_resource);
//         vma_ptr_t src_logit_vmaptr                  = dg::network_uma::get_vma_ptr(src_map_resource);

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
//             auto [src_logit_cudaptr, src_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(src_logit_vmaptr);
//             dg::network_tileops_cuda_poly::fwd_clone(dst_logit_cudaptr, dst_logit_cudaid, src_logit_cudaptr, src_logit_cudaid, tileops_dp_id);
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto dst_map_fsyshost_resource  = map_fsyshost(dst_logit_vmaptr); //weird name 
//             auto src_map_fsyshost_resource  = map_fsyshost(src_logit_vmaptr);
//             dg::network_tileops_host_poly::fwd_clone(dst_map_fsyshost_resource, src_map_fsyshost_resource, tileops_dp_id);
//             return true;
//         }
        
//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     auto backward_mono(uma_ptr_t src) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter; 

//         uam_ptr_t src_lck_addr  = get_mono_rculock_addr_nothrow(src);
//         uma_ptr_t dst           = {};
        
//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
//             dst = get_mono_src_nothrow(src);
//         }
        
//         auto dst_lck_addr                   = get_rculock_addr_nothrow(dst);
//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(dst_lck_addr, src_lck_addr);
//         operatable_id_t dst_operatable_id   = get_operatable_id_nothrow(dst);
//         operatable_id_t src_operatable_id   = get_mono_operatable_id_nothrow(src);

//         if (dst_operatable_id != src_operatable_id){
//             return false;
//         }

//         uma_ptr_t dst_grad_umaptr                   = get_grad_addr_nothrow(dst);
//         uma_ptr_t dst_logit_umaptr                  = get_logit_addr_nothrow(dst);
//         uma_ptr_t src_grad_umaptr                   = get_mono_grad_addr_nothrow(src);
//         dispatch_control_t dispatch_control         = get_mono_dispatch_control_nothrow(src);
//         auto [src_vd_id, dst_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_mono(dispatch_control);
//         auto [dst_grad_map_resource, dst_logit_map_resource, src_grad_map_resource] = dg::network_uma::mapsafe_recursivewait_many_nothrow<3>({{dst_grad_umaptr, dst_vd_id}, {dst_logit_umaptr, dst_vd_id}, {src_grad_umaptr, src_vd_id}});
//         vma_ptr_t dst_grad_vmaptr                   = dg::network_uma::get_vma_ptr(dst_grad_map_resource);
//         vma_ptr_t dst_logit_vmaptr                  = dg::network_uma::get_vma_ptr(dst_logit_map_resource);
//         vma_ptr_t src_grad_vmaptr                   = dg::network_uma::get_vma_ptr(src_grad_map_resource);

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             auto [dst_grad_cudaptr, dst_grad_cudaid]    = dg::network_virtual_device::devirtualize_cuda_ptr(dst_grad_vmaptr);
//             auto [dst_logit_cudaptr, dst_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(dst_logit_vmaptr);
//             auto [src_grad_cudaptr, src_grad_cudaid]    = dg::network_virtual_device::devirtualize_cuda_ptr(src_grad_vmaptr);
//             dg::network_tileops_cuda_poly::bwdzr_mono(dst_grad_cudaptr, dst_grad_cudaid, dst_logit_cudaptr, dst_logit_cudaid, src_grad_cudaptr, src_grad_cudaid, tileops_dp_id);
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto dst_grad_fsyshost_resource     = map_fsyshost(dst_grad_vmaptr); //weird
//             auto dst_logit_fsyshost_resource    = map_fsyshost(dst_logit_vmaptr); //weird 
//             auto src_grad_fsyshost_resource     = map_fsyshost(src_grad_vmaptr); //weird
//             dg::network_tileops_host_poly::bwdzr_mono(dst_grad_fsyshost_resource, dst_logit_fsyshost_resource, src_grad_fsyshost_resource, tileops_dp_id);
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             std::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     auto backward_pair(uma_ptr_t src) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter;

//         uma_ptr_t src_lck_addr  = get_pair_rculock_addr_nothrow(src);
//         uma_ptr_t lhs           = {};
//         uma_ptr_t rhs           = {};

//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
//             lhs = get_pair_lhs_nothrow(src);
//             rhs = get_pair_rhs_nothrow(src);
//         }
        
//         //I'll fix the tabs later
//         uma_ptr_t lhs_lck_addr              = get_rculock_addr_nothrow(lhs);
//         uma_ptr_t rhs_lck_addr              = get_rculock_addr_nothrow(rhs);
//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(src_lck_addr, lhs_lck_addr, rhs_lck_addr);
//         operatable_id_t src_operatable_id   = get_pair_operatable_id_nothrow(src);
//         operatable_id_t lhs_operatable_id   = get_operatable_id_nothrow(lhs);
//         operatable_id_t rhs_operatable_id   = get_operatable_id_nothrow(rhs);

//         if (!dg::network_genult::is_same_value(src_operatable_id, lhs_operatable_id, rhs_operatable_id)){
//             return false;
//         }

//         uma_ptr_t lhs_grad_umaptr           = get_grad_addr_nothrow(lhs);
//         uma_ptr_t lhs_logit_umaptr          = get_logit_addr_nothrow(lhs);
//         uma_ptr_t rhs_grad_umaptr           = get_grad_addr_nothrow(rhs);
//         uma_ptr_t rhs_logit_umaptr          = get_logit_addr_nothrow(rhs);
//         uma_ptr_t src_grad_umaptr           = get_pair_grad_addr_nothrow(src); //
//         dispatch_control_t dispatch_control = get_pair_dispatch_control_nothrow(src);
//         auto [src_vd_id, lhs_vd_id, rhs_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_pair(dispatch_control);
//         auto [lhs_grad_map_resource, lhs_logit_map_resource, rhs_grad_map_resource, rhs_logit_map_resource, src_grad_map_resource] = dg::network_uma::mapsafe_recursivewait_many_nothrow<4>({{lhs_grad_umaptr, lhs_vd_id}, {lhs_logit_umaptr, lhs_vd_id}, {rhs_grad_umaptr, rhs_vd_id}, {rhs_logit_umaptr, rhs_vd_id}, {src_grad_umaptr, src_vd_id}}); //too lengthy - syntax program fix this
        
//         vma_ptr_t lhs_grad_vmaptr           = dg::network_uma::get_vma_ptr(lhs_grad_map_resource);
//         vma_ptr_t lhs_logit_vmaptr          = dg::network_uma::get_vma_ptr(lhs_logit_map_resource);
//         vma_ptr_t rhs_grad_vmaptr           = dg::network_uma::get_vma_ptr(rhs_grad_map_resource);
//         vma_ptr_t rhs_logit_vmaptr          = dg::network_uma::get_vma_ptr(rhs_logit_map_resource);
//         vma_ptr_t src_grad_vmaptr           = dg::network_uma::get_vma_ptr(src_grad_map_resource);

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             auto [lhs_grad_cudaptr, lhs_grad_cudaid]    = dg::network_virtual_device::devirtualize_cuda_ptr(lhs_grad_vmaptr);
//             auto [lhs_logit_cudaptr, lhs_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(lhs_logit_vmaptr);
//             auto [rhs_grad_cudaptr, rhs_grad_cudaid]    = dg::network_virtual_device::devirtualize_cuda_ptr(rhs_grad_vmaptr);
//             auto [rhs_logit_cudaptr, rhs_logit_cudaid]  = dg::network_virtual_device::devirtualize_cuda_ptr(rhs_logit_vmaptr);
//             auto [src_grad_cudaptr, src_grad_cudaid]    = dg::network_virtual_device::devirtualize_cuda_ptr(src_grad_vmaptr);

//             dg::network_tileops_cuda_poly::bwdzr_pair(lhs_grad_cudaptr, lhs_grad_cudaid, lhs_logit_cudaptr, lhs_logit_cudaid, 
//                                                       rhs_grad_cudaptr, rhs_grad_cudaid, rhs_logit_cudaptr, rhs_logit_cudaid, 
//                                                       src_grad_cudaptr, src_grad_cudaid);
//             return true;
//         } 
        
//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto lhs_grad_fsyshost_resource     = map_fsyshost(lhs_grad_vmaptr);
//             auto lhs_logit_fsyshost_resource    = map_fsyshost(lhs_logit_vmaptr);
//             auto rhs_grad_fsyshost_resource     = map_fsyshost(rhs_grad_vmaptr);
//             auto rhs_logit_fsyshost_resource    = map_fsyshost(rhs_logit_vmaptr);
//             auto src_grad_fsyshost_resource     = map_fsyshost(src_grad_vmaptr); 

//             dg::network_tileops_host_poly::bwdzr_pair(lhs_grad_fsyshost_resource, lhs_logit_fsyshost_resource,
//                                                       rhs_grad_fsyshost_resource, rhs_logit_fsyshost_resource,
//                                                       src_grad_fsyshost_resource); //weird
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     auto backward_uacm(uma_ptr_t src) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter;
//         uma_ptr_t src_lck_addr                  = get_uacm_rculock_addr_nothrow(src);
//         std::array<uma_ptr_t, UACM_COUNT> dst   = {}; 

//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
//             dst = get_uacm_src_nothrow(src);
//         }

//         auto addr       = dg::network_genult::tuple_join(std::tuple(src), dst);
//         auto lck_addr   = dg::network_genult::tuple_transform(addr, get_rculock_addr_nothrow);
//         auto lck_grd    = dg::network_genult::tuple_invoke(dg::network_memops_uma::memlock_many_guard, lck_addr);
//         auto ops_id     = dg::network_genult::tuple_transform(addr, get_operatable_id_nothrow);

//         if (!dg::network_genult::tuple_invoke(is_same_value, ops_id)){
//             return false;
//         }

//         auto logit_addr                     = dg::network_genult::tuple_transform(addr, get_logit_addr_nothrow); //
//         auto grad_addr                      = dg::network_genult::tuple_transform(addr, get_grad_addr_nothrow); //
//         dispatch_control_t dispatch_control = get_uacm_dispatch_control_nothrow(src); 
//         auto vd_id                          = dg::network_dispatch_control::decode_uacm_vd_id(dispatch_control); 
//         auto tileops_dp_id                  = dg::network_dispatch_control::decode_uacm_tileops_dp_id(dispatch_control); 
//         auto zipped_logit_addr              = dg::network_genult::tuple_zip(logit_addr, vd_id);
//         auto zipped_grad_addr               = dg::network_genult::tuple_zip(grad_addr, vd_id); 

//     }

//     auto backward_pacm(uma_ptr_t src) noexcept -> bool{

//     }

//     auto backward_crit(uma_ptr_t src) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter;

//         uma_ptr_t src_lck_addr  = get_crit_rculock_addr_nothrow(src); 
//         uma_ptr_t dst           = {};

//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
//             dst = get_crit_src_nothrow(src);
//         }

//         uma_ptr_t dst_lck_addr              = get_rculock_addr_nothrow(dst); 
//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(src_lck_addr, dst_lck_addr);
//         operatable_id_t src_operatable_id   = get_crit_operatable_id_nothrow(src);
//         operatable_id_t dst_operatable_id   = get_operatable_id_nothrow(dst);

//         if (src_operatable_id != dst_operatable_id){
//             return false;
//         }

//         uma_ptr_t dst_logit_umaptr                  = get_logit_addr_nothrow(dst);
//         uma_ptr_t dst_grad_umaptr                   = get_grad_addr_nothrow(dst);
//         uma_ptr_t src_grad_umaptr                   = get_grad_addr_nothrow(src);
//         dispatch_control_t dispatch_control         = get_crit_dispatch_control_nothrow(src);
//         auto [src_vd_id, dst_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_crit(dispatch_control);
//         auto [dst_logit_map_resource, dst_grad_map_resource, src_grad_map_resource] = dg::network_uma::mapsafe_recursivewait_many_nothrow<3>({{dst_logit_umaptr, dst_vd_id}, {dst_grad_umaptr, dst_vd_id}, {src_grad_umaptr, src_vd_id}});
//         vma_ptr_t dst_grad_vmaptr                   = dg::network_uma::get_vma_ptr(dst_grad_map_resource);
//         vma_ptr_t src_grad_vmaptr                   = dg::network_uma::get_vma_ptr(src_grad_map_resource);
        

//         //cuda_id + cuda_handle_t are singleton managed by network_tileops_cuda_poly 
//         //cross device computation or in device computation are runtime-checked by the controller + internal corruption if failed 

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             auto [dst_grad_cudaptr, dst_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(dst_grad_vmaptr);
//             auto [src_grad_cudaptr, src_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(src_grad_vmaptr);
//             dg::network_tileops_cuda_poly::bwdzr_crit(dst_grad_cudaptr, dst_grad_cudaid, src_grad_cudaptr, src_grad_cudaid, tileops_dp_id);
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto dst_grad_fsyshost_resource = map_fsyshost(dst_grad_vmaptr);
//             auto src_grad_fsyshost_resource = get_fsyshost_reosurce(src_grad_vmaptr); 
//             dg::network_tileops_host_poly::bwdzr_crit(dst_grad_fsyshost_resource, src_grad_fsyshost_resource, tileops_dp_id); //implicit conversion is not allowed here - fsyshost resource is vma_ptr_t wrapper - should only implicitly convert to vma_ptr_t - placeholder for now
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     auto backward_extn(uma_ptr_t src) noexcept -> bool{

//     }

//     auto backward_msgrfwd(uma_ptr_t src) noexcept -> bool{

//         using namespace dg::network_tile_member_getsetter; 
        
//         uma_ptr_t src_lck_addr  = get_msgrfwd_rculock_addr_nothrow(src);
//         uma_ptr_t dst           = {};

//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
//             dst = get_msgrfwd_src_nothrow(src);
//         }

//         uma_ptr_t dst_lck_addr              = get_rculock_addr_nothrow(dst);
//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(src_lck_addr, dst_lck_addr);
//         operatable_id_t src_operatable_id   = get_msgrfwd_operatable_id_nothrow(src);
//         operatable_id_t dst_operatable_id   = get_operatable_id_nothrow(dst);

//         if (src_operatable_id != dst_operatable_id){
//             return false;
//         }

//         uma_ptr_t dst_logit_umaptr                  = get_logit_addr_nothrow(dst);
//         uma_ptr_t dst_grad_umaptr                   = get_grad_addr_nothrow(dst);
//         uam_ptr_t src_grad_umaptr                   = get_msgrfwd_grad_addr_nothrow(src);
//         dispatch_control_t dispatch_control         = get_msgrfwd_dispatch_control_nothrow(src);
//         auto [src_vd_id, dst_vd_id, dp_device, tileops_dp_id] = dg::network_dispatch_control::decode_msgrfwd(dispatch_control); //weird
//         auto [dst_logit_map_resource, dst_grad_map_resource, src_grad_map_resource] = dg::network_uma::mapsafe_recursivewait_many_nothrow<3>({{dst_logit_umaptr, dst_vd_id}, {dst_grad_umaptr, dst_vd_id}, {src_grad_umaptr, src_vd_id}});
//         vma_ptr_t dst_grad_vmaptr                   = dg::network_uma::get_vma_ptr(dst_grad_map_resource);
//         vma_ptr_t src_grad_vmaptr                   = dg::network_uma::get_vma_ptr(src_grad_map_resource);

//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){ //uma_ptr_t is cuda_dispatchable - not dp_device - change semantic
//             auto [dst_grad_cudaptr, dst_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(dst_grad_vmaptr);
//             auto [src_grad_cudaptr, src_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(src_grad_vmaptr);
//             dg::network_tileops_cuda_poly::bwdzr_clone(dst_grad_cudaptr, dst_grad_cudaid, src_grad_cudaptr, src_grad_cudaid, tileops_dp_id);
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto dst_grad_fsyshost_resource = map_fsyshost(dst_grad_vmaptr);
//             auto src_grad_fsyshost_resource = map_fsyshost(src_grad_vmaptr); 
//             dg::network_tileops_host_poly::bwdzr_clone(dst_grad_fsyshost_resource, src_grad_fsyshost_resource, tileops_dp_id);
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     auto backward_msgrbwd(uma_ptr_t src) noexcept -> bool{
        
//         using namespace dg::network_tile_member_getsetter; 

//         uma_ptr_t src_lck_addr  = get_msgrbwd_rculock_addr_nothrow(src);
//         uma_ptr_t dst           = {};

//         {
//             auto lck_grd = dg::network_memops_uma::memlock_guard(src_lck_addr);
//             dst = get_msgrbwd_src_nothrow(src);
//         }

//         uma_ptr_t dst_lck_addr              = get_rculock_addr_nothrow(dst);
//         auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(src_lck_addr, dst_lck_addr);
//         operatable_id_t src_operatable_id   = get_msgrbwd_operatable_id_nothrow(src);
//         operatable_id_t dst_operatable_id   = get_operatable_id_nothrow(dst);

//         if (src_operatable_id != dst_operatable_id){
//             return false;
//         }

//         uma_ptr_t dst_logit_umaptr          = get_logit_addr_nothrow(dst);
//         uma_ptr_t dst_grad_umaptr           = get_grad_addr_nothrow(dst);
//         uma_ptr_t src_grad_umaptr           = get_msgrbwd_grad_addr_nothrow(src);
//         dispatch_control_t dispatch_control = get_msgrbwd_dispatch_control_nothrow(src); 
//         auto [src_vd_id, dst_vd_id, dp_device, tileops_dp_id] =  dg::network_dispatch_control::decode_msgrbwd(dispatch_control);
//         auto [dst_logit_map_resource, dst_grad_map_resource, src_grad_map_resource] = dg::network_uma::mapsafe_recursivewait_many_nothrow<3>({{dst_logit_umaptr, dst_vd_id}, {dst_grad_umaptr, dst_vd_id}, {src_grad_umaptr, src_vd_id}});
//         vma_ptr_t dst_grad_vmaptr           = dg::network_uma::get_vma_ptr(dst_grad_map_resource);
//         vma_ptr_t src_grad_vmaptr           = dg::network_uma::get_vma_ptr(src_grad_map_resource);
        
//         if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
//             auto [dst_grad_cudaptr, dst_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(dst_grad_vmaptr);
//             auto [src_grad_cudaptr, src_grad_cudaid] = dg::network_virtual_device::devirtualize_cuda_ptr(src_grad_vmaptr);
//             dg::network_tileops_cuda_poly::bwdzr_clone(dst_grad_cudaptr, dst_grad_cudaid, src_grad_cudaptr, src_grad_cudaid, tileops_dp_id);
//             return true;
//         }

//         if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
//             auto dst_grad_fsyshost_resource = map_fsyshost(dst_grad_vmaptr);
//             auto src_grad_fsyshost_resource = map_fsyshost(src_grad_vmaptr); 
//             dg::network_tileops_host_poly::bwdzr_clone(dst_grad_fsyshost_resource, src_grad_fsyshost_resource, tileops_dp_id);
//             return true;
//         }

//         if constexpr(DEBUG_MODE_FLAG){
//             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
//             std::abort();
//         }

//         return false;
//     }

//     //----
//     void dispatch_fwdpong(uma_ptr_t dst) noexcept{

//     }

//     void dispatch_fwdmsgr(uma_ptr_t dst) noexcept{

//     }

//     void dispatch_bwdmsgr(uma_ptr_t dst) noexcept{

//     }


// } 

// #endif

namespace dg::network_memcommit_resolutor{

    struct UnifiedMemoryIPRetrieverInterface{
        virtual ~UnifiedMemoryIPRetrieverInterface() noexcept = default;
        virtual auto ip(uma_ptr_t) const noexcept -> Address = 0; 
    };

    struct HostIPRetrieverInterface{
        virtual ~HostIPRetrieverInterface() noexcept = default;
        virtual auto ip() const noexcept -> Address = 0;
    };

    struct ForeignTileAliasGetterInterface{
        virtual ~ExtnSrcIngestionPoolInterface() noexcept = default;
        virtual auto alias(uma_ptr_t) noexcept -> std::optional<uma_ptr_t> = 0; //reduce lock_collisions by using distributed hash_map
    };

    template <class T>
    struct Request{
        Address requestor;
        Address requestee;
        T content;
    };

    class ForwardPingLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        public:

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                (void) ptr_arr;
            }
    };

    class ForwardPingPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPingPairSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity); //there is a perf constraint here - in the sense of delivery locality - when we ping tiles - we want locality of accesses - and locality of deliveries

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(ptr);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(ptr);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t left_descendant   = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(ptr);
                        uma_ptr_t right_descendant  = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(ptr);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_ping_signal(left_descendant));
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_ping_signal(right_descendant));
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_request(left_descendant, ptr));
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_request(right_descendant, ptr));
                        dg::network_tile_member_getsetter::set_pair_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
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
    };

    class ForwardPingUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPingUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity); //there is a perf constraint here -

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(ptr);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(ptr);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        std::array<uma_ptr_t, UACM_ACM_SZ> descendant_arr = dg::network_tile_member_getsetter::get_uacm_descendant_nothrow(ptr);

                        for (size_t i = 0u; i < UACM_ACM_SZ; ++i){
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_ping_signal(descendant_arr[i]));
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_request(descendant_arr[i], ptr));
                        }

                        dg::network_tile_member_getsetter::set_uacm_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
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
    };

    class ForwardPingPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPingPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity); //there is a perf constraint here - 

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safe_pacm_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(ptr);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(ptr);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        std::array<uma_ptr_t, PACM_ACM_SZ> left_descendant_arr  = dg::network_tile_member_getsetter::get_pacm_left_descendant_nothrow(ptr);
                        std::array<uma_ptr_t, PACM_ACM_SZ> right_descendant_arr = dg::network_tile_member_getsetter::get_pacm_right_descendant_nothrow(ptr); 

                        for (size_t i = 0u; i < PACM_ACM_SZ; ++i){
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_ping_signal(left_descendant_arr[i]));
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_ping_signal(right_descendant_arr[i]));
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_request(left_descendant_arr[i], ptr));
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_request(right_descendant_arr[i], ptr));
                        }

                        dg::network_tile_member_getsetter::set_pacm_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
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
    };

    class ForwardPingExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPingExtnSrcSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity); //there is a perf constraint here - 

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(ptr);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(ptr);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t descendant = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(ptr);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_ping_signal(descendant));
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_request(descendant, ptr));
                        dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
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
    };

    class ForwardPingExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            const size_t delivery_capacity;

        public:

            DstExternalForwardPingSignalResolutor(std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                  std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                  std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<external_virtual_memory_event_t>> request_box,
                                                  size_t delivery_capacity): uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                             host_ip_retriever(std::move(host_ip_retriever)),
                                                                             request_box(std::move(request_box)),
                                                                             delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity); //there is a perf constraint here - 

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t requestee, dg::network_producer_consumer::DeliveryHandle<external_virtual_memory_event_t> * handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_guard(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t requestor     = dg::network_tile_member_getsetter::get_extndst_src_addr_nothrow(requestee);
                        Address requestor_ip    = this->uma_ip_retriever->ip(src_addr);
                        Address requestee_ip    = this->host_ip_retriever->ip();
                        Request<external_virtual_memory_event_t> ping_request{requestor_ip, requestee_ip, dg::network_external_memcommit_factory::make_event_forward_ping_signal(requestor)};
                        Request<external_virtual_memory_event_t> pong_request{requestor_ip, requestee_ip, dg::network_external_memcommit_factory::make_event_forward_pong_request(requestor, requestee)};
                        dg::network_producer_conumser::delvrsrv_deliver(handle, std::move(ping_request));
                        dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(pong_request));
                        dg::network_tile_member_getsetter::set_extndst_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
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
    };
    
    class ForwardPingCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPingCritSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity); //there is a perf constraint here - 

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                } 

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(ptr);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(ptr);
                
                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t descendant = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(ptr);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_ping_signal(descendant));
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_request(descendant, ptr));
                        dg::network_tile_member_getsetter::set_crit_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
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
    };

    class ForwardPingMsgrFwdResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_mmeory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPingMsgrFwdResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                        size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                            delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity); //there is a perf constraint here - 

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(ptr);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(ptr);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(ptr);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_ping_signal(descendant));
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_request(descendant, ptr));
                        dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
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
    };

    class ForwardPingMsgrBwdResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPingMsgrBwdResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                        size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                            delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity); //there is a perf constraint here - 

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(ptr);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(ptr);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(ptr);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_ping_signal(descendant));
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_request(descendant, ptr));
                        dg::network_tile_member_getsetter::set_msgrbwd_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
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
    };

    class ForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor;
            const size_t pair_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor;
            const size_t crit_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;

        public:

            ForwardPingSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor,
                                       size_t leaf_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor,
                                       size_t pair_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor,
                                       size_t uacm_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor,
                                       size_t pacm_dispatch_sz,
                                       std::unique_Ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor,
                                       size_t extnsrc_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor,
                                       size_t extndst_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor,
                                       size_t crit_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor,
                                       size_t msgrfwd_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor,
                                       size_t msgrbwd_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
                                                                             leaf_dispatch_sz(leaf_dispatch_sz),
                                                                             pair_resolutor(std::move(pair_resolutor)),
                                                                             pair_dispatch_sz(pair_dispatch_sz),
                                                                             uacm_resolutor(std::move(uacm_resolutor)),
                                                                             uacm_dispatch_sz(uacm_dispatch_sz),
                                                                             pacm_resolutor(std::move(pacm_resolutor)),
                                                                             pacm_dispatch_sz(pacm_dispatch_sz),
                                                                             extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                             extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                             extndst_resolutor(std::move(extndst_resolutor)),
                                                                             extndst_dispatch_sz(extndst_dispatch_sz),
                                                                             crit_resolutor(std::move(crit_resolutor)),
                                                                             crit_dispatch_sz(crit_dispatch_sz),
                                                                             msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                             msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                             msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                             msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto leaf_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_resolutor.get(), this->leaf_dispatch_sz);
                auto pair_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz);
                auto uacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz);
                auto pacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz);
                auto extnsrc_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz);
                auto extndst_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz);
                auto crit_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz);
                auto msgrfwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz);
                auto msgrbwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz);

                if (!dg::network_exception::conjunc_expect_has_value(leaf_delivery_handle, pair_delivery_handle, uacm_delivery_handle,
                                                                     pacm_delivery_handle, extnsrc_delivery_handle, extndst_delivery_handle,
                                                                     crit_delivery_handle, msgrfwd_delivery_handle, msgrbwd_delivery_handle)){

                    dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(ptr_arr[i]); 

                    if (!tile_kind.has_value()){ //this branch is never taken - so we don't worry - this takes at most 2-3 CPU cycle
                        dg::network_log_stackdump::error(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(leaf_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PAIR:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pair_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_UACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(uacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNSRC:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extnsrc_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNDST:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extndst_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_CRIT:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(crit_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRFWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrfwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRBWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrbwd_delivery_handle)->get(), ptr_arr[i]);
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
    };

    //

    class ForwardPongRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPongRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                        size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                            delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(requestee_rcu_addr);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        size_t observer_arr_sz                                  = dg::network_tile_member_getsetter::get_tile_observer_array_size_nothrow(requestee);
                        size_t observer_arr_idx                                 = observer_arr_sz % OBSERVER_ARRAY_SZ;
                        std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = dg::network_tile_member_getsetter::get_tile_observer_nothrow(requestee);
                        observer_arr[observer_arr_idx]                          = requestor;
                        size_t new_observer_arr_sz                              = observer_arr_idx + 1; //this needs to be an ordered_set - 

                        dg::network_tile_member_getsetter::set_tile_observer_nothrow(requestee, observer_arr);
                        dg::network_tile_member_getsetter::set_tile_observer_array_size_nothrow(requestee, new_observer_arr_sz);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_pong_signal(requestor, requestee);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(request));
                        break;
                    }
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
    };

    //

    class ForwardPongSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPongSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                       size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                           delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle.get());
                }
            }

        private:

            void resolve(uma_ptr_t signalee, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                uma_ptr_t signalee_rcu_addr         = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(signalee);
                dg::network_memops_uma::memlock_guard mem_grd(signalee_rcu_addr);
                init_status_t init_status           = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(signalee);

                switch (init_status){
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        pong_count_t pong_count         = dg::network_tile_member_getsetter::get_tile_pong_count_nothrow(signalee);
                        descendant_size_t descendant_sz = dg::network_tile_member_getsetter::get_tile_descendant_size_nothrow(signalee);

                        if (descendant_sz == pong_count){
                            virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_do_signal(signalee);
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(request));
                        } else{
                            pong_count += 1;
                            dg::network_tile_member_getsetter::set_tile_pong_count_nothrow(signalee, pong_count);
                            if (pong_count == descendant_sz){
                                virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_do_signal(signalee);
                                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(request));
                            }
                        }

                        break;
                    }
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
    };

    class ForwardInitMonoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardInitMonoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            //alright guys - things are complicated - we want to see if init_status_t == DECAYED - we then want to see if src is initialized - then we forward - then we decay init_signal -> pong_signal
            //5 flops/ dispatch is prolly a dream - we tried our best to reduce as many polymorphism overhead as possible - I think we better vectorize uma_ptr_t * dispatch to reduce cuda synchronization overheads here - rather than using "array" approaches - this is a bad approach as we already talked about this being not a quantifiable thing
            //think of the vectorizations as delvsrv_open_raiihandle - the only diff is we stop when std::vector<std::tuple<void * __restrict__, const void * __restrict__, const void * __restrict__>> contract is broken

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_lck_addr  = get_mono_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {}; 

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr); 
                    src = get_mono_src_nothrow(dst);
                }

                uma_ptr_t src_lck_addr = get_tile_rcu_addr_nothrow(src); //access_err
                dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr, src_lck_addr); //we dont want to use try_lock because it's not a good practice here - so let's actually do twice lock_guards - lock_guard does mmeory flush + everything - which is good

                //we want to combine some of these guys to avoid too many cache reads - we'll do that after implementing this - we cant rely on uma_ptr_t * being adjecent to offset the costs

                uma_ptr_t new_src                                           = get_mono_src_nothrow(dst);
                init_status_t dst_init_status                               = get_mono_init_status_nothrow(dst);
                operatable_id_t dst_operatable_id                           = get_mono_operatable_id_nothrow(dst);
                std::array<uma_ptr_t, OBSERVER_ARRAY_CAP> dst_observer_arr  = get_mono_observer_array(dst);
                size_t dst_observer_arr_sz                                  = get_mono_observer_array_size(dst);
                uma_ptr_t dst_logit_umaptr                                  = get_mono_logit_addr_nothrow(dst);
                dispatch_control_t dispatch_control                         = get_mono_dispatch_control_nothrow(dst);
                init_status_t src_init_status                               = get_tile_init_status_nothrow(src);
                operatable_id_t src_operatable_id                           = get_tile_operatable_id_nothrow(src);
                uma_ptr_t src_logit_umaptr                                  = get_tile_logit_addr_nothrow(src);

                if (new_src != src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return;
                }

                auto [dst_vd_id, src_vd_id, dp_device, tileops_dp]  = dg::network_dispatch_control::decode_mono(dispatch_control);
                auto [dst_map_resource, src_map_resource]           = dg::network_uma::lockmap_safewait_many<2u>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}}); //weird - I mean this could combined with vmamap - yet I have yet wanted to complicate this further
                auto dst_logit_vmaptr                               = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto src_logit_vmaptr                               = dg::network_uma::get_vma_ptr(src_map_resource); 
                auto dst_logit_vmamap                               = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
                auto src_logit_vmamap                               = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);
                
                //no-ops on errors

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                    dg::network_tileops_cuda_poly::fwd_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), tileops_dp);
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                    dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), tileops_dp);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                set_mono_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                for (size_t i = 0u; i < dst_observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(dst_observer_arr[i]));
                }
            }
    };

    class ForwardInitPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardInitPairSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                using namespace dg::network_tile_member_getsetter;
                auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error())); //print statemnt needs to be more descriptive
                    return;
                } 

                uma_ptr_t dst_lck_addr  = get_pair_rcu_addr_nothrow(dst);
                uma_ptr_t lhs           = {};
                uma_ptr_t rhs           = {};

                //fine - refactor later
                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr);
                    lhs = get_pair_left_descendant_nothrow(dst);
                    rhs = get_pair_right_descendant_nothrow(dst);
                }

                uma_ptr_t lhs_lck_addr  = get_tile_rcu_addr_nothrow(lhs);
                uma_ptr_t rhs_lck_addr  = get_tile_rcu_addr_nothrow(rhs);
                dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr, lhs_lck_addr, rhs_lck_addr); //access err

                uma_ptr_t new_lhs                                       = get_pair_left_descendant_nothrow(dst);
                uma_ptr_t new_rhs                                       = get_pair_right_descendant_nothrow(dst);
                uma_ptr_t dst_logit_umaptr                              = get_pair_logit_addr_nothrow(dst);
                operatable_id_t dst_operatable_id                       = get_pair_operatable_id_nothrow(dst);
                init_status_t dst_init_status                           = get_pair_init_status_nothrow(dst);
                dispatch_control_t dispatch_control                     = get_pair_dispatch_control_nothrow(dst);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = get_pair_observer_arr_nothrow(dst);
                size_t observer_arr_sz                                  = get_pair_observer_arr_size_nothrow(dst);
                operatable_id_t lhs_operatable_id                       = get_tile_operatable_id_nothrow(lhs);
                init_status_t lhs_init_status                           = get_tile_init_status_nothrow(lhs);
                uma_ptr_t lhs_logit_umaptr                              = get_tile_logit_addr_nothrow(lhs);
                operatable_id_t rhs_operatable_id                       = get_tile_operatable_id_nothrow(rhs);
                init_status_t rhs_init_status                           = get_tile_init_status_nothrow(rhs);
                uma_ptr_t rhs_logit_umaptr                              = get_tile_logit_addr_nothrow(rhs);

                if (lhs != new_lhs){
                    return;
                }

                if (rhs != new_rhs){
                    return;
                }

                if (dst_operatable_id != lhs_operatable_id || lhs_operatable_id != rhs_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return;
                }

                if (lhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (rhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                auto [dst_vd_id, lhs_vd_id, rhs_vd_id, dp_device_kind, tileops_dp_id] = dg::network_dispatch_control::decode_pair(dispatch_control);

                auto [dst_map_resource, lhs_map_resource, rhs_map_resource] = dg::network_uma::lockmap_safewait_many<3u>({{dst_logit_umaptr, dst_vd_id}, {lhs_logit_umaptr, lhs_vd_id}, {rhs_logit_umaptr, rhs_vd_id}});
                auto dst_logit_vmaptr   = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto lhs_logit_vmaptr   = dg::network_uma::get_vma_ptr(lhs_map_resource);
                auto rhs_logit_vmaptr   = dg::network_uma::get_vma_ptr(rhs_map_resource); 
                auto dst_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
                auto lhs_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(lhs_logit_vmaptr);
                auto rhs_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(rhs_logit_vmaptr); 

                //dispatch errs 
                //no-ops on errors

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_cuda_poly::fwd_pair(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(lhs_logit_vmamap), dg::network_vmamap::get_cuda_ptr(rhs_logit_vmamap), tileops_dp_id);
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    dg::network_tileops_host_poly::fwd_pair(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(lhs_logit_vmamap), dg::network_vmamap::get_host_ptr(rhs_logit_vmamap), tileops_dp_id);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                for (size_t i = 0u; i < observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i], dst));
                }

                set_pair_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
            }
    };

    class ForwardInitUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardInitUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

            }
    };

    class ForwardInitPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerIntterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardInitPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{
                
            }
    };

    class ForwardInitExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<Request<external_virtual_memory_event_t>>>> request_box;
            std::unique_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            std::unique_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const size_t delivery_capacity;

        public:

            SrcExternalForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                                  std::unique_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                  std::unique_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                  size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                      uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                      host_ip_retriever(std::move(host_ip_retriever)),
                                                                                      delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_lck_addr = get_extnsrc_rcu_addr_nothrow(dst);
                uma_ptr_t src = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr);
                    src = get_extnsrc_descendant_nothrow(src);
                }

                uma_ptr_t src_lck_addr = get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr, src_lck_addr);
             
                uma_ptr_t new_src                       = get_extnsrc_descendant_nothrow(dst);
                init_status_t dst_init_status           = get_extnsrc_init_status_nothrow(dst);
                operatable_id_t dst_operatable_id       = get_extnsrc_operatable_id_nothrow(dst);
                uma_ptr_t dst_logit_umaptr              = get_extnsrc_logit_addr_nothrow(dst); 
                dispatch_control_t dst_dispatch_control = get_extnsrc_dispatch_control_nothrow(dst);
                uma_ptr_t counterpart                   = get_extnsrc_counterpart_nothrow(dst);
                init_status_t src_init_status           = get_tile_init_status_nothrow(src);
                operatble_id_t src_operatable_id        = get_tile_operatable_id_nothrow(src);
                uma_ptr_t src_logit_umaptr              = get_tile_logit_addr_nothrow(src); 

                if (new_src != src){
                    return;
                }

                if (src_operatable_id != dst_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                auto [dst_vd_id, src_vd_id, dp_device_kind, tileops_dp_id]  = dg::network_dispatch_control::decode_extnsrc(dispatch_control);
                auto [dst_map_resource, src_map_resource]                   = dg::network_uma::lockmap_safewait_many<2u>({dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id});
                auto dst_logit_vmaptr   = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto src_logit_vmaptr   = dg::network_uma::get_vma_ptr(src_map_resource);
                auto dst_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
                auto src_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);

                //dispatch errs
                //no-ops on errors

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_cuda_poly::fwd_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), tileops_dp_id);
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), tileops_dp_id);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                set_extnsrc_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                dg::string serialized                           = serialize_extnsrc(dst);
                Address fr_addr                                 = this->host_ip_retriever->ip();
                Address to_addr                                 = this->uma_ip_retriever->ip(counterpart);
                external_virtual_memory_event_t inject_event    = dg::network_external_memcommit_factory::make_event_shadow_injection(dst, TILE_KIND_EXTNSRC, std::move(serialized));
                external_virtual_memory_event_t notify_event    = dg::network_external_memcommit_factory::make_event_forward_init_signal(counterpart);
                external_virtual_memory_event_t event           = dg::network_external_memcommit_factory::make_event_sequential(std::move(inject_event), std::move(notify_event));
                Request<external_virtual_memory_event_t> rq     = {to_addr, fr_addr, std::move(event)};

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(rq));
            }
    };

    class ForwardInitExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter;
            const size_t delivery_capacity;

        public:

            DstExternalForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                  std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter,
                                                  size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                      alias_getter(std::move(alias_getter)),
                                                                                      delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error())); //be more descriptive
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_extndst_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = get_extndst_descendant_nothrow(dst);
                }

                std::optional<uma_ptr_t> local_src = this->alias_getter->alias(src);

                if (!local_src.has_value()){
                    return;
                }

                uma_ptr_t local_src_rcu_addr                                = get_extnsrc_rcu_addr_nothrow(local_src.value()); //access_err
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, local_src_rcu_addr);
                uma_ptr_t new_src                                           = get_extndst_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id                           = get_extndst_operatable_id_nothrow(dst);
                init_status_t dst_init_status                               = get_extndst_init_status_nothrow(dst);
                dispatch_control_t dispatch_control                         = get_extndst_dispatch_control_nothrow(dst);
                uma_ptr_t dst_logit_umaptr                                  = get_extndst_logit_addr_nothrow(dst); 
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> dst_observer_arr    = get_extndst_observer_array_nothrow(dst);
                size_t dst_observer_arr_sz                                  = get_extndst_observer_array_size_nothrow(dst);
                operatable_id_t src_operatable_id                           = get_extnsrc_operatable_id_nothrow(local_src.value());
                init_status_t src_init_status                               = get_extnsrc_init_status_nothrow(local_src.value());
                uma_ptr_t src_logit_umaptr                                  = get_extnsrc_logit_addr_nothrow(local_src.value());

                if (new_src != src){
                    return; //no-ops no-error
                }

                if (dst_operatable_id != src_operatable_id){
                    return; //no-ops no-error
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return; //no-ops no-error
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return; //no-ops no-error
                }

                auto [dst_vd_id, src_vd_id, dp_device_kind, tileops_dp_kind]    = dg::network_dispatch_control::decode_extndst(dispatch_control);
                auto [dst_map_resource, src_map_resource]                       = dg::network_uma::lockmap_safewait_many<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                auto dst_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto src_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(src_map_resource);
                auto dst_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr); //weird invention
                auto src_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);

                //dispatch device confirms pre dispatch - no-ops no-error

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_cuda_poly::fwd_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmammap::get_cuda_ptr(src_logit_vmamap), tileops_dp_kind);
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmammap::get_host_ptr(src_logit_vmamap), tileops_dp_kind);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                set_extndst_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                for (size_t i = 0u; i < dst_observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(dst_observer_arr[i]));
                }
            }
    };

    class ForwardInitCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            CritForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{
                
                //I'm afraid for the ones that maintain this repo after me - there are tons of technical specs that are hard to understand - 
                //the concept of memory_lock needs to be changed - either in the direction of dependencies - acquire a member requires the rcu to be free, etc.
                //or in the direction of atomic_reference
                //due to performance constraints - we rather change the semantics of the memory_lock than to fix the implementations to fit the memory_lock semantics

                //first is the memory locks - it has to have a clear exit (ConsumerInterface<virtual_memory_event_t> is not a clear exit) - explicit exits during the lock to guarantee non-deadlock
                //second - if you acquire multiple locks - it has to be in the same order every single time in the program - such forms a tree of lock acquisitions
                //third - if you acquire locks that aren't in the same order - it has to be in a single payload
                //second and third can be mixed - such that you can acquired multiple locks that aren't in the same order that forms an hierarchical tree
                //example is memlock_guard(args...) then lockmap(args...)
                //error msgs need to be more descriptives 
                //there should be more error checking for tile inputs - and safely return with no-ops on error

                auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error())); //be more descriptive
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_crit_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = get_crit_descendant_nothrow(dst);
                }

                uma_ptr_t src_rcu_addr                                  = get_tile_rcu_addr_nothrow(src);
                dg::network_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                                       = get_crit_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id                       = get_crit_operatable_id_nothrow(dst);
                init_status_t dst_init_status                           = get_crit_init_status_nothrow(dst);
                dispatch_control_t dispatch_control                     = get_crit_dispatch_control_nothrow(dst);
                std::array<uma_ptr_t, OBSERVER_ARRAY_CAP> observer_arr  = get_crit_observer_array_nothrow(dst);
                size_t observer_arr_sz                                  = get_crit_observer_array_size_nothrow(dst);
                uma_ptr_t dst_logit_umaptr                              = get_crit_logit_addr_nothrow(dst);
                uma_ptr_t dst_grad_umaptr                               = get_crit_grad_addr_nothrow(dst); 
                uma_ptr_t dst_clogit_umaptr                             = get_crit_clogit_addr_nothrow(dst);
                crit_kind_t crit_kind                                   = get_crit_crit_kind(dst);
                crit_ratio_t crit_ratio                                 = get_crit_crit_ratio(dst);
                init_status_t src_init_status                           = get_tile_init_status_nothrow(src);
                operatable_id_t src_operatable_id                       = get_tile_operatable_id_nothrow(src);
                uma_ptr_t src_logit_umaptr                              = get_tile_logit_addr_nothrow(src);
                uma_ptr_t src_grad_umaptr                               = get_tile_grad_addr_nothrow(src); //access compromise - no-ops on errors 

                if (new_src != src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return;
                }

                auto dispatch_info = dg::network_dispatch_control::decode_crit(dispatch_control);

                {
                    auto [dst_logit_map_resource, src_logit_map_resource] = dg::network_uma::lockmap_safewait_many<2>({{dst_logit_umaptr, dispatch_info.fwd_dst_logit_vd_id}, {src_logit_umaptr, dispatch_info.fwd_src_logit_vd_id}});
                    auto dst_logit_vmaptr   = dg::network_uma::get_vma_ptr(dst_logit_map_resource);
                    auto src_logit_vmaptr   = dg::network_uma::get_vma_ptr(src_logit_map_resource);
                    auto dst_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr); //
                    auto src_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr); //

                    if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.fwd_dp_device_kind)){
                        dg::network_tileops_cuda_poly::fwd_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), dispatch_info.fwd_tileops_dp_kind);
                    } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.fwd_dp_device_kind)){
                        dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), dispathc_info.fwd_tileops_dp_kind);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                }

                {
                    auto [dst_logit_map_resource, dst_clogit_map_resource, dst_grad_map_resource] = dg::network_uma::lockmap_safewait_many<3>({{dst_logit_umaptr, dispatch_info.crit_dst_logit_vd_id}, {dst_clogit_umaptr, dispatch_info.crit_dst_clogit_vd_id}, {dst_grad_umaptr, dispatch_info.crit_dst_grad_vd_id}});
                    auto dst_logit_vmaptr   = dg::network_uma::get_vma_ptr(dst_logit_map_resource);
                    auto dst_clogit_vmaptr  = dg::network_uma::get_vma_ptr(dst_clogit_map_resource);
                    auto dst_grad_vmaptr    = dg::network_uma::get_vma_ptr(dst_grad_map_resource);

                    if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.crit_dp_device_kind)){

                    } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.crit_dp_device_kind)){

                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                }

                {
                    auto [src_grad_map_resource, src_logit_map_resource, dst_grad_map_resource] = dg::network_uma::lockmap_safewait_many<3>({{src_grad_umaptr, dispatch_info.bwd_src_grad_vd_id}, {src_logit_umaptr, dispatch_info.bwd_src_logit_vd_id}, {dst_grad_umaptr, dispatch_info.bwd_dst_grad_vd_id}});
                    auto src_grad_vmaptr    = dg::network_uma::get_vma_ptr(src_grad_map_resource);
                    auto src_logit_vmaptr   = dg::network_uma::get_vma_ptr(src_logit_map_resource);
                    auto dst_grad_vmaptr    = dg::network_uma::get_vma_ptr(dst_grad_map_resource);
                    auto src_grad_vmamap    = dg::network_vmamap::mapsafe_nothrow(src_grad_vmaptr);
                    auto src_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);
                    auto dst_grad_vmamap    = dg::network_vmamap::mapsafe_nothrow(dst_grad_vmaptr);

                    if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.bwd_dp_device_kind)){
                        dg::network_tileops_cuda_poly::bwd_mono(dg::network_vmamap::get_cuda_ptr(src_grad_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap), dispatch_info.bwd_tileops_dp_kind);
                    } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.bwd_dp_device_kind)){
                        dg::network_tileops_host_poly::bwd_mono(dg::network_vmamap::get_host_ptr(src_grad_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), dg::network_vmamap::get_host_ptr(dst_grad_vmamap), dispatch_info.bwd_tileops_dp_kind);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }

                }

                set_crit_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                for (size_t i = 0u; i < observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i]));
                }

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class ForwardInitMsgrFwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t request_box_delivery_capacity;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box;
            const size_t eu_packet_box_delivery_capacity;

        public:

            MsgrFwdForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserInterface>> eu_packet_box,
                                              size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                  eu_packet_box(std::move(eu_packet_box)),
                                                                                  delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto request_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);
                auto eu_packet_delivery_handle  = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->eu_packet_box.get(), this->eu_packet_box_delivery_capacity); 

                if (!request_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(request_delivery_handle.error()));
                    return;
                }

                if (!eu_packet_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(eu_packet_delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], request_delivery_handle->get(), eu_packet_delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, 
                         dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle,
                         dg::network_producer_consumer::DeliveryHandle<EndUserPacket> * eu_packet_delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_msgrfwd_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = get_msgrfwd_descendant_nothrow(dst);
                }

                uma_ptr_t src_rcu_addr                                      = get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                                           = get_msgrfwd_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id                           = get_msgrfwd_operatable_id_nothrow(dst);
                init_status_t dst_init_status                               = get_msgrfwd_init_status_nothrow(dst);
                dispatch_control_t dispatch_control                         = get_msgrfwd_dispatch_control_nothrow(dst);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> dst_observer_arr    = get_msgrfwd_observer_array_nothrow(dst);
                size_t dst_observer_arr_size                                = get_msgrfwd_observer_array_size_nothrow(dst);
                uma_ptr_t dst_logit_umaptr                                  = get_msgrfwd_logit_addr_nothrow(dst); 
                size_t msgr_retry_count                                     = get_msgrfwd_retry_count_nothrow(dst);
                dst_info_t msgr_dst                                         = get_msgrfwd_dst_info_nothrow(dst);
                logit_id_t msgr_logit_id                                    = get_msgrfwd_logit_id_nothrow(dst); 
                transmit_urgency_t msgr_transmit_urgency                    = get_msgrfwd_transmit_urgency_nothrow(dst);
                transmit_comm_t msgr_transmit_comm                          = get_msgrfwd_transmit_comm_nothrow(dst);
                operatable_id_t src_operatable_id                           = get_tile_operatable_id_nothrow(src);
                init_status_t src_init_status                               = get_tile_init_status_nothrow(src);
                uma_ptr_t src_logit_umaptr                                  = get_tile_logit_addr_nothrow(src); 

                if (new_src != src){
                    return; //no-ops no-err
                }

                if (dst_operatable_id != src_operatable_id){
                    return; //no-ops no-err
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return; //no-ops no-err
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return; //no-ops no-err
                }

                auto [dst_vd_id, src_vd_id, dp_device_kind, tileops_dp_kind]    = dg::network_dispatch_control::decode_msgrfwd(dispatch_control);
                auto [dst_map_resource, src_map_resource]                       = dg::network_uma::lockmap_safewait_many<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                auto dst_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto src_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(src_map_resource);
                auto dst_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
                auto src_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);

                //no-ops on errors

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_cuda_poly::forward_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), tileops_dp_kind);
                }  else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    dg::network_tileops_host_poly::forward_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), tileops_dp_kind);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                set_msgrfwd_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                for (size_t i = 0u; i < dst_observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(dst_observer_arr[i]));
                }

                EndUserPacket eu_packet{};

                eu_packet.kind          = EUPACKET_MSGRFWD_TRANSMIT;
                eu_packet.content       = dg::network_compact_serializer::serialize<dg::string>(LogitData{logit_id, get_msgrfwd_logit_nothrow(dst)});
                eu_packet.dst           = msgr_dst;
                eu_packet.retry_count   = msgr_retry_count;
                eu_packet.urgency       = msgr_transmit_urgency;
                eu_packet.comm          = msgr_transmit_comm;

                dg::network_producer_consumer::delvrsrv_deliver(eu_packet_delivery_handle, std::move(eu_packet));
            }
    };

    class ForwardInitMsgrBwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardInitMsgrBwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_msgrbwd_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = get_msgrbwd_descendant_nothrow(dst);
                }

                uma_ptr_t src_rcu_addr                                      = get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                                           = get_msgrbwd_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id                           = get_msgrbwd_operatable_id_nothrow(dst);
                dispatch_control_t dispatch_control                         = get_msgrbwd_dispatch_control_nothrow(dst);
                init_status_t dst_init_status                               = get_msgrbwd_init_status_nothrow(dst);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> dst_observer_arr    = get_msgrbwd_observer_array_nothrow(dst);
                size_t dst_observer_arr_sz                                  = get_msgrbwd_observer_array_size_nothrow(dst);
                uma_ptr_t dst_logit_umaptr                                  = get_msgrbwd_loigt_addr_nothrow(dst);
                init_status_t src_init_status                               = get_tile_init_status_nothrow(src);
                operatable_id_t src_operatable_id                           = get_tile_operatable_id_nothrow(src);
                uma_ptr_t src_logit_umaptr                                  = get_tile_logit_addr_nothrow(src);

                if (new_src != src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                auto [dst_vd_id, src_vd_id, dp_device_kind, tileops_dp_kind]    = dg::network_dispatch_control::decode_msgrbwd(dispatch_control);
                auto [dst_map_resource, src_map_resource]                       = dg::network_uma::lockmap_safewait_many<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                auto dst_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto src_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(src_map_resource);
                auto dst_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
                auto src_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);

                if (dg::network_dispatch_control::is_cuda_dispatch(tileops_dp_kind)){
                    dg::network_tileops_cuda_poly::forward_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), tileops_dp_kind);
                } else if (dg::network_dispatch_control::is_host_dispatch(tileops_dp_kind)){
                    dg::network_tileops_host_poly::forward_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), tileops_dp_kind);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                set_msgrbwd_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                for (size_t i = 0u; i < dst_observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(dst_observer_arr[i]));
                }
            }
    };

    class ForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> mono_resolutor;
            const size_t mono_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor;
            const size_t pair_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor;
            const size_t crit_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;


        public:

            ForwardInitSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> mono_resolutor,
                                       size_t mono_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor,
                                       size_t pair_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor,
                                       size_t uacm_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor,
                                       size_t pacm_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor,
                                       size_t extnsrc_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor,
                                       size_t extndst_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor,
                                       size_t crit_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor,
                                       size_t msgrfwd_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor,
                                       size_t msgrbwd_dispatch_sz): mono_resolutor(std::move(mono_resolutor)),
                                                                    mono_dispatch_sz(mono_dispatch_sz),
                                                                    pair_resolutor(std::move(pair_resolutor)),
                                                                    pair_dispatch_sz(pair_dispatch_sz),
                                                                    uacm_resolutor(std::move(uacm_resolutor)),
                                                                    uacm_dispatch_sz(uacm_dispatch_sz),
                                                                    pacm_resolutor(std::move(pacm_resolutor)),
                                                                    pacm_dispatch_sz(pacm_dispatch_sz),
                                                                    extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                    extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                    extndst_resolutor(std::move(extndst_resolutor)),
                                                                    extndst_dispatch_sz(extndst_dispatch_sz),
                                                                    crit_resolutor(std::move(crit_resolutor)),
                                                                    crit_dispatch_sz(crit_dispatch_sz),
                                                                    msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                    msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                    msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                    msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{
                
                auto mono_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->mono_resolutor.get(), this->mono_dispatch_sz);
                auto pair_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz);
                auto uacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz);
                auto pacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz);
                auto extnsrc_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz);
                auto extndst_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz);
                auto crit_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz);
                auto msgrfwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz);
                auto msgrbwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz); 

                //resource leak is expected here - we don't want to communicate the error because it would mess up the code very badly - let's assume that everything is "might", "maybe", and set a timeout for that - like network packet
                //users of the engines wait for msgrfwds to be intiialized within a certain windows - or deallocate and move on
                //error handlings in producer-consumer situation is not encouraged - and it is not a good practice also - this is an error that should be controlled in users' code flow

                if (!dg::network_exception::conjunc_expect_has_value(mono_delivery_handle, pair_delivery_handle, uacm_delivery_handle,
                                                                     pacm_delivery_handle, extnsrc_delivery_handle, extndst_delivery_handle,
                                                                     crit_delivery_handle, msgrfwd_delivery_handle, msgrbwd_delivery_handle)){

                    dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION)); //usually I would abort and throw kernel an exception - to restart the program but this is better to be a no-ops
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(ptr_arr[i]); //better to chk for error here - it is not expensive in terms of branching - because that branch is mostly not taken

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_MONO:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(mono_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PAIR:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pair_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_UACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(uacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNSRC:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extnsrc_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNDST:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extndst_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_CRIT:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(crit_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRFWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrfwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRBWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrbwd_delivery_handle)->get(), ptr_arr[i]);
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
                    };
                }
            }
    };

    //

    //we'll get these compiled then we'll write a linter that could turn our code -> C++ acceptable code - we don't care about that now
    //we care about easy to debug - easy to digest - after the program is theoretically working - then we'll lint the program
    //there are bugs that are hard to catch without inlining the entire programs - like global var accesses
    //when inlining the program, we have to assume that the global vars are concurrent vars - which requires us to use concurrent-safety-measures when we access those vars
    //when we not inlining the program, we accidentally, undefinely solve those problems by compiling those guys separately

    //alrights, we'll try to reduce the overhead flops, after get the grasp of the high level interface - we want to reduce worst case member_access -> one_cache_read + 1 memlock + try_acquire - increase locality of access by scheduling, etc.
    //the chance of lock acquisition must be unif for all tasks 

    //I think the scheduling algorithm is the state of the art for concurrent training
    //we want to schedule the tile -> max_init_time(descendants)

    //the further the forward hops or the backward hops - the lesser the scan frequency on such memregion
    //scan frequency ONLY makes sense relatively

    //lets say we have mono (32 Hz)-> pair (16Hz) -> uacm (8Hz) -> pacm (4Hz) -> mono (2Hz) -> msgfwd (1Hz) -> crit
    //we want to increase the locality of dispatches for forwards - not because of the tiles - but because of the locality access of the tile_members - every time we read 1 cache_line for an element of size 8 bytes - we are utilizing the memfetch at a rate of 8/CAHCE_LINE_SIZE = 8/64 = 1/8 - which is bad
    //the moral is simple - we don't care if we are the fastest forward or backward - that's meaningless - we care about given that GPU power and CPU power - we want maximum computation thruput for all the trainings or all the forwards and backwards

    //if you skip synchronization for backwards - you are wasting at least 20x GPU powers - the exact formula for no-synchro backprop is 1 << n + 2 * (1 << (n - 1)) + 4 * (1 << (n - 2)) + (8 * (1 << (n -3))) etc. 
    //so we are talking about height * cost(synchro)
    //so when people are training 1 model - we are training concurrently HEIGHT models with the same cost and extracting the derivatives of gradients
    //I'm trying to think of the problems for this kind of training
    //it is possible for bad training data - saturate the network and the compression pipeline - but it does not happen as bad for f(g(x)) -> x
    //is it possible to remove a compression node (f(g(x) -> x)) - replace it and get a better loss_rate? - hmm - this is up for debate

    //essentially - we want raw inputs of terabytes of data (electrical signals at every pixel on Earth) - and compress terabytes of data -> gigabytes of data or even 100 MBs of data
    //we want 5000 stack of f(g(x)) -> x

    //each compression node, a.k.a. f(g(x)) takes in a fixed context size - let's say 1 TB -> 900GB -> 810GB -> down to 1GB - with the decay rate of 10% or 5% or 1% 
    //we want data decay rate of 0.1%, so 99.9% ** 5000 compression stack should yield 0.00672111195 compression rate - or 0.6% compression rate
    //intelligent/intelligence is done by reversing the process - g(x) -> x
    //g(x) can be an arbitrary context token
    //we'll get there buddys - the day of democracy and a community run by the people, for the people

    class BackwardDoLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoLeafSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr                  = get_leaf_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                init_status_t init_status               = get_leaf_init_status_nothrow(dst);
                grad_status_t grad_status               = get_leaf_grad_status_nothrow(dst);
                uma_ptr_t logit_umaptr                  = get_leaf_logit_addr_nothrow(dst);
                uma_ptr_t grad_umaptr                   = get_leaf_grad_addr_nothrow(dst);
                dispatch_control_t dispatch_control     = get_leaf_grad_dispatch_control_nothrow(dst);

                if (init_status != TILE_INIT_STATUS_INITIALIZED){
                    return; //no-ops no-err
                }

                if (grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }      
                
                auto [logit_vd_id, grad_vd_id, dp_device_kind, tileops_dp_id]   = dg::network_dispatch_control::decode_gradupdate_leaf(dispatch_control);
                auto [logit_map_resource, grad_map_resource]                    = dg::network_uma::lockmap_safewait_many_nothrow<2u>({{logit_umaptr, logit_vd_id}, {grad_umaptr, grad_vd_id}});
                auto logit_vmaptr                                               = dg::network_uma::get_vma_ptr(logit_map_resource);
                auto grad_vmaptr                                                = dg::network_uma::get_vma_ptr(grad_map_resource);
                auto logit_vmamap                                               = dg::network_vmamap::mapsafe_nothrow(logit_vmaptr);
                auto grad_vmamap                                                = dg::network_vmamap::mapsafe_nothrow(grad_vmaptr);

                //so we have expected output - which we called clogit
                //we take the difference of clogit and logit - we propagate the difference and gradients
                //then we want to update the leaf logits by using the formula new_var = old_var + lr * dy/ (dy/dvar)
                //is there a clever way to just backprop the gradients?
                //if dy is always 1 - then 1dy/dy/dvar = dvar + var = new_var
                //what's the general formula for this? I think this is where we lack -

                //so there must exist a loss function to offset the cost - such loss function is with one increase in lf - difference/ratio increase in y
                //dy/dlf = difference/ratio
                //dlf/dy = ratio/difference
                //so its just two mono transforms then - because the mono is trainable - and crit is actually immu - so we could expect a dynamic training
                //so we did crit wrong - there is no clogit - only 11111111
                //so there also must exist a logit where backprop is deactivated - such is gradients do not prop through the values

                //because what we want is this f(x) -> y
                //f(x) - immu = difference
                //u(f(x)) =  f(x) * ratio/difference (this is what we want)
                //d u(f(x)) / d f(x) = ratio/difference
                //there is no d u(f(x)) / d difference/ratio

                //-----------------
                //alrights - let's do 1/d_grad update for now
                //thing is we try to const these at some point - either it is the learning rate or the backpropagation or etc. 
                //all these recursive definitions of training needs to have a recursive base

                //a dynamic training (loss_function) is something, in my opinion, not quantifiable - and I don't know if we should move in the direction to mess up the code - but we'll try the theories
                //a dynamic training can be done - via the usage of F(f(x), g(x)) -> y + crit prop back of memset(clogit, 1, sizeof(clogit_sz))
                //a static training could be done via storing immu logits on the crit tiles - which is fine - and has a finer grain of logic-binding

                //------------------
                //the rotor and gears problem could be solved by having multiple networks - 
                //the maximum layers a training network should have is 10-15 layers each layer is uacm pacm mono pair
                //we'll move in the direction of random sticks and paths to estimate the best possible network
                //because we are training compression f(g(x)) -> x - and not f(x) -> y - so there should be no problem with the output (intermediate layer) being "unreasonable"- for the ouput of any network is the semantic of its input
                //we'll try to stack 1000 networks and compress the thing - let's see how it goes
                //------------------

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_cuda_poly::grad_update_n_zero(dg::network_vmamap::get_cuda_ptr(logit_vmamap), dg::network_vmamap::get_cuda_ptr(grad_vmamap), tileops_dp_id);
                } else if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_host_poly::grad_update_n_zero(dg::network_vmamap::get_host_ptr(logit_vmamap), dg::network_vmamap::get_host_ptr(grad_vmamap), tileops_dp_id);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

            }
    };

    class BackwardDoMonoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            BackwardDoMonoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle){

                auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_mono_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = get_mono_descendant_nothrow(dst);
                }        

                uma_ptr_t src_rcu_addr  = get_tile_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(src_rcu_addr, dst_rcu_addr); 

                uma_ptr_t new_src                   = get_mono_descendant_nothrow(dst);
                init_status_t dst_init_status       = get_mono_init_status_nothrow(dst);
                grad_status_t dst_grad_status       = get_mono_grad_status_nothrow(dst);
                operatable_id_t dst_operatable_id   = get_mono_operatable_id_nothrow(dst); //operatable_id seems like an individual identifier - use operatable_group_id instead
                uma_ptr_t dst_grad_umaptr           = get_mono_grad_addr_nothrow(dst);
                dispatch_control_t dispatch_control = get_mono_backprop_dispatch_control_nothrow(dst);
                init_status_t src_init_status       = get_tile_init_status_nothrow(src); //ptr-access
                operatable_id_t src_operatable_id   = get_tile_operatable_id_nothrow(src);
                grad_status_t src_grad_status       = get_tile_grad_status_nothrow(src);
                uma_ptr_t src_grad_umaptr           = get_tile_grad_addr_nothrow(src);
                uma_ptr_t src_logit_umaptr          = get_tile_logit_addr_nothrow(src);

                if (src != new_src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }
                
                //refactor later
                auto [backwardee_grad_vd_id, backwardee_logit_vd_id, backwarder_grad_vd_id, dp_device_kind, tileops_dp_kind]    = dg::network_dispatch_control::decode_bwd_mono(dispatch_control); //dst src is ambiguous here
                auto [backwardee_grad_map_resource, backwardee_logit_map_resource, backwarder_grad_map_resouce]                 = dg::network_uma::lockmap_safewait_many_nothrow<3u>({{src_grad_umaptr, backwardee_grad_vd_id}, {src_logit_umaptr, backwardee_logit_vd_id}, {dst_grad_umaptr, backwarder_grad_vd_id}});
                auto backwardee_grad_vmaptr     = dg::network_uma::get_vma_ptr(backwardee_grad_map_resource);
                auto backwardee_logit_vmaptr    = dg::network_uma::get_vma_ptr(backwardee_logit_map_resource);
                auto backwarder_grad_vmaptr     = dg::network_uma::get_vma_ptr(backwarder_grad_map_resouce);

                auto backwardee_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(backwardee_grad_vmaptr);
                auto backwardee_logit_vmamap    = dg::network_vmamap::mapsafe_nothrow(backwardee_logit_vmaptr);
                auto backwarder_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(backwarder_grad_vmaptr);

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_cuda_poly::bwd_mono_zero_n_add(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_kind);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_cuda_poly::bwd_mono_zero_n_assign(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_kind);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_host_poly::bwd_mono_zero_n_add(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_kind);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_host_poly::bwd_mono_zero_n_assign(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_kind);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                switch (src_grad_status){
                    case TILE_GRAD_STATUS_EMPTY:
                        set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_INITIALIZED);
                        break;
                    case TILE_GRAD_STATUS_INITIALIZED:
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

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class BackwardDoPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            BackwardDoPairSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle){

                auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_pair_rcu_addr_nothrow(ptr);
                uma_ptr_t lhs           = {};
                uma_ptr_t rhs           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    lhs = get_pair_left_descendant_nothrow(dst);
                    rhs = get_pair_right_descendant_nothrow(dst);
                }

                uma_ptr_t lhs_rcu_addr              = get_tile_rcu_addr_nothrow(lhs);
                uma_ptr_t rhs_rcu_addr              = get_tile_rcu_addr_nothrow(rhs);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, lhs_rcu_addr, rhs_rcu_addr);

                uma_ptr_t new_lhs                   = get_pair_left_descendant_nothrow(dst);
                uma_ptr_t new_rhs                   = get_pair_right_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id   = get_pair_operatable_id_nothrow(dst);
                init_status_t dst_init_status       = get_pair_init_status_nothrow(dst);

                if (lhs != new_lhs){
                    return;
                }

                if (rhs != new_rhs){
                    return;
                }

                if (lhs_operatable_id != rhs_operatable_id || lhs_operatable_id != dst_operatable_id){
                    return;
                }

                if (lhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (rhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                auto dispatch_info = dg::network_dispatch_control::decode_bwd_pair(dispatch_control); 

                {
                    // auto [backwardee_grad_vd_id, backwardee_logit_vd_id, backwarder_grad_vd_id, other_logit_vd_id, dp_device_kind. tileops_dp_kind] = dg::network_dispatch_control::decode_bwd_pair(dispatch_control);
                    auto [lhs_grad_map_resource, lhs_logit_map_resource, rhs_logit_map_resource, backwarder_grad_map_resource] = dg::network_uma::lockmap_safewait_many_nothrow<4>({{lhs_grad_umaptr, dispatch_info.lhs_grad_vd_id}, {lhs_logit_umaptr, dispatch_info.lhs_logit_vd_id}, 
                                                                                                                                                                                    {rhs_logit_umaptr, dispatch_info.rhs_logit_vd_id}, {dst_grad_umaptr, dispatch_info.backwarder_grad_vd_id}});
                    
                    auto lhs_grad_vmamap        = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(lhs_grad_map_resource));
                    auto lhs_logit_vmamap       = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(lhs_logit_map_resource));
                    auto rhs_logit_vmamap       = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(rhs_logit_map_resource));
                    auto backwarder_grad_vmamap = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resource));

                    if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.lhs_dp_device_kind)){
                        if (lhs_grad_status == TILE_GRAD_STATUS_INITIALIZED){

                        } else if (lhs_grad_status == TILE_GRAD_STATUS_EMPTY){

                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.lhs_dp_device_kind)){
                        if (lhs_grad_status == TILE_GRAD_STATUS_INITIALIZED){

                        } else if (lhs_grad_status == TILE_GRAD_STATUS_EMPTY){

                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }

                    switch (lhs_grad_status){
                        case TILE_GRAD_STATUS_EMPTY:
                            set_tile_grad_status_nothrow(lhs, TILE_GRAD_STATUS_INITIALIZED);
                            break;
                        case TILE_GRAD_STATUS_INITIALIZED:
                            break;
                        default:
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                    }

                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(lhs));
                }

                {
                    auto [rhs_grad_map_resource, rhs_logit_map_resource, lhs_logit_map_resource, backwarder_grad_map_resource] = dg::network_uma::lockmap_safewait_many_nothrow<4>({{rhs_grad_umaptr, dispatch_info.rhs_grad_vd_id}, {rhs_logit_umaptr, dispatch_info.rhs_logit_vd_id}, 
                                                                                                                                                                                    {lhs_logit_umaptr, dispatch_info.lhs_logit_vd_id}, {dst_grad_umaptr, dispatch_info.backwarder_grad_vd_id}});
                    
                    auto rhs_grad_vmamap        = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(rhs_grad_map_resource));
                    auto rhs_logit_vmamap       = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(rhs_logit_map_resource));
                    auto lhs_logit_vmamap       = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(lhs_logit_map_resource));
                    auto backwarder_grad_vmamap = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resource));

                    if (dg::network_dispatch_control::is_cuda_dispatch()){

                    } else if (dg::network_dispatch_control::is_host_dispatch()){

                    } else{
                        if constexpr(DEBUG_MODE_FLAG){

                        } else{

                        }
                    }

                    switch (rhs_grad_status){
                        case TILE_GRAD_STATUS_EMPTY:
                            set_tile_grad_status_nothrow(rhs, TILE_GRAD_STATUS_INITIALIZED);
                            break;
                        case TILE_GRAD_STATUS_INITIALIZED:
                            break;
                        default:
                            if constexpr(DEBUG_MODE_FLAG){

                            } else{

                            }
                    }

                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(rhs));
                }

            }
    };

    class BackwardDoUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

            }
    };

    class BackwardDoPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

            }
    };

    class BackwardDoExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter;
            const size_t delivery_capacity;
        
        public:

            BackwardDoExtnSrcSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                             std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter, //
                                             size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                 alias_getter(std::move(alias_getter)),
                                                                                 delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{
                
                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get());

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{
                
                //a backward do on this tile == backward do on the counterpart tile
                //we have a cyclic external tile queue - we, essentially could assume all addresses be external - yet it would impose too much overhead
                //and a fixed_size hash_map with FIFO vector
                //then we want to concurrent the hash_map by using modulo - 
                //so there is a chance where the external tile cannot be backpropped - this is equivalent to "packet_loss"

                //what do we assume?
                //we assume that counterparts external tiles are identical - we could add fail-safes along the way - by doing no-ops on err 
                //we have to add fail-safes along the way - and we must assume all inputs are corrupted
                //incompatible dispatchs + unsafe memory access + etc. will be added later
                //we'll add the assumptions next round

                auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_extnsrc_rcu_addr_nothrow(dst);
                uma_ptr_t counterpart   = {};
                uma_ptr_t src           = {}; 

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    uma_ptr_t counterpart   = get_extnsrc_counterpart_nothrow(dst);
                    uma_ptr_t src           = get_extnsrc_descendant_nothrow(dst);
                }

                std::optional<uma_ptr_t> local_counterpart = this->alias_getter->alias(counterpart);

                if (!local_counterpart.has_value()){
                    return;
                }

                uma_ptr_t local_counterpart_rcu_addr    = get_extndst_rcu_addr_nothrow(local_counterpart.value());
                uma_ptr_t src_rcu_addr                  = get_tile_rcu_addr_nothrow(src);
                dg::network_uma_memops::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr, local_counterpart_rcu_addr);

                uma_ptr_t new_src                       = get_extnsrc_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id       = get_extnsrc_operatable_id_nothrow(dst);
                init_status_t init_status               = get_extnsrc_init_status_nothrow(dst);
                uma_ptr_t new_counterpart               = get_extnsrc_counterpart_nothrow(dst); 
                dispatch_control_t dispatch_control     = get_extnsrc_dispatch_control_nothrow(dst);

                uma_ptr_t dst_grad_umaptr               = get_extndst_logit_addr_nothrow(local_counterpart.value());
                uma_ptr_t external_counterpart          = get_extndst_counterpart_nothrow(local_counterpart.value());
                uma_ptr_t external_selfaddr             = get_extndst_selfaddr_nothrow(local_counterpart.value());
                uma_ptr_t external_grad_status          = get_extndst_grad_status_nothrow(local_counterpart.value());
                uma_ptr_t src_grad_umaptr               = get_tile_logit_addr_nothrow(src);
                uma_ptr_t src_logit_umaptr              = get_tile_grad_addr_nothrow(src);
                operatable_id_t src_operatable_id       = get_tile_operatable_id_nothrow(src);

                if (new_src != src){
                    return;
                }

                if (new_counterpart != external_selfaddr){
                    return;
                }

                if (external_counterpart != dst){
                    return;
                }

                if (src_operatable_id != dst_operatable_id){
                    return;
                }

                if (dst_operatable_id != external_operatable_id){
                    return;
                }

                if (external_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (external_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                auto [backwardee_grad_vd_id, backwardee_logit_vd_id, backwarder_grad_vd_id, dp_device_kind, tileops_dp_kind]    = dg::network_dispatch_control::decode_bwd_extnsrc(dispatch_control);
                auto [backwardee_grad_map_resource, backwardee_logit_map_resource, backwarder_grad_map_resource]                = dg::network_uma::lockmap_safewait_many_nothrow<3>({{src_grad_umaptr, backwardee_grad_vd_id}, {src_logit_umaptr, backwardee_logit_vd_id}, {dst_grad_umaptr, backwarder_grad_vd_id}}); 
                
                auto backwardee_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_grad_map_resource)); 
                auto backwardee_logit_vmamap    = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_logit_map_resource));
                auto backwarder_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resource));

                if (dg::network_dispatch_control::is_cuda_dispatch()){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){

                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){

                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else if (dg::network_dispatch_control::is_host_dispatch()){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){

                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){

                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                switch (src_grad_status){
                    case TILE_GRAD_STATUS_EMPTY:
                        set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_INITIALIZED);
                        break;
                    case TILE_GRAD_STATUS_INITIALIZED:
                        break;
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                            break;
                        }
                }

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class BackwardDoExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> requst_box;
            const size_t delivery_capacity;
            std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
        
        public:

            BackwardDoExtnDstSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                             size_t delivery_capacity,
                                             std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever) noexcept: request_box(std::move(request_box)),
                                                                                                                            delivery_capacity(delivery_capacity),
                                                                                                                            uma_ip_retriever(std::move(uma_ip_retriever)){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<Request<Request<external_virtual_memory_event_t>>> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()))
                    return;
                }

                uma_ptr_t dst_rcu_addr      = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                uma_ptr_t counterpart       = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(dst);
                init_status_t init_status   = dg::network_tile_member_getsetter::get_init_status_nothrow(dst);
                grad_status_t grad_status   = dg::network_tile_member_getsetter::get_grad_status_nothrow(dst);
                dg::string serialized       = dg::network_tile_member_getsetter::serialize_extndst(dst); 

                if (init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                external_virtual_memory_event_t injection_event     = dg::network_external_memcommit_factory::make_event_foreign_injection(dst, std::move(serialized));
                external_virtual_memory_event_t signal_event        = dg::network_external_memcommit_factory::make_event_backward_do_signal(counterpart);
                external_virtual_memory_event_t event               = dg::network_external_memcommit_factory::make_sequential_event(std::move(injection_event), std::move(signal_event));
                
                Request<external_virtual_memory_event_t> request{};
                request.requestor   = this->host_ip_retriever->ip();
                request.requestee   = this->uma_ip_retriever->ip(counterpart);
                request.content     = std::move(event);

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(request));
            }
    };

    class BackwardDoCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoCritSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(src);
                }

                uma_ptr_t src_rcu_addr              = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                   = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id   = dg::network_tile_member_getsetter::get_crit_operatable_id_nothrow(dst);
                init_status_t dst_init_status       = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(dst);
                dispatch_control_t dispatch_control = dg::network_tile_member_getsetter::get_crit_dispatch_control_nothrow(dst);
                uma_ptr_t dst_grad_umaptr           = dg::network_tile_member_getsetter::get_crit_grad_addr_nothrow(dst);
                grad_status_t dst_grad_status       = dg::network_tile_member_getsetter::get_crit_grad_status_nothrow(dst);

                operatable_id_t src_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_id_nothrow(src);
                init_status_t src_init_status       = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(src);
                grad_status_t src_grad_status       = dg::network_tile_member_getsetter::get_tile_grad_status_nothrow(src);

                if (src != new_src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                auto [backwardee_grad_vd_id, backwardee_logit_vd_id, backwarder_grad_vd_id, dp_device_kind, tileops_dp_id]  = dg::network_dispatch_control::decode_bwd_crit(dispatch_control);
                auto [backwardee_grad_map_resource, backwardee_logit_map_resource, backwarder_grad_map_resource]            = dg::network_uma::lockmap_safewait_many_nothrow<3u>({{src_grad_umaptr, backwardee_grad_vd_id}, {src_logit_umaptr, backwardee_logit_vd_id}, {dst_grad_umaptr, backwarder_grad_vd_id}});
                
                auto backwardee_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_grad_map_resource));
                auto backwardee_logit_vmamap    = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_logit_map_resource));
                auto backwarder_grad_vmamap     = dg::networK_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resource));

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){

                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){

                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){

                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){

                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                switch (src_grad_status){
                    case TILE_GRAD_STATUS_INITIALIZED:
                        break;
                    case TILE_GRAD_STATUS_EMPTY:
                        break;
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                }

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class BackwardDoMsgrFwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoMsgrFwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                             size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                 delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(dst);
                }

                uma_ptr_t src_rcu_addr                  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                       = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_msgrfwd_operatable_id_nothrow(dst);
                dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_msgrfwd_dispatch_control_nothrow(dst);
                init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(dst);

                if (new_src != src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                
            }
    };

    class BackwardDoMsgrBwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoMsgrBwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                             size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                 delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

            }
    };

    class BackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> mono_resolutor;
            const size_t mono_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor;
            const size_t pair_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor;
            const size_t crit_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;

        public:

            BackwardDoSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor,
                                      size_t leaf_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> mono_resolutor,
                                      size_t mono_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor,
                                      size_t pair_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor,
                                      size_t uacm_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor,
                                      size_t pacm_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor,
                                      size_t extnsrc_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor,
                                      size_t extndst_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor,
                                      size_t crit_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor,
                                      size_t msgrfwd_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor,
                                      size_t msgrbwd_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
                                                                            leaf_dispatch_sz(leaf_dispatch_sz),
                                                                            mono_resolutor(std;:move(mono_resolutor)),
                                                                            mono_dispatch_sz(mono_dispatch_sz),
                                                                            pair_resolutor(std::move(pair_resolutor)),
                                                                            pair_dispatch_sz(pair_dispatch_sz),
                                                                            uacm_resolutor(std::move(uacm_resolutor)),
                                                                            uacm_dispatch_sz(uacm_dispatch_sz),
                                                                            pacm_resolutor(std::move(pacm_resolutor)),
                                                                            pacm_dispatch_sz(pacm_dispatch_sz),
                                                                            extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                            extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                            extndst_resolutor(std::move(extndst_resolutor)),
                                                                            extndst_dispatch_sz(extndst_dispatch_sz),
                                                                            crit_resolutor(std::move(crit_resolutor)),
                                                                            crit_dispatch_sz(crit_dispatch_sz),
                                                                            msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                            msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                            msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                            msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto leaf_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_resolutor.get(), this->leaf_dispatch_sz);
                auto mono_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->mono_resolutor.get(), this->mono_dispatch_sz);
                auto pair_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz);
                auto uacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz);
                auto pacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz);
                auto extnsrc_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz);
                auto extndst_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz);
                auto crit_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz);
                auto msgrfwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz);
                auto msgrbwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz);

                if (!dg::network_exception::conjunc_expect_has_value(leaf_delivery_handle, mono_delivery_handle, pair_delivery_handle, 
                                                                     uacm_delivery_handle, pacm_delivery_handle, extnsrc_delivery_handle, 
                                                                     extndst_delivery_handle, crit_delivery_handle, msgrfwd_delivery_handle, 
                                                                     msgrbwd_delivery_handle)){

                    dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(ptr_arr[i]);

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(leaf_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MONO:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(mono_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PAIR:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pair_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_UACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(uacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNSRC:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extnsrc_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNDST:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extndst_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_CRIT:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(crit_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRFWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrfwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRBWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrbwd_delivery_handle)->get(), ptr_arr[i]);
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
    };

    //

    class MemCommitResolutor: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>> producer;
            const size_t producer_consume_capacity;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_ping_signal_resolutor;
            const size_t fwd_ping_delivery_capacity; 
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pong_request_resolutor;
            const size_t fwd_pong_request_delivery_capacity;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_pong_signal_resolutor;
            const size_t fwd_pong_signal_delivery_capacity;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_do_resolutor;
            const size_t fwd_do_delivery_capacity;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> bwd_do_resolutor;
            const size_t bwd_do_delivery_capacity;

        public:

            MemCommitResolutor(std::shared_ptr<dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>> producer,
                               size_t producer_consume_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_ping_signal_resolutor,
                               size_t fwd_ping_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pong_request_resolutor,
                               size_t fwd_pong_request_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_pong_signal_resolutor,
                               size_t fwd_pong_signal_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_do_resolutor,
                               size_t fwd_do_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> bwd_do_resolutor,
                               size_t bwd_do_delivery_capacity) noexcept: producer(std::move(producer)),
                                                                          producer_consume_capacity(producer_consume_capacity),
                                                                          fwd_ping_signal_resolutor(std::move(fwd_ping_signal_resolutor)),
                                                                          fwd_ping_delivery_capacity(fwd_ping_delivery_capacity),
                                                                          fwd_pong_request_resolutor(std::move(fwd_pong_request_resolutor)),
                                                                          fwd_pong_request_delivery_capacity(fwd_pong_request_delivery_capacity),
                                                                          fwd_pong_signal_resolutor(std::move(fwd_pong_signal_resolutor)),
                                                                          fwd_pong_signal_delivery_capacity(fwd_pong_signal_delivery_capacity),
                                                                          fwd_do_resolutor(std::move(fwd_do_resolutor)),
                                                                          fwd_do_delivery_capacity(fwd_do_delivery_capacity),
                                                                          bwd_do_resolutor(std::move(bwd_do_resolutor)),
                                                                          bwd_do_delivery_capacity(bwd_do_delivery_capacity){}

            bool run_one_epoch() noexcept{

                auto virtual_memory_event_arr   = std::make_unique<virtual_memory_event_t>(this->producer_consume_capacity); //TODOs: internalize allocations
                size_t virtual_memory_event_sz  = {};
                this->producer->get(virtual_memory_event_arr.get(), virtual_memory_event_sz, this->producer_consume_capacity);

                if (virtual_memory_event_sz == 0u){
                    return false;
                }

                //refactor - this logic is transform consumers

                auto fwd_ping_signal_resolution_array   = std::make_unique<uma_ptr_t[]>(this->fwd_ping_delivery_capacity);
                auto fwd_pong_request_resolution_array  = std::make_unique<std::tuple<uma_ptr_t, uma_ptr_t>[]>(this->fwd_pong_request_delivery_capacity);
                auto fwd_pong_signal_resolution_array   = std::make_unique<uma_ptr_t[]>(this->fwd_pong_signal_delivery_capacity);
                auto fwd_do_resolution_array            = std::make_unique<uma_ptr_t[]>(this->fwd_do_delivery_capacity);
                auto bwd_do_resolution_array            = std::make_unique<uma_ptr_t[]>(this->bwd_do_delivery_capacity);

                auto fwd_ping_signal_lambda_consumer    = [this, &](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    for (size_t i = 0u; i < arr_sz; ++i){
                        fwd_ping_signal_resolution_array[i] = dg::network_memcommit_factory::read_event_forward_ping_signal(event_arr[i]);
                    }

                    this->fwd_ping_signal_resolutor->push(fwd_ping_signal_resolution_array.get(), arr_sz);
                };

                auto fwd_pong_request_lambda_consumer   = [this, &](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    for (size_t i = 0u; i < arr_sz; ++i){
                        fwd_pong_request_resolution_array[i] = dg::network_memcommit_factory::read_event_forward_pong_request(event_arr[i]);
                    }

                    this->fwd_pong_request_resolutor->push(fwd_pong_request_resolution_array.get(), arr_sz);
                };

                auto fwd_pong_signal_lambda_consumer    = [this, &](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    for (size_t i = 0u; i < arr_sz; ++i){
                        fwd_pong_signal_resolution_array[i] = dg::network_memcommit_factory::read_event_forward_pong_signal(event_arr[i]);
                    }
                    
                    this->fwd_pong_signal_resolutor->push(fwd_pong_signal_resolution_array.get(), arr_sz);
                };

                auto fwd_do_lambda_consumer             = [this, &](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    for (size_t i = 0u; i < arr_sz; ++i){
                        fwd_do_resolution_array[i] = dg::network_memcommit_factory::read_event_forward_do_signal(event_arr[i]);
                    }

                    this->fwd_do_resolutor->push(fwd_do_resolution_array.get(), arr_sz);
                };

                auto bwd_do_lambda_consumer             = [this, &](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    for (size_t i = 0u; i < arr_sz; ++i){
                        bwd_do_resolution_array[i] = dg::network_memcommit_factory::read_event_backward_do_signal(event_arr[i]);
                    }

                    this->bwd_do_resolutor->push(bwd_do_resolution_array.get(), arr_sz);
                };

                auto fwd_ping_signal_virtual_consumer   = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_ping_signal_lambda_consumer)>(fwd_ping_signal_lambda_consumer);
                auto fwd_pong_request_virtual_consumer  = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_pong_request_lambda_consumer)>(fwd_pong_request_lambda_consumer);
                auto fwd_pong_signal_virtual_consumer   = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_pong_signal_lambda_consumer)>(fwd_pong_signal_lambda_consumer);
                auto fwd_do_virtual_consumer            = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_do_lambda_consumer)>(fwd_do_lambda_consumer);
                auto bwd_do_virtual_consumer            = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(bwd_do_lambda_consumer)>(bwd_do_lambda_consumer);

                stdx::seq_cst_guard seqcst_guard;

                auto fwd_ping_signal_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_ping_signal_virtual_consumer, this->fwd_ping_delivery_capacity);
                auto fwd_pong_request_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_pong_request_virtual_consumer, this->fwd_pong_request_delivery_capacity);
                auto fwd_pong_signal_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_pong_signal_virtual_consumer, this->fwd_pong_signal_delivery_capacity);
                auto fwd_do_delivery_handle             = dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_do_virtual_consumer, this->fwd_do_delivery_capacity);
                auto bwd_do_delivery_handle             = dg::network_producer_consumer::delvrsrv_open_raiihandle(&bwd_do_virtual_consumer, this->bwd_do_delivery_capacity);

                if (!dg::network_exception::conjunc_expect_has_value(fwd_ping_signal_delivery_handle, fwd_pong_request_delivery_handle,
                                                                     fwd_pong_signal_delivery_handle, fwd_do_delivery_handle,
                                                                     bwd_do_delivery_handle)){

                    dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    return false;
                }

                for (size_t i = 0u; i < virtual_memory_event_sz; ++i){
                    memory_event_kind_t event_kind = dg::network_memcommit_factory::read_event_kind(virtual_memory_event_arr[i]);

                    switch (event_kind){
                        case dg::network_memcommit_factory::event_kind_forward_ping_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_ping_signal_delivery_handle)->get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_forward_pong_request:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_pong_request_delivery_handle)->get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_forward_pong_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_pong_signal_delivery_handle)->get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_forward_do_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_do_delivery_handle)->get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_backward_do_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(bwd_do_delivery_handle)->get(), std::move(virtual_memory_event_arr[i]));
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

                return true;
            }
    };
}

#endif