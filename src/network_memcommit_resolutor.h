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

    struct ExternalAddressAliasGetterInterface{ //
        virtual ~ExternalAddressAliasGetterInterface() noexcept = default;
        virtual auto alias(uma_ptr_t) noexcept -> std::optional<uma_ptr_t> = 0; //change semantics
    };

    template <class T>
    struct Request{
        Address requestee;
        Address requestor;
        T content;
    };

    //alrights - we'll get these filled tmr - 

    class ForwardPingLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPingLeafSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(ptr);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(ptr);

                switch (init_status){
                    case TILE_INIT_STATUS_ORPHANED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        std::array<uma_ptr_t, MAX_DESCENDANT_SIZE> descendants{};
                        size_t descendant_size{};
                        std::tie(descendants, descendant_size) = dg::network_tile_member_getsetter::get_tile_descendants_nothrow(ptr); 

                        for (size_t i = 0u; i < descendant_size; ++i){
                            virtual_memory_event_t ping_request = dg::network_memcommit_factory::make_event_forward_ping_request(descendants[i], ptr);
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(ping_request));
                            virtual_memory_event_t pong_request = dg::network_memcommit_factory::make_event_forward_pong_request(descendants[i], ptr);
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(pong_request));
                        }

                        dg::network_tile_member_getsetter::set_initialization_status_nothrow(ptr, INIT_STATUS_DECAYED);
                        break;
                    }
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        break;
                    }
                    default:
                    {
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

    class ForwardPingPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class ForwardPingUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class ForwardPingPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class ForwardPingExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::unique_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            std::unique_ptr<HostIPRetrieverInterface> host_ip_retriever;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            const size_t delivery_capacity;

        public:

            DstExternalForwardPingSignalResolutor(std::unique_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                  std::unique_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                  std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<external_virtual_memory_event_t>> request_box,
                                                  size_t delivery_capacity): uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                             host_ip_retriever(std::move(host_ip_retriever)),
                                                                             request_box(std::move(request_box)),
                                                                             delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t requestee, dg::network_producer_consumer::DeliveryHandle<external_virtual_memory_event_t> * handle) noexcept{

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_dstextclone_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_guard(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_dstextclone_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_ORPHANED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t requestor     = dg::network_tile_member_getsetter::get_dstextclone_src_addr_nothrow(requestee);
                        Address requestor_ip    = this->uma_ip_retriever->ip(src_addr);
                        Address requestee_ip    = this->host_ip_retriever->ip();
                        Request<external_virtual_memory_event_t> ping_request{requestor_ip, requestee_ip, dg::network_external_memcommit_factory::make_event_external_forward_ping_signal(requestor)};
                        Request<external_virtual_memory_event_t> pong_request{requestor_ip, requestee_ip, dg::network_external_memcommit_factory::make_event_external_forward_pong_request(requestor, requestee)};
                        dg::network_producer_conumser::delvrsrv_deliver(handle, std::move(ping_request));
                        dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(pong_request));
                        dg::network_tile_member_getsetter::set_dstextclone_init_status_nothrow(requestee, INIT_STATUS_DECAYED);
                        break;
                    }
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        break;
                    }
                    default:
                    {
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

    class ForwardPingExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };
    
    class ForwardPingCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class ForwardPingMsgrFwdResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class ForwardPingMsgrBwdResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

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

            ForwardPingSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> ordinary_ping_resolutor,
                                       size_t ordinary_ping_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> dst_external_ping_resolutor,
                                       size_t dst_external_ping_dispatch_sz) noexcept: ordinary_ping_resolutor(std::move(ordinary_ping_resolutor)),
                                                                                       ordinary_ping_dispatch_sz(ordinary_ping_dispatch_sz),
                                                                                       dst_external_ping_resolutor(std::move(dst_external_ping_resolutor)),
                                                                                       dst_external_ping_dispatch_sz(dst_external_ping_dispatch_sz){}

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
                    this->internal_resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(requestee_rcu_addr);

                switch (init_status){
                    case TILE_INIT_STATUS_ORPHANED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_ADOPTED:
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
                    {
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
                    this->internal_resolve(ptr_arr[i], delivery_handle.get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t signalee, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                uma_ptr_t signalee_rcu_addr         = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(signalee);
                dg::network_memops_uma::memlock_guard mem_grd(signalee_rcu_addr);
                init_status_t init_status           = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(signalee);

                switch (init_status){
                    case TILE_INIT_STATUS_ORPHANED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        break;
                    }
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
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        break;
                    }
                    default:
                    {
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

    class ForwardInitMonoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class ForwardInitPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class ForwardInitUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class ForwardInitPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerIntterface<uma_ptr_t>{

    };

    class ForwardInitExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            std::unique_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            std::unique_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const size_t delivery_capacity;

        public:

            SrcExternalForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
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
                    this->internal_resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                bool forward_status = dg::network_tileops_host_handler::forward_srcextclone(ptr);

                if (!forward_status){
                    return;
                }

                uma_ptr_t ptr_rcu_addr      = dg::network_tile_member_getsetter::get_srcextclone_rcu_addr_nothrow(ptr);
                init_status_t init_status   = {};
                uma_ptr_t dst_addr          = {}; 

                {
                    dg::network_memops_uma::memlock_guard mem_grd(ptr_rcu_addr);
                    init_status = dg::network_tile_member_getsetter::get_srcexclone_init_status_nothrow(ptr);

                    if (init_status != TILE_INIT_STATUS_INITIALIZED){
                        return;
                    }

                    dst_addr = dg::network_tile_member_getsetter::get_srcexclone_dst_addr_nothrow(ptr);
                    //
                }
            }
    };

    class ForwardInitExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            std::shared_ptr<ExternalAddressAliasGetterInterface> alias_getter;
            const size_t delivery_capacity;

        public:

            DstExternalForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                  std::shared_ptr<ExternalAddressAliasGetterInterface> alias_getter,
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
                    this->internal_resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                std::optional<uma_ptr_t> alias = this->alias_getter->alias(ptr);
                
                if (!alias.has_value()){
                    return;
                }

                bool forward_status = dg::network_tileops_host_handler::forward_dstextclone(ptr, alias.value());

                if (!forward_status){
                    return;
                }

                uma_ptr_t ptr_lock_addr                                 = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(ptr);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = {};
                size_t observer_arr_sz                                  = {};
                init_status_t init_status                               = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(ptr_lock_addr);
                    init_status = dg::network_tile_member_getsetter::get_dstextclone_init_status_nothrow(ptr);

                    if (init_status != TILE_INIT_STATUS_INITIALIZED){
                        return;
                    }

                    observer_arr    = dg::network_tile_member_getsetter::get_dstextclone_observer_nothrow(ptr);
                    observer_arr_sz = dg::network_tile_member_getsetter::get_dstextclone_observer_array_size_nothrow(ptr);

                    for (size_t i = 0u; i < observer_arr_sz; ++i){
                        virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i]);
                        dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
                    }
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
                    this->internal_resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                bool forward_status = dg::network_tileops_host_handler::forward_crit(ptr);

                if (!forward_status){
                    return;
                }

                uma_ptr_t ptr_lock_addr                                 = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(ptr);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = {};
                size_t observer_arr_sz                                  = {};
                init_status_t init_status                               = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(ptr_lock_addr);
                    init_status = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(ptr);

                    if (init_status != TILE_INIT_STATUS_INITIALIZED){
                        return;
                    }

                    observer_arr        = dg::network_tile_member_getsetter::get_crit_observer_nothrow(ptr);
                    observer_arr_sz     = dg::network_tile_member_getsetter::get_crit_observer_array_size_nothrow(ptr);

                    for (size_t i = 0u; i < observer_arr_sz; ++i){
                        virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i]);
                        dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
                    }

                    virtual_memory_event_t request = dg::network_memcommit_factory::make_event_backward_do_signal(ptr);
                    dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
                }
            }
    };

    class ForwardInitMsgrFwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            MsgrFwdForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                bool forward_status = dg::network_tileops_host_handler::forward_msgrfwd(ptr);

                if (!forward_status){
                    return;
                }

                uma_ptr_t ptr_lock_addr                                 = dg::network_tile_member_get_setter::get_tile_rcu_addr_nothrow(ptr);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = {};
                size_t observer_arr_sz                                  = {};
                init_status_t init_status                               = {};
                dst_info_t dst_info                                     = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(ptr_lock_addr);
                    init_status     = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(ptr);

                    if (init_status != TILE_INIT_STATUS_INITIALIZED){
                        return;
                    }

                    observer_arr    = dg::network_tile_member_getsetter::get_msgrfwd_observer_nothrow(ptr);
                    observer_arr_sz = dg::network_tile_member_getsetter::get_msgrfwd_observer_array_size_nothrow(ptr);
                    dst_info        = dg::network_tile_member_getsetter::get_msgrfwd_dst_info_nothrow(ptr);

                    for (size_t i = 0u; i < observer_arr_sz; ++i){
                        virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i]);
                        dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
                    }


                } 
            }
    };

    class ForwardInitMsgrBwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

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

    class BackwardDoLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        // private:    

        //     std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
        //     const size_t delivery_capacity;
        
        // public:

        //     LeafBackwardDoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
        //                                   size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
        //                                                                       delivery_capacity(delivery_capacity){}
            
        //     void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{
                
        //         for (size_t i = 0u; i < sz; ++i){
        //             dg::network_tileops_handler::backward_leaf(ptr_arr[i]); 
        //         }
        //     }
    };

    class BackwardDoMonoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class BackwardDoPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class BackwardDoUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class BackwardDoPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class BackwardDoExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        // private:

        //     std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
        //     std::shared_ptr<ExternalAddressAliasGetterInterface> alias_getter;
        //     const size_t delivery_capacity;

        // public:

        //     SrcExternalBackwardDoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
        //                                          std::shared_ptr<ExternalAddressAliasGetterInterface> alias_getter,
        //                                          const size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
        //                                                                                    alias_getter(std::move(alias_getter)),
        //                                                                                    delivery_capacity(delivery_capacity){}

        //     void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

        //         auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

        //         if (!delivery_handle.has_value()){
        //             dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
        //             return;
        //         }

        //         for (size_t i = 0u; i < sz; ++i){
        //             this->internal_resolve(ptr_arr[i], delivery_handle->get());
        //         }
        //     }

        // private:

        //     void internal_resolve(uma_ptr_t backwarder, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

        //         std::optional<uma_ptr_t> alias = this->alias_getter->alias(backwarder);

        //         if (!alias.has_value()){
        //             return;
        //         }

        //         bool backward_status = dg::network_tileops_host_handler::backward(backwarder, alias.value());

        //         if (!backward_status){
        //             return;
        //         }

        //         uma_ptr_t backwarder_rcu_addr                               = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(backwarder);
        //         std::array<uma_ptr_t, MAX_DESCENDANT_SIZE> descendant_arr   = {};
        //         size_t descendant_arr_sz                                    = {};
        //         init_status_t init_status                                   = {};

        //         {
        //             dg::network_memops_uma::memlock_guard mem_grd(backwarder_rcu_addr);
        //             init_status = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(backwarder);

        //             switch (init_status){
        //                 case TILE_INIT_STATUS_ORPHANED:
        //                     break;
        //                 case TILE_INIT_STATUS_ADOPTED:
        //                     break;
        //                 case ITLE_INIT_STATUS_DECAYED:
        //                     break;
        //                 case TILE_INIT_STATUS_INITIALIZED:
        //                     descendant_arr      = dg::network_tile_member_getsetter::get_srcextclone_descendant_nothrow(backwarder);
        //                     descendant_arr_sz   = dg::network_tile_member_getsetter::get_srcextclone_descendant_size_nothrow(backwarder);
                            
        //                     for (size_t i = 0u; i < descendant_arr_sz; ++i){
        //                         virtual_memory_event_t request = dg::network_memcommit_factory::make_event_backward_do_request(descendant_arr[i]);
        //                         dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
        //                     }
        //                     break;
        //                 default:
        //                     if constexpr(DEBUG_MODE_FLAG){
        //                         dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        //                         std::abort();
        //                         break;
        //                     } else{
        //                         std::unreachable();
        //                         break;
        //                     }
        //             }
        //         }
        //     }
    };

    class BackwardDoExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        // private:

        //     std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<virtual_memory_event_t>>> request_box;
        //     const size_t delivery_capacity;
        //     std::shared_ptr<UnifiedMemoryIPRetrieverInterface>  uma_ip_retriever;
        
        // public:

        //     DstExternalBackwardDoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<virtual_memory_event_t>>> request_box,
        //                                          size_t delivery_capacity,
        //                                          std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever) noexcept: request_box(std::move(request_box)),
        //                                                                                                                         delivery_capacity(delivery_capacity),
        //                                                                                                                         uma_ip_retriever(std::move(uma_ip_retriever)){}
            

        //     void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

        //         auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

        //         if (!delivery_handle.has_value()){
        //             dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
        //             return;
        //         }

        //         for (size_t i = 0u; i < sz; ++i){
        //             this->internal_resolve(ptr_arr[i], delivery_handle->get());
        //         }
        //     }

        // private:

        //     void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * handle) noexcept{

        //     }
    };

    class BackwardDoCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class BackwardDoMsgrFwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class BackwardDoMsgrBwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

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