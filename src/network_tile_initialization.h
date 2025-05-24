#ifndef __DG_NETWORK_TILE_LIFETIME_H__
#define __DG_NETWORK_TILE_LIFETIME_H__

#include <stdint.h>
#include <stddef.h>
#include "network_exception.h" 
#include "network_exception_handler.h"
#include "network_uma.h"
#include "network_tile_member_getsetter.h"
#include "network_memops_uma.h"
#include "network_tile_member_access.h"
#include "stdx.h"
#include "network_pointer.h"

//alright, we'll try to get in the coding mode to write all of these within 1 week

//bijective mapping function f(x) -> y
//semantic space
//logit density mining
//logit mining patterns

namespace dg::network_tile_lifetime::concurrent_unsafe{

    using uma_ptr_t                 = dg::network_pointer::uma_ptr_t; 
    using group_operatable_id_t     = dg::network_tile_metadata::group_operatable_id_t; //sounds weird but it should be group_operatable_id_t not operatable_group_id_t - we want topology of things here
    using dispatch_control_t        = dg::network_tile_metadata::dispatch_control_t;
    using crit_kind_t               = dg::network_tile_metadata::crit_kind_t;
    using ClientDeliveryInfo        = dg::network_tile_metadata::ClientDeliveryInfo;
    using timein_t                  = dg::network_tile_metadata::timein_t;

    static inline constexpr UACM_ACM_SZ = dg::network_tile_metadata::UACM_ACM_SZ;
    static inline constexpr PACM_ACM_SZ = dg::network_tile_metadata::PACM_ACM_SZ;

    //we need three operatable ids to fence off the other users, we'll get the exclusivity of accesses via these three tickets
    //for reusages of leaf nodes, we'll need to orphan the upper nodes to vodify the signals
    //before we init_... again
    //we have settled for the design of optional<uma_ptr_t> as an accumulated signal tile, which is frequencized on the mempress, the otherwises are delivered directly to the dispatch_warehose

    //I'm tired of having to explain to my Mom about that differential is probably not the way, but TWO SUM problem is the classical physical problem
    //there is no way to make another variable out of thin air, without intercoursing two existing variables
    //if there are problems, the problem is the tile being not FAT enough
    //now think about the differential problem is probably not the way again
    //this is an incorrect statement
    //this is because our logit density miner + Taylor Series patterns database aren't rich enough

    auto init_leaf(uma_ptr_t ptr,
                   operatable_id_t forward_ops_id,
                   operatable_id_t backward_ops_id,
                   operatable_id_t memevent_ops_id,
                   const void * logit_value, uint64_t logit_value_sz,
                   bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_leaf_logit(logit_value, logit_value_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        //we'll need to rid of the nothrows, we are getting greedy, yet we are breaking the atomicity contract
        //which we dont really care about 
        //we'll see about that later ...
        //because a tile in a wrong state is VERY DANGEROUS, we are assuming atomicity of tiles to operate on literally EVERYTHING

        dg::network_tile_member_getsetter::set_leaf_operatable_memevent_id_nothrow(ptr, memevent_ops_id);
        dg::network_tile_member_getsetter::set_leaf_operatable_forward_id_nothrow(ptr, forward_ops_id);
        dg::network_tile_member_getsetter::set_leaf_operatable_backward_id_nothrow(ptr, backward_ops_id);
        dg::network_tile_member_getsetter::set_leaf_logit_nothrow(ptr, logit_value, logit_value_sz);
        dg::network_tile_member_getsetter::set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_INITIALIZED);
        dg::network_tile_member_getsetter::set_leaf_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_blkr(uma_ptr_t ptr,
                   uma_ptr_t src,
                   operatable_id_t forward_ops_id,
                   operatable_id_t backward_ops_id,
                   operatable_id_t memevent_ops_id, 
                   std::optional<uma_ptr_t> signal_accum_addr,
                   dispatch_control_t dispatch_control,
                   const ObserverData * observer_arr, uint64_t observer_arr_sz,
                   bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_blkr_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_blkr_descendant(src); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_blkr_dispatch_control(dispatch_control); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_blkr_observer_array(observer_arr, observer_arr_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_blkr_descendant_nothrow(ptr, src);
        dg::network_tile_member_getsetter::set_blkr_operatable_memevent_id_nothrow(ptr, memevent_ops_id);
        dg::network_tile_member_getsetter::set_blkr_operatable_forward_id_nothrow(ptr, forward_ops_id);
        dg::network_tile_member_getsetter::set_blkr_operatable_backward_id_nothrow(ptr, backward_ops_id);
        dg::network_tile_member_getsetter::set_blkr_signal_smph_addr_nothrow(ptr, signal_accum_addr);
        dg::network_tile_member_getsetter::set_blkr_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_blkr_observer_array_nothrow(ptr, observer_arr, observer_arr_sz);
        dg::network_tile_member_getsetter::set_blkr_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);

        return dg::network_exception::SUCCESS;
    }

    auto init_mono(uma_ptr_t ptr,
                   uma_ptr_t src,
                   operatable_id_t forward_ops_id,
                   operatable_id_t backward_ops_id,
                   operatable_id_t memevent_ops_id,
                   std::optional<uma_ptr_t> signal_accum_addr,
                   dispatch_control_t dispatch_control, 
                   const uma_ptr_t * observer_arr, uint64_t observer_arr_sz,
                   bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_mono_descendant(src); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_mono_dispatch_control(dispatch_control); dg::network_exception::is_failed(err)){ 
            return err;
        }

        if (exception_t err = check_mono_observer_array(observer_arr, observer_arr_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_mono_descendant_nothrow(ptr, src);
        dg::network_tile_member_getsetter::set_mono_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_mono_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_mono_observer_array_nothrow(ptr, observer_arr, observer_arr_sz);
        dg::network_tile_member_getsetter::set_mono_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        dg::network_tile_member_getsetter::set_mono_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        dg::network_tile_member_getsetter::set_mono_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    //the pong_sz is very bad for external interfaces, we'd rather having an observer array on the leafs
    //we'll work on "covering" the leafs, because of a lot of reasons, we can't really expose the external interface of pingpong low levels, observer, notifying addr is OK

    auto init_pair(uma_ptr_t ptr,
                   uma_ptr_t lhs, uma_ptr_t rhs,
                   operatable_id_t forward_ops_id,
                   operatable_id_t backward_ops_id,
                   operatable_id_t memevent_ops_id,
                   std::optional<uma_ptr_t> signal_accum_addr, 
                   dispatch_control_t dispatch_control,
                   const ObserverData * observer_arr, uint64_t observer_arr_sz,
                   bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_pair_descendant(lhs, rhs); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_pair_dispatch_control(dispatch_control); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_pair_observer_array(observer_arr, observer_arr_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_pair_pong_count(pong_count); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_pair_left_descendant_nothrow(ptr, lhs);
        dg::network_tile_member_getsetter::set_pair_right_descendant_nothrow(ptr, rhs);
        dg::network_tile_member_getsetter::set_pair_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_pair_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_pair_observer_array_nothrow(ptr, observer_arr, observer_arr_sz);
        dg::network_tile_member_getsetter::set_pair_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        dg::network_tile_member_getsetter::set_pair_pong_count_nothrow(ptr, pong_count);
        dg::network_tile_member_getsetter::set_pair_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_uacm(uma_ptr_t ptr, 
                   const uma_ptr_t * src_arr, uint64_t src_arr_sz,
                   operatable_id_t forward_ops_id,
                   operatable_id_t backward_ops_id,
                   operatable_id_t memevent_ops_id,
                   std::optional<uma_ptr_t> signal_accum_addr,
                   dispatch_control_t dispatch_control,
                   const ObserverData * observer_arr, uint64_t observer_arr_sz,
                   bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_uacm_descendant(src_arr, src_arr_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_uacm_dispatch_control(dispatch_control); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_uacm_observer_array(observer_arr, observer_arr_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_uacm_pong_count(pong_count); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_uacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        dg::network_tile_member_getsetter::set_uacm_observer_array_size_nothrow(ptr, 0u);
        dg::network_tile_member_getsetter::set_uacm_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_uacm_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_uacm_pong_count_nothrow(ptr, pong_count);
        dg::network_tile_member_getsetter::set_uacm_descendant_nothrow(ptr, src);
        dg::network_tile_member_getsetter::set_uacm_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_pacm(uma_ptr_t ptr,
                   const uma_ptr_t * lhs_arr, const uma_ptr_t * rhs_arr, uint64_t acm_sz, 
                   operatable_id_t forward_ops_id,
                   operatable_id_t backward_ops_id,
                   operatable_id_t memevent_ops_id,
                   std::optional<uma_ptr_t> signal_accum_addr,
                   dispatch_control_t dispatch_control, 
                   const ObserverData * observer_arr, uint64_t observer_arr_sz,
                   bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_pacm_descendant(lhs, rhs, acm_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_pacm_dispatch_control(dispatch_control); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_pacm_observer_array(observer_arr, observer_arr_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_pacm_pong_count(pong_count); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_pacm_left_descendant_nothrow(ptr, lhs, acm_sz);
        dg::network_tile_member_getsetter::set_pacm_right_descendant_nothrow(ptr, rhs, acm_sz);
        dg::network_tile_member_getsetter::set_pacm_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_pacm_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_pacm_observer_array_nothrow(ptr, observer_arr, observer_arr_sz);
        dg::network_tile_member_getsetter::set_pacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        dg::network_tile_member_getsetter::set_pacm_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        dg::network_tile_member_getsetter::set_pacm_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_crit(uma_ptr_t ptr,
                   uma_ptr_t src,
                   operatable_id_t forward_ops_id,
                   operatable_id_t backward_ops_id,
                   operatable_id_t memevent_ops_id,
                   std::optional<uma_ptr_t> signal_accum_addr,
                   dispatch_control_t dispatch_control,
                   learning_rate_t learning_rate,
                   const void * clogit_value, uint64_t clogit_value_sz,
                   const ObserverData * observer_arr, uint64_t observer_arr_sz,
                   bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_crit_descendant(src); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_crit_dispatch_control(dispatch_control); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_crit_learning_rate(learning_rate); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_crit_clogit(clogit_value, clogit_value_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_crit_observer_array(observer_arr, observer_arr_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_crit_descendant_nothrow(ptr, src);
        dg::network_tile_member_getsetter::set_crit_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_crit_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_crit_learning_rate_nothrow(ptr, learning_rate);
        dg::network_tile_member_getsetter::set_crit_clogit_nothrow(ptr, clogit_value, clogit_value_sz);
        dg::network_tile_member_getsetter::set_crit_observer_array_nothrow(ptr, observer_arr, observer_arr_sz);
        dg::network_tile_member_getsetter::set_crit_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        dg::network_tile_member_getsetter::set_crit_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        dg::network_tile_member_getsetter::set_crit_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_immu(uma_ptr_t ptr,
                   operatable_id_t forward_ops_id,
                   operatable_id_t backward_ops_id,
                   operatable_id_t memevent_ops_id,
                   const void * logit_value, uint64_t logit_value_sz,
                   bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_immu_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_immu_logit(logit_value, logit_value_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_immu_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_INITIALIZED);
        dg::network_tile_member_getsetter::set_immu_logit_nothrow(ptr, logit_value, logit_value_sz);
        dg::network_tile_member_getsetter::set_immu_group_operatable_id_nothrow(ptr, group_operatable_id);

        return dg::network_exception::SUCCESS;
    }

    auto init_msgrfwd(uma_ptr_t ptr,
                      uma_ptr_t src,
                      operatable_id_t forward_ops_id,
                      operatable_id_t backward_ops_id,
                      operatable_id_t memevent_ops_id,
                      std::optional<uma_ptr_t> signal_accum_addr,
                      dispatch_control_t dispatch_control,
                      ClientDeliveryInfo dst_info,
                      const ObserverData * observer_arr, uint64_t observer_arr_sz,
                      bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_msgrfwd_descendant(src); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_msgrfwd_dispatch_control(dispatch_control); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_msgrfwd_dst_info(dst_info); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_msgrfwd_observer_array(observer_arr, observer_arr_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        dg::network_tile_member_getsetter::set_msgrfwd_observer_array_size_nothrow(ptr, 0u);
        dg::network_tile_member_getsetter::set_msgrfwd_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_msgrfwd_operatable_id_nothrow(ptr, operatable_id);
        dg::network_tile_member_getsetter::set_msgrfwd_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_msgrfwd_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        dg::network_tile_member_getsetter::set_msgrfwd_descendant_nothrow(ptr, src);
        dg::network_tile_member_getsetter::set_msgrfwd_dst_info_nothrow(ptr, dst_info);
        dg::network_tile_member_getsetter::set_msgrfwd_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_msgrbwd(uma_ptr_t ptr,
                      uma_ptr_t src,
                      operatable_id_t forward_ops_id,
                      operatable_id_t backward_ops_id,
                      operatable_id_t memevent_ops_id,
                      std::optional<uma_ptr_t> signal_accum_addr,
                      dispatch_control_t dispatch_control,
                      ClientDeliveryInfo dst_info, 
                      const ObserverData * observer_arr, uint64_t observer_arr_sz,
                      bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_msgrbwd_descendant(src); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_msgrbwd_dispatch_control(dispatch_control); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_msgrbwd_timein(timein); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_msgrbwd_observer_array(observer_arr, observer_arr_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_msgrbwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        dg::network_tile_member_getsetter::set_msgrbwd_observer_array_size_nothrow(ptr, 0u);
        dg::network_tile_member_getsetter::set_msgrbwd_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_msgrbwd_operatable_id_nothrow(ptr, operatable_id);
        dg::network_tile_member_getsetter::set_msgrbwd_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_msgrbwd_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        dg::network_tile_member_getsetter::set_msgrbwd_descendant_nothrow(ptr, src);
        dg::network_tile_member_getsetter::set_msgrbwd_dst_info_nothrow(ptr, dst_info);
        dg::network_tile_member_getsetter::set_msgrbwd_timein_nothrow(ptr, timein);
        dg::network_tile_member_getsetter::set_msgrbwd_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_extnsrc(uma_ptr_t ptr,
                      uma_ptr_t src,
                      uma_ptr_t counterpart,
                      uma_ptr_t counterpart_shadow,
                      operatable_id_t forward_ops_id,
                      operatable_id_t backward_ops_id,
                      operatable_id_t memevent_ops_id,
                      std::optional<uma_ptr_t> signal_accum_addr,
                      dispatch_control_t dispatch_control,
                      const ObserverData * observer_arr, uint64_t observer_arr_sz,
                      bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_extnsrc_descendant(src); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_extnsrc_counterpart(src); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_extnsrc_dispatch_control(dispatch_control); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        dg::network_tile_member_getsetter::set_extnsrc_observer_array_size_nothrow(ptr, 0u);
        dg::network_tile_member_getsetter::set_extnsrc_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_extnsrc_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_extnsrc_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        dg::network_tile_member_getsetter::set_extnsrc_descendant_nothrow(ptr, src);
        dg::network_tile_member_getsetter::set_extnsrc_counterpart_nothrow(ptr, counterpart);
        dg::network_tile_member_getsetter::set_extnsrc_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_extnsrx(uma_ptr_t ptr,
                      operatable_id_t memevent_ops_id,
                      std::optional<uma_ptr_t> signal_accum_addr,
                      bool force_init_flag) noexcept -> exception_t{

    }

    auto init_extndst(uma_ptr_t ptr,
                      uma_ptr_t src,
                      uma_ptr_t counterpart,
                      uma_ptr_t counterpart_shadow,
                      operatable_id_t forward_ops_id,
                      operatable_id_t backward_ops_id,
                      operatable_id_t memevent_ops_id,
                      std::optional<uma_ptr_t> signal_accum_addr,
                      dispatch_control_t dispatch_control, 
                      const ObserverData * observer_arr, uint64_t observer_arr_sz,
                      bool force_init_flag) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_extndst_counterpart(counterpart); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_extndst_dispatch_control(dispatch_control); dg::network_exception::is_failed(err)){
            return err;
        }

        if (exception_t err = check_extndst_observer_array(observer_arr, observer_arr_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_extndst_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        dg::network_tile_member_getsetter::set_extndst_observer_array_size_nothrow(ptr, 0u);
        dg::network_tile_member_getsetter::set_extndst_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_extndst_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_extndst_counterpart_nothrow(ptr, counterpart);

        return dg::network_exception::SUCCESS;
    }

    auto init_extndsx(uma_ptr_t ptr,
                      operatable_id_t memevent_ops_id,
                      std::optional<uma_ptr_t> signal_accum_addr,
                      bool force_init_flag) noexcept -> exception_t{

    }

    //I admit things could be done better
    //we are objecting the polymorphic dispatch at the moment

    auto orphan_leaf(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_leaf_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED); 
        return dg::network_exception::SUCCESS;
    }

    auto orphan_blkr(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_blkr_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_blkr_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_blkr_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_mono(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_mono_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_mono_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_pair(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_pair_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_pair_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_uacm(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_uacm_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_uacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_pacm(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_pacm_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_pacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_crit(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_crit_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_crit_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_immu(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_immu_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_immu_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_immu_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_msgrfwd(uma_ptr_t ptr, 
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_msgrfwd_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_msgrbwd(uma_ptr_t ptr, 
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_msgrbwd_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_msgrbwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_extnsrc(uma_ptr_t ptr, 
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_extnsrc_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_extnsrx(uma_ptr_t ptr, 
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

    }

    auto orphan_extndst(uma_ptr_t ptr, 
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_extndst_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_extndst_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);
        return dg::network_exception::SUCCESS;
    }

    auto orphan_extndsx(uma_ptr_t ptr, 
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

    }

    auto deinit_leaf(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_leaf_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_EMPTY);
        dg::network_tile_member_getsetter::set_leaf_logit_nothrow(ptr, dg::network_tile_metadata::TILE_LOGIT_VALUE_DEFAULT);
        dg::network_tile_member_getsetter::set_leaf_grad_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_VALUE_DEFAULT);
        dg::network_tile_member_getsetter::set_leaf_observer_array_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        dg::network_tile_member_getsetter::set_leaf_observer_array_size_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_SIZE_DEFAULT);
        dg::network_tile_member_getsetter::set_leaf_group_operatable_id_nothrow(ptr, dg::network_tile_metadata::TILE_OPERATABLE_ID_DEFAULT);
        dg::network_tile_member_getsetter::set_leaf_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_EMPTY);
        
        return dg::network_exception::SUCCESS;
    }

    auto deinit_blkr(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_blkr_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_blkr_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_blkr_init_status_nothrow();
        dg::network_tile_member_getsetter::set_blkr_logit_nothrow();
        dg::network_tile_member_getsetter::set_blkr_observer_array_nothrow();
        dg::network_tile_member_getsetter::set_blkr_dispatch_control_nothrow();
        dg::network_tile_member_getsetter::set_blkr_descendant_nothrow();
        dg::network_tile_member_getsetter::set_blkr_group_operatable_id_nothrow();
        dg::network_tile_member_getsetter::set_blkr_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_mono(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_mono_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_mono_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_EMPTY);
        dg::network_tile_member_getsetter::set_mono_logit_nothrow(ptr, dg::network_tile_metadata::TILE_LOGIT_VALUE_DEFAULT);
        dg::network_tile_member_getsetter::set_mono_grad_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_VALUE_DEFAULT);
        dg::network_tile_member_getsetter::set_mono_observer_array_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        dg::network_tile_member_getsetter::set_mono_observer_array_size_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_SIZE_DEFAULT);
        dg::network_tile_member_getsetter::set_mono_dispatch_control_nothrow(ptr, dg::network_tile_metadata::TILE_DISPATCH_CONTROL_DEFAULT);
        dg::network_tile_member_getsetter::set_mono_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        dg::network_tile_member_getsetter::set_mono_descendant_nothrow(ptr, dg::network_tile_metadata::TILE_ADDRESS_DEFAULT);
        dg::network_tile_member_getsetter::set_mono_group_operatable_id_nothrow(ptr, dg::network_tile_metadata::TILE_OPERATABLE_ID_DEFAULT);
        dg::network_tile_member_getsetter::set_mono_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_EMPTY);

        return dg::network_exception::SUCCESS;
    }

    auto deinit_pair(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_pair_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_pair_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_EMPTY);
        dg::network_tile_member_getsetter::set_pair_logit_nothrow(ptr, dg::network_tile_metadata::TILE_LOGIT_VALUE_DEFAULT);
        dg::network_tile_member_getsetter::set_pair_grad_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_VALUE_DEFAULT);
        dg::network_tile_member_getsetter::set_pair_observer_array_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        dg::network_tile_member_getsetter::set_pair_observer_array_size_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_SIZE_DEFAULT);
        dg::network_tile_member_getsetter::set_pair_dispatch_control_nothrow(ptr, dg::network_tile_metadata::TILE_DISPATCH_CONTROL_DEFAULT);
        dg::network_tile_member_getsetter::set_pair_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        dg::network_tile_member_getsetter::set_pair_left_descendant_nothrow(ptr, dg::network_tile_metadata::TILE_ADDRESS_DEFAULT);
        dg::network_tile_member_getsetter::set_pair_right_descendant_nothrow(ptr, dg::network_tile_metadata::TILE_ADDRESS_DEFAULT);
        dg::network_tile_member_getsetter::set_pair_group_operatable_id_nothrow(ptr, dg::network_tile_metadata::TILE_OPERATABLE_ID_DEFAULT);
        dg::network_tile_member_getsetter::set_pair_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_EMPTY);
    
        return dg::network_exception::SUCCESS;
    }

    auto deinit_uacm(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_uacm_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_uacm_init_status_nothrow();
        dg::network_tile_member_getsetter::set_uacm_logit_nothrow();
        dg::network_tile_member_getsetter::set_uacm_grad_nothrow();
        dg::network_tile_member_getsetter::set_uacm_observer_array_nothrow();
        dg::network_tile_member_getsetter::set_uacm_observer_array_size_nothrow();
        dg::network_tile_member_getsetter::set_uacm_group_operatable_id_nothrow();
        dg::network_tile_member_getsetter::set_uacm_dispatch_control_nothrow();
        dg::network_tile_member_getsetter::set_uacm_pong_count_nothrow();
        dg::network_tile_member_getsetter::set_uacm_descendant_nothrow();
        dg::network_tile_member_getsetter::set_uacm_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_pacm(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{ 

        auto ptr_access = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_pacm_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_pacm_init_status_nothrow();
        dg::network_tile_member_getsetter::set_pacm_logit_nothrow();
        dg::network_tile_member_getsetter::set_pacm_grad_nothrow();
        dg::network_tile_member_getsetter::set_pacm_observer_array_size_nothrow();
        dg::network_tile_member_getsetter::set_pacm_group_operatable_id_nothrow();
        dg::network_tile_member_getsetter::set_pacm_dispatch_control_nothrow();
        dg::network_tile_member_getsetter::set_pacm_pong_count_nothrow();
        dg::network_tile_member_getsetter::set_pacm_left_descendant_nothrow();
        dg::network_tile_member_getsetter::set_pacm_right_descendant_nothrow();
        dg::network_tile_member_getsetter::set_pacm_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_crit(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }
        
        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_crit_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_crit_init_status_nothrow();
        dg::network_tile_member_getsetter::set_crit_logit_nothrow();
        dg::network_tile_member_getsetter::set_crit_grad_nothrow();
        dg::network_tile_member_getsetter::set_crit_clogit_nothrow();
        dg::network_tile_member_getsetter::set_crit_observer_array_nothrow();
        dg::network_tile_member_getsetter::set_crit_observer_array_size_nothrow();
        dg::network_tile_member_getsetter::set_crit_group_operatable_id_nothrow();
        dg::network_tile_member_getsetter::set_crit_dispatch_control_nothrow();
        dg::network_tile_member_getsetter::set_crit_pong_count_nothrow();
        dg::network_tile_member_getsetter::set_crit_descendant_nothrow();
        dg::network_tile_member_getsetter::set_crit_kind_nothrow();
        dg::network_tile_member_getsetter::set_crit_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_immu(uma_ptr_t ptr, 
                     operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_immu_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_immu_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_immu_init_status_nothrow();
        dg::network_tile_member_getsetter::set_immu_logit_nothrow();
        dg::network_tile_member_getsetter::set_immu_observer_array_nothrow();
        dg::network_tile_member_getsetter::set_immu_observer_array_size_nothrow();
        dg::network_tile_member_getsetter::set_immu_group_operatable_id_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_msgrfwd(uma_ptr_t ptr, 
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_msgrfwd_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow();
        dg::network_tile_member_getsetter::set_msgrfwd_logit_nothrow();
        dg::network_tile_member_getsetter::set_msgrfwd_grad_nothrow();
        dg::network_tile_member_getsetter::set_msgrfwd_observer_array_nothrow();
        dg::network_tile_member_getsetter::set_msgrfwd_observer_array_size_nothrow();
        dg::network_tile_member_getsetter::set_msgrfwd_group_operatable_id_nothrow();
        dg::network_tile_member_getsetter::set_msgrfwd_dispatch_control_nothrow();
        dg::network_tile_member_getsetter::set_msgrfwd_pong_count_nothrow();
        dg::network_tile_member_getsetter::set_msgrfwd_descendant_nothrow();
        dg::network_tile_member_getsetter::set_msgrfwd_dst_info_nothrow();
        dg::network_tile_member_getsetter::set_msgrfwd_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_msgrbwd(uma_ptr_t ptr, 
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{
        
        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_msgrbwd_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_msgrbwd_init_status_nothrow();
        dg::network_tile_member_getsetter::set_msgrbwd_logit_nothrow();
        dg::network_tile_member_getsetter::set_msgrbwd_grad_nothrow();
        dg::network_tile_member_getsetter::set_msgrbwd_observer_array_nothrow();
        dg::network_tile_member_getsetter::set_msgrbwd_observer_array_size_nothrow();
        dg::network_tile_member_getsetter::set_msgrbwd_group_operatable_id_nothrow();
        dg::network_tile_member_getsetter::set_msgrbwd_dispatch_control_nothrow();
        dg::network_tile_member_getsetter::set_msgrbwd_pong_count_nothrow();
        dg::network_tile_member_getsetter::set_msgrbwd_descendant_nothrow();
        dg::network_tile_member_getsetter::set_msgrbwd_dst_info_nothrow();
        dg::network_tile_member_getsetter::set_msgrbwd_timein_nothrow(); //
        dg::network_tile_member_getsetter::set_msgrbwd_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_extnsrc(uma_ptr_t ptr, 
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_extnsrc_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow();
        dg::network_tile_member_getsetter::set_extnsrc_logit_nothrow();
        dg::network_tile_member_getsetter::set_extnsrc_grad_nothrow();
        dg::network_tile_member_getsetter::set_extnsrc_observer_array_nothrow();
        dg::network_tile_member_getsetter::set_extnsrc_observer_array_size_nothrow();
        dg::network_tile_member_getsetter::set_extnsrc_group_operatable_id_nothrow();
        dg::network_tile_member_getsetter::set_extnsrc_dispatch_control_nothrow();
        dg::network_tile_member_getsetter::set_extnsrc_pong_count_nothrow();
        dg::network_tile_member_getsetter::set_extnsrc_descendant_nothrow();
        dg::network_tile_member_getsetter::set_extnsrc_counterpart_nothrow();
        dg::network_tile_member_getsetter::set_extnsrc_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_extnsrx(uma_ptr_t ptr,
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

    }

    auto deinit_extndst(uma_ptr_t ptr, 
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        group_operatable_id_t ops_id = dg::network_tile_member_getsetter::get_extndst_group_operatable_id_nothrow(ptr);

        if (ops_id != group_operatable_id){
            return dg::network_exception::BAD_ACCESS;
        }

        dg::network_tile_member_getsetter::set_extndst_init_status_nothrow();
        dg::network_tile_member_getsetter::set_extndst_observer_array_nothrow();
        dg::network_tile_member_getsetter::set_extndst_observer_array_size_nothrow();
        dg::network_tile_member_getsetter::set_extndst_group_operatable_id_nothrow();
        dg::network_tile_member_getsetter::set_extndst_dispatch_control_nothrow();
        dg::network_tile_member_getsetter::set_extndst_counterpart_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_extndsx(uma_ptr_t ptr,
                        operatable_id_t memevent_ops_id) noexcept -> exception_t{

    }
}

namespace dg::network_tile_lifetime::concurrent_safe_batch{

    struct InitLeafPayLoad{
        uma_ptr_t ptr;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        dg::string logit_value;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, forward_ops_id, backward_ops_id, memevent_ops_id, logit_value, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, forward_ops_id, backward_ops_id, memevent_ops_id, logit_value, force_init_flag);
        }
    };

    struct InitBlkrPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        dispatch_control_t dispatch_control;
        dg::vector<ObserverData> observer_vec;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }
    };

    struct InitMonoPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr,
        dispatch_control_t dispatch_control;
        dg::vector<ObserverData> observer_vec;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }
    };

    struct InitPairPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t lhs;
        uma_ptr_t rhs;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        dispatch_control_t dispatch_control;
        dg::vector<ObserverData> observer_vec;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, lhs, rhs, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, lhs, rhs, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }
    };

    struct InitUACMPayLoad{
        uma_ptr_t ptr;
        dg::vector<uma_ptr_t> src;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        dispatch_control_t dispatch_control;
        dg::vector<ObserverData> observer_vec;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }
    };

    struct InitPACMPayLoad{
        uma_ptr_t ptr;
        dg::vector<uma_ptr_t> lhs;
        dg::vector<uma_ptr_t> rhs;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        dispatch_control_t dispatch_control;
        dg::vector<ObserverData> observer_vec;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, lhs, rhs, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, lhs, rhs, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }
    };

    struct InitCritPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        dispatch_control_t dispatch_control;
        learning_rate_t learning_rate;
        dg::string clogit_value;
        dg::vector<ObserverData> observer_vec;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, learning_rate,
                      clogit_value, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, learning_rate,
                      clogit_value, observer_vec, force_init_flag);
        }
    };

    struct InitImmuPayLoad{
        uma_ptr_t ptr;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        dg::string logit_value;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, forward_ops_id, backward_ops_id, memevent_ops_id, logit_value, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, forward_ops_id, backward_ops_id, memevent_ops_id, logit_value, force_init_flag);
        }
    };

    struct InitMsgrFwdPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        dispatch_control_t dispatch_control;
        ClientDeliveryInfo dst_info;
        dg::vector<ObserverData> observer_vec;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, dst_info, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, dst_info, observer_vec, force_init_flag);
        }
    };

    struct InitMsgrBwdPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        dispatch_control_t dispatch_control;
        ClientDeliveryInfo dst_info;
        dg::vector<ObserverData> observer_vec;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, dst_info, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, dst_info, observer_vec, force_init_flag);
        }
    };

    struct InitExtnSrcPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        uma_ptr_t counterpart;
        uma_ptr_t counterpart_shadow;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        dispatch_control_t dispatch_control;
        dg::vector<ObserverData> observer_vec;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, src, counterpart, counterpart_shadow, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, src, counterpart, counterpart_shadow, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }
    };

    struct InitExtnSrxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id, signal_accum_addr, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id, signal_accum_addr, force_init_flag);
        }
    };

    struct InitExtnDstPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        uma_ptr_t counterpart;
        uma_ptr_t counterpart_shadow;
        operatable_id_t forward_ops_id;
        operatable_id_t backward_ops_id;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        dispatch_control_t dispatch_control;
        dg::vector<ObserverData> observer_vec;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, src, counterpart, counterpart_shadow, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, src, counterpart, counterpart_shadow, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }
    };

    struct InitExtnDsxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id, signal_accum_addr, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id, signal_accum_addr, force_init_flag);
        }
    };

    struct OrphanLeafPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanBlkrPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanMonoPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id; 

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanPairPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanUACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanPACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanCritPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanImmuPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanMsgrFwdPayLoad{
        uma_ptr_t ptr; 
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanMsgrBwdPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanExtnSrcPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanExtnSrxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;
        
        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanExtnDstPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanExtnDsxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitLeafPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitBlkrPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitMonoPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitPairPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitUACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitPACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitCritPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitImmuPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitMsgrFwdPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitMsgrBwdPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitExtnSrcPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitExtnSrxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitExtnDstPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitExtnDsxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ptr, memevent_ops_id);
        }
    };

    auto make_init_leaf_payload(uma_ptr_t ptr, 
                                operatable_id_t forward_ops_id,
                                operatable_id_t backward_ops_id,
                                operatable_id_t memevent_ops_id, 
                                void * logit_value, uint64_t logit_value_sz,
                                bool force_init_flag) noexcept -> std::expected<InitLeafPayLoad, exception_t>{
        
        std::expected<dg::string, exception_t> logit_buf = dg::network_exception::cstyle_initialize<dg::string>((char) 0, logit_value_sz); //I have yet to check the logit_value_sz before allocations YET, we dont know

        if (!logit_buf.has_value()){
            return std::unexpected(logit_buf.error());
        }

        std::memcpy(logit_buf->data(), logit_value, logit_value_sz);

        return InitLeafPayLoad{.ptr             = ptr,
                               .forward_ops_id  = forward_ops_id,
                               .backward_ops_id = backward_ops_id,
                               .memevent_ops_id = memevent_ops_id,
                               .logit_value     = std::move(logit_buf.value()),
                               .force_init_flag = force_init_flag};
    }

    auto make_init_blkr_payload(uma_ptr_t ptr,
                                uma_ptr_t src,
                                operatable_id_t forward_ops_id,
                                operatable_id_t backward_ops_id,
                                operatable_id_t memevent_ops_id,
                                std::optional<uma_ptr_t> signal_accum_addr,
                                dispatch_control_t dispatch_control,
                                ObserverData * observer_arr, uint64_t observer_arr_sz,
                                bool force_init_flag) noexcept -> std::expected<InitBlkrPayLoad, exception_t>{

        std::expected<dg::vector<ObserverData>, exception_t> observer_vec = dg::network_exception::cstyle_initialize<dg::vector<ObserverData>>(observer_arr_sz); //I have yet to ...

        if (!observer_vec.has_value()){
            return std::unexpected(obsrever_vec.error());
        }

        std::copy(observer_arr, std::next(observer_arr, observer_arr_sz), observer_vec->begin());

        return InitBlkrPayLoad{.ptr                 = ptr,
                               .src                 = src,
                               .forward_ops_id      = forward_ops_id,
                               .backward_ops_id     = backward_ops_id,
                               .memevent_ops_id     = memevent_ops_id,
                               .signal_accum_addr   = signal_accum_addr,
                               .dispatch_control    = dispatch_control,
                               .observer_vec        = std::move(observer_vec.value()),
                               .force_init_flag     = force_init_flag};
    }

    auto make_init_mono_payload(uma_ptr_t ptr, 
                                uma_ptr_t src,
                                operatable_id_t forward_ops_id,
                                operatable_id_t backward_ops_id,
                                operatable_id_t memevent_ops_id,
                                std::optional<uma_ptr_t> signal_accum_addr, 
                                dispatch_control_t dispatch_control, 
                                ObserverData * observer_arr, uint64_t observer_arr_sz,
                                bool force_init_flag) noexcept -> std::expected<InitMonoPayLoad, exception_t>{

        std::expected<dg::vector<ObserverData>, exception_t> observer_vec = dg::network_exception::cstyle_initialize<dg::vector<ObserverData>>(observer_arr_sz);

        if (!observer_vec.has_value()){
            return std::unexpected(observer_vec.error());
        }

        std::copy(observer_arr, std::next(observer_arr, observer_arr_sz), observer_vec->begin());

        return InitMonoPayLoad{.ptr                 = ptr,
                               .src                 = src,
                               .forward_ops_id      = forward_ops_id,
                               .backward_ops_id     = backward_ops_id,
                               .memevent_ops_id     = memevent_ops_id,
                               .signal_accum_addr   = signal_accum_addr,
                               .dispatch_control    = dispatch_control,
                               .observer_vec        = std::move(observer_vec.value()),
                               .force_init_flag     = force_init_flag};
    }

    auto make_init_pair_payload(uma_ptr_t ptr, 
                                uma_ptr_t lhs,
                                uma_ptr_t rhs,
                                operatable_id_t forward_ops_id,
                                operatable_id_t backward_ops_id,
                                operatable_id_t memevent_ops_id,
                                std::optional<uma_ptr_t> signal_accum_addr,
                                dispatch_control_t dispatch_control, 
                                ObserverData * observer_arr, uint64_t observer_arr_sz,
                                bool force_init_flag) noexcept -> std::expected<InitPairPayLoad, exception_t>{
        
        std::expected<dg::vector<ObserverData>, exception_t> observer_vec = dg::network_exception::cstyle_initialize<dg::vector<ObserverData>>(observer_arr_sz);

        if (!observer_vec.has_value()){
            return std::unexpected(observer_vec.error());
        }

        std::copy(observer_arr, std::next(observer_arr, observer_arr_sz), observer_vec->begin());

        return InitPairPayLoad{.ptr                 = ptr,
                               .lhs                 = lhs,
                               .rhs                 = rhs,
                               .forward_ops_id      = forward_ops_id,
                               .backward_ops_id     = backward_ops_id,
                               .signal_accum_addr   = signal_accum_addr,
                               .dispatch_control    = dispatch_control,
                               .observer_vec        = std::move(observer_vec.value()),
                               .force_init_flag     = force_init_flag};
    }

    auto make_init_uacm_payload(uma_ptr_t ptr, 
                                uma_ptr_t * src, uint64_t src_sz, 
                                operatable_id_t forward_ops_id,
                                operatable_id_t backward_ops_id,
                                operatable_id_t memevent_ops_id,
                                std::optional<uma_ptr_t> signal_accum_addr,
                                dispatch_control_t dispatch_control, 
                                ObserverData * observer_arr, uint64_t observer_arr_sz,
                                bool force_init_flag) noexcept -> std::expected<InitUACMPayLoad, exception_t>{
        
        std::expected<dg::vector<uma_ptr_t>, exception_t> src_vec = dg::network_exception::cstyle_initialize<dg::vector<uma_ptr_t>>(src_sz);

        if (!src_vec.has_value()){
            return std::unexpected(src_vec.error());
        }

        std::expected<dg::vector<ObserverData>, exception_t> observer_vec = dg::network_exception::cstyle_initialize<dg::vector<ObserverData>>(observer_arr_sz);

        if (!observer_vec.has_value()){
            return std::unexpected(observer_vec.error());
        }

        std::copy(src, std::next(src, src_sz), src_vec->begin());
        std::copy(observer_arr, std::next(observer_arr, observer_arr_sz), observer_vec->begin());

        return InitUACMPayLoad{.ptr                 = ptr,
                               .src                 = std::move(src_vec.value()),
                               .forward_ops_id      = forward_ops_id,
                               .backward_ops_id     = backward_ops_id,
                               .memevent_ops_id     = memevent_ops_id,
                               .signal_accum_addr   = signal_accum_addr,
                               .dispatch_control    = dispatch_control,
                               .observer_vec        = std::move(observer_vec.value()),
                               .force_init_flag     = force_init_flag};
    }

    auto make_init_pacm_payload(uma_ptr_t ptr,
                                uma_ptr_t * lhs, uma_ptr_t * rhs, uint64_t acm_sz,
                                operatable_id_t forward_ops_id,
                                operatable_id_t backward_ops_id,
                                operatable_id_t memevent_ops_id,
                                std::optional<uma_ptr_t> signal_accum_addr,
                                dispatch_control_t dispatch_control,
                                ObserverData * observer_arr, uint64_t observer_arr_sz,
                                bool force_init_flag) noexcept -> std::expected<InitPACMPayLoad, exception_t>{
        
        std::expected<dg::vector<uma_ptr_t>, exception_t> lhs_vec = dg::network_exception::cstyle_initialize<dg::vector<uma_ptr_t>>(acm_sz);

        if (!lhs_vec.has_value()){
            return std::unexpected(lhs_vec.error());
        }

        std::expected<dg::vector<uma_ptr_t>, exception_t> rhs_vec = dg::network_exception::cstyle_initialize<dg::vector<uma_ptr_t>>(acm_sz); 
        
        if (!rhs_vec.has_value()){
            return std::unexpected(rhs_vec.error());
        }

        std::expected<dg::vector<ObserverData>, exception_t> observer_vec = dg::network_exception::cstyle_initialize<dg::vector<ObserverData>>(observer_arr_sz);

        if (!observer_vec.has_value()){
            return std::unexpected(observer_vec.error());
        }

        std::copy(lhs, std::next(lhs, acm_sz), lhs_vec->begin());
        std::copy(rhs, std::next(rhs, acm_sz), rhs_vec->begin());
        std::copy(observer_arr, std::next(observer_arr, observer_arr_sz), observer_vec->begin());

        return InitPACMPayLoad{.ptr                 = ptr,
                               .lhs                 = std::move(lhs_vec.value()),
                               .rhs                 = std::move(rhs_vec.value()),
                               .forward_ops_id      = forward_ops_id,
                               .backward_ops_id     = backward_ops_id,
                               .memevent_ops_id     = memevent_ops_id,
                               .signal_accum_addr   = signal_accum_addr,
                               .dispatch_control    = dispatch_control,
                               .observer_vec        = std::move(observer_vec.value()),
                               .force_init_flag     = force_init_flag};
    }

    auto make_init_crit_payload(uma_ptr_t ptr,
                                uma_ptr_t src,
                                operatable_id_t forward_ops_id,
                                operatable_id_t backward_ops_id,
                                operatable_id_t memevent_ops_id,
                                std::optional<uma_ptr_t> signal_accum_addr,
                                dispatch_control_t dispatch_control,
                                learning_rate_t learning_rate,
                                void * clogit_value, uint64_t clogit_value_sz,
                                ObserverData * observer_arr, uint64_t observer_arr_sz,
                                bool force_init_flag) noexcept -> std::expected<InitCritPayLoad, exception_t>{
                            
        std::expected<dg::string, exception_t> clogit_buf = dg::network_exception::cstyle_initialize<dg::string>(clogit_value_sz);
        
        if (!clogit_buf.has_value()){
            return std::unexpected(clogit_buf.error());
        }

        std::memcpy(clogit_buf->data(), clogit_value, clogit_value_sz);

        std::expected<dg::vector<ObserverData>, exception_t> observer_vec = dg::network_exception::cstyle_initialize<dg::vector<ObserverData>>(observer_arr_sz);

        if (!observer_vec.has_value()){
            return std::unexpected(observer_vec.error());
        }

        std::copy(observer_arr, std::next(observer_arr, observer_arr_sz), observer_vec->begin());

        return InitCritPayLoad{.ptr                 = ptr,
                               .src                 = src,
                               .forward_ops_id      = forward_ops_id,
                               .backward_ops_id     = backward_ops_id,
                               .memevent_ops_id     = memevent_ops_id,
                               .signal_accum_addr   = signal_accum_addr,
                               .dispatch_control    = dispatch_control,
                               .learning_rate       = learning_rate,
                               .clogit_value        = std::move(clogit_buf.value()),
                               .observer_vec        = std::move(observer_vec.value()),
                               .force_init_flag     = force_init_flag};
    }

    auto make_init_immu_payload(uma_ptr_t ptr,
                                operatable_id_t forward_ops_id,
                                operatable_id_t backward_ops_id,
                                operatable_id_t memevent_ops_id, 
                                void * logit_value, uint64_t logit_value_sz,
                                bool force_init_flag) noexcept -> std::expected<InitImmuPayLoad, exception_t>{
        
        std::expected<dg::string, exception_t> logit_buf = dg::network_exception::cstyle_initialize<dg::string>(logit_value_sz);

        if (!logit_buf.has_value()){
            return std::unexpected(logit_buf.error());
        }

        std::memcpy(logit_buf->data(), logit_value, logit_value_sz);

        return InitImmuPayLoad{.ptr             = ptr,
                               .forward_ops_id  = forward_ops_id,
                               .backward_ops_id = backward_ops_id,
                               .memevent_ops_id = memevent_ops_id,
                               .logit_value     = std::move(logit_buf.value()),
                               .force_init_flag = force_init_flag};
    }

    auto make_init_msgrfwd_payload(uma_ptr_t ptr,
                                   uma_ptr_t src,
                                   operatable_id_t forward_ops_id,
                                   operatable_id_t backward_ops_id,
                                   operatable_id_t memevent_ops_id,
                                   std::optional<uma_ptr_t> signal_accum_addr,
                                   dispatch_control_t dispatch_control,
                                   ClientDeliveryInfo dst_info, 
                                   ObserverData * observer_arr, uint64_t observer_arr_sz,
                                   bool force_init_flag) noexcept -> std::expected<InitMsgrFwdPayLoad, exception_t>{
        
        std::expected<dg::vector<ObserverData>, exception_t> observer_vec = dg::network_exception::cstyle_initialize<dg::vector<ObserverData>>(observer_arr_sz);

        if (!observer_vec.has_value()){
            return std::unexpected(observer_vec.error());
        }

        std::copy(observer_arr, std::next(observer_arr, observer_arr_sz), observer_vec->begin());

        return InitMsgrFwdPayLoad{.ptr                  = ptr,
                                  .src                  = src,
                                  .forward_ops_id       = forward_ops_id,
                                  .backward_ops_id      = backward_ops_id,
                                  .memevent_ops_id      = memevent_ops_id,
                                  .signal_accum_addr    = signal_accum_addr,
                                  .dispatch_control     = dispatch_control,
                                  .dst_info             = dst_info,
                                  .observer_vec         = std::move(observer_vec.value()),
                                  .force_init_flag      = force_init_flag};
    }

    auto make_init_msgrbwd_payload(uma_ptr_t ptr,
                                   uma_ptr_t src,
                                   operatable_id_t forward_ops_id,
                                   operatable_id_t backward_ops_id,
                                   operatable_id_t memevent_ops_id,
                                   std::optional<uma_ptr_t> signal_accum_addr,
                                   dispatch_control_t dispatch_control,
                                   ClientDeliveryInfo dst_info, 
                                   ObserverData * observer_arr, uint64_t observer_arr_sz,
                                   bool force_init_flag) noexcept -> std::expected<InitMsgrBwdPayLoad, exception_t>{
        
        std::expected<dg::vector<ObserverData>, exception_t> observer_vec = dg::network_exception::cstyle_initialize<dg::vector<ObserverData>>(observer_arr_sz);

        if (!observer_vec.has_value()){
            return std::unexpected(observer_vec.error());
        }

        std::copy(observer_arr, std::next(observer_arr, observer_arr_sz), observer_vec->begin());

        return InitMsgrBwdPayLoad{.ptr                  = ptr,
                                  .src                  = src,
                                  .forward_ops_id       = forward_ops_id,
                                  .backward_ops_id      = backward_ops_id,
                                  .memevent_ops_id      = memevent_ops_id,
                                  .signal_accum_addr    = signal_accum_addr,
                                  .dispatch_control     = dispatch_control,
                                  .dst_info             = dst_info,
                                  .observer_vec         = std::move(observer_vec.value()),
                                  .force_init_flag      = force_init_flag};
    }

    auto make_init_extnsrc_payload(uma_ptr_t ptr,
                                   uma_ptr_t src,
                                   uma_ptr_t counterpart,
                                   uma_ptr_t counterpart_shadow,
                                   operatable_id_t forward_ops_id,
                                   operatable_id_t backward_ops_id,
                                   operatable_id_t memevent_ops_id,
                                   std::optional<uma_ptr_t> signal_accum_addr, 
                                   dispatch_control_t dispatch_control,
                                   ObserverData * observer_arr, uint64_t observer_arr_sz,
                                   bool force_init_flag) noexcept -> std::expected<InitExtnSrcPayLoad, exception_t>{
        
        std::expected<dg::vector<ObserverData>, exception_t> observer_vec = dg::network_exception::cstyle_initialize<dg::vector<ObserverData>>(observer_arr_sz);

        if (!observer_vec.has_value()){
            return std::unexpected(observer_vec.error());
        }

        std::copy(observer_arr, std::next(observer_arr, observer_arr_sz), observer_vec->begin());

        return InitExtnSrcPayLoad{.ptr                  = ptr,
                                  .src                  = src,
                                  .counterpart          = counterpart,
                                  .counterpart_shadow   = counterpart_shadow,
                                  .forward_ops_id       = forward_ops_id,
                                  .backward_ops_id      = backward_ops_id,
                                  .memevent_ops_id      = memevent_ops_id,
                                  .signal_accum_addr    = signal_accum_addr,
                                  .dispatch_control     = dispatch_control,
                                  .observer_vec         = std::move(observer_vec.value()),
                                  .force_init_flag      = force_init_flag};
    }

    auto make_init_extnsrx_payload(uma_ptr_t ptr,
                                   operatable_id_t memevent_ops_id,
                                   std::optional<uma_ptr_t> signal_accum_addr,
                                   bool force_init_flag) noexcept -> std::expected<InitExtnSrxPayLoad, exception_t>{
        
        return InitExtnSrxPayLoad{.ptr                  = ptr,
                                  .memevent_ops_id      = memevent_ops_id,
                                  .signal_accum_addr    = signal_accum_addr,
                                  .force_init_flag      = force_init_flag};
    }

    auto make_init_extndst_payload(uma_ptr_t ptr,
                                   uma_ptr_t src,
                                   uma_ptr_t counterpart,
                                   uma_ptr_t counterpart_shadow,
                                   operatable_id_t forward_ops_id,
                                   operatable_id_t backward_ops_id,
                                   operatable_id_t memevent_ops_id,
                                   std::optional<uma_ptr_t> signal_accum_addr,
                                   dispatch_control_t dispatch_control, 
                                   ObserverData * observer_arr, uint64_t observer_arr_sz,
                                   bool force_init_flag) noexcept -> std::expected<InitExtnDstPayLoad, exception_t>{
        
        std::expected<dg::vector<ObserverData>, exception_t> observer_vec = dg::network_exception::cstyle_initialize<dg::vector<ObserverData>>(observer_arr_sz);

        if (!observer_vec.has_value()){
            return std::unexpected(observer_vec.error());
        }

        std::copy(observer_arr, std::next(observer_arr, observer_arr_sz), observer_vec->begin());

        return InitExtnDstPayLoad{.ptr                  = ptr,
                                  .src                  = src,
                                  .counterpart          = counterpart,
                                  .counterpart_shadow   = counterpart_shadow,
                                  .forward_ops_id       = forward_ops_id,
                                  .backward_ops_id      = backward_ops_id,
                                  .memevent_ops_id      = memevent_ops_id,
                                  .signal_accum_addr    = signal_accum_addr,
                                  .dispatch_control     = dispatch_control,
                                  .observer_vec         = std::move(observer_vec.value()),
                                  .force_init_flag      = force_init_flag}
    }

    auto make_init_extndsx_payload(uma_ptr_t ptr,
                                   operatable_id_t memevent_ops_id,
                                   std::optional<uma_ptr_t> signal_accum_addr,
                                   bool force_init_flag) noexcept -> std::expected<InitExtnDsxPayLoad, exception_t>{
        
        return InitExtnDsxPayLoad{.ptr                  = ptr,
                                  .memevent_ops_id      = memevent_ops_id,
                                  .signal_accum_addr    = signal_accum_addr,
                                  .force_init_flag      = force_init_flag};
    }

    auto make_orphan_leaf_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanLeafPayLoad, exception_t>{
        
        return OrphanLeafPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_orphan_blkr_payload(uma_ptr_t ptr,
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanBlkrPayLoad, exception_t>{
        
        return OrphanBlkrPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_orphan_mono_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanMonoPayLoad, exception_t>{
        
        return OrphanMonoPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_orphan_pair_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanPairPayLoad, exception_t>{
        
        return OrphanPairPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_orphan_uacm_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanUACMPayLoad, exception_t>{
        
        return OrphanUACMPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_orphan_pacm_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanPACMPayLoad, exception_t>{
        
        return OrphanPACMPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_orphan_crit_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanCritPayLoad, exception_t>{
        
        return OrphanCritPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_orphan_immu_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanImmuPayLoad, exception_t>{

        return OrphanImmuPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_orphan_msgrfwd_payload(uma_ptr_t ptr, 
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanMsgrFwdPayLoad, exception_t>{

        return OrphanMsgrFwdPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_orphan_msgrbwd_payload(uma_ptr_t ptr, 
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanMsgrBwdPayLoad, exception_t>{

        return OrphanMsgrBwdPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_orphan_extnsrc_payload(uma_ptr_t ptr, 
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanExtnSrcPayLoad, exception_t>{

        return OrphanExtnSrcPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_orphan_extnsrx_payload(uma_ptr_t ptr,
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanExtnSrxPayLoad, exception_t>{

        return OrphanExtnSrxPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_orphan_extndst_payload(uma_ptr_t ptr, 
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanExtnDstPayLoad, exception_t>{

        return OrphanExtnDstPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_orphan_extndsx_payload(uma_ptr_t ptr,
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<OrphanExtnDsxPayLoad, exception_t>{

        return OrphanExtnDsxPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_deinit_leaf_payload(uma_ptr_t ptr,
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitLeafPayLoad, exception_t>{

        return DeinitLeafPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_deinit_blkr_payload(uma_ptr_t ptr,
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitBlkrPayLoad, exception_t>{

        return DeinitBlkrPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }
 
    auto make_deinit_mono_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitMonoPayLoad, exception_t>{

        return DeinitMonoPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_deinit_pair_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitPairPayLoad, exception_t>{

        return DeinitPairPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_deinit_uacm_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitUACMPayLoad, exception_t>{

        return DeinitUACMPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_deinit_pacm_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitPACMPayLoad, exception_t>{

        return DeinitPACMPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_deinit_crit_payload(uma_ptr_t ptr,
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitCritPayLoad, exception_t>{

        return DeinitCritPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_deinit_immu_payload(uma_ptr_t ptr, 
                                  operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitImmuPayLoad, exception_t>{

        return DeinitImmuPayLoad{.ptr               = ptr,
                                 .memevent_ops_id   = memevent_ops_id};
    }

    auto make_deinit_msgrfwd_payload(uma_ptr_t ptr,
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitMsgrFwdPayLoad, exception_t>{

        return DeinitMsgrFwdPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_deinit_msgrbwd_payload(uma_ptr_t ptr, 
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitMsgrBwdPayLoad, exception_t>{

        return DeinitMsgrBwdPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_deinit_extnsrc_payload(uma_ptr_t ptr, 
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitExtnSrcPayLoad, exception_t>{

        return DeinitExtnSrcPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_deinit_extnsrx_payload(uma_ptr_t ptr,
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitExtnSrxPayLoad, exception_t>{

        return DeinitExtnSrxPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_deinit_extndst_payload(uma_ptr_t ptr, 
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitExtnDstPayLoad, exception_t>{

        return DeinitExtnDstPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    auto make_deinit_extndsx_payload(uma_ptr_t ptr,
                                     operatable_id_t memevent_ops_id) noexcept -> std::expected<DeinitExtnDsxPayLoad, exception_t>{

        return DeinitExtnDsxPayLoad{.ptr                = ptr,
                                    .memevent_ops_id    = memevent_ops_id};
    }

    //the problem is that we are leaking resources, std::move_iterator<> on failure must be kept like before invoking the function
    //the move iterator is looking very badly for this particular payload job
    //the problem is that we dont want to do a move, each of the payload is already 1 cache fetch ...
    //so it's better to do a pointer delvrsrv instead of a move InitLeafPayload...

    void load_init_leaf_payload(const InitLeafPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8; //we'll fix this

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitLeafPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<uma_memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                            dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                            dg::network_uma::MEMORY_PLATFORM_CUDA}; //we are trying to catch as many use cases as possible but not all use cases - first usecase is cuda_runtime_ptr_t overlapped with cuda_transfer_ptr_t backed by cuda_runtime_ptr_t (circular), second usecase is cuda_runtime_ptr_t overlapped with host_ptr_t, third usecase is cuda_runtime_ptr_t overlapped with cuda_fsys_ptr_t backed by storage devices, fourth is cuda_fsys_ptr_t

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size()); //we'll fix this

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr]   = payload_arr[i];
                exception_t init_err                = dg::network_tile_lifetime::concurrent_unsafe::init_leaf(payload_ptr->ptr, 
                                                                                                              payload_ptr->forward_ops_id, 
                                                                                                              payload_ptr->backward_ops_id,
                                                                                                              payload_ptr->memevent_ops_id,
                                                                                                              payload_ptr->logit_value.data(),
                                                                                                              payload_ptr->logit_value.size(),
                                                                                                              payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_err)){
                    *exception_ptr = init_err;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitLeafPayLoad * payload_ptr             = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_arr, i); 
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr(payload_ptr->ptr); 

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    }

    void load_init_mono_payload(const InitMonoPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitMonoPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr]   = payload_arr[i];
                exception_t init_err                = dg::network_tile_lifetime::concurrent_unsafe::init_mono(payload_ptr->ptr,
                                                                                                              payload_ptr->src,
                                                                                                              payload_ptr->forward_ops_id,
                                                                                                              payload_ptr->backward_ops_id,
                                                                                                              payload_ptr->memevent_ops_id,
                                                                                                              payload_ptr->signal_accum_addr,
                                                                                                              payload_ptr->dispatch_control, 
                                                                                                              payload_ptr->observer_vec.data(), payload_ptr->observer_vec.size(),
                                                                                                              payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_err)){
                    *exception_ptr = init_err;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitMonoPayLoad * payload_ptr             = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr(payload_ptr->ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    } 

    void load_init_pair_payload(const InitPairPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitPairPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr]   = payload_arr[i];
                exception_t init_err                = dg::network_tile_lifetime::concurrent_unsafe::init_pair(payload_ptr->ptr,
                                                                                                              payload_ptr->lhs,
                                                                                                              payload_ptr->rhs,
                                                                                                              payload_ptr->forward_ops_id,
                                                                                                              payload_ptr->backward_ops_id,
                                                                                                              payload_ptr->memevent_ops_id,
                                                                                                              payload_ptr->signal_accum_addr,
                                                                                                              payload_ptr->dispatch_control,
                                                                                                              payload_ptr->observer_vec.data(), payload_ptr->observer_vec.size(),
                                                                                                              payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_err)){
                    *exception_ptr = init_err;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitPairPayLoad * payload_ptr             = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr(payload_ptr->ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    }

    void load_init_uacm_payload(const InitUACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitUACMPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr]   = payload_arr[i];
                exception_t init_err                = dg::network_tile_lifetime::concurrent_unsafe::init_uacm(payload_ptr->ptr, 
                                                                                                              payload_ptr->src.data(), payload_ptr->src.size(), 
                                                                                                              payload_ptr->forward_ops_id,
                                                                                                              payload_ptr->backward_ops_id,
                                                                                                              payload_ptr->memevent_ops_id,
                                                                                                              payload_ptr->signal_accum_addr,
                                                                                                              payload_ptr->dispatch_control, 
                                                                                                              payload_ptr->observer_vec.data(), payload_ptr->observer_vec.size(),
                                                                                                              payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_err)){
                    *exception_ptr = init_err;
                }
            }
        };

        dg::network_producer_consumer::LamdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(sz, LOCAL_VECTORIZATION_SZ); 
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitUACMPayLoad * payload_ptr             = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr(payload_ptr->ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    }

    void load_init_pacm_payload(const InitPACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitPACMPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = payload_arr[i]; 

                if (payload_ptr->lhs.size() != payload_ptr->rhs.size()){
                    *exception_ptr = dg::network_exception::INVALID_ARGUMENT;
                    continue;
                }

                exception_t init_err = dg::network_tile_lifetime::concurrent_unsafe::init_pacm(payload_ptr->ptr,
                                                                                               payload_ptr->lhs.data(), payload_ptr->rhs.data(), payload_ptr->lhs.size(),
                                                                                               payload_ptr->forward_ops_id,
                                                                                               payload_ptr->backward_ops_id,
                                                                                               payload_ptr->memevent_ops_id,
                                                                                               payload_ptr->signal_accum_addr,
                                                                                               payload_ptr->dispatch_control, 
                                                                                               payload_ptr->observer_vec.data(), payload_ptr->observer_vec.size(),
                                                                                               payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_err)){
                    *exception_ptr = init_err;
                }
            }
        };
        
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitPACMPayLoad * payload_ptr             = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_ptr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr(payload_ptr->ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    }

    void load_init_crit_payload(const InitCritPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitCritPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST, 
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr]   = payload_arr[i];
                exception_t init_status             = dg::network_tile_lifetime::concurrent_unsafe::init_crit(payload_ptr->ptr, 
                                                                                                              payload_ptr->src, 
                                                                                                              payload_ptr->forward_ops_id,
                                                                                                              payload_ptr->backward_ops_id,
                                                                                                              payload_ptr->memevent_ops_id,
                                                                                                              payload_ptr->signal_accum_addr,
                                                                                                              payload_ptr->dispatch_control,  
                                                                                                              payload_ptr->learning_rate, 
                                                                                                              payload_ptr->clogit_value.data(), payload_ptr->clogit_value.size(), 
                                                                                                              payload_ptr->observer_vec.data(), payload_ptr->observer_vec.size(),
                                                                                                              payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_status)){
                    *exception_ptr = init_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz); 
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitCritPayLoad * payload_ptr             = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr(payload_ptr->ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    }

    void load_init_immu_payload(const InitImmuPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitImmuPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr]   = payload_arr[i];
                exception_t init_status             = dg::network_tile_lifetime::concurrent_unsafe::init_immu(payload_ptr->ptr,
                                                                                                              payload_ptr->forward_ops_id,
                                                                                                              payload_ptr->backward_ops_id,
                                                                                                              payload_ptr->memevent_ops_id,
                                                                                                              payload_ptr->logit_value.data(), payload_ptr->logit_value.size(),
                                                                                                              payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_status)){
                    *exception_ptr = init_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitImmuPayLoad * payload_ptr             = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr(payload_ptr->ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    }

    void load_init_msgrfwd_payload(const InitMsgrFwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitMsgrFwdPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr]   = payload_arr[i];
                exception_t init_status             = dg::network_tile_lifetime::concurrent_unsafe::init_msgrfwd(payload_ptr->ptr, 
                                                                                                                 payload_ptr->src,
                                                                                                                 payload_ptr->forward_ops_id,
                                                                                                                 payload_ptr->backward_ops_id,
                                                                                                                 payload_ptr->memevent_ops_id,
                                                                                                                 payload_ptr->signal_accum_addr,
                                                                                                                 payload_ptr->dispatch_control,
                                                                                                                 payload_ptr->dst_info,
                                                                                                                 payload_ptr->observer_vec.data(), payload_ptr->observer_vec.size(),
                                                                                                                 payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_status)){
                    *exception_ptr = init_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitMsgrFwdPayLoad * payload_ptr          = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr(payload_ptr->ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    }

    void load_init_msgrbwd_payload(const InitMsgrBwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitMsgrBwdPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr]   = payload_arr[i];
                exception_t init_status             = dg::network_tile_lifetime::concurrent_unsafe::init_msgrbwd(payload_ptr->ptr,
                                                                                                                 payload_ptr->src,
                                                                                                                 payload_ptr->forward_ops_id,
                                                                                                                 payload_ptr->backward_ops_id,
                                                                                                                 payload_ptr->memevent_ops_id,
                                                                                                                 payload_ptr->signal_accum_addr,
                                                                                                                 payload_ptr->dispatch_control,
                                                                                                                 payload_ptr->dst_info,
                                                                                                                 payload_ptr->observer_vec.data(), payload_ptr->observer_vec.size(), 
                                                                                                                 payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_status)){
                    *exception_ptr = init_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitMsgrBwdPayLoad * payload_ptr          = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr(payload_ptr->ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    }

    void load_init_extnsrc_payload(const InitExtnSrcPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitExtnSrcPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr]   = payload_arr[i];
                exception_t init_status             = dg::network_tile_lifetime::concurrent_unsafe::init_extnsrc(payload_ptr->ptr, 
                                                                                                                 payload_ptr->src, 
                                                                                                                 payload_ptr->counterpart, 
                                                                                                                 payload_ptr->counterpart_shadow,
                                                                                                                 payload_ptr->forward_ops_id,
                                                                                                                 payload_ptr->backward_ops_id,
                                                                                                                 payload_ptr->memevent_ops_id,
                                                                                                                 payload_ptr->signal_accum_addr,
                                                                                                                 payload_ptr->dispatch_control,
                                                                                                                 payload_ptr->observer_vec.data(), payload_ptr->observer_vec.size(),
                                                                                                                 payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_status)){
                    *exception_ptr = init_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitExtnSrcPayLoad * payload_ptr          = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr(payload_ptr->ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    }

    void load_init_extndst_payload(const InitExtnDstPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<const InitExtnDstPayLoad *, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr]   = payload_arr[i];
                exception_t init_status             = dg::network_tile_lifetime::concurrent_unsafe::init_extndst(payload_ptr->ptr,
                                                                                                                 payload_ptr->src,
                                                                                                                 payload_ptr->counterpart,
                                                                                                                 payload_ptr->counterpart_shadow,
                                                                                                                 payload_ptr->forward_ops_id,
                                                                                                                 payload_ptr->backward_ops_id,
                                                                                                                 payload_ptr->memevent_ops_id,
                                                                                                                 payload_ptr->signal_accum_addr,
                                                                                                                 payload_ptr->dispatch_control,
                                                                                                                 payload_ptr->observer_vec.data(), payload_ptr->observer_vec.size(),
                                                                                                                 payload_ptr->force_init_flag);

                if (dg::network_exception::is_failed(init_status)){
                    *exception_ptr = init_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const InitExtnDstPayLoad * payload_ptr          = std::next(payload_arr, i);
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr(payload_ptr->ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_ptr, exception_ptr));
        }
    }

    void load_orphan_leaf_payload(const OrphanLeafPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanLeafPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_leaf(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanLeafPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_orphan_mono_payload(const OrphanMonoPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanMonoPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_mono(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanMonoPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            } 

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_orphan_pair_payload(const OrphanPairPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanPairPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_pair(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanPairPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exxception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_orphan_uacm_payload(const OrphanUACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;
        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanUACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_uacm(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanUACMPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_orphan_pacm_payload(const OrphanPACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanPACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_pacm(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanPACMPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_orphan_crit_payload(const OrphanCritPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanCritPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_crit(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanCritPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_orphan_msgrfwd_payload(const OrphanMsgrFwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanMsgrFwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_msgrfwd(payload.ptr, payload.memevent_ops_id);
                
                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanMsgrFwdPayLoad payload                    = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_orphan_msgrbwd_payload(const OrphanMsgrBwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanMsgrBwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_msgrbwd(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanMsgrBwdPayLoad payload                    = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_adr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_orphan_extnsrc_payload(const OrphanExtnSrcPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanExtnSrcPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_extnsrc(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanExtnSrcPayload payload                    = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_orphan_extndst_payload(const OrphanExtnDstPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanExtnDstPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_extndst(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };
        
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectrz);
        
        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanExtnDstPayLoad payload                    = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tule(payload, exception_ptr));
        }
    }

    void load_orphan_immu_payload(const OrphanImmuPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanImmuPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_immu(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanImmuPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_leaf_payload(const DeinitLeafPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitLeafPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_leaf(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitLeafPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr(payload.ptr)

            if (!rcu_addr.has_value());{
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_mono_payload(const DeinitMonoPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMonoPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_mono(payload.ptr, payload.memevent_ops_id);
                
                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitMonoPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_pair_payload(const DeinitPairPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitPairPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_pair(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitPairPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_uacm_payload(const DeinitUACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitUACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size()); 

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_uacm(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz); 
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitUACMPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            } 

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_pacm_payload(const DeinitPACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitPACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_pacm(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz); 
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitPACMPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_crit_payload(const DeinitCritPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitCritPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_crit(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);
        //g
        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz); 
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitCritPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_msgrfwd_payload(const DeinitMsgrFwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMsgrFwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_msgrfwd(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitMsgrFwdPayLoad payload                    = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_msgrbwd_payload(const DeinitMsgrBwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMsgrBwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_msgrbwd(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitMsgrBwdPayLoad payload                    = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_extnsrc_payload(const DeinitExtnSrcPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitExtnSrcPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_extnsrc(payload.ptr, payload.memevent_ops_id);
                
                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitExtnSrcPayLoad payload                    = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_extndst_payload(const DeinitExtnDstPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitExtnDstPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_extndst(payload.ptr, payload.memevent_ops_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitExtnDstPayLoad payload                    = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr(payload.dst);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }

    void load_deinit_immu_payload(const DeinitImmuPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;

        auto vectorizer = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitImmuPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            constexpr std::array<memory_advise_t, 5u> MEM_ADVISE_SET = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                        dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                        dg::network_uma::MEMORY_PLATFORM_CUDA};

            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_immu(payload.ptr, payload.memevent_ops_id);
                
                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_vectorizer(vectorizer);

        size_t trimmed_vectorization_sz = std::min(LOCAL_VECTORIZATION_SZ, sz);
        size_t allocation_cost          = dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectorizer, trimmed_vectorization_sz); 
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> buf(allocation_cost);
        auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectorizer, trimmed_vectorization_sz, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitImmuPayLoad payload                       = payload_arr[i];
            exception_t * exception_ptr                     = std::next(exception_arr, i);
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr(payload.dst);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr));
        }
    }
}

namespace dg::network_tile_lifetime::concurrent_safe_poly{

    using payload_kind_t   = uint8_t;

    enum enum_payload: payload_kind_t{
        payload_kind_init_leaf          = 0u,
        payload_kind_init_blkr          = 1u,
        payload_kind_init_mono          = 2u,
        payload_kind_init_pair          = 3u,
        payload_kind_init_uacm          = 4u,
        payload_kind_init_pacm          = 5u,
        payload_kind_init_crit          = 6u,
        payload_kind_init_immu          = 7u,
        payload_kind_init_poly          = 8u,
        payload_kind_init_msgrfwd       = 9u,
        payload_kind_init_msgrbwd       = 10u,
        payload_kind_init_extnsrc       = 11u,
        payload_kind_init_extnsrx       = 12u,
        payload_kind_init_extndst       = 13u,
        payload_kind_init_extndsx       = 14u,

        payload_kind_orphan_leaf        = 15u,
        payload_kind_orphan_blkr        = 16u,
        payload_kind_orphan_mono        = 17u,
        payload_kind_orphan_pair        = 18u,
        payload_kind_orphan_uacm        = 19u,
        payload_kind_orphan_pacm        = 20u,
        payload_kind_orphan_crit        = 21u,
        payload_kind_orphan_immu        = 22u,
        payload_kind_orphan_poly        = 23u,
        payload_kind_orphan_msgrfwd     = 24u,
        payload_kind_orphan_msgrbwd     = 25u,
        payload_kind_orphan_extnsrc     = 26u,
        payload_kind_orphan_extnsrx     = 27u,
        payload_kind_orphan_extndst     = 28u,
        payload_kind_orphan_extndsx     = 29u,

        payload_kind_deinit_leaf        = 30u,
        payload_kind_deinit_blkr        = 31u,
        payload_kind_deinit_mono        = 32u,
        payload_kind_deinit_pair        = 33u,
        payload_kind_deinit_uacm        = 34u,
        payload_kind_deinit_pacm        = 35u,
        payload_kind_deinit_crit        = 36u,
        payload_kind_deinit_immu        = 37u,
        payload_kind_deinit_poly        = 38u,
        payload_kind_deinit_msgrfwd     = 39u,
        payload_kind_deinit_msgrbwd     = 40u,
        payload_kind_deinit_extnsrc     = 41u,
        payload_kind_deinit_extnsrx     = 42u,
        payload_kind_deinit_extndst     = 43u,
        payload_kind_deinit_extndsx     = 44u
    };

    //these are used for internal errors + checkings, aren't supposed to be changed as secrets
    //these numbers are part of the structure definition, as if it is invisible to users yet crucial for serialization + deserialization
    //we'll provide the serialization format to external users via the public APIs, which they'd want to call a C file to run the serialization and get the payloads
    //parsing a JSON string is very hard to get correctly + very slow, we'll stick with the approach for now

    static inline constexpr uint32_t POLY_PAYLOAD_SERIALIZATION_SECRET              = 1015571905UL;
    static inline constexpr uint32_t INIT_LEAF_PAYLOAD_SERIALIZATION_SECRET         = 775110819UL;
    static inline constexpr uint32_t INIT_BLKR_PAYLOAD_SERIALIZATION_SECRET         = 1410260615UL;
    static inline constexpr uint32_t INIT_MONO_PAYLOAD_SERIALIZATION_SECRET         = 1279769031UL;
    static inline constexpr uint32_t INIT_PAIR_PAYLOAD_SERIALIZATION_SECRET         = 3319988850UL;
    static inline constexpr uint32_t INIT_UACM_PAYLOAD_SERIALIZATION_SECRET         = 2132729877UL;
    static inline constexpr uint32_t INIT_PACM_PAYLOAD_SERIALIZATION_SECRET         = 3463896483UL;
    static inline constexpr uint32_t INIT_CRIT_PAYLOAD_SERIALIZATION_SECRET         = 4285963311UL;
    static inline constexpr uint32_t INIT_IMMU_PAYLOAD_SERIALIZATION_SECRET         = 4162690786UL
    static inline constexpr uint32_t INIT_POLY_PAYLOAD_SERIALIZATION_SECRET         = 1031221616UL;
    static inline constexpr uint32_t INIT_MSGRFWD_PAYLOAD_SERIALIZATION_SECRET      = 206935149UL;
    static inline constexpr uint32_t INIT_MSGRBWD_PAYLOAD_SERIALIZATION_SECRET      = 3038720787UL;
    static inline constexpr uint32_t INIT_EXTNSRC_PAYLOAD_SERIALIZATION_SECRET      = 3070992396UL;
    static inline constexpr uint32_t INIT_EXTNSRX_PAYLOAD_SERIALIZATION_SECRET      = 3691088546UL;
    static inline constexpr uint32_t INIT_EXTNDST_PAYLOAD_SERIALIZATION_SECRET      = 1967035241UL;
    static inline constexpr uint32_t INIT_EXTNDSX_PAYLOAD_SERIALIZATION_SECRET      = 160838281UL;

    static inline constexpr uint32_t ORPHAN_LEAF_PAYLOAD_SERIALIZATION_SECRET       = 3153482732UL;
    static inline constexpr uint32_t ORPHAN_BLKR_PAYLOAD_SERIALIZATION_SECRET       = 45859177UL;
    static inline constexpr uint32_t ORPHAN_MONO_PAYLOAD_SERIALIZATION_SECRET       = 2239278293UL;
    static inline constexpr uint32_t ORPHAN_PAIR_PAYLOAD_SERIALIZATION_SECRET       = 4093082929UL;
    static inline constexpr uint32_t ORPHAN_UACM_PAYLOAD_SERIALIZATION_SECRET       = 650003605UL;
    static inline constexpr uint32_t ORPHAN_PACM_PAYLOAD_SERIALIZATION_SECRET       = 2264319474UL;
    static inline constexpr uint32_t ORPHAN_CRIT_PAYLOAD_SERIALIZATION_SECRET       = 1513593274UL;
    static inline constexpr uint32_t ORPHAN_IMMU_PAYLOAD_SERIALIZATION_SECRET       = 1416929944UL;
    static inline constexpr uint32_t ORPHAN_POLY_PAYLOAD_SERIALIZATION_SECRET       = 237134996UL;
    static inline constexpr uint32_t ORPHAN_MSGRFWD_PAYLOAD_SERIALIZATION_SECRET    = 2055475217UL;
    static inline constexpr uint32_t ORPHAN_MSGRBWD_PAYLOAD_SERIALIZATION_SECRET    = 2673192566UL;
    static inline constexpr uint32_t ORPHAN_EXTNSRC_PAYLOAD_SERIALIZATION_SECRET    = 1537480300UL;
    static inline constexpr uint32_t ORPHAN_EXTNSRX_PAYLOAD_SERIALIZATION_SECRET    = 25239095UL;
    static inline constexpr uint32_t ORPHAN_EXNTDST_PAYLOAD_SERIALIZATION_SECRET    = 3004954735UL;
    static inline constexpr uint32_t ORPHAN_EXTNDSX_PAYLOAD_SERIALIZATION_SECRET    = 957952478UL;

    static inline constexpr uint32_t DEINIT_LEAF_PAYLOAD_SERIALIZATION_SECRET       = 3383954495UL;
    static inline constexpr uint32_t DEINIT_BLKR_PAYLOAD_SERIALIZATION_SECRET       = 1994339924UL;
    static inline constexpr uint32_t DEINIT_MONO_PAYLOAD_SERIALIZATION_SECRET       = 1019424741UL;
    static inline constexpr uint32_t DEINIT_PAIR_PAYLOAD_SERIALIZATION_SECRET       = 2205608154UL;
    static inline constexpr uint32_t DEINIT_UACM_PAYLOAD_SERIALIZATION_SECRET       = 2406924483UL;
    static inline constexpr uint32_t DEINIT_PACM_PAYLOAD_SERIALIZATION_SECRET       = 4265611268UL;
    static inline constexpr uint32_t DEINIT_CRIT_PAYLOAD_SERIALIZATION_SECRET       = 2561344351UL;
    static inline constexpr uint32_t DEINIT_IMMU_PAYLOAD_SERIALIZATION_SECRET       = 2933283905UL;
    static inline constexpr uint32_t DEINIT_POLY_PAYLOAD_SERIALIZATION_SECRET       = 1350879522UL;
    static inline constexpr uint32_t DEINIT_MSGRFWD_PAYLOAD_SERIALIZATION_SECRET    = 2872102250UL;
    static inline constexpr uint32_t DEINIT_MSGRBWD_PAYLOAD_SERIALIZATION_SECRET    = 3996583514UL;
    static inline constexpr uint32_t DEINIT_EXTNSRC_PAYLOAD_SERIALIZATION_SECRET    = 2769127462UL;
    static inline constexpr uint32_t DEINIT_EXTNSRX_PAYLOAD_SERIALIZATION_SECRET    = 3359415902UL;
    static inline constexpr uint32_t DEINIT_EXTNDST_PAYLOAD_SERIALIZATION_SECRET    = 2732518174UL;
    static inline constexpr uint32_t DEINIT_EXTNDSX_PAYLOAD_SERIALIZATION_SECRET    = 440808470UL;

    struct VirtualPayLoad{
        payload_kind_t kind;
        dg::string content;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(kind, content);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(kind, content);
        }
    };

    auto virtualize_payload(const InitLeafPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitLeafPayLoad>)(payload, INIT_LEAF_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_leaf,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitBlkrPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitBlkrPayLoad>)(payload, INIT_BLKR_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_blkr,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitMonoPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitMonoPayLoad>)(payload, INIT_MONO_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_mono,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitPairPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitPairPayLoad>)(payload, INIT_PAIR_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_pair,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitUACMPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitUACMPayLoad>)(payload, INIT_UACM_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_uacm,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitPACMPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitPACMPayLoad>)(payload, INIT_PACM_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_pacm,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitCritPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitCritPayLoad>)(payload, INIT_CRIT_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_crit,
                              .content  = std::move(serialized_payload.value())};

    }

    auto virtualize_payload(const InitImmuPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitImmuPayLoad>)(payload, INIT_IMMU_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_immu,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitPolyPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitPolyPayLoad>)(payload, INIT_POLY_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_poly,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitMsgrFwdPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitMsgrFwdPayLoad>)(payload, INIT_MSGRFWD_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_msgrfwd,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitMsgrBwdPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitMsgrBwdPayLoad>)(payload, INIT_MSGRBWD_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_msgrbwd,
                              .content  = std::move(serialized_payload.value())};

    }

    auto virtualize_payload(const InitExtnSrcPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitExtnSrcPayLoad>)(payload, INIT_EXTNSRC_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_extnsrc,
                              .content  = std::move(serialized_payload.value())};

    }

    auto virtualize_payload(const InitExtnSrxPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitExtnSrxPayLoad>)(payload, INIT_EXTNSRX_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_extnsrx,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitExtnDstPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitExtnDstPayLoad>)(payload, INIT_EXTNDST_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_extndst,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const InitExtnDsxPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, InitExtnDsxPayLoad>)(payload, INIT_EXTNDSX_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_init_extndsx,
                              .content  = std::move(serialized_payload.value())};
    }

    //----

    auto virtualize_payload(const OrphanLeafPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanLeafPayLoad>)(payload, ORPHAN_LEAF_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_leaf,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanBlkrPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanBlkrPayLoad>)(payload, ORPHAN_BLKR_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_blkr,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanMonoPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanMonoPayLoad>)(payload, ORPHAN_MONO_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_mono,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanPairPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanPairPayLoad>)(payload, ORPHAN_PAIR_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_pair,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanUACMPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanUACMPayLoad>)(payload, ORPHAN_UACM_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_uacm,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanPACMPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanPACMPayLoad>)(payload, ORPHAN_PACM_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_pacm,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanCritPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanCritPayLoad>)(payload, ORPHAN_CRIT_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_crit,
                              .content  = std::move(serialized_payload.value())};

    }

    auto virtualize_payload(const OrphanPolyPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanPolyPayLoad>)(payload, ORPHAN_POLY_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_poly,
                              .content  = std::move(serialized_payload.value())};

    }

    auto virtualize_payload(const OrphanImmuPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanImmuPayLoad>)(payload, ORPHAN_IMMU_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_immu,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanMsgrFwdPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanMsgrFwdPayLoad>)(payload, ORPHAN_MSGRFWD_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_msgrfwd,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanMsgrBwdPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanMsgrBwdPayLoad>)(payload, ORPHAN_MSGRBWD_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_msgrbwd,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanExtnSrcPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanExtnSrcPayLoad>)(payload, ORPHAN_EXTNSRC_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_extnsrc,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanExtnSrxPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanExtnSrxPayLoad>)(payload, ORPHAN_EXTNSRX_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_extnsrx,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanExtnDstPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanExtnDstPayLoad>)(payload, ORPHAN_EXNTDST_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_extndst,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const OrphanExtnDsxPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, OrphanExtnDsxPayLoad>)(payload, ORPHAN_EXTNDSX_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_orphan_extndsx,
                              .content  = std::move(serialized_payload.value())};
    }

    //----

    auto virtualize_payload(const DeinitLeafPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitLeafPayLoad>)(payload, DEINIT_LEAF_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_leaf,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitBlkrPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitBlkrPayLoad>)(payload, DEINIT_BLKR_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_blkr,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitMonoPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitMonoPayLoad>)(payload, DEINIT_MONO_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_mono,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitPairPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitPairPayLoad>)(payload, DEINIT_PAIR_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_pair,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitUACMPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitUACMPayLoad>)(payload, DEINIT_UACM_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_uacm,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitPACMPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitPACMPayLoad>)(payload, DEINIT_PACM_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_pacm,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitCritPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitCritPayLoad>)(payload, DEINIT_CRIT_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_crit,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitImmuPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitImmuPayLoad>)(payload, DEINIT_IMMU_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_immu,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitPolyPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitPolyPayLoad>)(payload, DEINIT_POLY_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_poly,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitMsgrFwdPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitMsgrFwdPayLoad>)(payload, DEINIT_MSGRFWD_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_msgrfwd,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitMsgrBwdPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitMsgrBwdPayLoad>)(payload, DEINIT_MSGRBWD_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_msgrbwd,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitExtnSrcPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitExtnSrcPayLoad>)(payload, DEINIT_EXTNSRC_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_extnsrc,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitExtnSrxPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitExtnSrxPayLoad>)(payload, DEINIT_EXTNSRX_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_extnsrx,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitExtnDstPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitExtnDstPayLoad>)(payload, DEINIT_EXTNDST_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_extndst,
                              .content  = std::move(serialized_payload.value())};
    }

    auto virtualize_payload(const DeinitExtnDsxPayLoad& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{

        std::expected<dg::string, exception_t> serialized_payload = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, DeinitExtnDsxPayLoad>)(payload, DEINIT_EXTNDSX_PAYLOAD_SERIALIZATION_SECRET);

        if (!serialized_payload.has_value()){
            return std::unexpected(serialized_payload.error());
        }

        return VirtualPayLoad{.kind     = payload_kind_deinit_extndsx,
                              .content  = std::move(serialized_payload.value())};
    }

    //-------

    auto serialize_payload(const VirtualPayLoad& virtual_payload) noexcept -> std::expected<dg::string, exception_t>{

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_serialize<dg::string, VirtualPayLoad>)(virtual_payload, POLY_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto deserialize_payload(const dg::string& payload) noexcept -> std::expected<VirtualPayLoad, exception_t>{
        
        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<VirtualPayLoad, dg::string>)(payload, POLY_PAYLOAD_SERIALIZATION_SECRET);
    }

    //-----

    auto devirtualize_init_leaf_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitLeafPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_leaf){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitLeafPayLoad, dg::string>)(payload.content, INIT_LEAF_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_blkr_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitBlkrPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_blkr){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitBlkrPayLoad, dg::string>)(payload.content, INIT_BLKR_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_mono_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitMonoPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_mono){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitMonoPayLoad, dg::string>)(payload.content, INIT_MONO_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_pair_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitPairPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_pair){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitPairPayLoad, dg::string>)(payload.content, INIT_PAIR_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_uacm_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitUACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_uacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitUACMPayLoad, dg::string>)(payload.content, INIT_UACM_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_pacm_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitPACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_pacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitPACMPayLoad, dg::string>)(payload.content, INIT_PACM_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_crit_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitCritPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_crit){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitCritPayLoad, dg::string>)(payload.content, INIT_CRIT_PAYLOAD_SERIALIZATION_SECRET);
    } 

    auto devirtualize_init_immu_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitImmuPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_immu){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitImmuPayLoad, dg::string>)(payload.content, INIT_IMMU_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_msgrfwd_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitMsgrFwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_msgrfwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitMsgrFwdPayLoad, dg::string>)(payload.content, INIT_MSGRFWD_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_msgrbwd_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitMsgrBwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_msgrbwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitMsgrBwdPayLoad, dg::string>)(payload.content, INIT_MSGRBWD_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_extnsrc_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitExtnSrcPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_extnsrc){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitExtnSrcPayLoad, dg::string>)(payload.content, INIT_EXTNSRC_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_extnsrx_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitExtnSrxPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_extnsrx){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitExtnSrxPayLoad, dg::string>)(payload.content, INIT_EXTNSRX_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_extndst_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitExtnDstPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_extndst){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitExtnDstPayLoad, dg::string>)(payload.content, INIT_EXNTDST_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_init_extndst_payload(const VirtualPayLoad& payload) noexcept -> std::expected<InitExtnDsxPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_extndsx){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<InitExtnDsxPayLoad, dg::string>)(payload.content, INIT_EXNTDSX_PAYLOAD_SERIALIZATION_SECRET);
    }

    //

    auto devirtualize_orphan_leaf_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanLeafPayLoad, exception_t>{
        
        if (payload.kind != payload_kind_orphan_leaf){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanLeafPayLoad, dg::string>)(payload.content, ORPHAN_LEAF_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_blkr_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanBlkrPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_blkr){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanBlkrPayLoad, dg::string>)(payload.content, ORPHAN_BLKR_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_mono_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanMonoPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_mono){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanMonoPayLoad, dg::string>)(payload.content, ORPHAN_MONO_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_pair_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanPairPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_pair){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanPairPayLoad, dg::string>)(payload.content, ORPHAN_PAIR_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_uacm_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanUACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_uacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanUACMPayLoad, dg::string>)(payload.content, ORPHAN_UACM_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_pacm_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanPACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_pacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanPACMPayLoad, dg::string>)(payload.content, ORPHAN_PACM_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_crit_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanCritPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_crit){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanCritPayLoad, dg::string>)(payload.content, ORPHAN_CRIT_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_immu_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanImmuPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_immu){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanImmuPayLoad, dg::string>)(payload.content, ORPHAN_IMMU_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_msgrfwd_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanMsgrFwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_msgrfwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanMsgrFwdPayLoad, dg::string>)(payload.content, ORPHAN_MSGRFWD_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_msgrbwd_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanMsgrBwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_msgrbwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanMsgrBwdPayLoad, dg::string>)(payload.content, ORPHAN_MSGRBWD_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_extnsrc_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanExtnSrcPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_extnsrc){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanExtnSrcPayLoad, dg::string>)(payload.content, ORPHAN_EXTNSRC_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_extnsrx_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanExtnSrxPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_extnsrx){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanExtnSrxPayLoad, dg::string>)(payload.content, ORPHAN_EXTNSRX_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_extndst_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanExtnDstPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_extndst){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanExtnDstPayLoad, dg::string>)(payload.content, ORPHAN_EXTNDST_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_orphan_extndsx_payload(const VirtualPayLoad& payload) noexcept -> std::expected<OrphanExtnDsxPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_extndsx){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<OrphanExtnDsxPayLoad, dg::string>)(payload.content, ORPHAN_EXTNDSX_PAYLOAD_SERIALIZATION_SECRET);
    }

    //

    auto devirtualize_deinit_leaf_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitLeafPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_leaf){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitLeafPayLoad, dg::string>)(payload.content, DEINIT_LEAF_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_blkr_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitBlkrPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_blkr){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitBlkrPayLoad, dg::string>)(payload.content, DEINIT_BLKR_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_mono_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitMonoPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_mono){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitMonoPayLoad, dg::string>)(payload.content, DEINIT_MONO_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_pair_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitPairPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_pair){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitPairPayLoad, dg::string>)(payload.content, DEINIT_PAIR_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_uacm_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitUACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_uacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitUACMPayLoad, dg::string>)(payload.content, DEINIT_UACM_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_pacm_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitPACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_pacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitPACMPayLoad, dg::string>)(payload.content, DEINIT_PACM_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_crit_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitCritPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_crit){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitCritPayLoad, dg::string>)(payload.content, DEINIT_CRIT_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_immu_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitImmuPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_immu){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitImmuPayLoad, dg::string>)(payload.content, DEINIT_IMMU_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_msgrfwd_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitMsgrFwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_msgrfwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitMsgrFwdPayLoad, dg::string>)(payload.content, DEINIT_MSGRFWD_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_msgrbwd_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitMsgrBwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_msgrbwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitMsgrBwdPayLoad, dg::string>)(payload.content, DEINIT_MSGRBWD_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_extnsrc_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitExtnSrcPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_extnsrc){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitExtnSrcPayLoad, dg::string>)(payload.content, DEINIT_EXTNSRC_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_extnsrx_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitExtnSrxPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_extnsrx){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitExtnSrxPayLoad, dg::string>)(payload.content, DEINIT_EXTNSRX_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_extndst_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitExtnDstPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_extndst){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitExtnDstPayLoad, dg::string>)(payload.content, DEINIT_EXTNDST_PAYLOAD_SERIALIZATION_SECRET);
    }

    auto devirtualize_deinit_extndsx_payload(const VirtualPayLoad& payload) noexcept -> std::expected<DeinitExtnDsxPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_extndsx){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        return dg::network_exception::to_cstyle_function(dg::network_compact_serializer::dgstd_deserialize<DeinitExtnDsxPayLoad, dg::string>)(payload.content, DEINIT_EXTNDSX_PAYLOAD_SERIALIZATION_SECRET);
    }

    void load_virtual_payloads(const VirtualPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        //exception_t * need to be exactly 1 byte to be reasonably random-accessed

        constexpr size_t INIT_LEAF_DISPATCH_DELIVERY_CAP            = size_t{1} << 8; //this seems funny but each of these guys have different DISPATCH_DELIVERY_CAP (due to certain contraints of acquiring memregions + frens), we'll attempt to solve that problem by adding our ultimate PolyObject 
        constexpr size_t INIT_BLKR_DISPATCH_DELIVERY_CAP            = size_t{1} << 8;
        constexpr size_t INIT_MONO_DISPATCH_DELIVERY_CAP            = size_t{1} << 8;
        constexpr size_t INIT_PAIR_DISPATCH_DELIVERY_CAP            = size_t{1} << 8;
        constexpr size_t INIT_UACM_DISPATCH_DELIVERY_CAP            = size_t{1} << 8;
        constexpr size_t INIT_PACM_DISPATCH_DELIVERY_CAP            = size_t{1} << 8;
        constexpr size_t INIT_CRIT_DISPATCH_DELIVERY_CAP            = size_t{1} << 8;
        constexpr size_t INIT_IMMU_DISPATCH_DELIVERY_CAP            = size_t{1} << 8; 
        constexpr size_t INIT_POLY_DISPATCH_DELIVERY_CAP            = size_t{1} << 8; //this is the most important tile, we have considered every possible batching techniques, from fixed vectorization size, to fixed delivery cap per dispatch code to somewhere in between (we have come to a conclusion that this is for the best)
        constexpr size_t INIT_MSGRFWD_DISPATCH_DELIVERY_CAP         = size_t{1} << 8; 
        constexpr size_t INIT_MSGRBWD_DISPATCH_DELIVERY_CAP         = size_t{1} << 8;
        constexpr size_t INIT_EXTNSRC_DISPATCH_DELIVERY_CAP         = size_t{1} << 8;
        constexpr size_t INIT_EXTNSRX_DISPATCH_DELIVERY_CAP         = size_t{1} << 8;
        constexpr size_t INIT_EXTNDST_DISPATCH_DELIVERY_CAP         = size_t{1} << 8;
        constexpr size_t INIT_EXTNDSX_DISPATCH_DELIVERY_CAP         = size_t{1} << 8;

        constexpr size_t ORPHAN_LEAF_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t ORPHAN_BLKR_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t ORPHAN_MONO_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t ORPHAN_PAIR_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t ORPHAN_UACM_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t ORPHAN_PACM_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t ORPHAN_CRIT_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t ORPHAN_IMMU_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t ORPHAN_POLY_DISPATCH_DELIVERY_CAP          = size_t{1} << 8; 
        constexpr size_t ORPHAN_MSGRFWD_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;
        constexpr size_t ORPHAN_MSGRBWD_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;
        constexpr size_t ORPHAN_EXTNSRC_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;
        constexpr size_t ORPHAN_EXTNSRX_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;
        constexpr size_t ORPHAN_EXTNDST_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;
        constexpr size_t ORPHAN_EXTNDSX_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;

        constexpr size_t DEINIT_LEAF_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t DEINIT_BLKR_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t DEINIT_MONO_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t DEINIT_PAIR_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t DEINIT_UACM_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t DEINIT_PACM_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t DEINIT_CRIT_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t DEINIT_IMMU_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t DEINIT_POLY_DISPATCH_DELIVERY_CAP          = size_t{1} << 8;
        constexpr size_t DEINIT_MSGRFWD_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;
        constexpr size_t DEINIT_MSGRBWD_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;
        constexpr size_t DEINIT_EXTNSRC_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;
        constexpr size_t DEINIT_EXTNSRX_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;
        constexpr size_t DEINIT_EXTNDST_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;
        constexpr size_t DEINIT_EXTNDSX_DISPATCH_DELIVERY_CAP       = size_t{1} << 8;

        //we dont want to generalize the polymorphic dispatches, it looks reduntdant but we'd want to keep it that way, if there are issues, we can compromise the edit tree -> that specific dispatch only, such is the worst possible scenerio is that specific dispatch being not working

        auto init_leaf_dispatcher                   = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitLeafPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u; 

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitLeafPayLoad, exception_t> devirtualized_payload = devirtualize_init_leaf_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_leaf_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_blkr_dispatcher                   = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitBlkrPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitBlkrPayLoad, exception_t> devirtualized_payload = devirtualize_init_blkr_payload(*payload_ptr);
                
                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_blkr_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_mono_dispatcher                   = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitMonoPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitMonoPayLoad, exception_t> devirtualized_payload = devirtualize_init_mono_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_mono_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_pair_dispatcher                   = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitPairPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitPairPayLoad, exception_t> devirtualized_payload = devirtualize_init_pair_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_pair_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_uacm_dispatcher                   = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitUACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitUACMPayLoad, exception_t> devirtualized_payload = devirtualize_init_uacm_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_uacm_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_pacm_dispatcher                   = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitPACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitPACMPayLoad, exception_t> devirtualized_payload = devirtualize_init_pacm_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_pacm_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_crit_dispatcher                   = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitCritPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitCritPayLoad, exception_t> devirtualized_payload = devirtualize_init_crit_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_crit_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_immu_dispatcher                   = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitImmuPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitImmuPayLoad, exception_t> devirtualized_payload = devirtualize_init_immu_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_immu_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_poly_dispatcher                   = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitPolyPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitPolyPayLoad, exception_t> devirtualized_payload = devirtualize_init_poly_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_poly_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_msgrfwd_dispatcher                = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitMsgrFwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitMsgrFwdPayLoad, exception_t> devirtualized_payload = devirtualize_init_msgrfwd_payload(*payload_ptr);
                
                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_msgrfwd_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_msgrbwd_dispatcher                = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitMsgrBwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitMsgrBwdPayLoad, exception_t> devirtualized_payload = devirtualize_init_msgrbwd_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_msgrbwd_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_extnsrc_dispatcher                = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitExtnSrcPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitExtnSrcPayLoad, exception_t> devirtualized_payload = devirtualize_init_extnsrc_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_extnsrc_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_extnsrx_dispatcher                = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitExtnSrxPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitExtnSrxPayLoad, exception_t> devirtualized_payload = devirtualize_init_extnsrx_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_extnsrx_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_extndst_dispatcher                = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitExtnDstPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitExtnDstPayLoad, exception_t> devirtualized_payload = devirtualize_init_extndst_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_extndst_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto init_extndsx_dispatcher                = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitExtnDsxPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<InitExtnDsxPayLoad, exception_t> devirtualized_payload = devirtualize_init_extndsx_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_extndsx_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        //

        auto orphan_leaf_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanLeafPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanLeafPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_leaf_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_leaf_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_blkr_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanBlkrPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanBlkrPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_blkr_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_blkr_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_mono_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanMonoPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanMonoPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_mono_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_mono_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_pair_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanPairPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanPairPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_pair_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_pair_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_uacm_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanUACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;
           
            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanUACMPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_uacm_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_uacm_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_pacm_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanPACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanPACMPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_pacm_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_pacm_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_crit_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanCritPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanCritPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_crit_payload(*payload_ptr);
                
                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_crit_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_immu_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanImmuPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u; 

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanImmuPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_immu_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_immu_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i]))[
                    *exception_ptr_arr[i] = exception_arr[i];
                ]
            }
        };

        auto orphan_poly_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanPolyPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u; 

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanPolyPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_poly_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_poly_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i]))[
                    *exception_ptr_arr[i] = exception_arr[i];
                ]
            }
        };

        auto orphan_msgrfwd_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanMsgrFwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanMsgrFwdPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_msgrfwd_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_msgrfwd_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_msgrbwd_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanMsgrBwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanMsgrBwdPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_msgrbwd_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_msgrbwd_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_extnsrc_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanExtnSrcPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanExtnSrcPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_extnsrc_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u; 
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_extnsrc_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_extnsrx_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanExtnSrxPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanExtnSrxPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_extnsrx_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u; 
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_extnsrx_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_extndst_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanExtnDstPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanExtnDstPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_extndst_payload(*payload_ptr);
                
                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_extndst_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto orphan_extndsx_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanExtnDsxPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<OrphanExtnDsxPayLoad, exception_t> devirtualized_payload = devirtualize_orphan_extndsx_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_extndsx_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        //

        auto deinit_leaf_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitLeafPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitLeafPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_leaf_payload(*payload_ptr);
                
                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_leaf_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_blkr_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitBlkrPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitBlkrPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_blkr_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_blkr_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_mono_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitMonoPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitMonoPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_mono_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_mono_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_pair_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitPairPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitPairPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_pair_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_pair_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_uacm_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitUACMPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitUACMPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_uacm_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_uacm_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_pacm_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitPACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitPACMPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_pacm_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_pacm_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_crit_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitCritPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitCritPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_crit_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_crit_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_immu_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitImmuPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitImmuPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_immu_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_immu_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_immu_dispatcher                 = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitPolyPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitPolyPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_poly_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_poly_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_msgrfwd_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitMsgrFwdPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitMsgrFwdPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_msgrfwd_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_msgrfwd_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_msgrbwd_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitMsgrBwdPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitMsgrBwdPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_msgrbwd_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_msgrbwd_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_extnsrc_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitExtnSrcPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitExtnSrcPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_extnsrc_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr  = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_extnsrc_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_extnsrx_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitExtnSrxPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitExtnSrxPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_extnsrx_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_extnsrx_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i]))[
                    *exception_ptr_arr[i] = exception_arr[i];
                ]
            }
        };

        auto deinit_extndst_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitExtnDstPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitExtnDstPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_extndst_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_extndst_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        auto deinit_extndsx_dispatcher              = [](std::pair<const VirtualPayLoad *, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitExtnDsxPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<exception_t>[]> exception_ptr_arr(sz);

            size_t dispatch_sz = 0u;

            for (size_t i = 0u; i < sz; ++i){
                auto [payload_ptr, exception_ptr] = data_arr[i];
                std::expected<DeinitExtnDsxPayLoad, exception_t> devirtualized_payload = devirtualize_deinit_extndsx_payload(*payload_ptr);

                if (!devirtualized_payload.has_value()){
                    *exception_ptr = devirtualized_payload.error();
                    continue;
                }

                devirt_payload_arr[dispatch_sz] = std::move(devirtualized_payload.value());
                exception_ptr_arr[dispatch_sz]  = exception_ptr;
                dispatch_sz                     += 1u;
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_extndsx_payload(devirt_payload_arr.get(), exception_arr.get(), dispatch_sz);

            for (size_t i = 0u; i < dispatch_sz; ++i){
                if (dg::network_exception::is_failed(exception_arr[i])){
                    *exception_ptr_arr[i] = exception_arr[i];
                }
            }
        };

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_leaf_dispatcher(init_leaf_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_blkr_dispatcher(init_blkr_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_mono_dispatcher(init_mono_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_pair_dispatcher(init_pair_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_uacm_dispatcher(init_uacm_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_pacm_dispatcher(init_pacm_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_crit_dispatcher(init_crit_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_immu_dispatcher(init_immu_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_poly_dispatcher(init_poly_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_msgrfwd_dispatcher(init_msgrfwd_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_msgrbwd_dispatcher(init_msgrbwd_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_extnsrc_dispatcher(init_extnsrc_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_extnsrx_dispatcher(init_extnsrx_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_extndst_dispatcher(init_extndst_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_init_extndsx_dispatcher(init_extndsx_dispatcher);

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_leaf_dispatcher(orphan_leaf_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_blkr_dispatcher(orphan_blkr_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_mono_dispatcher(orphan_mono_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_pair_dispatcher(orphan_pair_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_uacm_dispatcher(orphan_uacm_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_pacm_dispatcher(orphan_pacm_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_crit_dispatcher(orphan_crit_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_immu_dispatcher(orphan_immu_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_poly_dispatcher(orphan_poly_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_msgrfwd_dispatcher(orphan_msgrfwd_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_msgrbwd_dispatcher(orphan_msgrbwd_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_extnsrc_dispatcher(orphan_extnsrc_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_extnsrx_dispatcher(orphan_extnsrx_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_extndst_dispatcher(orphan_extndst_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_orphan_extndsx_dispatcher(orphan_extndsx_dispatcher);

        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_leaf_dispatcher(deinit_leaf_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_blkr_dispatcher(deinit_blkr_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_mono_dispatcher(deinit_mono_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_pair_dispatcher(deinit_pair_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_uacm_dispatcher(deinit_uacm_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_pacm_dispatcher(deinit_pacm_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_crit_dispatcher(deinit_crit_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_immu_dispatcher(deinit_immu_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_poly_dispatcher(deinit_poly_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_msgrfwd_dispatcher(deinit_msgrfwd_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_msgrbwd_dispatcher(deinit_msgrbwd_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_extnsrc_dispatcher(deinit_extnsrc_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_extnsrx_dispatcher(deinit_extnsrx_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_extndst_dispatcher(deinit_extndst_dispatcher);
        dg::network_producer_consumer::LambdaWrappedConsumer virtual_deinit_extndsx_dispatcher(deinit_extndsx_dispatcher);

        //alrights - we want to split interface and link these guys by char[] here
        //stack allocations is probably one of the major optimization to reduce spin_lock overheads + allow true concurrency by using affined allocations

        size_t trimmed_init_leaf_dispatch_sz                    = std::min(INIT_LEAF_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_leaf_dispatch_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_leaf_dispatcher, trimmed_init_leaf_dispatch_sz); 
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_leaf_allocation(virtual_init_leaf_dispatch_allocation_cost);

        size_t trimmed_init_blkr_dispatch_sz                    = std::min(INIT_BLKR_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_blkr_dispatch_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_blkr_dispatcher, trimmed_init_blkr_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_blkr_allocation(virtual_init_blkr_dispatch_allocation_cost);

        size_t trimmed_init_mono_dispatch_sz                    = std::min(INIT_MONO_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_mono_dispatch_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_mono_dispatcher, trimmed_init_mono_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_mono_allocation(virtual_init_mono_dispatch_allocation_cost);

        size_t trimmed_init_pair_dispatch_sz                    = std::min(INIT_PAIR_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_pair_dispatch_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_pair_dispatcher, trimmed_init_pair_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_pair_allocation(virtual_init_pair_dispatch_allocation_cost);

        size_t trimmed_init_uacm_dispatch_sz                    = std::min(INIT_UACM_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_uacm_dispatch_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_uacm_dispatcher, trimmed_init_uacm_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_uacm_allocation(virtual_init_uacm_dispatch_allocation_cost);

        size_t trimmed_init_pacm_dispatch_sz                    = std::min(INIT_PACM_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_pacm_dispatch_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_pacm_mono_dispatcher, trimmed_init_pacm_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_pacm_allocation(virtual_init_pacm_dispatch_allocation_cost);

        size_t trimmed_init_crit_dispatch_sz                    = std::min(INIT_CRIT_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_crit_dispatch_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_crit_dispatcher, trimmed_init_crit_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_crit_allocation(virtual_init_crit_dispatch_allocation_cost);

        size_t trimmed_init_immu_dispatch_sz                    = std::min(INIT_IMMU_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_immu_dispatch_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_immu_dispatcher, trimmed_init_immu_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_immu_allocation(virtual_init_immu_dispatch_allocation_cost);

        size_t trimmed_init_poly_dispatch_sz                    = std::min(INIT_POLY_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_poly_dispatch_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_poly_dispatcher, trimmed_init_poly_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_immu_allocation(virtual_init_poly_dispatch_allocation_cost);

        size_t trimmed_init_msgrfwd_dispatch_sz                 = std::min(INIT_MSGRFWD_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_msgrfwd_dispatch_allocation_cost    = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_msgrfwd_dispatcher, trimmed_init_msgrfwd_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_msgrfwd_allocation(virtual_init_msgrfwd_dispatch_allocation_cost);

        size_t trimmed_init_msgrbwd_dispatch_sz                 = std::min(INIT_MSGRBWD_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_msgrbwd_dispatch_allocation_cost    = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_msgrbwd_dispatcher, trimmed_init_msgrbwd_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_msgrbwd_allocation(virtual_init_msgrbwd_dispatch_allocation_cost);

        size_t trimmed_init_extnsrc_dispatch_sz                 = std::min(INIT_EXTNSRC_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_extnsrc_dispatch_allocation_cost    = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_extnsrc_dispatcher, trimmed_init_extnsrc_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_extnsrc_allocation(virtual_init_extnsrc_dispatch_allocation_cost);

        size_t trimmed_init_extnsrx_dispatch_sz                 = std::min(INIT_EXTNSRX_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_extnsrx_dispatch_allocation_cost    = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_extnsrx_dispatcher, trimmed_init_extnsrx_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_extnsrx_allocation(virtual_init_extnsrx_dispatch_allocation_cost);

        size_t trimmed_init_extndst_dispatch_sz                 = std::min(INIT_EXTNDST_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_extndst_dispatch_allocation_cost    = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_extndst_dispatcher, trimmed_init_extndst_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_extndst_allocation(virtual_init_extndst_dispatch_allocation_cost);

        size_t trimmed_init_extndsx_dispatch_sz                 = std::min(INIT_EXTNDSX_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_init_extndsx_dispatch_allocation_cost    = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_init_extndsx_dispatcher, trimmed_init_extndsx_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_init_extndsx_allocation(virtual_init_extndsx_dispatch_allocation_cost);

        //

        size_t trimmed_orphan_leaf_dispatch_sz                  = std::min(ORPHAN_LEAF_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_leaf_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_leaf_dispatcher, trimmed_orphan_leaf_dispatch_sz); 
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_leaf_allocation(virtual_orphan_leaf_dispatch_allocation_cost);

        size_t trimmed_orphan_blkr_dispatch_sz                  = std::min(ORPHAN_BLKR_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_blkr_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_blkr_dispatcher, trimmed_orphan_blkr_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_blkr_allocation(virtual_orphan_blkr_dispatch_allocation_cost);

        size_t trimmed_orphan_mono_dispatch_sz                  = std::min(ORPHAN_MONO_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_mono_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_mono_dispatcher, trimmed_orphan_mono_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_mono_allocation(virtual_orphan_mono_dispatch_allocation_cost);

        size_t trimmed_orphan_pair_dispatch_sz                  = std::min(ORPHAN_PAIR_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_pair_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_pair_dispatcher, trimmed_orphan_pair_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_pair_allocation(virtual_orphan_pair_dispatch_allocation_cost);

        size_t trimmed_orphan_uacm_dispatch_sz                  = std::min(ORPHAN_UACM_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_uacm_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_uacm_dispatcher, trimmed_orphan_uacm_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_uacm_allocation(virtual_orphan_uacm_dispatch_allocation_cost);

        size_t trimmed_orphan_pacm_dispatch_sz                  = std::min(ORPHAN_PACM_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_pacm_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_pacm_mono_dispatcher, trimmed_orphan_pacm_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_pacm_allocation(virtual_orphan_pacm_dispatch_allocation_cost);

        size_t trimmed_orphan_crit_dispatch_sz                  = std::min(ORPHAN_CRIT_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_crit_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_crit_dispatcher, trimmed_orphan_crit_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_crit_allocation(virtual_orphan_crit_dispatch_allocation_cost);

        size_t trimmed_orphan_immu_dispatch_sz                  = std::min(ORPHAN_IMMU_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_immu_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_immu_dispatcher, trimmed_orphan_immu_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_immu_allocation(virtual_orphan_immu_dispatch_allocation_cost);

        size_t trimmed_orphan_poly_dispatch_sz                  = std::min(ORPHAN_POLY_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_poly_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_poly_dispatcher, trimmed_orphan_poly_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_immu_allocation(virtual_orphan_poly_dispatch_allocation_cost);

        size_t trimmed_orphan_msgrfwd_dispatch_sz               = std::min(ORPHAN_MSGRFWD_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_msgrfwd_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_msgrfwd_dispatcher, trimmed_orphan_msgrfwd_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_msgrfwd_allocation(virtual_orphan_msgrfwd_dispatch_allocation_cost);

        size_t trimmed_orphan_msgrbwd_dispatch_sz               = std::min(ORPHAN_MSGRBWD_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_msgrbwd_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_msgrbwd_dispatcher, trimmed_orphan_msgrbwd_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_msgrbwd_allocation(virtual_orphan_msgrbwd_dispatch_allocation_cost);

        size_t trimmed_orphan_extnsrc_dispatch_sz               = std::min(ORPHAN_EXTNSRC_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_extnsrc_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_extnsrc_dispatcher, trimmed_orphan_extnsrc_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_extnsrc_allocation(virtual_orphan_extnsrc_dispatch_allocation_cost);

        size_t trimmed_orphan_extnsrx_dispatch_sz               = std::min(ORPHAN_EXTNSRX_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_extnsrx_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_extnsrx_dispatcher, trimmed_orphan_extnsrx_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_extnsrx_allocation(virtual_orphan_extnsrx_dispatch_allocation_cost);

        size_t trimmed_orphan_extndst_dispatch_sz               = std::min(ORPHAN_EXTNDST_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_extndst_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_extndst_dispatcher, trimmed_orphan_extndst_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_extndst_allocation(virtual_orphan_extndst_dispatch_allocation_cost);

        size_t trimmed_orphan_extndsx_dispatch_sz               = std::min(ORPHAN_EXTNDSX_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_orphan_extndsx_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_orphan_extndsx_dispatcher, trimmed_orphan_extndsx_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_orphan_extndsx_allocation(virtual_orphan_extndsx_dispatch_allocation_cost);

        //

        size_t trimmed_deinit_leaf_dispatch_sz                  = std::min(DEINIT_LEAF_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_leaf_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_leaf_dispatcher, trimmed_deinit_leaf_dispatch_sz); 
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_leaf_allocation(virtual_deinit_leaf_dispatch_allocation_cost);

        size_t trimmed_deinit_blkr_dispatch_sz                  = std::min(DEINIT_BLKR_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_blkr_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_blkr_dispatcher, trimmed_deinit_blkr_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_blkr_allocation(virtual_deinit_blkr_dispatch_allocation_cost);

        size_t trimmed_deinit_mono_dispatch_sz                  = std::min(DEINIT_MONO_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_mono_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_mono_dispatcher, trimmed_deinit_mono_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_mono_allocation(virtual_deinit_mono_dispatch_allocation_cost);

        size_t trimmed_deinit_pair_dispatch_sz                  = std::min(DEINIT_PAIR_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_pair_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_pair_dispatcher, trimmed_deinit_pair_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_pair_allocation(virtual_deinit_pair_dispatch_allocation_cost);

        size_t trimmed_deinit_uacm_dispatch_sz                  = std::min(DEINIT_UACM_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_uacm_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_uacm_dispatcher, trimmed_deinit_uacm_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_uacm_allocation(virtual_deinit_uacm_dispatch_allocation_cost);

        size_t trimmed_deinit_pacm_dispatch_sz                  = std::min(DEINIT_PACM_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_pacm_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_pacm_mono_dispatcher, trimmed_deinit_pacm_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_pacm_allocation(virtual_deinit_pacm_dispatch_allocation_cost);

        size_t trimmed_deinit_crit_dispatch_sz                  = std::min(DEINIT_CRIT_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_crit_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_crit_dispatcher, trimmed_deinit_crit_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_crit_allocation(virtual_deinit_crit_dispatch_allocation_cost);

        size_t trimmed_deinit_immu_dispatch_sz                  = std::min(DEINIT_IMMU_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_immu_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_immu_dispatcher, trimmed_deinit_immu_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_immu_allocation(virtual_deinit_immu_dispatch_allocation_cost);

        size_t trimmed_deinit_poly_dispatch_sz                  = std::min(DEINIT_POLY_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_poly_dispatch_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_poly_dispatcher, trimmed_deinit_poly_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_immu_allocation(virtual_deinit_poly_dispatch_allocation_cost);

        size_t trimmed_deinit_msgrfwd_dispatch_sz               = std::min(DEINIT_MSGRFWD_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_msgrfwd_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_msgrfwd_dispatcher, trimmed_deinit_msgrfwd_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_msgrfwd_allocation(virtual_deinit_msgrfwd_dispatch_allocation_cost);

        size_t trimmed_deinit_msgrbwd_dispatch_sz               = std::min(DEINIT_MSGRBWD_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_msgrbwd_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_msgrbwd_dispatcher, trimmed_deinit_msgrbwd_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_msgrbwd_allocation(virtual_deinit_msgrbwd_dispatch_allocation_cost);

        size_t trimmed_deinit_extnsrc_dispatch_sz               = std::min(DEINIT_EXTNSRC_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_extnsrc_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_extnsrc_dispatcher, trimmed_deinit_extnsrc_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_extnsrc_allocation(virtual_deinit_extnsrc_dispatch_allocation_cost);

        size_t trimmed_deinit_extnsrx_dispatch_sz               = std::min(DEINIT_EXTNSRX_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_extnsrx_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_extnsrx_dispatcher, trimmed_deinit_extnsrx_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_extnsrx_allocation(virtual_deinit_extnsrx_dispatch_allocation_cost);

        size_t trimmed_deinit_extndst_dispatch_sz               = std::min(DEINIT_EXTNDST_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_extndst_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_extndst_dispatcher, trimmed_deinit_extndst_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_extndst_allocation(virtual_deinit_extndst_dispatch_allocation_cost);

        size_t trimmed_deinit_extndsx_dispatch_sz               = std::min(DEINIT_EXTNDSX_DISPATCH_DELIVERY_CAP, sz);
        size_t virtual_deinit_extndsx_dispatch_allocation_cost  = dg::network_producer_consumer::delvrsrv_allocation_cost(&virtual_deinit_extndsx_dispatcher, trimmed_deinit_extndsx_dispatch_sz);
        dg::network_stack_allocation::NoExceptRawAllocation<char[]> virtual_deinit_extndsx_allocation(virtual_deinit_extndsx_dispatch_allocation_cost);

        //

        auto init_leaf_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_leaf_dispatcher, trimmed_init_leaf_dispatch_sz, virtual_init_leaf_allocation.get()));
        auto init_blkr_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_blkr_dispatcher, trimmed_init_blkr_dispatch_sz, virtual_init_blkr_allocation.get()));
        auto init_mono_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_mono_dispatcher, trimmed_init_mono_dispatch_sz, virtual_init_mono_allocation.get()));
        auto init_pair_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_pair_dispatcher, trimmed_init_pair_dispatch_sz, virtual_init_pair_allocation.get()));
        auto init_uacm_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_uacm_dispatcher, trimmed_init_uacm_dispatch_sz, virtual_init_uacm_allocation.get()));
        auto init_pacm_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_pacm_dispatcher, trimmed_init_pacm_dispatch_sz, virtual_init_pacm_allocation.get()));
        auto init_crit_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_crit_dispatcher, trimmed_init_crit_dispatch_sz, virtual_init_crit_allocation.get()));
        auto init_immu_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_immu_dispatcher, trimmed_init_immu_dispatch_sz, virtual_init_immu_allocation.get()));
        auto init_poly_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_poly_dispatcher, trimmed_init_poly_dispatch_sz, virtual_init_immu_allocation.get()));        
        auto init_msgrfwd_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_msgrfwd_dispatcher, trimmed_init_msgrfwd_dispatch_sz, virtual_init_msgrfwd_allocation.get()));
        auto init_msgrbwd_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_msgrbwd_dispatcher, trimmed_init_msgrbwd_dispatch_sz, virtual_init_msgrbwd_allocation.get()));
        auto init_extnsrc_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_extnsrc_dispatcher, trimmed_init_extnsrc_dispatch_sz, virtual_init_extnsrc_allocation.get()));
        auto init_extnsrx_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_extnsrx_dispatcher, trimmed_init_extnsrx_dispatch_sz, virtual_init_extnsrx_allocation.get()));
        auto init_extndst_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_extndst_dispatcher, trimmed_init_extndst_dispatch_sz, virtual_init_extndst_allocation.get()));
        auto init_extndsx_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_init_extndsx_dispatcher, trimmed_init_extndsx_dispatch_sz, virtual_init_extndsx_allocation.get()));

        auto orphan_leaf_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_leaf_dispatcher, trimmed_orphan_leaf_dispatch_sz, virtual_orphan_leaf_allocation.get()));
        auto orphan_blkr_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_blkr_dispatcher, trimmed_orphan_blkr_dispatch_sz, virtual_orphan_blkr_allocation.get()));
        auto orphan_mono_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_mono_dispatcher, trimmed_orphan_mono_dispatch_sz, virtual_orphan_mono_allocation.get()));
        auto orphan_pair_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_pair_dispatcher, trimmed_orphan_pair_dispatch_sz, virtual_orphan_pair_allocation.get()));
        auto orphan_uacm_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_uacm_dispatcher, trimmed_orphan_uacm_dispatch_sz, virtual_orphan_uacm_allocation.get()));
        auto orphan_pacm_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_pacm_dispatcher, trimmed_orphan_pacm_dispatch_sz, virtual_orphan_pacm_allocation.get()));
        auto orphan_crit_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_crit_dispatcher, trimmed_orphan_crit_dispatch_sz, virtual_orphan_crit_allocation.get()));
        auto orphan_immu_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_immu_dispatcher, trimmed_orphan_immu_dispatch_sz, virtual_orphan_immu_allocation.get()));
        auto orphan_poly_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_poly_dispatcher, trimmed_orphan_poly_dispatch_sz, virtual_orphan_immu_allocation.get()));        
        auto orphan_msgrfwd_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_msgrfwd_dispatcher, trimmed_orphan_msgrfwd_dispatch_sz, virtual_orphan_msgrfwd_allocation.get()));
        auto orphan_msgrbwd_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_msgrbwd_dispatcher, trimmed_orphan_msgrbwd_dispatch_sz, virtual_orphan_msgrbwd_allocation.get()));
        auto orphan_extnsrc_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_extnsrc_dispatcher, trimmed_orphan_extnsrc_dispatch_sz, virtual_orphan_extnsrc_allocation.get()));
        auto orphan_extnsrx_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_extnsrx_dispatcher, trimmed_orphan_extnsrx_dispatch_sz, virtual_orphan_extnsrx_allocation.get()));
        auto orphan_extndst_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_extndst_dispatcher, trimmed_orphan_extndst_dispatch_sz, virtual_orphan_extndst_allocation.get()));
        auto orphan_extndsx_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_orphan_extndsx_dispatcher, trimmed_orphan_extndsx_dispatch_sz, virtual_orphan_extndsx_allocation.get()));

        auto deinit_leaf_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_leaf_dispatcher, trimmed_deinit_leaf_dispatch_sz, virtual_deinit_leaf_allocation.get()));
        auto deinit_blkr_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_blkr_dispatcher, trimmed_deinit_blkr_dispatch_sz, virtual_deinit_blkr_allocation.get()));
        auto deinit_mono_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_mono_dispatcher, trimmed_deinit_mono_dispatch_sz, virtual_deinit_mono_allocation.get()));
        auto deinit_pair_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_pair_dispatcher, trimmed_deinit_pair_dispatch_sz, virtual_deinit_pair_allocation.get()));
        auto deinit_uacm_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_uacm_dispatcher, trimmed_deinit_uacm_dispatch_sz, virtual_deinit_uacm_allocation.get()));
        auto deinit_pacm_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_pacm_dispatcher, trimmed_deinit_pacm_dispatch_sz, virtual_deinit_pacm_allocation.get()));
        auto deinit_crit_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_crit_dispatcher, trimmed_deinit_crit_dispatch_sz, virtual_deinit_crit_allocation.get()));
        auto deinit_immu_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_immu_dispatcher, trimmed_deinit_immu_dispatch_sz, virtual_deinit_immu_allocation.get()));
        auto deinit_poly_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_poly_dispatcher, trimmed_deinit_poly_dispatch_sz, virtual_deinit_immu_allocation.get()));        
        auto deinit_msgrfwd_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_msgrfwd_dispatcher, trimmed_deinit_msgrfwd_dispatch_sz, virtual_deinit_msgrfwd_allocation.get()));
        auto deinit_msgrbwd_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_msgrbwd_dispatcher, trimmed_deinit_msgrbwd_dispatch_sz, virtual_deinit_msgrbwd_allocation.get()));
        auto deinit_extnsrc_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_extnsrc_dispatcher, trimmed_deinit_extnsrc_dispatch_sz, virtual_deinit_extnsrc_allocation.get()));
        auto deinit_extnsrx_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_extnsrx_dispatcher, trimmed_deinit_extnsrx_dispatch_sz, virtual_deinit_extnsrx_allocation.get()));
        auto deinit_extndst_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_extndst_dispatcher, trimmed_deinit_extndst_dispatch_sz, virtual_deinit_extndst_allocation.get()));
        auto deinit_extndsx_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&virtual_deinit_extndsx_dispatcher, trimmed_deinit_extndsx_dispatch_sz, virtual_deinit_extndsx_allocation.get()));

        //it's incredibly hard to be able to push 100MM virtual dispatches/ second * core
        //we had to invent our own invention of "radix sort" of dispatches
        //our own locking routine, the memregion lock size needs to be reasonable, and all the operations in the duration must be reasonable (this is prescisely why the forward + backward dispatch operations must be known beforehand and carefully tuned -> a specific number)
        //we need to try_acquire at most 3 times to not pollute the atomic cmpexch instructions

        //we have considered every possible scenerios of batching
        //from dynamic size of specific dispatch -> fixed size of specific dispatch -> vectorization of blks

        //we have come to the conclusion that Poly type would solve most of the persisting problems that we haven't solved here YET
        //msgrfwd + msgrbwd + init_leaf + init_immu need special aggregation sizes to offset the cost of accessing the filesystem data
        //and it's incredibly complicated

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            const VirtualPayLoad * dispatching_payload  = std::next(payload_arr, i);
            auto payload_kind                           = dispatching_payload->kind;
            exception_t * cur_exception                 = std::next(exception_arr, i);

            if (!is_valid_payload_polymorphic_header(payload_kind)){
                exception_arr[i] = dg::network_exception::INVALID_FORMAT;
                continue;
            }

            auto delvrsrv_payload                       = std::make_pair(dispatching_payload, cur_exception);  

            switch (payload_kind){
                case payload_kind_init_leaf:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_leaf_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_blkr:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_blkr_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_mono:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_mono_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_pair:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_pair_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_uacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_uacm_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_pacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_pacm_delivery_handle.get(), sdelvrsrv_payload);
                    break;
                }
                case payload_kind_init_crit:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_crit_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_immu:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_immu_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_poly:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_poly_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_msgrfwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_msgrfwd_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_msgrbwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_msgrbwd_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_extnsrc:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_extnsrc_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_extnsrx:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_extnsrx_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_extndst:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_extndst_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_init_extndsx:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_extndsx_delivery_handle.get(), delvrsrv_payload);
                    break;
                };
                case payload_kind_orphan_leaf:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_leaf_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_blkr:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_blkr_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_mono:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_mono_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_pair:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_pair_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_uacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_uacm_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_pacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_pacm_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_crit:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_crit_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_immu:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_immu_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_poly:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_poly_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_msgrfwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_msgrfwd_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_msgrbwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_msgrbwd_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_extnsrc:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_extnsrc_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_extnsrx:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_extnsrx_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_extndst:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_extndst_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_orphan_extndsx:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_extndsx_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_leaf:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_leaf_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_blkr:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_blkr_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_mono:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_mono_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_pair:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_pair_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_uacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_uacm_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_pacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_pacm_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_poly:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_poly_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_msgrfwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_msgrfwd_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_msgrbwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_msgrbwd_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_extnsrc:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_extnsrc_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_extnsrx:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_extnsrx_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_extndst:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_extndst_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                case payload_kind_deinit_extndsx:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_extndsx_delivery_handle.get(), delvrsrv_payload);
                    break;
                }
                default:
                {
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }
            }
        }
    }
}

#endif
