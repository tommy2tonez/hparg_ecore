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
    using ClientDeliveryInfo    = dg::network_tile_metadata::ClientDeliveryInfo;
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, forward_ops_id, backward_ops_id, memevent_ops_id, logit_value, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, lhs, rhs, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, lhs, rhs, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, learning_rate, clogit_value, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, learning_rate, clogit_value, observer_vec, force_init_flag);
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, forward_ops_id, backward_ops_id, memevent_ops_id, logit_value, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, dst_info, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, dst_info, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, counterpart, counterpart_shadow, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, counterpart, counterpart_shadow, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }
    };

    struct InitExtnSrxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id, signal_accum_addr, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, counterpart, counterpart_shadow, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, counterpart, counterpart_shadow, forward_ops_id, backward_ops_id, memevent_ops_id, signal_accum_addr, dispatch_control, observer_vec, force_init_flag);
        }
    };

    struct InitExtnDsxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;
        std::optional<uma_ptr_t> signal_accum_addr;
        bool force_init_flag;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id, signal_accum_addr, force_init_flag);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id, signal_accum_addr, force_init_flag);
        }
    };

    struct OrphanLeafPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanBlkrPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanMonoPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id; 

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanPairPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanUACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanPACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanCritPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanImmuPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanMsgrFwdPayLoad{
        uma_ptr_t ptr; 
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanMsgrBwdPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanExtnSrcPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanExtnSrxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;
        
        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanExtnDstPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct OrphanExtnDsxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitLeafPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitBlkrPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitMonoPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitPairPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitUACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitPACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitCritPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitImmuPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitMsgrFwdPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitMsgrBwdPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitExtnSrcPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitExtnSrxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitExtnDstPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, memevent_ops_id);
        }
    };

    struct DeinitExtnDsxPayLoad{
        uma_ptr_t ptr;
        operatable_id_t memevent_ops_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, memevent_ops_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
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

    void load_orphan_crit_payload(std::move_iterator<OrphanCritPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t LOCAL_VECTORIZATION_SZ = size_t{1} << 8;
s
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

    void load_orphan_msgrfwd_payload(std::move_iterator<OrphanMsgrFwdPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

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
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload, exception_ptr);
        }
    }

    //---

    void load_orphan_msgrbwd_payload(std::move_iterator<OrphanMsgrBwdPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanMsgrBwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_msgrbwd(payload.ptr, payload.group_operatable_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanMsgrBwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanMsgrBwdPayLoad payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_adr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_extnsrc_payload(std::move_iterator<OrphanExtnSrcPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanExtnSrcPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_extnsrc(payload.ptr, payload.group_operatable_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanExtnSrcPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanExtnSrcPayload payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_extndst_payload(std::move_iterator<OrphanExtnDstPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanExtnDstPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_extndst(payload.ptr, payload.group_operatable_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanExtnDstPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanExtnDstPayLoad payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tule(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_immu_payload(std::move_iterator<OrphanImmuPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanImmuPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t orphan_status       = dg::network_tile_lifetime::concurrent_unsafe::orphan_immu(payload.ptr, payload.group_operatable_id);

                if (dg::network_exception::is_failed(orphan_status)){
                    *exception_ptr = orphan_status;
                }
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanImmuPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_memops_uma::delvrsrv_regionkv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_memops_uma::delvrsrv_regionkv_open_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            OrphanImmuPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_memops_uma::delvrsrv_regionkv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_leaf_payload(std::move_iterator<DeinitLeafPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};

        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitLeafPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_leaf(payload.ptr, payload.group_operatable_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitLeafPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitLeafPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr(payload.ptr)
            
            if (!rcu_addr.has_value());{
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_mono_payload(std::move_iterator<DeinitMonoPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};

        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMonoPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_mono(payload.ptr, payload.group_operatable_id);
                
                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitMonoPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitMonoPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }
            
            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_pair_payload(std::move_iterator<DeinitPairPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};

        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitPairPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_pair(payload.ptr, payload.group_operatable_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitPairPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitPairPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_uacm_payload(std::move_iterator<DeinitUACMPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};

        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitUACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size()); 

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_uacm(payload.ptr, payload.group_operatable_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitUACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitUACMPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            } 

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_pacm_payload(std::move_iterator<DeinitPACMPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};

        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitPACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());
            
            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_pacm(payload.ptr, payload.group_operatable_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitPACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitPACMPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_crit_payload(std::move_iterator<DeinitCritPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};

        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitCritPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_crit(payload.ptr, payload.group_operatable_id);

                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitCritPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitCritPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_msgrfwd_payload(std::move_iterator<DeinitMsgrFwdPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};

        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMsgrFwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_msgrfwd(payload.ptr, payload.group_operatable_id);
                
                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitMsgrFwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitMsgrFwdPayLoad payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_msgrbwd_payload(std::move_iterator<DeinitMsgrBwdPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};

        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMsgrBwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_msgrbwd(payload.ptr, payload.group_operatable_id);
                
                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitMsgrBwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitMsgrBwdPayLoad payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_extnsrc_payload(std::move_iterator<DeinitExtnSrcPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};

        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitExtnSrcPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_extnsrc(payload.ptr, payload.group_operatable_id);
                
                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitExtnSrcPayLoad, exception-t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitExtnSrcPayLoad payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_extndst_payload(std::move_iterator<DeinitExtnDstPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};

        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitExtnDstPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_extndst(payload.ptr, payload.group_operatable_id);
                
                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitExtnDstPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitExtnDstPayLoad payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr(payload.dst);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_immu_payload(std::move_iterator<DeinitImmuPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ                           = size_t{1} << 8;
        const std::array<memory_advise_t, 5u> MEM_ADVISE_SET    = {dg::network_uma::MEMORY_PLATFORM_CUTF, dg::network_uma::MEMORY_PLATFORM_CUFS, 
                                                                   dg::network_uma::MEMORY_PLATFORM_FSYS, dg::network_uma::MEMORY_PLATFORM_HOST,
                                                                   dg::network_uma::MEMORY_PLATFORM_CUDA};
        
        auto vectrz = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitImmuPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);
            exception_t advise_err = dg::network_uma::affined_advise_memory_platform(MEM_ADVISE_SET.data(), MEM_ADVISE_SET.size());

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr]  = payload_arr[i];
                exception_t deinit_status       = dg::network_tile_lifetime::concurrent_unsafe::deinit_immu(payload.ptr, payload.group_operatable_id);
                
                if (dg::network_exception::is_failed(deinit_status)){
                    *exception_ptr = deinit_status;
                }
            }

            if (dg::network_exception::is_success(advise_err)){
                dg::network_uma::affined_unadvise_memory_platform();
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitImmuPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

        for (size_t i = 0u; i < sz; ++i){
            DeinitImmuPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr(payload.dst);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
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
        payload_kind_init_msgrfwd       = 8u,
        payload_kind_init_msgrbwd       = 9u,
        payload_kind_init_extnsrc       = 10u,
        payload_kind_init_extndst       = 11u,
        payload_kind_orphan_leaf        = 12u,
        payload_kind_orphan_blkr        = 13u,
        payload_kind_orphan_mono        = 14u,
        payload_kind_orphan_pair        = 15u,
        payload_kind_orphan_uacm        = 16u,
        payload_kind_orphan_pacm        = 17u,
        payload_kind_orphan_crit        = 18u,
        payload_kind_orphan_immu        = 19u,
        payload_kind_orphan_msgrfwd     = 20u,
        payload_kind_orphan_msgrbwd     = 21u,
        payload_kind_orphan_extnsrc     = 22u,
        payload_kind_orphan_extndst     = 23u,
        payload_kind_deinit_leaf        = 24u,
        payload_kind_deinit_blkr        = 25u,
        payload_kind_deinit_mono        = 26u,
        payload_kind_deinit_pair        = 27u,
        payload_kind_deinit_uacm        = 28u,
        payload_kind_deinit_pacm        = 29u,
        payload_kind_deinit_crit        = 30u,
        payload_kind_deinit_immu        = 31u
        payload_kind_deinit_msgrfwd     = 32u,
        payload_kind_deinit_msgrbwd     = 33u,
        payload_kind_deinit_extnsrc     = 34u,
        payload_kind_deinit_extndst     = 35u
    };

    //static inline constexpr size_t VIRTUAL_PAYLOAD_CONTENT_SZ = size_t{1} << 5;
    //we probably will relax the no-exceptability of memory allocations - but for now - let's assume that memory allocations aren't errors
    //reality shows that aborting programs on memory exhaustion is better in most cases

    struct VirtualPayLoad{
        payload_kind_t kind;
        dg::sstring content; //we are using fat string - relies heavily on the stack buffers - because concurrency is very memory sensitive - especially host concurrency - where we want to minimize the memory operations and relies on CPU flops

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(kind, content);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(kind, content);
        }
    };

    auto virtualize_payload(InitLeafPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_leaf;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload); 

        return rs;
    }

    auto virtualize_payload(InitBlkrPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_blkr;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(InitMonoPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_mono;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(InitPairPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_pair;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(InitUACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_uacm;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(InitPACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_pacm;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(InitCritPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_crit;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);
    
        return rs;
    }
    
    auto virtualize_payload(InitImmuPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_immu;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(InitMsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_msgrfwd;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(InitMsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_msgrbwd;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(InitExtnSrcPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_extnsrc;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(InitExtnDstPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_init_extndst;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanLeafPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_leaf;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanBlkrPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_blkr;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanMonoPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_mono;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanPairPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_pair;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanUACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_uacm;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanPACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_pacm;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanCritPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_crit;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanMsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_msgrfwd;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanMsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_msgrbwd;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanExtnSrcPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_extnsrc;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanExtnDstPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_extndst;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(OrphanImmuPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_orphan_immu;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitLeafPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_leaf;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitBlkrPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_blkr;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitMonoPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_mono;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitPairPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_pair;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitUACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_uacm;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitPACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_pacm;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitCritPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_crit;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitImmuPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_immu;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitMsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_msgrfwd;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitMsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_msgrbwd;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitExtnSrcPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_extnsrc;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto virtualize_payload(DeinitExtnDstPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.kind     = payload_kind_deinit_extndst;
        rs.content  = dg::network_compact_serializer::serialize<dg::sstring>(payload);

        return rs;
    }

    auto devirtualize_init_leaf_payload(VirtualPayLoad payload) noexcept -> std::expected<InitLeafPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_leaf){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitLeafPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_blkr_payload(VirtualPayload payload) noexcept -> std::expected<InitBlkrPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_blkr){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitBlkrPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_mono_payload(VirtualPayLoad payload) noexcept -> std::expected<InitMonoPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_mono){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitMonoPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_pair_payload(VirtualPayLoad payload) noexcept -> std::expected<InitPairPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_pair){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitPairPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_uacm_payload(VirtualPayLoad payload) noexcept -> std::expected<InitUACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_uacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitUACMPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_pacm_payload(VirtualPayLoad payload) noexcept -> std::expected<InitPACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_pacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitPACMPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_crit_payload(VirtualPayLoad payload) noexcept -> std::expected<InitCritPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_crit){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitCritPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    } 

    auto devirtualize_init_immu_payload(VirtualPayLoad payload) noexcept -> std::expected<InitImmuPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_immu){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitImmuPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_msgrfwd_payload(VirtualPayLoad payload) noexcept -> std::expected<InitMsgrFwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_msgrfwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitMsgrFwdPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_msgrbwd_payload(VirtualPayLoad payload) noexcept -> std::expected<InitMsgrBwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_msgrbwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitMsgrBwdPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_extnsrc_payload(VirtualPayLoad payload) noexcept -> std::expected<InitExtnSrcPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_extnsrc){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitExtnSrcPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_extndst_payload(VirtualPayLoad payload) noexcept -> std::expected<InitExtnDstPayLoad, exception_t>{

        if (payload.kind != payload_kind_init_extndst){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        InitExtnDstPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    //

    auto devirtualize_orphan_leaf_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanLeafPayLoad, exception_t>{
        
        if (payload.kind != payload_kind_orphan_leaf){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanLeafPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_blkr_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanBlkrPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_blkr){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanBlkrPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_mono_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanMonoPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_mono){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanMonoPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_pair_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanPairPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_pair){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanPairPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_uacm_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanUACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_uacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanUACMPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_pacm_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanPACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_pacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanPACMPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_crit_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanCritPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_crit){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanCritPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_immu_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanImmuPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_immu){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanImmuPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_msgrfwd_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanMsgrFwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_msgrfwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanMsgrFwdPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_msgrbwd_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanMsgrBwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_msgrbwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanMsgrBwdPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_extnsrc_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanExtnSrcPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_extnsrc){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanExtnSrcPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_extndst_payload(VirtualPayLoad payload) noexcept -> std::expected<OrphanExtnDstPayLoad, exception_t>{

        if (payload.kind != payload_kind_orphan_extndst){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        OrphanExtnDstPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    //

    auto devirtualize_deinit_leaf_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitLeafPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_leaf){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitLeafPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_deinit_blkr_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitBlkrPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_blkr){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitBlkrPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_deinit_mono_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitMonoPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_mono){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitMonoPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_deinit_pair_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitPairPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_pair){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitPairPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_deinit_uacm_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitUACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_uacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitUACMPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());
        
        return rs;
    }

    auto devirtualize_deinit_pacm_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitPACMPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_pacm){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitPACMPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_deinit_crit_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitCritPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_crit){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitCritPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_deinit_immu_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitImmuPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_immu){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitImmuPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_deinit_msgrfwd_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitMsgrFwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_msgrfwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitMsgrFwdPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_deinit_msgrbwd_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitMsgrBwdPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_msgrbwd){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitMsgrBwdPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_deinit_extnsrc_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitExtnSrcPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_extnsrc){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitExtnSrcPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_deinit_extndst_payload(VirtualPayLoad payload) noexcept -> std::expected<DeinitExtnDstPayLoad, exception_t>{

        if (payload.kind != payload_kind_deinit_extndst){
            return std::unexpected(dg::network_exception::BAD_FORMAT);
        }

        DeinitExtnDstPayLoad rs{};
        dg::network_compact_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    void load_virtual_payloads(std::move_iterator<VirtualPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        //exception_t * need to be exactly 1 byte to be reasonably random-accessed

        constexpr size_t DISPATCH_DELIVERY_CAP      = size_t{1} << 16;

        auto init_leaf_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitLeafPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_leaf_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_leaf_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_blkr_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitBlkrPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_blkr_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_blkr_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_mono_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitMonoPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_mono_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_mono_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_pair_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitPairPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_pair_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_pair_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_uacm_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitUACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_uacm_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_uacm_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_pacm_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitPACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_pacm_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_pacm_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_crit_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitCritPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_crit_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_crit_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_immu_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitImmuPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_immu_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_immu_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_msgrfwd_dispatcher                = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitMsgrFwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_msgrfwd_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_msgrfwd_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_msgrbwd_dispatcher                = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitMsgrBwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_msgrbwd_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_msgrbwd_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_extnsrc_dispatcher                = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitExtnSrcPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_extnsrc_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_extnsrc_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_extndst_dispatcher                = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitExtnDstPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_extndst_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_extndst_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_leaf_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanLeafPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_leaf_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_leaf_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_blkr_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanBlkrPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_blkr_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_blkr_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_mono_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanMonoPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_mono_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_mono_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_pair_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanPairPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
            
            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_pair_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_pair_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_uacm_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanUACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
           
            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_uacm_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_uacm_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_pacm_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanPACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_pacm_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_pacm_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_crit_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanCritPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_crit_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_crit_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_immu_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanImmuPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_immu_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_immu_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_msgrfwd_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanMsgrFwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_msgrfwd_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_msgrfwd_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_msgrbwd_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanMsgrBwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
     
            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_msgrbwd_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_msgrbwd_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_extnsrc_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanExtnSrcPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_extnsrc_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_extnsrc_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_extndst_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<OrphanExtnDstPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_orphan_extndst_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_extndst_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_leaf_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitLeafPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_leaf_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_leaf_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_blkr_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitBlkrPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_blkr_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_blkr_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_mono_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitMonoPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_mono_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_mono_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_pair_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitPairPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_pair_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_pair_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_uacm_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitUACMPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_uacm_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_uacm_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_pacm_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitPACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz):

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_pacm_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_pacm_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_crit_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitCritPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_crit_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_crit_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_immu_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitImmuPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_immu_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_immu_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_msgrfwd_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitMsgrFwdPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_msgrfwd_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_msgrfwd_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_msgrbwd_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitMsgrBwdPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_msgrbwd_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_msgrbwd_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_extnsrc_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitExtnSrcPayLoad[]> devirt_payload_arr(sz):
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz):

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_extnsrc_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_extnsrc_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto deinit_extndst_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<DeinitExtnDstPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_deinit_extndst_payload(std::get<0>(data_arr[i])));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_deinit_extndst_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_leaf_consumer             = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_leaf_dispatcher)>(init_leaf_dispatcher);
        auto init_blkr_consumer             = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_blkr_dispatcher)>(init_blkr_dispatcher);
        auto init_mono_consumer             = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_mono_dispatcher)>(init_mono_dispatcher);
        auto init_pair_consumer             = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_pair_dispatcher)>(init_pair_dispatcher);
        auto init_uacm_consumer             = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_uacm_dispatcher)>(init_uacm_dispatcher);
        auto init_pacm_consumer             = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_pacm_dispatcher)>(init_pacm_dispatcher);
        auto init_crit_consumer             = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_crit_dispatcher)>(init_crit_dispatcher);
        auto init_immu_consumer             = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_immu_dispatcher)>(init_immu_dispatcher);
        auto init_msgrfwd_consumer          = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_msgrfwd_dispatcher)>(init_msgrfwd_dispatcher);
        auto init_msgrbwd_consumer          = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_msgrbwd_dispatcher)>(init_msgrbwd_dispatcher);
        auto init_extnsrc_consumer          = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_extnsrc_dispatcher)>(init_extnsrc_dispatcher);
        auto init_extndst_consumer          = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(init_extndst_dispatcher)>(init_extndst_dispatcher);

        auto orphan_leaf_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_leaf_dispatcher)>(orphan_leaf_dispatcher);
        auto orphan_blkr_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_blkr_dispatcher)>(orphan_blkr_dispatcher);
        auto orphan_mono_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_mono_dispatcher)>(orphan_mono_dispatcher);
        auto orphan_pair_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_pair_dispatcher)>(orphan_pair_dispatcher);
        auto orphan_uacm_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_uacm_dispatcher)> (orphan_uacm_dispatcher);
        auto orphan_pacm_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_pacm_dispatcher)>(orphan_pacm_dispatcher);
        auto orphan_crit_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_crit_dispatcher)>(orphan_crit_dispatcher);
        auto orphan_immu_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_immu_dispatcher)>(orphan_immu_dispatcher);
        auto orphan_msgrfwd_consumer        = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_msgrfwd_dispatcher)> (orphan_msgrfwd_dispatcher);
        auto orphan_msgrbwd_consumer        = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_msgrbwd_dispatcher)>(orphan_msgrbwd_dispatcher);
        auto orphan_extnsrc_consumer        = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_extnsrc_dispatcher)>(orphan_extnsrc_dispatcher);
        auto orphan_extndst_consumer        = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(orphan_extndst_dispatcher)>(orphan_extndst_dispatcher);

        auto deinit_leaf_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_leaf_dispatcher)>(deinit_leaf_dispatcher);
        auto deinit_blkr_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_blkr_dispatcher)>(deinit_blkr_dispatcher);
        auto deinit_mono_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_mono_dispatcher)>(deinit_mono_dispatcher);
        auto deinit_pair_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_pair_dispatcher)>(deinit_pair_dispatcher);
        auto deinit_uacm_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_uacm_dispatcher)>(deinit_uacm_dispatcher);
        auto deinit_pacm_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_pacm_dispatcher)>(deinit_pacm_dispatcher);
        auto deinit_crit_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_crit_dispatcher)>(deinit_crit_dispatcher);
        auto deinit_immu_consumer           = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_immu_dispatcher)>(deinit_immu_dispatcher);
        auto deinit_msgrfwd_consumer        = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_msgrfwd_dispatcher)>(deinit_msgrfwd_dispatcher);
        auto deinit_msgrbwd_consumer        = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_msgrbwd_dispatcher)>(deinit_msgrbwd_dispatcher);
        auto deinit_extnsrc_consumer        = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_extnsrc_dispatcher)>(deinit_extnsrc_dispatcher);
        auto deinit_extndst_consumer        = dg::network_producer_consumer::LambdaWrappedConsumer<std::pair<VirtualPayLoad, exception_t *>, decltype(deinit_extndst_dispatcher)>(deinit_extndst_dispatcher);

        //alrights - we want to split interface and link these guys by char[] here
        //stack allocations is probably one of the major optimization to reduce spin_lock overheads + allow true concurrency by using affined allocations

        dg::network_stack_allocation::NoExceptAllocation<char[]> init_leaf_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_leaf_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_blkr_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_blkr_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_mono_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_mono_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_pair_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_pair_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_uacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_uacm_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_pacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_pacm_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_crit_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_crit_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_immu_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_immu_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_msgrfwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_msgrfwd_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_msgrbwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_msgrbwd_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_extnsrc_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_extnsrc_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_extndst_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_extndst_consumer, DISPATCH_DELIVERY_CAP));

        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_leaf_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_leaf_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_blkr_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_blkr_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_mono_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_mono_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_pair_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_pair_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_uacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_uacm_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_pacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_pacm_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_crit_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_crit_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_immu_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_immu_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_msgrfwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_msgrfwd_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_msgrbwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_msgrbwd_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_extnsrc_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_extnsrc_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_extndst_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_extndst_consumer, DISPATCH_DELIVERY_CAP));  

        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_leaf_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_leaf_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_blkr_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_blkr_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_mono_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_mono_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_pair_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_pair_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_uacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_uacm_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_pacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_pacm_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_crit_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_crit_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_immu_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_immu_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_msgrfwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_msgrfwd_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_msgrbwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_msgrbwd_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_extnsrc_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_extnsrc_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> deinit_extndst_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&deinit_extndst_consumer, DISPATCH_DELIVERY_CAP));

        auto init_leaf_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_leaf_consumer, DISPATCH_DELIVERY_CAP, init_leaf_allocation.get()));
        auto init_blkr_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_blkr_consumer, DISPATCH_DELIVERY_CAP, init_blkr_allocation.get()));
        auto init_mono_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_mono_consumer, DISPATCH_DELIVERY_CAP, init_mono_allocation.get()));
        auto init_pair_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_pair_consumer, DISPATCH_DELIVERY_CAP, init_pair_allocation.get()));
        auto init_uacm_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_uacm_consumer, DISPATCH_DELIVERY_CAP, init_uacm_allocation.get()));
        auto init_pacm_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_pacm_consumer, DISPATCH_DELIVERY_CAP, init_pacm_allocation.get()));
        auto init_crit_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_crit_consumer, DISPATCH_DELIVERY_CAP, init_crit_allocation.get()));
        auto init_immu_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_immu_consumer, DISPATCH_DELIVERY_CAP, init_immu_allocation.get()));
        auto init_msgrfwd_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_msgrfwd_consumer, DISPATCH_DELIVERY_CAP, init_msgrfwd_allocation.get()));
        auto init_msgrbwd_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_msgrbwd_consumer, DISPATCH_DELIVERY_CAP, init_msgrbwd_allocation.get()));
        auto init_extnsrc_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_extnsrc_consumer, DISPATCH_DELIVERY_CAP, init_extnsrc_allocation.get()));
        auto init_extndst_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_extndst_consumer, DISPATCH_DELIVERY_CAP, init_extndst_allocation.get()));

        auto orphan_leaf_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_leaf_consumer, DISPATCH_DELIVERY_CAP, orphan_leaf_allocation.get()));
        auto orphan_blkr_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_blkr_consumer, DISPATCH_DELIVERY_CAP, orphan_blkr_allocation.get()));
        auto orphan_mono_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_mono_consumer, DISPATCH_DELIVERY_CAP, orphan_mono_allocation.get()));
        auto orphan_pair_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_pair_consumer, DISPATCH_DELIVERY_CAP, orphan_pair_allocation.get()));
        auto orphan_uacm_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_uacm_consumer, DISPATCH_DELIVERY_CAP, orphan_uacm_allocation.get()));
        auto orphan_pacm_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_pacm_consumer, DISPATCH_DELIVERY_CAP, orphan_pacm_allocation.get()));
        auto orphan_crit_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_crit_consumer, DISPATCH_DELIVERY_CAP, orphan_crit_allocation.get()));
        auto orphan_immu_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_immu_consumer, DISPATCH_DELIVERY_CAP, orphan_immu_allocation.get()));
        auto orphan_msgrfwd_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_msgrfwd_consumer, DISPATCH_DELIVERY_CAP, orphan_msgrfwd_allocation.get()));
        auto orphan_msgrbwd_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_msgrbwd_consumer, DISPATCH_DELIVERY_CAP, orphan_msgrbwd_allocation.get()));
        auto orphan_extnsrc_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_extnsrc_consumer, DISPATCH_DELIVERY_CAP, orphan_extnsrc_allocation.get()));
        auto orphan_extndst_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_extndst_consumer, DISPATCH_DELIVERY_CAP, orphan_extndst_allocation.get()));

        auto deinit_leaf_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_leaf_consumer, DISPATCH_DELIVERY_CAP, deinit_leaf_allocation.get()));
        auto deinit_blkr_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_blkr_consumer, DISPATCH_DELIVERY_CAP, deinit_blkr_allocation.get()));
        auto deinit_mono_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_mono_consumer, DISPATCH_DELIVERY_CAP, deinit_mono_allocation.get()));
        auto deinit_pair_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_pair_consumer, DISPATCH_DELIVERY_CAP, deinit_pair_allocation.get()));
        auto deinit_uacm_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_uacm_consumer, DISPATCH_DELIVERY_CAP, deinit_uacm_allocation.get()));
        auto deinit_pacm_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_pacm_consumer, DISPATCH_DELIVERY_CAP, deinit_pacm_allocation.get()));
        auto deinit_crit_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_crit_consumer, DISPATCH_DELIVERY_CAP, deinit_crit_allocation.get()));
        auto deinit_immu_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_immu_consumer, DISPATCH_DELIVERY_CAP, deinit_immu_allocation.get()));
        auto deinit_msgrfwd_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_msgrfwd_consumer, DISPATCH_DELIVERY_CAP, deinit_msgrfwd_allocation.get()));
        auto deinit_msgrbwd_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_msgrbwd_consumer, DISPATCH_DELIVERY_CAP, deinit_msgrbwd_allocation.get()));
        auto deinit_extnsrc_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_extnsrc_consumer, DISPATCH_DELIVERY_CAP, deinit_extnsrc_allocation.get()));
        auto deinit_extndst_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&deinit_extndst_consumer, DISPATCH_DELIVERY_CAP, deinit_extndst_allocation.get()));

        //we'll fix switch case later

        for (size_t i = 0u; i < sz; ++i){
            VirtualPayLoad dispatching_payload  = payload_arr[i];
            auto payload_kind                   = dispatching_payload.kind;
            exception_t * cur_exception         = std::next(exception_arr, i);

            switch (payload_kind){
                case payload_kind_init_leaf:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_leaf_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_blkr:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_blkr_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_mono:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_mono_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_pair:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_pair_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_uacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_uacm_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_pacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_pacm_delivery_handle.get(), sstd::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_crit:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_crit_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_immu:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_immu_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_msgrfwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_msgrfwd_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_msgrbwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_msgrbwd_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_extnsrc:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_extnsrc_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_init_extndst:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(init_extndst_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_leaf:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_leaf_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_blkr:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_blkr_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_mono:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_mono_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_pair:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_pair_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_uacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_uacm_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_pacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_pacm_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_crit:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_crit_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_immu:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_immu_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_msgrfwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_msgrfwd_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_msgrbwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_msgrbwd_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_extnsrc:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_extnsrc_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_orphan_extndst:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_extndst_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_deinit_leaf:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_leaf_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_deinit_blkr:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_blkr_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_deinit_mono:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_mono_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_deinit_pair:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_pair_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_deinit_uacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_uacm_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_deinit_pacm:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_pacm_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_deinit_msgrfwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_msgrfwd_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_deinit_msgrbwd:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_msgrbwd_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_deinit_extnsrc:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_extnsrc_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                }
                case payload_kind_deinit_extndst:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(deinit_extndst_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
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
