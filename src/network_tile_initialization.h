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

//alrights - we'll implement this tomorrow

namespace dg::network_tile_lifetime::concurrent_unsafe{

    using uma_ptr_t                 = dg::network_pointer::uma_ptr_t; 
    using group_operatable_id_t     = dg::network_tile_metadata::group_operatable_id_t; //sounds weird but it should be group_operatable_id_t not operatable_group_id_t - we want topology of things here
    using dispatch_control_t        = dg::network_tile_metadata::dispatch_control_t;
    using crit_kind_t               = dg::network_tile_metadata::crit_kind_t;
    using dst_info_t                = dg::network_tile_metadata::dst_info_t;
    using timein_t                  = dg::network_tile_metadata::timein_t;

    static inline constexpr UACM_ACM_SZ = dg::network_tile_metadata::UACM_ACM_SZ;
    static inline constexpr PACM_ACM_SZ = dg::network_tile_metadata::PACM_ACM_SZ;

    auto init_leaf(uma_ptr_t ptr, 
                   group_operatable_id_t group_operatable_id, 
                   void * logit_value, uint64_t logit_value_sz) noexcept -> exception_t{

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        if (exception_t err = check_leaf_logit(logit_value, logit_value_sz); dg::network_exception::is_failed(err)){
            return err;
        }

        dg::network_tile_member_getsetter::set_leaf_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_leaf_logit_nothrow(ptr, logit_value, logit_value_sz);
        dg::network_tile_member_getsetter::set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_INITIALIZED);
        dg::network_tile_member_getsetter::set_leaf_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_blkr(uma_ptr_t ptr, 
                   uma_ptr_t src, 
                   dispatch_control_t dispatch_control, 
                   group_operatable_id_t group_operatable_id, 
                   uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> exception_t{

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
        dg::network_tile_member_getsetter::set_blkr_dispatch_control_nothrow(ptr, dispatch_control);
        dg::network_tile_member_getsetter::set_blkr_group_operatable_id_nothrow(ptr, group_operatable_id);
        dg::network_tile_member_getsetter::set_blkr_observer_array_nothrow(ptr, observer_arr, observer_arr_sz);
        dg::network_tile_member_getsetter::set_blkr_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);

        return dg::network_exception::SUCCESS;
    }

    auto init_mono(uma_ptr_t ptr, 
                   uma_ptr_t src, 
                   dispatch_control_t dispatch_control, 
                   group_operatable_id_t group_operatable_id, 
                   uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> exception_t{

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

    auto init_pair(uma_ptr_t ptr, 
                   uma_ptr_t lhs, uma_ptr_t rhs, 
                   dispatch_control_t dispatch_control, 
                   group_operatable_id_t group_operatable_id, 
                   uma_ptr_t * observer_arr, uint64_t observer_arr_sz, 
                   pong_count_t pong_count) noexcept -> exception_t{

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
                   uma_ptr_t * src_arr, uint64_t src_arr_sz, 
                   dispatch_control_t dispatch_control, 
                   group_operatable_id_t group_operatable_id, 
                   uma_ptr_t * observer_arr, uint64_t observer_arr_sz, 
                   pong_count_t pong_count) noexcept -> exception_t{

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
                   uma_ptr_t * lhs_arr, uma_ptr_t * rhs_arr, uint64_t acm_sz, 
                   dispatch_control_t dispatch_control, 
                   group_operatable_id_t group_operatable_id, 
                   uma_ptr_t * observer_arr, uint64_t observer_arr_sz, 
                   pong_count_t pong_count) noexcept -> exception_t{

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
                   dispatch_control_t dispatch_control, 
                   group_operatable_id_t group_operatable_id, 
                   uint64_t reverse_learning_rate, 
                   void * clogit_value, uint64_t clogit_value_sz, 
                   uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> exception_t{

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

        if (exception_t err = check_crit_reverse_learning_rate(reverse_learning_rate); dg::network_exception::is_failed(err)){
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
        dg::network_tile_member_getsetter::set_crit_reverse_learning_rate_nothrow(ptr, reverse_learning_rate);
        dg::network_tile_member_getsetter::set_crit_clogit_nothrow(ptr, clogit_value, clogit_value_sz);
        dg::network_tile_member_getsetter::set_crit_observer_array_nothrow(ptr, observer_arr, observer_arr_sz);
        dg::network_tile_member_getsetter::set_crit_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        dg::network_tile_member_getsetter::set_crit_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        dg::network_tile_member_getsetter::set_crit_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_immu(uma_ptr_t ptr, 
                   group_operatable_id_t group_operatable_id, 
                   void * logit_value, uint64_t logit_value_sz) noexcept -> exception_t{

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
                      dispatch_control_t dispatch_control, 
                      group_operatable_id_t group_operatable_id,
                      operatable_id_t operatable_id, 
                      dst_info_t dst_info, 
                      uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> exception_t{

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
                      dispatch_control_t dispatch_control, 
                      group_operatable_id_t group_operatable_id,
                      operatable_id_t operatable_id, 
                      timein_t timein, 
                      dst_info_t dst_info, 
                      uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> exception_t{

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
                      dispatch_control_t dispatch_control, 
                      group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto init_extndst(uma_ptr_t ptr, 
                      uma_ptr_t counterpart, 
                      dispatch_control_t dispatch_control, 
                      group_operatable_id_t group_operatable_id, 
                      uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> exception_t{

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

    auto orphan_leaf(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto orphan_blkr(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{
        
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

    auto orphan_mono(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto orphan_pair(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto orphan_uacm(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto orphan_pacm(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto orphan_crit(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto orphan_immu(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto orphan_msgrfwd(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto orphan_msgrbwd(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto orphan_extnsrc(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto orphan_extndst(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto deinit_leaf(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto deinit_blkr(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto deinit_mono(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto deinit_pair(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto deinit_uacm(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto deinit_pacm(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{ 

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

    auto deinit_crit(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto deinit_immu(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto deinit_msgrfwd(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto deinit_msgrbwd(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{
        
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

    auto deinit_extnsrc(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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

    auto deinit_extndst(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> exception_t{

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
}

namespace dg::network_tile_lifetime::concurrent_safe_batch{

    struct InitLeafPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;
        dg::string logit_value;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id, logit_value);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id, logit_value);
        }
    };

    struct InitMonoPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        dispatch_control_t dispatch_control;
        group_operatable_id_t group_operatable_id;
        dg::svector<uma_ptr_t> observer_arr;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, group_operatable_id, observer_arr);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, group_operatable_id, observer_arr);
        }
    };

    struct InitPairPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t lhs;
        uma_ptr_t rhs;
        dispatch_control_t dispatch_control;
        group_operatable_id_t group_operatable_id;
        dg::svector<uma_ptr_t> observer_arr;
        pong_count_t pong_count;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, lhs, rhs, dispatch_control, group_operatable_id, observer_arr, pong_count);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, lhs, rhs, dispatch_control, group_operatable_id, observer_arr, pong_count);
        }
    };

    struct InitUACMPayLoad{
        uma_ptr_t ptr;
        dg::svector<uma_ptr_t> src;
        dispatch_control_t dispatch_control;
        group_operatable_id_t group_operatable_id;
        dg::svector<uma_ptr_t> observer_arr;
        pong_count_t pong_count;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, group_operatable_id, observer_arr, pong_count);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, group_operatable_id, observer_arr, pong_count);
        }
    };

    struct InitPACMPayLoad{
        uma_ptr_t ptr;
        dg::svector<uma_ptr_t> lhs;
        dg::svector<uma_ptr_t> rhs;
        dispatch_control_t dispatch_control;
        group_operatable_id_t group_operatable_id;
        dg::svector<uma_ptr_t> observer_arr;
        pong_count_t pong_count;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, lhs, rhs, dispatch_control, group_operatable_id, observer_arr, pong_count);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, lhs, rhs, dispatch_control, group_operatable_id, observer_arr, pong_count);
        }
    };

    struct InitCritPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        dispatch_control_t dispatch_control;
        group_operatable_id_t group_operatable_id;
        uint64_t reverse_learning_rate;
        dg::string clogit_value;
        dg::svector<uma_ptr_t> observer_arr;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, group_operatable_id, reverse_learning_rate, clogit_value, observer_arr);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, group_operatable_id, reverse_learning_rate, clogit_value, observer_arr);
        }
    };

    struct InitImmuPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;
        dg::string logit_value;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id, logit_value);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id, logit_value);
        }
    };

    struct InitMsgrFwdPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        dispatch_control_t dispatch_control;
        group_operatable_id_t group_operatable_id;
        operatable_id_t operatable_id;
        dst_info_t dst_info;
        dg::svector<uma_ptr_t> observer_arr;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, group_operatable_id, operatable_id, dst_info, observer_arr);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, group_operatable_id, operatable_id, dst_info, observer_arr);
        }
    };

    struct InitMsgrBwdPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        dispatch_control_t dispatch_control;
        group_operatable_id_t group_operatable_id;
        operatable_id_t operatable_id;
        timein_t timein;
        dst_info_t dst_info;
        dg::svector<uma_ptr_t> observer_arr;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, group_operatable_id, operatable_id, timein, dst_info, observer_arr);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, group_operatable_id, operatable_id, timein, dst_info, observer_arr);
        }
    };

    struct InitExtnSrcPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        uma_ptr_t counterpart;
        dispatch_control_t dispatch_control;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, counterpart, dispatch_control, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, counterpart, dispatch_control, group_operatable_id);
        }
    };

    struct InitExtnDstPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t counterpart;
        dispatch_control_t dispatch_control;
        group_operatable_id_t group_operatable_id;
        dg::svector<uma_ptr_t> observer_arr;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, counterpart, dispatch_control, group_operatable_id, observer_arr);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, counterpart, dispatch_control, group_operatable_id, observer_arr);
        }
    };

    struct OrphanLeafPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct OrphanMonoPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id; 

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct OrphanPairPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct OrphanUACMPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct OrphanPACMPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct OrphanCritPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct OrphanMsgrFwdPayLoad{
        uma_ptr_t ptr; 
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct OrphanMsgrBwdPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct OrphanExtnSrcPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct OrphanExtnDstPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct OrphanImmuPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitLeafPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitMonoPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitPairPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitUACMPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitPACMPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitCritPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitMsgrFwdPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitMsgrBwdPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitExtnSrcPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitExtnDstPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    struct DeinitImmuPayLoad{
        uma_ptr_t ptr;
        group_operatable_id_t group_operatable_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, group_operatable_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, group_operatable_id);
        }
    };

    auto make_init_leaf_payload(uma_ptr_t ptr, 
                                group_operatable_id_t id, 
                                void * logit_value, uint64_t logit_value_sz) noexcept -> InitLeafPayLoad{

        // return InitLeafPayLoad{ptr, id, std::move(logit_value)};
    }

    auto make_init_blkr_payload(uma_ptr_t ptr,
                                uma_ptr_t src,
                                dispatch_control_t dispatch_control,
                                group_operatable_id_t group_operatable_id,
                                uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> InitBlkrPayLoad{

    }

    auto make_init_mono_payload(uma_ptr_t ptr, 
                                uma_ptr_t src, 
                                dispatch_control_t dispatch_control, 
                                group_operatable_id_t group_operatable_id, 
                                uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> InitMonoPayLoad{

        // return InitMonoPayLoad{ptr, src, dispatch_control, group_operatable_id};
    }

    auto make_init_pair_payload(uma_ptr_t ptr, 
                                uma_ptr_t lhs, uma_ptr_t rhs, 
                                dispatch_control_t dispatch_control, 
                                group_operatable_id_t group_operatable_id, 
                                uma_ptr_t * observer_arr, uint64_t observer_arr_sz,
                                pong_count_t pong_count) noexcept -> InitPairPayLoad{

        // return InitPairPayLoad{ptr, lhs, rhs, dispatch_control, group_operatable_id};
    }

    auto make_init_uacm_payload(uma_ptr_t ptr, 
                                uma_ptr_t * src, uint64_t src_sz, 
                                dispatch_control_t dispatch_control, 
                                group_operatable_id_t group_operatable_id, 
                                uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> InitUACMPayLoad{

        // return InitUACMPayLoad{ptr, src, dispatch_control, group_operatable_id};
    }

    auto make_init_pacm_payload(uma_ptr_t ptr, 
                                uma_ptr_t * lhs, uma_ptr_t * rhs, uint64_t acm_sz, 
                                dispatch_control_t dispatch_control, group_operatable_id_t group_operatable_id, 
                                uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> InitPACMPayLoad{

        // return InitPACMPayLoad{ptr, lhs, rhs, dispatch_control, group_operatable_id};
    }

    auto make_init_crit_payload(uma_ptr_t ptr, 
                                uma_ptr_t src, 
                                dispatch_control_t dispatch_control, 
                                group_operatable_id_t group_operatable_id,
                                uint64_t reverse_learning_rate, 
                                void * clogit_value, uint64_t clogit_value_sz, 
                                uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> InitCritPayLoad{

        // return InitCritPayLoad{ptr, src, dispatch_control, group_operatable_id, std::move(clogit_value)};
    }

    auto make_init_immu_payload(uma_ptr_t ptr, 
                                group_operatable_id_t group_operatable_id, 
                                void * logit_value, uint64_t logit_value_sz) noexcept -> InitImmuPayLoad{

        // return InitImmuPayLoad{ptr, group_operatable_id};
    }

    auto make_init_msgrfwd_payload(uma_ptr_t ptr, 
                                   uma_ptr_t src, 
                                   dispatch_control_t dispatch_control, 
                                   group_operatable_id_t group_operatable_id, 
                                   operatable_id_t operatable_id,
                                   dst_info_t dst_info, 
                                   uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> InitMsgrFwdPayLoad{

        // return InitMsgrFwdPayLoad{ptr, src, dispatch_control, group_operatable_id, dst_info};
    }

    auto make_init_msgrbwd_payload(uma_ptr_t ptr, 
                                   uma_ptr_t src, 
                                   dispatch_control_t dispatch_control, 
                                   group_operatable_id_t group_operatable_id, 
                                   operatable_id_t operatable_id,
                                   timein_t timein, 
                                   dst_info_t dst_info, 
                                   uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> InitMsgrBwdPayLoad{

        // return InitMsgrBwdPayLoad{ptr, src, dispatch_control, group_operatable_id, timein, dst_info};
    }

    auto make_init_extnsrc_payload(uma_ptr_t ptr, 
                                   uma_ptr_t src, 
                                   uma_ptr_t counterpart, 
                                   dispatch_control_t dispatch_control, 
                                   group_operatable_id_t group_operatable_id) noexcept -> InitExtnSrcPayLoad{

        // return InitExtnsrcPayLoad{ptr, src, counterpart, dispatch_control, group_operatable_id};
    }

    auto make_init_extndst_payload(uma_ptr_t ptr, 
                                   uma_ptr_t counterpart, 
                                   dispatch_control_t dispatch_control, 
                                   group_operatable_id_t group_operatable_id, 
                                   uma_ptr_t * observer_arr, uint64_t observer_arr_sz) noexcept -> InitExtnDstPayLoad{

        // return InitSrcDstClonePayLoad{ptr, counterpart, dispatch_control, group_operatable_id};
    }

    auto make_orphan_leaf_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanLeafPayLoad{

        // return OrphanLeafPayLoad{ptr};
    }

    auto make_orphan_mono_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanMonoPayLoad{

        // return OrphanMonoPayLoad{ptr};
    }

    auto make_orphan_pair_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanPairPayLoad{

        // return OrphanPairPayLoad{ptr};
    }

    auto make_orphan_uacm_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanUACMPayLoad{

        // return OrphanUACMPayLoad{ptr};
    }

    auto make_orphan_pacm_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanPACMPayLoad{

        // return OrphanPACMPayLoad{ptr};
    }

    auto make_orphan_crit_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanCritPayLoad{

        // return OrphanCritPayLoad{ptr};
    }

    auto make_orphan_immu_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanImmuPayLoad{

        // return OrphanImmuPayLoad{ptr};
    }

    auto make_orphan_msgrfwd_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanMsgrFwdPayLoad{

        // return OrphanMsgrFwdPayLoad{ptr};
    }

    auto make_orphan_msgrbwd_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanMsgrBwdPayLoad{

        // return OrphanMsgrBwdPayLoad{ptr};
    }

    auto make_orphan_extnsrc_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanExtnSrcPayLoad{

        // return OrphanExtnSrcPayLoad{ptr};
    }

    auto make_orphan_extndst_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> OrphanExtnDstPayLoad{

        // return OrphanExtnDstPayLoad{ptr};
    }

    auto make_deinit_leaf_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitLeafPayLoad{

        // return DeinitLeafPayLoad{ptr};
    }

    auto make_deinit_mono_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitMonoPayLoad{

        // return DeinitMonoPayLoad{ptr};
    }

    auto make_deinit_pair_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitPairPayLoad{
        
        // return DeinitPairPayLoad{ptr};
    }

    auto make_deinit_uacm_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitUACMPayLoad{

        // return DeinitUACMPayLoad{ptr};
    }

    auto make_deinit_pacm_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitPACMPayLoad{
        
        // return DeinitPACMPayLoad{ptr};
    }

    auto make_deinit_crit_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitCritPayLoad{

        // return DeinitCritPayLoad{ptr};
    }

    auto make_deinit_immu_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitImmuPayLoad{

        // return DeinitImmuPayLoad{ptr};
    }

    auto make_deinit_msgrfwd_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitMsgrFwdPayLoad{

        // return DeinitMsgrFwdPayLoad{ptr};
    }

    auto make_deinit_msgrbwd_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitMsgrBwdPayLoad{

        // return DeinitMsgrBwdPayLoad{ptr};
    }

    auto make_deinit_extnsrc_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitExtnSrcPayLoad{

        // return DeinitExtnSrcPayLoad{ptr};
    }

    auto make_deinit_extndst_payload(uma_ptr_t ptr, group_operatable_id_t group_operatable_id) noexcept -> DeinitExtnDstPayLoad{

        // return DeinitExtnDstPayLoad{ptr};
    }

    void load_init_leaf_payload(std::move_iterator<InitLeafPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitLeafPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_leaf(payload.ptr, payload.group_operatable_id, payload.logit_value.data(), payload.logit_value.size());
            }
        };

        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitLeafPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            LeafPayload payload                             = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr(payload.ptr); 

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_init_mono_payload(std::move_iterator<InitMonoPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitMonoPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_mono(payload.ptr, payload.src, payload.dispatch_control, payload.group_operatable_id, payload.observer_vec.data(), payload.observer_vec.size());
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitMonoPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            InitMonoPayLoad payload                         = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    } 

    void load_init_pair_payload(std::move_iterator<InitPairPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitPairPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_pair(payload.ptr, payload.lhs, payload.rhs, payload.dispatch_control, payload.group_operatable_id, payload.observer_vec.data(), payload.observer_vec.size(), payload.pong_count);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitPairPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            InitPairPayLoad payload                         = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_init_uacm_payload(std::move_iterator<InitUACMPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitUACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_uacm(payload.ptr, payload.src.data(), payload.src.size(), payload.dispatch_control, payload.group_operatable_id, payload.observer_vec.data(), payload.observer_vec.size(), payload.pong_count);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitUACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            InitUACMPayLoad payload                         = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_init_pacm_payload(std::move_iterator<InitPACMPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitPACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i]; 

                if (payload.lhs.size() != payload.rhs.size()){
                    *exception_ptr = dg::network_exception::INVALID_ARGUMENT;
                } else{
                    *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_pacm(payload.ptr, payload.lhs.data(), payload.rhs.data(), payload.lhs.size(), 
                                                                                             payload.dispatch_control, payload.group_operatable_id, 
                                                                                             payload.observer_vec.data(), payload.observer_vec.size());
                }
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitPACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            InitPACMPayLoad payload                         = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_init_crit_payload(std::move_iterator<InitCritPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitCritPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_crit(payload.ptr, payload.src, payload.dispatch_control, payload.group_operatable_id, payload.reverse_learning_rate, 
                                                                                         payload.clogit_value.data(), payload.clogit_value.size(), payload.observer_vec.data(), payload.observer_vec.size());
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitCritPayLoad, exception-t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            InitCritPayLoad payload                         = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_init_immu_payload(std::move_iterator<InitImmuPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitImmuPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_immu(payload.ptr, payload.group_operatable_id, payload.logit_value.data(), payload.logit_value.size());
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitImmuPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            InitImmuPayLoad payload                         = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_init_msgrfwd_payload(std::move_iterator<InitMsgrFwdPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitMsgrFwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_msgrfwd(payload.ptr, payload.src, payload.dispatch_control, payload.group_operatable_id, 
                                                                                            payload.operatable_id, payload.dst_info, 
                                                                                            payload.observer_vec.data(), payload.observer_vec.size());
            }
        };

        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitMsgrFwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            InitMsgrFwdPayLoad payload                      = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_init_msgrbwd_payload(std::move_iterator<InitMsgrBwdPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitMsgrBwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_msgrbwd(payload.ptr, payload.src, payload.dispatch_control, payload.group_operatable_id, 
                                                                                            payload.operatable_id, payload.timein, payload.dst_info, 
                                                                                            payload.observer_vec.data(), payload.observer_vec.size());
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LamdaWrappedConsumer<std::tuple<InitMsgrBwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            InitMsgrBwdPayLoad payload                      = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_init_extnsrc_payload(std::move_iterator<InitExtnSrcPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitExtnSrcPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_extnsrc(payload.ptr, payload.src, payload.counterpart, 
                                                                                            payload.dispatch_control, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LamdaWrappedConsumer<std::tuple<InitExtnSrcPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            InitExtnSrcPayLoad payload                      = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_init_extndst_payload(std::move_iterator<InitExtnDstPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitExtnDstPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_extndst(payload.ptr, payload.counterpart, 
                                                                                            payload.dispatch_control, payload.group_operatable_id,
                                                                                            payload.observer_vec.data(), payload.observer_vec.size());
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitExtnDstPayLoad, exception_t * >, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            InitExtnDstPayLoad payload                      = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_leaf_payload(std::move_iterator<OrphanLeafPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t lck_addr, std::tuple<OrphanLeafPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_leaf(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanLeafPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanLeafPayload payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_mono_payload(std::move_iterator<OrphanMonoPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t lck_addr, std::tuple<OrphanMonoPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_mono(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanMonoPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanMonoPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            } 

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_pair_payload(std::move_iterator<OrphanPairPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanPairPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_pair(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanPairPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanPairPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_uacm_payload(std::move_iterator<OrphanUACMPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanUACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_uacm(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanUACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanUACMPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_pacm_payload(std::move_iterator<OrphanPACMPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanPACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_pacm(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanPACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanPACMPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_crit_payload(std::move_iterator<OrphanCritPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanCritPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_crit(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanCritPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanCritPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_msgrfwd_payload(std::move_iterator<OrphanMsgrFwdPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanMsgrFwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_msgrfwd(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanMsgrFwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanMsgrFwdPayLoad payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size()); //this is the most important optimization - we are managing all the allocations
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_msgrbwd_payload(std::move_iterator<OrphanMsgrBwdPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanMsgrBwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_msgrbwd(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanMsgrBwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanMsgrBwdPayLoad payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_adr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
            //people load kernel with 15M LOC Mom - we'll compile files later
        }
    }

    void load_orphan_extnsrc_payload(std::move_iterator<OrphanExtnSrcPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanExtnSrcPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_extnsrc(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanExtnSrcPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanExtnSrcPayload payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_extndst_payload(std::move_iterator<OrphanExtnDstPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanExtnDstPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_extndst(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanExtnDstPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanExtnDstPayLoad payload                    = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tule(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_orphan_immu_payload(std::move_iterator<OrphanImmuPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanImmuPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_immu(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanImmuPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            OrphanImmuPayLoad payload                       = payload_arr[i];
            std::expected<uma_ptr_t, exception_t> rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr(payload.ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(std::move(payload), std::next(exception_arr, i)));
        }
    }

    void load_deinit_leaf_payload(std::move_iterator<DeinitLeafPayLoad *> payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitLeafPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_leaf(payload.ptr, payload.group_operatable_id); 
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitLeafPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMonoPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_mono(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitMonoPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitPairPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_pair(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitPairPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitUACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_uacm(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitUACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitPACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_pacm(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitPACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

        auto VECTORIZATION_SZ           = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitCritPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_crit(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitCritPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMsgrFwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_msgrfwd(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitMsgrFwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMsgrBwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_msgrbwd(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitMsgrBwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitExtnSrcPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_extnsrc(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitExtnSrcPayLoad, exception-t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitExtnDstPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_extndst(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitExtnDstPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitImmuPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto& [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_immu(payload.ptr, payload.group_operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitImmuPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::NoExceptAllocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

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

    //these aren't neccessarily invoked by clients - these are for providing a high level interface of the payload format - we'll reiterate this later
    //we want to radix payload -> fast_payload and slow_payload which are handled by two different functions to avoid performance constraints by misuse of interfaces
    //I know things dg::string needs to extend stack buffer
    //we are aiming at least 1 << 25 init orders/core * second
    //assume 64KB tiles - we are allocating 1 << 41 bytes/ core * second = 2TB/core * second - this is acceptable
    //after debating - its best to combine this into one payload kind and relies on fat package
    //we must use all-stack memory in order to maximize the concurrent speed
    //most of the time, host_concurrency is bottlenecked by ram fetch - not CPU flops - so we must be very careful fetching memory here 

    //ideally - we want to use full cuda for forwarding + backwarding
    //we only use host for msgrbwd + msgrfwd + network packets + initializations + rest packets
    //we want to hyperthread to reduce spin overheads
    //we want to use stack memory whenever possible + delivery service to reduce dispatch overheads to exactly O(n) space - n being the memory of dispatching requests
    //if we stick to this approach - I think we will be successful - in the sense of building an efficient training system

    //we are treating stack allocation like C - we must use precautions to make sure that this aint stack overflowing - or program will abort on spot
    //we aren't processing exotic exceptions in this program - we preallocate - we run - we out of memory - we abort
    //for the reason being reaching OOM in the first place is already a bug
    //if we ain't reaching OOM - then OOM should be noexcept - proof by contradiction 

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

    //hmm - this should be std::expected<, exception_t> for best practices
    //this ain't precond
    //we can't optimize this for the reason being this is not worth optimizing
    //I mean it's fine either way - but we should adhere to best practices by not DEBUG aborting things, and production aborting things

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

    //we'll implement the feature Dad - even though orphan is already sufficient

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
        //

        constexpr size_t DISPATCH_DELIVERY_CAP      = size_t{1} << 16; //config this 

        auto init_mono_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::NoExceptAllocation<InitMonoPayLoad[]> devirt_payload_arr(sz); //we'll leverage concurrency and affinity to achieve the magic - we have to disable move cpy constructors + assignments
            dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = dg::network_exception_handler::nothrow_log(devirtualize_init_mono_payload(std::get<0>(data_arr[i]))); //we know that the only exception returned is the type exception - which is already enforced so we can do nothrow_log here
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

        auto init_mono_consumer                     = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_mono_dispatcher)>(std::move(init_mono_dispatcher));
        auto init_pair_consumer                     = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_pair_dispatcher)>(std::move(init_pair_dispatcher));
        auto init_uacm_consumer                     = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_uacm_dispatcher)>(std::move(init_uacm_dispatcher));
        auto init_pacm_consumer                     = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_pacm_dispatcher)>(std::move(init_pacm_dispatcher));
        auto init_msgrfwd_consumer                  = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_msgrfwd_dispatcher)> (std::move(init_msgrfwd_dispatcher));
        auto init_msgrbwd_consumer                  = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_msgrbwd_dispatcher)>(std::move(init_msgrbwd_dispatcher));
        auto init_extnsrc_consumer                  = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_extnsrc_dispatcher)>(std::move(init_extnsrc_dispatcher));
        auto init_extndst_consumer                  = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_extndst_dispatcher)>(std::move(init_extndst_dispatcher));
        auto orphan_leaf_consumer                   = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_leaf_dispatcher)>(std::move(orphan_leaf_dispatcher));
        auto orphan_mono_consumer                   = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_mono_dispatcher)>(std::move(orphan_mono_dispatcher));
        auto orphan_pair_consumer                   = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_pair_dispatcher)>(std::move(orphan_pair_dispatcher));
        auto orphan_uacm_consumer                   = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_uacm_dispatcher)> (std::move(orphan_uacm_dispatcher));
        auto orphan_pacm_consumer                   = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_pacm_dispatcher)>(std::move(orphan_pacm_dispatcher));
        auto orphan_crit_consumer                   = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_crit_dispatcher)>(std::move(orphan_crit_dispatcher));
        auto orphan_msgrfwd_consumer                = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_msgrfwd_dispatcher)> (std::move(orphan_msgrfwd_dispatcher));
        auto orphan_msgrbwd_consumer                = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_msgrbwd_dispatcher)>(std::move(orphan_msgrbwd_dispatcher));
        auto orphan_extnsrc_consumer                = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_extnsrc_dispatcher)>(std::move(orphan_extnsrc_dispatcher));
        auto orphan_extndst_consumer                = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_extndst_dispatcher)>(std::move(orphan_extndst_dispatcher));
        auto orphan_immu_consumer                   = dg::network_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_immu_dispatcher)>(std::move(orphan_immu_dispatcher));

        //alrights - we want to split interface and link these guys by char[] here
        //stack allocations is probably one of the major optimization to reduce spin_lock overheads + allow true concurrency by using affined allocations
 
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_mono_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_mono_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_pair_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_pair_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_uacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_uacm_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_pacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_pacm_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_msgrfwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_msgrfwd_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_msgrbwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_msgrbwd_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_extnsrc_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_extnsrc_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> init_extndst_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_extndst_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_leaf_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_leaf_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_mono_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_mono_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_pair_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_pair_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_uacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_uacm_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_pacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_pacm_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_crit_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_crit_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_msgrfwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_msgrfwd_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_msgrbwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_msgrbwd_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_extnsrc_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_extnsrc_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_extndst_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_extndst_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::NoExceptAllocation<char[]> orphan_immu_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_immu_consumer, DISPATCH_DELIVERY_CAP));  

        auto init_mono_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_mono_consumer, DISPATCH_DELIVERY_CAP, init_mono_allocation.get()));
        auto init_pair_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_pair_consumer, DISPATCH_DELIVERY_CAP, init_pair_allocation.get()));
        auto init_uacm_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_uacm_consumer, DISPATCH_DELIVERY_CAP, init_uacm_allocation.get()));
        auto init_pacm_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_pacm_consumer, DISPATCH_DELIVERY_CAP, init_pacm_allocation.get()));
        auto init_msgrfwd_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_msgrfwd_consumer, DISPATCH_DELIVERY_CAP, init_msgrfwd_allocation.get()));
        auto init_msgrbwd_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_msgrbwd_consumer, DISPATCH_DELIVERY_CAP, init_msgrbwd_allocation.get()));
        auto init_extnsrc_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_extnsrc_consumer, DISPATCH_DELIVERY_CAP, init_extnsrc_allocation.get()));
        auto init_extndst_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&init_extndst_consumer, DISPATCH_DELIVERY_CAP, init_extndst_allocation.get()));
        auto orphan_leaf_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_leaf_consumer, DISPATCH_DELIVERY_CAP, orphan_leaf_allocation.get()));
        auto orphan_mono_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_mono_consumer, DISPATCH_DELIVERY_CAP, orphan_mono_allocation.get()));
        auto orphan_pair_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_pair_consumer, DISPATCH_DELIVERY_CAP, orphan_pair_allocation.get()));
        auto orphan_uacm_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_uacm_consumer, DISPATCH_DELIVERY_CAP, orphan_uacm_allocation.get()));
        auto orphan_pacm_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_pacm_consumer, DISPATCH_DELIVERY_CAP, orphan_pacm_allocation.get()));
        auto orphan_crit_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_crit_consumer, DISPATCH_DELIVERY_CAP, orphan_crit_allocation.get()));
        auto orphan_msgrfwd_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_msgrfwd_consumer, DISPATCH_DELIVERY_CAP, orphan_msgrfwd_allocation.get()));
        auto orphan_msgrbwd_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_msgrbwd_consumer, DISPATCH_DELIVERY_CAP, orphan_msgrbwd_allocation.get()));
        auto orphan_extnsrc_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_extnsrc_consumer, DISPATCH_DELIVERY_CAP, orphan_extnsrc_allocation.get()));
        auto orphan_extndst_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_extndst_consumer, DISPATCH_DELIVERY_CAP, orphan_extndst_allocation.get()));
        auto orphan_immu_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&orphan_immu_consumer, DISPATCH_DELIVERY_CAP, orphan_immu_allocation.get()));

        //we'll fix switch case later

        for (size_t i = 0u; i < sz; ++i){

            VirtualPayLoad dispatching_payload  = payload_arr[i];
            auto payload_kind                   = dispatching_payload.kind;
            exception_t * cur_exception         = std::next(exception_arr, i);

            switch (payload_kind){
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
                case payload_kind_orphan_immu:
                {
                    dg::network_producer_consumer::delvrsrv_deliver(orphan_immu_delivery_handle.get(), std::make_pair(std::move(dispatching_payload), cur_exception));
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
