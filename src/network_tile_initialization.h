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

    using uma_ptr_t             = dg::network_pointer::uma_ptr_t; 
    using operatable_id_t       = dg::network_tile_metadata::operatable_id_t;
    using dispatch_control_t    = dg::network_tile_metadata::dispatch_control_t;
    using crit_kind_t           = dg::network_tile_metadata::crit_kind_t;
    using dst_info_t            = dg::network_tile_metadata::dst_info_t;
    using timein_t              = dg::network_tile_metadata::timein_t;

    static inline constexpr UACM_ACM_SZ = dg::network_tile_metadata::UACM_ACM_SZ;
    static inline constexpr PACM_ACM_SZ = dg::network_tile_metadata::PACM_ACM_SZ;

    auto init_leaf(uma_ptr_t ptr, operatable_id_t operatable_id, void * logit_value, size_t logit_value_sz, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        size_t pointing_logit_sz = get_leaf_logit_group_size_nothrow(ptr);

        if (pointing_logit_sz != logit_value_sz){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_INITIALIZED);
        set_leaf_logit_nothrow(ptr, logit_value);
        set_leaf_observer_array_size_nothrow(ptr, 0u);
        set_leaf_operatable_id_nothrow(ptr, operatable_id);
        set_leaf_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_mono(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_mono_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_mono_observer_array_size_nothrow(ptr, 0u);
        set_mono_dispatch_control_nothrow(ptr, dispatch_control);
        set_mono_operatable_id_nothrow(ptr, operatable_id);
        set_mono_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_mono_descendant_nothrow(ptr, src);
        set_mono_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);
    
        return dg::network_exception::SUCCESS;
    }

    auto init_pair(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_pair_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_pair_observer_array_size_nothrow(ptr, 0u);
        set_pair_operatable_id_nothrow(ptr, operatable_id);
        set_pair_dispatch_control_nothrow(ptr, dispatch_control);
        set_pair_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_pair_left_descendant_nothrow(ptr, lhs);
        set_pair_right_descendant_nothrow(ptr, rhs);
        set_pair_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_uacm(uma_ptr_t ptr, std::array<uma_ptr_t, UACM_ACM_SZ> src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_uacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_uacm_observer_array_size_nothrow(ptr, 0u);
        set_uacm_operatable_id_nothrow(ptr, operatable_id);
        set_uacm_dispatch_control_nothrow(ptr, dispatch_control);
        set_uacm_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_uacm_descendant_nothrow(ptr, src);
        set_uacm_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_pacm(uma_ptr_t ptr, std::array<uma_ptr_t, PACM_ACM_SZ> lhs, std::array<uma_ptr_t, PACM_ACM_SZ> rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_pacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_pacm_observer_array_size_nothrow(ptr, 0u);
        set_pacm_operatable_id_nothrow(ptr, operatable_id);
        set_pacm_dispatch_control_nothrow(ptr, dispatch_control);
        set_pacm_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_pacm_left_descendant_nothrow(ptr, lhs);
        set_pacm_right_descendant_nothrow(ptr, rhs);
        set_pacm_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_crit(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, crit_kind_t crit_kind, void * clogit_value, size_t clogit_value_sz, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        size_t pointing_clogit_value_sz = get_crit_clogit_group_size_nothrow(ptr);

        if (pointing_clogit_value_sz != clogit_value_sz){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        set_crit_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_crit_clogit_nothrow(ptr, clogit_value);
        set_crit_observer_array_size_nothrow(ptr, 0u);
        set_crit_operatable_id_nothrow(ptr, operatable_id);
        set_crit_dispatch_control_nothrow(ptr, dispatch_control);
        set_crit_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_crit_descendant_nothrow(ptr, src);
        set_crit_kind_nothrow(ptr, crit_kind);
        set_crit_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_msgrfwd(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, dst_info_t dst_info, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_msgrfwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_msgrfwd_observer_array_size_nothrow(ptr, 0u);
        set_msgrfwd_operatable_id_nothrow(ptr, operatable_id);
        set_msgrfwd_dispatch_control_nothrow(ptr, dispatch_control);
        set_msgrfwd_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_msgrfwd_descendant_nothrow(ptr, src);
        set_msgrfwd_dst_info_nothrow(ptr, dst_info);
        set_msgrfwd_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_msgrbwd(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, timein_t timein, dst_info_t dst_info, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_msgrbwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_msgrbwd_observer_array_size_nothrow(ptr, 0u);
        set_msgrbwd_operatable_id_nothrow(ptr, operatable_id);
        set_msgrbwd_dispatch_control_nothrow(ptr, dispatch_control);
        set_msgrbwd_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_msgrbwd_descendant_nothrow(ptr, src);
        set_msgrbwd_dst_info_nothrow(ptr, dst_info);
        set_msgrbwd_timein_nothrow(ptr, timein);
        set_msgrbwd_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_extnsrc(uma_ptr_t ptr, uma_ptr_t src, uma_ptr_t counterpart, dispatch_control_t dispatch_control, operatable_id_t operatable_id, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_extnsrc_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_extnsrc_observer_array_size_nothrow(ptr, 0u);
        set_extnsrc_operatable_id_nothrow(ptr, operatable_id);
        set_extnsrc_dispatch_control_nothrow(ptr, dispatch_control);
        set_extnsrc_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_extnsrc_descendant_nothrow(ptr, src);
        set_extnsrc_counterpart_nothrow(ptr, counterpart);
        set_extnsrc_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_extndst(uma_ptr_t ptr, uma_ptr_t counterpart, dispatch_control_t dispatch_control, operatable_id_t operatable_id, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_extndst_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_extndst_observer_array_size_nothrow(ptr, 0u);
        set_extndst_operatable_id_nothrow(ptr, operatable_id);
        set_extndst_dispatch_control_nothrow(ptr, dispatch_control);
        set_extndst_counterpart_nothrow(ptr, counterpart);

        return dg::network_exception::SUCCESS;
    }

    auto init_immu(uma_ptr_t ptr, operatable_id_t operatable_id, void * logit_value, size_t logit_value_sz, uma_ptr_t * observer_arr, size_t observer_arr_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_immu_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        size_t pointing_logit_value_sz = get_immu_logit_group_size(ptr);

        if (pointing_logit_value_sz != logit_value_sz){
            return dg::network_exception::INVALID_ARGUMENT;
        } 

        set_immu_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_INITIALIZED);
        set_immu_logit_nothrow(ptr, logit_value);
        set_immu_observer_array_size_nothrow(ptr, 0u);
        set_immu_operatable_id_nothrow(ptr, operatable_id);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_leaf(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED); 

        return dg::network_exception::SUCCESS;
    }

    auto orphan_mono(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_mono_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_pair(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_pair_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_uacm(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_uacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_pacm(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_pacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_crit(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_crit_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_msgrfwd(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_msgrfwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_msgrbwd(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_msgrbwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_extnsrc(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_extnsrc_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_extndst(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_extndst_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_immu(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_immu_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_immu_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto deinit_leaf(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_EMPTY);
        set_leaf_logit_nothrow(ptr, dg::network_tile_metadata::TILE_LOGIT_VALUE_DEFAULT);
        set_leaf_grad_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_VALUE_DEFAULT);
        set_leaf_observer_array_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_leaf_observer_array_size_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_SIZE_DEFAULT);
        set_leaf_operatable_id_nothrow(ptr, dg::network_tile_metadata::TILE_OPERATABLE_ID_DEFAULT);
        set_leaf_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_EMPTY);
        
        return dg::network_exception::SUCCESS;
    }

    auto deinit_mono(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_mono_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_EMPTY);
        set_mono_logit_nothrow(ptr, dg::network_tile_metadata::TILE_LOGIT_VALUE_DEFAULT);
        set_mono_grad_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_VALUE_DEFAULT);
        set_mono_observer_array_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_mono_observer_array_size_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_SIZE_DEFAULT);
        set_mono_dispatch_control_nothrow(ptr, dg::network_tile_metadata::TILE_DISPATCH_CONTROL_DEFAULT);
        set_mono_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_mono_descendant_nothrow(ptr, dg::network_tile_metadata::TILE_ADDRESS_DEFAULT);
        set_mono_operatable_id_nothrow(ptr, dg::network_tile_metadata::TILE_OPERATABLE_ID_DEFAULT);
        set_mono_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_EMPTY);

        return dg::network_exception::SUCCESS;
    }

    auto deinit_pair(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_pair_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_EMPTY);
        set_pair_logit_nothrow(ptr, dg::network_tile_metadata::TILE_LOGIT_VALUE_DEFAULT);
        set_pair_grad_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_VALUE_DEFAULT);
        set_pair_observer_array_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_pair_observer_array_size_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_SIZE_DEFAULT);
        set_pair_dispatch_control_nothrow(ptr, dg::network_tile_metadata::TILE_DISPATCH_CONTROL_DEFAULT);
        set_pair_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_pair_left_descendant_nothrow(ptr, dg::network_tile_metadata::TILE_ADDRESS_DEFAULT);
        set_pair_right_descendant_nothrow(ptr, dg::network_tile_metadata::TILE_ADDRESS_DEFAULT);
        set_pair_operatable_id_nothrow(ptr, dg::network_tile_metadata::TILE_OPERATABLE_ID_DEFAULT);
        set_pair_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_EMPTY);
    
        return dg::network_exception::SUCCESS;
    }

    auto deinit_uacm(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_uacm_init_status_nothrow();
        set_uacm_logit_nothrow();
        set_uacm_grad_nothrow();
        set_uacm_observer_array_nothrow();
        set_uacm_observer_array_size_nothrow();
        set_uacm_operatable_id_nothrow();
        set_uacm_dispatch_control_nothrow();
        set_uacm_pong_count_nothrow();
        set_uacm_descendant_nothrow();
        set_uacm_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_pacm(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{ 

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_pacm_init_status_nothrow();
        set_pacm_logit_nothrow();
        set_pacm_grad_nothrow();
        set_pacm_observer_array_size_nothrow();
        set_pacm_operatable_id_nothrow();
        set_pacm_dispatch_control_nothrow();
        set_pacm_pong_count_nothrow();
        set_pacm_left_descendant_nothrow();
        set_pacm_right_descendant_nothrow();
        set_pacm_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_crit(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }
        
        set_crit_init_status_nothrow();
        set_crit_logit_nothrow();
        set_crit_grad_nothrow();
        set_crit_clogit_nothrow();
        set_crit_observer_array_nothrow();
        set_crit_observer_array_size_nothrow();
        set_crit_operatable_id_nothrow();
        set_crit_dispatch_control_nothrow();
        set_crit_pong_count_nothrow();
        set_crit_descendant_nothrow();
        set_crit_kind_nothrow();
        set_crit_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_msgrfwd(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_msgrfwd_init_status_nothrow();
        set_msgrfwd_logit_nothrow();
        set_msgrfwd_grad_nothrow();
        set_msgrfwd_observer_array_nothrow();
        set_msgrfwd_observer_array_size_nothrow();
        set_msgrfwd_operatable_id_nothrow();
        set_msgrfwd_dispatch_control_nothrow();
        set_msgrfwd_pong_count_nothrow();
        set_msgrfwd_descendant_nothrow();
        set_msgrfwd_dst_info_nothrow();
        set_msgrfwd_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_msgrbwd(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{
        
        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_msgrbwd_init_status_nothrow();
        set_msgrbwd_logit_nothrow();
        set_msgrbwd_grad_nothrow();
        set_msgrbwd_observer_array_nothrow();
        set_msgrbwd_observer_array_size_nothrow();
        set_msgrbwd_operatable_id_nothrow();
        set_msgrbwd_dispatch_control_nothrow();
        set_msgrbwd_pong_count_nothrow();
        set_msgrbwd_descendant_nothrow();
        set_msgrbwd_dst_info_nothrow();
        set_msgrbwd_timein_nothrow(); //
        set_msgrbwd_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_extnsrc(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_extnsrc_init_status_nothrow();
        set_extnsrc_logit_nothrow();
        set_extnsrc_grad_nothrow();
        set_extnsrc_observer_array_nothrow();
        set_extnsrc_observer_array_size_nothrow();
        set_extnsrc_operatable_id_nothrow();
        set_extnsrc_dispatch_control_nothrow();
        set_extnsrc_pong_count_nothrow();
        set_extnsrc_descendant_nothrow();
        set_extnsrc_counterpart_nothrow();
        set_extnsrc_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_extndst(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_extndst_init_status_nothrow();
        set_extndst_observer_array_nothrow();
        set_extndst_observer_array_size_nothrow();
        set_extndst_operatable_id_nothrow();
        set_extndst_dispatch_control_nothrow();
        set_extndst_counterpart_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_immu(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_immu_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_immu_init_status_nothrow();
        set_immu_logit_nothrow();
        set_immu_observer_array_nothrow();
        set_immu_observer_array_size_nothrow();
        set_immu_operatable_id_nothrow();

        return dg::network_exception::SUCCESS;
    }
}

namespace dg::network_tile_lifetime::concurrent_safe_batch{

    //alrights - we are doing stack allocations - we'll reconsider the decision later 

    struct InitLeafPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;
        dg::string logit_value;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id, logit_value);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id, logit_value);
        }
    };

    struct InitMonoPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;
        std::array<uma_ptr_t, MAX_OBSERVER_ARRAY_SZ> observer_arr;
        uint16_t observer_arr_sz

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, observer_arr, observer_arr_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, observer_arr, observer_arr_sz);
        }
    };

    struct InitPairPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t lhs;
        uma_ptr_t rhs;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;
        std::array<uma_ptr_t, MAX_OBSERVER_ARRAY_SZ> observer_arr;
        uint16_t observer_arr_sz;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, lhs, rhs, dispatch_control, operatable_id, observer_arr, observer_arr_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, lhs, rhs, dispatch_control, operatable_id, observer_arr, observer_arr_sz);
        }
    };

    struct InitUACMPayLoad{
        uma_ptr_t ptr;
        std::array<uma_ptr_t, UACM_ACM_SZ> src;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;
        std::array<uma_ptr_t, MAX_OBSERVER_ARRAY_SZ> observer_arr;
        uint16_t observer_arr_sz;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, observer_arr, observer_arr_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, observer_arr, observer_arr_sz);
        }
    };

    struct InitPACMPayLoad{
        uma_ptr_t ptr;
        std::array<uma_ptr_t, PACM_ACM_SZ> left_descendant;
        std::array<uma_ptr_t, PACM_ACM_SZ> right_descendant;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;
        std::array<uma_ptr_t, MAX_OBSERVER_ARRAY_SZ> observer_arr;
        uint16_t observer_arr_sz;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, left_descendant, right_descendant, dispatch_control, operatable_id, observer_arr, observer_arr_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, left_descendant, right_descendant, dispatch_control, operatable_id, observer_arr, observer_arr_sz);
        }
    };

    struct InitCritPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;
        crit_kind_t crit_kind;
        dg::string clogit_value;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, crit_kind, clogit_value);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, crit_kind, clogit_value);
        }
    };

    struct InitMsgrFwdPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;
        dst_info_t dst_info;
        std::array<uma_ptr_t, MAX_OBSERVER_ARRAY_SZ> observer_arr;
        uint16_t observer_arr_sz;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, dst_info, observer_arr, observer_arr_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, dst_info, observer_arr, observer_arr_sz);
        }
    };

    struct InitMsgrBwdPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;
        timein_t timein;
        dst_info_t dst_info;
        std::array<uma_ptr_t, MAX_OBSERVER_ARRAY_SZ> observer_arr;
        uint16_t observer_arr_sz;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, timein, dst_info, observer_arr, observer_arr_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, timein, dst_info, observer_arr, observer_arr_sz);
        }
    };

    struct InitExtnSrcPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        uma_ptr_t counterpart;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, counterpart, dispatch_control, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, counterpart, dispatch_control, operatable_id);
        }
    };

    struct InitExtnDstPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t counterpart;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;
        std::array<uma_ptr_t, MAX_OBSERVER_ARRAY_SZ> observer_arr;
        uint16_t observer_arr_sz;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, counterpart, dispatch_control, operatable_id, observer_arr, observer_arr_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, counterpart, dispatch_control, operatable_id, observer_arr, observer_arr_sz);
        }
    };

    struct InitImmuPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;
        std::array<uma_ptr_t, MAX_OBSERVER_ARRAY_SZ> observer_arr;
        uint16_t observer_arr_sz;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id, observer_arr, observer_arr_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id, observer_arr, observer_arr_sz);
        }
    };

    struct OrphanLeafPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanMonoPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id; 

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanPairPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanUACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanPACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanCritPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanMsgrFwdPayLoad{
        uma_ptr_t ptr; 
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanMsgrBwdPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanExtnSrcPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanExtnDstPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanImmuPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitLeafPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitMonoPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitPairPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitUACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitPACMPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitCritPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitMsgrFwdPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitMsgrBwdPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitExtnSrcPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitExtnDstPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct DeinitImmuPayLoad{
        uma_ptr_t ptr;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    auto make_init_leaf_payload(uma_ptr_t ptr, operatable_id_t id, dg::string logit_value) noexcept -> InitLeafPayLoad{

        return InitLeafPayLoad{ptr, id, std::move(logit_value)};
    }

    auto make_init_mono_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitMonoPayLoad{

        return InitMonoPayLoad{ptr, src, dispatch_control, operatable_id};
    }

    auto make_init_pair_payload(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitPairPayLoad{

        return InitPairPayLoad{ptr, lhs, rhs, dispatch_control, operatable_id};
    }

    auto make_init_uacm_payload(uma_ptr_t ptr, std::array<uma_ptr_t, UACM_ACM_SZ> src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitUACMPayLoad{

        return InitUACMPayLoad{ptr, src, dispatch_control, operatable_id};
    }

    auto make_init_pacm_payload(uma_ptr_t ptr, std::array<uma_ptr_t, PACM_ACM_SZ> lhs, std::array<uma_ptr_t, PACM_ACM_SZ> rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitPACMPayLoad{

        return InitPACMPayLoad{ptr, lhs, rhs, dispatch_control, operatable_id};
    }

    auto make_init_crit_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, dg::string clogit_value) noexcept -> InitCritPayLoad{

        return InitCritPayLoad{ptr, src, dispatch_control, operatable_id, std::move(clogit_value)};
    }

    auto make_init_msgrfwd_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, dst_info_t dst_info) noexcept -> InitMsgrFwdPayLoad{

        return InitMsgrFwdPayLoad{ptr, src, dispatch_control, operatable_id, dst_info};
    }

    auto make_init_msgrbwd_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, timein_t timein, dst_info_t dst_info) noexcept -> InitMsgrBwdPayLoad{

        return InitMsgrBwdPayLoad{ptr, src, dispatch_control, operatable_id, timein, dst_info};
    }

    auto make_init_extnsrc_payload(uma_ptr_t ptr, uma_ptr_t src, uma_ptr_t counterpart, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitExtnSrcPayLoad{

        return InitExtnsrcPayLoad{ptr, src, counterpart, dispatch_control, operatable_id};
    }

    auto make_init_extndst_payload(uma_ptr_t ptr, uma_ptr_t counterpart, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitExtnDstPayLoad{

        return InitSrcDstClonePayLoad{ptr, counterpart, dispatch_control, operatable_id};
    }

    auto make_init_immu_payload(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> InitImmuPayLoad{

        return InitImmuPayLoad{ptr, operatable_id};
    }

    auto make_orphan_leaf_payload(uma_ptr_t ptr) noexcept -> OrphanLeafPayLoad{

        return OrphanLeafPayLoad{ptr};
    }

    auto make_orphan_mono_payload(uma_ptr_t ptr) noexcept -> OrphanMonoPayLoad{

        return OrphanMonoPayLoad{ptr};
    }

    auto make_orphan_pair_payload(uma_ptr_t ptr) noexcept -> OrphanPairPayLoad{

        return OrphanPairPayLoad{ptr};
    }

    auto make_orphan_uacm_payload(uma_ptr_t ptr) noexcept -> OrphanUACMPayLoad{

        return OrphanUACMPayLoad{ptr};
    }

    auto make_orphan_pacm_payload(uma_ptr_t ptr) noexcept -> OrphanPACMPayLoad{

        return OrphanPACMPayLoad{ptr};
    }

    auto make_orphan_crit_payload(uma_ptr_t ptr) noexcept -> OrphanCritPayLoad{

        return OrphanCritPayLoad{ptr};
    }

    auto make_orphan_msgrfwd_payload(uma_ptr_t ptr) noexcept -> OrphanMsgrFwdPayLoad{

        return OrphanMsgrFwdPayLoad{ptr};
    }

    auto make_orphan_msgrbwd_payload(uma_ptr_t ptr) noexcept -> OrphanMsgrBwdPayLoad{

        return OrphanMsgrBwdPayLoad{ptr};
    }

    auto make_orphan_extnsrc_payload(uma_ptr_t ptr) noexcept -> OrphanExtnSrcPayLoad{

        return OrphanExtnSrcPayLoad{ptr};
    }

    auto make_orphan_extndst_payload(uma_ptr_t ptr) noexcept -> OrphanExtnDstPayLoad{

        return OrphanExtnDstPayLoad{ptr};
    }

    auto make_orphan_immu_payload(uma_ptr_t ptr) noexcept -> OrphanImmuPayLoad{

        return OrphanImmuPayLoad{ptr};
    }

    auto make_orphan_payload(uma_ptr_t ptr) noexcept -> OrphanPayLoad{

        return OrphanPayLoad{ptr};
    }

    auto make_deinit_leaf_payload(uma_ptr_t ptr) noexcept -> DeinitLeafPayLoad{

        return DeinitLeafPayLoad{ptr};
    }

    auto make_deinit_mono_payload(uma_ptr_t ptr) noexcept -> DeinitMonoPayLoad{

        return DeinitMonoPayLoad{ptr};
    }

    auto make_deinit_pair_payload(uma_ptr_t ptr) noexcept -> DeinitPairPayLoad{
        
        return DeinitPairPayLoad{ptr};
    }

    auto make_deinit_uacm_payload(uma_ptr_t ptr) noexcept -> DeinitUACMPayLoad{

        return DeinitUACMPayLoad{ptr};
    }

    auto make_deinit_pacm_payload(uma_ptr_t ptr) noexcept -> DeinitPACMPayLoad{
        
        return DeinitPACMPayLoad{ptr};
    }

    auto make_deinit_crit_payload(uma_ptr_t ptr) noexcept -> DeinitCritPayLoad{

        return DeinitCritPayLoad{ptr};
    }

    auto make_deinit_msgrfwd_payload(uma_ptr_t ptr) noexcept -> DeinitMsgrFwdPayLoad{

        return DeinitMsgrFwdPayLoad{ptr};
    }

    auto make_deinit_msgrbwd_payload(uma_ptr_t ptr) noexcept -> DeinitMsgrBwdPayLoad{

        return DeinitMsgrBwdPayLoad{ptr};
    }

    auto make_deinit_extnsrc_payload(uma_ptr_t ptr) noexcept -> DeinitExtnSrcPayLoad{

        return DeinitExtnSrcPayLoad{ptr};
    }

    auto make_deinit_extndst_payload(uma_ptr_t ptr) noexcept -> DeinitExtnDstPayLoad{

        return DeinitExtnDstPayLoad{ptr};
    }

    auto make_deinit_immu_payload(uma_ptr_t ptr) noexcept -> DeinitImmuPayLoad{

        return DeinitImmuPayLoad{ptr};
    }

    auto make_deinit_payload(uma_ptr_t ptr) noexcept -> DeinitPayLoad{

        return DeinitPayLoad{ptr};
    }

    void load_init_leaf_payload(InitLeafPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitLeafPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_leaf(payload.ptr, payload.operatable_id, payload.logit_value.data(), payload.logit_value.size());
            }
        }; 
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitLeafPayLoad, exception_t *>, decltype(vectrz)>(vectrz); //devirtualization by ID required
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));
        
        for (size_t i = 0u; i < sz; ++i){ 
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_leaf_rcu_addr(payload_arr[i].ptr); 

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    } 

    void load_init_mono_payload(InitMonoPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitMonoPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_mono(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitMonoPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_mono_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    } 

    void load_init_pair_payload(InitPairPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitPairPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_pair(payload.ptr, payload.lhs, payload.rhs, payload.dispatch_control, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitPairPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_pair_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_init_uacm_payload(InitUACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitUACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_uacm(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitUACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));
    
        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_uacm_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_init_pacm_payload(InitPACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitPACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_pacm(payload.ptr, payload.lhs, payload.rhs, payload.dispatch_control, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitPACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_pacm_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_init_crit_payload(InitCritPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitCritPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_crit(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id, payload.crit_kind, payload.clogit_value.data(), payload.clogit_value.size());
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitCritPayLoad, exception-t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_crit_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_init_msgrfwd_payload(InitMsgrFwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitMsgrFwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_msgrfwd(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id, payload.dst_info);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitMsgrFwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_init_msgrbwd_payload(InitMsgrBwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitMsgrBwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_msgrbwd(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id, payload.timein, payload.dst_info);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LamdaWrappedConsumer<std::tuple<InitMsgrBwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_init_extnsrc_payload(InitExtnSrcPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitExtnSrcPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_extnsrc(payload.ptr, payload.src, payload.counterpart, payload.dispatch_control, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LamdaWrappedConsumer<std::tuple<InitExtnSrcPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_init_extndst_payload(InitExtnDstPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitExtnDstPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_extndst(payload.ptr, payload.counterpart, payload.dispatch_control, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitExtnDstPayLoad, exception_t * >, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_extndst_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_init_immu_payload(InitImmuPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<InitImmuPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::init_immu(payload.ptr, payload.operatable_id, payload.logit_value.data(), payload.logit_value.size());
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<InitImmuPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_immu_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_orphan_leaf_payload(OrphanLeafPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t lck_addr, std::tuple<OrphanLeafPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_leaf(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanLeafPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_leaf_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_orphan_mono_payload(OrphanMonoPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t lck_addr, std::tuple<OrphanMonoPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr]   = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_mono(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanMonoPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_mono_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            } 

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_orphan_pair_payload(OrphanPairPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanPairPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_pair(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanPairPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_pair_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_orphan_uacm_payload(OrphanUACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanUACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_uacm(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanUACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_uacm_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_orphan_pacm_payload(OrphanPACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanPACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_pacm(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanPACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_pacm_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_orphan_crit_payload(OrphanCritPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanCritPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_crit(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanCritPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_crit_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_orphan_msgrfwd_payload(OrphanMsgrFwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanMsgrFwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_msgrfwd(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanMsgrFwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size()); //this is the most important optimization - we are managing all the allocations
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_orphan_msgrbwd_payload(OrphanMsgrBwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanMsgrBwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_msgrbwd(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanMsgrBwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_adr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
            //people load kernel with 15M LOC Mom - we'll compile files later
        }
    }

    void load_orphan_extnsrc_payload(OrphanExtnSrcPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanExtnSrcPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_extnsrc(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanExtnSrcPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_orphan_extndst_payload(OrphanExtnDstPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanExtnDstPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_extndst(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanExtnDstPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_extndst_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tule(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_orphan_immu_payload(OrphanImmuPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<OrphanImmuPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::orphan_immu(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<OrphanImmuPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_immu_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_leaf_payload(DeinitLeafPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitLeafPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_leaf(payload.ptr, payload.operatable_id); 
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitLeafPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_leaf_rcu_addr(payload_arr[i].ptr)
            
            if (!rcu_addr.has_value());{
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_mono_payload(DeinitMonoPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMonoPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_mono(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitMonoPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_mono_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }
            
            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_pair_payload(DeinitPairPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitPairPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_pair(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitPairPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_pair_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_uacm_payload(DeinitUACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitUACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_uacm(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitUACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_uacm_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            } 

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_pacm_payload(DeinitPACMPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitPACMPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_pacm(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitPACMPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_pacm_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_crit_payload(DeinitCritPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        auto VECTORIZATION_SZ           = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitCritPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_crit(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitCritPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_crit_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_msgrfwd_payload(DeinitMsgrFwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMsgrFwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_msgrfwd(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitMsgrFwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_msgrbwd_payload(DeinitMsgrBwdPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitMsgrBwdPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_msgrbwd(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitMsgrBwdPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_extnsrc_payload(DeinitExtnSrcPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitExtnSrcPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_extnsrc(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitExtnSrcPayLoad, exception-t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr(payload_arr[i].ptr);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_extndst_payload(DeinitExtnDstPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitExtnDstPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_extndst(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitExtnDstPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_extndst_rcu_addr(payload_arr[i].dst);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }

    void load_deinit_immu_payload(DeinitImmuPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        const size_t VECTORIZATION_SZ   = size_t{1} << 8;
        auto vectrz                     = [](uma_ptr_t rcu_lck_addr, std::tuple<DeinitImmuPayLoad, exception_t *> * payload_arr, size_t sz) noexcept{
            dg::network_memops_uma::memlock_guard mem_grd(rcu_lck_addr);

            for (size_t i = 0u; i < sz; ++i){
                auto [payload, exception_ptr] = payload_arr[i];
                *exception_ptr = dg::network_tile_lifetime::concurrent_unsafe::deinit_immu(payload.ptr, payload.operatable_id);
            }
        };
        auto virtual_vectrz     = dg::network_producer_consumer::LambdaWrappedConsumer<uma_ptr_t, std::tuple<DeinitImmuPayLoad, exception_t *>, decltype(vectrz)>(vectrz);
        dg::network_stack_allocation::Allocation<char[]> buf(dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&virtual_vectz, VECTORIZATION_SZ));
        auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&virtual_vectrz, VECTORIZATION_SZ, buf.get()));

        for (size_t i = 0u; i < sz; ++i){
            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_immu_rcu_addr(payload_arr[i].dst);

            if (!rcu_addr.has_value()){
                exception_arr[i] = rcu_addr.error();
                continue;
            }

            uma_ptr_t lck_addr = dg::memult::region(rcu_addr.value(), dg::network_memops_uma::memlock_region_size());
            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, std::make_tuple(payload_arr[i], std::next(exception_arr, i)));
        }
    }
}

namespace dg::network_tile_lifetime::concurrent_safe_poly{

    //these aren't neccessarily invoked by clients - these are for providing a high level interface of the payload format - we'll reiterate this later
    //we want to radix payload -> fast_payload and slow_payload which are handled by two different functions to avoid performance constraints by misuse of interfaces

    using fast_payload_kind_t   = uint8_t;
    using slow_payload_kind_t   = uint8_t;

    enum enum_fast_payload: fast_payload_kind_t{
        payload_kind_init_mono          = 0u,
        payload_kind_init_pair          = 1u,
        payload_kind_init_uacm          = 2u,
        payload_kind_init_pacm          = 3u,
        payload_kind_init_msgrfwd       = 4u,
        payload_kind_init_msgrbwd       = 5u,
        payload_kind_init_extnsrc       = 6u,
        payload_kind_init_extndst       = 7u,
        payload_kind_orphan_leaf        = 8u,
        payload_kind_orphan_mono        = 9u,
        payload_kind_orphan_pair        = 10u,
        payload_kind_orphan_uacm        = 11u,
        payload_kind_orphan_pacm        = 12u,
        payload_kind_orphan_crit        = 13u,
        payload_kind_orphan_msgrfwd     = 14u,
        payload_kind_orphan_msgrbwd     = 15u,
        payload_kind_orphan_extnsrc     = 16u,
        payload_kind_orphan_extndst     = 17u,
        payload_kind_orphan_immu        = 18u
    };

    enum enum_slow_payload: slow_payload_kind_t{
        payload_kind_init_leaf          = 0u,
        payload_kind_init_crit          = 1u,
        payload_kind_init_immu          = 2u,
        payload_kind_deinit_leaf        = 3u,
        payload_kind_deinit_mono        = 4u,
        payload_kind_deinit_pair        = 5u,
        payload_kind_deinit_uacm        = 6u,
        payload_kind_deinit_pacm        = 7u,
        payload_kind_deinit_crit        = 8u,
        payload_kind_deinit_msgrfwd     = 9u,
        payload_kind_deinit_msgrbwd     = 10u,
        payload_kind_deinit_extnsrc     = 11u,
        payload_kind_deinit_extndst     = 12u,
        payload_kind_deinit_immu        = 13u
    };

    static inline constexpr size_t VIRTUAL_PAYLOAD_CONTENT_SZ = size_t{1} << 5;

    struct VirtualPayLoad{
        payload_kind_t kind;
        std::array<char, VIRTUAL_PAYLOAD_CONTENT_SZ> content;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(kind, content);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(kind, content);
        }
    };

    auto virtualize_payload(InitMonoPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_init_mono;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(InitPairPayLoad payload) noexcept -> VirtualPayLoad{
        
        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_init_pair;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(InitUACMPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);
        
        VirtualPayLoad rs{};
        rs.kind = payload_kind_init_uacm;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(InitPACMPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_init_pacm;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }
 
    auto virtualize_payload(InitMsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_init_msgrfwd;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(InitMsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_init_msgrbwd;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(InitExtnSrcPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_init_extnsrc;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(InitExtnDstPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_init_extndst;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanLeafPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_leaf;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanMonoPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_mono;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanPairPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_pair;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanUACMPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_uacm;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanPACMPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_pacm;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanCritPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_crit;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanMsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_msgrfwd;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanMsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_msgrbwd;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanExtnSrcPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_extnsrc;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanExtnDstPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_extndst;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto virtualize_payload(OrphanImmuPayLoad payload) noexcept -> VirtualPayLoad{

        static_assert(dg::network_trivial_serializer::size(payload) <= VIRTUAL_PAYLOAD_CONTENT_SZ);

        VirtualPayLoad rs{};
        rs.kind = payload_kind_orphan_immu;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

        return rs;
    }

    auto devirtualize_init_mono_payload(VirtualPayLoad payload) noexcept -> InitMonoPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_init_mono){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        InitMonoPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_pair_payload(VirtualPayLoad payload) noexcept -> InitPairPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_init_pair){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        InitPairPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_uacm_payload(VirtualPayLoad payload) noexcept -> InitUACMPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_init_uacm){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        InitUACMPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_pacm_payload(VirtualPayLoad payload) noexcept -> InitPACMPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_init_pacm){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        InitPACMPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_msgrfwd_payload(VirtualPayLoad payload) noexcept -> InitMsgrFwdPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_init_msgrfwd){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        InitMsgrFwdPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_msgrbwd_payload(VirtualPayLoad payload) noexcept -> InitMsgrBwdPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_init_msgrbwd){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        InitMsgrBwdPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_extnsrc_payload(VirtualPayLoad payload) noexcept -> InitExtnSrcPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_init_extnsrc){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        InitExtnSrcPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_init_extndst_payload(VirtualPayLoad payload) noexcept -> InitExtnDstPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_init_extndst){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        InitExtnDstPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_leaf_payload(VirtualPayLoad payload) noexcept -> OrphanLeafPayLoad{
        
        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_leaf){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanLeafPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_mono_payload(VirtualPayLoad payload) noexcept -> OrphanMonoPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_mono){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanMonoPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_pair_payload(VirtualPayLoad payload) noexcept -> OrphanPairPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_pair){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanPairPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_uacm_payload(VirtualPayLoad payload) noexcept -> OrphanUACMPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_uacm){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanUACMPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_pacm_payload(VirtualPayLoad payload) noexcept -> OrphanPACMPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_pacm){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanPACMPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_crit_payload(VirtualPayLoad payload) noexcept -> OrphanCritPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_crit){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanCritPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_msgrfwd_payload(VirtualPayLoad payload) noexcept -> OrphanMsgrFwdPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_msgrfwd){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanMsgrFwdPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_msgrbwd_payload(VirtualPayLoad payload) noexcept -> OrphanMsgrBwdPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_msgrbwd){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanMsgrBwdPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_extnsrc_payload(VirtualPayLoad payload) noexcept -> OrphanExtnSrcPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_extnsrc){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanExtnSrcPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_extndst_payload(VirtualPayLoad payload) noexcept -> OrphanExtnDstPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_extndst){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanExtnDstPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }

    auto devirtualize_orphan_immu_payload(VirtualPayLoad payload) noexcept -> OrphanImmuPayLoad{

        if constexpr(DEBUG_MODE_FLAG){
            if (payload.kind != payload_kind_orphan_immu){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        OrphanImmuPayLoad rs{};
        dg::network_trivial_serializer::deserialize_into(rs, payload.content.data());

        return rs;
    }
 
     // auto virtualize_payload(InitLeafPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload); 
    //     rs.kind             = payload_kind_init_leaf;

    //     return rs;
    // }


    // auto virtualize_payload(InitCritPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_init_crit;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);
    
    //     return rs;
    // }
    
    // auto virtualize_payload(InitImmuPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_init_immu;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitLeafPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_leaf;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitMonoPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_mono;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitPairPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_pair;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitUACMPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_uacm;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitPACMPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_pacm;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitCritPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_crit;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitMsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_msgrfwd;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitMsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_msgrbwd;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitExtnSrcPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_extnsrc;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitExtnDstPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_extndst;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto virtualize_payload(DeinitImmuPayLoad payload) noexcept -> VirtualPayLoad{

    //     VirtualPayLoad rs{};
    //     rs.kind = payload_kind_deinit_immu;
    //     dg::network_trivial_serializer::serialize_into(rs.content.data(), payload);

    //     return rs;
    // }

    // auto devirtualize_init_immu_payload(VirtualPayLoad payload) noexcept -> InitImmuPayLoad{

    //     if constexpr(DEBUG_MODE_FLAG){
    //         if (payload.kind != payload_kind_init_)
    //     }
    // }

    void load_virtual_payloads(VirtualPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        constexpr size_t DISPATCH_DELIVERY_CAP      = size_t{1} << 16; //config this

        auto init_mono_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<InitMonoPayLoad[]> devirt_payload_arr(sz); //we'll leverage concurrency and affinity to achieve the magic - we have to disable move cpy constructors + assignments
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_init_mono_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_mono_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_pair_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<InitPairPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_init_pair_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_pair_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_uacm_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<InitUACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_init_uacm_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_uacm_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_pacm_dispatcher                   = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<InitPACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_init_pacm_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_pacm_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_msgrfwd_dispatcher                = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<InitMsgrFwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_init_msgrfwd_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_msgrfwd_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_msgrbwd_dispatcher                = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<InitMsgrBwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_init_msgrbwd_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_msgrbwd_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_extnsrc_dispatcher                = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<InitExtnSrcPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_init_extnsrc_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_extnsrc_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto init_extndst_dispatcher                = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<InitExtnDstPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_init_extndst_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_init_extndst_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_leaf_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanLeafPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_leaf_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_leaf_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_mono_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanMonoPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_mono_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_mono_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_pair_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanPairPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);
            
            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_pair_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_pair_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_uacm_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanUACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);
           
            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_uacm_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_uacm_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_pacm_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanPACMPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_pacm_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_pacm_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_crit_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanCritPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_crit_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_crit_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_msgrfwd_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanMsgrFwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_msgrfwd_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_msgrfwd_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_msgrbwd_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanMsgrBwdPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);
     
            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_msgrbwd_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_msgrbwd_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_extnsrc_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanExtnSrcPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_extnsrc_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_extnsrc_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_extndst_dispatcher              = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanExtnDstPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_extndst_payload(std::get<0>(data_arr[i]));
            }

            dg::network_tile_lifetime::concurrent_safe_batch::load_orphan_extndst_payload(devirt_payload_arr.get(), exception_arr.get(), sz);

            for (size_t i = 0u; i < sz; ++i){
                *std::get<1>(data_arr[i]) = exception_arr[i];
            }
        };

        auto orphan_immu_dispatcher                 = [](std::pair<VirtualPayLoad, exception_t *> * data_arr, size_t sz) noexcept{
            dg::network_stack_allocation::Allocation<OrphanImmuPayLoad[]> devirt_payload_arr(sz);
            dg::network_stack_allocation::Allocation<exception_t[]> exception_arr(sz);

            for (size_t i = 0u; i < sz; ++i){
                devirt_payload_arr[i] = devirtualize_orphan_immu_payload(std::get<0>(data_arr[i]));
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
 
        dg::network_stack_allocation::Allocation<char[]> init_mono_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_mono_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::Allocation<char[]> init_pair_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_pair_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::Allocation<char[]> init_uacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_uacm_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::Allocation<char[]> init_pacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_pacm_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::Allocation<char[]> init_msgrfwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_msgrfwd_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::Allocation<char[]> init_msgrbwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_msgrbwd_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::Allocation<char[]> init_extnsrc_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_extnsrc_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::Allocation<char[]> init_extndst_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&init_extndst_consumer, DISPATCH_DELIVERY_CAP));
        dg::network_stack_allocation::Allocation<char[]> orphan_leaf_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_leaf_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::Allocation<char[]> orphan_mono_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_mono_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::Allocation<char[]> orphan_pair_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_pair_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::Allocation<char[]> orphan_uacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_uacm_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::Allocation<char[]> orphan_pacm_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_pacm_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::Allocation<char[]> orphan_crit_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_crit_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::Allocation<char[]> orphan_msgrfwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_msgrfwd_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::Allocation<char[]> orphan_msgrbwd_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_msgrbwd_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::Allocation<char[]> orphan_extnsrc_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_extnsrc_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::Allocation<char[]> orphan_extndst_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_extndst_consumer, DISPATCH_DELIVERY_CAP));  
        dg::network_stack_allocation::Allocation<char[]> orphan_immu_allocation(dg::network_producer_consumer::delvrsrv_allocation_cost(&orphan_immu_consumer, DISPATCH_DELIVERY_CAP));  

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
            auto payload_kind                   = payload_arr[i].kind;
            VirtualPayLoad dispatching_payload  = payload_arr[i];
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
