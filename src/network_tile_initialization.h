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

namespace dg::network_tile_lifetime::statix{

    using uma_ptr_t             = dg::network_pointer::uma_ptr_t; 
    using operatable_id_t       = dg::network_tile_metadata::operatable_id_t;
    using dispatch_control_t    = dg::network_tile_metadata::dispatch_control_t;
    using crit_kind_t           = dg::network_tile_metadata::crit_kind_t;
    using dst_info_t            = dg::network_tile_metadata::dst_info_t;
    using timein_t              = dg::network_tile_metadata::timein_t;

    static inline constexpr UACM_ACM_SZ = dg::network_tile_metadata::UACM_ACM_SZ;
    static inline constexpr PACM_ACM_SZ = dg::network_tile_metadata::PACM_ACM_SZ;

    auto init_leaf(uma_ptr_t ptr, operatable_id_t operatable_id, void * logit_value, size_t logit_value_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_leaf_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr); //this has to be a spinlock - and memory region needs to be large enough to avoid collision
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

    auto init_mono(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_mono_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

        set_mono_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_mono_observer_array_size_nothrow(ptr, 0u);
        set_mono_dispatch_control_nothrow(ptr, dispatch_control);
        set_mono_operatable_id_nothrow(ptr, operatable_id);
        set_mono_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_mono_descendant_nothrow(ptr, src);
        set_mono_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);
    
        return dg::network_exception::SUCCESS;
    }

    auto init_pair(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_pair_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

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

    auto init_uacm(uma_ptr_t ptr, std::array<uma_ptr_t, UACM_ACM_SZ> src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_uacm_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

        set_uacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_uacm_observer_array_size_nothrow(ptr, 0u);
        set_uacm_operatable_id_nothrow(ptr, operatable_id);
        set_uacm_dispatch_control_nothrow(ptr, dispatch_control);
        set_uacm_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_uacm_descendant_nothrow(ptr, src);
        set_uacm_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_pacm(uma_ptr_t ptr, std::array<uma_ptr_t, PACM_ACM_SZ> lhs, std::array<uma_ptr_t, PACM_ACM_SZ> rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_pacm_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

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

    auto init_crit(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, crit_kind_t crit_kind, void * clogit_value, size_t clogit_value_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_crit_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
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

    auto init_msgrfwd(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, dst_info_t dst_info) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_msgrfwd_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

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

    auto init_msgrbwd(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, timein_t timein, dst_info_t dst_info) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_msgrbwd_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

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

    auto init_srcextclone(uma_ptr_t ptr, uma_ptr_t src, uma_ptr_t counterpart, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_srcextclone_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_srcextclone_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

        set_srcextclone_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_srcextclone_observer_array_size_nothrow(ptr, 0u);
        set_srcextclone_operatable_id_nothrow(ptr, operatable_id);
        set_srcextclone_dispatch_control_nothrow(ptr, dispatch_control);
        set_srcextclone_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_srcextclone_descendant_nothrow(ptr, src);
        set_srcextclone_counterpart_nothrow(ptr, counterpart);
        set_srcextclone_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_dstextclone(uma_ptr_t ptr, uma_ptr_t counterpart, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_dstextclone_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_dstextclone_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

        set_dstextclone_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_dstextclone_observer_array_size_nothrow(ptr, 0u);
        set_dstextclone_operatable_id_nothrow(ptr, operatable_id);
        set_dstextclone_dispatch_control_nothrow(ptr, dispatch_control);
        set_dstextclone_counterpart_nothrow(ptr, counterpart);

        return dg::network_exception::SUCCESS;
    }

    auto init_immu(uma_ptr_t ptr, operatable_id_t operatable_id, void * logit_value, size_t logit_value_sz) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_immu_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_immu_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
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

    auto orphan_leaf(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_leaf_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED); 

        return dg::network_exception::SUCCESS;
    }

    auto orphan_mono(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_mono_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_mono_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_pair(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_pair_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_pair_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_uacm(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_uacm_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_uacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_pacm(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_pacm_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_pacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_crit(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_crit_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_crit_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_msgrfwd(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_msgrfwd_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_msgrfwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_msgrbwd(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_msgrbwd_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_msgrbwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_srcextclone(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_srcextclone_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_srcextclone_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_srcextclone_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_dstextclone(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_dstextclone_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_dstextclone_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_dstextclone_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_immu(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_immu_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_immu_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_immu_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_tile_ptr_access(ptr);
        
        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_tile_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_tile_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto deinit_leaf(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_leaf_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

        set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_EMPTY);
        set_leaf_logit_nothrow(ptr, dg::network_tile_metadata::TILE_LOGIT_VALUE_DEFAULT);
        set_leaf_grad_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_VALUE_DEFAULT);
        set_leaf_observer_array_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_leaf_observer_array_size_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_SIZE_DEFAULT);
        set_leaf_operatable_id_nothrow(ptr, dg::network_tile_metadata::TILE_OPERATABLE_ID_DEFAULT);
        set_leaf_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_EMPTY);
        
        return dg::network_exception::SUCCESS;
    }

    auto deinit_mono(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_mono_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

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

    auto deinit_pair(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_pair_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

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

    auto deinit_uacm(uma_ptr_t ptr) noexcept -> exception_t{

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

    auto deinit_pacm(uma_ptr_t ptr) noexcept -> exception_t{ 

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

    auto deinit_crit(uma_ptr_t ptr) noexcept -> exception_t{

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

    auto deinit_msgrfwd(uma_ptr_t ptr) noexcept -> exception_t{

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

    auto deinit_msgrbwd(uma_ptr_t ptr) noexcept -> exception_t{
        
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

    auto deinit_srcextclone(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_srcextclone_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_srcextclone_init_status_nothrow();
        set_srcextclone_logit_nothrow();
        set_srcextclone_grad_nothrow();
        set_srcextclone_observer_array_nothrow();
        set_srcextclone_observer_array_size_nothrow();
        set_srcextclone_operatable_id_nothrow();
        set_srcextclone_dispatch_control_nothrow();
        set_srcextclone_pong_count_nothrow();
        set_srcextclone_descendant_nothrow();
        set_srcextclone_counterpart_nothrow();
        set_srcextclone_grad_status_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_dstextclone(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_dstextclone_ptr_access(ptr);

        if (!ptr_access.has_value()){
            return ptr_access.error();
        }

        set_dstextclone_init_status_nothrow();
        set_dstextclone_observer_array_nothrow();
        set_dstextclone_observer_array_size_nothrow();
        set_dstextclone_operatable_id_nothrow();
        set_dstextclone_dispatch_control_nothrow();
        set_dstextclone_counterpart_nothrow();

        return dg::network_exception::SUCCESS;
    }

    auto deinit_immu(uma_ptr_t ptr) noexcept -> exception_t{

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

    auto deinit(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace dg::network_tile_member_getsetter;

        // auto ptr_access = 
    }
}

namespace dg::network_tile_lifetime::poly{

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

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, operatable_id);
        }
    };

    struct InitPairPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t lhs;
        uma_ptr_t rhs;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, lhs, rhs, dispatch_control, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, lhs, rhs, dispatch_control, operatable_id);
        }
    };

    struct InitUACMPayLoad{
        uma_ptr_t ptr;
        std::array<uma_ptr_t, UACM_ACM_SZ> src;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, operatable_id);
        }
    };

    struct InitPACMPayLoad{
        uma_ptr_t ptr;
        std::array<uma_ptr_t, PACM_ACM_SZ> left_descendant;
        std::array<uma_ptr_t, PACM_ACM_SZ> right_descendant;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, left_descendant, right_descendant, dispatch_control, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, left_descendant, right_descendant, dispatch_control, operatable_id);
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

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, dst_info);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, dst_info);
        }
    };

    struct InitMsgrBwdPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;
        timein_t timein;
        dst_info_t dst_info;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, timein, dst_info);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, timein, dst_info);
        }
    };

    struct InitSrcExtClonePayLoad{
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

    struct InitDstExtClonePayLoad{
        uma_ptr_t ptr;
        uma_ptr_t counterpart;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, counterpart, dispatch_control, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, counterpart, dispatch_control, operatable_id);
        }
    };

    struct InitImmuPayLoad{
        uma_ptr_t ptr;
        operaetable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, operatable_id);
        }
    };

    struct OrphanLeafPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanMonoPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanPairPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanUACMPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanPACMPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanCritPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanMsgrFwdPayLoad{
        uma_ptr_t ptr; 

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanMsgrBwdPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanSrcExtClonePayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanDstExtClonePayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanImmuPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct OrphanPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitLeafPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitMonoPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitPairPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitUACMPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitPACMPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitCritPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitMsgrFwdPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitMsgrBwdPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitSrcExtClonePayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitDstExtClonePayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitImmuPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    struct DeinitPayLoad{
        uma_ptr_t ptr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr);
        }
    };

    auto make_init_leaf_payload(uma_ptr_t ptr, operatable_id_t id, dg::string logit_value) noexcept -> InitLeafPayLoad{

        return InitLeafPayLoad{ptr, id, std::move(logit_value)};
    }

    auto load_init_leaf_payload(InitLeafPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_leaf(payload.ptr, payload.operatable_id, payload.logit_value.data(), payload.logit_value.size());
    } 

    auto make_init_mono_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitMonoPayLoad{

        return InitMonoPayLoad{ptr, src, dispatch_control, operatable_id};
    }

    auto load_init_mono_payload(InitMonoPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_mono(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id);
    } 

    auto make_init_pair_payload(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitPairPayLoad{

        return InitPairPayLoad{ptr, lhs, rhs, dispatch_control, operatable_id};
    }

    auto load_init_pair_payload(InitPairPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_mono(payload.ptr, payload.lhs, payload.rhs, payload.dispatch_control, payload.operatable_id);
    }

    auto make_init_uacm_payload(uma_ptr_t ptr, std::array<uma_ptr_t, UACM_ACM_SZ> src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitUACMPayLoad{

        return InitUACMPayLoad{ptr, src, dispatch_control, operatable_id};
    }

    auto load_init_uacm_payload(InitUACMPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_uacm(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id);
    }

    auto make_init_pacm_payload(uma_ptr_t ptr, std::array<uma_ptr_t, PACM_ACM_SZ> lhs, std::array<uma_ptr_t, PACM_ACM_SZ> rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitPACMPayLoad{

        return InitPACMPayLoad{ptr, lhs, rhs, dispatch_control, operatable_id};
    }

    auto load_init_pacm_payload(InitPACMPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_pacm(payload.ptr, payload.left_descendant, payload.right_descendant, payload.dispatch_control, payload.operatable_id);
    }
    
    auto make_init_crit_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, dg::string clogit_value) noexcept -> InitCritPayLoad{

        return InitCritPayLoad{ptr, src, dispatch_control, operatable_id, std::move(clogit_value)};
    }

    auto load_init_crit_payload(InitCritPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_crit(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id, payload.crit_kind, payload.clogit_value.data(), payload.clogit_value.size());
    }

    auto make_init_msgrfwd_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, dst_info_t dst_info) noexcept -> InitMsgrFwdPayLoad{

        return InitMsgrFwdPayLoad{ptr, src, dispatch_control, operatable_id, dst_info};
    }

    auto load_init_msgrfwd_payload(InitMsgrFwdPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_msgrfwd(ptr, src, dispatch_control, operatable_id, dst_info);
    }

    auto make_init_msgrbwd_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, timein_t timein, dst_info_t dst_info) noexcept -> InitMsgrBwdPayLoad{

        return InitMsgrBwdPayLoad{ptr, src, dispatch_control, operatable_id, timein, dst_info};
    }

    auto load_init_msgrbwd_payload(InitMsgrBwdPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_msgrbwd(ptr, src, dispatch_control, operatable_id, timein, dst_info);
    }

    auto make_init_srcextclone_payload(uma_ptr_t ptr, uma_ptr_t src, uma_ptr_t counterpart, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitSrcExtClonePayLoad{

        return InitSrcExtClonePayLoad{ptr, src, counterpart, dispatch_control, operatable_id};
    }

    auto load_init_srcextclone_payload(InitSrcExtClonePayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_srcextclone(payload.ptr, payload.src, payload.counterpart, payload.dispatch_control, payload.operatable_id);
    }

    auto make_init_dstextclone_payload(uma_ptr_t ptr, uma_ptr_t counterpart, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> InitDstExtClonePayLoad{

        return InitSrcDstClonePayLoad{ptr, counterpart, dispatch_control, operatable_id};
    }

    auto load_init_dstextclone_payload(InitDstExtClonePayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_dstextclone(payload.ptr, payload.counterpart, payload.dispatch_control, payload.operatable_id);
    }

    auto make_init_immu_payload(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> InitImmuPayLoad{

        return InitImmuPayLoad{ptr, operatable_id};
    }

    auto load_init_immu_payload(InitImmuPayload payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::init_immu(payload.ptr, payload.operatable_id);
    }

    auto make_orphan_leaf_payload(uma_ptr_t ptr) noexcept -> OrphanLeafPayLoad{

        return OrphanLeafPayLoad{ptr};
    }

    auto load_orphan_leaf_payload(OrphanLeafPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_leaf(payload.ptr);
    }

    auto make_orphan_mono_payload(uma_ptr_t ptr) noexcept -> OrphanMonoPayLoad{

        return OrphanMonoPayLoad{ptr};
    }

    auto load_orphan_mono_payload(OrphanMonoPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_mono(payload.ptr);
    }

    auto make_orphan_pair_payload(uma_ptr_t ptr) noexcept -> OrphanPairPayLoad{

        return OrphanPairPayLoad{ptr};
    }

    auto load_orphan_pair_payload(OrphanPairPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_pair(payload.ptr);
    }

    auto make_orphan_uacm_payload(uma_ptr_t ptr) noexcept -> OrphanUACMPayLoad{

        return OrphanUACMPayLoad{ptr};
    }

    auto load_orphan_uacm_payload(OrphanUACMPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_uacm(payload.ptr);
    }

    auto make_orphan_pacm_payload(uma_ptr_t ptr) noexcept -> OrphanPACMPayLoad{

        return OrphanPACMPayLoad{ptr};
    }

    auto load_orphan_pacm_payload(OrphanPACMPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_pacm(payload.ptr);
    }

    auto make_orphan_crit_payload(uma_ptr_t ptr) noexcept -> OrphanCritPayLoad{

        return OrphanCritPayLoad{ptr};
    }

    auto load_orphan_crit_payload(OrphanCritPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_crit(payload.ptr);
    }

    auto make_orphan_msgrfwd_payload(uma_ptr_t ptr) noexcept -> OrphanMsgrFwdPayLoad{

        return OrphanMsgrFwdPayLoad{ptr};
    }

    auto load_orphan_msgrfwd_payload(OrphanMsgrFwdPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_msgrfwd(payload.ptr);
    }

    auto make_orphan_msgrbwd_payload(uma_ptr_t ptr) noexcept -> OrphanMsgrBwdPayLoad{

        return OrphanMsgrBwdPayLoad{ptr};
    }

    auto load_orphan_msgrbwd_payload(OrphanMsgrBwdPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_msgrbwd(payload.ptr);
    }

    auto make_orphan_srcextclone_payload(uma_ptr_t ptr) noexcept -> OrphanSrcExtClonePayLoad{

        return OrphanSrcExtClonePayLoad{ptr};
    }
    
    auto load_orphan_srcextclone_payload(OrphanSrcExtClonePayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_srcextclone(payload.ptr);
    }

    auto make_orphan_dstextclone_payload(uma_ptr_t ptr) noexcept -> OrphanDstExtClonePayLoad{

        return OrphanDstExtClonePayLoad{ptr};
    }

    auto load_orphan_dstextclone_payload(OrphanDstExtClonePayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_dstextclone(payload.ptr);
    }

    auto make_orphan_immu_payload(uma_ptr_t ptr) noexcept -> OrphanImmuPayLoad{

        return OrphanImmuPayLoad{ptr};
    }

    auto load_orphan_immu_payload(OrphanImmuPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan_immu(payload.ptr);
    }

    auto make_orphan_payload(uma_ptr_t ptr) noexcept -> OrphanPayLoad{

        return OrphanPayLoad{ptr};
    }

    auto load_orphan_payload(OrphanPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::orphan(payload.ptr);
    }

    auto make_deinit_leaf_payload(uma_ptr_t ptr) noexcept -> DeinitLeafPayLoad{

        return DeinitLeafPayLoad{ptr};
    }

    auto load_deinit_leaf_payload(DeinitLeafPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::deinit_leaf(payload.ptr);
    }

    auto make_deinit_mono_payload(uma_ptr_t ptr) noexcept -> DeinitMonoPayLoad{

        return DeinitMonoPayLoad{ptr};
    }

    auto load_deinit_mono_payload(DeinitMonoPayLoad payload) noexcept -> exception_t{
        
        return dg::network_tile_lifetime::statix::deinit_mono(payload.ptr);
    }

    auto make_deinit_pair_payload(uma_ptr_t ptr) noexcept -> DeinitPairPayLoad{
        
        return DeinitPairPayLoad{ptr};
    }

    auto load_deinit_pair_payload(DeinitPairPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::deinit_pair(payload.ptr);
    }

    auto make_deinit_uacm_payload(uma_ptr_t ptr) noexcept -> DeinitUACMPayLoad{

        return DeinitUACMPayLoad{ptr};
    }

    auto load_deinit_uacm_payload(DeinitUACMPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::deinit_uacm(payload.ptr);
    }

    auto make_deinit_pacm_payload(uma_ptr_t ptr) noexcept -> DeinitPACMPayLoad{
        
        return DeinitPACMPayLoad{ptr};
    }

    auto load_deinit_pacm_payload(DeinitPACMPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::deinit_pacm(payload.ptr);
    }

    auto make_deinit_crit_payload(uma_ptr_t ptr) noexcept -> DeinitCritPayLoad{

        return DeinitCritPayLoad{ptr};
    }

    auto load_deinit_crit_payload(DeinitCritPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::deinit_crit(payload.ptr);
    }

    auto make_deinit_msgrfwd_payload(uma_ptr_t ptr) noexcept -> DeinitMsgrFwdPayLoad{

        return DeinitMsgrFwdPayLoad{ptr};
    }

    auto load_deinit_msgrfwd_payload(DeinitMsgrFwdPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::deinit_msgrfwd(payload.ptr);
    }

    auto make_deinit_msgrbwd_payload(uma_ptr_t ptr) noexcept -> DeinitMsgrBwdPayLoad{

        return DeinitMsgrBwdPayLoad{ptr};
    }

    auto load_deinit_msgrbwd_payload(DeinitMsgrBwdPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::deinit_msgrbwd(payload.ptr);
    }

    auto make_deinit_srcextclone_payload(uma_ptr_t ptr) noexcept -> DeinitSrcExtClonePayLoad{

        return DeinitSrcExtClonePayLoad{ptr};
    }

    auto load_deinit_srcextclone_payload(DeinitSrcExtClonePayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::deinit_srcextclone(payload.ptr);
    }

    auto make_deinit_dstextclone_payload(uma_ptr_t ptr) noexcept -> DeinitDstExtClonePayLoad{

        return DeinitDstExtClonePayLoad{ptr};
    }

    auto load_deinit_dstextclone_payload(DeinitDstExtClonePayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::deinit_dstextclone(payload.ptr);
    }

    auto make_deinit_immu_payload(uma_ptr_t ptr) noexcept -> DeinitImmuPayLoad{

        return DeinitImmuPayLoad{ptr};
    }

    auto load_deinit_immu_payload(DeinitImmuPayLoad payload) noexcept -> exception_t{

        return dg::network_tile_lifetime::statix::deinit_immu(payload.ptr);
    }

    auto make_deinit_payload(uma_ptr_t ptr) noexcept -> DeinitPayLoad{

        return DeinitPayLoad{ptr};
    }

    auto load_deinit_payload(DeinitPayLoad payload) noexcept -> exception_t{
        
        return dg::network_tile_lifetime::statix::deinit(payload.ptr);
    }

    using payload_kind_t = uint8_t;

    enum enum_payload: payload_kind_t{
        payload_kind_init_leaf          = 0u,
        payload_kind_init_mono          = 1u,
        payload_kind_init_pair          = 2u,
        payload_kind_init_uacm          = 3u,
        payload_kind_init_pacm          = 4u,
        payload_kind_init_crit          = 5u,
        payload_kind_init_msgrfwd       = 6u,
        payload_kind_init_msgrbwd       = 7u,
        payload_kind_init_srcextclone   = 8u,
        payload_kind_init_dstextclone   = 9u,
        payload_kind_init_immu          = 10u,
        payload_kind_orphan_leaf        = 11u,
        payload_kind_orphan_mono        = 12u,
        payload_kind_orphan_pair        = 13u,
        payload_kind_orphan_uacm        = 14u,
        payload_kind_orphan_pacm        = 15u,
        payload_kind_orphan_crit        = 16u,
        payload_kind_orphan_msgrfwd     = 17u,
        payload_kind_orphan_msgrbwd     = 18u,
        payload_kind_orphan_srcextclone = 19u,
        payload_kind_orphan_dstextclone = 20u,
        payload_kind_orphan_immu        = 21u,
        payload_kind_orphan             = 22u,
        payload_kind_deinit_leaf        = 23u,
        payload_kind_deinit_mono        = 24u,
        payload_kind_deinit_pair        = 25u,
        payload_kind_deinit_uacm        = 26u,
        payload_kind_deinit_pacm        = 27u,
        payload_kind_deinit_crit        = 28u,
        payload_kind_deinit_msgrfwd     = 29u,
        payload_kind_deinit_msgrbwd     = 30u,
        payload_kind_deinit_srcextclone = 31u,
        payload_kind_deinit_dstextclone = 32u,
        payload_kind_deinit_immu        = 33u,
        payload_kind_deinit             = 34u
    };

    struct VirtualPayLoad{
        payload_kind_t kind;
        dg::string payload_content;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(kind, payload_content);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(kind, payload_content);
        }  
    };

    auto virtualize_payload(InitLeafPayload payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload); 
        rs.kind             = payload_kind_init_leaf;

        return rs;
    }

    auto virtualize_payload(InitMonoPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload); 
        rs.kind             = payload_kind_init_mono;

        return rs;
    }

    auto virtualize_payload(InitPairPayLoad payload) noexcept -> VirtualPayLoad{
        
        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload); 
        rs.kind             = payload_kind_init_pair;
        
        return rs;
    }

    auto virtualize_payload(InitUACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_init_uacm;

        return rs;
    }

    auto virtualize_payload(InitPACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_init_pacm;

        return rs;
    }
    
    auto virtualize_payload(InitCritPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_init_crit;

        return rs;
    }

    auto virtualize_payload(InitMsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_init_msgrfwd;

        return rs;
    }

    auto virtualize_payload(InitMsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_init_msgrbwd;

        return rs;
    }

    auto virtualize_payload(InitSrcExtClonePayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_init_srcextclone;

        return rs;
    }

    auto virtualize_payload(InitDstExtClonePayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_init_dstextclone;

        return rs;
    }

    auto virtualize_payload(InitImmuPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_init_immu;

        return rs;
    }

    auto virtualize_payload(OrphanLeafPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_leaf;

        return rs;
    }

    auto virtualize_payload(OrphanMonoPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_mono;

        return rs;
    }

    auto virtualize_payload(OrphanPairPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_pair;

        return rs;
    }

    auto virtualize_payload(OrphanUACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_uacm;

        return rs;
    }

    auto virtualize_payload(OrphanPACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_pacm;

        return rs;
    }

    auto virtualize_payload(OrphanCritPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_crit;

        return rs;
    }

    auto virtualize_payload(OrphanMsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_msgrfwd;

        return rs;
    }

    auto virtualize_payload(OrphanMsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_msgrbwd;

        return rs;
    }

    auto virtualize_payload(OrphanSrcExtClonePayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_srcextclone;

        return rs;
    }

    auto virtualize_payload(OrphanDstExtClonePayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_dstextclone;

        return rs;
    }

    auto virtualize_payload(OrphanImmuPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan_immu;

        return rs;
    }

    auto virtualize_payload(OrphanPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_orphan;

        return rs;
    }

    auto virtualize_payload(DeinitLeafPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_leaf;

        return rs;
    }

    auto virtualize_payload(DeinitMonoPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_mono;

        return rs;
    }

    auto virtualize_payload(DeinitPairPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_pair;

        return rs;
    }

    auto virtualize_payload(DeinitUACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_uacm;

        return rs;
    }

    auto virtualize_payload(DeinitPACMPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_pacm;

        return rs;
    }

    auto virtualize_payload(DeinitCritPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_crit;

        return rs;
    }

    auto virtualize_payload(DeinitMsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_msgrfwd;

        return rs;
    }

    auto virtualize_payload(DeinitMsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_msgrbwd;

        return rs;
    }

    auto virtualize_payload(DeinitSrcExtClonePayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_srcextclone;

        return rs;
    }

    auto virtualize_payload(DeinitDstExtClonePayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_dstextclone;

        return rs;
    }

    auto virtualize_payload(DeinitImmuPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit_immu;

        return rs;
    }

    auto virtualize_payload(DeinitPayLoad payload) noexcept -> VirtualPayLoad{

        VirtualPayLoad rs{};
        rs.payload_content  = dg::network_compact_serializer::serialize<dg::string>(payload);
        rs.kind             = payload_kind_deinit;

        return rs;
    }

    auto load_virtual_payload(VirtualPayLoad payload) noexcept -> exception_t{

        switch (payload.kind){
            case payload_kind_init_leaf:
            {
                InitLeafPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_int_leaf_payload(std::move(devirt_payload));
            }
            case payload_kind_init_mono:
            {
                InitMonoPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_init_mono_payload(std::move(devirt_payload));
            }
            case payload_kind_init_pair:
            {
                InitPairPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_init_pair_payload(std::move(devirt_payload));
            }
            case payload_kind_init_uacm:
            {
                InitUACMPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_init_uacm_payload(std::move(devirt_payload));
            }
            case payload_kind_init_pacm:
            {
                InitPACMPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_init_pacm_payload(std::move(devirt_payload));
            }
            case payload_kind_init_crit:
            {
                InitCritPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_init_crit_payload(std::move(devirt_payload));
            }
            case payload_kind_init_msgrfwd:
            {
                InitMsgrFwdPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_init_msgrfwd_payload(std::move(devirt_payload));
            }
            case payload_kind_init_msgrbwd:
            {
                InitMsgrBwdPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_init_msgrbwd_payload(std::move(devirt_payload));
            }
            case payload_kind_init_srcextclone:
            {
                InitSrcExtClonePayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_init_srcextclone_payload(std::move(devirt_payload));
            }
            case payload_kind_init_dstextclone:
            {
                InitDstExtClonePayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_init_dstextclone_payload(std::move(devirt_payload));
            }
            case payload_kind_init_immu:
            {
                InitImmuPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_init_immu_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_leaf:
            {
                OrphanLeafPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_leaf_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_mono:
            {
                OrphanMonoPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_mono_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_pair:
            {
                OrphanPairPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_pair_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_uacm:
            {
                OrphanUACMPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_uacm_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_pacm:
            {
                OrphanPACMPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_pacm_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_crit:
            {
                OrphanCritPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_crit_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_msgrfwd:
            {
                OrhapnMsgrFwdPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_msgrfwd_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_msgrbwd:
            {
                OrphanMsgrBwdPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_msgrbwd_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_srcextclone:
            {
                OrphanSrcExtClonePayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_srcextclone_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_dstextclone:
            {
                OrphanDstExtClonePayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_dstextclone_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan_immu:
            {
                OrphanImmuPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_immu_payload(std::move(devirt_payload));
            }
            case payload_kind_orphan:
            {
                OrphanPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_orphan_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_leaf:
            {
                DeinitLeafPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_deinit_leaf_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_mono:
            {
                DeinitMonoPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_deinit_mono_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_pair:
            {
                DeinitPairPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_deinit_pair_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_uacm:
            {
                DeinitUACMPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_deinit_uacm_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_pacm:
            {
                DeinitPACMPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payhload.payload_content.data());
                return load_deinit_pacm_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_crit:
            {
                DeinitCritPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_deinit_crit_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_msgrfwd:
            {
                DeinitMsgrFwdPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_deinit_msgrfwd_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_msgrbwd:
            {
                DeinitMsgrBwdPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_deinit_msgrbwd_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_srcextclone:
            {
                DeinitSrcExtClonePayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_deinit_srcextclone_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_dstextclone:
            {
                DeinitDstExtClonePayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_deinit_dstextclone_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit_immu:
            {
                DeinitImmuPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_deinit_immu_payload(std::move(devirt_payload));
            }
            case payload_kind_deinit:
            {
                DeinitPayLoad devirt_payload{};
                dg::network_compact_serializer::deserialize_into(devirt_payload, payload.payload_content,data());
                return load_deinit_payload(std::move(devirt_payload));
            }
            default:
            {
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                } else{
                    std::unreachable();
                    return {};
                }
            }
        }
    }

    void load_virtual_payload_arr(VirtualPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        //this code looks dumb but this is an optimization trick that compilers are not allowed to make - it is called polymorphic dispatch - where we want to move the polymorphic up one level to void * - then we radix the dispatch by using polymorphic table(void *)
        //alright - this moves from 40 flops per table dispatch -> 3-5 flops - which is good - even though we are dispatching fat tiles - 64x64
        //we are actually facing a problem of 4GB of CPU flops/ 40TB of GPU flops - the ratio is 1/ 1 << 15
        //64x64 = 1 << 12 / 1 << 4 = 1 << 8, so we are still 1 << 7 shy
        //that statistic will tell you how hard it is to program tile logics
        //this is a very crucial part of synchronous brain

        //you probably see the tile_member_access and think it is not at all important
        //that is actually the hardest part of this program - is to fit all these getters in L1 cache and dispatch them to GPU - without that, this entire thing would be in L2-L3 and we defeat our purpose of dispatching the tiles to cuda in the first place

        constexpr size_t DISPATCH_DELIVERY_CAP = 32u;

        auto init_leaf_dispatcher           = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitLeafPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_leaf_payload(std::move(cur_payload));
            }
        };

        auto init_mono_dispatcher           = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitMonoPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_mono_payload(std::move(cur_payload));
            }
        };

        auto init_pair_dispatcher           = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitPairPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_pair_payload(std::move(cur_payload));
            }
        };

        auto init_uacm_dispatcher           = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitUACMPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_uacm_payload(std::move(cur_payload));
            }
        };

        auto init_pacm_dispatcher           = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitPACMPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_pacm_payload(std::move(cur_payload));
            }
        };

        auto init_crit_dispatcher           = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitCritPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_crit_payload(std::move(cur_payload));
            }
        };

        auto init_msgrfwd_dispatcher        = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitMsgrFwdPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_msgrfwd_payload(std::move(cur_payload));
            }
        };

        auto init_msgrbwd_dispatcher        = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitMsgrBwdPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_msgrbwd_payload(std::move(cur_payload));
            }
        };

        auto init_srcextclone_dispatcher    = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitSrcExtClonePayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_srcextclone_payload(std::move(cur_payload));
            }
        };

        auto init_dstextclone_dispatcher    = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitDstExtClonePayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_dstextclone_payload(std::move(cur_payload));
            }
        };

        auto init_immu_dispatcher           = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            InitImmuPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_init_immu_payload(std::move(cur_payload));
            }
        };

        auto orphan_leaf_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanLeafPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_leaf_payload(std::move(cur_payload));
            }
        };

        auto orphan_mono_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanMonoPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_mono_payload(std::move(cur_payload));
            }
        };

        auto orphan_pair_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanPairPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_pair_payload(std::move(cur_payload));
            }
        };

        auto orphan_uacm_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanUACMPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_uacm_payload(std::move(cur_payload));
            }
        };

        auto orphan_pacm_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanPACMPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_pacm_payload(std::move(cur_payload));
            }
        };

        auto orphan_crit_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanCritPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_crit_payload(std::move(cur_payload));
            }
        };

        auto orphan_msgrfwd_dispatcher      = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanMsgrFwdPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_msgrfwd_payload(std::move(cur_payload));
            }
        };

        auto orphan_msgrbwd_dispatcher      = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanMsgrBwdPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_msgrbwd_payload(std::move(cur_payload));
            }
        };

        auto orphan_srcextclone_dispatcher  = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanSrcExtClonePayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_srcextclone_payload(std::move(cur_payload));
            }
        };

        auto orphan_dstextclone_dispatcher  = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanDstExtClonePayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_dstextclone_payload(std::move(cur_payload));
            }
        };

        auto orphan_immu_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanImmuPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_immu_payload(std::move(cur_payload));
            }
        };

        auto orphan_dispatcher              = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            OrphanPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_orphan_payload(std::move(cur_payload));
            }
        };

        auto deinit_leaf_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitLeafPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_leaf_payload(std::move(cur_payload));
            }
        };

        auto deinit_mono_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitMonoPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_mono_payload(std::move(cur_payload));
            }
        };

        auto deinit_pair_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitPairPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first,payload_content.data());
                *vec_pair.second = load_deinit_pair_payload(std::move(cur_payload));
            }
        };

        auto deinit_uacm_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitUACMPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_uacm_payload(std::move(cur_payload));
            }
        };

        auto deinit_pacm_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitPACMPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_pacm_payload(std::move(cur_payload));
            }
        };

        auto deinit_crit_dispatcher         = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitCritPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_crit_payload(std::move(cur_payload));
            }
        };

        auto deinit_msgrfwd_dispatcher      = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitMsgrFwdPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_msgrfwd_payload(std::move(cur_payload));
            }
        };

        auto deinit_msgrbwd_dispatcher      = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitMsgrBwdPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_msgrbwd_payload(std::move(cur_payload));
            }
        };

        auto deinit_srcextclone_dispatcher  = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitSrcExtClonePayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_srcextclone_payload(std::move(cur_payload));
            }
        };

        auto deinit_dstextclone_dispatcher  = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitDstExtClonePayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_dstextclone_payload(std::move(cur_payload));
            }
        };
        
        auto deinit_immu_dispatcher         = []{dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec} noexcept{
            DeinitImmuPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_immu_payload(std::move(cur_payload));
            }
        };

        auto deinit_dispatcher              = [](dg::vector<std::pair<VirtualPayLoad, exception_t *>> vec) noexcept{
            DeinitPayLoad cur_payload{};
            for (const auto& vec_pair: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, vec_pair.first.payload_content.data());
                *vec_pair.second = load_deinit_payload(std::move(cur_payload));
            }
        };

        auto init_leaf_consumer                     = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_leaf_dispatcher)>(std::move(init_leaf_dispatcher));
        auto init_mono_consumer                     = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_mono_dispatcher)>(std::move(init_mono_dispatcher));
        auto init_pair_consumer                     = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_pair_dispatcher)>(std::move(init_pair_dispatcher));
        auto init_uacm_consumer                     = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_uacm_dispatcher)>(std::move(init_uacm_dispatcher));
        auto init_pacm_consumer                     = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_pacm_dispatcher)>(std::move(init_pacm_dispatcher));
        auto init_crit_consumer                     = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_crit_dispatcher)>(std::move(init_crit_dispatcher));
        auto init_msgrfwd_consumer                  = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_msgrfwd_dispatcher)> (std::move(init_msgrfwd_dispatcher));
        auto init_msgrbwd_consumer                  = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_msgrbwd_dispatcher)>(std::move(init_msgrbwd_dispatcher));
        auto init_srcextclone_consumer              = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_srcextclone_dispatcher)>(std::move(init_srcextclone_dispatcher));
        auto init_dstextclone_consumer              = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_dstextclone_dispatcher)>(std::move(init_dstextclone_dispatcher));
        auto init_immu_consumer                     = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_immu_dispatcher)>(std::move(init_immu_dispatcher));
        auto orphan_leaf_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_leaf_dispatcher)>(std::move(orphan_leaf_dispatcher));
        auto orphan_mono_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_mono_dispatcher)>(std::move(orphan_mono_dispatcher));
        auto orphan_pair_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_pair_dispatcher)>(std::move(orphan_pair_dispatcher));
        auto orphan_uacm_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_uacm_dispatcher)> (std::move(orphan_uacm_dispatcher));
        auto orphan_pacm_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_pacm_dispatcher)>(std::move(orphan_pacm_dispatcher));
        auto orphan_crit_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_crit_dispatcher)>(std::move(orphan_crit_dispatcher));
        auto orphan_msgrfwd_consumer                = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_msgrfwd_dispatcher)> (std::move(orphan_msgrfwd_dispatcher));
        auto orphan_msgrbwd_consumer                = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_msgrbwd_dispatcher)>(std::move(orphan_msgrbwd_dispatcher));
        auto orphan_srcextclone_consumer            = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_srcextclone_dispatcher)>(std::move(orphan_srcextclone_dispatcher));
        auto orphan_dstextclone_consumer            = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_dstextclone_dispatcher)>(std::move(orphan_dstextclone_dispatcher));
        auto orphan_immu_consumer                   = dg::networK_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_immu_dispatcher)>(std::move(orphan_immu_dispatcher));
        auto orphan_consumer                        = dg::networK_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_dispatcher)>(std::move(orphan_dispatcher));
        auto deinit_leaf_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_leaf_dispatcher)>(std::move(deinit_leaf_dispatcher));
        auto deinit_mono_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_mono_dispatcher)>(std::move(deinit_mono_dispatcher));
        auto deinit_pair_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_pair_dispatcher)>(std::move(deinit_pair_dispatcher));
        auto deinit_uacm_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_uacm_dispatcher)>(std::move(deinit_uacm_dispatcher));
        auto deinit_pacm_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_pacm_dispatcher)>(std::move(deinit_pacm_dispatcher));
        auto deinit_crit_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_crit_dispatcher)>(std::move(deinit_crit_dispatcher));
        auto deinit_msgrfwd_consumer                = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_msgrfwd_dispatcher)> (std::move(deinit_msgrfwd_dispatcher));
        auto deinit_msgrbwd_consumer                = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_msgrbwd_dispatcher)>(std::move(deinit_msgrbwd_dispatcher));
        auto deinit_srcextclone_consumer            = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_srcextclone_dispatcher)>(std::move(deinit_srcextclone_dispatcher));
        auto deinit_dstextclone_consumer            = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_dstextclone_dispatcher)>(std::move(deinit_dstextclone_dispatcher));
        auto deinit_immu_consumer                   = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_immu_dispatcher)>(std::move(deinit_immu_dispatcher));
        auto deinit_consumer                        = dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(deinit_dispatcher)>(std::move(deinit_dispatcher));

        stdx::seq_cst_guard memcst_guard; //this is to disallow compiler memory ordering - to deinitialize init... before delivery_handles 

        auto init_leaf_delivery_handle              = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_leaf_consumer, DISPATCH_DELIVERY_CAP);
        auto init_mono_delivery_handle              = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_mono_consumer, DISPATCH_DELIVERY_CAP);
        auto init_pair_delivery_handle              = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_pair_consumer, DISPATCH_DELIVERY_CAP);
        auto init_uacm_delivery_handle              = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_uacm_consumer, DISPATCH_DELIVERY_CAP);
        auto init_pacm_delivery_handle              = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_pacm_consumer, DISPATCH_DELIVERY_CAP);
        auto init_crit_delivery_handle              = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_crit_consumer, DISPATCH_DELIVERY_CAP);
        auto init_msgrfwd_delivery_handle           = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_msgrfwd_consumer, DISPATCH_DELIVERY_CAP);
        auto init_msgrbwd_delivery_handle           = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_msgrbwd_consumer, DISPATCH_DELIVERY_CAP);
        auto init_srcextclone_delivery_handle       = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_srcextclone_consumer, DISPATCH_DELIVERY_CAP);
        auto init_dstextclone_delivery_handle       = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_dstextclone_consumer, DISPATCH_DELIVERY_CAP);
        auto init_immu_delivery_handle              = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&init_immu_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_leaf_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_leaf_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_mono_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_mono_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_pair_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_pair_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_uacm_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_uacm_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_pacm_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_pacm_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_crit_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_crit_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_msgrfwd_delivery_handle         = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_msgrfwd_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_msgrbwd_delivery_handle         = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_msgrbwd_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_srcextclone_delivery_handle     = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_srcextclone_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_dstextclone_delivery_handle     = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_dstextclone_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_immu_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_immu_consumer, DISPATCH_DELIVERY_CAP);
        auto orphan_delivery_handle                 = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&orphan_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_leaf_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_leaf_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_mono_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_mono_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_pair_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_pair_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_uacm_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_uacm_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_pacm_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_pacm_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_crit_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_crit_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_msgrfwd_delivery_handle         = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_msgrfwd_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_msgrbwd_delivery_handle         = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_msgrbwd_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_srcextclone_delivery_handle     = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_srcextclone_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_dstextclone_delivery_handle     = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_dstextclone_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_immu_delivery_handle            = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_immu_consumer, DISPATCH_DELIVERY_CAP);
        auto deinit_delivery_handle                 = dg::network_raii_producer_consumer::delvrsrv_open_raiihandle(&deinit_consumer, DISPATCH_DELIVERY_CAP);

        if (!dg::network_exception::conjunc_expect_has_value(init_leaf_delivery_handle, init_mono_delivery_handle, init_pair_delivery_handle, init_uacm_delivery_handle,
                                                             init_pacm_delivery_handle, init_crit_delivery_handle, init_msgrfwd_delivery_handle, init_msgrbwd_delivery_handle,
                                                             init_srcextclone_delivery_handle, init_dstextclone_delivery_handle, init_immu_delivery_handle, orphan_leaf_delivery_handle,
                                                             orphan_mono_delivery_handle, orphan_pair_delivery_handle, orphan_uacm_delivery_handle, orphan_pacm_delivery_handle,
                                                             orphan_crit_delivery_handle, orphan_msgrfwd_delivery_handle, orphan_msgrbwd_delivery_handle, orphan_srcextclone_delivery_handle,
                                                             orphan_dstextclone_delivery_handle, orphan_immu_delivery_handle, orphan_delivery_handle, deinit_leaf_delivery_handle,
                                                             deinit_mono_delivery_handle, deinit_pair_delivery_handle, deinit_uacm_delivery_handle, deinit_pacm_delivery_handle,
                                                             deinit_crit_delivery_handle, deinit_msgrfwd_delivery_handle, deinit_msgrbwd_delivery_handle, deinit_srcextclone_delivery_handle,
                                                             deinit_dstextclone_delivery_handle, deinit_immu_delivery_handle, deinit_delivery_handle)){

            std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::RESOURCE_EXHAUSTION);
            return;
        }

        for (size_t i = 0u; i < sz; ++i){
            auto payload_kind                   = payload_arr[i].kind;
            VirtualPayLoad dispatching_payload  = payload_arr[i];
            exception_t * cur_exception         = std::next(exception_arr, i);

            switch (payload_kind){
                case payload_kind_init_leaf:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_leaf_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_init_mono:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_mono_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_init_pair:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_pair_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_init_uacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_uacm_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_init_pacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_pacm_delivery_handle)->get(), sstd::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_init_crit:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_crit_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_init_msgrfwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_msgrfwd_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_init_msgrbwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_msgrbwd_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_init_srcextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_srcextclone_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_init_dstextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_dstextclone_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_init_immu:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_immu_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_leaf:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_leaf_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_mono:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_mono_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_pair:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_pair_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_uacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_uacm_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_pacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_pacm_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_crit:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_crit_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_msgrfwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_msgrfwd_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_msgrbwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_msgrbwd_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_srcextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_srcextclone_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_dstextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_dstextclone_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan_immu:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_immu_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_orphan:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_leaf:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_leaf_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_mono:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_mono_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_pair:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_pair_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_uacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_uacm_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_pacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_pacm_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_crit:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_crit_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_msgrfwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_msgrfwd_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_msgrbwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_msgrbwd_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_srcextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_srcextclone_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_dstextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_dstextclone_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit_immu:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_immu_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
                    break;
                case payload_kind_deinit:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_delivery_handle)->get(), std::make_pair(std::move(dispatching_payload), cur_exception));
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
}

#endif