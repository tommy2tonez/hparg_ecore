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

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_leaf_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        size_t pointing_logit_sz = get_leaf_logit_group_size_nothrow(ptr); 

        if (pointing_logit_sz != logit_value_sz) [[unlikely]]{
            return dg::network_exception::INVALID_ARGUMENT;
        }

        set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_INITIALIZED);
        set_leaf_logit_nothrow(ptr, logit_value);
        set_leaf_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_leaf_operatable_id_nothrow(ptr, operatable_id);
        set_leaf_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_mono(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_mono_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        
        set_mono_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_mono_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_mono_dispatch_control_nothrow(ptr, dispatch_control);
        set_mono_operatable_id_nothrow(ptr, operatable_id);
        set_mono_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_mono_descendant_nothrow(ptr, src);
        set_mono_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);
    
        return dg::network_exception::SUCCESS;
    }

    auto init_pair(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }
        
        uma_ptr_t rcu_addr = get_pair_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

        set_pair_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_pair_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_pair_operatable_id_nothrow(ptr, operatable_id);
        set_pair_dispatch_control_nothrow(ptr, dispatch_control);
        set_pair_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_pair_left_descendant_nothrow(ptr, lhs);
        set_pair_right_descendant_nothrow(ptr, rhs);
        set_pair_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);
    
        return dg::network_exception::SUCCESS;
    }

    auto init_uacm(uma_ptr_t ptr, std::array<uma_ptr_t, UACM_ACM_SZ> src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_uacm_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

        set_uacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_uacm_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_uacm_operatable_id_nothrow(ptr, operatable_id);
        set_uacm_dispatch_control_nothrow(ptr, dispatch_control);
        set_uacm_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_uacm_descendant_nothrow(ptr, src);
        set_uacm_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);
    
        return dg::network_exception::SUCCESS;
    }

    auto init_pacm(uma_ptr_t ptr, std::array<uma_ptr_t, PACM_ACM_SZ> lhs, std::array<uma_ptr_t, PACM_ACM_SZ> rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_pacm_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        
        set_pacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_pacm_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_pacm_operatable_id_nothrow(ptr, operatable_id);
        set_pacm_dispatch_control_nothrow(ptr, dispatch_control);
        set_pacm_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_pacm_left_descendant_nothrow(ptr, lhs);
        set_pacm_right_descendant_nothrow(ptr, rhs);
        set_pacm_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_crit(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, crit_kind_t crit_kind, void * clogit_value, size_t clogit_value_sz) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_crit_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        size_t pointing_clogit_value_sz = get_crit_clogit_group_size_nothrow(ptr);

        if (pointing_clogit_value_sz != clogit_value_sz) [[unlikely]]{
            return dg::network_exception::INVALID_ARGUMENT;
        }

        set_crit_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_crit_clogit_nothrow(ptr, clogit_value);
        set_crit_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_crit_operatable_id_nothrow(ptr, operatable_id);
        set_crit_dispatch_control_nothrow(ptr, dispatch_control);
        set_crit_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_crit_descendant_nothrow(ptr, src);
        set_crit_kind_nothrow(ptr, crit_kind);
        set_crit_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_msgrfwd(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, dst_info_t dst_info) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_msgrfwd_rcu_addr_nothrow(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        
        set_msgrfwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_msgrfwd_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_msgrfwd_operatable_id_nothrow(ptr, operatable_id);
        set_msgrfwd_dispatch_control_nothrow(ptr, dispatch_control);
        set_msgrfwd_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_msgrfwd_descendant_nothrow(ptr, src);
        set_msgrfwd_dst_info_nothrow(ptr, dst_info);
        set_msgrfwd_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_msgrbwd(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, timein_t timein, dst_info_t dst_info) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_msgrbwd_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

        set_msgrbwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_msgrbwd_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
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

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_srcextclone_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_srcextclone_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

        set_srcextclone_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_srcextclone_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_srcextclone_operatable_id_nothrow(ptr, operatable_id);
        set_srcextclone_dispatch_control_nothrow(ptr, dispatch_control);
        set_srcextclone_pong_count_nothrow(ptr, dg::network_tile_metadata::TILE_PONG_COUNT_DEFAULT);
        set_srcextclone_descendant_nothrow(ptr, src);
        set_srcextclone_counterpart_nothrow(ptr, counterpart);
        set_srcextclone_grad_status_nothrow(ptr, dg::network_tile_metadata::TILE_GRAD_STATUS_UNINITIALIZED);

        return dg::network_exception::SUCCESS;
    }

    auto init_dstextclone(uma_ptr_t ptr, uma_ptr_t counterpart, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_dstextclone_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_dstextclone_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);

        set_dstextclone_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ADOPTED);
        set_dstextclone_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_dstextclone_operatable_id_nothrow(ptr, operatable_id);
        set_dstextclone_dispatch_control_nothrow(ptr, dispatch_control);
        set_dstextclone_counterpart_nothrow(ptr, counterpart);

        return dg::network_exception::SUCCESS;
    }

    auto init_immu(uma_ptr_t ptr, operatable_id_t operatable_id, void * logit_value, size_t logit_value_sz) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_immu_ptr_access(ptr);

        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_immu_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        size_t pointing_logit_value_sz = get_immu_logit_group_size(ptr);

        if (pointing_logit_value_sz != logit_value_sz) [[unlikely]]{
            return dg::network_exception::INVALID_ARGUMENT;
        } 

        set_immu_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_INITIALIZED);
        set_immu_logit_nothrow(ptr, logit_value);
        set_immu_observer_nothrow(ptr, dg::network_tile_metadata::TILE_OBSERVER_ARRAY_DEFAULT);
        set_immu_operatable_id_nothrow(ptr, operatable_id);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_leaf(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr);

        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_leaf_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_leaf_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED); 

        return dg::network_exception::SUCCESS;
    }

    auto orphan_mono(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_mono_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_mono_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_pair(uma_ptr_t ptr) noexcept -> exception_t{
        
        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_pair_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_pair_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_uacm(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_uacm_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_uacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_pacm(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_pacm_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_pacm_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_crit(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_crit_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_crit_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_msgrfwd(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_msgrfwd_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_msgrfwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_msgrbwd(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_msgrbwd_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_msgrbwd_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_srcextclone(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_srcextclone_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_srcextclone_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_srcextclone_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_dstextclone(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_dstextclone_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_dstextclone_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_dstextclone_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan_immu(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_immu_ptr_access(ptr);

        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_immu_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_immu_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto orphan(uma_ptr_t ptr) noexcept -> exception_t{

        using namespace network_tile_member_getsetter;

        auto ptr_access = dg::network_tile_member_access::safecthrow_tile_ptr_access(ptr);
        
        if (!ptr_access.has_value()) [[unlikely]]{
            return ptr_access.error();
        }

        uma_ptr_t rcu_addr = get_tile_rcu_addr(ptr);
        dg::network_memops_uma::memlock_guard lck_grd(rcu_addr);
        set_tile_init_status_nothrow(ptr, dg::network_tile_metadata::TILE_INIT_STATUS_ORPHANED);

        return dg::network_exception::SUCCESS;
    }

    auto deinit_leaf(uma_ptr_t ptr) noexcept -> exception_t{

    }

    auto deinit_mono(uma_ptr_t ptr) noexcept -> exception_t{

    }

    auto deinit_pair(uma_ptr_t ptr) noexcept -> exception_t{

    }

    auto deinit_uacm(uma_ptr_t ptr) noexcept -> exception_t{

    }

    auto deinit_pacm(uma_ptr_t ptr) noexcept -> exception_t{ 

    }

    auto deinit_crit(uma_ptr_t ptr) noexcept -> exception_t{

    }

    auto deinit_msgrfwd(uma_ptr_t ptr) noexcept -> exception_t{

    }

    auto deinit_msgrbwd(uma_ptr_t ptr) noexcept -> exception_t{
        
    }

    auto deinit_srcextclone(uma_ptr_t ptr) noexcept -> exception_t{

    }

    auto deinit_dstextclone(uma_ptr_t ptr) noexcept -> exception_t{

    }

    auto deinit_immu(uma_ptr_t ptr) noexcept -> exception_t{

    }

    auto deinit(uma_ptr_t ptr) noexcept -> exception_t{

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

    auto load_orphan_leaf_payload(OrphanLeafPayLoad) noexcept -> exception_t{

    }

    auto make_orphan_mono_payload(uma_ptr_t ptr) noexcept -> OrphanMonoPayLoad{

        return OrphanMonoPayLoad{ptr};
    }

    auto load_orphan_mono_payload(OrphanMonoPayLoad) noexcept -> exception_t{

    }

    auto make_orphan_pair_payload(uma_ptr_t ptr) noexcept -> OrphanPairPayLoad{

        return OrphanPairPayLoad{ptr};
    }

    auto load_orphan_pair_payload(OrphanPairPayLoad) noexcept -> exception_t{

    }

    auto make_orphan_uacm_payload(uma_ptr_t ptr) noexcept -> OrphanUACMPayLoad{

        return OrphanUACMPayLoad{ptr};
    }

    auto load_orphan_uacm_payload(OrphanUACMPayLoad) noexcept -> exception_t{

    }

    auto make_orphan_pacm_payload(uma_ptr_t ptr) noexcept -> OrphanPACMPayLoad{

        return OrphanPACMPayLoad{ptr};
    }

    auto load_orphan_pacm_payload(OrphanPACMPayLoad) noexcept -> exception_t{

    }

    auto make_orphan_crit_payload(uma_ptr_t ptr) noexcept -> OrphanCritPayLoad{

        return OrphanCritPayLoad{ptr};
    }

    auto load_orphan_crit_payload(OrphanCritPayLoad) noexcept -> exception_t{

    }

    auto make_orphan_msgrfwd_payload(uma_ptr_t ptr) noexcept -> OrphanMsgrFwdPayLoad{

        return OrphanMsgrFwdPayLoad{ptr};
    }

    auto load_orphan_msgrfwd_payload(OrphanMsgrFwdPayLoad) noexcept -> exception_t{

    }

    auto make_orphan_msgrbwd_payload(uma_ptr_t ptr) noexcept -> OrphanMsgrBwdPayLoad{

        return OrphanMsgrBwdPayLoad{ptr};
    }

    auto load_orphan_msgrbwd_payload(OrphanMsgrBwdPayLoad) noexcept -> exception_t{

    }

    auto make_orphan_srcextclone_payload(uma_ptr_t ptr) noexcept -> OrphanSrcExtClonePayLoad{

        return OrphanSrcExtClonePayLoad{ptr};
    }
    
    auto load_orphan_srcextclone_payload(OrphanSrcExtClonePayLoad) noexcept -> exception_t{

    }

    auto make_orphan_dstextclone_payload(uma_ptr_t ptr) noexcept -> OrphanDstExtClonePayLoad{

        return OrphanDstExtClonePayLoad{ptr};
    }

    auto load_orphan_dstextclone_payload(OrphanDstExtClonePayLoad) noexcept -> exception_t{

    }

    auto make_orphan_immu_payload(uma_ptr_t ptr) noexcept -> OrphanImmuPayLoad{

        return OrphanImmuPayLoad{ptr};
    }

    auto load_orphan_immu_payload(OrphanImmuPayLoad) noexcept -> exception_t{

    }

    auto make_orphan_payload(uma_ptr_t ptr) noexcept -> OrphanPayLoad{

        return OrphanPayLoad{ptr};
    }

    auto load_orphan_payload(OrphanPayLoad) noexcept -> exception_t{

    }

    static inline constexpr size_t MAX_PAYLOAD_CONTENT_SZ = size_t{1} << 6; 
    using payload_kind_t = uint8_t;

    enum enum_payload_kind: payload_kind_t{
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

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        // dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_init_leaf;

        return rs;
    }

    auto virtualize_payload(InitMonoPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        // dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_init_mono;

        return rs;
    }

    auto virtualize_payload(InitPairPayLoad payload) noexcept -> VirtualPayLoad{
        
        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        // dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_init_pair;
        
        return rs;
    }

    auto virtualize_payload(InitUACMPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);;
        VirtualPayLoad rs{};
        // dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_init_uacm;

        return rs;
    }

    auto virtualize_payload(InitPACMPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        // dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_init_pacm;

        return rs;
    }
    
    auto virtualize_payload(InitCritPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        // dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_init_crit;

        return rs;
    }

    auto virtualize_payload(InitMsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        // dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_init_msgrfwd;

        return rs;
    }

    auto virtualize_payload(InitMsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        // dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_init_msgrbwd;

        return rs;
    }
    
    auto load_virtual_payload(VirtualPayLoad payload) noexcept -> exception_t{

        switch (payload.kind){
            case payload_kind_init_leaf:
            {
                InitLeafPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_leaf_payload(devirt_payload);
            }
            case payload_kind_init_mono:
            {
                InitMonoPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_mono_payload(devirt_payload);
            }
            case payload_kind_init_pair:
            {
                InitPairPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_pair_payload(devirt_payload);
            }
            case payload_kind_init_uacm:
            {
                InitUACMPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_uacm_payload(devirt_payload);
            }
            case payload_kind_init_pacm:
            {
                InitPACMPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_pacm_payload(devirt_payload);
            }
            case payload_kind_init_crit:
            {
                InitCritPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_crit_payload(devirt_payload);
            }
            case payload_kind_init_msgrfwd:
            {
                InitMsgrFwdPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_msgrfwd_payload(devirt_payload);
            }
            case payload_kind_init_msgrbwd:
            {
                InitMsgrBwdPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_msgrbwd_payload(devirt_payload);
            }
            default:
            {
                return dg::network_exception::INVALID_ARGUMENT;
            }
        }
    }

    void load_virtual_payload_arr(VirtualPayLoad * payload_arr, exception_t * exception_arr, size_t sz) noexcept{

        auto init_leaf_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitLeafPayLoad cur_payload{};
            for (const auto& virt_payload: vec){ //optimizables - this is skewed load - this is bad - this is precisely why I dont want to init_leaf with heavy logit_value
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_leaf_payload(std::move(cur_payload));
            }
        };

        auto init_mono_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitMonoPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_mono_payload(std::move(cur_payload));
            }
        };

        auto init_pair_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitPairPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_pair_payload(std::move(cur_payload));
            }
        };

        auto init_uacm_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitUACMPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_uacm_payload(std::move(cur_payload));
            }
        };

        auto init_pacm_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitPACMPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_pacm_payload(std::move(cur_payload));
            }
        };

        auto init_crit_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitCritPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_crit_payload(std::move(cur_payload));
            }
        };

        auto init_msgrfwd_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitMsgrFwdPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_msgrfwd_payload(std::move(cur_payload));
            }
        };

        auto init_msgrbwd_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitMsgrBwdPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_msgrbwd_payload(std::move(cur_payload));
            }
        };

        auto init_srcextclone_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitSrcExtClonePayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_srcextclone_payload(std::move(cur_payload));
            }
        };

        auto init_dstextclone_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitDstExtClonePayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_dstextclone_payload(std::move(cur_payload));
            }
        };

        auto init_immu_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            InitImmuPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_init_immu_payload(std::move(cur_payload));
            }
        };

        auto orphan_leaf_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanLeafPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_leaf_payload(std::move(cur_payload));
            }
        };

        auto orphan_mono_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanMonoPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_mono_payload(std::move(cur_payload));
            }
        };

        auto orphan_pair_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanPairPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_pair_payload(std::move(cur_payload));
            }
        };

        auto orphan_uacm_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanUACMPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_uacm_payload(std::move(cur_payload));
            }
        };

        auto orphan_pacm_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanPACMPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_pacm_payload(std::move(cur_payload));
            }
        };

        auto orphan_crit_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanCritPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_crit_payload(std::move(cur_payload));
            }
        };

        auto orphan_msgrfwd_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanMsgrFwdPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_msgrfwd_payload(std::move(cur_payload));
            }
        };

        auto orphan_msgrbwd_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanMsgrBwdPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_msgrbwd_payload(std::move(cur_payload));
            }
        };

        auto orphan_srcextclone_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanSrcExtClonePayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_srcextclone_payload(std::move(cur_payload));
            }
        };

        auto orphan_dstextclone_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanDstExtClonePayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_dstextclone_payload(std::move(cur_payload));
            }
        };

        auto orphan_immu_dispatcher = [](dg::vector<VirtualPayLoad> vec) noexcept{
            OrphanImmuPayLoad cur_payload{};
            for (const auto& virt_payload: vec){
                dg::network_compact_serializer::deserialize_into(cur_payload, virt_payload.payload_content.data());
                load_orphan_immu_payload(std::move(cur_payload));
            }
        };

        auto deinit_leaf_dispatcher = []{};
        auto deinit_mono_dispatcher = []{};
        auto deinit_pair_dispatcher = []{};
        auto deinit_uacm_dispatcher = []{};
        auto deinit_crit_dispatcher = []{};
        auto deinit_msgrfwd_dispatcher = []{};
        auto deinit_msgrbwd_dispatcher = []{};
        auto deinit_srcextclone_dispatcher = []{};
        auto deinit_dstextclone_dispatcher = []{};
        auto deinit_immu_dispatcher = []{};

        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_leaf_dispatcher)> init_leaf_consumer(std::move(init_leaf_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_mono_dispatcher)> init_mono_consumer(std::move(init_mono_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_pair_dispatcher)> init_pair_consumer(std::move(init_pair_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_uacm_dispatcher)> init_uacm_consumer(std::move(init_uacm_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_pacm_dispatcher)> init_pacm_consumer(std::move(init_pacm_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_crit_dispatcher)> init_crit_consumer(std::move(init_crit_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_msgrfwd_dispatcher)> init_msgrfwd_consumer(std::move(init_msgrfwd_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_msgrbwd_dispatcher)> init_msgrbwd_consumer(std::move(init_msgrbwd_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_srcextclone_dispatcher)> init_srcextclone_consumer(std::move(init_srcextclone_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_dstextclone_dispatcher)> init_dstextclone_consumer(std::move(init_dstextclone_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(init_immu_dispatcher)> init_immu_consumer(std::move(init_immu_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_leaf_dispatcher)> orphan_leaf_consumer(std::move(orphan_leaf_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_mono_dispatcher)> orphan_mono_consumer(std::move(orphan_mono_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_pair_dispatcher)> orphan_pair_consumer(std::move(orphan_pair_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_uacm_dispatcher)> orphan_uacm_consumer(std::move(orphan_uacm_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_pacm_dispatcher)> orphan_pacm_consumer(std::move(orphan_pacm_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_crit_dispatcher)> orphan_crit_consumer(std::move(orphan_crit_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_msgrfwd_dispatcher)> orphan_msgrfwd_consumer(std::move(orphan_msgrfwd_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_msgrbwd_dispatcher)> orphan_msgrbwd_consumer(std::move(orphan_msgrbwd_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_srcextclone_dispatcher)> orphan_srcextclone_consumer(std::move(orphan_srcextclone_dispatcher));
        dg::network_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_dstextclone_dispatcher)> orphan_dstextclone_consumer(std::move(orphan_dstextclone_dispatcher));
        dg::networK_raii_producer_consumer::LambdaWrappedConsumer<VirtualPayLoad, decltype(orphan_immu_dispatcher)> orphan_immu_consumer(std::move(orphan_immu_dispatcher));        
        stdx::seq_cst_guard memcst_guard;

        auto err_unwind_guard =  stdx::resource_guard([=]() noexcept{
            for (size_t i = 0u; i < sz; ++i){
                exception_arr[i] = dg::network_exception::BAD_OPERATION;
            }
        });

        for (size_t i = 0u; i < sz; ++i){
            auto payload_kind = payload_arr[i];

            switch (payload_kind){
                case payload_kind_init_leaf:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_leaf_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_init_mono:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_mono_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_init_pair:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_pair_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_init_uacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_uacm_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_init_pacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_pacm_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_init_crit:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_crit_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_init_msgrfwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_msgrfwd_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_init_msgrbwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_msgrbwd_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_init_srcextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_srcextclone_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_init_dstextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_dstextclone_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_init_immu:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(init_immu_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_leaf:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_leaf_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_mono:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_mono_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_pair:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_pair_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_uacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_uacm_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_pacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_pacm_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_crit:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_crit_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_msgrfwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_msgrfwd_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_msgrbwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_msgrbwd_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_srcextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_srcextclone_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_dstextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_dstextclone_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan_immu:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_immu_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_orphan:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(orphan_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_leaf:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_leaf_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_mono:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_mono_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_pair:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_oair_delivery_hanlde)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_uacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_uacm_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_pacm:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_pacm_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_crit:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_crit_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_msgrfwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_msgrfwd_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_msgrbwd:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_msgrbwd_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_srcextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_srcextclone_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_dstextclone:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_dstextclone_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit_immu:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_immu_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
                    break;
                case payload_kind_deinit:
                    dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(deinit_delivery_handle)->get(), std::make_pair(std::move(payload_arr[i]), i));
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
        err_unwind_guard.release();
    }
}

#endif