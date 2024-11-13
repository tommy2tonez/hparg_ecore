#ifndef __DG_NETWORK_TILE_INITIALIZATION_H__
#define __DG_NETWORK_TILE_INITIALIZATION_H__

#include <stdint.h>
#include <stddef.h>
#include "network_exception.h" 
#include "network_exception_handler.h"
#include "network_uma.h"
#include "network_tile_member_getsetter.h"
#include "network_memops_uma.h"
#include "network_tile_member_access.h"
#include "stdx.h"

namespace dg::network_tile_initialization::statix{
    
    //idk why people see concurrency as a hard thing to understand
    //it's not - make sure that the compile control flows reach the fences in and out if you are in a concurrent transaction
    //the function is not responsible for its arguments - but is responsible for concurrent_transaction_block seq_cst property - if the concurrent blk is induced by the function 
    //if you assume that the function is not for concurrent usage - and there is no concurrent transaction induced by the function - see tile_member_get_setter - then it is not responsible for such
    //that's it, it's that simple - make sure that the transaction is ((always_inline)), otherwise you are very very F
    //usually that must be the lock_guard responsibility - always - but it's too tedious to change the thing now - let's leave that to the compiler
    //the most important thing is to always use seq_cst - even if you have a perfect understanding of the thing - yeah - because seq_cst MUST emit a cpu fence to flush the buffer where other instructions do not
    //don't try to be smart in C++, it's a broken language, driven by proven-to-be-working methods

    void init_leaf(uma_ptr_t ptr, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;

        ptr                 = dg::network_tile_member_access::throwsafe_leaf_ptr_access(ptr);
        uma_ptr_t rcu_addr  = get_leaf_rcu_addr_nothrow(ptr);
        auto lck_grd        = dg::network_memops_uma::memlock_guard(rcu_addr);
        stdx::memtransaction_guard transaction_grd(); //fences all the following in and out | reachable by compiler

        set_leaf_init_status_nothrow(ptr, DEFAULT_INIT_STATUS);
        set_leaf_observer_nothrow(ptr, DEFAULT_OBSERVER);
        set_leaf_operatable_id_nothrow(ptr, operatable_id);
        set_leaf_pong_count_nothrow(ptr, DEFAULT_PONG_COUNT);
    }

    void init_mono(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;

        ptr                 = dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr);
        uma_ptr_t rcu_addr  = get_mono_rcu_addr_nothrow(ptr);
        auto lck_grd        = dg::network_memops_uma::memlock_guard(rcu_addr);
        stdx::memtransaction_guard transaction_grd();

        set_mono_init_status_nothrow(ptr, DEFAULT_INIT_STATUS);
        set_mono_observer_nothrow(ptr, DEFAULT_OBSERVER);
        set_mono_dispatch_control_nothrow(ptr, dispatch_control_id);
        set_mono_operatable_id_nothrow(ptr, operatable_id);
        set_mono_pong_count_nothrow(ptr, DEFAULT_PONG_COUNT);
        set_mono_descendant_nothrow(ptr, src);
    }

    void init_pair(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;

        ptr                 = dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr);
        uma_ptr_t rcu_addr  = get_pair_rcu_addr_nothrow(ptr);
        auto lck_grd        = dg::network_memops_uma::memlock_guard(rcu_addr);
        stdx::memtransaction_guard transaction_grd();

        set_pair_init_status_nothrow(ptr, DEFAULT_INIT_STATUS);
        set_pair_observer_nothrow(ptr, DEFAULT_OBSERVER);
        set_pair_operatable_id_nothrow(ptr, operatable_id);
        set_pair_dispatch_control_nothrow(ptr, dispatch_control_id);
        set_pair_pong_count_nothrow(ptr, DEFAULT_PONG_COUNT);
        set_pair_left_descendant_nothrow(ptr, DEFAULT_DESCENDANT);
        set_pair_right_descendant_nothrow(ptr, DEFAULT_DESCENDANT);
    }

    void init_uacm(uma_ptr_t ptr, std::array<uma_ptr_t, UACM_ACM_SZ> src, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;

        ptr                 = dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr);
        uma_ptr_t rcu_addr  = get_uacm_rcu_addr_nothrow(ptr);
        auto lck_grd        = dg::network_memops_uma::memlock_guard(rcu_addr);
        stdx::memtransaction_guard transaction_grd();

        set_uacm_init_status_nothrow(ptr, DEFAULT_INIT_STATUS);
        set_uacm_observer_nothrow(ptr, DEFAULT_OBSERVER);
        set_uacm_operatable_id_nothrow(ptr, operatable_id);
        set_uacm_dispatch_control_nothrow(ptr, dispatch_control_id);
        set_uacm_pong_count_nothrow(ptr, DEFAULT_PONG_COUNT);
        set_uacm_descendant_nothrow(ptr, src);
    }

    void init_pacm(uma_ptr_t ptr, std::array<uma_ptr_t, PACM_ACM_SZ> lhs, std::array<uma_ptr_t, PACM_ACM_SZ> rhs, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;

        ptr                 = dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr);
        uma_ptr_t rcu_addr  = get_pacm_rcu_addr_nothrow(ptr);
        auto lck_grd        = dg::network_memops_uma::memlock_guard(rcu_addr);
        stdx::memtransaction_guard transaction_grd();
        
        set_pacm_init_status_nothrow(ptr, DEFAULT_INIT_STATUS);
        set_pacm_observer_nothrow(ptr, DEFAULT_OBSERVER);
        set_pacm_operatable_id_nothrow(ptr, operatable_id);
        set_pacm_dispatch_control_nothrow(ptr, dispatch_control_id);
        set_pacm_pong_count_nothrow(ptr, DEFAULT_PACM_PONG_COUNT);
        set_pacm_left_descendant_nothrow(ptr, lhs);
        set_pacm_right_descendant_nothrow(ptr, rhs);
    }

    void init_crit(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id, crit_kind_t crit_kind){

        using namespace network_tile_member_getsetter;

        ptr                 = dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr);
        uma_ptr_t rcu_addr  = get_crit_rcu_addr_nothrow(ptr);
        auto lck_grd        = dg::network_memops_uma::memlock_guard(rcu_addr);
        stdx::memtransaction_guard transaction_grd();
        
        set_crit_init_status_nothrow(ptr, DEFAULT_INIT_STATUS);
        set_crit_observer_nothrow(ptr, DEFAULT_OBSERVER);
        set_crit_operatable_id_nothrow(ptr, operatable_id);
        set_crit_dispatch_control_nothrow(ptr, dispatch_control_id);
        set_crit_pong_count_nothrow(ptr, DEFAULT_CRIT_PONG_COUNT);
        set_crit_descendant_nothrow(ptr, src);
        set_crit_kind_nothrow(ptr, crit_kind);
    }

    void init_msgrfwd(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id, dst_info_t dst_info){

        using namespace network_tile_member_getsetter;

        ptr                 = dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr);
        uma_ptr_t rcu_addr  = get_msgrfwd_rcu_addr_nothrow(ptr);
        auto lck_grd        = dg::network_memops_uma::memlock_guard(rcu_addr);
        stdx::memtransaction_guard transaction_grd();
        
        set_msgrfwd_init_status_nothrow(ptr, DEFAULT_INIT_STATUS);
        set_msgrfwd_observer_nothrow(ptr, DEFAULT_OBSERVER);
        set_msgrfwd_operatable_id_nothrow(ptr, operatable_id);
        set_msgrfwd_dispatch_control_nothrow(ptr, dispatch_control_id);
        set_msgrfwd_pong_count_nothrow(ptr, DEFAULT_MSGRFWD_PONG_COUNT);
        set_msgrfwd_descendant_nothrow(ptr, src);
        set_msgrfwd_dst_info_nothrow(ptr, dst_info);
    }

    void init_msgrbwd(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id, timein_t timein, dst_info_t dst_info){

        using namespace network_tile_member_getsetter;

        ptr                 = dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr);
        uma_ptr_t rcu_addr  = get_msgrbwd_rcu_addr(ptr);
        auto lck_grd        = dg::network_memops_uma::memlock_guard(rcu_addr);
        stdx::memtransaction_guard transaction_grd();

        set_msgrbwd_init_status_nothrow(ptr, DEFAULT_INIT_STATUS);
        set_msgrbwd_observer_nothrow(ptr, DEFAULT_OBSERVER);
        set_msgrbwd_operatable_id_nothrow(ptr, operatable_id);
        set_msgrbwd_dispatch_control_nothrow(ptr, dispatch_control_id);
        set_msgrbwd_pong_count_nothrow(ptr, DEFAULT_MSGRBWD_PONG_COUNT);
        set_msgrbwd_descendant_nothrow(ptr, src);
        set_msgrbwd_dst_info_nothrow(ptr, dst_info);
        set_msgrbwd_timein_nothrow(ptr, timein);
    }
}

namespace dg::network_tile_initialization::poly{
    
    struct LeafPayLoad{
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

    struct MonoPayLoad{
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

    struct PairPayLoad{
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

    struct UACMPayLoad{
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

    struct PACMPayLoad{
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

    struct CritPayLoad{
        uma_ptr_t ptr;
        uma_ptr_t src;
        dispatch_control_t dispatch_control;
        operatable_id_t operatable_id;
        crit_kind_t crit_kind;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, crit_kind);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ptr, src, dispatch_control, operatable_id, crit_kind);
        }
    };

    struct MsgrFwdPayLoad{
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

    struct MsgrBwdPayLoad{
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

    auto make_leaf_payload(uma_ptr_t ptr, operatable_id_t id) noexcept -> LeafPayLoad{

        return LeafPayLoad{ptr, id};
    }

    auto load_leaf_payload(LeafPayLoad payload) noexcept -> exception_t{
        
        try{
            dg::network_tile_initialization::statix::init_leaf(payload.ptr, payload.operatable_id);
            return dg::network_excepion::SUCCESS;
        } catch (,,,){
            return dg::network_exception::wrap_std_exception(std::current_exception());
        }
    } 

    auto make_mono_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> MonoPayLoad{

        return MonoPayLoad{ptr, src, dispatch_control, operatable_id};
    }

    auto load_mono_payload(MonoPayLoad payload) noexcept -> exception_t{

        try{
            dg::network_tile_initialization::statix::init_mono(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id);
            return dg::network_exception::SUCCESS;
        } catch (...){
            return dg::network_exception::wrap_std_exception(std::current_exception());
        }
    } 

    auto make_pair_payload(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> PairPayLoad{

        return PairPayLoad{ptr, lhs, rhs, dispatch_control, operatable_id};        
    }

    auto load_pair_payload(PairPayLoad payload) noexcept -> exception_t{

        try{
            dg::network_tile_initialization::statix::init_mono(payload.ptr, payload.lhs, payload.rhs, payload.dispatch_control, payload.operatable_id);
            return dg::network_exception::SUCCESS;
        } catch (...){
            return dg::network_exception::wrap_std_exception(std::current_exception());
        }
    }

    auto make_uacm_payload(uma_ptr_t ptr, std::array<uma_ptr_t, UACM_ACM_SZ> src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> UACMPayLoad{

        return UACMPayLoad{ptr, src, dispatch_control, operatable_id};
    }

    auto load_uacm_payload(UACMPayLoad payload) noexcept -> exception_t{

        try{
            dg::network_tile_initialization::statix::init_uacm(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id);
            return dg::network_exception::SUCCESS;
        } catch (...){
            return dg::network_exception::wrap_std_exception(std::current_exception());
        }
    }

    auto make_pacm_payload(uma_ptr_t ptr, std::array<uma_ptr_t, PACM_ACM_SZ> lhs, std::array<uma_ptr_t, PACM_ACM_SZ> rhs, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> PACMPayLoad{

        return PACMPayLoad{ptr, lhs, rhs, dispatch_control, operatable_id};
    }

    auto load_pacm_payload(PACMPayLoad payload) noexcept -> exception_t{

        try{
            dg::network_tile_initialization::statix::init_pacm(payload.ptr, payload.left_descendant, payload.right_descendant, payload.dispatch_control, payload.operatable_id);
            return dg::network_exception::SUCCESS;
        } catch (...){
            return dg::network_exception::wrap_std_exception(std::current_exception());
        }
    }
    
    auto make_crit_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id) noexcept -> CritPayLoad{

        return CritPayLoad{ptr, src, dispatch_control, operatable_id};
    }

    auto load_crit_payload(CritPayLoad payload) noexcept -> exception_t{

        try{
            dg::network_tile_initialization::statix::init_crit(payload.ptr, payload.src, payload.dispatch_control, payload.operatable_id, payload.crit_kind);
            return dg::network_exception::SUCCESS;
        } catch (...){
            return dg::network_exception::wrap_std_exception(std::current_exception());
        }
    }

    auto make_msgrfwd_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, dst_info_t dst_info) noexcept -> MsgrFwdPayLoad{

        return MsgrFwdPayLoad{ptr, src, dispatch_control, operatable_id, dst_info};
    }

    auto load_msgrfwd_payload(MsgrFwdPayLoad payload) noexcept -> exception_t{

        try{
            dg::network_tile_initialization::statix::init_msgrfwd(ptr, src, dispatch_control, operatable_id, dst_info);
            return dg::network_exception::SUCCESS;
        } catch (...){
            return dg::network_exception::wrap_std_exception(std::current_exception());
        }
    }

    auto make_msgrbwd_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control, operatable_id_t operatable_id, timein_t timein, dst_info_t dst_info) noexcept -> MsgrBwdPayLoad{

        return MsgrBwdPayLoad{ptr, src, dispatch_control, operatable_id, timein, dst_info};
    }

    auto load_msgrbwd_payload(MsgrBwdPayLoad payload) noexcept -> exception_t{

        try{
            dg::network_tile_initialization::statix::init_msgrbwd(ptr, src, dispatch_control, operatable_id, timein, dst_info);
            return dg::network_exception::SUCCESS;
        } catch (...){
            return dg::network_exception::wrap_std_exception(std::current_exception());
        }
    }

    static inline constexpr size_t MAX_PAYLOAD_CONTENT_SZ = size_t{1} << 6; 
    using payload_kind_t = uint8_t;

    enum enum_payload_kind: payload_kind_t{
        payload_kind_leaf       = 0u,
        payload_kind_mono       = 1u,
        payload_kind_pair       = 2u,
        payload_kind_uacm       = 3u,
        payload_kind_pacm       = 4u,
        payload_kind_crit       = 5u,
        payload_kind_msgrfwd    = 6u,
        payload_kind_msgrbwd    = 7u
    };

    struct VirtualPayLoad{
        payload_kind_t kind;
        std::array<char, MAX_PAYLOAD_CONTENT_SZ> payload_content;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(kind, payload_content);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(kind, payload_content);
        }  
    };

    auto virtualize_payload(LeafPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_leaf;

        return rs;
    }

    auto virtualize_payload(MonoPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_mono;

        return rs;
    }

    auto virtualize_payload(PairPayLoad payload) noexcept -> VirtualPayLoad{
        
        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_pair;
        
        return rs;
    }

    auto virtualize_payload(UACMPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);;
        VirtualPayLoad rs{};
        dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_uacm;

        return rs;
    }

    auto virtualize_payload(PACMPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_pacm;

        return rs;
    }
    
    auto virtualize_payload(CritPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_crit;

        return rs;
    }

    auto virtualize_payload(MsgrFwdPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_msgrfwd;

        return rs;
    }

    auto virtualize_payload(MsgrBwdPayLoad payload) noexcept -> VirtualPayLoad{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(payload);
        static_assert(SERIALIZATION_SZ <= MAX_PAYLOAD_CONTENT_SZ);
        VirtualPayLoad rs{};
        dg::network_trivial_serializer::serialize_into(rs.payload_content.data(), payload);
        rs.kind = payload_kind_msgrbwd;

        return rs;
    }

    auto load_virtual_payload(VirtualPayLoad payload) noexcept -> exception_t{

        switch (payload.kind){
            case payload_kind_leaf:
            {
                LeafPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_leaf_payload(devirt_payload);
            }
            case payload_kind_mono:
            {
                MonoPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_mono_payload(devirt_payload);
            }
            case payload_kind_pair:
            {
                PairPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_pair_payload(devirt_payload);
            }
            case payload_kind_uacm:
            {
                UACMPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_uacm_payload(devirt_payload);
            }
            case payload_kind_pacm:
            {
                PACMPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_pacm_payload(devirt_payload);
            }
            case payload_kind_crit:
            {
                CritPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_crit_payload(devirt_payload);
            }
            case payload_kind_msgrfwd:
            {
                MsgrFwdPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_msgrfwd_payload(devirt_payload);
            }
            case payload_kind_msgrbwd:
            {
                MsgrBwdPayLoad devirt_payload{};
                dg::network_trivial_serializer::deserialize_into(devirt_payload, payload.payload_content.data());
                return load_msgrbwd_payload(devirt_payload);
            }
            default:
            {
                return dg::network_exception::INVALID_ARGUMENT;
            }
        }
    }
} 

#endif