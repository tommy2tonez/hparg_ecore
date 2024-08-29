#ifndef __DG_NETWORK_TILE_INITIALIZATION_H__
#define __DG_NETWORK_TILE_INITIALIZATION_H__

#include <stdint.h>
#include <stddef.h>
#include "network_exception.h" 
#include "network_exception_handler.h"
#include "network_uma.h"
#include "network_tile_member_getsetter.h"

//need to decide whether C or C++ approach - exception_t or throw exception()
//C++ approach is cleaner - yet requires extra dyanmic memory allocation for throwing exception - this is a design problem 
//C approach is explicit, legacy
//I would say that this core should be somewhere from 100-200K lines of code 
//takes me probably 3 months if I code days and nights
//

namespace dg::network_tile_initialization_static{
    
    void init_leaf(uma_ptr_t ptr, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;
        using namespace network_tile_member_getsetguard; 

        uma_ptr_t rcu_addr  = get_leaf_rcu_addr(ptr);
        auto lck_grd        = network_uma::memacquire_guard(rcu_addr);
        auto reverter_grd   = dieguard_leaf_reverter(ptr, SET_OPS_LEAF_OPERATABLE_ID | SET_OPS_LEAF_PONG_ADDR);
        
        set_leaf_operatable_id(ptr, operatable_id);
        set_leaf_pong_addr(ptr, DEFAULT_LEAF_PONG_ADDR);

        reverter_grd.release();
    }

    void init_mono(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;
        using namespace network_tile_member_getsetguard; 

        uma_ptr_t rcu_addr  = get_mono_rcu_addr(ptr);
        auto lck_grd        = network_uma::memacquire_guard(rcu_addr);
        auto reverter_grd   = dieguard_mono_reverter(ptr, SET_OPS_MONO_SRC | SET_OPS_MONO_DISPATCH_CONTROL | SET_OPS_MONO_OPERATABLE_ID | SET_OPS_MONO_PONG_COUNT | SET_OPS_MONO_PONG_ADDR);
        
        set_mono_src(ptr, src);
        set_mono_dispatch_control(ptr, dispatch_control_id);
        set_mono_operatable_id(ptr, operatable_id);
        set_mono_pong_count(ptr, DEFAULT_MONO_PONG_COUNT);
        set_mono_pong_addr(ptr, DEFAULT_MONO_PONG_ADDR); 

        reverter_grd.release();
    }

    void init_pair(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;
        using namespace network_tile_member_getsetguard; 

        uma_ptr_t rcu_addr  = get_pair_rcu_addr(ptr);
        auto lck_grd        = network_uma::memacquire_guard(rcu_addr);
        auto reverter_grd   = dieguard_pair_reverter(ptr, SET_OPS_PAIR_LHS | SET_OPS_PAIR_RHS | SET_OPS_PAIR_DISPATCH_CONTROL | SET_OPS_PAIR_OPERATABLE_ID | SET_OPS_PAIR_PONG_COUNT | SET_OPS_PAIR_PONG_ADDR);
        
        set_pair_lhs(ptr, lhs);
        set_pair_rhs(ptr, rhs);
        set_pair_dispatch_control(ptr, dispatch_control_id);
        set_pair_operatable_id(ptr, operatable_id);
        set_pair_pong_count(ptr, DEFAULT_PAIR_PONG_COUNT);
        set_pair_pong_addr(ptr, DEFAULT_PAIR_PONG_ADDR);
        
        reverter_grd.release();
    }

    void init_uacm(uma_ptr_t ptr, std::array<uma_ptr_t, UACM_COUNT> src, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;
        using namespace network_tile_member_getsetguard; 

        uma_ptr_t rcu_addr  = get_uacm_rcu_addr(ptr);
        auto lck_grd        = network_uma::memacquire_guard(rcu_addr);
        auto reverter_grd   = dieguard_uacm_reverter(ptr, SET_OPS_UACM_SRC | SET_OPS_UACM_DISPATCH_CONTROL | SET_OPS_UACM_OPERATABLE_ID | SET_OPS_UACM_PONG_COUNT | SET_OPS_UACM_PONG_ADDR);
        
        set_uacm_src(ptr, src);
        set_uacm_dispatch_control(ptr, dispatch_control_id);
        set_uacm_operatable_id(ptr, operatable_id);
        set_uacm_pong_count(ptr, DEFAULT_UACM_PONG_COUNT);
        set_uacm_pong_addr(ptr, DEFAULT_UACM_PONG_ADDR);
        
        reverter_grd.release();
    }

    void init_pacm(uma_ptr_t ptr, std::array<uma_ptr_t, PACM_COUNT> lhs, std::array<uma_ptr_t, PACM_COUNT> rhs, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;
        using namespace network_tile_member_getsetguard; 

        uma_ptr_t rcu_addr  = get_pacm_rcu_addr(ptr);
        auto lck_grd        = network_uma::memacquire_guard(rcu_addr);
        auto reverter_grd   = dieguard_pacm_reverter(ptr, SET_OPS_PACM_LHS | SET_OPS_PACM_RHS | SET_OPS_PACM_DISPATCH_CONTROL | SET_OPS_PACM_OPERATABLE_ID | SET_OPS_PACM_PONG_COUNT | SET_OPS_PACM_PONG_ADDR);
        
        set_pacm_lhs(ptr, lhs);
        set_pacm_rhs(ptr, rhs);
        set_pacm_dispatch_control(ptr, dispatch_control_id);
        set_pacm_operatable_id(ptr, operatable_id);
        set_pacm_pong_count(ptr, DEFAULT_PACM_PONG_COUNT);
        set_pacm_pong_addr(ptr, DEFAULT_PACM_PONG_ADDR);
        
        reverter_grd.release();
    }

    void init_crit(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id){

        using namespace network_tile_member_getsetter;
        using namespace network_tile_member_getsetguard; 

        uma_ptr_t rcu_addr  = get_crit_rcu_addr(ptr);
        auto lck_grd        = network_uma::memacquire_guard(rcu_addr);
        auto reverter_grd   = dieguard_crit_reverter(ptr, SET_OPS_CRIT_SRC | SET_OPS_CRIT_DISPATCH_CONTROL | SET_OPS_CRIT_OPERATABLE_ID | SET_OPS_CRIT_PONG_COUNT | SET_OPS_CRIT_PONG_ADDR);
        
        set_crit_src(ptr, src);
        set_crit_dispatch_control(ptr, dispatch_control_id);
        set_crit_operatable_id(ptr, operatable_id);
        set_crit_pong_count(ptr, DEFAULT_CRIT_PONG_COUNT);
        set_crit_pong_addr(ptr, DEFAULT_CRIT_PONG_ADDR);
        
        reverter_grd.release();
    }

    void init_msgrfwd(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id, injection_info_t injection_info){

        using namespace network_tile_member_getsetter;
        using namespace network_tile_member_getsetguard; 

        uma_ptr_t rcu_addr  = get_msgrfwd_rcu_addr(ptr);
        auto lck_grd        = network_uma::memacquire_guard(rcu_addr);
        auto reverter_grd   = dieguard_msgrfwd_reverter(ptr, SET_OPS_MSGRFWD_SRC | SET_OPS_MSGRFWD_DISPATCH_CONTROL | SET_OPS_MSGRFWD_INJECTION_INFO | SET_OPS_MSGRFWD_PONG_COUNT | SET_OPS_MSGRFWD_PONG_ADDR);
        
        set_msgrfwd_src(ptr, src);
        set_msgrfwd_dispatch_control(ptr, dispatch_control_id);
        set_msgrfwd_operatable_id(ptr, operatable_id);
        set_msgrfwd_injection_info(ptr, injection_info);
        set_msgrfwd_pong_count(ptr, DEFAULT_MSGRFWD_PONG_COUNT);
        set_msgrfwd_pong_addr(ptr, DEFAULT_MSGRFWD_PONG_ADDR);
        
        reverter_grd.release();
    }

    void init_msgrbwd(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_control_id, operatable_id_t operatable_id, gradient_rc_t least, injection_info_t injection_info){

        using namespace network_tile_member_getsetter;
        using namespace network_tile_member_getsetguard; 

        uma_ptr_t rcu_addr  = get_msgrbwd_rcu_addr(ptr);
        auto lck_grd        = network_uma::memacquire_guard(rcu_addr);
        auto reverter_grd   = dieguard_msgrbwd_reverter(ptr, SET_OPS_MSGRBWD_SRC | SET_OPS_MSGRBWD_DISPATCH_CONTROL | SET_OPS_MSGRBWD_OPERATABLE_ID | SET_OPS_MSGRBWD_GBPC | SET_OPS_MSGRBWD_INJECTION_INFO | SET_OPS_MSGRBWD_PONG_ADDR);
        
        set_msgrbwd_src(ptr, src);
        set_msgrbwd_dispatch_control(ptr, dispatch_control_id);
        set_msgrbwd_operatable_id(ptr, operatable_id);
        set_msgrbwd_gbpc(ptr, least); 
        set_msgrbwd_injection_info(ptr, injection_info);
        set_msgrbwd_pong_count(ptr, DEFAULT_MSGRBWD_PONG_COUNT);
        set_msgrbwd_pong_addr(ptr, DEFAULT_MSGRBWD_PONG_ADDR);
        
        reverter_grd.release();
    }

    //having these methods as placeholder for future extension - require code duplication to actually recover recoverable 

    void init_leaf_nothrow(uma_ptr_t ptr, operatable_id_t id) noexcept{

        dg::network_exception_handler::nothrow_log(init_leaf(ptr, id));
    }

    void init_mono_nothrow(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept{

        dg::network_exception_handler::nothrow_log(init_mono(ptr, src, dispatch_id, operatable_id));
    }

    void init_pair_nothrow(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept{

        dg::network_exception_handler::nothrow_log(init_pair(ptr, lhs, rhs, dispatch_id, operatable_id));
    }

    void init_uacm_nothrow(uma_ptr_t ptr, uma_ptr_t * src, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept{

        dg::network_exception_handler::nothrow_log(init_uacm(ptr, src, dispatch_id, operatable_id));
    }

    void init_pacm_nothrow(uma_ptr_t ptr, uma_ptr_t * lhs, uma_ptr_t * rhs, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept{

        dg::network_exception_handler::nothrow_log(init_pacm(ptr, lhs, rhs, dispatch_id, operatable_id));
    }

    void init_crit_nothrow(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept{

        dg::network_exception_handler::nothrow_log(init_crit(ptr, src, dispatch_id, operatable_id));
    }

    void init_msgrfwd_nothrow(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id, injection_info_t injection_info) noexcept{

        dg::network_exception_handler::nothrow_log(init_msgrfwd(ptr, src, dispatch_id, operatable_id, injection_info));
    }

    void init_msgrbwd_nothrow(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id, gradient_rc_t least_rc, injection_info_t injection_info) noexcept{

        dg::network_exception_handler::nothrow_log(init_msgrbwd(ptr, src, dispatch_id, operatable_id, least_rc, injection_info));
    }
}

namespace dg::network_tile_initialization_poly{

    using virtual_payload_t = size_t;
    using exception_t       = dg::network_exception::exception_t;
    using payload_option_t  = uint8_t;

    static inline constexpr size_t PAYLOAD_SZ = size_t{1} << 6; 
    using payload_t = std::array<char, PAYLOAD_SZ>; 

    auto make_leaf_payload(uma_ptr_t ptr, operatable_id_t id) noexcept -> payload_t{

        constexpr size_t SERI_SZ = dg::compact_serializer::size(std::tuple(ptr, id)); 
        static_assert(SERI_SZ <= PAYLOAD_SZ);
        payload_t rs{};
        dg::compact_serializer::serialize_into(rs.data(), std::tuple(ptr, id));

        return rs;
    }

    auto make_mono_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept -> payload_t{

        constexpr size_t SERI_SZ = dg::compact_serializer::size(std::tuple(ptr, src, dispatch_id, operatable_id));
        static_assert(SERI_SZ <= PAYLOAD_SZ);
        payload_t rs{};
        dg::compact_serializer::serialize_into(rs.data(), std::tuple(ptr, src, dispatch_id, operatable_id));

        return rs;
    }

    auto make_pair_payload(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept -> payload_t{

        constexpr size_t SERI_SZ = dg::compact_serializer::size(std::tuple(ptr, lhs, rhs, dispatch_id, operatable_id));
        static_assert(SERI_SZ <= PAYLOAD_SZ);
        payload_t rs{};
        dg::compact_serializer::serialize_into(rs.data(), std::tuple(ptr, lhs, rhs, dispatch_id, operatable_id));

        return rs;
    }

    auto make_uacm_payload(uma_ptr_t ptr, std::array<uma_ptr_t, UACM_COUNT> src, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept -> payload_t{

        constexpr size_t SERI_SZ = dg::compact_serializer::size(std::tuple(ptr, src, dispatch_id, operatable_id));
        static_assert(SERI_SZ <= PAYLOAD_SZ);
        payload_t rs{};
        dg::compact_serializer::serialize_into(rs.data(), std::tuple(ptr, src, dispatch_id, operatable_id)); 

        return rs;
    }

    auto make_pacm_payload(uma_ptr_t ptr, std::array<uma_ptr_t, PACM_COUNT> lhs, std::array<uma_ptr_t, PACM_COUNT> rhs, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept -> payload_t{

        constexpr size_t SERI_SZ = dg::compact_serializer::size(std::tuple(ptr, lhs, rhs, dispatch_id, operatable_id));
        static_assert(SERI_SZ <= PAYLOAD_SZ);
        payload_t rs{};
        dg::compact_serializer::serialize_into(rs.data(), std::tuple(ptr, lhs, rhs, dispatch_id, operatable_id));

        return rs;
    }

    auto make_crit_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept -> payload_t{

        constexpr size_t SERI_SZ = dg::compact_serializer::size(std::tuple(ptr, src, dispatch_id, operatable_id));
        static_assert(SERI_SZ <= PAYLOAD_SZ);
        payload_t rs{};
        dg::compact_serializer::serialize_into(rs.data(), std::tuple(ptr, src, dispatch_id, operatable_id));

        return rs;
    }

    auto make_msgrfwd_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id, injection_info_t injection_info) noexcept -> payload_t{

        constexpr size_t SERI_SZ = dg::compact_serializer::size(std::tuple(ptr, src, dispatch_id, operatable_id, injection_info));
        static_assert(SERI_SZ <= PAYLOAD_SZ);
        payload_t rs{};
        dg::compact_serializer::serialize_into(rs.data(), std::tuple(ptr, src, dispatch_id, operatable_id, injection_info));

        return rs;
    }

    auto make_msgrbwd_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id, gradient_rc_t least_rc, injection_info_t injection_info) noexcept -> payload_t{

        constexpr size_t SERI_SZ = dg::compact_serializer::size(std::tuple(ptr, src, dispatch_id, operatable_id, least_rc, injection_info));
        static_assert(SERI_SZ <= PAYLOAD_SZ);
        payload_t rs{};
        dg::compact_serializer::serialize_into(rs.data(), std::tuple(ptr, src, dispatch_id, operatable_id, least_rc, injection_info));

        return rs;
    }

    auto read_leaf_payload(payload_t payload) noexcept -> std::tuple<uma_ptr_t, operatable_id_t>{

        auto rs = std::tuple<uma_ptr_t, operatable_id_t>{};
        dg::compact_serializer::deserialize_into(rs, payload.data());

        return rs;
    }

    auto read_mono_payload(payload_t payload) noexcept -> std::tuple<uma_ptr_t, uma_ptr_t, dispatch_control_t, operatable_id_t>{

        auto rs = std::tuple<uma_ptr_t, uma_ptr_t, dispatch_control_t, operatable_id_t>{};
        dg::compact_serializer::deserialize_into(rs, payload.data());

        return rs;
    }

    auto read_pair_payload(payload_t payload) noexcept -> std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t, dispatch_control_t, operatable_id_t>{

        auto rs = std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t, dispatch_control_t, operatable_id_t>{};
        dg::compact_serializer::deserialize_into(rs, payload.data());

        return rs;
    }

    auto read_uacm_payload(payload_t payload) noexcept -> std::tuple<uma_ptr_t, std::array<uma_ptr_t, UACM_COUNT>, dispatch_control_t, operatable_id_t>{

        auto rs = std::tuple<uma_ptr_t, std::array<uma_ptr_t, UACM_COUNT>, dispatch_control_t, operatable_id_t>{};
        dg::compact_serializer::deserialize_into(rs, payload.data());

        return rs;
    }

    auto read_pacm_payload(payload_t payload) noexcept -> std::tuple<uma_ptr_t, std::array<uma_ptr_t, PACM_COUNT>, std::array<uma_ptr_t, PACM_COUNT>, dispatch_control_t, operatable_id_t>{

        auto rs = std::tuple<uma_ptr_t, std::array<uma_ptr_t, PACM_COUNT>, std::array<uma_ptr_t, PACM_COUNT>, dispatch_control_t, operatable_id_t>{};
        dg::compact_serializer::deserialize_into(rs, payload.data());

        return rs;
    }

    auto read_crit_payload(payload_t payload) noexcept -> std::tuple<uma_ptr_t, uma_ptr_t, dispatch_control_t, operatable_id_t>{

        auto rs = std::tuple<uma_ptr_t, uma_ptr_t, dispatch_control_t, operatable_id_t>{};
        dg::compact_serializer::deserialize_into(rs, payload.data());

        return rs;
    }

    auto read_msgrfwd_payload(payload_t payload) noexcept -> std::tuple<uma_ptr_t, uma_ptr_t, dispatch_control_t, operatable_id_t, injection_info_t>{

        auto rs = std::tuple<uma_ptr_t, uma_ptr_t, dispatch_control_t, operatable_id_t, injection_info_t>{};
        dg::compact_serializer::deserialize_into(rs, payload.data());

        return rs;
    }

    auto read_msgrbwd_payload(payload_t payload) noexcept -> std::tuple<uma_ptr_t, uma_ptr_t, dispatch_control_t, operatable_id_t, gradient_rc_t, injection_info_t>{

        auto rs = std::tuple<uma_ptr_t, uma_ptr_t, dispatch_control_t, operatable_id_t, gradient_rc_t, injection_info_t>{};
        dg::compact_serializer::deserialize_into(rs, payload.data());

        return rs;
    }

    enum virtual_payload_option: payload_option_t{
        leaf_id         = 0u,
        mono_id         = 1u,
        pair_id         = 2u,
        uacm_id         = 3u,
        pacm_id         = 4u,
        crit_id         = 5u,
        msgrfwd_id      = 6u,
        msgrbwd_id      = 7u
    };

    struct VirtualPayload{
        payload_option_t opt;
        payload_t content;
    };

    auto make_leaf_virtual_payload(uma_ptr_t ptr, operatable_id_t id) noexcept -> VirtualPayload{

        return {leaf_id, make_leaf_payload(ptr, id)};
    }

    auto make_mono_virtual_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept -> VirtualPayload{

        return {mono_id, make_mono_payload(ptr, src, dispatch_id, operatable_id)};
    }

    auto make_pair_virtual_payload(uma_ptr_t ptr, uma_ptr_t lhs, uma_ptr_t rhs, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept -> VirtualPayload{

        return {pair_id, make_pair_payload(ptr, lhs, rhs, dispatch_id, operatable_id)};
    }

    auto make_uacm_virtual_payload(uma_ptr_t ptr, std::array<uma_ptr_t, UACM_COUNT> src, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept -> VirtualPayload{

        return {uacm_id, make_uacm_payload(ptr, src, dispatch_id, operatable_id)};
    }

    auto make_pacm_virtual_payload(uma_ptr_t ptr, std::array<uma_ptr_t, PACM_COUNT> lhs, std::array<uma_ptr_t, PACM_COUNT> rhs, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept -> VirtualPayload{

        return {pacm_id, make_pacm_payload(ptr, lhs, rhs, dispatch_id, operatable_id)};
    }

    auto make_crit_virtual_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id) noexcept -> VirtualPayload{

        return {crit_id, make_crit_payload(ptr, src, dispatch_id, operatable_id)};
    }

    auto make_msgrfwd_virtual_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id, injection_info_t injection_info) noexcept -> VirtualPayload{

        return {msgrfwd_id, make_msgrfwd_payload(ptr, src, dispatch_id, operatable_id, injection_info)};
    }

    auto make_msgrbwd_virtual_payload(uma_ptr_t ptr, uma_ptr_t src, dispatch_control_t dispatch_id, operatable_id_t operatable_id, gradient_rc_t least_rc, injection_info_t injection_info) noexcept -> VirtualPayload{

        return {msgrbwd_id, make_msgrbwd_payload(ptr, dispatch_id, operatable_id, least_rc, injection_info)};
    }

    auto load(VirtualPayload payload) noexcept -> exception_t{

        switch (payload.opt){
            case leaf_id:
                return dg::functional::tuple_invoke(dg::network_tile_initialization_static::init_leaf, read_leaf_payload(payload.content));
            case mono_id:
                return dg::functional::tuple_invoke(dg::network_tile_initialization_static::init_mono, read_mono_payload(payload.content));
            case pair_id:
                return dg::functional::tuple_invoke(dg::network_tile_initialization_static::init_pair, read_pair_payload(payload.content));
            case uacm_id:
                return dg::functional::tuple_invoke(dg::network_tile_initialization_static::init_uacm, read_uacm_payload(payload.content));
            case crit_id:
                return dg::functional::tuple_invoke(dg;:network_tile_initialization_static::init_crit, read_crit_payload(payload.content));
            case msgrfwd_id:
                return dg::functional::tuple_invoke(dg::network_tile_initialization_static::init_msgrfwd, read_msgrfwd_payload(payload.content));
            case msgrbwd_id:
                return dg::functional::tuple_invoke(dg::network_tile_initialization_static::init_msgrbwd, read_msgrbwd_payload(payload.content));
            default:
                return dg::network_exception::INVALID_TABLE_DISPATCH_CODE;
        }
    }

    void load_nothrow(VirtualPayload payload) noexcept{

        dg::network_exception_handler::nothrow_log(load(payload));
    }
} 

#endif