#ifndef __DG_NETWORK_TILE_MEMBER_GETSETTER_H__
#define __DG_NETWORK_TILE_MEMBER_GETSETTER_H__

#include "network_exception.h"
#include "network_container_unsigned_bitset.h"
#include <memory>
#include "network_exception_handler.h"
#include "network_memops_uma.h"
#include "network_tile_member_access.h"

namespace dg::network_tile_member_getsetter::internal_unsafe{
    
    void set_leaf_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr); 
            void * src      = &operatable_id; 
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_leaf_ptr_access(ptr));
    }

    void set_leaf_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        // dg::network_exception_handler::nothrow_log(set_leaf_operatable_id(ptr, operatable_id));
    }

    void set_leaf_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_leaf_ptr_access(ptr));
    }

    void set_leaf_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        // dg::network_exception_handler::nothrow_log(set_leaf_pong_addr(ptr, pong_addr));
    }

    void set_mono_src(uma_ptr_t ptr, uma_ptr_t src){

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_src_addr(ptr);
            void * ssrc     = &src;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, ssrc, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr));
    }

    void set_mono_src_nothrow(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        // dg::network_exception_handler::nothrow_log(set_mono_src(ptr, src));
    }

    void set_mono_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control_id){

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr));
    }

    void set_mono_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        // dg::network_exception_handler::nothrow_log(set_mono_dispatch_control(ptr, dispatch_control));
    }

    void set_mono_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t)); 
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr));
    }

    void set_mono_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        // dg::network_exception_handler::nothrow_log(set_mono_operatable_id(ptr, operatable_id));
    }

    void set_mono_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr));
    }

    void set_mono_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        // dg::network_exception_handler::nothrow_log(set_mono_pong_count(ptr, pong_count));
    } 

    void set_mono_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr));
    }

    void set_mono_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        // dg::network_exception_handler::nothrow_log(set_mono_pong_addr(ptr, pong_addr));
    }

    void set_pair_lhs(uma_ptr_t ptr, uma_ptr_t lhs){

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_lhs_addr(ptr);
            void * src      = &lhs;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
    }
    
    void set_pair_lhs_nothrow(uma_ptr_t ptr, uma_ptr_t lhs) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pair_lhs(ptr, lhs));
    }

    void set_pair_rhs(uma_ptr_t ptr, uma_ptr_t rhs){

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_rhs_addr(ptr);
            void * src      = &rhs;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
    }

    void set_pair_rhs_nothrow(uma_ptr_t ptr, uma_ptr_t rhs) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pair_rhs(ptr, rhs));
    }

    void set_pair_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control_id){

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); 
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
    }

    void set_pair_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pair_dispatch_control(ptr, dispatch_control));
    }

    void set_pair_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
    }

    void set_pair_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pair_operatable_id(ptr, operatable_id));
    }

    void set_pair_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
    }

    void set_pair_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pair_pong_count(ptr, pong_count));
    }

    void set_pair_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
    }

    void set_pair_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pair_pong_addr(ptr, pong_addr));
    }

    void set_uacm_src(uma_ptr_t ptr, uma_ptr_t src){

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_src_addr(ptr);
            void * ssrc     = &src;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr));
    }

    void set_uacm_src_nothrow(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        // dg::network_exception_handler::nothrow_log(set_uacm_src(ptr, src));
    }

    void set_uacm_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lamda, dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr));
    }

    void set_uacm_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        // dg::network_exception_handler::nothrow_log(set_uacm_dispatch_control(ptr, dispatch_control));
    } 

    void set_uacm_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lamda, dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr));
    }

    void set_uacm_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        // dg::network_exception_handler::nothrow_log(set_uacm_operatable_id(ptr, operatable_id));
    }

    void set_uacm_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr));
    }

    void set_uacm_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        // dg::network_exception_handler::nothrow_log(set_uacm_pong_count(ptr, pong_count));
    }

    void set_uacm_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>)); 
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr));
    }

    void set_uacm_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        // dg::network_exception_handler::nothrow_log(set_uacm_pong_addr(ptr, pong_addr));
    }

    void set_pacm_lhs(uma_ptr_t ptr, uma_ptr_t lhs){

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_lhs_addr(ptr);
            void * src      = &lhs;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
    }

    void set_pacm_lhs_nothrow(uma_ptr_t ptr, uma_ptr_t lhs) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pacm_lhs(ptr, lhs));
    }

    void set_pacm_rhs(uma_ptr_t ptr, uma_ptr_t rhs){

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_rhs_addr(ptr);
            void * src      = &rhs;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
    }

    void set_pacm_rhs_nothrow(uma_ptr_t ptr, uma_ptr_t rhs) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pacm_rhs(ptr, rhs));
    }

    void set_pacm_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control_id){

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); 
        };
        
        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
    }

    void set_pacm_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pacm_dispatch_control(ptr, dispatch_control));
    }

    void set_pacm_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id; 
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
    }

    void set_pacm_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pacm_operatable_id(ptr, operatable_id));
    }

    void set_pacm_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t)); 
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
    }

    void set_pacm_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        // dg::network_exception_handler::nothrow_log(set_pacm_pong_count(ptr, pong_count));
    }

    void set_pacm_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>)); 
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
    }

    void set_pacm_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{ 

        // dg::network_exception_handler::nothrow_log(set_pacm_pong_addr(ptr, pong_addr));
    } 

    void set_crit_src(uma_ptr_t ptr, uma_ptr_t src){

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_src_addr(ptr);
            void * ssrc     = &src;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr));
    }

    void set_crit_src_nothrow(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        // dg::network_exception_handler::nothrow_log(set_crit_src(ptr, src));
    }

    void set_crit_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control_id){

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr));
    }

    void set_crit_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        // dg::network_exception_handler::nothrow_log(set_crit_dispatch_control(ptr, dispatch_control));
    }

    void set_crit_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t)); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr));
    }

    void set_crit_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        // dg::network_exception_handler::nothrow_log(set_crit_operatable_id(ptr, operatable_id));
    }

    void set_crit_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t)); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr));
    }

    void set_crit_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        // dg::network_exception_handler::nothrow_log(set_crit_pong_count(ptr, pong_count));
    } 

    void set_crit_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>)); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr));
    }

    void set_crit_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        // dg::network_exception_handler::nothrow_log(set_crit_pong_addr(ptr, pong_addr));
    }

    void set_msgrfwd_src(uma_ptr_t ptr, uma_ptr_t src){

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_src_addr(ptr);
            void * ssrc     = &src;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
    }

    void set_msgrfwd_src_nothrow(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrfwd_src(ptr, src));
    } 

    void set_msgrfwd_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control_id){

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
    }

    void set_msgrfwd_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrfwd_dispatch_control(ptr, dispatch_control));
    } 

    void set_msgrfwd_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t)); 
        };     

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
    }

    void set_msgrfwd_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrfwd_operatable_id(ptr, operatable_id));
    }

    void set_msgrfwd_injection_info(uma_ptr_t ptr, injection_info_t injection_info){

        static_assert(std::is_trivially_copyable_v<injection_info_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_injection_info_addr(ptr);
            void * src      = &injection_info;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(injection_info_t)); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr)); 
    }

    void set_msgrfwd_injection_info_nothrow(uma_ptr_t ptr, injection_info_t injection_info) noexcept{

        // dg::network_exception_hanlder::nothrow_log(set_msgrfwd_injection_info(ptr, injection_info));
    }

    void set_msgrfwd_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
    }

    void set_msgrfwd_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrfwd_pong_count(ptr, pong_count));
    }

    void set_msgrfwd_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>)); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
    }

    void set_msgrfwd_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrfwd_pong_addr(ptr, pong_addr));
    }

    void set_msgrbwd_src(uma_ptr_t ptr, uma_ptr_t src){

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_src_addr(ptr);
            void * ssrc     = &src;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, ssrc, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
    }

    void set_msgrbwd_src_nothrow(uma_ptr_t ptr, uma_ptr_t src) noexcept{
        
        // dg::network_exception_handler::nothrow_log(set_msgrbwd_src(ptr, src));
    }

    void set_msgrbwd_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control_id){

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
    }

    void set_msgrbwd_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrbwd_dispatch_control(ptr, dispatch_control));
    } 

    void set_msgrbwd_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
    }

    void set_msgrbwd_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrbwd_operatable_id(ptr, operatable_id));
    }

    void set_msgrbwd_gbpc(uma_ptr_t ptr, gradient_count_t least){ //gradient backprop count

        static_assert(std::is_trivially_copyable_v<gradient_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_gbpc_addr(ptr);
            void * src      = &least;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(gradient_count_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
    }

    void set_msgrbwd_gbpc_nothrow(uma_ptr_t ptr, gradient_count_t least) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrbwd_gbpc(ptr, least));
    }

    void set_msgrbwd_injection_info(uma_ptr_t ptr, injection_info_t injection_info){

        static_assert(std::is_trivially_copyable_v<injection_info_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_injection_info_addr(ptr);
            void * src      = &injection_info;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(injection_info_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
    }

    void set_msgrbwd_injection_info_nothrow(uma_ptr_t ptr, injection_info_t injection_info) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrbwd_injection_info(ptr, injection_info));
    }

    void set_msgrbwd_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
    }

    void set_msgrbwd_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrbwd_pong_count(ptr, pong_count));
    }

    void set_msgrbwd_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr)); //equivalent to dynamic_cast<Interface *>(void *)->set_member(...);
    }

    void set_msgrbwd_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        // dg::network_exception_handler::nothrow_log(set_msgrbwd_pong_addr(ptr, pong_addr));
    }

    void set_tile_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        using namespace dg::network_tile_member_access;
        auto id = poly_typeid(throwsafe_tile_ptr_access(ptr));

        if (is_leaf_tile(id)){
            set_leaf_operatable_id_nothrow(ptr, operatable_id);
            return;
        }

        if (is_mono_tile(id)){
            set_mono_operatable_id_nothrow(ptr, operatable_id);
            return;
        }

        if (is_pair_tile(id)){
            set_pair_operatable_id_nothrow(ptr, operatable_id);
            return;
        }

        if (is_uacm_tile(id)){
            set_uacm_operatable_id_nothrow(ptr, operatable_id);
            return;
        }

        if (is_pacm_tile(id)){
            set_pacm_operatable_id_nothrow(ptr, operatable_id);
            return;
        }

        if (is_crit_tile(id)){
            set_crit_operatable_id_nothrow(ptr, operatable_id);
            return;
        }

        if (is_msgrfwd_tile(id)){
            set_msgrfwd_operatable_id_nothrow(ptr, operatable_id);
            return;
        }

        if (is_msgrbwd_tile(id)){
            set_msgrbwd_operatable_id_nothrow(ptr, operatable_id);
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    void set_tile_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    void set_tile_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        using namespace dg::network_tile_member_access;
        auto id = poly_typeid(throwsafe_tile_ptr_access(ptr));

        if (is_leaf_tile(id)){
            dg::network_exception::throw_exception(dg::network_exception::BAD_TILE_MEMBER_ACCESS);
            return;
        }

        if (is_mono_tile(id)){
            set_mono_dispatch_control_nothrow(ptr, dispatch_control);
            return;
        }

        if (is_pair_tile(id)){
            set_pair_dispatch_control_nothrow(ptr, dispatch_control);
            return;
        }

        if (is_uacm_tile(id)){
            set_uacm_dispatch_control_nothrow(ptr, dispatch_control);
            return;
        }

        if (is_pacm_tile(id)){
            set_pacm_dispatch_control_nothrow(ptr, dispatch_control);
            return;
        }

        if (is_crit_tile(id)){
            set_crit_dispatch_control_nothrow(ptr, dispatch_control);
            return;
        }

        if (is_msgrfwd_tile(id)){
            set_msgrfwd_dispatch_control_nothrow(ptr, dispatch_control);
            return;
        }

        if (is_msgrbwd_tile(id)){
            set_msgrbwd_dispatch_control_nothrow(ptr, dispatch_control);
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    void set_tile_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

    }

    void set_tile_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        using namespace dg::network_tile_member_access; 
        auto id = dg_typeid(throwsafe_tile_ptr_access(ptr));

        if (is_leaf_tile(id)){
            dg::network_exception::throw_exception(dg::network_exception::BAD_TILE_MEMBER_ACCESS);
            return;
        }

        if (is_mono_tile(id)){
            set_mono_pong_count_nothrow(ptr, pong_count);
            return;
        }

        if (is_pair_tile(id)){
            set_pair_pong_count_nothrow(ptr, pong_count);
            return;
        }

        if (is_uacm_tile(id)){
            set_uacm_pong_count_nothrow(ptr, pong_count);
            return;
        }

        if (is_pacm_tile(id)){
            set_pacm_pong_count_nothrow(ptr, pong_count);
            return;
        }

        if (is_crit_tile(id)){
            set_crit_pong_count_nothrow(ptr, pong_count);
            return;
        }

        if (is_msgrfwd_tile(id)){
            set_msgrfwd_pong_count_nothrow(ptr, pong_count);
            return;
        }

        if (is_msgrbwd_tile(id)){
            set_msgrbwd_pong_count_nothrow(ptr, pong_count);
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }
    
    void set_tile_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

    } 

    void set_tile_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        using namespace dg::network_tile_member_access;
        auto id = dg_typeid(throwsafe_tile_ptr_access(ptr));

        if (is_leaf_tile(id){
            set_leaf_pong_addr_nothrow(ptr, pong_addr);
            return;
        })

        if (is_mono_tile(id)){
            set_mono_pong_addr_nothrow(ptr, pong_addr);
            return;
        }

        if (is_pair_tile(id)){
            set_pair_pong_addr_nothrow(ptr, pong_addr);
            return;
        }

        if (is_uacm_tile(id)){
            set_uacm_pong_addr_nothrow(ptr, pong_addr);
            return;
        }

        if (is_pacm_tile(id)){
            set_pacm_pong_addr_nothrow(ptr, pong_addr);
            return;
        }

        if (is_crit_tile(id)){
            set_crit_pong_addr_nothrow(ptr, pong_addr);
            return;
        }

        if (is_msgrfwd_tile(id)){
            set_msgrfwd_pong_addr_nothrow(ptr, pong_addr);
            return;
        }

        if (is_msgrbwd_tile(id)){
            set_msgrbwd_pong_addr_nothrow(ptr, pong_addr);
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    void set_tile_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

    }

    auto get_leaf_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs; 
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_leaf_ptr_access(ptr));
        return rs;
    }

    auto get_leaf_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

    }

    auto get_leaf_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs; 
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_leaf_ptr_access(ptr));
        return rs;
    }

    auto get_leaf_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

    }

    auto get_mono_src(uma_ptr_t ptr) -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_src_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr));
        return rs;
    }

    auto get_mono_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    }

    auto get_mono_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr));
        return rs;
    }

    auto get_mono_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

    }

    auto get_mono_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr));
        return rs;
    }

    auto get_mono_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

    }

    auto get_mono_pong_count(uma_ptr_t ptr) -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t)); 
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr));
        return rs;
    }

    auto get_mono_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

    } 

    auto get_mono_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_mono_ptr_access(ptr));
        return rs;
    }

    auto get_mono_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

    }

    auto get_pair_lhs(uma_ptr_t ptr) -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_lhs_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lamda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
        return rs;
    }

    auto get_pair_lhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    }

    auto get_pair_rhs(uma_ptr_t ptr) -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_rhs_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
        return rs;
    }

    auto get_pair_rhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    }

    auto get_pair_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
        return rs;
    }

    auto get_pair_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

    }

    auto get_pair_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
        return rs;
    }

    auto get_pair_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

    }

    auto get_pair_pong_count(uma_ptr_t ptr) -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lamda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
        return rs;
    }

    auto get_pair_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

    }

    auto get_pair_pong_addr(uma_ptr_t) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pair_ptr_access(ptr));
        return rs;
    }

    auto get_pair_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

    }

    auto get_uacm_src(uma_ptr_t ptr) -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_src_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lamda, dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr));
        return rs;
    }

    auto get_uacm_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    } 

    auto get_uacm_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr));
        return rs;
    }

    auto get_uacm_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

    }

    auto get_uacm_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lamda, dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr));
        return rs;
    }

    auto get_uacm_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

    }

    auto get_uacm_pong_count(uma_ptr_t ptr) -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lamda   = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr));
        return rs;
    }

    auto get_uacm_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

    }

    auto get_uacm_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_uacm_ptr_access(ptr));
        return rs;
    }

    auto get_uacm_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

    }

    auto get_pacm_lhs(uma_ptr_t ptr) -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_lhs_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
        return rs;
    }

    auto get_pacm_lhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    } 

    auto get_pacm_rhs(uma_ptr_t ptr) -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_rhs_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
        return rs;
    }

    auto get_pacm_rhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    } 

    auto get_pacm_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
        return rs;
    }

    auto get_pacm_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

    }

    auto get_pacm_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
        return rs;
    }

    auto get_pacm_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

    }

    auto get_pacm_pong_count(uma_ptr_t ptr) -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
        return rs;
    }

    auto get_pacm_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

    }

    auto get_pacm_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lamda, dg::network_tile_member_access::throwsafe_pacm_ptr_access(ptr));
        return rs;
    }

    auto get_pacm_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

    }

    auto get_crit_src(uma_ptr_t ptr) -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_src_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr));
        return rs;
    }

    auto get_crit_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    }

    auto get_crit_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr));
        return rs;
    }

    auto get_crit_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

    } 

    auto get_crit_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr));
        return rs;
    }

    auto get_crit_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

    } 

    auto get_crit_pong_count(uma_ptr_t ptr) -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr));
        return rs;
    }

    auto get_crit_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

    }

    auto get_crit_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_crit_ptr_access(ptr));
        return rs;
    }

    auto get_crit_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        // return dg::network_exception_handler::nothrow_log(get_crit_pong_addr(ptr));
    }

    auto get_msgrfwd_src(uma_ptr_t ptr) -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_src_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrfwd_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        // return dg::network_exception_handler::nothrow_log(get_msgrfwd_src(ptr));
    }

    auto get_msgrfwd_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrfwd_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

    }

    auto get_msgrfwd_operatable_id(uma_ptr_t) -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };     

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrfwd_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        // return dg::network_exception_handler::nothrow_log(get_msgrfwd_operatable_id(ptr));
    }

    auto get_msgrfwd_injection_info(uma_ptr_t ptr) -> injection_info_t{

        static_assert(std::is_trivially_copyable_v<injection_info_t>);
        auto rs         = injection_info_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_injection_info_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(injection_info_t));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrfwd_injection_info_nothrow(uma_ptr_t ptr) noexcept -> injection_info_t{

        // return dg::network_exception_handler::nothrow_log(get_msgrfwd_injection_info(ptr));
    }

    auto get_msgrfwd_pong_count(uma_ptr_t ptr) -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrfwd_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        // return dg::network_exception_handler::nothrow_log(get_msgrfwd_pong_count(ptr));
    } 

    auto get_msgrfwd_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrfwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrfwd_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        // return dg::network_exception_handler::nothrow_log(get_msgrfwd_pong_addr(ptr));
    }

    auto get_msgrbwd_src(uma_ptr_t) -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_src_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrbwd_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        // return dg::network_exception_handler::nothrow_log(get_msgrbwd_src(ptr));
    }

    auto get_msgrbwd_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrbwd_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        // return dg::network_exception_handler::nothrow_log(get_msgrbwd_dispatch_control(ptr));
    }

    auto get_msgrbwd_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        //I won't copy paste my code - its bad practice - I want to make sure that practice is actually being practiced

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrbwd_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        // return dg::network_exception_handler::nothrow_log(get_msgrbwd_operatable_id(ptr));
    }

    auto get_msgrbwd_gbpc(uma_ptr_t ptr) -> gradient_count_t{

        static_assert(std::is_trivially_copyable_v<gradient_count_t>);
        auto rs         = gradient_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_gbpc_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(gradient_count_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrbwd_gbpc_nothrow(uma_ptr_t ptr) noexcept -> gradient_count_t{

        // return dg::network_exception_handler::nothrow_log(get_msgrbwd_gbpc(ptr));
    }

    auto get_msgrbwd_injection_info(uma_ptr_t ptr) -> injection_info_t{

        static_assert(std::is_trivially_copyable_v<injection_info_t>);
        auto rs         = injection_info_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_injection_info_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(injection_info_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrbwd_injection_info_nothrow(uma_ptr_t ptr) noexcept -> injection_info_t{

        // return dg::network_exception_handler::nothrow_log(get_msgrbwd_injection_info(ptr));
    }

    auto get_msgrbwd_pong_count(uma_ptr_t ptr) -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrbwd_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        // return dg::network_exception_handler::nothrow_log(get_msgrbwd_pong_count(ptr));
    }

    auto get_msgrbwd_pong_addr(uma_ptr_t) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::throwsafe_msgrbwd_ptr_access(ptr));
        return rs;
    }

    auto get_msgrbwd_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        // return dg::network_exception_handler::nothrow_log(get_msgrbwd_pong_addr(ptr));
    }

    auto get_tile_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        using namespace dg::network_tile_member_access;
        auto id = dg_typeid(throwsafe_tile_ptr_access(ptr));

        if (is_leaf_tile(id)){
            return get_leaf_operatable_id_nothrow(ptr);
        } 

        if (is_mono_tile(id)){
            return get_mono_operatable_id_nothrow(ptr);
        }

        if (is_pair_tile(id)){
            return get_pair_operatable_id_nothrow(ptr);
        }

        if (is_uacm_tile(id)){
            return get_uacm_operatable_id_nothrow(ptr);
        }

        if (is_pacm_tile(id)){
            return get_pacm_operatable_id_nothrow(ptr);
        }

        if (is_crit_tile(id)){
            return get_crit_operatable_id_nothrow(ptr);
        }

        if (is_msgrfwd_tile(id)){
            return get_msgrfwd_operatable_id_nothrow(ptr);
        }

        if (is_msgrbwd_tile(id)){
            return get_msgrbwd_operatable_id_nothrow(ptr);
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    auto get_tile_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

    }

    auto get_tile_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        using namespace dg::network_tile_member_access;
        auto id = dg_typeid(throwsafe_tile_ptr_access(ptr));
        
        if (is_leaf_tile(id)){
            return get_leaf_dispatch_control_nothrow(ptr);
        }

        if (is_mono_tile(id)){
            return get_mono_dispatch_control_nothrow(ptr);
        }

        if (is_pair_tile(id)){
            return get_pair_dispatch_control_nothrow(ptr);
        }

        if (is_uacm_tile(id)){
            return get_uacm_dispatch_control_nothrow(ptr);
        }

        if (is_pacm_tile(id)){
            return get_pacm_dispatch_control_nothrow(ptr);
        }

        if (is_crit_tile(id)){
            return get_crit_dispatch_control_nothrow(ptr);
        }

        if (is_msgrfwd_tile(id)){
            return get_msgrfwd_dispatch_control_nothrow(ptr);
        }

        if (is_msgrbwd_tile(id)){
            return get_msgrbwd_dispatch_control_nothrow(ptr);
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    auto get_tile_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

    }

    auto get_tile_pong_count(uma_ptr_t ptr) -> pong_count_t{

        using namespace dg::network_tile_member_access;
        auto id = dg_typeid(throwsafe_tile_ptr_access(ptr));

        if (is_leaf_tile(id)){
            return get_leaf_pong_count_nothrow(ptr);
        }

        if (is_mono_tile(id)){
            return get_mono_pong_count_nothrow(ptr);
        }

        if (is_pair_tile(id)){
            return get_pair_pong_count_nothrow(ptr); 
        }

        if (is_uacm_tile(id)){
            return get_uacm_pong_count_nothrow(ptr);
        }

        if (is_pacm_tile(id)){
            return get_pacm_pong_count_nothrow(ptr);
        }

        if (is_crit_tile(id)){
            return get_crit_pong_count_nothrow(ptr);
        }

        if (is_msgrfwd_tile(id)){
            return get_msgrfwd_pong_count_nothrow(ptr);
        }

        if (is_msgrbwd_tile(id)){
            return get_msgrbwd_pong_count_nothrow(ptr);
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    auto get_tile_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

    }

    auto get_tile_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        using namespace dg::network_tile_member_access;
        auto id = dg_typeid(throwsafe_tile_ptr_access(ptr));

        if (is_leaf_tile(id)){
            return get_leaf_pong_addr_nothrow(ptr);
        }

        if (is_mono_tile(id)){
            return get_mono_pong_addr_nothrow(ptr);
        }

        if (is_pair_tile(id)){
            return get_pair_pong_addr_nothrow(ptr);
        }

        if (is_uacm_tile(id)){
            return get_uacm_pong_addr_nothrow(ptr);
        }

        if (is_pacm_tile(id)){
            return get_pacm_pong_addr_nothrow(ptr);
        }

        if (is_crit_tile(id)){
            return get_crit_pong_addr_nothrow(ptr);
        }

        if (is_msgrfwd_tile(id)){
            return get_msgrfwd_pong_addr_nothrow(ptr);
        }

        if (is_msgrbwd_tile(id)){
            return get_msgrbwd_pong_addr_nothrow(ptr);
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    auto get_pong_addr_nothrow(uma_ptr_t) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

    }
}

namespace dg::network_tile_member_getsetter::iu_guard{

    //not yet know - if this is necessary 
    //this is here for future extension - remove if no longer necessary - a future decision
    //on the one hand, I think that preconds of dependency injection via setter and getter should be the dependencies' responsibility
    //on the other hand, that would introduce many types + remove polymorphic property of those getter + setter
    //an easy fix was to create a shadow type whose existence is only for precond (shadow_type <> static_cast <-> org_type)
    //such type is only used for the setter method arguments, not for the class members

    using uma_ptr_t     = uint64_t; 

    static inline constexpr uint64_t SET_OPS_LEAF_OPERATABLE_ID         = uint64_t{};
    static inline constexpr uint64_t SET_OPS_LEAF_PONG_ADDR             = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MONO_SRC                   = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MONO_DISPATCH_CONTROL      = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MONO_OPERATABLE_ID         = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MONO_PONG_COUNT            = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MONO_PONG_ADDR             = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PAIR_LHS                   = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PAIR_RHS                   = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PAIR_DISPATCH_CONTROL      = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PAIR_OPERATABLE_ID         = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PAIR_PONG_COUNT            = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PAIR_PONG_ADDR             = uint64_t{};
    static inline constexpr uint64_t SET_OPS_UACM_SRC                   = uint64_t{};
    static inline constexpr uint64_t SET_OPS_UACM_DISPATCH_CONTROL      = uint64_t{};
    static inline constexpr uint64_t SET_OPS_UACM_OPERATABLE_ID         = uint64_t{};
    static inline constexpr uint64_t SET_OPS_UACM_PONG_COUNT            = uint64_t{};
    static inline constexpr uint64_t SET_OPS_UACM_PONG_ADDR             = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PACM_LHS                   = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PACM_RHS                   = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PACM_DISPATCH_CONTROL      = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PACM_PONG_COUNT            = uint64_t{};
    static inline constexpr uint64_t SET_OPS_PACM_PONG_ADDR             = uint64_t{};
    static inline constexpr uint64_t SET_OPS_CRIT_SRC                   = uint64_t{};
    static inline constexpr uint64_t SET_OPS_CRIT_DISPATCH_CONTROL      = uint64_t{};
    static inline constexpr uint64_t SET_OPS_CRIT_OPERATABLE_ID         = uint64_t{};
    static inline constexpr uint64_t SET_OPS_CRIT_PONG_COUNT            = uint64_t{};
    static inline constexpr uint64_t SET_OPS_CRIT_PONG_ADDR             = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRFWD_SRC                = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRFWD_DISPATCH_CONTROL   = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRFWD_INJECTION_INFO     = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRFWD_PONG_COUNT         = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRFWD_PONG_ADDR          = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRBWD_SRC                = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRBWD_DISPATCH_CONTROL   = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRBWD_OPERATABLE_ID      = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRBWD_GBPC               = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRBWD_INJECTION_INFO     = uint64_t{};
    static inline constexpr uint64_t SET_OPS_MSGRBWD_PONG_ADDR          = uint64_t{};

    inline auto dieguard_leaf_reverter(uma_ptr_t ptr, uint64_t option){

        static int i    = 0;
        auto resource   = get_leaf_set_resource(ptr, option);
        auto destructor = [=](int *) noexcept{
            size_t cur{};
            uint64_t tmp_option = option;

            while (!dg::network_container::unsigned_bitset::empty(tmp_option)){
                dg::network_container::unsigned_bitset::pop(tmp_option, cur);

                switch (cur){
                    case SET_OPS_LEAF_OPERATABLE_ID:
                        network_tile_member_getsetter::set_leaf_operatable_id_nothrow(ptr, std::get<SET_OPS_LEAF_OPERATABLE_ID>(resource));
                        break;
                    case SET_OPS_LEAF_PONG_ADDR:
                        network_tile_member_getsetter::set_leaf_pong_addr_nothrow(ptr, std::get<SET_OPS_LEAF_PONG_ADDR>(resource));
                        break;
                    default:
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        break;
                }
            }
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    inline auto dieguard_mono_reverter(uma_ptr_t ptr, uint64_t option){

        static int i    = 0;
        auto resource   = get_mono_set_resource(ptr, option);
        auto destructor = [=](int *) noexcept{
            size_t cur{}; 
            uint64_t tmp_option = option; 

            while (!dg::network_container::unsigned_bitset::empty(tmp_option)){
                dg::network_container::unsigned_bitset::pop(tmp_option, cur);

                switch (cur){
                    case SET_OPS_MONO_SRC:
                        network_tile_member_getsetter::set_mono_src_nothrow(ptr, std::get<SET_OPS_MONO_SRC>(resource));
                        break;
                    case SET_OPS_MONO_PONG_COUNT:
                        network_tile_member_getsetter::set_mono_pong_count_nothrow(ptr, std::get<SET_OPS_MONO_PONG_COUNT>(resource));
                        break;
                    case SET_OPS_MONO_PONG_ADDR:
                        network_tile_member_getsetter::set_mono_pong_addr_nothrow(ptr, std::get<SET_OPS_MONO_PONG_ADDR>(resource));
                        break;
                    case SET_OPS_MONO_OPERATABLE_ID:
                        network_tile_member_getsetter::set_mono_operatable_id_nothrow(ptr, std::get<SET_OPS_MONO_OPERATABLE_ID>(resource));
                        break;
                    case SET_OPS_MONO_DISPATCH_CONTROL:
                        network_tile_member_getsetter::set_mono_dispatch_control_nothrow(ptr, std::get<SET_OPS_MONO_DISPATCH_CONTROL>(resource));
                        break;
                    default:
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        break;
                }
            }
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    inline auto dieguard_pair_reverter(uma_ptr_t ptr, uint64_t option){

        static int i    = 0;
        auto resource   = get_pair_set_resource(ptr, option);
        auto destructor = [=](int *) noexcept{
            size_t cur{};
            uint64_t tmp_option = option;

            while (!dg::network_exception::unsigned_bitset::empty(tmp_option)){
                dg::network_container::unsigned_bitset::pop(tmp_option, cur);

                switch (cur){
                    case SET_OPS_PAIR_DISPATCH_CONTROL:
                        network_tile_member_getsetter::set_pair_dispatch_control_nothrow(ptr, std::get<SET_OPS_PAIR_DISPATCH_CONTROL>(resource));
                        break;
                    case SET_OPS_PAIR_LHS:
                        network_tile_member_getsetter::set_pair_lhs_nothrow(ptr, std::get<SET_OPS_PAIR_LHS>(resource));
                        break;
                    case SET_OPS_PAIR_OPERATABLE_ID:
                        network_tile_member_getsetter::set_pair_operatable_id_nothrow(ptr, std::get<SET_OPS_PAIR_OPERATABLE_ID>(resource));
                        break;
                    case SET_OPS_PAIR_PONG_ADDR:
                        network_tile_member_getsetter::set_pair_pong_addr_nothrow(ptr, std::get<SET_OPS_PAIR_PONG_ADDR>(resource));
                        break;
                    case SET_OPS_PAIR_PONG_COUNT:
                        network_tile_member_getsetter::set_pair_pong_count_nothrow(ptr, std::get<SET_OPS_PAIR_PONG_COUNT>(resource));
                        break;
                    case SET_OPS_PAIR_RHS:
                        network_tile_member_getsetter::set_pair_rhs_nothrow(ptr, std::get<SET_OPS_PAIR_RHS>(resource));
                        break;
                    default:
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        break;
                }
            }
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    inline auto dieguard_uacm_reverter(uma_ptr_t ptr, uint64_t option){
        
        static int i    = 0;
        auto resource   = get_uacm_set_resource(ptr, option);
        auto destructor = [=](int *) noexcept{
            size_t cur{};
            uint64_t tmp_option = option;

            while (!dg::network_container::unsigned_bitset::empty(tmp_option)){
                dg::network_container::unsigned_bitset::pop(tmp_option, cur);

                switch (cur){
                    case SET_OPS_UACM_DISPATCH_CONTROL:
                        network_tile_member_getsetter::set_uacm_dispatch_control_nothrow(ptr, std::get<SET_OPS_UACM_DISPATCH_CONTROL>(resource));
                        break;
                    case SET_OPS_UACM_OPERATABLE_ID:
                        network_tile_member_getsetter::set_uacm_operatable_id_nothrow(ptr, std::get<SET_OPS_UACM_OPERATABLE_ID>(resource));
                        break;
                    case SET_OPS_UACM_PONG_ADDR:
                        network_tile_member_getsetter::set_uacm_pong_addr_nothrow(ptr, std::get<SET_OPS_UACM_PONG_ADDR>(resource));
                        break;
                    case SET_OPS_UACM_PONG_COUNT:
                        network_tile_member_getsetter::set_uacm_pong_count_nothrow(ptr, std::get<SET_OPS_UACM_PONG_COUNT>(resource));
                        break;
                    case SET_OPS_UACM_SRC:
                        network_tile_member_getsetter::set_uacm_src_nothrow(ptr, std::get<SET_OPS_UACM_SRC>(resource));
                        break;
                    default:
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        break;
                }
            }
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    inline auto dieguard_pacm_reverter(uma_ptr_t ptr, uint64_t option){

        static int i    = 0;
        auto resource   = get_pacm_set_resource(ptr, option);
        auto destructor = [=](int *) noexcept{
            size_t cur{};
            uint64_t tmp_option = option;

            while (!dg::network_container::unsigned_bitset::empty(tmp_option)){
                dg::network_container::unsigned_bitset::pop(tmp_option, cur);

                switch (cur){
                    case SET_OPS_PACM_DISPATCH_CONTROL:
                        network_tile_member_getsetter::set_pacm_dispatch_control_nothrow(ptr, std::get<SET_OPS_PACM_DISPATCH_CONTROL>(resource));
                        break;
                    case SET_OPS_PACM_LHS:
                        network_tile_member_getsetter::set_pacm_lhs_nothrow(ptr, std::get<SET_OPS_PACM_LHS>(resource));
                        break;
                    case SET_OPS_PACM_PONG_ADDR:
                        network_tile_member_getsetter::set_pacm_pong_addr_nothrow(ptr, std::get<SET_OPS_PACM_PONG_ADDR>(resource));
                        break;
                    case SET_OPS_PACM_PONG_COUNT:
                        network_tile_member_getsetter::set_pacm_pong_count_nothrow(ptr, std::get<SET_OPS_PACM_PONG_COUNT>(resource));
                        break;
                    case SET_OPS_PACM_RHS:
                        network_tile_member_getsetter::set_pacm_rhs(ptr, std::get<SET_OPS_PACM_RHS>(resource));
                        break;
                    default:
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        break;
                }
            }
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    inline auto dieguard_crit_reverter(uma_ptr_t ptr, uint64_t option){

        static int i    = 0;
        auto resource   = get_crit_set_resource(ptr, option);
        auto destructor = [=](int *) noexcept{
            size_t cur{};
            uint64_t tmp_option = option;

            while (!dg::network_container::unsigned_bitset::empty(tmp_option)){
                dg::network_container::unsigned_bitset::pop(tmp_option, cur);

                switch (cur){
                    case SET_OPS_CRIT_DISPATCH_CONTROL:
                        network_tile_member_getsetter::set_crit_dispatch_control_nothrow(ptr, std::get<SET_OPS_CRIT_DISPATCH_CONTROL>(resource));
                        break;
                    case SET_OPS_CRIT_OPERATABLE_ID:
                        network_tile_member_getsetter::set_crit_operatable_id_nothrow(ptr, std::get<SET_OPS_CRIT_OPERATABLE_ID>(resource));
                        break;
                    case SET_OPS_CRIT_PONG_ADDR:
                        network_tile_member_getsetter::set_crit_pong_addr_nothrow(ptr, std::get<SET_OPS_CRIT_PONG_ADDR>(resource));
                        break;
                    case SET_OPS_CRIT_PONG_COUNT:
                        network_tile_member_getsetter::set_crit_pong_count_nothrow(ptr, std::get<SET_OPS_CRIT_PONG_COUNT>(resource));
                        break;
                    case SET_OPS_CRIT_SRC:
                        network_tile_member_getsetter::set_crit_src_nothrow(ptr, std::get<SET_OPS_CRIT_SRC>(resource));
                        break;
                    default:
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        break;
                }
            }
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    inline auto dieguard_msgrfwd_reverter(uma_ptr_t ptr, uint64_t option){

        static int i    = 0;
        auto resource   = get_msgrfwd_set_resource(ptr, option);
        auto destructor = [=](int *) noexcept{
            size_t cur{};
            uint64_t tmp_option = option;

            while (!dg::network_container::unsigned_bitset::empty(tmp_option)){
                dg::network_container::unsigned_bitset::pop(tmp_option, cur);

                switch (cur){
                    case SET_OPS_MSGRFWD_DISPATCH_CONTROL:
                        network_tile_member_getsetter::set_msgrfwd_dispatch_control_nothrow(ptr, std::get<SET_OPS_MSGRFWD_DISPATCH_CONTROL>(resource));
                        break;
                    case SET_OPS_MSGRFWD_INJECTION_INFO:
                        network_tile_member_getsetter::set_msgrfwd_injection_info_nothrow(ptr, std::get<SET_OPS_MSGRFWD_INJECTION_INFO>(resource));
                        break;
                    case SET_OPS_MSGRFWD_PONG_ADDR:
                        network_tile_member_getsetter::set_msgrfwd_pong_addr_nothrow(ptr, std::get<SET_OPS_MSGRFWD_PONG_ADDR>(resource));
                        break;
                    case SET_OPS_MSGRFWD_PONG_COUNT:
                        network_tile_member_getsetter::set_msgrfwd_pong_count_nothrow(ptr, std::get<SET_OPS_MSGRFWD_PONG_COUNT>(resource));
                        break;
                    case SET_OPS_MSGRFWD_SRC:
                        network_tile_member_getsetter::set_msgrfwd_src_nothrow(ptr, std::get<SET_OPS_MSGRFWD_SRC>(resource));
                        break;
                    default:
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        break;
                }
            }
        }

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    inline auto dieguard_msgrbwd_reverter(uma_ptr_t ptr, uint64_t option){

        static int i    = 0;
        auto resource   = get_msgrbwd_set_resource(ptr, option);
        auto destructor = [=](int *) noexcept{
            size_t cur{};
            uint64_t tmp_option = option;

            while (!dg::network_container::unsigned_bitset::empty(tmp_option)){
                dg::network_container::unsigned_bitset::pop(tmp_option, cur);

                switch (cur){
                    case SET_OPS_MSGRBWD_DISPATCH_CONTROL:
                        network_tile_member_getsetter::set_msgrbwd_dispatch_control_nothrow(ptr, std::get<SET_OPS_MSGRBWD_DISPATCH_CONTROL>(resource));
                        break;
                    case SET_OPS_MSGRBWD_GBPC:
                        network_tile_member_getsetter::set_msgrbwd_gbpc_nothrow(ptr, std::get<SET_OPS_MSGRBWD_GBPC>(resource));
                        break;
                    case SET_OPS_MSGRBWD_INJECTION_INFO:
                        network_tile_member_getsetter::set_msgrbwd_injection_info_nothrow(ptr, std::get<SET_OPS_MSGRBWD_INJECTION_INFO>(resource));
                        break;
                    case SET_OPS_MSGRBWD_OPERATABLE_ID:
                        network_tile_member_getsetter::set_msgrbwd_operatable_id_nothrow(ptr, std::get<SET_OPS_MSGRBWD_OPERATABLE_ID>(resource));
                        break;
                    case SET_OPS_MSGRBWD_PONG_ADDR:
                        network_tile_member_getsetter::set_msgrbwd_pong_addr_nothrow(ptr, std::get<SET_OPS_MSGRBWD_PONG_ADDR>(resource));
                        break;
                    case SET_OPS_MSGRBWD_SRC:
                        network_tile_member_getsetter::set_msgrbwd_src_nothrow(ptr, std::get<SET_OPS_MSGRBWD_SRC>(resource));
                        break;
                    default:
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        break;
                }
            }

            return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
        };
    }
}

namespace dg::network_tile_member_getsetter::global_unsafe{

}

namespace dg::network_tile_member_getsetter::gu_guard{
    
}

#endif