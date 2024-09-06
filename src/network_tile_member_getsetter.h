#ifndef __DG_NETWORK_TILE_MEMBER_GETSETTER_H__
#define __DG_NETWORK_TILE_MEMBER_GETSETTER_H__

#include "network_exception.h"
#include "network_container_unsigned_bitset.h"
#include <memory>
#include "network_exception_handler.h"
#include "network_memops_uma.h"
#include "network_tile_member_access.h"
#include "network_tile_tlb.h"

namespace dg::network_tile_member_getsetter{

    void internal_set_leaf_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr); 
            void * src      = &operatable_id; 
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_leaf_operatable_id(uma_ptr_t ptr, operatable_id operatable_id){

        internal_set_leaf_operatable_id(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr), operatable_id);
    } 

    void set_leaf_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        internal_set_leaf_operatable_id(dg::network_tile_member_access::safe_leaf_ptr_access(ptr), operatable_id);
    }

    void internal_set_leaf_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_leaf_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        internal_set_leaf_pong_addr(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr), pong_addr);
    }

    void set_leaf_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        internal_set_leaf_pong_addr(dg::network_tile_member_access::safe_leaf_ptr_access(ptr), pong_addr); 
    }

    void internal_set_mono_src(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_src_addr(ptr);
            void * ssrc     = &src;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, ssrc, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_mono_src(uma_ptr_t ptr, uma_ptr_t src){

        internal_set_mono_src(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), src);
    }

    void set_mono_src_nothrow(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        internal_set_mono_src( dg::network_tile_member_access::safe_mono_ptr_access(ptr), src);
    }

    void internal_set_mono_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_mono_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        internal_set_mono_dispatch_control(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), dispatch_control);
    } 

    void set_mono_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        internal_set_mono_dispatch_control(dg::network_tile_member_access::safe_mono_ptr_access(ptr), dispatch_control);
    }

    void internal_set_mono_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t)); 
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_mono_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        internal_set_mono_operatable_id(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), operatable_id);
    }

    void set_mono_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        internal_set_mono_operatable_id(dg::network_tile_member_access::safe_mono_ptr_access(ptr), operatable_id);
    }

    void internal_set_mono_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_mono_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        internal_set_mono_pong_count(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), pong_count);
    } 

    void set_mono_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        internal_set_mono_pong_count(dg::network_tile_member_access::safe_mono_ptr_access(ptr), pong_count);
    } 

    void internal_set_mono_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_mono_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        internal_set_mono_pong_addr(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), pong_addr);
    }

    void set_mono_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        internal_set_mono_pong_addr(dg::network_tile_member_access::safe_mono_ptr_access(ptr), pong_addr);
    }

    void internal_set_pair_lhs(uma_ptr_t ptr, uma_ptr_t lhs) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_lhs_addr(ptr);
            void * src      = &lhs;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
    }
    
    void set_pair_lhs(uma_ptr_t ptr, uma_ptr_t lhs){

        internal_set_pair_lhs(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), lhs);
    }

    void set_pair_lhs_nothrow(uma_ptr_t ptr, uma_ptr_t lhs) noexcept{

        internal_set_pair_lhs(dg::network_tile_member_access::safe_pair_ptr_access(ptr), lhs);
    }

    void internal_set_pair_rhs(uma_ptr_t ptr, uma_ptr_t rhs) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_rhs_addr(ptr);
            void * src      = &rhs;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_pair_rhs(uma_ptr_t ptr, uma_ptr_t rhs){

        internal_set_pair_rhs(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), rhs);
    } 

    void set_pair_rhs_nothrow(uma_ptr_t ptr, uma_ptr_t rhs) noexcept{

        internal_set_pair_rhs(dg::network_tile_member_access::safe_pair_ptr_access(ptr), rhs);
    }

    void internal_set_pair_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); 
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_pair_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        internal_set_pair_dispatch_control(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), dispatch_control);
    } 

    void set_pair_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        internal_set_pair_dispatch_control(dg::network_tile_member_access::safe_pair_ptr_access(ptr), dispatch_control);
    }

    void internal_set_pair_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_pair_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        internal_set_pair_operatable_id(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), operatable_id);
    } 

    void set_pair_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        internal_set_pair_operatable_id(dg::network_tile_member_access::safe_pair_ptr_access(ptr), operatable_id);
    }

    void internal_set_pair_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_pair_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        internal_set_pair_pong_count(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), pong_count);
    } 

    void set_pair_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        internal_set_pair_pong_count(dg::network_tile_member_access::safe_pair_ptr_access(ptr), pong_count);
    }

    void internal_set_pair_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_pair_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        internal_set_pair_pong_addr(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), pong_addr);
    } 

    void set_pair_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        internal_set_pair_pong_addr(dg::network_tile_member_access::safe_pair_ptr_access(ptr), pong_addr);
    }

    void internal_set_uacm_src(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_src_addr(ptr);
            void * ssrc     = &src;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_uacm_src(uma_ptr_t ptr, uma_ptr_t src){

        internal_set_uacm_src(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), src);
    } 

    void set_uacm_src_nothrow(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        internal_set_uacm_src(dg::network_tile_member_access::safe_uacm_ptr_access(ptr), src);
    }

    void internal_set_uacm_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lamda, ptr);
    }

    void set_uacm_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        internal_set_uacm_dispatch_control(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), dispatch_control);
    } 

    void set_uacm_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        internal_set_uacm_dispatch_control(dg::network_tile_member_access::safe_uacm_ptr_access(ptr), dispatch_control);
    } 

    void internal_set_uacm_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lamda, ptr);
    }

    void set_uacm_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        internal_set_uacm_operatable_id(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), operatable_id);
    } 

    void set_uacm_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        internal_set_uacm_operatable_id(dg::network_tile_member_access::safe_uacm_ptr_access(ptr), operatable_id);
    }

    void internal_set_uacm_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_uacm_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        internal_set_uacm_pong_count(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), pong_count);
    } 

    void set_uacm_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        internal_set_uacm_pong_count(dg::network_tile_member_access::safe_uacm_ptr_access(ptr), pong_count);
    }

    void internal_set_uacm_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>)); 
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_uacm_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        internal_set_uacm_pong_addr(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), pong_addr);
    } 

    void set_uacm_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        internal_set_uacm_pong_addr(dg::network_tile_member_access::safe_uacm_ptr_access(ptr), pong_addr);
    }

    void internal_set_pacm_lhs(uma_ptr_t ptr, uma_ptr_t lhs) noexcept{ //fix

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_lhs_addr(ptr);
            void * src      = &lhs;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_pacm_lhs(uma_ptr_t ptr, uma_ptr_t lhs){ //fix

        internal_set_pacm_lhs(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), lhs);
    } 

    void set_pacm_lhs_nothrow(uma_ptr_t ptr, uma_ptr_t lhs) noexcept{ //fix

        internal_set_pacm_lhs(dg::network_tile_member_access::safe_pacm_ptr_access(ptr), lhs);
    }

    void internal_set_pacm_rhs(uma_ptr_t ptr, uma_ptr_t rhs) noexcept{ //fix

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_rhs_addr(ptr);
            void * src      = &rhs;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_pacm_rhs(uma_ptr_t ptr, uma_ptr_t rhs){ //fix

        internal_set_pacm_rhs(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), rhs);
    } 

    void set_pacm_rhs_nothrow(uma_ptr_t ptr, uma_ptr_t rhs) noexcept{ //fix

        internal_set_pacm_rhs(dg::network_tile_member_access::safe_pacm_ptr_access(ptr), rhs);
    }

    void internal_set_pacm_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); 
        };
        
        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_pacm_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        internal_set_pacm_dispatch_control(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), dispatch_control);
    }

    void set_pacm_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        internal_set_pacm_dispatch_control(dg::network_tile_member_access::safe_pacm_ptr_access(ptr), dispatch_control);
    }

    void internal_set_pacm_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id; 
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
    }
    
    void set_pacm_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        internal_set_pacm_operatable_id(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), operatable_id);
    } 

    void set_pacm_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        internal_set_pacm_operatable_id(dg::network_tile_member_access::safe_pacm_ptr_access(ptr), operatable_id);
    }

    void internal_set_pacm_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t)); 
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_pacm_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        internal_set_pacm_pong_count(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), pong_count); 
    } 

    void set_pacm_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        internal_set_pacm_pong_count(dg::network_tile_member_access::safe_pacm_ptr_access(ptr), pong_count);
    }
    
    void internal_set_pacm_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>)); 
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_pacm_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        internal_set_pacm_pong_addr(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), pong_addr);
    } 

    void set_pacm_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{ 

        internal_set_pacm_pong_addr(dg::network_tile_member_access::safe_pacm_ptr_access(ptr), pong_addr);
    } 

    void internal_set_crit_src(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_src_addr(ptr);
            void * ssrc     = &src;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_crit_src(uma_ptr_t ptr, uma_ptr_t src){

        internal_set_crit_src(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), src);
    } 

    void set_crit_src_nothrow(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        internal_set_crit_src(dg::network_tile_member_access::safe_crit_ptr_access(ptr), src);
    }

    void internal_set_crit_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_crit_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        internal_set_crit_dispatch_control(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), dispatch_control);
    } 

    void set_crit_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        internal_set_crit_dispatch_control(dg::network_tile_member_access::safe_crit_ptr_access(ptr), dispatch_control);
    }

    void internal_set_crit_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t)); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_crit_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        internal_set_crit_operatable_id(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), operatable_id);
    } 

    void set_crit_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        internal_set_crit_operatable_id(dg::network_tile_member_access::safe_crit_ptr_access(ptr), operatable_id);
    }

    void internal_set_crit_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t)); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_crit_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        internal_set_crit_pong_count(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), pong_count);
    }

    void set_crit_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        internal_set_crit_pong_count(dg::network_tile_member_access::safe_crit_ptr_access(ptr), pong_count);
    } 

    void internal_set_crit_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>)); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_crit_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        internal_set_crit_pong_addr(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), pong_addr);
    } 

    void set_crit_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        internal_set_crit_pong_addr(dg::network_tile_member_access::safe_crit_ptr_access(ptr), pong_addr);
    }

    void internal_set_msgrfwd_src(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_src_addr(ptr);
            void * ssrc     = &src;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, src);
    }

    void set_msgrfwd_src(uma_ptr_t ptr, uma_ptr_t src){

        internal_set_msgrfwd_src(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), src);
    } 

    void set_msgrfwd_src_nothrow(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        internal_set_msgrfwd_src(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr), src);
    } 

    void internal_set_msgrfwd_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrfwd_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        internal_set_msgrfwd_dispatch_control(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), dispatch_control);
    } 

    void set_msgrfwd_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        internal_set_msgrfwd_dispatch_control(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr), dispatch_control);
    } 

    void internal_set_msgrfwd_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t)); 
        };     

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrfwd_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        internal_set_msgrfwd_operatable_id(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), operatable_id);
    } 

    void set_msgrfwd_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        internal_set_msgrfwd_operatable_id(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr), operatable_id);
    }

    void internal_set_msgrfwd_injection_info(uma_ptr_t ptr, injection_info_t injection_info) noexcept{

        static_assert(std::is_trivially_copyable_v<injection_info_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_injection_info_addr(ptr);
            void * src      = &injection_info;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(injection_info_t)); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr); 
    }

    void set_msgrfwd_injection_info(uma_ptr_t ptr, injection_info_t injection_info){

        internal_set_msgrfwd_injection_info(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), injection_info);
    } 

    void set_msgrfwd_injection_info_nothrow(uma_ptr_t ptr, injection_info_t injection_info) noexcept{

        internal_set_msgrfwd_injection_info(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr), injection_info);
    }

    void internal_set_msgrfwd_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrfwd_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        internal_set_msgrfwd_pong_count(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), pong_count);
    } 

    void set_msgrfwd_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        internal_set_msgrfwd_pong_count(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr), pong_count);
    }

    void internal_set_msgrfwd_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>)); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrfwd_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        internal_set_msgrfwd_pong_addr(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), pong_addr);
    } 

    void set_msgrfwd_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        internal_set_msgrfwd_pong_addr(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr), pong_addr);
    }

    void internal_set_msgrbwd_src(uma_ptr_t ptr, uma_ptr_t src) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_src_addr(ptr);
            void * ssrc     = &src;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, ssrc, sizeof(uma_ptr_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrbwd_src(uma_ptr_t ptr, uma_ptr_t src){

        internal_set_msgrbwd_src(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), src);
    } 

    void set_msgrbwd_src_nothrow(uma_ptr_t ptr, uma_ptr_t src) noexcept{
        
        internal_set_msgrbwd_src(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr), src);
    }

    void internal_set_msgrbwd_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_dispatch_control_addr(ptr);
            void * src      = &dispatch_control;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrbwd_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        internal_set_msgrbwd_dispatch_control(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), dispatch_control);
    } 

    void set_msgrbwd_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        internal_set_msgrbwd_dispatch_control(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr), dispatch_control);
    } 

    void internal_set_msgrbwd_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_operatable_id_addr(ptr);
            void * src      = &operatable_id;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrbwd_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        internal_set_msgrbwd_operatable_id(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), operatable_id);
    } 

    void set_msgrbwd_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        internal_set_msgrbwd_operatable_id(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr), operatable_id);
    }

    void internal_set_msgrbwd_gbpc(uma_ptr_t ptr, gradient_count_t least) noexcept{ //gradient backprop count

        static_assert(std::is_trivially_copyable_v<gradient_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_gbpc_addr(ptr);
            void * src      = &least;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(gradient_count_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrbwd_gbpc(uma_ptr_t ptr, gradient_count_t least){

        internal_set_msgrbwd_gbpc(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), least);
    } 

    void set_msgrbwd_gbpc_nothrow(uma_ptr_t ptr, gradient_count_t least) noexcept{

        internal_set_msgrbwd_gbpc(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr), least);
    }

    void internal_set_msgrbwd_injection_info(uma_ptr_t ptr, injection_info_t injection_info) noexcept{

        static_assert(std::is_trivially_copyable_v<injection_info_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_injection_info_addr(ptr);
            void * src      = &injection_info;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(injection_info_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrbwd_injection_info(uma_ptr_t ptr, injection_info_t injection_info){

        internal_set_msgrbwd_injection_info(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), injection_info);
    } 

    void set_msgrbwd_injection_info_nothrow(uma_ptr_t ptr, injection_info_t injection_info) noexcept{

        internal_set_msgrbwd_injection_info(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr), injection_info);
    }

    void internal_set_msgrbwd_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_count_addr(ptr);
            void * src      = &pong_count;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrbwd_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        internal_set_msgrbwd_pong_count(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), pong_count);
    } 

    void set_msgrbwd_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        internal_set_msgrbwd_pong_count(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr), pong_count);
    }

    void internal_set_msgrbwd_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        auto cb_lambda = [&]<class Accessor>(const Accessor){
            uma_ptr_t dst   = Accessor::get_pong_addr_addr(ptr);
            void * src      = &pong_addr;
            dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>)); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
    }

    void set_msgrbwd_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        internal_set_msgrbwd_pong_addr(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), pong_addr);
    } 

    void set_msgrbwd_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        internal_set_msgrbwd_pong_addr(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr), pong_addr);
    }

    void internal_set_tile_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        using namespace dg::network_tile_member_access;
        auto id = poly_typeid(ptr);

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

    void set_tile_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){
        
        internal_set_tile_operatable_id(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), operatable_id);
    } 

    void set_tile_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        internal_set_tile_operatable_id(dg::network_tile_member_access::safe_tile_ptr_access(ptr), operatable_id);
    }

    void internal_set_tile_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{ 

        using namespace dg::network_tile_member_access;
        auto id = poly_typeid(ptr);

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

    void set_tile_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        internal_set_tile_dispatch_control(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), dispatch_control);
    } 

    void set_tile_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        internal_set_tile_dispatch_control(dg::network_tile_member_access::safe_tile_ptr_access(ptr), dispatch_control);
    }

    void internal_set_tile_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        using namespace dg::network_tile_member_access; 
        auto id = dg_typeid(ptr);

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
    
    void set_tile_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        internal_set_tile_pong_count(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), pong_count);
    } 

    void set_tile_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        internal_set_tile_pong_count(dg::network_tile_member_access::safe_tile_ptr_access(ptr), pong_count);
    } 

    void internal_set_tile_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        using namespace dg::network_tile_member_access;
        auto id = dg_typeid(ptr);

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

    void set_tile_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        internal_set_tile_pong_addr(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), pong_addr);
    } 

    void set_tile_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

        internal_set_tile_pong_addr(dg::network_tile_member_access::safe_tile_ptr_access(ptr), pong_addr);
    }

    auto internal_get_leaf_operatable_id(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs; 
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_leaf_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return internal_get_leaf_operatable_id(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr));
    } 

    auto get_leaf_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        return internal_get_leaf_operatable_id(dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
    }

    auto internal_get_leaf_pong_addr(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs; 
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_leaf_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_leaf_pong_addr(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr));
    } 

    auto get_leaf_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_leaf_pong_addr(dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
    }

    auto internal_get_mono_src(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_src_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_mono_src(uma_ptr_t ptr) -> uma_ptr_t{

        return internal_get_mono_src(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    } 

    auto get_mono_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        return internal_get_mono_src(dg::network_tile_member_access::safe_mono_ptr_access(ptr));
    }

    auto internal_get_mono_dispatch_control(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_mono_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return internal_get_mono_dispatch_control(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    auto get_mono_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        return internal_get_mono_dispatch_control(dg::network_tile_member_access::safe_mono_ptr_access(ptr));
    }

    auto internal_get_mono_operatable_id(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_mono_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return internal_get_mono_operatable_id(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    auto get_mono_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        return internal_get_mono_operatable_id(dg::network_tile_member_access::safe_mono_ptr_access(ptr));
    }

    auto internal_get_mono_pong_count(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t)); 
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_mono_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return internal_get_mono_pong_count(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    auto get_mono_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        return internal_get_mono_pong_count(dg::network_tile_member_access::safe_mono_ptr_access(ptr));
    } 

    auto internal_get_mono_pong_addr(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_mono_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_mono_pong_addr(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    } 

    auto get_mono_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_mono_pong_addr(dg::network_tile_member_access::safe_mono_ptr_access(ptr));
    }

    auto internal_get_pair_lhs(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_lhs_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lamda, ptr);
        return rs;
    }

    auto get_pair_lhs(uma_ptr_t ptr) -> uma_ptr_t{

        return internal_get_pair_lhs(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    } 

    auto get_pair_lhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        return internal_get_pair_lhs(dg::network_tile_member_access::safe_pair_ptr_access(ptr));
    }

    auto internal_get_pair_rhs(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_rhs_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_pair_rhs(uma_ptr_t ptr) -> uma_ptr_t{

        return internal_get_pair_rhs(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    }

    auto get_pair_rhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        return internal_get_pair_rhs(dg::network_tile_member_access::safe_pair_ptr_access(ptr));
    }

    auto internal_get_pair_dispatch_control(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_pair_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return internal_get_pair_dispatch_control(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    }

    auto get_pair_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        return internal_get_pair_dispatch_control(dg::network_tile_member_access::safe_pair_ptr_access(ptr));
    }

    auto internal_get_pair_operatable_id(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_pair_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return internal_get_pair_operatable_id(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    } 

    auto get_pair_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        return internal_get_pair_operatable_id(dg::network_tile_member_access::safe_pair_ptr_access(ptr));
    }

    auto internal_get_pair_pong_count(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lamda, ptr);
        return rs;
    }

    auto get_pair_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return internal_get_pair_pong_count(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    } 

    auto get_pair_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        return internal_get_pair_pong_count(dg::network_tile_member_access::safe_pair_ptr_access(ptr));
    }

    auto internal_get_pair_pong_addr(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_pair_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_pair_pong_addr(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    } 

    auto get_pair_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_pair_pong_addr(dg::network_tile_member_access::safe_pair_ptr_access(ptr));
    }

    auto internal_get_uacm_src(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_src_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lamda, ptr);
        return rs;
    }

    auto get_uacm_src(uma_ptr_t ptr) -> uma_ptr_t{

        return internal_get_uacm_src(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    } 

    auto get_uacm_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        return internal_get_uacm_src(dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
    } 

    auto internal_get_uacm_dispatch_control(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_uacm_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return internal_get_uacm_dispatch_control(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    } 

    auto get_uacm_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        return internal_get_uacm_dispatch_control(dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
    }

    auto internal_get_uacm_operatable_id(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lamda, ptr);
        return rs;
    }

    auto get_uacm_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return internal_get_uacm_operatable_id(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    } 

    auto get_uacm_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        return internal_get_uacm_operatable_id(dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
    }

    auto internal_get_uacm_pong_count(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lamda   = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_uacm_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return internal_get_uacm_pong_count(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    } 

    auto get_uacm_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        return internal_get_uacm_pong_count(dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
    }

    auto internal_get_uacm_pong_addr(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_uacm_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_uacm_pong_addr(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    } 

    auto get_uacm_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_uacm_pong_addr(dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
    }

    auto internal_get_pacm_lhs(uma_ptr_t ptr) noexcept -> uma_ptr_t{ //fix

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_lhs_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_pacm_lhs(uma_ptr_t ptr) -> uma_ptr_t{ //fix

        return internal_get_pacm_lhs(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    } 

    auto get_pacm_lhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{ //fix

        return internal_get_pacm_lhs(dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
    } 

    auto internal_get_pacm_rhs(uma_ptr_t ptr) noexcept -> uma_ptr_t{ //fix

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_rhs_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_pacm_rhs(uma_ptr_t ptr) -> uma_ptr_t{ //fix

        return internal_get_pacm_rhs(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    }

    auto get_pacm_rhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{ //fix

        return internal_get_pacm_rhs(dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
    } 

    auto internal_get_pacm_dispatch_control(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_pacm_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return internal_get_pacm_dispatch_control(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    } 

    auto get_pacm_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        return internal_get_pacm_dispatch_control(dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
    }

    auto internal_get_pacm_operatable_id(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_pacm_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return internal_get_pacm_operatable_id(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    } 

    auto get_pacm_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        return internal_get_pacm_operatable_id(dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
    }

    auto internal_get_pacm_pong_count(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_pacm_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return internal_get_pacm_pong_count(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    } 

    auto get_pacm_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        return internal_get_pacm_pong_count(dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
    }

    auto internal_get_pacm_pong_addr(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lamda, ptr);
        return rs;
    }

    auto get_pacm_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_pacm_pong_addr(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    } 

    auto get_pacm_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_pacm_pong_addr(dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
    }

    auto internal_get_crit_src(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_src_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_crit_src(uma_ptr_t ptr) -> uma_ptr_t{

        return internal_get_crit_src(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    } 

    auto get_crit_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        return internal_get_crit_src(dg::network_tile_member_access::safe_crit_ptr_access(ptr));
    }

    auto internal_get_crit_dispatch_control(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_crit_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return internal_get_crit_dispatch_control(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    } 

    auto get_crit_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        return internal_get_crit_dispatch_control(dg::network_tile_member_access::safe_crit_ptr_access(ptr));
    } 

    auto internal_get_crit_operatable_id(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_crit_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return internal_get_crit_operatable_id(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    } 

    auto get_crit_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        return internal_get_crit_operatable_id(dg::network_tile_member_access::safe_crit_ptr_access(ptr));
    } 

    auto internal_get_crit_pong_count(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_crit_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return internal_get_crit_pong_count(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    } 

    auto get_crit_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        return internal_get_crit_pong_count(dg::network_tile_member_access::safe_crit_ptr_access(ptr));
    }

    auto internal_get_crit_pong_addr(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_crit_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_crit_pong_addr(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    } 

    auto get_crit_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_crit_pong_addr(dg::network_tile_member_access::safe_crit_ptr_access(ptr));
    }

    auto internal_get_msgrfwd_src(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_src_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrfwd_src(uma_ptr_t ptr) -> uma_ptr_t{

        return internal_get_msgrfwd_src(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrfwd_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        return internal_get_msgrfwd_src(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
    }

    auto internal_get_msgrfwd_dispatch_control(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrfwd_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return internal_get_msgrfwd_dispatch_control(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrfwd_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        return internal_get_msgrfwd_dispatch_control(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
    }

    auto internal_get_msgrfwd_operatable_id(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };     

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrfwd_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return internal_get_msgrfwd_operatable_id(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrfwd_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        return internal_get_msgrfwd_operatable_id(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
    }

    auto internal_get_msgrfwd_injection_info(uma_ptr_t ptr) noexcept -> injection_info_t{

        static_assert(std::is_trivially_copyable_v<injection_info_t>);
        auto rs         = injection_info_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_injection_info_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(injection_info_t));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrfwd_injection_info(uma_ptr_t ptr) -> injection_info_t{

        return internal_get_msgrfwd_injection_info(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    }

    auto get_msgrfwd_injection_info_nothrow(uma_ptr_t ptr) noexcept -> injection_info_t{

        return internal_get_msgrfwd_injection_info(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
    }

    auto internal_get_msgrfwd_pong_count(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrfwd_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return internal_get_msgrfwd_pong_count(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrfwd_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        return internal_get_msgrfwd_pong_count(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
    } 

    auto internal_get_msgrfwd_pong_addr(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrfwd_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_msgrfwd_pong_addr(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrfwd_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_msgrfwd_pong_addr(dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
    }

    auto internal_get_msgrbwd_src(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_src_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrbwd_src(uma_ptr_t ptr) -> uma_ptr_t{

        return internal_get_msgrbwd_src(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    auto get_msgrbwd_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        return internal_get_msgrbwd_src(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
    }

    auto internal_get_msgrbwd_dispatch_control(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        auto rs         = dispatch_control_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_dispatch_control_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrbwd_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return internal_get_msgrbwd_dispatch_control(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    auto get_msgrbwd_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        return internal_get_msgrbwd_dispatch_control(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
    }

    auto internal_get_msgrbwd_operatable_id(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        auto rs         = operatable_id_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_operatable_id_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrbwd_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return internal_get_msgrbwd_operatable_id(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    auto get_msgrbwd_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        return internal_get_msgrbwd_operatable_id(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
    }

    auto internal_get_msgrbwd_gbpc(uma_ptr_t ptr) noexcept -> gradient_count_t{

        static_assert(std::is_trivially_copyable_v<gradient_count_t>);
        auto rs         = gradient_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_gbpc_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(gradient_count_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrbwd_gbpc(uma_ptr_t ptr) -> gradient_count_t{

        return internal_get_msgrbwd_gbpc(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    auto get_msgrbwd_gbpc_nothrow(uma_ptr_t ptr) noexcept -> gradient_count_t{

        return internal_get_msgrbwd_gbpc(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
    }

    auto internal_get_msgrbwd_injection_info(uma_ptr_t ptr) noexcept -> injection_info_t{

        static_assert(std::is_trivially_copyable_v<injection_info_t>);
        auto rs         = injection_info_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_injection_info_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(injection_info_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrbwd_injection_info(uma_ptr_t ptr) -> injection_info_t{

        return internal_get_msgrbwd_injection_info(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    auto get_msgrbwd_injection_info_nothrow(uma_ptr_t ptr) noexcept -> injection_info_t{

        return internal_get_msgrbwd_injection_info(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
    }

    auto internal_get_msgrbwd_pong_count(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        auto rs         = pong_count_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_count_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrbwd_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return internal_get_msgrbwd_pong_count(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    auto get_msgrbwd_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        return internal_get_msgrbwd_pong_count(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
    }

    auto internal_get_msgrbwd_pong_addr(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);
        auto rs         = std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            void * dst      = &rs;
            uma_ptr_t src   = Accessor::get_pong_addr_addr(ptr);
            dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        return rs;
    }

    auto get_msgrbwd_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_msgrbwd_pong_addr(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    auto get_msgrbwd_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_msgrbwd_pong_addr(dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
    }

    auto internal_get_tile_operatable_id(uma_ptr_t ptr) noexcept -> operatable_id_t{

        using namespace dg::network_tile_member_access;
        auto id = dg_typeid(ptr);

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

    auto get_tile_operatable_id(uma_ptr_t ptr) -> operatable_id{

        return internal_get_tile_operatable_id(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    } 
    
    auto get_tile_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        return internal_get_tile_operatable_id(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
    }

    auto internal_get_tile_dispatch_control(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        using namespace dg::network_tile_member_access;
        auto id = dg_typeid(ptr);
        
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

    auto get_tile_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return internal_get_tile_dispatch_control(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    } 

    auto get_tile_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        return internal_get_tile_dispatch_control(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
    }

    auto internal_get_tile_pong_count(uma_ptr_t ptr) noexcept -> pong_count_t{

        using namespace dg::network_tile_member_access;
        auto id = dg_typeid(ptr);

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

    auto get_tile_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return internal_get_tile_pong_count(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    } 

    auto get_tile_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        return internal_get_tile_pong_count(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
    }

    auto internal_get_tile_pong_addr(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        using namespace dg::network_tile_member_access;
        auto id = dg_typeid(ptr);

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

    auto get_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return internal_get_tile_pong_addr(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    } 

    auto get_pong_addr_nothrow(uma_ptr_t) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{
        
        return internal_get_tile_pong_addr(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
    }
}

#endif