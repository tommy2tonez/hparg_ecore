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

    void set_leaf_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    void set_leaf_operatable_id(uma_ptr_t ptr, operatable_id operatable_id){

        set_leaf_operatable_id_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr), operatable_id);
    } 

    template <size_t ARR_IDX>
    void set_leaf_observer_addr_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(std::array<uma_ptr_t, MAX_PONGADDR_COUNT>));
    }

    template <size_t ARR_IDX>
    void set_leaf_observer_addr(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>){

        set_leaf_observer_addr_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    }

    void set_mono_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t descendant) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &descendant;
        auto cb_lambda = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    void set_mono_descendant(uma_ptr_t ptr, uma_ptr_t src){

        set_mono_descendant_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), src);
    }

    void set_mono_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); //it's unified fileio responsibility to make sure this is nothrow ops - it's important to stop the stack unwinding of memcpy here - at getter and setter - it's hardly useful otherwise
    }

    void set_mono_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        set_mono_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), dispatch_control);
    }

    void set_mono_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    void set_mono_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        set_mono_operatable_id_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), operatable_id);
    } 

    void set_mono_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        uma_ptr_t dst   = {};
        void * src      = &pong_count;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    } 

    void set_mono_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        set_mono_pong_count_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), pong_count);
    } 

    template <size_t ARR_IDX>
    void set_mono_observer_addr_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>) noexcept{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    template <size_t ARR_IDX>
    void set_mono_observer_addr(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>){

        set_mono_observer_addr_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    }

    void set_pair_left_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::left_descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    void set_pair_left_descendant(uma_ptr_t ptr, uma_ptr_t addr){

        set_pair_left_descendant_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), addr);
    }

    void set_pair_right_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            dst = Accessor::right_descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
    }

    void set_pair_right_descendant(uma_ptr_t ptr, uma_ptr_t addr){

        set_pair_right_descendant_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), addr);
    } 

    void set_pair_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            dst = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    void set_pair_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        set_pair_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), dispatch_control);
    } 

    void set_pair_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            dst = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    void set_pair_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        set_pair_operatable_id_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), operatable_id);
    }

    void set_pair_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        uma_ptr_t dst   = {};
        void * src      = &pong_count;
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            dst = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    }

    void set_pair_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        set_pair_pong_count_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), pong_count);
    } 

    template <size_t ARR_IDX>
    void set_pair_observer_addr_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        
        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            dst = Accessor::observer_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    template <size_t ARR_IDX>
    void set_pair_observer_addr(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>){

        set_pair_observer_addr_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    } 

    template <size_t ACM_IDX>
    void set_uacm_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ACM_IDX>) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::descendant_addr(ptr, std::integral_constant<size_t, ACM_IDX>{});
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    template <size_t ACM_IDX>
    void set_uacm_descendant(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ACM_IDX>){

        set_uacm_descendant_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), addr, std::integral_constant<size_t, ACM_IDX>{});
    } 

    void set_uacm_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lamda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    void set_uacm_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        set_uacm_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), dispatch_control);
    } 

    void set_uacm_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lamda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    void set_uacm_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        set_uacm_operatable_id_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), operatable_id);
    } 

    void set_uacm_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        uma_ptr_t dst   = {};
        void * src      = &pong_count; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    }

    void set_uacm_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        set_uacm_pong_count_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), pong_count);
    }

    template <size_t ARR_IDX>
    void set_uacm_observer_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>) noexcept{

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    template <size_t ARR_IDX>
    void set_uacm_observer(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>){

        set_uacm_observer_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    }

    template <size_t ACM_IDX>
    void set_pacm_left_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ACM_IDX>) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::left_descendant(ptr, std::integral_constant<size_t, ACM_IDX>{});
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }
    
    template <size_t ACM_IDX>
    void set_pacm_left_descendant(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ACM_IDX>){

        set_pacm_left_descendant_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), addr, std::integral_constant<size_t, ACM_IDX>{});
    } 

    template <size_t ACM_IDX>
    void set_pacm_right_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ACM_IDX>) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        
        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::right_descendant_addr(ptr, std::integral_constant<size_t, ACM_IDX>{});
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    template <size_t ACM_IDX>
    void set_pacm_right_descendant(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ACM_IDX>){ //fix

        set_pacm_right_descendant_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), addr, std::integral_constant<size_t, ACM_IDX>{});
    }

    void set_pacm_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };
        
        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    void set_pacm_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        set_pacm_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), dispatch_control);
    }

    void set_pacm_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }
    
    void set_pacm_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        set_pacm_operatable_id_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), operatable_id);
    } 

    void set_pacm_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        uma_ptr_t dst   = {};
        void * src      = &pong_count;
        auto cb_lambda  = [&]<class Accessor>(const Accessor){
            dst = Accessor::get_pong_count_addr(ptr); 
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    }

    void set_pacm_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        set_pacm_pong_count_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), pong_count); 
    }

    template <size_t ARR_IDX>
    void set_pacm_observer_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>) noexcept{ 

        static_assert(std::is_trivially_copyable_v<std::array<uma_ptr_t, MAX_PONGADDR_COUNT>>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
    } 

    template <size_t ARR_IDX>
    void set_pacm_observer(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>){

        set_pacm_observer_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    } 
    
    void set_crit_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::descendant_addr(ptr); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    void set_crit_descendant(uma_ptr_t ptr, uma_ptr_t addr){

        set_crit_descendant_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), addr);
    } 

    void set_crit_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::get_dispatch_control_addr(ptr); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    void set_crit_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        set_crit_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), dispatch_control);
    } 

    void set_crit_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    void set_crit_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        set_crit_operatable_id_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), operatable_id);
    } 

    void set_crit_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        uma_ptr_t dst   = {};
        void * src      = &pong_count;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::pong_count_addr(ptr); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    }

    void set_crit_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        set_crit_pong_count_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), pong_count);
    }

    template <size_t ARR_IDX>
    void set_crit_observer_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{}); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    template <size_t ARR_IDX>
    void set_crit_observer(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>){

        set_crit_observer_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    } 

    void set_msgrfwd_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::descendant_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, src);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    void set_msgrfwd_descendant(uma_ptr_t ptr, uma_ptr_t addr){

        set_msgrfwd_descendant_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), addr);
    }

    void set_msgrfwd_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    void set_msgrfwd_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        set_msgrfwd_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), dispatch_control);
    } 

    void set_msgrfwd_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };     

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    void set_msgrfwd_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        set_msgrfwd_operatable_id_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), operatable_id);
    } 

    void set_msgrfwd_dst_info_nothrow(uma_ptr_t ptr, dst_info_t dst_info) noexcept{

        static_assert(std::is_trivially_copyable_v<dst_info_t>);

        uma_ptr_t dst   = {};
        void * src      = &dst_info;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dst_info_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr); 
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dst_info_t));
    }

    void set_msgrfwd_injection_info(uma_ptr_t ptr, dst_info_t dst_info){

        set_msgrfwd_injection_info_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), dst_info);
    } 

    void set_msgrfwd_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        uma_ptr_t dst   = {};
        void * src      = &pong_count; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor noexcept){
            dst = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    }
    
    void set_msgrfwd_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        set_msgrfwd_pong_count_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), pong_count);
    } 

    template <size_t ARR_IDX>
    void set_msgrfwd_observer_addr_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    void set_msgrfwd_observer_addr(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>){

        set_msgrfwd_observer_addr_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    } 

    void set_msgrbwd_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{
        
        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::descendant_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    void set_msgrbwd_descendant(uma_ptr_t ptr, uma_ptr_t src){

        set_msgrbwd_descendant_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), src);
    } 
    
    void set_msgrbwd_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    void set_msgrbwd_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        set_msgrbwd_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), dispatch_control);
    } 

    void set_msgrbwd_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    void set_msgrbwd_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){

        set_msgrbwd_operatable_id_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), operatable_id);
    } 

    void set_msgrbwd_timein_nothrow(uma_ptr_t ptr, timein_t timein) noexcept{/

        static_assert(std::is_trivially_copyable_v<timein_t>);

        uma_ptr_t dst   = {};
        void * src      = &timein;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::timein_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(timein_t));
    }

    void set_msgrbwd_timein(uma_ptr_t ptr, timein_t timein){

        set_msgrbwd_timein_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), timein);
    }

    void set_msgrbwd_dst_info_nothrow(uma_ptr_t ptr, dst_info_t dst_info) noexcept{

        static_assert(std::is_trivially_copyable_v<dst_info_t>);

        uma_ptr_t dst   = {};
        void * src      = &dst_info;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dst_info_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dst_info_t));
    }

    void set_msgrbwd_dst_info(uma_ptr_t ptr, dst_info_t dst_info){

        set_msgrbwd_dst_info_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), dst_info);
    } 

    void set_msgrbwd_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        
        uma_ptr_t dst   = {};
        void * src      = &pong_count;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::pong_count_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    }

    void set_msgrbwd_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        set_msgrbwd_pong_count_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), pong_count);
    } 

    template <size_t ARR_IDX>
    void set_msgrbwd_observer_addr_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    template <size_t ARR_IDX>
    void set_msgrbwd_observer_addrr(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>){

        set_msgrbwd_observer_addr_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    }

    void set_tile_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        using namespace dg::network_tile_member_access;
        
        const auto id   = poly_typeid(ptr);
        uma_ptr_t dst   = {};
        void * src      = &operatable_id; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);    
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_crit_tile(id)){
            get_crit_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_msgrfwd_tile(id)){
            get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_msgrbwd_tile(id)){
            get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }

        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    void set_tile_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id){
        
        set_tile_operatable_id_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), operatable_id);
    }

    void set_tile_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        using namespace dg::network_tile_member_access;
        const auto id   = poly_typeid(ptr);
        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = []<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };

        if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_crit_tile(id)){
            get_crit_static_polymorphic_accessor(cb_lambda ptr, id);
        } else if (is_msgrfwd_tile(id)){
            get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_msgrbwd_tile(id)){
            get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr, id);
            return;
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }

        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    void set_tile_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        set_tile_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), dispatch_control);
    } 

    void set_tile_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        using namespace dg::network_tile_member_access; 
        const auto id   = dg_typeid(ptr);
        uma_ptr_t dst   = {};
        void * src      = &pong_count;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::pong_count_addr(ptr);
        };

        if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_crit_tile(id)){
            get_crit_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_msgrfwd_tile(id)){
            get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_msgrbwd_tile(id)){
            get_msgrbwd_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }

        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    }
    
    void set_tile_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        set_tile_pong_count_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), pong_count);
    } 

    void set_tile_pong_addr_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr) noexcept{

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

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        } else{
            std::unreachable();
        }
    }

    void set_tile_pong_addr(uma_ptr_t ptr, std::array<uma_ptr_t, MAX_PONGADDR_COUNT> pong_addr){

        set_tile_pong_addr_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), pong_addr);
    }

    auto get_leaf_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

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

        return get_leaf_operatable_id_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr));
    } 

    auto get_leaf_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

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

        return get_leaf_pong_addr_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr));
    } 

    auto get_mono_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

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

        return get_mono_src_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    } 

    auto get_mono_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

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

        return get_mono_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    auto get_mono_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

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

        return get_mono_operatable_id_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    auto get_mono_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

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

        return get_mono_pong_count_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    auto get_mono_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

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

        return get_mono_pong_addr_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    } 

    auto get_pair_lhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

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

        return get_pair_lhs_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    } 

    auto get_pair_rhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

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

        return get_pair_rhs_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    }

    auto get_pair_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

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

        return get_pair_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    }

    auto get_pair_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

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

        return get_pair_operatable_id_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    } 

    auto get_pair_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

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

        return get_pair_pong_count_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    } 

    auto get_pair_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

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

        return get_pair_pong_addr_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    } 

    auto get_uacm_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

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

        return get_uacm_src_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    } 

    auto get_uacm_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

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

        return get_uacm_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    } 

    auto get_uacm_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

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

        return get_uacm_operatable_id_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    } 

    auto get_uacm_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

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

        return get_uacm_pong_count_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    } 

    auto get_uacm_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

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

        return get_uacm_pong_addr_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    } 


    auto get_pacm_lhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{ //fix

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

        return get_pacm_lhs_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    } 

    auto get_pacm_rhs_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{ //fix

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

        return get_pacm_rhs_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    }

    auto get_pacm_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

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

        return get_pacm_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    } 

    auto get_pacm_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

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

        return get_pacm_operatable_id_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    } 

    auto get_pacm_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

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

        return get_pacm_pong_count_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    } 

    auto get_pacm_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

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

        return get_pacm_pong_addr_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    } 

    auto get_crit_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

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

        return get_crit_src_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    } 

    auto get_crit_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

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

        return get_crit_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    }

    auto get_crit_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

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

        return get_crit_operatable_id_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    } 

    auto get_crit_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

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

        return get_crit_pong_count_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    } 

    auto get_crit_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

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

        return get_crit_pong_addr_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    } 

    auto get_msgrfwd_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

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

        return get_msgrfwd_src_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrfwd_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

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

        return get_msgrfwd_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrfwd_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

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

        return get_msgrfwd_operatable_id_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrfwd_injection_info_nothrow(uma_ptr_t ptr) noexcept -> injection_info_t{

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

        return get_msgrfwd_injection_info_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    }

    auto get_msgrfwd_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

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

        return get_msgrfwd_pong_count_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrfwd_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

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

        return get_msgrfwd_pong_addr_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrbwd_src_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

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

        return get_msgrbwd_src_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    auto get_msgrbwd_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

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

        return get_msgrbwd_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    auto get_msgrbwd_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

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

        return get_msgrbwd_operatable_id_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    auto get_msgrbwd_gbpc_nothrow(uma_ptr_t ptr) noexcept -> gradient_count_t{

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

        return get_msgrbwd_gbpc_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    auto get_msgrbwd_injection_info_nothrow(uma_ptr_t ptr) noexcept -> injection_info_t{

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

        return get_msgrbwd_injection_info_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    auto get_msgrbwd_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

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

        return get_msgrbwd_pong_count_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    auto get_msgrbwd_pong_addr_nothrow(uma_ptr_t ptr) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

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

        return get_msgrbwd_pong_addr_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    auto get_tile_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        //this should directly call the get_leaf_static_polymorphic_accessor for optimization - 2 nested for-loops are optimized -> a table dispatch by a decent compiler - important to force inline here
        //if there is no holes in the dispatch code - this is an important note

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

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        } else{
            std::unreachable();
        }
    }

    auto get_tile_operatable_id(uma_ptr_t ptr) -> operatable_id{

        return get_tile_operatable_id_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    } 
    
    auto get_tile_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

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

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        } else{
            std::unreachable();
        }
    }

    auto get_tile_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return get_tile_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    }

    auto get_tile_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

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

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        } else{
            std::unreachable();
        }
    }

    auto get_tile_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return get_tile_pong_count_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    }

    auto get_pong_addr_nothrow(uma_ptr_t) noexcept -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{
        
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

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        } else{
            std::unreachable();
        }
    }

    auto get_pong_addr(uma_ptr_t ptr) -> std::array<uma_ptr_t, MAX_PONGADDR_COUNT>{

        return get_pong_addr_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    }
}

#endif