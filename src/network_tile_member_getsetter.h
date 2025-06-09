#ifndef __DG_NETWORK_TILE_MEMBER_GETSETTER_H__
#define __DG_NETWORK_TILE_MEMBER_GETSETTER_H__

#include "network_exception.h"
#include "network_exception_handler.h"
#include "network_memops_uma.h"
#include "network_tile_member_access.h"
#include "network_pointer.h"
#include "network_tile_metadata.h"

namespace dg::network_tile_member_getsetter{

    //alright, I made some terrible terrible mistakes
    //which we'd attempt to fix by using poly

    //first is instruction cache
    //second is "frequently accessed datatypes" at a point should be packed into a struct
    //third is that we have to do some sort of fast path of unified memory address to get the data, I dont know how to solve this without breaking the readability

    //we'd attempt to storage raw storage layout on the memaccess
    //we'd attempt to semanticalize + desemanticalize (deserialize + serialize the underlying buffer here)

    //we are still thinking that the partitioning forward + backward sigagg responsibility is the sigagg tile, we dont really want to introduce another member for ALL of the forwarding tiles
    //that's not an option

    //I'm still very dont know if smph_addr for forward + backward + memevent_id needs a separate one
    //logically speaking, yes
    //conveniently speaking, no
    //we'll be back tmr to implement this

    static inline constexpr size_t OBSERVER_ARR_SZ = dg::network_tile_metadata::OBSERVER_ARRAY_SZ; 

    using uma_ptr_t             = dg::network_pointer::uma_ptr_t;
    using init_status_t         = dg::network_tile_metadata::init_status_t;
    using observer_t            = dg::network_tile_metadata::observer_t;
    using operatable_id_t       = dg::network_tile_metadata::operatable_id_t;
    using dispatch_control_t    = dg::network_tile_metadata::dispatch_control_t;
    using pong_count_t          = dg::network_tile_metadata::pong_count_t;
    using crit_kind_t           = dg::network_tile_metadata::crit_kind_t;
    using dst_info_t            = dg::network_tile_metadata::dst_info_t;
    using timein_t              = dg::network_tile_metadata::timein_t;

    //

    void set_immu_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{

    }

    auto set_immu_init_status(uma_ptr_t ptr, init_status_t init_status) noexcept -> exception_t{

    }

    void set_immu_logit_nothrow(uma_ptr_t ptr, void * logit_ptr, size_t logit_bsz) noexcept{

    }

    auto set_immu_logit(uma_ptr_t ptr, void * logit_ptr, size_t logit_bsz) noexcept -> exception_t{

    }

    void set_immu_memevent_operatable_id_nothrow(uma_ptr_t, operatable_id_set_t operatable_id) noexcept{

    }

    auto set_immu_memevent_operatable_id(uma_ptr_t, operatable_id_set_t operatable_id) noexcept -> exception_t{

    }

    void set_immu_forward_operatable_id_nothrow(uma_ptr_t, operatable_id_t operatable_id) noexcept{

    }

    auto set_immu_forward_operatable_id(uma_ptr_t, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    //

    void set_leaf_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        uma_ptr_t dst   = {};
        void * src      = &init_status;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(init_status_t));
    }

    auto set_leaf_init_status(uma_ptr_t ptr, init_status_t init_status) noexcept -> exception_t{

        set_leaf_init_status_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr), init_status);
    }

    void set_leaf_logit_nothrow(uma_ptr_t ptr, void * logit_ptr, size_t logit_bsz) noexcept{

        uma_ptr_t dst       = {};
        void * src          = addr;
        size_t cpy_sz       = {};
        auto cb_lambda      = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_leaf_logit(uma_ptr_t ptr, void * logit_ptr, size_t logit_bsz) noexcept -> exception_t{

        if (!addr){
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }

        set_leaf_logit_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr), addr);
    }

    void set_leaf_grad_nothrow(uma_ptr_t ptr, void * grad_ptr, size_t grad_bsz) noexcept{
 
        uma_ptr_t dst       = {};
        void * src          = addr;
        size_t cpy_sz       = {};
        auto cb_lambda      = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_leaf_grad(uma_ptr_t ptr, void * grad_ptr, size_t grad_bsz) noexcept -> exception_t{

        if (!addr){
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }

        set_leaf_grad_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr), addr);
    }

    void set_leaf_grad_status_nothrow(uma_ptr_t ptr, grad_status_t) noexcept{

    }

    auto set_leaf_grad_status(uma_ptr_t ptr, grad_status_t) noexcept -> exception_t{

    }

    void set_leaf_memevent_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_set_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    auto set_leaf_memevent_operatable_id(uma_ptr_t ptr, operatable_id_set_t operatable_id) noexcept -> exception_t{

    }

    void set_leaf_forward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_leaf_forward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_leaf_backward_operatable_id_nothrow(uma_ptr_t, operatable_id_t operatable_id) noexcept{

    }

    auto set_leaf_backward_operatable_id(uma_ptr_t, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    //

    void set_mono_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        uma_ptr_t dst   = {};
        void * src      = &init_status;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(init_status_t));
    }

    auto set_mono_init_status(uma_ptr_t ptr, init_status_t init_status) noexcept -> exception_t{

        set_mono_init_status_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), init_status);
    }

    void set_mono_observer_nothrow(uma_ptr_t ptr, const TileObserver& observer, size_t idx) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_mono_observer(uma_ptr_t ptr, const TileObserver& observer, size_t idx) noexcept -> exception_t{

        set_mono_observer_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    }

    void set_mono_observer_size_nothrow(uma_ptr_t ptr, size_t observer_sz) noexcept{

    }

    auto set_mono_observer_size(uma_ptr_t ptr, size_t observer_sz) noexcept -> exception_t{

    } 

    void set_mono_logit_nothrow(uma_ptr_t ptr, void * logit_ptr, size_t logit_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_mono_logit(uma_ptr_t ptr, void * logit_ptr, size_t logit_bsz) noexcept -> exception_t{

        set_mono_logit_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), addr);
    }

    void set_mono_grad_nothrow(uma_ptr_t ptr, void * grad_ptr, size_t grad_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_mono_grad(uma_ptr_t ptr, void * grad_ptr, size_t grad_bsz) noexcept -> exception_t{

    }

    void set_mono_signal_smph_addr_nothrow(uma_ptr_t, std::optional<uma_ptr_t>) noexcept{

    }

    auto set_mono_signal_smph_addr(uma_ptr_t, std::optional<uma_ptr_t>) noexcept -> exception_t{

    }

    void set_mono_memevent_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    auto set_mono_memevent_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        set_mono_operatable_id_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), operatable_id);
    }

    void set_mono_forward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    } 

    auto set_mono_forward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }
    
    void set_mono_backward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_mono_backward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_mono_forward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t)); //it's unified fileio responsibility to make sure this is nothrow ops - it's important to stop the stack unwinding of memcpy here - at getter and setter - it's hardly useful otherwise
    }

    auto set_mono_forward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) -> exception_t{

        set_mono_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), dispatch_control);
    }

    void set_mono_backward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

    }

    auto set_mono_backward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

    }

    void set_mono_grad_status_nothrow(uma_ptr_t ptr, grad_status_t grad_status) noexcept{

    }

    auto set_mono_grad_status(uma_ptr_t ptr, grad_status_t grad_status) noexcept -> exception_t{

    }

    void set_mono_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t descendant) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &descendant;
        auto cb_lambda = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_mono_descendant(uma_ptr_t ptr, uma_ptr_t src) noexcept -> exception_t{

        set_mono_descendant_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), src);
    }

    //

    //blkr

    void set_blkr_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{

    }

    auto set_blkr_init_status(uma_ptr_t ptr, init_status_t init_status) noexcept -> exception_t{

    }

    void set_blkr_observer_nothrow(uma_ptr_t ptr, const TileObserver& observer, size_t idx) noexcept{

    }

    auto set_blkr_observer(uma_ptr_t ptr, const TileObserver& observer, size_t idx) noexcept -> exception_t{

    }

    void set_blkr_observer_size_nothrow(uma_ptr_t ptr, size_t sz) noexcept{

    }

    auto set_blkr_observer_size(uma_ptr_t ptr, size_t sz) noexcept -> exception_t{

    }

    void set_blkr_logit_nothrow() noexcept{

    }

    auto set_blkr_logit() noexcept -> exception_t{

    }

    void set_blkr_signal_smph_addr_nothrow() noexcept{

    }

    auto set_blkr_signal_smph_addr() noexcept -> exception_t{

    }

    void set_blkr_memevent_operatable_id_nothrow() noexcept{

    }

    auto set_blkr_memevent_operatable_id() noexcept -> exception_t{

    }

    void set_blkr_forward_operatable_id_nothrow() noexcept{

    }

    auto set_blkr_forward_operatable_id() noexcept -> exception_t{

    }

    void set_blkr_forward_dispatch_control_nothrow() noexcept{

    }

    auto set_blkr_forward_dispatch_control() noexcept -> exception_t{

    }

    void set_blkr_descendant_nothrow() noexcept{

    }

    auto set_blkr_descendant() noexcept -> exception_t{

    }

    //

    void set_pair_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{
        
        static_assert(std::is_trivially_copyable_v<init_status_t>);

        uma_ptr_t dst   = {};
        void * src      = &init_status;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(init_status_t));
    } 

    auto set_pair_init_status(uma_ptr_t ptr, init_status_t init_status) noexcept -> exception_t{

        set_pair_init_status_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), init_status);
    }

    void set_pair_observer_nothrow(uma_ptr_t ptr, const TileObserver& addr, size_t idx) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        
        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_pair_observer(uma_ptr_t ptr, const TileObserver& addr, size_t idx) noexcept -> exception_t{

        set_pair_observer_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    } 

    void set_pair_observer_size_nothrow(uma_ptr_t ptr, size_t sz) noexcept{

    }

    auto set_pair_observer_size(uma_ptr_t ptr, size_t sz) noexcept -> exception_t{

    } 

    void set_pair_logit_nothrow(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }
    
    auto set_pair_logit(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept -> exception_t{

        set_pair_logit_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), addr);
    }

    void set_pair_grad_nothrow(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_pair_grad(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept -> exception_t{

        set_pair_grad_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), addr);
    }

    void set_pair_signal_smph_addr_nothrow(uma_ptr_t ptr, std::optional<uma_ptr_t> smph_addr) noexcept{

    }

    auto set_pair_signal_smph_addr(uma_ptr_t ptr, std::optional<uma_ptr_t> smph_addr) noexcept -> exception_t{

    } 

    void set_pair_memevent_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    auto set_pair_memevent_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        set_pair_operatable_id_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), operatable_id);
    }

    void set_pair_forward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{
    
    }

    auto set_pair_forward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_pair_backward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_pair_backward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_pair_forward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    auto set_pair_forward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

        set_pair_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), dispatch_control);
    }

    void set_pair_backward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

    }

    auto set_pair_backward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

    } 

    void set_pair_grad_status_nothrow(uma_ptr_t ptr, grad_status_t grad_status) noexcept{

    }

    auto set_pair_grad_status(uma_ptr_t ptr, grad_status_t grad_status) noexcept -> exception_t{

    } 

    void set_pair_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        uma_ptr_t dst   = {};
        void * src      = &pong_count;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    }

    auto set_pair_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept -> exception_t{

        set_pair_pong_count_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), pong_count);
    }

    void set_pair_dispatch_major_nothrow(uma_ptr_t ptr, dispatch_major_t dispatch_major) noexcept{

    }

    auto set_pair_dispatch_major(uma_ptr_t, dispatch_major_t dispatch_major) noexcept -> exception_t{

    } 

    void set_pair_left_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::left_descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_pair_left_descendant(uma_ptr_t ptr, uma_ptr_t addr) noexcept -> exception_t{

        set_pair_left_descendant_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), addr);
    }

    void set_pair_right_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::right_descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
    }

    auto set_pair_right_descendant(uma_ptr_t ptr, uma_ptr_t addr) noexcept -> exception_t{

        set_pair_right_descendant_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), addr);
    } 

    //poly does another type check to return get/setter errors
    //the errors are intentional to actually disable certain forward + backward + features getters + setters
    //so it's important to set_poly_poly_type first

    //this is important

    void set_poly_init_status_nothrow(uma_ptr_t, init_status_t init_status) noexcept{

    }

    auto set_poly_init_status(uma_ptr_t, init_status_t) noexcept -> exception_t{

    }

    void set_poly_observer_nothrow(uma_ptr_t, const TileObserver&, size_t idx) noexcept{

    }

    auto set_poly_observer(uma_ptr_t, const TileObserver&, size_t idx) noexcept -> exception_t{

    }

    void set_poly_observer_size_nothrow(uma_ptr_t, size_t) noexcept{

    }

    auto set_poly_observer_size(uma_ptr_t, size_t) noexcept -> exception_t{

    }

    void set_poly_poly_type_nothrow(uma_ptr_t, poly_tile_t) noexcept{

    }

    auto set_poly_poly_type(uma_ptr_t, poly_tile_t) noexcept -> exception_t{

    }

    void set_poly_logit_nothrow(uma_ptr_t, void * logit_addr, size_t logit_bsz) noexcept{

    }

    auto set_poly_logit(uma_ptr_t, void * logit_addr, size_t logit_bsz) noexcept -> exception_t{

    }

    void set_poly_grad_nothrow(uma_ptr_t, void * grad_addr, size_t grad_bsz) noexcept{

    }

    auto set_poly_grad(uma_ptr_t, void * grad_addr, size_t grad_bsz) noexcept -> exception_t{

    }

    void set_poly_signal_smph_addr_nothrow(uma_ptr_t ptr, std::optional<uma_ptr_t> smph_addr) noexcept{

    }

    auto set_poly_signal_smph_addr(uma_ptr_t ptr, std::optional<uma_ptr_t> smph_addr) noexcept -> exception_t{

    }

    void set_poly_memevent_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_poly_memevent_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_poly_forward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_poly_forward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_poly_backward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_poly_backward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_poly_forward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

    }

    auto set_poly_forward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

    }

    void set_poly_backward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

    }

    auto set_poly_backward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

    }

    void set_poly_grad_status_nothrow(uma_ptr_t ptr, grad_status_t grad_status) noexcept{

    }

    auto set_poly_grad_status(uma_ptr_t ptr, grad_status_t grad_status) noexcept -> exception_t{

    }

    void set_poly_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

    }

    auto set_poly_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept -> exception_t{

    }

    void set_poly_expected_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

    }

    auto set_poly_expected_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept -> exception_t{

    }

    void set_poly_dispatch_major_nothrow(uma_ptr_t, dispatch_major_t dispatch_major) noexcept{

    }

    auto set_poly_dispatch_major(uma_ptr_t, dispatch_major_t dispatch_major) noexcept -> exception_t{

    }

    void set_poly_left_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{

    }

    auto set_poly_left_descendant(uma_ptr_t ptr, uma_ptr_t addr) noexcept -> exception_t{

    }

    void set_poly_right_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{

    }

    auto set_poly_right_descendant(uma_ptr_t ptr, uma_ptr_t addr) noexcept -> exception_t{

    }

    //

    void set_uacm_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        uma_ptr_t dst   = {};
        void * src      = &init_status;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(init_status_t));
    }

    auto set_uacm_init_status(uma_ptr_t ptr, init_status_t init_status) noexcept -> exception_t{

        set_uacm_init_status_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), init_status);
    }

    void set_uacm_observer_nothrow(uma_ptr_t ptr, const TileObserver& addr, size_t idx) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_uacm_observer(uma_ptr_t ptr, const TileObserver& addr, size_t idx) noexcept -> exception_t{

        set_uacm_observer_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    }

    void set_uacm_observer_size_nothrow(uma_ptr_t ptr, size_t sz) noexcept{

    }

    auto set_uacm_observer_size(uma_ptr_t ptr, size_t sz) noexcept -> exception_t{

    } 

    void set_uacm_logit_nothrow(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_uacm_logit(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept -> exception_t{

        set_uacm_logit_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), addr);
    }

    void set_uacm_grad_nothrow(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_uacm_grad(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept -> exception_t{

        set_uacm_grad_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), addr);
    } 

    void set_uacm_signal_smph_addr_nothrow(uma_ptr_t ptr, std::optional<uma_ptr_t> smph_addr) noexcept{

    }

    auto set_uacm_signal_smph_addr(uma_ptr_t ptr, std::optional<uma_ptr_t> smph_addr) noexcept -> exception_t{

    }

    void set_uacm_memevent_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    auto set_uacm_memevent_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        set_uacm_operatable_id_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), operatable_id);
    }

    void set_uacm_forward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_uacm_forward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_uacm_backward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_uacm_backward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_uacm_forward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    auto set_uacm_forward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

        set_uacm_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), dispatch_control);
    }

    void set_uacm_backward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{
    
    }

    auto set_uacm_backward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

    }

    void set_uacm_grad_status_nothrow(uma_ptr_t ptr, grad_status_t grad_status) noexcept{

    }

    auto set_uacm_grad_status(uma_ptr_t ptr, grad_status_t grad_status) noexcept -> exception_t{

    } 

    void set_uacm_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        uma_ptr_t dst   = {};
        void * src      = &pong_count; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    }

    void set_uacm_pong_count(uma_ptr_t ptr, pong_count_t pong_count){

        set_uacm_pong_count_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), pong_count);
    }

    void set_uacm_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr, size_t idx) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::descendant_addr(ptr, std::integral_constant<size_t, ACM_IDX>{});
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_uacm_descendant(uma_ptr_t ptr, uma_ptr_t addr, size_t idx) noexcept -> exception_t{

        set_uacm_descendant_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), addr, std::integral_constant<size_t, ACM_IDX>{});
    } 

    //

    void set_pacm_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        uma_ptr_t dst   = {};
        void * src      = &init_status;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(init_status_t));
    }

    auto set_pacm_init_status(uma_ptr_t ptr, init_status_t init_status) noexcept -> exception_t{

        set_pacm_init_status_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), init_status);
    }

    void set_pacm_observer_nothrow(uma_ptr_t ptr, const TileObserver& addr, size_t idx) noexcept{ 

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t)); 
    }

    auto set_pacm_observer(uma_ptr_t ptr, const TileObserver& addr, size_t idx) noexcept -> exception_t{

        set_pacm_observer_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    }

    void set_pacm_observer_size_nothrow(uma_ptr_t ptr, size_t sz) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = arr.data();
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, 0u>{});
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, OBSERVER_ARR_SZ * sizeof(uma_ptr_t));
    }

    auto set_pacm_observer_size(uma_ptr_t ptr, size_t sz) noexcept -> exception_t{

        set_pacm_observer_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), arr);
    }

    void set_pacm_logit_nothrow(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_pacm_logit(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept -> exception_t{

        set_pacm_logit_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), addr);
    }

    void set_pacm_grad_nothrow(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_pacm_grad(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept -> exception_t{

        set_pacm_grad_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), addr);
    }

    void set_pacm_signal_smph_addr_nothrow(uma_ptr_t ptr, std::optional<uma_ptr_t>) noexcept{

    }

    auto set_pacm_signal_smph_addr(uma_ptr_t ptr, std::optional<uma_ptr_t>) noexcept -> exception_t{

    }

    void set_pacm_memevent_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    auto set_pacm_memevent_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        set_pacm_operatable_id_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), operatable_id);
    }

    void set_pacm_forward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_pacm_forward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_pacm_backward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_pacm_backward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_pacm_forward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };
        
        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    auto set_pacm_forward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

        set_pacm_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), dispatch_control);
    }

    void set_pacm_backward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

    }

    auto set_pacm_backward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

    }

    void set_pacm_grad_status_nothrow(uma_ptr_t ptr, grad_status_t grad_status) noexcept{

    }

    auto set_pacm_grad_status(uma_ptr_t ptr, grad_status_t grad_status) noexcept -> exception_t{

    } 

    void set_pacm_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        uma_ptr_t dst   = {};
        void * src      = &pong_count;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::pong_count_addr(ptr); 
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(pong_count_t));
    }

    auto set_pacm_pong_count(uma_ptr_t ptr, pong_count_t pong_count) noexcept -> exception_t{

        set_pacm_pong_count_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), pong_count); 
    }

    void set_pacm_left_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr, size_t idx) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::left_descendant_addr(ptr, std::integral_constant<size_t, ACM_IDX>{});
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_pacm_left_descendant(uma_ptr_t ptr, uma_ptr_t addr, size_t idx) noexcept -> exception_t{

        set_pacm_left_descendant_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), addr, std::integral_constant<size_t, ACM_IDX>{});
    } 

    void set_pacm_right_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr, size_t idx) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        
        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::right_descendant_addr(ptr, std::integral_constant<size_t, ACM_IDX>{});
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_pacm_right_descendant(uma_ptr_t ptr, uma_ptr_t addr, size_t idx) noexcept -> exception_t{

        set_pacm_right_descendant_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), addr, std::integral_constant<size_t, ACM_IDX>{});
    }
    
    //

    void set_crit_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        uma_ptr_t dst   = {};
        void * src      = &init_status;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(init_status_t));
    }

    auto set_crit_init_status(uma_ptr_t ptr, init_status_t init_status) noexcept -> exception_t{

        set_crit_init_status_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), init_status);
    }

    void set_crit_observer_nothrow(uma_ptr_t ptr, const TileObserver& observer, size_t idx) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{}); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_crit_observer(uma_ptr_t ptr, const TileObserver& observer, size_t idx) noexcept -> exception_t{

        set_crit_observer_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    } 

    void set_crit_observer_size_nothrow(uma_ptr_t ptr, size_t sz) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = arr.data();
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, 0u>{});
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, OBSERVER_ARR_SZ * sizeof(uma_ptr_t));
    }

    auto set_crit_observer_size(uma_ptr_t ptr, size_t sz) noexcept -> exception_t{

        set_crit_observer_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), arr);
    }

    void set_crit_logit_nothrow(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_crit_logit(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept -> exception_t{

        set_crit_logit_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), addr);
    }

    void set_crit_clogit_nothrow(uma_ptr_t ptr, void * clogit_addr, size_t clogit_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_clogit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_crit_clogit(uma_ptr_t ptr, void * clogit_addr, size_t clogit_bsz) noexcept -> exception_t{
        
        set_crit_clogit_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), addr);
    }

    void set_crit_grad_nothrow(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_crit_grad(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept -> exception_t{

        set_crit_grad_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), addr);
    }

    void set_crit_signal_smph_addr_nothrow(uma_ptr_t ptr, std::optional<uma_ptr_t> smph_addr) noexcept{

    }

    auto set_crit_signal_smph_addr(uma_ptr_t ptr, std::optional<uma_ptr_t> smph_addr) noexcept -> exception_t{

    }

    void set_crit_memevent_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    auto set_crit_memevent_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        set_crit_operatable_id_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), operatable_id);
    }

    void set_crit_forward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_crit_forward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_crit_backward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_crit_backward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_crit_forward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    auto set_crit_forward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

        set_crit_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), dispatch_control);
    }

    void set_crit_backward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

    }

    auto set_crit_backward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

    }

    void set_crit_grad_status_nothrow(uma_ptr_t ptr, grad_status_t grad_status) noexcept{

    }

    auto set_crit_grad_status(uma_ptr_t ptr, grad_status_t grad_status) noexcept -> exception_t{

    }

    void set_crit_learning_rate_nothrow(uma_ptr_t ptr, crit_learning_rate_t learning_rate) noexcept{

    }

    auto set_crit_learning_rate(uma_ptr_t ptr, crit_learning_rate_t learning_rate) noexcept -> exception_t{

    } 

    void set_crit_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::descendant_addr(ptr); 
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_crit_descendant(uma_ptr_t ptr, uma_ptr_t addr) noexcept -> exception_t{

        set_crit_descendant_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), addr);
    }

    //extnsrc (extnsrc compresses, extnsrx is the shadow of extnsrc, extndst decompresses)

    void set_extnsrc_init_status_nothrow(){}

    void set_extnsrc_observer_nothrow(){}

    void set_extnsrc_observer_size_nothrow(){}

    void set_extnsrc_counterpart_nothrow(){}

    void set_extnsrc_counterpart_shadow_nothrow(){}

    void set_extnsrc_request_retry_count_nothrow(){}

    void set_extnsrc_request_timeout_nothrow(){}

    void set_extnsrc_request_logid_nothrow(){}

    void set_extnsrc_logit_nothrow(){}

    void set_extnsrc_grad_nothrow(){}

    void set_extnsrc_signal_smph_addr_nothrow(){}

    void set_extnsrc_memevent_operatable_id_nothrow(){}

    void set_extnsrc_forward_operatable_id_nothrow(){}

    void set_extnsrc_backward_operatable_id_nothrow(){}

    void set_extnsrc_forward_dispatch_control_nothrow(){}

    void set_extnsrc_backward_dispatch_control_nothrow(){}

    void set_extnsrc_grad_status_nothrow(){}

    void set_extnsrc_descendant_nothrow(){}

    //extnsrx

    void set_extnsrx_init_status_nothrow(){}

    void set_extnsrx_logit_nothrow(){}

    // void set_extnsrx_signal_smph_addr_nothrow(){} 

    void set_extnsrx_memevent_operatable_id_nothrow(){}

    void set_extnsrx_forward_operatable_id_nothrow(){}

    // void set_extnsrx_counterpart_nothrow(){}

    //extndst

    void set_extndst_init_status_nothrow(){}

    void set_extndst_observer_nothrow(){}

    void set_extndst_observer_size_nothrow(){}

    void set_extndst_counterpart_nothrow(){}

    void set_extndst_counterpart_shadow_nothrow(){}

    void set_extndst_request_retry_count_nothrow(){}

    void set_extndst_request_timeout_nothrow(){}

    void set_extndst_logit_nothrow(){}

    void set_extndst_grad_nothrow(){}

    void set_extndst_signal_smph_addr_nothrow(){}

    void set_extndst_memevent_operatable_id_nothrow(){}

    void set_extndst_forward_operatable_id_nothrow(){}

    void set_extndst_backward_operatable_id_nothrow(){}

    void set_extndst_forward_dispatch_control_nothrow(){}

    void set_extndst_backward_dispatch_control_nothrow(){}

    void set_extndst_grad_status_nothrow(){}

    void set_extndst_descendant_nothrow(){}

    //extndsx, we have considered very carefully, the extndsx backprop must be unique, guaranteed by the SigaggSmph to make sure we are not overriding + over transmitting, though that is allowed
    //we just dont do grad accumulation @ extndsx, it's very messy

    void set_extndsx_init_status_nothrow(){}

    void set_extndsx_grad_nothrow(){}

    // void set_extndsx_signal_smph_addr_nothrow(){}

    void set_extndsx_memevent_operatable_id_nothrow(){}

    void set_extndsx_backward_operatable_id_nothrow(){}

    void set_extndsx_backward_dispatch_control_nothrow(){} 

    void set_extndsx_grad_status_nothrow(){} 

    void set_extndsx_descendant_nothrow(){}

    //--

    void set_msgrfwd_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        uma_ptr_t dst   = {};
        void * src      = &init_status;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(init_status_t));
    }

    auto set_msgrfwd_init_status(uma_ptr_t ptr, init_status_t init_status) noexcept -> exception_t{

        set_msgrfwd_init_status_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), init_status);
    }

    void set_msgrfwd_observer_nothrow(uma_ptr_t ptr, uma_ptr_t addr, size_t idx) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{}); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_msgrfwd_observer(uma_ptr_t ptr, uma_ptr_t addr, size_t idx) noexcept -> exception_t{

        set_msgrfwd_observer_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    }

    void set_msgrfwd_observer_size_nothrow(uma_ptr_t ptr, size_t sz) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = arr.data();
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, 0u>{});
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_uma::memcpy_host_to_uma_nothrow(dst, src, OBSERVER_ARR_SZ * sizeof(uma_ptr_t));
    }

    auto set_msgrfwd_observer_size(uma_ptr_t ptr, size_t sz) noexcept -> exception_t{

        set_msgrfwd_observer_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), arr);
    }

    void set_msgrfwd_logit_nothrow(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_msgrfwd_logit(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept -> exception_t{

        set_msgrfwd_logit_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), addr);
    }

    void set_msgrfwd_grad_nothrow(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    } 

    auto set_msgrfwd_grad(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept -> exception_t{

        set_msgrfwd_grad_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), addr);
    }

    void set_msgrfwd_signal_smph_addr_nothrow(uma_ptr_t ptr, std::optional<uma_ptr_t> smph_addr) noexcept{

    } 

    auto set_msgrfwd_signal_smph_addr(uma_ptr_t ptr, std::optional<uma_ptr_t> smph_addr) noexcept -> exception_t{

    }

    void set_msgrfwd_memevent_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };     

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    auto set_msgrfwd_memevent_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        set_msgrfwd_operatable_id_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), operatable_id);
    }

    void set_msgrfwd_forward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_msgrfwd_forward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    void set_msgrfwd_backward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_msgrfwd_backward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_msgrfwd_forward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    auto set_msgrfwd_forward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

        set_msgrfwd_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), dispatch_control);
    }

    void set_msgrfwd_backward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

    }

    auto set_msgrfwd_backward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

    } 

    void set_msgrfwd_grad_status_nothrow(uma_ptr_t ptr, grad_status_t grad_status) noexcept{

    }

    auto set_msgrfwd_grad_status(uma_ptr_t ptr, grad_status_t grad_status) noexcept -> exception_t{

    } 

    void set_msgrfwd_dst_info_nothrow(uma_ptr_t ptr, client_delivery_info_t dst_info) noexcept{

        static_assert(std::is_trivially_copyable_v<dst_info_t>);

        uma_ptr_t dst   = {};
        void * src      = &dst_info;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dst_info_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dst_info_t));
    }

    auto set_msgrfwd_dst_info(uma_ptr_t ptr, client_delivery_info_t dst_info) noexcept -> exception_t{

        set_msgrfwd_dst_info_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), dst_info);
    } 

    void set_msgrfwd_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::descendant_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_msgrfwd_descendant(uma_ptr_t ptr, uma_ptr_t addr) noexcept -> exception_t{

        set_msgrfwd_descendant_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), addr);
    }

    //

    void set_msgrbwd_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        uma_ptr_t dst   = {};
        void * src      = &init_status;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(init_status_t));
    }

    auto set_msgrbwd_init_status(uma_ptr_t ptr, init_status_t init_status) noexcept -> exception_t{

        set_msgrbwd_init_status_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), init_status);
    }

    void set_msgrbwd_observer_nothrow(uma_ptr_t ptr, uma_ptr_t addr, size_t idx) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_msgrbwd_observer(uma_ptr_t ptr, uma_ptr_t addr, size_t idx) noexcept -> exception_t{

        set_msgrbwd_observer_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    }

    void set_msgrbwd_observer_size_nothrow(uma_ptr_t ptr, size_t sz) noexcept{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = arr.data();
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, 0u>{});
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, OBSERVER_ARR_SZ * sizeof(uma_ptr_t));
    }

    auto set_msgrbwd_observer_size(uma_ptr_t ptr, size_t sz) noexcept -> exception_t{

        set_msgrbwd_observer_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), arr);
    }

    void set_msgrbwd_logit_nothrow(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_msgrbwd_logit(uma_ptr_t ptr, void * logit_addr, size_t logit_bsz) noexcept -> exception_t{

        set_msgrbwd_logit_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), addr);
    }

    void set_msgrbwd_grad_nothrow(uma_ptr_t ptr, void * grad_addr, size_t grad_bsz) noexcept{

        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    auto set_msgrbwd_grad(uma_ptr_t dst, void * grad_addr, size_t grad_bsz) noexcept -> exception_t{

        set_msgrbwd_grad_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(dst), addr);
    } 

    void set_msgrbwd_signal_smph_addr_nothrow(uma_ptr_t dst, std::optional<uma_ptr_t> smph_addr) noexcept{

    }

    auto set_msgrbwd_signal_smph_addr(uma_ptr_t dst, std::optional<uma_ptr_t> smph_addr) noexcept -> exception_t{

    }

    void set_msgrbwd_memevent_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        uma_ptr_t dst   = {};
        void * src      = &operatable_id;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(operatable_id_t));
    }

    auto set_msgrbwd_memevent_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

        set_msgrbwd_operatable_id_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), operatable_id);
    }

    void set_msgrbwd_forward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

    }

    auto set_msgrbwd_forward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_msgrbwd_backward_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{
        
    }

    auto set_msgrbwd_backward_operatable_id(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept -> exception_t{

    }

    void set_msgrbwd_forward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    auto set_msgrbwd_forward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

        set_msgrbwd_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), dispatch_control);
    }

    void set_msgrbwd_backward_dispatch_control_nothrow(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept{

    }

    auto set_msgrbwd_backward_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control) noexcept -> exception_t{

    }

    void set_msgrbwd_grad_status_nothrow(uma_ptr_t ptr, grad_status_t grad_status) noexcept{

    }

    auto set_msgrbwd_grad_status(uma_ptr_t ptr, grad_status_t grad_status) noexcept -> exception_t{

    }

    void set_msgrbwd_is_delivered_nothrow(uma_ptr_t ptr, bool is_delivered) noexcept{

    }

    auto set_msgrbwd_is_delivered(uma_ptr_t ptr, bool is_delivered) noexcept -> exception_t{

    }

    void set_msgrbwd_dst_info_nothrow(uma_ptr_t ptr, client_delivery_info_t dst_info) noexcept{

        static_assert(std::is_trivially_copyable_v<dst_info_t>);

        uma_ptr_t dst   = {};
        void * src      = &dst_info;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dst_info_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dst_info_t));
    }

    void set_msgrbwd_dst_info(uma_ptr_t ptr, client_delivery_info_t dst_info){

        set_msgrbwd_dst_info_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), dst_info);
    } 

    void set_msgrbwd_descendant_nothrow(uma_ptr_t ptr, uma_ptr_t addr) noexcept{
        
        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::descendant_addr(ptr); 
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    auto set_msgrbwd_descendant(uma_ptr_t ptr, uma_ptr_t addr) noexcept -> exception_t{

        set_msgrbwd_descendant_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), src);
    } 

    //

    void set_tile_init_status_nothrow(uma_ptr_t ptr, init_status_t init_status) noexcept{

        using namespace dg::network_tile_member_access;

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        const auto id   = dg_typeid(safe_tile_ptr_access(ptr));
        uma_ptr_t dst   = {};
        void * src      = &init_status;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::init_status_addr(ptr);
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(init_status_t));
    }

    void set_tile_init_status(uma_ptr_t ptr, init_status_t init_status){

        set_tile_init_status_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), init_status);
    }

    template <size_t ARR_IDX>
    void set_tile_observer_nothrow(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>) noexcept{

        using namespace dg::network_tile_member_access;
    
        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
      
        const auto id   = dg_typeid(safe_tile_ptr_access(ptr));
        uma_ptr_t dst   = {};
        void * src      = &addr;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(uma_ptr_t));
    }

    template <size_t ARR_IDX>
    void set_tile_observer(uma_ptr_t ptr, uma_ptr_t addr, const std::integral_constant<size_t, ARR_IDX>){

        set_tile_observer_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), addr, std::integral_constant<size_t, ARR_IDX>{});
    }

    void set_tile_observer_nothrow(uma_ptr_t ptr, std::array<uma_ptr_t, OBSERVER_ARR_SZ> arr) noexcept{

        using namespace dg::network_tile_member_access;
    
        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
      
        const auto id   = dg_typeid(safe_tile_ptr_access(ptr));
        uma_ptr_t dst   = {};
        void * src      = arr.data();
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::observer_addr(ptr, std::integral_constant<size_t, 0u>{});
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, OBSERVER_ARR_SZ * sizeof(uma_ptr_t));
    }

    void set_tile_observer(uma_ptr_t ptr, std::array<uma_ptr_t, OBSERVER_ARR_SZ> arr){

        set_tile_observer_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), arr);
    }

    void set_tile_logit_nothrow(uma_ptr_t ptr, void * addr) noexcept{

        using namespace dg::network_tile_member_access; 

        const auto id   = dg_typeid(safe_tile_ptr_access(ptr));
        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    void set_tile_logit(uma_ptr_t ptr, void * addr){

        set_tile_logit_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), addr);
    }

    void set_tile_grad_nothrow(uma_ptr_t ptr, void * addr) noexcept{

        using namespace dg::network_tile_member_access;

        const auto id   = dg_typeid(safe_tile_ptr_access(ptr));
        uma_ptr_t dst   = {};
        void * src      = addr;
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, cpy_sz);
    }

    void set_tile_grad(uma_ptr_t ptr, void * addr){

        set_tile_grad_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), addr);
    }

    void set_tile_operatable_id_nothrow(uma_ptr_t ptr, operatable_id_t operatable_id) noexcept{

        using namespace dg::network_tile_member_access;

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        const auto id   = dg_typeid(safe_tile_ptr_access(ptr));
        uma_ptr_t dst   = {};
        void * src      = &operatable_id; 
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::operatable_id_addr(ptr);
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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
       
        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        const auto id   = dg_typeid(safe_tile_ptr_access(ptr));
        uma_ptr_t dst   = {};
        void * src      = &dispatch_control;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::dispatch_control_addr(ptr);
        };

        if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_host_to_uma_nothrow(dst, src, sizeof(dispatch_control_t));
    }

    void set_tile_dispatch_control(uma_ptr_t ptr, dispatch_control_t dispatch_control){

        // set_tile_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), dispatch_control);
    }

    void set_tile_pong_count_nothrow(uma_ptr_t ptr, pong_count_t pong_count) noexcept{

        using namespace dg::network_tile_member_access; 
       
        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        const auto id   = dg_typeid(safe_tile_ptr_access(ptr));
        uma_ptr_t dst   = {};
        void * src      = &pong_count;
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            dst = Accessor::pong_count_addr(ptr);
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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
    
    //

    auto get_leaf_rcu_addr_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        uma_ptr_t rs    = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            rs = Accessor::rcu_lock_addr(ptr);
        };
        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        
        return rs;
    }

    auto get_leaf_rcu_addr(uma_ptr_t ptr) -> uma_ptr_t{

        return get_leaf_rcu_addr_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr));
    }

    auto get_leaf_init_status_nothrow(uma_ptr_t ptr) noexcept -> init_status_t{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        auto rs         = init_status_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(init_status_t));

        return rs;
    }

    auto get_leaf_init_status(uma_ptr_t ptr) -> init_status_t{

        return get_leaf_init_status_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr));
    }

    template <size_t ARR_IDX>
    auto get_leaf_observer_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    template <size_t ARR_IDX>
    auto get_leaf_observer(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) -> uma_ptr_t{

        return get_leaf_observer_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr), std::integral_constant<size_t, ARR_IDX>{});
    }

    void get_leaf_logit_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_leaf_logit(uma_ptr_t ptr, void * dst){

        get_leaf_logit_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr), dst);
    }

    void get_leaf_grad_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_leaf_grad(uma_ptr_t ptr, void * dst){

        get_leaf_grad_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr), dst);
    }

    auto get_leaf_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        
        auto rs         = operatable_id_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));

        return rs;
    }

    auto get_leaf_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return get_leaf_operatable_id_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr));
    } 
    
    auto get_leaf_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto rs         = dispatch_control_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));

        return rs;
    }

    auto get_leaf_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return get_leaf_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr));
    }

    auto get_leaf_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto rs         = pong_count_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_leaf_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_leaf_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));

        return rs;
    }

    auto get_leaf_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return get_leaf_pong_count_nothrow(dg::network_tile_member_access::safethrow_leaf_ptr_access(ptr));
    }

    //

    auto get_mono_rcu_addr_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        uma_ptr_t rs    = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            rs = Accessor::rcu_lock_addr(ptr);
        };
        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        
        return rs;
    }

    auto get_mono_rcu_addr(uma_ptr_t ptr) -> uma_ptr_t{

        return get_mono_rcu_addr_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    auto get_mono_init_status_nothrow(uma_ptr_t ptr) noexcept -> init_status_t{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        auto rs         = init_status_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(init_status_t));

        return rs;
    }

    auto get_mono_init_status(uma_ptr_t ptr) -> init_status_t{

        return get_mono_init_status_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    template <size_t ARR_IDX>
    auto get_mono_observer_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    template <size_t ARR_IDX>
    auto get_mono_observer(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

        return get_mono_observer_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), std::integral_constant<size_t, ARR_IDX>{});
    } 

    void get_mono_logit_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);

    }

    void get_mono_logit(uma_ptr_t ptr, void * dst){

        get_mono_logit_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), dst);
    }

    void get_mono_grad_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_mono_grad(uma_ptr_t ptr, void * dst){

        get_mono_grad_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr), dst);
    }

    auto get_mono_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        
        auto rs         = operatable_id_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));

        return rs;
    }
    
    auto get_mono_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return get_mono_operatable_id_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    auto get_mono_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
     
        auto rs         = dispatch_control_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));

        return rs;
    }

    auto get_mono_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return get_mono_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    auto get_mono_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        
        auto rs         = pong_count_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::pong_count_addr(ptr); 
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));

        return rs;
    }

    auto get_mono_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return get_mono_pong_count_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }

    auto get_mono_descendant_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_mono_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_mono_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    auto get_mono_descendant(uma_ptr_t ptr) -> uma_ptr_t{

        return get_mono_descendant_nothrow(dg::network_tile_member_access::safethrow_mono_ptr_access(ptr));
    }
        
    //
    
    auto get_pair_rcu_addr_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        uma_ptr_t rs    = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            rs = Accessor::rcu_lock_addr(ptr);
        };
        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        
        return rs;
    }

    auto get_pair_rcu_addr(uma_ptr_t ptr) -> uma_ptr_t{

        return get_pair_rcu_addr_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    }

    auto get_pair_init_status_nothrow(uma_ptr_t ptr) noexcept -> init_status_t{

        static_assert(std::is_trivially_copyable_v<init_status_t>);
        
        auto rs         = init_status_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(init_status_t));

        return rs;
    }

    auto get_pair_init_status(uma_ptr_t ptr) -> init_status_t{

        return get_pair_init_status_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    }

    template <size_t ARR_IDX>
    auto get_pair_observer_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    template <size_t ARR_IDX>
    auto get_pair_observer(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) -> uma_ptr_t{

        return get_pair_observer_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), std::integral_constant<size_t, ARR_IDX>{});
    }

    void get_pair_logit_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host(dst, src, cpy_sz);
    }

    void get_pair_logit(uma_ptr_t ptr, void * dst){

        get_pair_logit_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), dst);
    }

    void get_pair_grad_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host(dst, src, cpy_sz);
    }

    void get_pair_grad(uma_ptr_t ptr, void * dst){

        get_pair_grad_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr), dst);
    }

    auto get_pair_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        
        auto rs         = operatable_id_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));

        return rs;
    }

    auto get_pair_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return get_pair_operatable_id_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    }    

    auto get_pair_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto rs         = dispatch_control_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));

        return rs;
    }

    auto get_pair_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return get_pair_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    }

    auto get_pair_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        
        auto rs         = pong_count_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));

        return rs;
    }

    auto get_pair_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return get_pair_pong_count_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    }

    auto get_pair_left_descendant_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        
        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::left_descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    auto get_pair_left_descendant(uma_ptr_t ptr) -> uma_ptr_t{

        return get_pair_left_descendant_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    } 

    auto get_pair_right_descendant_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::right_descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_pair_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pair_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    auto get_pair_right_descendant(uma_ptr_t ptr) -> uma_ptr_t{

        return get_pair_right_descendant_nothrow(dg::network_tile_member_access::safethrow_pair_ptr_access(ptr));
    }

    //

    auto get_uacm_rcu_addr_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        uma_ptr_t rs    = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            rs = Accessor::rcu_lock_addr(ptr);
        };
        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));

        return rs;
    }

    auto get_uacm_rcu_addr(uma_ptr_t ptr) -> uma_ptr_t{

        return get_uacm_rcu_addr_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    }

    auto get_uacm_init_status_nothrow(uma_ptr_t ptr) noexcept -> init_status_t{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        auto rs         = init_status_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(init_status_t));

        return rs;
    }

    auto get_uacm_init_status(uma_ptr_t ptr) -> init_status_t{

        return get_uacm_init_status_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    }

    template <size_t ARR_IDX>
    auto get_uacm_observer_addr_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    template <size_t ARR_IDX>
    auto get_uacm_observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) -> uma_ptr_t{

        return get_uacm_observer_addr_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), std::integral_constant<size_t, ARR_IDX>{});
    }

    void get_uacm_logit_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_uacm_logit(uma_ptr_t ptr, void * dst){

        get_uacm_logit_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), dst);
    }

    void get_uacm_grad_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_uacm_grad(uma_ptr_t ptr, void * dst){

        get_uacm_grad_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), dst);
    }

    auto get_uacm_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        
        auto rs         = operatable_id_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));

        return rs;
    }

    auto get_uacm_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return get_uacm_operatable_id_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    }

    auto get_uacm_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);
        
        auto rs         = dispatch_control_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));

        return rs;
    }

    auto get_uacm_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return get_uacm_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    }

    auto get_uacm_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        
        auto rs         = pong_count_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));

        return rs;
    }

    auto get_uacm_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return get_uacm_pong_count_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr));
    }

    template <size_t ACM_IDX>
    auto get_uacm_descendant_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::descendant_addr(ptr, std::integral_constant<size_t, ACM_IDX>{});
        };

        dg::network_tile_member_access::get_uacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_uacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    template <size_t ACM_IDX>
    auto get_uacm_descendant(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) -> uma_ptr_t{

        return get_uacm_descendant_nothrow(dg::network_tile_member_access::safethrow_uacm_ptr_access(ptr), std::integral_constant<size_t, ACM_IDX>{});
    }
    
    //

    auto get_pacm_rcu_addr_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        uma_ptr_t rs    = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            rs = Accessor::rcu_lock_addr(ptr);
        };
        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));

        return rs;
    }

    auto get_pacm_rcu_addr(uma_ptr_t ptr) -> uma_ptr_t{

        return get_pacm_rcu_addr_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    }

    auto get_pacm_init_status_nothrow(uma_ptr_t ptr) noexcept -> init_status_t{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        auto rs         = init_status_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(init_status_t));

        return rs;
    }

    auto get_pacm_init_status(uma_ptr_t ptr) -> init_status_t{

        return get_pacm_init_status_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    }

    template <size_t ACM_IDX>
    auto get_pacm_observer_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::observer_addr(ptr, std::integral_constant<size_t, ACM_IDX>{});
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    template <size_t ACM_IDX>
    auto get_pacm_observer(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) -> uma_ptr_t{

        return get_pacm_observer_nothrow(ptr, std::integral_constant<size_t, ACM_IDX>{});
    }

    void get_pacm_logit_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_pacm_logit(uma_ptr_t ptr, void * dst){

        get_pacm_logit_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), dst);
    }

    void get_pacm_grad_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_pacm_grad(uma_ptr_t ptr, void * dst){
        
        get_pacm_grad_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), dst);
    }

    auto get_pacm_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        
        auto rs         = operatable_id_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));

        return rs;
    }

    auto get_pacm_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return get_pacm_operatable_id_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    }

    auto get_pacm_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto rs         = dispatch_control_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));

        return rs;
    }

    auto get_pacm_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return get_pacm_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    }

    auto get_pacm_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        
        auto rs         = pong_count_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));

        return rs;
    }

    auto get_pacm_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return get_pacm_pong_count_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr));
    }

    template <size_t ACM_IDX>
    auto get_pacm_left_descendant_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::left_descendant_addr(ptr, std::integral_constant<size_t, ACM_IDX>{});
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    } 

    template <size_t ACM_IDX>
    auto get_pacm_left_descendant(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) -> uma_ptr_t{

        return get_pacm_left_descendant_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), std::integral_constant<size_t, ACM_IDX>{});
    } 

    template <size_t ACM_IDX>
    auto get_pacm_right_descendant_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::right_descendant_addr(ptr, std::integral_constant<size_t, ACM_IDX>{});
        };

        dg::network_tile_member_access::get_pacm_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_pacm_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    } 

    template <size_t ACM_IDX>
    auto get_pacm_right_descendant(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) -> uma_ptr_t{

        return get_pacm_right_descendant_nothrow(dg::network_tile_member_access::safethrow_pacm_ptr_access(ptr), std::integral_constant<size_t, ACM_IDX>{});
    }
 
    //

    auto get_crit_rcu_addr_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        uma_ptr_t rs    = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            rs = Accessor::rcu_lock_addr(ptr);
        };
        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));

        return rs;
    }

    auto get_crit_rcu_addr(uma_ptr_t ptr) -> uma_ptr_t{

        return get_crit_rcu_addr_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    }

    auto get_crit_init_status_nothrow(uma_ptr_t ptr) noexcept -> init_status_t{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        auto rs         = init_status_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(init_status_t));

        return rs;
    }

    auto get_crit_init_status(uma_ptr_t ptr) -> init_status_t{

        return get_crit_init_status_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    }

    template <size_t ARR_IDX>
    auto get_crit_observer_addr_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    template <size_t ARR_IDX>
    auto get_crit_observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) -> uma_ptr_t{

        return get_crit_observer_addr_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), std::integral_constant<size_t, ARR_IDX>{});
    }

    void get_crit_logit_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_crit_logit(uma_ptr_t ptr, void * dst){

        get_crit_logit_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), dst);
    }

    void get_crit_clogit_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_clogit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_crit_clogit(uma_ptr_t ptr, void * dst){

        get_crit_clogit_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), dst);
    }

    void get_crit_grad_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_crit_grad(uma_ptr_t ptr, void * dst){

        get_crit_grad_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr), dst);
    }

    auto get_crit_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);
        
        auto rs         = operatable_id_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));

        return rs;
    }

    auto get_crit_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return get_crit_operatable_id_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    }

    auto get_crit_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto rs         = dispatch_control_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));

        return rs;
    }

    auto get_crit_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return get_crit_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    }

    auto get_crit_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);
        
        auto rs         = pong_count_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));

        return rs;
    }

    auto get_crit_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return get_crit_pong_count_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    }

    auto get_crit_descendant_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_crit_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_crit_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    auto get_crit_descendant(uma_ptr_t ptr) -> uma_ptr_t{

        return get_crit_descendant_nothrow(dg::network_tile_member_access::safethrow_crit_ptr_access(ptr));
    } 

    //

    auto get_msgrfwd_rcu_addr_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        uma_ptr_t rs    = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            rs = Accessor::rcu_lock_addr(ptr);
        };
        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        
        return rs;
    }

    auto get_msgrfwd_rcu_addr(uma_ptr_t ptr) -> uma_ptr_t{

        return get_msgrfwd_rcu_addr_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    }

    auto get_msgrfwd_init_status_nothrow(uma_ptr_t ptr) noexcept -> init_status_t{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        auto rs         = init_status_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(init_status_t));

        return rs;
    }

    auto get_msgrfwd_init_status(uma_ptr_t ptr) -> init_status_t{

        return get_msgrfwd_init_status_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    }

    template <size_t ARR_IDX>
    auto get_msgrfwd_observer_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    template <size_t ARR_IDX>
    auto get_msgrfwd_observer(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) -> uma_ptr_t{

        return get_msgrfwd_observer_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), std::integral_constant<size_t, ARR_IDX>{});
    }

    void get_msgrfwd_logit_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_msgrfwd_logit(uma_ptr_t ptr, void * dst){
        
        get_msgrfwd_logit_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), dst);
    }

    void get_msgrfwd_grad_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_msgrfwd_grad(uma_ptr_t ptr, void * dst){

        get_msgrfwd_grad_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr), dst);
    }

    auto get_msgrfwd_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto rs         = operatable_id_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::operatable_id_addr(ptr);
        };     

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));

        return rs;
    }

    auto get_msgrfwd_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return get_msgrfwd_operatable_id_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    }

    auto get_msgrfwd_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto rs         = dispatch_control_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, ptr);
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));

        return rs;
    }

    auto get_msgrfwd_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return get_msgrfwd_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    } 

    auto get_msgrfwd_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto rs         = pong_count_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));

        return rs;
    }

    auto get_msgrfwd_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return get_msgrfwd_pong_count_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    }

    auto get_msgrfwd_descendant_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        
        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    auto get_msgrfwd_descendant(uma_ptr_t ptr) -> uma_ptr_t{

        return get_msgrfwd_descendant_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    }

    auto get_msgrfwd_dst_info_nothrow(uma_ptr_t ptr) noexcept -> dst_info_t{

        static_assert(std::is_trivially_copyable_v<dst_info_t>);

        auto rs         = dst_info_t{};
        void * dst      = {};
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dst_info_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrfwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrfwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dst_info_t));

        return rs;
    }

    auto get_msgrfwd_dst_info(uma_ptr_t ptr) -> dst_info_t{

        return get_msgrfwd_dst_info_nothrow(dg::network_tile_member_access::safethrow_msgrfwd_ptr_access(ptr));
    }

    //

    auto get_msgrbwd_rcu_addr_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        uma_ptr_t rs    = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            rs = Accessor::rcu_lock_addr(ptr);
        };
        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        
        return rs;
    }

    auto get_msgrbwd_rcu_addr(uma_ptr_t ptr) -> uma_ptr_t{

        return get_msgrbwd_rcu_addr_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    auto get_msgrbwd_init_status_nothrow(uma_ptr_t ptr) noexcept -> init_status_t{

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        auto rs         = init_status_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::init_status_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(init_status_t));

        return rs;
    }

    auto get_msgrbwd_init_status(uma_ptr_t ptr) -> init_status_t{

        return get_msgrbwd_init_status_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    template <size_t ARR_IDX>
    auto get_msgrbwd_observer_addr_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);
        
        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    template <size_t ARR_IDX>
    auto get_msgrbwd_observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) -> uma_ptr_t{

        return get_msgrbwd_observer_addr_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), std::integral_constant<size_t, ARR_IDX>{});
    } 

    void get_msgrbwd_logit_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_msgrbwd_logit(uma_ptr_t ptr, void * dst){

        get_msgrbwd_logit_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), dst);
    }

    void get_msgrbwd_grad_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_msgrbwd_grad(uma_ptr_t ptr, void * dst){

        get_msgrbwd_grad_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr), dst);
    }

    auto get_msgrbwd_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        auto rs         = operatable_id_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::operatable_id_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));

        return rs;
    }

    auto get_msgrbwd_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return get_msgrbwd_operatable_id_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    auto get_msgrbwd_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        auto rs         = dispatch_control_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dispatch_control_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));

        return rs;
    }

    auto get_msgrbwd_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return get_msgrbwd_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    auto get_msgrbwd_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        auto rs         = pong_count_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::pong_count_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));

        return rs;
    }

    auto get_msgrbwd_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return get_msgrbwd_pong_count_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    auto get_msgrbwd_descendant_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::descendant_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));

        return rs;
    }

    auto get_msgrbwd_descendant(uma_ptr_t ptr) -> uma_ptr_t{

        return get_msgrbwd_descendant_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    auto get_msgrbwd_dst_info_nothrow(uma_ptr_t ptr) noexcept -> dst_info_t{

        static_assert(std::is_trivially_copyable_v<dst_info_t>);

        auto rs         = dst_info_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dst_info_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dst_info_t));

        return rs;
    }

    auto get_msgrbwd_dst_info(uma_ptr_t ptr) -> dst_info_t{

        return get_msgrbwd_dst_info_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    }

    auto get_msgrbwd_timein_nothrow(uma_ptr_t ptr) noexcept -> timein_t{

        static_assert(std::is_trivially_copyable_v<timein_t>);
        
        auto rs         = timein_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::timein_addr(ptr);
        };

        dg::network_tile_member_access::get_msgrbwd_static_polymorphic_accessor(cb_lambda, dg::network_tile_member_access::safe_msgrbwd_ptr_access(ptr));
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(timein_t));

        return rs;
    }

    auto get_msgrbwd_timein(uma_ptr_t ptr) -> timein_t{

        return get_msgrbwd_timein_nothrow(dg::network_tile_member_access::safethrow_msgrbwd_ptr_access(ptr));
    } 

    //

    auto get_tile_rcu_addr_nothrow(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        using namespace dg::network_tile_member_access;

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        const auto id   = dg_typeid(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
        auto rs         = uma_ptr_t{};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            rs = Accessor::rcu_lock_addr(ptr);
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        return rs;        
    }

    auto get_tile_rcu_addr(uma_ptr_t ptr) -> uma_ptr_t{

        return get_tile_rcu_addr_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    }

    auto get_tile_init_status_nothrow(uma_ptr_t ptr) noexcept -> init_status_t{
        
        using namespace dg::network_tile_member_access;

        static_assert(std::is_trivially_copyable_v<init_status_t>);

        const auto id   = dg_typeid(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
        auto rs         = init_status_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::init_status_addr(ptr);
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(init_status_t));
        return rs;        
    }

    auto get_tile_init_status(uma_ptr_t ptr) -> init_status_t{

        return get_tile_init_status_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    }

    template <size_t ARR_IDX>
    auto get_tile_observer_nothrow(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{
        
        using namespace dg::network_tile_member_access;

        static_assert(std::is_trivially_copyable_v<uma_ptr_t>);

        const auto id   = dg_typeid(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
        auto rs         = uma_ptr_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::observer_addr(ptr, std::integral_constant<size_t, ARR_IDX>{});
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(uma_ptr_t));
        return rs;
    }

    template <size_t ARR_IDX>
    auto get_tile_observer(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) -> uma_ptr_t{

        return get_tile_observer_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), std::integral_constant<size_t, ARR_IDX>{});
    }

    void get_tile_logit_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        using namespace dg::network_tile_member_access; 

        const auto id   = dg_typeid(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_logit_addr(ptr);
            cpy_sz  = Accessor::logit_group_size();
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_tile_logit(uma_ptr_t ptr, void * dst){

        get_tile_logit_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), dst);
    }

    void get_tile_grad_nothrow(uma_ptr_t ptr, void * dst) noexcept{

        using namespace dg::network_tile_member_access;

        const auto id   = dg_typeid(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
        uma_ptr_t src   = {};
        size_t cpy_sz   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src     = Accessor::tile_grad_addr(ptr);
            cpy_sz  = Accessor::grad_group_size();
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, cpy_sz);
    }

    void get_tile_grad(uma_ptr_t ptr, void * dst){

        get_tile_grad_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr), dst);
    }

    auto get_tile_operatable_id_nothrow(uma_ptr_t ptr) noexcept -> operatable_id_t{

        using namespace dg::network_tile_member_access;
        
        static_assert(std::is_trivially_copyable_v<operatable_id_t>);

        const auto id   = dg_typeid(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
        auto rs         = operatable_id_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::operatable_id_addr(ptr);
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(operatable_id_t));
        return rs;
    }

    auto get_tile_operatable_id(uma_ptr_t ptr) -> operatable_id_t{

        return get_tile_operatable_id_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    } 
    
    auto get_tile_dispatch_control_nothrow(uma_ptr_t ptr) noexcept -> dispatch_control_t{

        using namespace dg::network_tile_member_access;

        static_assert(std::is_trivially_copyable_v<dispatch_control_t>);

        const auto id   = dg_typeid(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
        auto rs         = dispatch_control_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::dispatch_control_addr(ptr);
        };

        if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(dispatch_control_t));
        return rs;
    }

    auto get_tile_dispatch_control(uma_ptr_t ptr) -> dispatch_control_t{

        return get_tile_dispatch_control_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    }

    auto get_tile_pong_count_nothrow(uma_ptr_t ptr) noexcept -> pong_count_t{

        using namespace dg::network_tile_member_access;
        
        static_assert(std::is_trivially_copyable_v<pong_count_t>);

        const auto id   = dg_typeid(dg::network_tile_member_access::safe_tile_ptr_access(ptr));
        auto rs         = pong_count_t{};
        void * dst      = &rs;
        uma_ptr_t src   = {};
        auto cb_lambda  = [&]<class Accessor>(const Accessor) noexcept{
            src = Accessor::pong_count_addr(ptr);
        };

        if (is_leaf_tile(id)){
            get_leaf_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_mono_tile(id)){
            get_mono_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_uacm_tile(id)){
            get_uacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pacm_tile(id)){
            get_pacm_static_polymorphic_accessor(cb_lambda, ptr, id);
        } else if (is_pair_tile(id)){
            get_pair_static_polymorphic_accessor(cb_lambda, ptr, id);
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

        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(pong_count_t));
        return rs;
    }

    auto get_tile_pong_count(uma_ptr_t ptr) -> pong_count_t{

        return get_tile_pong_count_nothrow(dg::network_tile_member_access::safethrow_tile_ptr_access(ptr));
    }
}

#endif