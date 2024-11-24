#ifndef __NETWORK_MEMOPS_UMA_H__
#define __NETWORK_MEMOPS_UMA_H__

#include "network_memlock.h"
#include "network_uma.h"
#include "network_memops.h"
#include "network_exception_handler.h"
#include <atomic>
#include "stdx.h"
#include "network_pointer.h"

namespace dg::network_memops_uma{

    struct signature_dg_network_memops_uma; 

    using uma_ptr_t         = dg::network_pointer::uma_ptr_t;
    using uma_lock_instance = dg::network_memlock_impl1::Lock<signature_dg_network_memops_uma, std::integral_constant<size_t, dg::network_pointer::MEMREGION_SZ>, uma_ptr_t>; 

    void init(uma_ptr_t first, uma_ptr_t last){

        stdx::memtransaction_guard grd;
        uma_lock_instance::init(first, last);
    }

    void deinit() noexcept{

        stdx::memtransaction_guard grd;
        uma_lock_instance::deinit();
    }

    template <class ...Args>
    class memlock_guard{

        private:

            decltype(dg::network_memlock::recursive_lock_guard_many(uma_lock_instance{}, std::declval<Args>()...)) resource;
        
        public:

            using self = memlock_guard;

            inline __attribute__((always_inline)) memlock_guard(Args ...args) noexcept{

                std::atomic_signal_fence(std::memory_order_acquire);
                this->resource = dg::network_memlock::recursive_lock_guard_many(uma_lock_instance{}, args...);
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }

            inline __attribute__((always_inline)) ~memlock_guard() noexcept{

                std::atomic_thread_fence(std::memory_order_seq_cst);
            }

            memlock_guard(const self&) = delete;
            memlock_guard(self&&) = delete;

            memlock_guard& operator =(const self&) = delete;
            memlock_guard& operator =(self&&) = delete;
    };

    auto memcpy_uma_to_vma(vma_ptr_t dst, uma_ptr_t src, size_t n) noexcept -> exception_t{

        auto src_map_rs = dg::network_uma::map_wait_safe(src);

        if (!src_map_rs.has_value()){
            return src_map_rs.error();
        }

        vma_ptr_t src_vptr  = dg::network_uma::get_vma_const_ptr(src_map_rs.value().value());        
        return dg::network_memops_virt::memcpy(dst, src_vptr, n);
    }

    void memcpy_uma_to_vma_nothrow(vma_ptr_t dst, uma_ptr_t src, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_uma_to_vma(dst, src, n));
    }

    auto memcpy_vma_to_uma(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept -> exception_t{

        auto dst_map_rs = dg::network_uma::map_wait_safe(dst);

        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        vma_ptr_t dst_vptr  = dg::network_uma::get_vma_ptr(dst_map_rs.value().value());
        return dg::network_memops_virt::memcpy(dst_vptr, src, n);
    }

    void memcpy_vma_to_uma_nothrow(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_vma_to_uma(dst, src, n));
    }

    //this function does not guarantee atomicity - such that a failed operation leave the writing uma_ptr_t in an undefined state 
    auto memcpy_vma_to_uma_directall(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept -> exception_t{

        std::expected<size_t, exception_t> device_range = dg::network_uma::device_count(dst);

        if (!device_range.has_value()){
            return device_range.error();
        } 

        for (size_t i = 0u; i < device_range.value(); ++i){
            std::expected<device_id_t, exception_t> device_id = dg::network_uma::device_at(dst, i);

            if (!device_id.has_value()){
                return device_id.error();
            }

            std::expected<vma_ptr_t, exception_t> dst_vptr = dg::network_uma::map_direct(device_id.value(), dst);

            if (!dst_vptr.has_value()){
                return dst_vptr.error();
            }

            exception_t memcpy_err = dg::network_memops_virt::memcpy(dst_vptr.value(), src, n);

            if (dg::network_exception::is_failed(memcpy_err)){
                return memcpy_err;
            }
        }

        return dg::network_exception::SUCCESS;
    }

    void memcpy_vma_to_uma_directall_nothrow(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept{

        size_t device_range = dg::network_uma::device_count_nothrow(dst);

        for (size_t i = 0u; i < device_range; ++i){
            device_id_t id      = dg::network_uma::device_at_nothrow(dst, i);
            vma_ptr_t dst_vptr  = dg::network_uma::map_direct_nothrow(id, dst);
            dg::network_memops_virt::memcpy_nothrow(dst_vptr, src, n);
        }
    } 

    auto memset(uma_ptr_t dst, int c, size_t n) noexcept -> exception_t{

        auto dst_map_rs = dg::network_uma::map_wait_safe(dst);
        
        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        vma_ptr_t dst_vptr  = dg::network_uma::get_vma_ptr(dst_map_rs.value().value());
        return dg::network_memops_virt::memset(dst_vptr, c, n);
    }

    void memset_nothrow(uma_ptr_t dst, int c, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memset(dst, c, n));
    }

    //this function does not guarantee atomicity - such that a failed operation leave the writing uma_ptr_t in an undefined state 
    auto memset_directall(uma_ptr_t dst, int c, size_t n) noexcept -> exception_t{
        
        std::expected<size_t, exception_t> device_range = dg::network_uma::device_count(dst);

        if (!device_range.has_value()){
            return device_range.error();
        }

        for (size_t i = 0u; i < device_range.value(); ++i){
            std::expected<device_id_t, exception_t> device_id = dg::network_uma::device_at(dst, i);

            if (!device_id.has_value()){
                return device_id.error();
            }

            std::expected<vma_ptr_t, exception_t> dst_vptr = dg::network_uma::map_direct(device_id.value(), dst);

            if (!dst_vptr.has_value()){
                return dst_vptr.error();
            } 

            exception_t memset_err = dg::network_memops_virt::memset(dst_vptr.value(), c, n);

            if (dg::network_exception::is_failed(memset_err)){
                return memset_err;
            }
        }

        return dg::network_exception::SUCCESS;
    }

    void memset_directall_nothrow(uma_ptr_t dst, int c, size_t n) noexcept{

        size_t device_range = dg::network_uma::device_count_nothrow(dst);

        for (size_t i = 0u; i < device_range; ++i){
            device_id_t id      = dg::network_uma::device_at_nothrow(dst, i);
            vma_ptr_t dst_vptr  = dg::network_uma::map_direct_nothrow(id, dst);
            dg::network_memops_virt::memset_nothrow(dst_vptr, c, n);
        }
    }
}

namespace dg::network_memops_umax{

    auto memcpy_uma_to_host(void * dst, uma_ptr_t src, size_t n) noexcept -> exception_t{

        vma_ptr_t dst_vptr = dg::network_virtual_device::virtualize_host_ptr(dst); 
        return dg::network_memops_uma::memcpy_uma_to_vma(dst_vptr, src, n);
    }

    void memcpy_uma_to_host_nothrow(void * dst, uma_ptr_t src, size_t n) noexcept{

        vma_ptr_t dst_vptr = dg::network_virtual_device::virtualize_host_ptr(dst); 
        dg::network_memops_uma::memcpy_uma_to_vma_nothrow(dst_vptr, src, n);
    }

    auto memcpy_host_to_uma(uma_ptr_t dst, void * src, size_t n) noexcept -> exception_t{ //remove constness for now - next iteration 

        vma_ptr_t src_vptr = dg::network_virtual_device::virtualize_host_ptr(src);
        return dg::network_memops_uma::memcpy_vma_to_uma(dst, src_vptr, n);
    }

    void memcpy_host_to_uma_nothrow(uma_ptr_t dst, void * src, size_t n) noexcept{

        vma_ptr_t src_vptr = dg::network_virtual_device::virtualize_host_ptr(src);
        dg::network_memops_uma::memcpy_vma_to_uma_nothrow(dst, src_vptr, n);
    }

    auto memcpy_host_to_uma_directall(uma_ptr_t dst, void * src, size_t n) noexcept -> exception_t{ //remove constness for now - next iteration

        vma_ptr_t src_vptr = dg::network_virtual_device::virtualize_host_ptr(src);
        return dg::network_memops_uma::memcpy_vma_to_uma_directall(dst, src_vptr, n);
    }

    void memcpy_host_to_uma_directall_nothrow(uma_ptr_t dst, void * src, size_t n) noexcept{

        vma_ptr_t src_vptr = dg::network_virtual_device::virtualize_host_ptr(src);
        dg::network_memops_uma::memcpy_vma_to_uma_directall_nothrow(dst, src_vptr, n);
    }
}

#endif