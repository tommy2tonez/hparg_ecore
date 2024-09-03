#ifndef __NETWORK_MEMOPS_UMA_H__
#define __NETWORK_MEMOPS_UMA_H__

#include "network_memlock.h"
#include "network_uma.h"
#include "network_memops.h"
#include "network_exception_handler.h"

namespace dg::network_memops_uma{

    using uma_lock_instance = dg::network_memlock_host::Lock<signature_dg_network_uma, std::integral_constant<size_t, MEMREGION_SZ>, uma_ptr_t>; 
 
    void init(){ //weird - yet it's anti-pattern otherwise - all memory operation should be global for bridge + join problems

    }

    auto memlock_guard(uma_ptr_t ptr) noexcept{

        return dg::network_memlock_utility::recursive_lock_guard(uma_lock_instance{}, ptr);
    } 
    
    template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args, uma_ptr_t>...>, bool> = true>
    auto memlock_guard_many(Args... args) noexcept{

        return dg::network_memlock_utility::recursive_lock_guard_many(uma_lock_instance{}, args...);
    }

    auto memcpy_uma_to_vma(vma_ptr_t dst, uma_ptr_t src, size_t n) noexcept -> exception_t{

        auto src_maptoken   = dg::network_uma::map_wait(src);

        if (!src_maptoken.has_value()){
            return src_maptoken.error();
        }

        auto relgrd         = dg::network_uma::map_relguard(src_maptoken.value());
        vma_ptr_t src_vptr  = dg::network_uma::get_vma_const_ptr(src_maptoken.value());
        
        return dg::network_memops_virt::memcpy(dst, src_vptr, n);
    }

    void memcpy_uma_to_vma_nothrow(vma_ptr_t dst, uma_ptr_t src, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_uma_to_vma(dst, src, n));
    }

    auto memcpy_vma_to_uma(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept -> exception_t{

        auto dst_maptoken   = dg::network_uma::map_wait(dst);

        if (!dst_maptoken.has_value()){
            return dst_maptoken.error();
        }

        auto relgrd         = dg::network_uma::map_relguard(dst_maptoken.value());
        vma_ptr_t dst_vptr  = dg::network_uma::get_vma_ptr(dst_maptoken.value());

        return dg::network_memops_virt::memcpy(dst_vptr, src, n);
    }

    void memcpy_vma_to_uma_nothrow(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_vma_to_uma(dst, src, n));
    }

    auto memcpy_vma_to_uma_directall_bypass_qualifier_nothrow(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept{

        size_t dc = dg::network_uma::device_count_nothrow(dst); 

        for (size_t i = 0; i < dc; ++i){
            auto id             = dg::network_uma::device_at_nothrow(dst, i) 
            vma_ptr_t dst_vptr  = dg::network_uma::map_direct_nothrow(id, dst);
            dg::network_memops_virt::memcpy_nothrow(dst_vptr, src, n);
        }
    }

    auto memset(uma_ptr_t dst, int c, size_t n) noexcept -> exception_t{

        auto dst_maptoken   = dg::network_uma::map_wait(dst);
        
        if (!dst_maptoken.has_value()){
            return dst_maptoken.error();
        }

        auto relgrd         = dg::network_uma::map_relguard(dst_maptoken.value());
        vma_ptr_t dst_vptr  = dg::network_uma::get_vma_ptr(dst_maptoken.value());

        return dg::network_memops_virt::memset(dst_vptr, c, n);
    }

    void memset_nothrow(uma_ptr_t dst, int c, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memset(dst, c, n));
    }

    void memset_directall_bypass_qualifier_nothrow(uma_ptr_t dst, int c, size_t n) noexcept{

        size_t dc = dg::network_uma::device_count_nothrow(dst);

        for (size_t i = 0; i < dc; ++i){
            auto id             = dg::network_uma::device_at_nothrow(dst, i);
            vma_ptr_t dst_vptr  = dg::network_uma::map_direct_nothrow(id, dst);
            dg::network_memops_virt::memset_nothrow(dst_vptr, c, n);
        }
    }
}

namespace dg::network_memops_umax{

    auto memcpy_uma_to_host(void * dst, uma_ptr_t src, size_t n) noexcept -> exception_t{

        vma_ptr_t dst_vptr = dg::network_virtual_device::virtualize_host_ptr(dst, dg::network_virtual_device::HOST_MAIN_ID); 
        return dg::network_memops_uma::memcpy_uma_to_vma(dst_vptr, src, n);
    }

    void memcpy_uma_to_host_nothrow(void * dst, uma_ptr_t src, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_uma_to_host(dst, src, n));
    }

    auto memcpy_host_to_uma(uma_ptr_t dst, void * src, size_t n) noexcept -> exception_t{ //remove constness for now - next iteration 

        vma_ptr_t src_vptr = dg::network_virtual_device::virtualize_host_ptr(src, dg::network_virtual_device::HOST_MAIN_ID);
        return dg::network_memops_uma::memcpy_vma_to_uma(dst, src_vptr, n);
    }

    void memcpy_host_to_uma_nothrow(uma_ptr_t dst, void * src, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_host_to_uma(dst, src, n));
    }

    void memcpy_host_to_uma_directall_bypass_qualifier_nothrow(uma_ptr_t dst, void * src, size_t n) noexcept{ //remove constness for now - next iteration

        vma_ptr_t src_vptr  = dg::network_virtual_device::virtualize_host_ptr(src, dg::network_virtual_device::HOST_MAIN_ID);
        dg::network_memops_uma::memcpy_vma_to_uma_directall_bypass_qualifier_nothrow(dst, src_vptr, n);
    }
}

#endif