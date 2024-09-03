#ifndef __NETWORK_MEMOPS_VIRTUAL_DEVICE_H__
#define __NETWORK_MEMOPS_VIRTUAL_DEVICE_H__

#include <stdlib.h>
#include <stddef.h>
#include "network_exception.h"
#include "network_virtual_device.h"
#include "network_kernelmap_x.h" 

namespace dg::network_memops_clib{

    using cuda_ptr_t        = void *; 
    using exception_t       = dg::network_exception::exception_t; 
    using cuda_device_id_t  = int; 

    #ifdef __DG_NETWORK_CUDA_FLAG__

    inline auto memcpy_host_to_cuda(cuda_ptr_t, cuda_device_id_t, void *, size_t) noexcept -> exception_t{

        // cudaMemcpy();
    }

    inline auto memcpy_cuda_to_host(void *, cuda_ptr_t, cuda_device_id_t, size_t) noexcept -> exception_t{

    }

    inline auto memcpy_cuda_to_cuda(cuda_ptr_t, cuda_device_id_t, cuda_ptr_t, cuda_device_id_t, size_t) noexcept -> exception_t{

    }

    inline auto memset_cuda(cuda_ptr_t, cuda_device_id_t, int, size_t) noexcept -> exception_t{

    }

    #else 

    inline auto memcpy_host_to_cuda(...) noexcept -> exception_t{

        return dg::network_exception::CUDA_DEVICE_NOT_SUPPORTED;
    }

    inline auto memcpy_cuda_to_host(...) noexcept -> exception_t{

        return dg::network_exception::CUDA_DEVICE_NOT_SUPPORTED;
    }

    inline auto memcpy_cuda_to_cuda(...) noexcept -> exception_t{

        return dg::network_exception::CUDA_DEVICE_NOT_SUPPORTED;
    }

    inline auto memset_cuda(...) noexcept -> exception_t{

        return dg::network_exception::CUDA_DEVICE_NOT_SUPPORTED;
    }

    #endif

    inline auto memcpy_host_to_host(void * dst, const void * src, size_t sz) noexcept -> exception_t{

        std::memcpy(dst, src, sz);
        return dg::network_exception::SUCCESS;
    }

    inline auto memset_host(void * dst, int c, size_t sz) noexcept -> exception_t{

        std::memset(dst, c, sz);
        return dg::network_exception::SUCCESS;
    }

    inline void memcpy_host_to_cuda_nothrow(cuda_ptr_t dst, cuda_device_id_t dst_id, void * src, size_t sz) noexcept{
        
        dg::network_error_handler::nothrow_log(memcpy_host_to_cuda(dst, dst_id, src, sz));
    }

    inline void memcpy_cuda_to_host_nothrow(void * dst, cuda_ptr_t src, cuda_device_id_t src_id, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_cuda_to_host(dst, src, src_id, sz));
    }

    inline void memcpy_cuda_to_cuda_nothrow(cuda_ptr_t dst, cuda_device_id_t dst_id, cuda_ptr_t src, cuda_device_id_t src_id, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_cuda_to_cuda(dst, dst_id, src, src_id, sz));
    }
    
    inline void memcpy_host_to_host_nothrow(void * dst, const void * src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_host_to_host(dst, src, sz));
    }

    inline void memset_cuda_nothrow(cuda_ptr_t dst, cuda_device_id_t dst_id, int c, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memset_cuda(dst, dst_id, c, sz));
    }

    inline void memset_host_nothrow(void * dst, int c, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memset_host(dst, c, sz));
    }
}

namespace dg::network_memops_fsys{

    inline auto memcpy_host_to_fsys(fsys_ptr_t dst, const void * src, size_t sz) noexcept -> exception_t{

        auto map_rs             = dg::network_kernelmap_x::map_wait(dst);
        auto map_guard          = dg::network_kernelmap_x::map_guard(map_rs);
        void * dst_cptr         = dg::network_kernelmap_x::get_host_ptr(map_rs);
        exception_t cpy_err     = dg::network_memops_clib::memcpy_host_to_host(dst_cptr, src, sz);

        if (dg::network_exception::is_failed(cpy_err)){
            return cpy_err;
        }

        return dg::network_exception::SUCCESS;
    }

    inline auto memcpy_cuda_to_fsys(fsys_ptr_t dst, cuda_ptr_t src, cuda_device_id_t src_id, size_t sz) noexcept -> exception_t{

        auto map_rs             = dg::network_kernelmap_x::map_wait(dst);
        auto map_guard          = dg::network_kernelmap_x::map_relguard(map_rs);
        void * dst_cptr         = dg::network_kernelmap_x::get_host_ptr(map_rs);
        exception_t cpy_err     = dg::network_memops_clib::memcpy_cuda_to_host(dst_cptr, src, src_id, sz);

        if (dg::network_exception::is_failed(cpy_err)){
            return cpy_err;
        }

        return dg::network_exception::SUCCESS;
    }

    inline auto memcpy_fsys_to_host(void * dst, fsys_ptr_t src, size_t sz) noexcept -> exception_t{

        auto map_rs             = dg::network_kernelmap_x::map_wait(src);
        auto map_guard          = dg::network_kernelmap_x::map_relguard(map_rs);
        const void * src_cptr   = dg::network_kernelmap_x::get_host_const_ptr(map_rs);
        exception_t cpy_err     = dg::network_memops_clib::memcpy_host_to_hst(dst, src_cptr, sz);

        if (dg::network_exception::is_failed(cpy_err)){
            return cpy_err;
        }

        return dg::network_exception::SUCCESS;
    }

    inline auto memcpy_fsys_to_cuda(cuda_ptr_t dst, cuda_device_id_t dst_id, fsys_ptr_t src, size_t sz) noexcept -> exception_t{

        auto map_rs             = dg::network_kernelmap_x::map_wait(src);
        auto map_guard          = dg::network_kernelmap_x::map_relguard(map_rs);
        const void * src_cptr   = dg::network_kernelmap_x::get_host_const_ptr(map_rs);
        exception_t cpy_err     = dg::network_memops_clib::memcpy_host_to_cuda(dst, dst_id, src_cptr, sz);

        if (dg::network_exception::is_failed(cpy_err)){
            return cpy_err;
        }

        return dg::network_exception::SUCCESS;
    }

    inline auto memcpy_fsys_to_fsys(fsys_ptr_t dst, fsys_ptr_t src, size_t sz) noexcept -> exception_t{

        auto dst_map_rs         = dg::network_kernelmap_x::map_wait(dst);
        auto src_map_rs         = dg::network_kernelmap_x::map_wait(src);
        auto dst_map_guard      = dg::network_kernelmap_x::map_relguard(dst_map_rs);
        auto src_map_guard      = dg::network_kernelmap_x::map_relguard(src_map_rs);
        void * dst_cptr         = dg::network_kernelmap_x::get_host_ptr(dst_map_rs);
        const void * src_cptr   = dg::network_kernelmap_x::get_host_const_ptr(src_map_rs);
        exception_t cpy_err     = dg::network_memops_clib::memcpy_host_to_host(dst_cptr, src_cptr, sz);

        if (dg::network_exception::is_failed(cpy_err)){
            return cpy_err;
        }

        return dg::network_exception::SUCCESS;
    }

    inline auto memset_fsys(fsys_ptr_t dst, int c, size_t sz) noexcept -> exception_t{
        
        auto dst_map_rs         = dg::network_kernelmap_x::map_wait(dst);
        auto dst_map_guard      = dg::network_kernelmap_x::map_guard(dst_map_rs);
        void * dst_cptr         = dg::network_kernelmap_x::get_host_ptr(dst_map_rs);
        exception_t set_err     = dg::network_memops_clib::memset_host(dst_cptr, c, sz);

        if (dg::network_exception::is_failed(set_err)){
            return set_err;
        }

        return dg::network_exception::SUCCESS;
    }

    inline void memcpy_host_to_fsys_nothrow(fsys_ptr_t dst, const void * src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_host_to_fsys(dst, src, sz));
    }

    inline void memcpy_cuda_to_fsys_nothrow(fsys_ptr_t dst, cuda_ptr_t src, cuda_device_id_t src_id, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_cuda_to_fsys(dst, src, src_id, sz));
    }

    inline void memcpy_fsys_to_host_nothrow(void * dst, fsys_ptr_t src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_fsys_to_host(dst, src, sz));
    }

    inline void memcpy_fsys_to_cuda_nothrow(cuda_ptr_t dst, cuda_device_id_t dst_id, fsys_ptr_t src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_fsys_to_cuda(dst, dst_id, src, sz));
    }

    inline void memcpy_fsys_to_fsys_nothrow(fsys_ptr_t dst, fsys_ptr_t src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_fsys_to_fsys(dst, src, sz));
    }

    inline void memset_fsys_nothrow(fsys_ptr_t dst, int c, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memset_fsys(dst, c, sz));
    }
}

namespace dg::network_memops_virt{

    using vma_ptr_t = uint32_t; 

    inline auto memcpy(vma_ptr_t dst, vma_ptr_t src, size_t sz) noexcept -> exception_t{

        using namespace dg::network_virtual_device; 
        
        if (is_host_ptr(dst) && is_host_ptr(src)){
            auto [dst_ptr, dst_id] = devirtualize_host_ptr(dst);
            auto [src_ptr, src_id] = devirtualize_host_ptr(src);
            return network_memops_clib::memcpy_host_to_host(dst_ptr, src_ptr, sz);
        } 

        if (is_host_ptr(dst) && is_cuda_ptr(src)){
            auto [dst_ptr, dst_id] = devirtualize_host_ptr(dst);
            auto [src_ptr, src_id] = devirtualize_cuda_ptr(src);
            return network_memops_clib::memcpy_cuda_to_host(dst_ptr, src_ptr, src_id, sz);
        }
        
        if (is_host_ptr(dst) && is_fsys_ptr(src)){
            auto [dst_ptr, dst_id] = devirtualize_host_ptr(dst);
            auto [src_ptr, src_id] = devirtualize_fsys_ptr(src);
            return network_memops_fsys::memcpy_fsys_to_host(dst_ptr, src_ptr, sz);
        }

        if (is_cuda_ptr(dst) && is_host_ptr(src)){
            auto [dst_ptr, dst_id] = devirtualize_cuda_ptr(dst);
            auto [src_ptr, src_id] = devirtualize_host_ptr(src);
            return network_memops_clib::memcpy_host_to_cuda(dst_ptr, dst_id, src_ptr, sz);
        }

        if (is_cuda_ptr(dst) && is_cuda_ptr(src)){
            auto [dst_ptr, dst_id] = devirtualize_cuda_ptr(dst);
            auto [src_ptr, src_id] = devirtualize_cuda_ptr(src);
            return network_memops_clib::memcpy_cuda_to_cuda(dst_ptr, dst_id, src_ptr, src_id, sz);
        }

        if (is_cuda_ptr(dst) && is_fsys_ptr(src)){
            auto [dst_ptr, dst_id] = devirtualize_cuda_ptr(dst);
            auto [src_ptr, src_id] = devirtualize_fsys_ptr(src);
            return network_memops_fsys::memcpy_fsys_to_cuda(dst_ptr, dst_id, src, sz);
        }

        if (is_fsys_ptr(dst) && is_host_ptr(src)){
            auto [dst_ptr, dst_id] = devirtualize_fsys_ptr(dst);
            auto [src_ptr, src_id] = devirtualize_host_ptr(src);
            return network_memops_fsys::memcpy_host_to_fsys(dst_ptr, src_ptr, sz);
        }

        if (is_fsys_ptr(dst) && is_cuda_ptr(src)){
            auto [dst_ptr, dst_id] = devirtualize_fsys_ptr(dst);
            auto [src_ptr, src_id] = devirtualize_cuda_ptr(src);
            return network_memops_fsys::memcpy_cuda_to_fsys(dst_ptr, src_ptr, src_id, sz);
        }

        if (is_fsys_ptr(dst) && is_fsys_ptr(src)){
            auto [dst_ptr, dst_id] = devirtualize_fsys_ptr(dst);
            auto [src_ptr, src_id] = devirtualize_fsys_ptr(src);
            return network_memops_fsys::memcpy_fsys_to_fsys(dst_ptr, src_ptr, sz);
        }

        return dg::network_exception::INVALID_VMAPTR_FORMAT;
    }

    inline void memcpy_nothrow(vma_ptr_t dst, vma_ptr_t src, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy(dst, src, sz));
    }

    inline auto memset(vma_ptr_t dst, int c, size_t sz) noexcept -> exception_t{

        using namespace dg::network_virtual_device;

        if (is_host_ptr(dst)){
            auto [dst_ptr, dst_id] = devirtualize_host_ptr(dst);
            return network_memops_clib::memset_host(dst_ptr, c, sz);
        } 

        if (is_cuda_ptr(dst)){
            auto [dst_ptr, dst_id] = devirtualize_cuda_ptr(dst);
            return network_memops_clib::memset_cuda(dst_ptr, dst_id, c, sz);
        }

        if (is_fsys_ptr(dst)){
            auto [dst_ptr, dst_id] = devirtualize_fsys_ptr(dst);
            return network_memops_fsys::memset_fsys(dst, c, sz);
        }

        return dg::network_exception::INVALID_SERIALIZATION_FORMAT;
    }

    inline void memset_nothrow(vma_ptr_t dst, int c, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memset(dst, c, sz));
    }
} 

#endif