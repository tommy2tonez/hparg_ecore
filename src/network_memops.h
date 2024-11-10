#ifndef __NETWORK_MEMOPS_VIRTUAL_DEVICE_H__
#define __NETWORK_MEMOPS_VIRTUAL_DEVICE_H__

#include <stdlib.h>
#include <stddef.h>
#include "network_exception.h"
#include "network_virtual_device.h"
#include "network_kernelmap_x.h" 
#include "stdx.h"

namespace dg::network_memops_clib{

    using cuda_ptr_t        = void *; 
    using exception_t       = dg::network_exception::exception_t; 
    using cuda_device_id_t  = int; 

    #ifdef __DG_NETWORK_CUDA_FLAG__

    auto memcpy_host_to_cuda(cuda_ptr_t, void *, size_t) noexcept -> exception_t{

        // cudaMemcpy();
    }

    auto memcpy_cuda_to_host(void *, cuda_ptr_t, size_t) noexcept -> exception_t{

    }

    auto memcpy_cuda_to_cuda(cuda_ptr_t, cuda_ptr_t, size_t) noexcept -> exception_t{

    }

    auto memset_cuda(cuda_ptr_t, int, size_t) noexcept -> exception_t{

    }

    #else 

    auto memcpy_host_to_cuda(...) noexcept -> exception_t{

        return dg::network_exception::CUDA_DEVICE_NOT_SUPPORTED;
    }

    auto memcpy_cuda_to_host(...) noexcept -> exception_t{

        return dg::network_exception::CUDA_DEVICE_NOT_SUPPORTED;
    }

    auto memcpy_cuda_to_cuda(...) noexcept -> exception_t{

        return dg::network_exception::CUDA_DEVICE_NOT_SUPPORTED;
    }

    auto memset_cuda(...) noexcept -> exception_t{

        return dg::network_exception::CUDA_DEVICE_NOT_SUPPORTED;
    }

    #endif


    //this is defined - because I wrote it
    //remember - launder is the savior and launder is the devil - this is ONLY defined if you synchronize all read and write at the MEMCPY
    //not read through dereferncing the uint64_5 * or uint32_t * or uint16_t *
    //the ONLY pointer you should have in the program is either void * or uintptr_t - YEAH
    //because when you launder a pointer - the compiler assumes that it does not alias with ANY OTHER POINTER - it's like a newly allocated pointer
    //so when you write to a uint32_t *, the other laundered uint64_t * won't see the result and, congratulations, you have invoked undefined behavior
    //this is the main reason std::start_lifetime_as_array is not implemented - it's not feasible - from the perspective of a compiler engineer
    //the only other defined case, in conjunction with the above use case, is when you have a restricted void * pointer - you want to launder that restricted pointer to an arithmetic ptr type
    //and you only do basic operations on the newly created arithmetic pointer - and the lifetime of such pointer - you might not call the above memcpy - and you cannot launder another pointer
    //it's that hard - really

    auto memcpy_host_to_host(void * dst, const void * src, size_t sz) noexcept -> exception_t{

        auto grd = stdx::memtransaction_optional_guard();
        std::memcpy(dst, stdx::launder_pointer<void>(src), sz);
        return dg::network_exception::SUCCESS;
    }

    auto memset_host(void * dst, int c, size_t sz) noexcept -> exception_t{

        auto grd = stdx::memtransaction_optional_guard();
        std::memset(dst, c, sz);
        return dg::network_exception::SUCCESS;
    }

    void memcpy_host_to_cuda_nothrow(cuda_ptr_t dst, void * src, size_t sz) noexcept{
        
        dg::network_error_handler::nothrow_log(memcpy_host_to_cuda(dst, src, sz));
    }

    void memcpy_cuda_to_host_nothrow(void * dst, cuda_ptr_t src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_cuda_to_host(dst, src, sz));
    }

    void memcpy_cuda_to_cuda_nothrow(cuda_ptr_t dst, cuda_ptr_t src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_cuda_to_cuda(dst, src, sz));
    }
    
    void memcpy_host_to_host_nothrow(void * dst, const void * src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_host_to_host(dst, src, sz));
    }

    void memset_cuda_nothrow(cuda_ptr_t dst, int c, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memset_cuda(dst, c, sz));
    }

    void memset_host_nothrow(void * dst, int c, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memset_host(dst, c, sz));
    }
}

namespace dg::network_memops_fsys{

    auto memcpy_host_to_fsys(fsys_ptr_t dst, const void * src, size_t sz) noexcept -> exception_t{

        auto map_rs             = dg::network_kernelmap_x::map_safe(dst);
        
        if (!map_rs.has_value()){
            return map_rs.error();
        }

        void * dst_cptr         = dg::network_kernelmap_x::get_host_ptr(map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_host_to_host(dst_cptr, src, sz);

        return err;
    }

    auto memcpy_cuda_to_fsys(fsys_ptr_t dst, cuda_ptr_t src, size_t sz) noexcept -> exception_t{

        auto map_rs             = dg::network_kernelmap_x::map_safe(dst);

        if (!map_rs.has_value()){
            return map_rs.error();
        }

        void * dst_cptr         = dg::network_kernelmap_x::get_host_ptr(map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_cuda_to_host(dst_cptr, src, sz);

        return err;
    }

    auto memcpy_fsys_to_host(void * dst, fsys_ptr_t src, size_t sz) noexcept -> exception_t{

        auto map_rs             = dg::network_kernelmap_x::map_safe(src);

        if (!map_rs.has_value()){
            return map_rs.error();
        }

        const void * src_cptr   = dg::network_kernelmap_x::get_host_ptr(map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_host_to_host(dst, src_cptr, sz);

        return err;
    }

    auto memcpy_fsys_to_cuda(cuda_ptr_t dst, fsys_ptr_t src, size_t sz) noexcept -> exception_t{

        auto map_rs             = dg::network_kernelmap_x::map_safe(src);

        if (!map_rs.has_value()){
            return map_rs.error();
        }

        const void * src_cptr   = dg::network_kernelmap_x::get_host_ptr(map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_host_to_cuda(dst, src_cptr, sz);

        return err;
    }

    auto memcpy_fsys_to_fsys(fsys_ptr_t dst, fsys_ptr_t src, size_t sz) noexcept -> exception_t{

        auto dst_map_rs         = dg::network_kernelmap_x::map_safe(dst);

        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        auto src_map_rs         = dg::network_kernelmap_x::map_safe(src);

        if (!src_map_rs.has_value()){
            return src_map_rs.error();
        }

        void * dst_cptr         = dg::network_kernelmap_x::get_host_ptr(dst_map_rs.value());
        const void * src_cptr   = dg::network_kernelmap_x::get_host_ptr(src_map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_host_to_host(dst_cptr, src_cptr, sz);

        return err;
    }

    auto memset_fsys(fsys_ptr_t dst, int c, size_t sz) noexcept -> exception_t{
        
        auto dst_map_rs         = dg::network_kernelmap_x::map_safe(dst);

        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        void * dst_cptr         = dg::network_kernelmap_x::get_host_ptr(dst_map_rs.value());
        exception_t err         = dg::network_memops_clib::memset_host(dst_cptr, c, sz);

        return err;
    }

    void memcpy_host_to_fsys_nothrow(fsys_ptr_t dst, const void * src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_host_to_fsys(dst, src, sz));
    }

    void memcpy_cuda_to_fsys_nothrow(fsys_ptr_t dst, cuda_ptr_t src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_cuda_to_fsys(dst, src, sz));
    }

    void memcpy_fsys_to_host_nothrow(void * dst, fsys_ptr_t src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_fsys_to_host(dst, src, sz));
    }

    void memcpy_fsys_to_cuda_nothrow(cuda_ptr_t dst, fsys_ptr_t src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_fsys_to_cuda(dst, src, sz));
    }

    void memcpy_fsys_to_fsys_nothrow(fsys_ptr_t dst, fsys_ptr_t src, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memcpy_fsys_to_fsys(dst, src, sz));
    }

    void memset_fsys_nothrow(fsys_ptr_t dst, int c, size_t sz) noexcept{

        dg::network_error_handler::nothrow_log(memset_fsys(dst, c, sz));
    }
}

namespace dg::network_memops_cufs{

    auto memcpy_host_to_cufs(cufs_ptr_t dst, const void * src, size_t sz) noexcept -> exception_t{

        auto dst_map_rs         = dg::network_cudafsmap_x::map_safe(dst);

        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        cuda_ptr_t dst_cuptr    = dg::network_cudafsmap_x::get_cuda_ptr(dst_map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_host_to_cuda(dst_cuptr, src, sz);

        return err;
    } 

    auto memcpy_cuda_to_cufs(cufs_ptr_t dst, cuda_ptr_t src, size_t sz) noexcept -> exception_t{

        auto dst_map_rs         = dg::network_cudafsmap_x::map_safe(dst);

        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        cuda_ptr_t dst_cuptr    = dg::network_cudafsmap_x::get_cuda_ptr(dst_map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_cuda_to_cuda(dst_cuptr, src, sz);

        return err;
    }

    auto memcpy_fsys_to_cufs(cufs_ptr_t dst, fsys_ptr_t src, size_t sz) noexcept -> exception_t{

        auto dst_map_rs         = dg::network_cudafsmap_x::map_safe(dst);

        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        auto src_map_rs         = dg::network_kernelmap_x::map_safe(src);

        if (!src_map_rs.has_value()){
            return src_map_rs.error();
        }

        cuda_ptr_t dst_cuptr    = dg::network_cudafsmap_x::get_cuda_ptr(dst_map_rs.value());
        void * src_hostptr      = dg::network_kernelmap_x::get_host_ptr(src_map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_host_to_cuda(dst_cuptr, src_hostptr, sz);

        return err;
    }

    auto memcpy_cufs_to_cufs(cufs_ptr_t dst, cufs_ptr_t src, size_t sz) noexcept -> exception_t{

        auto dst_map_rs         = dg::network_cudafsmap_x::map_safe(dst);

        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        } 

        auto src_map_rs         = dg::network_cudafsmap_x::map_safe(src);

        if (!src_map_rs.has_value()){
            return src_map_rs.error();
        }

        cuda_ptr_t dst_cuptr    = dg::network_cudafsmap_x::get_cuda_ptr(dst_map_rs.value());
        cuda_ptr_t src_cuptr    = dg::network_cudafsmap_x::get_cuda_ptr(src_map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_cuda_to_cuda(dst_cuptr, src_cuptr, sz);

        return err;
    }

    auto memcpy_cufs_to_host(void * dst, cufs_ptr_t src, size_t sz) noexcept -> exception_t{

        auto src_map_rs         = dg::network_cudafsmap_x::map_safe(src);

        if (!src_map_rs.has_value()){
            return src_map_rs.error();
        } 

        cuda_ptr_t src_cuptr    = dg::network_cudafsmap_x::get_cuda_ptr(src_map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_cuda_to_host(dst, src_cuptr, sz);

        return err;
    }

    auto memcpy_cufs_to_cuda(cuda_ptr_t dst, cufs_ptr_t src, size_t sz) noexcept -> exception_t{

        auto src_map_rs         = dg::network_cudafsmap_x::map_safe(src);

        if (!src_map_rs.has_value()){
            return src_map_rs.error();
        }

        cuda_ptr_t src_cuptr    = dg::network_cudafsmap_x::get_cuda_ptr(src_map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_cuda_to_cuda(dst, src_cuptr, sz);

        return err; 
    }

    auto memcpy_cufs_to_fsys(fsys_ptr_t dst, cufs_ptr_t src, size_t sz) noexcept -> exception_t{

        auto dst_map_rs         = dg::network_kernelmap_x::map_safe(dst);

        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        auto src_map_rs         = dg::network_cudafsmap_x::map_safe(src);

        if (!src_map_rs.has_value()){
            return src_map_rs.error();
        }

        void * dst_hostptr      = dg::network_kernelmap_x::get_host_ptr(dst_map_rs.value());
        cuda_ptr_t src_cuptr    = dg::network_cudafsmap_x::get_cuda_ptr(src_map_rs.value());
        exception_t err         = dg::network_memops_clib::memcpy_cuda_to_host(dst_hostptr, src_cuptr, sz);

        return err;
    }

    auto memset_cufs(cufs_ptr_t dst, int c, size_t sz) noexcept -> exception_t{

        auto dst_map_rs         = dg::network_cudafsmap_x::map_safe(dst);

        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        cuda_ptr_t dst_cuptr    = dg::network_cudafsmap_x::get_cuda_ptr(dst_map_rs.value());
        exception_t err         = dg::network_memops_clib::memset_cuda(dst_cuptr, c, sz);

        return err;
    }

    void memcpy_host_to_cufs_nothrow(cufs_ptr_t dst, const void * src, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_host_to_cufs(dst, src, sz));
    }

    void memcpy_cuda_to_cufs_nothrow(cufs_ptr_t dst, cuda_ptr_t src, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_cuda_to_cufs(dst, src, sz));
    }

    void memcpy_fsys_to_cufs_nothrow(cufs_ptr_t dst, fsys_ptr_t src, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_fsys_to_cufs(dst, src, sz));
    }

    void memcpy_cufs_to_cufs_nothrow(cufs_ptr_t dst, cufs_ptr_t src, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_cufs_to_cufs(dst, src, sz));
    }

    void memcpy_cufs_to_host_nothrow(void * dst, cufs_ptr_t src, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_cufs_to_host(dst, src, sz));
    }

    void memcpy_cufs_to_cuda_nothrow(cuda_ptr_t dst, cufs_ptr_t src, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_cufs_to_cuda(dst, src, sz));
    }

    void memcpy_cufs_to_fsys_nothrow(fsys_ptr_t dst, cufs_ptr_t src, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_cufs_to_fsys(dst, src, sz));
    }

    void memset_cufs_nothrow(cufs_ptr_t dst, int c, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memset_cufs(dst, c, sz));
    }
} 

namespace dg::network_memops_virt{

    using vma_ptr_t = uint32_t; 

    auto memcpy(vma_ptr_t dst, vma_ptr_t src, size_t sz) noexcept -> exception_t{

        using namespace dg::network_virtual_device; 
        
        if (is_host_ptr(dst) && is_host_ptr(src)){
            auto dst_ptr = devirtualize_host_ptr(dst);
            auto src_ptr = devirtualize_host_ptr(src);
            return network_memops_clib::memcpy_host_to_host(dst_ptr, src_ptr, sz);
        } 

        if (is_host_ptr(dst) && is_cuda_ptr(src)){
            auto dst_ptr = devirtualize_host_ptr(dst);
            auto src_ptr = devirtualize_cuda_ptr(src);
            return network_memops_clib::memcpy_cuda_to_host(dst_ptr, src_ptr, sz);
        }
        
        if (is_host_ptr(dst) && is_fsys_ptr(src)){
            auto dst_ptr = devirtualize_host_ptr(dst);
            auto src_ptr = devirtualize_fsys_ptr(src);
            return network_memops_fsys::memcpy_fsys_to_host(dst_ptr, src_ptr, sz);
        }

        if (is_host_ptr(dst) && is_cufs_ptr(src)){
            auto dst_ptr = devirtualize_host_ptr(dst);
            auto src_ptr = devirtualize_cufs_ptr(src);
            return network_memops_cufs::memcpy_cufs_to_host(dst_ptr, src_ptr, sz);
        }
        
        if (is_cuda_ptr(dst) && is_host_ptr(src)){
            auto dst_ptr = devirtualize_cuda_ptr(dst);
            auto src_ptr = devirtualize_host_ptr(src);
            return network_memops_clib::memcpy_host_to_cuda(dst_ptr, src_ptr, sz);
        }

        if (is_cuda_ptr(dst) && is_cuda_ptr(src)){
            auto dst_ptr = devirtualize_cuda_ptr(dst);
            auto src_ptr = devirtualize_cuda_ptr(src);
            return network_memops_clib::memcpy_cuda_to_cuda(dst_ptr, src_ptr, sz);
        }

        if (is_cuda_ptr(dst) && is_fsys_ptr(src)){
            auto dst_ptr = devirtualize_cuda_ptr(dst);
            auto src_ptr = devirtualize_fsys_ptr(src);
            return network_memops_fsys::memcpy_fsys_to_cuda(dst_ptr, src_ptr, sz);
        }

        if (is_cuda_ptr(dst) && is_cufs_ptr(src)){
            auto dst_ptr = devirtualize_cuda_ptr(dst);
            auto src_ptr = devirtualize_cufs_ptr(src);
            return network_memops_cufs::memcpy_cufs_to_cuda(dst_ptr, src_ptr, sz);
        }

        if (is_fsys_ptr(dst) && is_host_ptr(src)){
            auto dst_ptr = devirtualize_fsys_ptr(dst);
            auto src_ptr = devirtualize_host_ptr(src);
            return network_memops_fsys::memcpy_host_to_fsys(dst_ptr, src_ptr, sz);
        }

        if (is_fsys_ptr(dst) && is_cuda_ptr(src)){
            auto dst_ptr = devirtualize_fsys_ptr(dst);
            auto src_ptr = devirtualize_cuda_ptr(src);
            return network_memops_fsys::memcpy_cuda_to_fsys(dst_ptr, src_ptr, sz);
        }

        if (is_fsys_ptr(dst) && is_fsys_ptr(src)){
            auto dst_ptr = devirtualize_fsys_ptr(dst);
            auto src_ptr = devirtualize_fsys_ptr(src);
            return network_memops_fsys::memcpy_fsys_to_fsys(dst_ptr, src_ptr, sz);
        }

        if (is_fsys_ptr(dst) && is_cufs_ptr(src)){
            auto dst_ptr = devirtualize_fsys_ptr(dst);
            auto src_ptr = devirtualize_cufs_ptr(src);
            return network_memops_cufs::memcpy_cufs_to_fsys(dst_ptr, src_ptr, sz);
        }

        if (is_cufs_ptr(dst) && is_host_ptr(src)){
            auto dst_ptr = devirtualize_cufs_ptr(dst);
            auto src_ptr = devirtualize_host_ptr(src);
            return network_memops_cufs::memcpy_host_to_cufs(dst_ptr, src_ptr, sz);
        }

        if (is_cufs_ptr(dst) && is_cuda_ptr(src)){
            auto dst_ptr = devirtualize_cufs_ptr(dst);
            auto src_ptr = devirtualize_cuda_ptr(src);
            return network_memops_cufs::memcpy_cuda_to_cufs(dst_ptr, src_ptr, sz);
        }

        if (is_cufs_ptr(dst) && is_fsys_ptr(src)){
            auto dst_ptr = devirtualize_cufs_ptr(dst);
            auto src_ptr = devirtualize_fsys_ptr(src);
            return network_memops_cufs::memcpy_fsys_to_cufs(dst_ptr, src_ptr, sz);
        }

        if (is_cufs_ptr(dst) && is_cufs_ptr(src)){
            auto dst_ptr = devirtualize_cufs_ptr(dst);
            auto src_ptr = devirtualize_cufs_ptr(src);
            return network_memops_cufs::memcpy_cufs_to_cufs(dst_ptr, src_ptr, sz);
        }

        return dg::network_exception::INVALID_SERIALIZATION_FORMAT;
    }

    void memcpy_nothrow(vma_ptr_t dst, vma_ptr_t src, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy(dst, src, sz));
    }

    auto memset(vma_ptr_t dst, int c, size_t sz) noexcept -> exception_t{

        using namespace dg::network_virtual_device;

        if (is_host_ptr(dst)){
            auto dst_ptr = devirtualize_host_ptr(dst);
            return network_memops_clib::memset_host(dst_ptr, c, sz);
        } 

        if (is_cuda_ptr(dst)){
            auto dst_ptr = devirtualize_cuda_ptr(dst);
            return network_memops_clib::memset_cuda(dst_ptr, c, sz);
        }

        if (is_fsys_ptr(dst)){
            auto dst_ptr = devirtualize_fsys_ptr(dst);
            return network_memops_fsys::memset_fsys(dst_ptr, c, sz);
        }

        if (is_cufs_ptr(dst)){
            auto dst_ptr = devirtualize_cufs_ptr(dst);
            return network_memops_cufs::memset_cufs(dst_ptr, c, sz);
        }

        return dg::network_exception::INVALID_SERIALIZATION_FORMAT;
    }

    void memset_nothrow(vma_ptr_t dst, int c, size_t sz) noexcept{

        dg::network_exception_handler::nothrow_log(memset(dst, c, sz));
    }
} 

#endif