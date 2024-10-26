#ifndef __NETWORK_ALLOCATION_CLIB_H__
#define __NETWORK_ALLOCATION_CLIB_H__

#include <cstring> 
#include <stdint.h>
#include <stdlib.h>
#include <exception>
#include "network_exception.h"
#include "network_log.h"
#include <memory>
#include "network_utility.h"
#include "stdx.h"

namespace dg::network_allocation_clib{

    static inline constexpr size_t DEFAULT_CUDA_ALIGNMENT_SZ = size_t{1} << 10; 
    using cuda_ptr_header_t = uint64_t; 

    auto c_malloc(size_t blk_sz) -> void *{

        if (blk_sz == 0u){
            return nullptr;
        }

        void * rs = std::malloc(blk_sz);

        if (!rs){
            throw std::bad_alloc();
        }

        return rs;
    }

    auto c_aligned_malloc(size_t alignment_sz, size_t blk_sz) -> void *{

        if constexpr(DEBUG_MODE_FLAG){
            if (!dg::memult::is_pow2(alignment_sz)){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        if (blk_sz == 0u){
            return nullptr;
        }

        void * rs = std::aligned_alloc(alignment_sz, blk_sz); //alignment_sz promotion is std::aligned_alloc responsibility - std does not specify upperbound for alignment_sz - need to verify this

        if (!rs){
            throw std::bad_alloc(); //bad_alloc always semantically equivalents to not enough memory - important to make this a sole exception for OOM - alignment is a precond - not an exception
        }

        return rs;
    }

    void c_free(void * ptr) noexcept{
        
        if (!ptr){
            return;
        }

        std::free(ptr); //std::free accidentally takes nullptr as valid arg - this could be a premature optimization
    }

    auto cuda_aligned_malloc(size_t alignment_sz, size_t blk_sz) -> void *{

        if constexpr(DEBUG_MODE_FLAG){
            if (!dg::memult::is_pow2(alignment_sz)){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        if (blk_sz == 0u){
            return nullptr;
        }

        size_t adjusted_blk_sz = blk_sz + (alignment_sz - 1) + sizeof(cuda_ptr_header_t); //this guarantee that align(fwd(buf, cuda_ptr_header_t), alignment_sz) return aligned ptr - and write cuda_ptr_header_t to std::advance(ptr, -sizeof(cuda_ptr_header_t)) 
        void * rs{};
        exception_t err = dg::network_cuda_controller::cuda_malloc(&rs, adjusted_blk_sz);

        if (dg::network_exception::is_failed(err)){
            throw std::bad_alloc();
        }

        auto mem_backout    = [rs]() noexcept{
            dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::cuda_free(rs));
        };
        auto mem_guard      = stdx::resource_guard(std::move(mem_backout));
        void * aligned_ptr  = dg::memult::align(dg::memult::badvance(rs, sizeof(cuda_ptr_header_t)), alignment_sz);
        void * header_ptr   = dg::memult::badvance(aligned_ptr, -static_cast<intmax_t>(sizeof(cuda_ptr_header_t)));
        auto header         = static_cast<cuda_ptr_header_t>(dg::memult::distance(rs, aligned_ptr));
        exception_t cpy_err = dg::network_cuda_controller::cuda_memcpy(header_ptr, &header, sizeof(cuda_ptr_header_t), cudaMemcpyHostToDevice);

        if (dg::network_exception::is_failed(cpy_err)){
            throw std::bad_alloc();
        }

        mem_guard.release();
        return aligned_ptr;
    }

    auto cuda_malloc(size_t blk_sz) -> void *{

        return cuda_aligned_malloc(DEFAULT_CUDA_ALIGNMENT_SZ, blk_sz);
    }

    void cuda_free(void * ptr) noexcept{

        if (!ptr){
            return;
        }

        void * header_ptr   = dg::memult::badvance(ptr, -static_cast<intmax_t>(sizeof(cuda_ptr_header_t)));
        auto header         = cuda_ptr_header_t{};
        exception_t err     = dg::network_cuda_controller::cuda_memcpy(&header, header_ptr, sizeof(cuda_ptr_header_t), cudaMemcpyDeviceToHost);
        dg::network_exception_handler::nothrow_log(err);
        void * org_ptr      = dg::memult::badvance(ptr, -static_cast<intmax_t>(header));
        err                 = dg::network_cuda_controller::cuda_free(org_ptr);
        dg::network_exception_handler::nothrow_log(err);
    }

    auto craii_malloc(size_t blk_sz) -> std::unique_ptr<char[], decltype(&c_free)>{

        return {static_cast<char *>(c_malloc(blk_sz)), c_free};
    }

    auto craii_aligned_malloc(size_t alignment_sz, size_t blk_sz) -> std::unique_ptr<char[], decltype(&c_free)>{

        return {static_cast<char *>(c_aligned_malloc(alignment_sz, blk_sz)), c_free};
    }

    auto cudaraii_malloc(size_t blk_sz) -> std::unique_ptr<void, decltype(&cuda_free)>{

        return {cuda_malloc(blk_sz), cuda_free};
    }

    auto cudaraii_aligned_malloc(size_t alignment_sz, size_t blk_sz) -> std::unique_ptr<void, decltype(&cuda_free)>{

        return {cuda_aligned_malloc(alignment_sz, blk_sz), cuda_free};
    }
}

namespace dg::network_allocation_cuda_x{

    struct CudaGlobalPtr{
        int device_id;
        void * dev_ptr;
    };

    using cuda_ptr_t = CudaGlobalPtr;
  
    auto cuda_malloc(int device_id, size_t blk_sz) -> CudaGlobalPtr{

        bool status = dg::network_exception_handler::throw_nolog(dg::network_cuda_controller::cuda_is_valid_device(&device_id, 1u)); 

        if (!status){
            dg::network_exception::throw_exception(dg::network_exception::BAD_CUDA_DEVICE_ACCESS);
        }

        auto cuda_device_grd = dg::network_cuda_controller::lock_env_guard(&device_id, 1u);  
        void * mem = dg::network_allocation_clib::cuda_malloc(blk_sz); 

        return CudaGlobalPtr{device_id, mem};
    }

    auto cuda_aligned_malloc(int device_id, size_t alignment_sz, size_t blk_sz) -> CudaGlobalPtr{

        bool status = dg::network_exception_handler::throw_nolog(dg::network_cuda_controller::cuda_is_valid_device(&device_id, 1u));

        if (!status){
            dg::network_exception::throw_exception(dg::network_exception::BAD_CUDA_DEVICE_ACCESS);
        }

        auto cuda_device_grd = dg::network_cuda_controller::lock_env_guard(&device_id, 1u);
        void * mem = dg::network_allocation_clib::cuda_aligned_malloc(alignment_sz, blk_sz);

        return CudaGlobalPtr{device_id, mem};
    }

    auto cuda_free(CudaGlobalPtr ptr) noexcept{

        auto cuda_device_grd = dg::network_cuda_controller::lock_env_guard(&ptr.device_id, 1u);
        dg::network_allocation_clib::cuda_free(ptr.dev_ptr);
    }

    void cuda_memset(CudaGlobalPtr ptr, int c, size_t sz){

        auto cuda_device_grd = dg::network_cuda_controller::lock_env_guard(&ptr.device_id, 1u);
        dg::network_exception_handler::throw_nolog(dg::network_cuda_controller::cuda_memset(ptr.dev_ptr, c, sz)); //
    }

    void cuda_memcpy(CudaGlobalPtr dst, CudaGlobalPtr src, size_t sz){

        dg::network_exception_handler::throw_nolog(dg::network_cuda_controller::cuda_mempy_peer(dst.dev_ptr, dst.device_id, src.dev_ptr, src.device_id, sz));
    } 

    auto cudaraii_malloc(int device_id, size_t blk_sz) -> dg::network_genult::nothrow_immutable_unique_raii_wrapper<CudaGlobalPtr, decltype(&cuda_free)>{ //i hate the nothrow_unique_raii yet this is the way

        return {cuda_malloc(device_id, blk_sz), cuda_free};
    }

    auto cudaraii_aligned_malloc(int device_id, size_t alignment_sz, size_t blk_sz) -> dg::network_genult::nothrow_immutable_unique_raii_wrapper<CudaGlobalPtr, decltype(&cuda_free)>{

        return {cuda_aligned_malloc(device_id, alignment_sz, blk_sz), cuda_free};
    }
}

#endif