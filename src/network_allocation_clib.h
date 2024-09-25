#ifndef __NETWORK_ALLOCATION_CLIB_H__
#define __NETWORK_ALLOCATION_CLIB_H__

#include <cstring> 
#include <stdint.h>
#include <stdlib.h>
#include <exception>
#include "network_exception.h"
#include "network_log.h"
#include <memory>

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
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMalloc(&rs, adjusted_blk_sz));

        if (dg::network_exception::is_failed(err)){
            dg::network_exception::throw_exception(err);
        }

        auto mem_backout    = [rs]() noexcept{
            exception_t err = dg::network_exception::wrap_cuda_exception(cudaFree(rs));
            dg::network_exception_handler::nothrow_log(err);
        };
        auto mem_guard      = dg::network_genult::resource_guard(std::move(mem_backout));
        void * aligned_ptr  = dg::memult::align(dg::memult::badvance(rs, sizeof(cuda_ptr_header_t)), alignment_sz);
        void * header_ptr   = dg::memult::badvance(aligned_ptr, -static_cast<intmax_t>(sizeof(cuda_ptr_header_t)));
        auto header         = static_cast<cuda_ptr_header_t>(dg::memult::distance(rs, aligned_ptr));

        err = dg::network_exception::wrap_cuda_exception(cudaMemcpy(header_ptr, &header, sizeof(cuda_ptr_header_t), cudaMemcpyHostToDevice));

        if (dg::network_exception::is_failed(err)){
            dg::network_exception::throw_exception(err);
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
        exception_t err     = dg::network_exception::wrap_cuda_exception(cudaMemcpy(&header, header_ptr, sizeof(cuda_ptr_header_t), cudaMemcpyDeviceToHost));
        dg::network_exception_handler::nothrow_log(err);
        void * org_ptr      = dg::memult::badvance(ptr, -static_cast<intmax_t>(header));
        err                 = dg::network_exception::wrap_cuda_exception(cudaFree(org_ptr));
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

#endif