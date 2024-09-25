#ifndef __DG_NETWORK_CUDA_CONTROLLER_H__
#define __DG_NETWORK_CUDA_CONTROLLER_H__

#include "network_utility.h"
#include <atomic>
#include <mutex>

namespace dg::network_cuda_controller{
    
    struct ControllerResource{
        std::vector<int> prev_dev;
        std::vector<int> cur_dev; 
        std::recursive_mutex mtx;
    };

    inline ControllerResource controller_resource{};

    auto set_cuda_device(int * device, size_t sz) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(controller_resource.mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(device, sz));

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        controller_resource.prev_dev    = std::move(controller_resource.cur_dev);
        controller_resource.cur_dev     = std::vector<int>(device, device + sz);
    }

    auto cuda_malloc(void ** ptr, size_t blk_sz) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(controller_resource.mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMalloc(ptr, blk_sz));

        return err;
    }

    auto cuda_free(void * ptr) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(controller_resource.mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaFree(ptr));

        return err;
    }

    auto cuda_memset(void * ptr, int value, size_t sz) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(controller_resource.mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMemset(ptr, value, sz));

        return err;
    } 

    auto cuda_memcpy(void * dst, const void * src, size_t sz, cudaMemcpyKind kind) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(controller_resource.mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMemcpy(dst, src, sz, kind));

        return err;
    } 

    auto cuda_memcpy_peer(void * dst, int dst_id, const void * src, size_t src_id, size_t sz) noexcept -> exception_t{

        return dg::network_exception::wrap_cuda_exception(cudaMemcpyPeer(dst, dst_id, src, src_id, sz));
    }

    //--
    auto get_cuda_device_guard(int device_id){ // this is hard to implement correctly

        auto resource_backout = []() noexcept{
            controller_resource.cur_dev     = std::move(controller_resource.prev_dev);
            controller_resource.prev_dev    = {};
            exception_t err                 = dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(controller_resource.cur_dev.data(), controller_resource.cur_dev.size())); 
            dg::network_exception_handler::nothrow_log(err);
            controller_resource.mtx.unlock();
        };

        controller_resource.mtx.lock();
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaSetDevice(device_id)); 

        if (dg::network_exception::is_failed(err)){
            controller_resource.mtx.unlock();
            dg::network_exception::throw_exception(err);
        }

        controller_resource.prev_dev    = std::move(controller_resource.cur_dev);
        controller_resource.cur_dev     = {device_id}; //has to be a no-throw operation

        return dg::network_genult::resource_guard(resource_backout);
    }

}

#endif