#ifndef __DG_NETWORK_CUDA_CONTROLLER_H__
#define __DG_NETWORK_CUDA_CONTROLLER_H__

#include "network_utility.h"
#include <atomic>
#include <mutex>
#include "network_std_container.h"
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <deque>
#include "network_concurrency.h"
#include "network_exception.h"
#include "stdx.h"

namespace dg::network_cuda_stream{

    static inline constexpr uint8_t SYNC_FLAG = 0b001;

    struct CudaStreamHandle{
        cudaStream_t cuda_stream;
        uint8_t flags;
    };

    auto cuda_stream_create(uint8_t flags) noexcept -> std::expected<CudaStreamHandle, exception_t>{

        cudaStream_t cuda_stream    = {};
        exception_t err             = dg::network_exception::wrap_cuda_exception(cudaStreamCreate(&cuda_stream));

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return CudaStreamHandle{cuda_stream, flags};
    }

    void cuda_stream_close(CudaStreamHandle handle) noexcept{

        bool has_synchronization = (handle.flags & SYNC_FLAG) != 0u;

        if (has_synchronization){
            dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaStreamSynchronize(handle.cuda_stream)));
        }

        dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaStreamDestroy(handle.cuda_stream)));
    }
 
    auto cuda_stream_raiicreate(uint8_t flags) noexcept -> std::expected<dg::network_genult::unique_resource<CudaStreamHandle, decltype(&cuda_stream_close)>, exception_t>{

        std::expected<CudaStreamHandle, exception_t> handle = cuda_stream_create(flags);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return dg::network_genult::unique_resource<CudaStreamHandle, decltype(&cuda_stream_close)>(std::move(handle.value()), cuda_stream_close);
    }

    auto cuda_stream_get_legacy(CudaStreamHandle handle) noexcept -> cudaStream_t{

        return handle.cuda_stream;
    }
} 

namespace dg::network_cuda_controller{
    
    //this is the sole interface to communicate with cuda_runtime lib - to allow synchronized accesses to cuda_runtime lib 
    //this is fine for the first draft - be back for improvement later

    struct ControllerResource{
        dg::vector<int> device;
        const size_t total_device_count;
        std::unique_ptr<std::recursive_mutex> mtx;
    };

    inline std::unique_ptr<ControllerResource> controller_resource{};

    auto init() noexcept -> exception_t{

        int count{};

        exception_t err = dg::network_exception::wrap_cuda_exception(cudaGetDeviceCount(&count));

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        if (count <= 0){
            return dg::network_exception::CUDA_NOT_SUPPORTED;
        }

        exception_t err = dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(nullptr, 0u)); 

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        controller_resource = std::make_unique<ControllerResource>(ControllerResource{{}, count, std::make_unique<std::recursive_mutex>()}); //resource allocation error need to be isolated and made noexcept - instead of relying on init noexcept (bad practice)
        return dg::network_exception::SUCCESS;
    }

    void deinit() noexcept{

        //cuda does not build this for deinitialization - sorry but that's the truth - rather this to be program-lifetime than to add redundant logics here
    }

    auto cuda_is_valid_device(int * device, size_t sz) noexcept -> std::expected<bool, exception_t>{ // fine - this is for interface consistency - does not neccessarily need to return error

        if (sz == 0u){
            return true; //is default according to MAN
        }

        dg::unordered_set<int> device_set(device, device + sz, sz);
        
        if (device_set.size() != sz){ //MAN does not specify whether device *, size_t has to be as valid set or not - stricter req
            return false;
        }

        const int MIN_DEVICE_ID = 0;
        const int MAX_DEVICE_ID = controller_resource->total_device_count - 1;
        auto unmet_cond         = [=](int cur){return std::clamp(cur, MIN_DEVICE_ID, MAX_DEVICE_ID) != cur;};

        return std::find_if(device_set.begin(), device_set.end(), unmet_cond) != device_set.end();
    }

    auto cuda_set_device(int * device, size_t sz) noexcept -> exception_t{

        stdx::xlock_guard<std::recursive_mutex> lck_grd(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(device, sz));

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        controller_resource->device = dg::vector<int>(device, device + sz);
        return dg::network_exception::SUCCESS;
    }

    auto cuda_malloc(void ** ptr, size_t blk_sz) noexcept -> exception_t{

        stdx::xlock_guard<std::recursive_mutex> lck_grd(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMalloc(ptr, blk_sz));

        return err;
    }

    auto cuda_free(void * ptr) noexcept -> exception_t{

        stdx::xlock_guard<std::recursive_mutex> lck_grd(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaFree(ptr));

        return err;
    }

    auto cuda_memset(void * dst, int c, size_t sz) noexcept -> exception_t{

        stdx::xlock_guard<std::recursive_mutex> lck_grd(*controller_resource->mtx);
        auto stream     = dg::network_cuda_stream::cuda_stream_raiicreate(dg::network_cuda_stream::SYNC_FLAG);

        if (!stream.has_value()){
            return stream.error();
        }

        return dg::network_exception::wrap_cuda_exception(cudaMemsetAsync(dst, c, sz, dg::network_cuda_stream::cuda_stream_get_legacy(stream.value())));
    } 

    auto cuda_memcpy(void * dst, const void * src, size_t sz, cudaMemcpyKind kind) noexcept -> exception_t{

        stdx::xlock_guard<std::recursive_mutex> lck_grd(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMemcpy(dst, src, sz, kind));

        return err;
    } 

    auto cuda_memcpy_peer(void * dst, int dst_id, const void * src, size_t src_id, size_t sz) noexcept -> exception_t{

        auto stream = network_cuda_stream::cuda_stream_raiicreate(dg::network_cuda_stream::SYNC_FLAG);

        if (!stream.has_value()){
            return stream.error();
        } 

        return dg::network_exception::wrap_cuda_exception(cudaMemcpyPeerAsync(dst, dst_id, src, src_id, sz, dg::network_cuda_stream::cuda_stream_get_legacy(stream.value())));
    }
   
    auto cuda_synchronize() noexcept -> exception_t{

        stdx::xlock_guard<std::recursive_mutex> lck_grd(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaDeviceSynchronize());

        return err;
    }

    //this is protected interface - don't invoke if cannot guarantee lock hierarchical order - risking deadlock if done otherwise - bad practice

    auto lock_env_guard(int * device, size_t sz) noexcept{

        controller_resource->mtx->lock();
        //UB
        auto old_device = controller_resource->device;

        auto resource_backout = [old_device]() noexcept{
            controller_resource->device = old_device;
            dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(controller_resource->device.data(), controller_resource->device.size())));
            //UB
            controller_resource->mtx->unlock();
        };

        dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(device, sz)));
        return stdx::resource_guard(std::move(resource_backout)); //not semantically accurate - yet functionally accurate - improvement required
    }
    //----
}

#endif