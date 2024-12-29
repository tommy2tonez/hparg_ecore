#ifndef __NETWORK_CUDAMAP_X_H__
#define __NETWORK_CUDAMAP_X_H__

#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include "stdx.h"
#include "network_cudamap_x_impl1.h"

namespace dg::network_cudamap_x{

    //alright - consider this scenrio
    //this has specialized application
    //we have uma_ptr_t -> <vma_ptr_t1, vma_ptr_t2> vma_ptr_t1 -> cuda_ptr_t, vma_ptr_t2 -> cutf_ptr_t (bijective the cuda_ptr_t) -> cupm_ptr_t
    //assume the memregion segments are injective w.r.t uma_ptr_t
    //we are wasting memory transferring cutf_ptr_t -> cuda_ptr_t and cuda_ptr_t -> cutf_ptr_t
    //the logic gets messy if we try to solve it here (which is fine - even though we aren't doing this cleanly - uma_ptr_t acquisition guarantees serialization accesses for that case)
    //we need to add the logic evict memory for cudamap_x
    //we are maintaining evict_try for this map but not other map because it would limit future implementations

    inline std::unique_ptr<dg::network_cuda_impl1::interface::ConcurrentMapInterface> map_instance; 
    using map_resource_handle_t = dg::network_cudamap_impl1::model::ConcurrentMapResource; 

    void init(){

        stdx::memtransaction_guard tx_grd;
    }

    void deinit() noexcept{

        stdx::memtransaction_guard tx_grd;
        map_instance = nullptr;
    }

    auto get_map_instance() noexcept -> dg::network_cuda_impl1::interface::ConcurrentMapInterface *{

        std::atomic_signal_fence(std::memory_order_acquire); //
        return map_instance.get();
    }

    auto map(cuda_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        return get_map_instance()->map(ptr);
    }

    auto evict_try(cuda_ptr_t ptr) noexcept -> std::expected<bool, exception_t>{

        return get_map_instance()->evict_try(ptr);
    }

    auto map_nothrow(cuda_ptr_t ptr) noexcept -> map_resource_handle_t{

        return dg::network_exception_handler::nothrow_log(get_map_instance()->map(ptr));
    }

    void map_release(map_resource_handle_t map_resource) noexcept{

        get_map_instance()->unmap(map_resource);
    }

    auto remap_try(map_resource_handle_t map_resource, cuda_ptr_t new_ptr) noexcept -> std::expected<std::optional<map_resource_handle_t>, exception_t>{

        return get_map_instance()->remap_try(map_resource, new_ptr);
    }

    auto remap(map_resource_handle_t map_resource, cuda_ptr_t new_ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        auto _map_instance  = get_map_instance();
        auto remap_try_rs   = _map_instance->remap_try(map_resource, new_ptr);

        if (remap_try_rs.has_value() && remap_try_rs.value().has_value()){ //this is precisely the problem with try catch - we prefer if else to handle errors
            return remap_try_rs.value().value();
        }

        auto new_map_rs = _map_instance->map(new_ptr);

        if (new_map_rs.has_value()){
            _map_instance->unmap(map_resource);
        }

        return new_map_rs;
    }

    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        network_cudamap_x::map_release(map_resource);
    };

    auto map_safe(cuda_ptr_t ptr) noexcept -> std::expected<dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        auto map_rs = get_map_instance()->map(ptr);

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto map_safe_nothrow(cuda_ptr_t ptr) noexcept -> dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>{

        return dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>(dg::network_exception_handler::nothrow_log(get_map_instance()->map(ptr)), map_release_lambda);
    }

    auto get_cupm_ptr(map_resource_handle_t map_resource) noexcept -> cupm_ptr_t{

        return map_resource.ptr();
    }
}

#endif