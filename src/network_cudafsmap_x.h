#ifndef __NETWORK_CUDAFSMAP_X_H__
#define __NETWORK_CUDAFSMAP_X_H__

#include "network_cudafsmap_x_impl1.h"
#include "network_exception_handler.h"
#include "network_std_container.h"

namespace dg::network_cudafsmap_x{

    using cufs_ptr_t            = dg::network_pointer::cufs_ptr_t; 
    using map_resource_handle_t = dg::network_cudafsmap_x_impl1::model::ConcurrentMapResource; 

    inline dg::network_cudafsmap_x_impl1::interface::ConcurrentMapInterface * volatile map_instance;  

    void init(const dg::unordered_map<cufs_ptr_t, std::filesystem::path>& bijective_alias_map, 
              const dg::unordered_map<std::filesystem::path, int>& gpu_platform_map,
              size_t memregion_sz, 
              double ram_to_disk_ratio, 
              size_t distribution_factor){

        stdx::memtransaction_guard tx_grd;
        auto tmp_map_instance   = dg::network_cudafsmap_x_impl1::make(bijective_alias_map, gpu_platform_map, memregion_sz, ram_to_disk_ratio, distribution_factor);
        map_instance            = tmp_map_instance.get();
        tmp_map_instance.release();
    }

    void deinit() noexcept{

        stdx::memtransaction_guard tx_grd;
        delete map_instance;
    }

    auto get_map_instance() noexcept -> dg::network_cudafsmap_x_impl1::interface::ConcurrentMapInterface *{

        std::atomic_signal_fence(std::memory_order_acquire); //we are doing double protection
        return map_instance;
    } 

    auto map(cufs_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        return get_map_instance()->map(ptr);
    }

    auto map_nothrow(cufs_ptr_t ptr) noexcept -> map_resource_handle_t{

        return dg::network_exception_handler::nothrow_log(get_map_instance()->map(ptr));
    }

    void map_release(map_resource_handle_t map_resource) noexcept{

        get_map_instance()->unmap(map_resource);
    }

    auto remap_try(map_resource_handle_t map_resource, cufs_ptr_t ptr) noexcept -> std::expected<std::optional<map_resource_handle_t>, exception_t>{

        return get_map_instance()->remap_try(map_resource, ptr);
    }

    auto remap(map_resource_handle_t map_resource, cufs_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        auto _map_instance  = get_map_instance();
        auto remap_rs       = _map_instance->remap_try(map_resource, ptr); 

        if (remap_rs.has_value() && remap_rs.value().has_value()){
            return map_rs.value().value();
        }
        
        std::expected<map_resource_handle_t, exception_t> new_map_resource = _map_instance->map(ptr);

        if (new_map_resource.has_value()){
            _map_instance->unmap(map_resource);
        }

        return new_map_resource;
    }

    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        map_release(map_resource);
    };

    auto map_safe(cufs_ptr_t ptr) noexcept -> std::expected<dg::network_genult::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        std::expected<map_resource_handle_t, exception_t> map_rs = map(ptr);

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return dg::network_genult::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto map_nothrow_safe(cufs_ptr_t ptr) noexcept -> dg::network_genult::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>{

        return {map_nothrow(ptr), map_release_lambda};
    }

    auto get_cuda_ptr(map_resource_handle_t map_resource) noexcept -> cuda_ptr_t{

        return map_resource.ptr();
    }
}

#endif