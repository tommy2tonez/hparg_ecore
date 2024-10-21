#ifndef __NETWORK_CUDAFSMAP_X_H__
#define __NETWORK_CUDAFSMAP_X_H__

#include "network_cudafsmap_x_impl1.h"
#include "network_exception_handler.h"

namespace dg::network_cudafsmap_x{

    using cufs_ptr_t            = dg::network_pointer::cufs_ptr_t; 
    using map_resource_handle_t = dg::network_cudafsmap_x_impl1::model::ConcurrentMapResource; 

    inline std::unique_ptr<dg::network_cudafsmap_x_impl1::interface::ConcurrentMapInterface> map_instance{};  

    void init(const dg::unordered_map<cufs_ptr_t, std::filesystem::path>& bijective_map, size_t memregion_sz, double ram_to_disk_ratio, size_t distribution_factor){

        map_instance = dg::network_cudafsmap_x_impl1::make(bijective_map, memregion_sz, ram_to_disk_ratio, distribution_factor);
    }

    void deinit() noexcept{

        map_instance = nullptr;
    }

    auto map(cufs_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        return map_instance->map(ptr);
    }

    auto map_nothrow(cufs_ptr_t ptr) noexcept -> map_resource_handle_t{

        return dg::network_exception_handler::nothrow_log(map_instance->map(ptr));
    }

    void map_release(map_resource_handle_t map_resource) noexcept{

        map_instance->map_release(map_resource);
    }

    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        map_release(map_resource);
    };

    auto map_safe(cufs_ptr_t ptr) noexcept -> std::expected<dg::network_genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        std::expected<map_resource_handle_t, exception_t> map_rs = map(ptr);

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return dg::network_genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto map_nothrow_safe(cufs_ptr_t ptr) noexcept -> dg::network_genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>{

        return {map_nothrow(ptr), map_release_lambda};
    }

    auto get_cuda_ptr(map_resource_handle_t map_resource) noexcept -> cuda_ptr_t{

        return map_resource.ptr();
    }
}

#endif