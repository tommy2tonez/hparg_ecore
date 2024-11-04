#ifndef __NETWORK_KERNEL_MAP_X_H__
#define __NETWORK_KERNEL_MAP_X_H__

//define HEADER_CONTROL 10

#include <stdint.h>
#include <stddef.h> 
#include <filesystem>
#include "network_exception.h"
#include "network_kernelmap_x_impl1.h" 
#include "network_raii_x.h"
#include "network_type_traits_x.h"
#include "network_std_container.h"

namespace dg::network_kernelmap_x{
    
    using fsys_ptr_t                = dg::network_pointer::fsys_ptr_t;   
    using map_resource_handle_t     = dg::network_kernelmap_x_impl1::model::ConcurrentMapResource;

    inline std::unique_ptr<dg::network_kernelmap_x_impl1::interface::ConcurrentMapInterface> map_instance{}; 

    void init(const dg::unordered_map<fsys_ptr_t, std::filesystem::path>& bijective_alias_map, size_t memregion_sz, double ram_to_disk_ratio, size_t distribution_factor){

        map_instance = dg::network_kernelmap_x_impl1::make(bijective_alias_map, memregion_sz, ram_to_disk_ratio, distribution_factor);
    }

    void deinit() noexcept{
        
        map_instance = nullptr;
    }

    auto map(fsys_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        return map_instance->map(ptr);
    }

    auto map_nothrow(fsys_ptr_t ptr) noexcept -> map_resource_handle_t{

        return dg::network_exception_handler::nothrow_log(network_kernelmap_x::map(ptr));
    }

    void map_release(map_resource_handle_t map_resource) noexcept{

        map_instance->unmap(map_resource);
    }
    
    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        network_kernelmap_x::map_release(map_resource);
    };

    auto map_safe(fsys_ptr_t ptr) noexcept -> std::expected<dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        auto map_rs = network_kernelmap_x::map(ptr);

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    } 

    auto map_nothrow_safe(fsys_ptr_t ptr) noexcept -> dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>{

        return dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(network_kernelmap_x::map_nothrow(ptr), map_release_lambda);
    }

    auto get_host_ptr(map_resource_handle_t map_resource) noexcept -> void *{

        return map_resource.ptr();
    }
}

#endif