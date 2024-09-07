#ifndef __NETWORK_KERNEL_MAP_X_H__
#define __NETWORK_KERNEL_MAP_X_H__

#include <stdint.h>
#include <stddef.h> 
#include <filesystem>
#include "network_exception.h"
#include "network_kernelmap_x_impl1.h" 
#include "network_utility.h"
#include "network_type_traits_x.h"

namespace dg::network_kernelmap_x{
    
    using fsys_ptr_t                = uint32_t; 
    using exception_t               = dg::network_exception::exception_t;  
    using map_resource_handle_t     = dg::network_kernelmap_x_impl1::model::MapResource;

    static_assert(dg::is_immutable_resource_handle_v<map_resource_handle_t>);

    inline std::unique_ptr<dg::network_kernelmap_x_impl1::interface::MapInterface> map_instance{}; 

    template <size_t MEMREGION_SZ>
    void init(fsys_ptr_t * region, std::filesystem::path * path, fsys_device_id_t * device_id, size_t n, std::integral_constant<size_t, MEMREGION_SZ>){

        auto logger     = dg::network_log_scope::critical_terminate();
        map_instance    = dg::network_kernelmap_x_impl1::make(region, path, device_id, n, std::integral_constant<size_t, MEMREGION_SZ>{});
        logger.release();
    }

    auto map(fsys_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        return map_instance->map(ptr);
    }

    auto map_nothrow(fsys_ptr_t ptr) noexcept -> map_resource_handle_t{

        return dg::network_exception_handler::nothrow_log(map(ptr));
    }

    void map_release(map_resource_handle_t map_resource) noexcept{

        map_instance->map_release(map_resource);
    }
    
    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        map_release(map_resource);
    };

    auto map_relguard(map_resource_handle_t map_resource) noexcept{
    
        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            map_release(map_resource);
        }

        return std::unique_ptr<int, decltype(destructor)>(&i, std::move(destructor));
    }

    auto map_safe(fsys_ptr_t ptr) noexcept -> std::expected<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        auto map_rs = map(ptr);

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    } 

    auto map_nothrow_safe(fsys_ptr_t ptr) noexcept -> dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>{

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_nothrow(ptr), map_release_lambda);
    }

    auto get_host_ptr(map_resource_handle_t map_resource) noexcept -> void *{

        return map_resource.ptr();
    }

    auto get_host_const_ptr(map_resource_handle_t map_resource) noexcept -> const void *{ //deprecate usage of const void * - for interface consistency

        return map_resource.const_ptr();
    }
}

#endif