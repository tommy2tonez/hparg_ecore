#ifndef __NETWORK_KERNEL_MAP_X_H__
#define __NETWORK_KERNEL_MAP_X_H__

#include <stdint.h>
#include <stddef.h> 
#include <filesystem>
#include "network_exception.h"
#include "network_kernelmap_x_impl1.h"

namespace dg::network_kernelmap_x{
    
    using fsys_ptr_t                = uint32_t; 
    using exception_t               = dg::network_exception::exception_t;  
    using map_resource_handler_t    = dg::network_kernelmap_x_impl1::model::MapResource;

    inline std::unique_ptr<dg::network_kernelmap_x_impl1::interface::MapInterface> map_instance{}; 

    template <size_t MEMREGION_SZ>
    void init(fsys_ptr_t * region, std::filesystem::path * path, fsys_device_id_t * device_id, size_t n, std::integral_constant<size_t, MEMREGION_SZ>){

        auto logger     = dg::network_log_scope::critical_terminate();
        map_instance    = dg::network_kernelmap_x_impl1::make(region, path, device_id, n, std::integral_constant<size_t, MEMREGION_SZ>{});
        logger.release();
    }

    auto map_try(fsys_ptr_t ptr) noexcept -> std::expected<map_resource_handler_t, exception_t>{

        return map_instance->map_try(ptr);
    }

    auto map_wait(fsys_ptr_t ptr) noexcept -> map_resource_handler_t{

        return map_instance->map_wait(ptr);
    }

    void map_release(map_resource_handler_t map_resource) noexcept{

        map_instance->map_release(map_resource);
    }

    auto map_relguard(map_resource_handler_t map_resource) noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            map_release(map_resource);
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    auto get_host_ptr(map_resource_handler_t map_resource) noexcept -> void *{

        return map_resource.ptr();
    }

    auto get_host_const_ptr(map_resource_handler_t map_resource) noexcept -> const void *{

        return map_resource.const_ptr();
    }
}

#endif