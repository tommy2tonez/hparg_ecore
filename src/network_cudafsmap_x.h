#ifndef __NETWORK_CUDAFSMAP_X_H__
#define __NETWORK_CUDAFSMAP_X_H__

#include "network_cudafsmap_x_impl1.h"
#include "network_exception_handler.h"

namespace dg::network_cudafsmap_x{

    inline std::unique_ptr<dg::network_cudamap_x_impl1::interface::ConcurrentMapInterface> map_instance{}; 
    using map_resource_handle_t = dg::network_cudamap_x_impl1::model::ConcurrentMapResource; 

    void init(){

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

    auto map_safe(cufs_ptr_t ptr) noexcept -> std::expected<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(&map_release)>, exception_t>{

        std::expected<map_resource_handle_t, exception_t> map_rs = map(ptr);

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return {std::in_place_t{}, map_rs.value(), map_release}; 
    }

    auto map_nothrow_safe(cufs_ptr_t ptr) noexcept -> dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(&map_release)>{

        return {map_nothrow(ptr), map_release};
    }

    auto get_cuda_ptr(map_resource_handle_t map_resource) noexcept -> cuda_ptr_t{

        return map_resource.ptr();
    }
}

#endif