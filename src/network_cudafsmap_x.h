#ifndef __NETWORK_CUDAFSMAP_X_H__
#define __NETWORK_CUDAFSMAP_X_H__

namespace dg::network_cudafsmap_x{

    void init(){

    }

    auto map(cufs_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

    }

    auto map_nothrow(cufs_ptr_t ptr) noexcept -> map_resource_handle_t{

    }

    void map_release(map_resource_handle_t) noexcept{

    }

    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        map_release(map_resource);
    };

    auto map_safe(cufs_ptr_t ptr) noexcept -> std::expected<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

    }

    auto map_nothrow_safe(cufs_ptr_t ptr) noexcept -> dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>{

    }

    auto get_cuda_ptr(map_resource_handle_t) noexcept -> cuda_ptr_t{

    }
}

#endif