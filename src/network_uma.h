#ifndef __NETWORK_UMA_H__
#define __NETWORK_UMA_H__

#include <stdint.h>
#include <stddef.h>
#include "network_uma_definition.h"
#include "network_uma_tlb.h"
#include "network_exception.h"
#include "network_exception_handler.h"
#include "network_randomizer.h" 
#include "stdx.h"
#include "network_raii_x.h"

namespace dg::network_uma{

    using device_id_t           = uint8_t;
    using uma_ptr_t             = uint64_t;
    using device_ptr_t          = void *;
    
    static inline constexpr size_t MEMREGION_SZ         = size_t{0u};
    static inline constexpr size_t PROXY_COUNT          = size_t{0u};
    static inline constexpr size_t MAX_PROXY_PER_REGION = 32u;

    struct signature_dg_network_uma{}; 
      
    using tlb_factory                   = dg::network_uma_tlb::v1::Factory<signature_dg_network_uma, device_id_t, uma_ptr_t, device_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>; 
    using tlb_instance                  = typename tlb_factory::tlb;
    using tlbdirect_instance            = typename tlb_factory::tlb_direct;
    using uma_ptr_access                = typename tlb_factory::uma_ptr_access;
    using metadata_getter               = typename tlb_factory::uma_metadata; 
    using map_resource_handle_t         = typename tlb_factory::map_resource_handle_t; 

    static_assert(std::is_trivial_v<map_resource_handle_t>);
    static_assert(dg::is_immutable_resource_handle_v<map_resource_handle_t>);

    void init(uma_ptr_t * host_region, vma_ptr_t * device_region, device_id_t * device_id, size_t n){

        // tlb_factory::init(host_region, device_region, device_id, n);
    }
    
    void deinit() noexcept{

    }
    
    auto map_direct(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<vma_ptr_t, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(device_id, ptr); 

        if (dg::network_exception::is_failed(ptrchk)){ [[unlikely]]
            return std::unexpected(ptrchk);
        }

        return tlbdirect_instance::map(device_id, ptr);
    }

    auto map_try(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<std::optional<map_resource_handle_t>, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(device_id, ptr);

        if (dg::network_exception::is_failed(ptrchk)){ [[unlikely]]
            return std::unexpected(ptrchk);
        }
        
        return tlb_instance::map_try(device_id, ptr);
    }

    auto map_wait(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(device_id, ptr);

        if (dg::network_exception::is_failed(ptrchk)){ [[unlikely]]
            return std::unexpected(ptrchk);
        }

        return tlb_instance::map_wait(device_id, ptr);
    }

    auto map_wait(uma_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(ptr);

        if (dg::network_exception::is_failed(ptrchk)){ [[unlikely]]
            return std::unexpected(ptrchk);
        } 

        while (true){
            size_t random_value     = dg::network_randomizer::randomize_xrange(std::integral_constant<size_t, MAX_PROXY_PER_REGION>{}); 
            size_t device_sz        = metadata_getter::device_count(ptr);
            size_t idx              = stdx::pow2mod_unsigned(random_value, device_sz); //this is actually hard to tell
            device_id_t device_id   = metadata_getter::device_at(ptr, idx);

            if (auto rs = tlb_instance::map_try(device_id, ptr); rs.has_value()){
                return rs.value();
            }
        }
    }

    void map_release(map_resource_handle_t map_resource) noexcept{

        tlb_instance::map_release(map_resource);
    }

    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        map_release(map_resource);
    };

    auto get_vma_ptr(map_resource_handle_t map_resource) noexcept -> vma_ptr_t{

        return tlb_instance::get_vma_ptr(map_resource);
    }

    auto map_safewait(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        auto map_rs = network_uma::map_wait(device_id, ptr); 

        if (!map_rs.has_value()){ [[unlikely]]
            return std::unexpected(map_rs.error());
        }

        return dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto map_safewait(uma_ptr_t ptr) noexcept -> std::expected<dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        auto map_rs = network_uma::map_wait(ptr);

        if (!map_rs.has_value()){ [[unlikely]]
            return std::unexpected(map_rs.error());
        }

        return dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto device_count(uma_ptr_t ptr) noexcept -> std::expected<size_t, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(ptr);

        if (dg::network_exception::is_failed(ptrchk)){ [[unlikely]]
            return std::unexpected(ptrchk);
        }

        return metadata_getter::device_count(ptr);
    } 
    
    auto device_at(uma_ptr_t ptr, size_t idx) noexcept -> std::expected<device_id_t, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(ptr);

        if (dg::network_exception::is_failed(ptrchk)){ [[unlikely]]
            return std::unexpected(ptrchk);
        }

        return metadata_getter::device_at(ptr, idx);
    }
 
    //--
    auto map_direct_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> vma_ptr_t{

        uma_ptr_access::safe_access(device_id, ptr);
        return tlbdirect_instance::map(device_id, ptr); 
    } 

    auto map_try_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<map_resource_handle_t, exception_t>{

        uma_ptr_access::safe_access(device_id, ptr);
        return tlb_instance::map_try(device_id, ptr);
    }

    auto map_wait_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> map_resource_handle_t{

        uma_ptr_access::safe_access(device_id, ptr);
        return tlb_instance::map_wait(device_id, ptr);
    }

    auto map_safewait_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>{

        return dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_wait_nothrow(device_id, ptr), map_release_lambda);
    }

    auto map_safetry_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>>{

        auto rs = network_uma::map_try_nothrow(device_id, ptr); 

        if (!rs.has_value()){
            return std::nullopt;
        }

        return dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(rs.value(), map_release_lambda);
    } 

    auto device_count_nothrow(uma_ptr_t ptr) noexcept -> size_t{

        uma_ptr_access::safe_access(ptr);
        return metatdata_getter::device_count(ptr);
    }

    auto device_at_nothrow(uma_ptr_t ptr, size_t idx) noexcept -> device_id_t{

        uma_ptr_access::safe_access(ptr);
        return metadata_getter::device_at(ptr, idx);
    }

    auto map_wait_nothrow(uma_ptr_t ptr) noexcept -> map_resource_handle_t{

        uma_ptr_access::safe_access(ptr);

         while (true){
            size_t random_value     = dg::network_randomizer::randomize_xrange(std::integral_constant<size_t, MAX_PROXY_PER_REGION>{});
            size_t device_sz        = metadata_getter::device_count(ptr);
            size_t idx              = stdx::pow2mod_unsigned(random_value, device_sz);
            device_id_t device_id   = metadata_getter::device_at(ptr, idx);

            if (auto rs = tlb_instance::map_try(device_id, ptr); rs.has_value()){
                return rs.value();
            }
        }
    }

    auto map_wait_safe_nothrow(uma_ptr_t ptr) noexcept -> dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>{

        return dg::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_wait_nothrow(ptr), map_release_lambda);
    }
}

#endif