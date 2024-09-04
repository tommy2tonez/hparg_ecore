#ifndef __NETWORK_UMA_H__
#define __NETWORK_UMA_H__

#include <stdint.h>
#include <stddef.h>
#include "network_uma_definition.h"
#include "network_uma_tlb.h"
#include "network_exception_handler.h"

namespace dg::network_uma{
    
    struct signature_dg_network_uma{}; 

    using namespace dg::network_uma_definition; 
    using namespace dg::network_uma_tlb::memqualifier_taxonomy;  
    
    using tlb_factory           = dg::network_uma_tlb::v1::Factory<signature_dg_network_uma, device_id_t, uma_ptr_t, device_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>; 
    using tlb_instance          = typename tlb_factory::tlb;
    using tlbdirect_instance    = typename tlb_factory::tlb_direct;
    using uma_metadata_instance = typename tlb_factory::uma_metadata; 

    void init(uma_ptr_t * host_region, vma_ptr_t * device_region, device_id_t * device_id, memqualifier_t * qualifier, size_t n){

        tlb_factory::init(host_region, device_region, device_id, qualifier, n);
    }
    
    auto map_direct(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<vma_ptr_t, exception_t>{

    }

    auto map_direct_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> vma_ptr_t{

        return tlbdirect_instance::map_wait(device_id, ptr);
    } 

    auto map_try(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

    }

    auto map_try_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<map_resource_handle_t, exception_t>{

    }

    auto map_wait(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

    }

    auto map_wait_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> map_resource_handle_t{

    }

    void map_release(map_resource_handle_t map_resource) noexcept{

        // tlb_instance::map_release(device_id, ptr);
    }

    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        map_release(map_resource);
    };

    auto map_wait_safe(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        auto map_rs = map_wait(device_id, ptr); 

        if (!map_rs.has_value()){
            return map_rs.error();
        }

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto map_wait_nothrow_safe(device_id_t device_id, uma_ptr_t ptr) noexcept -> dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>{

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_wait_nothrow(device_id, ptr), map_release_lambda);
    }

    auto map_relguard(map_resource_handle_t map_resource) noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            map_release(map_resource);
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    } 

    auto get_vma_ptr(map_resource_handle_t map_resource) noexcept -> vma_ptr_t{

    } 

    auto get_vma_const_ptr(map_resource_handle_t map_resource) noexcept -> vma_ptr_t{ 

    } 

    auto device_count(uma_ptr_t ptr) noexcept -> std::expected<size_t, exception_t>{

    } 

    auto device_count_nothrow(uma_ptr_t ptr) noexcept -> size_t{

        // return uma_metadata_instance::device_reference_count(ptr);
    }

    auto device_at(uma_ptr_t ptr, size_t idx) noexcept -> std::expected<device_id_t, exception_t>{

    } 

    auto device_at_nothrow(uma_ptr_t ptr, size_t idx) noexcept -> device_id_t{

        // return uma_metadata_instance::device_at(ptr, idx);
    }

    auto device_strictest_at(uma_ptr_t ptr) noexcept -> std::expected<device_id_t, exception_t>{
>
        return uma_metadata_instance::device_strictest_at(ptr);
    }

    auto device_strictest_at_nothrow(uma_ptr_t ptr) noexcept -> device_id_t{

    } 

    auto device_recent_at(uma_ptr_t ptr) noexcept -> std::expected<device_id_t, exception_t>{

    }

    auto device_recent_at_nothrow(uma_ptr_t ptr) noexcept -> device_id_t{

    }

    auto map_wait(uma_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        while (true){
            device_id_t id  = device_recent_at(ptr);
            auto map_token  = map_try(id, ptr);

            if (map_token.has_value()){
                return map_token.value();
            }

            if (map_token.error() == dg::network_exception::OCCUPIED_MEMREGION){
                continue;
            }

            return std::unexpected(map_token.error());
        }
    }

    auto map_wait_nothrow(uma_ptr_t ptr) noexcept -> map_resource_handle_t{

        return dg::network_exception_handler::nothrow_log(map_wait(ptr));
    }

    auto map_wait_safe(uma_ptr_t ptr) noexcept -> std::expected<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        auto map_rs = map_wait(ptr);

        if (!map_rs.has_value()){
            return map_rs.error();
        }

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto map_wait_safe_nothrow(uma_ptr_t ptr) noexcept -> dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>{

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_wait_nothrow(ptr), map_release_lambda);
    }
}

#endif