#ifndef __NETWORK_UMA_H__
#define __NETWORK_UMA_H__

#include <stdint.h>
#include <stddef.h>
#include "network_uma_definition.h"
#include "network_uma_tlb.h"

namespace dg::network_uma{

    //
    
    struct signature_dg_network_uma{}; 

    using namespace dg::network_uma_definition; 
    using namespace dg::network_uma_tlb::memqualifier_taxonomy;  
    
    using tlb_factory           = dg::network_uma_tlb::v1::Factory<signature_dg_network_uma, device_id_t, uma_ptr_t, device_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>; 
    using tlb_instance          = typename tlb_factory::tlb;
    using tlbdirect_instance    = typename tlb_factory::tlb_direct;
    using uma_metadata_instance = typename tlb_factory::uma_metadata; 

    void init(uma_ptr_t * host_region, vma_ptr_t * device_region, device_id_t * device_id, memqualifier_t * qualifier, size_t n){

        tlb_factory::init(host_ptr, device_ptr, device_id, qualifier, n);
    }
    
    auto map_direct(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

        return tlbdirect_instance::map_wait(device_id, host_ptr);
    } 

    auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> std::expected<map_resource_handler_t, exception_t>{ //it's clearer this way - strange but I just feel like it

        // return tlb_instance::map_try(device_id, host_ptr);
    } 

    auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> map_resource_handler_t{

        // return tlb_instance::map_wait(device_id, host_ptr);
    }

    void map_release(map_resource_handler_t map_resource) noexcept{

        // tlb_instance::map_release(device_id, host_ptr);
    }

    auto map_relguard(map_resource_handler_t map_resource) noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            map_release(map_resource);
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    } 

    auto get_vma_ptr(map_resource_handler_t map_resource) noexcept -> vma_ptr_t{

    } 

    auto get_vma_const_ptr(map_resource_handler_t map_resource) noexcept -> vma_ptr_t{ 

    } 

    //-----
    
    auto safe_ptr_access(uma_ptr_t) -> uma_ptr_t{

    }

    auto safe_ptr_access_nothrow(uma_ptr_t) noexcept -> uma_ptr_t{
        
    }

    //-----
    
    auto memacquire_try(uma_ptr_t ptr) noexcept -> bool{

    }

    void memacquire_wait(uma_ptr_t ptr) noexcept{

    }

    void memacquire_release(uma_ptr_t ptr) noexcept{

    }

    auto memacquire_guard(uma_ptr_t ptr) noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            memacquire_release(ptr);
        };

        memacquire_wait(ptr);
        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    } 

    //-----

    auto device_count(uma_ptr_t host_ptr) noexcept -> size_t{

        return uma_metadata_instance::device_reference_count(host_ptr);
    }

    auto device_at(uma_ptr_t host_ptr, size_t idx) noexcept -> device_id_t{

        return uma_metadata_instance::device_at(host_ptr, idx);
    }

    auto device_strictest_at(uma_ptr_t host_ptr) noexcept -> device_id_t{

        return uma_metadata_instance::device_strictest_at(host_ptr);
    }

    //-----
    //lengthy - consider dispatch_t
    void memset_synchronous_alldevice_bypass_qualifier(uma_ptr_t dst, int c, size_t sz) noexcept{

    } 

    void memset_synchronous(uma_ptr_t dst, int c, size_t sz) noexcept{

    }

    void memset(uma_ptr_t dst, int c, size_t sz) noexcept{

    }

    void memcpy_uma_to_device_synchronous(vma_ptr_t dst, uma_ptr_t src, size_t sz) noexcept{

    }

    void memcpy_uma_to_device(vma_ptr_t dst, uma_ptr_t src, size_t sz) noexcept{

    } 
    
    void memcpy_device_to_uma_synchronous_alldevice_bypass_qualifier(uma_ptr_t dst, vma_ptr_t src, size_t sz) noexcept{

    } 

    void memcpy_device_to_uma_synchronous(uma_ptr_t dst, vma_ptr_t src, size_t sz) noexcept{

    }

    void memcpy_device_to_uma(uma_ptr_t dst, vma_ptr_t src, size_t sz) noexcept{

    }
}

#endif