#ifndef __NETWORK_VMA_OPERATION_H__
#define __NETWORK_VMA_OPERATION_H__

#include <stdint.h>
#include <stddef.h>
#include "network_vma_definition.h"
#include "network_vma_tlb.h"

namespace dg::network_vma{
    
    struct signature_dg_network_vma{}; 

    using namespace dg::network_vma_definition; 
    using namespace dg::network_vma_tlb::memqualifier_taxonomy;  
    
    using tlb_factory           = dg::network_vma_tlb::v1::Factory<signature_dg_network_vma, device_id_t, vma_ptr_t, device_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>; 
    using tlb_instance          = typename tlb_factory::tlb;
    using tlbdirect_instance    = typename tlb_factory::tlb_direct;
    using vma_metadata_instance = typename tlb_factory::vma_metadata; 

    struct map_resource_handler{
        device_id_t device_id;
        vma_ptr_t host_ptr;
    };

    void init(vma_ptr_t * host_ptr, device_ptr_t * device_ptr, device_id_t * device_id, memqualifier_t * qualifier, size_t n){

        tlb_factory::init(host_ptr, device_ptr, device_id, qualifier, n);
    }
    
    inline auto map_direct(device_id_t device_id, vma_ptr_t host_ptr) noexcept -> device_ptr_t{

        return tlbdirect_instance::map_wait(device_id, host_ptr);
    } 

    inline auto map_try(device_id_t device_id, vma_ptr_t host_ptr) noexcept -> std::optional<map_resource_handler>{

        // return tlb_instance::map_try(device_id, host_ptr);
    } 

    inline auto map_wait(device_id_t device_id, vma_ptr_t host_ptr) noexcept -> map_resource_handler{

        // return tlb_instance::map_wait(device_id, host_ptr);
    }

    inline void map_release(map_resource_handler map_resource) noexcept{

        // tlb_instance::map_release(device_id, host_ptr);
    }

    inline auto remap_try(map_resource_handler map_resource, device_id_t device_id, vma_ptr_t new_host_ptr) noexcept -> std::optional<map_resource_handler>{

        // return tlb_instance::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
    }
    
    inline auto remap_wait(map_resource_handler map_resource, device_id_t device_id, vma_ptr_t new_host_ptr) noexcept -> map_resource_handler{

        // return tlb_instance::remap_wait(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
    } 

    inline auto device_ptr(map_resource_handler map_resource) noexcept -> device_ptr_t{

    } 

    inline auto device_const_ptr(map_resource_handler map_resource) noexcept -> device_ptr_t{
s
    } 

    inline auto device_reference_count(vma_ptr_t host_ptr) noexcept -> size_t{

        return vma_metadata_instance::device_reference_count(host_ptr);
    }

    inline auto device_at(vma_ptr_t host_ptr, size_t idx) noexcept -> device_id_t{

        return vma_metadata_instance::device_at(host_ptr, idx);
    }

    inline auto device_strictest_at(vma_ptr_t host_ptr) noexcept -> device_id_t{

        return vma_metadata_instance::device_strictest_at(host_ptr);
    }

    inline void memset_synchronous_bypass_qualifier(vma_ptr_t dst, int c, size_t sz) noexcept{

    } 

    inline void memset_synchronous(vma_ptr_t dst, int c, size_t sz) noexcept{

    }

    inline void memset(vma_ptr_t dst, int c, size_t sz) noexcept{

    }

    inline void memcpy_vma_to_device_synchronous(device_ptr_t dst, device_id_t dst_id, vma_ptr_t src, size_t sz) noexcept{

    }

    inline void memcpy_vma_to_device(device_ptr_t dst, device_id_t dst_id, vma_ptr_t src, size_t sz) noexcept{

    } 
    
    inline void memcpy_device_to_vma_synchronous_bypass_qualifier(vma_ptr_t dst, device_ptr_t src, device_id_t src_id, size_t sz) noexcept{

    } 

    inline void memcpy_device_to_vma_synchronous(vma_ptr_t dst, device_ptr_t src, device_id_t src_id, size_t sz) noexcept{

    }

    inline void memcpy_device_to_vma(vma_ptr_t dst, device_ptr_t src, device_id_t src_id, size_t sz) noexcept{

    }
}

#endif