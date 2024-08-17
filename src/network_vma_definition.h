#ifndef __NETWORK_VMA_DEFINITION_H__
#define __NETWORK_VMA_DEFINITION_H__

#include <stdint.h>
#include <stddef.h>

namespace dg::network_vma_definition{

    using device_id_t           = uint8_t;
    using vma_ptr_t             = uint64_t; //convert vma_ptr_t -> object
    using device_ptr_t          = void *;  //convert device_ptr_t -> arithmetic + uint8_t to virtualize device_ptr_t, explicit cast will convert to the org type Cudeviceptr, etc...
    using virtual_device_ptr_t  = uint64_t;
    
    static inline constexpr vma_ptr_t NULL_VMA_PTR          = 0u;
    static inline constexpr device_ptr_t NULL_DEVICE_PTR    = nullptr;
    static inline constexpr size_t MEMREGION_SZ             = size_t{0u};
    static inline constexpr size_t PROXY_COUNT              = size_t{0u};
} 

#endif