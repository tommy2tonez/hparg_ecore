#ifndef __NETWORK_POINTER_H__
#define __NETWORK_POINTER_H__

#include <stdint.h>
#include <stdlib.h>

namespace dg::network_pointer{

    struct CufsPtr{
        uint64_t device_numerical_addr;
        int gpu_device;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(device_numerical_addr, gpu_device);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(device_numerical_addr, gpu_device);
        }
    };

    using cufs_ptr_t = CufsPtr;

    auto cufs_ptr_device_id(CufsPtr ptr) noexcept -> int{
        
        return ptr.gpu_device;
    }
} 

#endif