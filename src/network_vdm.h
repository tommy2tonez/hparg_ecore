s#ifndef __NETWORK_MEMORY_PROXY_H__
#define __NETWORK_MEMORY_PROXY_H_

#include <type_traits>
#include <utility> 
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include "network_bitset.h"
#include "network_exception.h"
#include "network_log.h" 

namespace dg::network_vma_memproxy{

    template <class ID>
    class DeviceMetadataGetter{

        private:

            struct DeviceDescription{
                device_t device;
                id_t device_id;
            };

            static inline DeviceDescription * table{}; 

        public:

            static void init(id_t * host_device_id, id_t * device_id, device_t * device, size_t sz) noexcept{
                
                static_assert(std::is_unsigned_v<id_t>);
                auto log_scope = dg::network_log_scope::critical_error_catch("dg::network_memproxy::DeviceMetatdataGetter::init(id_t *, id_t *, device_t *, size_t)"); 

                if (sz == 0u){
                    throw dg::network_exception::invalid_init();
                }

                size_t table_sz   = *std::max_element(host_device_id, host_device_id + sz) + 1; 
                table             = new DeviceDescription[table_sz];

                for (size_t i = 0; i < sz; ++i){
                    table[host_device_id[i]].device     = device[i];
                    table[host_device_id[i]].device_id  = device_id[i];
                }

                log_scope.release();
            }

            static inline auto get_device_type(id_t host_device_id) noexcept -> device_option_t{

                return table[host_device_id].device;
            }

            static inline auto get_device_id(id_t host_device_id) noexcept -> id_t{

                return table[host_device_id].device_id;
            }
    };

    template <class ID, class T>
    class HostCudaMemoryTransferDevice{};

    template <class ID, class T>
    class HostCudaMemoryTransferDevice<ID, DeviceMetadataGetterInterface<T>>{

        using id_translator     = DeviceMetadataGetterInterface<T>; 
        using cuda_device_t     = int;

        static inline void memcpy(void * dst, size_t dst_device_id, const void * src, size_t src_device_id, size_t n) noexcept{

            device_option_t dst_device_opt = id_translator::get_device_type(dst_device_id);
            device_option_t src_device_opt = id_translator::get_device_type(src_device_id);

            if (dst_device_opt == cuda){
                if (src_device_opt == cuda){
                    cuda_device_t dst_cuda_device_id    = id_translator::get_device_id(dst_device_id); 
                    cuda_device_t src_cuda_device_id    = id_translator::get_device_id(src_device_id);
                    dg::network_error_handler::cuda_noexcept_log(cudaMemcpyPeer(dst, dst_cuda_device_id, src, src_cuda_device_id, n)); //
                } else if (src_device_opt == cpu){
                    dg::network_error_handler::cuda_noexcept_log(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
                } else{
                    dg::network_log::critical_error("function: HostCudaMemoryTransferDevice::memcpy, src codec not found");
                    std::abort();
                }
            } else if (dst_device_opt == cpu){
                if (src_device_opt == cuda){
                    dg::network_error_handler::cuda_noexcept_log(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
                } else if (src_device_opt == cpu){
                    std::memcpy(dst, src, n);
                } else{
                    dg::network_log::critical_error("function: HostCudaMemoryTransferDevice::memcpy, src codec not found");
                    dg::network_temination::abort();
                }
            } else{
                dg::network_log::critical_error("function: HostCudaMemoryTransferDevice::memcpy, dst codec not found");
                std::abort();
            }
        }
    };
}

namespace dg::network_vdm{

    using device_t                  = uint8_t;
    using device_id_t               = uint64_t; 
    using virtual_device_id_t       = uint64_t;  
    using virtual_device_ptr_t      = void *; 

    enum device_option: device_t{
        cpu     = 0,
        cuda    = 1
    };

    void init(virtual_device_id_t * virtual_device_id, device_id_t * device_id, device_t * device){

    }

    inline auto virtualize_device_id(device_id_t, device_t) noexcept -> virtual_device_id_t{

    }

    inline auto devirtualize_device_id(virtual_device_id_t) noexcept -> std::pair<device_id_t, device_t>{

    } 

    inline auto virtualize_device_ptr(device_ptr_t, device_id_t, device_t) noexcept -> virtual_device_ptr_t{

    } 

    inline auto devirtualize_device_ptr(virtual_device_ptr_t) noexcept -> std::tuple<device_ptr_t, device_id_t, device_t>{

    } 

    inline void memcpy(virtual_device_ptr_t dst, virtual_device_ptr_t src, size_t n) noexcept{ 

    }

    inline void memset(virtual_device_ptr_t dst, int c, size_t n) noexcept{

    }
}

#endif
