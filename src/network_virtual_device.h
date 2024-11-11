#ifndef __NETWORK_MEMORY_PROXY_H__
#define __NETWORK_MEMORY_PROXY_H_

#include <type_traits>
#include <utility> 
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include "network_exception.h"
#include "network_log.h" 
#include "network_exception_handler.h"

namespace dg::network_device_virtualizer::incremental{

    template <class ID, class device_t, class device_id_t>
    class IncrementalVirtualizer{

        private:

            struct DeviceDescription{
                device_t device;
                device_id_t device_id;
            };

            static inline DeviceDescription * table{}; 

        public:

            static_assert(std::is_unsigned_v<device_t>);
            static_assert(std::is_unsigned_v<device_id_t>);

            using virtual_device_id_t = DeviceDescription;

            static void init(device_t * device, device_id_t * device_id, size_t sz){
                
            }

            static auto virtualize(device_t device, device_id_t device_id) noexcept -> std::expected<virtual_device_id_t, exception_t>{

                auto encoded_key    = encode(device, device_id);
                auto dict_ptr       = table.find(encoded_key);

                if (dict_ptr == table.end()){
                    return std::unexpected(dg::network_exception::INVALID_DICTIONARY_KEY);
                }

                return dict_ptr->second;
            }

            static auto devirtualize(virtual_device_id_t virtual_device_id) noexcept -> std::tuple<device_t, device_id_t>{

                return {virtual_device_id.device, virtual_device_id.device_id};
            }
    };

    template <class ID, class cuda_device_id_t, class fsys_device_id_t, class host_device_id_t>
    class StdDeviceVirtualizer{

        private:

            using device_t          = uint8_t; 
            using cast_device_id_t  = intmax_t; //
            using unif_device_id_t  = uintmax_t;

            static inline constexpr device_t CUDA_DEVICE    = 0u;
            static inline constexpr device_t HOST_DEVICE    = 1u;
            static inline constexpr device_t FSYS_DEVICE    = 2u;   

        public:

            static_assert(std::numeric_limits<cuda_device_id_t>::is_integer);
            static_assert(std::numeric_limits<fsys_device_id_t>::is_integer);
            static_assert(std::numeric_limits<host_device_id_t>::is_integer);

            static auto virtualize_cuda(cuda_device_id_t cuda_id) noexcept -> std::expected<virtual_device_id_t, exception_t>{

                cast_device_id_t tmp    = static_cast<cast_device_id_t>(cuda_id);
                unif_device_id_t id     = std::bit_cast<unif_device_id_t>(tmp);

                return base::virtualize(CUDA_DEVICE, id); 
            } 

            static auto virtualize_host(host_device_id_t host_id) noexcept -> std::expected<virtual_device_id_t, exception_t>{

                cast_device_id_t tmp    = static_cast<cast_device_id_t>(host_id);
                unif_device_id_t id     = std::bit_cast<unif_device_id_t>(tmp);

                return base::virtualize(HOST_DEVICE, id);
            }

            static auto virtualize_fsys(fsys_device_id_t fsys_id) noexcept -> std::expected<virtual_device_id_t, exception_t>{

                cast_device_id_t tmp    = static_cast<cast_device_id_t>(fsys_id);
                unif_device_id_t id     = std::bit_cast<unif_device_id_t>(tmp);

                return base::virtualize(FSYS_DEVICE, id);
            }

            static auto is_cuda(virtual_device_id_t id) noexcept -> bool{

                auto [device, device_id] = base::devirtualize(id); 
                return device == CUDA_DEVICE;
            } 

            static auto is_host(virtual_device_id_t id) noexcept -> bool{

                auto [device, device_id] = base::devirtualize(id);
                return device == HOST_DEVICE;
            }

            static auto is_fsys(virtual_device_id_t id) noexcept -> bool{

                auto [device, device_id] = base::devirtualize(id);
                return device == FSYS_DEVICE;
            }

            static auto devirtualize_cuda(virtual_device_id_t id) noexcept -> cuda_device_id_t{

                auto [device, device_id] = base::devirtualize(id);

                if (device != CUDA_DEVICE){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                return static_cast<cuda_device_id_t>(std::bit_cast<cast_device_id_t>(device_id))
            }

            static auto devirtualize_host(virtual_device_id_t id) noexcept -> host_device_id_t{

                auto [device, device_id] = base::devirtualize(id);

                if (device != HOST_DEVICE){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                return static_cast<host_device_id_t>(std::bit_cast<cast_device_id_t>(device_id));
            }

            static auto devirtualize_fsys(virtual_device_id_t id) noexcept -> fsys_device_id_t{

                auto [device, device_id] = base::devirtualize(id);

                if (device != FSYS_DEVICE){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                return static_cast<fsys_device_id_t>(std::bit_cast<cast_device_id_t>(device_id));
            }

    };

    template <class ID, class cuda_ptr_t, class cuda_device_id_t, class fsys_ptr_t, class fsys_device_id_t, class host_ptr_t, class host_device_id_t>
    class PointerVirtualizer{

        private:

            using ptr_arithemtic_t  = dg::ptr_info<>::max_unsigned_t;
            using device_id_t       = intmax_t; //
            using device_t          = uint8_t; 

            static inline constexpr device_t CUDA_DEVICE    = 0u;
            static inline constexpr device_t FSYS_DEVICE    = 1u;
            static inline constexpr device_t HOST_DEVICE    = 2u; 

            struct VirtualPtr{
                ptr_arithmetic_t ptr_value;
                device_id_t device_id;
                device_t device;
            };

        public:

            static_assert(dg::is_ptr_v<cuda_ptr_t>);
            static_assert(dg::is_ptr_v<fsys_ptr_t>);
            static_assert(dg::is_ptr_v<host_ptr_t>);
            static_assert(std::numeric_limits<cuda_device_id_t>::is_integer);
            static_assert(std::numeric_limits<fsys_device_id_t>::is_integer);
            static_assert(std::numeric_limits<host_device_id_t>::is_integer);
            
            using virtual_ptr_t = VirtualPtr;

            static auto is_cuda_ptr(VirtualPtr ptr) noexcept -> bool{

                return ptr.device == CUDA_DEVICE;
            }

            static auto is_fsys_ptr(VirtualPtr ptr) noexcept -> bool{

                return ptr.device == FSYS_DEVICE;
            }

            static auto is_host_ptr(VirtualPtr ptr) noexcept -> bool{

                return ptr.device == HOST_DEVICE;
            }

            static auto virtualize_cuda_ptr(cuda_ptr_t ptr, cuda_device_id_t id) noexcept -> VirtualPtr{

                return {pointer_cast<ptr_arithemtic_t>(ptr), static_cast<device_id_t>(id), CUDA_DEVICE};
            }

            static auto virtualize_fsys_ptr(fsys_ptr_t ptr, fsys_device_id_t id) noexcept -> VirtualPtr{

                return {pointer_cast<ptr_arithemtic_t>(ptr), static_cast<device_id_t>(id), FSYS_DEVICE};
            }

            static auto virtualize_host_ptr(host_ptr_t ptr, host_device_id_t id) noexcept -> VirtualPtr{

                return {pointer_cast<ptr_arithemtic_t>(ptr), static_cast<device_id_t>(id), HOST_DEVICE};
            }

            static auto devirtualize_cuda_ptr(VirtualPtr ptr) noexcept -> std::tuple<cuda_ptr_t, cuda_device_id_t>{
                
                if constexpr(SAFE_PTR_ACCESS_ENABLED){
                    if (!is_cuda_ptr(ptr)){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return {pointer_cast<cuda_ptr_t>(ptr.value), static_cast<cuda_device_id_t>(ptr.device_id)};
            }

            static auto devirtualize_fsys_ptr(VirtualPtr ptr) noexcept -> std::tuple<fsys_ptr_t, fsys_device_id_t>{
                
                if constexpr(SAFE_PTR_ACCESS_ENABLED){
                    if (!is_fsys_ptr(ptr)){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return {pointer_cast<fsys_ptr_t>(ptr.ptr_value), static_cast<fsys_device_id_t>(ptr.device_id)};
            }

            static auto devirtualize_host_ptr(VirtualPtr ptr) noexcept -> std::tuple<host_ptr_t, host_device_id_t>{

                if constexpr(SAFE_PTR_ACCESS_ENABLED){
                    if (!is_host_ptr(ptr)){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } 
                }
 
                return {pointer_cast<host_ptr_t>(ptr.ptr_value), static_cast<host_device_id_t>(ptr.device_id)};
            };
    };
}

namespace dg::network_virtual_device{
        
    struct signature_dg_network_virtual_device{}; 

    using namespace dg::network_device_virtualizer::device_info;
    using virtualizer           = dg::network_device_virtualizer::incremental::Virtualizer<signature_dg_network_virtual_device>;
    using virtual_device_id_t   = typename virtualizer::virtual_device_id_t;
    using virtual_ptr_t         = typename virtualizer::virtual_ptr_t;
    using ptr_arithmetic_t      = typename dg::ptr_info<>::max_unsigned_t; 

    static inline constexpr virtual_device_id_t HOST_VIRTUAL_DEVICE_ID = 0u; 
    static inline constexpr size_t HOST_PTR_FLAG = 0u;
    static inline constexpr size_t CUDA_PTR_FLAG = 0u;
    static inline constexpr size_t FSYS_PTR_FLAG = 0u; 

    // void init(device_t * device, device_id_t * device_id, size_t n){

    //     virtualizer::init(device, device_id, n);
    // }

    void init(){

    }

    auto is_cuda_id(virtual_device_id_t) noexcept -> bool{

        return virtualizer::is_cuda_id(id);
    } 

    auto is_host_id(virtual_device_id_t id) noexcept -> bool{

        return virtualizer::is_host_id(id);
    }

    auto is_fsys_id(virtual_device_id_t id) noexcept -> bool{

    }

    auto virtualize_cuda_id(cuda_device_id_t) noexcept -> std::expected<virtual_device_id_t, exception_t>{

        return virtualizer::virtualize_cuda_id(id);
    }

    auto virtualize_host_id(host_device_id_t) noexcept -> std::expected<virtual_device_id_t, exception_t>{

        return virtualizer::virtualize_host_id();
    }
    
    auto virtualize_fsys_id(fsys_device_id_t) noexcept -> std::expected<virtual_device_id_t, exception_t>{

    }

    auto virtualize_cuda_id_nothrow(cuda_device_id_t) noexcept -> virtual_device_id_t{

    }

    auto virtualize_host_id_nothrow(host_device_id_t) noexcept -> virtual_device_id_t{

    }

    auto virtualize_fsys_id_nothrow(fsys_device_id_t) noexcept -> virtual_device_id_t{

    }

    auto devirtualize_cuda_id(virtual_device_id_t id) noexcept -> cuda_device_id_t{

        return virtualizer::devirtualize_cuda_id(id);
    }

    auto devirtualize_host_id(virtual_device_id_t id) noexcept -> host_device_id_t{

    }

    auto devirtualize_fsys_id(virtual_device_id_t id) noexcept -> fsys_device_id_t{

    } 

    //------

    auto is_cuda_ptr(virtual_ptr_t) noexcept -> bool{

    } 

    auto is_host_ptr(virtual_ptr_t) noexcept -> bool{

    }

    auto is_fsys_ptr(virtual_ptr_t) noexcept -> bool{

    }

    auto is_ptr(virtual_ptr_t, ptr_check_flag_t) noexcept -> bool{

    }

    auto virtualize_cuda_ptr(cuda_ptr_t ptr) noexcept -> virtual_ptr_t{

    }

    auto virtualize_host_ptr(host_ptr_t ptr) noexcept -> virtual_ptr_t{

    } 

    auto virtualize_fsys_ptr(fsys_ptr_t ptr) noexcept -> virtual_ptr_t{

    }

    auto devirtualize_cuda_ptr(virtual_ptr_t ptr) noexcept -> cuda_ptr_t{

    }

    auto devirtualize_host_ptr(virtual_ptr_t ptr) noexcept -> host_ptr_t{

    }

    auto devirtualize_fsys_ptr(virtual_ptr_t ptr) noexcept -> fsys_ptr_t{

    }
}

#endif
