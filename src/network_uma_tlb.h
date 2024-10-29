#ifndef __DG_NETWORK_UMA_TLB_H__
#define __DG_NETWORK_UMA_TLB_H__

#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <type_traits>
#include "network_log.h"
#include "network_uma_tlb_impl1.h"

namespace dg::network_uma_tlb::interface{

    template <class T>
    struct ProxyTLBInterface{

        using device_id_t           = typename T::device_id_t;
        using uma_ptr_t             = typename T::uma_ptr_t;
        using vma_ptr_t             = typename T::vma_ptr_t;
        using map_resource_handle_t = typename T::map_resource_handle_t;

        static_assert(std::conjunction_v<std::is_trivial<device_id_t>, dg::is_ptr<uma_ptr_t>, dg::is_ptr<vma_ptr_t>, std::is_trivial<map_resource_handle_t>>);

        static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> std::optional<map_resource_handle_t>{

            return T::map_try(device_id, host_ptr);
        }

        static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> map_resource_handle_t{

            return T::map_wait(device_id, host_ptr);
        }

        static void map_release(map_resource_handle_t map_resource) noexcept{

            T::map_release(map_resource);
        }
    
        static auto get_vma_ptr(map_resource_handle_t map_resource) noexcept -> vma_ptr_t{

            return T::get_vma_ptr(map_resource);
        }
    };

    template <class T>
    struct DirectTLBInterface{

        using device_id_t           = typename T::device_id_t;
        using uma_ptr_t             = typename T::uma_ptr_t;
        using vma_ptr_t             = typename T::vma_ptr_t;

        static_assert(std::conjunction_v<std::is_trivial<device_id_t>, dg::is_ptr<uma_ptr_t>, dg::is_ptr<vma_ptr_t>);
        
        static auto map(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

            return T::map(device_id, host_ptr);
        }
    };

    template <class T>
    struct MetadataGetterInterface{

        using device_id_t           = typename T::device_id_t;
        using uma_ptr_t             = typename T::uma_ptr_t;

        static_assert(std::is_trivial_v<device_id_t>);
        static_assert(std::is_trivial_v<uma_ptr_t>);

        static auto device_count(uma_ptr_t host_ptr) noexcept -> size_t{

            return T::device_count(device_id, host_ptr);
        }

        static auto device_at(uma_ptr_t host_ptr, size_t idx) noexcept -> device_id_t{

            return T::device_at(host_ptr, idx);
        }
    };

    template <class T>
    struct SafePtrAccessInterface{

        using device_id_t   = typename T::device_id_t
        using uma_ptr_t     = typename T::uma_ptr_t;

        static_assert(std::is_trivial_v<device_id_t>);
        static_assert(dg::is_ptr<uma_ptr_t>);

        static auto safecthrow_access(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> exception_t{

            return T::access(device_id, host_ptr);
        }

        static auto safecthrow_access(uma_ptr_t host_ptr) noexcept -> exception_t{

            return T::access(host_ptr);
        }

        static void safe_access(device_id_t device_id, uma_ptr_t host_ptr) noexcept{

            T::safe_access(device_id, host_ptr);
        }

        static void safe_access(uma_ptr_t host_ptr) noexcept{

            T::safe_access(host_ptr);
        }
    };
}

namespace dg::network_uma_tlb::wrapper{

    using namespace interface;

    template <class T>
    struct DirectWrappedProxyTLB{};

    template <class T>
    struct DirectWrappedProxyTLB<dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>>: DirectTLBInterface<DirectWrappedProxyTLB<dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>>>{

        //it's the factory responsibility to make sure that map_release is (void) function  

        using base          = dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>;
        using device_id_t   = typename base::device_id_t;
        using uma_ptr_t     = typename base::uma_ptr_t;
        using vma_ptr_t     = typename base::vma_ptr_t;

        static auto map(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

            return base::map_wait(device_id, host_ptr); 
        }
    };

    template <class T>
    struct ResourceWrappedProxyTLB{};  

    template <class T>
    struct ResourceWrappedProxyTLB<dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>>: ProxyTLBInterface<ResourceWrappedProxyTLB<dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>>>{

        public:

            using base          = dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>;
            using device_id_t   = typename base::device_id_t;
            using uma_ptr_t     = typename base::uma_ptr_t;
            using vma_ptr_t     = typename base::vma_ptr_t;

        private:

            struct MapResource{
                device_id_t arg_id;
                uma_ptr_t arg_ptr;
                vma_ptr_t map_ptr;
            };
        
        public:

            using map_resource_handle_t = MapResource;

            static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> std::optional<MapResource>{

                vma_ptr_t rs = base::map_try(device_id, host_ptr);

                if (dg::memult::is_nullptr(rs)){
                    return std::nullopt;
                }

                return MapResource{device_id, host_ptr, rs};
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> MapResource{

                vma_ptr_t rs = base::map_wait(device_id, host_ptr);
                return MapResource{device_id, host_ptr, rs};
            }

            static void map_release(MapResource map_resource) noexcept{

                base::map_release(map_resource.arg_id, map_resource.arg_ptr);
            }

            static auto get_vma_ptr(MapResource map_resource) noexcept -> vma_ptr_t{

                return map_resource.map_ptr;
            }
    };

    template <class ID, class UmaPtrType, class DeviceIdType, size_t MEMREGION_SZ>
    struct StdSafePtrRegionAccess: SafePtrAccessInterface<StdSafePtrRegionAccess<ID, UmaPtrtype, DeviceIdType, MEMREGION_SZ>>{

        public:

            using uma_ptr_t     = UmaPtrType;
            using device_id_t   = DeviceIdType;

        private:

            static inline std::unordered_set<std::pair<uma_ptr_t, device_id_t>> umadevice_hash_set{}; //consider vector_table for fast lookup later - in profiling phase - also rid of static if deems necessary - 
            static inline std::unordered_set<uma_ptr_t> uma_hash_set{}; //consider vector_table for fast lookup later - in profiling phase - also rid of static if deems necessary - 

        public:

            static void init(std::unordered_set<std::pair<uma_ptr_t, device_id_t>> umadevice_hash_set_arg,
                             std::unordered_set<uma_ptr_t> uma_hash_set_arg) noexcept{
                
                umadevice_hash_set = std::move(umadevice_hash_set_arg);
                uma_hash_set = std::move(uma_hash_set_arg);
            }

            static auto safecthrow_access(device_id_t id, uma_ptr_t ptr) noexcept -> exception_t{
                
                auto region_ptr = dg::memult::region(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
                auto set_ptr    = umadevice_hash_set.find(std::make_pair(region_ptr, id));

                if (set_ptr == umadevice_hash_set.end()){
                    return dg::network_exception::BAD_PTR_ACCESS;
                }

                return dg::network_exception::SUCCESS;
            }

            static auto safecthrow_access(uma_ptr_t ptr) noexcept -> exception_t{
                
                auto region_ptr = dg::memult::region(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
                auto set_ptr    = uma_hash_set.find(region_ptr);

                if (set_ptr == uma_hash_set.end()){
                    return dg::network_exception::BAD_PTR_ACCESS;
                }

                return dg::network_exception::SUCCESS;
            }

            static void safe_access(device_id_t id, uma_ptr_t ptr) noexcept{

                if constexpr(IS_SAFE_ACCESS_ENABLED){
                    dg::network_exception_handler::nothrow_log(safecthrow_access(id, ptr));
                } else{
                    (void) id;
                }
            }

            static void safe_access(uma_ptr_t ptr) noexcept{

                if constexpr(IS_SAFE_ACCESS_ENABLED){
                    dg::network_exception_handler::nothrow_log(safecthrow_access(ptr));
                } else{
                    (void) ptr;
                }
            }
    };
} 

#endif