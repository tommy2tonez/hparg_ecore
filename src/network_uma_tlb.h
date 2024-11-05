#ifndef __DG_NETWORK_UMA_TLB_H__
#define __DG_NETWORK_UMA_TLB_H__

#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <type_traits>
#include "network_log.h"
#include "network_std_container.h"
#include "network_exception_handler.h"

namespace dg::network_uma_tlb::interface{
    
    template <class T>
    struct MutexTLBInterface{

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using device_id_t           = typename T1::device_id_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using uma_ptr_t             = typename T1::uma_ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using vma_ptr_t             = typename T1::vma_ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using map_resource_handle_t = typename T1::map_resource_handle_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto map_try(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept -> std::optional<typename T1::map_resource_handle_t>{

            return T::map_try(device_id, host_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto map_wait(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept -> typename T1::map_resource_handle_t{

            return T::map_wait(device_id, host_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void map_release(typename T1::map_resource_handle_t map_resource) noexcept{

            T::map_release(map_resource);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto remap_try(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr, typename T1::map_resource_handle_t resource) noexcept -> std::optional<typename T1::map_resource_handle_t>{

            return T::remap_try(device_id, host_ptr, resource);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto remap_wait(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr, typename T1::map_resource_handle_t resource) noexcept -> typename T1::map_resource_handle_t{

            return T::remap_wait(device_id, host_ptr, resource);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto get_vma_ptr(typename T1::map_resource_handle_t map_resource) noexcept -> typename T1::vma_ptr_t{

            return T::get_vma_ptr(map_resource);
        }
    };

    template <class T>
    struct MutexRegionTLBInterface: MutexTLBInterface<T>{

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto memregion_size() noexcept -> size_t{

            return T::memregion_size();
        }
    };

    template <class T>
    struct DirectTLBInterface{

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using device_id_t           = typename T1::device_id_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using uma_ptr_t             = typename T1::uma_ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using vma_ptr_t             = typename T1::vma_ptr_t;
        
        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto map(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept -> typename T1::vma_ptr_t{

            return T::map(device_id, host_ptr);
        }
    };

    template <class T>
    struct MetadataGetterInterface{

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using device_id_t           = typename T1::device_id_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using uma_ptr_t             = typename T1::uma_ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto device_count(typename T1::uma_ptr_t host_ptr) noexcept -> size_t{

            return T::device_count(host_ptr);
        }
        
        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto device_at(typename T1::uma_ptr_t host_ptr, size_t idx) noexcept -> typename T1::device_id_t{

            return T::device_at(host_ptr, idx);
        }
    };

    template <class T>
    struct SafePtrAccessInterface{

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using device_id_t   = typename T1::device_id_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using uma_ptr_t     = typename T1::uma_ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto safecthrow_access(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept -> exception_t{

            return T::safecthrow_access(device_id, host_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto safecthrow_access(typename T1::uma_ptr_t host_ptr) noexcept -> exception_t{

            return T::safecthrow_access(host_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void safe_access(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept{

            T::safe_access(device_id, host_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void safe_access(typename T1::uma_ptr_t host_ptr) noexcept{

            T::safe_access(host_ptr);
        }
    };

    template <class T>
    auto recursive_raiimap_try(const MutexRegionTLBInterface<T>, 
                               typename MutexRegionTLBInterface<T>::device_id_t device_id, 
                               typename MutexRegionTLBInterface<T>::uma_ptr_t ptr) noexcept -> std::optional<dg::unique_resource<typename MutexRegionTLBInterface<T>::map_resource_handle_t, void (*)(typename MutexRegionTLBInterface<T>::map_resource_handle_t) noexcept>>{
        
        using tlb_ins       = MutexRegionTLBInterface<T>;
        using resource_ins  = RecursiveMapResource<MutexRegionTLBInterface<T>>;
        using ptr_t         = typename tlb_ins:uma_ptr_t;
        using resource_t    = typename tlb_ins::map_resource_handle_t;

        dg::unordered_unstable_set<ptr_t>& ptr_set = resource_ins::get();
        ptr_t ptr_region = dg::memult::region(ptr);
        
        if (ptr_set.contains(ptr_region)){
            return dg::unique_resource<typename MutexRegionTLBInterface<T>::map_resource_handle_t, void (*)(typename MutexRegionTLBInterface<T>::map_resource_handle_t) noexcept>{};
        }

        auto destructor = [](resource_t resource_arg) noexcept{
            resource_ins::get().erase();
            tlb_ins::map_release(resource_arg);
        };

        if (auto rs = resource_ins::map_try(device_id, ptr); rs.has_value()){
            return dg::unique_resource<typename MutexRegionTLBInterface<T>::map_resource_handle_t, void (*)(typename MutexRegionTLBInterface<T>::map_resource_handle_t) noexcept>{rs.value(), destructor};
        }
        
        return std::nullopt;
    }
}

namespace dg::network_uma_tlb::access{

    using namespace interface;

    static inline constexpr bool IS_SAFE_ACCESS_ENABLED = true;

    template <class ID, class UMAPtrType, class DeviceIdType, size_t MEMREGION_SZ>
    struct StdSafeRegionAccess: SafePtrAccessInterface<StdSafeRegionAccess<ID, UMAPtrType, DeviceIdType, MEMREGION_SZ>>{

        public:

            using uma_ptr_t     = UMAPtrType;
            using device_id_t   = DeviceIdType;

        private:

            static inline dg::unordered_unstable_set<std::pair<uma_ptr_t, device_id_t>> umadevice_hash_set{};
            static inline dg::unordered_unstable_set<uma_ptr_t> uma_hash_set{};

        public:

            static void init(uma_ptr_t * uma_region_arr, device_id_t * device_id_arr, size_t sz){
                
                for (size_t i = 0u; i < sz; ++i){
                    umadevice_hash_set.insert(std::make_pair(uma_region_arr[i], device_id_arr[i]));
                    uma_hash_set.insert(uma_region_arr[i]);
                }
            }

            static void deinit() noexcept{

                umadevice_hash_set = {};
                uma_hash_set = {};
            }

            static auto safecthrow_access(device_id_t id, uma_ptr_t ptr) noexcept -> exception_t{
                
                auto region_ptr = dg::memult::region(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
                
                if (!umadevice_hash_set.contains(std::make_pair(ptr, id))){ [[unlikely]]
                    return dg::network_exception::BAD_ACCESS;
                }

                return dg::network_exception::SUCCESS;
            }

            static auto safecthrow_access(uma_ptr_t ptr) noexcept -> exception_t{
                
                auto region_ptr = dg::memult::region(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});

                if (!uma_hash_set.contains(ptr)){ [[unlikely]]
                    return dg::network_exception::BAD_ACCESS;
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