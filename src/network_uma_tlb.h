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
    
    //I've been trying to find a way to deinitialize interface - it's hard - impossible - I guess that the price for static linking - it's fine - it's an object - just not deallocatable - program-bound  
    //I mean you could do std::vector<std::shared_ptr<void>> as arguments - but who do that - really ?
    //I've been coding bare metal for years now
    //there are two ways - static-linking + crtp + static inline <root_object> + decomposite by inheritance
    //or this - static crtp
    //out of what's worse - I choose this to be the better way to do dirty C++ things

    template <class T>
    struct ProxyTLBInterface{

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
        static auto remap_try(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr, typename T1::map_resource_handle_t resource) -> std::optional<typename T1::map_resource_handle_t>{

            return T::remap_try(device_id, host_ptr, resource);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto remap_wait(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr, typename T1::map_resource_handle_t resource) -> typename T1::map_resource_handle_t{

            return T::remap_wait(device_id, host_ptr, resource);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto get_vma_ptr(typename T1::map_resource_handle_t map_resource) noexcept -> typename T1::vma_ptr_t{

            return T::get_vma_ptr(map_resource);
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


    //^^^
    //-the interface is consistent yet the implementation of a specific map (memregion) should not be here - for future extension
    //this is messy

    // struct RecusriveMapResource{
    //     map_resource_handle_t handle;
    //     uma_ptr_t region_ptr;
    //     size_t offset;
    //     bool responsibility_flag;
    // };

    // using map_recusrive_resource_handle_t = RecusriveMapResource;

    // void map_recursive_release(map_recusrive_resource_handle_t map_resource) noexcept{

    //     if (map_resource.responsibility_flag){
    //         map_release(map_resource.handle);

    //         dg::unordered_map<uma_ptr_t, std::pair<device_id_t, map_resource_handle_t>>& map_ins = map_recursive_resource_instance::get();
    //         auto map_ptr = map_ins.find(map_resource.region_ptr);

    //         if (map_ptr == map_ins.end()){ //DEBUG_FLAG here - all internal error should be disable-able
    //             dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_ERROR));
    //             std::abort();
    //         }

    //         map_ins.erase(map_ptr);
    //     }
    // }

    // static inline auto map_recursive_release_lambda = [](map_recursive_resource_handle_t map_resource) noexcept{
    //     map_recursive_release(map_resource);
    // };

    // auto mapsafe_recusivetry_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>>{
        
    //     dg::unordered_map<uma_ptr_t, std::pair<device_id_t, map_resource_handle_t>>& map_ins = map_recursive_resource_instance::get();
    //     uma_ptr_t region_ptr        = dg::memult::region(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
    //     size_t region_off           = dg::memult::region_offset(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
    //     auto map_ptr                = map_ins.find(region_ptr);

    //     if (map_ptr == map_ins.end()){
    //         auto rs = map_try_nothrow(device_id, region_ptr);

    //         if (!static_cast<bool>(rs)){
    //             return std::nullopt;
    //         }

    //         map_ins[region_ptr]     = std::make_pair(device_id, rs.value());
    //         auto handle             = map_recursive_resource_handle_t{rs.value(), region_ptr, region_off, true};
    //         return dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>(handle, map_recursive_release_lambda);
    //     }

    //     auto [bucket_device_id, bucket_resource] = *map_ptr;

    //     if (bucket_device_id != device_id){ //DEBUG_FLAG here - all internal error should be disable-able
    //         dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
    //         std::abort();
    //     }

    //     auto handle = map_recursive_resource_handle_t{bucket_resource, region_ptr, region_off, false};  //region_ptr should be std::optional - yet I find it more intuitive to have a separate bool flag
    //     return dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>(handle, map_recursive_release_lambda);
    // }

    // auto mapsafe_recursivewait_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>{

    //     while (true){
    //         if (auto rs = mapsafe_recusivetry_nothrow(device_id, ptr); static_cast<bool>(rs)){
    //             return rs.value();
    //         }
    //     }
    // } 

    // template <size_t SZ>
    // auto mapsafe_recursivetry_many_nothrow(std::array<std::pair<device_id_t, uma_ptr_t>, SZ> arg) noexcept -> std::optional<std::array<dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>, SZ>>{

    //     std::array<dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>, SZ> rs{};

    //     for (size_t i = 0; i < SZ; ++i){
    //         rs[i] = dg::network_genult::tuple_invoke(mapsafe_recusivetry_nothrow, arg[i]);

    //         if (!static_cast<bool>(rs[i])){
    //             return std::nullopt;
    //         }
    //     }

    //     return rs;
    // }

    // template <size_t SZ>
    // auto mapsafe_recursivewait_many_nothrow(std::array<std::pair<device_id_t, uma_ptr_t>, SZ> arg) noexcept -> std::array<dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>, SZ>{

    //     while (true){
    //         if (auto rs = mapsafe_recursivetry_many_nothrow(arg); static_cast<bool>(rs)){
    //             return rs.value();
    //         }
    //     }
    // }
    // //vvv

    //undefined otherwise
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