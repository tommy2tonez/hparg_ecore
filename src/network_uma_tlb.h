#ifndef __DG_NETWORK_UMA_TLB_H__
#define __DG_NETWORK_UMA_TLB_H__

#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <type_traits>
#include "network_log.h"
#include "network_std_container.h"
#include "network_exception_handler.h"
#include "network_concurrency.h"
#include <tuple>
#include <array>
#include "stdx.h"
#include "network_type_traits_x.h"

namespace dg::network_uma_tlb::interface{

    template <class T>
    struct MutexTLBInterface{

        using interface_t           = MutexTLBInterface<T>;

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
        static auto map_try_strong(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept -> std::optional<typename T1::map_resource_handle_t>{

            return T::map_try_strong(device_id, host_ptr);
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
        static void map_waitonly(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept{

            T::map_waitonly(device_id, host_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void map_waitonly_release_responsibility(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept{

            T::map_waitonly_release_responsibility(device_id, host_ptr);
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
        static auto get_vma_ptr(stdx::add_const_reference_t<typename T1::map_resource_handle_t> map_resource) noexcept -> typename T1::vma_ptr_t{

            return T::get_vma_ptr(map_resource);
        }
    };

    template <class T>
    struct MutexRegionTLBInterface: MutexTLBInterface<T>{

        using interface_t = MutexRegionTLBInterface<T>; 

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
}

namespace dg::network_uma_tlb::rec_lck{

    using namespace interface;

    template <class TLBInterface>
    struct MapResource{
        typename TLBInterface::uma_ptr_t<> region;
        typename TLBInterface::map_resource_handle_t<> map_resource;
        bool responsibility_flag;
        size_t offset;
    };

    template <class T>
    struct RecursiveMapController{};

    template <class T>
    struct RecursiveMapController<MutexRegionTLBInterface<T>>{

        public:

            static inline constexpr size_t MAX_KEY_PER_THREAD = size_t{1} << 16; 

            using tlb_ins           = MutexRegionTLBInterface<T>;
            using key_t             = typename tlb_ins::uma_ptr_t<>;
            using value_t           = std::pair<typename tlb_ins::device_id_t<>, typename tlb_ins::map_resource_handle_t<>>;

        private:

            using self              = RecursiveMapController;
            using singleton_object  = stdx::singleton<self, std::array<dg::unordered_unstable_map<key_t, value_t>, dg::network_concurrency::THREAD_COUNT>>;

        public:

            static void insert(key_t key, value_t value) noexcept{
                
                auto& map = singleton_object::get()[dg::network_concurrency::this_thread_idx()];

                if (map.size() == MAX_KEY_PER_THREAD + 1){ 
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    std::abort();
                }

                map[key] = value;
            }

            static auto map(key_t key) noexcept -> std::optional<value_t>{

                const auto& map_ins = singleton_object::get()[dg::network_concurrency::this_thread_idx()];
                auto ptr            = map_ins.find(key);

                if (ptr == map_ins.end()){
                    return std::nullopt;
                }

                return ptr->second;
            }

            static void erase(key_t key) noexcept{

                auto& map = singleton_object::get()[dg::network_concurrency::this_thread_idx()];
                map.erase(key);
            }
    };

    template <class device_id_t, class arg_ptr_t, size_t SZ>
    constexpr auto sort_ptr_array(const std::array<std::pair<device_id_t, arg_ptr_t>, SZ>& inp) noexcept -> std::array<std::pair<device_id_t, arg_ptr_t>, SZ>{

        static_assert(dg::ptr_info<T>::is_pointer);

        // std::array<T, SZ> rs = inp;
        // std::sort(rs.begin(), rs.end());

        return rs;
    }

    template <class T>
    auto recursive_resource_type(const MutexRegionTLBInterface<T>) -> MapResource<MutexRegionTLBInterface<T>>;

    template <class T>
    auto recursive_lockmap_try(const MutexRegionTLBInterface<T>, 
                               typename MutexRegionTLBInterface<T>::device_id_t<> device_id,
                               typename MutexRegionTLBInterface<T>::uma_ptr_t<> ptr) noexcept{

        using tlb_ins                   = MutexRegionTLBInterface<T>;
        using controller_ins            = RecursiveMapController<MutexRegionTLBInterface<T>>;
        using map_ptr_t                 = typename tlb_ins::uma_ptr_t<>;
        using device_id_t               = typename tlb_ins::device_id_t<>;
        using map_resource_t            = typename tlb_ins::map_resource_handle_t<>; 
        using recursive_map_resource_t  = MapResource<tlb_ins>;

        map_ptr_t region                                                = dg::memult::region(ptr, tlb_ins::memregion_size());
        size_t offset                                                   = dg::memult::region_offset(ptr, tlb_ins::memregion_size());
        std::optional<std::pair<device_id_t, map_resource_t>> mapped    = controller_ins::map(region);

        auto destructor = [](recursive_map_resource_t arg) noexcept{
            if (arg.responsibility_flag){
                controller_ins::erase(arg.region);
                tlb_ins::map_release(arg.map_resource);
            }
        };

        if (mapped.has_value()){
            if constexpr(DEBUG_MODE_FLAG){
                if (mapped->first != device_id){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }    
            }

            recursive_map_resource_t recursive_resource{.region                 = region,
                                                        .map_resource           = mapped->second,
                                                        .responsibility_flag    = false,
                                                        .offset                 = offset};  

            return std::optional<dg::unique_resource<recursive_map_resource_t, decltype(destructor)>>(dg::unique_resource<recursive_map_resource_t, decltype(destructor)>(recursive_resource, std::move(destructor)));
        }

        std::optional<map_resource_t> map_rs = tlb_ins::map_try(device_id, region);

        if (map_rs.has_value()){
            controller_ins::insert(region, std::make_pair(device_id, map_rs.value()));
            
            recursive_map_resource_t recursive_resource{.region                 = region,
                                                        .map_resource           = map_rs.value(),
                                                        .responsibility_flag    = true,
                                                        .offset                 = offset};

            return std::optional<dg::unique_resource<recursive_map_resource_t, decltype(destructor)>>(dg::unique_resource<recursive_map_resource_t, decltype(destructor)>(recursive_resource, std::move(destructor)));
        }

        return std::optional<dg::unique_resource<recursive_map_resource_t, decltype(destructor)>>(std::nullopt);
    }

    template <class T>
    auto recursive_lockmap_wait(const MutexRegionTLBInterface<T> tlb, typename MutexRegionTLBInterface<T>::device_id_t<> device_id, typename MutexRegionTLBInterface<T>::uma_ptr_t<> ptr) noexcept{

        using tlb_ins                   = MutexRegionTLBInterface<T>;
        using controller_ins            = RecursiveMapController<MutexRegionTLBInterface<T>>;
        using map_ptr_t                 = typename tlb_ins::uma_ptr_t<>;
        using device_id_t               = typename tlb_ins::device_id_t<>;
        using map_resource_t            = typename tlb_ins::map_resource_handle_t<>; 
        using recursive_map_resource_t  = MapResource<tlb_ins>;

        map_ptr_t region                                                = dg::memult::region(ptr, tlb_ins::memregion_size());
        size_t offset                                                   = dg::memult::region_offset(ptr, tlb_ins::memregion_size());
        std::optional<std::pair<device_id_t, map_resource_t>> mapped    = controller_ins::map(region);

        auto destructor = [](recursive_map_resource_t arg) noexcept{
            if (arg.responsibility_flag){
                controller_ins::erase(arg.region);
                tlb_ins::map_release(arg.map_resource);
            }
        };

        if (mapped.has_value()){
            if constexpr(DEBUG_MODE_FLAG){
                if (mapped->first != device_id){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }    
            }

            recursive_map_resource_t recursive_resource{.region                 = region,
                                                        .map_resource           = mapped->second,
                                                        .responsibility_flag    = false,
                                                        .offset                 = offset};  

            return dg::unique_resource<recursive_map_resource_t, decltype(destructor)>(recursive_resource, std::move(destructor));
        }

        map_resource_t map_rs = tlb_ins::map_wait(device_id, region);
        controller_ins::insert(region, std::make_pair(device_id, map_rs));

        recursive_map_resource_t recursive_resource{.region                 = region,
                                                    .map_resource           = map_rs,
                                                    .responsibility_flag    = true,
                                                    .offset                 = offset};

        return dg::unique_resource<recursive_map_resource_t, decltype(destructor)>(recursive_resource, std::move(destructor));
    }

    template <size_t SZ, class T>
    auto recursive_lockmap_try_array(const MutexRegionTLBInterface<T> tlb, 
                                     const std::array<std::pair<typename MutexRegionTLBInterface<T>::device_id_t<>, typename MutexRegionTLBInterface<T>::uma_ptr_t<>>, SZ>& args){

        using element_t         = decltype(rec_lck::recursive_lockmap_try(tlb, typename MutexRegionTLBInterface<T>::device_id_t<>{}, typename MutexRegionTLBInterface<T>::uma_ptr_t<>{}));
        using resource_arr_t    = std::array<element_t, SZ>;
        resource_arr_t rs       = {};
        auto sorted_args        = rec_lck::sort_ptr_array(args);

        for (size_t i = 0u; i < sorted_args.size(); ++i){
            rs[i] = rec_lck::recursive_lockmap_try(tlb, sorted_args[i].first, sorted_args[i].second);

            if (!rs[i].has_value()){
                return std::optional<resource_arr_t>(std::nullopt);
            }
        }

        return std::optional<resource_arr_t>(std::move(rs));
    }

    //we have yet want to do the waiting + frens here YET
    //it's extremely very complicated to change that
    //we wont be super greedy for now
    //eventloop should solve the problem
    //we'll try to do uma_tlb_memregion_sz == memlock_memregion_sz, because the reference is fishy in the sense of fifo
    //so we'll do our best to make things worked, yet we'll offload this responsibility to the memlock shoulder

    //we have not solved the fifo problems of this guy
    //such is the very common case of shared_lock + exclusive lock for databases 
    //this is precisely why we always want the exclusive lock for the FIFO fling
    //YET we'd want to leverage shared_lock here for the reason being that we are mostly on the region and the priority of the sameregionness should be prioritized because the transfer is really very extra expensive 
    //this is extremely hard to implement, so we'd just tackle problems by problems, we'll try to compromise that at some point (the memlock)

    template <size_t SZ, class T>
    auto recursive_lockmap_wait_many(const MutexRegionTLBInterface<T> tlb,
                                     const std::array<std::pair<typename MutexRegionTLBInterface<T>::device_id_t<>, typename MutexRegionTLBInterface<T>::uma_ptr_t<>>, SZ>& args){


        static_assert(SZ != 0u);

        if constexpr(SZ == 1u){
            return rec_lck::recursive_lockmap_wait(tlb, args[0].first, args[0].second);
        } else{
            using try_lockmap_resource_t                                    = std::variant<decltype(rec_lck::recursive_lockmap_try(tlb, args[0].first, args[0].second)), decltype(rec_lck::recursive_lockmap_wait(tlb, args[0].first, args[0].second))>;
            constexpr size_t INNER_LOOP_BUSY_WAIT_MAX_EXPONENT              = 5u;
            constexpr size_t CYCLIC_EXPBACKOFF_SPIN_SZ                      = 16u;
            constexpr std::chrono::nanoseconds CYCLIC_EXPBACKOFF_WAIT_TIME  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds(100));
            constexpr size_t CYCLIC_EXPBACKOFF_REVOLUTION                   = 8u;

            auto sorted_args                                                = rec_lck::sort_ptr_array(args);
            std::optional<size_t> wait_idx                                  = std::nullopt;
            std::array<try_lockmap_resource_t, SZ> rs;

            auto task = [&]() noexcept{
                rs                                      = {};
                bool was_thru                           = true;
                std::optional<size_t> responsible_idx   = std::nullopt;
                size_t retry_exponent                   = 0u;

                if (wait_idx.has_value()){
                    MutexRegionTLBInterface<T>::map_waitonly(sorted_args[wait_idx.value()].first, sorted_args[wait_idx.value()].second);
                    responsible_idx = wait_idx;
                    wait_idx        = std::nullopt;
                }

                for (size_t i = 0u; i < SZ; ++i){
                    if (i != 0u && sorted_args[i] != sorted_args[i - 1]){ //
                        retry_exponent += 1u; //border index detection
                    }

                    auto inner_loop_task = [&]() noexcept{
                        rs[i] = rec_lck::recursive_lockmap_try(tlb, sorted_args[i].first, sorted_args[i].second);
                        return std::get<0>(rs[i]).has_value();
                    };

                    size_t retry_sz = size_t{1} << std::min(retry_exponent, INNER_LOOP_BUSY_WAIT_MAX_EXPONENT);
                    stdx::eventloop_competitive_spin(inner_loop_task, retry_sz);

                    if (responsible_idx.has_value()){
                        if (responsible_idx.value() == i){
                            responsible_idx = std::nullopt;
                        }
                    }

                    if (!std::get<0>(rs[i]).has_value()){
                        wait_idx    = i;
                        was_thru    = false;
                        break;
                    }
                }

                if (!was_thru){
                    for (size_t i = 0u; i < SZ; ++i){
                        size_t back_idx                         = (SZ - 1u) - i;
                        *stdx::volatile_access(&rs[back_idx])   = {};
                    }

                    if (responsible_idx.has_value()){
                        MutexRegionTLBInterface<T>::map_waitonly_release_responsibility(sorted_args[responsible_idx.value()].first, sorted_args[responsible_idx.value()].second);
                        responsible_idx = std::nullopt;
                    }

                    return false;
                }

                return true; //OK, good
            };

            bool was_expbackoff_thru = stdx::eventloop_cyclic_expbackoff_spin(task, CYCLIC_EXPBACKOFF_SPIN_SZ, CYCLIC_EXPBACKOFF_WAIT_TIME, CYCLIC_EXPBACKOFF_REVOLUTION);
            
            if (was_expbackoff_thru){
                return rs;
            }

            rs = {};

            for (size_t i = 0u; i < SZ; ++i){
                rs[i] = rec_lck::recursive_lockmap_wait(tlb, sorted_args[i].first, sorted_args[i].second);
            }

            return rs;
        }
    }

    template <class TLBInterface>
    auto get_vma_ptr(const MapResource<TLBInterface>& resource) noexcept -> typename TLBInterface::vma_ptr_t<>{

        auto region = dg::memult::region(TLBInterface::get_vma_ptr(resource.map_resource), TLBInterface::memregion_size()); 
        auto rs     = dg::memult::advance(region, resource.offset);
        
        return rs;
    }
}

namespace dg::network_uma_tlb::access{

    using namespace interface;

    static inline constexpr bool IS_SAFE_ACCESS_ENABLED = true;

    template <class ID, class DeviceIdType, class UMAPtrType, size_t MEMREGION_SZ>
    class MetadataGetter: public MetadataGetterInterface<MetadataGetter<ID, DeviceIdType, UMAPtrType, MEMREGION_SZ>>{

        public:

            using device_id_t   = DeviceIdType;
            using uma_ptr_t     = UMAPtrType;

        private:

            static inline dg::unordered_unstable_map<uma_ptr_t, dg::vector<device_id_t>> region_device_map{};  

        public:

            static void init(uma_ptr_t * uma_region_arr, device_id_t * device_id_arr, size_t n){

                for (size_t i = 0u; i < n; ++i){
                    region_device_map[uma_region_arr[i]].push_back(device_id_arr[i]);
                }
            }

            static void deinit() noexcept{

                region_device_map = {};
            }

            static auto device_count(uma_ptr_t host_ptr) noexcept -> size_t{

                uma_ptr_t region    = dg::memult::region(host_ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
                auto map_ptr        = stdx::to_const_reference(region_device_map).find(region);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == stdx::to_const_reference(region_device_map).end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return map_ptr->second.size();
            }

            static auto device_at(uma_ptr_t host_ptr, size_t idx) noexcept -> device_id_t{

                uma_ptr_t region    = dg::memult::region(host_ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
                auto map_ptr        = stdx::to_const_reference(region_device_map).find(region);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == stdx::to_const_reference(region_device_map).end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                } 

                return map_ptr->second[idx];
            }
    };

    template <class ID, class UMAPtrType, class DeviceIdType, size_t MEMREGION_SZ>
    class StdSafeRegionAccess: public SafePtrAccessInterface<StdSafeRegionAccess<ID, UMAPtrType, DeviceIdType, MEMREGION_SZ>>{

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
                
                if (!stdx::to_const_reference(umadevice_hash_set).contains(std::make_pair(ptr, id))){
                    return dg::network_exception::BAD_ACCESS;
                }

                return dg::network_exception::SUCCESS;
            }

            static auto safecthrow_access(uma_ptr_t ptr) noexcept -> exception_t{
                
                auto region_ptr = dg::memult::region(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});

                if (!stdx::to_const_reference(uma_hash_set).contains(ptr)){
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
