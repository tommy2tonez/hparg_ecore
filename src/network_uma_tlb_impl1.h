#ifndef __NETWORK_UMA_TLB_IMPL1_H__
#define __NETWORK_UMA_TLB_IMPL1_H__

#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <type_traits>
#include <atomic>
#include <optional>
#include <iterator>
#include "network_memult.h" 
#include "network_memlock_proxyspin.h" 
#include <limits.h>
#include "network_log.h"
#include "network_segcheck_bound.h"
#include "network_exception.h"
#include "network_type_traits_x.h" 
#include "network_uma_tlb.h" 
#include "stdx.h"
#include "dense_hash_map/dense_hash_map.hpp" 

namespace dg::network_uma_tlb_impl1::interface{

    template <class T>
    struct ProxyTLBInterface{

        using interface_t   = ProxyTLBInterface<T>;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using device_id_t   = typename T1::device_id_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using uma_ptr_t     = typename T1::uma_ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using vma_ptr_t     = typename T1::vma_ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto map_try(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept -> typename T1::vma_ptr_t{

            return T::map_try(device_id, host_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto map_wait(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept -> typename T1::vma_ptr_t{

            return T::map_wait(device_id, host_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void map_release(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept{

            T::map_release(device_id, host_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto remap_try(typename T1::device_id_t device_id, typename T1::uma_ptr_t new_host_ptr, typename T1::device_id_t old_device_id, typename T1::uma_ptr_t old_host_ptr) noexcept -> typename T1::vma_ptr_t{

            return T::remap_try(device_id, new_host_ptr, old_device_id, old_host_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto remap_wait(typename T1::device_id_t device_id, typename T1::uma_ptr_t new_host_ptr, typename T1::device_id_t old_device_id, typename T1::uma_ptr_t old_host_ptr) noexcept -> typename T1::vma_ptr_t{

            return T::remap_wait(device_id, new_host_ptr, old_device_id, old_host_ptr);
        }
    };

    template <class T>
    struct MemoryCopyDeviceInterface{

        using interface_t   = MemoryCopyDeviceInterface<T>; 

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using ptr_t         = typename T1::ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void memcpy(typename T1::ptr_t dst, typename T1::ptr_t src, size_t n) noexcept{
            
            T::memcpy(dst, src, n);
        }
    };
}

namespace dg::network_uma_tlb_impl1::translation_table_impl{

    static inline constexpr bool IS_SAFE_ACCESS_ENABLED     = true;
    static inline constexpr size_t DG_MAX_DEVICE            = 128;

    template <class ID, class SrcPtrType, class DstPtrType, size_t MEMREGION_SZ>
    class TranslationTable{

        public:

            static_assert(MEMREGION_SZ != 0u);

            using src_ptr_t = SrcPtrType;
            using dst_ptr_t = DstPtrType;

        private:

            static inline std::unique_ptr<dst_ptr_t[]> translation_table{};
            static inline src_ptr_t region_first{}; 

            using self              = TranslationTable;
            using segcheck_ins      = dg::network_segcheck_bound::StdAccess<self, src_ptr_t>;

            static inline auto memregion_slot(src_ptr_t ptr) noexcept -> size_t{
 
                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static inline auto memregion_offset(src_ptr_t ptr) noexcept -> size_t{

                using src_uptr_t = typename dg::ptr_info<src_ptr_t>::max_unsigned_t;
                return dg::pointer_cast<src_uptr_t>(ptr) % MEMREGION_SZ;
            }

        public:

            static_assert(stdx::is_pow2(MEMREGION_SZ));

            static void init(src_ptr_t * src_region_arr, dst_ptr_t * dst_region_arr, size_t n){

                if (n == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                for (size_t i = 0u; i < n; ++i){
                    using src_uptr_t = typename dg::ptr_info<src_ptr_t>::max_unsigned_t;
                    using dst_uptr_t = typename dg::ptr_info<dst_ptr_t>::max_unsigned_t; 

                    if (dg::pointer_cast<src_uptr_t>(src_region_arr[i]) % MEMREGION_SZ != 0u){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }

                    if (dg::pointer_cast<dst_uptr_t>(dst_region_arr[i]) % MEMREGION_SZ != 0u){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }

                    if (src_region_arr[i] == dg::ptr_limits<src_ptr_t>::null_value()){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }

                    if (dst_region_arr[i] == dg::ptr_limits<dst_ptr_t>::null_value()){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }
                }

                src_ptr_t first_src_region  = *std::min_element(src_region_arr, src_region_arr + n, dg::memult::ptrcmpless_lambda);
                src_ptr_t last_src_region   = dg::memult::advance(*std::max_element(src_region_arr, src_region_arr + n, dg::memult::ptrcmpless_lambda), MEMREGION_SZ);
                size_t table_sz             = dg::memult::distance(first_src_region, last_src_region) / MEMREGION_SZ;
                translation_table           = std::make_unique<dst_ptr_t[]>(table_sz); //kmalloc - to avoid automatic virtual memory allocation by kernel
                region_first                = first_src_region;

                for (size_t i = 0u; i < n; ++i){
                    size_t table_idx = dg::memult::distance(first_src_region, src_region_arr[i]) / MEMREGION_SZ;
                    translation_table[table_idx] = dst_region_arr[i];
                }

                segcheck_ins::init(first_src_region, last_src_region);
            }

            static void deinit() noexcept{

                translation_table = nullptr;
            }

            static auto translate(src_ptr_t ptr) noexcept -> dst_ptr_t{
                
                ptr         = segcheck_ins::access(ptr);
                size_t idx  = memregion_slot(ptr);
                size_t off  = memregion_offset(ptr);

                if constexpr(IS_SAFE_ACCESS_ENABLED){
                    if (translation_table[idx] == dg::ptr_limits<dst_ptr_t>::null_value()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::SEGFAULT));
                        std::abort();
                    }
                }

                return dg::memult::advance(translation_table[idx], off);
            }
    };

    template <class ID, class UMAPtrType, class VMAPtrType, class DeviceIDType, size_t MEMREGION_SZ>
    class UMATranslationTable{

        public:

            using uma_ptr_t         = UMAPtrType;
            using vma_ptr_t         = VMAPtrType;
            using device_id_t       = DeviceIDType;

        private:

            using self              = UMATranslationTable;
            using virtual_ptr_t     = typename dg::ptr_info<>::max_unsigned_t;

            static inline jg::dense_hash_map<virtual_ptr_t, vma_ptr_t> translation_table{};

            static inline auto to_virtual_region(uma_ptr_t region, device_id_t device_id) noexcept -> virtual_ptr_t{

                return dg::memult::advance(dg::pointer_cast<virtual_ptr_t>(dg::pointer_cast<typename dg::ptr_info<uma_ptr_t>::max_unsigned_t>(region) * DG_MAX_DEVICE), stdx::safe_integer_cast<size_t>(device_id) * MEMREGION_SZ);
            }

        public:

            static_assert(stdx::is_pow2(MEMREGION_SZ));
            static_assert(std::numeric_limits<device_id_t>::is_integer);

            static void init(uma_ptr_t * uma_region_arr, vma_ptr_t * vma_region_arr, device_id_t * device_id_arr, size_t n){

                for (size_t i = 0u; i < n; ++i){
                    if (!dg::memult::is_region(uma_region_arr[i], std::integral_constant<size_t, MEMREGION_SZ>{})){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }

                    if (!dg::memult::is_region(vma_region_arr[i], std::integral_constant<size_t, MEMREGION_SZ>{})){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }

                    if (device_id_arr[i] < 0){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }

                    translation_table[to_virtual_region(uma_region_arr[i], device_id_arr[i])] = vma_region_arr[i];
                }
            }

            static void deinit() noexcept{

                translation_table = {};
            }

            static auto translate(device_id_t device_id, uma_ptr_t ptr) noexcept -> vma_ptr_t{

                auto map_key    = to_virtual_region(dg::memult::region(ptr, std::integral_constant<size_t, MEMREGION_SZ>{}), device_id);
                auto map_ptr    = stdx::to_const_reference(translation_table).find(map_key);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == stdx::to_const_reference(translation_table).end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return dg::memult::advance(map_ptr->second, dg::memult::region_offset(ptr, std::integral_constant<size_t, MEMREGION_SZ>{}));
            }
    };
}

namespace dg::network_uma_tlb_impl1::exclusive{

    using namespace interface;

    template <class ID, class MemCopyDevice, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize>
    struct ProxyTLB{};

    template <class ID, class T, class DeviceIDType, class UMAPtrType, class VMAPtrType, size_t MEMREGION_SZ>
    struct ProxyTLB<ID, MemoryCopyDeviceInterface<T>, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>>: ProxyTLBInterface<ProxyTLB<ID, MemoryCopyDeviceInterface<T>, DeviceIDType, VMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>>>{                                                                                                                                               

        public:

            using device_id_t           = DeviceIDType;
            using uma_ptr_t             = UMAPtrType;
            using vma_ptr_t             = dg::network_type_traits_x::mono_reduction_type_t<VMAPtrType, typename MemoryCopyDeviceInterface<T>::ptr_t<>>;

        private:
            
            using self                  = ProxyTLB;
            using memcopy_device        = MemoryCopyDeviceInterface<T>;
            using uma_proxy_lock        = dg::network_memlock_proxyspin::ReferenceLock<self, std::integral_constant<size_t, MEMREGION_SZ>, uma_ptr_t, device_id_t>;
            using translation_table     = translation_table_impl::UMATranslationTable<self, uma_ptr_t, vma_ptr_t, device_id_t, MEMREGION_SZ>;

            static auto memregion(uma_ptr_t ptr) noexcept -> uma_ptr_t{
                
                using uptr_t = typename dg::ptr_info<uma_ptr_t>::max_unsigned_t;
                constexpr uptr_t BITMASK = ~static_cast<uptr_t>(MEMREGION_SZ - 1);
                return dg::pointer_cast<uma_ptr_t>(dg::pointer_cast<uptr_t>(ptr) & BITMASK);
            }

            static auto steal_try(device_id_t stealer_id, uma_ptr_t host_ptr) noexcept -> bool{

                uma_ptr_t host_region = memregion(host_ptr);
                std::optional<device_id_t> potential_stealee_id = uma_proxy_lock::acquire_try(host_region);  

                if (!potential_stealee_id.has_value()){
                    return false;
                }

                //mem tx payload begin
                std::atomic_thread_fence(std::memory_order_acquire);
                device_id_t stealee_id = potential_stealee_id.value();

                if (stealee_id != stealer_id){
                    vma_ptr_t dst   = translation_table::translate(stealer_id, host_region);
                    vma_ptr_t src   = translation_table::translate(stealee_id, host_region);
                    size_t cpy_sz   = MEMREGION_SZ;
                    memcopy_device::memcpy(dst, src, cpy_sz);
                }

                std::atomic_thread_fence(std::memory_order_release);
                //mem tx payload end

                std::atomic_signal_fence(std::memory_order_seq_cst);
                uma_proxy_lock::acquire_release(host_region, stealer_id, dg::network_memlock_proxyspin::increase_reference_tag{}); //make sure to release this post the payload, by calling the signal_fence_seq_cst

                return true;
            }

        public:

            static_assert(stdx::is_pow2(MEMREGION_SZ));

            static void init(uma_ptr_t * uma_region_arr, vma_ptr_t * vma_region_arr, device_id_t * device_id_arr, bool * is_proxy_rep_arr, size_t n){

                auto uma_rep_region_arr     = std::make_unique<uma_ptr_t[]>(n);
                auto uma_rep_device_id_arr  = std::make_unique<device_id_t[]>(n);
                size_t rep_sz               = 0u; 

                for (size_t i = 0u; i < n; ++i){
                    if (is_proxy_rep_arr[i]){
                        uma_rep_region_arr[rep_sz]      = uma_region_arr[i];
                        uma_rep_device_id_arr[rep_sz]   = device_id_arr[i];
                        ++rep_sz;
                    }
                }

                translation_table::init(uma_region_arr, vma_region_arr, device_id_arr, n);
                uma_proxy_lock::init(uma_rep_region_arr.get(), uma_rep_device_id_arr.get(), rep_sz);
            }

            static void deinit() noexcept{

                translation_table::deinit();
                uma_proxy_lock::deinit();
            }

            static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                if (uma_proxy_lock::reference_try(host_ptr, device_id)){
                    return translation_table::translate(device_id, host_ptr);
                }

                if (steal_try(device_id, host_ptr)){
                    return translation_table::translate(device_id, host_ptr);
                }

                return dg::ptr_limits<vma_ptr_t>::null_value();
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                while (true){
                    if (auto rs = map_try(device_id, host_ptr); rs != dg::ptr_limits<vma_ptr_t>::null_value()){
                        return rs;
                    }
                }
            }

            //what is the problem with THIS
            //vma_ptr_t is written, compiler cant prove that vma_ptr_t is related to device_id + host_ptr, vma_ptr_t is now not updated
            //it seems like a seq_cst block problem

            static void map_release(device_id_t device_id, uma_ptr_t host_ptr) noexcept{

                // std::atomic_thread_fence(std::memory_order_release);
                // std::atomic_signal_fence(std::memory_order_seq_cst);

                //we are not doing memmory safe operations
                //because this is a reference operation, which does not guarantee anything except for returning the pointer, another serialization instruction is not this component responsibility
                //this is an accepted answer, because if this component is doing locks + releases operation, then the memory ordering is this guy responsibility, it is actually still not this guy responsibility, but the lock_guard responsibility
                //we have been working on this problem for the longest time EVER
                //I'm telling you that this is harder to split than we think
                //if we are to put a memory ordering here, it's entirely wrong

                uma_proxy_lock::reference_release(host_ptr);
            }

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, device_id_t old_device_id, uma_ptr_t old_host_ptr) noexcept -> vma_ptr_t{

                if (memregion(old_host_ptr) == memregion(new_host_ptr) && device_id == old_device_id){
                    return translation_table::translate(device_id, new_host_ptr);
                }

                return dg::ptr_limits<vma_ptr_t>::null_value();
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, device_id_t old_device_id, uma_ptr_t old_host_ptr) noexcept -> vma_ptr_t{

                if (auto rs = remap_try(device_id, new_host_ptr, old_device_id, old_host_ptr); rs != dg::ptr_limits<vma_ptr_t>::null_value()){
                    return rs;
                }

                map_release(old_device_id, old_host_ptr);
                return map_wait(device_id, new_host_ptr);
            }
    };
}

namespace dg::network_uma_tlb_impl1::direct{

    using namespace interface; 

    template <class ID, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize>
    class ProxyTLB{};

    template <class ID, class DeviceIDType, class UMAPtrType, class VMAPtrType, size_t MEMREGION_SZ>
    class ProxyTLB<ID, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>>: public ProxyTLBInterface<ProxyTLB<ID, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>>>{

        public:

            using device_id_t       = DeviceIDType;
            using uma_ptr_t         = UMAPtrType;
            using vma_ptr_t         = VMAPtrType; 

        private:

            using self              = ProxyTLB;
            using translation_table = translation_table_impl::UMATranslationTable<self, uma_ptr_t, vma_ptr_t, device_id_t, MEMREGION_SZ>;

        public:

            static void init(uma_ptr_t * uma_region_arr, vma_ptr_t * vma_region_arr, device_id_t * device_id_arr, size_t n){
                
                translation_table::init(uma_region_arr, vma_region_arr, device_id_arr, n);
            }

            static void deinit() noexcept{

                translation_table::deinit();
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                translation_table::translate(device_id, host_ptr);
            }

            static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                return translation_table::translate(device_id, host_ptr);
            }

            static void map_release(device_id_t arg, uma_ptr_t) noexcept{

                (void) arg;
            } 

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, device_id_t old_device_id, uma_ptr_t old_host_ptr) noexcept -> vma_ptr_t{

                return translation_table::translate(device_id, new_host_ptr);
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, device_id_t old_device_id, uma_ptr_t old_host_ptr) noexcept -> vma_ptr_t{
                
                return translation_table::translate(device_id, new_host_ptr);
            }
    };
} 

namespace dg::network_uma_tlb_impl1::bijective{
    
    using namespace interface;

    template <class ID, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize>
    class ProxyTLB{};

    template <class ID, class DeviceIDType, class UMAPtrType, class VMAPtrType, size_t MEMREGION_SZ>
    class ProxyTLB<ID, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>>: public ProxyTLBInterface<ProxyTLB<ID, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>>>{

        public:

            using device_id_t       = DeviceIDType;
            using uma_ptr_t         = UMAPtrType;
            using vma_ptr_t         = VMAPtrType;

        private:

            using self              = ProxyTLB;
            using translation_table = translation_table_impl::TranslationTable<self, uma_ptr_t, vma_ptr_t, MEMREGION_SZ>; 

        public:

            static void init(uma_ptr_t * uma_region_arr, vma_ptr_t * vma_region_arr, size_t n){
                
                translation_table::init(uma_region_arr, vma_region_arr, n); 
            }

            static void deinit() noexcept{

                translation_table::deinit();
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                return translation_table::translate(host_ptr);
            }

            static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                return translation_table::translate(host_ptr);
            }

            static void map_release(device_id_t arg, uma_ptr_t) noexcept{

                (void) arg;
            }

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, device_id_t old_device_id, uma_ptr_t old_host_ptr) noexcept -> vma_ptr_t{

                return translation_table::translate(new_host_ptr);
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, device_id_t old_device_id, uma_ptr_t old_host_ptr) noexcept -> vma_ptr_t{
                
                return translation_table::translate(new_host_ptr);
            }
    };
}

namespace dg::network_uma_tlb_impl1::biex{

    using namespace interface;
    
    template <class ID, class MemCopyDevice, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize>
    class ProxyTLB{}; 

    template <class ID, class T, class DeviceIDType, class UMAPtrType, class VMAPtrType, size_t MEMREGION_SZ>
    class ProxyTLB<ID, MemoryCopyDeviceInterface<T>, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>>: public ProxyTLBInterface<ProxyTLB<ID, MemoryCopyDeviceInterface<T>, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>>>{

        public:

            using device_id_t           = DeviceIDType; 
            using uma_ptr_t             = UMAPtrType;
            using vma_ptr_t             = VMAPtrType;

        private:

            using self                  = ProxyTLB;
            using bijective_tlb         = network_uma_tlb_impl1::bijective::ProxyTLB<self, device_id_t, uma_ptr_t, vma_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>>; 
            using exclusive_tlb         = network_uma_tlb_impl1::exclusive::ProxyTLB<self, MemoryCopyDeviceInterface<T>, device_id_t, uma_ptr_t, vma_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>>;
            using segcheck_ins          = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline constexpr uint8_t DISPATCH_CODE_BIJECTIVE     = 0u;
            static inline constexpr uint8_t DISPATCH_CODE_EXCLUSIVE     = 1u;
            static inline std::unique_ptr<uint8_t[]> dispatch_table     = nullptr;
            static inline uma_ptr_t first_region                        = {}; 

            static auto memregion_slot(uma_ptr_t ptr) noexcept -> size_t{
 
                return dg::memult::distance(first_region, ptr) / MEMREGION_SZ;
            }

        public:

            static_assert(stdx::is_pow2(MEMREGION_SZ));

            static void init(uma_ptr_t * uma_region_arr, vma_ptr_t * vma_region_arr, device_id_t * device_id_arr, bool * is_proxy_rep_arr, size_t n){

                if (n == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                for (size_t i = 0u; i < n; ++i){
                    using uma_uptr_t = dg::ptr_info<uma_ptr_t>::max_unsigned_t; 

                    if (dg::pointer_cast<uma_uptr_t>(uma_region_arr[i]) % MEMREGION_SZ != 0u){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }

                    if (uma_region_arr[i] == dg::ptr_limits<uma_ptr_t>::null_value()){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }
                }

                uma_ptr_t uma_region_first      = *std::min_element(uma_region_arr, uma_region_arr + n, dg::memult::ptrcmpless_lambda);
                uma_ptr_t uma_region_last       = dg::memult::advance(*std::max_element(uma_region_arr, uma_region_arr + n, dg::memult::ptrcmpless_lambda), MEMREGION_SZ);
                auto counter_map                = std::unordered_map<uma_ptr_t, size_t>{};
                auto bijective_uma_region_arr   = std::make_unique<uma_ptr_t[]>(n);
                auto bijective_vma_region_arr   = std::make_unique<vma_ptr_t[]>(n);
                size_t bijective_arr_sz         = 0u;
                size_t dispatch_table_sz        = dg::memult::distance(uma_region_first, uma_region_last) / MEMREGION_SZ;
                dispatch_table                  = std::make_unique<uint8_t[]>(dispatch_table_sz);
                first_region                    = uma_region_first;

                for (size_t i = 0u; i < n; ++i){
                    counter_map[uma_region_arr[i]] += 1;
                }

                for (size_t i = 0u; i < n; ++i){
                    size_t table_idx = dg::memult::distance(uma_region_first, uma_region_arr[i]) / MEMREGION_SZ;

                    if (counter_map[uma_region_arr[i]] == 1u){
                        bijective_uma_region_arr[bijective_arr_sz] = uma_region_arr[i];
                        bijective_vma_region_arr[bijective_arr_sz] = vma_region_arr[i];
                        bijective_arr_sz += 1;
                        dispatch_table[table_idx] = DISPATCH_CODE_BIJECTIVE;
                    } else{
                        dispatch_table[table_idx] = DISPATCH_CODE_EXCLUSIVE;
                    }
                }

                exclusive_tlb::init(uma_region_arr, vma_region_arr, device_id_arr, is_proxy_rep_arr, n);
                bijective_tlb::init(bijective_uma_region_arr.get(), bijective_vma_region_arr.get(), bijective_arr_sz);
                segcheck_ins::init(uma_region_first, uma_region_last);
            }

            static void deinit() noexcept{

                bijective_tlb::deinit();
                exclusive_tlb::deinit();
                dispatch_table = {};
            }

            static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                uint8_t dispatch_code = dispatch_table[memregion_slot(segcheck_ins::access(host_ptr))];

                if (dispatch_code == DISPATCH_CODE_BIJECTIVE){
                    return bijective_tlb::map_try(device_id, host_ptr);
                }

                return exclusive_tlb::map_try(device_id, host_ptr);
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                uint8_t dispatch_code = dispatch_table[memregion_slot(segcheck_ins::access(host_ptr))];

                if (dispatch_code == DISPATCH_CODE_BIJECTIVE){
                    return bijective_tlb::map_wait(device_id, host_ptr);
                }

                return exclusive_tlb::map_wait(device_id, host_ptr);
            }

            static void map_release(device_id_t device_id, uma_ptr_t host_ptr) noexcept{

                uint8_t dispatch_code = dispatch_table[memregion_slot(segcheck_ins::access(host_ptr))];

                if (dispatch_code == DISPATCH_CODE_BIJECTIVE){
                    bijective_tlb::map_release(device_id, host_ptr);
                    return;
                }

                exclusive_tlb::map_release(device_id, host_ptr);
            }

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, device_id_t old_device_id, uma_ptr_t old_host_ptr) noexcept -> vma_ptr_t{
                
                uint8_t old_dispatch_code   = dispatch_table[memregion_slot(segcheck_ins::access(old_host_ptr))];
                uint8_t new_dispatch_code   = dispatch_table[memregion_slot(segcheck_ins::access(new_host_ptr))];

                if (old_dispatch_code != new_dispatch_code){
                    return dg::ptr_limits<vma_ptr_t>::null_value(); 
                }

                if (old_dispatch_code == DISPATCH_CODE_BIJECTIVE){
                    return bijective_tlb::remap_try(device_id, new_host_ptr, old_device_id, old_host_ptr);
                }

                return exclusive_tlb::remap_try(device_id, new_host_ptr, old_device_id, old_host_ptr);
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, device_id_t old_device_id, uma_ptr_t old_host_ptr) noexcept -> vma_ptr_t{

                if (auto rs = remap_try(device_id, new_host_ptr, old_device_id, old_host_ptr); rs != dg::ptr_limits<vma_ptr_t>::null_value()){
                    return rs;
                }

                map_release(old_device_id, old_host_ptr);
                return map_wait(device_id, new_host_ptr);
            }
    };
}

namespace dg::network_uma_tlb_impl1::generic{

    template <class ID, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize>
    class DirectTLB: public dg::network_uma_tlb::interface::DirectTLBInterface<DirectTLB<ID, DeviceIDType, UMAPtrType, VMAPtrType, MemregionSize>>{

        private:

            using self = DirectTLB;
            using base = network_uma_tlb_impl1::direct::ProxyTLB<self, DeviceIDType, UMAPtrType, VMAPtrType, MemregionSize>;

        public:

            using device_id_t   = DeviceIDType;
            using uma_ptr_t     = UMAPtrType;
            using vma_ptr_t     = VMAPtrType; 

            static void init(uma_ptr_t * uma_region_arr, vma_ptr_t * vma_region_arr, device_id_t * device_id_arr, size_t n){

                base::init(uma_region_arr, vma_region_arr, device_id_arr, n);
            }

            static void deinit() noexcept{

                base::deinit();
            }

            static auto map(device_id_t device_id, uma_ptr_t ptr) noexcept -> vma_ptr_t{

                return base::map_wait(device_id, ptr);
            }
    };

    template <class ID, class MemCopyDevice, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize>
    class BiexTLB: public dg::network_uma_tlb::interface::MutexRegionTLBInterface<BiexTLB<ID, MemCopyDevice, DeviceIDType, UMAPtrType, VMAPtrType, MemregionSize>>{

        private:

            using self  = BiexTLB;
            using base  = network_uma_tlb_impl1::biex::ProxyTLB<self, MemCopyDevice, DeviceIDType, UMAPtrType, VMAPtrType, MemregionSize>;

        public:

            using device_id_t           = DeviceIDType;
            using uma_ptr_t             = UMAPtrType;
            using vma_ptr_t             = VMAPtrType;

            struct MapResource{
                device_id_t device_id;
                uma_ptr_t mapping_ptr;
                vma_ptr_t mapped_ptr;
            };

            using map_resource_handle_t = MapResource;

            static void init(uma_ptr_t * uma_region_arr, vma_ptr_t * vma_region_arr, device_id_t * device_id_arr, bool * is_proxy_rep_arr, size_t n){

                base::init(uma_region_arr, vma_region_arr, device_id_arr, is_proxy_rep_arr, n);
            }

            static void deinit() noexcept{

                base::deinit();
            }

            static auto map_try(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<MapResource>{

                vma_ptr_t rs = base::map_try(device_id, ptr);

                if (rs == dg::ptr_limits<vma_ptr_t>::null_value()){
                    return std::nullopt;
                }

                return MapResource{device_id, ptr, rs};
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t ptr) noexcept -> MapResource{

                return MapResource{device_id, ptr, base::map_wait(device_id, ptr)};
            }

            static void map_release(MapResource map_resource) noexcept{

                base::map_release(map_resource.device_id, map_resource.mapping_ptr);
            }

            static auto remap_try(device_id_t device_id, uma_ptr_t ptr, MapResource resource) noexcept -> std::optional<MapResource>{
                
                vma_ptr_t rs = base::remap_try(device_id, ptr, resource.device_id, resource.mapping_ptr);
                
                if (rs == dg::ptr_limits<vma_ptr_t>::null_value()){
                    return std::nullopt;
                }

                return MapResource{device_id, ptr, rs};
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t ptr, MapResource resource) noexcept -> MapResource{

                vma_ptr_t rs = base::remap_wait(device_id, ptr, resource.device_id, resource.mapping_ptr);
                return MapResource{device_id, ptr, rs};
            }

            static auto get_vma_ptr(MapResource resource) noexcept -> vma_ptr_t{

                return resource.mapped_ptr;
            }

            static auto memregion_size() noexcept -> size_t{

                return MemregionSize{}; 
            }
    };
}

#endif