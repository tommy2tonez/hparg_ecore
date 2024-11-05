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

//ssd + RAID + friends are very fast - DDR4, DDR5 are expensive - so it's the program responsibility to optimize locality to achieve maximum speed and minimum expensive storage
//cuda + friends are of different league - it's 1 << 20 faster - so it's important to not waste any flop here - this is a bare metal programming of unified mutual exclusive memory address - not recommend anyone to read and understand this
//tensor combinatorial operations are expensive (linear - all_row (lhs) combines all columns (rhs)) - sometimes more expensive than a load from a slow speed storage - so the usage of slow storage devices are recommended 

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
        static auto remap_try(typename T1::device_id_t device_id, typename T1::uma_ptr_t new_host_ptr, typename T1::uma_ptr_t old_host_ptr, typename T1::vma_ptr_t old_device_ptr) noexcept -> typename T1::vma_ptr_t{

            return T::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto remap_wait(typename T1::device_id_t device_id, typename T1::uma_ptr_t new_host_ptr, typename T1::uma_ptr_t old_host_ptr, typename T1::vma_ptr_t old_device_ptr) noexcept -> typename T1::vma_ptr_t{

            return T::remap_wait(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
        }
    };

    template <class T>
    struct MemoryTransferDeviceInterface{

        using interface_t   = MemoryTransferDeviceInterface<T>; 

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using vma_ptr_t     = typename T1::vma_ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void memcpy(typename T1::vma_ptr_t dst, typename T1::vma_ptr_t src, size_t n) noexcept{
            
            T::memcpy(dst, src, n);
        }
    };
}

namespace dg::network_uma_tlb_impl1::translation_table_impl{

    static inline constexpr bool IS_SAFE_ACCESS_ENABLED = true;

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
                translation_table           = std::make_unique<dst_ptr_t[]>(table_sz);
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

    //it's better to be device_id * MEM_WIDTH + static_cast<uptr_t>(ptr) - its more cache-efficient this way - if memregion is reordered to achieve adjecency - otherwise use unordered_unstable_map
    //the problem with the multiplication there is that - it's a pipeline halt - but still better than a cache miss

    template <class ID, class UMAPtrType, class VMAPtrType, class DeviceIDType, size_t MEMREGION_SZ, size_t DEVICE_COUNT>
    class UMATranslationTable{

        public:

            using uma_ptr_t         = UMAPtrType;
            using vma_ptr_t         = VMAPtrType;
            using device_id_t       = DeviceIDType;

        private:

            using self              = UMATranslationTable;
            using virtual_ptr_t     = typename dg::ptr_info<>::max_unsigned_t;
            using base              = TranslationTable<self, virtual_ptr_t, vma_ptr_t, MEMREGION_SZ>;
            
            static auto to_virtual_ptr(uma_ptr_t ptr, device_id_t device_id) noexcept -> virtual_ptr_t{

                using uma_uptr_t = typename dg::ptr_info<uma_ptr_t>::max_unsigned_t;
                
                size_t idx                  = dg::pointer_cast<uma_uptr_t>(ptr) / MEMREGION_SZ;
                size_t off                  = dg::pointer_cast<uma_uptr_t>(ptr) % MEMREGION_SZ;
                size_t virtual_idx          = idx * DEVICE_COUNT + device_id;
                virtual_ptr_t virtual_ptr   = virtual_idx * MEMREGION_SZ + off;

                return virtual_ptr;
            } 

        public:

            static void init(uma_ptr_t * uma_ptr_arr, vma_ptr_t * vma_ptr_arr, device_id_t * device_id_arr, size_t n){

                auto virtual_ptr_arr = std::make_unique<virtual_ptr_t[]>(n);
                
                for (size_t i = 0u; i < n; ++i){
                    virtual_ptr_arr[i] = to_virtual_ptr(uma_ptr_arr[i], device_id_arr[i]);
                }

                base::init(virtual_ptr_arr.get(), vma_ptr_arr, n);
            }

            static void deinit() noexcept{

                base::deinit();
            }

            static auto translate(device_id_t device_id, uma_ptr_t ptr) noexcept -> vma_ptr_t{

                return base::translate(device_id, to_virtual_ptr(ptr, device_id));
            }
    };
} 

namespace dg::network_uma_tlb_impl1::exclusive{

    using namespace interface;

    template <class ID, class MemTransferDevice, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize, class ProxyCount>
    struct ProxyTLB{};

    template <class ID, class T, class DeviceIDType, class UMAPtrType, class VMAPtrType, size_t MEMREGION_SZ, size_t PROXY_COUNT>
    struct ProxyTLB<ID, MemoryTransferDeviceInterface<T>, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>: ProxyTLBInterface<ProxyTLB<ID, MemoryTransferDeviceInterface<T>, DeviceIDType, VMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>>{                                                                                                                                               

        public:

            using device_id_t           = DeviceIDType;
            using uma_ptr_t             = UMAPtrType;
            using vma_ptr_t             = dg::network_type_traits_x::mono_reduction_type_t<VMAPtrType, typename MemoryTransferDeviceInterface<T>::vma_ptr_t<>>;

        private:
            
            using self                  = ProxyTLB;
            using memtransfer_device    = MemoryTransferDeviceInterface<T>;
            using uma_proxy_lock        = dg::network_memlock_proxyspin::ReferenceLock<self, std::integral_constant<size_t, MEMREGION_SZ>, uma_ptr_t, device_id_t>;
            using translation_table     = translation_table_impl::UMATranslationTable<self, uma_ptr_t, vma_ptr_t, device_id_t, MEMREGION_SZ, PROXY_COUNT>;

            static auto memregion_slot(uma_ptr_t ptr) noexcept -> size_t{

                using uptr_t = typename dg::ptr_info<uma_ptr_t>::max_unsigned_t; 
                return dg::pointer_cast<uptr_t>(ptr) / MEMREGION_SZ;
            }

            static auto memregion(uma_ptr_t ptr) noexcept -> uma_ptr_t{
                
                using uptr_t = typename dg::ptr_info<uma_ptr_t>::max_unsigned_t;
                constexpr uptr_t BITMASK = ~static_cast<uptr_t>(MEMREGION_SZ - 1);
                return dg::pointer_cast<uma_ptr_t>(dg::pointer_cast<uptr_t>(ptr) & BITMASK);
            }

            static void dispatch_steal(device_id_t stealer_id, uma_ptr_t host_ptr) noexcept{

                uma_ptr_t host_region = memregion(host_ptr);
                std::optional<device_id_t> potential_stealee_id = uma_proxy_lock::acquire_try(host_region);  

                if (!potential_stealee_id.has_value()){
                    return;
                }

                device_id_t stealee_id = potential_stealee_id.value();

                if (stealee_id != stealer_id){
                    vma_ptr_t dst   = translation_table::translate(stealer_id, host_region);
                    vma_ptr_t src   = translation_table::translate(stealee_id, host_region);
                    size_t cpy_sz   = MEMREGION_SZ;
                    memtransfer_device::memcpy(dst, src, cpy_sz);
                }

                uma_proxy_lock::acquire_release(stealer_id, host_region);
            }

        public:

            static_assert(memult::is_pow2(MEMREGION_SZ));

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

                if (uma_proxy_lock::reference_try(device_id, host_ptr)){
                    return translation_table::translate(device_id, host_ptr);
                }

                dispatch_steal(device_id, host_ptr);
                return dg::ptr_limits<vma_ptr_t>::null_value();
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                while (true){
                    if (auto rs = map_try(device_id, host_ptr); rs != dg::ptr_limits<vma_ptr_t>::null_value()){
                        return rs;
                    }
                }
            }

            static void map_release(device_id_t device_id, uma_ptr_t host_ptr) noexcept{

                uma_proxy_lock::reference_release(host_ptr);
            }

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (memregion_slot(old_host_ptr) == memregion_slot(new_host_ptr)){
                    return dg::memult::advance(old_device_ptr, dg::memult::distance(old_host_ptr, new_host_ptr));
                }

                return dg::ptr_limits<vma_ptr_t>::null_value();
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (auto rs = remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr); rs != dg::ptr_limits<vma_ptr_t>::null_value()){
                    return rs;
                }

                map_release(device_id, old_host_ptr);
                return map_wait(device_id, new_host_ptr);
            }
    };
}

namespace dg::network_uma_tlb_impl1::direct{

    using namespace interface; 

    template <class ID, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize, class ProxyCount>
    class ProxyTLB{};

    template <class ID, class DeviceIDType, class UMAPtrType, class VMAPtrType, size_t MEMREGION_SZ, size_t PROXY_COUNT>
    class ProxyTLB<ID, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>: public ProxyTLBInterface<ProxyTLB<ID, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>>{

        public:

            using device_id_t       = DeviceIDType;
            using uma_ptr_t         = UMAPtrType;
            using vma_ptr_t         = VMAPtrType; 

        private:

            using self              = ProxyTLB;
            using translation_table = translation_table_impl::UMATranslationTable<self, uma_ptr_t, vma_ptr_t, device_id_t, MEMREGION_SZ, PROXY_COUNT>;

            static auto memregion_slot(uma_ptr_t ptr) noexcept -> size_t{
    
                using uptr_t = typename dg::ptr_info<uma_ptr_t>::max_unsigned_t; 
                return pointer_cast<uptr_t>(ptr) / MEMREGION_SZ;
            }

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

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (memregion_slot(old_host_ptr) == memregion_slot(new_host_ptr)){
                    return dg::memult::advance(old_device_ptr, dg::memult::distance(old_host_ptr, new_host_ptr));
                }

                return dg::ptr_limits<vma_ptr_t>::null_value();
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (auto rs = remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr); rs != dg::ptr_limits<vma_ptr_t>::null_value()){
                    return rs;
                }

                map_release(device_id, old_host_ptr);
                return map_wait(device_id, new_host_ptr);
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

            static auto memregion_slot(uma_ptr_t ptr) noexcept -> size_t{

                using uptr_t = typename dg::ptr_info<uma_ptr_t>::max_unsigned_t; 
                return dg::pointer_cast<uptr_t>(ptr) / MEMREGION_SZ;
            }

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

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (memregion_slot(old_host_ptr) == memregion_slot(new_host_ptr)){
                    return memult::advance(old_device_ptr, memult::distance(old_host_ptr, new_host_ptr));
                }

                return dg::ptr_limits<vma_ptr_t>::null_value();
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (auto rs = remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr); rs != dg::ptr_limits<vma_ptr_t>::null_value()){
                    return rs;
                }

                map_release(device_id, old_host_ptr);
                return map_wait(device_id, new_host_ptr);
            }
    };
}

namespace dg::network_uma_tlb_impl1::biex{

    using namespace interface;
    
    template <class ID, class MemTransferDevice, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize, class ProxyCount>
    class ProxyTLB{}; 

    template <class ID, class T, class DeviceIDType, class UMAPtrType, class VMAPtrType, size_t MEMREGION_SZ, size_t PROXY_COUNT>
    class ProxyTLB<ID, MemoryTransferDeviceInterface<T>, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>: public ProxyTLBInterface<ProxyTLB<ID, MemoryTransferDeviceInterface<T>, DeviceIDType, UMAPtrType, VMAPtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>>{

        public:

            using device_id_t           = DeviceIDType; 
            using uma_ptr_t             = UMAPtrType;
            using vma_ptr_t             = VMAPtrType;

        private:

            using self                  = ProxyTLB;
            using bijective_tlb         = network_uma_tlb_impl1::bijective::ProxyTLB<self, device_id_t, uma_ptr_t, vma_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>>; 
            using exclusive_tlb         = network_uma_tlb_impl1::exclusive::ProxyTLB<self, MemoryTransferDeviceInterface<T>, device_id_t, uma_ptr_t, vma_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>;
            using segcheck_ins          = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline constexpr uint8_t DISPATCH_CODE_BIJECTIVE     = 0u;
            static inline constexpr uint8_t DISPATCH_CODE_EXCLUSIVE     = 1u;
            static inline std::unique_ptr<uint8_t[]> dispatch_table     = nullptr;
            static inline uma_ptr_t first_region                        = {}; 

            static auto memregion_slot(uma_ptr_t ptr) noexcept -> size_t{
 
                return dg::memult::distance(first_region, ptr) / MEMREGION_SZ;
            }

        public:

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

                if (dispatch_code == DISPATCH_CODE_BIJECTIVE){ //== 0 is heavily optimized
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

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{
                
                uint8_t old_dispatch_code   = dispatch_table[memregion_slot(segcheck_ins::access(old_host_ptr))];
                uint8_t new_dispatch_code   = dispatch_table[memregion_slot(segcheck_ins::access(new_host_ptr))];

                if (old_dispatch_code != new_dispatch_code){
                    return dg::ptr_limits<vma_ptr_t>::null_value(); 
                }
                
                if (old_dispatch_code == DISPATCH_CODE_BIJECTIVE){
                    return bijective_tlb::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
                }

                return exclusive_tlb::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{
                
                if (auto rs = remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr); rs != dg::ptr_limits<vma_ptr_t>::null_value()){
                    return rs;
                }

                map_release(device_id, old_host_ptr);
                return map_wait(device_id, new_host_ptr);
            }
    };
}

namespace dg::network_uma_tlb_impl1::generic{

    template <class ID, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize, class ProxyCount>
    class DirectTLB: public dg::network_uma_tlb::interface::DirectTLBInterface<DirectTLB<ID, DeviceIDType, UMAPtrType, VMAPtrType, MemregionSize, ProxyCount>>{

        private:

            using self = DirectTLB;
            using base = network_uma_tlb_impl1::direct::ProxyTLB<self, DeviceIDType, UMAPtrType, VMAPtrType, MemregionSize, ProxyCount>;

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

    template <class ID, class MemTransferDevice, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemregionSize, class ProxyCount>
    class BiexTLB: public dg::network_uma_tlb::interface::MutexTLBInterface<BiexTLB<ID, MemTransferDevice, DeviceIDType, UMAPtrType, VMAPtrType, MemregionSize, ProxyCount>>{

        private:

            using self  = BiexTLB;
            using base  = network_uma_tlb_impl1::biex::ProxyTLB<self, MemTransferDevice, DeviceIDType, UMAPtrType, VMAPtrType, MemregionSize, ProxyCount>;

        public:

            using device_id_t           = DeviceIDType;
            using uma_ptr_t             = UMAPtrType;
            using vma_ptr_t             = VMAPtrType;
            using map_resource_handle_t = std::tuple<device_id_t, uma_ptr_t, vma_ptr_t>;

            static void init(uma_ptr_t * uma_region_arr, vma_ptr_t * vma_region_arr, device_id_t * device_id_arr, bool * is_proxy_rep_arr, size_t n){

                base::init(uma_region_arr, vma_region_arr, device_id_arr, is_proxy_rep_arr, n);
            }

            static void deinit() noexcept{

                base::deinit();
            }

            static auto map_try(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<map_resource_handle_t>{

                vma_ptr_t rs = base::map_try(device_id, ptr);

                if (rs == dg::ptr_limits<vma_ptr_t>::null_value()){
                    return std::nullopt;
                }

                return std::make_tuple(device_id, ptr, rs);
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t ptr) noexcept -> map_resource_handle_t{

                return std::make_tuple(device_id, ptr, base::map_wait(device_id, ptr));
            }

            static void map_release(map_resource_handle_t map_resource) noexcept{

                base::map_release(std::get<0>(map_resource), std::get<1>(map_resource));
            }

            static auto remap_try(device_id_t device_id, uma_ptr_t ptr, map_resource_handle_t resource) noexcept -> std::optional<map_resource_handle_t>{
                
                if (std::get<0>(resource) != device_id){
                    return std::nullopt;
                }

                vma_ptr_t rs = base::remap_try(device_id, ptr, std::get<1>(resource), std::get<2>(resource));
                
                if (rs == dg::ptr_limits<vma_ptr_t>::null_value()){
                    return std::nullopt;
                }

                return std::make_tuple(device_id, ptr, rs);
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t ptr, map_resource_handle_t resource) noexcept -> map_resource_handle_t{

                if (std::get<0>(resource) != device_id){
                    map_release(resource);
                    return map_wait(device_id, ptr);
                }
                
                vma_ptr_t rs = base::remap_wait(device_id, ptr, std::get<1>(resource), std::get<2>(resource));
                return std::make_tuple(device_id, ptr, rs);
            }

            static auto get_vma_ptr(map_resource_handle_t resource) noexcept -> vma_ptr_t{

                return std::get<2>(resource);
            }
    };
}

#endif