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
        static auto map_try(typename T1::device_id_t device_id, typename T1::uma_ptr_t host_ptr) noexcept -> typename T1::vma_ptr_t{ //I never liked the idea of NULLPTR - its bad - buggy - optional<ptr_t> is the way yet I think there's performance constraints - nullptr is too heavily optimized such that 0 cmp is SO FAST than other type of cmps

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

    template <class T>
    struct MemregionQualiferGetterInterface{

        using interface_t           = MemregionQualiferGetterInterface<T>;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using uma_ptr_t             = typename T1::uma_ptr_t;
        
        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using memqualifier_kind_t   = typename T1::memqualifier_kind_t;  

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto get(typename T1::uma_ptr_t host_ptr) noexcept -> typename T1::memqualifier_kind_t{

            return T1::get(host_ptr);
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
                }

                src_ptr_t first_src_region  = *std::min_element(src_region_arr, src_region_arr + n);
                src_ptr_t last_src_region   = *std::max_element(src_region_arr, src_region_arr + n); 
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

    template <class ID, class MemTransferDevice, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemRegionSize, class ProxyCount>                                    
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

            static void init(uma_ptr_t * uma_region_arr, vma_ptr_t * vma_region_arr, device_id_t * device_id, bool * is_proxy_rep, size_t n){

                auto uma_rep_region     = std::make_unique<uma_ptr_t[]>(n);
                auto uma_rep_device_id  = std::make_unique<device_id_t[]>(n);
                size_t rep_sz           = 0u; 

                for (size_t i = 0; i < n; ++i){
                    if (is_proxy_rep[i]){
                        uma_rep_region[rep_sz]      = uma_region_arr[i];
                        uma_rep_device_id[rep_sz]   = device_id[i];
                        ++rep_sz;
                    }
                }
                
                translation_table::init(uma_region_arr, vma_region_arr, device_id, n); 
                uma_proxy_lock::init(uma_rep_region.get(), uma_rep_device_id.get(), rep_sz);
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

    template <class ID, class DeviceIDType, class UMAPtrType, class VMAPtrType, class MemRegionSize, class ProxyCount>
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

            static void init(uma_ptr_t * host_region, vma_ptr_t * device_region, device_id_t * device_id, size_t n){
                
                translation_table::init(host_region, device_region, device_id, n);
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

            static void init(uma_ptr_t * host_region, vma_ptr_t * device_region, size_t n){
                
                translation_table::init(host_region, device_region, n); 
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

namespace dg::network_uma_tlb_impl1::dbe{ //direct - bijective - exclusive

    using namespace interface;
    using memregion_qualifier_t = uint8_t;

    enum memregion_qualifier_option: memregion_qualifier_t{
        memqualifier_readonly   = 0u,
        memqualifier_bijective  = 1u,
        memqualifier_default    = 2u
    };
    
    template <class ID, class DirectTLB, class BijectiveTLB, class ExclusiveTLB, class QualifierTable>
    class ProxyTLB{}; 

    template <class ID, class T, class T1, class T2, class T3>
    class ProxyTLB<ID, ProxyTLBInterface<T>, ProxyTLBInterface<T1>, ProxyTLBInterface<T2>, MemregionQualiferGetterInterface<T3>>: public ProxyTLBInterface<ProxyTLB<ID, ProxyTLBInterface<T>, ProxyTLBInterface<T1>, ProxyTLBInterface<T2>, MemregionQualiferGetterInterface<T3>>>{

        private:

            using readonly_tlb          = ProxyTLBInterface<T>;
            using bijective_tlb         = ProxyTLBInterface<T1>;
            using default_tlb           = ProxyTLBInterface<T2>;
            using qualifier_table       = MemregionQualiferGetterInterface<T3>; 

        public:

            using device_id_t           = dg::network_type_traits_x::mono_reduction_type_t<typename readonly_tlb::device_id_t<>, typename bijective_tlb::device_id_t<>, typename default_tlb::device_id_t<>>; 
            using uma_ptr_t             = dg::network_type_traits_x::mono_reduction_type_t<typename readonly_tlb::uma_ptr_t<>, typename bijective_tlb::uma_ptr_t<>, typename default_tlb::uma_ptr_t<>, typename qualifier_table::uma_ptr_t<>>;
            using vma_ptr_t             = dg::network_type_traits_x::mono_reduction_type_t<typename readonly_tlb::vma_ptr_t<>, typename bijective_tlb::vma_ptr_t<>, typename default_tlb::vma_ptr_t<>>;

            static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                auto memqualifier = qualifier_table::get(host_ptr); 

                switch (memqualifier){
                    case memqualifier_readonly:
                        return readonly_tlb::map_try(device_id, host_ptr);
                    case memqualifier_bijective:
                        return bijective_tlb::map_try(device_id, host_ptr);
                    case memqualifier_default:
                        return default_tlb::map_try(device_id, host_ptr);
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            return {};
                        } else{
                            std::unreachable();
                            return {};
                        }
                }        
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                auto memqualifier = qualifier_table::get(host_ptr); 

                switch (memqualifier){
                    case memqualifier_readonly:
                        return readonly_tlb::map_wait(device_id, host_ptr);
                    case memqualifier_bijective:
                        return bijective_tlb::map_wait(device_id, host_ptr);
                    case memqualifier_default:
                        return default_tlb::map_wait(device_id, host_ptr);
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            return {};
                        } else{
                            std::unreachable();
                            return {};
                        }
                }           
            }

            static void map_release(device_id_t device_id, uma_ptr_t host_ptr) noexcept{

                auto memqualifier = qualifier_table::get(host_ptr); 

                switch (memqualifier){
                    case memqualifier_readonly:
                        readonly_tlb::map_release(device_id, host_ptr);
                        return;
                    case memqualifier_bijective:
                        bijective_tlb::map_release(device_id, host_ptr);
                        return;
                    case memqualifier_default:
                        default_tlb::map_release(device_id, host_ptr);
                        return;
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            return;
                        } else{
                            std::unreachable();
                            return;
                        }
                }           
            }

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{
                
                auto new_qualifier  = qualifier_table::get(new_host_ptr);
                auto old_qualifier  = qualifier_table::get(old_host_ptr); 

                if (old_qualifier != new_qualifier){
                    return dg::ptr_limits<vma_ptr_t>::null_value();
                }

                switch (old_qualifier){
                    case memqualifier_readonly:
                        return readonly_tlb::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
                    case memqualifier_bijective:
                        return bijective_tlb::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
                    case memqualifier_default:
                        return default_tlb::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            return {};
                        } else{
                            std::unreachable();
                            return {};
                        }
                }           
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

namespace dg::network_uma_tlb_impl1::wrapper{

    // template <class T>
    // struct DirectWrappedProxyTLB{};

    // template <class T>
    // struct DirectWrappedProxyTLB<dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>>: DirectTLBInterface<DirectWrappedProxyTLB<dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>>>{

    //     //it's the factory responsibility to make sure that map_release is (void) function  

    //     using base          = dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>;
    //     using device_id_t   = typename base::device_id_t;
    //     using uma_ptr_t     = typename base::uma_ptr_t;
    //     using vma_ptr_t     = typename base::vma_ptr_t;

    //     static auto map(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

    //         return base::map_wait(device_id, host_ptr); 
    //     }
    // };

    // template <class T>
    // struct ResourceWrappedProxyTLB{};  

    // template <class T>
    // struct ResourceWrappedProxyTLB<dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>>: ProxyTLBInterface<ResourceWrappedProxyTLB<dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>>>{

    //     public:

    //         using base          = dg::network_uma_tlb_impl1::interface::ProxyTLBInterface<T>;
    //         using device_id_t   = typename base::device_id_t;
    //         using uma_ptr_t     = typename base::uma_ptr_t;
    //         using vma_ptr_t     = typename base::vma_ptr_t;

    //     private:

    //         struct MapResource{
    //             device_id_t arg_id;
    //             uma_ptr_t arg_ptr;
    //             vma_ptr_t map_ptr;
    //         };
        
    //     public:

    //         using map_resource_handle_t = MapResource;

    //         static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> std::optional<MapResource>{

    //             vma_ptr_t rs = base::map_try(device_id, host_ptr);

    //             if (dg::memult::is_nullptr(rs)){
    //                 return std::nullopt;
    //             }

    //             return MapResource{device_id, host_ptr, rs};
    //         }

    //         static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> MapResource{

    //             vma_ptr_t rs = base::map_wait(device_id, host_ptr);
    //             return MapResource{device_id, host_ptr, rs};
    //         }

    //         static void map_release(MapResource map_resource) noexcept{

    //             base::map_release(map_resource.arg_id, map_resource.arg_ptr);
    //         }

    //         static auto get_vma_ptr(MapResource map_resource) noexcept -> vma_ptr_t{

    //             return map_resource.map_ptr;
    //         }
    // };
}

namespace dg::network_uma_tlb_impl1{

    // using namespace dg::network_uma_tlb::memqualifier_taxonomy; 
    // using internal_memqualifier_t = uint8_t; 

    // enum internal_memqualifier_option: internal_memqualifier_t{
    //     memqualifier_readonly   = 0b0001,
    //     memqualifier_direct     = 0b0010,
    //     memqualifier_exclusive  = 0b0100,
    //     memqualifier_bijective  = 0b1000
    // };

    // template <class ID, class DeviceIDType, class VMAPtrType, class DevicePtrType, class MemRegionSize, class ProxyCount>
    // struct Factory{}; 

    // template <class ID, class DeviceIDType, class VMAPtrType, class DevicePtrType, size_t MEMREGION_SZ, size_t PROXY_COUNT>
    // struct Factory<ID, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>{

    //     private:

    //         using self                          = Factory;
    //         using internal_memtransfer_device   = MemTransferDevice<DevicePtrType, DeviceIDType>;
    //         using internal_memqualifier_table   = void *;

    //         using internal_tlb_readonly         = direct::ProxyTLB<tags<self, std::integral_constant<size_t, 0>>, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>;
    //         using internal_tlb_direct           = direct::ProxyTLB<self, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>;
    //         using internal_tlb_exclusive        = exclusive::ProxyTLB<self, typename internal_memtransfer_device::interface_t, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>;
    //         using internal_tlb_bijective        = bijective::ProxyTLB<self, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>>;
    //         using internal_tlb                  = dbe::ProxyTLB<self, typename internal_tlb_readonly::interface_t, typename internal_tlb_bijective::interface_t, typename internal_tlb_exclusive::interface_t, typename internal_memqualifier_table::interface_t>;
    //         using internal_tlbx                 = dbe::ProxyTLBX<self, typename internal_tlb::interface_t>; 

    //     public:

    //         using tlb                           = typename internal_tlb::interface_t;
    //         using tlbx                          = typename internal_tlbx::interface_t;
    //         using tlb_direct                    = typename internal_tlb_direct::interface_t;

    //         static void init(uma_ptr_t * host_ptr, vma_ptr_t * device_ptr, device_id_t * device_id, memqualifier_t * mem_qualifier, size_t n){
                
    //             auto logger                 = dg::network_log_scope::critical_error_terminate();
    //             auto memqualifier_table     = to_memqualifier_table(host_ptr, device_ptr, device_id, mem_qualifier, n); 

    //             auto readonly_host_ptr      = std::make_unique<uma_ptr_t[]>(n);
    //             auto readonly_device_ptr    = std::make_unique<vma_ptr_t[]>(n);
    //             auto readonly_device_id     = std::make_unique<device_id_t[]>(n); 
    //             size_t readonly_sz          = partition_readonly_qualifier_to(readonly_host_ptr.get(), readonly_device_ptr.get(), readonly_device_id.get(), host_ptr, device_ptr, device_id, n, memqualifier_table);
    //             internal_tlb_readonly::init(readonly_host_ptr.get(), readonly_device_ptr.get(), readonly_device_id.get(), readonly_sz);

    //             auto direct_host_ptr        = std::make_unique<uma_ptr_t[]>(n);
    //             auto direct_device_ptr      = std::make_unique<vma_ptr_t[]>(n);
    //             auto direct_device_id       = std::make_unique<device_id_t[]>(n);
    //             size_t direct_sz            = partition_direct_qualifier_to(direct_host_ptr.get(), direct_device_ptr.get(), direct_device_id.get(), host_ptr, device_ptr, device_id, n, memqualifier_table); 
    //             internal_tlb_direct::init(direct_host_ptr.get(), direct_device_ptr.get(), direct_device_id.get(), direct_sz);

    //             auto exclusive_host_ptr     = std::make_unique<uma_ptr_t[]>(n);
    //             auto exclusive_device_ptr   = std::make_unique<vma_ptr_t[]>(n);
    //             auto exclusive_device_id    = std::make_unique<device_id_t[]>(n);
    //             auto exclusive_proxy_flag   = std::make_unique<bool[]>(n); 
    //             size_t exclusive_sz         = partition_exclusive_qualifier_to(exclusive_host_ptr.get(), exclusive_device_ptr.get(), exclusive_device_id.get(), exclusive_proxy_flag.get(), host_ptr, device_ptr, device_id, n, memqualifier_table); 
    //             internal_tlb_exclusive::init(exclusive_host_ptr.get(), exclusive_device_ptr.get(), exclusive_device_id.get(), exclusive_proxy_flag.get(), exclusive_sz);

    //             auto bijective_host_ptr     = std::make_unique<uma_ptr_t[]>(n);
    //             auto bijective_device_ptr   = std::make_unique<vma_ptr_t[]>(n);
    //             size_t bijective_sz         = partition_bijective_qualifier_to(bijective_host_ptr.get(), bijective_device_ptr.get(0, host_ptr, device_ptr, device_id, n, memqualifier_table)); 
    //             internal_tlb_bijective::init(bijective_host_ptr.get(), bijective_device_ptr.get(), bijective_sz); 

    //             logger.release();
    //         }
    // };
}

#endif