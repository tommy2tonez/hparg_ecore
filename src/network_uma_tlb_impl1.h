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
#include "network_lang_x.h"
#include "network_log.h"

#include "network_segcheck_bound.h"
#include "network_exception.h"

namespace dg::network_uma_tlb_impl1::interface{

    template <class T>
    struct ProxyTLBInterface{

        using interface_t   = ProxyTLBInterface<T>;
        using device_id_t   = typename T::device_id_t;
        using uma_ptr_t     = typename T::uma_ptr_t;
        using vma_ptr_t     = typename T::vma_ptr_t;

        static_assert(std::is_unsigned_v<device_id_t>);
        static_assert(dg::is_ptr_v<uma_ptr_t>);
        static_assert(dg::is_ptr_v<vma_ptr_t>);

        static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

            return T::map_try(device_id, host_ptr);
        }

        static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

            return T::map_wait(device_id, host_ptr);
        }

        static void map_release(device_id_t device_id, uma_ptr_t host_ptr) noexcept{

            T::map_release(device_id, host_ptr);
        }

        static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

            return T::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
        }

        static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

            return T::remap_wait(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
        }
    };

    template <class T>
    struct MemoryTransferDeviceInterface{

        using vma_ptr_t = typename T::vma_ptr_t;
        static_assert(dg::is_ptr_v<vma_ptr_t>);

        static void memcpy(vma_ptr_t dst, vma_ptr_t src, size_t n) noexcept{
            
            T::memcpy(dst, src, n);
        }
    };

    template <class T>
    struct MemregionQualiferGetterInterface{

        using uma_ptr_t = typename T::uma_ptr_t; 
        static_assert(dg::is_ptr_v<uma_ptr_t>);

        static auto get(uma_ptr_t host_ptr) noexcept -> memregion_qualifier_t[

            return T::get(host_ptr);
        ]
    };
}

namespace dg::network_uma_tlb_impl1::memqualifier_taxonomy{

    using mem_qualifier_t = uint8_t;

    enum memqualifier_option: mem_qualifier_t{
        read    = 0u,
        write   = 1u,
        sync    = 2u,
        nosync  = 3u
    };
} 

namespace dg::network_uma_tlb_impl1::translation_table{

    using namespace interface;

    static inline constexpr bool IS_SAFE_ACCESS_ENABLED = true;

    template <class ID, class SrcPtrType, class DstPtrType, size_t MEMREGION_SZ>
    class InjectiveTranslationTable{

        public:

            static_assert(MEMREGION_SZ != 0u);

            using src_ptr_t = SrcPtrType;
            using dst_ptr_t = DstPtrType;

        private:

            static inline dst_ptr_t * translation_table{};
            
            using self              = InjectiveTranslationTable;
            using segcheck_ins      = dg::network_segcheck_bound::StdAccess<self, src_ptr_t>;
            using ptr_arithmetic_t  = typename dg::ptr_info<src_ptr_t>::max_unsigned_t; 

            static inline auto memregion_slot(src_ptr_t ptr) noexcept -> size_t{

                return pointer_cast<ptr_arithmetic_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto memregion_offset(src_ptr_t ptr) noexcept -> size_t{
                
                return pointer_cast<ptr_arithmetic_t>(ptr) % MEMREGION_SZ;
            }

        public:

            static void init(src_ptr_t * host_region, dst_ptr_t * device_region, size_t n){

                auto logger = dg::network_log_scope::critical_terminate();

                if (n == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_INIT_ARG);
                }

                src_ptr_t max_ptr   = *std::max_element(host_region, host_region + n);
                src_ptr_t last_ptr  = memult::advance(max_ptr, MEMREGION_SZ);
                size_t table_sz     = pointer_cast<ptr_arithmetic_t>(last_ptr) / MEMREGION_SZ;
                translation_table   = new dst_ptr_t[table_sz];

                std::fill(translation_table, translation_table + table_sz, dg::pointer_limits<src_ptr_t>::null_value());
                segcheck_ins::init(dg::pointer_limits<src_ptr_t>::min(), last_ptr);

                for (size_t i = 0u; i < n; ++i){
                    if (memregion_offset(host_region[i]) != 0u || memregion_offset(device_region[i]) != 0u || memregion_slot(host_region[i]) == 0u || memregion_slot(device_region[i]) == 0u){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_INIT_ARG);
                    }
                    translation_table[memregion_slot(host_region[i])] = device_region[i];
                }

                logger.release();
            }

            static auto translate(src_ptr_t ptr) noexcept -> dst_ptr_t{
                
                ptr         = segcheck_ins::access(ptr);
                size_t idx  = memregion_slot(ptr);
                size_t off  = memregion_offset(ptr);

                if constexpr(IS_SAFE_ACCESS_ENABLED){
                    if (translation_table[idx] == dg::pointer_limits<dst_ptr_t>::null_value()){
                        dg::network_log_stackdump::critical_error(dg::network_exception::verbose(dg::network_exception::SEGFAULT));
                        std::abort();
                    }
                }

                return memult::advance(translation_table[idx], off);
            }
    };

    template <class ID, class SrcPtrType, class DstPtrType, class DeviceIDType, size_t MEMREGION_SZ, size_t DEVICE_COUNT>
    class UMATranslationTable{

        public:

            using src_ptr_t         = SrcPtrType;
            using dst_ptr_t         = DstPtrType;
            using device_id_t       = DeviceIDType;

        private:

            using self              = UMATranslationTable;
            using virtual_vma_ptr_t = typename dg::ptr_info<>::max_unsigned_t;
            using base              = InjectiveTranslationTable<self, virtual_vma_ptr_t, dst_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>>;

            static inline auto memregion_slot(src_ptr_t ptr) noexcept -> size_t{

                using ptr_arithmetic_t = typename dg::ptr_info<src_ptr_t>::max_unsigned_t;
                return pointer_cast<ptr_arithmetic_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto memregion_offset(src_ptr_t ptr) noexcept -> size_t{

                using ptr_arithmetic_t = typename dg::ptr_info<src_ptr_t>::max_unsigned_t;
                return pointer_cast<ptr_arithmetic_t>(ptr) % MEMREGION_SZ;
            }
            
            static inline auto to_virtual_ptr(src_ptr_t ptr, device_id_t device_id) noexcept -> virtual_vma_ptr_t{

                size_t idx                      = memregion_slot(ptr);
                size_t off                      = memregion_offset(ptr);
                size_t virtual_idx              = idx * DEVICE_COUNT + device_id;
                virtual_vma_ptr_t virtual_ptr   = static_cast<virtual_vma_ptr_t>(virtual_idx * MEMREGION_SZ + off);

                return virtual_ptr;
            } 

        public:

            static void init(src_ptr_t * host_region, dst_ptr_t * device_region, device_id_t * device_id, size_t n){

                auto logger                 = dg::network_log_scope::critical_terminate();
                auto virtual_host_region    = std::make_unique<virtual_vma_ptr_t[]>(n);
                
                for (size_t i = 0u; i < n; ++i){
                    virtual_host_region[i] = to_virtual_ptr(host_region[i], device_id[i]);
                }

                base::init(virtual_host_region.get(), device_region, n);
                logger.release();
            }

            static auto translate(device_id_t device_id, vma_ptr_t ptr) noexcept -> device_ptr_t{

                return base::translate(device_id, to_virtual_ptr(ptr, device_id));
            }
    };
} 

namespace dg::network_uma_tlb_impl1::exclusive{

    using namespace interface;

    template <class ID, class T, class DeviceIDType, class VMAPtrType, class DevicePtrType, class MemRegionSize, class ProxyCount>                                    
    struct ProxyTLB{};

    template <class ID, class T, class DeviceIDType, class VMAPtrType, class DevicePtrType, size_t MEMREGION_SZ, size_t PROXY_COUNT>
    struct ProxyTLB<ID, MemoryTransferDeviceInterface<T>, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>: ProxyTLBInterface<ProxyTLB<ID, MemoryTransferDeviceInterface<T>, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>>{                                                                                                                                               

        public:


            using device_id_t           = DeviceIDType;
            using uma_ptr_t             = VMAPtrType;
            using vma_ptr_t             = dg::mono_type_reduction_t<DevicePtrType, typename memtransfer_device::vma_ptr_t>;
            
        private:

            using self                  = ProxyTLB;
            using memtransfer_device    = MemoryTransferDeviceInterface<T>; 
            using vna_lock              = dg::network_memlock_proxyspin::ReferenceLock<self, std::integral_constant<size_t, MEMREGION_SZ>, uma_ptr_t>;
            using translation_table     = translation_table::UMATranslationTable<self, uma_ptr_t, vma_ptr_t, device_id_t, MEMREGION_SZ, PROXY_COUNT>;

            static auto memregion_slot(uma_ptr_t ptr) noexcept -> size_t{

                using ptr_arithmetic_t = typename dg::ptr_info<uma_ptr_t>::max_unsigned_t; 
                return pointer_cast<ptr_arithmetic_t>(ptr) / MEMREGION_SZ;
            }

            static auto memregion(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                static_assert(memult::is_pow2(MEMREGION_SZ));
                using ptr_arithmetic_t = typename dg::ptr_info<uma_ptr_t>::max_unsigned_t; 
                constexpr ptr_arithmetic_t BITMASK = ~(static_cast<ptr_arithmetic_t>(MEMREGION_SZ) - 1);
                
                return pointer_cast<uma_ptr_t>(pointer_cast<ptr_arithmetic_t>(ptr) & BITMASK);
            }

            static void steal_try(device_id_t stealer_id, uma_ptr_t host_ptr) noexcept{
                
                uma_ptr_t host_region = memregion(host_ptr);
                std::optional<device_id_t> potential_stealee_id = vna_lock::acquire_try(host_region);  

                if (!static_cast<bool>(potential_stealee_id)){
                    return;
                }

                device_id_t stealee_id = potential_stealee_id.value();

                if (stealee_id != stealer_id){
                    memtransfer_device::memcpy(translation_table::translate(stealer_id, host_region), translation_table::translate(stealee_id, host_region), MEMREGION_SZ);
                }

                vna_lock::acquire_release(stealer_id, host_ptr);
            }

        public:

            static void init(uma_ptr_t * host_region, vma_ptr_t * device_region, device_id_t * device_id, bool * is_proxy_rep, size_t n){ //this does look bad - combining factory + component - violate single responsibility - yet its fine for the purpose

                auto logger                     = dg::network_log_scope::critical_terminate();
                auto initial_rep_host_region    = std::make_unique<uma_ptr_t[]>(n);
                auto initial_rep_device_id      = std::make_unique<device_id_t[]>(n);
                size_t initial_rep_sz           = 0u; 

                for (size_t i = 0; i < n; ++i){
                    if (is_proxy_rep[i]){
                        initial_rep_host_region[initial_rep_sz] = host_region[i];
                        initial_rep_device_id[initial_rep_sz]   = device_id[i];
                        ++initial_rep_sz;
                    }
                }
                
                translation_table::init(host_region, device_region, device_id, n); 
                vna_lock::init(initial_rep_host_region.get(), initial_rep_device_id.get(), initial_rep_sz);
                logger.release();
            }

            static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                if (vna_lock::reference_try(device_id, host_ptr)){
                    return translation_table::translate(device_id, host_ptr);
                }

                steal_try(device_id, host_ptr);
                return dg::pointer_limits<vma_ptr_t>::null_value();
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                while (true){
                    if (auto rs = map_try(device_id, host_ptr); memult::is_validptr(rs)){
                        return rs;
                    }
                }
            }

            static void map_release(device_id_t device_id, uma_ptr_t host_ptr) noexcept{

                vna_lock::reference_release(host_ptr);
            }

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (memregion_slot(old_host_ptr) == memregion_slot(new_host_ptr)){
                    return memult::forward(old_device_ptr, memult::distance(old_host_ptr, new_host_ptr));
                }

                return dg::pointer_limits<vma_ptr_t>::null_value();
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (auto rs = remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr); memult::is_validptr(rs)){
                    return rs;
                }

                map_release(device_id, old_host_ptr);
                return map_wait(device_id, new_host_ptr);
            }
    };
}

namespace dg::network_uma_tlb_impl1::direct{

    using namespace interface; 

    template <class ID, class DeviceIDType, class VMAPtrType, class DevicePtrType, class MemRegionSize, class ProxyCount>
    class ProxyTLB{};

    template <class ID, class DeviceIDType, class VMAPtrType, class DevicePtrType, size_t MEMREGION_SZ, size_t PROXY_COUNT>
    class ProxyTLB<ID, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>: public ProxyTLBInterface<ProxyTLB<ID, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>>{

        public:

            using device_id_t   = DeviceIDType;
            using uma_ptr_t     = VMAPtrType;
            using vma_ptr_t     = DevicePtrType; 

        private:

            using self              = ProxyTLB;
            using translation_table = translation_table::UMATranslationTable<self, uma_ptr_t, vma_ptr_t, MEMREGION_SZ, PROXY_COUNT>;

            static auto memregion_slot(uma_ptr_t ptr) noexcept -> size_t{
    
                using ptr_arithmetic_t = typename dg::ptr_info<uma_ptr_t>::max_unsigned_t; 
                return pointer_cast<ptr_arithmetic_t>(ptr) / MEMREGION_SZ;
            }

        public:

            static void init(uma_ptr_t * host_region, vma_ptr_t * device_region, device_id_t * device_id, size_t n){
                
                translation_table::init(host_region, device_region, device_id, n);
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                translation_table::translate(device_id, host_ptr);
            }

            static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                return map_wait(device_id, host_ptr);
            }

            static void map_release(device_id_t arg, uma_ptr_t) noexcept{

                (void) arg;
            } 

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (memregion_slot(old_host_ptr) == memregion_slot(new_host_ptr)){
                    return memult::advance(old_device_ptr, memult::distance(old_host_ptr, new_host_ptr));
                }

                return dg::pointer_limits<vma_ptr_t>::null_value();
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (auto rs = remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr); memult::is_validptr(rs)){
                    return rs;
                }

                map_release(device_id, old_host_ptr);
                return map_wait(device_id, new_host_ptr);
            }
    };
} 

namespace dg::network_uma_tlb_impl1::bijective{
    
    using namespace interface;

    template <class ID, class DeviceIDType, class VMAPtrType, class DevicePtrType, size_t MemRegionSize>
    class ProxyTLB{};

    template <class ID, class DeviceIDType, class VMAPtrType, class DevicePtrType, size_t MEMREGION_SZ>
    class ProxyTLB<ID, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>>: public ProxyTLBInterface<ProxyTLB<ID, DeviceIDType, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>>>{

        public:

            using device_id_t   = DeviceIDType;
            using uma_ptr_t     = VMAPtrType;
            using vma_ptr_t     = DevicePtrType;

        private:

            using self              = ProxyTLB;
            using translation_table = translation_table::InjectiveTranslationTable<self, uma_ptr_t, vma_ptr_t, MEMREGION_SZ>; 

            static auto memregion_slot(uma_ptr_t ptr) noexcept -> size_t{
    
                using ptr_arithmetic_t = typename dg::ptr_info<uma_ptr_t>::max_unsigned_t; 
                return pointer_cast<ptr_arithmetic_t>(ptr) / MEMREGION_SZ;
            }

        public:

            static void init(uma_ptr_t * host_region, vma_ptr_t * device_region, size_t n){ //this is not significantly faster yet its good to have this here for future extension - 
                
                translation_table::init(host_region, device_region, n); 
            }

            static auto map_wait(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                return translation_table::translate(host_ptr);
            }

            static auto map_try(device_id_t device_id, uma_ptr_t host_ptr) noexcept -> vma_ptr_t{

                return map_wait(device_id, host_ptr);                
            }

            static void map_release(device_id_t arg, uma_ptr_t) noexcept{

                (void) arg;
            }

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (memregion_slot(old_host_ptr) == memregion_slot(new_host_ptr)){
                    return memult::advance(old_device_ptr, memult::distance(old_host_ptr, new_host_ptr));
                }

                return dg::pointer_limits<vma_ptr_t>::null_value();
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (auto rs = remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr); memult::is_validptr(rs)){
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

            using device_id_t           = dg::mono_type_reduction_t<typename readonly_tlb::device_id_t, typename bijective_tlb::device_id_t, typename default_tlb::device_id_t>; 
            using uma_ptr_t             = dg::mono_type_reduction_t<typename readonly_tlb::uma_ptr_t, typename bijective_tlb::uma_ptr_t, typename default_tlb::uma_ptr_t, typename qualifier_table::uma_ptr_t>;
            using vma_ptr_t             = dg::mono_type_reduction_t<typename readonly_tlb::vma_ptr_t, typename bijective_tlb::vma_ptr_t, typename default_tlb::vma_ptr_t>;

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
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        return {};
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
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        return {};
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
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        return;
                }           
            }

            static auto remap_try(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{
                
                auto new_qualifier  = qualifier_table::get(new_host_ptr);
                auto old_qualifier  = qualifier_table::get(old_host_ptr); 

                if (old_qualifier != new_qualifier){
                    return dg::pointer_limits<vma_ptr_t>::null_value();
                }

                switch (old_qualifier){
                    case memqualifier_readonly:
                        return readonly_tlb::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
                    case memqualifier_bijective:
                        return bijective_tlb::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
                    case memqualifier_default:
                        return default_tlb::remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr);
                    default:
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                        return {};
                }           
            }

            static auto remap_wait(device_id_t device_id, uma_ptr_t new_host_ptr, uma_ptr_t old_host_ptr, vma_ptr_t old_device_ptr) noexcept -> vma_ptr_t{

                if (auto rs = remap_try(device_id, new_host_ptr, old_host_ptr, old_device_ptr); memult::is_validptr(rs)){
                    return rs;
                }

                map_release(device_id, old_host_ptr);
                return map_wait(device_id, new_host_ptr);
            }
    };
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