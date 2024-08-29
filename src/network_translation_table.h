#ifndef __DG_NETWORK_TRANSLATION_TABLE_H__
#define __DG_NETWORK_TRANSLATION_TABLE_H__

#include "network_segcheck_bound.h"
#include "network_exception.h"

namespace dg::network_translation_table_bijective{

    template <class T>
    struct TranslationTableInterface{

        using vma_ptr_t     = typename T::vma_ptr_t;
        using device_ptr_t  = typename T::device_ptr_t;
        
        static_assert(dg::is_ptr_v<vma_ptr_t>);
        static_assert(dg::is_ptr_v<device_ptr_t>);

        static inline auto translate(vma_ptr_t ptr) noexcept -> device_ptr_t{

            return T::translate(ptr);
        }
    };

    template <class ID, class VMAPtrType, class DevicePtrType, class MemRegionSize, class IsSafeTranslation>
    class TranslationTable{}; 

    template <class ID, class VMAPtrType, class DevicePtrType, size_t MEMREGION_SZ>
    class TranslationTable<ID, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<bool, true>>: public TranslationTableInterface<TranslationTable<ID, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<bool, true>>>{

        public:

            static_assert(MEMREGION_SZ != 0u);

            using vma_ptr_t         = VMAPtrType;
            using device_ptr_t      = DevicePtrType;

        private:

            static inline device_ptr_t * translation_table{};
            
            using self              = TranslationTable;
            using segcheck_ins      = dg::network_segcheck_bound::SafeAlignedAccess<self, vma_ptr_t>;
            using ptr_arithmetic_t  = typename dg::ptr_info<vma_ptr_t>::max_unsigned_t; 

            static inline auto memregion_slot(vma_ptr_t ptr) noexcept -> size_t{

                return pointer_cast<ptr_arithmetic_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto memregion_offset(vma_ptr_t ptr) noexcept -> size_t{
                
                return pointer_cast<ptr_arithmetic_t>(ptr) % MEMREGION_SZ;
            }

        public:

            static void init(vma_ptr_t * host_region, device_ptr_t * device_region, size_t n){

                auto logger = dg::network_log_scope::critical_error_terminate();

                if (n == 0u){
                    throw dg::network_exception::invalid_arg();
                }

                vma_ptr_t max_ptr   = *std::max_element(host_region, host_region + n);
                vma_ptr_t last_ptr  = memult::advance(max_ptr, MEMREGION_SZ);
                size_t table_sz     = pointer_cast<ptr_arithmetic_t>(last_ptr) / MEMREGION_SZ;
                translation_table   = new device_ptr_t[table_sz];

                std::fill(translation_table, translation_table + table_sz, dg::pointer_limits<vma_ptr_t>::null_value());
                segcheck_ins::init(dg::pointer_limits<vma_ptr_t>::min(), last_ptr);

                for (size_t i = 0u; i < n; ++i){
                    if (memregion_offset(host_region[i]) != 0u || memregion_offset(device_region[i]) != 0u || memregion_slot(host_region[i]) == 0u || memregion_slot(device_region[i]) == 0u){
                        throw dg::network_exception::invalid_arg();
                    }
                    translation_table[memregion_slot(host_region[i])] = device_region[i];
                }

                logger.release();
            }

            static inline auto translate(vma_ptr_t ptr) noexcept -> device_ptr_t{
                
                ptr         = segcheck_ins::access(ptr);
                size_t idx  = memregion_slot(ptr);
                size_t off  = memregion_offset(ptr);

                if (translation_table[idx] == dg::pointer_limits<device_ptr_t>::null_value()){
                    dg::network_log_stackdump::critical_error(dg::network_exception::SEGFAULT_CSTR);
                    std::abort();
                }

                return memult::advance(translation_table[idx], off);
            }
    };

    template <class ID, class VMAPtrType, class DevicePtrType, size_t MEMREGION_SZ>
    class TranslationTable<ID, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<bool, false>>: public TranslationTableInterface<TranslationTable<ID, VMAPtrType, DevicePtrType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<bool, false>>>{

        public:

            static_assert(MEMREGION_SZ != 0u);
            
            using vma_ptr_t     = VMAPtrType;
            using device_ptr_t  = DevicePtrType;
        
        private:

            static inline device_ptr_t * translation_table{};
            using ptr_arithmetic_t = typename dg::ptr_info<vma_ptr_t>::max_unsigned_t; 

            static inline auto memregion_slot(vma_ptr_t ptr) noexcept -> size_t{

                return pointer_cast<ptr_arithmetic_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto memregion_offset(vma_ptr_t ptr) noexcept -> size_t{
                
                return pointer_cast<ptr_arithmetic_t>(ptr) % MEMREGION_SZ;
            }

        public:

            static void init(vma_ptr_t * host_region, device_ptr_t * device_region, size_t n){

                auto logger = dg::network_log_scope::critical_error_terminate();

                if (n == 0u){
                    throw dg::network_exception::invalid_arg();
                }

                vma_ptr_t max_ptr   = *std::max_element(host_region, host_region + n);
                vma_ptr_t last_ptr  = memult::advance(max_ptr, MEMREGION_SZ);
                size_t table_sz     = pointer_cast<ptr_arithmetic_t>(last_ptr) / MEMREGION_SZ;
                translation_table   = new device_ptr_t[table_sz]; 

                for (size_t i = 0u; i < n; ++i){
                    if (memregion_offset(host_region[i]) != 0u || memregion_offset(device_region[i]) != 0u || memregion_slot(host_region[i]) == 0u || memregion_slot(device_region[i]) == 0u){
                        throw dg::network_exception::invalid_arg();
                    }
                    translation_table[memregion_slot(host_region[i])] = device_region[i];
                }

                logger.release();
            }

            static inline auto translate(vma_ptr_t ptr) noexcept -> device_ptr_t{

                size_t idx  = memregion_slot(ptr);
                size_t off  = memregion_offset(ptr);

                return memult::advance(translation_table[idx], off);
            }
    };

    static inline constexpr bool IS_SAFE_ACCESS_ENABLED = true;

    template <class ID, class VMAPtrType, class DevicePtrType, class MemRegionSize>
    using StdTranslationTable = RegionInjectiveTranslationTable<ID, VMAPtrType, DevicePtrType, MemRegionSize, std::integral_constant<bool, IS_SAFE_ACCESS_ENABLED>>;

}

namespace dg::network_translation_table_proxy{

    template <class T>
    struct TranslationTableInterface{

        using device_id_t   = typename T::device_id_t;
        using vma_ptr_t     = typename T::vma_ptr_t;
        using device_ptr_t  = typename T::device_ptr_t;

        static_assert(std::is_unsigned_v<device_id_t>);
        static_assert(dg::is_ptr_v<vma_ptr_t>);
        static_assert(dg::is_ptr_v<device_ptr_t>);

        static inline auto translate(device_id_t device_id, vma_ptr_t ptr) noexcept -> device_ptr_t{

            return T::translate(device_id, ptr);
        }
    };

    template <class ID, class VMAPtrType, class DevicePtrType, class DeviceIDType, class MemRegionSize, class DeviceCount>
    class TranslationTable{};

    template <class ID, class VMAPtrType, class DevicePtrType, class DeviceIDType, size_t MEMREGION_SZ, size_t DEVICE_COUNT>
    class TranslationTable<ID, VMAPtrType, DevicePtrType, DeviceIDType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, DEVICE_COUNT>>: TranslationTableInterface<TranslationTable<ID, VMAPtrType, DevicePtrType, DeviceIDType, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, DEVICE_COUNT>>>{

        public:

            using vma_ptr_t         = VMAPtrType;
            using device_ptr_t      = DevicePtrType;
            using device_id_t       = DeviceIDType;

        private:

            using self              = TranslationTable;
            using base              = dg::network_translation_table_bijective::StdTranslationTable<self, virtual_vma_ptr_t, device_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>>;
            using virtual_vma_ptr_t = typename dg::ptr_info<>::max_unsigned_t;

            static inline auto memregion_slot(vma_ptr_t ptr) noexcept -> size_t{

                using ptr_arithmetic_t = typename dg::ptr_info<vma_ptr_t>::max_unsigned_t;
                return pointer_cast<ptr_arithmetic_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto memregion_offset(vma_ptr_t ptr) noexcept -> size_t{

                using ptr_arithmetic_t = typename dg::ptr_info<vma_ptr_t>::max_unsigned_t;
                return pointer_cast<ptr_arithmetic_t>(ptr) % MEMREGION_SZ;
            }
            
            static inline auto to_virtual_ptr(vma_ptr_t ptr, device_id_t device_id) noexcept -> virtual_vma_ptr_t{

                size_t idx                      = memregion_slot(ptr);
                size_t off                      = memregion_offset(ptr);
                size_t virtual_idx              = idx * DEVICE_COUNT + device_id;
                virtual_vma_ptr_t virtual_ptr   = static_cast<virtual_vma_ptr_t>(virtual_idx * MEMREGION_SZ + off);

                return virtual_ptr;
            } 

        public:

            static void init(vma_ptr_t * host_region, device_ptr_t * device_region, device_id_t * device_id, size_t n){

                auto logger                 = dg::network_log_scope::critical_error_terminate();
                auto virtual_host_region    = std::make_unique<virtual_vma_ptr_t[]>(n);
                
                for (size_t i = 0u; i < n; ++i){
                    virtual_host_region[i] = to_virtual_ptr(host_region[i], device_id[i]);
                }

                base::init(virtual_host_region.get(), device_region, n);
                logger.release();
            }

            static inline auto translate(device_id_t device_id, vma_ptr_t ptr) noexcept -> device_ptr_t{

                return base::translate(device_id, to_virtual_ptr(ptr, device_id));
            }
    };
} 

#endif