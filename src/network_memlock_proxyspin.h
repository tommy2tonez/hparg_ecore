#ifndef __NETWORK_MEMLOCKPX_H__
#define __NETWORK_MEMLOCKPX_H__

#include <stdlib.h>
#include <stdint.h>
#include <type_traits>
#include <atomic>
#include <limits.h>
#include "network_log.h" 
#include "network_segcheck_bound.h"
#include "network_lang_x.h"
#include "network_exception.h"
#include <mutex>
#include <optional>
#include "stdx.h"

namespace dg::network_memlock_proxyspin{

    using namespace dg::network_lang_x;

    //maybe I should consider removing static - thats a decision that will be made in the future - when I do optimization and profiling 

    using Lock = std::conditional_t<IS_ATOMIC_OPERATION_PREFERRED,
                                    std::atomic_flag,
                                    std::mutex>; 

    template <class T>
    struct ReferenceLockInterface{

        using proxy_id_t    = typename T::proxy_id_t; 
        using ptr_t         = typename T::ptr_t;

        static_assert(std::is_unsigned_v<proxy_id_t>);
        static_assert(dg::is_pointer_v<ptr_t>);

        static inline auto acquire_try(ptr_t ptr) noexcept -> std::optional<proxy_id_t>{

            return T::acquire_try(ptr);
        }

        static inline auto acquire_wait(ptr_t ptr) noexcept -> proxy_id_t{

            return T::acquire_wait(ptr);
        }

        static inline void acquire_release(proxy_id_t new_proxy_id, ptr_t ptr) noexcept{

            T::acquire_release(new_proxy_id, ptr);
        } 

        static inline auto reference_try(proxy_id_t expected_proxy_id, ptr_t ptr) noexcept -> bool{

            return T::reference_try(expected_proxy_id, ptr);
        } 

        static inline void reference_release(ptr_t ptr) noexcept{

            T::reference_release(ptr);
        }
    };

    template <class ProxyIDType, class RefCountType>
    struct AtomicLockStateController{
        
        using proxy_id_t    = ProxyIDType;
        using refcount_t    = RefCountType;
        using lock_state_t  = uint64_t; 

        static_assert(sizeof(proxy_id_t) + sizeof(refcount_t) <= sizeof(lock_state_t));
        static_assert(std::is_unsigned_v<proxy_id_t>);
        static_assert(std::is_unsigned_v<refcount_t>);

        static inline constexpr refcount_t REFERENCE_EMPTY_VALUE    = refcount_t{0u};
        static inline constexpr refcount_t REFERENCE_ACQUIRED_VALUE = ~refcount_t{0u};

        static constexpr auto make(proxy_id_t proxy_id, refcount_t refcount) noexcept -> lock_state_t{
            
            return (static_cast<lock_state_t>(proxy_id) << (sizeof(refcount_t) * CHAR_BIT)) | static_cast<lock_state_t>(refcount);
        }

        static constexpr auto proxy_id(lock_state_t state) noexcept -> proxy_id_t{

            return state >> (sizeof(refcount_t) * CHAR_BIT);
        }

        static constexpr auto refcount(lock_state_t state) noexcept -> refcount_t{

            constexpr lock_state_t BITMASK = (lock_state_t{1} << (sizeof(refcount_t) * CHAR_BIT)) - 1; 
            return state & BITMASK;
        }
    };

    template <class ID, class MemRegionSize, class ProxyIDType, class RefCountType, class PtrType>
    struct AtomicReferenceLock{};

    template <class ID, size_t MEMREGION_SZ, class ProxyIDType, class RefCountType, class PtrType>
    struct AtomicReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, ProxyIDType, RefCountType, PtrType>: ReferenceLockInterface<AtomicReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, ProxyIDType, RefCountType, PtrType>>{

        public:

            using proxy_id_t    = ProxyIDType;
            using ptr_t         = PtrType;  

        private:

            using self          = AtomicReferenceLock;
            using controller    = AtomicLockStateController<ProxyIDType, RefCountType>; 
            using lock_state_t  = typename controller::lock_state_t; 
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>;

            static inline std::atomic<lock_state_t> * lck_table{};

            static inline auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto to_table_idx(ptr_t ptr) noexcept -> size_t{

                return memregion_slot(segcheck_ins::access(ptr));
            } 

            static inline auto internal_acquire_try(size_t table_idx) noexcept -> std::optional<proxy_id_t>{

                lock_state_t cur = lck_table[table_idx].load(std::memory_order_acquire);

                if (controller::refcount(cur) != controller::REFERENCE_EMPTY_VALUE){
                    return std::nullopt;
                }

                proxy_id_t cur_proxy    = controller::proxy_id(cur);
                lock_state_t nxt        = controller::make(cur_proxy, controller::REFERENCE_ACQUIRED_VALUE);

                if (!lck_table[table_idx].compare_exchange_weak(cur, nxt, std::memory_order_acq_rel)){
                    return std::nullopt;
                }

                return cur_proxy;
            }

            static inline void internal_acquire_wait(size_t table_idx) noexcept -> proxy_id_t{

                while (true){
                    if (auto rs = internal_acquire_try(table_idx); static_cast<bool>(rs)){
                        return rs.value();
                    }
                }
            } 

            static inline void internal_acquire_release(size_t table_idx, proxy_id_t new_proxy_id) noexcept{

                lck_table[table_idx].exchange(controller::make(new_proxy_id, controller::REFERENCE_EMPTY_VALUE), std::memory_order_acq_rel);
            }
            
            static inline auto internal_reference_try(size_t table_idx, proxy_id_t expected_proxy_id) noexcept -> bool{

                lock_state_t cur = lck_table[table_idx].load(std::memory_order_acquire);

                if (controller::proxy_id(cur) != expected_proxy_id){
                    return false;
                }

                if (controller::refcount(cur) == controller::REFERENCE_ACQUIRED_VALUE){
                    return false;
                }

                lock_state_t nxt = controller::make(expected_proxy_id, controller::refcount(cur) + 1);
                return lck_table[table_idx].compare_exchange_weak(cur, nxt, std::memory_order_acq_rel);
            } 

            static inline void internal_reference_wait(size_t table_idx, proxy_id_t proxy_id) noexcept{

                while (!internal_reference_try(table_idx, proxy_id)){}
            } 

            static inline void internal_reference_release(size_t table_idx) noexcept{

                lck_table[table_idx].fetchSub(1u, std::memory_order_acq_rel);
            }

        public:

            static void init(ptr_t * region, proxy_id_t * initial_proxy, size_t n){

                auto logger = dg::network_log_scope::critical_terminate(); 
                
                if (n == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                ptr_t back_region       = *std::max_element(region, region + n);
                ptr_t last_region       = memult::advance(back_region, MEMREGION_SZ);
                size_t lck_table_sz     = pointer_cast<typename dg::pointer_info<ptr_t>::max_unsigned_t>(last_region) / MEMREGION_SZ; 
                lck_table               = new std::atomic<lock_state_t>[lck_table_sz];

                segcheck_ins::init(dg::pointer_limits<ptr_t>::min(), last_region);

                for (size_t i = 0u; i < n; ++i){
                    if (memregion_offset(region[i]) != 0u || memregion_slot(region[i]) == 0u){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }
                    internal_acquire_release(to_table_idx(region[i]), initial_proxy[i]);
                }

                log_scope.release();
            }

            static inline auto acquire_try(ptr_t ptr) noexcept -> std::optional<proxy_id_t>{
                
                return internal_acquire_try(to_table_idx(ptr));
            } 

            static inline auto acquire_wait(ptr_t ptr) noexcept -> proxy_id_t{

                return internal_acquire_wait(to_table_idx(ptr));
            }

            static inline void acquire_release(proxy_id_t new_proxy_id, ptr_t ptr) noexcept{

                internal_acquire_release(to_table_idx(ptr), new_proxy_id);
            }

            static inline auto reference_try(proxy_id_t expected_proxy_id, ptr_t ptr) noexcept -> bool{

                return internal_reference_try(to_table_idx(ptr), expected_proxy_id);
            } 

            static inline void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(to_table_idx(ptr));
            }
    };

    template <class ID, class MemRegionSize, class ProxyIDType, class RefCountType, class PtrType>
    struct MtxReferenceLock{}; 

    template <class ID, size_t MEMREGION_SZ, class ProxyIDType, class RefCountType, class PtrType>
    struct MtxReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, ProxyIDType, RefCountType, PtrType>: ReferenceLockInterface<MtxReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, ProxyIDType, RefCountType, PtrType>>{

        public:

            using proxy_id_t    = ProxyIDType;
            using ptr_t         = PtrType;
        
        private:

            using refcount_t    = RefCountType; 
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>;

            static_assert(std::is_unsigned_v<refcount_t>);

            static inline constexpr refcount_t REFERENCE_ACQUIRED_VALUE = ~refcount_t{0u}; 
            static inline constexpr refcount_t REFERENCE_EMPTY_VALUE    = refcount_t{0u};
    
            struct ControlUnit{
                Lock lck;
                proxy_id_t proxy_id;
                refcount_t refcount;
            };

            static inline auto ControlUnit * lck_table{};
            
            static inline auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto to_table_idx(ptr_t ptr) noexcept -> size_t{

                return memregion_slot(segcheck_ins::access(ptr));
            }

            static inline auto internal_acquire_try(size_t table_idx) noexcept -> std::optional<proxy_id_t>{

                auto lck_grd = stdx::lock_guard(lck_table[table_idx].lck);

                if (lck_table[table_idx].refcount != REFERENCE_EMPTY_VALUE){
                    return std::nullopt;
                }

                lck_table[table_idx].refcount = REFERENCE_ACQUIRED_VALUE;
                return lck_table[table_idx].proxy_id;
            } 

            static inline auto internal_acquire_wait(size_t table_idx) noexcept -> proxy_id_t{

                while (true){
                    if (auto rs = internal_acquire_try(table_idx); static_cast<bool>(rs)){
                        return rs.value();
                    }
                }
            }

            static inline void internal_acquire_release(size_t table_idx, proxy_id_t new_proxy_id) noexcept{

                auto lck_grd = stdx::lock_guard(lck_table[table_idx].lck);
                lck_table[table_idx].proxy_id = new_proxy_id;
                lck_table[table_idx].refcount = REFERENCE_EMPTY_VALUE;
            }

            static inline auto internal_reference_try(size_t table_idx, proxy_id_t expected_proxy_id) noexcept -> bool{

                auto lck_grd = stdx::lock_guard(lck_table[table_idx].lck);

                if (lck_table[table_idx].proxy_id != expected_proxy_id){
                    return false;
                }

                if (lck_table[table_idx].refcount == REFERENCE_ACQUIRED_VALUE){
                    return false; 
                }

                lck_table[table_idx].refcount += 1;
                return true;
            }

            static inline void internal_reference_wait(size_t table_idx, proxy_id_t expected_proxy_id) noexcept{

                while (!internal_reference_try(table_idx, expected_proxy_id)){}
            }

            static inline void internal_reference_release(size_t table_idx) noexcept{

                auto lck_grd = stdx::lock_guard(lck_table[table_idx].lck);
                lck_table[table_idx].refcount -= 1;
            }

        public:

            static void init(ptr_t * region, proxy_id_t * initial_proxy, size_t n){

                auto log_scope  = dg::network_log_scope::critical_terminate();

                if (n == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                ptr_t back_region       = *std::max_element(region, region + n);
                ptr_t last_region       = memult::advance(back_region, MEMREGION_SZ);
                size_t lck_table_sz     = pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(last_region) / MEMREGION_SZ; 
                lck_table               = new ControlUnit[lck_table_sz];

                segcheck_ins::init(dg::pointer_limits<ptr_t>::min(), last_region);

                for (size_t i = 0u; i < n; ++i){
                    if (memregion_offset(region[i]) != 0u || memregion_slot(region[i]) == 0u){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }
                    internal_acquire_release(to_table_idx(region[i]), initial_proxy[i]);
                }

                log_scope.release();
            }

            static inline auto acquire_try(ptr_t ptr) noexcept -> std::optional<proxy_id_t>{
                
                return internal_acquire_try(to_table_idx(ptr));
            } 

            static inline auto acquire_wait(ptr_t ptr) noexcept -> proxy_id_t{

                return internal_acquire_wait(to_table_idx(ptr));
            }

            static inline void acquire_release(proxy_id_t new_proxy_id, ptr_t ptr) noexcept{

                internal_acquire_release(to_table_idx(ptr), new_proxy_id);
            }

            static inline auto reference_try(proxy_id_t expected_proxy_id, ptr_t ptr) noexcept -> bool{

                return internal_reference_try(to_table_idx(ptr), expected_proxy_id);
            } 

            static inline void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(to_table_idx(ptr));
            }
    }

    static inline constexpr bool IS_ATOMIC_OPERATION_PREFERRED = true;

    template <class ID, class MemRegionSize, class PtrType = std::add_pointer_t<const void>, class ProxyIDType = uint32_t, class RefCountType = uint32_t>
    using ReferenceLock = std::conditional_t<IS_ATOMIC_OPERATION_PREFERRED, 
                                             AtomicReferenceLock<ID, MemRegionSize, ProxyIDType, RefCountType, PtrType>,
                                             MtxReferenceLock<ID, MemRegionSize, ProxyIDType, RefCountType, PtrType>>; 


} 

#endif 