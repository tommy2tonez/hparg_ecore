#ifndef __NETWORK_MEMLOCKPX_H__
#define __NETWORK_MEMLOCKPX_H__

#include <stdlib.h>
#include <stdint.h>
#include <type_traits>
#include <atomic>
#include <limits.h>
#include "network_segcheck_bound.h"
#include "network_exception.h"
#include <mutex>
#include <optional>
#include "stdx.h"
#include "network_pointer.h"

namespace dg::network_memlock_proxyspin{

    struct increase_reference_tag{}; 

    //just got a feedback guys
    //we have to implement that acquire_waitnolock + acquire_waitnolock_release_responsibility
    //this is not my decision, I would rather do that at the memlock yet this does not solve the problem of acquiring memregions

    template <class T>
    struct ReferenceLockInterface{

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using proxy_id_t    = typename T1::proxy_id_t; 
        
        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using ptr_t         = typename T1::ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto acquire_try(typename T1::ptr_t ptr) noexcept -> std::optional<typename T1::proxy_id_t>{

            return T::acquire_try(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto acquire_try_strong(typename T1::ptr_t ptr) noexcept -> std::optional<typename T1::proxy_id_t>{

            return T::acquire_try_strong(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto acquire_wait(typename T1::ptr_t ptr) noexcept -> typename T1::proxy_id_t{

            return T::acquire_wait(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void acquire_release(typename T1::ptr_t ptr, typename T1::proxy_id_t new_proxy_id) noexcept{

            T::acquire_release(ptr, new_proxy_id);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void acquire_release(typename T1::ptr_t ptr, typename T1::proxy_id_t new_proxy_id, const increase_reference_tag){

            T::acquire_release(ptr, new_proxy_id, increase_reference_tag{});
        } 

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void acquire_waitnolock(typename T1::ptr_t ptr) noexcept{

            T::acquire_waitnolock(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void acquire_waitnolock_release_responsibility(typename T1::ptr_t ptr) noexcept{

            T::acquire_waitnolock_release_responsibility(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto reference_try(typename T1::ptr_t ptr, typename T1::proxy_id_t expected_proxy_id) noexcept -> bool{

            return T::reference_try(ptr, expected_proxy_id);
        } 

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto reference_try_strong(typename T1::ptr_t ptr, typename T1::proxy_id_t expected_proxy_id) noexcept -> bool{

            return T::reference_try_strong(ptr, expected_proxy_id);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void reference_release(typename T1::ptr_t ptr) noexcept{

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

        static inline constexpr refcount_t REFERENCE_EMPTY_VALUE        = 0u;
        static inline constexpr refcount_t REFERENCE_ACQUIRED_VALUE     = std::numeric_limits<refcount_t>::max();
        static inline constexpr refcount_t REFERENCE_INTERMEDIATE_VALUE = std::numeric_limits<refcount_t>::max() - 1u;

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
            using uptr_t        = typename dg::ptr_info<ptr_t>::max_unsigned_t;

            static inline constexpr size_t FOREHEAD_SPIN_SZ                             = 16u;
            static inline constexpr std::chrono::nanoseconds FOREHEAD_SPIN_PERIOD       = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds(10));

            static inline constexpr size_t COMPETITIVE_SPIN_SZ                          = 32u;
            static inline constexpr std::chrono::nanoseconds COMPETITIVE_SPIN_PERIOD    = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds(10)); 

            static inline std::unique_ptr<stdx::hdi_container<std::atomic<lock_state_t>>[]> lck_table{};
            static inline std::unique_ptr<stdx::hdi_container<std::atomic_flag>[]> acquirability_table{};
            static inline ptr_t region_first{}; 

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static auto internal_acquire_try(size_t table_idx) noexcept -> std::optional<proxy_id_t>{

                lock_state_t cur = lck_table[table_idx].value.load(std::memory_order_relaxed);

                if (controller::refcount(cur) != controller::REFERENCE_EMPTY_VALUE){
                    return std::nullopt;
                }

                proxy_id_t cur_proxy    = controller::proxy_id(cur);
                lock_state_t nxt        = controller::make(cur_proxy, controller::REFERENCE_ACQUIRED_VALUE); 

                if (!lck_table[table_idx].value.compare_exchange_strong(cur, nxt, std::memory_order_relaxed)){
                    return std::nullopt;
                }

                acquirability_table[table_idx].value.clear(std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                return cur_proxy;
            }

            static auto internal_acquire_try_strong(size_t table_idx) noexcept -> std::optional<proxy_id_t>{

                return internal_acquire_try(table_idx);
            }

            static auto internal_acquire_wait(size_t table_idx) noexcept -> proxy_id_t{

                proxy_id_t rs   = {}; 

                auto lambda     = [&]() noexcept{
                    std::optional<proxy_id_t> proxy = internal_acquire_try(table_idx);

                    if (!proxy.has_value()){
                        return false;
                    }

                    rs = proxy.value();
                    return true;
                };

                bool was_thru = stdx::eventloop_expbackoff_spin(lambda, FOREHEAD_SPIN_SZ, FOREHEAD_SPIN_PERIOD);

                if (was_thru){
                    return;
                }

                while (true){
                    was_thru = lambda();

                    if (was_thru){
                        break;
                    }

                    acquirability_table[table_idx].value.wait(false, std::memory_order_relaxed);
                }

                return rs;
            }

            static void internal_acquire_release(size_t table_idx, proxy_id_t new_proxy_id) noexcept{

                std::atomic_signal_fence(std::memory_order_seq_cst);
                acquirability_table[table_idx].value.test_and_set(std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                lck_table[table_idx].value.exchange(controller::make(new_proxy_id, controller::REFERENCE_EMPTY_VALUE), std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                acquirability_table[table_idx].value.notify_one();

                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            static void internal_acquire_release(size_t table_idx, proxy_id_t new_proxy_id, const increase_reference_tag) noexcept{

                std::atomic_signal_fence(std::memory_order_seq_cst);
                lck_table[table_idx].value.exchange(controller::make(new_proxy_id, 1u), std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            static void internal_acquire_waitnolock(size_t table_idx) noexcept{

                acquirability_table[table_idx].value.wait(false, std::memory_order_relaxed);
            }

            static void internal_acquire_waitnolock_release_responsibility(size_t table_idx) noexcept{

                acquirability_table[table_idx].value.notify_one();
            }

            static auto internal_reference_try(size_t table_idx, proxy_id_t expected_proxy_id) noexcept -> bool{

                lock_state_t cur = lck_table[table_idx].value.load(std::memory_order_relaxed);

                if (controller::proxy_id(cur) != expected_proxy_id){
                    return false;
                }

                if (controller::refcount(cur) == controller::REFERENCE_ACQUIRED_VALUE){
                    return false;
                }

                if (controller::refcount(cur) == controller::REFERENCE_INTERMEDIATE_VALUE){
                    return false;
                }

                lock_state_t nxt    = controller::make(expected_proxy_id, controller::refcount(cur) + 1);

                if (!lck_table[table_idx].value.compare_exchange_strong(cur, nxt, std::memory_order_relaxed)){
                    return false;
                }

                acquirability_table[table_idx].value.clear(std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                return true;
            }

            static auto internal_reference_try_strong(size_t table_idx, proxy_id_t expected_proxy_id) noexcept -> bool{

                bool strong_result = {}; 

                auto lambda = [&]() noexcept{
                    lock_state_t cur = lck_table[table_idx].value.load(std::memory_order_relaxed);

                    if (controller::proxy_id(cur) != expected_proxy_id){
                        strong_result = false;
                        return true;
                    }

                    if (controller::refcount(cur) == controller::REFERENCE_ACQUIRED_VALUE){
                        strong_result = false;
                        return true;
                    }

                    if (controller::refcount(cur) == controller::REFERENCE_INTERMEDIATE_VALUE){
                        return false;
                    }

                    lock_state_t nxt    = controller::make(expected_proxy_id, controller::refcount(cur) + 1);

                    if (!lck_table[table_idx].value.compare_exchange_weak(cur, nxt, std::memory_order_relaxed)){
                        return false;
                    }

                    acquirability_table[table_idx].value.clear(std::memory_order_relaxed);
                    std::atomic_signal_fence(std::memory_order_seq_cst);

                    strong_result = true;
                    return true;    
                };

                stdx::eventloop_cyclic_expbackoff_spin(lambda, COMPETITIVE_SPIN_SZ, COMPETITIVE_SPIN_PERIOD);
                return strong_result;
            }

            static void internal_reference_release(size_t table_idx) noexcept{

                auto lambda = [&]() noexcept{
                    lock_state_t cur        = lck_table[table_idx].value.load(std::memory_order_relaxed);
                    proxy_id_t cur_proxy    = controller::proxy_id(cur);
                    lock_state_t inter      = controller::make(cur_proxy, controller::REFERENCE_INTERMEDIATE_VALUE); 
                    lock_state_t nxt        = controller::make(cur_proxy, controller::refcount(cur) - 1u);

                    if (!lck_table[table_idx].value.compare_exchange_weak(cur, inter, std::memory_order_relaxed)){
                        return false;
                    }

                    if (controller::refcount(cur) == 1u){
                        acquirability_table[table_idx].value.test_and_set(std::memory_order_relaxed);
                    }

                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    lck_table[table_idx].value.exchange(nxt, std::memory_order_relaxed);

                    return true;
                };

                stdx::eventloop_cyclic_expbackoff_spin(lambda, COMPETITIVE_SPIN_SZ, COMPETITIVE_SPIN_PERIOD);
                acquirability_table[table_idx].value.notify_one();
                std::atomic_signal_fence(std::memory_order_seq_cst); //this is unnecessary
            }

        public:

            static void init(ptr_t * region_arr, proxy_id_t * initial_proxy_arr, size_t n){

                if (n == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);    
                }

                for (size_t i = 0u; i < n; ++i){
                    uptr_t uregion = pointer_cast<uptr_t>(region_arr[i]);

                    if (uregion % MEMREGION_SZ != 0u){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }

                    if (region_arr[i] == dg::ptr_limits<ptr_t>::null_value()){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }
                }

                ptr_t first_region      = *std::min_element(region_arr, region_arr + n, dg::memult::ptrcmpless_lambda);
                ptr_t last_region       = dg::memult::advance(*std::max_element(region_arr, region_arr + n, dg::memult::ptrcmpless_lambda), MEMREGION_SZ);
                size_t lck_table_sz     = dg::memult::distance(first_region, last_region) / MEMREGION_SZ;
                lck_table               = std::make_unique<stdx::hdi_container<std::atomic<lock_state_t>>[]>(lck_table_sz);
                acquirability_table     = std::make_unique<stdx::hdi_container<std::atomic_flag>[]>(lck_table_sz);
                region_first            = first_region;

                for (size_t i = 0u; i < n; ++i){
                    size_t table_idx                        = memregion_slot(region_arr[i]);
                    lck_table[table_idx].value              = controller::make(initial_proxy_arr[i], controller::REFERENCE_EMPTY_VALUE);
                    acquirability_table[table_idx].value    = false;
                    acquirability_table[table_idx].value    = true;
                }
            
                segcheck_ins::init(first_region, last_region);
            }

            static void deinit() noexcept{

                lck_table               = nullptr;
                acquirability_table     = nullptr;
            }

            static auto acquire_try(ptr_t ptr) noexcept -> std::optional<proxy_id_t>{
                
                return internal_acquire_try(memregion_slot(segcheck_ins::access(ptr)));
            } 

            static auto acquire_try_strong(ptr_t ptr) noexcept -> std::optional<proxy_id_t>{

                return internal_acquire_try_strong(memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto acquire_wait(ptr_t ptr) noexcept -> proxy_id_t{

                return internal_acquire_wait(memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_release(ptr_t ptr, proxy_id_t new_proxy_id) noexcept{

                internal_acquire_release(memregion_slot(segcheck_ins::access(ptr)), new_proxy_id);
            }

            static void acquire_release(ptr_t ptr, proxy_id_t new_proxy_id, const increase_reference_tag) noexcept{

                internal_acquire_release(memregion_slot(segcheck_ins::access(ptr)), new_proxy_id, increase_reference_tag{});
            }

            static void acquire_waitnolock(ptr_t ptr) noexcept{
                
                internal_acquire_waitnolock(memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_waitnolock_release_responsibility(ptr_t ptr) noexcept{

                internal_acquire_waitnolock_release_responsibility(segcheck_ins::access(ptr));
            }

            static auto reference_try(ptr_t ptr, proxy_id_t expected_proxy_id) noexcept -> bool{

                return internal_reference_try(memregion_slot(segcheck_ins::access(ptr)), expected_proxy_id);
            }

            static auto reference_try_strong(ptr_t ptr, proxy_id_t expected_proxy_id) noexcept -> bool{

                return internal_reference_try_strong(memregion_slot(segcheck_ins::access(ptr)), expected_proxy_id);
            }

            static void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(memregion_slot(segcheck_ins::access(ptr)));
            }
    };

    template <class ID, class MemRegionSize, class PtrType = std::add_pointer_t<const void>, class ProxyIDType = uint32_t, class RefCountType = uint32_t>
    using ReferenceLock = AtomicReferenceLock<ID, MemRegionSize, ProxyIDType, RefCountType, PtrType>;
} 

#endif 