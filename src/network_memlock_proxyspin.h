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

    //I've been working on the notify + lock features for acquire + acquire reference + acquire_reference

    //can we prove that for every state change, from false -> true, true -> false, a notification in between is sufficient?
    //this proof is hard to write, we already proved that the atomic_lock + notify + cmp_exch_strong would work, how about that in this case?
    //it seems like the logic of that is carried over to this
    //the China-man keymaker came to me in my dream

    //except for that we need a dedicated boolean for each of the feature, and making sure that the acquisition has the reverse operation (such responsibility is created upon acquisition)
    //today we are implementing a very hard implementation

    //alright guys, I got a curse such that I could not see more than the immediate obvious, due to certain environmental problems within recent years

    //we'll attempt to implement this notify_one() and we'll prove the completeness such that if we notify after every true -> false border cross, we'll be fine
    //this is you weak on the floor so you call her cell
    //we aren't weak fellas, we are stronk
    //why a notification every post border cross is sufficient?
    //assume that notify_one() successfully wakes up one of the waitee - fine
    //assume that notify_one() does not successfully wake up one of the waitee, then there is another guy in progress (this can only be guaranteed by compare exchange stronk), which would guarantee that our next wait() would be before or after the last notifying event

    //we'll be focusing on acquirability + referenceability

    //acquirability after demoted will have two values acquirable or not acquirable
    //referenceability after demoted will also have two values referenceable and not referenceable

    //in another approach
    //we need to be able to prove the if state1 => state2
    //state1 being the acquirability or referencebility, state2 means the future promise of calling notify(), false => promise will notify, true => does not promise notifability, we are to thru
    //the problem is that we need to be able to do the state1 in between the state2

    //alright fellas, I think the only way that this would work is to do the traditional approach
    //if this is implemented correctly
    //all the memlocks + uma_locks
    //memlock retried 3 times for each memregion + wait for the 4th times ...
    //uma_lock wait (we have not found a better way to do this than to match the memregion_lock -> uma_region)
    //we'd be blazingly fast

    //we are looking for an implementation such that: memevents -> warehouse (direct, without signal_smph_addr)
    //                                                                       (direct, with signal_smph_addr)

    //                                                memevents -> mempress (only for signal_smph_addr)

    //forward only (by implementing observer_data * + observer_arr_sz)
    //search implementation... how to???
    //this uma_region is our most proud product, from the CUTF -> CUDA -> HOST -> SSD_FILE_SYSTEM -> etc.
    //as long as every operation on the uma region is through, we can prove that this has no deadlocks 
    //this solves so many problems that the cuda unified memory address could not solve, namely the bottlenecked MEMBUS 
    //we are having the full CUDA speed + platform transferability without actually compromising the speed !!!

    //we have yet to convert the logics of green despair -> desert scream or bloody dice YET
    //we are hopeful because we know that the logic is inter-convertible

    //we are searching + moving towards the coordinate by using unit displacement vector
    //we are using very fat tiles, 32KB -> 64KB containing a compressed crit dataset of 1MM training data points, we are searching the coordinate on CUDA -> convert to the "traditional backward" value
    //we are very limited by the technology of our times, we'd hope that we could deploy this on 1BB devices in a forseeable future
    //we can't really do search + set absolute values because of certain training constraints

    static inline constexpr bool IS_ATOMIC_OPERATION_PREFERRED = false;

    struct increase_reference_tag{}; 

    //I have never implemented a single more confusing thing than this lock 
    //I admit the heap allocation is very confusing

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
        static auto reference_try(typename T1::ptr_t ptr, typename T1::proxy_id_t expected_proxy_id) noexcept -> bool{

            return T::reference_try(ptr, expected_proxy_id);
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

            static inline std::unique_ptr<stdx::hdi_container<std::atomic<lock_state_t>>[]> lck_table{};
            static inline std::unique_ptr<stdx::hdi_container<std::atomic_flag>[]> acquirability_table{};
            static inline std::unique_ptr<stdx::hdi_container<std::atomic_flag>[]> referenceability_table{};

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

                //thru, we are to set the acquirability -> false and referenceability -> false

                acquirability_table[table_idx].value.clear(std::memory_order_relaxed);
                referenceability_table[table_idx].value.clear(std::memory_order_relaxed);

                std::atomic_signal_fence(std::memory_order_seq_cst);

                return cur_proxy;
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

                while (true){
                    bool was_thru = stdx::eventloop_spin_expbackoff(lambda, stdx::SPINLOCK_SIZE_MAGIC_VALUE);

                    if (was_thru){
                        break;
                    }

                    acquirability_table[table_idx].value.wait(false, std::memory_order_relaxed); //can we, you, me prove that this is thru ?, assume that notify_one() does not wake up, last value is true, current value is false, other dude will notify, it is fine to not thru
                                                                                                 //assume that current value is true, last value is false, we are to take the "call of duty", compare exchange strong is thru => OK, compare exchange strong is not thru
                                                                                                 //compare exchange strong is only for the FIRST GUY, that's why the you weak on the floor so you call her cell, 8 calls all you
                                                                                                 //=> other guy is in progress, the next wait is gonna be guaranteed to be woken up by the last notify_one(), assume wait() is before last notify_one() => at worst woken by last notify_one() 
                                                                                                 //assume wait() is after the last notify_one(), guaranteed to be woken up by the self wait(false)

                                                                                                 //the overview of this fling is a guy taking responsibily of decrementing the waiting queue
                                                                                                 //this responsibility is notify_one(), if another guy got in the way, then the guy is taking the responsibility of decrementing the queue

                                                                                                 //the most important hinge of this exercise is the last notifying guy and the current wait()
                                                                                                 //the punchline is the notify_one(), before the notify_one() will be at worst woken by the notify_one(), proof above, false => will notify, true => does not guarantee notifiability
                                                                                                 //                                                                                                       at call: false (other guy will take the responsibility of decrementing the queue -> 0 => converge to case number 2 (prove this) => notify), 
                                                                                                 //                                                                                                                true => the next guy's gonna take the responsibility of decrementing the queue -> 0 after decrementing the queue => notify)
                                                                                                 //                                                                                                                        the next guy does not take the cmpexch right, other guy's gonna take the responsibility, cyclic recursive definition => 1
                                                                                                 //                                                                                                                        the next guy takes the cmpexch right, the guy's gonna take the responsibility

                                                                                                 //after the notify_one() will be self woken
                                                                                                 //proof by contradiction, assume that after the notify_one() is false
                                                                                                 //then there guaranteed to be a notify_one() after the point, the notify_one() is not the last notify_one()
                                                                                                 //this exercise if not tackled by the right angle would definitely ruin the scholarability of yall fellas
                                                                                                 //it is very important to find the hinges to prove these exercises

                                                                                                 //how about notify all, the only difference is that there are not only before or after for notify all, there are also in between notify all
                                                                                                 //if inbetween notify all => false => the notify_all() is not the last notify all
                                                                                                 //the last notify_all() would wake all the waiting fellas 
                }

                return rs;
            }

            static void internal_acquire_release(size_t table_idx, proxy_id_t new_proxy_id) noexcept{

                //this is the punchline

                std::atomic_signal_fence(std::memory_order_seq_cst);
                acquirability_table[table_idx].value.test_and_set(std::memory_order_relaxed);
                referenceability_table[table_idx].value.test_and_set(std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                lck_table[table_idx].value.exchange(controller::make(new_proxy_id, controller::REFERENCE_EMPTY_VALUE), std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                acquirability_table[table_idx].value.notify_one();
                referenceability_table[table_idx].value.notify_all();

                std::atomic_signal_fence(std::memory_order_seq_cst); //this is unnecessary
            }

            static void internal_acquire_release(size_t table_idx, proxy_id_t new_proxy_id, const increase_reference_tag){

                std::atomic_signal_fence(std::memory_order_seq_cst);
                referenceability_table[table_idx].value.test_and_set(std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                lck_table[table_idx].value.exchange(controller::make(new_proxy_id, 1u), std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                referenceability_table[table_idx].value.notify_all();

                std::atomic_signal_fence(std::memory_order_seq_cst); //this is unnecessary
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

            static void internal_reference_wait(size_t table_idx, proxy_id_t expected_proxy_id) noexcept{

                auto lambda = [&]() noexcept{
                    return internal_reference_try(table_idx, expected_proxy_id);
                };

                while (true){
                    bool was_thru = stdx::eventloop_spin_expbackoff(lambda, stdx::SPINLOCK_SIZE_MAGIC_VALUE);

                    if (was_thru){
                        break;
                    }

                    referenceability_table[table_idx].value.wait(false, std::memory_order_relaxed);
                }
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

                stdx::eventloop_spin_expbackoff(lambda);
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
                referenceability_table  = std::make_unique<stdx::hdi_container<std::atomic_flag>[]>(lck_table_sz);
                region_first            = first_region;

                for (size_t i = 0u; i < n; ++i){
                    size_t table_idx                        = memregion_slot(region_arr[i]);
                    lck_table[table_idx].value              = controller::make(initial_proxy_arr[i], controller::REFERENCE_EMPTY_VALUE);
                    acquirability_table[table_idx].value    = false;
                    acquirability_table[table_idx].value    = true;
                    referenceability_table[table_idx].value = false;
                    referenceability_table[table_idx].value = true;
                }
            
                segcheck_ins::init(first_region, last_region);
            }

            static void deinit() noexcept{

                lck_table               = nullptr;
                acquirability_table     = nullptr;
                referenceability_table  = nullptr;
            }

            static auto acquire_try(ptr_t ptr) noexcept -> std::optional<proxy_id_t>{
                
                return internal_acquire_try(memregion_slot(segcheck_ins::access(ptr)));
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

            static auto reference_try(ptr_t ptr, proxy_id_t expected_proxy_id) noexcept -> bool{

                return internal_reference_try(memregion_slot(segcheck_ins::access(ptr)), expected_proxy_id);
            } 

            static void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(memregion_slot(segcheck_ins::access(ptr)));
            }
    };

    template <class ID, class MemRegionSize, class ProxyIDType, class RefCountType, class MutexT, class PtrType>
    struct MtxReferenceLock{}; 

    template <class ID, size_t MEMREGION_SZ, class ProxyIDType, class RefCountType, class MutexT, class PtrType>
    struct MtxReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, ProxyIDType, RefCountType, MutexT, PtrType>: ReferenceLockInterface<MtxReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, ProxyIDType, RefCountType, MutexT, PtrType>>{

        public:

            using proxy_id_t    = ProxyIDType;
            using ptr_t         = PtrType;
        
        private:

            using self          = MtxReferenceLock;
            using uptr_t        = typename dg::ptr_info<ptr_t>::max_unsigned_t;
            using refcount_t    = RefCountType; 
            using mutex_t       = MutexT;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>;

            static_assert(std::is_unsigned_v<refcount_t>);

            static inline constexpr refcount_t REFERENCE_EMPTY_VALUE    = 0u;
            static inline constexpr refcount_t REFERENCE_ACQUIRED_VALUE = std::numeric_limits<refcount_t>::max(); 
    
            struct ControlUnit{
                mutex_t lck;
                proxy_id_t proxy_id;
                refcount_t refcount;
            };

            static inline std::unique_ptr<ControlUnit[]> lck_table{};
            static inline ptr_t region_first{}; 

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static auto internal_acquire_try(size_t table_idx) noexcept -> std::optional<proxy_id_t>{

                stdx::xlock_guard<mutex_t> lck_grd(lck_table[table_idx].lck);

                if (lck_table[table_idx].refcount != REFERENCE_EMPTY_VALUE){
                    return std::nullopt;
                }

                lck_table[table_idx].refcount = REFERENCE_ACQUIRED_VALUE;
                return lck_table[table_idx].proxy_id;
            } 

            static auto internal_acquire_wait(size_t table_idx) noexcept -> proxy_id_t{
                
                std::optional<proxy_id_t> rs{};
                auto lambda = [&]() noexcept{
                    rs = internal_acquire_try(table_idx);
                    return rs.has_value();
                };
                stdx::eventloop_spin_expbackoff(lambda);

                return rs.value();
            }

            static void internal_acquire_release(size_t table_idx, proxy_id_t new_proxy_id) noexcept{

                stdx::xlock_guard<mutex_t> lck_grd(lck_table[table_idx].lck);

                lck_table[table_idx].proxy_id = new_proxy_id;
                lck_table[table_idx].refcount = REFERENCE_EMPTY_VALUE;
            }

            static void internal_acquire_release(size_t table_idx, proxy_id_t new_proxy_id, const increase_reference_tag){

                stdx::xlock_guard<mutex_t> lck_grd(lck_table[table_idx].lck);

                lck_table[table_idx].proxy_id = new_proxy_id;
                lck_table[table_idx].refcount = 1;
            } 

            static auto internal_reference_try(size_t table_idx, proxy_id_t expected_proxy_id) noexcept -> bool{

                stdx::xlock_guard<mutex_t> lck_grd(lck_table[table_idx].lck);

                if (lck_table[table_idx].proxy_id != expected_proxy_id){
                    return false;
                }

                if (lck_table[table_idx].refcount == REFERENCE_ACQUIRED_VALUE){
                    return false; 
                }

                lck_table[table_idx].refcount += 1;
                return true;
            }

            static void internal_reference_wait(size_t table_idx, proxy_id_t expected_proxy_id) noexcept{

                auto lambda = [&]() noexcept{
                    return internal_reference_try(table_idx, expected_proxy_id);
                };

                stdx::eventloop_spin_expbackoff(lambda);
            }

            static void internal_reference_release(size_t table_idx) noexcept{

                stdx::xlock_guard<mutex_t> lck_grd(lck_table[table_idx].lck);
                lck_table[table_idx].refcount -= 1;
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

                ptr_t first_region  = *std::min_element(region_arr, region_arr + n, dg::memult::ptrcmpless_lambda);
                ptr_t last_region   = dg::memult::advance(*std::max_element(region_arr, region_arr + n, dg::memult::ptrcmpless_lambda), MEMREGION_SZ);
                size_t lck_table_sz = dg::memult::distance(first_region, last_region) / MEMREGION_SZ;
                lck_table           = std::make_unique<ControlUnit[]>(lck_table_sz); 
                region_first        = first_region;

                for (size_t i = 0u; i < n; ++i){
                    size_t table_idx                = memregion_slot(region_arr[i]); 
                    lck_table[table_idx].proxy_id   = initial_proxy_arr[i];
                    lck_table[table_idx].refcount   = REFERENCE_EMPTY_VALUE;
                }

                segcheck_ins::init(first_region, last_region);
            }

            static void deinit() noexcept{

                lck_table = nullptr;
            }

            static auto acquire_try(ptr_t ptr) noexcept -> std::optional<proxy_id_t>{
                
                return internal_acquire_try(memregion_slot(segcheck_ins::access(ptr)));
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

            static auto reference_try(ptr_t ptr, proxy_id_t expected_proxy_id) noexcept -> bool{

                return internal_reference_try(memregion_slot(segcheck_ins::access(ptr)), expected_proxy_id);
            } 

            static void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(memregion_slot(segcheck_ins::access(ptr)));
            }
    };

    template <class ID, class MemRegionSize, class PtrType = std::add_pointer_t<const void>, class ProxyIDType = uint32_t, class RefCountType = uint32_t>
    using ReferenceLock = std::conditional_t<IS_ATOMIC_OPERATION_PREFERRED, 
                                             AtomicReferenceLock<ID, MemRegionSize, ProxyIDType, RefCountType, PtrType>,
                                             MtxReferenceLock<ID, MemRegionSize, ProxyIDType, RefCountType, std::atomic_flag, PtrType>>; 


} 

#endif 