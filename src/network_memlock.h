#ifndef __DG_NETWORK_MEMLOCK_H__
#define __DG_NETWORK_MEMLOCK_H__

#include <stdint.h>
#include <stddef.h>
#include <atomic>
#include <type_traits>
#include <utility>
#include <tuple>
#include <cstring>
#include <array>
#include "network_memult.h" 
#include <mutex>
#include "network_exception.h"
#include "network_segcheck_bound.h"
#include "stdx.h"
#include <cstdlib>
#include "network_raii_x.h"
#include <optional>

namespace dg::network_memlock{

    template <class T>
    struct MemoryLockInterface{

        using interface_t   = MemoryLockInterface<T>;
        
        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using ptr_t         = typename T1::ptr_t;
        
        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto acquire_try(typename T1::ptr_t ptr) noexcept -> bool{

            return T::acquire_try(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void acquire_wait(typename T1::ptr_t ptr) noexcept{

            T::acquire_wait(ptr);
        } 

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void acquire_release(typename T1::ptr_t ptr) noexcept{

            T::acquire_release(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto acquire_transfer_try(typename T1::ptr_t new_ptr, typename T1::ptr_t old_ptr) noexcept -> bool{

            return T::transfer_try(new_ptr, old_ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void acquire_transfer_wait(typename T1::ptr_t new_ptr, typename T1::ptr_t old_ptr) noexcept{

            T::acquire_transfer_wait(new_ptr, old_ptr);
        }
    };

    template <class T>
    struct MemoryRegionLockInterface: MemoryLockInterface<T>{

        using interface_t = MemoryRegionLockInterface<T>; 

        static auto memregion_size() noexcept -> size_t{

            return T::memregion_size();
        }
    };

    template <class T>
    struct MemoryReferenceLockInterface: MemoryLockInterface<T>{
        
        using interface_t   = MemoryReferenceLockInterface<T>;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        using ptr_t         = typename T1::ptr_t;

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto reference_try(typename T1::ptr_t ptr) noexcept -> bool{

            return T::reference_try(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void reference_wait(typename T1::ptr_t ptr) noexcept{

            T::reference_wait(ptr);
        } 

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void reference_release(typename T1::ptr_t ptr) noexcept{

            T::reference_release(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto reference_transfer_try(typename T1::ptr_t new_ptr, typename T1::ptr_t old_ptr) noexcept -> bool{

            return T::reference_transfer_try(new_ptr, old_ptr);
        } 

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void reference_transfer_wait(typename T1::ptr_t new_ptr, typename T1::ptr_t old_ptr) noexcept{

            T::reference_transfer_wait(new_ptr, old_ptr);
        }
    };

    template <class T>
    class lock_guard{

        private:

            typename dg::network_memlock::MemoryLockInterface<T>::ptr_t<> ptr; 

        public:

            using self = lock_guard;

            inline __attribute__((always_inline)) lock_guard(const dg::network_memlock::MemoryLockInterface<T>,
                                                             typename dg::network_memlock::MemoryLockInterface<T>::ptr_t<> ptr) noexcept: ptr(ptr){
                
                std::atomic_signal_fence(std::memory_order_seq_cst);
                dg::network_memlock::MemoryLockInterface<T>::acquire_wait(this->ptr);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } else{
                    std::atomic_thread_fence(std::memory_order_acquire);
                }
            }

            inline __attribute__((always_inline)) ~lock_guard() noexcept{

                if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } else{
                    std::atomic_thread_fence(std::memory_order_release);
                }

                std::atomic_signal_fence(std::memory_order_seq_cst);
                dg::network_memlock::MemoryLockInterface<T>::acquire_release(this->ptr);
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            lock_guard(const self&) = delete;
            lock_guard(self&&) = delete;

            self& operator =(const self&) = delete;
            self& operator =(self&&) = delete;
    };

    //there is not a single more confusing implementation of locks than this
    //try_guard -> .has_value() or not has_value(), default initializable
    //guard -> a mysterious default initializable datatype that does RAII (we dont know what, why, interfaces)

    template <class T>
    struct RecursiveLockResource{};

    template <class T>
    struct RecursiveLockResource<dg::network_memlock::MemoryRegionLockInterface<T>>{

        private:

            using self                  = RecursiveLockResource; 
            using id                    = self;
            using ptr_t                 = typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<>;
            using resource_t            = dg::unordered_unstable_set<ptr_t>;
            using singleton_obj_t       = std::array<resource_t, dg::network_concurrency::THREAD_COUNT>;
            using singleton_container   = stdx::singleton<id, singleton_obj_t>;
        
        public:

            static inline auto get() noexcept -> dg::unordered_unstable_set<ptr_t>&{

                return singleton_container::get()[dg::network_concurrency::this_thread_idx()];
            }
    };

    template <class T>
    auto recursive_trylock_guard(const dg::network_memlock::MemoryRegionLockInterface<T> lock_ins, typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<> ptr) noexcept{

        using memlock_ins   = dg::network_memlock::MemoryRegionLockInterface<T>;
        using lock_ptr_t    = typename memlock_ins::ptr_t<>;
        using resource_ins  = RecursiveLockResource<dg::network_memlock::MemoryRegionLockInterface<T>>;

        lock_ptr_t ptr_region   = dg::memult::region(ptr, memlock_ins::memregion_size());
        auto destructor         = [](lock_ptr_t arg) noexcept{
            resource_ins::get().erase(arg);
            memlock_ins::acquire_release(arg);
        };

        using rs_type = std::optional<dg::unique_resource<lock_ptr_t, decltype(destructor)>>;

        if (resource_ins::get().contains(ptr_region)){
            return rs_type(dg::unique_resource<lock_ptr_t, decltype(destructor)>());
        }

        if (memlock_ins::acquire_try(ptr_region)){
            resource_ins::get().insert(ptr_region);
            return rs_type(dg::unique_resource<lock_ptr_t, decltype(destructor)>(ptr_region, std::move(destructor)));
        }

        return rs_type(std::nullopt);
    }

    template <class T>
    auto recursive_lock_guard(const dg::network_memlock::MemoryRegionLockInterface<T> lock_ins, typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<> ptr) noexcept{

        using memlock_ins   = dg::network_memlock::MemoryRegionLockInterface<T>;
        using lock_ptr_t    = typename memlock_ins::ptr_t<>;
        using resource_ins  = RecursiveLockResource<dg::network_memlock::MemoryRegionLockInterface<T>>;

        lock_ptr_t ptr_region = dg::memult::region(ptr, memlock_ins::memregion_size());
        auto destructor = [](lock_ptr_t arg) noexcept{
            resource_ins::get().erase(arg);
            memlock_ins::acquire_release(arg);
        };

        if (resource_ins::get().contains(ptr_region)){
            return dg::unique_resource<lock_ptr_t, decltype(destructor)>();
        }

        memlock_ins::acquire_wait(ptr_region);
        resource_ins::get().insert(ptr_region);
        return dg::unique_resource<lock_ptr_t, decltype(destructor)>(ptr_region, std::move(destructor));
    }

    template <class T, size_t SZ>
    auto recursive_trylock_guard_array(const dg::network_memlock::MemoryRegionLockInterface<T> lock_ins,
                                       const std::array<typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<>, SZ>& lock_ptr_arr){

        static_assert(SZ != 0u);

        using lock_ptr_t        = typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<>;
        using lock_resource_t   = decltype(recursive_trylock_guard(lock_ins, lock_ptr_t{}));
        auto resource_arr       = std::array<lock_resource_t, SZ>{};

        for (size_t i = 0u; i < SZ; ++i){
            resource_arr[i] = recursive_trylock_guard(lock_ins, lock_ptr_arr[i]);

            if (!static_cast<bool>(resource_arr[i])){
                return std::optional<std::array<lock_resource_t, SZ>>(std::nullopt);
            }
        }

        return std::optional<std::array<lock_resource_t, SZ>>(std::move(resource_arr));
    }

    template <class T, class ...Args>
    auto recursive_trylock_guard_many(const dg::network_memlock::MemoryRegionLockInterface<T> lock_ins, Args... args) noexcept{

        using lock_ptr_t        = typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<>;
        auto lock_ptr_arr       = std::array<lock_ptr_t, sizeof...(Args)>{args...};

        return recursive_trylock_guard_array(lock_ins, lock_ptr_arr);
    }

    template <class T, size_t SZ>
    auto recursive_lock_guard_array(const dg::network_memlock::MemoryRegionLockInterface<T> lock_ins,
                                    const std::array<typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<>, SZ>& lock_ptr_arr){

        static_assert(SZ != 0u);

        if constexpr(SZ == 1u){
            return recursive_lock_guard(lock_ins, lock_ptr_arr[0]);
        } else{
            using waiting_lock_guard_resource_t     = decltype(recursive_lock_guard(lock_ins, lock_ptr_arr[0]));
            using spinning_lock_guard_resource_t    = decltype(recursive_trylock_guard(lock_ins, lock_ptr_arr[0]));

            auto waiting_lock_guard_resource        = waiting_lock_guard_resource_t{};
            auto spinning_lock_guard_resource_array = std::array<spinning_lock_guard_resource_t, SZ>{};
            size_t waitable_idx                     = {}; 
            bool was_thru                           = true;

            for (size_t i = 0u; i < SZ; ++i){
                spinning_lock_guard_resource_array[i] = recursive_lock_guard(lock_ins, lock_ptr_arr[i]);

                if (!spinning_lock_guard_resource_array[i].has_value()){
                    waitable_idx    = i; 
                    was_thru        = false;
                    break;
                }
            }

            if (was_thru){
                return std::make_pair(std::move(waiting_lock_guard_resource), std::move(spinning_lock_guard_resource_array));
            }

            while (true){
                waiting_lock_guard_resource         = {};
                spinning_lock_guard_resource_array  = {};
                waiting_lock_guard_resource         = recursive_lock_guard(lock_ins, lock_ptr_arr[waitable_idx]);
                was_thru                            = true; 

                for (size_t i = 0u; i < SZ; ++i){
                    if (i != waitable_idx){
                        spinning_lock_guard_resource_array[i] = recursive_trylock_guard(lock_ins, lock_ptr_arr[i]);

                        if (!spinning_lock_guard_resource_array[i].has_value()){
                            waitable_idx    = i;
                            was_thru        = false;
                            break;
                        }
                    }
                }

                if (was_thru){
                    return std::make_pair(std::move(waiting_lock_guard_resource), std::move(spinning_lock_guard_resource_array));
                }
            }
        }
    }

    template <class T, class ...Args>
    auto recursive_lock_guard_many(const dg::network_memlock::MemoryRegionLockInterface<T> lock_ins, Args... args) noexcept{

        using lock_ptr_t        = typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<>;
        auto lock_ptr_arr       = std::array<lock_ptr_t, sizeof...(Args)>{args...};

        return recursive_lock_guard_array(lock_ins, lock_ptr_arr);
    }

    template <class T, class ...Args>
    class recursive_lock_guard_many_x{

        private:

            decltype(recursive_lock_guard_many(dg::network_memlock::MemoryRegionLockInterface<T>{}, std::declval<Args>()...)) resource;

        public:

            using self = recursive_lock_guard_many_x;
            
            inline __attribute__((always_inline)) recursive_lock_guard_many_x(const dg::network_memlock::MemoryRegionLockInterface<T> ins,
                                                                              Args... args) noexcept{

                std::atomic_signal_fence(std::memory_order_seq_cst);
                this->resource = recursive_lock_guard_many(ins, std::memory_order_relaxed, args...);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } else{
                    std::atomic_thread_fence(std::memory_order_acquire);
                }       
            }

            inline __attribute__((always_inline)) ~recursive_lock_guard_many_x() noexcept{

                if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } else{
                    std::atomic_thread_fence(std::memory_order_release);
                }

                std::atomic_signal_fence(std::memory_order_seq_cst);
                this->resource = std::nullopt;
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            recursive_lock_guard_many_x(const self&) = delete;
            recursive_lock_guard_many_x(self&&) = delete;

            self& operator =(const self&) = delete;
            self& operator =(self&&) = delete;
    };
}

namespace dg::network_memlock_impl1{

    using namespace dg::network_memlock;

    template <class ID, class MemRegionSize, class PtrT = std::add_pointer_t<const void>>
    struct AtomicFlagLock{}; 

    //I was trying to get the hinge of lock + schedules
    //I was proving the notify + nodeadlocks

    //it is the problem of having at least 1 guy to continue the process after release
    //what does that even mean?

    //assume notify()
    //assume another guy got in the way before notify() + got the lock -> OK

    //assume notify() is notifying a list of subscribed subscribers
    //notify() -> subscriber, not triggered because of false -> OK, another guy got in the way
    //notify() -> subscriber, triggered because of true, guy is spinning loop -> OK, another guy got in the way (cmpexch_strong is precisely for this)
    //                                                   guy is not spinning loop -> OK, we got in 

    //notify() -> no_subscriber -> next guy's gonna read false -> OK, another guy got in the way
    //                          -> next guy's gonna read true -> OK, the lucky guy got in the way

    //is there a possiblity of wait() deadlock
    //no, why?

    //we need to look at the very important hinge, the last notify() and the wait() invoke

    //if the wait() is sequenced before the notify(), it is subscribed, then we are guaranteed to have 1 guy to continue the process (as explained above)
    //can we prove that the wait() will be woken (in the case of tentatively_freed_lock) if it is sequenced before the notify()?

    //if it is sequenced before the notify, then it is guaranteeing the subscriber list to be >= 1
    //<the not having a next guy to continue the process> (tentatively_freed_lock) only happens when the subscriber list == 0, which means the wait() has been woken
    //so there is no such case of tentatively_freed_lock + not_woke_already_subscribed

    //this is complicated 

    //if the wait() is sequenced after the notify(), then we are guaranteed to read true, as unblocked by the notifier

    //this is a very important note

    //due to technical constraints of UMA proxyspin locks (we cant find a clean implementation of that)
    //we would want to reduce the lock contention by making the memlock_region_sz == uma_region_sz
    //recall that we are not actually processing the WO
    //we are offloading that -> the asynchronous devices
    //so the pros of acquiring different parts of a region is not actually a quantifiable plus  
    //we dont want to excuse our implementation, yet we'd invest our time in improving the UMA proxy spinlock implementation should there be usecases

    template <class ID, size_t MEMREGION_SZ, class PtrT>
    struct AtomicFlagLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>: MemoryRegionLockInterface<AtomicFlagLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>>{

        public:

            using ptr_t = PtrT; 

        private:

            using self          = AtomicFlagLock;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>;;
            using uptr_t        = typename dg::ptr_info<ptr_t>::max_unsigned_t; 

            static inline std::unique_ptr<stdx::hdi_container<std::atomic_flag>[]> lck_table{};
            static inline ptr_t region_first{}; 

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                return lck_table[table_idx].value.test_and_set(std::memory_order_relaxed) == false;
            }

            static void internal_acquire_wait(size_t table_idx) noexcept{

                auto lambda = [&]() noexcept{
                    return lck_table[table_idx].value.test_and_set(std::memory_order_relaxed) == false;
                };

                while (true){
                    bool is_success = stdx::eventloop_spin_expbackoff(lambda, stdx::SPINLOCK_SIZE_MAGIC_VALUE);

                    if (is_success){
                        return;
                    }

                    lck_table[table_idx].value.wait(true, std::memory_order_relaxed);
                }
            }

            static void internal_acquire_release(size_t table_idx) noexcept{

                lck_table[table_idx].value.clear(std::memory_order_relaxed);
                lck_table[table_idx].value.notify_one();
            }

        public:

            static_assert(stdx::is_pow2(MEMREGION_SZ));

            static void init(ptr_t first, ptr_t last){ 
                
                uptr_t ufirst   = pointer_cast<uptr_t>(first);
                uptr_t ulast    = pointer_cast<uptr_t>(last);

                if (ulast < ufirst){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (ufirst % MEMREGION_SZ != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (ulast % MEMREGION_SZ != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                size_t lck_table_sz = (ulast - ufirst) / MEMREGION_SZ;
                lck_table           = std::make_unique<stdx::hdi_container<std::atomic_flag>[]>(lck_table_sz);
                region_first        = first;
                segcheck_ins::init(first, last);
            }

            static void deinit() noexcept{

                lck_table = nullptr;
            }

            static auto memregion_size() noexcept -> size_t{

                return MEMREGION_SZ;
            }

            static auto acquire_try(ptr_t ptr) noexcept -> bool{

                return internal_acquire_try(memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_wait(ptr_t ptr) noexcept{

                internal_acquire_wait(memregion_slot(segcheck_ins::access(ptr)));
            } 

            static void acquire_release(ptr_t ptr) noexcept{

                internal_acquire_release(memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return memregion_slot(segcheck_ins::access(new_ptr)) == memregion_slot(segcheck_ins::access(old_ptr));
            } 

            static void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (acquire_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                acquire_release(old_ptr);
                acquire_wait(new_ptr);
            }
    };

    template <class ID, class MemRegionSize, class PtrT = std::add_pointer_t<const void>>
    struct MtxLock{};

    template <class ID, size_t MEMREGION_SZ, class PtrT>
    struct MtxLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>: MemoryRegionLockInterface<MtxLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>>{

        public:

            using ptr_t = PtrT;

        private:

            using self          = MtxLock;
            using uptr_t        = typename dg::ptr_info<ptr_t>::max_unsigned_t;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>; 

            static inline std::unique_ptr<stdx::hdi_container<std::mutex>[]> lck_table{};
            static inline ptr_t region_first{};

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

        public:

            static void init(ptr_t first, ptr_t last){
                
                uptr_t ufirst   = pointer_cast<uptr_t>(first);
                uptr_t ulast    = pointer_cast<uptr_t>(last);

                if (ulast < ufirst){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (ufirst % MEMREGION_SZ != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (ulast % MEMREGION_SZ != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                size_t lck_table_sz     = (ulast - ufirst) / MEMREGION_SZ;
                lck_table               = std::make_unique<stdx::hdi_container<std::mutex>[]>(lck_table_sz);
                region_first            = first;
                segcheck_ins::init(first, last);
            }

            static void deinit() noexcept{

                lck_table = nullptr;
            }

            static auto memregion_size() noexcept -> size_t{

                return MEMREGION_SZ;
            }

            static auto acquire_try(ptr_t ptr) noexcept -> bool{

                return lck_table[memregion_slot(segcheck_ins::access(ptr))].value.try_lock(); //mutex is a sufficient memory barrier
            }

            static void acquire_wait(ptr_t ptr) noexcept{

                lck_table[memregion_slot(segcheck_ins::access(ptr))].value.lock();
            }

            static void acquire_release(ptr_t ptr) noexcept{

                lck_table[memregion_slot(segcheck_ins::access(ptr))].value.unlock();
            }

            static auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return memregion_slot(segcheck_ins::access(new_ptr)) == memregion_slot(segcheck_ins::access(old_ptr));
            }

            static void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (acquire_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                acquire_release(old_ptr);
                acquire_wait(new_ptr);
            }
    };

    template <class ID, class MemRegionSize, class PtrT = std::add_pointer_t<const void>>
    struct AtomicReferenceLock{};

    template <class ID, size_t MEMREGION_SZ, class PtrT>
    struct AtomicReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>: MemoryReferenceLockInterface<AtomicReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>>{

        public:

            using ptr_t = PtrT;

        private:

            using atomic_lock_t = uint64_t; 
            using self          = AtomicReferenceLock;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>; 
            using uptr_t        = typename dg::ptr_info<ptr_t>::max_unsigned_t;

            static inline constexpr atomic_lock_t MEMREGION_EMP_STATE = 0u;
            static inline constexpr atomic_lock_t MEMREGION_ACQ_STATE = std::numeric_limits<atomic_lock_t>::max();

            static inline std::unique_ptr<stdx::hdi_container<std::atomic<atomic_lock_t>>[]> lck_table{};    
            static inline ptr_t region_first{};

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                atomic_lock_t expected = MEMREGION_EMP_STATE;
                bool rs = lck_table[table_idx].value.compare_exchange_weak(expected, MEMREGION_ACQ_STATE);

                return rs;
            } 

            static void internal_acquire_wait(size_t table_idx) noexcept{

                auto lambda = [&]() noexcept{
                    atomic_lock_t expected = MEMREGION_EMP_STATE;
                    return lck_table[table_idx].value.compare_exchange_weak(expected, MEMREGION_ACQ_STATE, std::memory_order_relaxed);
                };

                stdx::eventloop_spin_expbackoff(lambda);
            }

            static void internal_acquire_release(size_t table_idx) noexcept{

                lck_table[table_idx].value.exchange(MEMREGION_EMP_STATE, std::memory_order_relaxed);
            }

            static auto internal_reference_try(size_t table_idx) noexcept -> bool{

                atomic_lock_t cur_state  = lck_table[table_idx].value.load(std::memory_order_relaxed);

                if (cur_state == MEMREGION_ACQ_STATE){
                    return false;
                }

                atomic_lock_t nxt_state  = cur_state + 1;
                bool rs = lck_table[table_idx].value.compare_exchange_weak(cur_state, nxt_state, std::memory_order_relaxed);

                return rs;
            }

            static void internal_reference_wait(size_t table_idx) noexcept{

                auto lambda = [&]() noexcept{
                    atomic_lock_t cur_state  = lck_table[table_idx].value.load(std::memory_order_relaxed);
                    atomic_lock_t nxt_state  = cur_state + 1;

                    return cur_state != MEMREGION_ACQ_STATE && lck_table[table_idx].value.compare_exchange_weak(cur_state, nxt_state, std::memory_order_relaxed);
                };

                stdx::eventloop_spin_expbackoff(lambda);
            }

            static void internal_reference_release(size_t table_idx) noexcept{

                lck_table[table_idx].value.fetch_sub(1, std::memory_order_relaxed);
            }

        public:

            static_assert(stdx::is_pow2(MEMREGION_SZ));

            static void init(ptr_t first, ptr_t last){
                
                uptr_t ufirst   = pointer_cast<uptr_t>(first);
                uptr_t ulast    = pointer_cast<uptr_t>(last);

                if (ulast < ufirst){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                } 

                if (ufirst % MEMREGION_SZ != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (ulast % MEMREGION_SZ != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                size_t lck_table_sz = (ulast - ufirst) / MEMREGION_SZ;
                lck_table           = std::make_unique<stdx::hdi_container<std::atomic<atomic_lock_t>>[]>(lck_table_sz);
                region_first        = first;

                for (size_t i = 0u; i < lck_table_sz; ++i){
                    lck_table[i].value = MEMREGION_EMP_STATE;
                }

                segcheck_ins::init(first, last);
            }

            static void deinit() noexcept{

                lck_table = nullptr;    
            }

            static auto transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return memregion_slot(segcheck_ins::access(new_ptr)) == memregion_slot(segcheck_ins::access(old_ptr));
            }

            static auto acquire_try(ptr_t ptr) noexcept -> bool{

                return internal_acquire_try(memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_wait(ptr_t ptr) noexcept{

                internal_acquire_wait(memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_release(ptr_t ptr) noexcept{

                internal_acquire_release(memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return transfer_try(new_ptr, old_ptr);
            } 

            static void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (acquire_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                acquire_release(old_ptr);
                acquire_wait(new_ptr);
            }

            static auto reference_try(ptr_t ptr, std::memory_order success = std::memory_order_seq_cst) noexcept -> bool{

                return internal_reference_try(memregion_slot(segcheck_ins::access(ptr)), success);
            }

            static void reference_wait(ptr_t ptr) noexcept{

                internal_reference_wait(memregion_slot(segcheck_ins::access(ptr)));
            } 

            static void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto reference_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return transfer_try(new_ptr, old_ptr);
            } 

            static void reference_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (reference_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                reference_release(old_ptr);
                reference_wait(new_ptr);
            }
    };

    template <class ID, class MemRegionSize, class MutexT = std::atomic_flag, class PtrT = std::add_pointer_t<const void>>
    struct MtxReferenceLock{};    

    template <class ID, size_t MEMREGION_SZ, class MutexT, class PtrT>
    struct MtxReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, MutexT, PtrT>: MemoryReferenceLockInterface<MtxReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, MutexT, PtrT>>{

        public:

            using ptr_t = PtrT; 

        private:

            using refcount_t    = uint64_t;
            using self          = MtxReferenceLock;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>;
            using uptr_t        = typename dg::ptr_info<ptr_t>::max_unsigned_t;

            static constexpr inline refcount_t REFERENCE_EMPTY_STATE    = 0u;
            static constexpr inline refcount_t REFERENCE_ACQUIRED_STATE = std::numeric_limits<refcount_t>::max();

            struct LockUnit{
                MutexT lck;
                refcount_t refcount;
            };

            static inline std::unique_ptr<stdx::hdi_container<LockUnit>[]> lck_table{};
            static inline ptr_t region_first{}; 

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                stdx::xlock_guard<MutexT> lck_grd(lck_table[table_idx].value.lck);
                
                if (lck_table[table_idx].value.refcount != REFERENCE_EMPTY_STATE){
                    return false;
                }

                lck_table[table_idx].value.refcount = REFERENCE_ACQUIRED_STATE;
                return true;
            }

            static void internal_acquire_wait(size_t table_idx) noexcept{

                while (!internal_acquire_try(table_idx)){}
            }

            static void internal_acquire_release(size_t table_idx) noexcept{

                stdx::xlock_guard<MutexT> lck_grd(lck_table[table_idx].value.lck);
                lck_table[table_idx].refcount = REFERENCE_EMPTY_STATE;
            }

            static auto internal_reference_try(size_t table_idx) noexcept -> bool{

                stdx::xlock_guard<MutexT> lck_grd(lck_table[table_idx].value.lck);
                
                if (lck_table[table_idx].value.refcount == REFERENCE_ACQUIRED_STATE){
                    return false;
                }

                ++lck_table[table_idx].value.refcount;
                return true;
            }

            static auto internal_reference_wait(size_t table_idx) noexcept{

                while (!internal_reference_try(table_idx)){}
            }

            static auto internal_reference_release(size_t table_idx) noexcept{

                stdx::xlock_guard<MutexT> lck_grd(lck_table[table_idx].value.lck);
                --lck_table[table_idx].value.refcount;
            }

        public:

            static void init(ptr_t first, ptr_t last){ 
                
                uptr_t ufirst   = pointer_cast<uptr_t>(first);
                uptr_t ulast    = pointer_cast<uptr_t>(last);

                if (ulast < ufirst){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (ufirst % MEMREGION_SZ != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (ulast % MEMREGION_SZ != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                size_t lck_table_sz = (ulast - ufirst) / MEMREGION_SZ;
                lck_table           = std::make_unique<stdx::hdi_container<LockUnit>[]>(lck_table_sz);
                region_first        = first;
                
                for (size_t i = 0u; i < lck_table_sz; ++i){
                    lck_table[i].value.refcount = REFERENCE_EMPTY_STATE;
                }

                segcheck_ins::init(first, last);
            }

            static void deinit() noexcept{

                lck_table = nullptr;
            }

            static auto transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{
                
                return memregion_slot(segcheck_ins::access(new_ptr)) == memregion_slot(segcheck_ins::access(old_ptr));
            }

            static auto acquire_try(ptr_t ptr) noexcept -> bool{

                return internal_acquire_try(memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_wait(ptr_t ptr) noexcept{

                internal_acquire_wait(memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_release(ptr_t ptr) noexcept{

                internal_acquire_release(memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return transfer_try(new_ptr, old_ptr);
            } 

            static void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (acquire_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                acquire_release(old_ptr);
                acquire_wait(new_ptr);
            }

            static auto reference_try(ptr_t ptr) noexcept -> bool{

                return internal_reference_try(memregion_slot(segcheck_ins::access(ptr)));
            }

            static void reference_wait(ptr_t ptr) noexcept{

                internal_reference_wait(memregion_slot(segcheck_ins::access(ptr)));
            } 

            static void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto reference_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return transfer_try(new_ptr, old_ptr);
            }

            static void reference_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (reference_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                reference_release(old_ptr);
                reference_wait(new_ptr);
            }
    };

    static inline constexpr bool IS_ATOMIC_OPERATION_PREFERRED = true; 
    
    template <class ID, class MemRegionSize, class PtrT = std::add_pointer_t<const void>>
    using Lock = std::conditional_t<IS_ATOMIC_OPERATION_PREFERRED,
                                    AtomicFlagLock<ID, MemRegionSize, PtrT>,
                                    MtxLock<ID, MemRegionSize, PtrT>>;
    

    template <class ID, class MemRegionSize, class MutexT = std::atomic_flag, class PtrT = std::add_pointer_t<const void>>
    using ReferenceLock = std::conditional_t<IS_ATOMIC_OPERATION_PREFERRED,
                                             AtomicReferenceLock<ID, MemRegionSize, PtrT>,
                                             MtxReferenceLock<ID, MemRegionSize, MutexT, PtrT>>;

}

#endif 