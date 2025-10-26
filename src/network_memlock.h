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
        static auto acquire_try_strong(typename T1::ptr_t ptr) noexcept -> bool{

            return T::acquire_try_strong(ptr);
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
        static void acquire_waitnolock(typename T1::ptr_t ptr) noexcept{

            T::acquire_waitnolock(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static void acquire_waitnolock_release_responsibility(typename T1::ptr_t ptr) noexcept{

            T::acquire_waitnolock_release_responsibility(ptr);
        }

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto acquire_transfer_try(typename T1::ptr_t new_ptr, typename T1::ptr_t old_ptr) noexcept -> bool{

            return T::acquire_transfer_try(new_ptr, old_ptr);
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

    template <class T, size_t SZ>
    constexpr auto sort_ptr_array(const std::array<T, SZ>& inp) noexcept -> std::array<T, SZ>{

        // static_assert(dg::ptr_info<T>::is_pointer);

        std::array<T, SZ> rs = inp;
        // std::sort(rs.begin(), rs.end());

        return rs;
    }

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

        if (memlock_ins::acquire_try_strong(ptr_region)){
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

        lock_ptr_t ptr_region   = dg::memult::region(ptr, memlock_ins::memregion_size());
        auto destructor         = [](lock_ptr_t arg) noexcept{
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
                                       const std::array<typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<>, SZ>& arg_lock_ptr_arr){

        static_assert(SZ != 0u);

        using lock_ptr_t        = typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<>;
        using lock_resource_t   = decltype(recursive_trylock_guard(lock_ins, lock_ptr_t{}));
        auto resource_arr       = std::array<lock_resource_t, SZ>{};
        auto lock_ptr_arr       = sort_ptr_array(arg_lock_ptr_arr);

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
                                    const std::array<typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t<>, SZ>& arg_lock_ptr_arr){

        static_assert(SZ != 0u);

        if constexpr(SZ == 1u){
            return recursive_lock_guard(lock_ins, arg_lock_ptr_arr[0]);
        } else{
            using try_lock_guard_resource_t                                 = std::variant<decltype(recursive_trylock_guard(lock_ins, arg_lock_ptr_arr[0])), decltype(recursive_lock_guard(lock_ins, arg_lock_ptr_arr[0]))>;
            constexpr size_t INNER_LOOP_BUSY_WAIT_MAX_EXPONENT              = 5u;

            constexpr size_t CYCLIC_EXPBACKOFF_SPIN_SZ                      = 16u;
            constexpr std::chrono::nanoseconds CYCLIC_EXPBACKOFF_WAIT_TIME  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds(100));
            constexpr size_t CYCLIC_EXPBACKOFF_REVOLUTION                   = 8u;

            auto lock_ptr_arr                                               = sort_ptr_array(arg_lock_ptr_arr); 
            std::optional<size_t> wait_idx                                  = std::nullopt;
            std::array<try_lock_guard_resource_t, SZ> rs;

            auto task = [&]() noexcept{
                rs                                      = {};
                bool was_thru                           = true;
                std::optional<size_t> responsible_idx   = std::nullopt; 
                size_t retry_exponent                   = 0u; 

                if (wait_idx.has_value()){
                    dg::network_memlock::MemoryRegionLockInterface<T>::acquire_waitnolock(lock_ptr_arr[wait_idx.value()]);
                    responsible_idx = wait_idx;
                    wait_idx        = std::nullopt;
                }

                for (size_t i = 0u; i < SZ; ++i){
                    if (i != 0u && lock_ptr_arr[i] != lock_ptr_arr[i - 1]){ //bad logics, people dont like this for etc. reasons
                        retry_exponent += 1u; //border index detection
                    }

                    auto inner_loop_task = [&]() noexcept{
                        rs[i] = recursive_trylock_guard(lock_ins, lock_ptr_arr[i]);
                        return std::get<0>(rs[i]).has_value();
                    };

                    size_t retry_sz = size_t{1} << std::min(retry_exponent, INNER_LOOP_BUSY_WAIT_MAX_EXPONENT);
                    stdx::eventloop_competitive_spin(inner_loop_task, retry_sz);

                    if (responsible_idx.has_value()){
                        if (responsible_idx.value() == i){
                            responsible_idx = std::nullopt; //responsibility of reversing the acquire_waitnolock transferred -> the rs[i], whether that was thru or not thru, we need to guarantee acquire_try is strong, this is very important
                        }
                    }

                    if (!std::get<0>(rs[i]).has_value()){
                        wait_idx    = i; //acquire_try strong acquisition does not thru, wait_idx == i, hint the next iteration of the waiting idx 
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
                        dg::network_memlock::MemoryRegionLockInterface<T>::acquire_waitnolock_release_responsibility(lock_ptr_arr[responsible_idx.value()]); //we are still responsible for the waitnolock, we need to release the responsibility
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
                rs[i] = recursive_lock_guard(lock_ins, lock_ptr_arr[i]);
            }

            return rs;
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
                this->resource = recursive_lock_guard_many(ins, args...);
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

    template <class ID, size_t MEMREGION_SZ, class PtrT>
    struct AtomicFlagLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>: MemoryRegionLockInterface<AtomicFlagLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>>{

        public:

            using ptr_t = PtrT; 

        private:

            using self          = AtomicFlagLock;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>;
            using uptr_t        = typename dg::ptr_info<ptr_t>::max_unsigned_t; 

            static inline std::unique_ptr<stdx::hdi_container<std::atomic_flag>[]> lck_table;
            static inline ptr_t region_first;

            static inline constexpr size_t FOREHEAD_SPIN_SIZE                       = 16u;
            static inline constexpr std::chrono::nanoseconds FOREHEAD_SPIN_PERIOD   = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds(10));

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                return lck_table[table_idx].value.test_and_set(std::memory_order_relaxed) == false;
            }

            static auto internal_acquire_try_strong(size_t table_idx) noexcept -> bool{

                return lck_table[table_idx].value.test_and_set(std::memory_order_seq_cst) == false;
            }

            static void internal_acquire_wait(size_t table_idx) noexcept{

                auto lambda = [&]() noexcept{
                    return lck_table[table_idx].value.test_and_set(std::memory_order_relaxed) == false;
                };

                bool is_success = stdx::eventloop_expbackoff_spin(lambda, FOREHEAD_SPIN_SIZE, FOREHEAD_SPIN_PERIOD);

                if (is_success){
                    return;
                }

                while (true){
                    is_success = lambda();

                    if (is_success){
                        return;
                    }

                    lck_table[table_idx].value.wait(true, std::memory_order_relaxed);
                }
            }

            static void internal_acquire_waitnolock(size_t table_idx) noexcept{

                lck_table[table_idx].value.wait(true, std::memory_order_relaxed);
            }

            static void internal_acquire_waitnolock_release_responsibility(size_t table_idx) noexcept{

                lck_table[table_idx].value.notify_one();
            }

            static void internal_acquire_release(size_t table_idx) noexcept{

                lck_table[table_idx].value.clear(std::memory_order_relaxed);
                lck_table[table_idx].value.notify_one();
            }

        public:

            static_assert(stdx::is_pow2(MEMREGION_SZ));

            static void init(ptr_t first, ptr_t last){ 

                uptr_t ufirst   = dg::pointer_cast<uptr_t>(first);
                uptr_t ulast    = dg::pointer_cast<uptr_t>(last);

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

                for (size_t i = 0u; i < lck_table_sz; ++i){
                    lck_table[i].value.clear(std::memory_order_relaxed);
                }

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

                return self::internal_acquire_try(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto acquire_try_strong(ptr_t ptr) noexcept -> bool{

                return self::internal_acquire_try_strong(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_wait(ptr_t ptr) noexcept{

                self::internal_acquire_wait(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_waitnolock(ptr_t ptr) noexcept{

                self::internal_acquire_waitnolock(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_waitnolock_release_responsibility(ptr_t ptr) noexcept{

                self::internal_acquire_waitnolock_release_responsibility(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_release(ptr_t ptr) noexcept{

                self::internal_acquire_release(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return self::memregion_slot(segcheck_ins::access(new_ptr)) == self::memregion_slot(segcheck_ins::access(old_ptr));
            } 

            static void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (self::acquire_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                self::acquire_release(old_ptr);
                self::acquire_wait(new_ptr);
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

            static inline constexpr atomic_lock_t MEMREGION_EMP_STATE                   = 0u;
            static inline constexpr atomic_lock_t MEMREGION_ACQ_STATE                   = std::numeric_limits<atomic_lock_t>::max();
            static inline constexpr atomic_lock_t MEMREGION_MID_STATE                   = std::numeric_limits<atomic_lock_t>::max() - 1u;

            static inline constexpr size_t FOREHEAD_SPIN_SZ                             = 16u;
            static inline constexpr std::chrono::nanoseconds FOREHEAD_SPIN_PERIOD       = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds(10));

            static inline constexpr size_t COMPETITIVE_SPIN_SZ                          = 32u;
            static inline constexpr std::chrono::nanoseconds COMPETITIVE_SPIN_PERIOD    = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds(10)); 

            static inline std::unique_ptr<stdx::hdi_container<std::atomic<atomic_lock_t>>[]> lck_table;
            static inline std::unique_ptr<stdx::hdi_container<std::atomic_flag>[]> acquirability_table;
            static inline std::unique_ptr<stdx::hdi_container<std::atomic_flag>[]> referenceability_table;
            static inline ptr_t region_first;

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                atomic_lock_t expected = MEMREGION_EMP_STATE;
                bool rs = lck_table[table_idx].value.compare_exchange_strong(expected, MEMREGION_ACQ_STATE);

                if (rs){
                    acquirability_table[table_idx].value.clear(std::memory_order_relaxed);
                    referenceability_table[table_idx].value.clear(std::memory_order_relaxed);
                }

                return rs;
            }

            static auto internal_acquire_try_strong(size_t table_idx) noexcept -> bool{

                return self::internal_acquire_try(table_idx);
            }

            static void internal_acquire_wait(size_t table_idx) noexcept{

                auto lambda     = [&]() noexcept{
                    return self::internal_acquire_try(table_idx);
                };

                bool was_thru   = stdx::eventloop_expbackoff_spin(lambda, FOREHEAD_SPIN_SZ, FOREHEAD_SPIN_PERIOD);

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
            }

            static void internal_acquire_waitnolock(size_t table_idx) noexcept{

                acquirability_table[table_idx].value.wait(false, std::memory_order_relaxed);
            }

            static void internal_acquire_waitnolock_release_responsibility(size_t table_idx) noexcept{

                acquirability_table[table_idx].value.notify_one();
            }

            static void internal_acquire_release(size_t table_idx) noexcept{

                acquirability_table[table_idx].value.test_and_set(std::memory_order_relaxed);
                referenceability_table[table_idx].value.test_and_set(std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                lck_table[table_idx].value.exchange(MEMREGION_EMP_STATE, std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                acquirability_table[table_idx].value.notify_one();
                referenceability_table[table_idx].value.notify_all();

                std::atomic_signal_fence(std::memory_order_seq_cst); //not necessary
            }

            static auto internal_reference_try(size_t table_idx) noexcept -> bool{

                bool was_referenced = {}; 

                auto task = [&]() noexcept{
                    atomic_lock_t cur_state  = lck_table[table_idx].value.load(std::memory_order_relaxed);

                    if (cur_state == MEMREGION_ACQ_STATE){
                        was_referenced = false;
                        return true;
                    }

                    if (cur_state == MEMREGION_MID_STATE){
                        return false;
                    }

                    atomic_lock_t nxt_state  = cur_state + 1;
                    bool rs = lck_table[table_idx].value.compare_exchange_strong(cur_state, nxt_state, std::memory_order_relaxed);

                    if (!rs){
                        return false;
                    }

                    was_referenced = true;
                    acquirability_table[table_idx].value.clear(std::memory_order_relaxed);

                    return true;
                };

                stdx::eventloop_cyclic_expbackoff_spin(task, COMPETITIVE_SPIN_SZ, COMPETITIVE_SPIN_PERIOD);
                return was_referenced;
            }

            static void internal_reference_wait(size_t table_idx) noexcept{

                auto lambda = [&]() noexcept{
                    return self::internal_reference_try(table_idx);
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

                    referenceability_table[table_idx].value.wait(false, std::memory_order_relaxed);
                }
            }

            static void internal_reference_release(size_t table_idx) noexcept{

                atomic_lock_t old_value = {};

                auto lambda = [&]() noexcept{
                    old_value = lck_table[table_idx].value.exchange(MEMREGION_MID_STATE);
                    return old_value != MEMREGION_MID_STATE;
                };

                stdx::eventloop_cyclic_expbackoff_spin(lambda, COMPETITIVE_SPIN_SZ, COMPETITIVE_SPIN_PERIOD);

                std::atomic_signal_fence(std::memory_order_seq_cst);
                acquirability_table[table_idx].value.test_and_set(std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                lck_table[table_idx].value.exchange(old_value - 1u, std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                acquirability_table[table_idx].value.notify_one();
                std::atomic_signal_fence(std::memory_order_seq_cst); //not necessary
            }

        public:

            static_assert(stdx::is_pow2(MEMREGION_SZ));

            static void init(ptr_t first, ptr_t last){
                
                uptr_t ufirst   = dg::pointer_cast<uptr_t>(first);
                uptr_t ulast    = dg::pointer_cast<uptr_t>(last);

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
                lck_table               = std::make_unique<stdx::hdi_container<std::atomic<atomic_lock_t>>[]>(lck_table_sz);
                acquirability_table     = std::make_unique<stdx::hdi_container<std::atomic_flag>[]>(lck_table_sz);
                referenceability_table  = std::make_unique<stdx::hdi_container<std::atomic_flag>[]>(lck_table_sz);
                region_first            = first;

                for (size_t i = 0u; i < lck_table_sz; ++i){
                    lck_table[i].value = MEMREGION_EMP_STATE;
                    acquirability_table[i].value.test_and_set(std::memory_order_seq_cst);
                    referenceability_table[i].value.test_and_set(std::memory_order_seq_cst);
                }

                segcheck_ins::init(first, last);
            }

            static void deinit() noexcept{

                lck_table = nullptr;    
            }

            static auto transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return self::memregion_slot(segcheck_ins::access(new_ptr)) == self::memregion_slot(segcheck_ins::access(old_ptr));
            }

            static auto acquire_try(ptr_t ptr) noexcept -> bool{

                return self::internal_acquire_try(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto acquire_try_strong(ptr_t ptr) noexcept -> bool{

                return self::internal_acquire_try_strong(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_wait(ptr_t ptr) noexcept{

                self::internal_acquire_wait(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_release(ptr_t ptr) noexcept{

                self::internal_acquire_release(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static void acquire_waitnolock(ptr_t ptr) noexcept{

                self::internal_acquire_waitnolock(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return self::transfer_try(new_ptr, old_ptr);
            } 

            static void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (self::acquire_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                self::acquire_release(old_ptr);
                self::acquire_wait(new_ptr);
            }

            static auto reference_try(ptr_t ptr) noexcept -> bool{

                return self::internal_reference_try(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static void reference_wait(ptr_t ptr) noexcept{

                self::internal_reference_wait(self::memregion_slot(segcheck_ins::access(ptr)));
            } 

            static void reference_release(ptr_t ptr) noexcept{

                self::internal_reference_release(self::memregion_slot(segcheck_ins::access(ptr)));
            }

            static auto reference_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return self::transfer_try(new_ptr, old_ptr);
            } 

            static void reference_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (self::reference_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                self::reference_release(old_ptr);
                self::reference_wait(new_ptr);
            }
    };
    
    template <class ID, class MemRegionSize, class PtrT = std::add_pointer_t<const void>>
    using Lock = AtomicFlagLock<ID, MemRegionSize, PtrT>;

    template <class ID, class MemRegionSize, class MutexT = std::atomic_flag, class PtrT = std::add_pointer_t<const void>>
    using ReferenceLock = AtomicReferenceLock<ID, MemRegionSize, PtrT>;

}

#endif 
