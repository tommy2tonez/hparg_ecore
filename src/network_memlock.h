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
#include "network_std_container.h"
#include "stdx.h"

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
        static auto acquire_wait(typename T1::ptr_t ptr) noexcept{

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

        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static auto transfer_try(typename T1::ptr_t new_ptr, typename T1::ptr_t old_ptr) noexcept -> bool{
            
            return T::transfer_try(new_ptr, old_ptr);
        }
    };

    template <class T, class ptr_t>
    auto lock_guard(const dg::network_memlock::MemoryLockInterface<T>, ptr_t ptr) noexcept{

        using memlock_ins   = dg::network_memlock::MemoryLockInterface<T>;
        using lock_ptr_t    = typename memlock_ins::ptr_t; 
        static_assert(std::is_same_v<lock_ptr_t, ptr_t>);

        static int i        = 0;
        auto destructor     = [=](int *) noexcept{
            memlock_ins::acquire_release(ptr);
        };

        memlock_ins::acquire_wait(ptr);
        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

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
    
    template <class T, class ptr_t>
    auto recursive_trylock_guard(const dg::network_memlock::MemoryRegionLockInterface<T> lock_ins, ptr_t ptr) noexcept{
        
        using memlock_ins   = dg::network_memlock::MemoryRegionLockInterface<T>;
        using lock_ptr_t    = typename memlock_ins::ptr_t<>;
        using resource_ins  = RecursiveLockResource<dg::network_memlock::MemoryRegionLockInterface<T>>;
        static_assert(std::is_same_v<lock_ptr_t, ptr_t>);

        dg::unordered_unstable_set<lock_ptr_t> * ptr_set    = &resource_ins::get();
        lock_ptr_t ptr_region                               = dg::memult::region(ptr, memlock_ins::memregion_size()); 
        bool responsibility_flag                            = {};
        bool try_success_flag                               = {};

        if (ptr_set->find(ptr_region) == ptr_set->end()){
            if (memlock_ins::acquire_try(ptr_region)){
                responsibility_flag = true;
                try_success_flag    = true;
                ptr_set->insert(ptr_region);
            } else{
                responsibility_flag = false;
                try_success_flag    = false;
            }
        } else{
            responsibility_flag = false;
            try_success_flag    = true;
        }

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            if (responsibility_flag){
                ptr_set->erase(ptr_region);
                memlock_ins::acquire_release(ptr_region);
            }
        };
        
        if (try_success_flag){
            return std::unique_ptr<int, decltype(destructor)>(&i, std::move(destructor)); 
        }

        return std::unique_ptr<int, decltype(destructor)>(nullptr, std::move(destructor));
    }

    template <class T, class ptr_t>
    auto recursive_lock_guard(const dg::network_memlock::MemoryRegionLockInterface<T> lock_ins, ptr_t ptr) noexcept{

        while (true){
            if (auto rs = recursive_trylock_guard(lock_ins, ptr); static_cast<bool>(rs)){
                return rs;
            }
        }
    }

    template <class T, class ...Args>
    auto recursive_lock_guard_many(const dg::network_memlock::MemoryRegionLockInterface<T> lock_ins, Args... args) noexcept{

        auto args_tup   = std::make_tuple(args...);
        auto try_lambda = [=]<class Self, size_t IDX>(Self& self, const std::integral_constant<size_t, IDX>){
            if constexpr(IDX == sizeof...(Args)){
                return bool{true};
            } else{                
                auto cur_lck            = recursive_trylock_guard(lock_ins, std::get<IDX>(args_tup));
                using successor_ret_t   = decltype(self(self, std::integral_constant<size_t, IDX + 1>{}));
                using lck_t             = decltype(cur_lck);
                using ret_t             = std::pair<lck_t, successor_ret_t>;
                using opt_ret_t         = std::optional<ret_t>;

                if (!static_cast<bool>(cur_lck)){
                    return opt_ret_t{std::nullopt};
                } else{
                    auto successor_rs = self(self, std::integral_constant<size_t, IDX + 1>{});
                    
                    if (!static_cast<bool>(successor_rs)){
                        return opt_ret_t{std::nullopt};
                    } else{
                        return opt_ret_t{ret_t{std::move(cur_lck), std::move(successor_rs)}};
                    }
                }
            }
        };

        while (true){
            if (auto rs = try_lambda(try_lambda, std::integral_constant<size_t, 0u>{}); static_cast<bool>(rs)){
                return rs;
            }
        }
    }
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
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>;;
            using uptr_t        = typename dg::ptr_info<ptr_t>::max_unsigned_t; 

            static inline std::unique_ptr<std::atomic_flag[]> lck_table{};
            static inline ptr_t region_first{}; 

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                bool rs = lck_table[table_idx].test_and_set(std::memory_order_acquire);
                stdx::atomic_optional_thread_fence();
                return rs;
            }

            static void internal_acquire_wait(size_t table_idx) noexcept{

                while (!internal_acquire_try(table_idx)){}
            }

            static void internal_acquire_release(size_t table_idx) noexcept{
                
                stdx::atomic_optional_thread_fence();
                lck_table[table_idx].clear(std::memory_order_release);
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
                lck_table           = std::make_unique<std::atomic_flag[]>(lck_table_sz);
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

            static inline std::unique_ptr<std::mutex[]> lck_table{};
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
                lck_table               = std::make_unique<std::mutex[]>(lck_table_sz);
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

                bool rs = lck_table[memregion_slot(segcheck_ins::access(ptr))].try_lock();
                stdx::atomic_optional_thread_fence();
                return rs;
            }

            static void acquire_wait(ptr_t ptr) noexcept{

                lck_table[memregion_slot(segcheck_ins::access(ptr))].lock();
                stdx::atomic_optional_thread_fence();
            }
            
            static void acquire_release(ptr_t ptr) noexcept{

                stdx::atomic_optional_thread_fence();
                lck_table[memregion_slot(segcheck_ins::access(ptr))].unlock();
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

            static inline std::unique_ptr<std::atomic<atomic_lock_t>[]> lck_table{};    
            static inline ptr_t region_first{};

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{
                
                atomic_lock_t expected = MEMREGION_EMP_STATE;
                bool rs = lck_table[table_idx].compare_exchange_weak(expected, MEMREGION_ACQ_STATE, std::memory_order_acq_rel);
                stdx::atomic_optional_thread_fence();
                
                return rs;
            } 

            static void internal_acquire_wait(size_t table_idx) noexcept{

                while (!internal_acquire_try(table_idx)){}
            }

            static void internal_acquire_release(size_t table_idx) noexcept{

                stdx::atomic_optional_thread_fence();
                lck_table[table_idx].exchange(MEMREGION_EMP_STATE, std::memory_order_release);
            }

            static auto internal_reference_try(size_t table_idx) noexcept -> bool{

                atomic_lock_t cur_state  = lck_table[table_idx].load(std::memory_order_relaxed);

                if (cur_state == MEMREGION_ACQ_STATE){
                    return false;
                }

                atomic_lock_t nxt_state  = cur_state + 1;
                bool rs = lck_table[table_idx].compare_exchange_weak(cur_state, nxt_state, std::memory_order_acq_rel);
                stdx::atomic_optional_thread_fence();

                return rs;
            }

            static void internal_reference_wait(size_t table_idx) noexcept{

                while (!internal_reference_try(table_idx)){}
            }

            static void internal_reference_release(size_t table_idx) noexcept{

                stdx::atomic_optional_thread_fence();
                lck_table[table_idx].fetch_sub(1, std::memory_order_release);
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
                lck_table           = std::make_unique<std::atomic<atomic_lock_t>[]>(lck_table_sz);
                region_first        = first;

                for (size_t i = 0u; i < lck_table_sz; ++i){
                    lck_table[i] = MEMREGION_EMP_STATE;
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

            static inline std::unique_ptr<LockUnit[]> lck_table{};
            static inline ptr_t region_first{}; 

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(region_first, ptr) / MEMREGION_SZ;
            }

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                auto lck_grd = stdx::lock_guard(lck_table[table_idx].lck);
                
                if (lck_table[table_idx].refcount != REFERENCE_EMPTY_STATE){
                    return false;
                }

                lck_table[table_idx].refcount = REFERENCE_ACQUIRED_STATE;
                return true;
            }

            static auto internal_acquire_wait(size_t table_idx) noexcept{

                while (!internal_acquire_try(table_idx)){}
            }

            static auto internal_acquire_release(size_t table_idx) noexcept{

                auto lck_grd = stdx::lock_guard(lck_table[table_idx].lck);
                lck_table[table_idx].refcount = REFERENCE_EMPTY_STATE;
            }

            static auto internal_reference_try(size_t table_idx) noexcept -> bool{

                auto lck_grd = stdx::lock_guard(lck_table[table_idx].lck);
                
                if (lck_table[table_idx].refcount == REFERENCE_ACQUIRED_STATE){
                    return false;
                }

                ++lck_table[table_idx].refcount;
                return true;
            }

            static auto internal_reference_wait(size_t table_idx) noexcept{

                while (!internal_reference_try(table_idx)){}
            }

            static auto internal_reference_release(size_t table_idx) noexcept{

                auto lck_grd = stdx::lock_guard(lck_table[table_idx].lck);
                --lck_table[table_idx].refcount;
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
                lck_table           = std::make_unique<LockUnit[]>(lck_table_sz);
                region_first        = first;
                
                for (size_t i = 0u; i < lck_table_sz; ++i){
                    lck_table[i].refcount = REFERENCE_EMPTY_STATE;
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

            static auto acquire_wait(ptr_t ptr) noexcept{

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