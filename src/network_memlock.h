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
#include "network_log.h" 
#include "network_exception.h"
#include "network_segcheck_bound.h"
#include <vector>
#include <unordered_set>

namespace dg::network_memlock{
    
    template <class T>
    struct MemoryLockInterface{

        using interface_t   = MemoryLockInterface<T>;
        using ptr_t         = typename T::ptr_t;
        static_assert(dg::is_ptr_v<ptr_t>); 

        static auto acquire_try(ptr_t ptr) noexcept -> bool{

            return T::acquire_try(ptr);
        }

        static auto acquire_wait(ptr_t ptr) noexcept{

            T::acquire_wait(ptr);
        } 

        static void acquire_release(ptr_t ptr) noexcept{

            T::acquire_release(ptr);
        }

        static auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

            return T::transfer_try(new_ptr, old_ptr);
        } 

        static void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

            T::acquire_transfer_wait(new_ptr, old_ptr);
        }
    };

    template <class T>
    struct MemoryRegionLockInterface: MemoryLockInterface<T>{

        static auto memregion_size() noexcept -> size_t{

            return T::memregion_size();
        }
    };

    template <class T>
    struct MemoryReferenceInterface{

        using interface_t   = MemoryReferenceInterface<T>; 
        using ptr_t         = typename T::ptr_t;
        static_assert(dg::is_ptr_v<ptr_t>); 

        static auto reference_try(ptr_t ptr) noexcept -> bool{

            return T::reference_try(ptr);
        }

        static void reference_wait(ptr_t ptr) noexcept {

            T::reference_wait(ptr);
        } 

        static void reference_release(ptr_t ptr) noexcept{

            T::reference_release(ptr);
        }

        static auto reference_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

            return T::reference_transfer_try(new_ptr, old_ptr);
        } 

        static void reference_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

            T::reference_transfer_wait(new_ptr, old_ptr);
        }
    };

    template <class T>
    struct MemoryReferenceLockInterface: MemoryLockInterface<T>, MemoryReferenceInterface<T>{
        
        using interface_t   = MemoryReferenceLockInterface<T>;
        using ptr_t         = typename T::ptr_t;

        static auto transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{
            
            return T::transfer_try(new_ptr, old_ptr);
        }
    };
}

namespace dg::network_memlock_host{

    using namespace dg::network_memlock;
    using namespace dg::network_lang_x;

    template <class ID, class MemRegionSize, class PtrT = std::add_pointer_t<const void>>
    struct AtomicFlagLock{}; 

    template <class ID, size_t MEMREGION_SZ, class PtrT>
    struct AtomicFlagLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>: MemoryRegionLockInterface<AtomicFlagLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>>{
        
        public:

            using ptr_t = PtrT; 

        private:

            using self          = AtomicFlagLock;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>;;

            static inline std::atomic_flag * lck_table{};
            
            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) / MEMREGION_SZ;
            }

            static auto to_table_idx(ptr_t ptr) noexcept -> size_t{

                return memregion_slot(segcheck_ins::access(ptr));
            } 

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                return lck_table[table_idx].test_and_set(std::memory_order_acq_rel); //acquire
            }

            static void internal_acquire_wait(size_t table_idx) noexcept{

                while (!internal_acquire_try(table_idx)){}
            }

            static void internal_acquire_release(size_t table_idx) noexcept{

                lck_table[table_idx].clear(std::memory_order_acq_rel); //release
            }

        public:

            static void init(ptr_t ptr, size_t sz){

                auto log_scope = dg::network_log_scope::critical_error_terminate(); 

                if (pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) % MEMREGION_SZ != 0u || sz % MEMREGION_SZ != 0u || pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) == 0u || sz == 0u){
                    throw dg::network_exception::invalid_arg();
                }

                size_t lck_table_sz = (pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) + sz) / MEMREGION_SZ;
                lck_table           = new std::atomic_flag[lck_table_sz];
                segcheck_ins::init(ptr, memult::advance(ptr, sz));
                log_scope.release();
            } 

            static auto memregion_size() noexcept -> size_t{

                return MEMREGION_SZ;
            }

            static auto acquire_try(ptr_t ptr) noexcept -> bool{
                
                return internal_acquire_try(to_table_idx(ptr));
            }

            static void acquire_wait(ptr_t ptr) noexcept{

                internal_acquire_wait(to_table_idx(ptr));
            } 

            static void acquire_release(ptr_t ptr) noexcept{

                internal_acquire_release(to_table_idx(ptr));
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
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>; 

            static inline std::mutex * lck_table{};
        
            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) / MEMREGION_SZ;
            }

            static auto to_table_idx(ptr_t ptr) noexcept -> size_t{

                return memregion_slot(segcheck_ins::access(ptr));
            }
        
        public:

            static void init(ptr_t ptr, size_t sz){
                
                auto log_scope = dg::network_log_scope::critical_error_terminate(); 

                if (pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) % MEMREGION_SZ != 0u || sz % MEMREGION_SZ != 0u || pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) == 0u || sz == 0u){
                    throw dg::network_exception::invalid_arg();
                }

                size_t lck_table_sz = (pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) + sz) / MEMREGION_SZ;
                lck_table           = new std::mutex[lck_table_sz];
                segcheck_ins::init(ptr, memult::advance(ptr, sz));
                log_scope.release();
            }

            static auto acquire_try(ptr_t ptr) noexcept -> bool{

                return lck_table[to_table_idx(ptr)].try_lock();
            }

            static auto memregion_size() noexcept -> size_t{

                return MEMREGION_SZ;
            }

            static void acquire_wait(ptr_t ptr) noexcept{

                lck_table[to_table_idx(ptr)].lock();
            } 

            static void acquire_release(ptr_t ptr) noexcept{

                lck_table[to_table_idx(ptr)].unlock();
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

            static_assert(MEMREGION_SZ != 0u);
            static_assert((MEMREGION_SZ & (MEMREGION_SZ - 1)) == 0u);

            using atomic_lock_t = uint64_t; 
            using self          = AtomicReferenceLock;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>; 

            static inline std::atomic<atomic_lock_t> * lck_table{};    
            static inline constexpr atomic_lock_t MEMREGION_EMP_STATE = 0u;
            static inline constexpr atomic_lock_t MEMREGION_ACQ_STATE = ~PAGE_EMP_STATE;

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) / MEMREGION_SZ;
            }

            static auto to_table_idx(ptr_t ptr) noexcept -> size_t{

                return memregion_slot(segcheck_ins::access(ptr));
            } 

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                return lck_table[table_idx].compare_exchange_weak(MEMREGION_EMP_STATE, MEMREGION_ACQ_STATE, std::memory_order_acq_rel);
            } 

            static void internal_acquire_wait(size_t table_idx) noexcept{

                while (!acquire_try(table_idx)){}
            }

            static void internal_acquire_release(size_t table_idx) noexcept{

                lck_table[table_idx].exchange(MEMREGION_EMP_STATE, std::memory_order_acq_rel); //release
            }

            static auto internal_reference_try(size_t table_idx) noexcept -> bool{

                atomic_lock_t cur_state  = lck_table[table_idx].load(std::memory_order_acquire); //relaxed

                if (cur_state == MEMREGION_ACQ_STATE){
                    return false;
                }

                atomic_lock_t nxt_state  = cur_state + 1;
                return lck_table[table_idx].compare_exchange_weak(cur_state, nxt_state, std::memory_order_acq_rel);
            }

            static void internal_reference_wait(size_t table_idx) noexcept{

                while (!reference_try(table_idx)){}
            }

            static void internal_reference_release(size_t table_idx) noexcept{
                
                lck_table[table_idx].fetch_sub(1, std::memory_order_acq_rel);
            }

        public:

            static void init(ptr_t ptr, size_t sz){
                
                auto log_scope = dg::network_log_scope::critical_error_terminate(); 

                if (pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) % MEMREGION_SZ != 0u || pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) == 0u || sz % MEMREGION_SZ != 0u || sz == 0u){
                    throw dg::network_exception::invalid_arg();
                }

                size_t lck_table_sz = (pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) + sz) / MEMREGION_SZ;
                lck_table           = new std::atomic<atomic_lock_t>[lck_table_sz];

                for (size_t i = 0u; i < lck_table_sz; ++i){
                    lck_table[i] = MEMREGION_EMP_STATE;
                }

                segcheck_ins::init(ptr, memult::advance(ptr, sz));
                log_scope.release();
            }

            static auto transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return memregion_slot(segcheck_ins::access(new_ptr)) == memregion_slot(segcheck_ins::access(old_ptr));
            }

            static auto acquire_try(ptr_t ptr) noexcept -> bool{

                return internal_acquire_try(to_table_idx(ptr));
            }

            static void acquire_wait(ptr_t ptr) noexcept{

                internal_acquire_wait(to_table_idx(ptr));
            }

            static void acquire_release(ptr_t ptr) noexcept{

                internal_acquire_release(to_table_idx(ptr));
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

                return internal_reference_try(to_table_idx(ptr));
            }

            static void reference_wait(ptr_t ptr) noexcept{

                internal_reference_wait(to_table_idx(ptr));
            } 

            static void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(to_table_idx(ptr));
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

    template <class ID, class MemRegionSize, class PtrT = std::add_pointer_t<const void>>
    struct MtxReferenceLock{};    

    template <class ID, size_t MEMREGION_SZ, class PtrT>
    struct MtxReferenceLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>: MemoryReferenceLockInterface<MtxReferenceLocK<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>>{

        public:

            using ptr_t = PtrT; 

        private:

            using refcount_t    = uint32_t;
            using self          = MtxReferenceLock;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>;

            static constexpr inline refcount_t REFERENCE_EMPTY_STATE    = 0u;
            static constexpr inline refcount_t REFERENCE_ACQUIRED_STATE = ~REFERENCE_EMPTY_STATE;

            struct LockUnit{
                std::mutex lck;
                refcount_t refcount;
            };

            static inline LockUnit * lck_table{};

            static auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) / MEMREGION_SZ;
            }

            static auto to_table_idx(ptr_t ptr) noexcept -> size_t{

                return memregion_slot(segcheck_ins::access(ptr));
            } 

            static auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                std::lock_guard<std::mutex> lck_grd{lck_table[table_idx].lck};
                
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

                std::lock_guard<std::mutex> lck_grd{lck_table[table_idx].lck};
                lck_table[table_idx].refcount = REFERENCE_EMPTY_STATE;
            }

            static auto internal_reference_try(size_t table_idx) noexcept -> bool{

                std::lock_guard<std::mutex> lck_grd{lck_table[table_idx].lck};
                
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

                std::lock_guard<std::mutex> lck_grd{lck_table[table_idx].lck};
                --lck_table[table_idx].refcount;
            }

        public:

            static void init(ptr_t ptr, size_t sz){ 

                auto log_scope = dg::network_log_scope::critical_error_terminate(); 

                if (pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) == 0u || sz == 0u || pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) % MEMREGION_SZ != 0u || sz % MEMREGION_SZ != 0u){
                    throw dg::network_exception::invalid_arg();
                }

                size_t lck_table_sz     = (pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) + sz) / MEMREGION_SZ;
                lck_table               = new LockUnit[lck_table_sz];
                segcheck_ins::init(ptr, memult::advance(ptr, sz));

                log_scope.release();
            } 

            static auto transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{
                
                return memregion_slot(segcheck_ins::access(new_ptr)) == memregion_slot(segcheck_ins::access(old_ptr));
            }

            static auto acquire_try(ptr_t ptr) noexcept -> bool{

                return internal_acquire_try(to_table_idx(ptr));
            }

            static auto acquire_wait(ptr_t ptr) noexcept{

                internal_acquire_wait(to_table_idx(ptr));
            } 

            static void acquire_release(ptr_t ptr) noexcept{

                internal_acquire_release(to_table_idx(ptr));
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

                return internal_reference_try(to_table_idx(ptr));
            }

            static void reference_wait(ptr_t ptr) noexcept{

                internal_reference_wait(to_table_idx(ptr));
            } 

            static void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(to_table_idx(ptr));
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
    

    template <class ID, class MemRegionSize, class PtrT = std::add_pointer_t<const void>>
    using ReferenceLock = std::conditional_t<IS_ATOMIC_OPERATION_PREFERRED,
                                             AtomicReferenceLock<ID, MemRegionSize, PtrT>,
                                             MtxReferenceLock<ID, MemRegionSize, PtrT>>;

} 

namespace dg::network_memlock_utility{

    template <class T, class ptr_t>
    auto lock_guard(const dg::network_memlock::MemoryLockInterface<T>, ptr_t ptr) noexcept{

        using memlock_ins   = dg::network_memlock::MemoryLockInterface<T>;
        using lock_ptr_t    = typename memlock_ins::ptr_t; 
        static_assert(std::is_same_v<lock_ptr_t, ptr_t>);

        static int i        = 0;
        auto destructor = [=](int *) noexcept{
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
            using ptr_t                 = typename dg::network_memlock::MemoryRegionLockInterface<T>::ptr_t;
            using resource_t            = std::unordered_set<ptr_t>;
            using singleton_obj_t       = std::array<resource_t, dg::network_concurrency::THREAD_COUNT>;
            using singleton_container   = dg::network_genult::singleton<id, singleton_obj_t>;
        
        public:

            static inline auto get() noexcept -> std::unordered_set<ptr_t>&{

                return singleton_container::get()[dg::network_concurrency::this_thread_idx()];
            }
    };

    template <class T, class ptr_t>
    auto recursive_trylock_guard(const dg::network_memlock::MemoryRegionLockInterface<T>, ptr_t ptr) noexcept{
        
        using memlock_ins   = dg::network_memlock::MemoryRegionLockInterface<T>;
        using lock_ptr_t    = typename memlock_ins::ptr_t;
        using resource_ins  = RecursiveLockResource<dg::network_memlock::MemoryRegionLockInterface<T>>;
        static_assert(std::is_same_v<lock_ptr_t, ptr_t>);

        std::unordered_set<lock_ptr_t> * ptr_set = &resource_ins::get();
        bool responsibility_flag    = false;
        lock_ptr_t ptr_region       = dg::memult::region(ptr, memlock_ins::memregion_size()); 
        bool try_success_flag       = true;

        if (ptr_set->find(ptr_region) == ptr_set->end()){            
            if (memlock_ins::acquire_try(ptr_region)){
                responsibility_flag = true;
                ptr_set->add(ptr_region);
            } else{
                try_success_flag = false; 
            }
        }

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            if (responsibility_flag){
                ptr_set->remove(ptr_region);
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
        auto try_lambda = [=]<class Self, size_t IDX>(Self self, const std::integral_constant<size_t, IDX>){
            if constexpr(IDX == sizeof...(Args)){
                return bool{true};
            } else{                
                using successor_ret_t   = decltype(self(self, std::integral_constant<size_t, IDX + 1>{}));
                using lck_t             = decltype(recursive_trylock_guard(lock_ins, std::get<IDX>(args_tup)));
                using ret_t             = std::pair<lck_t, successor_ret_t>;
                using opt_ret_t         = std::optional<ret_t>;
                auto successor_rs       = self(self, std::integral_constant<size_t, IDX + 1>{});

                if (!static_cast<bool>(successor_rs)){
                    return opt_ret_t{std::nullopt};
                } else{
                    auto cur_lck = recursive_trylock_guard(lock_ins, std::get<IDX>(args_tup));
                    if (!static_cast<bool>(cur_lck)){
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

#endif 