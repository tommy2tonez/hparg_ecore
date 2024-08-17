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

namespace dg::network_memlock{
    
    template <class T>
    struct MemoryLockInterface{

        using interface_t   = MemoryLockInterface<T>;
        using ptr_t         = typename T::ptr_t;
        static_assert(dg::is_pointer_v<ptr_t>); 

        static inline auto acquire_try(ptr_t ptr) noexcept -> bool{

            return T::acquire_try(ptr);
        }

        static inline auto acquire_wait(ptr_t ptr) noexcept{

            T::acquire_wait(ptr);
        } 

        static inline void acquire_release(ptr_t ptr) noexcept{

            T::acquire_release(ptr);
        }

        static inline auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

            return T::transfer_try(new_ptr, old_ptr);
        } 

        static inline void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

            T::acquire_transfer_wait(new_ptr, old_ptr);
        }
    };

    template <class T>
    struct MemoryReferenceInterface{

        using interface_t   = MemoryReferenceInterface<T>; 
        using ptr_t         = typename T::ptr_t;
        static_assert(dg::is_pointer_v<ptr_t>); 

        static inline auto reference_try(ptr_t ptr) noexcept -> bool{

            return T::reference_try(ptr);
        }

        static inline void reference_wait(ptr_t ptr) noexcept {

            T::reference_wait(ptr);
        } 

        static inline void reference_release(ptr_t ptr) noexcept{

            T::reference_release(ptr);
        }

        static inline auto reference_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

            return T::reference_transfer_try(new_ptr, old_ptr);
        } 

        static inline void reference_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

            T::reference_transfer_wait(new_ptr, old_ptr);
        }
    };

    template <class T>
    struct MemoryReferenceLockInterface: MemoryLockInterface<T>, MemoryReferenceInterface<T>{
        
        using interface_t   = MemoryReferenceLockInterface<T>;
        using ptr_t         = typename T::ptr_t;

        static inline auto transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{
            
            return T::transfer_try(new_ptr, old_ptr);
        }
    };

    template <class T>
    struct MemoryLockXInterface{

        using interface_t   = MemoryLockXInterface; 
        using ptr_t         = typename T::ptr_t;
        static_assert(dg::is_pointer_v<ptr_t>); 

        static inline auto acquire_try(ptr_t * ptr) noexcept -> bool{

            return T::acquire_try(ptr);
        }

        static inline void acquire_wait(ptr_t * ptr) noexcept{

            T::acquire_wait(ptr);
        } 

        static inline void acquire_release(ptr_t * ptr) noexcept{

            T::acquire_release(ptr);
        }

        static inline auto acquire_transfer_try(ptr_t * new_ptr, ptr_t * old_ptr) noexcept -> bool{

            return T::transfer_try(new_ptr, old_ptr);
        } 

        static inline void acquire_transfer_wait(ptr_t * new_ptr, ptr_t * old_ptr) noexcept{

            T::acquire_transfer_wait(new_ptr, old_ptr);
        }
    };

    template <class T>
    struct MemoryReferenceXInterface{

        using interface_t   = MemoryReferenceXInterface<T>; 
        using ptr_t         = typename T::ptr_t;
        static_assert(dg::is_pointer_v<ptr_t>); 

        static inline auto reference_try(ptr_t * ptr) noexcept -> bool{

            return T::reference_try(ptr);
        }

        static inline void reference_wait(ptr_t * ptr) noexcept {

            T::reference_wait(ptr);
        } 

        static inline void reference_release(ptr_t * ptr) noexcept{

            T::reference_release(ptr);
        }

        static inline auto reference_transfer_try(ptr_t * new_ptr, ptr_t * old_ptr) noexcept -> bool{

            return T::reference_transfer_try(new_ptr, old_ptr);
        } 

        static inline void reference_transfer_wait(ptr_t * new_ptr, ptr_t * old_ptr) noexcept{

            T::reference_transfer_wait(new_ptr, old_ptr);
        }
    };
} 

namespace dg::network_memlock_host{

    using namespace dg::network_memlock;
    using namespace dg::network_lang_x;

    template <class ID, class MemRegionSize, class PtrT = std::add_pointer_t<const void>>
    struct AtomicFlagLock{}; 

    template <class ID, size_t MEMREGION_SZ, class PtrT>
    struct AtomicFlagLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>: MemoryLockInterface<AtomicFlagLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>>{
        
        public:

            using ptr_t = PtrT; 

        private:

            using self          = AtomicFlagLock;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>;;

            static inline std::atomic_flag * lck_table{};
            
            static inline auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto to_table_idx(ptr_t ptr) noexcept -> size_t{

                return memregion_slot(segcheck_ins::access(ptr));
            } 

            static inline auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                return lck_table[table_idx].test_and_set(std::memory_order_acq_rel);
            }

            static inline void internal_acquire_wait(size_t table_idx) noexcept{

                while (!internal_acquire_try(table_idx)){}
            }

            static inline void internal_acquire_release(size_t table_idx) noexcept{

                lck_table[table_idx].clear(std::memory_order_acq_rel);
            }

        public:

            static void init(ptr_t ptr, size_t sz){

                auto log_scope = dg::network_log_scope::critical_error_terminate(); 

                if (pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) % MEMREGION_SZ != 0u || sz % MEMREGION_SZ != 0u || pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) == 0u || sz == 0u){
                    throw dg::network_exception::invalid_arg();
                }

                size_t lck_table_sz = (pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) + sz) / MEMREGION_SZ;
                lck_table           = new std::atomic_flag[lck_table_sz];
                segcheck_ins::init(ptr, sz);
                log_scope.release();
            } 

            static inline auto acquire_try(ptr_t ptr) noexcept -> bool{
                
                return internal_acquire_try(to_table_idx(ptr));
            }

            static inline void acquire_wait(ptr_t ptr) noexcept{

                internal_acquire_wait(to_table_idx(ptr));
            } 

            static inline void acquire_release(ptr_t ptr) noexcept{

                internal_acquire_release(to_table_idx(ptr));
            }

            static inline auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return memregion_slot(segcheck_ins::access(new_ptr)) == memregion_slot(segcheck_ins::access(old_ptr));
            } 

            static inline void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

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
    struct MtxLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>: MemoryLockInterface<MtxLock<ID, std::integral_constant<size_t, MEMREGION_SZ>, PtrT>>{

        public:

            using ptr_t = PtrT; 

        private:

            using self          = MtxLock;
            using segcheck_ins  = dg::network_segcheck_bound::StdAccess<self, ptr_t>; 

            static inline std::mutex * lck_table{};
        
            static inline auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto to_table_idx(ptr_t ptr) noexcept -> size_t{

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

            static inline auto acquire_try(ptr_t ptr) noexcept -> bool{

                return lck_table[to_table_idx(ptr)].try_lock();
            }

            static inline void acquire_wait(ptr_t ptr) noexcept{

                lck_table[to_table_idx(ptr)].lock();
            } 

            static inline void acquire_release(ptr_t ptr) noexcept{

                lck_table[to_table_idx(ptr)].unlock();
            }

            static inline auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return memregion_slot(segcheck_ins::access(new_ptr)) == memregion_slot(segcheck_ins::access(old_ptr));
            }

            static inline void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

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
            static inline atomic_lock_t MEMREGION_EMP_STATE = 0u;
            static inline atomic_lock_t MEMREGION_ACQ_STATE = ~PAGE_EMP_STATE;

            static inline auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto to_table_idx(ptr_t ptr) noexcept -> size_t{

                return memregion_slot(segcheck_ins::access(ptr));
            } 

            static inline auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                return lck_table[table_idx].compare_exchange_weak(MEMREGION_EMP_STATE, MEMREGION_ACQ_STATE, std::memory_order_acq_rel);
            } 

            static inline void internal_acquire_wait(size_t table_idx) noexcept{

                while (!acquire_try(table_idx)){}
            }

            static inline void internal_acquire_release(size_t table_idx) noexcept{

                lck_table[table_idx].exchange(MEMREGION_EMP_STATE, std::memory_order_acq_rel);
            }

            static inline auto internal_reference_try(size_t table_idx) noexcept -> bool{

                atomic_lock_t cur_state  = lck_table[table_idx].load(std::memory_order_acquire);

                if (cur_state == MEMREGION_ACQ_STATE){
                    return false;
                }

                atomic_lock_t nxt_state  = cur_state + 1;
                return lck_table[table_idx].compare_exchange_weak(cur_state, nxt_state, std::memory_order_acq_rel);
            }

            static inline void internal_reference_wait(size_t table_idx) noexcept{

                while (!reference_try(table_idx)){}
            }

            static inline void internal_reference_release(size_t table_idx) noexcept{
                
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

            static inline auto transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return memregion_slot(segcheck_ins::access(new_ptr)) == memregion_slot(segcheck_ins::access(old_ptr));
            }

            static inline auto acquire_try(ptr_t ptr) noexcept -> bool{

                return internal_acquire_try(to_table_idx(ptr));
            }

            static inline void acquire_wait(ptr_t ptr) noexcept{

                internal_acquire_wait(to_table_idx(ptr));
            }

            static inline void acquire_release(ptr_t ptr) noexcept{

                internal_acquire_release(to_table_idx(ptr));
            }

            static inline auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return transfer_try(new_ptr, old_ptr);
            } 

            static inline void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (acquire_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                acquire_release(old_ptr);
                acquire_wait(new_ptr);
            }

            static inline auto reference_try(ptr_t ptr) noexcept -> bool{

                return internal_reference_try(to_table_idx(ptr));
            }

            static inline void reference_wait(ptr_t ptr) noexcept{

                internal_reference_wait(to_table_idx(ptr));
            } 

            static inline void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(to_table_idx(ptr));
            }

            static inline auto reference_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return transfer_try(new_ptr, old_ptr);
            } 

            static inline void reference_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

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

            static inline constexpr refcount_t REFERENCE_EMPTY_STATE    = 0u;
            static inline constexpr refcount_t REFERENCE_ACQUIRED_STATE = ~REFERENCE_EMPTY_STATE;

            struct LockUnit{
                std::mutex lck;
                refcount_t refcount;
            };

            static inline LockUnit * lck_table{};

            static inline auto memregion_slot(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<typename dg::ptr_info<ptr_t>::max_unsigned_t>(ptr) / MEMREGION_SZ;
            }

            static inline auto to_table_idx(ptr_t ptr) noexcept -> size_t{

                return memregion_slot(segcheck_ins::access(ptr));
            } 

            static inline auto internal_acquire_try(size_t table_idx) noexcept -> bool{

                std::lock_guard<std::mutex> lck_grd{lck_table[table_idx].lck};
                
                if (lck_table[table_idx].refcount != REFERENCE_EMPTY_STATE){
                    return false;
                }

                lck_table[table_idx].refcount = REFERENCE_ACQUIRED_STATE;
                return true;
            }

            static inline auto internal_acquire_wait(size_t table_idx) noexcept{

                while (!internal_acquire_try(table_idx)){}
            }

            static inline auto internal_acquire_release(size_t table_idx) noexcept{

                std::lock_guard<std::mutex> lck_grd{lck_table[table_idx].lck};
                lck_table[table_idx].refcount = REFERENCE_EMPTY_STATE;
            }

            static inline auto internal_reference_try(size_t table_idx) noexcept -> bool{

                std::lock_guard<std::mutex> lck_grd{lck_table[table_idx].lck};
                
                if (lck_table[table_idx].refcount == REFERENCE_ACQUIRED_STATE){
                    return false;
                }

                ++lck_table[table_idx].refcount;
                return true;
            }

            static inline auto internal_reference_wait(size_t table_idx) noexcept{

                while (!internal_reference_try(table_idx)){}
            }

            static inline auto internal_reference_release(size_t table_idx) noexcept{

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

            static inline auto transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{
                
                return memregion_slot(segcheck_ins::access(new_ptr)) == memregion_slot(segcheck_ins::access(old_ptr));
            }

            static inline auto acquire_try(ptr_t ptr) noexcept -> bool{

                return internal_acquire_try(to_table_idx(ptr));
            }

            static inline auto acquire_wait(ptr_t ptr) noexcept{

                internal_acquire_wait(to_table_idx(ptr));
            } 

            static inline void acquire_release(ptr_t ptr) noexcept{

                internal_acquire_release(to_table_idx(ptr));
            }

            static inline auto acquire_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return transfer_try(new_ptr, old_ptr);
            } 

            static inline void acquire_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (acquire_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                acquire_release(old_ptr);
                acquire_wait(new_ptr);
            }

            static inline auto reference_try(ptr_t ptr) noexcept -> bool{

                return internal_reference_try(to_table_idx(ptr));
            }

            static inline void reference_wait(ptr_t ptr) noexcept{

                internal_reference_wait(to_table_idx(ptr));
            } 

            static inline void reference_release(ptr_t ptr) noexcept{

                internal_reference_release(to_table_idx(ptr));
            }

            static inline auto reference_transfer_try(ptr_t new_ptr, ptr_t old_ptr) noexcept -> bool{

                return transfer_try(new_ptr, old_ptr);
            }

            static inline void reference_transfer_wait(ptr_t new_ptr, ptr_t old_ptr) noexcept{

                if (reference_transfer_try(new_ptr, old_ptr)){
                    return;
                }

                reference_release(old_ptr);
                reference_wait(new_ptr);
            }  
    };

    template <class ID, class T, class MaxArgSize>
    struct LockX_Unique{};

    template <class ID, class T, size_t MAX_ARG_SIZE>
    struct LockX_Unique<ID, MemoryLockInterface<T>, std::integral_constant<size_t, MAX_ARG_SIZE>>: MemoryLockXInterface<LockX_Unique<ID, MemoryLockInterface<T>, std::integral_constant<size_t, MAX_ARG_SIZE>>>{

        using base  = MemoryLockInterface<T>;
        using ptr_t = typename base::ptr_t;

        static inline auto acquire_try(ptr_t * ptr) noexcept -> bool{

            std::array<ptr_t, MAX_ARG_SIZE> acquired_arr{};
            ptr_t * last = acquired_arr.data();

            while (static_cast<bool>(*ptr)){
                if (!base::acquire_try(*ptr)){
                    break;
                }

                *(last++) = *(ptr++);
            }

            if (!static_cast<bool>(*ptr)){
                return true;
            }
            
            for (auto i = acquired_arr.data(); i != last; ++i){
                base::acquire_release(*i);
            }

            return false;
        }

        static inline void acquire_wait(ptr_t * ptr) noexcept{

            while (!acquire_try_many(ptr)){};
        }

        static inline void acquire_release(ptr_t * ptr) noexcept{

            while (static_cast<bool>(*ptr)){
                base::acquire_release(*ptr);
                ++ptr;
            }
        } 

        static inline auto acquire_transfer_try(ptr_t * dst, ptr_t * src) noexcept -> bool{
            
            while (static_cast<bool>(*dst)){
                if (!base::transfer_try(*dst, *src)){
                    return false;
                }

                ++dst;
                ++src;
            }

            return true;
        }

        static inline void acquire_transfer_wait(ptr_t * dst, ptr_t * src) noexcept{

            if (transfer_try_many(dst, src)){
                return;
            }

            acquire_release(src);
            acquire_wait(dst);
        }
    };

    template <class ID, class T, class MaxArgSize>
    struct ReferenceX_Unique{};

    template <class ID, class T, size_t MAX_ARG_SIZE>
    struct ReferenceX_Unique<ID, MemoryReferenceInterface<T>, std::integral_constant<size_t, MAX_ARG_SIZE>>: MemoryReferenceXInterface<ReferenceX_Unique<ID, MemoryReferenceInterface<T>, std::integral_constant<size_t, MAX_ARG_SIZE>>>{

        using base  = MemoryReferenceInterface<T>;
        using ptr_t = typename base::ptr_t;
        
        static inline auto reference_try(ptr_t * ptr) noexcept -> bool{

            std::array<ptr_t, MAX_ARG_SIZE> acquired_arr{};
            ptr_t * last = acquired_arr.data();

            while (static_cast<bool>(*ptr)){
                if (!base::reference_try(*ptr)){
                    break;
                }

                *(last++) = *(ptr++);
            }

            if (!static_cast<bool>(*ptr)){
                return true;
            }
            
            for (auto i = acquired_arr.data(); i != last; ++i){
                base::reference_release(*i);
            }
            
            return false;
        }

        static inline void reference_wait(ptr_t * ptr) noexcept{

            while (!reference_try_many(ptr)){}
        }

        static inline void reference_release(ptr_t * ptr) noexcept{

            while (static_cast<bool>(*ptr)){
                base::reference_release(*ptr);
                ++ptr;
            }
        }

        static inline auto reference_transfer_try(ptr_t * dst, ptr_t * src) noexcept -> bool{
            
            while (static_cast<bool>(*dst)){
                if (!base::transfer_try(*dst, *src)){
                    return false;
                }

                ++dst;
                ++src;
            }

            return true;
        }

        static inline void reference_transfer_wait(ptr_t * dst, ptr_t * src) noexcept{

            if (reference_transfer_try(dst, src)){
                return;
            }

            reference_release(src);
            reference_wait(dst);
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

#endif 