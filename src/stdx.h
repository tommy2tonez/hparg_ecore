#ifndef __STD_X_H__
#define __STD_X_H__

//define HEADER_CONTROL 0

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <deque>
#include <atomic>
#include <mutex>
#include <functional>
#include <chrono>
#include "network_raii_x.h"
#include <stdfloat>
#include <immintrin.h>
#include <utility>

namespace stdx{

    //to check the get_ptr_arithemtic, we need to focus on the bit representations, such is two complements bit pattern when we are in the negative territory, we would want to increment, and extract the direction
    //we need to make sure that this is the cases by doing compile-time measurements
    //things got complicated when pointer is unsigned or contains unsigned addresses, we'll break assumptions

    using ptr_bitcastable_arithmetic_t = std::conditional_t<sizeof(void *) == 4u, 
                                                            uint32_t,
                                                            std::conditional_t<sizeof(void *) == 8u,
                                                                               uint64_t,
                                                                               void>>; 

    inline __attribute__((always_inline)) auto try_lock(std::atomic_flag& mtx, std::memory_order order = std::memory_order_seq_cst) noexcept -> bool{
        return mtx.test_and_set(order) == false;
    }

    inline __attribute__((always_inline)) auto try_lock(std::mutex& mtx) noexcept -> bool{
        return mtx.try_lock();
    }

    inline __attribute__((always_inline)) void lock_yield(std::chrono::nanoseconds lapsed) noexcept{

        (void) lapsed;
    }

    struct polymorphic_launderer{
        virtual auto ptr() noexcept -> void *{
            return nullptr;
        }
    };

    template <class T>
    struct launderer{}; 

    template <>
    struct launderer<uint8_t>: polymorphic_launderer{
        uint8_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<uint16_t>: polymorphic_launderer{
        uint16_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<uint32_t>: polymorphic_launderer{
        uint32_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<uint64_t>: polymorphic_launderer{
        uint64_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<char>: polymorphic_launderer{
        char * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int8_t>: polymorphic_launderer{
        int8_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int16_t>: polymorphic_launderer{
        int16_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int32_t>: polymorphic_launderer{
        int32_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<int64_t>: polymorphic_launderer{
        int64_t * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<float>: polymorphic_launderer{
        float * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<double>: polymorphic_launderer{
        double * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    template <>
    struct launderer<void>: polymorphic_launderer{
        void * value;

        virtual auto ptr() noexcept -> void *{
            return value;
        }
    };

    struct polymorphic_const_launderer{
        virtual auto ptr() noexcept -> const void *{
            return nullptr;
        }
    };

    template <class T>
    struct const_launderer{};

    template <>
    struct const_launderer<uint8_t>: polymorphic_const_launderer{
        const uint8_t * value;

        virtual auto ptr() noexcept -> const void *{
            return value;
        }
    };

    template <>
    struct const_launderer<uint16_t>: polymorphic_const_launderer{
        const uint16_t * value;

        virtual auto ptr() noexcept -> const void *{
            return value;
        }
    };

    template <>
    struct const_launderer<uint32_t>: polymorphic_const_launderer{
        const uint32_t * value;

        virtual auto ptr() noexcept -> const void *{
            return value;
        }
    };

    template <>
    struct const_launderer<uint64_t>: polymorphic_const_launderer{
        const uint64_t * value;

        virtual auto ptr() noexcept -> const void *{
            return value;
        }
    };

    template <>
    struct const_launderer<char>: polymorphic_const_launderer{
        const char * value;

        virtual auto ptr() noexcept -> const void *{
            return value;
        }
    };

    template <>
    struct const_launderer<int8_t>: polymorphic_const_launderer{
        const int8_t * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        } 
    };

    template <>
    struct const_launderer<int16_t>: polymorphic_const_launderer{
        const int16_t * value;

        virtual auto ptr() noexcept -> const void *{
            
            return value;
        }
    };

    template <>
    struct const_launderer<int32_t>: polymorphic_const_launderer{
        const int32_t * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    template <>
    struct const_launderer<int64_t>: polymorphic_const_launderer{
        const int64_t * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    template <>
    struct const_launderer<float>: polymorphic_const_launderer{
        const float * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        } 
    };

    template <>
    struct const_launderer<double>: polymorphic_const_launderer{
        const double * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    template <>
    struct const_launderer<void>: polymorphic_const_launderer{
        const void * value;

        virtual auto ptr() noexcept -> const void *{

            return value;
        }
    };

    using uint8_launderer           = launderer<uint8_t>;
    using uint16_launderer          = launderer<uint16_t>;
    using uint32_launderer          = launderer<uint32_t>;
    using uint64_launderer          = launderer<uint64_t>;
    using char_launderer            = launderer<char>; //
    using int8_launderer            = launderer<int8_t>;
    using int16_launderer           = launderer<int16_t>;
    using int32_launderer           = launderer<int32_t>;
    using int64_launderer           = launderer<int64_t>;
    using float_launderer           = launderer<float>;
    using double_launderer          = launderer<double>; 
    using void_launderer            = launderer<void>;
    using uint8_const_launderer     = const_launderer<uint8_t>;
    using uint16_const_launderer    = const_launderer<uint16_t>;
    using uint32_const_launderer    = const_launderer<uint32_t>;
    using uint64_const_launderer    = const_launderer<uint64_t>;
    using char_const_launderer      = const_launderer<char>; //
    using int8_const_launderer      = const_launderer<int8_t>;
    using int16_const_launderer     = const_launderer<int16_t>;
    using int32_const_launderer     = const_launderer<int32_t>;
    using int64_const_launderer     = const_launderer<int64_t>;
    using float_const_launderer     = const_launderer<float>;
    using double_const_launderer    = const_launderer<double>; 
    using void_const_launderer      = const_launderer<void>;

    static inline constexpr bool IS_SAFE_MEMORY_ORDER_ENABLED                                   = false;
    static inline constexpr bool IS_SAFE_INTEGER_CONVERSION_ENABLED                             = true;
    static inline constexpr bool IS_ATOMIC_FLAG_AS_SPINLOCK                                     = true;

    static inline constexpr size_t SPINLOCK_SIZE_MAGIC_VALUE                                    = 16u;
    static inline constexpr size_t EXPBACKOFF_MUTEX_SPINLOCK_SIZE                               = 16u; 

    static inline constexpr std::chrono::nanoseconds EXPBACKOFF_DEFAULT_SPIN_PERIOD             = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds(10));
    static inline constexpr std::chrono::nanoseconds EXPBACKOFF_MUTEX_SPIN_PERIOD               = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::microseconds(10));

    using spin_lock_t = std::conditional_t<IS_ATOMIC_FLAG_AS_SPINLOCK,
                                           std::atomic_flag,
                                           std::mutex>; 

    // template <class Lambda>
    // inline void eventloop_expbackoff_spin(Lambda&& lambda) noexcept(noexcept(lambda())){

    //     const size_t BASE                   = 2u;
    //     const size_t MAX_SEQUENTIAL_PAUSE   = 64u;
    //     size_t current_sequential_pause     = 1u;

    //     while (true){
    //         if (lambda()){
    //             return;
    //         }

    //         for (size_t i = 0u; i < current_sequential_pause; ++i){
    //             _mm_pause();
    //         }

    //         current_sequential_pause = std::min(MAX_SEQUENTIAL_PAUSE, current_sequential_pause * BASE);
    //     }
    // }

    //recall the eqn: x^0 + x^1 + x^n = f(x) = (x^ (n + 1) - 1) / (x - 1)
    //we are to find x

    template <class Lambda>
    inline bool eventloop_expbackoff_spin(Lambda&& lambda, 
                                          size_t spin_sz,
                                          std::chrono::nanoseconds period) noexcept(noexcept(lambda())){

        const size_t BASE                   = 2u;
        const size_t MAX_SEQUENTIAL_PAUSE   = 64u;
        size_t current_sequential_pause     = 1u;

        for (size_t i = 0u; i < spin_sz; ++i){
            if (lambda()){
                return true;
            }

            for (size_t i = 0u; i < current_sequential_pause; ++i){
                _mm_pause();
            }

            current_sequential_pause = std::min(MAX_SEQUENTIAL_PAUSE, current_sequential_pause * BASE);
        }

        return false;
    }

    template <class Lambda>
    inline void eventloop_competitive_spin(Lambda&& lambda) noexcept(noexcept(lambda())){

        lambda();
    }

    template <class Lambda>
    inline bool eventloop_competitive_spin(Lambda&& lambda, size_t sz) noexcept(noexcept(lambda())){

        return true;
    } 

    template <class Lambda>
    inline void eventloop_cyclic_expbackoff_spin(Lambda&& lambda,
                                                 size_t spin_sz,
                                                 std::chrono::nanoseconds period) noexcept(noexcept(lambda())){

        lambda();
    } 

    template <class Lambda>
    inline bool eventloop_cyclic_expbackoff_spin(Lambda&& lambda, 
                                                 size_t spin_sz,
                                                 std::chrono::nanoseconds period,
                                                 size_t revolution) noexcept(noexcept(lambda())){

        return true;
    }

    inline __attribute__((always_inline)) bool atomic_flag_memsafe_try_lock(std::atomic_flag * volatile mtx) noexcept{

        //fencing the before transaction, this is very important
        std::atomic_signal_fence(std::memory_order_seq_cst);

        bool is_success = mtx->test_and_set(std::memory_order_relaxed) == false;

        if (!is_success){
            return false;
        }

        //the test_and_set is guaranteed to be sequenced before this line, because there is a branch inferring the relaxed operation

        if constexpr(STRONG_MEMORY_ORDERING_FLAG){
            std::atomic_thread_fence(std::memory_order_seq_cst);
        } else{
            std::atomic_thread_fence(std::memory_order_acquire);
        }
    } 

    inline __attribute__((always_inline)) void atomic_flag_memsafe_lock(std::atomic_flag * volatile mtx) noexcept{

        //fencing the before transaction, this is very important
        std::atomic_signal_fence(std::memory_order_seq_cst);

        auto job = [&]() noexcept{
            return mtx->test_and_set(std::memory_order_relaxed) == false;
        };

        if (!job()){ //fast_path
            while (true){
                if (eventloop_expbackoff_spin(job, EXPBACKOFF_MUTEX_SPINLOCK_SIZE, EXPBACKOFF_MUTEX_SPIN_PERIOD)){
                    break;
                }

                mtx->wait(true, std::memory_order_relaxed); //slow path
            }
        }

        //the test_and_set is guaranteed to be sequenced before this line, because there is a branch inferring the relaxed operation

        if constexpr(STRONG_MEMORY_ORDERING_FLAG){
            std::atomic_thread_fence(std::memory_order_seq_cst);
        } else{
            std::atomic_thread_fence(std::memory_order_acquire);
        }
    }

    inline __attribute__((always_inline)) void atomic_flag_memsafe_unlock(std::atomic_flag * volatile mtx) noexcept{
        
        if constexpr(STRONG_MEMORY_ORDERING_FLAG){
            std::atomic_thread_fence(std::memory_order_seq_cst);
        } else{
            std::atomic_thread_fence(std::memory_order_release);
        }

        //ok, memory-wise OK
        //we are to make sure that the relaxed operation is sequenced after this, 

        std::atomic_signal_fence(std::memory_order_seq_cst);
        mtx->clear(std::memory_order_relaxed);
        mtx->notify_one(); //we are to notify, notify is guaranteed to be sequenced after clear, 
        std::atomic_signal_fence(std::memory_order_seq_cst); //we are to guard the transaction of clear + notify one
    }

    template <class Lock>
    class xlock_guard_base{};

    template <>
    class xlock_guard_base<std::atomic_flag>{

        private:

            std::atomic_flag * volatile mtx; 

        public:

            using self = xlock_guard_base;

            inline __attribute__((always_inline)) xlock_guard_base(std::atomic_flag& mtx) noexcept: mtx(&mtx){

                atomic_flag_memsafe_lock(this->mtx);
           }

            xlock_guard_base(const self&) = delete;
            xlock_guard_base(self&&) = delete;

            inline __attribute__((always_inline)) ~xlock_guard_base() noexcept{

                atomic_flag_memsafe_unlock(this->mtx);
            }

            self& operator =(const self&) = delete;
            self& operator =(self&&) = delete;
    };

    template <>
    class xlock_guard_base<std::mutex>{

        private:

            std::mutex * volatile mtx;

        public:

            using self = xlock_guard_base;

            inline __attribute__((always_inline)) xlock_guard_base(std::mutex& mtx) noexcept: mtx(&mtx){

                this->mtx->lock();
            }

            xlock_guard_base(const self&) = delete;
            xlock_guard_base(self&&) = delete;

            inline __attribute__((always_inline)) ~xlock_guard_base() noexcept{

                this->mtx->unlock();
            }

            self& operator =(const self&) = delete;
            self& operator =(self&&) = delete;
    };

    template <class Lock>
    struct xlock_guard_chooser{};

    template <>
    struct xlock_guard_chooser<std::atomic_flag>{
        using type = xlock_guard_base<std::atomic_flag>;
    };

    template <>
    struct xlock_guard_chooser<std::mutex>{
        using type = std::lock_guard<std::mutex>;
    };

    template <class Lock>
    using xlock_guard = typename xlock_guard_chooser<Lock>::type;

    //we rather use std::lock_guard for max compatibility

    template <class Lock>
    class unlock_guard{};

    template <>
    class unlock_guard<std::mutex>{

        private:

            std::mutex * volatile mtx; 
        
        public:

            using self = unlock_guard; 

            inline __attribute__((always_inline)) unlock_guard(std::mutex& mtx) noexcept: mtx(&mtx){}

            unlock_guard(const self&) = delete;
            unlock_guard(self&&) = delete;

            inline __attribute__((always_inline)) ~unlock_guard() noexcept{

                this->mtx->unlock();
            }

            self& operator =(const self&) = delete;
            self& operator =(self&&) = delete;
    };

    template <>
    class unlock_guard<std::atomic_flag>{

        private:

            std::atomic_flag * volatile mtx; 
        
        public:

            using self = unlock_guard; 

            inline __attribute__((always_inline)) unlock_guard(std::atomic_flag& mtx) noexcept: mtx(&mtx){}

            unlock_guard(const self&) = delete;
            unlock_guard(self&&) = delete;

            inline __attribute__((always_inline)) ~unlock_guard() noexcept{

                atomic_flag_memsafe_unlock(this->mtx);
            }

            self& operator =(const self&) = delete;
            self& operator =(self&&) = delete;
    };

    class seq_cst_guard{

        public:

            inline __attribute__((always_inline)) seq_cst_guard() noexcept{
            
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            seq_cst_guard(const seq_cst_guard&) = delete;
            seq_cst_guard(seq_cst_guard&&) = delete;

            inline __attribute__((always_inline)) ~seq_cst_guard() noexcept{

                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            seq_cst_guard& operator =(const seq_cst_guard&) = delete;
            seq_cst_guard& operator =(seq_cst_guard&&) = delete;
    };
    
    class memtransaction_guard{

        public:

            inline __attribute__((always_inline)) memtransaction_guard() noexcept{

                if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } else{
                    std::atomic_thread_fence(std::memory_order_acquire);
                }
            }

            memtransaction_guard(const memtransaction_guard&) = delete;
            memtransaction_guard(memtransaction_guard&&) = delete;

            inline __attribute__((always_inline)) ~memtransaction_guard() noexcept{

                if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } else{
                    std::atomic_thread_fence(std::memory_order_release);
                }
            }

            memtransaction_guard& operator =(const memtransaction_guard&) = delete;
            memtransaction_guard& operator =(memtransaction_guard&&) = delete;
    };

    inline __attribute__((always_inline)) void atomic_optional_thread_fence(std::memory_order order = std::memory_order_seq_cst) noexcept{

        if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
            std::atomic_thread_fence(order); 
        } else{
            (void) order;
        }
    }

    inline __attribute__((always_inline)) void atomic_optional_signal_fence(std::memory_order order = std::memory_order_seq_cst) noexcept{

        if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
            std::atomic_signal_fence(order);
        } else{
            (void) order;
        }
    }

    template <class T>
    inline __attribute__((always_inline)) auto safe_ptr_access(T * ptr) noexcept -> T *{

        if (!ptr) [[unlikely]]{
            std::abort();
        }

        return ptr;
    }

    template <class T>
    inline __attribute__((always_inline)) auto launder_pointer(void * volatile ptr) noexcept -> T *{

        static_assert(std::disjunction_v<std::is_same<T, uint8_t>, std::is_same<T, uint16_t>, std::is_same<T, uint32_t>, std::is_same<T, uint64_t>,
                                         std::is_same<T, int8_t>, std::is_same<T, int16_t>, std::is_same<T, int32_t>, std::is_same<T, int64_t>,
                                         std::is_same<T, float>, std::is_same<T, double>, std::is_same<T, void>, std::is_same<T, char>,
                                         std::is_same<T, std::float16_t>, std::is_same<T, std::bfloat16_t>, std::is_same<T, std::float32_t>, 
                                         std::is_same<T, std::float64_t>>);

        std::atomic_signal_fence(std::memory_order_seq_cst);
        launderer<void> launder_machine{};
        launder_machine.value = ptr;
        T * rs = static_cast<T *>(std::launder(&launder_machine)->ptr()); //any sane compiler MUST read the polymorphic header - this is due to the definition given by STD - so it is clean right after launder() - because the function is aliased to all arithmetic_t
        std::atomic_signal_fence(std::memory_order_seq_cst);
        
        return rs;
    }

    template <class T>
    inline __attribute__((always_inline)) auto launder_pointer(const void * volatile ptr) noexcept -> const T *{

        static_assert(std::disjunction_v<std::is_same<T, uint8_t>, std::is_same<T, uint16_t>, std::is_same<T, uint32_t>, std::is_same<T, uint64_t>,
                                         std::is_same<T, int8_t>, std::is_same<T, int16_t>, std::is_same<T, int32_t>, std::is_same<T, int64_t>,
                                         std::is_same<T, float>, std::is_same<T, double>, std::is_same<T, void>, std::is_same<T, char>,
                                         std::is_same<T, std::float16_t>, std::is_same<T, std::bfloat16_t>, std::is_same<T, std::float32_t>, 
                                         std::is_same<T, std::float64_t>>);

        std::atomic_signal_fence(std::memory_order_seq_cst);
        const_launderer<void> launder_machine{};
        launder_machine.value = ptr;
        const T * rs = static_cast<const T *>(std::launder(&launder_machine)->ptr()); //any sane compiler MUST read the polymorphic header - this is due to the definition given by STD - so it is clean right after launder() - because the function is aliased to all arithmetic_t
        std::atomic_signal_fence(std::memory_order_seq_cst);

        return rs;
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto is_pow2(T value){

        return value != 0u && (value & static_cast<T>(value - 1)) == 0u;
    }

    constexpr auto align_ptr(char * buf, uintptr_t alignment_sz) noexcept -> char *{

        assert(is_pow2(alignment_sz));

        uintptr_t arithmetic_buf        = reinterpret_cast<uintptr_t>(buf);
        uintptr_t FWD_SZ                = alignment_sz - 1u;
        uintptr_t MASK_VALUE            = ~FWD_SZ;
        uintptr_t fwd_arithmetic_buf    = (arithmetic_buf + FWD_SZ) & MASK_VALUE;

        return reinterpret_cast<char *>(fwd_arithmetic_buf);
    }

    constexpr auto align_ptr(const char * buf, uintptr_t alignment_sz) noexcept -> const char *{

        assert(is_pow2(alignment_sz));

        uintptr_t arithmetic_buf        = reinterpret_cast<uintptr_t>(buf);
        uintptr_t FWD_SZ                = alignment_sz - 1u;
        uintptr_t MASK_VALUE            = ~FWD_SZ;
        uintptr_t fwd_arithmetic_buf    = (arithmetic_buf + FWD_SZ) & MASK_VALUE;

        return reinterpret_cast<const char *>(fwd_arithmetic_buf);
    }

    template <class T>
    inline __attribute__((always_inline)) auto to_const_reference(T& obj) noexcept -> decltype(auto){

        return std::as_const(obj);
    }

    template <class Destructor>
    inline auto resource_guard(Destructor destructor) noexcept{
        
        static_assert(std::is_nothrow_move_constructible_v<Destructor>);
        static_assert(std::is_nothrow_invocable_v<Destructor>);
        
        auto backout_ld = [destructor_arg = std::move(destructor)](int) noexcept{
            destructor_arg();
        };

        return dg::unique_resource<int, decltype(backout_ld)>(int{0}, std::move(backout_ld));
    }

    template <class T, class T1>
    constexpr auto pow2mod_unsigned(T lhs, T1 rhs) noexcept -> std::conditional_t<(sizeof(T) > sizeof(T1)), T, T1>{

        static_assert(std::is_unsigned_v<T>);
        static_assert(std::is_unsigned_v<T1>);
        
        using promoted_t = std::conditional_t<(sizeof(T) > sizeof(T1)), T, T1>;
        return static_cast<promoted_t>(lhs) & static_cast<promoted_t>(rhs - 1);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto ulog2_aligned(T val) noexcept -> size_t{

        return std::countr_zero(val);
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto ulog2(T val) noexcept -> T{

        return static_cast<T>(sizeof(T) * CHAR_BIT - 1u) - static_cast<T>(std::countl_zero(val));
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static constexpr auto ceil2(T val) noexcept -> T{

        if (val < 2u) [[unlikely]]{
            return 1u;
        } else [[likely]]{
            T uplog_value = ulog2(static_cast<T>(val - 1u)) + 1u;
            return T{1u} << uplog_value;
        }
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto least_pow2_greater_equal_than(T val) noexcept -> T{

        return stdx::ceil2(val);
    } 

    template <class T1, class T>
    constexpr auto safe_integer_cast(T value) noexcept -> T1{

        static_assert(std::numeric_limits<T>::is_integer);
        static_assert(std::numeric_limits<T1>::is_integer);

        if constexpr(IS_SAFE_INTEGER_CONVERSION_ENABLED){
            if constexpr(std::is_unsigned_v<T> && std::is_unsigned_v<T1>){
                (void) value;
            } else if constexpr(std::is_signed_v<T> && std::is_signed_v<T1>){
                (void) value;
            } else{
                if constexpr(std::is_signed_v<T>){
                    if constexpr(sizeof(T) > sizeof(T1)){
                        (void) value;
                    } else{
                        if (value < 0){
                            std::abort();
                        } else{
                            return value; //sizeof(signed) <= sizeof(unsigned)
                        }
                    }
                } else{
                    if constexpr(sizeof(T1) > sizeof(T)){
                        (void) value;
                    } else{
                        if (value > std::numeric_limits<T1>::max()){
                            std::abort();
                        } else{
                            return value; //sizeof(unsigned) >= sizeof(signed)
                        }
                    }
                }
            }

            if (value > std::numeric_limits<T1>::max()){
                std::abort();
            }

            if (value < std::numeric_limits<T1>::min()){
                std::abort();
            }

            return value;
        } else{
            return value;
        }
    }

    template <size_t BIT_SZ, class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto low_bit(T value) noexcept -> T{

        constexpr size_t MAX_BIT_CAP = sizeof(T) * CHAR_BIT;
        static_assert(BIT_SZ <= MAX_BIT_CAP);

        if constexpr(BIT_SZ == MAX_BIT_CAP){
            return std::numeric_limits<T>::max(); 
        } else{
            constexpr T low_mask = (T{1u} << BIT_SZ) - 1;
            return value & low_mask;
        }
    }

    template <class T, size_t BIT_SIZE, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    consteval auto lowones_bitgen(const std::integral_constant<size_t, BIT_SIZE>) noexcept -> T{

        static_assert(BIT_SIZE <= std::numeric_limits<T>::digits);

        if constexpr(BIT_SIZE == std::numeric_limits<T>::digits){
            return std::numeric_limits<T>::max();
        } else{
            return (T{1} << BIT_SIZE) - 1u;
        }
    }

    template <class T>
    struct safe_integer_cast_wrapper{

        static_assert(std::numeric_limits<T>::is_integer);
        T value;

        template <class U>
        constexpr operator U() const noexcept{

            return stdx::safe_integer_cast<U>(this->value);
        }
    };

    template <class T>
    constexpr auto wrap_safe_integer_cast(T value) noexcept{

        return stdx::safe_integer_cast_wrapper<T>{value};
    }

    template <class Iterator>
    constexpr auto advance(Iterator it, intmax_t diff) noexcept(noexcept(std::advance(it, diff))) -> Iterator{

        static_assert(std::is_nothrow_move_constructible_v<Iterator>);
        std::advance(it, diff); //I never knew what drug std was on
        return it;
    }

    template <class T>
    struct is_chrono_dur: std::false_type{};

    template <class ...Args>
    struct is_chrono_dur<std::chrono::duration<Args...>>: std::true_type{};

    template <class T>
    static inline constexpr bool is_chrono_dur_v = is_chrono_dur<T>::value;

    struct safe_timestamp_cast_wrapper{

        std::chrono::nanoseconds caster;

        constexpr safe_timestamp_cast_wrapper(std::chrono::nanoseconds caster) noexcept: caster(std::move(caster)){}

        template <class U, std::enable_if_t<stdx::is_chrono_dur_v<U>, bool> = true>
        constexpr operator U() const noexcept{

            return std::chrono::duration_cast<U>(this->caster);
        }

        template <class U, std::enable_if_t<std::numeric_limits<U>::is_integer, bool> = true>
        constexpr operator U() const noexcept{
            auto counter = caster.count();
            static_assert(std::numeric_limits<decltype(counter)>::is_integer);
            return safe_integer_cast<U>(counter);
        }
    };

    auto utc_timestamp() noexcept -> std::chrono::nanoseconds{

        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::utc_clock::now().time_since_epoch());
    }

    auto unix_timestamp() noexcept -> std::chrono::nanoseconds{

        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
    }
    
    auto unix_low_resolution_timestamp() noexcept -> std::chrono::nanoseconds{

        return {};
    }
    
    auto timestamp_conversion_wrap(std::chrono::nanoseconds dur) noexcept -> safe_timestamp_cast_wrapper{

        return safe_timestamp_cast_wrapper(dur);
    } 

    template <class ...Args>
    struct vector_convertible{

        private:

            std::tuple<Args...> tup;
        
        public:

            vector_convertible(std::tuple<Args...> tup) noexcept: tup(std::move(tup)){}

            template <class ...AArgs>
            operator std::vector<AArgs...>(){

                auto rs = std::vector<AArgs...>();
                rs.reserve(sizeof...(Args));

                [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                    (
                        [&]{
                            rs.emplace_back(std::move(std::get<IDX>(this->tup)));
                        }(), ...
                    );
                }(std::make_index_sequence<sizeof...(Args)>{});

                return rs;
            }
    };

    template <class ...Args>
    inline auto make_vector_convertible(Args ...args) noexcept -> vector_convertible<Args...>{
        
        static_assert(std::conjunction_v<std::is_nothrow_move_constructible<Args>...>);

        auto tup = std::make_tuple(std::move(args)...);
        vector_convertible rs(std::move(tup));

        return rs;
    }

    class basicstr_converitble{

        private:

            std::string_view view;

        public:

            constexpr basicstr_converitble() = default;

            constexpr basicstr_converitble(std::string_view view) noexcept: view(view){}

            template <class ...Args>
            operator std::basic_string<Args...>() const{
                
                std::basic_string<Args...> rs(view.begin(), view.end());
                return rs;
            }
    };

    inline auto to_basicstr_convertible(std::string_view view) noexcept -> basicstr_converitble{

        return basicstr_converitble(view);
    }

    template <class ...Args>
    auto backsplit_str(std::basic_string<Args...> s, size_t sz) -> std::pair<std::basic_string<Args...>, std::basic_string<Args...>>{

        size_t rhs_sz = std::min(s.size(), sz); 
        std::basic_string<Args...> rhs(rhs_sz, ' ');

        for (size_t i = rhs_sz; i != 0u; --i){
            rhs[i - 1] = s.back();
            s.pop_back();
        }

        return std::make_pair(std::move(s), std::move(rhs));
    }

    template <class ID, class T>
    class singleton{

        private:

            using self = singleton;
            static inline T * volatile obj = new T(); //this is the most important global access operation

        public:

            static inline auto get() noexcept -> T&{

                std::atomic_signal_fence(std::memory_order_seq_cst);
                return *self::obj;
            }
    };

    class VirtualResourceGuard{

        public:

            virtual ~VirtualResourceGuard() noexcept = default;
            virtual void release() noexcept = 0;
    };

    template <class ...Args>
    class UniquePtrVirtualGuard: public virtual VirtualResourceGuard{

        private:

            dg::unique_resource<Args...> resource;
        
        public:

            UniquePtrVirtualGuard(dg::unique_resource<Args...> resource) noexcept: resource(std::move(resource)){}

            void release() noexcept{

                static_assert(noexcept(this->resource.release()));
                this->resource.release();
            }
    };

    template <class Destructor>
    auto vresource_guard(Destructor destructor) -> std::unique_ptr<VirtualResourceGuard>{ //mem-exhaustion is not an error here - it's bad to have it as an error

        auto resource_grd = resource_guard(std::move(destructor));
        UniquePtrVirtualGuard virt_guard(std::move(resource_grd));

        return std::make_unique<decltype(virt_guard)>(std::move(virt_guard));
    }

    static consteval auto hdi_size() noexcept -> size_t{

        return std::max(std::hardware_destructive_interference_size, alignof(std::max_align_t));
    }

    template <size_t SZ>
    static consteval auto round_hdi_size(std::integral_constant<size_t, SZ>) noexcept -> size_t{

        size_t multiplier = SZ / hdi_size() + static_cast<size_t>(SZ % hdi_size() != 0u);
        return hdi_size() * multiplier;
    } 

    template <class T>
    struct hdi_container{
        T value;
    };

    template <class T>
    union inplace_hdi_container{
        alignas(stdx::hdi_size()) T value;
        alignas(stdx::hdi_size()) char shape[round_hdi_size(std::integral_constant<size_t, sizeof(T)>{})];

        template <class ...Args>
        inplace_hdi_container(const std::in_place_t, Args&& ...args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>): value(std::forward<Args>(args)...){}
    };

    void high_resolution_sleep(std::chrono::nanoseconds) noexcept{

    }

    template <class ...Args>
    __attribute__((noipa)) void empty_noipa(Args&& ...args) noexcept{

        (((void) args), ...);
    }

    template <class Task, class ...Args>
    __attribute__((noipa)) auto noipa_do_task(Task&& task, Args&& ...args) noexcept(std::is_nothrow_invocable_v<Task&&, Args&&...>) -> decltype(auto){

        if constexpr(std::is_same_v<decltype(task(std::forward<Args>(args)...)), void>){
            task(std::forward<Args>(args)...);
        } else{
            return task(std::forward<Args>(args)...);
        }    
    }

    //this is a very dangerous yet languagely accurate
    //we taint the value of arg and returns a pointer that could be of any values (restrictness has not been tainted, maybe ...)
    //what happens to the point and the arg value thereafter is ... in the mercy of the compiler
    //how do we make this official, by using a volatile container, such is every operation must be through the volatile container to be defined
    //this is hard
    //this is undefined as specified by Dad, we dont have a better approach

    template <class T, class ...ConsumingArgs>
    __attribute__((noipa)) auto volatile_access(T * volatile arg, ConsumingArgs& ...consuming_args) noexcept -> T *{

        (((void) consuming_args), ...);
        return arg;
    }

    //this is used as a last resort to access the immediate containee, not their dependencies, we'll mess with the compiler escape analysis very badly
    //these hacks are immediate patches, not to be used in production, including the usage of launder<>    
    //one special case for launder is producer consumer queue, where we offload the restrictness of access -> the callee, such voids all the bad accesses thru the intermediate containees

    template <class T>
    struct volatile_container{

        private:

            alignas(T) std::byte s[sizeof(T)]; //we have a major bug of destruction, the compiler does not taint the value of this even if it is marked as tained by volatile_access
                                               //the only way to make this works is to use inplace_container, std::byte
                                               //there is no such thing that makes me feel headache as overcoming compiler escape analysis + restrictness of access (we can compromise the bugs at every access level of the object)
        public:

            template <class ...Args>
            volatile_container(const std::in_place_t, Args&& ...args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>){

                new (&this->s) T(std::forward<Args>(args)...);
            }

            ~volatile_container() noexcept(std::is_nothrow_destructible_v<T>){

                T * laundered_ptr   = std::launder(reinterpret_cast<T *>(&this->s));
                T * volatiled_ptr   = stdx::volatile_access(laundered_ptr); 

                std::destroy_at(volatiled_ptr);
            }

            inline auto value() noexcept -> T *{

                return stdx::volatile_access(std::launder(reinterpret_cast<T *>(&this->s)));
            } 
    };
}

#endif
