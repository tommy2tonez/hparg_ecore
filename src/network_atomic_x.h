#ifndef __NETWORK_ATOMIC_X_H__
#define __NETWORK_ATOMIC_X_H__

#include <atomic>
#include <type_traits>
#include <stdlib.h>
#include <stdint.h>

namespace dg::network_atomic_x{
    
    template <class = void>
    static inline constexpr bool FALSE_VAL                  = false;

    static inline constexpr bool IS_STRONG_UB_PRUNE_ENABLED = true;
    static inline constexpr auto dg_memory_order_acqrel     = std::integral_constant<size_t, 0>{};
    static inline constexpr auto dg_memory_order_release    = std::integral_constant<size_t, 1>{};
    static inline constexpr auto dg_memory_order_acquire    = std::integral_constant<size_t, 2>{};
    static inline constexpr auto dg_memory_order_seqcst     = std::integral_constant<size_t, 3>{};
    static inline constexpr auto dg_memory_order_relaxed    = std::integral_constant<size_t, 4>{}; 

    void dg_thread_fence() noexcept{

        std::atomic_thread_fence(std::memory_order_acq_rel);
    }

    void dg_thread_fence_optional() noexcept{

        if constexpr(IS_STRONG_UB_PRUNE_ENABLED){
            dg_thread_fence();
        } else{
            (void) dg_thread_fence_optional;
        }
    }

    template <class T>
    auto dg_internal_compare_exchange_strong_acqrel(std::atomic<T>& obj, T expected, T new_value) noexcept -> bool{

        static_assert(std::is_trivial_v<T>); //this is a stricter req to catch performance constraints + force noexceptability
        dg_thread_fence_optional();
        bool rs = obj.compare_exchange_strong(obj, expected, new_value, std::memory_order_acq_rel);
        dg_thread_fence_optional();

        return rs;
    }

    template <class T>
    auto dg_internal_compare_exchange_strong_release(std::atomic<T>& obj, T expected, T new_value) noexcept -> bool{

        static_assert(std::is_trivial_v<T>); //this is a stricter req to catch performance constraints + force noexceptability
        dg_thread_fence_optional();

        return obj.compare_exchange_strong(obj, expected, new_value, std::memory_order_release);
    } 

    template <class T, size_t DISPATCH_CODE = static_cast<size_t>(dg_memory_order_seqcst)>
    auto dg_compare_exchange_strong(std::atomic<T>& obj, T expected, T new_value, const std::integral_constant<size_t, DISPATCH_CODE> = std::integral_constant<size_t, DISPATCH_CODE>{}) noexcept -> bool{

        if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_acqrel)){
            return dg_internal_compare_exchange_strong_acqrel(obj, expected, new_value);
        } else if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_seqcst)){
            return dg_internal_compare_exchange_strong_acqrel(obj, expected, new_value);
        } else if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_release)){
            return dg_internal_compare_exchange_strong_release(obj, expected, new_value);
        } else{
            static_assert(FALSE_VAL<>); //PRECOND: only compatible with gcc (fix)
        }
    }

    template <class T>
    auto dg_internal_compare_exchange_weak_acqrel(std::atomic<T>& obj, T expected, T new_value) noexcept -> bool{

        static_assert(std::is_trivial_v<T>); //this is a stricter req to catch performance constraints + force noexceptability
        dg_thread_fence_optional();
        bool rs = obj.compare_exchange_weak(obj, expected, new_value, std::memory_order_acq_rel);
        dg_thread_fence_optional();

        return rs;
    } 

    template <class T>
    auto dg_internal_compare_exchange_weak_release(std::atomic<T>& obj, T expected, T new_value) noexcept -> bool{

        static_assert(std::is_trivial_v<T>); //this is a stricter req to catch performance constraints + force noexceptability
        dg_thread_fence_optional();
        bool rs = obj.compare_exchange_weak(obj, expected, new_value, std::memory_order_release);

        return rs;
    }

    template <class T, size_t DISPATCH_CODE = static_cast<size_t>(dg_memory_order_seqcst)>
    auto dg_compare_exchange_weak(std::atomic<T>& obj, T expected, T new_value, const std::integral_constant<size_t, DISPATCH_CODE> = std::integral_constant<size_t, DISPATCH_CODE>{}) noexcept -> bool{

        if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_acqrel)){
            return dg_internal_compare_exchange_weak_acqrel(obj, expected, new_value);
        } else if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_seqcst)){
            return dg_internal_compare_exchange_weak_acqrel(obj, expected, new_value);
        } else if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_release)){
            return dg_internal_compare_exchange_weak_release(obj, expected, new_value);
        } else{
            static_assert(FALSE_VAL<>);
        }
    }

    template <class T>
    auto dg_load_relaxed(std::atomic<T>& obj) noexcept -> T{

        static_assert(std::is_trivial_v<T>); //this is a stricter req to catch performance constraints + force noexceptability
        return obj.load(std::memory_order_relaxed);
    }

    template <class T>
    auto dg_load_acqrel(std::atomic<T>& obj) noexcept -> T{

        static_assert(std::is_trivial_v<T>); //this is a stricter req to catch performance constraints + force noexceptability
        dg_thread_fence_optional();
        T rs = obj.load(std::memory_order_acq_rel);
        dg_thread_fence_optional();
        
        return rs;
    }

    template <class T>
    auto dg_load_acquire(std::atomic<T>& obj) noexcept -> T{

        static_assert(std::is_trivial_v<T>); //this is a stricter req to catch performance constraints + force noexceptability
        T rs = obj.load(std::memory_order_acquire);
        dg_thread_fence_optional();
        
        return rs;
    }

    template <class T, size_t DISPATCH_CODE = static_cast<size_t>(dg_memory_order_seqcst)>
    auto dg_load(std::atomic<T>& obj, const std::integral_constant<size_t, DISPATCH_CODE> = std::integral_constant<size_t, DISPATCH_CODE>{}) noexcept -> T{

        if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_relaxed)){
            return dg_load_relaxed(obj);
        } else if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_acqrel)){
            return dg_load_acqrel(obj);
        } else if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_acquire)){
            return dg_load_acquire(obj);
        } else if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_seqcst)){
            return dg_load_acquire(obj);
        } else{
            static_assert(FALSE_VAL<>); //PRECOND: only compatible with gcc (fix)
        }
    }

    template <class T, class Arg>
    auto dg_internal_exchange_relaxed(std::atomic<T>& obj, Arg&& arg) noexcept -> T{

        static_assert(std::is_trivial_v<T>); //trivial should suffice here
        return obj.exchange(std::forward<Arg>(arg), std::memory_order_relaxed);
    }

    template <class T, class Arg>
    auto dg_internal_exchange_acqrel(std::atomic<T>& obj, Arg&& arg) noexcept -> T{

        static_assert(std::is_trivial_v<T>); //this is a stricter req to catch performance constraints + force noexceptability
        dg_thread_fence_optional();
        T rs = obj.exchange(std::forward<Arg>(arg), std::memory_order_acq_rel);
        dg_thread_fence_optional();

        return rs;
    } 

    template <class T, class Arg>
    auto dg_internal_exchange_release(std::atomic<T>& obj, Arg&& arg) noexcept -> T{

        static_assert(std::is_trivial_v<T>); //this is a stricter req to catch performance constraints + force noexceptability
        dg_thread_fence_optional();

        return obj.exchange(std::forward<Arg>(arg), std::memory_order_release);
    }

    template <class T, class Arg, size_t DISPATCH_CODE = static_cast<size_t>(dg_memory_order_seqcst)>
    auto dg_exchange(std::atomic<T>& obj, Arg&& arg, const std::integral_constant<size_t, DISPATCH_CODE> = std::integral_constant<size_t, DISPATCH_CODE>{}) noexcept -> T{

        if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_relaxed)){
            return dg_internal_exchange_relaxed(obj, std::forward<Arg>(arg));
        } else if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_acqrel)){
            return dg_internal_exchange_acqrel(obj, std::forward<Arg>(arg));
        } else if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_release)){
            return dg_internal_exchange_release(obj, std::forward<Arg>(arg));
        } else if constexpr(DISPATCH_CODE == static_cast<size_t>(dg_memory_order_seqcst)){
            return dg_internal_exchange_acqrel(obj, std::forward<Arg>(arg)); //this actually does rcu
        } else{
            static_assert(FALSE_VAL<>); //PRECOND: only compatible with gcc (fix)
        }
    }
} 

#endif