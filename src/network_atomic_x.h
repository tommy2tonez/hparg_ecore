#ifndef __NETWORK_ATOMIC_X_H__
#define __NETWORK_ATOMIC_X_H__

#include <atomic>

namespace dg::network_atomic_x{

    //atomicity is an extremely hard problem that std and the implementors did not take the right turn
    //atomic variable and friends could be loosely explained as quantum entanglement
    //when the state of a variable is "deducable" by another state of variable via the usage of release + acquire
    
    //release guarantees that the releasing local variable could be used for deducing their friends in the global_context
    //acquire guarantees that the acquiring global variable could be used for deducing their friends in the local_context

    //the friends definition is not done correctly by the std and the implementors
    //it's almost extremely hard - impossible to build a depedency tree with the current implementation

    static inline constexpr bool IS_STRONG_UB_PRUNE_ENABLED = true;

    void dg_thread_fence_optional() noexcept{

        if constexpr(IS_STRONG_UB_PRUNE_ENABLED){
            std::atomic_thread_fence(std::memory_order_acq_rel);
        } else{
            (void) dg_thread_fence_optional;
        }
    }

    void dg_thread_fence() noexcept{

        std::atomic_thread_fence(std::memory_order_acq_rel);
    }

    template <class T>
    auto dg_compare_exchange_strong_acqrel(std::atomic<T>& obj, T expected, T new_value) noexcept -> bool{

        static_assert(std::is_trivial_v<T>);
        dg_thread_fence_optional();
        decltype(auto) rs = obj.compare_exchange_strong(obj, expected, new_value, std::memory_order_acq_rel);
        dg_thread_fence_optional();

        return rs;
    }

    template <class T>
    auto dg_compare_exchange_strong_release(std::atomic<T>& obj, T expected, T new_value) noexcept -> bool{

        static_assert(std::is_trivial_v<T>);
        dg_thread_fence_optional();

        return obj.compare_exchange_strong(obj, expected, new_value, std::memory_order_release);
    } 

    template <class T>
    auto dg_compare_exchange_weak_acqrel(std::atomic<T>& obj, T expected, T new_value) noexcept -> bool{

        static_assert(std::is_trivial_v<T>);
        dg_thread_fence_optional();
        decltype(auto) rs =  obj.compare_exchange_weak(obj, expected, new_value, std::memory_order_acq_rel);
        dg_thread_fence_optional();

        return rs;
    } 

    template <class T>
    auto dg_compare_exchange_weak_release(std::atomic<T>& obj, T expected, T new_value) noexcept -> bool{

        static_assert(std::is_trivial_v<T>);
        dg_thread_fence_optional();
        decltype(auto) rs = obj.compare_exchange_weak(obj, expected, new_value, std::memory_order_release);
        dg_thread_fence_optional();

        return rs;
    }

    template <class T>
    auto dg_load_relaxed(std::atomic<T>& obj) noexcept -> T{

        static_assert(std::is_trivial_v<T>);
        return obj.load(std::memory_order_relaxed);
    }

    template <class T>
    auto dg_load_acqrel(std::atomic<T>& obj) noexcept -> T{

        static_assert(std::is_trivial_v<T>);
        dg_thread_fence_optional();
        decltype(auto) rs = obj.load(std::memory_order_acq_rel);
        dg_thread_fence_optional();
        
        return rs;
    }

    template <class T>
    auto dg_load_acquire(std::atomic<T>& obj) noexcept -> T{

        static_assert(std::is_trivial_v<T>);
        decltype(auto) rs = obj.load(std::memory_order_acquire);
        dg_thread_fence_optional();
        
        return rs;
    }

    template <class T, class Arg>
    auto dg_exchange_relaxed(std::atomic<T>& obj, Arg&& arg) noexcept -> T{

        static_assert(std::is_trivial_v<T>);
        static_assert(std::is_trivial_v<std::remove_reference_t<Arg>>);
        return obj.exchange(std::forward<Arg>(arg), std::memory_order_relaxed);
    }

    template <class T, class Arg>
    auto dg_exchange_acqrel(std::atomic<T>& obj, Arg&& arg) noexcept -> T{

        static_assert(std::is_trivial_v<T>);
        static_assert(std::is_trivial_v<std::remove_reference_t<Arg>>);
        dg_thread_fence_optional();
        decltype(auto) rs = obj.exchange(std::forward<Arg>(arg), std::memory_order_acq_rel);
        dg_thread_fence_optional();

        return rs;
    } 

    template <class T, class Arg>
    auto dg_exchange_release(std::atomic<T>& obj, Arg&& arg) noexcept -> T{

        static_assert(std::is_trivial_v<T>);
        static_assert(std::is_trivial_v<std::remove_reference_t<Arg>>);
        dg_thread_fence_optional();
      
        return obj.exchange(std::forward<Arg>(arg), std::memory_order_release);
    }
} 

#endif