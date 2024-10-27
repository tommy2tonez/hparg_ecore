#ifndef __STD_X_H__
#define __STD_X_H__

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

namespace stdx{
    
    //verison controls are cross-the-t-dot-the-i tasks - not to worry now 

    using max_signed_t = __int128_t; //macro

    static inline constexpr bool IS_SAFE_MEMORY_ORDER_ENABLED       = true; 
    static inline constexpr bool IS_SAFE_INTEGER_CONVERSION_ENABLED = true;

    #if __cplusplus >= 202002L
        #if __cplusplus <= 202302L

        template <class T>
        struct NoExceptAllocator: public std::allocator<T>{
            
            constexpr auto allocate(size_t n) -> T *{
                
                try{
                    return std::allocator<T>::allocate(n);
                } catch (std::bad_alloc& e){
                    std::abort();
                }

                return {};
            }
            
            constexpr void deallocate(T * p, size_t n){
                
                std::allocator<T>::deallocate(p, n);
            }
        };

        #endif
    #endif
    
    template <class T>
    using vector            = std::vector<T, NoExceptAllocator<T>>;

    using string            = std::basic_string<char, std::char_traits<char>, NoExceptAllocator<char>>;

    template <class Key, class Value, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>>
    using unordered_map     = std::unordered_map<Key, Value, Hasher, Pred, NoExceptAllocator<std::pair<const Key, Value>>>;

    template <class Key, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>>
    using unordered_set     = std::unordered_set<Key, Hasher, Pred, NoExceptAllocator<Key>>;

    template <class Key, class Value, class Cmp = std::less<Key>>
    using map               = std::map<Key, Value, Cmp, NoExceptAllocator<std::pair<const Key, Value>>>;

    template <class Key, class Cmp = std::less<Key>>
    using set               = std::set<Key, Cmp, NoExceptAllocator<Key>>;

    template <class T>
    using deque             = std::deque<T, NoExceptAllocator<T>>;

    template <class T, class Deleter = std::default_delete<T>>
    using unique_ptr        = std::unique_ptr<T, Deleter>;
    
    template <class T>
    using shared_ptr        = std::shared_ptr<T>;

    template <class T, class ...Args>
    auto make_unique(Args&& ...args){

        return std::make_unique<T>(std::forward<Args>(args)...);
    }

    template <class T, class ...Args>
    auto make_shared(Args&& ...args){

        return std::make_shared<T>(std::forward<Args>(args)...);
    }

    auto lock_guard(std::atomic_flag& lck) noexcept{

        static int i    = 0u;
        auto destructor = [&](int *) noexcept{
            if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
                std::atomic_thread_fence(std::memory_order_acq_rel);
            }
            lck.clear(std::memory_order_release);
        };

        while (!lck.test_and_set(std::memory_order_acquire)){}
        if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
            std::atomic_thread_fence(std::memory_order_acq_rel);
        }  

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    auto lock_guard(std::mutex& lck) noexcept{

        static int i    = 0u;
        auto destructor = [&](int *) noexcept{
            if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
                std::atomic_thread_fence(std::memory_order_acq_rel);
            }
            lck.unlock();
        };

        lck.lock();
        if constexpr(IS_SAFE_MEMORY_ORDER_ENABLED){
            std::atomic_thread_fence(std::memory_order_acq_rel);
        }

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    }

    template <class Destructor>
    auto resource_guard(Destructor destructor) noexcept{
        
        static_assert(std::is_nothrow_move_constructible_v<Destructor>);
        static_assert(std::is_nothrow_invocable_v<Destructor>);

        static int i    = 0;
        auto backout_ld = [destructor_arg = std::move(destructor)](int *) noexcept{
            destructor_arg();
        };

        return std::unique_ptr<int, decltype(backout_ld)>(&i, std::move(backout_ld));
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
    constexpr auto ulog2(T val) noexcept -> size_t{

        return static_cast<size_t>(sizeof(T) * CHAR_BIT - 1) - static_cast<size_t>(std::countl_zero(val));
    }

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto is_pow2(T val) noexcept -> bool{

        return val != 0u && (val & (val - 1)) == 0u;
    } 

    template <class T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto least_pow2_greater_equal_than(T val) noexcept -> T{

        if (val == 0u){ [[unlikely]]
            return 1u;
        }

        size_t max_log2     = ulog2(val);
        size_t min_log2     = std::countr_zero(val);
        size_t cand_log2    = max_log2 + ((max_log2 ^ min_log2) != 0u);

        return T{1u} << cand_log2; 
    } 

    template <class T, class T1>
    constexpr auto safe_integer_cast(T1 value) noexcept -> T{

        static_assert(std::numeric_limits<T>::is_integer);
        static_assert(std::numeric_limits<T1>::is_integer);

        if constexpr(IS_SAFE_INTEGER_CONVERSION_ENABLED){
            using promoted_t = stdx::max_signed_t; 

            static_assert(sizeof(promoted_t) > sizeof(T));
            static_assert(sizeof(promoted_t) > sizeof(T1));

            if (std::clamp(static_cast<promoted_t>(value), static_cast<promoted_t>(std::numeric_limits<T>::min()), static_cast<promoted_t>(std::numeric_limits<T>::max())) != static_cast<promoted_t>(value)){
                std::abort();
            }
        }

        return value;
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

    template <class T>
    struct safe_integer_cast_wrapper{

        static_assert(std::numeric_limits<T>::is_integer);
        T value;

        template <class U>
        constexpr operator U() const noexcept{

            return safe_integer_cast<U>(this->value);
        }
    };

    template <class T>
    constexpr auto wrap_safe_integer_cast(T value) noexcept{

        return safe_integer_cast_wrapper<T>{value};
    }

    template <class Iterator>
    constexpr auto advance(Iterator it, intmax_t diff) noexcept -> Iterator{

        std::advance(it, diff); //I never knew what drug was std on
        return it;
    }

    auto utc_timestamp() noexcept -> std::chrono::nanoseconds{

    }

    auto unix_timestamp() noexcept -> std::chrono::nanoseconds{

    }
}

#endif