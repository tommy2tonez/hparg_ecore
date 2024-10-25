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

namespace stdx{
    
    static inline constexpr bool IS_SAFE_MEMORY_ORDER_ENABLED = true; 

    template <class T>
    struct NoExceptAllocator: private std::allocator<T>{
        
        using value_type                                = T;
        using pointer                                   = T *;
        using const_pointer                             = const T *;
        using reference                                 = T&;
        using const_reference                           = const T&;
        using size_type                                 = size_t;
        using difference_type                           = intmax_t;
        using is_always_equal                           = std::true_type;
        using propagate_on_container_move_assignment    = std::true_type;
        
        template <class U>
        struct rebind{
            using other = NoExceptAllocator<U>;
        };

        auto address(reference x) const noexcept -> pointer{

            return std::allocator<T>::address(x);
        }

        auto address(const_reference x) const noexcept -> const_pointer{

            return std::allocator<T>::address(x);
        }
        
        auto allocate(size_t n, const void * hint) -> pointer{ //noexcept is guaranteed internally - this is to comply with std

            if (n == 0u){
                return nullptr;
            }

            pointer rs = std::allocator<T>::allocate(n, hint);

            if (!rs){                
                std::abort();
            }

            return rs;
        }

        auto allocate(size_t n) -> pointer{
            
            return std::allocator<T>::allocate(n);
        }
        
        //according to std - deallocate arg is valid ptr - such that allocate -> std::optional<ptr_type>, void deallocate(ptr_type)
        void deallocate(pointer p, size_t n){ //noexcept is guaranteed internally - this is to comply with std

            if (n == 0u){
                return;
            }

            std::allocator<T>::deallocate(p, n);
        }

        consteval auto max_size() const noexcept -> size_type{

            return std::allocator<T>::max_size();
        }
        
        template <class U, class... Args>
        void construct(U * p, Args&&... args) noexcept(std::is_nothrow_constructible_v<U, Args...>){

            return std::allocator<T>::construct(p, std::forward<Args>(args)...);
        }

        template <class U>
        void destroy(U * p) noexcept(std::is_nothrow_destructible_v<U>){

            std::allocator<T>::destroy(p);
        }
    };

    template <class T>
    using vector            = std::vector<T, NoExceptAllocator<T>>;

    template <class T>
    using string            = std::basic_string<char, std::char_traits<char>, NoExceptAllocator<char>>;

    template <class Key, class Value, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>>
    using unordered_map     = std::unordered_map<Key, Value, Haser, Pred, NoExceptAllocator<std::pair<const Key, Value>>>;

    template <class Key, class Hasher = std::hash<Key>, class Pred = std::equal_to<Key>>
    using unordered_set     = std::unordered_set<Key, Hasher, Pred, NoExceptAllocator<Key>>;

    template <class Key, class Value, class Cmp = std::less<Key>>
    using map               = std::map<Key, Value, Cmp, NoExceptAllocator<std::pair<const Key, Value>>;

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

        //atomic operations don't work - its a lame implementation of mutual exclusion in most operating system
        //use atomic flag and do your atomic operations foo - this way you compromise concurrent memory access at lock_guard which can do memory flush for you  

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
}

#endif