#ifndef __STD_X_H__
#define __STD_X_H__

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace stdx{

    template <class T>
    struct NoExceptAllocator: protected std::allocator<T>{
        
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


}

#endif