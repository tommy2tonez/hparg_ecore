#ifndef __DG_NETWORK_UTILITY_H__
#define __DG_NETWORK_UTILITY_H__

#include <atomic>
#include <memory>
#include <mutex>
#include "network_log.h"
#include "network_exception.h" 
#include <type_traits>

namespace dg::network_genult{

    class unix_timepoint{

        private:

            std::chrono::nanoseconds ts;

            friend auto unix_timestamp() noexcept -> unix_timepoint; 
            explicit unix_timepoint(std::chrono::nanoseconds ts) noexcept: ts(std::move(ts)){}

        public:

            operator std::chrono::nanoseconds() const noexcept{

                return this->ts;
            }

            operator std::chrono::microseconds() const noexcept{

                return std::chrono::duration_cast<std::chrono::microseconds>(this->ts);
            }

            operator std::chrono::milliseconds() const noexcept{

                return std::chrono::duration_cast<std::chrono::milliseconds>(this->ts);
            }

            operator std::chrono::seconds() const noexcept{

                return std::chrono::duration_cast<std::chrono::seconds>(this->ts);
            }
    };

    auto unix_timestamp() noexcept -> unix_timepoint{

        return unix_timepoint(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()));
    }

    template <class T>
    auto safe_ptr_access(T * ptr) noexcept -> T *{

        if (ptr == nullptr){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::BAD_PTR_ACCESS));
            std::abort();
        }

        return ptr;
    }

    template <class T>
    auto safe_optional_access(std::optional<T>& obj) noexcept -> std::optional<T>&{

        if (!obj.has_value()){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::BAD_OPTIONAL_ACCESS));
            std::abort();
        }

        return obj;
    } 

    //defined for every std::lock_guard use cases
    //undefined otherwise
    auto lock_guard(std::atomic_flag& lck) noexcept{

        static int i    = 0;
        auto destructor = [&](int *) noexcept{
            lck.clear(std::memory_order_acq_rel);
        };

        while (!lck.test_and_set(std::memory_order_acq_rel)){}
        return std::unique_ptr<int, decltype(destructor)>(&i, std::move(destructor));
    }

    //defined for every std::lock_guard use cases
    //undefined otherwise
    auto lock_guard(std::mutex& lck) noexcept{

        static int i    = 0;
        auto destructor = [&](int *) noexcept{
            lck.unlock();
        };

        lck.lock();
        return std::unique_ptr<int, decltype(destructor)>(&i, std::move(destructor));
    }

    template <class ID, class T>
    class singleton{

        private:

            static inline T obj{};
        
        public:

            static inline auto get() noexcept -> T&{

                return obj;
            }
    }

    template <class Functor, class Tup, class = void>
    struct is_nothrow_invokable_helper: std::false_type{};

    template <class Functor, class ...Args>
    struct is_nothrow_invokable_helper<Functor, std::tuple<Args...>, std::enable_if_t<noexcept(std::declval<Functor>()(std::declval<Args>()...))>>: std::true_type{}; 

    template <class Functor, class ...Args>
    struct is_nothrow_invokable: is_nothrow_invokable_helper<Functor, std::tuple<Args...>>{}; 

    template <class Functor, class ...Args>
    static inline constexpr bool is_nothrow_invokable_v = is_nothrow_invokable<Functor, Args...>::value; 
    
    template <class ResourceType, class ResourceDeallocator, class = void>
    class nothrow_immutable_unique_raii_wrapper{}; 

    template <class ResourceType, class ResourceDeallocator>
    class nothrow_immutable_unique_raii_wrapper<ResourceType, ResourceDeallocator, std::void_t<std::enable_if_t<std::conjunction_v<std::is_trivial<ResourceType>,
                                                                                                                                   is_nothrow_invokable<ResourceDeallocator, std::add_lvalue_reference_t<ResourceType>>, 
                                                                                                                                   std::is_nothrow_move_constructible<ResourceDeallocator>>>>>{

        private:

            ResourceType resource;
            ResourceDeallocator deallocator;
            bool responsibility_flag

        public:

            using self = nothrow_immutable_unique_raii_wrapper;

            explicit nothrow_immutable_unique_raii_wrapper(ResourceType resource,
                                                           ResourceDeallocator deallocator) noexcept: resource(resource),
                                                                                                      deallocator(std::move(deallocator)),
                                                                                                      responsibility_flag(true){}
            
            nothrow_immutable_unique_raii_wrapper(const self&) = delete;
            nothrow_immutable_unique_raii_wrapper(self&& other) noexcept: resource(other.resource),
                                                                          deallocator(std::move(other.deallocator)),
                                                                          responsibility_flag(other.responsibility_flag){
                other.responsibility_flag = false;
            }

            self& operator =(const self&) = delete;

            self& operator =(self&& other) noexcept{

                if (this != std::addressof(other)){
                    this->release_responsibility();
                    this->resource              = other.resource;
                    this->deallocator           = std::move(other.deallocator);
                    this->responsibility_flag   = other.responsibility_flag;
                    other.responsibility_flag   = false;
                }

                return *this;
            }

            ~nothrow_immutable_unique_raii_wrapper() noexcept{

                this->release_responsibility();
            }

            auto value() const noexcept -> ResourceType{

                if (!this->responsibility_flag){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::BAD_RAII_ACCESS));
                    std::abort();
                }

                return this->resource;
            }

            auto has_value() const noexcept -> bool{

                return this->responsibility_flag;
            }
        
        private:

            void release_responsibility() noexcept{
                                
                if (this->responsibility_flag){
                    this->deallocator(this->resource);
                    this->responsibility_flag = false;
                }
            }
    };
}

#endif