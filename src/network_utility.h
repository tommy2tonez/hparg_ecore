#ifndef __DG_NETWORK_UTILITY_H__
#define __DG_NETWORK_UTILITY_H__

#include <atomic>
#include <memory>
#include <mutex>
#include "network_log.h"
#include "network_exception.h" 
#include <type_traits>
#include "network_atomic_x.h"
#include "network_type_traits_x.h"

namespace dg::network_genult{

    static inline constexpr bool IS_SAFE_ACCESS_ENABLED = true; 

    class unix_timepoint{

        private:

            std::chrono::nanoseconds ts;

            friend auto unix_timestamp() noexcept -> unix_timepoint; 
            friend auto utc_timestamp() noexcept -> unix_timepoint;
            explicit unix_timepoint(std::chrono::nanoseconds ts) noexcept: ts(std::move(ts)){}

        public:

            constexpr operator std::chrono::nanoseconds() const noexcept{

                return this->ts;
            }

            constexpr operator std::chrono::microseconds() const noexcept{

                return std::chrono::duration_cast<std::chrono::microseconds>(this->ts);
            }

            constexpr operator std::chrono::milliseconds() const noexcept{

                return std::chrono::duration_cast<std::chrono::milliseconds>(this->ts);
            }

            constexpr operator std::chrono::seconds() const noexcept{

                return std::chrono::duration_cast<std::chrono::seconds>(this->ts);
            }
    };

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

    auto unix_timestamp() noexcept -> unix_timepoint{

        return unix_timepoint(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()));
    }

    auto utc_timestamp() noexcept -> unix_timepoint{

    }

    template <class T>
    auto safe_ptr_access(T * ptr) noexcept -> T *{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            if (ptr == nullptr){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::BAD_PTR_ACCESS));
                std::abort();
            }
        }

        return ptr;
    }

    template <class T, class T1>
    constexpr auto safe_integer_cast(T1 value) noexcept -> T{

        static_assert(std::numeric_limits<T>::is_integer);
        static_assert(std::numeric_limits<T1>::is_integer);

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            using promoted_t = dg::max_signed_t; 

            static_assert(sizeof(promoted_t) > sizeof(T));
            static_assert(sizeof(promoted_t) > sizeof(T1));

            if (std::clamp(static_cast<promoted_t>(value), static_cast<promoted_t>(std::numeric_limits<T>::min()), static_cast<promoted_t>(std::numeric_limits<T>::max())) != static_cast<promoted_t>(value)){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                std::abort();
            }
        }

        return value;
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

    template <class T>
    auto safe_optional_access(std::optional<T>& obj) noexcept -> std::optional<T>&{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            if (!obj.has_value()){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::BAD_OPTIONAL_ACCESS));
                std::abort();
            }
        }

        return obj;
    } 

    template <class ID, class T>
    class singleton{

        private:

            static_assert(std::is_nothrow_default_constructible_v<T>);
            static inline std::unique_ptr<T> obj = std::make_unique<T>();
        
        public:

            static inline auto get() noexcept -> T&{

                return *obj;
            }
    };

    template <class ID, class T>
    class serialized_singleton: private singleton<ID, T>{

        private:

            using base = singleton<ID, T>; 
            static inline std::mutex mtx{}; 

        public:

            template <class U>
            static inline auto assign(U&& arg) noexcept(std::is_nothrow_assignable_v<T&, U>){
                
                auto lck_grd = lock_guard(mtx);
                base::get() = std::forward<U>(arg);
            }

            template <class Functor, class ...Args>
            static inline auto invoke(Functor functor, Args&& ...args) noexcept(noexcept((base::get().*functor)(std::forward<Args>(args...)))) -> decltype(auto){

                using ret_t = decltype((base::get().*functor)(std::forward<Args>(args)...)); 
                auto lck_grd = lock_guard(mtx); 

                if constexpr(std::is_same_v<ret_t, void>){
                    (base::get().*functor)(std::forward<Args>(args)...);
                } else{
                    return (base::get().*functor)(std::forward<Args>(args)...);
                }
            }
    };

    template <class T, class = void>
    struct mono_reduce_or{};

    template <class T>
    struct mono_reduce_or<T, std::void_t<std::enable_if_t<std::numeric_limits<T>::is_integer>>>{
 
        constexpr auto operator()(T lhs, T rhs) const noexcept -> T{

            return lhs | rhs;
        }
    };

    template <class T, class = void>
    struct mono_reduce_and{};

    template <class T>
    struct mono_reduce_and<T, std::void_t<std::enable_if_t<std::numeric_limits<T>::is_integer>>>{

        constexpr auto operator()(T lhs, T rhs) const noexcept -> T{

            return lhs & rhs;
        }
    };

    //correct:
    //sfinae are usually used for two things: 
    //first:    validate if the providing args are compile-time valid for a specific function
    //second:   different dispatch for different types

    //(first):  if pure auto return type == no header precond required (std::enable_if_t<precond, bool> = true)
    //(first):  if explicit return type then header precond declaration is required
    //(second): if the function name is tuple_reduce, second usage of sfinae is not required(precond as function signature)  
    //(second): if the function name is reduce - then sfinae is required for tuple dispatch - in this case function reduce with sfinae header invokes tuple_reduce 
    //(second): never do type overloading when sfinae header could be used - to disable implicit init (this is where most developer fails)
    //add(double, dobule), add(size_t, size_t) -> add(T, T) enable_if_t<std::is_same_v<T, double>, bool> = true

    template <class T, class Functor>
    auto tuple_reduce(T&& tup, Functor&& functor){ //noexcept(auto) - feature request 
        
        using tup_t = dg::network_type_traits_x::base_type_t<T>;
        static_assert(std::tuple_size_v<tup_t> != 0u);

        auto lambda = [&]<class Self, size_t IDX>(Self self, const std::integral_constant<size_t, IDX>){ //Self -> Self& next iteration, feature request decltype(auto) return type - wrong otherwise
            if constexpr(IDX == std::tuple_size_v<tup_t> - 1){
                return std::get<IDX>(tup);
            } else{
                return functor(std::get<IDX>(tup), lambda(self, std::integral_constant<size_t, IDX + 1>{}));
            }
        };

        return lambda(lambda, std::integral_constant<size_t, 0u>{});
    }

    template <class ...Args>
    auto tuple_join(Args&& ...args){ //noexcept(auto) - feature request

        static_assert(sizeof...(Args) != 0u);

        auto fwd_tup    = std::forward_as_tuple(std::forward<Args>(args)...); 
        auto lambda     = [&]<class Self, size_t IDX>(Self self, const std::integral_constant<size_t, IDX>){ //capture fwd_tup as [=], & is slow - next iteration, Self -> Self&
            if constexpr(IDX == sizeof...(Args) - 1){
                //good to put a precond here - to catch undefined behavior compile-time - this, however, is not required - tuple_join semantically means that (type(args) == type(tuple_and_friends))...
                static_assert(dg::network_type_traits_x::is_tuple_v<dg::network_type_traits_x::base_type_t<std::tuple_element_t<IDX, decltype(fwd_tup)>>);
                return std::get<IDX>(fwd_tup);
            } else{
                auto successor                      = self(self, std::integral_constant<size_t, IDX + 1>{});
                auto cur                            = std::get<IDX>(fwd_tup); //should be decltype(auto) - not necessary
                using successor_t                   = decltype(successor);
                using cur_t                         = decltype(cur);
                constexpr size_t successor_tuple_sz = std::tuple_size_v<successor_t>;
                constexpr size_t cur_tuple_sz       = std::tuple_size_v<cur_t>>;

                return [&]<size_t ...LHS_IDX, size_t ...RHS_IDX>(const std::index_sequence<LHS_IDX...>, const std::index_sequence<RHS_IDX...>){
                    static_assert(std::conjunction_v<dg::network_type_traits_x::is_base_type<std::tuple_element_t<LHS_IDX, successor_t>>...>);
                    static_assert(std::conjunction_v<dg::network_type_traits_x::is_base_type<std::tuple_element_t<RHS_IDX, cur_t>>...>);
                    return std::make_tuple(std::get<LHS_IDX>(successor)..., std::get<RHS_IDX>(cur)...); //make sure that all joining element_types are base_type - lesser the requirement next iteration 
                }(std::make_index_sequence<successor_tuple_sz>{}, std::make_index_sequence<cur_tuple_sz>{});
            }
        };

        return lambda(lambda, std::integral_constant<size_t, 0u>{});
    }

    template <class T, size_t SZ>
    auto tuple_peek(T&& tup, const std::integral_constant<size_t, SZ>){ //noexcept(auto) - feature request

        static_assert(SZ != 0u);
        using tup_t = dg::network_type_traits_x::base_type_t<T>; 

        return [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
            static_assert(std::conjunction_v<dg::network_type_traits_x::is_base_type<std::tuple_element_t<IDX, tup_t>>...>);
            return std::make_tuple(std::get<IDX>(tup)...); //consider forward_as_tuple next iteration -
        }(std::make_index_sequence<SZ>{});
    }

    template <class T, class FirstIArg, class ...IArgs>
    auto tuple_peek_many(T&& arg, FirstIArg first_iarg, IArgs... iargs){ //noexcept(auto) - feature request

        if constexpr(sizeof...(IArgs) == 0u){
            return tuple_peek(arg, first_iarg);
        } else{
            return tuple_join(std::make_tuple(tuple_peek(arg, first_iarg)), tuple_peek_many(arg, iargs...));
        }
    }

    template <class T, class T1>
    auto tuple_zip(T&& lhs, T1&& rhs){ //noexcept(auto) - feature request 

        using lhs_tup_t = dg::network_type_traits_x::base_type_t<T>;
        using rhs_tup_t = dg::network_type_traits_x::base_type_t<T1>;

        static_assert(std::tuple_size_v<lhs_tup_t> == std::tuple_size_v<rhs_tup_t>);

        return [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
            static_assert(std::conjunction_v<dg::network_type_traits_x::is_base_type<std::tuple_element_t<IDX, lhs_tup_t>>...>); //stricter req - remove next iteration
            static_assert(std::conjunction_v<dg::network_type_traits_x::is_base_type<std::tuple_element_t<IDX, rhs_tup_t>>...>); //stricter req - remove next iteration
            return std::make_tuple(std::make_tuple(std::get<IDX>(lhs), std::get<IDX>(rhs))...);
        }(std::make_index_sequence<std::tuple_size_v<lhs_tup_t>>{});
    }

    template <class T, class Functor>
    auto tuple_transform(T&& tup, Functor&& functor){ //noexcept(auto) - feature request

        using tup_t = dg::network_type_traits_x::base_type_t<T>; //
        
        return [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
            static_assert(std::conjunction_v<std::is_same<dg::network_type_traits_x::base_type_t<decltype(functor(std::get<IDX>(tup)))>, decltype(functor(std::get<IDX>(tup)))>...>);
            return std::make_tuple(functor(std::get<IDX>(tup))...);
        }(std::make_index_sequence<std::tuple_size_v<tup_t>>{});
    }

    template <class T>
    auto tuple_to_array(T&& tup){ //noexcept(auto) - feature request
        
        using tup_t = dg::network_type_traits_x::base_type_t<T>;

        return [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
            using type = dg::network_type_traits_x::mono_reduction_type_t<std::tuple_element_t<IDX, tup_t>...>; 
            static_assert(std::is_same_v<type, dg::network_type_traits_x::base_type_t<type>>); //stricter_req - remove next iteration
            return std::array<type, std::tuple_size_v<tup_t>>{std::get<IDX>(tup)...};
        }(std::make_index_sequence<std::tuple_size_v<tup_t>>{});
    }

    template <class ...Args>
    auto is_same_value(Args&& ...args) -> bool{ //noexcept(auto) - feature request
        
        auto fwd_tup    = std::forward_as_tuple(args...); 
        auto lambda     = [&]<class Self, size_t IDX>(Self self, const std::integral_constant<size_t, IDX>){
            if constexpr(IDX == sizeof...(Args)){
                return true;
            } else{
                if constexpr(IDX != 0u){
                    using ret_t = decltype(std::get<IDX - 1>(fwd_tup) == std::get<IDX>(fwd_tup));
                    static_assert(std::is_same_v<ret_t, bool>);
                    return (std::get<IDX - 1>(fwd_tup) == std::get<IDX>(fwd_tup)) && self(self, std::integral_constant<size_t, IDX + 1>{}); 
                } else{
                    return self(self, std::integral_constant<size_t, IDX + 1>{});
                }
            }
        };

        return lambda(lambda, std::integral_constant<size_t, 0u>{});
    }

    template <class Functor, class T, size_t ...IDX>
    auto internal_tuple_invoke(Functor&& f, T&& tup, const std::index_sequence<IDX...>) -> decltype(auto){ //noexcept(auto) - feature request, decltype(auto) lambda - feature request

        using ret_t = decltype(f(std::get<IDX>(tup)...));
   
        if constexpr(std::is_same_v<ret_t, void>){
            f(std::get<IDX>(tup)...);
        } else{
            return f(std::get<IDX>(tup)...);
        }
    }

    template <class Functor, class T>
    auto tuple_invoke(Functor&& f, T&& tup) -> decltype(auto){ //noexcept(auto) - feature request

        return internal_tuple_invoke(f, tup, std::make_index_sequence<sizeof...(Args)>{});
    }

    template <class ResourceType, class ResourceDeallocator, class = void>
    class nothrow_immutable_unique_raii_wrapper{}; 
    
    template <class ResourceType, class ResourceDeallocator>
    class nothrow_immutable_unique_raii_wrapper<ResourceType, ResourceDeallocator, std::void_t<std::enable_if_t<std::conjunction_v<std::is_trivial<ResourceType>,
                                                                                                                                   std::is_nothrow_invocable<ResourceDeallocator, std::add_lvalue_reference_t<ResourceType>>, 
                                                                                                                                   std::is_nothrow_move_constructible<ResourceDeallocator>,
                                                                                                                                   dg::network_type_traits_x::is_base_type_v<ResourceDeallocator>>>>>{

        private:

            ResourceType resource;
            ResourceDeallocator deallocator;
            bool responsibility_flag

        public:

            using self = nothrow_immutable_unique_raii_wrapper;

            template <class DelArg = ResourceDeallocator, std::enable_if_t<std::is_nothrow_default_constructible_v<DelArg>, bool> = true>
            nothrow_immutable_unique_raii_wrapper() noexcept: resource(), 
                                                              deallocator(), 
                                                              responsibility_flag(false){}

            nothrow_immutable_unique_raii_wrapper(ResourceType resource,
                                                  ResourceDeallocator deallocator) noexcept: resource(resource),
                                                                                             deallocator(std::move(deallocator)),
                                                                                             responsibility_flag(true){}
            
            nothrow_immutable_unique_raii_wrapper(const self&) = delete;
            nothrow_immutable_unique_raii_wrapper(self&& other) noexcept: resource(other.resource),
                                                                          deallocator(std::move(other.deallocator)),
                                                                          responsibility_flag(other.responsibility_flag){
                other.responsibility_flag = false;
            }

            ~nothrow_immutable_unique_raii_wrapper() noexcept{

                if (this->responsibility_flag){
                    this->deallocator(this->resource);
                }            
            }

            self& operator =(const self&) = delete;

            self& operator =(self&& other) noexcept{

                if (this != &other){
                    if (this->responsibility_flag){
                        this->deallocator(this->resource);
                    }      
                    this->resource              = other.resource;
                    this->deallocator           = std::move(other.deallocator);
                    this->responsibility_flag   = other.responsibility_flag;
                    other.responsibility_flag   = false;
                }

                return *this;
            }

            auto value() const noexcept -> ResourceType{

                if constexpr(IS_SAFE_ACCESS_ENABLED){
                    if (!this->responsibility_flag){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::BAD_RAII_ACCESS));
                        std::abort();
                    }
                }

                return this->resource;
            }

            operator ResourceType() const noexcept{

                return this->value();
            }

            auto has_value() const noexcept -> bool{

                return this->responsibility_flag;
            }
    };

    template <class ResourceType, class ResourceDeallocator, class = void>
    class nothrow_unique_raii_wrapper{};

    template <class ResourceType, class ResourceDeallocator>
    class nothrow_unique_raii_wrapper<ResourceType, ResourceDeallocator, std::void_t<std::enable_if_t<std::conjunction_v<std::is_nothrow_move_constructible<ResourceType>,
                                                                                                                         std::is_nothrow_move_constructible<ResourceDeallocator>,
                                                                                                                         std::is_nothrow_invocable<ResourceDeallocator, std::add_rvalue_reference_t<ResourceType>>,
                                                                                                                         dg::network_type_traits_x::is_base_type_v<ResourceType>,
                                                                                                                         dg::network_type_traits_x::is_base_type_v<ResourceDeallocator>>>>>{
        
        private:

            ResourceType resource;
            ResourceDeallocator deallocator;
            bool responsibility_flag;
        
        public:

            using self = nothrow_unique_raii_wrapper; 

            template <class ResourceArg = ResourceType, class DelArg = ResourceDeallocator, std::enable_if_t<std::conjunction_v<std::is_nothrow_default_constructible<ResourceArg>, std::is_nothrow_default_constructible<DelArg>>, bool> = true>
            nothrow_unique_raii_wrapper() noexcept: resource(),
                                                    deallocator(),
                                                    responsibility_flag(false){}

            nothrow_unique_raii_wrapper(ResourceType resource, 
                                        ResourceDeallocator deallocator) noexcept: resource(std::move(resource)),
                                                                                   deallocator(std::move(deallocator)),
                                                                                   responsibility_flag(true){}

            nothrow_unique_raii_wrapper(const self& other) = delete;

            nothrow_unique_raii_wrapper(self&& other) noexcept: resource(std::move(other.resource)),
                                                                deallocator(std::move(other.deallocator)),
                                                                responsibility_flag(other.responsibility_flag){
                
                other.responsibility_flag = false;
            }

            ~nothrow_unique_raii_wrapper() noexcept{

                if (this->responsibility_flag){
                    this->deallocator(std::move(this->resource));
                }
            }

            self& operator =(const self&) = delete;

            self& operator =(self&& other) noexcept{

                if (this != &other){
                    if (this->responsibility_flag){
                        this->deallocator(std::move(this->resource));
                    }

                    this->resource              = std::move(other.resource);
                    this->deallocator           = std::move(other.deallocator);
                    this->responsibility_flag   = other.responsibility_flag;
                    other.responsibility_flag   = false;
                }

                return *this;
            }

            auto value() const noexcept -> const ResourceType&{

                if constexpr(IS_SAFE_ACCESS_ENABLED){
                    if (!this->responsibility_flag){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    }
                }

                return this->resource;
            }

            auto value() noexcept -> ResourceType&{

                if constexpr(IS_SAFE_ACCESS_ENABLED){
                    if (!this->responsibility_flag){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    }
                }

                return this->resource;
            }

            auto has_value() const noexcept -> bool{

                return this->responsibility_flag;
            }
    };
}

#endif