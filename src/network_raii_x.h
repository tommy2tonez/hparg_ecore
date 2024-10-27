#ifndef __NETWORK_RAII_X_H__
#define __NETWORK_RAII_X_H__

#include <type_traits> 
#include "network_type_traits_x.h"
#include "network_log.h"

namespace dg{

    template <class ResourceType, class ResourceDeallocator, class = void>
    class nothrow_immutable_unique_raii_wrapper{}; 
    
    template <class ResourceType, class ResourceDeallocator>
    class nothrow_immutable_unique_raii_wrapper<ResourceType, ResourceDeallocator, std::void_t<std::enable_if_t<std::conjunction_v<std::is_trivial<ResourceType>,
                                                                                                                                   std::is_nothrow_invocable<ResourceDeallocator, std::add_lvalue_reference_t<ResourceType>>, 
                                                                                                                                   std::is_nothrow_move_constructible<ResourceDeallocator>,
                                                                                                                                   dg::network_type_traits_x::is_base_type<ResourceDeallocator>>>>>{

        private:

            ResourceType resource;
            ResourceDeallocator deallocator;
            bool responsibility_flag;

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

                if constexpr(DEBUG_MODE_FLAG){
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

                if constexpr(DEBUG_MODE_FLAG){
                    if (!this->responsibility_flag){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    }
                }

                return this->resource;
            }

            auto value() noexcept -> ResourceType&{

                if constexpr(DEBUG_MODE_FLAG){
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