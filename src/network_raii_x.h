#ifndef __NETWORK_RAII_X_H__
#define __NETWORK_RAII_X_H__

#include <type_traits> 
#include "network_type_traits_x.h"

namespace dg{

    template <class ResourceType, class ResourceDeallocator>
    class nothrow_immutable_unique_raii_wrapper{
    
        private:

            ResourceType resource;
            ResourceDeallocator deallocator;
            bool responsibility_flag;

        public:

            static_assert(std::is_trivial_v<ResourceType>);
            static_assert(std::is_nothrow_invocable_v<ResourceDeallocator, std::add_lvalue_reference_t<ResourceType>>);
            static_assert(std::is_nothrow_move_constructible_v<ResourceDeallocator>);
            static_assert(dg::network_type_traits_x::is_base_type_v<ResourceDeallocator>);

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
}

#endif