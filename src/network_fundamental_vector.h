#ifndef __DG_NETWORK_BUFFER_CONTAINER_H__
#define __DG_NETWORK_BUFFER_CONTAINER_H__

#include <type_traits>
#include <stdint.h>
#include <stddef.h>
#include <cstring>
#include <memory>
#include "network_memory_utility.h"

namespace dg::network_fundamental_vector{

    struct init_tag{}; 

    template <class ContaineeType, size_t CAPACITY, std::enable_if_t<std::is_fundamental_v<ContaineeType>, bool> = true>
    class fixed_fundamental_vector_view{

        private:

            size_t * sz; 
            ContaineeType * arr;

        public:

            static_assert(alignof(size_t) >= alignof(ContaineeType));

            constexpr fixed_fundamental_vector_view() noexcept = default; 

            explicit fixed_fundamental_vector_view(char * buf) noexcept{
                
                void * aligned_buf      = dg::memory_utility::align(buf, alignof(size_t));
                this->arr               = dg::memory_utility::start_lifetime_as_array<ContaineeType>(static_cast<char *>(aligned_buf) + sizeof(size_t), CAPACITY);
                this->sz                = dg::memory_utility::start_lifetime_as<size_t>(aligned_buf);
            }

            explicit fixed_fundamental_vector_view(char * buf, const init_tag) noexcept: fixed_fundamental_vector_view(buf){

                *this->sz = 0u;
            }

            void clear() noexcept{

                *this->sz = 0u;
            }
            
            void resize(size_t new_sz) noexcept{

                *this->sz = new_sz;    
            }

            auto size() const noexcept -> size_t{

                return *this->sz;
            }

            void push_back(ContaineeType containee) noexcept{
                
                this->arr[(*this->sz)++] = containee;
            }

            void push_back(ContaineeType * first, ContaineeType * last) noexcept{
                
                std::copy(first, last, this->end());
                *this->sz += std::distance(first, last);
            } 

            auto get(size_t idx) const noexcept -> const ContaineeType&{

                return this->arr[idx];
            }

            auto get(size_t idx) noexcept -> ContaineeType&{

                return this->arr[idx];
            }

            auto begin() noexcept -> ContaineeType *{

                return this->arr;
            } 

            auto end() noexcept -> ContaineeType *{

                return this->arr + (*this->sz);
            } 

            auto begin() const noexcept -> const ContaineeType *{

                return this->arr;
            } 

            auto end() const noexcept -> const ContaineeType *{

                return this->arr + (*this->sz);
            }

            auto data() const noexcept -> const ContaineeType *{

                return this->arr;
            }

            auto data() noexcept -> ContaineeType *{

                return this->arr;
            }

            static consteval auto flat_byte_size() noexcept -> size_t{

                return static_cast<size_t>(sizeof(size_t)) + (CAPACITY * sizeof(ContaineeType)) + alignof(size_t); 
            }

            static consteval auto capacity() noexcept -> size_t{

                return CAPACITY;
            }
    };

    template <class ContaineeType, size_t CAPACITY>
    class fixed_fundamental_vector: public fixed_fundamental_vector_view<ContaineeType, CAPACITY>{

        private:

            using base = fixed_fundamental_vector_view<ContaineeType, CAPACITY>;
            std::unique_ptr<char[]> buf;
            
        public:

            fixed_fundamental_vector(): base(){

                this->buf = std::make_unique<char[]>(base::flat_byte_size());
                static_cast<base&>(*this) = base(this->buf.get(), init_tag{});    
            }
    };
}

#endif