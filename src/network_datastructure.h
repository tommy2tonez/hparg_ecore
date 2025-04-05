#ifndef __DG_NETWORK_DATASTRUCTURE_H__
#define __DG_NETWORK_DATASTRUCTURE_H__

#include <stdint.h>
#include <stdlib.h>
#include "network_exception.h"
// #include "network_log.h"

namespace dg::network_datastructure::cyclic_queue{

    //this only works if the default state of T does not hold extra semantic meaning rather than a truely empty, meaningless representation
    //std::vector<> actually uses inplace construction to make sure that is not the case
    //this is precisely the reason std::vector<> is not qualified for raw pointer arithmetic operation
    //because it is not constructed by using new[] contiguous memory operation
    //we are not doing that yet

    class pow2_cyclic_queue_index_getter_device{

        private:

            size_t off;
            size_t cap;
        
        public:

            constexpr pow2_cyclic_queue_index_getter_device() = default;
            constexpr pow2_cyclic_queue_index_getter_device(size_t off, size_t cap) noexcept: off(off), cap(cap){}

            constexpr auto operator()(size_t virtual_idx) const noexcept -> size_t{

                return (this->off + virtual_idx) & (this->cap - 1u);
            }
    };

    template <class BaseIterator>
    class pow2_cyclic_queue_iterator{

        private:

            BaseIterator iter_head;
            intmax_t virtual_idx;
            pow2_cyclic_queue_index_getter_device index_getter;
        
        public:

            using self              = pow2_cyclic_queue_iterator; 
            using difference_type   = std::ptrdiff_t;
            using value_type        = typename BaseIterator::value_type; 

            template <class T = BaseIterator, std::enable_if_t<std::is_nothrow_default_constructible_v<T>, bool> = true>
            constexpr pow2_cyclic_queue_iterator(): iter_head(), 
                                                    virtual_idx(), 
                                                    index_getter(){}

            constexpr pow2_cyclic_queue_iterator(BaseIterator iter_head,
                                                 intmax_t virtual_idx,
                                                 pow2_cyclic_queue_index_getter_device index_getter) noexcept(std::is_nothrow_move_constructible_v<BaseIterator>): iter_head(std::move(iter_head)),
                                                                                                                                                                   virtual_idx(virtual_idx),
                                                                                                                                                                   index_getter(index_getter){} 

            constexpr auto operator ++() noexcept -> self&{

                this->virtual_idx += 1;
                return *this;
            }

            constexpr auto operator ++(int) noexcept -> self{

                static_assert(std::is_nothrow_copy_constructible_v<self>);

                self rs = *this;
                this->virtual_idx += 1;
                return rs;
            }

            constexpr auto operator --() noexcept -> self&{

                this->virtual_idx -= 1;
                return *this;
            }

            constexpr auto operator --(int) noexcept -> self{

                static_assert(std::is_nothrow_copy_constructible_v<self>);

                self rs = *this;
                this->virtual_idx -= 1;
                return rs;
            }

            constexpr auto operator +(difference_type off) const noexcept -> self{

                //
                return self(this->iter_head, this->virtual_idx + off, this->index_getter);
            }

            constexpr auto operator +=(difference_type idx) noexcept -> self&{

                *this = *this + idx;
                return *this;
            }

            constexpr auto operator -(difference_type off) const noexcept -> self{

                //
                return self(this->iter_head, this->virtual_idx - off, this->index_getter);
            }

            constexpr auto operator -(const self& other) const noexcept -> difference_type{

                return this->virtual_idx - other.virtual_idx;
            }
        
            constexpr auto operator -=(difference_type idx) noexcept -> self&{

                *this = *this - idx;
                return *this;
            }

            constexpr auto operator ==(const self& other) const noexcept -> bool{

                return (this->iter_head == other.iter_head) && (this->virtual_idx == other.virtual_idx);
            }

            constexpr auto operator !=(const self& other) const noexcept -> bool{

                return (this->iter_head != other.iter_head) || (this->virtual_idx != other.virtual_idx);
            }

            constexpr auto operator *() const noexcept -> decltype(auto){

                size_t actual_idx = this->index_getter(this->virtual_idx);
                return std::next(this->iter_head, actual_idx).operator *();
            }

            constexpr auto operator ->() const noexcept -> decltype(auto){

                size_t actual_idx = this->index_getter(this->virtual_idx);
                return std::next(this->iter_head, actual_idx).operator ->();
            }
    };

    template <class T, class Allocator = std::allocator<T>>
    class pow2_cyclic_queue{

        private:

            std::vector<T, Allocator> data_arr;
            size_t off;
            size_t sz;
            size_t cap;

            using self              = pow2_cyclic_queue;

        public:

            using value_type        = T;
            using iterator          = pow2_cyclic_queue_iterator<typename std::vector<T, Allocator>::iterator>;
            using const_iterator    = pow2_cyclic_queue_iterator<typename std::vector<T, Allocator>::const_iterator>;
            using size_type         = std::size_t;

            static inline constexpr size_t DEFAULT_POW2_EXPONENT = 10u; 

            pow2_cyclic_queue(): pow2_cyclic_queue(DEFAULT_POW2_EXPONENT){}

            pow2_cyclic_queue(size_t pow2_exponent): data_arr(size_t{1} << pow2_exponent),
                                                     off(0u),
                                                     sz(0u),
                                                     cap(size_t{1} << pow2_exponent){}

            constexpr auto front() const noexcept -> const T&{

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr = this->to_index(0u);
                return this->data_arr[ptr];
            }

            constexpr auto front() noexcept -> T&{

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr = this->to_index(0u);
                return this->data_arr[ptr];
            }

            constexpr auto back() const noexcept -> const T&{

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr = this->to_index(this->sz - 1u);
                return this->data_arr[ptr];
            }

            constexpr auto back() noexcept -> T&{

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr = this->to_index(this->sz - 1u);
                return this->data_arr[ptr];
            }

            constexpr auto empty() const noexcept -> bool{

                return this->sz == 0u;                
            }

            constexpr auto begin() const noexcept -> const_iterator{

                return const_iterator(this->data_arr.begin(), 0u, this->get_index_getter_device());
            }

            constexpr auto end() const noexcept -> const_iterator{

                return const_iterator(this->data_arr.begin(), this->sz, this->get_index_getter_device());
            }

            constexpr auto begin() noexcept -> iterator{
                
                return iterator(this->data_arr.begin(), 0u, this->get_index_getter_device());
            }

            constexpr auto end() noexcept -> iterator{

                return iterator(this->data_arr.begin(), this->sz, this->get_index_getter_device());
            }

            constexpr auto size() const noexcept -> size_t{

                return this->sz;
            }

            constexpr auto capacity() const noexcept -> size_t{

                return this->cap;
            }

            constexpr auto operator[](size_t idx) const noexcept -> const T&{

                return this->data_arr[this->to_index(idx)];
            }

            constexpr auto operator[](size_t idx) noexcept -> T&{

                return this->data_arr[this->to_index(idx)];
            }

            constexpr auto at(size_t idx) const noexcept -> const T&{

                return (*this)[idx];
            }

            constexpr auto at(size_t idx) noexcept -> T&{

                return (*this)[idx];
            }

            constexpr auto resize(size_t new_sz) noexcept -> exception_t{

                static_assert(std::is_nothrow_default_constructible_v<T> && std::is_nothrow_assignable_v<T&, T&&>);

                if (new_sz > this->cap){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                size_t erase_first  = std::min(this->sz, new_sz);
                size_t erase_last   = this->sz;

                for (size_t i = erase_first; i < erase_last; ++i){
                    this->operator[](i) = T{};
                }

                this->sz            = new_sz;

                return dg::network_exception::SUCCESS;
            }

            template <class ValueLike>
            constexpr auto push_back(ValueLike&& value) noexcept -> exception_t{

                if (this->sz == this->cap){
                    return dg::network_exception::QUEUE_FULL;
                }

                size_t ptr = this->to_index(this->sz);

                if constexpr(std::is_nothrow_assignable_v<T&, ValueLike&&>){
                    this->data_arr[ptr] = std::forward<ValueLike>(value);
                    this->sz            += 1u;
                    return dg::network_exception::SUCCESS; 
                } else{
                    try{
                        this->data_arr[ptr] = std::forward<ValueLike>(value);
                        this->sz            += 1u;
                        return dg::network_exception::SUCCESS;
                    } catch (...){
                        return dg::network_exception::wrap_std_exception(std::current_exception());
                    }
                }
            }

            constexpr void pop_front() noexcept{

                static_assert(std::is_nothrow_default_constructible_v<T> && std::is_nothrow_assignable_v<T&, T&&>);

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr          = this->to_index(0u);
                this->data_arr[ptr] = T{};
                this->off           += 1u;
                this->sz            -= 1u;
            }
 
            constexpr void pop_back() noexcept{

                static_assert(std::is_nothrow_default_constructible_v<T> && std::is_nothrow_assignable_v<T&, T&&>);

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->sz == 0u){
                        // dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t ptr          = this->to_index(this->sz - 1u);
                this->data_arr[ptr] = T{};
                this->sz            -= 1u;
            } 

            constexpr void erase_front_range(size_t sz) noexcept{
                
                for (size_t i = 0u; i < sz; ++i){
                    pop_front();
                }
            }

            constexpr void erase_back_range(size_t sz) noexcept{

                for (size_t i = 0u; i < sz; ++i){
                    pop_back();
                }
            }

            constexpr auto operator ==(const self& other) const noexcept -> bool{

                return std::equal(this->begin(), this->end(), other.begin(), other.end());
            }

        private:

            constexpr auto to_index(size_t virtual_offset) const noexcept -> size_t{

                return (this->off + virtual_offset) & (this->cap - 1u);
            }

            constexpr auto get_index_getter_device() const noexcept -> pow2_cyclic_queue_index_getter_device{

                return pow2_cyclic_queue_index_getter_device(this->off, this->cap);
            }
    };
}

#endif