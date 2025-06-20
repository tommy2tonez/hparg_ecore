#ifndef __DG_NETWORK_MEMPRESS_IMPL1_H__
#define __DG_NETWORK_MEMPRESS_IMPL1_H__

#include "network_mempress_interface.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <memory>
#include "assert.h"
#include <type_traits>
#include "network_exception.h"
#include "network_log.h" 
#include "network_pointer.h" 
#include <functional>
#include <utility>
#include <algorithm>
#include "network_concurrency_x.h"
#include "stdx.h"
#include <new>

namespace dg::network_mempress_impl1{

    using uma_ptr_t = dg::network_pointer::uma_ptr_t;
    using event_t   = uint64_t;

    struct BatchBucket{
        dg::pow2_cyclic_queue<dg::vector<event_t>> event_container;
        std::unique_ptr<std::atomic_flag> lck; //we'd have to use std::unique_ptr<> for now, for etc. reasons, we dont wanna talk about this
        std::unique_ptr<std::atomic_flag> is_empty_concurrent_var;
    };

    //this is precisely why we would want to rewrite our container literally everytime
    //each of the container has their own virtue of optimizations that only the container could provide
    //we'd want to implement two mempress, one with dg::vector<> and one without
    //because our client is very strict about this

    class BatchPress: public virtual dg::network_mempress::MemoryPressInterface{

        private:

            const size_t _memregion_sz_2exp;
            const uma_ptr_t _first;
            const uma_ptr_t _last;
            const size_t _submit_cap;
            dg::vector<BatchBucket> region_vec;

        public:

            BatchPress(size_t _memregion_sz_2exp,
                       uma_ptr_t _first,
                       uma_ptr_t _last, 
                       size_t _submit_cap,
                       dg::vector<BatchBucket> region_vec) noexcept: _memregion_sz_2exp(_memregion_sz_2exp),
                                                                      _first(_first),
                                                                      _last(_last),
                                                                      _submit_cap(_submit_cap),
                                                                      region_vec(std::move(region_vec)){}

            auto first() const noexcept -> uma_ptr_t{

                return this->_first;
            }

            auto last() const noexcept -> uma_ptr_t{

                return this->_last;
            }

            auto memregion_size() const noexcept -> size_t{

                return size_t{1} << this->_memregion_sz_2exp;
            }

            auto is_busy(uma_ptr_t ptr) noexcept -> bool{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                return this->region_vec[bucket_idx].lck->test(std::memory_order_relaxed);
            } 

            auto push(uma_ptr_t ptr, dg::vector<event_t>&& event_vec) noexcept -> std::expected<bool, exception_t>{

                if (event_vec.empty()){
                    return true;
                }

                if (event_vec.size() > this->max_consume_size()){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::atomic_flag> lck_grd(*this->region_vec[bucket_idx].lck);

                if (this->region_vec[bucket_idx].event_container.size() == this->region_vec[bucket_idx].event_container.capacity()){
                    return false;
                }

                dg::network_exception_handler::nothrow_log(this->region_vec[bucket_idx].event_container.push_back(std::move(event_vec)));
                this->region_vec[bucket_idx].is_empty_concurrent_var.value.clear(std::memory_order_relaxed);

                return true;
            }

            void push(uma_ptr_t ptr, std::move_iterator<event_t *> event_arr, size_t event_arr_sz, exception_t * exception_arr) noexcept{

                if (event_arr_sz == 0u){
                    return;
                }

                if constexpr(DEBUG_MODE_FLAG){
                    if (event_arr_sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                std::expected<dg::vector<event_t>, exception_t> payload = dg::network_exception::cstyle_initialize<dg::vector<event_t>>(event_arr_sz);

                if (!payload.has_value()){
                    std::fill(exception_arr, std::next(exception_arr, event_arr_sz), dg::network_exception::RESOURCE_EXHAUSTION); //I would rather having an explicit error even though it could be not maintainable
                    return;
                }

                event_t * base_event_arr = event_arr.base();
                std::copy(std::make_move_iterator(base_event_arr), std::make_move_iterator(std::next(base_event_arr, event_arr_sz)), payload->begin());
                std::expected<bool, exception_t> response = this->push(ptr, std::move(payload.value()));

                if (!response.has_value()){
                    std::copy(std::make_move_iterator(payload->begin()), std::make_move_iterator(payload->end()), base_event_arr);
                    std::fill(exception_arr, std::next(exception_arr, event_arr_sz), response.error());

                    return;
                }

                if (!response.value()){
                    std::copy(std::make_move_iterator(payload->begin()), std::make_move_iterator(payload->end()), base_event_arr);
                    std::fill(exception_arr, std::next(exception_arr, event_arr_sz), dg::network_exception::QUEUE_FULL);

                    return;
                }

                std::fill(exception_arr, std::next(exception_arr, event_arr_sz), dg::network_exception::SUCCESS);
            }

            auto try_collect(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept -> bool{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                if (!stdx::try_lock(*this->region_vec[bucket_idx].lck, std::memory_order_relaxed)){
                    return false;
                }

                dg::sensitive_vector<dg::vector<event_t>> tmp_vec = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::sensitive_vector<dg::vector<event_t>>>());

                {
                    stdx::unlock_guard<std::atomic_flag> lck_grd(*this->region_vec[bucket_idx].lck);
                    this->unsafe_do_collect(bucket_idx, tmp_vec, dst_cap);
                }

                dst_sz = std::distance(dst, this->steal_contiguous(dst, std::move(tmp_vec)));

                return true;
            }

            //I have a hinge that this could be part of the try_collect, as if is_empty == true + dst_sz == 0 
            //having this as is_collectable is ... fine

            auto is_collectable(uma_ptr_t ptr) noexcept -> bool{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                return this->region_vec[bucket_idx].is_empty_concurrent_var.value.test(std::memory_order_relaxed) == false;
            }

            void collect(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                dg::sensitive_vector<dg::vector<event_t>> tmp_vec = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::sensitive_vector<dg::vector<event_t>>>());

                {
                    stdx::xlock_guard<std::atomic_flag> lck_grd(*this->region_vec[bucket_idx].lck);
                    this->unsafe_do_collect(bucket_idx, tmp_vec, dst_cap);
                }

                dst_sz = std::distance(dst, this->steal_contiguous(dst, std::move(tmp_vec)));
            }

            auto max_consume_size() noexcept -> size_t{

                return this->_submit_cap;
            }

            auto minimum_collect_cap() noexcept -> size_t{

                return this->_submit_cap;
            }

        private:

            void unsafe_do_collect(size_t bucket_idx, dg::sensitive_vector<dg::vector<event_t>>& output_vec, size_t event_cap) noexcept{

                size_t event_sz = 0u; 

                while (true){
                    if (this->region_vec[bucket_idx].event_container.empty()){
                        break;
                    }

                    if (event_sz + this->region_vec[bucket_idx].event_container.front().size() > event_cap){
                        break;
                    }

                    output_vec.push_back(std::move(this->region_vec[bucket_idx].event_container.front()));
                    this->region_vec[bucket_idx].event_container.pop_front();
                    event_sz += output_vec.back().size();
                }

                if (this->region_vec[bucket_idx].event_container.empty()){
                    this->region_vec[bucket_idx].is_empty_concurrent_var.test_and_set(std::memory_order_relaxed);
                }
            }

            auto steal_contiguous(event_t * dst, dg::vector<dg::vector<event_t>>&& src_vec) noexcept -> event_t *{

                event_t * cpy_dst = dst;

                for (dg::vector<event_t>& src: src_vec){
                    cpy_dst = std::copy(std::make_move_iterator(src.begin()), std::make_move_iterator(src.end()), cpy_dst);
                }

                src_vec.clear();
                return cpy_dst;
            }
    };

    struct PressBucket{
        dg::pow2_cyclic_queue<event_t> event_container;
        std::unique_ptr<std::atomic_flag> lck;
        std::unique_ptr<std::atomic_flag> is_empty_concurrent_var;
    };

    class MemoryPress: public virtual dg::network_mempress::MemoryPressInterface{

        private:

            const size_t _memregion_sz_2exp;
            const uma_ptr_t _first;
            const uma_ptr_t _last;
            const size_t _submit_cap;
            dg::vector<BatchBucket> region_vec;
        
        public:
            
            MemoryPress(size_t _memregion_sz_2exp,
                        uma_ptr_t _first,
                        uma_ptr_t _last,
                        size_t _submit_cap,
                        dg::vector<PressBucket> region_vec) noexcept: _memregion_sz_2exp(_memregion_sz_2exp),
                                                                      _first(_first),
                                                                      _last(_last),
                                                                      _submit_cap(_submit_cap),
                                                                      region_vec(std::move(region_vec)){}

            auto first() const noexcept -> uma_ptr_t{

                return this->_first;
            }

            auto last() const noexcept -> uma_ptr_t{

                return this->_last;
            }

            auto memregion_size() const noexcept -> size_t{

                return size_t{1} << this->_memregion_sz_2exp;
            }

            auto is_busy(uma_ptr_t ptr) noexcept -> bool{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                return this->region_vec[bucket_idx].lck->test(std::memory_order_relaxed);
            }

            void push(uma_ptr_t ptr, std::move_iterator<event_t *> event_arr, size_t event_arr_sz, exception_t * exception_arr){

                event_t * base_event_arr    = event_arr.base();
                size_t bucket_idx           = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::networK_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::atomic_flag> lck_grd(*this->region_vec[bucket_idx].lck);

                size_t app_cap  = this->region_vec[bucket_idx].event_container.capacity() - this->region_vec[bucket_idx].event_container.size();
                size_t push_sz  = std::min(event_arr_sz, app_cap);
                size_t old_sz   = this->region_vec[bucket_idx].event_container.size();
                size_t new_sz   = old_sz + push_sz;

                this->region_vec[bucket_idx].event_container.resize(new_sz);

                std::copy(std::make_move_iterator(base_event_arr),
                          std::make_move_iterator(std::next(base_event_arr, push_sz)),
                          std::next(this->region_vec[bucket_idx].event_container.begin(), old_sz));

                std::fill(exception_arr, std::next(exception_arr, push_sz), dg::network_exception::SUCCESS);
                std::fill(std::next(exception_arr, push_sz), std::next(exception_arr, event_arr_sz), dg::network_exception::QUEUE_FULL);

                if (new_sz != 0u){
                    this->region_vec[bucket_idx].is_empty_concurrent_var.value.clear(std::memory_order_relaxed);
                }
            }

            auto try_collect(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept -> bool{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }                    
                }

                if (!stdx::try_lock(*this->region_vec[bucket_idx].lck, std::memory_order_relaxed)){
                    return false;
                }

                stdx::unlock_guard<std::atomic_flag> lck_grd(*this->region_vec[bucket_idx].lck);

                dst_sz = std::min(dst_cap, static_cast<size_t>(this->region_vec[bucket_idx].event_container.size()));

                std::copy(std::make_move_iterator(this->region_vec[bucket_idx].event_container.begin()),
                          std::make_move_iterator(std::next(this->region_vec[bucket_idx].event_container.begin(), dst_sz)),
                          dst);

                this->region_vec[bucket_idx].event_container.erase_front_range(dst_sz);

                if (this->region_vec[bucket_idx].event_container.empty()){
                    this->region_vec[bucket_idx].is_empty_concurrent_var.value.test_and_set(std::memory_order_relaxed);
                }

                return true;
            }

            auto is_collectable(uma_ptr_t ptr) noexcept -> bool{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                return this->region_vec[bucket_idx].is_empty_concurrent_var.value.test(std::memory_order_relaxed) == false;
            }

            void collect(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical();
                    }
                }

                stdx::lock_guard<std::atomic_flag> lck_grd(*this->region_vec[bucket_idx].lck);

                dst_sz = std::min(dst_cap, static_cast<size_t>(this->region_vec[bucket_idx].event_container.size()));

                std::copy(std::make_move_iterator(this->region_vec[bucket_idx].event_container.begin()),
                          std::make_move_iterator(std::next(this->region_vec[bucket_idx].event_container.begin(), dst_sz)),
                          dst);

                this->region_vec[bucket_idx].event_container.erase_front_range(dst_sz);

                if (this->region_vec[bucket_idx].event_container.empty()){
                    this->region_vec[bucket_idx].is_empty_concurrent_var.value.test_and_set(std::memory_order_relaxed);
                }

                return true;
            }

            auto max_consume_size() noexcept -> size_t{

                return this->_submit_cap;
            }

            auto minimum_collect_cap() noexcept -> size_t{

                return 1u;
            }
    };

    //we have desperately trying to "reduce" the time spending in the mutex, by using an aggregation technique of batch press
    //this is "proven" to be faster than the normal_press alone, not the optimal optimal that we are talking about, we'll see about that

    class FastPress: public virtual dg::network_mempress::MemoryPressInterface{

        private:

            std::unique_ptr<dg::network_mempress::MemoryPressInterface> batch_press;
            std::unique_ptr<dg::network_mempress::MemoryPressInterface> normal_press;
            size_t batch_trigger_threshold;

        public:

            FastPress(std::unique_ptr<dg::network_mempress::MemoryPressInterface> batch_press,
                      std::unique_ptr<dg::network_mempress::MemoryPressInterface> normal_press,
                      size_t batch_trigger_threshold) noexcept: batch_press(std::move(batch_press)),
                                                                normal_press(std::move(normal_press)),
                                                                batch_trigger_threshold(batch_trigger_threshold){

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->batch_press == nullptr){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }

                    if (this->normal_press == nullptr){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }

                    if (this->batch_press->first() != this->normal_press->first()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }

                    if (this->batch_press->last() != this->normal_press->last()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }

                    if (this->batch_press->memregion_size() != this->normal_press->memregion_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }

                    //we dont want complexities

                    if (this->batch_press->max_consume_size() != this->normal_press->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }
            }

            auto first() const noexcept -> uma_ptr_t{

                return this->normal_press->first;
            }

            auto last() const noexcept -> uma_ptr_t{

                return this->normal_press->last;
            }

            auto memregion_size() const noexcept -> uma_ptr_t{

                return this->normal_press->memregion_size();
            }

            auto is_busy(uma_ptr_t ptr) noexcept -> bool{
                
                return this->normal_press->is_busy() || this->batch_press->is_busy(); //escalation, we'll see about this
            }

            void push(uma_ptr_t ptr, std::move_iterator<event_t *> event_arr, size_t event_arr_sz, exception_t * exception_arr) noexcept{

                if (event_arr_sz >= this->batch_trigger_threshold){
                    this->batch_press->push(ptr, event_arr, event_arr_sz, exception_arr);
                } else{
                    this->normal_press->push(ptr, event_arr, event_arr_sz, exception_arr);
                }
            }

            auto try_collect(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept -> bool{

                //attempt to collect the batch_press
                //we are not being "fair"

                bool random_value = dg::network_randomizer::randomize_int<bool>();
                
                if (random_value){
                    return this->try_collect_1(ptr, dst, dst_sz, dst_cap);
                } else{
                    return this->try_collect_2(ptr, dst, dst_sz, dst_cap);
                }
            }

            void collect(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                //attempt to collect the batch_press
                //we are not being "fair"

                bool random_value = dg::network_randomizer::randomize_int<bool>();

                if (random_value){
                    this->collect_1(ptr, dst, dst_sz, dst_cap);
                } else{
                    this->collect_2(ptr, dst, dst_sz, dst_cap);
                }
            }

            auto is_collectable(uma_ptr_t ptr) noexcept -> bool{

                return this->normal_press->is_collectable(ptr) || this->batch_press->is_collectable(ptr);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->normal_press->max_consume_size();
            }

            auto minimum_collect_cap() noexcept -> size_t{

                return std::max(this->normal_press->minimum_collect_cap(), this->batch_press->minimum_collect_cap()); //escalation
            }
        
        private:
            
            auto try_collect_1(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept -> bool{

                dst_sz                  = 0u;

                event_t * batch_dst     = dst;
                size_t batch_dst_sz     = 0u;
                size_t batch_dst_cap    = dst_cap; 

                event_t * normal_dst    = dst;
                size_t normal_dst_sz    = 0u;
                size_t normal_dst_cap   = dst_cap;

                bool batch_try_result   = this->batch_press->try_collect(ptr, batch_dst, batch_dst_sz, batch_dst_cap);

                if (batch_try_result){
                    dst_sz          += batch_dst_sz;
                    std::advance(normal_dst, batch_dst_sz);
                    normal_dst_cap  -= batch_dst_sz;
                }

                bool normal_try_result  = this->normal_press->try_collect(ptr, normal_dst, normal_dst_sz, normal_dst_cap);

                if (normal_try_result){
                    dst_sz          += normal_dst_sz;
                }

                return batch_try_result || normal_try_result;
            }

            auto try_collect_2(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept -> bool{

                dst_sz                  = 0u;

                event_t * normal_dst    = dst;
                size_t normal_dst_sz    = 0u;
                size_t normal_dst_cap   = dst_cap;

                event_t * batch_dst     = dst;
                size_t batch_dst_sz     = 0u;
                size_t batch_dst_cap    = dst_cap; 

                bool normal_try_result  = this->normal_press->try_collect(ptr, normal_dst, normal_dst_sz, normal_dst_cap);

                if (normal_try_result){
                    dst_sz          += normal_dst_sz;
                    std::advance(batch_dst, normal_dst_sz);
                    batch_dst_cap   -= normal_dst_sz;
                }

                if (batch_dst_cap < this->batch_press->minimum_collect_cap()){
                    return normal_try_result;
                }

                bool batch_try_result   = this->batch_press->try_collect(ptr, batch_dst, batch_dst_sz, batch_dst_cap);

                if (batch_try_result){
                    dst_sz          += batch_dst_sz;
                }

                return batch_try_result || normal_try_result;
            }

            void collect_1(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                event_t * batch_dst     = dst;
                size_t batch_dst_sz     = 0u;
                size_t batch_dst_cap    = dst_cap;

                this->batch_press->collect(ptr, batch_dst, batch_dst_sz, batch_dst_cap);

                event_t * normal_dst    = std::next(dst, batch_dst_sz);
                size_t normal_dst_sz    = 0u;
                size_t normal_dst_cap   = dst_cap - batch_dst_sz;

                this->normal_press->collect(ptr, normal_dst, normal_dst_sz, normal_dst_cap);

                dst_sz                  = batch_dst_sz + normal_dst_sz;
            }

            void collect_2(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                event_t * normal_dst    = dst;
                size_t normal_dst_sz    = 0u;
                size_t normal_dst_cap   = dst_cap;

                this->normal_press->collect(ptr, normal_dst, normal_dst_sz, normal_dst_cap);

                event_t * batch_dst     = std::next(dst, normal_dst_sz);
                size_t batch_dst_sz     = 0u;
                size_t batch_dst_cap    = dst_cap - normal_dst_sz;

                if (batch_dst_cap < this->batch_press->minimum_collect_cap()){
                    dst_sz = normal_dst_sz;
                    return;
                }

                this->batch_press->collect(ptr, batch_dst, batch_dst_sz, batch_dst_cap);

                dst_sz                  = batch_dst_sz + normal_dst_sz;

            }
    };

    //I guess the exhaustion controlled is the connector problem, not the memory press problem, we'd just have it here
    //we dont have a patch for the memory vectorization, it'd just use a lot of memory allocations that we'd think harmful
    //we'd use an affined exhaustion controlled guard at the infretry device to solve this problem
    //we dont have an immediate patch for this 

    class ExhaustionControlledBatchPress: public virtual MemoryPressInterface{

        private:

            std::unique_ptr<BatchPress> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;

        public:

            ExhaustionControlledBatchPress(std::unique_ptr<BatchPress> base,
                                           std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) noexcept: base(std::move(base)),
                                                                                                                                      executor(std::move(executor)){}

            auto first() const noexcept -> uma_ptr_t{

                return this->base->first();
            }

            auto last() const noexcept -> uma_ptr_t{

                return this->base->last();
            }

            auto memregion_size() const noexcept -> size_t{

                return this->base->memregion_size();
            }

            auto is_busy(uma_ptr_t ptr) noexcept -> bool{

                return this->base->is_busy(ptr);
            }

            void push(uma_ptr_t region, std::move_iterator<event_t *> event_arr, size_t event_arr_sz, exception_t * exception_arr) noexcept{

                event_t * base_event_arr = event_arr.base();

                if (event_arr_sz == 0u){
                    return;
                }

                if constexpr(DEBUG_MODE_FLAG){
                    if (event_arr_sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                std::expected<dg::vector<event_t>, exception_t> payload = dg::network_exception::cstyle_initialize<dg::vector<event_t>>(event_arr_sz); 

                if (!payload.has_value()){
                    std::fill(exception_arr, std::next(exception_arr, event_arr_sz), dg::network_exception::RESOURCE_EXHAUSTION); //I would rather having an explicit error even though it could be not maintainable
                    return;
                }

                std::copy(std::make_move_iterator(base_event_arr), std::make_move_iterator(std::next(base_event_arr, event_arr_sz)), payload->begin());
                std::expected<bool, exception_t> response = std::unexpected(dg::network_exception::EXPECTED_NOT_INITIALIZED);

                auto task = [&, this]() noexcept{
                    response = this->base->push(std::move(payload.value()));
                    return !repsonse.has_value() || response.value();
                };

                dg::network_concurrency_infretry_x::ExecutableWrapper virtual_task(task);
                this->executor->exec(virtual_task);

                if (!response.has_value()){
                    std::fill(exception_arr, std::next(exception_arr, event_arr_sz), response.error());
                    std::copy(std::make_move_iterator(payload->begin()), std::make_move_iterator(payload->end()), base_event_arr);
                    return;
                }

                if (!response.value()){
                    std::fill(exception_arr, std::next(exception_arr, event_arr_sz), dg::network_exception::QUEUE_FULL);
                    std::copy(std::make_move_iterator(payload->begin()), std::make_move_iterator(payload->end()), base_event_arr);
                    return;
                }

                std::fill(exception_arr, std::next(exception_arr, event_arr_sz), dg::network_exception::SUCCESS);
            }

            auto try_collect(uma_ptr_t region, event_t * event_arr, size_t& event_arr_sz, size_t event_arr_cap) noexcept -> bool{

                return this->base->try_collect(region, event_arr, event_arr_sz, event_arr_cap);
            }

            auto is_collectable(uma_ptr_t ptr) noexcept -> bool{

                return this->base->is_collectable(ptr);
            }

            void collect(uma_ptr_t region, event_t * event_arr, size_t& event_arr_sz, size_t event_arr_cap) noexcept{

                this->base->collect(region, event_arr, event_arr_sz, event_arr_cap);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }

            auto minimum_collect_cap() noexcept -> size_t{

                return this->base->minimum_collect_cap();
            }
    };

    class ExhaustionControlledMemoryPress: public virtual dg::network_mempress::MemoryPressInterface{

        private:

            std::unique_ptr<MemoryPress> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
        
        public:

            ExhaustionControlledMemoryPress(std::unique_ptr<MemoryPress> base,
                                            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) noexcept: base(std::move(base)),
                                                                                                                                       executor(std::move(executor)){}

            auto first() const noexcept -> uma_ptr_t{

                return this->base->first();
            }

            auto last() const noexcept -> uma_ptr_t{

                return this->base->last();
            }

            auto memregion_size() const noexcept -> size_t{

                return this->base->memregion_size();
            }

            auto is_busy(uma_ptr_t ptr) noexcept -> bool{

                return this->base->is_busy(ptr);
            }

            void push(uma_ptr_t region, std::move_iterator<event_t *> event_arr, size_t event_arr_sz, exception_t * exception_arr) noexcept{

                event_t * first_event_arr           = event_arr.base();
                event_t * last_event_arr            = std::next(first_event_arr, event_arr_sz);

                exception_t * first_exception_arr   = exception_arr;
                exception_t * last_exception_arr    = std::next(first_exception_arr, event_arr_sz);

                this->base->push(region, event_arr, event_arr_sz, exception_arr);

                auto task = [&, this]() noexcept{
                    exception_t * first_retriable_arr   = std::find(first_exception_arr, last_exception_arr, dg::network_exception::QUEUE_FULL);
                    exception_t * last_retriable_arr    = std::find_if(first_retriable_arr, last_exception_arr, [](exception_t err) noexcept{return err != dg::network_exception::QUEUE_FULL;});

                    size_t sliding_window_sz            = std::distance(first_retriable_arr, last_retriable_arr);

                    if (sliding_window_sz == 0u){
                        return true;
                    }

                    size_t relative_offset_sz           = std::distance(first_exception_arr, first_retriable_arr);

                    std::advance(first_event_arr, relative_offset_sz);
                    std::advance(first_exception_arr, relative_offset_sz);

                    this->base->push(region, std::make_move_iterator(first_event_arr), sliding_window_sz, first_exception_arr);
                };

                dg::network_concurrency_infretry_x::ExecutableWrapper virtual_task(task);
                this->executor->exec(virtual_task);
            }

            auto try_collect(uma_ptr_t region, event_t * event_arr, size_t& event_arr_sz, size_t event_arr_cap) noexcept -> bool{

                return this->base->try_collect(region, event_arr, event_arr_sz, event_arr_cap);
            }

            auto is_collectable(uma_ptr_t ptr) noexcept -> bool{

                return this->base->is_collectable(ptr);
            }

            void collect(uma_ptr_t region, event_t * event_arr, size_t& event_arr_sz, size_t event_arr_cap) noexcept{

                this->base->collect(region, event_arr, event_arr_sz, event_arr_cap);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }

            auto minimum_collect_cap() noexcept -> size_t{

                return this->base->minimum_collect_cap();
            }
    };

    //I'll be right back, wait a second

    struct Factory{

        static auto spawn_batchpress(uma_ptr_t first, uma_ptr_t last,
                                     size_t submit_cap, 
                                     size_t region_vec_cap, 
                                     size_t memregion_sz) -> std::unique_ptr<dg::network_mempress::MemoryPressInterface>{

            const size_t MIN_SUBMIT_CAP     = 1u;
            const size_t MAX_SUBMIT_CAP     = size_t{1} << 30;
            const size_t MIN_REGION_VEC_CAP = 1u;
            const size_t MAX_REGION_VEC_CAP = size_t{1} << 30;

            using uptr_t    = typename dg::ptr_info<uma_ptr_t>::unsigned_t;
            uptr_t ufirst   = dg::pointer_cast<uptr_t>(first);
            uptr_t ulast    = dg::pointer_cast<uptr_t>(last);

            if (ulast < ufirst){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ufirst % memregion_sz != 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ulast % memregion_sz != 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(submit_cap, MIN_SUBMIT_CAP, MAX_SUBMIT_CAP) != submit_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(region_vec_cap, MIN_REGION_VEC_CAP, MAX_REGION_VEC_CAP) != region_vec_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(region_vec_cap)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t vec_sz   = (ulast - ufirst) / memregion_sz;
            auto vec        = dg::vector<BatchBucket>(vec_sz);

            for (BatchBucket& bucket: vec){
                bucket.event_container          = dg::pow2_cyclic_queue<dg::vector<event_t>>(stdx::ulog2(region_vec_cap));
                bucket.lck                      = std::make_unique<std::atomic_flag>(false); //TODOs: hdi
                bucket.is_empty_concurrent_var  = std::make_unique<std::atomic_flag>(true); //TODOs: hdi
            }

            return std::make_unique<BatchPress>(stdx::ulog2(memregion_sz),
                                                first,
                                                last,
                                                submit_cap,
                                                std::move(vec));
        }

        static auto spawn_exhaustion_controlled_batchpress(uma_ptr_t first, uma_ptr_t last,
                                                           size_t submit_cap,
                                                           size_t region_vec_cap,
                                                           size_t memregion_sz,
                                                           std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) -> std::unique_ptr<dg::network_mempress::MemoryPressInterface> {

            const size_t MIN_SUBMIT_CAP     = 1u;
            const size_t MAX_SUBMIT_CAP     = size_t{1} << 30;
            const size_t MIN_REGION_VEC_CAP = 1u;
            const size_t MAX_REGION_VEC_CAP = size_t{1} << 30;

            using uptr_t    = typename dg::ptr_info<uma_ptr_t>::unsigned_t;
            uptr_t ufirst   = dg::pointer_cast<uptr_t>(first);
            uptr_t ulast    = dg::pointer_cast<uptr_t>(last);

            if (ulast < ufirst){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ufirst % memregion_sz != 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ulast % memregion_sz != 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(submit_cap, MIN_SUBMIT_CAP, MAX_SUBMIT_CAP) != submit_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(region_vec_cap, MIN_REGION_VEC_CAP, MAX_REGION_VEC_CAP) != region_vec_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(region_vec_cap)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t vec_sz   = (ulast - ufirst) / memregion_sz;
            auto vec        = dg::vector<BatchBucket>(vec_sz);

            for (BatchBucket& bucket: vec){
                bucket.event_container          = dg::pow2_cyclic_queue<dg::vector<event_t>>(stdx::ulog2(region_vec_cap));
                bucket.lck                      = std::make_unique<std::atomic_flag>(false); //TODOs: hdi
                bucket.is_empty_concurrent_var  = std::make_unique<std::atomic_flag>(true); //TODOs: hdi
            }

            auto batch_press    =  std::make_unique<BatchPress>(stdx::ulog2(memregion_sz),
                                                                first,
                                                                last,
                                                                submit_cap,
                                                                std::move(vec));

            return std::make_unique<ExhaustionControlledBatchPress>(std::move(batch_press),
                                                                    std::move(executor));
        } 

        static auto spawn_mempress(uma_ptr_t first, uma_ptr_t last,
                                   size_t submit_cap, size_t region_cap, size_t memregion_sz) -> std::unique_ptr<dg::network_mempress::MemoryPressInterface>{

            const size_t MIN_SUBMIT_CAP = 1u;
            const size_t MAX_SUBMIT_CAP = size_t{1} << 30;
            const size_t MIN_REGION_CAP = 1u;
            const size_t MAX_REGION_CAP = size_t{1} << 30;

            using uptr_t    = typename dg::ptr_info<uma_ptr_t>::unsigned_t;
            uptr_t ufirst   = dg::pointer_cast<uptr_t>(first);
            uptr_t ulast    = dg::pointer_cast<uptr_t>(last);

            if (ulast < ufirst){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ufirst % memregion_sz != 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ulast % memregion_sz != 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(submit_cap, MIN_SUBMIT_CAP, MAX_SUBMIT_CAP) != submit_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(region_cap, MIN_REGION_CAP, MAX_REGION_CAP) != region_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (submit_cap > region_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(region_cap)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t vec_sz   = (ulast - ufirst) / memregion_sz;
            auto vec        = std::vector<PressBucket>(vec_sz);

            for (PressBucket& bucket: vec){
                bucket.event_container          = dg::pow2_cyclic_queue<event_t>(stdx::ulog2(region_cap));
                bucket.lck                      = std::make_unique<std::atomic_flag>(false); //TODOs: hdi
                bucket.is_empty_concurrent_var  = std::make_unique<std::atomic_flag>(true); //TODOs: hdi
            }

            return std::make_unique<MemoryPress>(stdx::ulog2(memregion_sz),
                                                 first,
                                                 last,
                                                 submit_cap,
                                                 std::move(vec));
        }

        static auto spawn_exhaustion_controlled_mempress(std::unique_ptr<dg::network_mempress::MemoryPressInterface> base,
                                                         std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor){
                        
            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledMemoryPress>(std::move(base),
                                                                     std::move(executor));
        } 

        static auto spawn_fastpress(uma_ptr_t first, uma_ptr_t last,
                                    size_t submit_cap, size_t region_cap,
                                    size_t region_vec_cap,
                                    size_t memregion_sz,
                                    size_t batch_trigger_threshold) -> std::unique_ptr<dg::network_mempress::MemoryPressInterface>{

            return std::make_unique<FastPress>(spawn_batchpress(first, last, submit_cap, region_vec_cap, memregion_sz),
                                               spwan_mempress(first, last, submit_cap, region_cap, memregion_sz),
                                               batch_trigger_threshold);
        }
    };
}

#endif