#ifndef __DG_NETWORK_MEMPRESS_H__
#define __DG_NETWORK_MEMPRESS_H__

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

namespace dg::network_mempress{

    using uma_ptr_t = dg::network_pointer::uma_ptr_t;
    using event_t   = uint64_t;

    //this is a very tough component to write correctly
    //I'm telling yall that I have rewritten this the 20th time already
    //we'll settle with what we have for now

    struct MemoryPressInterface{
        virtual ~MemoryPressInterface() noexcept = default;
        virtual auto first() const noexcept -> uma_ptr_t = 0;
        virtual auto last() const noexcept -> uma_ptr_t = 0;
        virtual auto memregion_size() const noexcept -> size_t = 0;
        virtual void push(uma_ptr_t, std::move_iterator<event_t *>, size_t, exception_t *) noexcept = 0; //the problem is here, yet I think this is the right decision in terms of resolutor, not the interface,
                                                                                                         //we cant really log the exhaustion (-> user_id) due to performance + technical constraints
                                                                                                         //yet we could log the exhaustion as a global error (because that's not a performance contraint)

        virtual auto try_collect(uma_ptr_t, event_t *, size_t&, size_t) noexcept -> bool = 0;
        virtual void collect(uma_ptr_t, event_t *, size_t&, size_t) noexcept = 0;
        virtual auto is_collectable(uma_ptr_t) noexcept -> bool = 0; 
        virtual auto max_consume_size() noexcept -> size_t = 0;
        virtual auto minimum_collect_cap() noexcept -> size_t = 0;
    };

    template <class lock_t>
    struct RegionBucket{
        dg::pow2_cyclic_queue<dg::vector<event_t>> event_container;
        std::unique_ptr<lock_t> lck;
        stdx::inplace_hdi_container<std::atomic<bool>> is_empty_concurrent_var;
    };

    //this is precisely why we would want to rewrite our container literally everytime
    //each of the container has their own virtue of optimizations that only the container could provide

    template <class lock_t>
    class MemoryPress: public virtual MemoryPressInterface{

        private:

            const size_t _memregion_sz_2exp;
            const uma_ptr_t _first;
            const uma_ptr_t _last;
            const size_t _submit_cap;
            const size_t _collect_tmp_vec_cap;
            dg::vector<RegionBucket<lock_t>> region_vec;

        public:

            MemoryPress(size_t _memregion_sz_2exp,
                        uma_ptr_t _first,
                        uma_ptr_t _last, 
                        size_t _submit_cap,
                        size_t _collect_tmp_vec_cap,
                        dg::vector<RegionBucket<lock_t>> region_vec) noexcept: _memregion_sz_2exp(_memregion_sz_2exp),
                                                                               _first(_first),
                                                                               _last(_last),
                                                                               _submit_cap(_submit_cap),
                                                                               _collect_tmp_vec_cap(_collect_tmp_vec_cap),
                                                                               region_vec(std::move(region_vec)){}

            auto memregion_size() const noexcept -> size_t{

                return size_t{1} << this->_memregion_sz_2exp;
            }

            auto first() const noexcept -> uma_ptr_t{

                return this->_first;
            }

            auto last() const noexcept -> uma_ptr_t{

                return this->_last;
            }

            auto push(uma_ptr_t ptr, dg::vector<event_t>&& event_vec) -> std::expected<bool, exception_t>{

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

                stdx::xlock_guard<lock_t> lck_grd(*this->region_vec[bucket_idx].lck);

                if (this->region_vec[bucket_idx].event_container.size() == this->region_vec[bucket_idx].event_container.capacity()){
                    return false;
                }

                this->region_vec[bucket_idx].event_container.push_back(std::move(event_vec));    
                this->update_concurrent_is_empty(bucket_idx);

                return true;
            } 

            void push(uma_ptr_t ptr, std::move_iterator<event_t *> event_arr, size_t event_sz, exception_t * exception_arr) noexcept{

                if (event_sz == 0u){
                    return;
                }

                if constexpr(DEBUG_MODE_FLAG){
                    if (event_sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                std::expected<dg::vector<event_t>, exception_t> payload = dg::network_exception::cstyle_initialize<dg::vector<event_t>>(event_sz); 

                if (!payload.has_value()){
                    std::fill(exception_arr, std::next(exception_arr, event_sz), dg::network_exception::RESOURCE_EXHAUSTION); //I would rather having an explicit error even though it could be not maintainable
                    return;
                }

                event_t * base_event_arr = event_arr.base();

                std::copy(std::make_move_iterator(base_event_arr), std::make_move_iterator(std::next(base_event_arr, event_sz)), payload->begin());
                std::expected<bool, exception_t> response = this->push(ptr, std::move(payload.value()));

                if (!response.has_value() || !response.value()){
                    std::copy(std::make_move_iterator(payload->begin()), std::make_move_iterator(payload->end()), base_event_arr);

                    if (!response.has_value()){
                        std::fill(exception_arr, std::next(exception_arr, event_sz), response.error());
                    } else{
                        std::fill(exception_arr, std::next(exception_arr, event_sz), dg::network_exception::QUEUE_FULL);                        
                    }
                } else{
                    std::fill(exception_arr, std::next(exception_arr, event_sz), dg::network_exception::SUCCESS);
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

                //sequenced after the is_empty, does not need a fence

                if (!stdx::try_lock(*this->region_vec[bucket_idx].lck)){ //OK, what happens, worst case: try_lock is noipa, not seen by compiler (expected), the std::atomic_thread_fence() kicks in the hardware, the if fences the acquire, the following statements post the if must be after the if
                    return false;
                }

                // dg::network_stack_allocation::NoExceptAllocation<std::optional<dg::vector<event_t>>[]> tmp_vec(this->_collect_tmp_vec_cap); //consider _collect_tmp_vec_cap as a reservation technique (this is a very important optimizable, we cant really do resize + etc., its not good)
                //this is incredibly difficult to write, we'll stick with noexcept measurements for now

                dg::sensitive_vector<dg::vector<event_t>> tmp_vec = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::sensitive_vector<dg::vector<event_t>>>());

                {
                    stdx::unlock_guard<lock_t> lck_grd(*this->region_vec[bucket_idx].lck);
                    this->do_collect(bucket_idx, tmp_vec, dst_cap);    
                }

                std::atomic_signal_fence(std::memory_order_seq_cst);
                dst_sz = std::distance(dst, this->write_contiguous(dst, std::move(tmp_vec)));

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

                return !this->region_vec[bucket_idx].is_empty_concurrent_var.value.load(std::memory_order_relaxed);
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
                    stdx::xlock_guard<lock_t> lck_grd(*this->region_vec[bucket_idx].lck);
                    this->do_collect(bucket_idx, tmp_vec, dst_cap);    
                }

                std::atomic_signal_fence(std::memory_order_seq_cst);
                dst_sz = std::distance(dst, this->write_contiguous(dst, std::move(tmp_vec)));
            }

            auto max_consume_size() noexcept -> size_t{

                return this->_submit_cap;
            }

            auto minimum_collect_cap() noexcept -> size_t{

                return this->_submit_cap;
            }

        private:

            void update_concurrent_is_empty(size_t bucket_idx) noexcept{

                stdx::seq_cst_guard tx_grd;

                RegionBucket<lock_t>& bucket = this->region_vec[bucket_idx];
                bucket.is_empty_concurrent_var.value.exchange(bucket.event_container.empty(), std::memory_order_relaxed); //is this expensive ??? people are trying to collect, we are introducing serialization @ the variable
            }

            void do_collect(size_t bucket_idx, dg::sensitive_vector<dg::vector<event_t>>& output_vec, size_t event_cap) noexcept{

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

                this->update_concurrent_is_empty(bucket_idx);
            }

            auto write_contiguous(event_t * dst, dg::sensitive_vector<dg::vector<event_t>>&& src_vec) noexcept -> event_t *{

                for (dg::vector<event_t>& src: src_vec){
                    dst = std::copy(std::make_move_iterator(src.begin()), std::make_move_iterator(src.end()), dst);
                }

                return dst;
            }
    };

    class ExhaustionControlledMemoryPress: public virtual MemoryPressInterface{

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

            void push(uma_ptr_t region, std::move_iterator<event_t *> event_arr, size_t event_arr_sz, exception_t * exception_arr) noexcept{

                event_t * base_event_arr = event_arr.base();

                if (event_sz == 0u){
                    return;
                }

                if constexpr(DEBUG_MODE_FLAG){
                    if (event_sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                std::expected<dg::vector<event_t>, exception_t> payload = dg::network_exception::cstyle_initialize<dg::vector<event_t>>(event_sz); 

                if (!payload.has_value()){
                    std::fill(exception_arr, std::next(exception_arr, event_sz), dg::network_exception::RESOURCE_EXHAUSTION); //I would rather having an explicit error even though it could be not maintainable
                    return;
                }

                std::copy(std::make_move_iterator(base_event_arr), std::make_move_iterator(std::next(base_event_arr, event_sz)), payload->begin());
                std::expected<bool, exception_t> response = std::unexpected(dg::network_exception::EXPECTED_NOT_INITIALIZED);
                
                auto task = [&, this]() noexcept{
                    response = this->base->push(std::move(payload.value()));
                    return !repsonse.has_value() || response.value() == true;
                };

                dg::network_concurrency_infretry_x::ExecutableWrapper virtual_task(task);
                this->executor->exec(virtual_task);

                if (!response.has_value()){
                    std::fill(exception_arr, std::next(exception_arr, event_sz), response.error());
                    std::copy(std::make_move_iterator(payload->begin()), std::make_move_iterator(payload->end()), base_event_arr);
                    return;
                }

                if (!response.value()){
                    std::fill(exception_arr, std::next(exception_arr, event_sz), dg::network_exception::QUEUE_FULL);
                    std::copy(std::make_move_iterator(payload->begin()), std::make_move_iterator(payload->end()), base_event_arr);
                    return;
                }

                std::fill(exception_arr, std::next(exception_arr, event_sz), dg::network_exception::SUCCESS);
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

    struct Factory{

        static auto spawn_mempress(uma_ptr_t first, uma_ptr_t last,
                                   size_t submit_cap, size_t region_cap, size_t memregion_sz, 
                                   std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) -> std::unique_ptr<MemoryPressInterface>{

            const size_t MIN_SUBMIT_CAP = 1u;
            const size_t MAX_SUBMIT_CAP = size_t{1} << 30;
            const size_t MIN_REGION_CAP = 1u;
            const size_t MAX_REGION_CAP = size_t{1} << 30;

            using uptr_t    = typename dg::ptr_info<>::max_unsigned_t;
            uptr_t ufirst   = dg::pointer_cast<uptr_t>(first);
            uptr_t ulast    = dg::pointer_cast<uptr_t>(last);

            if (ulast < ufirst){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!dg::memult::is_pow2(memregion_sz)){
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

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t region_vec_len   = (ulast - ufirst) / memregion_sz;
            auto region_vec         = std::vector<RegionBucket<stdx::spin_lock_t>>(region_vec_len);

            for (size_t i = 0u; i < region_vec_len; ++i){
                region_vec[i].event_container.reserve(region_cap);
            }

            return std::make_unique<MemoryPress<stdx::spin_lock_t>>(stdx::ulog2(memregion_sz), first, last, submit_cap, std::move(region_vec), std::move(executor));
        }
    };
}

#endif