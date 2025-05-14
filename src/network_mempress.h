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

    struct MemoryPressInterface{
        virtual ~MemoryPressInterface() noexcept = default;
        virtual auto first() const noexcept -> uma_ptr_t = 0;
        virtual auto last() const noexcept -> uma_ptr_t = 0;
        virtual auto memregion_size() const noexcept -> size_t = 0;
        virtual void push(uma_ptr_t, event_t *, size_t, exception_t *) noexcept = 0; //the problem is here, yet I think this is the right decision in terms of resolutor, not the interface,
                                                                                     //we cant really log the exhaustion (-> user_id) due to performance + technical constraints
                                                                                     //yet we could log the exhaustion as a global error (because that's not a performance contraint)

        virtual auto try_collect(uma_ptr_t, event_t *, size_t&, size_t) noexcept -> bool = 0; 
        virtual void collect(uma_ptr_t, event_t *, size_t&, size_t) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    template <class lock_t>
    struct alignas(std::max(alignof(std::max_align_t), std::hardware_destructive_interference_size)) RegionBucket{
        std::vector<event_t> event_container;
        size_t event_container_cap;
        lock_t lck;
    };

    template <class lock_t>
    class MemoryPress: public virtual MemoryPressInterface{

        private:

            const size_t _memregion_sz_2exp;
            const uma_ptr_t _first;
            const uma_ptr_t _last;
            const size_t _submit_cap;
            std::vector<RegionBucket<lock_t>> region_vec;

        public:

            MemoryPress(size_t _memregion_sz_2exp,
                        uma_ptr_t _first,
                        uma_ptr_t _last, 
                        size_t _submit_cap,
                        std::vector<RegionBucket<lock_t>> region_vec) noexcept: _memregion_sz_2exp(_memregion_sz_2exp),
                                                                                _first(_first),
                                                                                _last(_last),
                                                                                _submit_cap(_submit_cap),
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

            void push(uma_ptr_t ptr, event_t * event, size_t event_sz, exception_t * exception_arr) noexcept{

                size_t bucket_idx = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){

                    //this is fishy, we'll change this later
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }

                    if (event_sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                stdx::xlock_guard<lock_t> lck_grd(this->region_vec[bucket_idx].lck);

                size_t old_sz   = this->region_vec[bucket_idx].event_container.size();
                size_t app_cap  = this->region_vec[bucket_idx].event_container_cap - old_sz;
                size_t app_sz   = std::min(event_sz, app_cap);
                size_t new_sz   = old_sz + app_sz;

                this->region_vec[bucket_idx].event_container.resize(new_sz);
                std::copy(event, std::next(event, app_sz), std::next(this->region_vec[bucket_idx].event_container.begin(), old_sz));

                std::fill(exception_arr, std::next(exception_arr, app_sz), dg::network_exception::SUCCESS);
                std::fill(std::next(exception_arr, app_sz), std::next(exception_arr, event_sz), dg::network_exception::RESOURCE_EXHAUSTION);
            }

            auto try_collect(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept -> bool{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                if (!stdx::try_lock(this->region_vec[bucket_idx].lck)){
                    return false;
                }

                stdx::unlock_guard<lock_t> lck_grd(this->region_vec[bucket_idx].lck);

                dst_sz              = std::min(dst_cap, static_cast<size_t>(this->region_vec[bucket_idx].event_container.size()));
                size_t rem_sz       = this->region_vec[bucket_idx].event_container.size() - dst_sz;

                auto opit_first     = std::next(this->region_vec[bucket_idx].event_container.begin(), rem_sz); 
                auto opit_last      = this->region_vec[bucket_idx].event_container.end();

                std::copy(opit_first, opit_last, dst);
                this->region_vec[bucket_idx].event_container.erase(opit_first, opit_last);

                return true;
            }

            void collect(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_sz_2exp;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                stdx::xlock_guard<lock_t> lck_grd(this->region_vec[bucket_idx].lck);

                dst_sz              = std::min(dst_cap, static_cast<size_t>(this->region_vec[bucket_idx].event_container.size()));
                size_t rem_sz       = this->region_vec[bucket_idx].event_container.size() - dst_sz;
   
                auto opit_first     = std::next(this->region_vec[bucket_idx].event_container.begin(), rem_sz); 
                auto opit_last      = this->region_vec[bucket_idx].event_container.end();

                std::copy(opit_first, opit_last, dst);
                this->region_vec[bucket_idx].event_container.erase(opit_first, opit_last);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->_submit_cap;
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