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
        virtual void push(uma_ptr_t, event_t *, size_t) noexcept = 0;
        virtual auto try_collect(uma_ptr_t, event_t *, size_t&, size_t) noexcept -> bool = 0; 
        virtual void collect(uma_ptr_t, event_t *, size_t&, size_t) noexcept = 0;
    };
    
    template <class lock_t>
    struct alignas(std::max(alignof(std::max_align_t), std::hardware_destructive_interference_size)) RegionBucket{
        std::vector<event_t> event_container;
        lock_t lck;
    };

    template <class lock_t>
    class MemoryPress: public virtual MemoryPressInterface{

        private:

            const size_t _memregion_pow2_value;
            const uma_ptr_t _first;
            const uma_ptr_t _last;
            const size_t _submit_cap;
            std::vector<RegionBucket<lock_t>> region_vec;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;

        public:

            MemoryPress(size_t _memregion_pow2_value,
                        uma_ptr_t _first,
                        uma_ptr_t _last, 
                        size_t _submit_cap,
                        std::vector<RegionBucket<lock_t>> region_vec,
                        std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) noexcept: _memregion_pow2_value(_memregion_pow2_value),
                                                                                                                   _first(_first),
                                                                                                                   _last(_last),
                                                                                                                   _submit_cap(_submit_cap),
                                                                                                                   region_vec(std::move(region_vec)),
                                                                                                                   executor(std::move(executor)){}

            auto memregion_size() const noexcept -> size_t{

                return size_t{1} << this->_memregion_pow2_value;
            }

            auto first() const noexcept -> uma_ptr_t{

                return this->_first;
            }

            auto last() const noexcept -> uma_ptr_t{

                return this->_last;
            }

            void push(uma_ptr_t ptr, event_t * event, size_t event_sz) noexcept{

                while (event_sz != 0u){
                    size_t submit_sz = std::min(event_sz, this->_submit_cap);
                    this->internal_push(ptr, event, submit_sz);
                    event_sz -= submit_sz;
                    std::advance(event, submit_sz);
                }
            }

            auto try_collect(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept -> bool{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_pow2_value;

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
                auto opit_first     = this->region_vec[bucket_idx].event_container.begin() + rem_sz; 
                auto opit_last      = this->region_vec[bucket_idx].event_container.end();

                std::copy(opit_first, opit_last, dst);
                this->region_vec[bucket_idx].event_container.erase(opit_first, opit_last);

                return true;
            }

            void collect(uma_ptr_t ptr, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                size_t bucket_idx   = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_pow2_value;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                stdx::xlock_guard<lock_t> lck_grd(this->region_vec[bucket_idx].lck);
                dst_sz              = std::min(dst_cap, static_cast<size_t>(this->region_vec[bucket_idx].event_container.size()));
                size_t rem_sz       = this->region_vec[bucket_idx].event_container.size() - dst_sz;
                auto opit_first     = this->region_vec[bucket_idx].event_container.begin() + rem_sz; 
                auto opit_last      = this->region_vec[bucket_idx].event_container.end();

                std::copy(opit_first, opit_last, dst);
                this->region_vec[bucket_idx].event_container.erase(opit_first, opit_last);
            }

        private:

            void internal_push(uma_ptr_t ptr, event_t * event, size_t event_sz) noexcept{

                auto task = [&]() noexcept{
                    return this->retry_push(ptr, event, event_sz);
                };

                dg::network_concurrency_infretry_x::ExecutableWrapper virtual_task(std::move(task));
                this->executor->exec(virtual_task);
            }

            auto retry_push(uma_ptr_t ptr, event_t * event, size_t event_sz) noexcept -> bool{

                size_t bucket_idx = stdx::safe_integer_cast<size_t>(dg::memult::distance(this->_first, ptr)) >> this->_memregion_pow2_value;

                if constexpr(DEBUG_MODE_FLAG){
                    if (bucket_idx >= this->region_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        std::abort();
                    }
                }

                stdx::xlock_guard<lock_t> lck_grd(this->region_vec[bucket_idx].lck);
                size_t old_sz = this->region_vec[bucket_idx].event_container.size();
                size_t new_sz = old_sz + event_sz;

                if (new_sz > this->region_vec[bucket_idx].event_container.capacity()){
                    return false;
                }

                this->region_vec[bucket_idx].event_container.insert(this->region_vec[bucket_idx].event_container.end(), event, event + event_sz);
                return true;
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