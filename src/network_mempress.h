#ifndef __DG_NETWORK_MEMPRESS_H__
#define __DG_NETWORK_MEMPRESS_H__

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <chrono>
#include <array>
#include <memory>
#include <cstring>
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

namespace dg::network_mempress{

    static inline constexpr bool IS_SPINLOCK_PREFERRED = true; 

    using uma_ptr_t = dg::network_pointer::uma_ptr_t;
    using event_t   = dg::network_memcommit::virtual_mmeory_event_t;
    using Lock      = std::conditional_t<IS_SPINLOCK_PREFERRED, 
                                         std::atomic_flag,
                                         std::mutex>;

    struct MemoryPressInterface{
        virtual ~MemoryPressInterface() noexcept = default;
        virtual auto first() const noexcept -> uma_ptr_t = 0;
        virtual auto last() const noexcept -> uma_ptr_t = 0;
        virtual auto memregion_size() const noexcept -> size_t = 0;
        virtual void push(uma_ptr_t, event_t *, size_t) noexcept = 0;
        virtual void collect(uma_ptr_t, event_t *, size_t&, size_t) noexcept = 0;
    };

    struct RegionBucket{
        dg::vector<event_t> event_container;
        std::unique_ptr<Lock> lck; //unique_ptr here is automatic HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE
    };

    class MemoryPress: public virtual MemoryPressInterface{

        private:

            const size_t _memregion_sz;
            const uma_ptr_t _first;
            const uma_ptr_t _last;
            const size_t max_submit_size_per_region;
            std::vector<RegionBucket> region_vec; 
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;

        public:

            MemoryPress(size_t _memregion_sz,
                        uma_ptr_t _first,
                        uma_ptr_t _last, 
                        size_t max_submit_size_per_region,
                        std::vector<RegionBucket> region_vec,
                        std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) noexcept: _memregion_sz(_memregion_sz),
                                                                                                                   _first(_first),
                                                                                                                   _last(_last),
                                                                                                                   max_submit_size_per_region(max_submit_size_per_region),
                                                                                                                   region_vec(std::move(region_vec)),
                                                                                                                   executor(std::move(executor)){}
            
            auto memregion_size() const noexcept -> size_t{

                return this->_memregion_sz;
            }

            auto first() const noexcept -> uma_ptr_t{

                return this->_first;
            }

            auto last() const noexcept -> uma_ptr_t{

                return this->_last;
            }

            void push(uma_ptr_t region, event_t * event, size_t event_sz) noexcept{
                
                while (event_sz != 0u){
                    size_t submit_sz = std::min(event_sz, this->max_submit_size_per_region);
                    this->internal_push(region, event, submit_sz);
                    event_sz -= submit_sz;
                    std::advance(event, submit_sz);
                }
            }

            void collect(uma_ptr_t region, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                size_t bucket_idx   = dg::memult::distance(this->_first, region) / this->_memregion_sz;
                auto lck_grd        = stdx::lock_guard(*this->region_vec[bucket_idx].lck);
                dst_sz              = std::min(dst_cap, static_cast<size_t>(this->region_vec[bucket_idx].event_container.size()));
                size_t rem_sz       = this->region_vec[bucket_idx].event_container.size() - dst_sz;
                auto opit_first     = this->region_vec[bucket_idx].event_container.begin() + rem_sz; 
                auto opit_last      = this->region_vec[bucket_idx].event_container.end();

                std::copy(opit_first, opit_last, dst);
                this->region_vec[bucket_idx].event_container.erase(opit_first, opit_last);
            }
        
        private:

            void internal_push(uma_ptr_t region, event_t * event, size_t event_sz) noexcept{

                auto task = [&]() noexcept{
                    return this->retry_push(region, event, event_sz);
                };

                auto virtual_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(task);
                this->executor->exec(virtual_task);
            }

            auto retry_push(uma_ptr_t region, event_t * event, size_t event_sz) noexcept -> bool{

                size_t bucket_idx   = dg::memult::distance(this->_first, region) / this->_memregion_sz;
                auto lck_grd        = stdx::lock_guard(*this->region_vec[bucket_idx].lck);
                size_t old_sz       = this->region_vec[bucket_idx].event_container.size();
                size_t new_sz       = old_sz + event_sz;

                if (new_sz > this->region_vec[bucket_idx].event_container.capacity()){
                    return false;
                }

                std::copy(event, event + event_sz, std::back_inserter(this->region_vec[bucket_idx].event_container));
                return true;
            }
    };

    struct Factory{

        static auto spawn_mempress(uma_ptr_t first, uma_ptr_t last, 
                                   size_t region_cap, size_t memregion_sz, 
                                   std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) -> std::unique_ptr<MemoryPressInterface>{
            
            const size_t MIN_REGION_CAP = size_t{1} << 5; 
            const size_t MAX_REGION_CAP = size_t{1} << 30;
            const double SUBMIT_RATIO   = double{0.1f};

            using uptr_t    = dg::ptr_info<>::max_unsigned_t;
            uptr_t ufirst   = pointer_cast<uptr_t>(first);
            uptr_t ulast    = pointer_cast<uptr_t>(last);

            if (dg::memult::distance(ufirst, ulast) < 0){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ufirst % memregion_sz != 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ulast % memregion_sz != 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!dg::memult::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(region_cap, MIN_REGION_CAP, MAX_REGION_CAP) != region_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t max_submit_size  = region_cap * SUBMIT_RATIO;
            size_t region_vec_len   = dg::memult::distance(ufirst, ulast) / memregion_sz; 
            auto region_vec         = std::vector<RegionBucket>(region_vec_len);

            for (size_t i = 0u; i < region_vec_len; ++i){
                region_vec[i].event_container.reserve(region_cap);
                region_vec[i].lck = std::make_unique<Lock>();
            }
            
            return std::make_unique<MemoryPress>(memregion_sz, first, last, max_submit_size, std::move(region_vec), std::move(executor));
        }
    };
}

#endif