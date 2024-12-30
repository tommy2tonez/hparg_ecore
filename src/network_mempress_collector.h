#ifndef __NETWORK_MEMPRESS_COLLECTOR_H__
#define __NETWORK_MEMPRESS_COLLECTOR_H__

#include <stddef.h>
#include <stdint.h>
#include <chrono>
#include <ratio>
#include "network_producer_consumer.h"
#include "network_mempress.h"
#include "network_concurrency.h" 
#include "network_randomizer.h"
#include "network_std_container.h"
#include "stdx.h"
#include <functional>
#include <memory>

namespace dg::network_mempress_collector{

    using uma_ptr_t = dg::network_pointer::uma_ptr_t;
    using event_t   = uint64_t;

    struct RangePressInterface{
        virtual ~RangePressInterface() noexcept = default;
        virtual auto size() const noexcept -> size_t = 0;
        virtual auto try_get(size_t idx, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept -> bool = 0;
        virtual void get(size_t idx, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept = 0;
    };

    class MemoryRangePress: public virtual RangePressInterface{

        private:

            std::vector<uma_ptr_t> region_table;
            std::shared_ptr<dg::network_mempress::MemoryPressInterface> mempress;
        
        public:

            MemoryRangePress(std::vector<uma_ptr_t> region_table,
                             std::shared_ptr<dg::network_mempress::MemoryPressInterface> mempress) noexcept: region_table(std::move(region_table)),
                                                                                                             mempress(std::move(mempress)){}

            auto size() const noexcept -> size_t{

                return this->region_table.size();
            }

            auto try_get(size_t idx, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept -> bool{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->region_table.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                uma_ptr_t region = stdx::to_const_reference(this->region_table)[idx];
                return this->mempress->try_collect(region, dst, dst_sz, dst_cap);
            } 

            void get(size_t idx, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->region_table.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                uma_ptr_t region = stdx::to_const_reference(this->region_table)[idx];
                this->mempress->collect(region, dst, dst_sz, dst_cap);
            }
    };

    class TryCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::chrono::nanoseconds last_updated;
            const std::chrono::nanoseconds update_interval;
            const size_t collect_cap;
            const size_t delivery_cap;
            
        public:

            TryCollector(std::shared_ptr<RangePressInterface> range_press,
                         std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                         std::chrono::nanoseconds last_updated,
                         std::chrono::nanoseconds update_interval,
                         size_t collect_cap,
                         size_t delivery_cap) noexcept: range_press(std::move(range_press)),
                                                        consumer(std::move(consumer)),
                                                        last_updated(last_updated),
                                                        update_interval(update_interval),
                                                        collect_cap(collect_cap),
                                                        delivery_cap(delivery_cap){}

            auto run_one_epoch() noexcept -> bool{

                std::chrono::nanoseconds now    = stdx::unix_timestamp();
                std::chrono::nanoseconds diff   = now - this->last_updated;

                if (diff < this->update_interval){
                    return false;
                }

                auto delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->consumer.get(), this->delivery_cap);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return false;
                }

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);
                size_t event_arr_sz     = 0u;
                size_t range_sz         = this->range_press->size();

                for (size_t i = 0u; i < range_sz; ++i){
                    if (this->range_press->try_get(i, event_arr.get(), event_arr_sz, this->collect_cap)){
                        std::for_each(event_arr.get(), std::next(event_arr.get(), event_arr_sz), std::bind_front(dg::network_producer_consumer::delvrsrv_deliver_lambda, delivery_handle->get()));
                    }
                }

                this->last_updated = stdx::unix_timestamp();
                return true;
            }
    };

    struct ClockData{
        std::chrono::nanoseconds last_updated;
        std::chrono::nanoseconds update_interval;
    };

    class ClockCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::chrono::nanoseconds last_updated;
            const std::chrono::nanoseconds update_interval;
            std::vector<ClockData> clock_data_table;
            const size_t collect_cap;
            const size_t delivery_cap;

        public:

            ClockCollector(std::shared_ptr<RangePressInterface> range_press,
                           std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                           std::chrono::nanoseconds last_updated,
                           std::chrono::nanoseconds update_interval,
                           std::vector<ClockData> clock_data_table,
                           size_t collect_cap,
                           size_t delivery_cap) noexcept: range_press(std::move(range_press)),
                                                          consumer(std::move(consumer)),
                                                          last_updated(last_updated),
                                                          update_interval(update_interval),
                                                          clock_data_table(std::move(clock_data_table)),
                                                          collect_cap(collect_cap),
                                                          delivery_cap(delivery_cap){}

            auto run_one_epoch() noexcept -> bool{

                std::chrono::nanoseconds now    = stdx::unix_timestamp();
                std::chrono::nanoseconds diff   = now - this->last_updated;

                if (diff < this->update_interval){
                    return false;
                }

                auto delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->consumer.get(), this->delivery_cap);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return false;
                }

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);
                size_t event_arr_sz     = 0u;
                size_t range_sz         = this->range_press->size(); 

                for (size_t i = 0u; i < range_sz; ++i){
                    std::chrono::nanoseconds local_now  = stdx::unix_low_resolution_timestamp();
                    std::chrono::nanoseconds local_diff = local_now - this->clock_data_table[i].last_updated;

                    if (local_diff < this->clock_data_table[i].update_interval){
                        continue;
                    }

                    this->range_press->get(i, event_arr.get(), event_arr_sz, this->collect_cap);
                    std::for_each(event_arr.get(), std::next(event_arr.get(), event_arr_sz), std::bind_front(dg::network_producer_consumer::delvrsrv_deliver_lambda, delivery_handle->get()));
                    this->clock_data_table[i].last_updated = stdx::unix_low_resolution_timestamp();
                }

                this->last_updated = stdx::unix_timestamp();
                return true;
            }
    };

    struct RevolutionData{
        uint32_t current_revolution;
        uint32_t pow2_update_revolution;
    };

    class RevolutionCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::chrono::nanoseconds last_updated;
            const std::chrono::nanoseconds update_interval;
            std::vector<RevolutionData> revolution_table;
            const size_t collect_cap;
            const size_t delivery_cap;

        public:

            RevolutionCollector(std::shared_ptr<RangePressInterface> range_press,
                                std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                std::chrono::nanoseconds last_updated,
                                std::chrono::nanoseconds update_interval,
                                std::vector<RevolutionData> revolution_table,
                                size_t collect_cap,
                                size_t delivery_cap) noexcept: range_press(std::move(range_press)),
                                                               consumer(std::move(consumer)),
                                                               last_updated(last_updated),
                                                               update_interval(update_interval),
                                                               revolution_table(std::move(revolution_table)),
                                                               collect_cap(collect_cap),
                                                               delivery_cap(delivery_cap){}

            auto run_one_epoch() noexcept -> bool{

                std::chrono::nanoseconds now    = stdx::unix_timestamp();
                std::chrono::nanoseconds diff   = now - this->last_updated;

                if (diff < this->update_interval){
                    return false;
                }

                auto delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->consumer.get(), this->delivery_cap);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return false;
                }

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);
                size_t event_arr_sz     = {};
                size_t range_sz         = this->range_press->size();

                for (size_t i = 0u; i < range_sz; ++i){
                    bool update_cond = stdx::pow2mod_unsigned(this->revolution_table[i].current_revolution, this->revolution_table[i].pow2_update_revolution) == 0u; 

                    if (update_cond){
                        this->range_press->get(i, event_arr.get(), event_arr_sz, this->collect_cap);
                        std::for_each(event_arr.get(), std::next(event_arr.get(), event_arr_sz), std::bind_front(dg::network_producer_consumer::delvrsrv_deliver_lambda, delivery_handle->get()));
                    }

                    this->revolution_table[i].current_revolution += 1u;
                }

                this->last_updated = stdx::unix_timestamp();
                return true;
            }
    };

    struct Factory{

        static auto spawn_range_press(std::shared_ptr<dg::network_mempress::MemoryPressInterface> mem_press) -> std::unique_ptr<RangePressInterface>{

            if (mem_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::vector<uma_ptr_t> region_vec{};

            for (auto region = mem_press->first(); region != mem_press->last(); region = dg::memult::advance(region, mem_press->memregion_size())){
                region_vec.push_back(region);
            }

            return std::make_unique<MemoryRangePress>(std::move(region_vec), std::move(mem_press));
        }
 
        static auto spawn_try_collector(std::shared_ptr<RangePressInterface> range_press, 
                                        std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                        std::chrono::nanoseconds update_interval,
                                        size_t collect_cap,
                                        size_t delivery_cap) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{

            const std::chrono::nanoseconds MIN_UPDATE_INTERVAL  = std::chrono::nanoseconds(1);
            const std::chrono::nanoseconds MAX_UPDATE_INTERVAL  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(1));
            const size_t MIN_COLLECT_CAP                        = 1u;
            const size_t MAX_COLLECT_CAP                        = size_t{1} << 30;
            const size_t MIN_DELIVERY_CAP                       = 1u;
            const size_t MAX_DELIVERY_CAP                       = size_t{1} << 30;

            if (range_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (consumer == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(update_interval, MIN_UPDATE_INTERVAL, MAX_UPDATE_INTERVAL) != update_interval){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(collect_cap, MIN_COLLECT_CAP, MAX_COLLECT_CAP) != collect_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(delivery_cap, MIN_DELIVERY_CAP, MAX_DELIVERY_CAP) != delivery_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<TryCollector>(std::move(range_press),
                                                  std::move(consumer),
                                                  stdx::unix_timestamp(),
                                                  update_interval,
                                                  collect_cap,
                                                  delivery_cap);
        }

        static auto spawn_clock_collector(std::shared_ptr<RangePressInterface> range_press, 
                                          std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                          std::chrono::nanoseconds update_interval,
                                          std::vector<std::chrono::nanoseconds> update_interval_table,
                                          size_t collect_cap,
                                          size_t delivery_cap) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{

            const std::chrono::nanoseconds MIN_UPDATE_INTERVAL  = std::chrono::nanoseconds(1);
            const std::chrono::nanoseconds MAX_UPDATE_INTERVAL  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(1));
            const size_t MIN_COLLECT_CAP                        = 1u;
            const size_t MAX_COLLECT_CAP                        = size_t{1} << 30;
            const size_t MIN_DELIVERY_CAP                       = 1u;
            const size_t MAX_DELIVERY_CAP                       = size_t{1} << 30; 

            if (range_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (consumer == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(update_interval, MIN_UPDATE_INTERVAL, MAX_UPDATE_INTERVAL) != update_interval){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (range_press->size() != update_interval_table.size()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(collect_cap, MIN_COLLECT_CAP, MAX_COLLECT_CAP) != collect_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(delivery_cap, MIN_DELIVERY_CAP, MAX_DELIVERY_CAP) != delivery_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::vector<ClockData> clock_data_table{};

            for (std::chrono::nanoseconds update_interval_entry: update_interval_table){
                clock_data_table.push_back(ClockData{stdx::unix_low_resolution_timestamp(), update_interval_entry});
            }

            return std::make_unique<ClockCollector>(std::move(range_press),
                                                    std::move(consumer),
                                                    stdx::unix_timestamp(),
                                                    update_interval,
                                                    std::move(clock_data_table),
                                                    collect_cap,
                                                    delivery_cap);
        }
        
        static auto spawn_revolution_collector(std::shared_ptr<RangePressInterface> range_press,
                                               std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                               std::chrono::nanoseconds update_interval,
                                               std::vector<uint32_t> update_revolution_table,
                                               size_t collect_cap,
                                               size_t delivery_cap) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{

            const std::chrono::nanoseconds MIN_UPDATE_INTERVAL  = std::chrono::nanoseconds(1);
            const std::chrono::nanoseconds MAX_UPDATE_INTERVAL  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(1));
            const size_t MIN_COLLECT_CAP                        = 1u;
            const size_t MAX_COLLECT_CAP                        = size_t{1} << 30;
            const size_t MIN_DELIVERY_CAP                       = 1u;
            const size_t MAX_DELIVERY_CAP                       = size_t{1} << 30;

            if (range_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (consumer == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(update_interval, MIN_UPDATE_INTERVAL, MAX_UPDATE_INTERVAL) != update_interval){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (update_revolution_table.size() != range_press->size()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(collect_cap, MIN_COLLECT_CAP, MAX_COLLECT_CAP) != collect_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(delivery_cap, MIN_DELIVERY_CAP, MAX_DELIVERY_CAP) != delivery_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            for (uint32_t update_revolution: update_revolution_table){
                if (!stdx::is_pow2(update_revolution)){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }
            }

            std::vector<RevolutionData> revolution_table{};

            for (uint32_t update_revolution: update_revolution_table){
                revolution_table.push_back(RevolutionData{0u, update_revolution});
            }

            return std::make_unique<RevolutionCollector>(std::move(range_press), 
                                                         std::move(consumer), 
                                                         stdx::unix_timestamp(), 
                                                         update_interval, 
                                                         std::move(revolution_table), 
                                                         collect_cap, 
                                                         delivery_cap);
        }
    };
}

#endif 