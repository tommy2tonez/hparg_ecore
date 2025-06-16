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

    //we'll work on this component today
    //strategize
    //we scan memory regions, 16K memory region, each 1MB == 16GB worth of memories

    //we'll attempt to try_get the memory regions, we'll leverage locality of region data cache fetch by using delvrsrv
    //

    using uma_ptr_t = dg::network_pointer::uma_ptr_t;
    using event_t   = uint64_t;

    struct RangePressInterface{
        virtual ~RangePressInterface() noexcept = default;
        virtual auto size() const noexcept -> size_t = 0;
        virtual auto try_get(size_t idx, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept -> bool = 0;
        virtual auto is_empty(size_t idx) noexcept -> bool = 0;
        virtual auto is_busy(size_t idx) noexcept -> bool = 0;
        virtual void get(size_t idx, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept = 0;
    };

    struct BonsaiFrequencierInterface{
        virtual ~BonsaiFrequencierInterface() noexcept = default;
        virtual auto frequencize(uint32_t) noexcept -> exception_t = 0;
        virtual void reset() noexcept = 0;
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

                uma_ptr_t region = std::as_const(this->region_table)[idx];
                return this->mempress->try_collect(region, dst, dst_sz, dst_cap);
            } 

            auto is_empty(size_t idx) noexcept -> bool{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->region_table.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                uma_ptr_t region = std::as_const(this->region_table)[idx];
                return !this->mempress->is_collectable(region);
            }

            auto is_busy(size_t idx) noexcept -> bool{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->region_table.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                uma_ptr_t region = std::as_const(this->region_table)[idx];
                return this->mempress->is_busy(region);
            }

            void get(size_t idx, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->region_table.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                uma_ptr_t region = std::as_const(this->region_table)[idx];
                this->mempress->collect(region, dst, dst_sz, dst_cap);
            }
    };

    class WareHouseConnector: public virtual dg::network_producer_consumer::ConsumerInterface<event_t>{

        private:

            std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse;
            size_t warehouse_ingestion_sz; 

        public:

            WareHouseConnector(std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse,
                               size_t warehouse_ingestion_sz): warehouse(std::move(warehouse)),
                                                               warehouse_ingestion_sz(warehouse_ingestion_sz){}

            void push(std::move_iterator<event_t *> event_arr, size_t event_arr_sz) noexcept{

                event_t * base_event_arr                = event_arr.base();
                size_t trimmed_warehouse_ingestion_sz   = std::min(this->warehouse_ingestion_sz, this->warehouse->max_consume_size());
                size_t discretization_sz                = trimmed_warehouse_ingestion_sz;

                if constexpr(DEBUG_MODE_FLAG){
                    if (discretization_sz == 0u){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t iterable_sz                      = event_arr_sz / discretization_sz + size_t{event_arr_sz % discretization_sz != 0u};  

                for (size_t i = 0u; i < iterable_sz; ++i){
                    size_t first    = discretization_sz * i;
                    size_t last     = std::min(static_cast<size_t>(discretization_sz * (i + 1)), event_arr_sz);
                    size_t vec_sz   = last - first;

                    std::expected<dg::vector<event_t>, exception_t> vec = dg::network_exception::cstyle_initialize<dg::vector<event_t>>(vec_sz);

                    if (!vec.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(vec.error()));
                        continue;
                    }

                    std::copy(std::make_move_iterator(std::next(base_event_arr, first)), std::make_move_iterator(std::next(base_event_arr, last)), vec->begin());
                    std::expected<bool, exception_t> push_err = this->warehouse->push(std::move(vec.value()));

                    if (!push_err.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(push_err.error()));
                        continue;
                    }

                    if (!push_err.value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(dg::network_exception::QUEUE_FULL));
                        continue;
                    }
                }
            }
    };

    class ExhaustionControllerWareHouseConnector: public virtual dg::network_producer_consumer::ConsumerInterface<event_t>{

        private:

            std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device;
            size_t warehouse_ingestion_sz;

        public:

            ExhaustionControllerWareHouseConnector(std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse,
                                                   std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device,
                                                   size_t warehouse_ingestion_sz) noexcept: warehouse(std::move(warehouse)),
                                                                                            infretry_device(std::move(infretry_device)),
                                                                                            warehouse_ingestion_sz(warehouse_ingestion_sz){}

            void push(std::move_iterator<event_t *> event_arr, size_t event_arr_sz) noexcept{

                //we probably retry 1 time, 2 times or inf times
                //we dont know, that's why we need abstractions

                event_t * base_event_arr                = event_arr.base();
                size_t trimmed_warehouse_ingestion_sz   = std::min(this->warehouse_ingestion_sz, this->warehouse->max_consume_size());
                size_t discretization_sz                = trimmed_warehouse_ingestion_sz;

                if constexpr(DEBUG_MODE_FLAG){
                    if (discretization_sz == 0u){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t iterable_sz                      = event_arr_sz / discretization_sz + size_t{event_arr_sz % discretization_sz != 0u};

                for (size_t i = 0u; i < iterable_sz; ++i){
                    size_t first    = discretization_sz * i;
                    size_t last     = std::min(static_cast<size_t>(discretization_sz * (i + 1)), event_arr_sz);
                    size_t vec_sz   = last - first;

                    std::expected<dg::vector<event_t>, exception_t> vec = dg::network_exception::cstyle_initialize<dg::vector<event_t>>(vec_sz);

                    if (!vec.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vec.error())); //serious error
                        continue;
                    }

                    std::copy(std::make_move_iterator(std::next(base_event_arr, first)), std::make_move_iterator(std::next(base_event_arr, last)), vec->begin());
                    std::expected<bool, exception_t> push_err = std::unexpected(dg::network_exception::EXPECTED_NOT_INITIALIZED);

                    auto task = [&, this]() noexcept{
                        push_err = this->warehouse->push(std::move(vec.value()));                       

                        if (!push_err.has_value()){
                            return true;
                        }

                        return push_err.value();
                    };

                    dg::network_concurrency_infretry_x::ExecutableWrapper virtual_task(task);
                    this->infretry_device->exec(virtual_task);

                    //there are a lot of stuffs could happen
                    //we are on finite retriables or infinite retriables
                    //upon exit, it's possibly thru, excepted, or not thru due to producer-consumer exhaustion or finite retriables ran out
                    //we can only clue our transactional push_err, which is guaranteed to be correct
                    //yet we could guarantee one thing, that if it hits push_err.error() or thru, it would not continue to retry

                    if (!push_err.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(push_err.error())); //serious error
                        continue;
                    }

                    if (!push_err.value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::QUEUE_FULL)); //serious error
                        continue;
                    }
                }
            }
    };

    class BonsaiFrequencier: public virtual BonsaiFrequencierInterface{

        private:

            std::optional<std::chrono::time_point<std::chrono::steady_clock>> last_tick;

        public:

            BonsaiFrequencier() noexcept: last_tick(std::nullopt){}

            auto frequencize(uint32_t frequency) noexcept -> exception_t{

                if (frequency == 0u){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                if (!this->last_tick.has_value()){
                    this->last_tick = std::chrono::steady_clock::now();
                    return;
                }

                std::chrono::nanoseconds period                             = this->frequency_to_period(frequency);
                std::chrono::time_point<std::chrono::steady_clock> expiry   = this->last_tick.value() + period;
                std::chrono::time_point<std::chrono::steady_clock> now      = std::chrono::steady_clock::now();

                if (now >= expiry){
                    this->last_tick = now;
                    return;
                }

                std::chrono::nanoseconds diff   = std::chrono::duration_cast<std::chrono::nanoseconds>(expiry - now);
                stdx::high_resolution_sleep(diff);
                this->last_tick                 = std::chrono::steady_clock::now();
            }

            void reset() noexcept{

                this->last_tick = std::nullopt;
            }

        private:

            static constexpr auto frequency_to_period(uint32_t frequency) noexcept -> std::chrono::nanoseconds{

                if constexpr(DEBUG_MODE_FLAG){
                    if (frequency == 0u){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                uint32_t SECOND_NANOSECONDS = 1000000000UL;
                uint32_t round_period       = SECOND_NANOSECONDS / frequency;

                return std::chrono::nanoseconds(round_period);
            }
    };

    class TryCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::shared_ptr<BonsaiFrequencierInterface> frequencizer;
            const size_t collect_cap;
            const size_t delivery_cap;
            const uint32_t scan_frequency;

        public:

            TryCollector(std::shared_ptr<RangePressInterface> range_press,
                         std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                         std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                         size_t collect_cap,
                         size_t delivery_cap,
                         uint32_t scan_frequency) noexcept: range_press(std::move(range_press)),
                                                            consumer(std::move(consumer)),
                                                            frequencizer(std::move(frequencizer)),
                                                            collect_cap(collect_cap),
                                                            delivery_cap(delivery_cap),
                                                            scan_frequency(scan_frequency){}

            auto run_one_epoch() noexcept -> bool{

                dg::network_exception_handler::nothrow_log(this->frequencizer->frequencize(this->scan_frequency));

                size_t dh_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->consumer.get(), this->delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->consumer.get(), this->delivery_cap, dh_mem.get()));

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);

                size_t range_sz             = this->range_press->size();

                for (size_t i = 0u; i < range_sz; ++i){
                    if (this->range_press->is_empty(i)){
                        continue;
                    }

                    size_t event_arr_sz = {};
                    bool get_rs         = this->range_press->try_get(i, event_arr.get(), event_arr_sz, this->collect_cap);

                    if (!get_rs){
                        continue;
                    }

                    for (size_t j = 0u; j < event_arr_sz; ++j){
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), std::move(event_arr[j]));
                    }
                }

                return true;
            }
    };

    class CompetitiveTryCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::shared_ptr<BonsaiFrequencierInterface> frequencizer;
            std::vector<size_t> suffix_array;
            const size_t collect_cap;
            const size_t delivery_cap;
            const uint32_t scan_frequency;

        public:

            CompetitiveTryCollector(std::shared_ptr<RangePressInterface> range_press,
                                    std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                    std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                                    std::vector<size_t> suffix_array,
                                    size_t collect_cap,
                                    size_t delivery_cap,
                                    uint32_t scan_frequency) noexcept: range_press(std::move(range_press)),
                                                                       consumer(std::move(consumer)),
                                                                       frequencizer(std::move(frequencizer)),
                                                                       suffix_array(std::move(suffix_array)),
                                                                       collect_cap(collect_cap),
                                                                       delivery_cap(delivery_cap),
                                                                       scan_frequency(scan_frequency){}
            
            auto run_one_epoch() noexcept -> bool{

                dg::network_exception_handler::nothrow_log(this->frequencizer->frequencize(this->scan_frequency));

                size_t dh_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->consumer.get(), this->delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->consumer.get(), this->delivery_cap, dh_mem.get()));

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);

                if constexpr(DEBUG_MODE_FLAG){
                    size_t range_sz = this->range_press->size();

                    if (range_sz != this->suffix_array.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t max_log2_exp         = stdx::ulog2(stdx::ceil2(this->suffix_array.size())); //max_log2_exp is guaranteed to be >= suffix_array.size()
                size_t iterable_log2_sz     = max_log2_exp + 1u;

                for (size_t i = 0u; i < iterable_log2_sz; ++i){
                    size_t first        = 0u;
                    size_t last         = std::min(size_t{1} << i, static_cast<size_t>(this->suffix_array.size()));
                    size_t failed_sz    = 0u;

                    for (size_t j = first; j < last; ++j){
                        size_t region_idx   = this->suffix_array[j]; 
                        size_t event_arr_sz = {};

                        if (this->range_press->is_empty(region_idx)){
                            continue;
                        }

                        bool get_rs = this->range_press->try_get(region_idx, event_arr.get(), event_arr_sz, this->collect_cap);

                        if (!get_rs){
                            std::swap(this->suffix_array[j], this->suffix_array[first + failed_sz]);
                            failed_sz += 1u;
                            continue;
                        }

                        for (size_t z = 0u; z < event_arr_sz; ++z){
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), std::move(event_arr[z]));
                        }
                    }
                }

                return true;
            }
    };

    struct ClockData{
        std::optional<std::chrono::time_point<std::chrono::steady_clock>> last_updated;
        std::chrono::nanoseconds update_interval;
    };

    class ClockCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::shared_ptr<BonsaiFrequencierInterface> frequencizer;
            std::vector<ClockData> clock_data_table;
            const size_t ticking_clock_resolution;
            const size_t collect_cap;
            const size_t delivery_cap;
            const uint32_t scan_frequency;

        public:

            ClockCollector(std::shared_ptr<RangePressInterface> range_press,
                           std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                           std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                           std::vector<ClockData> clock_data_table,
                           size_t ticking_clock_resolution,
                           size_t collect_cap,
                           size_t delivery_cap,
                           uint32_t scan_frequency) noexcept: range_press(std::move(range_press)),
                                                              consumer(std::move(consumer)),
                                                              frequencizer(std::move(frequencizer)),
                                                              clock_data_table(std::move(clock_data_table)),
                                                              ticking_clock_resolution(ticking_clock_resolution),
                                                              collect_cap(collect_cap),
                                                              delivery_cap(delivery_cap),
                                                              scan_frequency(scan_frequency){}

            auto run_one_epoch() noexcept -> bool{

                dg::network_exception_handler::nothrow_log(this->frequencizer->frequencize(this->scan_frequency));

                size_t dh_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->consumer.get(), this->delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->consumer.get(), this->delivery_cap, dh_mem.get()));

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);
                size_t range_sz             = this->range_press->size();
                auto ticking_steady_clock   = dg::ticking_clock<std::chrono::steady_clock>(this->ticking_clock_resolution); 

                for (size_t i = 0u; i < range_sz; ++i){
                    std::chrono::time_point<std::chrono::steady_clock> local_now = ticking_steady_clock.get();

                    if constexpr(DEBUG_MODE_FLAG){
                        if (i >= this->clock_data_table.size()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    std::optional<std::chrono::time_point<std::chrono::steady_clock>> last_updated = this->clock_data_table[i].last_updated;

                    if (last_updated.has_value()){
                        std::chrono::nanoseconds lifetime = std::chrono::duration_cast<std::chrono::nanoseconds>(local_now - last_updated.value());

                        if (lifetime < this->clock_data_table[i].update_interval){ //not expired
                            continue;
                        }
                    }

                    size_t event_arr_sz = {};
                    this->range_press->get(i, event_arr.get(), event_arr_sz, this->collect_cap);

                    for (size_t j = 0u; j < event_arr_sz; ++j){
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), std::move(event_arr[j]));
                    }

                    this->clock_data_table[i].last_updated = ticking_steady_clock.get();
                }

                return true;
            }
    };

    //these guys have different virtues of try_get optimizations that I have yet to explore
    //an attempt to polymorphic the solution is not recommended
    //so we would want to see the howtos later
 
    class ClockTryCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::shared_ptr<BonsaiFrequencierInterface> frequencizer;
            std::vector<ClockData> clock_data_table;
            const size_t ticking_clock_resolution;
            const size_t collect_cap;
            const size_t delivery_cap;
            const uint32_t scan_frequency;

        public:

            ClockTryCollector(std::shared_ptr<RangePressInterface> range_press,
                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                              std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                              std::vector<ClockData> clock_data_table,
                              size_t ticking_clock_resolution,
                              size_t collect_cap,
                              size_t delivery_cap,
                              uint32_t scan_frequency) noexcept: range_press(std::move(range_press)),
                                                                 consumer(std::move(consumer)),
                                                                 frequencizer(std::move(frequencizer)),
                                                                 clock_data_table(std::move(clock_data_table)),
                                                                 ticking_clock_resolution(ticking_clock_resolution),
                                                                 collect_cap(collect_cap),
                                                                 delivery_cap(delivery_cap),
                                                                 scan_frequency(scan_frequency){}

            auto run_one_epoch() noexcept -> bool{

                dg::network_exception_handler::nothrow_log(this->frequencizer->frequencize(this->scan_frequency));

                size_t dh_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->consumer.get(), this->delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->consumer.get(), this->delivery_cap, dh_mem.get()));

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);
                size_t range_sz             = this->range_press->size(); 
                auto clock                  = dg::ticking_clock<std::chrono::steady_clock>(this->ticking_clock_resolution);

                for (size_t i = 0u; i < range_sz; ++i){
                    std::chrono::time_point<std::chrono::steady_clock> local_now = clock.get();

                    if constexpr(DEBUG_MODE_FLAG){
                        if (i >= this->clock_data_table.size()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    std::optional<std::chrono::time_point<std::chrono::steady_clock>> last_updated = this->clock_data_table[i].last_updated;

                    if (last_updated.has_value()){
                        std::chrono::nanoseconds lifetime = local_now - last_updated.value();

                        if (lifetime < this->clock_data_table[i].update_interval){ //not expired
                            continue;
                        }
                    }

                    if (this->range_press->is_empty(i)){
                        this->clock_data_table[i].last_updated = clock.get();    
                        continue;
                    }

                    size_t event_arr_sz = {};
                    bool get_rs         = this->range_press->try_get(i, event_arr.get(), event_arr_sz, this->collect_cap);

                    if (!get_rs){
                        continue;
                    }

                    for (size_t j = 0u; j < event_arr_sz; ++j){
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), std::move(event_arr[j]));
                    }

                    this->clock_data_table[i].last_updated = clock.get();    
                }

                return true;
            }
    };

    class ClockSuffixData{
        std::optional<std::chrono::time_point<std::chrono::steady_clock>> last_updated;
        std::chrono::nanoseconds update_interval;
        uint32_t suffix_idx;
    };

    //
    class ClockCompetitiveTryCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::shared_ptr<BonsaiFrequencierInterface> frequencizer;
            std::vector<ClockSuffixData> clock_data_table;
            const size_t ticking_clock_resolution;
            const size_t collect_cap;
            const size_t delivery_cap;
            const uint32_t scan_frequency;
        
        public:

            ClockCompetitiveTryCollector(std::shared_ptr<RangePressInterface> range_press,
                                         std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                         std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                                         std::vector<ClockSuffixData> clock_data_table,
                                         size_t ticking_clock_resolution,
                                         size_t collect_cap,
                                         size_t delivery_cap,
                                         uint32_t scan_frequency) noexcept: range_press(std::move(range_press)),
                                                                            consumer(std::move(consumer)),
                                                                            frequencizer(std::move(frequencizer)),
                                                                            clock_data_table(std::move(clock_data_table)),
                                                                            ticking_clock_resolution(ticking_clock_resolution),
                                                                            collect_cap(collect_cap),
                                                                            delivery_cap(delivery_cap),
                                                                            scan_frequency(scan_frequency){}

            auto run_one_epoch() noexcept -> bool{

                dg::network_exception_handler::nothrow_log(this->frequencizer->frequencize(this->scan_frequency));

                size_t dh_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->consumer.get(), this->delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->consumer.get(), this->delivery_cap, dh_mem.get()));

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);

                if constexpr(DEBUG_MODE_FLAG){
                    size_t range_sz = this->range_press->size();

                    if (range_sz != this->clock_data_table.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto clock              = dg::ticking_clock<std::chrono::steady_clock>(this->ticking_clock_resolution); 

                size_t max_log2_exp     = stdx::ulog2(stdx::ceil2(this->clock_data_table.size()));
                size_t iterable_sz      = max_log2_exp + 1u;

                for (size_t i = 0u; i < iterable_sz; ++i){
                    size_t first        = 0u;
                    size_t last         = std::min(size_t{1} << i, static_cast<size_t>(this->clock_data_table.size()));
                    size_t failed_sz    = 0u;

                    for (size_t j = first; j < last; ++j){
                        size_t region_idx                                       = this->clock_data_table[j].suffix_idx; 
                        std::chrono::nanoseconds update_interval                = this->clock_data_table[j].update_interval; //this is very expensive
                        std::chrono::time_point<std::chrono::steady_clock> now  = clock.get();

                        std::optional<std::chrono::time_point<std::chrono::steady_clock>> last_updated = this->clock_data_table[j].last_updated;

                        if (last_updated.has_value()){
                            std::chrono::nanoseconds lifetime   = now - last_updated;

                            if (lifetime < update_interval){
                                continue;
                            }
                        }

                        if (this->range_press->is_empty(region_idx)){
                            this->clock_data_table[j].last_updated = clock.get();
                            continue;
                        }

                        size_t event_arr_sz = {};
                        bool get_rs         = this->range_press->try_get(region_idx, event_arr.get(), event_arr_sz, this->collect_cap);  

                        if (!get_rs){
                            std::swap(this->clock_data_table[j], this->clock_data_table[first + failed_sz]);
                            failed_sz += 1;
                            continue;
                        }

                        for (size_t z = 0u; z < event_arr_sz; ++z){
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), std::move(event_arr[z]));
                        }

                        this->clock_data_table[j].last_updated = clock.get();
                    }
                }
            }
    };

    //we have yet want to abstract this -> ClockCollector
    struct RevolutionData{
        uint32_t current_revolution;
        uint32_t revolution_update_threshold;
    };

    class RevolutionCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::shared_ptr<BonsaiFrequencierInterface> frequencizer;
            std::vector<RevolutionData> revolution_table;
            const size_t collect_cap;
            const size_t delivery_cap;
            const uint32_t scan_frequency;

        public:

            RevolutionCollector(std::shared_ptr<RangePressInterface> range_press,
                                std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                                std::vector<RevolutionData> revolution_table,
                                size_t collect_cap,
                                size_t delivery_cap,
                                uint32_t scan_frequency) noexcept: range_press(std::move(range_press)),
                                                                   consumer(std::move(consumer)),
                                                                   frequencizer(std::move(frequencizer)),
                                                                   revolution_table(std::move(revolution_table)),
                                                                   collect_cap(collect_cap),
                                                                   delivery_cap(delivery_cap),
                                                                   scan_frequency(scan_frequency){}

            auto run_one_epoch() noexcept -> bool{

                dg::networK_exception_handler::nothrow_log(this->frequencizer->frequencize(this->scan_frequency));

                size_t dh_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->consumer.get(), this->delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->consumer.get(), this->delivery_cap, dh_mem.get()));

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);
                size_t range_sz             = this->range_press->size();

                for (size_t i = 0u; i < range_sz; ++i){
                    if constexpr(DEBUG_MODE_FLAG){
                        if (i >= this->revolution_table.size()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    size_t& current_revolution  = this->revolution_table[i].current_revolution;
                    size_t update_revolution    = this->revolution_table[i].revolution_update_threshold; 
                    current_revolution          += 1;

                    if (current_revolution == update_revolution){
                        size_t event_arr_sz = {};
                        this->range_press->get(i, event_arr.get(), event_arr_sz, this->collect_cap);

                        for (size_t j = 0u; j < event_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), std::move(event_arr[j]));
                        }
                        
                        current_revolution  = 0u;
                    }
                }

                return true;
            }
    };

    struct Factory{

        static auto spawn_range_press(std::shared_ptr<dg::network_mempress::MemoryPressInterface> mempress,
                                      const std::vector<uma_ptr_t>& region_table) -> std::unique_ptr<RangePressInterface>{

            if (mempress == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            using uptr_t = dg::ptr_info<uma_ptr_t>::max_unsigned_t;

            for (uma_ptr_t region: region_table){
                uptr_t uptr         = dg::pointer_cast<uptr_t>(region);
                size_t memregion_sz = mempress->memregion_size();
                uptr_t ufirst       = dg::pointer_cast<uptr_t>(mempress->first());
                uptr_t ulast        = dg::pointer_cast<uptr_t>(mempress->last()); 

                if (uptr % memregion_sz != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (uptr < ufirst){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (uptr >= ulast){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }
            }

            return std::make_unique<MemoryRangePress>(region_table, std::move(mempress));
        }

        static auto spawn_range_press(std::shared_ptr<dg::network_mempress::MemoryPressInterface> mempress) -> std::unique_ptr<RangePressInterface>{

            if (mem_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::vector<uma_ptr_t> region_vec{};

            for (uma_ptr_t region = mem_press->first(); region != mem_press->last(); region = dg::memult::next(region, mem_press->memregion_size())){
                region_vec.push_back(region);
            }

            return spawn_range_press(mempress, region_vec);
        }

        static auto spawn_warehouse_connector(std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse,
                                              size_t warehouse_ingestion_sz) -> std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>>{

            if (warehouse == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            const size_t MIN_WAREHOUSE_INGESTION_SZ     = 1u;
            const size_t MAX_WAREHOUSE_INGESTION_SZ     = warehouse->max_consume_size(); 

            if (std::clamp(warehouse_ingestion_sz, MIN_WAREHOUSE_INGESTION_SZ, MAX_WAREHOUSE_INGESTION_SZ) != warehouse_ingestion_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<WareHouseConnector>(std::move(warehouse), warehouse_ingestion_sz);
        }

        static auto spawn_exhaustion_controlled_warehouse_connector(std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse,
                                                                    std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device,
                                                                    size_t warehouse_ingestion_sz) -> std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>>{
                        
            if (warehouse == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (infretry_device == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            const size_t MIN_WAREHOUSE_INGESTION_SZ     = 1u;
            const size_t MAX_WAREHOUSE_INGESTION_SZ     = warehouse->max_consume_size();

            if (std::clamp(warehouse_ingestion_sz, MIN_WAREHOUSE_INGESTION_SZ, MAX_WAREHOUSE_INGESTION_SZ) != warehouse_ingestion_sz){
                dg::network_exception::throw_exception(dg::networK_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControllerWareHouseConnector>(std::move(warehouse), std::move(infretry_device), warehouse_ingestion_sz);
        }

        static auto spawn_bonsai_frequencizer() -> std::unique_ptr<BonsaiFrequencierInterface>{

            return std::make_unique<BonsaiFrequencier>();
        }

        static auto spawn_try_collector(std::shared_ptr<RangePressInterface> range_press, 
                                        std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                        size_t collect_cap,
                                        size_t delivery_cap,
                                        uint32_t scan_frequency) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{

            const size_t MIN_COLLECT_CAP        = 1u;
            const size_t MAX_COLLECT_CAP        = size_t{1} << 30;
            const size_t MIN_DELIVERY_CAP       = 1u;
            const size_t MAX_DELIVERY_CAP       = size_t{1} << 30;
            const uint32_t MIN_SCAN_FREQUENCY   = 1u;
            const uint32_t MAX_SCAN_FREQUENCY   = 1000000000UL; 

            if (range_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (consumer == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(collect_cap, MIN_COLLECT_CAP, MAX_COLLECT_CAP) != collect_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(delivery_cap, MIN_DELIVERY_CAP, MAX_DELIVERY_CAP) != delivery_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(scan_frequency, MIN_SCAN_FREQUENCY, MAX_SCAN_FREQUENCY) != scan_frequency){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<TryCollector>(std::move(range_press),
                                                  std::move(consumer),
                                                  spawn_bonsai_frequencizer(),
                                                  collect_cap,
                                                  delivery_cap,
                                                  scan_frequency);
        }

        static auto spawn_competitive_try_collector(std::shared_ptr<RangePressInterface> range_press,
                                                    std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                                    size_t collect_cap,
                                                    size_t delivery_cap,
                                                    size_t scan_frequency) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            const size_t MIN_COLLECT_CAP        = 1u;
            const size_t MAX_COLLECT_CAP        = size_t{1} << 30;
            const size_t MIN_DELIVERY_CAP       = 1u;
            const size_t MAX_DELIVERY_CAP       = size_t{1} << 30;
            const uint32_t MIN_SCAN_FREQUENCY   = 1u;
            const uint32_t MAX_SCAN_FREQUENCY   = 1000000000UL;

            if (range_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (consumer == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(collect_cap, MIN_COLLECT_CAP, MAX_COLLECT_CAP) != collect_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(delivery_cap, MIN_DELIVERY_CAP, MAX_DELIVERY_CAP) != delivery_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(scan_frequency, MIN_SCAN_FREQUENCY, MAX_SCAN_FREQUENCY) != scan_frequency){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::vector<size_t> suffix_array(range_press->size());
            std::iota(suffix_array.begin(), suffix_array.end(), 0u);

            return std::make_unique<CompetitiveTryCollector>(std::move(range_press),
                                                             std::move(consumer),
                                                             spawn_bonsai_frequencizer(),
                                                             std::move(suffix_array),
                                                             collect_cap,
                                                             delivery_cap,
                                                             scan_frequency);
        } 

        static auto spawn_clock_collector(std::shared_ptr<RangePressInterface> range_press,
                                          std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                          const std::vector<std::chrono::nanoseconds>& update_interval_table,
                                          size_t ops_clock_resolution,
                                          size_t collect_cap,
                                          size_t delivery_cap,
                                          uint32_t scan_frequency) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{

            const std::chrono::nanoseconds MIN_UPDATE_INTERVAL  = std::chrono::nanoseconds(1);
            const std::chrono::nanoseconds MAX_UPDATE_INTERVAL  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::days(1));
            const size_t MIN_OPS_CLOCK_RESOLUTION               = 0u;
            const size_t MAX_OPS_CLOCK_RESOLUTION               = std::numeric_limits<size_t>::max();
            const size_t MIN_COLLECT_CAP                        = 1u;
            const size_t MAX_COLLECT_CAP                        = size_t{1} << 30;
            const size_t MIN_DELIVERY_CAP                       = 1u;
            const size_t MAX_DELIVERY_CAP                       = size_t{1} << 30; 
            const uint32_t MIN_SCAN_FREQUENCY                   = 1u;
            const uint32_t MAX_SCAN_FREQUENCY                   = 1000000000UL; 

            if (range_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (consumer == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (range_press->size() != update_interval_table.size()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(ops_clock_resolution, MIN_OPS_CLOCK_RESOLUTION, MAX_OPS_CLOCK_RESOLUTION) != ops_clock_resolution){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(collect_cap, MIN_COLLECT_CAP, MAX_COLLECT_CAP) != collect_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(delivery_cap, MIN_DELIVERY_CAP, MAX_DELIVERY_CAP) != delivery_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(scan_frequency, MIN_SCAN_FREQUENCY, MAX_SCAN_FREQUENCY)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::vector<ClockData> clock_data_table{};

            for (std::chrono::nanoseconds update_interval_entry: update_interval_table){
                if (std::clamp(update_interval_entry, MIN_UPDATE_INTERVAL, MAX_UPDATE_INTERVAL) != update_interval_entry){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);                    
                }

                clock_data_table.push_back(ClockData{.last_updated      = std::nullopt,
                                                     .update_interval   = update_interval_entry});
            }

            return std::make_unique<ClockCollector>(std::move(range_press),
                                                    std::move(consumer),
                                                    spawn_bonsai_frequencizer(),
                                                    std::move(clock_data_table),
                                                    ops_clock_resolution,
                                                    collect_cap,
                                                    delivery_cap,
                                                    scan_frequency);
        }

        static auto spawn_clock_try_collector(std::shared_ptr<RangePressInterface> range_press,
                                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                              const std::vector<std::chrono::nanoseconds>& update_interval_table,
                                              size_t ops_clock_resolution,
                                              size_t collect_cap,
                                              size_t delivery_cap,
                                              uint32_t scan_frequency) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            const std::chrono::nanoseconds MIN_UPDATE_INTERVAL  = std::chrono::nanoseconds(1);
            const std::chrono::nanoseconds MAX_UPDATE_INTERVAL  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::days(1));
            const size_t MIN_OPS_CLOCK_RESOLUTION               = 0u;
            const size_t MAX_OPS_CLOCK_RESOLUTION               = std::numeric_limits<size_t>::max();
            const size_t MIN_COLLECT_CAP                        = 1u;
            const size_t MAX_COLLECT_CAP                        = size_t{1} << 30;
            const size_t MIN_DELIVERY_CAP                       = 1u;
            const size_t MAX_DELIVERY_CAP                       = size_t{1} << 30;
            const uint32_t MIN_SCAN_FREQUENCY                   = 1u;
            const uint32_t MAX_SCAN_FREQUENCY                   = 1000000000UL; 

            if (range_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (consumer == nullptr){
                dg::networK_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (range_press->size() != update_interval_table.size()){
                dg::networK_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(ops_clock_resolution, MIN_OPS_CLOCK_RESOLUTION, MAX_OPS_CLOCK_RESOLUTION) != ops_clock_resolution){
                dg::networK_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(collect_cap, MIN_COLLECT_CAP, MAX_COLLECT_CAP) != collect_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(delivery_cap, MIN_DELIVERY_CAP, MAX_DELIVERY_CAP) != delivery_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(scan_frequency, MIN_SCAN_FREQUENCY, MAX_SCAN_FREQUENCY) != scan_frequency){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::vector<ClockData> clock_data_table{};

            for (std::chrono::nanoseconds update_interval_entry: update_interval_table){
                if (std::clamp(update_interval_entry, MIN_UPDATE_INTERVAL, MAX_UPDATE_INTERVAL) != update_interval_entry){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                clock_data_table.push_back(ClockData{.last_updated      = std::nullopt,
                                                     .update_interval   = update_interval_entry});
            }

            return std::make_unique<ClockTryCollector>(std::move(range_press),
                                                       std::move(consumer),
                                                       spawn_bonsai_frequencizer(),
                                                       std::move(clock_data_table),
                                                       ops_clock_resolution,
                                                       collect_cap,
                                                       delivery_cap,
                                                       scan_frequency);
        }

        static auto spawn_clock_competitive_try_collector(std::shared_ptr<RangePressInterface> range_press,
                                                          std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                                          const std::vector<std::chrono::nanoseconds>& update_interval_table,
                                                          size_t ops_clock_resolution,
                                                          size_t collect_cap,
                                                          size_t delivery_cap,
                                                          uint32_t scan_frequency) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{

            const std::chrono::nanoseconds MIN_UPDATE_INTERVAL  = std::chrono::nanoseconds(1);
            const std::chrono::nanoseconds MAX_UPDATE_INTERVAL  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::days(1));
            const size_t MIN_OPS_CLOCK_RESOLUTION               = 0u;
            const size_t MAX_OPS_CLOCK_RESOLUTION               = std::numeric_limits<size_t>::max();
            const size_t MIN_COLLECT_CAP                        = 1u;
            const size_t MAX_COLLECT_CAP                        = size_t{1} << 30;
            const size_t MIN_DELIVERY_CAP                       = 1u;
            const size_t MAX_DELIVERY_CAP                       = size_t{1} << 30;
            const uint32_t MIN_SCAN_FREQUENCY                   = 1u;
            const uint32_t MAX_SCAN_FREQUENCY                   = 1000000000UL;

            if (range_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (consumer == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (range_press->size() != update_interval_table.size()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(ops_clock_resolution, MIN_OPS_CLOCK_RESOLUTION, MAX_OPS_CLOCK_RESOLUTION) != ops_clock_resolution){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(collect_cap, MIN_COLLECT_CAP, MAX_COLLECT_CAP) != collect_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(delivery_cap, MIN_DELIVERY_CAP, MAX_DELIVERY_CAP) != delivery_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(scan_frequency, MIN_SCAN_FREQUENCY, MAX_SCAN_FREQUENCY) != scan_frequency){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::vector<ClockSuffixData> clock_data_table{};

            for (std::chrono::nanoseconds update_interval_entry: update_interval_table){
                if (std::clamp(update_interval_entry, MIN_UPDATE_INTERVAL, MAX_UPDATE_INTERVAL) != update_interval_entry){
                    dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
                }

                clock_data_table.push_back(ClockSuffixData{.last_updated    = std::nullopt,
                                                           .update_interval = update_interval_entry,
                                                           .suffix_idx      = clock_data_table.size()});
            }

            return std::make_unique<ClockCompetitiveTryCollector>(std::move(range_press),
                                                                  std::move(consumer),
                                                                  spawn_bonsai_frequencizer(),
                                                                  std::move(clock_data_table),
                                                                  ops_clock_resolution,
                                                                  collect_cap,
                                                                  delivery_cap,
                                                                  scan_frequency);
        } 

        static auto spawn_revolution_collector(std::shared_ptr<RangePressInterface> range_press,
                                               std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                               const std::vector<uint32_t>& update_revolution_table,
                                               size_t collect_cap,
                                               size_t delivery_cap,
                                               uint32_t scan_frequency) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{

            const size_t MIN_COLLECT_CAP        = 1u;
            const size_t MAX_COLLECT_CAP        = size_t{1} << 30;
            const size_t MIN_DELIVERY_CAP       = 1u;
            const size_t MAX_DELIVERY_CAP       = size_t{1} << 30;
            const uint32_t MIN_SCAN_FREQUENCY   = 1u;
            const uint32_t MAX_SCAN_FREQUENCY   = 1000000000UL;

            if (range_press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (consumer == nullptr){
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

            if (std::clamp(scan_frequency, MIN_SCAN_FREQUENCY, MAX_SCAN_FREQUENCY) != scan_frequency){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::vector<RevolutionData> revolution_table{};

            for (uint32_t update_revolution: update_revolution_table){
                revolution_table.push_back(RevolutionData{.current_revolution           = 0u, 
                                                          .revolution_update_threshold  = update_revolution});
            }

            return std::make_unique<RevolutionCollector>(std::move(range_press), 
                                                         std::move(consumer), 
                                                         spawn_bonsai_frequencizer(),  
                                                         std::move(revolution_table), 
                                                         collect_cap, 
                                                         delivery_cap,
                                                         scan_frequency);
        }
    };
}

#endif 