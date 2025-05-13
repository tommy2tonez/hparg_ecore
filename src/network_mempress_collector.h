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
        virtual void get(size_t idx, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept = 0;
    };

    struct BonsaiFrequencierInterface{
        virtual ~BonsaiFrequencierInterface() noexcept = default;
        virtual void tick() noexcept = 0;
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

    //let's see whats going on

    //we have consumer
    //we have range_press
    //we are trying to try_collect the memregions as fast as possible
    //if it is not successfully try_collected, we are fine, it's being operated on by other guys (either the producer or consumer 50% 50%, we dont really know)
    //the problem is the immediate ConsumerInterface<event_t>
    //we are having lock contention problem at the point of ConsumerInterface<event_t>
    //we are context switching + polluting all the cache fetch of the up-to-point memregions
    //we'll attempt to solve that by using a bouncing container
    //alright, this is now the question of dynamic memory allocations or stack_allocations whether vector + friends are better (in terms of polluting RAM), we'll be talking about that
    //the problem with stack is that it is already IN_CACHE, so we dont have the problem of polluting RAM, yet the problem of wasting CPU flops

    //we'll implement a frequencizer, this is a fancy name to meet a frequency requirement without reading the update interval literally every nanoseconds
    //we implement a first version of "prioritized" collector
    //by leveraging pivot + reverse levelwise tree traversal

    //0 -> 1
    //0 -> 3
    //0 -> 7
    //0 -> 15

    //the number is very sound when we have a heap of 15 height
    //we fit everything in the CPU cache

    //what is a good number
    //1024 mempress regions, the only reason we are not serializing the access of 1024 mempress -> 1 mempress is the problem of serialization overheads
    //the number of mempress regions are only to reduce the serialization overheads, does not represent the memlocks or etc

    //1024 mempress regions, with an iterable sz of 2036, average frequencied region iterations per scan == 3 -> 4
    //so the actual iterable sz reduced -> 512
    //the number is sound
    //plus, we established a basic retry_prioritized for lock contention

    //I think this should suffice

    //I was thinking about ClockCompetitiveTryCollector ->  ClockCollector

    //ClockCompetitiveTryCollector should suffice for most of the cases
    //ClockCollector would prune the minor cases (the absolute worst cases)

    //CompetitiveTryCollector should suffice for most of the cases that require ASAP notification (without temporal)
    //and ClockCollector would prune the minor cases (the absolute worst cases)

    class TryCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::shared_ptr<BonsaiFrequencierInterface> frequencizer;
            const size_t collect_cap;
            const size_t delivery_cap;

        public:

            TryCollector(std::shared_ptr<RangePressInterface> range_press,
                         std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                         std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                         size_t collect_cap,
                         size_t delivery_cap) noexcept: range_press(std::move(range_press)),
                                                        consumer(std::move(consumer)),
                                                        frequencizer(std::move(frequencizer)),
                                                        collect_cap(collect_cap),
                                                        delivery_cap(delivery_cap){}

            auto run_one_epoch() noexcept -> bool{

                this->frequencizer->tick();

                size_t dh_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->consumer.get(), this->delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->consumer.get(), this->delivery_cap, dh_mem.get()));

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);

                size_t range_sz             = this->range_press->size();

                for (size_t i = 0u; i < range_sz; ++i){
                    size_t event_arr_sz = {};

                    if (this->range_press->try_get(i, event_arr.get(), event_arr_sz, this->collect_cap)){
                        std::for_each(event_arr.get(), std::next(event_arr.get(), event_arr_sz), std::bind_front(dg::network_producer_consumer::delvrsrv_deliver_lambda, delivery_handle->get()));
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

        public:

            CompetitiveTryCollector(std::shared_ptr<RangePressInterface> range_press,
                                    std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_>> consumer,
                                    std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                                    std::vector<size_t> suffix_array,
                                    size_t collect_cap,
                                    size_t delivery_cap) noexcept: range_press(std::move(range_press)),
                                                                   consumer(std::move(consumer)),
                                                                   frequencizer(std::move(frequencizer)),
                                                                   suffix_array(std::move(suffix_array)),
                                                                   collect_cap(collect_cap),
                                                                   delivery_cap(delivery_cap){}
            
            auto run_one_epoch() noexcept -> bool{

                //strategize
                //1 -> log2 == 0
                //iterable_sz == [0, 0 + 1)

                this->frequencizer->tick();

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

                    //this is complicated, we'd want to try_collect once every run_one_epoch
                    //do we ?
                    //I dont really know

                    for (size_t j = first; j < last; ++j){
                        size_t region_idx   = this->suffix_array[j]; 
                        size_t event_arr_sz = {};

                        if (this->range_press->try_get(region_idx, event_arr.get(), event_arr_sz, this->collect_cap)){
                            std::for_each(event_arr.get(), std::next(event_arr.get(), event_arr_sz), std::bind_front(dg::network_producer_consumer::delvrsrv_deliver_lambda, delivery_handle->get()));
                        } else{
                            std::swap(this->suffix_array[j], this->suffix_array[first + failed_sz]);
                            failed_sz += 1;
                        }
                    }
                }

                return true;
            }
    };

    struct ClockData{
        std::chrono::time_point<std::chrono::steady_clock> last_updated;
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

        public:

            ClockCollector(std::shared_ptr<RangePressInterface> range_press,
                           std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                           std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                           std::vector<ClockData> clock_data_table,
                           size_t ticking_clock_resolution,
                           size_t collect_cap,
                           size_t delivery_cap) noexcept: range_press(std::move(range_press)),
                                                          consumer(std::move(consumer)),
                                                          frequencizer(std::move(frequencizer)),
                                                          clock_data_table(std::move(clock_data_table)),
                                                          ticking_clock_resolution(ticking_clock_resolution),
                                                          collect_cap(collect_cap),
                                                          delivery_cap(delivery_cap){}

            auto run_one_epoch() noexcept -> bool{

                this->frequencizer->tick();

                size_t dh_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->consumer.get(), this->delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->consumer.get(), this->delivery_cap, dh_mem.get()));

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);
                size_t range_sz             = this->range_press->size(); 
                auto ticking_steady_clock   = dg::ticking_steady_clock(this->ticking_clock_resolution); 

                for (size_t i = 0u; i < range_sz; ++i){
                    std::chrono::time_point<std::chrono::steady_clock> local_now = ticking_steady_clock.get();

                    if constexpr(DEBUG_MODE_FLAG){
                        if (i >= this->clock_data_table.size()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    std::chrono::nanoseconds lifetime = local_now - this->clock_data_table[i].last_updated;

                    if (lifetime < this->clock_data_table[i].update_interval){ //not expired
                        continue;
                    }

                    size_t event_arr_sz = {};
                    this->range_press->get(i, event_arr.get(), event_arr_sz, this->collect_cap); //
                    std::for_each(event_arr.get(), std::next(event_arr.get(), event_arr_sz), std::bind_front(dg::network_producer_consumer::delvrsrv_deliver_lambda, delivery_handle->get()));

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

        public:

            ClockTryCollector(std::shared_ptr<RangePressInterface> range_press,
                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                              std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                              std::vector<ClockData> clock_data_table,
                              size_t ticking_clock_resolution,
                              size_t collect_cap,
                              size_t delivery_cap) noexcept: range_press(std::move(range_press)),
                                                             consumer(std::move(consumer)),
                                                             frequencizer(std::move(frequencizer)),
                                                             clock_data_table(std::move(clock_data_table)),
                                                             ticking_clock_resolution(ticking_clock_resolution),
                                                             collect_cap(collect_cap),
                                                             delivery_cap(delivery_cap){}

            auto run_one_epoch() noexcept -> bool{

                this->frequencizer->tick();

                size_t dh_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->consumer.get(), this->delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->consumer.get(), this->delivery_cap, dh_mem.get()));

                dg::network_stack_allocation::NoExceptAllocation<event_t[]> event_arr(this->collect_cap);
                size_t range_sz             = this->range_press->size(); 
                auto clock                  = dg::ticking_steady_clock(this->ticking_clock_resolution); 

                for (size_t i = 0u; i < range_sz; ++i){
                    std::chrono::time_point<std::chrono::steady_clock> local_now = clock.get();

                    if constexpr(DEBUG_MODE_FLAG){
                        if (i >= this->clock_data_table.size()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    std::chrono::nanoseconds lifetime = local_now - this->clock_data_table[i].last_updated;

                    if (lifetime < this->clock_data_table[i].update_interval){ //not expired
                        continue;
                    }

                    size_t event_arr_sz = {};

                    if (this->range_press->try_get(i, event_arr.get(), event_arr_sz, this->collect_cap)){
                        std::for_each(event_arr.get(), std::next(event_arr.get(), event_arr_sz), std::bind_front(dg::network_producer_consumer::delvrsrv_deliver_lambda, delivery_handle->get()));
                        this->clock_data_table[i].last_updated = clock.get();    
                    }
                }

                return true;
            }
    };

    class ClockSuffixData{
        std::chrono::time_point<std::chrono::steady_clock> last_updated;
        std::chrono::nanoseconds update_interval;
        uint32_t suffix_idx;
    };

    class ClockCompetitiveTryCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::shared_ptr<BonsaiFrequencierInterface> frequencizer;
            std::vector<ClockSuffixData> clock_data_table;
            const size_t ticking_clock_resolution;
            const size_t collect_cap;
            const size_t delivery_cap;
        
        
        public:

            ClockCompetitiveTryCollector(std::shared_ptr<RangePressInterface> range_press,
                                         std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                         std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                                         std::vector<ClockSuffixData> clock_data_table,
                                         size_t ticking_clock_resolution,
                                         size_t collect_cap,
                                         size_t delivery_cap) noexcept: range_press(std::move(range_press)),
                                                                        consumer(std::move(consumer)),
                                                                        frequencizer(std::move(frequencizer)),
                                                                        clock_data_table(std::move(clock_data_table)),
                                                                        ticking_clock_resolution(ticking_clock_resolution),
                                                                        collect_cap(collect_cap),
                                                                        delivery_cap(delivery_cap){}

            auto run_one_epoch() noexcept -> bool{

                this->frequencizer->tick();

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

                auto clock              = dg::ticking_steady_clock(this->ticking_clock_resolution); 

                size_t max_log2_exp     = stdx::ulog2(stdx::ceil2(this->clock_data_table.size()));
                size_t iterable_sz      = max_log2_exp + 1u;
                
                for (size_t i = 0u; i < iterable_sz; ++i){
                    size_t first        = 0u;
                    size_t last         = std::min(size_t{1} << i, static_cast<size_t>(this->clock_data_table.size()));  
                    size_t failed_sz    = 0u; 

                    for (size_t j = first; j < last; ++j){
                        size_t region_idx                                               = this->clock_data_table[j].suffix_idx; 
                        std::chrono::nanoseconds update_interval                        = this->clock_data_table[j].update_interval; //this is very expensive
                        std::chrono::time_point<std::chrono::steady_clock> last_updated = this->clock_data_table[j].last_updated;
                        std::chrono::time_point<std::chrono::steady_clock> now          = clock.get();
                        std::chrono::nanoseconds lifetime                               = now - last_updated;

                        if (lifetime < update_interval){
                            continue;
                        }

                        size_t event_arr_sz = {};

                        if (this->range_press->try_get(region_idx, event_arr.get(), event_arr_sz, this->collect_cap)){
                            std::for_each(event_arr.get(), std::next(event_arr.get(), event_arr_sz), std::bind_front(dg::network_producer_consumer::delvrsrv_deliver_lambda, delivery_handle->get()));
                            this->clock_data_table[j].last_updated = clock.get();        
                        } else{
                            std::swap(this->clock_data_table[j], this->clock_data_table[first + failed_sz]);
                            failed_sz += 1;
                        }
                    }
                }
            }
    };

    //we have yet want to abstract this -> ClockCollector
    struct RevolutionData{
        uint32_t current_revolution;
        uint32_t update_sz_pow2_exp; 
    };

    class RevolutionCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RangePressInterface> range_press;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::shared_ptr<BonsaiFrequencierInterface> frequencizer;
            std::vector<RevolutionData> revolution_table;
            const size_t collect_cap;
            const size_t delivery_cap;

        public:

            RevolutionCollector(std::shared_ptr<RangePressInterface> range_press,
                                std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                                std::shared_ptr<BonsaiFrequencierInterface> frequencizer,
                                std::vector<RevolutionData> revolution_table,
                                size_t collect_cap,
                                size_t delivery_cap) noexcept: range_press(std::move(range_press)),
                                                               consumer(std::move(consumer)),
                                                               frequencizer(std::move(frequencizer)),
                                                               revolution_table(std::move(revolution_table)),
                                                               collect_cap(collect_cap),
                                                               delivery_cap(delivery_cap){}

            auto run_one_epoch() noexcept -> bool{

                this->frequencizer->tick();

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
                    size_t update_revolution    = size_t{1} << this->revolution_table[i].update_sz_pow2_exp; 
                    current_revolution          += 1;

                    if (current_revolution == update_revolution){
                        size_t event_arr_sz = {};
                        this->range_press->get(i, event_arr.get(), event_arr_sz, this->collect_cap);
                        std::for_each(event_arr.get(), std::next(event_arr.get(), event_arr_sz), std::bind_front(dg::network_producer_consumer::delvrsrv_deliver_lambda, delivery_handle->get()));
                        current_revolution  = 0u;
                    }
                }

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