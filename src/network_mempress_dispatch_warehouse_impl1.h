#ifndef __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_IMPL1_H__
#define __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_IMPL1_H__

#include "network_mempress_dispatch_warehouse_interface.h"
#include "network_memcommit_model.h" 
#include <stdint.h>
#include <stdlib.h>
#include "network_std_container.h"
#include <optional>
#include "stdx.h"
#include <atomic> 

namespace dg::network_mempress_dispatch_warehouse_impl1{

    using event_t = dg::network_memcommit_factory::virtual_memory_event_t;

    class NormalWareHouse: public virtual dg::network_mempress_dispatch_warehouse::WareHouseInterface{

        private:

            dg::pow2_cyclic_queue<dg::vector<event_t>> production_queue;
            dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, std::optional<dg::vector<event_t>> *>> waiting_queue;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> event_consume_sz;
            stdx::inplace_hdi_container<std::atomic<size_t>> sz_concurrent_var;

        public:

            NormalWareHouse(dg::pow2_cyclic_queue<dg::vector<event_t>> production_queue,
                            dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, std::optional<dg::vector<event_t>> *>> waiting_queue,
                            std::unique_ptr<std::mutex> mtx,
                            size_t event_consume_sz,
                            size_t sz_concurrent_var) noexcept: production_queue(std::move(production_queue)),
                                                                waiting_queue(std::move(waiting_queue)),
                                                                mtx(std::move(mtx)),
                                                                event_consume_sz(stdx::hdi_container<size_t>{event_consume_sz}),
                                                                sz_concurrent_var(std::in_place_t{}, sz_concurrent_var){}

            auto push(dg::vector<event_t>&& event_vec) noexcept -> std::expected<bool, exception_t>{

                if (event_vec.empty()){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                if (event_vec.size() > this->max_consume_size()){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                std::binary_semaphore * releasing_smp   = nullptr;

                std::expected<bool, exception_t> rs     = [&, this]() noexcept{
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->waiting_queue.empty()){
                        auto [fetching_smp, fetching_addr] = this->waiting_queue.front();
                        this->waiting_queue.pop_front();
                        *fetching_addr  = std::optional<dg::vector<event_t>>(std::move(event_vec));
                        std::atomic_signal_fence(std::memory_order_seq_cst);
                        releasing_smp   = fetching_smp; 

                        return std::expected<bool, exception_t>(true);
                    }

                    if (this->production_queue.size() == this->production_queue.capacity()){
                        return std::expected<bool, exception_t>(false);
                    }

                    dg::network_exception_handler::nothrow_log(this->production_queue.push_back(std::move(event_vec)));
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    this->sz_concurrent_var.value.fetch_add(1u, std::memory_order_relaxed);

                    return std::expected<bool, exception_t>(true);
                }();

                if (releasing_smp != nullptr){
                    releasing_smp->release();
                }

                return rs;
            }

            auto pop() noexcept -> dg::vector<event_t>{

                std::binary_semaphore fetching_smp(0);
                std::optional<dg::vector<event_t>> fetching_data(std::nullopt); 

                {
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->production_queue.empty()){
                        auto rs         = std::move(this->production_queue.front());
                        this->production_queue.pop_front();
                        this->sz_concurrent_var.value.fetch_sub(1u, std::memory_order_relaxed);

                        return rs;
                    }

                    if constexpr(DEBUG_MODE_FLAG){
                        if (this->waiting_queue.size() == this->waiting_queue.capacity()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    dg::network_exception_handler::nothrow_log(this->waiting_queue.push_back(std::make_pair(&fetching_smp, &fetching_data)));
                }

                fetching_smp.acquire();
                return dg::vector<event_t>(std::move(fetching_data.value()));
            }

            auto pop_nowait() noexcept -> std::optional<dg::vector<event_t>>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (this->production_queue.empty()){
                    return std::nullopt;
                }

                auto rs = std::move(this->production_queue.front());
                this->production_queue.pop_front();

                return std::optional<dg::vector<event_t>>(std::move(rs));
            }

            auto size() noexcept -> size_t{

                return this->sz_concurrent_var.value.load(std::memory_order_relaxed);
            } 

            auto max_consume_size() noexcept -> size_t{

                return this->event_consume_sz.value;
            }
    };

    //the only and sole problem that we are looking forward to solve is the waiting queue distribution

    //we can prove that by using induction, if the queue is uniformly distributed, a probably uniformly distributed move would put it in a probably uniformly distributed state
    //it is missing an equilibrium factor, which is, despite my sincerest efforts, I have been unable to eliminate from, what is, otherwise the perfection of mathematical harmony
    //alright, we'll have a periodic transmissioner to bring the number of waiting threads -> 0
    //every 100 seconds for example

    //how could we prove that if we exhaust the waiting queue at a certain point (rescue_packet_sz = warehouse_concurrency_sz * actual_concurrency_sz), it is guaranteed that the waiting state is uniformly distributed 
    //we are back to the "probably" uniformly distributed again, we aren't certain, we are playing with random statistics, and adding an "equilibrium factor" to make sure that our induction is complete

    //in order for this to work, we have to make sure that the empty_curious_pop_sz has to empty the queue before doing actual waiting, so our waiting state is not skewed 
    //this is probably very hard to write

    //we'd have to implement that, even though we'd not use that if there are not indicated performance contraints

    class DistributedWareHouse: public virtual dg::network_mempress_dispatch_warehouse::WareHouseInterface{

        private:

            std::unique_ptr<std::unique_ptr<NormalWareHouse>[]> warehouse_arr;
            size_t pow2_warehouse_arr_sz;
            size_t empty_curious_pop_sz;

        public:

            DistributedWareHouse(std::unique_ptr<std::unique_ptr<NormalWareHouse>[]> warehouse_arr,
                                 size_t pow2_warehouse_arr_sz,
                                 size_t empty_curious_pop_sz): warehouse_arr(std::move(warehouse_arr)),
                                                               pow2_warehouse_arr_sz(pow2_warehouse_arr_sz),
                                                               empty_curious_pop_sz(empty_curious_pop_sz){}

            auto push(dg::vector<event_t>&& event_vec) noexcept -> std::expected<bool, exception_t>{

                size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                size_t idx          = random_value & (this->pow2_warehouse_arr_sz - 1u);

                return this->warehouse_arr[idx]->push(std::move(event_vec));
            }

            auto pop() noexcept -> dg::vector<event_t>{

                for (size_t i = 0u; i < this->empty_curious_pop_sz; ++i){
                    size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                    size_t idx          = random_value & (this->pow2_warehouse_arr_sz - 1u);

                    std::optional<dg::vector<event_t>> rs = this->warehouse_arr[idx]->pop_nowait();

                    if (rs.has_value()){
                        return dg::vector<event_t>(std::move(rs.value()));
                    }
                }

                //can we prove that this is uniformly distributed, yes, because the next operation is "likely" to be waiting after a sufficient number of iterations (or samplings)
                //so we can prove our likelyhood of uniform distribution of subscripted waiting threads (random_value is uniformly distributed => waiting_queue is uniformly distributed by using induction of previous state being already uniformly distributed)

                size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                size_t idx          = random_value & (this->pow2_warehouse_arr_sz - 1u);

                return this->warehouse_arr[idx]->pop();
            }
    };

    struct Factory{

        static auto spawn_warehouse(size_t production_queue_cap,
                                    size_t max_concurrency_sz,
                                    size_t unit_consumption_sz) -> std::unique_ptr<NormalWareHouse>{
        
            const size_t MIN_PRODUCTION_QUEUE_CAP   = 1u;
            const size_t MAX_PRODUCTION_QUEUE_CAP   = size_t{1} << 30;
            const size_t MIN_MAX_CONCURRENCY_SZ     = 1u;
            const size_t MAX_MAX_CONCURRENCY_SZ     = size_t{1} << 30;
            const size_t MIN_UNIT_CONSUMPTION_SZ    = 1u;
            const size_t MAX_UNIT_CONSUMPTION_SZ    = size_t{1} << 30;

            if (std::clamp(production_queue_cap, MIN_PRODUCTION_QUEUE_CAP, MAX_PRODUCTION_QUEUE_CAP) != production_queue_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(max_concurrency_sz, MIN_MAX_CONCURRENCY_SZ, MAX_MAX_CONCURRENCY_SZ) != max_concurrency_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(unit_consumption_sz, MIN_UNIT_CONSUMPTION_SZ, MAX_UNIT_CONSUMPTION_SZ) != unit_consumption_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(production_queue_cap)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(max_concurrency_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT)
            }

            return std::make_unique<NormalWareHouse>(dg::pow2_cyclic_queue<dg::vector<event_t>>(stdx::ulog2(production_queue_cap)),
                                                     dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, std::optional<dg::vector<event_t>> *>>(stdx::ulog2(max_concurrency_sz)),
                                                     std::make_unique<std::mutex>(),
                                                     unit_consumption_sz,
                                                     0u);
        }

        static auto spawn_distributed_warehouse(size_t production_queue_cap,
                                                size_t max_concurrency_sz,
                                                size_t unit_consumption_sz,
                                                size_t warehouse_concurrency,
                                                size_t empty_curious_pop_sz) -> std::unique_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface>{
                    
            const size_t MIN_WAREHOUSE_CONCURRENCY  = 1u;
            const size_t MAX_WAREHOUSE_CONCURRENCY  = size_t{1} << 30;
            const size_t MIN_EMPTY_CURIOUS_POP_SZ   = 0u;
            const size_t MAX_EMPTY_CURIOUS_POP_SZ   = size_t{1} << 30;
            
            if (std::clamp(warehouse_concurrency, MIN_WAREHOUSE_CONCURRENCY, MAX_WAREHOUSE_CONCURRENCY) != warehouse_concurrency){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(warehouse_concurrency)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(empty_curious_pop_sz, MIN_EMPTY_CURIOUS_POP_SZ, MAX_EMPTY_CURIOUS_POP_SZ) != empty_curious_pop_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::unique_ptr<std::unique_ptr<NormalWareHouse>[]> warehouse_arr = std::make_unique<std::unique_ptr<NormalWareHouse>[]>(warehouse_concurrency);

            for (size_t i = 0u; i < warehouse_concurrency; ++i){
                warehouse_arr[i] = spawn_warehouse(production_queue_cap, max_concurrency_sz, unit_consumption_sz);
            }

            return std::make_unique<DistributedWareHouse>(std::move(warehouse_arr), warehouse_concurrency, empty_curious_pop_sz);
        }
    };
}

#endif