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

                if (event_vec.size() == 0u){
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
                this->sz_concurrent_var.value.fetch_sub(1u, std::memory_order_relaxed);

                return rs;
            }

            auto size() noexcept -> size_t{

                return this->sz_concurrent_var.value.load(std::memory_order_relaxed);
            } 

            auto max_consume_size() noexcept -> size_t{

                return this->event_consume_sz.value;
            }
    };

    //As I was considering every possible approach for how this could work
    //This is the best approach in terms of no forever waiting + affinity + latency + etc
    //the unordered_map rendered useless if it is not affined + const which defeats the initial purpose of doing distributed warehouse (we are synchronizing accesses, which is the NormalWareHouse responsibility)

    //this is not the socket problem where we could use complex reactor to force all of the waiting queue to be less than a certain size and use a forward cut every interval to make sure that we aren't missing out the bottom guys
    //we didn't aim for latency in that case
    //we aim for absolute latency in this case

    //we made the resolutor such that its only responsibility is to dispatch to asynchronous devices
    //so we dont have to worry about the "worklife balance" of those guys
    //except for some simple get/setters or ping pong which is "in the acceptable range"
    //we however, could implement an empty_curious to partly solve that

    //we'd want to solve two problems: 
    //the problem of pushing into an empty container -> OK, thru
    //the problem of popping a not empty container -> OK, thru

    //otherwise, we'd need empty_curious 

    class HybridAffinedWareHouse: public virtual dg::network_mempress_dispatch_warehouse::WareHouseInterface{

        private:

            std::unique_ptr<std::unique_ptr<dg::network_mempress_dispatch_warehouse::NormalWareHouse>[]> warehouse_arr;
            size_t pow2_warehouse_arr_sz;
            dg::unordered_unstable_map<size_t, size_t> dedicated_thread_to_warehouse_consumption_map;
            size_t event_consume_sz;
            size_t notempty_curious_push_sz;
            size_t empty_curious_pop_sz;

        public:

            HybridAffinedWareHouse(std::unique_ptr<std::unique_ptr<dg::network_mempress_dispatch_warehouse::NormalWareHouse>[]> warehouse_arr,
                                   size_t pow2_warehouse_arr_sz,
                                   dg::unordered_unstable_map<size_t, size_t> dedicated_thread_to_warehouse_consumption_map,
                                   size_t event_consume_sz,
                                   size_t notempty_curious_push_sz,
                                   size_t empty_curious_pop_sz) noexcept: warehouse_arr(std::move(warehouse_arr)),
                                                                          pow2_warehouse_arr_sz(pow2_warehouse_arr_sz),
                                                                          dedicated_thread_to_warehouse_consumption_map(std::move(dedicated_thread_to_warehouse_consumption_map)),
                                                                          event_consume_sz(event_consume_sz),
                                                                          notempty_curious_push_sz(notempty_curious_push_sz),
                                                                          empty_curious_pop_sz(empty_curious_pop_sz){}

            auto push(dg::vector<event_t>&& event_vec) noexcept -> std::expected<bool, exception_t>{

                std::optional<size_t> clue_idx  = std::nullopt;
                size_t clue_sz                  = {};

                for (size_t i = 0u; i < this->notempty_curious_push_sz; ++i){
                    size_t random_value     = dg::network_randomizer::randomize_int<size_t>();
                    size_t idx              = random_value & (this->pow2_warehouse_arr_sz - 1u); 
                    size_t cur_sz           = this->warehouse_arr[idx]->size();

                    if (cur_sz == 0u){
                        return this->warehouse_arr[idx]->push(std::move(event_vec));
                    }

                    if (!clue_idx.has_value()){
                        clue_idx    = i;
                        clue_sz     = cur_sz;
                    }

                    if (cur_sz < clue_sz){
                        clue_idx    = i;
                        clue_sz     = cur_sz;
                    }
                }

                if (clue_idx.has_value()){
                    return this->warehouse_arr[clue_idx.value()]->push(std::move(event_vec));                  
                }

                size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                size_t idx          = random_value & (this->pow2_warehouse_arr_sz - 1u);

                return this->warehouse_arr[idx]->push(std::move(event_vec));
            }

            auto pop() noexcept -> dg::vector<event_t>{

                auto map_ptr = std::as_const(this->dedicated_thread_to_warehouse_consumption_map).find(dg::network_concurrency::this_thread_idx());

                if (map_ptr != std::as_const(this->dedicated_thread_to_warehouse_consumption_map).end()){
                    return this->warehouse_arr[map_ptr->second]->pop();
                }

                for (size_t i = 0u; i < this->empty_curious_pop_sz; ++i){
                    size_t random_value     = dg::network_randomizer::randomize_int<size_t>();
                    size_t idx              = random_value & (this->pow2_warehouse_arr_sz - 1u);
                    size_t cur_sz           = this->warehouse_arr[idx]->size();

                    if (cur_sz != 0u){
                        std::optional<dg::vector<event_t>> rs = this->warehouse_arr[idx]->pop_nowait();

                        if (rs.has_value()){
                            return dg::vector<event_t>(std::move(rs.value()));
                        }
                    }
                }

                size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                size_t idx          = random_value & (this->pow2_warehouse_arr_sz - 1u);

                return this->warehouse_arr[idx]->pop();
            }

            auto max_consume_size() noexcept -> size_t{

                return this->event_consume_sz;
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

        static auto spawn_hybrid_distributed_warehouse(size_t base_production_warehouse_cap,
                                                       size_t base_unit_consumption_sz,
                                                       size_t max_concurrency_sz,
                                                       size_t warehouse_concurrency_sz,
                                                       size_t notempty_curious_push_sz,
                                                       size_t empty_curious_pop_sz,
                                                       const std::unordered_map<size_t, size_t>& dedicated_thread_id_to_warehouse_idx_map) -> std::unique_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface>{

            const size_t MIN_WAREHOUSE_CONCURRENCY_SZ   = 1u;
            const size_t MAX_WAREHOUSE_CONCURRENCY_SZ   = size_t{1} << 20;
            
            const size_t MIN_NOTEMPTY_CURIOUS_PUSH_SZ   = 0u;
            const size_t MAX_NOTEMPTY_CURIOUS_PUSH_SZ   = size_t{1} << 20;
            
            const size_t MIN_EMPTY_CURIOUS_POP_SZ       = 0u;
            const size_t MAX_EMPTY_CURIOUS_POP_SZ       = size_t{1} << 20;

            if (std::clamp(warehouse_concurrency_sz, MIN_WAREHOUSE_CONCURRENCY_SZ, MAX_WAREHOUSE_CONCURRENCY_SZ) != warehouse_concurrency_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(warehouse_concurrency_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(notempty_curious_push_sz, MIN_NOTEMPTY_CURIOUS_PUSH_SZ, MAX_NOTEMPTY_CURIOUS_PUSH_SZ) != notempty_curious_push_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(empty_curious_pop_sz, MIN_EMPTY_CURIOUS_POP_SZ, MAX_EMPTY_CURIOUS_POP_SZ) != empty_curious_pop_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            dg::unordered_unstable_map<size_t, size_t> internal_map(dedicated_thread_id_to_warehouse_idx_map.begin(), dedicated_thread_id_to_warehouse_idx_map.end(), 1u);

            for (const auto& _map_pair: internal_map){
                if (_map_pair.second >= this->warehouse_concurrency_sz){
                    dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
                }
            }

            std::unique_ptr<std::unique_ptr<NormalWareHouse>[]> warehouse_arr = std::make_unique<std::unique_ptr<NormalWareHouse>[]>(warehouse_concurrency_sz);

            for (size_t i = 0u; warehouse_concurrency_sz; ++i){
                warehouse_arr[i] = spawn_warehouse(base_production_warehouse_cap,
                                                   max_concurrency_sz,
                                                   base_unit_consumption_sz);
            }

            return std::make_unique<HybridAffinedWareHouse>(std::move(warehouse_arr),
                                                            warehouse_concurrency_sz,
                                                            std::move(internal_map),
                                                            base_unit_consumption_sz,
                                                            notempty_curious_push_sz,
                                                            empty_curious_pop_sz);
        }
    };
}

#endif