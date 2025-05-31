#ifndef __DG_NETWORK_MEMCOMMIT_MESSENGER_H__
#define __DG_NETWORK_MEMCOMMIT_MESSENGER_H__

#include "network_mempress.h" 
#include "network_mempress_dispatch_warehouse.h"
#include "network_producer_consumer.h"

namespace dg::network_memcommit_messenger{

    class MemregionRadixerInterface{

        public:

            using memregion_kind_t = uint8_t;

            static inline constexpr uint8_t EXPRESS_HIGH_LATENCY_REGION = 0u;
            static inline constexpr uint8_t EXPRESS_MID_LATENCY_REGION  = 1u;
            static inline constexpr uint8_t EXPRESS_LOW_LATENCY_REGION  = 2u;
            static inline constexpr uint8_t NOMRAL_REGION               = 3u;

            virtual ~MemregionRadixerInterface() noexcept = default;
            virtual auto radix(uma_ptr_t) noexcept -> std::expected<memregion_kind_t, exception_t> = 0;
    };

    class WareHouseIngestionConnectorInterface{

        public:

            virtual ~WareHouseIngestionConnectorInterface() noexcept = defautl;
            virtual auto push(dg::vector<virtual_memory_event_t>&&) noexcept -> std::expected<bool, exception_t> = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    //the problem with software is that we just keep writing literally, build abstraction, keep writing
    //we dont really have time to ask why this why that

    class NormalWareHouseConnector: public virtual WareHouseIngestionConnectorInterface{

        private:

            std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse;
        
        public:

            NormalWareHouseConnector(std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse) noexcept: warehouse(std::move(warehouse)){}

            auto push(dg::vector<virtual_memory_event_t>&& event_vec) noexcept -> std::expected<bool, exception_t>{

                return this->warehouse->push(std::move(event_vec));
            }

            auto max_consume_size() noexcept -> size_t{

                return this->warehouse->max_consume_size();
            }
    };

    class ExhaustionControlledWareHouseConnector: public virtual WareHouseIngestionConnectorInterface{

        private:

            std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device;

        public:

            ExhaustionControlledWareHouseConnector(std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse,
                                                   std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device) noexcept: warehouse(std::move(warehouse)),
                                                                                                                                                     infretry_device(std::move(infretry_device)){}

            auto push(dg::vector<virtual_memory_event_t>&& event_vec) noexcept -> std::expected<bool, exception_t>{

                std::expected<bool, exception_t> rs = std::unexpected(dg::network_exception::EXPECTED_NOT_INITIALIZED);
                
                auto task = [&, this]() noexcept{
                    rs = this->warehouse->push(static_cast<dg::vector<virtual_memory_event_t>&&>(event_vec));

                    if (!rs.has_value()){
                        return true;
                    }

                    return rs.value();
                };

                dg::network_concurrency_infretry_x::ExecutableWrapper virtual_task(task);
                this->infretry_device->exec(virtual_task);

                return rs;
            }

            auto max_consume_size() noexcept -> size_t{

                return this->warehouse->max_consume_size();
            }
    };

    class MemeventMessenger: public virtual dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>{

        private:

            std::shared_ptr<MemregionRadixerInterface> memregion_express_radixer;
            std::shared_ptr<dg::network_mempress::MemoryPressInterface> press;
            size_t press_vectorization_sz;
            std::unique_ptr<WareHouseIngestionConnectorInterface> high_latency_warehouse;
            size_t high_latency_warehouse_feed_cap;
            std::unique_ptr<WareHouseIngestionConnectorInterface> mid_latency_warehouse;
            size_t mid_latency_warehouse_feed_cap;
            std::unique_ptr<WareHouseIngestionConnectorInterface> low_latency_warehouse;
            size_t low_latency_warehouse_feed_cap;

        public:

            MemeventMessenger(std::shared_ptr<MemregionRadixerInterface> memregion_express_radixer,
                              std::shared_ptr<dg::network_mempress::MemoryPressInterface> press,
                              size_t press_vectorization_sz,
                              std::unique_ptr<WareHouseIngestionConnectorInterface> high_latency_warehouse,
                              size_t high_latency_warehouse_feed_cap,
                              std::unique_ptr<WareHouseIngestionConnectorInterface> mid_latency_warehouse,
                              size_t mid_latency_warehouse_feed_cap,
                              std::unique_ptr<WareHouseIngestionConnectorInterface> low_latency_warehouse,
                              size_t low_latency_warehouse_feed_cap) noexcept: memregion_express_radixer(std::move(memregion_express_radixer)),
                                                                               press(std::move(press)),
                                                                               press_vectorization_sz(press_vectorization_sz),
                                                                               high_latency_warehouse(std::move(high_latency_warehouse)),
                                                                               high_latency_warehouse_feed_cap(high_latency_warehouse_feed_cap),
                                                                               mid_latency_warehouse(std::move(mid_latency_warehouse)),
                                                                               mid_latency_warehouse_feed_cap(mid_latency_warehouse_feed_cap),
                                                                               low_latency_warehouse(std::move(low_latency_warehouse)),
                                                                               low_latency_warehouse_feed_cap(low_latency_warehouse_feed_cap){}

            void push(std::move_iterator<virtual_memory_event_t *> data_arr, size_t sz) noexcept{

                //despite my sincerest effort, we decide to devirtualize the memory events here, we wont be using the tricks, because the branching prediction is pretty decent for these guys (we'll be evaluating the performance later)
                //the structure is too heavy to be moved around
                //we are expecting a structure of size 24 - 32, to leverage cache fetch + friends
                //the problem is that the messenger can't really notify the customers about their packages going missing or not delivered, this is expected in real-life, because the overhead of doing so exceeds the benefits of doing so

                virtual_memory_event_t * base_data_arr = data_arr.base();

                auto press_feed_resolutor                               = InternalPressFeedResolutor{};
                press_feed_resolutor.dst                                = this->press.get();
                
                size_t trimmed_press_vectorization_sz                   = std::min(std::min(this->press_vectorization_sz, this->press->max_consume_size()), sz);
                size_t press_feeder_allocation_cost                     = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&press_feed_resolutor, trimmed_press_vectorization_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> press_feeder_mem(press_feeder_allocation_cost);
                auto press_feeder                                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&press_feed_resolutor,
                                                                                                                                                                                             trimmed_press_vectorization_sz,
                                                                                                                                                                                             press_feeder_mem.get()));
                
                //------------------------

                auto high_latency_warehouse_feed_resolutor              = InternalWareHouseFeedResolutor{};
                high_latency_warehoues_feed_resolutor.dst               = this->high_latency_warehouse.get();

                size_t trimmed_high_latency_warehouse_feed_cap          = std::min(std::min(this->high_latency_warehouse_feed_cap, this->high_latency_warehouse->max_consume_size()), sz);
                size_t high_latency_warehouse_feeder_allocation_cost    = dg::network_producer_consumer::delvrsrv_allocation_cost(&high_latency_warehouse_feed_resolutor, trimmed_high_latency_warehouse_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> high_latency_warehouse_feeder_mem(high_latency_warehouse_feeder_allocation_cost);
                auto high_latency_warehouse_feeder                      = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&high_latency_warehouse_feed_resolutor, 
                                                                                                                                                                                          trimmed_high_latency_warehouse_feed_cap,
                                                                                                                                                                                          high_latency_warehouse_feeder_mem.get()));

                //------------------------

                auto mid_latency_warehouse_feed_resolutor               = InternalWareHouseFeedResolutor{};
                mid_latency_warehouse_feed_resolutor.dst                = this->mid_latency_warehouse.get();

                size_t trimmed_mid_latency_warehouse_feed_cap           = std::min(std::min(this->mid_latency_warehouse_feed_cap, this->mid_latency_warehouse->max_consume_size()), sz);
                size_t mid_latency_warehouse_feeder_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&mid_latency_warehouse_feed_resolutor, trimmed_mid_latency_warehouse_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> mid_latency_warehouse_feeder_mem(mid_latency_warehouse_feeder_allocation_cost);
                auto mid_latency_warehouse_feeder                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&mid_latency_warehouse_feed_resolutor,
                                                                                                                                                                                          trimmed_mid_latency_warehouse_feed_cap,
                                                                                                                                                                                          mid_latency_warehouse_feeder_mem.get()));

                //------------------------

                auto low_latency_warehouse_feed_resolutor               = InternalWareHouseFeedResolutor{};
                low_latency_warehouse_feed_resolutor.dst                = this->low_latency_warehouse.get();

                size_t trimmed_low_latency_warehouse_feed_cap           = std::min(std::min(this->low_latency_warehouse_feed_cap, this->low_latency_warehouse->max_consume_size()), sz);
                size_t low_latency_warehouse_feeder_allocation_cost     = dg::network_producer_consumer::delvrsrv_allocation_cost(&low_latency_warehouse_feed_resolutor, trimmed_low_latency_warehouse_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> low_latency_warehouse_feeder_mem(low_latency_warehouse_feeder_allocation_cost);
                auto low_latency_warehouse_feeder                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&low_latency_warehouse_feed_resolutor,
                                                                                                                                                                                          trimmed_low_latency_warehouse_feed_cap,
                                                                                                                                                                                          low_latency_warehouse_feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    uma_ptr_t notifying_addr = this->extract_notifying_addr(base_data_arr[i]);
                    std::expected<MemregionLatencyRadixerInterface::memregion_kind_t, exception_t> memregion_kind = this->memregion_express_radixer->radix(notifying_addr);

                    if (!memregion_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(memregion_kind.error()));
                        continue;
                    }

                    switch (memregion_kind.value()){
                        case MemregionLatencyRadixerInterface::EXPRESS_HIGH_LATENCY_REGION:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(high_latency_warehouse_feeder.get(), std::move(base_data_arr[i]));
                            break;
                        }
                        case MemregionLatencyRadixerInterface::EXPRESS_MID_LATENCY_REGION:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(mid_latency_warehouse_feeder.get(), std::move(base_data_arr[i]));
                            break;
                        }
                        case MemregionLatencyRadixerInterface::EXPRESS_LOW_LATENCY_REGION:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(low_latency_warehouse_feeder.get(), std::move(base_data_arr[i]));
                            break;
                        }
                        case MemregionLatencyRadixerInterface::NOMRAL_REGION
                        {
                            uma_ptr_t notifying_region = dg::memult::region(notifying_addr, this->press->memregion_size());
                            dg::network_producer_consumer::delvrsrv_kv_deliver(press_feeder.get(), notifying_region, std::move(base_data_arr[i]));
                            break;
                        }
                        default:
                        {
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    }
                }
            }

        private:

            static constexpr auto extract_notifying_addr(const virtual_memory_event_t& memevent) noexcept -> uma_ptr_t{

                memory_event_kind_t event_kind = dg::network_memcommit_factory::read_virtual_event_kind(memevent);

                //switch case branch prediction is really good because of the locality of msgr context
                //

                switch (event_kind){
                    case dg::network_memcommit_factory::event_kind_forward_ping_signal:
                    {
                        return dg::network_memcommit_factory::devirtualize_forward_ping_signal_event(memevent).dst;
                    }
                    case dg::network_memcommit_factory::event_kind_forward_pong_request:
                    {
                        return dg::network_memcommit_factory::devirtualize_forward_pong_request_event(memevent).requestee;
                    }
                    case dg::network_memcommit_factory::event_kind_forward_pingpong_request:
                    {
                        return dg::network_memcommit_factory::devirtualize_forward_pingpong_request_event(memevent).requestee;
                    }
                    case dg::network_memcommit_factory::event_kind_forward_do_signal:
                    {
                        return dg::network_memcommit_factory::devirtualize_forward_do_signal_event(memevent).dst;
                    }
                    case dg::network_memcommit_factory::event_kind_backward_do_signal:
                    {
                        return dg::network_memcommit_factory::devirtualize_backward_do_signal_event(memevent).dst;
                    }
                    case dg::network_memcommit_factory::event_kind_signal_aggregation_signal:
                    {
                        // virtual_sigagg_event_t sigagg_event     = dg::network_memcommit_factory::devirtualize_sigagg_signal_event(memevent);
                        // sigagg_event_kind_t sigagg_kind         = dg::network_memcommit_factory::read_aggregation_kind(sigagg_event);

                        // switch (sigagg_kind){
                        //     case dg::network_memcommit_factory::aggregation_kind_forward_ping_signal:
                        //     {
                        //         return dg::network_memcommit_factory::devirtualize_forward_ping_signal_aggregation_event(sigagg_event).sigagg_addr;
                        //     }
                        //     case dg::network_memcommit_factory::aggregation_kind_forward_pong_request:
                        //     {
                        //         return dg::network_memcommit_factory::devirtualize_forward_pong_signal_aggregation_event(sigagg_event).sigagg_addr;
                        //     }
                        //     case dg::network_memcommit_factory::aggregation_kind_forward_pingpong_request:
                        //     {
                        //         return dg::network_memcommit_factory::devirtualize_forward_pingpong_request_aggregation_event(sigagg_event).sigagg_addr;
                        //     }
                        //     case dg::network_memcommit_factory::aggregation_kind_forward_do_signal:
                        //     {
                        //         return dg::network_memcommit_factory::devirtualize_forward_do_signal_aggregation_event(sigagg_event).sigagg_addr;
                        //     }
                        //     case dg::network_memcommit_factory::aggregation_kind_backward_do_signal:
                        //     {
                        //         return dg::network_memcommit_factory::devirtualize_backward_do_signal_aggregation_event(sigagg_event).sigagg_addr;
                        //     }
                        //     default:
                        //     {
                        //         std::unreachable();
                        //     }
                        // }
                    }
                    default:
                    {
                        std::unreachable();
                    }
                }
            }

            struct InternalPressFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, virtual_memory_event_t>{

                dg::network_mempress::MemoryPressInterface * dst;

                void push(const uma_ptr_t& region, std::move_iterator<virtual_memory_event_t *> event_arr, size_t sz) noexcept{

                    if (region < this->dst->first() || region >= this->dst->last()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::OUT_OF_BOUND_ACCESS));
                        return;
                    }

                    if constexpr(DEBUG_MODE_FLAG){
                        if (sz > this->dst->max_consume_size()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    this->dst->push(region, event_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

            struct InternalWareHouseFeedResolutor: dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>{

                WareHouseIngestionConnectorInterface * dst;

                void push(std::move_iterator<virtual_emmory_event_t *> event_arr, size_t sz) noexcept{

                    if constexpr(DEBUG_MODE_FLAG){
                        if (sz > this->dst->max_consume_size()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    std::expected<dg::vector<virtual_memory_event_t>, exception_t> vec = dg::network_exception::cstyle_initialize<dg::vector<virtual_memory_event_t>>(sz);

                    if (!vec.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        return;
                    }

                    std::copy(event_arr, std::next(event_arr, sz), vec->begin());
                    std::expected<bool, exception_t> rs = this->dst->push(std::move(vec.value())); //alright Martian, we wont lose the crops this time (or rocket with no cap), it seems to me that the warehouse retryable responsibility is THIS guy responsibility, we'll attempt to do a warehouse connector again

                    if (!rs.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(rs.error()));
                        return;
                    }

                    if (!rs.value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::QUEUE_FULL));
                        return;
                    }
                }
            };
    };
} 

#endif