#ifndef __DG_NETWORK_MEMCOMMIT_MESSENGER_H__
#define __DG_NETWORK_MEMCOMMIT_MESSENGER_H__

#include "network_mempress.h" 
#include "network_mempress_dispatch_warehouse.h"
#include "network_producer_consumer.h"

namespace dg::network_memcommit_messenger{

    using virtual_memory_event_t = dg::network_memcommit_factory::virtual_memory_event_t; 

    class MemregionRadixerInterface{

        public:

            using memregion_kind_t = uint8_t;

            static inline constexpr uint8_t EXPRESS_REGION  = 0u;
            static inline constexpr uint8_t NOMRAL_REGION   = 1u;

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

    class MemregionRadixer: public virtual MemregionRadixerInterface{

        private:

            dg::unordered_unstable_map<uma_ptr_t, MemregionRadixerInterface::memregion_kind_t> region_kind_map;
            size_t pow2_memregion_sz;
        
        public:

            MemregionRadixer(dg::unordered_unstable_map<uma_ptr_t, MemregionRadixerInterface::memregion_kind_t> region_kind_map,
                             size_t pow2_memregion_sz) noexcept: region_kind_map(std::move(region_kind_map)),
                                                                 pow2_memregion_sz(pow2_memregion_sz){}

            auto radix(uma_ptr_t ptr) noexcept -> std::expected<MemregionRadixerInterface::memregion_kind_t, exception_t>{

                uma_ptr_t ptr_region    = dg::memult::region(ptr, this->pow2_memregion_sz);
                auto map_ptr            = std::as_const(this->region_kind_map).find(ptr_region);

                if (map_ptr == std::as_const(this->region_kind_map).end()){
                    return std::unexpected(dg::network_exception::OUT_OF_BOUND_ACCESS);
                }

                MemregionRadixerInterface::memregion_kind_t memregion_kind = map_ptr->second;

                if constexpr(DEBUG_MODE_FLAG){
                    if (memregion_kind != MemregionRadixerInterface::EXPRESS_REGION && memregion_kind != MemregionRadixerInterface::NOMRAL_REGION){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return memregion_kind;
            }
    };

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
            std::shared_ptr<WareHouseIngestionConnectorInterface> warehouse_connector;
            size_t warehouse_aggregation_sz;

        public:

            MemeventMessenger(std::shared_ptr<MemregionRadixerInterface> memregion_express_radixer,
                              std::shared_ptr<dg::network_mempress::MemoryPressInterface> press,
                              size_t press_vectorization_sz,
                              std::shared_ptr<WareHouseIngestionConnectorInterface> warehouse_connector,
                              size_t warehouse_aggregation_sz) noexcept: memregion_express_radixer(std::move(memregion_express_radixer)),
                                                                         press(std::move(press)),
                                                                         press_vectorization_sz(press_vectorization_sz),
                                                                         warehouse_connector(std::move(warehouse_connector)),
                                                                         warehouse_aggregation_sz(warehouse_aggregation_sz){}

            void push(std::move_iterator<virtual_memory_event_t *> data_arr, size_t sz) noexcept{

                virtual_memory_event_t * base_data_arr  = data_arr.base();

                auto press_feed_resolutor               = InternalPressFeedResolutor{};
                press_feed_resolutor.dst                = this->press.get();

                size_t trimmed_press_vectorization_sz   = std::min(std::min(this->press_vectorization_sz, this->press->max_consume_size()), sz);
                size_t press_feeder_allocation_cost     = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&press_feed_resolutor, trimmed_press_vectorization_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> press_feeder_mem(press_feeder_allocation_cost);
                auto press_feeder                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&press_feed_resolutor,
                                                                                                                                                                             trimmed_press_vectorization_sz,
                                                                                                                                                                             press_feeder_mem.get()));

                //------------------------

                auto warehouse_feed_resolutor           = InternalWareHouseFeedResolutor{};
                warehoues_feed_resolutor.dst            = this->warehouse_connector.get();

                size_t trimmed_warehouse_feed_cap       = std::min(std::min(this->warehouse_aggregation_sz, this->warehouse_connector->max_consume_size()), sz);
                size_t warehouse_feeder_allocation_cost = dg::network_producer_consumer::delvrsrv_allocation_cost(&warehouse_feed_resolutor, trimmed_warehouse_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> warehouse_feeder_mem(warehouse_feeder_allocation_cost);
                auto warehouse_feeder                   = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&warehouse_feed_resolutor, 
                                                                                                                                                                          trimmed_warehouse_feed_cap,
                                                                                                                                                                          warehouse_feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    uma_ptr_t notifying_addr = this->extract_notifying_addr(base_data_arr[i]);
                    std::expected<MemregionRadixerInterface::memregion_kind_t, exception_t> memregion_kind = this->memregion_express_radixer->radix(notifying_addr);

                    if (!memregion_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(memregion_kind.error()));
                        continue;
                    }

                    switch (memregion_kind.value()){
                        case MemregionRadixerInterface::EXPRESS_REGION:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(warehouse_feeder.get(), std::move(base_data_arr[i]));
                            break;
                        }
                        case MemregionRadixerInterface::NOMRAL_REGION:
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
                        return dg::network_memcommit_factory::devirtualize_sigagg_signal_event(memevent).smph_addr;
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

                    if (sz == 0u){
                        return;
                    }

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

    struct Factory{

        static auto spawn_memregion_radixer(const std::vector<uma_ptr_t>& express_region_vec,
                                            const std::vector<uma_ptr_t>& normal_region_vec,
                                            size_t memregion_sz) -> std::unique_ptr<MemregionRadixerInterface>{
                
            if (!stdx::is_pow2(memregion_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            using uptr_t = dg::ptr_info<uma_ptr_t>::max_unsigned_t; 

            dg::unordered_unstable_map<uma_ptr_t, MemregionRadixerInterface::memregion_kind_t> region_kind_map{};

            for (uma_ptr_t normal_region: normal_region_vec){
                uptr_t uptr = dg::pointer_cast<uptr_t>(normal_region); 

                if (uptr % memregion_sz != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                region_kind_map[normal_region] = MemregionRadixerInterface::NOMRAL_REGION;
            }

            for (uma_ptr_t express_region: express_region_vec){
                uptr_t uptr = dg::pointer_cast<uptr_t>(express_region);

                if (uptr % memregion_sz != 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                region_kind_map[express_region] = MemregionRadixerInterface::EXPRESS_REGION;
            }

            return std::make_unique<MemregionRadixer>(std::move(region_kind_map), memregion_sz);
        }

        static auto spawn_warehouse_connector(std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse) -> std::unique_ptr<WareHouseIngestionConnectorInterface>{

            if (warehouse == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<NormalWareHouseConnector>(std::move(warehouse));
        }

        static auto spawn_exhaustion_controlled_warehouse_connector(std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface> warehouse,
                                                                    std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device) -> std::unique_ptr<WareHouseIngestionConnectorInterface>{
                        
            if (warehouse == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (infretry_device == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledWareHouseConnector>(std::move(warehouse), std::move(infretry_device));
        }

        static auto spawn_memevent_messenger(std::shared_ptr<MemregionRadixerInterface> memregion_radixer,
                                             std::shared_ptr<dg::network_mempress::MemoryPressInterface> press,
                                             size_t press_vectorization_sz,
                                             std::shared_ptr<WareHouseIngestionConnectorInterface> warehouse_connector,
                                             size_t warehouse_aggregation_sz) -> std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_emmory_event_st>>{
            
            if (memregion_radixer == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (press == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (warehouse_connector == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            const size_t MIN_PRESS_VECTORIZATION_SZ     = 1u;
            const size_t MAX_PRESS_VECTORIZATION_SZ     = press->max_consume_size();

            const size_t MIN_WAREHOUSE_AGGREGATION_SZ   = 1u;
            const size_t MAX_WAREHOUSE_AGGREGATION_SZ   = warehouse_connector->max_consume_size();
            
            if (std::clamp(press_vectorization_sz, MIN_PRESS_VECTORIZATION_SZ, MAX_PRESS_VECTORIZATION_SZ) != press_vectorization_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(warehouse_aggregation_sz, MIN_WAREHOUSE_AGGREGATION_SZ, MAX_WAREHOUSE_AGGREGATION_SZ)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<MemeventMessenger>(std::move(memregion_radixer),
                                                       std::move(press),
                                                       press_vectorization_sz,
                                                       std::move(warehouse_connector),
                                                       warehouse_aggregation_sz);
        }
    };
} 

#endif