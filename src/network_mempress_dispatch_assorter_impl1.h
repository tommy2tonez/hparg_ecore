#ifndef __NETWORK_MEMPRESS_DISPATCH_ASSORTER_IMPL1_H__
#define __NETWORK_MEMPRESS_DISPATCH_ASSORTER_IMPL1_H__

#include "network_mempress_dispatch_warehouse_interface.h" 

namespace dg::network_mempress_dispatch_assorter_impl1{

    //what optimizables can we do?
    //same_regionness => OK thru to 1 resolutor

    //different regionness => cap the max number of partition based on the "concurrency factor"
    //what is the generic, all round first approach?
    //this is a very important optimizable, because we'd want to rely on our smph tile to do all the aggregating magics, yet we can't transfer the responsibility of "efficient dispatch of a bunch" -> the smph tile because that'd be implementation-specific, whose knowledge is unknown to the users
    //our only and sole clue in the world is probably statistics, we have to do our best to push all the data thru possible, give the user a window of probability of an arbitrary packet travelling from A -> B  

    //I was thinking of the levelorder traversal of trees
    
    //this tree does not point to the immediate next level but also the next level etc., this is not an injective property but rather a multi_link tree
    //how would I, precisely, partition the smph tile to forward + backward in the most efficient fashion

    //we don't answer that question yet
    //the question now is given the info of a batch dispatch, what's the most rational decision that we could make, without compromising the overall speed?

    //let's make a decision tree

    //it's possible that every region deserves their own "dispatcher" or "resolutor"

    //problem, we are wasting a lot of memory orderings

    //immediate patch, increase tile size -> 64KB, problem solved 
    //other patch, we only marked a region as "deserved-to-be" when the region_events >= certain threshold
    //alright, how about the others not "deserved-to-be"
    //the generic dispatches need to have a smooth, reasonable batch size to be well distributed among the resolutors (we haven't solve this just yet)

    //remember, we are concerned about the latency of completion, because we are "waiting" for a batch of memregion events to be completed, we aren't concerned about the total number on the region, that's the asynchronous device responsibility

    class WareHouseExtractionConnectorInterface{

        public:

            virtual ~WareHouseExtractionConnectorInterface() noexcept = default;
            virtual auto pop() noexcept -> dg::vector<event_t> = 0;
    };

    class WareHouseIngestionConnectorInterface{

        public:

            virtual ~WareHouseIngestionConnectorInterface() noexcept = default;
            virtual auto push(dg::vector<event_t>&&) noexcept -> exception_t = 0;
    };

    class NormalWareHouseExtractionConnector: public virtual WareHouseExtractionConnectorInterface{

        private:

            std::shared_ptr<dg::network_mempress_dispatch_warehouse_interface::WareHouseInterface> base;
        
        public:

            NormalWareHouseExtractionConnector(std::shared_ptr<dg::network_mempress_dispatch_warehouse_interface::WareHouseInterface> base) noexcept: base(std::move(base)){}

            auto pop() noexcept -> dg::vector<event_t>{

                return this->base->pop();
            }
    };

    class NormalWareHouseIngestionConnector: public virtual WareHouseIngestionConnectorInterface{

        private:

            std::shared_ptr<dg::network_mempress_dispatch_warehouse_interface::WareHouseInterface> base;
        
        public: 

            NormalWareHouseIngestionConnector(std::shared_ptr<dg::network_mempress_dispatch_warehouse_interface::WareHouseInterface> base) noexcept: base(std::move(base)){}

            auto push(dg::vector<event_t>&& event_vec) noexcept -> exception_t{

                return this->base->push(std::move(event_vec));
            }
    };

    class ExhaustionControlledWareHouseIngestionConnector: public virtual WareHouseIngestionConnectorInterface{

        private:

            std::shared_ptr<dg::network_mempress_dispatch_warehouse_interface::WareHouseInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
        
        public:

            ExhaustionControlledWareHouseIngestionConnector(std::shared_ptr<dg::network_mempress_dispatch_warehouse_interface::WareHouseInterface> base,
                                                            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) noexcept: base(std::move(base)),
                                                                                                                                                       executor(std::move(executor)){}
            
            auto push(dg::vector<event_t>&& event_vec) noexcept -> exception_t{

                exception_t err;

                auto task = [&]() noexcept{
                    err = this->base->push(static_cast<dg::vector<event_t>&&>(event_vec));

                    if (dg::network_exception::is_success(err)){
                        return true;
                    }

                    if (err != dg::network_exception::QUEUE_FULL){
                        return true;
                    }

                    return false;
                };

                dg::network_concurrency_infretry_x::ExecutableWrapper virtual_task(task);
                this->executor->exec(virtual_task);

                return err;
            }
    };

    class Assorter: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WareHouseExtractionConnectorInterface> unassorted_warehouse;
            std::shared_ptr<WareHouseIngestionConnectorInterface> assorted_warehouse;
            size_t affined_region_process_threshold;
            size_t region_vectorization_sz;
            size_t generic_vectorization_sz;

        public:

            Assorter(std::shared_ptr<WareHouseExtractionConnectorInterface> unassorted_warehouse,
                     std::shared_ptr<WareHouseIngestionConnectorInterface> assorted_warehouse,
                     size_t affined_region_process_threshold,
                     size_t region_vectorization_sz,
                     size_t generic_vectorization_sz) noexcept: unassorted_warehouse(std::move(unassorted_warehouse)),
                                                                assorted_warehouse(std::move(assorted_warehouse)),
                                                                affined_region_process_threshold(affined_region_process_threshold),
                                                                region_vectorization_sz(region_vectorization_sz),
                                                                generic_vectorization_sz(generic_vectorization_sz){}

            bool run_one_epoch() noexcept{

                dg::vector<event_t> event_vec               = this->unassorted_warehouse->pop();

                if (event_vec.empty()){
                    return true;
                }

                auto generic_internal_resolutor             = GenericInternalResolutor{}; 
                generic_internal_resolutor.warehouse        = this->assorted_warehouse.get();

                size_t trimmed_generic_vectorization_sz     = std::min(this->generic_vectorization_sz, static_cast<size_t>(event_vec.size()));
                size_t generic_feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&generic_internal_resolutor, trimmed_generic_vectorization_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> generic_feeder_mem(generic_feeder_allocation_cost);
                auto generic_feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&generic_internal_resolutor, trimmed_generic_vectorization_sz, generic_feeder_mem.get()));

                auto region_internal_resolutor              = RegionInternalResolutor{};
                region_internal_resolutor.warehouse         = this->assorted_warehouse.get();
                region_internal_resolutor.generic_feeder    = generic_feeder.get();
                region_internal_resolutor.threshold         = this->affined_region_process_threshold;

                size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, static_cast<size_t>(event_vec.size()));
                size_t region_feeder_allocation_cost        = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&region_internal_resolutor, trimmed_region_vectorization_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> region_feeder_mem(region_feeder_allocation_cost);
                auto region_feeder                          = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&region_internal_resolutor, trimmed_region_vectorization_sz, region_feeder_mem.get()));

                for (const event_t& event: event_vec){
                    uma_ptr_t event_ptr = this->internal_get_event_ptr(event);
                    uma_ptr_t region    = dg::memult::region(event_ptr, dg::network_memops_uma::memlock_region_size()); //

                    dg::network_producer_consumer::delvrsrv_kv_deliver(region_feeder.get(), region, event);
                }

                return true;
            }
        
        private:

            static inline auto internal_get_event_ptr(const virtual_memory_event_t& event) noexcept -> uma_ptr_t{

                memory_event_kind_t event_kind = dg::network_memcommit_factory::read_virtual_event_kind(event);

                switch (event_kind){
                    case dg::network_memcommit_factory::event_kind_forward_ping_signal:
                    {
                        return dg::network_memcommit_factory::devirtualize_forward_ping_signal(event).dst;
                    }
                    case dg::network_memcommit_factory::event_kind_forward_pong_request:
                    {
                        return dg::network_memcommit_factory::devirtualize_forward_pong_request(event).requestee;
                    }
                    case dg::network_memcommit_factory::event_kind_forward_pingpong_request:
                    {
                        return dg::network_memcommit_factory::devirtualize_forward_pingpong_request(event).requestee;
                    }
                    case dg::network_memcommit_factory::event_kind_forward_do_signal:
                    {
                        return dg::network_memcommit_factory::devirtualize_forward_do_signal(event).dst;
                    }
                    case dg::network_memcommit_factory::event_kind_backward_do_signal:
                    {
                        return dg::network_memcommit_factory::devirtualize_backward_do_signal(event).dst;
                    }
                    case dg::network_memcommit_factory::event_kind_signal_aggregation_signal:
                    {
                        return dg::network_memcommit_factory::devirtualize_signal_aggregation_signal(event).smph_addr;
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

            struct GenericInternalResolutor: dg::network_producer_consumer::ConsumerInterface<event_t>{

                WareHouseIngestionConnectorInterface * warehouse;

                void push(std::move_iterator<event_t *> event_arr, size_t sz) noexcept{

                    if (sz == 0u){
                        return;
                    }

                    std::expected<dg::vector<event_t>, exception_t> event_vec = dg::network_exception::cstyle_initialize<dg::vector<event_t>>(sz);

                    if (!event_vec.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(event_vec.error()));
                        return;
                    }

                    std::copy(event_arr, std::next(event_arr, sz), event_vec->begin());
                    exception_t err = this->warehouse->push(std::move(event_vec.value()));

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(err));
                        return;
                    }
                }
            };

            struct RegionInternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, event_t>{

                WareHouseIngestionConnectorInterface * warehouse;
                dg::network_producer_consumer::DeliveryHandle<event_t> * generic_feeder;
                size_t threshold;

                void push(const uma_ptr_t& key, std::move_iterator<event_t *> event_arr, size_t sz) noexcept{

                    event_t * base_event_arr = event_arr.base();

                    if (sz < this->threshold){
                        for (size_t i = 0u; i < sz; ++i){
                            dg::network_producer_consumer::delvrsrv_deliver(this->generic_feeder, std::move(base_event_arr[i]));
                        }

                        return;
                    }

                    std::expected<dg::vector<event_t>, exception_t> event_vec = dg::network_exception::cstyle_initialize<dg::vector<event_t>>(sz);

                    if (!event_vec.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(event_vec.error()));
                        return;
                    }

                    std::copy(event_arr, std::next(event_arr, sz), event_vec->begin());
                    exception_t err = this->warehouse->push(std::move(event_vec.value()));

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(err));
                        return;
                    }
                }
            };
    };

} 

#endif
