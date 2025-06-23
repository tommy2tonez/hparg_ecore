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

    //I dont have the hinge for this component, I feel like this is bad, not well implemented
    //what precisely are we solving

    //we have a batch of workorders
    //we need to complete the workorders as fast as possible in the sense of allocating enough correct waiters to dispatch the workorders 

    //in the totally random case, we are on finite waiters, we need to partition that in a chunk of uniform processing unit

    //in the memregion cases, we need to allocate a right amount of waiters to wait
    //best scenerio, each waiter waits exactly one region
    //problems, we dont have that number of waiters, which would also increase the lock contention dramatically which is bad  

    //what's the right amount, region1 region2 on worker 1, region3 region4 on worker 2, etc.
    //it seems like the order of the kv matters

    //assume that we are sorting the memregions

    //region1 (repeat 100 times), region2 (repeat 10 times), region3 (repeat 1 time), region 4 (repeat 1 time)
    //we'd want to take a uniform slice region1 -> region4 or region4 -> region1

    //our first implementation would sounds like that, the normal unit size scenerio, a border extend of the unit (if the border lands on the body of a contiguous region, we'd be moving that to the last), a keyvalue sort based on value size

    //alright, now we are doing the hot-cold problem, what's the worst latency scenerio of discretization_sz == 16, it's 1 1 1 ... 1 (16 times), because a worker would have to wait on a memregion to complete 16 times
    //how about we solve the problem by using engineered dispatches, we'd make sure that our memregion hits would be 10 + 6 == 16, so a worker would have to wait on a memregion to complete 2 times

    class Assorter: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WareHouseExtractionConnectorInterface> unassorted_warehouse;
            std::shared_ptr<WareHouseIngestionConnectorInterface> assorted_warehouse;
            size_t region_vectorization_sz;
            size_t warehouse_expected_ingestion_sz; 
            size_t warehouse_max_ingestion_sz;

        public:

            Assorter(std::shared_ptr<WareHouseExtractionConnectorInterface> unassorted_warehouse,
                     std::shared_ptr<WareHouseIngestionConnectorInterface> assorted_warehouse,
                     size_t region_vectorization_sz,
                     size_t warehouse_expected_ingestion_sz,
                     size_t warehouse_max_ingestion_sz) noexcept: unassorted_warehouse(std::move(unassorted_warehouse)),
                                                                  assorted_warehouse(std::move(assorted_warehouse)),
                                                                  region_vectorization_sz(region_vectorization_sz),
                                                                  warehouse_expected_ingestion_sz(warehouse_expected_ingestion_sz),
                                                                  warehouse_max_ingestion_sz(warehouse_max_ingestion_sz){}

            bool run_one_epoch() noexcept{

                dg::vector<event_t> event_vec               = this->unassorted_warehouse->pop();

                if (event_vec.empty()){
                    return true;
                }

                auto generic_internal_resolutor             = GenericInternalResolutor{}; 
                generic_internal_resolutor.warehouse        = this->assorted_warehouse.get();
                generic_internal_resolutor.expected_unit_sz = this->warehouse_expected_ingestion_sz;
                generic_internal_resolutor.max_unit_sz      = this->warehouse_max_ingestion_sz;

                size_t trimmed_generic_vectorization_sz     = std::min(this->region_vectorization_sz, static_cast<size_t>(event_vec.size()));
                size_t generic_feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&generic_internal_resolutor, trimmed_generic_vectorization_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> generic_feeder_mem(generic_feeder_allocation_cost);
                auto generic_feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&generic_internal_resolutor, trimmed_generic_vectorization_sz, generic_feeder_mem.get()));

                auto region_internal_resolutor              = RegionInternalResolutor{};
                region_internal_resolutor.generic_feeder    = generic_feeder.get();

                size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, static_cast<size_t>(event_vec.size()));
                size_t region_feeder_allocation_cost        = dg::network_producer_consumer::delvrsrv_orderedkv_allocation_cost(&region_internal_resolutor, trimmed_region_vectorization_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> region_feeder_mem(region_feeder_allocation_cost);
                auto region_feeder                          = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_orderedkv_open_preallocated_raiihandle(&region_internal_resolutor, trimmed_region_vectorization_sz, region_feeder_mem.get()));

                for (const event_t& event: event_vec){
                    uma_ptr_t event_ptr = this->internal_get_event_ptr(event);
                    uma_ptr_t region    = dg::memult::region(event_ptr, dg::network_memops_uma::memlock_region_size()); //

                    dg::network_producer_consumer::delvrsrv_orderedkv_deliver(region_feeder.get(), region, event);
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
                size_t expected_unit_sz;
                size_t max_unit_sz;

                void push(std::move_iterator<event_t *> event_arr, size_t sz) noexcept{

                    size_t discretization_sz    = this->expected_unit_sz;
                    event_t * first             = event_arr.base();
                    event_t * last              = std::next(first, sz); 

                    while (first != last){
                        size_t rem_sz               = std::distance(first, last);
                        size_t discretization_sz    = this->expected_unit_sz;
                        size_t tentative_sz         = std::min(discretization_sz, rem_sz);
                        event_t * tentative_last    = std::next(first, tentative_sz);

                        while (tentative_last != last){
                            if (tentative_last->region != std::prev(tentative_last)->region){
                                break;
                            }

                            if (std::distance(first, tentative_last) == this->max_unit_sz){
                                break;
                            }

                            std::advance(tentative_last, 1u);
                        }

                        size_t ingestion_sz                 = std::distance(first, tentative_last);
                        dg::vector<event_t ingesting_vec    = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::vector<event_t>>(ingestion_sz));

                        for (size_t i = 0u; i < ingestion_sz; ++i){
                            ingesting_vec[i] = std::move(first[i]);
                        }

                        exception_t err = this->warehouse->push(std::move(ingesting_vec.value()));

                        if (dg::network_exception::is_failed(err)){
                            dg::network_log_stackdump::error(dg::network_exception::verbose(err));
                        }

                        first = tentative_last;
                    }
                }
            };

            struct RegionInternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, event_t>{

                dg::network_producer_consumer::DeliveryHandle<event_t> * generic_feeder;

                void push(const uma_ptr_t& key, std::move_iterator<event_t *> event_arr, size_t sz) noexcept{

                    event_t * base_event_arr = event_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_producer_consumer::delvrsrv_deliver(this->generic_feeder, std::move(base_event_arr[i]));
                    }
                }
            };
    };

} 

#endif
