#ifndef __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_0_INSTANCE_H__
#define __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_0_INSTANCE_H__

#include "network_mempress_dispatch_warehouse_impl1.h"
#include "network_mempress_dispatch_warehouse_interface.h"
#include "stdx.h"

namespace dg::network_mempress_dispatch_warehouse_0_instance{

    //we are missing an intermediate step, we have yet to know to detach this intermediate warehouse (the assorting warehouse to reduce latency) or including the responsibility -> the messenger
    //because the messenger does not have such a knowledge about the warehouse, we have to take care of the business here 

    //we are assorting the events based on memregions
    //this is very important because we have to load balance the latency also
    //even though we detached the responsibility of "doing actual work" from the memevent dispatchers -> the asynchronous device, we can't guarantee the on-timeness or the latency of a batch of memevents from smph tile which could cause some cores running hot and some cores running cold while waiting for the memregion workorders to be completed

    //we dont like the usage of singletons, yet it's extremely very hard to bridge + extend without singleton

    struct DispatchWareHouse0Signature{};

    using warehouse_singleton = stdx::singleton<DispatchWareHouse0Signature, std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface>>; 

    struct Config{
        size_t production_queue_cap;
        size_t max_concurrency_sz;
        size_t unit_consumption_sz;
        bool has_distributed_warehouse;
        size_t distributed_warehouse_concurrency_sz;
        size_t distributed_warehouse_empty_curious_pop_sz;
    };

    void init(Config config){

        if (!config.has_distributed_warehouse){
            warehouse_singleton::get() = dg::network_mempress_dispatch_warehouse_impl1::Factory::spawn_warehouse(config.production_queue_cap,
                                                                                                                 config.max_concurrency_sz,
                                                                                                                 config.unit_consumption_sz);
        } else{
            warehouse_singleton::get() = dg::network_mempress_dispatch_warehouse_impl1::Factory::spawn_distributed_warehouse(config.production_queue_cap,
                                                                                                                             config.max_concurrency_sz,
                                                                                                                             config.unit_consumption_sz,
                                                                                                                             config.distributed_warehouse_concurrency_sz,
                                                                                                                             config.distributed_warehouse_empty_curious_pop_sz);
        }
    }

    void deinit() noexcept{

        warehouse_singleton::get() = nullptr;
    }

    auto get_instance() noexcept -> const std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface>{

        return warehouse_singleton::get();
    }
}

#endif