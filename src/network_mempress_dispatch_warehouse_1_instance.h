#ifndef __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_1_INSTANCE_H__
#define __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_1_INSTANCE_H__

#include "network_mempress_dispatch_warehouse_impl1.h"
#include "network_mempress_dispatch_warehouse_interface.h"
#include "stdx.h"

namespace dg::network_mempress_dispatch_warehouse_1_instance{

    struct DispatchWareHouse1Signature{};

    using warehouse_singleton = stdx::singleton<DispatchWareHouse1Signature, std::shared_ptr<dg::network_mempress_dispatch_warehouse::WareHouseInterface>>;
    
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