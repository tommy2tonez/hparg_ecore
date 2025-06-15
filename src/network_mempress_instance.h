#ifndef __DG_NETWORK_MEMPRESS_H__
#define __DG_NETWORK_MEMPRESS_H__

#include "network_mempress_interface.h"
#include "network_mempress_impl1.h"
#include "stdx.h"

namespace dg::network_mempress_instance{

    //we'll attempt to "singleton" the instance, for various reasons, first is the bridge problem
    //second is the reference problem
    //problem is ... we cannot deinit this once initialized, well ...
    //we "just" turn on the switches as components and glue them together via singletons

    struct mempress_instance_signature{}; 

    using mempress_singleton = stdx::singleton<mempress_instance_signature, std::shared_ptr<dg::network_mempress::MemoryPressInterface>>; 

    struct Config{
        uma_ptr_t first;
        uma_ptr_t last;
        size_t submit_cap;
        size_t region_cap;
        size_t memregion_sz;

        bool has_fast_container;
        size_t fast_container_region_vec_cap;
        size_t fast_container_trigger_threshold;
    };

    void init(Config config){

        if (config.has_fast_container){
            mempress_singleton::get() = dg::network_mempress_impl1::Factory::spawn_fastpress(config.first, config.last,
                                                                                             config.submit_cap, config.region_cap,
                                                                                             config.fast_container_region_vec_cap,
                                                                                             config.memregion_sz,
                                                                                             config.fast_container_trigger_threshold);
        } else{
            mempress_singleton::get() = dg::network_mempress_impl1::Factory::spawn_mempress(config.first, config.last,
                                                                                            config.submit_cap, config.region_cap,
                                                                                            config.memregion_sz);
        }
    }

    void deinit() noexcept{

        mempress_singleton::get() = nullptr;
    } 

    auto get_instance() noexcept -> const std::shared_ptr<dg::network_mempress::MemoryPressInterface>&{

        return mempress_singleton::get();
    } 
}