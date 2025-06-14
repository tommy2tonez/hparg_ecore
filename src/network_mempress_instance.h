#ifndef __DG_NETWORK_MEMPRESS_H__
#define __DG_NETWORK_MEMPRESS_H__

namespace dg::network_mempress_instance{

    //we'll attempt to "singleton" the instance, for various reasons, first is the bridge problem
    //second is the reference problem
    //problem is ... we cannot deinit this once initialized, well ...
    //we "just" turn on the switches as components and glue them together via singletons

    void init(){

    }

    void deinit(){

    } 

    auto get_instance() -> std::shared_ptr<dg::network_mempress::MemoryPressInterface>{

    } 
}