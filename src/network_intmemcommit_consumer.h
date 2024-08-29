#ifndef __EVENT_DISPATCHER_H__
#define __EVENT_DISPATCHER_H__

#include <stdint.h>
#include <stddef.h>
#include <network_addr_lookup.h>
#include "network_function_concurrent_buffer.h"
#include "network_tile_member_access.h"   
#include "network_memcommit_factory.h" 

namespace dg::network_memcommit_consumer{
    
    template <class ...Args>
    struct tags{};

    using virtual_memory_event_t = uint64_t;

    template <class ResolveCapacity, class DeliveryHandlerSpawner>
    struct ForwardPingSignalConsumer{}; 

    template <size_t RESOLVE_CAPACITY, class T>
    struct ForwardPingSignalConsumer<std::integral_constant<size_t, RESOLVE_CAPACITY>, dg::network_producer_consumer::DeliveryHandlerSpawnerInterface<T>>{

        using delivery_spawner          = dg::network_producer_consumer::DeliveryHandlerSpawnerInterface<T>;
        using virtual_memory_event_t    = dg::mono_type_reduction_t<dg::network_memcommit_factory::virtual_memory_event_t, typename delivery_spawner::event_t>;
        
        static inline void resolve(virtual_memory_event_t * event, size_t sz) noexcept{

        }

        static inline auto capacity() noexcept -> size_t{

            return RESOLVE_CAPACITY;
        } 
    };

}

#endif