#ifndef __NETWORK_MEMCOMMIT_DROPBOX_H__
#define __NETWORK_MEMCOMMIT_DROPBOX_H__

#include <stdint.h>
#include <stdlib.h>
#include "network_producer_consumer.h"
#include "network_memcommit_model.h"

namespace dg::network_intmemcommit_dropbox{

    //1hr
    
    template <class ID>
    struct DropBox{

        using virtual_memory_event_t    = network_memcommit_factory::virtual_memory_event_t;  
        using memory_event_t            = network_memcommit_factory::memory_event_t;

        static inline void submit(virtual_memory_event_t *, size_t) noexcept{

        }

        static inline auto capacity() noexcept -> size_t{

        } 

        static inline void recv(virtual_memory_event_t * virtual_event, memory_event_t event, size_t& sz, size_t cap) noexcept{

        }
    };

}

namespace dg::network_intmemcommit_dropbox_wrapper{

    template <class DropBox>
    struct LimitConsumerWrapper{};

    template <class ID>
    struct LimitConsumerWrapper<network_intmemcommit_dropbox::DropBox<ID>>: dg::network_producer_consumer::LimitConsumerInterface<LimitConsumerWrapper<network_intmemcommit_dropbox::DropBox<ID>>>{

        using dropbox   = network_intmemcommit_dropbox::DropBox<ID>;
        using event_t   = typename dropbox::virtual_memory_event_t;

        static inline void push(event_t * events, size_t sz) noexcept{

            dropbox::submit(events, sz);
        }

        static inline auto capacity() noexcept -> size_t{

            return dropbox::capacity();
        }
    };

    using ConsumerWrapper = typename dg::network_producer_consumer::ConsumerWrapper<std::void_t<>, typename LimitConsumerWrapper::interface_t>::interface_t;

    
}

#endif