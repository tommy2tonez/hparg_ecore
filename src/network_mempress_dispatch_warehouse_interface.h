#ifndef __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_INTERFACE_H__
#define __NETWORK_MEMPRESS_DISPATCH_WAREHOUSE_INTERFACE_H__

#include "network_std_container.h"
#include "network_exception.h"
#include "network_memevent_model.h"

namespace dg::network_mempress_dispatch_warehouse{
    
    using event_t = dg::network_memevent_factory::virtual_memory_event_t

    struct WareHouseInterface{

        virtual ~WareHouseInterface() = default;
        virtual auto push(dg::vector<event_t>&&) noexcept -> std::expected<bool, exception_t> = 0;
        virtual auto pop() noexcept -> dg::vector<event_t> = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };
}

#endif