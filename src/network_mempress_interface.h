#ifndef __DG_NETWORK_MEMPRESS_INTERFACE_H__
#define __DG_NETWORK_MEMPRESS_INTERFACE_H__

#include "network_pointer.h"
#include <stdint.h>
#include <stdlib.h>
#include "network_exception.h"
#include <iterator>

namespace dg::network_mempress{

    using uma_ptr_t = dg::network_pointer::uma_ptr_t;
    using event_t   = uint64_t;

    struct MemoryPressInterface{
        virtual ~MemoryPressInterface() noexcept = default;
        virtual auto first() const noexcept -> uma_ptr_t = 0;
        virtual auto last() const noexcept -> uma_ptr_t = 0;
        virtual auto memregion_size() const noexcept -> size_t = 0;
        virtual auto is_busy(uma_ptr_t ptr) noexcept -> bool = 0;
        virtual void push(uma_ptr_t ptr, std::move_iterator<event_t *> event_arr, size_t event_arr_sz, exception_t * exception_arr) noexcept = 0;
        virtual auto try_collect(uma_ptr_t ptr, event_t * event_arr, size_t& event_arr_sz, size_t event_arr_cap) noexcept -> bool = 0;
        virtual void collect(uma_ptr_t ptr, event_t * event_arr, size_t& event_arr_sz, size_t event_arr_cap) noexcept = 0;
        virtual auto is_collectable(uma_ptr_t ptr) noexcept -> bool = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
        virtual auto minimum_collect_cap() noexcept -> size_t = 0;
    };
}

#endif