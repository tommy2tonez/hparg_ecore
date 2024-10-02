#ifndef __NETWORK_EXTERNAL_MEMCOMMIT_MODEL_H__
#define __NETWORK_EXTERNAL_MEMCOMMIT_MODEL_H__

#include <variant>
#include <stdlib.h>
#include <stdint.h>

namespace dg::network_extmemcommit_model{

    using event_kind_t = uint8_t;

    enum event_option: event_kind_t{
        signal_event_kind               = 0u,
        inject_event_kind               = 1u,
        conditional_inject_event_kind   = 2u,
        init_event_kind                 = 3u
    };

    struct SignalEvent{
        dg::network_tile_signal_poly::virtual_payload_t payload;
    };

    struct InjectEvent{
        dg::network_tile_injection_poly::virtual_payload_t payload;
    };

    struct ConditionalInjectEvent{
        dg::network_tile_condinjection_poly::virtual_payload_t payload;
    };

    struct InitEvent{
        dg::network_tile_initialization_poly::virtual_payload_t payload;
    };

    using poly_event_t = std::variant<SignalEvent, ConditionalInjectEvent, InjectEvent, InitEvent>;

    auto is_signal_event(const poly_event_t& event){

        return std::holds_alternative<SignalEvent>(event);
    }

    auto is_inject_event(const poly_event_t& event){

        return std::holds_alternative<InjectEvent>(event);
    }

    auto is_conditional_inject_event(const poly_event_t& event){

        return std::holds_alternativec<ConditionalInjectEvent>(event);
    }

    auto is_init_event(const poly_event_t& event){

        return std::holds_alternative<InitEvent>(event);
    }
}

#endif