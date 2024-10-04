#ifndef __NETWORK_EXTERNAL_MEMCOMMIT_MODEL_H__
#define __NETWORK_EXTERNAL_MEMCOMMIT_MODEL_H__

#include <variant>
#include <stdlib.h>
#include <stdint.h>

namespace dg::network_extmemcommit_model{

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

    auto get_signal_event(poly_event_t event) noexcept -> dg::network_tile_signal_poly::virtual_payload_t{

        auto rs = std::move(std::get<SignalEvent>(event).payload);
        return rs;
    }

    auto get_inject_event(poly_event_t event) noexcept -> dg::network_tile_injection_poly::virtual_payload_t{

        auto rs = std::move(std::get<InjectEvent>(event).payload);
        return rs;
    }

    auto get_condinject_event(poly_event_t event) noexcept -> dg::network_tile_condinjection_poly::virtual_payload_t{

        auto rs = std::move(std::get<ConditionalInjectEvent>(event).payload);
        return rs;
    }

    auto get_init_event(poly_event_t event) noexcept -> dg::network_tile_initialization_poly::virtual_payload_t{

        auto rs = std::move(std::get<InitEvent>(event).payload);
        return rs;
    }
}

#endif