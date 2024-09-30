#ifndef __NETWORK_EXTERNAL_MEMCOMMIT_MODEL_H__
#define __NETWORK_EXTERNAL_MEMCOMMIT_MODEL_H__

#include <variant>

namespace dg::network_external_memcommit_model{

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
}

#endif