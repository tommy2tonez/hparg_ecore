#ifndef __DG_NETWORK_MEMCOMMIT_MODEL_H__
#define __DG_NETWORK_MEMCOMMIT_MODEL_H__

#include <stdint.h>
#include <stdlib.h>
#include <tuple>
#include "network_trivial_serializer.h" 
#include <array>
#include "network_pointer.h"

namespace dg::network_memcommit_factory{

    using memory_event_kind_t       = uint8_t;
    using uma_ptr_t                 = dg::network_pointer::uma_ptr_t;

    enum enum_memory_event: memory_event_kind_t{
        event_kind_forward_ping_signal      = 0u,
        event_kind_forward_pong_request     = 1u,
        event_kind_forward_pingpong_request = 2u,
        event_kind_forward_pong_signal      = 3u,
        event_kind_forward_do_signal        = 4u,
        event_kind_backward_do_signal       = 5u
    };

    struct ForwardPingSignalEvent{
        uma_ptr_t dst;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(dst);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(dst);
        }
    };

    struct ForwardPongSignalEvent{
        uma_ptr_t dst;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(dst);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(dst);
        }
    };

    struct ForwardPongRequestEvent{
        uma_ptr_t requestee;
        uma_ptr_t requestor;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(requestee, requestor);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(requestee, requestor);
        }
    };

    struct ForwardPingPongRequestEvent{
        uma_ptr_t requestee;
        uma_ptr_t requestor;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(requestee, requestor);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(requestee, requestor);
        }
    };

    struct ForwardDoSignalEvent{
        uma_ptr_t dst;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(dst);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(dst);
        }
    };

    struct BackwardDoSignalEvent{
        uma_ptr_t dst;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(dst);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(dst);
        }
    };

    static inline constexpr size_t VIRTUAL_EVENT_BUFFER_SZ = size_t{1} << 5; 

    struct VirtualEvent{
        memory_event_kind_t event_kind;
        std::array<char, VIRTUAL_EVENT_BUFFER_SZ> content;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(event_kind, content);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(event_kind, content);
        }
    };

    using virtual_memory_event_t    = VirtualEvent;

    constexpr auto make_event_forward_ping_signal(uma_ptr_t dst) noexcept -> ForwardPingSignalEvent{

        return ForwardPingSignalEvent{dst};
    }

    constexpr auto make_event_forward_pong_request(uma_ptr_t requestee, uma_ptr_t requestor) noexcept -> ForwardPongRequestEvent{

        return ForwardPongRequestEvent{requestee, requestor};
    }

    constexpr auto make_event_forward_pingpong_request(uma_ptr_t requestee, uma_ptr_t requestor) noexcept -> ForwardPingPongRequestEvent{

        return ForwardPingPongRequestEvent{requestee, requestor};
    }

    constexpr auto make_event_forward_pong_signal(uma_ptr_t dst) noexcept -> ForwardPongSignalEvent{

        return ForwardPongSignalEvent{dst};
    } 

    constexpr auto make_event_forward_do_signal(uma_ptr_t dst) noexcept -> ForwardDoSignalEvent{

        return ForwardDoSignalEvent{dst};
    }

    constexpr auto make_event_backward_do_signal(uma_ptr_t dst) noexcept -> BackwardDoSignalEvent{

        return BackwardDoSignalEvent{dst};
    }

    constexpr auto virtualize_event(ForwardPingSignalEvent event) noexcept -> VirtualEvent{
        
        static_assert(dg::network_trivial_serializer::size(event) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs{};
        rs.event_kind = event_kind_forward_ping_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(ForwardPongSignalEvent event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(event) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs{};
        rs.event_kind = event_kind_forward_pong_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(ForwardPongRequestEvent event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(event) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs{};
        rs.event_kind = event_kind_forward_pong_request;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(ForwardPingPongRequestEvent event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(event) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs{};
        rs.event_kind = event_kind_forward_pingpong_request;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(ForwardDoSignalEvent event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(event) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs{};
        rs.event_kind = event_kind_forward_do_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(BackwardDoSignalEvent event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(event) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs{};
        rs.event_kind = event_kind_backward_do_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto read_virtual_event_kind(VirtualEvent event) noexcept -> memory_event_kind_t{

        return event.event_kind;
    }

    constexpr auto devirtualize_forward_ping_signal_event(VirtualEvent event) noexcept -> ForwardPingSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_forward_ping_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPingSignalEvent rs{};
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_pong_signal_event(VirtualEvent event) noexcept -> ForwardPongSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_forward_pong_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPongSignalEvent rs{};
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_pong_request_event(VirtualEvent event) noexcept -> ForwardPongRequestEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_forward_pong_request){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPongRequestEvent rs{};
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_pinppong_request_event(VirtualEvent event) noexcept -> ForwardPingPongRequestEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_forward_pingpong_request){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPingPongRequestEvent rs{};
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_do_signal_event(VirtualEvent event) noexcept -> ForwardDoSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_forward_do_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardDoSignalEvent rs{};
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_backward_do_signal_event(VirtualEvent event) noexcept -> BackwardDoSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_backward_do_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        BackwardDoSignalEvent rs{};
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }
}

#endif