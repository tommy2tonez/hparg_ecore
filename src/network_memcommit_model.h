#ifndef __DG_NETWORK_MEMCOMMIT_MODEL_H__
#define __DG_NETWORK_MEMCOMMIT_MODEL_H__

#include <stdint.h>
#include <stdlib.h>
#include <tuple>
#include "network_trivial_serializer.h" 
#include <array>
#include "network_pointer.h"

namespace dg::network_memcommit_factory{

    //I feel like we are doing variants, YET this is the faster version of variants
    //the problem is that we can have direct access to the polymorphic type, which we can guarantee that there is no dispatch hole in between (this is not guaranteed by the implementation)
    //due to the delvrsrv_polymorphic_dispatch, we can actually optimize this to O(1) access, as if the devirtualization has no cost at all

    //this is incredibly complicated
    //we have iterated every possible way to forward + backward, I was not kidding when I said that the proposed approach is the most optimal approach for most of the cases 
    //we need to keep the virtualized structure <= 32 bytes or 16 bytes
    //and we need to actually keep the code size reasonable, so the compiler can know our intentions + optimize the code in the most optimal way

    using uma_ptr_t = dg::network_pointer::uma_ptr_t;

    struct ForwardPingSignalEvent{
        uma_ptr_t dst;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(dst, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(dst, operatable_id);
        }
    };

    struct ForwardPongSignalEvent{
        uma_ptr_t dst;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(dst, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(dst, operatable_id);
        }
    };

    struct ForwardPongRequestEvent{
        uma_ptr_t requestee;
        uma_ptr_t requestor;
        operatable_id_t operatable_id;
        std::optional<uma_ptr_t> notify_addr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(requestee, requestor, operatable_id, notify_addr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(requestee, requestor, operatable_id, notify_addr);
        }
    };

    struct ForwardPingPongRequestEvent{
        uma_ptr_t requestee;
        uma_ptr_t requestor;
        operatable_id_t operatable_id;
        std::optional<uma_ptr_t> notify_addr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(requestee, requestor, operatable_id, notify_addr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(requestee, requestor, operatable_id, notify_addr);
        }
    };

    struct ForwardDoSignalEvent{
        uma_ptr_t dst;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(dst, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(dst, operatable_id);
        }
    };

    struct BackwardDoSignalEvent{
        uma_ptr_t dst;
        operatable_id_t operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(dst, operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(dst, operatable_id);
        }
    };

    constexpr auto make_event_forward_ping_signal(uma_ptr_t dst, 
                                                  operatable_id_t operatable_id) noexcept -> ForwardPingSignalEvent{

        return ForwardPingSignalEvent{.dst = dst, 
                                      .operatable_id = operatable_id};
    }

    constexpr auto make_event_forward_pong_signal(uma_ptr_t dst, 
                                                  operatable_id_t operatable_id) noexcept -> ForwardPongSignalEvent{

        return ForwardPongSignalEvent{.dst = dst, 
                                      .operatable_id = operatable_id};
    } 

    constexpr auto make_event_forward_pong_request(uma_ptr_t requestee, 
                                                   uma_ptr_t requestor, 
                                                   operatable_id_t operatable_id, 
                                                   std::optional<uma_ptr_t> notify_addr) noexcept -> ForwardPongRequestEvent{

        return ForwardPongRequestEvent{.requestee       = requestee, 
                                       .requestor       = requestor, 
                                       .operatable_id   = operatable_id,
                                       .notify_addr     = notify_addr};
    }

    constexpr auto make_event_forward_pingpong_request(uma_ptr_t requestee, 
                                                       uma_ptr_t requestor, 
                                                       operatable_id_t operatable_id, 
                                                       std::optional<uma_ptr_t> notify_addr) noexcept -> ForwardPingPongRequestEvent{

        return ForwardPingPongRequestEvent{.requestee       = requestee, 
                                           .requestor       = requestor, 
                                           .operatable_id   = operatable_id,
                                           .notify_addr     = notify_addr};
    }

    constexpr auto make_event_forward_do_signal(uma_ptr_t dst, 
                                                operatable_id_t operatable_id) noexcept -> ForwardDoSignalEvent{

        return ForwardDoSignalEvent{.dst            = dst, 
                                    .operatable_id  = operatable_id};
    }

    constexpr auto make_event_backward_do_signal(uma_ptr_t dst, 
                                                 operatable_id_t operatable_id) noexcept -> BackwardDoSignalEvent{

        return BackwardDoSignalEvent{.dst           = dst, 
                                     .operatable_id = operatable_id};
    }

    //

    struct ForwardPingSignalAggregationEvent{
        uma_ptr_t sigagg_addr;
        ForwardPingSignalEvent event;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(sigagg_addr, event);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(sigagg_addr, event);
        }
    };

    struct ForwardPongSignalAggregationEvent{
        uma_ptr_t sigagg_addr;
        ForwardPongSignalEvent event;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(sigagg_addr, event);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(sigagg_addr, event);
        }
    };

    struct ForwardPongRequestAggregationEvent{
        uma_ptr_t sigagg_addr;
        ForwardPongRequestEvent event;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(sigagg_addr, event);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(sigagg_addr, event);
        }
    };

    struct ForwardPingPongRequestAggregationEvent{
        uma_ptr_t sigagg_addr;
        ForwardPingPongRequestEvent event;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(sigagg_addr, event);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(sigagg_addr, event);
        }
    };

    struct ForwardDoSignalAggregationEvent{
        uma_ptr_t sigagg_addr;
        ForwardDoSignalEvent event;

        template <class Reflectgor>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(sigagg_addr, event);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(sigagg_addr, event);
        }
    };

    struct BackwardDoSignalAggregationEvent{
        uma_ptr_t sigagg_addr;
        BackwardDoSignalEvent event;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(sigagg_addr, event);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(sigagg_addr, event);
        }
    };

    constexpr auto make_sigagg_forward_ping_signal(uma_ptr_t sigagg_addr, 
                                                   uma_ptr_t dst, 
                                                   operatable_id_t operatable_id) noexcept -> ForwardPingSignalAggregationEvent{
        
        return ForwardPingSignalAggregationEvent{.sigagg_addr   = sigagg_addr,
                                                 .event         = ForwardPingSignalEvent{.dst           = dst,
                                                                                         .operatable_id = operatable_id}};
    }

    constexpr auto make_sigagg_forward_pong_signal(uma_ptr_t sigagg_addr, 
                                                   uma_ptr_t dst, 
                                                   operatable_id_t operatable_id) noexcept -> ForwardPongSignalAggregationEvent{
        
        return ForwardPongSignalAggregationEvent{.sigagg_addr  = sigagg_addr,
                                                 .event        = ForwardPongSignalEvent{.dst            = dst,
                                                                                        .operatable_id  = operatable_id}};
    }

    constexpr auto make_sigagg_forward_pong_request(uma_ptr_t sigagg_addr, 
                                                    uma_ptr_t requestee, 
                                                    uma_ptr_t requestor,
                                                    operatable_id_t operatable_id, 
                                                    std::optional<uma_ptr_t> notify_addr) noexcept -> ForwardPongRequestAggregationEvent{
            
        return ForwardPongRequestAggregationEvent{.sigagg_addr  = sigagg_addr,
                                                  .event        = ForwardPongRequestEvent{.requestee        = requestee,
                                                                                          .requestor        = requestor,
                                                                                          .operatable_id    = operatable_id,
                                                                                          .notify_addr      = notify_addr}};
    }

    constexpr auto make_sigagg_forward_pingpong_request(uma_ptr_t sigagg_addr, 
                                                        uma_ptr_t requestee, 
                                                        uma_ptr_t requestor, 
                                                        operatable_id_t operatable_id, 
                                                        std::optional<uma_ptr_t> notify_addr) noexcept -> ForwardPingPongRequestAggregationEvent{

        return ForwardPingPongRequestAggregationEvent{.sigagg_addr  = sigagg_addr,
                                                      .event        = ForwardPingPongRequestEvent{.requestee        = requestee,
                                                                                                  .requestor        = requestor,
                                                                                                  .operatable_id    = operatable_id,
                                                                                                  .notify_addr      = notify_addr}};
    }

    constexpr auto make_sigagg_forward_do_signal(uma_ptr_t sigagg_addr, 
                                                 uma_ptr_t dst, 
                                                 operatable_id_t operatable_id) noexcept -> ForwardDoSignalAggregationEvent{
                        
        return ForwardDoSignalAggregationEvent{.sigagg_addr = sigagg_addr,
                                               .event       = ForwardDoSignalEvent{.dst             = dst, 
                                                                                   .operatable_id   = operatable_id}};
    }

    constexpr auto make_sigagg_backward_do_signal(uma_ptr_t sigagg_addr, 
                                                  uma_ptr_t dst, 
                                                  operatable_id_t operatable_id) noexcept -> BackwardDoSignalAggregationEvent{

        return BackwardDoSignalAggregationEvent{.sigagg_addr    = sigagg_addr,
                                                .event          = BackwardDoSignalEvent{.dst            = dst,
                                                                                        .operatable_id  = operatable_id}};
    }

    //

    static inline constexpr size_t VIRTUAL_AGGREGATION_BUFFER_SZ = size_t{1} << 5; 

    using aggregation_event_kind_t = uint8_t; 

    enum enum_aggregation_event: aggregation_event_kind_t{
        aggregation_kind_forward_ping_signal        = 0u,
        aggregation_kind_forward_pong_request       = 1u,
        aggregation_kind_forward_pingpong_request   = 2u,
        aggregation_kind_forward_pong_signal        = 3u,
        aggregation_kind_forward_do_signal          = 4u,
        aggregation_kind_backward_do_signal         = 5u
    };

    struct VirtualSignalAggregationEvent{
        aggregation_event_kind_t aggregation_kind;
        std::array<char, VIRTUAL_AGGREGATION_BUFFER_SZ> content;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(aggregation_kind, content);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(aggregation_kind, content);
        }
    };

    constexpr auto virtualize_sigagg(const ForwardPingSignalAggregationEvent& event) noexcept -> VirtualSignalAggregationEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardPongSignalAggregationEvent{}) <= VIRTUAL_AGGREGATION_BUFFER_SZ);

        VirtualSignalAggregationEvent rs;
        rs.aggregation_kind = aggregation_kind_forward_ping_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_sigagg(const ForwardPongSignalAggregationEvent& event) noexcept -> VirtualSignalAggregationEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardPongSignalAggregationEvent{}) <= VIRTUAL_AGGREGATION_BUFFER_SZ);

        VirtualSignalAggregationEvent rs;
        rs.aggregation_kind = aggregation_kind_forward_pong_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_sigagg(const ForwardPongRequestAggregationEvent& event) noexcept -> VirtualSignalAggregationEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardPongRequestAggregationEvent{}) <= VIRTUAL_AGGREGATION_BUFFER_SZ);

        VirtualSignalAggregationEvent rs;
        rs.aggregation_kind = aggregation_kind_forward_pong_request;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_sigagg(const ForwardPingPongRequestAggregationEvent& event) noexcept -> VirtualSignalAggregationEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardPingPongRequestAggregationEvent{}) <= VIRTUAL_AGGREGATION_BUFFER_SZ);

        VirtualSignalAggregationEvent rs;
        rs.aggregation_kind = aggregation_kind_forward_pingpong_request;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_sigagg(const ForwardDoSignalAggregationEvent& event) noexcept -> VirtualSignalAggregationEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardDoSignalAggregationEvent{}) <= VIRTUAL_AGGREGATION_BUFFER_SZ);

        VirtualSignalAggregationEvent rs;
        rs.aggregation_kind = aggregation_kind_forward_do_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_sigagg(const BackwardDoSignalAggregationEvent& event) noexcept -> VirtualSignalAggregationEvent{

        static_assert(dg::network_trivial_serializer::size(BackwardDoSignalAggregationEvent{}) <= VIRTUAL_AGGREGATION_BUFFER_SZ);

        VirtualSignalAggregationEvent rs;
        rs.aggregation_kind = aggregation_kind_backward_do_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto read_aggregation_kind(const VirtualSignalAggregationEvent& event) noexcept -> aggregation_event_kind_t{

        return event.aggregation_kind;
    }

    constexpr auto devirtualize_forward_ping_signal_aggregation_event(const VirtualSignalAggregationEvent& event) noexcept -> ForwardPingSignalAggregationEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.aggregation_kind != aggregation_kind_forward_ping_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPingSignalAggregationEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_pong_signal_aggregation_event(const VirtualSignalAggregationEvent& event) noexcept -> ForwardPongSignalAggregationEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.aggregation_kind != aggregation_kind_forward_pong_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPongSignalAggregationEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_pong_request_aggregation_event(const VirtualSignalAggregationEvent& event) noexcept -> ForwardPongRequestAggregationEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.aggregation_kind != aggregation_kind_forward_pong_request){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPongRequestAggregationEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_pingpong_request_aggregation_event(const VirtualSignalAggregationEvent& event) noexcept -> ForwardPingPongRequestAggregationEvent{
    
        if constexpr(DEBUG_MODE_FLAG){
            if (event.aggregation_kind != aggregation_kind_forward_pingpong_request){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPingPongRequestAggregationEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_do_signal_aggregation_event(const VirtualSignalAggregationEvent& event) noexcept -> ForwardDoSignalAggregationEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.aggregation_kind != aggregation_kind_forward_do_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardDoSignalAggregationEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_backward_do_signal_aggregation_event(const VirtualSignalAggregationEvent& event) noexcept -> BackwardDoSignalAggregationEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.aggregation_kind != aggregation_kind_backward_do_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        BackwardDoSignalAggregationEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    //

    static inline constexpr size_t VIRTUAL_EVENT_BUFFER_SZ = size_t{1} << 5; 

    using memory_event_kind_t = uint8_t;

    enum enum_memory_event: memory_event_kind_t{
        event_kind_forward_ping_signal      = 0u,
        event_kind_forward_pong_request     = 1u,
        event_kind_forward_pingpong_request = 2u,
        event_kind_forward_pong_signal      = 3u,
        event_kind_forward_do_signal        = 4u,
        event_kind_backward_do_signal       = 5u,
        event_kind_signal_aggregation       = 6u
    };

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

    constexpr auto virtualize_event(const ForwardPingSignalEvent& event) noexcept -> VirtualEvent{
        
        static_assert(dg::network_trivial_serializer::size(ForwardPingSignalEvent{}) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs;
        rs.event_kind = event_kind_forward_ping_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(const ForwardPongSignalEvent& event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardPongSignalEvent{}) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs;
        rs.event_kind = event_kind_forward_pong_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(const ForwardPongRequestEvent& event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardPongRequestEvent{}) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs;
        rs.event_kind = event_kind_forward_pong_request;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(const ForwardPingPongRequestEvent& event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardPingPongRequestEvent{}) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs;
        rs.event_kind = event_kind_forward_pingpong_request;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(const ForwardDoSignalEvent& event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardDoSignalEvent{}) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs;
        rs.event_kind = event_kind_forward_do_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(const BackwardDoSignalEvent& event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(BackwardDoSignalEvent{}) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs;
        rs.event_kind = event_kind_backward_do_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto virtualize_event(const VirtualSignalAggregationEvent& event) noexcept -> VirtualEvent{

        static_assert(dg::network_trivial_serializer::size(VirtualSignalAggregationEvent{}) <= VIRTUAL_EVENT_BUFFER_SZ);

        VirtualEvent rs;
        rs.event_kind = event_kind_signal_aggregation;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto read_virtual_event_kind(const VirtualEvent& event) noexcept -> memory_event_kind_t{

        return event.event_kind;
    }

    constexpr auto devirtualize_forward_ping_signal_event(const VirtualEvent& event) noexcept -> ForwardPingSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_forward_ping_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPingSignalEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_pong_signal_event(const VirtualEvent& event) noexcept -> ForwardPongSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_forward_pong_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPongSignalEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_pong_request_event(const VirtualEvent& event) noexcept -> ForwardPongRequestEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_forward_pong_request){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPongRequestEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_pingpong_request_event(const VirtualEvent& event) noexcept -> ForwardPingPongRequestEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_forward_pingpong_request){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPingPongRequestEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_forward_do_signal_event(const VirtualEvent& event) noexcept -> ForwardDoSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_forward_do_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardDoSignalEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_backward_do_signal_event(const VirtualEvent& event) noexcept -> BackwardDoSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_backward_do_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        BackwardDoSignalEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto devirtualize_sigagg_event(const VirtualEvent& event) noexcept -> VirtualSignalAggregationEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_signal_aggregation){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        VirtualSignalAggregationEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }
}

#endif