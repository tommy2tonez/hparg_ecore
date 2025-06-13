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

    //we'll do compressions later at the virtualization phase if there are usecases, yet the semantic layer looks like this

    struct ForwardPongRequestEvent{
        uma_ptr_t requestee;
        uma_ptr_t requestor;
        operatable_id_t operatable_id;
        std::optional<uma_ptr_t> notify_addr;
        operatable_id_t notify_addr_operatable_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(requestee, requestor, operatable_id, notify_addr, notify_addr_operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(requestee, requestor, operatable_id, notify_addr, notify_addr_operatable_id);
        }
    };

    struct ForwardPingPongRequestEvent{
        uma_ptr_t requestee;
        uma_ptr_t requestor;
        operatable_id_t operatable_id;
        std::optional<uma_ptr_t> notify_addr;
        operatable_id_t notify_addr_operatable_id; 

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(requestee, requestor, operatable_id, notify_addr, notify_addr_operatable_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(requestee, requestor, operatable_id, notify_addr, notify_addr_operatable_id);
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

    constexpr auto make_event_forward_pong_request(uma_ptr_t requestee, 
                                                   uma_ptr_t requestor, 
                                                   operatable_id_t operatable_id, 
                                                   std::optional<uma_ptr_t> notify_addr
                                                   operatable_id_t notify_addr_operatable_id) noexcept -> ForwardPongRequestEvent{

        return ForwardPongRequestEvent{.requestee                   = requestee, 
                                       .requestor                   = requestor, 
                                       .operatable_id               = operatable_id,
                                       .notify_addr                 = notify_addr,
                                       .notify_addr_operatable_id   = notify_addr_operatable_id};
    }

    constexpr auto make_event_forward_pingpong_request(uma_ptr_t requestee, 
                                                       uma_ptr_t requestor, 
                                                       operatable_id_t operatable_id, 
                                                       std::optional<uma_ptr_t> notify_addr,
                                                       operatable_id_t notify_addr_operatable_id) noexcept -> ForwardPingPongRequestEvent{

        return ForwardPingPongRequestEvent{.requestee                   = requestee, 
                                           .requestor                   = requestor, 
                                           .operatable_id               = operatable_id,
                                           .notify_addr                 = notify_addr,
                                           .notify_addr_operatable_id   = notify_addr_operatable_id};
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

    static inline constexpr size_t SIGAGG_VIRTUAL_EVENT_BUFFER_SZ = size_t{1} << 5;

    using sigagg_virtual_event_kind_t = uint8_t; 

    enum enum_sigagg_memory_event: sigagg_virtual_event_kind_t{
        sigagg_event_kind_forward_ping_signal           = 0u,
        sigagg_event_kind_forward_pong_request          = 1u,
        sigagg_event_kind_forward_pingpong_request      = 2u,
        sigagg_event_kind_forward_do_signal             = 3u,
        sigagg_event_kind_backward_do_signal            = 4u,
        sigagg_event_kind_self_decay_signal             = 5u
    };

    struct SigaggVirtualEvent{
        sigagg_virtual_event_kind_t event_kind;
        std::array<char, SIGAGG_VIRTUAL_EVENT_BUFFER_SZ> content;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(event_kind, content);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(event_kind, content);
        }
    };

    struct SignalAggregationEvent{
        uma_ptr_t smph_addr;
        operatable_id_t operatable_id;
        SigaggVirtualEvent sigagg_virtual_event;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(smph_addr, operatable_id, sigagg_virtual_event);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(smph_addr, operatable_id, sigagg_virtual_event);
        }
    };

    using sigagg_event_t = SignalAggregationEvent;

    constexpr auto sigagg_virtualize_event(const ForwardPingSignalEvent& event) noexcept -> SigaggVirtualEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardPingSignalEvent{}) <= SIGAGG_VIRTUAL_EVENT_BUFFER_SZ);

        SigaggVirtualEvent rs;
        rs.event_kind = sigagg_event_kind_forward_ping_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto sigagg_virtualize_event(const ForwardPongRequestEvent& event) noexcept -> SigaggVirtualEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardPongRequestEvent{}) <= SIGAGG_VIRTUAL_EVENT_BUFFER_SZ);

        SigaggVirtualEvent rs;
        rs.event_kind = sigagg_event_kind_forward_pong_request;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto sigagg_virtualize_event(const ForwardPingPongRequestEvent& event) noexcept -> SigaggVirtualEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardPingPongRequestEvent{}) <= SIGAGG_VIRTUAL_EVENT_BUFFER_SZ);

        SigaggVirtualEvent rs;
        rs.event_kind = sigagg_event_kind_forward_pingpong_request;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto sigagg_virtualize_event(const ForwardDoSignalEvent& event) noexcept -> SigaggVirtualEvent{

        static_assert(dg::network_trivial_serializer::size(ForwardDoSignalEvent{}) <= SIGAGG_VIRTUAL_EVENT_BUFFER_SZ);

        SigaggVirtualEvent rs;
        rs.event_kind = sigagg_event_kind_forward_do_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    constexpr auto sigagg_virtualize_event(const BackwardDoSignalEvent& event) noexcept -> SigaggVirtualEvent{

        static_assert(dg::network_trivial_serializer::size(BackwardDoSignalEvent{}) <= SIGAGG_VIRTUAL_EVENT_BUFFER_SZ);

        SigaggVirtualEvent rs;
        rs.event_kind = sigagg_event_kind_backward_do_signal;
        dg::network_trivial_serializer::serialize_into(rs.content.data(), event);

        return rs;
    }

    static inline constexpr auto sigagg_virtualize_event_lambda = []<class ...Args>(Args&& ...args) noexcept(noexcept(sigagg_virtualize_event(std::declval<Args&&>()...))){
        return sigagg_virtualize_event(std::forward<Args>(args)...);
    };

    constexpr auto sigagg_devirtualize_forward_ping_signal_event(const SigaggVirtualEvent& event) noexcept -> ForwardPingSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != sigagg_event_kind_forward_ping_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPingSignalEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto sigagg_devirtualize_forward_pong_request_event(const SigaggVirtualEvent& event) noexcept -> ForwardPongRequestEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != sigagg_event_kind_forward_pong_request){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPongRequestEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto sigagg_devirtualize_forward_pingpong_request_event(const SigaggVirtualEvent& event) noexcept -> ForwardPingPongRequestEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != sigagg_event_kind_forward_pingpong_request){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardPingPongRequestEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto sigagg_devirtualize_forward_do_signal_event(const SigaggVirtualEvent& event) noexcept -> ForwardDoSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != sigagg_event_kind_forward_do_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        ForwardDoSignalEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto sigagg_devirtualize_backward_do_signal_event(const SigaggVirtualEvent& event) noexcept -> BackwardDoSignalEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != sigagg_event_kind_backward_do_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        BackwardDoSignalEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    struct SigaggVirtualEventAdaptiveContainer{
        SigaggVirtualEvent value;

        constexpr SigaggVirtualEventAdaptiveContainer() = default;

        constexpr SigaggVirtualEventAdaptiveContainer(const SigaggVirtualEvent& sigagg) noexcept: value(sigagg){}

        template <class T, std::enable_if_t<std::is_invocable_v<decltype(sigagg_virtualize_event_lambda), T&&>, bool> = true>
        constexpr SigaggVirtualEventAdaptiveContainer(T&& event) noexcept: value(sigagg_virtualize_event_lambda(std::forward<T>(event))){}
    };

    constexpr auto make_sigagg(uma_ptr_t smph_addr,
                               operatable_id_t operatable_id,
                               const SigaggVirtualEventAdaptiveContainer& event) noexcept -> SignalAggregationEvent{

        return SignalAggregationEvent{.smph_addr            = smph_addr,
                                      .operatable_id        = operatable_id,
                                      .signal_virtual_event = event.value};
    }

    //

    static inline constexpr size_t VIRTUAL_EVENT_BUFFER_SZ = size_t{1} << 5; 

    using memory_event_kind_t = uint8_t;

    enum enum_memory_event: memory_event_kind_t{
        event_kind_forward_ping_signal          = 0u,
        event_kind_forward_pong_request         = 1u,
        event_kind_forward_pingpong_request     = 2u,
        event_kind_forward_do_signal            = 3u,
        event_kind_backward_do_signal           = 4u,
        event_kind_signal_aggregation_signal    = 5u
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
        rs.event_kind = event_kind_signal_aggregation_signal;
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

    constexpr auto devirtualize_sigagg_signal_event(const VirtualEvent& event) noexcept -> VirtualSignalAggregationEvent{

        if constexpr(DEBUG_MODE_FLAG){
            if (event.event_kind != event_kind_signal_aggregation_signal){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        VirtualSignalAggregationEvent rs;
        dg::network_trivial_serializer::deserialize_into(rs, event.content.data());

        return rs;
    }

    constexpr auto to_virtual_event(const SigaggVirtualEvent& event) noexcept -> std::expected<VirtualEvent, exception_t>{

        switch (event.event_kind){
            case sigagg_event_kind_forward_ping_signal:
            {
                return virtualize_event(sigagg_devirtualize_forward_ping_signal_event(event));
            }
            case sigagg_event_kind_forward_pong_request:
            {
                return virtualize_event(sigagg_devirtualize_forward_pong_request_event(event));
            }
            case sigagg_event_kind_forward_pingpong_request:
            {
                return virtualize_event(sigagg_devirtualize_forward_pingpong_request_event(event));
            }
            case sigagg_event_kind_forward_do_signal:
            {
                return virtualize_event(sigagg_devirtualize_forward_do_signal_event(event));
            }
            case sigagg_event_kind_backward_do_signal:
            {
                return virtualize_event(sigagg_devirtualize_backward_do_signal_event(event));
            }
            default:
            {
                return std::unexpected(dg::network_exception::BAD_OPERATION);
            }
        }
    }
}

#endif