#ifndef __DG_NETWORK_MEMCOMMIT_MODEL_H__
#define __DG_NETWORK_MEMCOMMIT_MODEL_H__

#include <stdint.h>
#include <stdlib.h>
#include <tuple>
#include "network_trivial_serializer.h" 
#include <array>

namespace dg::network_memcommit_taxonomy{

    using memory_event_taxonomy_t = uint8_t;

    enum memory_event_option: memory_event_taxonomy_t{
        forward_ping_signal             = 0u,
        forward_pong_request            = 1u,
        forward_pong_signal             = 2u,
        forward_do_signal               = 3u,
        backward_do_signal              = 4u,
    };
}

namespace dg::network_memcommit_factory{

    static inline constexpr size_t VIRTUAL_EVENT_BUFFER_SZ = size_t{1} << 5; 

    using namespace dg::network_memcommit_taxonomy;
    using uma_ptr_t                 = uint64_t;
    using virtual_memory_event_t    = std::array<char, VIRTUAL_EVENT_BUFFER_SZ>;

    auto make_event_forward_ping_signal(uma_ptr_t signalee) noexcept -> virtual_memory_event_t{

        constexpr size_t SERILIAZATION_SZ = dg::network_trivial_serializer::size(memory_event_taxonomy_t{}) + dg::network_trivial_serializer::size(std::tuple<uma_ptr_t>{}); 
        static_assert(SERILIAZATION_SZ <= VIRTUAL_EVENT_BUFFER_SZ);
        virtual_memory_event_t rs{};
        char * nxt  = dg::network_trivial_serializer::serialize_into(rs.data(), forward_ping_signal);
        char * nnxt = dg::network_trivial_serializer::serialize_into(nxt, std::make_tuple(signalee));

        return rs;
    }

    auto make_event_forward_pong_request(uma_ptr_t requestee, uma_ptr_t requestor) noexcept -> virtual_memory_event_t{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(memory_event_taxonomy_t{}) + dg::network_trivial_serializer::size(std::tuple<uma_ptr_t, uma_ptr_t>{});
        static_assert(SERIALIZATION_SZ <= VIRTUAL_EVENT_BUFFER_SZ);
        virtual_memory_event_t rs{};
        char * nxt  = dg::network_trivial_serializer::serialize_into(rs.data(), forward_pong_request);
        char * nnxt = dg::network_trivial_serializer::serialize_into(nxt, std::make_tuple(requestee, requestor));

        return rs; 

    }

    auto make_event_forward_pong_signal(uma_ptr_t signalee, uma_ptr_t signaler) noexcept -> virtual_memory_event_t{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(memory_event_taxonomy_t{}) + dg::network_trivial_serializer::size(std::tuple<uma_ptr_t, uma_ptr_t>{});
        static_assert(SERIALIZATION_SZ <= VIRTUAL_EVENT_BUFFER_SZ);
        virtual_memory_event_t rs{};
        char * nxt  = dg::network_trivial_serializer::serialize_into(rs.data(), forward_pong_signal);
        char * nnxt = dg::network_trivial_serializer::serialize_into(nxt, std::make_tuple(signalee, signaler));

        return rs;
    } 

    auto make_event_forward_do_signal(uma_ptr_t signalee) noexcept -> virtual_memory_event_t{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(memory_event_taxonomy_t{}) + dg::network_trivial_serializer::size(std::tuple<uma_ptr_t>{});
        static_assert(SERIALIZATION_SZ <= VIRTUAL_EVENT_BUFFER_SZ);;
        virtual_memory_event_t rs{};
        char * nxt  = dg::network_trivial_serializer::serialize_into(rs.data(), forward_do_signal);
        char * nnxt = dg::network_trivial_serializer::serialize_into(nxt, std::make_tuple(signalee));

        return rs;
    }

    auto make_event_backward_do_signal(uma_ptr_t signalee) noexcept -> virtual_memory_event_t{

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(memory_event_taxonomy_t{}) + dg::network_trivial_serializer::size(std::tuple<uma_ptr_t, uma_ptr_t>{});
        static_assert(SERIALIZATION_SZ <= VIRTUAL_EVENT_BUFFER_SZ);
        virtual_memory_event_t rs{};
        char * nxt  = dg::network_trivial_serializer::serialize_into(rs.data(), backward_do_signal);
        char * nnxt = dg::network_trivial_serializer::serialize_into(nxt, std::make_tuple(signalee));

        return rs;
    }

    auto read_event_forward_ping_signal(virtual_memory_event_t event) noexcept -> std::tuple<uma_ptr_t>{

        const char * buf = event.data() + dg::network_trivial_serializer::size(memory_event_taxonomy_t{});
        std::tuple<uma_ptr_t> rs{};
        dg::network_trivial_serializer::deserialize_into(rs, buf); 

        return rs;
    }

    auto read_event_forward_pong_request(virtual_memory_event_t event) noexcept -> std::tuple<uma_ptr_t, uma_ptr_t>{

        const char * buf = event.data() + dg::network_trivial_serializer::size(memory_event_taxonomy_t{});
        std::tuple<uma_ptr_t, uma_ptr_t> rs{};
        dg::network_trivial_serializer::deserialize_into(rs, buf);

        return rs;
    } 

    auto read_event_forward_pong_signal(virtual_memory_event_t event) noexcept -> std::tuple<uma_ptr_t, uma_ptr_t>{

        const char * buf = event.data() + dg::network_trivial_serializer::size(memory_event_taxonomy_t{});
        std::tuple<uma_ptr_t, uma_ptr_t> rs{};
        dg::network_trivial_serializer::deserialize_into(rs, buf);

        return rs;
    }

    auto read_event_forward_do_signal(virtual_memory_event_t event) noexcept -> std::tuple<uma_ptr_t>{

        const char * buf = event.data() + dg::network_trivial_serializer::size(memory_event_taxonomy_t{});
        std::tuple<uma_ptr_t> rs{};
        dg::network_trivial_serializer::deserialize_into(rs, buf);

        return rs;
    }

    auto read_event_backward_do_signal(virtual_memory_event_t event) noexcept -> std::tuple<uma_ptr_t, uma_ptr_t>{

        const char * buf = event.data() + dg::network_trivial_serializer::size(memory_event_taxonomy_t{});
        std::tuple<uma_ptr_t, uma_ptr_t> rs{};
        dg::network_trivial_serializer::deserialize_into(rs, buf);

        return rs;
    }
}

#endif