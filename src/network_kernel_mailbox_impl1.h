#ifndef __DG_SENDERLESS_MAILBOX_H__
#define __DG_SENDERLESS_MAILBOX_H__

//define HEADER_CONTROL 8

#include <stdint.h>
#include <vector>
#include <chrono>
#include <optional>
#include <memory>
#include <string>
#include "network_compact_serializer.h"
#include <mutex>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <thread>
#include <deque>
#include <sched.h>
#include <pthread.h>
#include <math.h>
#include "assert.h"
#include "network_std_container.h"
#include <array>
#include "network_log.h"
#include "network_exception.h"
#include "network_concurrency_x.h"
#include "stdx.h"
#include <chrono>
#include <array>
#include "network_randomizer.h"

namespace dg::network_kernel_mailbox_impl1::types{

    using factory_id_t          = std::array<char, 32>;
    using local_packet_id_t     = uint64_t;
}

namespace dg::network_kernel_mailbox_impl1::model{

    using namespace dg::network_kernel_mailbox_impl1::types;
    
    struct SocketHandle{
        int kernel_sock_fd;
        int sin_fam;
        int comm;
        int protocol;
    };
    
    struct IP{
        std::array<char, 4> ipv4;
        std::array<char, 16> ipv6;
        bool flag;

        auto data() const noexcept -> const char *{

            if (this->flag){
                return this->ipv4.data();
            } else{
                return this->ipv6.data();
            }
        }

        auto sin_fam() const noexcept{

            if (this->flag){
                return AF_INET;
            } else{
                return AF_INET6;
            }
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ipv4, ipv6, flag);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ipv4, ipv6, flag);
        }
    };

    struct Address{
        IP ip;
        uint16_t port;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ip, port);
        } 

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ip, port);
        }
    };

    struct GlobalPacketIdentifier{
        local_packet_id_t local_packet_id;
        factory_id_t factory_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(local_packet_id, factory_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(local_packet_id, factory_id);
        }
    };

    using global_packet_id_t = GlobalPacketIdentifier;

    struct PacketHeader{
        Address fr_addr;
        Address to_addr; 
        global_packet_id_t id;
        uint8_t retransmission_count;
        uint8_t priority;
        uint8_t kind; 
        std::array<uint64_t, 16> port_utc_stamps;
        uint8_t port_stamp_sz; 

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(fr_addr, to_addr, id, retransmission_count, priority, kind, port_utc_stamps, port_stamp_sz);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(fr_addr, to_addr, id, retransmission_count, priority, kind, port_utc_stamps, port_stamp_sz);
        }
    };

    struct Packet: PacketHeader{
        dg::string content;
    };

    struct ScheduledPacket{
        Packet pkt;
        std::chrono::nanoseconds sched_time;
    };
}

namespace dg::network_kernel_mailbox_impl1::constants{

    using namespace std::literals::chrono_literals;
    using namespace std::chrono;

    enum packet_kind: uint8_t{
        rts_ack = 0,
        request = 1
    };

    static inline constexpr size_t MAXIMUM_MSG_SIZE = size_t{1} << 10;  
}

namespace dg::network_kernel_mailbox_impl1::data_structure{

    template <class T>
    class unordered_set_interface{ //this is fine - if the interface is only referenced by an object then devirt is automatically performed by compiler - no overhead

        public:

            static_assert(std::is_trivial_v<T>);

            virtual ~unordered_set_interface() noexcept = default;
            virtual void insert(T key) noexcept = 0;
            virtual auto contains(const T& key) const noexcept -> bool = 0;
    };
} 

namespace dg::network_kernel_mailbox_impl1::packet_controller{

    using namespace dg::network_kernel_mailbox_impl1::model;
    
    class SchedulerInterface{

        public:

            virtual ~SchedulerInterface() noexcept = default;
            virtual auto schedule(Address)  noexcept -> std::chrono::nanoseconds = 0;
            virtual auto feedback(Address, std::chrono::nanoseconds) noexcept -> exception_t = 0;
    };

    class UpdatableInterface{

        public:

            virtual ~UpdatableInterface() noexcept = default;
            virtual void update() noexcept = 0;
    };
    
    class IDGeneratorInterface{

        public:

            virtual ~IDGeneratorInterface() noexcept = default;
            virtual auto get() noexcept -> global_packet_id_t = 0;
    };

    class PacketGeneratorInterface{

        public:

            virtual ~PacketGeneratorInterface() noexcept = default;
            virtual auto get(Address to_addr, dg::string content) noexcept -> Packet = 0;
    };

    class RetransmissionManagerInterface{

        public:

            virtual ~RetransmissionManagerInterface() noexcept = default;
            virtual auto add_retriable(Packet) noexcept -> exception_t = 0;
            virtual void ack(global_packet_id_t) noexcept = 0;
            virtual auto get_retriables() noexcept -> dg::vector<Packet> = 0;
    };

    class PacketContainerInterface{
        
        public:
            
            virtual ~PacketContainerInterface() noexcept = default;
            virtual void push(Packet) noexcept = 0; //the reason this is void is because it looks dumb if auto push(Packet) noexcept -> std::expected<Packet, exception_t> - yeah the language is hard - noexcept -> exception_t is only the tip of the iceberg (https://fouronnes.github.io/cppiceberg/)
            virtual auto pop() noexcept -> std::optional<Packet> = 0;
    };

    class InBoundIDControllerInterface{

        public:

            virtual ~InBoundIDControllerInterface() noexcept = default;
            virtual auto thru(global_packet_id_t) noexcept -> std::expected<bool, exception_t> = 0; //std::expected<bool, exception_t> is a good practice (look std::filesystem::exists)- they are used for representing different things - bool = false denotes that the thru packet is of valid format
    };

    class InBoundTrafficControllerInterface{

        public:

            virtual ~InBoundTrafficControllerInterface() noexcept = default;
            virtual auto thru(Address) noexcept -> std::expected<bool, exception_t> = 0;
    };
}

namespace dg::network_kernel_mailbox_impl1::core{

    using namespace dg::network_kernel_mailbox_impl1::model;

    class MailboxInterface{
        
        public: 

            virtual ~MailboxInterface() noexcept = default;
            virtual void send(Address, dg::string) noexcept = 0;
            virtual auto recv() noexcept -> std::optional<dg::string> = 0;
    };
}

namespace dg::network_kernel_mailbox_impl1::utility{

    using namespace dg::network_kernel_mailbox_impl1::model;

    static auto to_factory_id(Address addr) noexcept -> factory_id_t{

        static_assert(dg::network_trivial_serializer::size(Address{}) <= dg::network_trivial_serializer::size(factory_id_t{}));
        static_assert(std::has_unique_object_representations_v<factory_id_t>);

        factory_id_t rs{};
        dg::network_trivial_serializer::serialize_into(reinterpret_cast<char *>(&rs), addr); //-> &rs - fine - this is defined according to std

        return rs;
    } 

    static auto serialize_packet(Packet packet) noexcept -> dg::string{

        size_t header_sz    = dg::network_compact_serializer::integrity_size(static_cast<const PacketHeader&>(packet));  //assure that PacketHeader is constexpr sz
        size_t content_sz   = packet.content.size();
        size_t total_sz     = content_sz + header_sz;
        dg::string bstream = std::move(packet.content);
        bstream.resize(total_sz);
        char * header_ptr   = bstream.data() + content_sz;
        dg::network_compact_serializer::integrity_serialize_into(header_ptr, static_cast<const PacketHeader&>(packet));

        return bstream;
    }

    static auto deserialize_packet(dg::string bstream) noexcept -> std::expected<Packet, exception_t>{

        auto header_sz      = dg::network_compact_serializer::integrity_size(PacketHeader{});
        Packet rs           = {};
        auto [left, right]  = stdx::backsplit_str(std::move(bstream), header_sz);
        rs.content          = std::move(left);
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<PacketHeader>)(static_cast<PacketHeader&>(rs), right.data(), right.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }
}

namespace dg::network_kernel_mailbox_impl1::socket_service{

    using namespace dg::network_kernel_mailbox_impl1::model;

    using socket_close_t = void (*)(SocketHandle *) noexcept; 

    template <class T>
    static auto legacy_struct_default_init() noexcept -> T{

        static_assert(std::is_trivial_v<T>);
        T rs{}; //list initializer is important for C++, UB otherwise
        std::memset(&rs, 0, sizeof(T));

        return rs;
    }

    static auto open_socket(int sin_fam, int comm, int protocol) noexcept -> std::expected<std::unique_ptr<SocketHandle, socket_close_t>, exception_t>{

        auto destructor = [](SocketHandle * sock) noexcept{
            if (close(sock->kernel_sock_fd) == -1){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_kernel_error(errno)));
                std::abort();
            };

            delete sock;
        };

        int sock = socket(sin_fam, comm, protocol);
        
        if (sock == -1){
            return std::unexpected(dg::network_exception::wrap_kernel_error(errno));
        }

        return std::unique_ptr<SocketHandle, socket_close_t>(new SocketHandle{sock, sin_fam, comm, protocol}, destructor);
    }

    static auto port_socket_ipv6(SocketHandle sock, uint16_t port) noexcept -> exception_t{

        if constexpr(DEBUG_MODE_FLAG){
            if (sock.sin_fam != AF_INET6){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        struct sockaddr_in6 server  = legacy_struct_default_init<struct sockaddr_in6>();
        server.sin6_family          = AF_INET6;
        server.sin6_addr            = in6addr_any;
        server.sin6_port            = htons(port);

        if (bind(sock.kernel_sock_fd, (struct sockaddr *) &server, sizeof(struct sockaddr_in6)) == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        return dg::network_exception::SUCCESS;
    }

    static auto port_socket_ipv4(SocketHandle sock, uint16_t port) noexcept -> exception_t{

        if constexpr(DEBUG_MODE_FLAG){
            if (sock.sin_fam != AF_INET){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        struct sockaddr_in server   = legacy_struct_default_init<struct sockaddr_in>();
        server.sin_family           = AF_INET;
        server.sin_addr.s_addr      = INADDR_ANY;
        server.sin_port             = htons(port);

        if (bind(sock.kernel_sock_fd, (struct sockaddr *) &server, sizeof(struct sockaddr_in)) == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        return dg::network_exception::SUCCESS;
    }

    static auto port_socket(SocketHandle sock, uint16_t port) noexcept -> exception_t{

        if (sock.sin_fam == AF_INET6){
            return port_socket_ipv6(sock, port);
        }

        if (sock.sin_fam == AF_INET){
            return port_socket_ipv4(sock, port);
        }

        return dg::network_exception::INVALID_ARGUMENT;
    }

    static auto send_noblock_ipv6(SocketHandle sock, model::Address to_addr, const void * buf, size_t sz) noexcept -> exception_t{

        struct sockaddr_in6 server = legacy_struct_default_init<struct sockaddr_in6>();
        
        if constexpr(DEBUG_MODE_FLAG){
            if (to_addr.ip.sin_fam() != AF_INET6){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }

            if (sock.sin_fam != AF_INET6){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();   
            }

            if (sz > constants::MAXIMUM_MSG_SIZE){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }

            if (sock.protocol != SOCK_DGRAM){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        if (inet_pton(AF_INET6, to_addr.ip.data(), &server.sin6_addr) == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        server.sin6_family  = AF_INET6;
        server.sin6_port    = htons(to_addr.port);
        auto n              = sendto(sock.kernel_sock_fd, buf, stdx::wrap_safe_integer_cast(sz), MSG_DONTWAIT, (const struct sockaddr *) &server, sizeof(struct sockaddr_in6));

        if (n == -1){ //this is defined - 
            return dg::network_exception::wrap_kernel_error(errno);
        }

        if (n != sz){ //this is defined -
            return dg::network_exception::RUNTIME_SOCKETIO_ERROR;
        }

        return dg::network_exception::SUCCESS;
    }

    static auto send_noblock_ipv4(SocketHandle sock, model::Address to_addr, const void * buf, size_t sz) noexcept -> exception_t{

        struct sockaddr_in server = legacy_struct_default_init<struct sockaddr_in>();
        
        if constexpr(DEBUG_MODE_FLAG){
            if (to_addr.ip.sin_fam() != AF_INET){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
            if (sock.sin_fam != AF_INET){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();   
            }

            if (sz > constants::MAXIMUM_MSG_SIZE){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }

            if (sock.protocol != SOCK_DGRAM){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        if (inet_pton(AF_INET, to_addr.ip.data(), &server.sin_addr) == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        server.sin_family   = AF_INET;
        server.sin_port     = htons(to_addr.port);
        auto n              = sendto(sock.kernel_sock_fd, buf, stdx::wrap_safe_integer_cast(sz), MSG_DONTWAIT, (const struct sockaddr *) &server, sizeof(struct sockaddr_in));

        if (n == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        if (n != sz){
            return dg::network_exception::RUNTIME_SOCKETIO_ERROR;
        }

        return dg::network_exception::SUCCESS;
    } 

    static auto send_noblock(SocketHandle sock, model::Address to_addr, const void * buf, size_t sz) noexcept -> exception_t{

        //TODOS:
        //another kernel protocol is required to saturate network bandwidth - either reimplementation or SO_REUSEPORT + SO_ATTACH + disable gro + unload ip tables + disable validation + disable audit + skip id calculation
        //on top of all those configurations, use multiple ports to spam TCP packets - make sure that there's no lock congestion

        if (sock.sin_fam == AF_INET6){
            return send_noblock_ipv6(sock, to_addr, buf, sz);
        }
    
        if (sock.sin_fam == AF_INET){
            return send_noblock_ipv4(sock, to_addr, buf, sz);
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return {};
    }

    static auto recv_block(SocketHandle sock, void * dst, size_t& dst_sz, size_t dst_cap) noexcept -> exception_t{

        //TODOS:
        //another kernel protocol is required to saturate network bandwidth - either reimplementation or SO_REUSEPORT + SO_ATTACH + disable gro + unload ip tables + disable validation + disable audit + skip id calculation
        //on top of all those configurations, use multiple ports to spam TCP packets - make sure that there's no lock congestion

        if constexpr(DEBUG_MODE_FLAG){
            if (sock.protocol != SOCK_DGRAM){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        struct sockaddr_storage from    = legacy_struct_default_init<struct sockaddr_storage>();
        socklen_t length                = sizeof(from);
        auto n                          = recvfrom(sock.kernel_sock_fd, dst, stdx::wrap_safe_integer_cast(dst_cap), 0, (struct sockaddr *) &from, &length);

        if (n == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }
        
        dst_sz = stdx::safe_integer_cast<size_t>(n);
        return dg::network_exception::SUCCESS;
    }
}

namespace dg::network_kernel_mailbox_impl1::data_structure{

    template <class T>
    class temporal_unordered_set: public virtual unordered_set_interface<T>{

        private:

            dg::unordered_set<T> hashset;
            dg::deque<T> entries;
            size_t cap;

        public:

            temporal_unordered_set(dg::unordered_set<T> hashset,
                                   dg::deque<T> entries, 
                                   size_t cap) noexcept: hashset(std::move(hashset)),
                                                         entries(std::move(entries)),
                                                         cap(cap){}

            void insert(T key) noexcept{

                if (this->hashset.contains(key)){
                    return;
                }

                if (this->entries.size() == this->cap){
                    size_t half_cap = this->cap >> 1;

                    for (size_t i = 0u; i < half_cap; ++i){
                        T cur = this->entries.front();
                        this->entries.pop_front();
                        this->hashset.erase(cur);
                    }
                }

                this->hashset.insert(key);
                this->entries.push_back(key);
            }

            auto contains(const T& key) const noexcept -> bool{

                return this->hashset.contains(key);
            }
    };

    struct Factory{

        template <class T>
        static auto get_temporal_unordered_set(size_t capacity) -> std::unique_ptr<unordered_set_interface<T>>{

            const size_t MINIMUM_CAPACITY   = size_t{1} << 1;
            const size_t MAXIMUM_CAPACITY   = size_t{1} << 30;
            
            if (std::clamp(capacity, MINIMUM_CAPACITY, MAXIMUM_CAPACITY) != capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto hashset    = dg::unordered_set<T>{};
            hashset.reserve(capacity);
            auto entries    = dg::deque<T>{};

            return std::make_unique<temporal_unordered_set<T>>(std::move(hashset), std::move(entries), capacity);
        }
    };
}

namespace dg::network_kernel_mailbox_impl1::packet_service{

    using namespace dg::network_kernel_mailbox_impl1::model;
    
    static auto request_to_ack(const model::Packet& pkt) noexcept -> model::Packet{

        Packet rs{};

        rs.to_addr              = pkt.fr_addr;
        rs.fr_addr              = pkt.to_addr;
        rs.id                   = pkt.id;
        rs.retransmission_count = pkt.retransmission_count;
        rs.priority             = pkt.priority;
        rs.kind                 = constants::rts_ack;
        rs.port_utc_stamps      = pkt.port_utc_stamps;
        rs.port_stamp_sz        = pkt.port_stamp_sz;

        return rs;
    }

    static auto get_transit_time(const model::Packet& pkt) noexcept -> std::expected<std::chrono::nanoseconds, exception_t>{
        
        using namespace std::chrono_literals; 

        if (pkt.port_stamp_sz % 2 != 0){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        const std::chrono::nanoseconds MIN_LAPSED = std::chrono::duration_cast<std::chrono::nanoseconds>(1ns);
        const std::chrono::nanoseconds MAX_LAPSED = std::chrono::duration_cast<std::chrono::nanoseconds>(100s);
        std::chrono::nanoseconds lapsed{};

        for (size_t i = 1; i < pkt.port_stamp_sz; i += 2){
            std::chrono::nanoseconds cur    = std::chrono::nanoseconds(pkt.port_utc_stamps[i]);
            std::chrono::nanoseconds prev   = std::chrono::nanoseconds(pkt.port_utc_stamps[i - 1]);

            if (cur < prev){
                return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
            }

            std::chrono::nanoseconds diff   = cur - prev; 

            if (std::clamp(diff, MIN_LAPSED, MAX_LAPSED) != diff){
                return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
            }

            lapsed += diff;
        }

        return lapsed;
    }

    static auto port_stamp(model::Packet& pkt) noexcept -> exception_t{

        if (pkt.port_stamp_sz >= pkt.port_utc_stamps.size()){
            return dg::network_exception::RESOURCE_EXHAUSTION;
        }

        pkt.port_utc_stamps[pkt.port_stamp_sz++] = stdx::timestamp_conversion_wrap(stdx::utc_timestamp());
        return dg::network_exception::SUCCESS;
    }
}

namespace dg::network_kernel_mailbox_impl1::packet_controller{
        
    struct WAPIntervalData{
        size_t outbound_sz;
        dg::vector<std::chrono::nanoseconds> rtt_vec;
        std::chrono::nanoseconds ideal_lapse;
        std::chrono::nanoseconds last;
    };

    struct WAPStatisticValue{
        size_t outbound_sz;
        size_t inbound_sz;
    };

    struct WAPStatisticModel{
        dg::unordered_unstable_map<uint32_t, dg::unordered_unstable_map<uint32_t, WAPStatisticValue>> model;
    };

    class WAPScheduler: public virtual SchedulerInterface, public virtual UpdatableInterface{

        private:

            dg::unordered_unstable_map<Address, WAPIntervalData> interval_data_map;
            dg::unordered_unstable_map<Address, WAPStatisticModel> statistic_data_map;
            uint32_t rtt_discretization_sz;
            std::chrono::nanoseconds rtt_minbound;
            std::chrono::nanoseconds rtt_maxbound;
            uint32_t schedule_discretization_sz;
            std::chrono::nanoseconds schedule_minbound;
            std::chrono::nanoseconds schedule_maxbound;
            std::chrono::nanoseconds max_schedule_time;
            std::chrono::nanoseconds last_updated_time;
            std::chrono::nanoseconds min_update_interval;
            std::chrono::nanoseconds last_reset_time;
            std::chrono::nanoseconds min_reset_interval;
            size_t interval_data_map_capacity;
            size_t rtt_vec_capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            WAPScheduler(dg::unordered_unstable_map<Address, WAPIntervalData> interval_data_map,
                         dg::unordered_unstable_map<Address, WAPStatisticModel> statistic_data_map,
                         uint32_t rtt_discretization_sz,
                         std::chrono::nanoseconds rtt_minbound,
                         std::chrono::nanoseconds rtt_maxbound,
                         uint32_t schedule_discretization_sz,
                         std::chrono::nanoseconds schedule_minbound,
                         std::chrono::nanoseconds schedule_maxbound,
                         std::chrono::nanoseconds max_schedule_time,
                         std::chrono::nanoseconds last_updated_time,
                         std::chrono::nanoseconds min_update_interval,
                         std::chrono::nanoseconds last_reset_time,
                         std::chrono::nanoseconds min_reset_interval,
                         size_t interval_data_map_capacity,
                         size_t rtt_vec_capacity,
                         std::unique_ptr<std::mutex> mtx) noexcept: interval_data_map(std::move(interval_data_map)),
                                                                    statistic_data_map(std::move(statistic_data_map)),
                                                                    rtt_discretization_sz(rtt_discretization_sz),
                                                                    rtt_minbound(rtt_minbound),
                                                                    rtt_maxbound(rtt_maxbound),
                                                                    schedule_discretization_sz(schedule_discretization_sz),
                                                                    schedule_minbound(schedule_minbound),
                                                                    schedule_maxbound(schedule_maxbound),
                                                                    max_schedule_time(max_schedule_time),
                                                                    last_updated_time(last_updated_time),
                                                                    min_update_interval(min_update_interval),
                                                                    last_reset_time(last_reset_time),
                                                                    min_reset_interval(min_reset_interval),
                                                                    interval_data_map_capacity(interval_data_map_capacity),
                                                                    rtt_vec_capacity(rtt_vec_capacity),
                                                                    mtx(std::move(mtx)){}

            auto schedule(Address addr) noexcept -> std::chrono::nanoseconds{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->interval_data_map.find(addr);

                if (map_ptr == this->interval_data_map.end()){
                    return stdx::utc_timestamp();
                }
                
                std::chrono::nanoseconds now        = stdx::utc_timestamp();
                std::chrono::nanoseconds worst      = now + this->max_schedule_time;
                std::chrono::nanoseconds tentative  = map_ptr->second.last + map_ptr->second.ideal_lapse;
                std::chrono::nanoseconds chosen     = std::clamp(tentative, now, worst);
                map_ptr->second.last                = chosen;
                map_ptr->second.outbound_sz         += 1;

                return chosen;
            }

            auto feedback(Address addr, std::chrono::nanoseconds lapsed) noexcept -> exception_t{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->interval_data_map.find(addr);

                if (map_ptr == this->interval_data_map.end()){
                    if (this->interval_data_map.size() == this->interval_data_map_capacity){
                        return dg::network_exception::RESOURCE_EXHAUSTION;
                    }

                    if (this->rtt_vec_capacity == 0u){
                        return dg::network_exception::RESOURCE_EXHAUSTION;
                    }

                    auto [emplace_ptr, status] = this->interval_data_map.emplace(std::make_pair(addr, WAPIntervalData{}));
                    dg::network_exception_handler::dg_assert(status);
                    auto [_emplace_ptr, _status] = this->statistic_data_map.emplace(std::make_pair(addr, WAPStatisticModel{}));
                    dg::network_exception_handler::dg_assert(_status);
                    map_ptr = emplace_ptr; 
                }
                
                if (map_ptr->second.rtt_vec.size() == this->rtt_vec_capacity){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                map_ptr->second.rtt_vec.push_back(lapsed);
                return dg::network_exception::SUCCESS;
            }

            void update() noexcept{

                auto lck_grd                            = stdx::lock_guard(*this->mtx);
                std::chrono::nanoseconds now            = stdx::utc_timestamp();
                std::chrono::nanoseconds reset_lapsed   = now - this->last_reset_time;
                std::chrono::nanoseconds update_lapsed  = now - this->last_updated_time;

                if (reset_lapsed > this->min_reset_interval){
                    this->interval_data_map.clear();
                    this->statistic_data_map.clear();
                    this->last_reset_time = now;
                    return;
                } 

                if (update_lapsed < this->min_update_interval){
                    return;
                }

                for (auto& interval_data_pair: this->interval_data_map){
                    std::expected<std::pair<uint32_t, uint32_t>, exception_t> key = this->make_statistic_lookup_key(interval_data_pair.second);
                    
                    if (!key.has_value()){
                        continue;
                    }

                    auto [rtt_idx, sched_idx]               = key.value(); 
                    auto& statistical_bucket                = this->statistic_data_map[interval_data_pair.first].model[rtt_idx][sched_idx];
                    statistical_bucket.outbound_sz          += interval_data_pair.second.outbound_sz;
                    statistical_bucket.inbound_sz           += interval_data_pair.second.rtt_vec.size();
                    interval_data_pair.second.outbound_sz   = 0u;
                    interval_data_pair.second.ideal_lapse   = this->get_ideal_or_random_lapsed(this->statistic_data_map[interval_data_pair.first].model[rtt_idx]);
                    interval_data_pair.second.last          = stdx::utc_timestamp();
                    interval_data_pair.second.rtt_vec.clear();
                }

                this->last_updated_time = now;
            }
        
        private:

            auto make_statistic_lookup_key(const WAPIntervalData& interval_data) const noexcept -> std::expected<std::pair<uint32_t, uint32_t>, exception_t>{

                if (interval_data.rtt_vec.empty()){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                std::chrono::nanoseconds last_lapsed                = std::clamp(interval_data.ideal_lapse, this->schedule_minbound, this->schedule_maxbound);
                std::chrono::nanoseconds discrete_sched_interval    = (this->schedule_maxbound - this->schedule_minbound) / this->schedule_discretization_sz;
                size_t lapsed_idx                                   = static_cast<size_t>(stdx::timestamp_conversion_wrap(last_lapsed - this->schedule_minbound)) / static_cast<size_t>(stdx::timestamp_conversion_wrap(discrete_sched_interval));
                std::chrono::nanoseconds rtt_avg                    = std::accumulate(interval_data.rtt_vec.begin(), interval_data.rtt_vec.end(), std::chrono::nanoseconds(0u), std::plus<>{}) / interval_data.rtt_vec.size();
                std::chrono::nanoseconds rtt                        = std::clamp(rtt_avg, this->rtt_minbound, this->rtt_maxbound);
                std::chrono::nanoseconds discrete_rtt_interval      = (this->rtt_maxbound - this->rtt_minbound) / this->rtt_discretization_sz;
                size_t rtt_idx                                      = static_cast<size_t>(stdx::timestamp_conversion_wrap(rtt - this->rtt_minbound)) / static_cast<size_t>(stdx::timestamp_conversion_wrap(discrete_rtt_interval)); 

                return std::make_pair(std::min(static_cast<size_t>(this->rtt_discretization_sz - 1), rtt_idx), std::min(static_cast<size_t>(this->schedule_discretization_sz - 1), lapsed_idx));
            }

            auto get_ideal_or_random_lapsed(const dg::unordered_unstable_map<uint32_t, WAPStatisticValue>& sched_discrete_idx_wapstat_map) noexcept -> std::chrono::nanoseconds{

                constexpr size_t DICE_SZ                            = 32u;
                size_t dice_value                                   = dg::network_randomizer::randomize_xrange(std::integral_constant<size_t, DICE_SZ>{});
                std::chrono::nanoseconds discrete_sched_interval    = (this->schedule_maxbound - this->schedule_minbound) / this->schedule_discretization_sz;

                if (dice_value == 0u){
                    size_t sched_discrete_idx = dg::network_randomizer::randomize_int<uint32_t>() % this->schedule_discretization_sz;
                    return std::min(static_cast<std::chrono::nanoseconds>(this->schedule_minbound + (discrete_sched_interval * sched_discrete_idx)), this->schedule_maxbound);
                }

                double max_cursor           = 0u;
                size_t sched_discrete_idx   = this->schedule_discretization_sz - 1;

                for (const auto& map_pair: sched_discrete_idx_wapstat_map){
                    if (map_pair.second.outbound_sz == 0u){
                        continue;
                    }

                    double success_rate = static_cast<double>(map_pair.second.inbound_sz) / map_pair.second.outbound_sz;

                    if (max_cursor < success_rate){
                        max_cursor          = success_rate;
                        sched_discrete_idx  = map_pair.first;
                    }
                }

                return std::min(static_cast<std::chrono::nanoseconds>(this->schedule_minbound + (discrete_sched_interval * sched_discrete_idx)), this->schedule_maxbound);
            }
    };

    class ASAPScheduler: public virtual SchedulerInterface{

        public:

            auto schedule(Address) noexcept -> std::chrono::nanoseconds{

                return stdx::utc_timestamp();
            }

            auto feedback(Address, std::chrono::nanoseconds) noexcept -> exception_t{

                return dg::network_exception::SUCCESS;
            }
    };

    class IDGenerator: public virtual IDGeneratorInterface{
        
        private:

            local_packet_id_t last_pkt_id;
            factory_id_t factory_id;
            std::unique_ptr<std::mutex> mtx;

        public:

            IDGenerator(local_packet_id_t last_pkt_id,
                        factory_id_t factory_id,
                        std::unique_ptr<std::mutex> mtx) noexcept: last_pkt_id(std::move(last_pkt_id)),
                                                                   factory_id(std::move(factory_id)),
                                                                   mtx(std::move(mtx)){}

            auto get() noexcept -> GlobalPacketIdentifier{

                auto lck_grd        = stdx::lock_guard(*this->mtx);
                auto rs             = model::GlobalPacketIdentifier{this->last_pkt_id, this->factory_id};
                this->last_pkt_id   += 1;

                return rs;
            }
    };
    
    class PacketGenerator: public virtual PacketGeneratorInterface{

        private:

            std::unique_ptr<IDGeneratorInterface> id_gen;
            Address host_addr; 

        public:

            PacketGenerator(std::unique_ptr<IDGeneratorInterface> id_gen,
                            Address host_addr) noexcept: id_gen(std::move(id_gen)),
                                                         host_addr(std::move(host_addr)){}

            auto get(Address to_addr, dg::string content) noexcept -> Packet{

                model::Packet pkt           = {};
                pkt.fr_addr                 = host_addr;
                pkt.to_addr                 = std::move(to_addr);
                pkt.id                      = this->id_gen->get();
                pkt.content                 = std::move(content);
                pkt.retransmission_count    = 0u;
                pkt.priority                = 0u;
                pkt.kind                    = constants::request;
                pkt.port_utc_stamps         = {};
                pkt.port_stamp_sz           = 0u;

                return pkt;
            }
    };

    class RetransmissionManager: public virtual RetransmissionManagerInterface{

        private:

            dg::deque<std::pair<std::chrono::nanoseconds, Packet>> pkt_deque;
            std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> acked_id_hashset;
            std::chrono::nanoseconds transmission_delay_time;
            size_t max_retransmission;
            size_t capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            RetransmissionManager(dg::deque<std::pair<std::chrono::nanoseconds, Packet>> pkt_deque,
                                  std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> acked_id_hashset,
                                  std::chrono::nanoseconds transmission_delay_time,
                                  size_t max_retransmission,
                                  size_t capacity,
                                  std::unique_ptr<std::mutex> mtx) noexcept: pkt_deque(std::move(pkt_deque)),
                                                                             acked_id_hashset(std::move(acked_id_hashset)),
                                                                             transmission_delay_time(transmission_delay_time),
                                                                             max_retransmission(max_retransmission),
                                                                             capacity(capacity),
                                                                             mtx(std::move(mtx)){}

            auto add_retriable(Packet pkt) noexcept -> exception_t{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (this->pkt_deque.size() == this->capacity){
                    return dg::network_exception::BAD_RETRANSMISSION;
                }

                if (pkt.retransmission_count >= this->max_retransmission){
                    return dg::network_exception::BAD_RETRANSMISSION;
                }

                pkt.retransmission_count += 1;
                pkt.priority += 1;
                this->pkt_deque.push_back(std::make_pair(stdx::utc_timestamp(), std::move(pkt))); 

                return dg::network_exception::SUCCESS;
            }

            void ack(global_packet_id_t pkt_id) noexcept{
                
                auto lck_grd = stdx::lock_guard(*this->mtx);
                this->acked_id_hashset->insert(std::move(pkt_id));
            }

            auto get_retriables() noexcept -> dg::vector<Packet>{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto ts         = stdx::utc_timestamp() - this->transmission_delay_time;
                auto lb_key     = std::make_pair(ts, Packet{});
                auto last       = std::lower_bound(this->pkt_deque.begin(), this->pkt_deque.end(), lb_key, [](const auto& lhs, const auto& rhs){return lhs.first < rhs.first;});
                auto rs         = dg::vector<Packet>(); 
                
                for (auto it = this->pkt_deque.begin(); it != last; ++it){
                    if (!this->acked_id_hashset->contains(it->second.id)){
                        rs.push_back(std::move(it->second)); 
                    }
                }

                this->pkt_deque.erase(this->pkt_deque.begin(), last);
                return rs;
            }
    };

    class PrioritizedPacketContainer: public virtual PacketContainerInterface{

        private:
            
            dg::vector<Packet> packet_vec;
            std::unique_ptr<std::mutex> mtx;

        public:

            PrioritizedPacketContainer(dg::vector<Packet> packet_vec,
                                 std::unique_ptr<std::mutex> mtx) noexcept: packet_vec(std::move(packet_vec)),
                                                                            mtx(std::move(mtx)){}

            void push(Packet pkt) noexcept{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto less       = [](const Packet& lhs, const Packet& rhs){return lhs.priority < rhs.priority;};
                this->packet_vec.push_back(std::move(pkt)); 
                std::push_heap(this->packet_vec.begin(), this->packet_vec.end(), less);
            }     

            auto pop() noexcept -> std::optional<Packet>{
                
                auto lck_grd    = stdx::lock_guard(*this->mtx);

                if (this->packet_vec.empty()){
                    return std::nullopt;
                }
                
                auto less = [](const Packet& lhs, const Packet& rhs){return lhs.priority < rhs.priority;};
                std::pop_heap(this->packet_vec.begin(), this->packet_vec.end(), less);
                auto rs = std::move(this->packet_vec.back());
                this->packet_vec.pop_back();

                return rs;
            }   
    };

    class ScheduledPacketContainer: public virtual PacketContainerInterface{

        private:

            dg::vector<ScheduledPacket> packet_vec;
            std::shared_ptr<SchedulerInterface> scheduler;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            ScheduledPacketContainer(dg::vector<ScheduledPacket> packet_vec, 
                                     std::shared_ptr<SchedulerInterface> scheduler,
                                     std::unique_ptr<std::mutex> mtx) noexcept: packet_vec(std::move(packet_vec)),
                                                                                scheduler(std::move(scheduler)),
                                                                                mtx(std::move(mtx)){}
            
            void push(Packet pkt) noexcept{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto appender   = ScheduledPacket{std::move(pkt), this->scheduler->schedule(pkt.to_addr)};
                auto greater    = [](const auto& lhs, const auto& rhs){return lhs.sched_time > rhs.sched_time;};
                this->packet_vec.push_back(std::move(appender));
                std::push_heap(this->packet_vec.begin(), this->packet_vec.end(), greater);
            }

            auto pop() noexcept -> std::optional<Packet>{

                auto lck_grd    = stdx::lock_guard(*this->mtx);

                if (this->packet_vec.empty()){
                    return std::nullopt;
                }

                if (this->packet_vec.front().sched_time > stdx::utc_timestamp()){
                    return std::nullopt;
                }

                auto greater    = [](const auto& lhs, const auto& rhs){return lhs.sched_time > rhs.sched_time;};
                std::pop_heap(this->packet_vec.begin(), this->packet_vec.end(), greater);
                auto rs = std::move(this->packet_vec.back().pkt);
                this->packet_vec.pop_back();

                return rs;
            }
    };

    class OutboundPacketContainer: public virtual PacketContainerInterface{

        private:

            std::unique_ptr<PacketContainerInterface> ack_container;
            std::unique_ptr<PacketContainerInterface> pkt_container;
            std::unique_ptr<std::mutex> mtx; 

        public:

            OutboundPacketContainer(std::unique_ptr<PacketContainerInterface> ack_container,
                                 std::unique_ptr<PacketContainerInterface> pkt_container,
                                 std::unique_ptr<std::mutex> mtx) noexcept: ack_container(std::move(ack_container)),
                                                                            pkt_container(std::move(pkt_container)),
                                                                            mtx(std::move(mtx)){}

            void push(Packet pkt) noexcept{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (pkt.kind == constants::rts_ack){
                    this->ack_container->push(std::move(pkt));
                    return;
                }
                
                this->pkt_container->push(std::move(pkt));
            }

            auto pop() noexcept -> std::optional<Packet>{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (auto rs = this->ack_container->pop(); rs){
                    return rs;
                }

                return this->pkt_container->pop();
            }
    };

    class InBoundIDController: public virtual InBoundIDControllerInterface{
        
        private:

            std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> id_hashset;
            std::unique_ptr<std::mutex> mtx;

        public:

            InBoundIDController(std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> id_hashset,
                              std::unique_ptr<std::mutex> mtx) noexcept: id_hashset(std::move(id_hashset)),
                                                                         mtx(std::move(mtx)){}
            
            auto thru(global_packet_id_t packet_id) noexcept -> std::expected<bool, exception_t>{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (this->id_hashset->contains(packet_id)){
                    return false;
                }

                this->id_hashset->insert(packet_id);
                return true;
            }
    };

    class ExhaustionControlledPacketContainer: public virtual PacketContainerInterface{

        private:

            std::unique_ptr<PacketContainerInterface> base;
            size_t size;
            size_t capacity;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledPacketContainer(std::unique_ptr<PacketContainerInterface> base,
                                                size_t size,
                                                size_t capacity,
                                                std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                           size(size),
                                                                                           capacity(capacity),
                                                                                           executor(std::move(executor)),
                                                                                           mtx(std::move(mtx)){}

            void push(Packet pkt) noexcept{

                auto lambda = [&]() noexcept{
                    return this->internal_push(pkt);
                };
                auto exe = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(lambda)>(std::move(lambda));
                this->executor->exec(exe);
            } 

            auto pop() noexcept -> std::optional<Packet>{

                return this->internal_pop();
            }

        private:

            auto internal_push(Packet& pkt) noexcept -> bool{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (this->size == this->capacity){
                    return false;
                }

                this->base->push(std::move(pkt));
                this->size += 1;

                return true;
            }

            auto internal_pop() noexcept -> std::optional<Packet>{

                auto lck_grd = stdx::lock_guard(*this->mtx);
                auto rs = this->base->pop();

                if (rs.has_value()){
                    this->size -= 1;
                }

                return rs;
            }
    };

    class InBoundTrafficController: public virtual InBoundTrafficControllerInterface, public virtual UpdatableInterface{

        private:

            dg::unordered_unstable_map<Address, size_t> address_counter_map;
            size_t address_cap;
            size_t global_cap;
            size_t map_cap;
            size_t global_counter;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            InBoundTrafficController(dg::unordered_unstable_map<Address, size_t> address_counter_map,
                                     size_t address_cap,
                                     size_t global_cap,
                                     size_t map_cap,
                                     size_t global_counter,
                                     std::unique_ptr<std::mutex> mtx) noexcept: address_counter_map(std::move(address_counter_map)),
                                                                                address_cap(address_cap),
                                                                                global_cap(global_cap),
                                                                                map_cap(map_cap),
                                                                                global_counter(global_counter),
                                                                                mtx(std::move(mtx)){}

            auto thru(Address addr) noexcept -> std::expected<bool, exception_t>{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (this->global_counter == this->global_cap){
                    return false;
                }

                auto map_ptr = this->address_counter_map.find(addr);

                if (map_ptr == this->address_counter_map.end()){
                    if (this->address_counter_map.size() == this->map_cap){
                        return false;
                    }

                    if (this->address_cap == 0u){
                        return false;
                    }

                    auto [emplace_ptr, status] = this->address_counter_map.emplace(std::make_pair(addr, 0u));
                    dg::network_exception_handler::dg_assert(status);
                    map_ptr = emplace_ptr;
                }

                if (map_ptr->second == this->address_cap){
                    return false;
                }

                map_ptr->second += 1;
                this->global_counter += 1;

                return true;
            }

            void update() noexcept{

                auto lck_grd = stdx::lock_guard(*this->mtx);
                this->address_counter_map.clear();
                this->global_counter = 0u;
            }
    };

    struct ComponentFactory{

        static auto get_wap_scheduler(std::chrono::nanoseconds rtt_minbound, std::chrono::nanoseconds rtt_maxbound, size_t rtt_discretization_sz,
                                      std::chrono::nanoseconds sched_minbound, std::chrono::nanoseconds sched_maxbound, size_t sched_discretization_sz,
                                      std::chrono::nanoseconds max_sched_time, std::chrono::nanoseconds update_interval, std::chrono::nanoseconds reset_interval,
                                      size_t map_capacity, size_t rtt_vec_capacity) -> std::unique_ptr<WAPScheduler>{
            
            using namespace std::chrono_literals;

            const std::chrono::nanoseconds MIN_RTT_MINBOUND     = std::chrono::duration_cast<std::chrono::nanoseconds>(1us);
            const std::chrono::nanoseconds MAX_RTT_MINBOUND     = std::chrono::duration_cast<std::chrono::nanoseconds>(30s);
            const std::chrono::nanoseconds MIN_RTT_MAXBOUND     = std::chrono::duration_cast<std::chrono::nanoseconds>(1us);
            const std::chrono::nanoseconds MAX_RTT_MAXBOUND     = std::chrono::duration_cast<std::chrono::nanoseconds>(30s);
            const size_t MIN_RTT_DISCRETE_SZ                    = 1u;
            const size_t MAX_RTT_DISCRETE_SZ                    = size_t{1} << 10;
            const std::chrono::nanoseconds MIN_SCHED_MINBOUND   = std::chrono::duration_cast<std::chrono::nanoseconds>(1us);
            const std::chrono::nanoseconds MAX_SCHED_MINBOUND   = std::chrono::duration_cast<std::chrono::nanoseconds>(30s);
            const std::chrono::nanoseconds MIN_SCHED_MAXBOUND   = std::chrono::duration_cast<std::chrono::nanoseconds>(1us);
            const std::chrono::nanoseconds MAX_SCHED_MAXBOUND   = std::chrono::duration_cast<std::chrono::nanoseconds>(30s);
            const size_t MIN_SCHED_DISCRETE_SZ                  = 1u;
            const size_t MAX_SCHED_DISCRETE_SZ                  = size_t{1} << 10;
            const std::chrono::nanoseconds MIN_MAX_SCHED_TIME   = std::chrono::duration_cast<std::chrono::nanoseconds>(1us);
            const std::chrono::nanoseconds MAX_MAX_SCHED_TIME   = std::chrono::duration_cast<std::chrono::nanoseconds>(30s);
            const std::chrono::nanoseconds MIN_UPDATE_INTERVAL  = std::chrono::duration_cast<std::chrono::nanoseconds>(1us);
            const std::chrono::nanoseconds MAX_UPDATE_INTERVAL  = std::chrono::duration_cast<std::chrono::nanoseconds>(100s);
            const std::chrono::nanoseconds MIN_RESET_INTERVAL   = std::chrono::duration_cast<std::chrono::nanoseconds>(1s);
            const std::chrono::nanoseconds MAX_RESET_INTERVAL   = std::chrono::duration_cast<std::chrono::nanoseconds>(3600s);
            const size_t MIN_MAP_CAPACITY                       = 0u;
            const size_t MAX_MAP_CAPACITY                       = size_t{1} << 25;
            const size_t MIN_RTT_VEC_CAPACITY                   = 0u;
            const size_t MAX_RTT_VEC_CAPACITY                   = size_t{1} << 8;

            if (std::clamp(rtt_minbound, MIN_RTT_MINBOUND, MAX_RTT_MINBOUND) != rtt_minbound){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(rtt_maxbound, MIN_RTT_MAXBOUND, MAX_RTT_MAXBOUND) != rtt_maxbound){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(rtt_discretization_sz, MIN_RTT_DISCRETE_SZ, MAX_RTT_DISCRETE_SZ) != rtt_discretization_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (rtt_maxbound < rtt_minbound){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::chrono::nanoseconds discrete_interval_size = (rtt_maxbound - rtt_minbound) / rtt_discretization_sz;
            size_t udiscrete_interval_size = stdx::timestamp_conversion_wrap(discrete_interval_size); 

            if (udiscrete_interval_size == 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(sched_minbound, MIN_SCHED_MINBOUND, MAX_SCHED_MINBOUND) != sched_minbound){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(sched_maxbound, MIN_SCHED_MAXBOUND, MAX_SCHED_MAXBOUND) != sched_maxbound){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(sched_discretization_sz, MIN_SCHED_DISCRETE_SZ, MAX_SCHED_DISCRETE_SZ) != sched_discretization_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (sched_maxbound < sched_minbound){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::chrono::nanoseconds discrete_sched_interval_size = (sched_maxbound - sched_minbound) / sched_discretization_sz;
            size_t udiscrete_sched_interval_size = stdx::timestamp_conversion_wrap(discrete_sched_interval_size);

            if (udiscrete_sched_interval_size == 0u){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(max_sched_time, MIN_MAX_SCHED_TIME, MAX_MAX_SCHED_TIME) != max_sched_time){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(update_interval, MIN_UPDATE_INTERVAL, MAX_UPDATE_INTERVAL) != update_interval){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(reset_interval, MIN_RESET_INTERVAL, MAX_RESET_INTERVAL) != reset_interval){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(map_capacity, MIN_MAP_CAPACITY, MAX_MAP_CAPACITY) != map_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(rtt_vec_capacity, MIN_RTT_VEC_CAPACITY, MAX_RTT_VEC_CAPACITY) != rtt_vec_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto interval_data_map  = dg::unordered_unstable_map<Address, WAPIntervalData>{};
            auto statistic_data_map = dg::unordered_unstable_map<Address, WAPStatisticModel>{};
            auto mtx                = std::make_unique<std::mutex>();
            interval_data_map.reserve(map_capacity);
            statistic_data_map.reserve(map_capacity);

            return std::make_unique<WAPScheduler>(std::move(interval_data_map), std::move(statistic_data_map),
                                                  rtt_discretization_sz, rtt_minbound, rtt_maxbound,
                                                  sched_discretization_sz, sched_minbound, sched_maxbound, max_sched_time, 
                                                  stdx::utc_timestamp(), update_interval,
                                                  stdx::utc_timestamp(), reset_interval,
                                                  map_capacity, rtt_vec_capacity, std::move(mtx));
        }

        static auto get_asap_scheduler() -> std::unique_ptr<SchedulerInterface>{

            return std::make_unique<ASAPScheduler>();
        }

        static auto get_id_generator(factory_id_t factory_id) -> std::unique_ptr<IDGeneratorInterface>{
            
            return std::make_unique<IDGenerator>(dg::network_randomizer::randomize_int<local_packet_id_t>(), 
                                                 factory_id, 
                                                 std::make_unique<std::mutex>());
        }

        static auto get_packet_gen(factory_id_t factory_id, Address factory_addr) -> std::unique_ptr<PacketGeneratorInterface>{

            return std::make_unique<PacketGenerator>(get_id_generator(factory_id), factory_addr);
        }
     
        static auto get_retransmission_manager(std::chrono::nanoseconds delay, size_t max_retransmission, 
                                               size_t idhashset_cap, size_t retransmission_cap) -> std::unique_ptr<RetransmissionManagerInterface>{

            using namespace std::chrono_literals; 

            const std::chrono::nanoseconds MIN_DELAY    = std::chrono::duration_cast<std::chrono::nanoseconds>(1us);
            const std::chrono::nanoseconds MAX_DELAY    = std::chrono::duration_cast<std::chrono::nanoseconds>(60s);
            const size_t MIN_MAX_RETRANSMISSION         = 0u;
            const size_t MAX_MAX_RETRANSMISSION         = 32u;
            const size_t MIN_RETRANSMISSION_CAP         = 0u;
            const size_t MAX_RETRANSMISSION_CAP         = size_t{1} << 20;

            if (std::clamp(delay, MIN_DELAY, MAX_DELAY) != delay){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }
            
            if (std::clamp(max_retransmission, MIN_MAX_RETRANSMISSION, MAX_MAX_RETRANSMISSION) != max_retransmission){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(retransmission_cap, MIN_RETRANSMISSION_CAP, MAX_RETRANSMISSION_CAP) != retransmission_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<RetransmissionManager>(dg::deque<std::pair<std::chrono::nanoseconds, Packet>>{},
                                                           data_structure::Factory::get_temporal_unordered_set<global_packet_id_t>(idhashset_cap),
                                                           delay, 
                                                           max_retransmission,
                                                           retransmission_cap,
                                                           std::make_unique<std::mutex>());
        }

        static auto get_prioritized_packet_container() -> std::unique_ptr<PacketContainerInterface>{

            return std::make_unique<PrioritizedPacketContainer>(dg::vector<Packet>{}, 
                                                                std::make_unique<std::mutex>());
        }

        static auto get_scheduled_packet_container(std::shared_ptr<SchedulerInterface> scheduler) -> std::unique_ptr<PacketContainerInterface>{

            if (scheduler == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ScheduledPacketContainer>(dg::vector<ScheduledPacket>{}, 
                                                              scheduler, 
                                                              std::make_unique<std::mutex>());
        }

        static auto get_inbound_id_controller(size_t idhashset_cap) -> std::unique_ptr<InBoundIDControllerInterface>{

            return std::make_unique<InBoundIDController>(data_structure::Factory::get_temporal_unordered_set<global_packet_id_t>(idhashset_cap), 
                                                         std::make_unique<std::mutex>());
        }

        static auto get_inbound_traffic_controller(size_t addr_capacity, size_t global_capacity, size_t max_address) -> std::unique_ptr<InBoundTrafficController>{

            const size_t MIN_ADDR_CAPACITY      = 0u;
            const size_t MAX_ADDR_CAPACITY      = size_t{1} << 20;
            const size_t MIN_GLOBAL_CAPACITY    = 0u;
            const size_t MAX_GLOBAL_CAPACITY    = size_t{1} << 25;
            const size_t MIN_MAX_ADDRESS        = 0u;
            const size_t MAX_MAX_ADDRESS        = size_t{1} << 20;

            if (std::clamp(addr_capacity, MIN_ADDR_CAPACITY, MAX_ADDR_CAPACITY) != addr_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(global_capacity, MIN_GLOBAL_CAPACITY, MAX_GLOBAL_CAPACITY) != global_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(max_address, MIN_MAX_ADDRESS, MAX_MAX_ADDRESS) != max_address){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto address_counter_map    = dg::unordered_unstable_map<Address, size_t>{};
            auto mtx                    = std::make_unique<std::mutex>(); 
            size_t global_counter       = 0u; 

            address_counter_map.reserve(max_address);
            
            return std::make_unique<InBoundTrafficController>(std::move(address_counter_map), addr_capacity, global_capacity, max_address, global_counter, std::move(mtx));
        } 

        static auto get_outbound_packet_container(std::shared_ptr<SchedulerInterface> scheduler) -> std::unique_ptr<PacketContainerInterface>{

            return std::make_unique<OutboundPacketContainer>(get_prioritized_packet_container(), 
                                                             get_scheduled_packet_container(scheduler), 
                                                             std::make_unique<std::mutex>());
        }

        static auto get_exhaustion_controlled_packet_container(std::unique_ptr<PacketContainerInterface> base, 
                                                               std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> retry_device, 
                                                               size_t capacity) -> std::unique_ptr<PacketContainerInterface>{

            const size_t MIN_CAP  = size_t{1};
            const size_t MAX_CAP  = size_t{1} << 20; 
    
            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (retry_device == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(capacity, MIN_CAP, MAX_CAP) != capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }
            
            return std::make_unique<ExhaustionControlledPacketContainer>(std::move(base), size_t{0u}, capacity, std::move(retry_device), std::make_unique<std::mutex>());
        }
    };
}

namespace dg::network_kernel_mailbox_impl1::worker{

    using namespace dg::network_kernel_mailbox_impl1::model; 
    
    class OutBoundWorker: public virtual dg::network_concurrency::WorkerInterface{
        
        private:

            std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container;
            std::shared_ptr<model::SocketHandle> socket;

        public:

            OutBoundWorker(std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container,
                           std::shared_ptr<model::SocketHandle> socket) noexcept: outbound_packet_container(std::move(outbound_packet_container)),
                                                                                  socket(std::move(socket)){}

            bool run_one_epoch() noexcept{

                std::optional<Packet> cur = this->outbound_packet_container->pop();

                if (!cur.has_value()){
                    return false;
                } 

                exception_t stamp_err = packet_service::port_stamp(cur.value());

                if (dg::network_exception::is_failed(stamp_err)){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(stamp_err));
                }

                dg::string bstream      = utility::serialize_packet(std::move(cur.value()));
                exception_t sock_err    = socket_service::send_noblock(*this->socket, cur->to_addr, bstream.data(), bstream.size());
                
                if (dg::network_exception::is_failed(sock_err)){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(sock_err));
                    return false;
                }

                return true;
            }
    };

    class RetransmissionWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager;
            std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container;

        public:

            RetransmissionWorker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                                 std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container) noexcept: retransmission_manager(std::move(retransmission_manager)),
                                                                                                                                   outbound_packet_container(std::move(outbound_packet_container)){}

            bool run_one_epoch() noexcept{
                
                dg::vector<Packet> packets = this->retransmission_manager->get_retriables();

                if (packets.empty()){
                    return false;
                }

                for (Packet& packet: packets){
                    this->outbound_packet_container->push(packet);
                    exception_t err = this->retransmission_manager->add_retriable(std::move(packet));

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    }
                }

                return true;
            }
    };

    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager;
            std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container;
            std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container;
            std::shared_ptr<packet_controller::InBoundIDControllerInterface> inbound_id_controller;
            std::shared_ptr<packet_controller::InBoundTrafficControllerInterface> inbound_traffic_controller;
            std::shared_ptr<packet_controller::SchedulerInterface> scheduler;
            std::shared_ptr<model::SocketHandle> socket;
        
        public:

            InBoundWorker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                          std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container,
                          std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container,
                          std::shared_ptr<packet_controller::InBoundIDControllerInterface> inbound_id_controller,
                          std::shared_ptr<packet_controller::InBoundTrafficControllerInterface> inbound_traffic_controller,
                          std::shared_ptr<packet_controller::SchedulerInterface> scheduler,
                          std::shared_ptr<model::SocketHandle> socket) noexcept: retransmission_manager(std::move(retransmission_manager)),
                                                                                 ob_packet_container(std::move(ob_packet_container)),
                                                                                 ib_packet_container(std::move(ib_packet_container)),
                                                                                 inbound_id_controller(std::move(inbound_id_controller)),
                                                                                 inbound_traffic_controller(std::move(inbound_traffic_controller)),
                                                                                 scheduler(std::move(scheduler)),
                                                                                 socket(std::move(socket)){}
            
            bool run_one_epoch() noexcept{
                
                model::Packet pkt   = {};
                size_t sz           = {};
                auto bstream        = dg::string(constants::MAXIMUM_MSG_SIZE, ' '); //this is an optimizable - custom string implementation that only does std::malloc() - instead of calloc
                exception_t err     = socket_service::recv_block(*this->socket, bstream.data(), sz, constants::MAXIMUM_MSG_SIZE);

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    return false;
                }

                bstream.resize(sz);
                std::expected<Packet, exception_t> epkt = utility::deserialize_packet(std::move(bstream)); 
                
                if (!epkt.has_value()){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(epkt.error()));
                    return true;
                }
                
                pkt = std::move(epkt.value());
                exception_t stamp_err = packet_service::port_stamp(pkt);

                if (dg::network_exception::is_failed(stamp_err)){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(stamp_err));
                }

                std::expected<bool, exception_t> is_thru_traffic = this->inbound_traffic_controller->thru(pkt.fr_addr);
                
                if (!is_thru_traffic.has_value()){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(is_thru_traffic.error()));
                    return true;
                }

                if (!is_thru_traffic.value()){
                    return true;
                }

                std::expected<bool, exception_t> is_thru_id = this->inbound_id_controller->thru(pkt.id); 

                if (!is_thru_id.has_value()){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(is_thru_id.error()));
                    return true;
                }

                if (!is_thru_id.value()){   
                    if (pkt.kind == constants::rts_ack){
                        std::expected<std::chrono::nanoseconds, exception_t> transit_time = packet_service::get_transit_time(pkt);
                        
                        if (!transit_time.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(transit_time.error()));
                        } else{
                            exception_t fb_err = this->scheduler->feedback(pkt.fr_addr, transit_time.value());
                            if (dg::network_exception::is_failed(fb_err)){
                                dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(fb_err));
                            }
                        }
                    } else if (pkt.kind == constants::request){
                        auto ack_pkt = packet_service::request_to_ack(pkt);
                        this->ob_packet_container->push(std::move(ack_pkt));
                    } else{
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(dg::network_exception::INVALID_FORMAT));
                    }

                    return true;
                }

                if (pkt.kind == constants::rts_ack){
                    this->retransmission_manager->ack(pkt.id);
                    std::expected<std::chrono::nanoseconds, exception_t> transit_time = packet_service::get_transit_time(pkt);

                    if (!transit_time.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(transit_time.error()));
                    } else{
                        exception_t fb_err = this->scheduler->feedback(pkt.fr_addr, transit_time.value());
                    
                        if (dg::network_exception::is_failed(fb_err)){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(fb_err));
                        }
                    }
                } else if (pkt.kind == constants::request){
                    auto ack_pkt = packet_service::request_to_ack(pkt);
                    this->ib_packet_container->push(std::move(pkt));
                    this->ob_packet_container->push(std::move(ack_pkt));
                } else{
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(dg::network_exception::INVALID_FORMAT));
                }

                return true;
            }
    };

    class UpdateWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::UpdatableInterface> updatable;
            std::chrono::nanoseconds wait_dur;
        
        public:

            UpdateWorker(std::shared_ptr<packet_controller::UpdatableInterface> updatable,
                         std::chrono::nanoseconds wait_dur) noexcept: updatable(std::move(updatable)),
                                                                      wait_dur(std::move(wait_dur)){}

            bool run_one_epoch() noexcept{

                this->updatable->update();
                std::this_thread::sleep_for(this->wait_dur);
                return true;
            }
    };

    struct ComponentFactory{

        static auto spawn_outbound_worker(std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container,
                                          std::shared_ptr<model::SocketHandle> socket) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            if (outbound_packet_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (socket == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<OutBoundWorker>(std::move(outbound_packet_container), std::move(socket));
        }

        static auto spawn_retransmission_worker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager, 
                                                std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            if (retransmission_manager == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (outbound_packet_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<RetransmissionWorker>(std::move(retransmission_manager), std::move(outbound_packet_container));
        }

        static auto spawn_inbound_worker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager, 
                                         std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container,
                                         std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container,
                                         std::shared_ptr<packet_controller::InBoundIDControllerInterface> ib_id_controller,
                                         std::shared_ptr<packet_controller::InBoundTrafficControllerInterface> ib_traffic_controller,
                                         std::shared_ptr<packet_controller::SchedulerInterface> scheduler, 
                                         std::shared_ptr<SocketHandle> socket) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            if (retransmission_manager == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ob_packet_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ib_packet_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ib_id_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ib_traffic_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (scheduler == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (socket == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<InBoundWorker>(std::move(retransmission_manager), std::move(ob_packet_container), 
                                                   std::move(ib_packet_container), std::move(ib_id_controller),
                                                   std::move(ib_traffic_controller), std::move(scheduler), 
                                                   std::move(socket));
        }

        static auto spawn_update_worker(std::shared_ptr<packet_controller::UpdatableInterface> updatable,
                                        std::chrono::nanoseconds traffic_dur) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            using namespace std::chrono_literals; 

            const std::chrono::nanoseconds MIN_TRAFFIC_DUR  = std::chrono::duration_cast<std::chrono::nanoseconds>(1us); 
            const std::chrono::nanoseconds MAX_TRAFFIC_DUR  = std::chrono::duration_cast<std::chrono::nanoseconds>(60s);

            if (updatable == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(traffic_dur, MIN_TRAFFIC_DUR, MAX_TRAFFIC_DUR) != traffic_dur){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<UpdateWorker>(std::move(updatable), traffic_dur);
        }
    };
}

namespace dg::network_kernel_mailbox_impl1::core{

    class RetransmittableMailBoxController: public virtual MailboxInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen;
            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager;
            std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container;
            std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container;

        public:

            RetransmittableMailBoxController(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec, 
                                             std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen,
                                             std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                                             std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container,
                                             std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                                                                                         packet_gen(std::move(packet_gen)),
                                                                                                                                         retransmission_manager(std::move(retransmission_manager)),
                                                                                                                                         ob_packet_container(std::move(ob_packet_container)),
                                                                                                                                         ib_packet_container(std::move(ib_packet_container)){}

            void send(Address dst, dg::string msg) noexcept{

                model::Packet pkt = this->packet_gen->get(std::move(dst), std::move(msg));
                this->ob_packet_container->push(pkt);
                exception_t err = this->retransmission_manager->add_retriable(std::move(pkt));

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                }            
            }

            auto recv() noexcept -> std::optional<dg::string>{

                std::optional<Packet> pkt = this->ib_packet_container->pop();

                if (!pkt.has_value()){
                    return std::nullopt;
                }
                
                dg::string rs = std::move(pkt->content);
                return rs;
            }
    };

    struct ComponentFactory{

        static auto get_retransmittable_mailbox_controller(std::unique_ptr<packet_controller::InBoundIDControllerInterface> ib_id_controller,
                                                           std::unique_ptr<packet_controller::InBoundTrafficController> ib_traffic_controller,
                                                           std::shared_ptr<packet_controller::WAPScheduler> scheduler,
                                                           std::unique_ptr<model::SocketHandle, socket_service::socket_close_t> socket,
                                                           std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen,
                                                           std::unique_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                                                           std::unique_ptr<packet_controller::PacketContainerInterface> ob_packet_container,
                                                           std::unique_ptr<packet_controller::PacketContainerInterface> ib_packet_container,
                                                           std::chrono::nanoseconds traffic_reset_dur,
                                                           std::chrono::nanoseconds scheduler_update_dur,
                                                           size_t num_inbound_worker,
                                                           size_t num_outbound_worker,
                                                           size_t num_retry_worker) -> std::unique_ptr<MailboxInterface>{
            
            const size_t MIN_WORKER_SIZE    = size_t{1u};
            const size_t MAX_WORKER_SIZE    = size_t{1024u}; 

            if (ib_id_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ib_traffic_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (scheduler == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (socket == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (packet_gen == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (retransmission_manager == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ob_packet_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (ib_packet_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(num_inbound_worker, MIN_WORKER_SIZE, MAX_WORKER_SIZE) != num_inbound_worker){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(num_outbound_worker, MIN_WORKER_SIZE, MAX_WORKER_SIZE) != num_outbound_worker){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(num_retry_worker, MIN_WORKER_SIZE, MAX_WORKER_SIZE) != num_retry_worker){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::shared_ptr<packet_controller::InBoundIDControllerInterface> ib_id_controller_sp            = std::move(ib_id_controller);
            std::shared_ptr<packet_controller::InBoundTrafficController> ib_traffic_controller_sp           = std::move(ib_traffic_controller);
            std::shared_ptr<packet_controller::WAPScheduler> scheduler_sp                                   = std::move(scheduler);
            std::shared_ptr<model::SocketHandle> socket_sp                                                  = std::move(socket);
            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager_sp    = std::move(retransmission_manager);
            std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container_sp             = std::move(ob_packet_container);
            std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container_sp             = std::move(ib_packet_container);
            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec                            = {};
 
            for (size_t i = 0u; i < num_inbound_worker; ++i){
                auto worker_ins     = worker::ComponentFactory::spawn_inbound_worker(retransmission_manager_sp, ob_packet_container_sp, ib_packet_container_sp, ib_id_controller_sp, ib_traffic_controller_sp, scheduler_sp, socket_sp);
                auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker_ins)));
                daemon_vec.emplace_back(std::move(daemon_handle));
            }

            for (size_t i = 0u; i < num_outbound_worker; ++i){
                auto worker_ins     = worker::ComponentFactory::spawn_outbound_worker(ob_packet_container_sp, socket_sp);
                auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::COMPUTING_DAEMON, std::move(worker_ins)));
                daemon_vec.emplace_back(std::move(daemon_handle));
            }

            for (size_t i = 0u; i < num_retry_worker; ++i){
                auto worker_ins     = worker::ComponentFactory::spawn_retransmission_worker(retransmission_manager_sp, ob_packet_container_sp);
                auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::COMPUTING_DAEMON, std::move(worker_ins)));
                daemon_vec.emplace_back(std::move(daemon_handle));
            }

            auto traffic_update_ins     = worker::ComponentFactory::spawn_update_worker(ib_traffic_controller_sp, traffic_reset_dur);
            auto traffic_daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::HEARTBEAT_DAEMON, std::move(traffic_update_ins)));
            auto sched_update_ins       = worker::ComponentFactory::spawn_update_worker(scheduler_sp, scheduler_update_dur);
            auto sched_daemon_handle    = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::HEARTBEAT_DAEMON, std::move(sched_update_ins)));

            daemon_vec.push_back(std::move(traffic_daemon_handle));
            daemon_vec.push_back(std::move(sched_daemon_handle)); 

            return std::make_unique<RetransmittableMailBoxController>(std::move(daemon_vec), std::move(packet_gen), 
                                                                      std::move(retransmission_manager_sp), std::move(ob_packet_container_sp),
                                                                      std::move(ib_packet_container_sp));
        }

    };
}

namespace dg::network_kernel_mailbox_impl1{

    //scheduler - this is actually hard to implement
    //discretization_sz - is is uniform discretization or log discretization
    //statistics data might be saturated and become irrelevant after a certain period of time
    //Address need to be cleared intervally - this should be solved internally rather than default instantiation
    //sampling needs to be fast enough and not becoming an overhead
    //update_interval and max_sched_time are highly correlated - this is to tell whether outbound_sz | inbound_sz are affected by the sched_time
    //update_interval guarantees inbound of all outbounds from [0, update_interval - max_sched_time - C] to be not affected - this is an important note - because the algorithm depends on success rate solely 
    //the fuzzy interval from [update_interval - max_sched_time - C, update_interval] is uncertainty - so - the higher update_interval/ max_sched_time ratio - the lower the uncertainty
    //sampling bias - given random time slices A, B of two random statistical points. Are they truely uniform distribution - if so - in what space does the uniform distribution happens?
    //the probabilistic outbound distribution of the two random time slices should be identical
    //it seems like max_update_interval is irrelevant - because only min_update_interval is required to reduce uncertainty

    struct Config{
        size_t num_inbound_worker;
        size_t num_outbound_worker;
        size_t num_retry_worker;
        int sin_fam;  
        int comm;
        int protocol;
        model::IP host_ip;
        uint16_t host_port;
        std::chrono::nanoseconds retransmission_delay; 
        size_t retransmission_count;
        size_t retransmission_cap;
        size_t inbound_exhaustion_control_cap;
        size_t outbound_exhaustion_control_cap;
        size_t global_id_flush_cap;
        size_t inbound_traffic_addr_cap;
        size_t inbound_traffic_global_cap;
        size_t inbound_traffic_max_address;
        std::chrono::nanoseconds traffic_reset_dur;
        std::chrono::nanoseconds sched_rtt_minbound;
        std::chrono::nanoseconds sched_rtt_maxbound;
        size_t sched_rtt_discretization_sz;
        std::chrono::nanoseconds sched_adjecent_minbound;
        std::chrono::nanoseconds sched_adjecent_maxbound;
        size_t sched_adjecent_discretization_sz;
        std::chrono::nanoseconds sched_outgoing_maxbound;
        std::chrono::nanoseconds sched_update_interval;
        std::chrono::nanoseconds sched_reset_interval;
        size_t sched_map_cap;
        size_t sched_rtt_vec_cap;
        std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> retry_device;
    };

    auto spawn(Config config) -> std::unique_ptr<core::MailboxInterface>{
        
        using namespace dg::network_kernel_mailbox_impl1::model;

        std::shared_ptr<packet_controller::WAPScheduler> scheduler{};
        std::unique_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager{};
        std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen{};
        std::unique_ptr<packet_controller::PacketContainerInterface> ib_packet_container{};
        std::unique_ptr<packet_controller::PacketContainerInterface> ob_packet_container{}; 
        std::unique_ptr<packet_controller::InBoundIDControllerInterface> ib_id_controller{};
        std::unique_ptr<packet_controller::InBoundTrafficController> ib_traffic_controller{};

        scheduler               = packet_controller::ComponentFactory::get_wap_scheduler(config.sched_rtt_minbound, config.sched_rtt_maxbound, config.sched_rtt_discretization_sz,
                                                                                         config.sched_adjecent_minbound, config.sched_adjecent_maxbound, config.sched_adjecent_discretization_sz,
                                                                                         config.sched_outgoing_maxbound, config.sched_update_interval, config.sched_reset_interval,
                                                                                        config.sched_map_cap, config.sched_rtt_vec_cap);

        retransmission_manager  = packet_controller::ComponentFactory::get_retransmission_manager(config.retransmission_delay, config.retransmission_count, config.global_id_flush_cap, config.retransmission_cap);

        packet_gen              = packet_controller::ComponentFactory::get_packet_gen(utility::to_factory_id(Address{config.host_ip, config.host_port}), Address{config.host_ip, config.host_port});

        ib_packet_container     = packet_controller::ComponentFactory::get_exhaustion_controlled_packet_container(packet_controller::ComponentFactory::get_outbound_packet_container(scheduler), 
                                                                                                                  config.retry_device,
                                                                                                                  config.inbound_exhaustion_control_cap);
                                                                                                                  
        ob_packet_container     = packet_controller::ComponentFactory::get_exhaustion_controlled_packet_container(packet_controller::ComponentFactory::get_prioritized_packet_container(),
                                                                                                                  config.retry_device,
                                                                                                                  config.outbound_exhaustion_control_cap);
        ib_id_controller        = packet_controller::ComponentFactory::get_inbound_id_controller(config.global_id_flush_cap);

        ib_traffic_controller   = packet_controller::ComponentFactory::get_inbound_traffic_controller(config.inbound_traffic_addr_cap, config.inbound_traffic_global_cap, config.inbound_traffic_max_address);

        if (config.protocol != SOCK_DGRAM){
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }

        std::unique_ptr<model::SocketHandle, socket_service::socket_close_t> sock_handle = dg::network_exception_handler::throw_nolog(socket_service::open_socket(config.sin_fam, config.comm, config.protocol));
        dg::network_exception_handler::throw_nolog(socket_service::port_socket(*sock_handle, config.host_port));

        return core::ComponentFactory::get_retransmittable_mailbox_controller(std::move(ib_id_controller), std::move(ib_traffic_controller),
                                                                              scheduler, std::move(sock_handle), 
                                                                              std::move(packet_gen), std::move(retransmission_manager),
                                                                              std::move(ob_packet_container), std::move(ib_packet_container),
                                                                              config.traffic_reset_dur, config.sched_update_interval, 
                                                                              config.num_inbound_worker, config.num_outbound_worker, 
                                                                              config.num_retry_worker);
    }
}

#endif