#ifndef __DG_SENDERLESS_MAILBOX_H__
#define __DG_SENDERLESS_MAILBOX_H__

//define HEADER_CONTROL 9

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
#include "network_exception_handler.h"

//alright - we'll focus on this heavily this week
//goals: reduce mutual exclusion overhead

//       establish basic massive parallel self-sufficient transmission controlled protocol by using round-trip time + sched_time as keys 
//          - round-trip time does not have to be from NIC (network interface controller) - because we don't need the meter precision - in fact, we need to include all factors including kernel queuing - we need the differences - and the uncertainty in non-datacenter environments is not like one in datacenter environments
//          - round-trip time + transmit_frequency would form a inbound/ outbound ratio (success rate)
//          - and we want to choose the best transmit_frequency (sched_time) based on the round-trip time (which we gather from the "recent" transmissions)
//          - how we define "best" is yet to know - the equation is not just simply maximizing success rate - yet has to take other factors into considerations - like the number of outbounds to reach the success rate which would heavily determine the congestion 
//          - what do we actually want - in the perspective of a server, we want to maximize thruput + uniform distribution of thruput or skewed distribution of thruput (IP-based)
//          - the uniformity of thruput is not simply determined based on RTT solely - this information must be from the server to determine the transmit frequency 
//          - we have to be able to answer this question this week

//       patch every leak possible
//       hopefully reach maximum network bandwidth
//       we dont really care about how kernel actually handles their packets and their internal memory ordering virtue - we must do good on our parts and pray that kernel will come to their senses of not polluting the system (this is the sole purpose of the kernel) - this includes minimizing mutex calls and accumulating kernel recv + delvrsrv - this requires a heartbeat self ping to avoid unconsumed packets
//       things would be very nasty if the physical core thread scales to 1024 or 2048 - at this level - hardsync or even acquire or release would be a disaster (1024 memory_order_acquire/core*s * 1024 = 1 << 20 acquire/s ~= 100ms of overhead - things are bad)
//       we try to implement the doables first and try to relax every operation possible
//       the optimizables we are implementing are: (1) vectorization of ack packets + IP by using kv_delvrsrv (2x optimizables)
//                                                 (2) reducing mutex acquisitions + memory ordering by using batching techniques
//                                                 (3) "noblock" the UDP recv by using self packet ping
//                                                 (4) maximizing bandwidth by using affined ports or SO_REUSEPORT + SO_ATTACH - we'll be doing actual benchs later this week - stay tuned
//

//lets see what we could do
//recall how CPU works - CPU does not stop - they have a "side" buffer to see where branches might go - they go forward in the tentative direction - if things work out fine - fine - if not reverse and correct their route
//they also have a statistic calculator - these guys work independently - a guy does not stop - a guy updates the heuristics - and a guy to inform the guys that aren't stopping
//so this is an affinity problem
//in the case of scheduler - or the case of producer + consumer - we need to accumulate the orders in order to reduce the mutual exclusion overheads (this is important)

//otherwise we are forever giga chads - this breaks practices - yet I guess that must be done - we must bring the number of acquire + release (or std::memory_order_seq_cst for that matter) operation -> < 1000 per second for the entire system - we'll scale to improve the latency
//                                    - this is another radix of optimization that I dont think is related to the all relaxed + transactional open-close cache invalidation - if we are moving in the latency direction - we must do things that way
//                                                                                                                                                                          - if we are moving in the thruput direction - we must batch EVERYTHING
//                                                                                                                                                                          - the right answer is not either or but both must be achieved
//                                                                                                                                                                          - people already done the all relaxed - it's called volatile - yet its deprecating - we dont know if volatile even passes code review now - we'll try our best - we leave the rest to the advancement of technology
//                                                                                                                                                                          - it is complicated - if we decided to relax everything (including container) - it must be of size std::array<1024, char> for every relaxed read (to avoid the overhead of relaxed) - this can be achieved - or must be achieved if we dont want to contaminate other threads' computation 

//let's start simple
//we use exponential backoff strategy for memregion_lock acquisitions - to increase the number of relaxed operation as many as possible - followed by a hardsync std::atomic_thread_fence(std::memory_order_seq_cst) post the successful acquisition operation
//we'll try to increase the number of push/ pop for producer consumer -> 1024 - by using delvrsrv + other strategies
//as for one simple std::mutex acquisition - we'll leave the matter to the kernel + std implementation
//if it is spinlock - then we must self implement

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
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(fr_addr, to_addr, id, retransmission_count, priority, kind, port_utc_stamps, port_stamp_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(fr_addr, to_addr, id, retransmission_count, priority, kind, port_utc_stamps, port_stamp_sz);
        }
    };

    struct Packet: PacketHeader{
        dg::string content;
    };

    struct ScheduledPacket{
        Packet pkt;
        std::chrono::time_point<std::chrono::utc_clock> sched_time;
    };

    struct QueuedPacket{
        Packet pkt;
        std::chrono::time_point<std::chrono::utc_clock> queued_time;            
    };

    struct SchedulerFeedBack{
        Address fr;
        std::chrono::nanoseconds rtt;
    };

    struct MailBoxArgument{
        Address to;
        dg::string content;
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

    //alright - this is the interface we agreed upon
    //max_consume_size is the abort error - exception_t * would return resource_exhaustion - which is resolvable by infretry_device
    //we'll try to implement this
    //scheduler is like allocator - we "preallocate" the schedules - exhaust the allocations - then we reallocate the schedules
    //we try to stay affined + not invoking the memory_ordering for as long as possible - then we invoke free on the remaining buffer - the way we are doing allocations allows us to do that
    //we'll talk in depth about memory allocations

    //alright - let's see what we'd implement today 
    //first is heartbeat packet to unblock UDP recv
    //second is vectorization of recv + accumulation of ack packet based on IP
    //third is a first draft of a machine learning load balancing model for network packet - we are aiming for maximum thruput + uniformity of bandwidth based on rtt + transmit_frequency (this is part of the Packet)
    //we dont want to heavily load balance - just a normalization layer to allow upstream optimizations
    //fourth is SO_REUSEPORT + SO_ATTACH
    //fifth is to patch every leak possible - including retriables
    //we'll try to minimize the memory ordering along the way

    //why cant we generalize - one is many, many is one?
    //because that's how kernel works - 8K unit is the best packet size possible - and there must be an accumulation of recv in order to reach the maximum bandwidth
    //kernel does not wait for UDP buffer - so we must empty the queue as fast as possible - we want vectorization of 1024 continuous 8K buffer before doing one big push into the containers
    //                                                                                     - we vectorize the ack based on IP along the way to avoid 2x overhead (we can only rely on the temporal characteristics of data transfer)
    //                                                                                     - we probably want to throttle the memory ordering invokes -> 5000/second for all network workers - otherwise we are contaminating the system
    //                                                                                     - we do that by increasing vectorization_sz and sleep time
    //                                                                                     - we want to profile the critical sections - like binary tree push - we probably want to do key sort first before the mutex acquisition and do one AVL batch push - this should be very fast
    //                                                                                     - if that affects code quality - we want to increase affinity of tasks - like using affined ports or increasing the number of containers + randomization

    class BatchSchedulerInterface{

        public:

            virtual ~BatchSchedulerInterface() noexcept = default;
            virtual void schedule(Address *, size_t, std::expected<std::chrono::time_point<std::chrono::utc_clock>, exception_t> *) noexcept = 0;
            virtual void feedback(SchedulerFeedBack *, size_t, exception_t *) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class SchedulerInterface{

        public:

            virtual ~SchedulerInterface() noexcept = default;
            virtual auto schedule(Address) noexcept -> std::expected<std::chrono::time_point<std::chrono::utc_clock>, exception_t> = 0;
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
            virtual auto get(MailBoxArgument) noexcept -> std::expected<Packet, exception_t> = 0;
    };

    class RetransmissionManagerInterface{

        public:

            virtual ~RetransmissionManagerInterface() noexcept = default;
            virtual void add_retriables(std::move_iterator<Packet *>, size_t, exception_t *) noexcept = 0;
            virtual void ack(global_packet_id_t *, size_t, exception_t *) noexcept = 0;
            virtual void get_retriables(Packet *, size_t&, size_t) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class PacketContainerInterface{

        public:

            virtual ~PacketContainerInterface() noexcept = default;
            virtual void push(std::move_iterator<Packet *>, size_t, exception_t *) noexcept = 0;
            virtual void pop(Packet *, size_t&, size_t) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class InBoundIDControllerInterface{

        public:

            virtual ~InBoundIDControllerInterface() noexcept = default;
            virtual void thru(global_packet_id_t *, size_t, std::expected<bool, exception_t> *) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class InBoundTrafficControllerInterface{

        public:

            virtual ~InBoundTrafficControllerInterface() noexcept = default;
            virtual void thru(Address *, size_t, std::expected<bool, exception_t> *) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };
}

namespace dg::network_kernel_mailbox_impl1::core{

    using namespace dg::network_kernel_mailbox_impl1::model;

    class MailboxInterface{

        public: 

            virtual ~MailboxInterface() noexcept = default;
            virtual void send(std::move_iterator<MailBoxArgument *>, size_t, exception_t *) noexcept = 0;
            virtual void recv(dg::string *, size_t&, size_t) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
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

        size_t header_sz    = dg::network_compact_serializer::integrity_size(PacketHeader{});
        size_t content_sz   = packet.content.size();
        size_t total_sz     = content_sz + header_sz;
        dg::string bstream = std::move(packet.content);
        bstream.resize(total_sz);
        char * header_ptr   = bstream.data() + content_sz;
        dg::network_compact_serializer::integrity_serialize_into(header_ptr, static_cast<const PacketHeader&>(packet));

        return bstream;
    }

    static auto deserialize_packet(dg::string bstream) noexcept -> std::expected<Packet, exception_t>{

        size_t header_sz    = dg::network_compact_serializer::integrity_size(PacketHeader{});
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

        if (sock.sin_fam != AF_INET6){
            return dg::network_exception::INVALID_ARGUMENT;
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

        if (sock.sin_fam != AF_INET){
            return dg::network_exception::INVALID_ARGUMENT;
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
        
        if (to_addr.ip.sin_fam() != AF_INET6){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        if (sock.sin_fam != AF_INET6){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        if (sz > constants::MAXIMUM_MSG_SIZE){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        if (sock.comm != SOCK_DGRAM){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        if (inet_pton(AF_INET6, to_addr.ip.data(), &server.sin6_addr) == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        server.sin6_family  = AF_INET6;
        server.sin6_port    = htons(to_addr.port);
        auto n              = sendto(sock.kernel_sock_fd, buf, stdx::wrap_safe_integer_cast(sz), MSG_DONTWAIT, (const struct sockaddr *) &server, sizeof(struct sockaddr_in6));

        if (n == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        if (stdx::safe_integer_cast<size_t>(n) != sz){
            return dg::network_exception::RUNTIME_SOCKETIO_ERROR;
        }

        return dg::network_exception::SUCCESS;
    }

    static auto send_noblock_ipv4(SocketHandle sock, model::Address to_addr, const void * buf, size_t sz) noexcept -> exception_t{

        struct sockaddr_in server = legacy_struct_default_init<struct sockaddr_in>();
        
        if (to_addr.ip.sin_fam() != AF_INET){
            return dg::network_exception::INVALID_ARGUMENT;
        }
        if (sock.sin_fam != AF_INET){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        if (sz > constants::MAXIMUM_MSG_SIZE){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        if (sock.comm != SOCK_DGRAM){
            return dg::network_exception::INVALID_ARGUMENT;
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

        if (stdx::safe_integer_cast<size_t>(n) != sz){
            return dg::network_exception::RUNTIME_SOCKETIO_ERROR;
        }

        return dg::network_exception::SUCCESS;
    } 

    static auto send_noblock(SocketHandle sock, model::Address to_addr, const void * buf, size_t sz) noexcept -> exception_t{

        if (sock.sin_fam == AF_INET6){
            return send_noblock_ipv6(sock, to_addr, buf, sz);
        }
    
        if (sock.sin_fam == AF_INET){
            return send_noblock_ipv4(sock, to_addr, buf, sz);
        }

        return dg::network_exception::INVALID_ARGUMENT;
    }

    static auto recv_block(SocketHandle sock, void * dst, size_t& dst_sz, size_t dst_cap) noexcept -> exception_t{

        if (sock.comm != SOCK_DGRAM){
            return dg::network_exception::INVALID_ARGUMENT;
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

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
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

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
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

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                std::chrono::nanoseconds now            = stdx::utc_timestamp();
                std::chrono::nanoseconds reset_lapsed   = now - this->last_reset_time;
                std::chrono::nanoseconds update_lapsed  = now - this->last_updated_time;

                if (reset_lapsed > this->min_reset_interval){
                    this->interval_data_map.clear();
                    this->statistic_data_map.clear();
                    this->last_reset_time   = now;
                    this->last_updated_time = now;
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

            std::atomic<local_packet_id_t> last_pkt_id;
            stdx::hdi_container<factory_id_t> factory_id;

        public:

            IDGenerator(std::atomic<local_packet_id_t> last_pkt_id,
                        stdx::hdi_container<factory_id_t> factory_id) noexcept: last_pkt_id(std::move(last_pkt_id)),
                                                                                factory_id(std::move(factory_id)){}

            auto get() noexcept -> GlobalPacketIdentifier{

                return model::GlobalPacketIdentifier{this->last_pkt_id.value.fetch_add(1u, std::memory_order_relaxed), this->factory_id.value};
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

            auto get(MailBoxArgument arg) noexcept -> std::expected<Packet, exception_t>{

                model::Packet pkt           = {};
                pkt.fr_addr                 = host_addr;
                pkt.to_addr                 = std::move(arg.to);
                pkt.id                      = this->id_gen->get();
                pkt.content                 = std::move(arg.content);
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

            dg::deque<QueuedPacket> pkt_deque;
            std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> acked_id_hashset;
            std::chrono::nanoseconds transmission_delay_time;
            size_t max_retransmission;
            size_t pkt_deque_capacity;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            RetransmissionManager(dg::deque<QueuedPacket> pkt_deque,
                                  std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> acked_id_hashset,
                                  std::chrono::nanoseconds transmission_delay_time,
                                  size_t max_retransmission,
                                  size_t pkt_deque_capacity,
                                  std::unique_ptr<std::mutex> mtx,
                                  stdx::hdi_container<size_t> consume_sz_per_load) noexcept: pkt_deque(std::move(pkt_deque)),
                                                                                             acked_id_hashset(std::move(acked_id_hashset)),
                                                                                             transmission_delay_time(transmission_delay_time),
                                                                                             max_retransmission(max_retransmission),
                                                                                             pkt_deque_capacity(pkt_deque_capacity),
                                                                                             mtx(std::move(mtx)),
                                                                                             consume_sz_per_load(std::move(consume_sz_per_load)){}

            void add_retriables(std::move_iterator<Packet> * pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                const size_t RETICK_OPS_SZ  = 1024u; 
                auto clock                  = dg::tickonops_clock<std::chrono::utc_clock>(RETICK_OPS_SZ); 
                Packet * base_pkt_arr       = pkt_arr.base(); 

                for (size_t i = 0u; i < sz; ++i){
                    if (this->pkt_deque.size() == this->pkt_deque_capacity){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    if (base_pkt_arr[i].retransmission_count >= this->max_retransmission){ //it seems like this is the packet responsibility yet I think this is the retransmission responsibility - to avoid system flooding
                        exception_arr[i] = dg::network_exception::NOT_RETRANSMITTABLE;
                        continue;
                    }

                    Packet pkt                  = std::move(base_pkt_arr[i]);
                    pkt.retransmission_count    += 1;
                    QueuedPacket queued_pkt     = {};
                    queued_pkt.pkt              = std::move(pkt);
                    queued_pkt.queued_time      = clock.now();
                    this->pkt_deque.push_back(std::move(queued_pkt));
                    exception_arr[i]            = dg::network_exception::SUCCESS;
                }
            }

            void ack(global_packet_id_t * pkt_id_arr, size_t sz, exception_t * exception_arr) noexcept{
                
                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->acked_id_hashset->insert(pkt_id_arr[i]);
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            void get_retriables(Packet * output_pkt_arr, size_t& sz, size_t output_pkt_arr_cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                std::chrono::time_point<std::chrono::utc_clock> time_bar = std::chrono::utc_clock::now() - this->transmission_delay_time;
                auto key            = QueuedPacket{};
                key.queued_time     = time_bar;
                auto last           = std::lower_bound(this->pkt_deque.begin(), this->pkt_deque.end(), key, [](const auto& lhs, const auto& rhs){return lhs.queued_time < rhs.queued_time;});
                size_t barred_sz    = std::distance(this->pkt_deque.begin(), last);
                sz                  = std::min(barred_sz, output_pkt_arr_cap); 
                auto new_last       = std::next(this->pkt_deque.begin(), sz);
                auto out_iter       = output_pkt_arr;

                for (auto it = this->pkt_deque.begin(); it != new_last; ++it){
                    if (!this->acked_id_hashset->contains(it->pkt.id)){
                        *out_iter = std::move(it->pkt);
                        std::advance(out_iter, 1u);
                    }
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }
    };

    class ExhaustionControlledRetransmissionManager: public virtual RetransmissionManagerInterface{

        private:

            std::unique_ptr<RetransmissionManagerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
        
        public:

            ExhaustionControlledRetransmissionManager(std::unique_ptr<RetransmissionManagerInterface> base,
                                                      std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) noexcept: base(std::move(base)),
                                                                                                                                                 executor(std::move(executor)){}

            void add_retriables(std::move_iterator<Packet *> pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                Packet * pkt_arr_base               = pkt_arr.base();
                Packet * pkt_arr_first              = pkt_arr_base;
                Packet * pkt_arr_last               = std::next(pkt_arr_first, sz);
                exception_t * exception_arr_first   = exception_arr;
                exception_t * exception_arr_last    = std::next(exception_arr_first, sz);
                size_t sliding_window_sz            = sz; 

                auto task = [&, this]() noexcept{
                    this->base->add_retriables(std::make_move_iterator(pkt_arr_first), sliding_window_sz, exception_arr_first);

                    exception_t * retriable_eptr_first  = std::find(exception_arr_first, exception_arr_last, dg::network_exception::QUEUE_FULL);
                    exception_t * retriable_eptr_last   = std::find_if(retriable_eptr_first, exception_arr_last, [](exception_t err){return e != dg::network_exception::QUEUE_FULL;});
                    size_t relative_offset              = std::distance(exception_arr_first, retriable_eptr_first);
                    sliding_window_sz                   = std::distance(retriable_eptr_first, retriable_eptr_last);

                    std::advance(pkt_arr_first, relative_offset);
                    std::advance(exception_arr_first, relative_offset);

                    return pkt_arr_first == pkt_arr_last;
                };

                auto virtual_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(std::move(task));
                this->executor->exec(virtual_task);
            }

            void ack(global_packet_id_t * packet_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                this->base->ack(packet_id_arr, sz, exception_arr);
            }

            void get_retriables(Packet * output_pkt_arr, size_t& sz, size_t output_pkt_arr_cap) noexcept{

                this->base->get_retriables(output_pkt_arr, sz, output_pkt_arr_cap);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    class PrioritizedPacketContainer: public virtual PacketContainerInterface{

        private:

            dg::vector<Packet> packet_vec;
            size_t packet_vec_capacity;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            PrioritizedPacketContainer(dg::vector<Packet> packet_vec,
                                       size_t packet_vec_capacity,
                                       std::unique_ptr<std::mutex> mtx,
                                       stdx::hdi_container<size_t> consume_sz_per_load) noexcept: packet_vec(std::move(packet_vec)),
                                                                                                  packet_vec_capacity(packet_vec_capacity),
                                                                                                  mtx(std::move(mtx)),
                                                                                                  consume_sz_per_load(std::move(consume_sz_per_load)){}

            void push(std::move_iterator<Packet> * pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto less = [](const Packet& lhs, const Packet& rhs){return lhs.priority < rhs.priority;};

                for (size_t i = 0u; i < sz; ++i){
                    if (this->packet_vec.size() == this->packet_vec_capacity){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    this->packet_vec.push_back(pkt_arr[i]);
                    std::push_heap(this->packet_vec.begin(), this->packet_vec.end(), less);
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }     

            void pop(Packet * output_pkt_arr, size_t& sz, size_t output_pkt_arr_capacity) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto less   = [](const Packet& lhs, const Packet& rhs){return lhs.priority < rhs.priority;};
                sz          = std::min(this->output_pkt_arr_capacity, this->packet_vec.size());
                auto out_it = output_pkt_arr; 

                for (size_t i = 0u; i < sz; ++i){
                    std::pop_heap(this->packet_vec.begin(), this->packet_vec.end(), less);
                    *out_it = std::move(this->packet_vec.back());
                    this->packet_vec.pop_back();
                    std::advance(out_it, 1u);
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }   
    };

    class ScheduledPacketContainer: public virtual PacketContainerInterface{

        private:

            dg::vector<ScheduledPacket> packet_vec;
            std::shared_ptr<SchedulerInterface> scheduler;
            size_t packet_vec_capacity;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load; 

        public:

            ScheduledPacketContainer(dg::vector<ScheduledPacket> packet_vec, 
                                     std::shared_ptr<SchedulerInterface> scheduler,
                                     size_t packet_vec_capacity,
                                     std::unique_ptr<std::mutex> mtx,
                                     stdx::hdi_container<size_t> consume_sz_per_load) noexcept: packet_vec(std::move(packet_vec)),
                                                                                                scheduler(std::move(scheduler)),
                                                                                                packet_vec_capacity(packet_vec_capacity),
                                                                                                mtx(std::move(mtx)),
                                                                                                consume_sz_per_load(std::move(consume_sz_per_load)){}

            void push(std::move_iterator<Packet *> pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto greater            = [](const auto& lhs, const auto& rhs){return lhs.sched_time > rhs.sched_time;};
                Packet * base_pkt_arr   = pkt_arr.base();

                for (size_t i = 0u; i < sz; ++i){
                    if (this->packet_vec.size() == this->packet_vec_capacity){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    std::expected<std::chrono::timepoint<std::chrono::utc_clock>, exception_t> sched_time = this->scheduler->schedule(base_pkt_arr[i].to_addr);

                    if (!sched_time.has_value()){
                        exception_arr[i] = sched_time.error();
                        continue;
                    }

                    auto sched_packet       = ScheduledPacket{};
                    sched_packet.pkt        = std::move(base_pkt_arr[i]);
                    sched_packet.sched_time = sched_time.value();

                    this->packet_vec.push_back(std::move(sched_packet));
                    std::push_heap(this->packet_vec.begin(), this->packet_vec.end(), greater);
                    exception_arr[i]        = dg::network_exception::SUCCESS;
                }
            }

            void pop(Packet * output_pkt_arr, size_t& sz, size_t output_pkt_arr_capacity) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto greater    = [](const auto& lhs, const auto& rhs){return lhs.sched_time > rhs.sched_time;};
                auto time_bar   = std::chrono::utc_clock::now();

                for (size_t i = 0u; i < output_pkt_arr_capacity; ++i){
                    if (this->packet_vec.empty()){
                        return;
                    }

                    if (this->packet_vec.front().sched_time > time_bar){
                        return;
                    }

                    std::pop_heap(this->packet_vec.begin(), this->packet_vec.end(), greater);
                    output_pkt_arr[sz++] = std::move(this->packet_vec.back().pkt);
                    this->packet_vec.pop_back();
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }
    };

    class OutboundPacketContainer: public virtual PacketContainerInterface{

        private:

            std::unique_ptr<PacketContainerInterface> ack_container;
            std::unique_ptr<PacketContainerInterface> req_container;
            size_t ack_accum_sz;
            size_t req_accum_sz; 
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            OutboundPacketContainer(std::unique_ptr<PacketContainerInterface> ack_container,
                                    std::unique_ptr<PacketContainerInterface> req_container,
                                    size_t ack_accum_sz,
                                    size_t req_accum_sz,
                                    stdx::hdi_container<size_t> consume_sz_per_load) noexcept: ack_container(std::move(ack_container)),
                                                                                               req_container(std::move(req_container)),
                                                                                               ack_accum_sz(ack_accum_sz),
                                                                                               req_accum_sz(req_accum_sz),
                                                                                               consume_sz_per_load(std::move(consume_sz_per_load)){}

            void push(std::move_iterator<Packet *> pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                Packet * base_pkt_arr       = pkt_arr.base();
                auto ack_push_resolutor     = InternalPushResolutor{};
                ack_push_resolutor.dst      = this->ack_container.get();

                size_t trimmed_ack_accum_sz = std::min(std::min(this->ack_accum_sz, sz), this->ack_container->max_consume_size());
                size_t ack_accumulator_size = dg::network_producer_consumer::delvrsrv_allocation_cost(&ack_push_resolutor, trimmed_ack_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> ack_accumulator_buf(ack_accumulator_size);
                auto ack_accumulator        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ack_push_resolutor, trimmed_ack_accum_sz, ack_accumulator_buf.get()));

                auto req_push_resolutor     = InternalPushResolutor{};
                req_push_resolutor.dst      = this->req_container.get();

                size_t trimmed_req_accum_sz = std::min(std::min(this->req_accum_sz, sz), this->req_container->max_consume_size());
                size_t req_accumulator_size = dg::network_producer_consumer::delvrsrv_allocation_cost(&req_push_resolutor, trimmed_req_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> req_accumulator_buf(req_accumulator_size);
                auto req_accumulator        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&req_push_resolutor, trimmed_req_accum_sz, req_accumulator_buf.get()));

                std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

                for (size_t i = 0u; i < sz; ++i){
                    auto delivery_arg           = DeliveryArgument{};
                    uint8_t kind                = base_pkt_arr[i].kind;
                    delivery_arg.pkt_ptr        = std::next(base_pkt_arr, i);
                    delivery_arg.exception_ptr  = std::next(exception_arr, i);
                    delivery_arg.pkt            = std::move(base_pkt_arr[i]);

                    switch (kind){
                        case constants::rts_ack:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(ack_accumulator.get(), std::move(delivery_arg));
                            break;
                        }
                        case constants::request:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(req_accumulator.get(), std::move(delivery_arg));
                            break;
                        }
                        default:
                        {
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    }
                }
            }

            void pop(Packet * output_pkt_arr, size_t& sz, size_t output_pkt_arr_capacity) noexcept{

                size_t cur_sz               = {};
                this->ack_container->pop(output_pkt_arr, cur_sz, output_pkt_arr_capacity);
                Packet * new_output_pkt_arr = std::next(output_pkt_arr, cur_sz);
                size_t new_cap              = output_pkt_arr_capacity - cur_sz;
                size_t new_sz               = {};
                this->req_container->pop(new_output_pkt_arr, new_sz, new_cap); 
                sz                          = cur_sz + new_sz;
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }

        private:

            struct DeliveryArgument{
                Packet * pkt_ptr;
                exception_t * exception_ptr;
                Packet pkt;
            };

            struct InternalPushResolutor: dg::network_producer_consumer::ConsumerInterface<DeliveryArgument>{
                PacketContainerInterface * dst;

                void push(std::move_iterator<DeliveryArgument *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<Packet[]> pkt_arr(sz); //whatever - we aren't excepting this - string is default initialized as a char array - and this should not overflow the stack buffer
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    DeliveryArgument * raw_data_arr = data_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        pkt_arr[i] = std::move(raw_data_arr[i].pkt);                        
                    }

                    this->dst->push(pkt_arr.get(), exception_arr.get(), sz);

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            *raw_data_arr[i].pkt_ptr        = std::move(pkt_arr[i]);
                            *raw_data_arr[i].exception_ptr  = exception_arr[i];
                        }
                    }
                }
            };
    };

    class InBoundIDController: public virtual InBoundIDControllerInterface{
        
        private:

            std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> id_hashset;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            InBoundIDController(std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> id_hashset,
                                std::unique_ptr<std::mutex> mtx,
                                stdx::hdi_container<size_t> consume_sz_per_load) noexcept: id_hashset(std::move(id_hashset)),
                                                                                           mtx(std::move(mtx)),
                                                                                           consume_sz_per_load(std::move(consume_sz_per_load)){}

            void thru(global_packet_id_t * packet_id_arr, size_t sz, std::expected<bool, exception_t> * op) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                for (size_t i = 0u; i < sz; ++i){
                    if (this->id_hashset->contains(packet_id_arr[i])){
                        op[i] = false;
                        continue;
                    }

                    this->id_hashset->insert(packet_id_arr[i]);
                    op[i] = true;
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }
    };

    class ExhaustionControlledPacketContainer: public virtual PacketContainerInterface{

        private:

            std::unique_ptr<PacketContainerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;

        public:

            ExhaustionControlledPacketContainer(std::unique_ptr<PacketContainerInterface> base,
                                                std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) noexcept: base(std::move(base)),
                                                                                                                                           executor(std::move(executor)){}

            void push(std::move_iterator<Packet *> pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                Packet * pkt_arr_raw                = pkt_arr.base();
                Packet * pkt_arr_first              = pkt_arr_raw;
                Packet * pkt_arr_last               = std::next(pkt_arr_first, sz);
                exception_t * exception_arr_first   = exception_arr;
                exception_t * exception_arr_last    = std::next(exception_arr_first, sz);
                size_t sliding_window_sz            = sz;

                auto task = [&, this]() noexcept{
                    this->base->push(std::make_move_iterator(pkt_arr_first), sliding_window_sz, exception_arr_first);

                    exception_t * retriable_eptr_first  = std::find(exception_arr_first, exception_arr_last, dg::network_exception::QUEUE_FULL);
                    exception_t * retriable_eptr_last   = std::find_if(retriable_eptr_first, exception_arr_last, [](exception_t err){return err != dg::network_exception::QUEUE_FULL;});
                    size_t relative_offset              = std::distance(exception_arr_first, retriable_eptr_first);
                    sliding_window_sz                   = std::distance(retriable_eptr_first, retriable_eptr_last);

                    std::advance(pkt_arr_first, relative_offset);
                    std::advance(exception_arr_first, relative_offset);

                    return pkt_arr_first == pkt_arr_last;
                };

                auto virtual_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(std::move(task));
                this->executor->exec(virtual_task);
            } 

            void pop(Packet * output_pkt_arr, size_t& sz, size_t output_pkt_arr_capacity) noexcept{

                this->base->pop(output_pkt_arr, sz, output_pkt_arr_capacity);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
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
            stdx::hdi_container<size_t> consume_sz_per_load;
        
        public:

            InBoundTrafficController(dg::unordered_unstable_map<Address, size_t> address_counter_map,
                                     size_t address_cap,
                                     size_t global_cap,
                                     size_t map_cap,
                                     size_t global_counter,
                                     std::unique_ptr<std::mutex> mtx,
                                     stdx::hdi_container<size_t> consume_sz_per_load) noexcept: address_counter_map(std::move(address_counter_map)),
                                                                                                address_cap(address_cap),
                                                                                                global_cap(global_cap),
                                                                                                map_cap(map_cap),
                                                                                                global_counter(global_counter),
                                                                                                mtx(std::move(mtx)),
                                                                                                consume_sz_per_load(std::move(consume_sz_per_load)){}

            auto thru(Address addr) noexcept -> std::expected<bool, exception_t>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

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

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                this->address_counter_map.clear();
                this->global_counter = 0u;
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
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

                Address dst             = cur->to_addr;
                dg::string bstream      = utility::serialize_packet(std::move(cur.value()));
                exception_t sock_err    = socket_service::send_noblock(*this->socket, dst, bstream.data(), bstream.size());
                
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
                auto bstream        = dg::string(constants::MAXIMUM_MSG_SIZE, ' '); //this is an optimizable - custom string implementation that only does std::malloc() - instead of calloc - it's actually hard to tell - this could be prefetch - depends on the kernel implementation of recv
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
                                        std::chrono::nanoseconds update_dur) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            using namespace std::chrono_literals; 

            const std::chrono::nanoseconds MIN_UPDATE_DUR   = std::chrono::duration_cast<std::chrono::nanoseconds>(1us); 
            const std::chrono::nanoseconds MAX_UPDATE_DUR   = std::chrono::duration_cast<std::chrono::nanoseconds>(3600s);

            if (updatable == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(update_dur, MIN_UPDATE_DUR, MAX_UPDATE_DUR) != update_dur){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<UpdateWorker>(std::move(updatable), update_dur);
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

        ib_packet_container     = packet_controller::ComponentFactory::get_exhaustion_controlled_packet_container(packet_controller::ComponentFactory::get_prioritized_packet_container(),
                                                                                                                  config.retry_device,
                                                                                                                  config.inbound_exhaustion_control_cap);
                                                                                                                  
        ob_packet_container     = packet_controller::ComponentFactory::get_exhaustion_controlled_packet_container(packet_controller::ComponentFactory::get_outbound_packet_container(scheduler), 
                                                                                                                  config.retry_device,
                                                                                                                  config.outbound_exhaustion_control_cap);
                                                                                                                  

        ib_id_controller        = packet_controller::ComponentFactory::get_inbound_id_controller(config.global_id_flush_cap);
        
        ib_traffic_controller   = packet_controller::ComponentFactory::get_inbound_traffic_controller(config.inbound_traffic_addr_cap, config.inbound_traffic_global_cap, config.inbound_traffic_max_address);

        if (config.sin_fam != AF_INET && config.sin_fam != AF_INET6){
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }

        if (config.comm != SOCK_DGRAM){
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