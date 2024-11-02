#ifndef __DG_SENDERLESS_MAILBOX_H__
#define __DG_SENDERLESS_MAILBOX_H__

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
#include <algorithm>
#include <math.h>
#include <iostream>
#include "assert.h"
#include "network_std_container.h"
#include <array>
#include "network_log.h"
#include "network_exception.h"
#include "network_concurrency_x.h"
#include "stdx.h"
#include <chrono>
#include <array>

namespace dg::network_kernel_mailbox_impl1::types{

    using timepoint_t           = uint64_t;
    using timelapsed_t          = int64_t;
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
        // dg::fixed_cap_vector<timepoint_t, MAX_STAMP_SZ> port_stamps;  //TODOs:
        std::array<timepoint_t, 16> port_stamps;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(fr_addr, to_addr, id, retransmission_count, priority, kind, port_stamps);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(fr_addr, to_addr, id, retransmission_count, priority, kind, port_stamps);
        }
    };

    struct Packet: PacketHeader{
        dg::string content;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{

            reflector(content);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){

            reflector(content);
        }
    };

    struct ScheduledPacket{
        Packet pkt;
        timepoint_t sched_time;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(pkt, sched_time);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(pkt, sched_time);
        }
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
            virtual auto schedule(Address)  noexcept -> timepoint_t = 0;
            virtual void feedback(Address, timelapsed_t) noexcept = 0;
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
            virtual void add_retriable(Packet) noexcept = 0;
            virtual void ack(global_packet_id_t) noexcept = 0;
            virtual auto get_retriables() noexcept -> dg::vector<Packet> = 0;
    };

    class PacketContainerInterface{
        
        public:
            
            virtual ~PacketContainerInterface() noexcept = default;
            virtual void push(Packet) noexcept = 0;
            virtual auto pop() noexcept -> std::optional<Packet> = 0;
    };

    class InBoundControllerInterface{

        public:

            virtual ~InBoundControllerInterface() noexcept = default;
            virtual auto thru(global_packet_id_t) noexcept -> bool = 0;
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

    static auto unix_timestamp() noexcept -> uint64_t{

        std::chrono::nanoseconds ts = stdx::utc_timestamp();
        return ts.count();
    }

    static auto subtract_timepoint(uint64_t tp, int64_t dur) noexcept -> uint64_t{

        return static_cast<int64_t>(tp) - dur;
    }

    static auto frequency_to_period(double f) noexcept -> int64_t{
        
        using namespace std::literals::chrono_literals;
        using namespace std::chrono;

        return duration_cast<nanoseconds>(1s).count() / f;
    }

    template <class ...Args>
    static auto to_timelapsed(std::chrono::duration<Args...> dur) noexcept -> int64_t{

        return std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
    }

    static auto timelapsed(uint64_t fr, uint64_t to) -> int64_t{

        return static_cast<int64_t>(fr) - static_cast<int64_t>(to);
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
        rs.port_stamps          = pkt.port_stamps;

        return rs;
    }

    static auto get_transit_time(const model::Packet& pkt) noexcept -> types::timelapsed_t{

        if constexpr(DEBUG_MODE_FLAG){
            if (pkt.port_stamps.size() % 2 != 0){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        types::timelapsed_t lapsed{};

        for (size_t i = 1; i < pkt.port_stamps.size(); i += 2){
            lapsed += utility::timelapsed(pkt.port_stamps[i - 1], pkt.port_stamps[i]);
        }

        return lapsed;
    }
}

namespace dg::network_kernel_mailbox_impl1::packet_controller{
    
    class StdScheduler: public virtual SchedulerInterface{

        private:

            dg::unordered_map<Address, timepoint_t> last_sched;
            dg::unordered_map<Address, double> frequency;
            dg::unordered_map<Address, timelapsed_t> min_rtt;
            dg::unordered_map<Address, timelapsed_t> total_rtt;
            dg::unordered_map<Address, size_t> rtt_count;
            double max_frequency;
            double min_frequency;
            double learning_rate;
            double epsilon;
            timelapsed_t max_q_time;
            std::unique_ptr<std::mutex> mtx;

        public:

            StdScheduler(dg::unordered_map<Address, timepoint_t> last_sched, 
                         dg::unordered_map<Address, double> frequency,
                         dg::unordered_map<Address, timelapsed_t> min_rtt,
                         dg::unordered_map<Address, timelapsed_t> total_rtt,
                         dg::unordered_map<Address, size_t> rtt_count,
                         double max_frequency,
                         double min_frequency,
                         double learning_rate,
                         double epsilon,
                         timelapsed_t max_q_time,
                         std::unique_ptr<std::mutex> mtx) noexcept: last_sched(std::move(last_sched)),
                                                                    frequency(std::move(frequency)),
                                                                    min_rtt(std::move(min_rtt)),
                                                                    total_rtt(std::move(total_rtt)),
                                                                    rtt_count(std::move(rtt_count)),
                                                                    max_frequency(max_frequency),
                                                                    min_frequency(min_frequency),
                                                                    learning_rate(learning_rate),
                                                                    epsilon(epsilon),
                                                                    max_q_time(max_q_time),
                                                                    mtx(std::move(mtx)){}

            auto schedule(Address addr) noexcept -> timepoint_t{

                auto lck_grd = stdx::lock_guard(*this->mtx);
                
                if (this->frequency.find(addr) == this->frequency.end()){
                    this->frequency[addr] = this->max_frequency;
                }

                if (this->last_sched.find(addr) == this->last_sched.end()){
                    this->last_sched[addr] = utility::unix_timestamp();
                }

                auto tentative_sched    = this->last_sched[addr] + utility::frequency_to_period(this->frequency[addr]);
                auto MIN_SCHED          = utility::unix_timestamp();
                auto MAX_SCHED          = MIN_SCHED + this->max_q_time;
                this->last_sched[addr]    = std::clamp(tentative_sched, MIN_SCHED, MAX_SCHED);

                return this->last_sched[addr];
            }

            void feedback(Address addr, timelapsed_t lapsed) noexcept{
                
                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (this->frequency.find(addr) == this->frequency.end()){
                    this->frequency[addr] = this->max_frequency;
                }

                if (this->min_rtt.find(addr) == this->min_rtt.end()){
                    this->min_rtt[addr] = lapsed;
                }
         
                if (lapsed < this->min_rtt[addr]){
                    this->min_rtt[addr] += (lapsed - this->min_rtt[addr]) * this->learning_rate; 
                }

                this->total_rtt[addr] += lapsed;
                this->rtt_count[addr] += 1;

                double avg_rtt      = this->total_rtt[addr] / this->rtt_count[addr]; 
                double dist         = std::max(this->epsilon, (avg_rtt - this->min_rtt[addr]) * 2);
                double perc         = static_cast<double>(lapsed - this->min_rtt[addr]) / dist; 
                double f            = this->max_frequency - (this->max_frequency - this->min_frequency) * perc;

                this->frequency[addr] += (f - this->frequency[addr]) * this->learning_rate;
                this->frequency[addr] = std::clamp(this->frequency[addr], this->min_frequency, this->max_frequency);
            }
    };  

    class ASAPScheduler: public virtual SchedulerInterface{

        public:

            auto schedule(Address) noexcept -> timepoint_t{

                return utility::unix_timestamp();
            }

            void feedback(Address addr, timelapsed_t) noexcept{

                (void) addr;
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
                pkt.port_stamps             = {};

                return pkt;
            }
    };

    //max_retransmission is extensible by ping/pong (packet acked) - this is user-configurable - yet this will leak bad - because unlimited retransmission == unlinmited memory_pool - it's not easy to solve this - really 
    //packet drop is real - it is a mandatory

    class RetransmissionManager: public virtual RetransmissionManagerInterface{

        private:

            dg::deque<std::pair<timepoint_t, Packet>> pkt_deque;
            std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> acked_id_hashset;
            timelapsed_t transmission_delay_time;
            size_t max_retransmission;
            std::unique_ptr<std::mutex> mtx;

        public:

            RetransmissionManager(dg::deque<std::pair<timepoint_t, Packet>> pkt_deque,
                                  std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> acked_id_hashset,
                                  timelapsed_t transmission_delay_time,
                                  size_t max_retransmission,
                                  std::unique_ptr<std::mutex> mtx) noexcept: pkt_deque(std::move(pkt_deque)),
                                                                             acked_id_hashset(std::move(acked_id_hashset)),
                                                                             transmission_delay_time(transmission_delay_time),
                                                                             max_retransmission(max_retransmission),
                                                                             mtx(std::move(mtx)){}

            void add_retriable(Packet pkt) noexcept{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (pkt.retransmission_count == this->max_retransmission){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(dg::network_exception::LOST_RETRANSMISSION));
                    return;
                }

                pkt.retransmission_count += 1;
                pkt.priority += 1;
                timepoint_t ts = utility::unix_timestamp();
                this->pkt_deque.push_back(std::make_pair(std::move(ts), std::move(pkt))); 
            }

            void ack(global_packet_id_t pkt_id) noexcept{
                
                auto lck_grd = stdx::lock_guard(*this->mtx);
                this->acked_id_hashset->insert(std::move(pkt_id));
            }

            auto get_retriables() noexcept -> dg::vector<Packet>{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto ts         = utility::unix_timestamp();
                auto lb_key     = std::make_pair(utility::subtract_timepoint(ts, this->transmission_delay_time), Packet{});
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

                if (this->packet_vec.front().sched_time > utility::unix_timestamp()){
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

    class InBoundController: public virtual InBoundControllerInterface{
        
        private:

            std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> id_hashset;
            std::unique_ptr<std::mutex> mtx;

        public:

            InBoundController(std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> id_hashset,
                              std::unique_ptr<std::mutex> mtx) noexcept: id_hashset(std::move(id_hashset)),
                                                                         mtx(std::move(mtx)){}
            
            auto thru(global_packet_id_t packet_id) noexcept -> bool{

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

    struct ComponentFactory{

        static auto get_std_scheduler(double max_frequency, double min_frequency,
                                      double learning_rate, double epsilon, 
                                      timelapsed_t max_q_time) -> std::unique_ptr<SchedulerInterface>{
                                    
            using namespace std::chrono_literals; 

            const double MIN_MAX_FREQUENCY      = 1;
            const double MAX_MAX_FREQUENCY      = size_t{1} << 30;
            const double MIN_MIN_FREQUENCY      = 1;
            const double MAX_MIN_FREQUENCY      = size_t{1} << 30;
            const double MIN_LEARNING_RATE      = 0.01;
            const double MAX_LEARNING_RATE      = 0.99;
            const double MIN_EPSILON            = 0.001;
            const double MAX_EPSILON            = 0.01;
            const timelapsed_t MIN_MAX_Q_TIME   = utility::to_timelapsed(1ns);  
            const timelapsed_t MAX_MAX_Q_TIME   = utility::to_timelapsed(30s);

            if (std::clamp(max_frequency, MIN_MAX_FREQUENCY, MAX_MAX_FREQUENCY) != max_frequency){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(min_frequency, MIN_MIN_FREQUENCY, MIN_MAX_FREQUENCY) != min_frequency){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(learning_rate, MIN_LEARNING_RATE, MAX_LEARNING_RATE) != learning_rate){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(epsilon, MIN_EPSILON, MAX_EPSILON) != epsilon){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(max_q_time, MIN_MAX_Q_TIME, MAX_MAX_Q_TIME) != max_q_time){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (min_frequency > max_frequency){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<StdScheduler>(dg::unordered_map<Address, timepoint_t>{}, 
                                                  dg::unordered_map<Address, double>{},
                                                  dg::unordered_map<Address, timelapsed_t>{},
                                                  dg::unordered_map<Address, timelapsed_t>{},
                                                  dg::unordered_map<Address, size_t>{},
                                                  max_frequency,
                                                  min_frequency,
                                                  learning_rate,
                                                  epsilon,
                                                  max_q_time,
                                                  std::make_unique<std::mutex>());
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

        static auto get_retransmission_manager(timelapsed_t delay, size_t max_retransmission, size_t idhashset_cap) -> std::unique_ptr<RetransmissionManagerInterface>{

            using namespace std::chrono_literals; 

            const timelapsed_t MIN_DELAY        = utility::to_timelapsed(1s);
            const timelapsed_t MAX_DELAY        = utility::to_timelapsed(60s);
            const size_t MIN_MAX_RETRANSMISSION = 0u;
            const size_t MAX_MAX_RETRANSMISSION = 32u;

            if (std::clamp(delay, MIN_DELAY, MAX_DELAY) != delay){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }
            
            if (std::clamp(max_retransmission, MIN_MAX_RETRANSMISSION, MAX_MAX_RETRANSMISSION) != max_retransmission){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<RetransmissionManager>(dg::deque<std::pair<timepoint_t, Packet>>{},
                                                           data_structure::Factory::get_temporal_unordered_set<global_packet_id_t>(idhashset_cap),
                                                           delay, 
                                                           max_retransmission,
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

        static auto get_inbound_controller(size_t idhashset_cap) -> std::unique_ptr<InBoundControllerInterface>{

            return std::make_unique<InBoundController>(data_structure::Factory::get_temporal_unordered_set<global_packet_id_t>(idhashset_cap), 
                                                       std::make_unique<std::mutex>());
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

                // cur->port_stamps.push_back(utility::unix_timestamp()); //TODOs:
                dg::string bstream = utility::serialize_packet(std::move(cur.value()));
                exception_t err = socket_service::send_noblock(*this->socket, cur->to_addr, bstream.data(), bstream.size());
                
                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
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
                    this->retransmission_manager->add_retriable(std::move(packet));
                }

                return true;
            }
    };

    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager;
            std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container;
            std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container;
            std::shared_ptr<packet_controller::InBoundControllerInterface> ib_controller;
            std::shared_ptr<packet_controller::SchedulerInterface> scheduler;
            std::shared_ptr<model::SocketHandle> socket;
        
        public:

            InBoundWorker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                          std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container,
                          std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container,
                          std::shared_ptr<packet_controller::InBoundControllerInterface> ib_controller,
                          std::shared_ptr<packet_controller::SchedulerInterface> scheduler,
                          std::shared_ptr<model::SocketHandle> socket) noexcept: retransmission_manager(std::move(retransmission_manager)),
                                                                                 ob_packet_container(std::move(ob_packet_container)),
                                                                                 ib_packet_container(std::move(ib_packet_container)),
                                                                                 ib_controller(std::move(ib_controller)),
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
                
                if (!this->ib_controller->thru(pkt.id)){
                    if (pkt.kind == constants::request){
                        auto ack_pkt = packet_service::request_to_ack(pkt);
                        this->ob_packet_container->push(std::move(ack_pkt));
                    }
                    return true;
                }

                // pkt.port_stamps.push_back(utility::unix_timestamp());  //TODOs:

                if (pkt.kind == constants::rts_ack){
                    this->retransmission_manager->ack(pkt.id); //I was thinking about vectorization of ack packet - yet I think that's a premature optimization not yet to make (after profiling - second cut) - because the overhead of ack_packet / true_packet ~= 10% - 15% which will continue to decrease in the future
                    this->scheduler->feedback(pkt.fr_addr, packet_service::get_transit_time(pkt));
                    return true;
                }

                if (pkt.kind == constants::request){
                    auto ack_pkt = packet_service::request_to_ack(pkt);
                    this->ib_packet_container->push(std::move(pkt));
                    this->ob_packet_container->push(std::move(ack_pkt));
                    return true;
                }
                
                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

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
                                         std::shared_ptr<packet_controller::InBoundControllerInterface> ib_controller,
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

            if (ib_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (scheduler == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (socket == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<InBoundWorker>(std::move(retransmission_manager), std::move(ob_packet_container), 
                                                   std::move(ib_packet_container), std::move(ib_controller),
                                                   std::move(scheduler), std::move(socket));
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
                this->retransmission_manager->add_retriable(std::move(pkt));
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

        static auto get_retransmittable_mailbox_controller(std::unique_ptr<packet_controller::InBoundControllerInterface> ib_controller,
                                                           std::shared_ptr<packet_controller::SchedulerInterface> scheduler, //fine - scheduler is external injection - tons of optimization could be done with scheduler - this needs a right model to approx congestion - not the current one - of course
                                                           std::unique_ptr<model::SocketHandle, socket_service::socket_close_t> socket,
                                                           std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen,
                                                           std::unique_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                                                           std::unique_ptr<packet_controller::PacketContainerInterface> ob_packet_container,
                                                           std::unique_ptr<packet_controller::PacketContainerInterface> ib_packet_container,
                                                           size_t num_inbound_worker,
                                                           size_t num_outbound_worker,
                                                           size_t num_retry_worker) -> std::unique_ptr<MailboxInterface>{
            
            const size_t MIN_WORKER_SIZE    = size_t{1u};
            const size_t MAX_WORKER_SIZE    = size_t{1024u}; 

            if (ib_controller == nullptr){
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

            std::shared_ptr<packet_controller::InBoundControllerInterface> ib_controller_sp = std::move(ib_controller);
            std::shared_ptr<packet_controller::SchedulerInterface> scheduler_sp = std::move(scheduler);
            std::shared_ptr<model::SocketHandle> socket_sp = std::move(socket);
            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager_sp = std::move(retransmission_manager);
            std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container_sp = std::move(ob_packet_container);
            std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container_sp = std::move(ib_packet_container);
            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec = {};
 
            for (size_t i = 0u; i < num_inbound_worker; ++i){
                auto worker_ins     = worker::ComponentFactory::spawn_inbound_worker(retransmission_manager_sp, ob_packet_container_sp, ib_packet_container_sp, ib_controller_sp, scheduler_sp, socket_sp);
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

            return std::make_unique<RetransmittableMailBoxController>(std::move(daemon_vec), std::move(packet_gen), 
                                                                      std::move(retransmission_manager_sp), std::move(ob_packet_container_sp),
                                                                      std::move(ib_packet_container_sp));
        }

    };
}

namespace dg::network_kernel_mailbox_impl1{

    struct RuntimeRTTSchedulerConfig{
        double rtt_epsilon;
        double min_frequency;
        double max_frequency;
        double learning_rate;
        std::chrono::nanoseconds max_q_time;
    };

    struct ASAPSchedulerConfig{};

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
        std::variant<RuntimeRTTSchedulerConfig, ASAPSchedulerConfig> scheduler;
        size_t inbound_exhaustion_control_cap;
        size_t outbound_exhaustion_control_cap;
        size_t global_id_flush_cap;
        std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> retry_device;
    };

    auto spawn(Config config) -> std::unique_ptr<core::MailboxInterface>{
        
        using namespace dg::network_kernel_mailbox_impl1::model;

        std::shared_ptr<packet_controller::SchedulerInterface> scheduler{};
        std::unique_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager{};
        std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen{};
        std::unique_ptr<packet_controller::PacketContainerInterface> ib_packet_container{};
        std::unique_ptr<packet_controller::PacketContainerInterface> ob_packet_container{}; 
        std::unique_ptr<packet_controller::InBoundControllerInterface> ib_controller{};

        if (std::holds_alternative<RuntimeRTTSchedulerConfig>(config.scheduler)){
            auto sched_config = std::get<RuntimeRTTSchedulerConfig>(config.scheduler); 
            scheduler = packet_controller::ComponentFactory::get_std_scheduler(sched_config.max_frequency, sched_config.min_frequency,
                                                                               sched_config.learning_rate, sched_config.rtt_epsilon, 
                                                                               utility::to_timelapsed(sched_config.max_q_time));
        } else if (std::holds_alternative<ASAPSchedulerConfig>(config.scheduler)){
            scheduler = packet_controller::ComponentFactory::get_asap_scheduler();
        } else{
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }

        retransmission_manager  = packet_controller::ComponentFactory::get_retransmission_manager(utility::to_timelapsed(config.retransmission_delay), config.retransmission_count, config.global_id_flush_cap);
        packet_gen              = packet_controller::ComponentFactory::get_packet_gen(utility::to_factory_id(Address{config.host_ip, config.host_port}), Address{config.host_ip, config.host_port});
        ib_packet_container     = packet_controller::ComponentFactory::get_exhaustion_controlled_packet_container(packet_controller::ComponentFactory::get_outbound_packet_container(scheduler), 
                                                                                                                  config.retry_device,
                                                                                                                  config.inbound_exhaustion_control_cap);
        ob_packet_container     = packet_controller::ComponentFactory::get_exhaustion_controlled_packet_container(packet_controller::ComponentFactory::get_prioritized_packet_container(),
                                                                                                                  config.retry_device,
                                                                                                                  config.outbound_exhaustion_control_cap);
        ib_controller           = packet_controller::ComponentFactory::get_inbound_controller(config.global_id_flush_cap);

        if (config.protocol != SOCK_DGRAM){
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }

        auto sock_handle = dg::network_exception_handler::throw_nolog(socket_service::open_socket(config.sin_fam, config.comm, config.protocol));
        dg::network_exception_handler::throw_nolog(socket_service::port_socket(*sock_handle, config.host_port));

        return core::ComponentFactory::get_retransmittable_mailbox_controller(std::move(ib_controller), scheduler, std::move(sock_handle), 
                                                                              std::move(packet_gen), std::move(retransmission_manager),
                                                                              std::move(ob_packet_container), std::move(ib_packet_container), 
                                                                              config.num_inbound_worker, config.num_outbound_worker, 
                                                                              config.num_retry_worker);
    }
}

#endif