#ifndef __DG_SENDERLESS_MAILBOX_H__
#define __DG_SENDERLESS_MAILBOX_H__

#include <stdint.h>
#include <vector>
#include <chrono>
#include <optional>
#include <memory>
#include <string>
#include "serialization.h"
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

namespace dg::network_kernel_mailbox_ack::types{

    using global_packet_id_t     = uint64_t;
    using local_packet_id_t      = uint32_t;
    using port_t                 = uint16_t;
    using ip_t                   = std::string;
    using packet_content_t       = std::string;
    using timepoint_t            = uint64_t;
    using timelapsed_t           = int64_t;
}

namespace dg::network_kernel_mailbox_ack::model{

    using namespace dg::network_kernel_mailbox_ack::types;
    
    struct SocketHandle{
        int kernel_sock_fd;
        int sin_fam;
        int comm;
        int protocol;
    };
    
    struct Address{
        ip_t ip;
        uint16_t port;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(ip, port);
        } 

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(ip, port);
        }
    };

    struct Packet{
        Address fr_addr;
        Address to_addr; 
        global_packet_id_t id;
        uint8_t transmission_count;
        uint8_t priority;
        uint8_t taxonomy; 
        packet_content_t content;
        std::vector<timepoint_t> port_stamps;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(fr_addr, to_addr, id, transmission_count, priority, taxonomy, content, port_stamps);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(fr_addr, to_addr, id, transmission_count, priority, taxonomy, content, port_stamps);
        }
    };

    struct ScheduledPacket{
        Packet pkt;
        timepoint_t sched_time;
    };
}

namespace dg::network_kernel_mailbox_ack::constants{

    using namespace std::literals::chrono_literals;
    using namespace std::chrono;

    enum packet_taxonomy: uint8_t{
        rts_ack = 0,    
        request = 1
    };

    static inline constexpr size_t MAXIMUM_MSG_SIZE = size_t{1} << 14;  
}

namespace dg::network_kernel_mailbox_ack::memory{

    class Allocatable{

        public:

            virtual ~Allocatable() noexcept = default;
            virtual char * malloc(size_t) noexcept = 0; //extend responsibility of malloc here - force noexcept - recoverability is component's optional responsibility
            virtual void free(void *) noexcept = 0;
    };
} 

namespace dg::network_kernel_mailbox_ack::data_structure{

    template <class T>
    class trivial_unordered_set_interface{

        public:

            static_assert(std::is_trivial_v<T>);

            virtual ~unordered_set_interface() noexcept = default;
            virtual void insert(T key) noexcept = 0;
            virtual auto contains(const T& key) const noexcept -> bool = 0;
    };
} 

namespace dg::network_kernel_mailbox_ack::packet_controller{

    using namespace dg::network_kernel_mailbox_ack::types;
    using namespace dg::network_kernel_mailbox_ack::model;
    
    class SchedulerInterface{

        public:

            virtual ~SchedulerInterface() noexcept = default;
            virtual auto schedule(ip_t)  noexcept -> timepoint_t = 0;
            virtual void feedback(ip_t, timelapsed_t) noexcept = 0;
    };

    class IDGeneratorInterface{

        public:

            virtual ~IDGeneratorInterface() noexcept = default;
            virtual auto get() noexcept -> global_packet_id_t = 0;
    };

    class RetransmissionManagerInterface{

        public:

            virtual ~RetransmissionManagerInterface() noexcept = default;
            virtual void add_retriable(Packet) noexcept = 0;
            virtual void ack(global_packet_id_t) noexcept = 0;
            virtual auto get_retriables() noexcept -> std::vector<Packet> = 0;
    };

    class PacketCenterInterface{
        
        public:
            
            virtual ~PacketCenterInterface() noexcept = default;
            virtual void push(Packet) noexcept = 0;
            virtual auto pop() noexcept -> std::optional<Packet> = 0;
    };
}

namespace dg::network_kernel_mailbox_ack::core{

    using namespace dg::network_kernel_mailbox_ack::types;
    using namespace dg::network_kernel_mailbox_ack::model;

    class MailboxInterface{
        
        public: 

            virtual ~MailboxInterface() noexcept = default;
            virtual void send(Address, packet_content_t) noexcept = 0;
            virtual auto recv() noexcept -> std::optional<packet_content_t> = 0;
    };
}

namespace dg::network_kernel_mailbox_ack::utility{

    using namespace dg::network_kernel_mailbox_ack::types;
    
    static auto malloc(size_t n, std::shared_ptr<memory::Allocatable> allocator) noexcept -> std::shared_ptr<char[]>{

        char * buf      = allocator->malloc(n);
        auto destructor = [=](char * ptr) noexcept{
            allocator->free(ptr);
        };

        return std::unique_ptr<char[], decltype(destructor)>(buf, destructor);
    }

    static auto ipv4toi(std::string ip) noexcept -> uint32_t{

        const char * frmt = "%u.%u.%u.%u";

        uint32_t q1{};
        uint32_t q2{};
        uint32_t q3{};
        uint32_t q4{};
        
        if (sscanf(ip.c_str(), frmt, &q1, &q2, &q3, &q4) == EOF){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        uint32_t combined = q1 | q2 | q3 | q4;

        if (combined > std::numeric_limits<uint8_t>::max()){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return (((((q1 << CHAR_BIT) | q2) << CHAR_BIT) | q3) << CHAR_BIT) | q4;
    }

    static auto ipv4_hostip() noexcept -> std::string{ //do not invoke without synchronization - legacy code

        struct hostent * hp                 = nullptr;
        constexpr size_t TMP_BUF_LEN        = 1024; 
        std::unique_ptr<char[]> hostname    = std::make_unique<char[]>(TMP_BUF_LEN);
        std::unique_ptr<char[]> buffer      = std::make_unique<char[]>(TMP_BUF_LEN); 

        if (gethostname(hostname.get(), TMP_BUF_LEN) == -1){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        if (hp = gethostbyname(hostname.get()); !hp){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        if (!inet_ntop(AF_INET, static_cast<char *>(hp->h_addr), buffer.get(), INET_ADDRSTRLEN)){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        std::string s(buffer.get());
        return s;
    } 

    static inline std::string ipv4_hostip_val = ipv4_hostip();

    static auto unix_timestamp() noexcept -> timepoint_t{

        using namespace std::chrono;
        return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
    }

    static auto subtract_timepoint(timepoint_t tp, timelapsed_t dur) noexcept -> timepoint_t{

        return tp - dur; //
    }

    static auto frequency_to_period(double f) noexcept -> timelapsed_t{
        
        using namespace std::literals::chrono_literals;
        using namespace std::chrono;

        return duration_cast<nanoseconds>(1s).count() / f;
    } 
}

namespace dg::network_kernel_mailbox_ack::socket_service{

    using namespace dg::network_kernel_mailbox_ack::types;
    using namespace dg::network_kernel_mailbox_ack::model;

    using socket_fclose_t = void (*)(SocketHandle *) noexcept; 

    static auto open_socket(int sin_fam, int comm, int protocol) noexcept -> std::expected<std::unique_ptr<SocketHandle, socket_fclose_t>, exception_t>{

        auto destructor = [](SocketHandle * sock) noexcept{
            if (close(sock->kernel_sock_fd) == -1){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_kernel_error(errno)));
                std::abort();
            };

            delete sock;
        };

        int sock = socket(sin_fam, comm, protocol);
        
        if (sock == -1){
            return std::unexpected(dg::network_exception::wrap_kernel_exception(errno));
        }

        return {std::in_place_t{}, new SocketHandle{sock, sin_fam, comm, protocol}, destructor};
    }

    static auto bind_socket_to_port(SocketHandle sock, uint16_t port) noexcept -> exception_t{

        struct sockaddr_in server{};

        server.sin_family       = sock.sin_fam;
        server.sin_addr.s_addr  = INADDR_ANY;
        server.sin_port         = htons(port);

        if (bind(sock.kernel_sock_fd, (struct sockaddr *) &server, sizeof(server)) == -1){
            return dg::network_exception::wrap_kernel_exception(errno);
        }

        return dg::network_exception::SUCCESS;
    }

    static auto nonblocking_send(SocketHandle sock, model::Address to_addr, const void * buf, size_t sz) noexcept -> exception_t{

        struct sockaddr_in server{};
        
        if (sz > constants::MAXIMUM_MSG_SIZE){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        std::string ip      = to_addr.ip;
        uint16_t to_port    = to_addr.port;

        if (inet_pton(sock.sin_fam, ip.data(), &server.sin_addr) == -1){
            return dg::network_exception::wrap_kernel_exception(errno);
        }

        server.sin_family   = sock.sin_fam;
        server.sin_port     = htons(to_port);
        auto n              = sendto(sock.kernel_sock_fd, buf, dg::network_genult::wrap_safe_integer_cast(sz), MSG_DONTWAIT, (const struct sockaddr *) &server, sizeof(server));

        if (n == -1){
            return dg::network_exception::wrap_kernel_exception(errno);
        }

        if (n != sz{
            return dg::network_exception::RUNTIME_SOCKETIO_ERROR; 
        })

        return dg::network_exception::SUCCESS;
    }

    static auto blocking_recv(SocketHandle sock, void * dst, size_t& dst_sz, size_t dst_cap) noexcept -> exception_t{

        auto n = recv(sock.kernel_sock_fd, dst, dg::network_genult::wrap_safe_integer_cast(dst_cap), 0);

        if (n == -1){
            return dg::network_exception::wrap_kernel_exception(errno);
        }
        
        dst_sz = dg::network_genult::safe_integer_cast<size_t>(n);
        return dg::network_exception::SUCCESS;
    }
}

namespace dg::network_kernel_mailbox_ack::data_structure{

    template <class T>
    class trivial_temporal_unordered_set: public virtual trivial_unordered_set_interface<T>{

        private:

            std::unordered_set<T> hashset;
            std::deque<T> entries;
            size_t cap;

        public:

            trivial_temporal_unordered_set(std::unordered_set<T> hashset,
                                           std::deque<T> entries, 
                                           size_t cap) noexcept: hashset(std::move(hashset)),
                                                                 entries(std::move(entries)),
                                                                 cap(cap){}

            void insert(T key) noexcept{

                if (this->contains(key)){
                    return;
                }

                if (this->entries.size() == this->cap){
                    size_t half_cap = this->cap >> 1;
                    assert(half_cap != 0u);

                    for (size_t i = 0u; i < half_cap; ++i){
                        T cur = this->entries.front();
                        this->entries.pop_front();
                        this->hashset.remove(cur);
                    }
                }

                this->hashset.insert(key);
                this->entries.push_back(key);
            }

            auto contains(const T& key) const noexcept -> bool{

                return this->hashset.find(key) != this->hashset.end();
            }
    };

    template <class T>
    class std_trivial_unordered_set: public virtual trivial_unordered_set_interface<T>{

        private:

            std::unordered_set<T> hashset;
        
        public:

            std_trivial_unordered_set(std::unordered_set<T> hashset) noexcept: hashset(std::move(hashset)){}

            void insert(T key) noexcept{

                this->hashset.insert(std::move(key));
            }

            auto contains(const T& key) const noexcept -> bool{

                return this->hashset.find(key) != this->hashset.end();
            }
    };
}

namespace dg::network_kernel_mailbox_ack::packet_service{

    using namespace dg::network_kernel_mailbox_ack::types;
    using namespace dg::network_kernel_mailbox_ack::model;

    static auto make_ipv4_request(Address to_addr, uint16_t src_port, global_packet_id_t id, types::packet_content_t content) noexcept -> model::Packet{

        model::Packet pkt{};

        pkt.fr_addr             = (utility::ipv4_hostip_val, src_port);
        pkt.to_addr             = std::move(to_addr);
        pkt.id                  = id;
        pkt.content             = std::move(content);
        pkt.priority            = 0u;
        pkt.transmission_count  = 0u;
        pkt.taxonomy            = constants::request;
        pkt.port_stamps         = {};

        return pkt;
    } 

    static auto make_ipv6_request(Address to_addr, global_packet_id_t id, types::packet_content_t content) noexcept -> model::Packet{

        return {};
    }
    
    static auto request_to_ack(const model::Packet& pkt) noexcept -> model::Packet{

        Packet rs{};

        rs.to_addr              = pkt.fr_addr;
        rs.fr_addr              = pkt.to_addr;
        rs.id                   = pkt.id;
        rs.transmission_count   = pkt.transmission_count;
        rs.priority             = pkt.priority;
        rs.taxonomy             = pkt.taxonomy;
        rs.port_stamps          = pkt.port_stamps;

        return rs;
    } 

    static auto get_transit_time(const model::Packet& pkt) noexcept -> types::timelapsed_t{

        if (pkt.port_stamps.size() % 2 != 0){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        types::timelapsed_t lapsed{};

        for (size_t i = 1; i < pkt.port_stamps.size(); i += 2){
            lapsed += pkt.port_stamps[i] - pkt.port_stamps[i - 1]; //need safe_cast here to avoid overflow ub
        }

        return lapsed;
    }
}

namespace dg::network_kernel_mailbox_ack::packet_controller{
    
    class StdScheduler: public virtual SchedulerInterface{

        private:

            std::unordered_map<ip_t, timepoint_t> last_sched;
            std::unordered_map<ip_t, double> frequency;
            std::unordered_map<ip_t, timelapsed_t> min_rtt;
            std::unordered_map<ip_t, timelapsed_t> total_rtt;
            std::unordered_map<ip_t, size_t> rtt_count;
            double outbound_schedule_max_frequency;
            double outbound_schedule_min_frequency;
            double outbound_schedule_learning_rate;
            double epsilon;
            timelapsed_t outbound_schedule_max_q_time;
            std::unique_ptr<std::mutex> mtx;

        public:

            StdScheduler(std::unordered_map<ip_t, timepoint_t> last_sched, 
                         std::unordered_map<ip_t, double> frequency,
                         std::unordered_map<ip_t, timelapsed_t> min_rtt,
                         std::unordered_map<ip_t, timelapsed_t> total_rtt,
                         std::unordered_map<ip_t, size_t> rtt_count,
                         double outbound_schedule_max_frequency,
                         double outbound_schedule_min_frequency,
                         double outbound_schedule_learning_rate,
                         double epsilon,
                         timelapsed_t outbound_schedule_max_q_time,
                         std::unique_ptr<std::mutex> mtx) noexcept: last_sched(std::move(last_sched)),
                                                                    frequency(std::move(frequency)),
                                                                    min_rtt(std::move(min_rtt)),
                                                                    total_rtt(std::move(total_rtt)),
                                                                    rtt_count(std::move(rtt_count)),
                                                                    outbound_schedule_max_frequency(outbound_schedule_max_frequency),
                                                                    outbound_schedule_min_frequency(outbound_schedule_min_frequency),
                                                                    outbound_schedule_learning_rate(outbound_schedule_learning_rate),
                                                                    epsilon(epsilon),
                                                                    outbound_schedule_max_q_time(outbound_schedule_max_q_time),
                                                                    mtx(std::move(mtx)){}

            auto schedule(ip_t ip) noexcept -> timepoint_t{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                
                if (this->frequency.find(ip) == this->frequency.end()){
                    this->frequency[ip] = this->outbound_schedule_max_frequency;
                }

                if (this->last_sched.find(ip) == this->last_sched.end()){
                    this->last_sched[ip] = utility::unix_timestamp();
                }

                auto tentative_sched    = this->last_sched[ip] + utility::frequency_to_period(this->frequency[ip]);
                auto MIN_SCHED          = utility::unix_timestamp();
                auto MAX_SCHED          = MIN_SCHED + this->outbound_schedule_max_q_time;
                this->last_sched[ip]    = std::clamp(tentative_sched, MIN_SCHED, MAX_SCHED);

                return this->last_sched[ip];
            }

            void feedback(ip_t ip, timelapsed_t lapsed) noexcept{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->frequency.find(ip) == this->frequency.end()){
                    this->frequency[ip] = this->outbound_schedule_max_frequency;
                }

                if (this->min_rtt.find(ip) == this->min_rtt.end()){
                    this->min_rtt[ip] = lapsed;
                }
         
                if (lapsed < this->min_rtt[ip]){
                    this->min_rtt[ip] += (lapsed - this->min_rtt[ip]) * this->outbound_schedule_learning_rate; 
                }

                this->total_rtt[ip] += lapsed;
                this->rtt_count[ip] += 1;

                double avg_rtt      = this->total_rtt[ip] / this->rtt_count[ip]; 
                double dist         = std::max(this->epsilon, (avg_rtt - this->min_rtt[ip]) * 2);
                double perc         = static_cast<double>(lapsed - this->min_rtt[ip]) / dist; 
                double f            = this->outbound_schedule_max_frequency - (this->outbound_schedule_max_frequency - this->outbound_schedule_min_frequency) * perc;

                this->frequency[ip] += (f - this->frequency[ip]) * this->outbound_schedule_learning_rate;
                this->frequency[ip] = std::clamp(this->frequency[ip], this->outbound_schedule_min_frequency, this->outbound_schedule_max_frequency);
            }
    };  

    template <class = void>
    class IPv4IDGenerator{};

    template <>
    class IPv4IDGenerator<std::void_t<std::enable_if_t<std::conjunction_v<std::is_unsigned<global_packet_id_t>, 
                                                                          std::is_unsigned<local_packet_id_t> 
                                                                          std::bool_constant<sizeof(global_packet_id_t) == 8u>, 
                                                                          std::bool_constant<sizeof(local_packet_id_t) == 4u>>>>>: public virtual IDGeneratorInterface{
        private:

            local_packet_id_t pkt_count;
            std::unique_ptr<std::mutex> mtx;

        public:

            IPv4IDGenerator(local_packet_id_t pkt_count,
                            std::unique_ptr<std::mutex> mtx) noexcept: pkt_count(pkt_count),
                                                                       mtx(std::move(mtx)){}

            auto get() noexcept -> global_packet_id_t{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                return (static_cast<global_packet_id_t>(this->pkt_count++) << 32) | static_cast<global_packet_id_t>(utility::ipv4toi(utility::ipv4_hostip_val));
            }
    
    }

    template<>
    class ConcurrentIPv4IDGenerator<std::void_t<std::enable_if_t<std::conjunction_v<std::is_unsigned<global_packet_id_t>,
                                                                                    std::is_unsigned<local_packet_id_t>,
                                                                                    std::bool_constant<sizeof(global_packet_id_t) == 8u>,
                                                                                    std::bool_constant<sizeof(local_packet_id_t) == 4u>>>>> : public virtual IDGeneratorInterface{
        private:

            std::vector<local_packet_id_t> pkt_vec_count;
        
        public:

            ConcurrentIPv4IDGenerator(std::vector<local_packet_id_t> pkt_vec_count) noexcept: pkt_vec_count(std::move(pkt_vec_count)){}

            auto get() noexcept -> global_packet_id_t{
                
                size_t thread_idx = dg::network_concurrency::this_thread_idx();
                return (static_cast<global_packet_id_t>(this->pkt_vec_count[thread_idx]++) << 32) | static_cast<global_packet_id_t>(utility::ipv4toi(utility::ipv4_hostip_val));
            }
    };

    class RetransmissionManager: public virtual RetransmissionManagerInterface{

        private:

            std::deque<std::pair<timepoint_t, Packet>> pkt_deque;
            std::unique_ptr<datastructure::trivial_unordered_set_interface<global_packet_id_t>> acked_id;
            timelapsed_t transmission_delay_time;
            size_t max_transmission;
            std::unique_ptr<std::mutex> mtx;

        public:

            RetransmissionManager(std::deque<std::pair<timepoint_t, Packet>> pkt_deque,
                                  std::unique_ptr<datastructure::trivial_unordered_set_interface<global_packet_id_t>> acked_id,
                                  timelapsed_t transmission_delay_time,
                                  size_t max_transmission,
                                  std::unique_ptr<std::mutex> mtx) noexcept: pkt_deque(std::move(pkt_deque)),
                                                                             acked_id(std::move(acked_id)),
                                                                             transmission_delay_time(transmission_delay_time),
                                                                             max_transmission(max_transmission),
                                                                             mtx(std::move(mtx)){}

            void add_retriable(Packet pkt) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (pkt.transmission_count == this->max_transmission){
                    dg::network_log_stackdump::error_optional_fast(dg::network_exception::verbose(dg::network_exception::LOST_RETRANMISSION));
                    return;
                }

                pkt.transmission_count += 1;
                auto ts = utility::unix_timestamp();
                this->pkt_deque.push_back(std::make_pair(std::move(ts), std::move(pkt))); 
            }

            void ack(global_packet_id_t pkt_id_t) noexcept{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->acked_id->insert(std::move(pkt_id_t));
            }

            auto get_retriables() noexcept -> std::vector<Packet>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto ts         = utility::unix_timestamp();
                auto lb_key     = std::make_pair(utility::subtract_timepoint(ts, this->transmission_delay_time), Packet{});
                auto last       = std::lower_bound(this->pkt_deque.begin(), this->pkt_deque.end(), lb_key, [](const auto& lhs, const auto& rhs){return lhs.first < rhs.first;});
                auto rs         = std::vector<Packet>(); 
                
                for (auto it = this->pkt_deque.begin(); it != last; ++it){
                    if (!this->acked_id->contains(it->second.id)){
                        rs.push_back(std::move(it->second)); 
                    }
                }

                this->pkt_deque.erase(this->pkt_deque.begin(), last);
                return rs;
            }
    };

    class PriorityPacketCenter: public virtual PacketCenterInterface{

        private:
            
            std::vector<Packet> packet_vec;
            std::unique_ptr<std::mutex> mtx;

        public:

            PriorityPacketCenter(std::vector<Packet> packet_vec,
                                 std::unique_ptr<std::mutex> mtx) noexcept: packet_vec(std::move(packet_vec)),
                                                                            mtx(std::move(mtx)){}

            void push(Packet pkt) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto less       = [](const Packet& lhs, const Packet& rhs){return lhs.priority < rhs.priority;};
                this->packet_vec.push_back(std::move(pkt)); 
                std::push_heap(this->packet_vec.begin(), this->packet_vec.end(), less);
            }     

            auto pop() noexcept -> std::optional<Packet>{
                
                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);

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

    class ScheduledPacketCenter: public virtual PacketCenterInterface{

        private:

            std::vector<ScheduledPacket> packet_vec;
            std::shared_ptr<SchedulerInterface> scheduler;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            ScheduledPacketCenter(std::vector<ScheduledPacket> packet_vec, 
                                  std::shared_ptr<SchedulerInterface> scheduler,
                                  std::unique_ptr<std::mutex> mtx) noexcept: packet_vec(std::move(packet_vec)),
                                                                             scheduler(std::move(scheduler)),
                                                                             mtx(std::move(mtx)){}
            
            void push(Packet pkt) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto appender   = ScheduledPacket{std::move(pkt), this->scheduler->schedule(pkt.to_addr.ip)};
                auto greater    = [](const auto& lhs, const auto& rhs){return lhs.sched_time > rhs.sched_time;};
                this->packet_vec.push_back(std::move(appender));
                std::push_heap(this->packet_vec.begin(), this->packet_vec.end(), greater);
            }

            auto pop() noexcept -> std::optional<Packet>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);

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

    class OutboundPacketCenter: public virtual PacketCenterInterface{

        private:

            std::unique_ptr<PacketCenterInterface> ack_center;
            std::unique_ptr<PacketCenterInterface> pkt_center;
            std::unique_ptr<std::mutex> mtx; 

        public:

            OutboundPacketCenter(std::unique_ptr<PacketCenterInterface> ack_center,
                                 std::unique_ptr<PacketCenterInterface> pkt_center,
                                 std::unique_ptr<std::mutex> mtx) noexcept: ack_center(std::move(ack_center)),
                                                                            pkt_center(std::move(pkt_center)),
                                                                            mtx(std::move(mtx)){}

            void push(Packet pkt) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (pkt.taxonomy == constants::rts_ack){
                    this->ack_center->push(std::move(pkt));
                    return;
                }
                
                this->pkt_center->push(std::move(pkt));
            }

            auto pop() noexcept -> std::optional<Packet>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (auto rs = this->ack_center->pop(); rs){
                    return rs;
                }

                return this->pkt_center->pop();
            }
    };

    class InboundPacketCenter: public virtual PacketCenterInterface{

        private:

            std::unique_ptr<PacketCenterInterface> base_pkt_center;
            std::unique_ptr<datastructure::trivial_unordered_set_interface<global_packet_id_t>> id_set;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            InboundPacketCenter(std::unique_ptr<PacketCenterInterface> base_pkt_center, 
                                std::unique_ptr<datastructure::trivial_unordered_set_interface<global_packet_id_t>> id_set,
                                std::unique_ptr<std::mutex> mtx) noexcept: base_pkt_center(std::move(base_pkt_center)),
                                                                           id_set(std::move(id_set)),
                                                                           mtx(std::move(mtx)){}

            void push(Packet pkt) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->id_set->containsz(pkt.id)){
                    return;
                }

                this->id_set->insert(pkt.id);
                this->base_pkt_center->push(std::move(pkt));
            }

            auto pop() noexcept -> std::optional<Packet>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                return this->base_pkt_center->pop();
            }
    };

    class ExhaustionControlledPacketCenter: public virtual PacketCenterInterface{

        private:

            size_t cur_byte_count;
            size_t cap_byte_count;
            std::unique_ptr<PacketCenterInterface> base_packet_center;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledPacketCenter(size_t cur_byte_count,
                                             size_t cap_byte_count,
                                             std::unique_ptr<PacketCenterInterface> base_packet_center,
                                             std::unique_ptr<std::mutex> mtx) noexcept: cur_byte_count(cur_byte_count),
                                                                                        cap_byte_count(cap_byte_count),
                                                                                        base_packet_center(std::move(base_packet_center)),
                                                                                        mtx(std::move(mtx)){}

            void push(Packet pkt) noexcept{

                while (!internal_push(pkt)){}
            } 

            auto pop() noexcept -> std::optional<Packet>{

                return this->internal_pop();
            }

        private:

            auto internal_push(Packet& pkt) noexcept -> bool{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                size_t pkt_sz   = dg::compact_serializer::count(pkt.content); 

                if (this->cur_byte_count + pkt_sz > this->cap_byte_count){
                    return false;
                }

                this->cur_byte_count += pkt_sz;
                this->base_packet_center->push(std::move(pkt));
                
                return true;
            }

            auto internal_pop() noexcept -> std::optional<Packet>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto rs         = this->base_packet_center->pop();

                if (!static_cast<bool>(rs)){
                    return std::nullopt;
                }

                size_t pkt_sz   = dg::compact_serializer::count(rs->content);
                this->cur_byte_count -= pkt_sz;

                return rs;
            }
    };

    struct ComponentFactory{

        static auto get_std_scheduler(double outbound_schedule_max_frequency, double outbound_schedule_min_frequency,
                                      double outbound_schedule_learning_rate, double epsilon, 
                                      timelapsed_t outbound_schedule_max_q_time) noexcept -> std::expected<std::unique_ptr<SchedulerInterface>, exception_t>{ //I'd love to stay legacy for this project

            //preconds

            return std::make_unique<StdScheduler>(std::unordered_map<ip_t, timepoint_t>{}, 
                                                  std::unordered_map<ip_t, double>{},
                                                  std::unordered_map<ip_t, timelapsed_t>{},
                                                  std::unordered_map<ip_t, timelapsed_t>{},
                                                  std::unordered_map<ip_t, size_t>{},
                                                  outbound_schedule_max_frequency,
                                                  outbound_schedule_min_frequency,
                                                  outbound_schedule_learning_rate,
                                                  epsilon,
                                                  outbound_schedule_max_q_time,
                                                  std::make_unique<std::mutex>());
        } 

        static auto get_id_generator() -> std::unique_ptr<IDGeneratorInterface>{

            return std::make_unique<IPv4IDGenerator>(local_packet_id_t{0u}, 
                                                    std::make_unique<std::mutex>()); //abstractize std::mutex - by using crtp to avoid polymorphic access overhead - not a bad practice because returning an interface
        }

        static auto get_retransmission_manager(timelapsed_t delay, size_t max_transmission) noexcept -> std::expected<std::unique_ptr<RetransmissionManagerInterface>, exception_t>{

            //preconds

            return std::make_unique<RetransmissionManager>(std::deque<std::pair<timepoint_t, Packet>>{}, 
                                                           data_structure::Factory::get_trivial_std_unordered_set<global_packet_id_t>(), 
                                                           delay, 
                                                           max_transmission,
                                                           std::make_unique<std::mutex>()); //abstractize std::mutex - by using crtp to avoid polymorphic access overhead - not a bad practice because returning an interface
        }

        static auto get_noleak_retransmission_manager(timelapsed_t deay, size_t max_transmission, size_t clear_id_cap) noexcept -> std::expected<std::unique_ptr<RetransmissionManagerInterface>, exception_t>{

            //preconds 
            
            return std::make_unique<RetransmissionManager>(std::deque<std::pair<timepoint_t, Packet>>{},
                                                           data_structure::Factory::get_trivial_temporal_unordered_set<global_packet_id_t>(clear_id_cap),
                                                           delay, 
                                                           max_transmission,
                                                           std::make_unique<std::mutex>()); //abstractize std::mutex - by using crtp to avoid polymorphic access overhead - not a bad practice because returning an interface
        } 

        static auto get_priority_packet_center() noexcept -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<PriorityPacketCenter>(std::vector<Packet>{}, std::make_unique<std::mutex>());
        }

        static auto get_scheduled_packet_center(std::shared_ptr<SchedulerInterface> scheduler) noexcept -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<ScheduledPacketCenter>(std::vector<ScheduledPacket>{}, scheduler, std::make_unique<std::mutex>());
        }

        static auto get_inbound_packet_center() noexcept -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<InboundPacketCenter>(get_priority_packet_center(), std::unordered_set<global_packet_id_t>{}, std::make_unique<std::mutex>());
        }

        static auto get_outbound_packet_center(std::shared_ptr<SchedulerInterface> scheduler) noexcept -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<OutboundPacketCenter>(get_priority_packet_center(), get_scheduled_packet_center(scheduler), std::make_unique<std::mutex>());
        }
    };
}

namespace dg::network_kernel_mailbox_ack::worker{

    using namespace dg::network_kernel_mailbox_ack::model; 
    
    class OutBoundWorker: public virtual dg::network_concurrency::WorkerInterface{
        
        private:

            std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center;
            std::shared_ptr<model::SocketHandle> socket;
            std::shared_ptr<memory::Allocatable> allocator;

        public:

            OutBoundWorker(std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center,
                           std::shared_ptr<model::SocketHandle> socket,
                           std::shared_ptr<memory::Allocatable> allocator) noexcept: outbound_packet_center(std::move(outbound_packet_center)),
                                                                                     socket(std::move(socket)),
                                                                                     allocator(std::move(allocator)){}

            bool run_one_epoch()() noexcept{

                std::optional<Packet> cur = this->outbound_packet_center->pop();

                if (!static_cast<bool>(cur)){
                    return false;
                } 

                cur->port_stamps.push_back(utility::unix_timestamp());
                size_t sz = dg::compact_serializer::core::integrity_count(cur.value());
                std::shared_ptr<char[]> buf = utility::malloc(sz, this->allocator);
                dg::compact_serializer::core::integrity_serialize(cur.value(), buf.get());
                exception_t err = socket_service::nonblocking_send(*this->socket, cur->to_addr, buf.get(), sz); //exception_handling is optional - exception could be solved by using heartbeat broadcast approach (a component is good if all other heartbeats are good) - optional log here
                
                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_optional_fast(dg::network_exception::verbose(err));
                }

                return true;
            }
    };

    class RetransmissionWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager;
            std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center;

        public:

            RetransmissionWorker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                                 std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center) noexcept: retransmission_manager(std::move(retransmission_manager)),
                                                                                                                             outbound_packet_center(std::move(outbound_packet_center)){}

            bool run_one_epoch()() noexcept{
                
                std::vector<Packet> packets = this->retransmission_manager->get_retriables();

                if (packets.empty()){
                    return false;
                }

                for (Packet& packet: packets){
                    this->outbound_packet_center->push(packet);
                    this->retransmission_manager->add_retriable(packet);
                }

                return true;
            }
    };

    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager;
            std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center;
            std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center;
            std::shared_ptr<packet_controller::SchedulerInterface> scheduler;
            std::shared_ptr<model::SocketHandle> socket;
            std::shared_ptr<memory::Allocatable> allocator;
        
        public:

            InBoundWorker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                          std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                          std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center,
                          std::shared_ptr<packet_controller::SchedulerInterface> scheduler,
                          std::shared_ptr<model::SocketHandle> socket,
                          std::shared_ptr<memory::Allocatable> allocator) noexcept: retransmission_manager(std::move(retransmission_manager)),
                                                                                    ob_packet_center(std::move(ob_packet_center)),
                                                                                    ib_packet_center(std::move(ib_packet_center)),
                                                                                    scheduler(std::move(scheduler)),
                                                                                    socket(std::move(socket)),
                                                                                    allocator(std::move(allocator)){}
            
            bool run_one_epoch()() noexcept{
                
                model::Packet pkt   = {};
                size_t sz           = {};
                auto buf            = utility::malloc(constants::MSG_MAX_SIZE, this->allocator);
                exception_t err     = socket_service::blocking_recv(*this->socket, buf.get(), sz, constants::MSG_MAX_SIZE);

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_optional_fast(dg::network_exception::verbose(err));
                    return false;
                }

                err = dg::compact_serializer::core::integrity_deserialize(buf.get(), sz, pkt); 
                
                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_optional_fast(dg::network_exception::verbose(err));
                    return true;
                }

                pkt.port_stamps.push_back(utility::unix_timestamp());

                if (pkt.taxonomy == constants::rts_ack){
                    this->retransmission_manager->ack(pkt.id);
                    this->scheduler->feedback(pkt.fr_addr.ip, packet_service::get_transit_time(pkt));
                    return true;
                }

                if (pkt.taxonomy == constants::request){
                    auto ack_pkt = packet_service::request_to_ack(pkt);
                    this->ib_packet_center->push(std::move(pkt));
                    this->ob_packet_center->push(std::move(ack_pkt));
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

        static auto spawn_outbound_worker(std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center,
                                          std::shared_ptr<model::SocketHandle> socket,
                                          std::shared_ptr<memory::Allocatable> allocator) noexcept -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            return std::make_unique<OutBoundWorker>(std::move(outbound_packet_center), std::move(socket), std::move(allocator));
        }

        static auto spawn_retransmission_worker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager, 
                                                std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center) noexcept -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            return std::make_unique<RetransmissionWorker>(std::move(retransmission_manager), std::move(outbound_packet_center));
        }

        static auto spawn_inbound_worker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager, 
                                         std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                                         std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center, 
                                         std::shared_ptr<packet_controller::SchedulerInterface> scheduler, 
                                         std::shared_ptr<SocketHandle> socket,
                                         std::shared_ptr<memory::Allocatable> allocator) noexcept-> std::unique_ptr<Workable>{

            return std::make_unique<InBoundWorker>(std::move(retransmission_manager), std::move(ob_packet_center), 
                                                   std::move(ib_packet_center), std::move(scheduler), 
                                                   std::move(socket), std::move(allocator))>;
        }
    };
}

namespace dg::network_kernel_mailbox_ack::core{

    class IPv4MailBoxController: public virtual MailboxInterface{

        private:

            std::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
            std::unique_ptr<packet_controller::IDGeneratorInterface> id_gen;
            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager;
            std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center;
            std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center;
            uint16_t src_port;

        public:

            IPv4MailBoxController(std::vector<std::unique_ptr<dg::network_concurrency::daemon_raii_handle_t>> daemons, 
                                  std::unique_ptr<packet_controller::IDGeneratorInterface> id_gen,
                                  std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                                  std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                                  std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center,
                                  uint16_t src_port) noexcept: daemons(std::move(daemons)),
                                                               id_gen(std::move(id_gen)),
                                                               retransmission_manager(std::move(retransmission_manager)),
                                                               ob_packet_center(std::move(ob_packet_center)),
                                                               ib_packet_center(std::move(ib_packet_center)),
                                                               src_port(src_port){}

            void send(Address addr, packet_content_t buf) noexcept{

                global_packet_id_t id   = this->id_gen->get();
                model::Packet pkt       = packet_service::make_ipv4_request(std::move(addr), this->src_port, std::move(id), std::move(buf));
                this->retransmission_manager->add_retriable(pkt);
                this->ob_packet_center->push(std::move(pkt));
            }

            auto recv() noexcept -> std::optional<packet_content_t>{

                std::optional<Packet> pkt = this->ib_packet_center->pop();

                if (!static_cast<bool>(pkt)){
                    return std::nullopt;
                }

                return pkt->content;
            }
    };

    struct ComponentFactory{

    };
}

namespace dg::network_kernel_mailbox_ack{

    struct Config{
        size_t outbound_worker_count;
        size_t inbound_worker_count;
        size_t retranmission_worker_count; 
        int sin_fam;  
        int comm;
        uint16_t port; 
        uint64_t retransmission_delay; 
        uint64_t transmission_count;
        double outbound_schedule_min_frequency;
        double outbound_schedule_max_frequency;
        double outbound_schedule_learning_rate;
        double outbound_schedule_max_q_time;
        std::optional<size_t> inbound_exhaustion_control_sz;
        std::optional<size_t> outbound_exhaustion_control_sz;
        std::shared_ptr<memory::Allocatable> allocator;
    };

    auto spawn(Config config) -> std::unique_ptr<core::MailboxInterface>{

    }
}

#endif