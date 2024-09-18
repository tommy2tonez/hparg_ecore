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

namespace dg::network_kernel_mailbox_impl1::types{

    using timepoint_t   = uint64_t;
    using timelapsed_t  = int64_t;
    using factory_id_t  = std::array<char, 24>;
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

        auto data() const noexcept -> const void *{

            if (this->flag){
                return this->ipv4.data();
            } else{
                return this->ipv6.data();
            }
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const{
            reflector(ipv4, ipv6, flag);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector){
            reflector(ipv4, ipv6, flag);
        }
    };

    using ip_t = IP; 

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

    struct GlobalPacketIdentifier{
        local_packet_id_t local_packet_id;
        factory_id_t factory_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const{
            reflector(local_packet_id, factory_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector){
            reflector(local_packet_id, factory_id);
        }
    };

    using global_packet_id_t = GlobalPacketIdentifier;

    struct Packet{
        Address fr_addr;
        Address to_addr; 
        global_packet_id_t id;
        uint8_t retransmission_count;
        uint8_t priority;
        uint8_t taxonomy; 
        dg::network_std_container::string content;
        dg::network_std_container::vector<timepoint_t> port_stamps;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(fr_addr, to_addr, id, retransmission_count, priority, taxonomy, content, port_stamps);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(fr_addr, to_addr, id, retransmission_count, priority, taxonomy, content, port_stamps);
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

    enum packet_taxonomy: uint8_t{
        rts_ack = 0,    
        request = 1
    };

    static inline constexpr size_t MAXIMUM_MSG_SIZE = size_t{1} << 14;  
}

namespace dg::network_kernel_mailbox_impl1::data_structure{

    template <class T>
    class unordered_set_interface{

        public:

            static_assert(std::is_trivial_v<T>);

            virtual ~unordered_set_interface() noexcept = default;
            virtual void insert(T key) noexcept = 0;
            virtual auto contains(const T& key) const noexcept -> bool = 0;
    };
} 

namespace dg::network_kernel_mailbox_impl1::packet_controller{

    using namespace dg::network_kernel_mailbox_impl1::types;
    using namespace dg::network_kernel_mailbox_impl1::model;
    
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

    class PacketGeneratorInterface{

        public:

            virtual ~PacketGeneratorInterface() noexcept = default;
            virtual auto get(Address to_addr, dg::network_std_container::string content) noexcept -> Packet = 0;
    };

    class RetransmissionManagerInterface{

        public:

            virtual ~RetransmissionManagerInterface() noexcept = default;
            virtual void add_retriable(Packet) noexcept = 0;
            virtual void ack(global_packet_id_t) noexcept = 0;
            virtual auto get_retriables() noexcept -> dg::network_std_container::vector<Packet> = 0;
    };

    class PacketCenterInterface{
        
        public:
            
            virtual ~PacketCenterInterface() noexcept = default;
            virtual void push(Packet) noexcept = 0;
            virtual auto pop() noexcept -> std::optional<Packet> = 0;
    };
}

namespace dg::network_kernel_mailbox_impl1::core{

    using namespace dg::network_kernel_mailbox_impl1::types;
    using namespace dg::network_kernel_mailbox_impl1::model;

    class MailboxInterface{
        
        public: 

            virtual ~MailboxInterface() noexcept = default;
            virtual void send(Address, dg::network_std_container::string) noexcept = 0;
            virtual auto recv() noexcept -> std::optional<dg::network_std_container::string> = 0;
    };
}

namespace dg::network_kernel_mailbox_impl1::utility{

    using namespace dg::network_kernel_mailbox_impl1::types;

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

        std::chrono::nanoseconds ts = dg::network_genult::utc_timestamp();
        return ts.count();
    }

    static auto subtract_timepoint(timepoint_t tp, timelapsed_t dur) noexcept -> timepoint_t{

        return tp - dur;
    }

    static auto frequency_to_period(double f) noexcept -> timelapsed_t{
        
        using namespace std::literals::chrono_literals;
        using namespace std::chrono;

        return duration_cast<nanoseconds>(1s).count() / f;
    }

    template <class ...Args>
    static auto to_timelapsed(std::chrono::duration<Args...> dur) noexcept -> timelapsed_t{

        return std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count();
    }
}

namespace dg::network_kernel_mailbox_impl1::socket_service{

    using namespace dg::network_kernel_mailbox_impl1::types;
    using namespace dg::network_kernel_mailbox_impl1::model;

    using socket_close_t = void (*)(SocketHandle *) noexcept; 

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
        
        if constexpr(DEBUG_MODE_FLAG){
            if (sz > constants::MAXIMUM_MSG_SIZE){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        if (inet_pton(sock.sin_fam, to_addr.ip.data(), &server.sin_addr) == -1){
            return dg::network_exception::wrap_kernel_exception(errno);
        }

        server.sin_family   = sock.sin_fam;
        server.sin_port     = htons(to_addr.port);
        auto n              = sendto(sock.kernel_sock_fd, buf, dg::network_genult::wrap_safe_integer_cast(sz), MSG_DONTWAIT, (const struct sockaddr *) &server, sizeof(server));

        if (n == -1){
            return dg::network_exception::wrap_kernel_exception(errno);
        }

        if (n != sz){
            return dg::network_exception::RUNTIME_SOCKETIO_ERROR;
        }

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

namespace dg::network_kernel_mailbox_impl1::data_structure{

    template <class T>
    class temporal_unordered_set: public virtual unordered_set_interface<T>{

        private:

            dg::network_std_container::unordered_set<T> hashset;
            std::deque<T> entries;
            size_t cap;

        public:

            temporal_unordered_set(dg::network_std_container::unordered_set<T> hashset,
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
    class std_unordered_set: public virtual unordered_set_interface<T>{

        private:

            dg::network_std_container::unordered_set<T> hashset;
        
        public:

            std_unordered_set(dg::network_std_container::unordered_set<T> hashset) noexcept: hashset(std::move(hashset)){}

            void insert(T key) noexcept{

                this->hashset.insert(std::move(key));
            }

            auto contains(const T& key) const noexcept -> bool{

                return this->hashset.find(key) != this->hashset.end();
            }
    };
}

namespace dg::network_kernel_mailbox_impl1::packet_service{

    using namespace dg::network_kernel_mailbox_impl1::types;
    using namespace dg::network_kernel_mailbox_impl1::model;
    
    static auto request_to_ack(const model::Packet& pkt) noexcept -> model::Packet{

        Packet rs{};

        rs.to_addr              = pkt.fr_addr;
        rs.fr_addr              = pkt.to_addr;
        rs.id                   = pkt.id;
        rs.priority             = pkt.priority;
        rs.retransmission_count = pkt.retransmission_count;
        rs.taxonomy             = pkt.taxonomy;
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
            lapsed += dg::network_genult::timelapsed(pkt.port_stamps[i], pkt.port_stamps[i - 1]);
        }

        return lapsed;
    }
}

namespace dg::network_kernel_mailbox_impl1::packet_controller{
    
    //reimplementation required - need to include success rate
    //success rate for time slice [a, b] = schedule_sz(ip, a, b) / feedback_sz(ip, a, b)
    //abstractize lck -> atomic_flag, mutex, nolock - by using Lock<T> - and lock guard takes in Lock<T>& as arg

    class StdScheduler: public virtual SchedulerInterface{

        private:

            dg::network_std_container::unordered_map<ip_t, timepoint_t> last_sched;
            dg::network_std_container::unordered_map<ip_t, double> frequency;
            dg::network_std_container::unordered_map<ip_t, timelapsed_t> min_rtt;
            dg::network_std_container::unordered_map<ip_t, timelapsed_t> total_rtt;
            dg::network_std_container::unordered_map<ip_t, size_t> rtt_count;
            double max_frequency;
            double min_frequency;
            double learning_rate;
            double epsilon;
            timelapsed_t max_q_time;
            std::unique_ptr<std::mutex> mtx;

        public:

            StdScheduler(dg::network_std_container::unordered_map<ip_t, timepoint_t> last_sched, 
                         dg::network_std_container::unordered_map<ip_t, double> frequency,
                         dg::network_std_container::unordered_map<ip_t, timelapsed_t> min_rtt,
                         dg::network_std_container::unordered_map<ip_t, timelapsed_t> total_rtt,
                         dg::network_std_container::unordered_map<ip_t, size_t> rtt_count,
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

            auto schedule(ip_t ip) noexcept -> timepoint_t{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                
                if (this->frequency.find(ip) == this->frequency.end()){
                    this->frequency[ip] = this->max_frequency;
                }

                if (this->last_sched.find(ip) == this->last_sched.end()){
                    this->last_sched[ip] = utility::unix_timestamp();
                }

                auto tentative_sched    = this->last_sched[ip] + utility::frequency_to_period(this->frequency[ip]);
                auto MIN_SCHED          = utility::unix_timestamp();
                auto MAX_SCHED          = MIN_SCHED + this->max_q_time;
                this->last_sched[ip]    = std::clamp(tentative_sched, MIN_SCHED, MAX_SCHED);

                return this->last_sched[ip];
            }

            void feedback(ip_t ip, timelapsed_t lapsed) noexcept{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->frequency.find(ip) == this->frequency.end()){
                    this->frequency[ip] = this->max_frequency;
                }

                if (this->min_rtt.find(ip) == this->min_rtt.end()){
                    this->min_rtt[ip] = lapsed;
                }
         
                if (lapsed < this->min_rtt[ip]){
                    this->min_rtt[ip] += (lapsed - this->min_rtt[ip]) * this->learning_rate; 
                }

                this->total_rtt[ip] += lapsed;
                this->rtt_count[ip] += 1;

                double avg_rtt      = this->total_rtt[ip] / this->rtt_count[ip]; 
                double dist         = std::max(this->epsilon, (avg_rtt - this->min_rtt[ip]) * 2);
                double perc         = static_cast<double>(lapsed - this->min_rtt[ip]) / dist; 
                double f            = this->max_frequency - (this->max_frequency - this->min_frequency) * perc;

                this->frequency[ip] += (f - this->frequency[ip]) * this->learning_rate;
                this->frequency[ip] = std::clamp(this->frequency[ip], this->min_frequency, this->max_frequency);
            }
    };  

    class ASAPScheduler: public virtual SchedulerInterface{

        public:

            auto schedule(ip_t) noexcept -> timepoint_t{

                return utility::unix_timestamp();
            }

            void feedback(ip_t, timelapsed_t) noexcept{

                (void) feedback;
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
                        std::unique_ptr<std::mutex> mtx) noexcept: last_pkt_id(last_pkt_id),
                                                                   factory_id(factory_id),
                                                                   mtx(std::move(mtx)){}

            auto get() noexcept -> global_packet_id_t{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                return model::GlobalPacketIdentifier{this->last_pkt_id++, this->factory_id};
            }
    };
    
    class PacketGenerator: public virtual PacketGeneratorInterface{

        private:

            std::unique_ptr<IDGeneratorInterface> id_gen;
            std::string host_ip;
            uint16_t src_port; 

        public:

            PacketGenerator(std::unique_ptr<IDGeneratorInterface> id_gen,
                            std::string host_ip,
                            uint16_t src_port) noexcept: id_gen(std::move(id_gen)),
                                                         host_ip(std::move(host_ip)),
                                                         src_port(src_port){}

            auto get(Address to_addr, dg::network_std_container::string content) noexcept -> Packet{

                model::Packet pkt           = {};
                pkt.fr_addr                 = (this->host_ip, this->src_port);
                pkt.to_addr                 = std::move(to_addr);
                pkt.id                      = this->id_gen->get();
                pkt.content                 = std::move(content);
                pkt.priority                = 0u;
                pkt.retransmission_count    = 0u;
                pkt.taxonomy                = constants::request;
                pkt.port_stamps             = {};

                return pkt;
            }
    };

    class RetransmissionManager: public virtual RetransmissionManagerInterface{

        private:

            std::deque<std::pair<timepoint_t, Packet>> pkt_deque;
            std::unique_ptr<datastructure::unordered_set_interface<global_packet_id_t>> acked_id;
            timelapsed_t transmission_delay_time;
            size_t max_transmission;
            std::unique_ptr<std::mutex> mtx;

        public:

            RetransmissionManager(std::deque<std::pair<timepoint_t, Packet>> pkt_deque,
                                  std::unique_ptr<datastructure::unordered_set_interface<global_packet_id_t>> acked_id,
                                  timelapsed_t transmission_delay_time,
                                  size_t max_transmission,
                                  std::unique_ptr<std::mutex> mtx) noexcept: pkt_deque(std::move(pkt_deque)),
                                                                             acked_id(std::move(acked_id)),
                                                                             transmission_delay_time(transmission_delay_time),
                                                                             max_transmission(max_transmission),
                                                                             mtx(std::move(mtx)){}

            void add_retriable(Packet pkt) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (pkt.retransmission_count == this->max_transmission){
                    dg::network_log_stackdump::error_optional_fast(dg::network_exception::verbose(dg::network_exception::LOST_RETRANSMISSION));
                    return;
                }

                pkt.retransmission_count += 1;
                auto ts = utility::unix_timestamp();
                this->pkt_deque.push_back(std::make_pair(std::move(ts), std::move(pkt))); 
            }

            void ack(global_packet_id_t pkt_id) noexcept{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->acked_id->insert(std::move(pkt_id));
            }

            auto get_retriables() noexcept -> dg::network_std_container::vector<Packet>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto ts         = utility::unix_timestamp();
                auto lb_key     = std::make_pair(utility::subtract_timepoint(ts, this->transmission_delay_time), Packet{});
                auto last       = std::lower_bound(this->pkt_deque.begin(), this->pkt_deque.end(), lb_key, [](const auto& lhs, const auto& rhs){return lhs.first < rhs.first;});
                auto rs         = dg::network_std_container::vector<Packet>(); 
                
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
            
            dg::network_std_container::vector<Packet> packet_vec;
            std::unique_ptr<std::mutex> mtx;

        public:

            PriorityPacketCenter(dg::network_std_container::vector<Packet> packet_vec,
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

            dg::network_std_container::vector<ScheduledPacket> packet_vec;
            std::shared_ptr<SchedulerInterface> scheduler;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            ScheduledPacketCenter(dg::network_std_container::vector<ScheduledPacket> packet_vec, 
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
            std::unique_ptr<datastructure::unordered_set_interface<global_packet_id_t>> id_set;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            InboundPacketCenter(std::unique_ptr<PacketCenterInterface> base_pkt_center, 
                                std::unique_ptr<datastructure::unordered_set_interface<global_packet_id_t>> id_set,
                                std::unique_ptr<std::mutex> mtx) noexcept: base_pkt_center(std::move(base_pkt_center)),
                                                                           id_set(std::move(id_set)),
                                                                           mtx(std::move(mtx)){}

            void push(Packet pkt) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->id_set->contains(pkt.id)){
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

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->cur_byte_count + constants::MSG_MAX_SZ > this->cap_byte_count){
                    return false;
                }

                this->cur_byte_count += constants::MSG_MAX_SZ;
                this->base_packet_center->push(std::move(pkt));

                return true;
            }

            auto internal_pop() noexcept -> std::optional<Packet>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                auto rs = this->base_packet_center->pop();

                if (!static_cast<bool>(rs)){
                    return std::nullopt;
                }

                this->cur_byte_count -= constants::MSG_MAX_SZ;

                return rs;
            }
    };

    //this is most likely not necessary - a packet is usually 32kb in size - so concurrency is not playing a crucial role here - 
    //yet it's fine to have it here to approx true concurrency

    template <size_t PACKET_CENTER_SZ>
    class ConcurrentPacketCenter: public virtual PacketCenterInterface{

        private:

            dg::network_std_container<std::unique_ptr<PacketCenterInterface>> packet_center;
        
        public:

            ConcurrentPacketCenter(dg::network_std_container<std::unique_ptr<PacketCenterInterface>> packet_center,
                                   const std::integral_constant<size_t, PACKET_CENTER_SZ>) noexcept: packet_center(std::move(packet_center)){} //weird - yet it's important to do const prop here - inheriting an interface is not a bad practice
            
            void push(Packet pkt) noexcept{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, PACKET_CENTER_SZ>{});
                this->packet_center[idx]->push(std::move(pkt));
            }

            auto pop() noexcept -> std::optional<Packet>{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, PACKET_CENTER_SZ>{});
                return this->packet_center[idx]->pop();
            }
    };

    struct ComponentFactory{

        static auto get_std_scheduler(double max_frequency, double min_frequency,
                                      double learning_rate, double epsilon, 
                                      timelapsed_t max_q_time) -> std::unique_ptr<SchedulerInterface>{
                                    
            using namespace dg::chrono::literals; 

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

            return std::make_unique<StdScheduler>(dg::network_std_container::unordered_map<ip_t, timepoint_t>{}, 
                                                  dg::network_std_container::unordered_map<ip_t, double>{},
                                                  dg::network_std_container::unordered_map<ip_t, timelapsed_t>{},
                                                  dg::network_std_container::unordered_map<ip_t, timelapsed_t>{},
                                                  dg::network_std_container::unordered_map<ip_t, size_t>{},
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

        static auto get_ipv4_id_generator() -> std::unique_ptr<IDGeneratorInterface>{

            // return std::make_unique<IPv4IDGenerator>(utility::ipv4_hostip_val, local_packet_id_t{0u}, std::make_unique<std::mutex>());
        }

        static auto get_ipv6_id_generator() -> std::unique_ptr<IDGeneratorInterface>{

            //
            // return std::make_unique<IPv6IDGenerator>(utility::ipv6_hostip_val, local_packet_id_t{0u}, std::make_unique<std::mutex>());
        }

        static auto get_ipv4_packet_gen(uint16_t port) -> std::unique_ptr<PacketGeneratorInterface>{

            return std::make_unique<PacketGenerator>(get_ipv4_id_generator(), utility::ipv4_hostip_val, port);
        }

        static auto get_ipv6_packet_gen(uint16_t port) -> std::unique_ptr<PacketGeneratorInterface>{

            return std::make_unique<PacketGenerator>(get_ipv6_id_generator(), utility::ipv6_hostip_val, port);
        }

        static auto get_retransmission_manager(timelapsed_t delay, size_t max_transmission) -> std::unique_ptr<RetransmissionManagerInterface>{

            using namespace std::chrono::literals; 

            const timelapsed_t MIN_DELAY        = utility::to_timelapsed(1s);
            const timelapsed_t MAX_DELAY        = utility::to_timelapsed(60s);
            const size_t MIN_MAX_RETRANSMISSION = 0u;
            const size_t MAX_MAX_RETRANSMISSION = 5u;
            const size_t HASHSET_CAP            = size_t{1} << 25; // 1 << 25 * 1 << 16 = 1 << 41 = 1 TB = 50s assuming 20GB / s 

            if (std::clamp(delay, MIN_DELAY, MAX_DELAY) != delay){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }
            
            if (std::clamp(max_transmission, MIN_MAX_RETRANSMISSION, MAX_MAX_RETRANSMISSION) != max_transmission){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<RetransmissionManager>(std::deque<std::pair<timepoint_t, Packet>>{},
                                                           data_structure::Factory::get_trivial_temporal_unordered_set<global_packet_id_t>(HASHSET_CAP),
                                                           delay, 
                                                           max_transmission,
                                                           std::make_unique<std::mutex>());
        } 

        static auto get_priority_packet_center() -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<PriorityPacketCenter>(dg::network_std_container::vector<Packet>{}, std::make_unique<std::mutex>());
        }

        static auto get_scheduled_packet_center(std::shared_ptr<SchedulerInterface> scheduler) -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<ScheduledPacketCenter>(dg::network_std_container::vector<ScheduledPacket>{}, scheduler, std::make_unique<std::mutex>());
        }

        static auto get_inbound_packet_center() -> std::unique_ptr<PacketCenterInterface>{

            const size_t HASHSET_CAP = size_t{1} << 25;
            return std::make_unique<InboundPacketCenter>(get_priority_packet_center(), 
                                                         data_structure::Factory::get_trivial_temporal_unordered_set<global_packet_id_t>(HASHSET_CAP), 
                                                         std::make_unique<std::mutex>());
        }

        static auto get_outbound_packet_center(std::shared_ptr<SchedulerInterface> scheduler) -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<OutboundPacketCenter>(get_priority_packet_center(), get_scheduled_packet_center(scheduler), std::make_unique<std::mutex>());
        }

        static auto get_exhaustion_controlled_packet_center(std::unique_ptr<PacketCenterInterface> base, size_t byte_capacity) -> std::unique_ptr<PacketCenterInterface>{

            size_t MIN_CAP  = constants::MAXIMUM_MSG_SIZE;
            size_t MAX_CAP  = std::max(MIN_CAP, size_t{1} << 40); 
            
            if (std::clamp(byte_capacity, MIN_CAP, MAX_CAP) != byte_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledPacketCenter>(size_t{0u}, byte_capacity, std::move(base), std::make_unique<std::mutex>());
        } 
    };
}

namespace dg::network_kernel_mailbox_impl1::worker{

    using namespace dg::network_kernel_mailbox_impl1::model; 
    
    class OutBoundWorker: public virtual dg::network_concurrency::WorkerInterface{
        
        private:

            std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center;
            std::shared_ptr<model::SocketHandle> socket;

        public:

            OutBoundWorker(std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center,
                           std::shared_ptr<model::SocketHandle> socket) noexcept: outbound_packet_center(std::move(outbound_packet_center)),
                                                                                  socket(std::move(socket)),
                                                                                  allocator(std::move(allocator)){}

            bool run_one_epoch() noexcept{

                std::optional<Packet> cur = this->outbound_packet_center->pop();

                if (!static_cast<bool>(cur)){
                    return false;
                } 

                cur->port_stamps.push_back(utility::unix_timestamp());
                size_t sz   = dg::network_compact_serializer::integrity_size(cur.value());
                auto buf    = dg::network_std_container::string(sz); 
                dg::network_compact_serializer::integrity_serialize_into(buf.data(), cur.value());
                exception_t err = socket_service::nonblocking_send(*this->socket, cur->to_addr, buf.data(), sz);
                
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

            bool run_one_epoch() noexcept{
                
                dg::network_std_container::vector<Packet> packets = this->retransmission_manager->get_retriables();

                if (packets.empty()){
                    return false;
                }

                for (Packet& packet: packets){
                    this->outbound_packet_center->push(packet);
                    this->retransmission_manager->add_retriable(std::move(packet));
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
        
        public:

            InBoundWorker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                          std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                          std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center,
                          std::shared_ptr<packet_controller::SchedulerInterface> scheduler,
                          std::shared_ptr<model::SocketHandle> socket) noexcept: retransmission_manager(std::move(retransmission_manager)),
                                                                                    ob_packet_center(std::move(ob_packet_center)),
                                                                                    ib_packet_center(std::move(ib_packet_center)),
                                                                                    scheduler(std::move(scheduler)),
                                                                                    socket(std::move(socket)){}
            
            bool run_one_epoch() noexcept{
                
                model::Packet pkt   = {};
                size_t sz           = {};
                auto buf            = dg::network_std_container::string(constants::MAXIMUM_MSG_SIZE);
                exception_t err     = socket_service::blocking_recv(*this->socket, buf.data(), sz, constants::MAXIMUM_MSG_SIZE);

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_optional_fast(dg::network_exception::verbose(err));
                    return false;
                }

                auto expt_nxt_ptr   = dg::network_compact_serializer::integrity_deserialize_into(buf.data(), pkt, sz); 
                
                if (!expt_nxt_ptr.has_value()){
                    dg::network_log_stackdump::error_optional_fast(dg::network_exception::verbose(expt_nxt_ptr.error()));
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
                                          std::shared_ptr<model::SocketHandle> socket) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            return std::make_unique<OutBoundWorker>(std::move(outbound_packet_center), std::move(socket));
        }

        static auto spawn_retransmission_worker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager, 
                                                std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            return std::make_unique<RetransmissionWorker>(std::move(retransmission_manager), std::move(outbound_packet_center));
        }

        static auto spawn_inbound_worker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager, 
                                         std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                                         std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center, 
                                         std::shared_ptr<packet_controller::SchedulerInterface> scheduler, 
                                         std::shared_ptr<SocketHandle> socket) -> std::unique_ptr<Workable>{

            return std::make_unique<InBoundWorker>(std::move(retransmission_manager), std::move(ob_packet_center), 
                                                   std::move(ib_packet_center), std::move(scheduler), 
                                                   std::move(socket))>;
        }
    };
}

namespace dg::network_kernel_mailbox_impl1::core{

    class RetransmittableMailBoxController: public virtual MailboxInterface{

        private:

            dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
            std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen;
            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager;
            std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center;
            std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center;

        public:

            RetransmittableMailBoxController(dg::network_std_container::vector<std::unique_ptr<dg::network_concurrency::daemon_raii_handle_t>> daemons, 
                                             std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen,
                                             std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                                             std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                                             std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center) noexcept: daemons(std::move(daemons)),
                                                                                                                                   packet_gen(std::move(packet_gen)),
                                                                                                                                   retransmission_manager(std::move(retransmission_manager)),
                                                                                                                                   ob_packet_center(std::move(ob_packet_center)),
                                                                                                                                   ib_packet_center(std::move(ib_packet_center)){}

            void send(Address addr, dg::network_std_container::string buf) noexcept{

                model::Packet pkt = this->packet_gen->get(std::move(addr), std::move(buf));
                this->ob_packet_center->push(pkt);
                this->retransmission_manager->add_retriable(std::move(pkt));
            }

            auto recv() noexcept -> std::optional<dg::network_std_container::string>{

                std::optional<Packet> pkt = this->ib_packet_center->pop();

                if (!static_cast<bool>(pkt)){
                    return std::nullopt;
                }

                return std::move(pkt->content);
            }
    };

    struct ComponentFactory{

        static auto spawn_retransmittable_mailbox_controller(dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemons, 
                                                             std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen,
                                                             std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                                                             std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                                                             std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center) -> std::unique_ptr<MailBoxInterface>{
            
            return std::make_unique<RetransmittableMailBoxController>(std::move(daemons), std::move(packet_gen), 
                                                                      std::move(retransmission_manager), std::move(ob_packet_center),
                                                                      std::move(ib_packet_center));
        }

    };
}

namespace dg::network_kernel_mailbox_impl1{

    struct RuntimeRTTScheduler{
        double rtt_epsilon;
        double min_frequency;
        double max_frequency;
        double learning_rate;
        double max_q_time;
    };

    struct ASAPScheduler{};

    struct Config{
        size_t outbound_worker_count;
        size_t inbound_worker_count;
        size_t retransmission_worker_count; 
        int sin_fam;  
        int comm;
        int protocol;
        uint16_t port;
        uint64_t retransmission_delay_in_nanosecond; 
        size_t retransmission_count;
        std::variant<RuntimeRTTScheduler, ASAPScheduler> scheduler;
        std::optional<size_t> inbound_exhaustion_control_sz;
        std::optional<size_t> outbound_exhaustion_control_sz;
    };

    auto spawn(Config config) -> std::unique_ptr<core::MailboxInterface>{
        
        dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemons{};
        std::shared_ptr<model::SocketHandle> sock_handle{}; 
        std::shared_ptr<packet_controller::SchedulerInterface> scheduler{};
        std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager{};
        std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen{};
        std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center{};
        std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center{}; 

        if (std::holds_alternative<RuntimeRTTScheduler>(config.scheduler)){
            auto sched_config = std::get<RuntimeRTTScheduler>(config.scheduler); 
            scheduler = packet_controller::ComponentFactory::get_std_scheduler(sched_config.max_frequency, sched_config.min_frequency,
                                                                               sched_config.learning_rate, sched_config.rtt_epsilon, 
                                                                               sched_config.max_q_time);
        } else if (std::holds_alternative<ASAPScheduler>(config.scheduler)){
            scheduler = packet_controller::ComponentFactory::get_asap_scheduler();
        } else{
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }
        
        if (config.sin_fam == AF_INET){
            packet_gen = packet_controller::ComponentFactory::get_ipv4_packet_gen();
        } else if (config.sin_fam == AF_INET6){
            packet_gen = packet_controller::ComponentFactory::get_ipv6_packet_gen();
        } else{
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }

        retransmission_manager = packet_controller::ComponentFactory::get_transmission_manager(config.retransmission_delay_in_nanosecond, config.retransmission_count);

        if (config.inbound_exhaustion_control_sz){
            ib_packet_center = packet_controller::ComponentFactory::get_exhaustion_controlled_packet_center(packet_controller::ComponentFactory::get_inbound_packet_center(), config.inbound_exhaustion_control_sz.value());
        } else{
            ib_packet_center = packet_controller::ComponentFactory::get_inbound_packet_center();
        }

        if (config.outbound_exhaustion_control_sz){
            ob_packet_center = packet_controller::ComponentFactory::get_exhaustion_controlled_packet_center(packet_controller::ComponentFactory::get_outbound_packet_center(scheduler), config.outbound_exhaustion_control_sz.value());
        } else{
            ob_packet_center = packet_controller::ComponentFactory::get_outbound_packet_center(scheduler);
        }

        if (config.protocol != SOCK_DGRAM){
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }

        sock_handle = dg::network_exception_handler::throw_nolog(socket_service::open_socket(config.sin_fam, config.comm, config.protocol));
        dg::network_exception_handler::throw_nolog(socket_service::bind_socket_to_port(*sock_handle, config.port));

        for (size_t i = 0u; i < config.outbound_worker_count; ++i){
            auto worker_ins     = worker::ComponentFactory::spawn_outbound_worker(ob_packet_center, sock_handle);
            auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker_ins)));
            daemons.push_back(std::move(daemon_handle));
        }

        for (size_t i = 0u; i < config.inbound_worker_count; ++i){
            auto worker_ins     = worker::ComponentFactory::spawn_inbound_worker(retransmission_manager, ob_packet_center, ib_packet_center, scheduler, sock_handle);
            auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker_ins)));
            daemons.push_back(std::move(daemon_handle));
        }

        for (size_t i = 0u; i < config.retransmission_worker_count; ++i){
            auto worker_ins     = worker::ComponentFactory::spawn_retransmission_worker(rm, ob_packet_center);
            auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker_ins)));
            daemons.push_back(std::move(daemon_handle));
        }

        return core::ComponentFactory::spawn_retransmittable_mailbox_controller(std::move(daemons), std::move(packet_gen), 
                                                                                std::move(retransmission_manager), std::move(ob_packet_center), 
                                                                                std::move(ib_packet_center));
    }

    auto cspawn(Config config) noexcept -> std::expected<std::unique_ptr<core::MailboxInterface>, exception_t>{

        auto functor = dg::network_exception::to_cstyle_function(spawn);
        return functor(std::move(config));
    }
}

#endif