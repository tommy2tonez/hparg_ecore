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

namespace dg::network_kernel_mailbox_ack::types{

    using global_packet_id_t     = uint64_t;
    using local_packet_id_t      = uint32_t;
    using ip_t                   = std::string;
    using port_t                 = uint16_t;
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

    struct Packet{
        ip_t fr_addr;
        ip_t to_addr; 
        global_packet_id_t id;
        uint8_t retry_count;
        uint8_t priority;
        uint8_t taxonomy; 
        packet_content_t content;
        std::vector<timepoint_t> port_stamps;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(fr_addr, to_addr, id, retry_count, priority, taxonomy, content, port_stamps);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(fr_addr, to_addr, id, retry_count, priority, taxonomy, content, port_stamps);
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

    //consider runtimize these and precond - in future when deems appropriate (placeholder for now) 
    static inline constexpr auto SIN_FAMLIY                             = AF_INET;  
    static inline constexpr auto COMM_TYPE                              = SOCK_DGRAM;
    static inline constexpr auto PORT                                   = uint16_t{65530}; 
    static inline constexpr auto MAXIMUM_MSG_SIZE                       = size_t{1} << 12;  
    static inline constexpr auto RETRANSMISSION_DELAY_TIME              = uint64_t{duration_cast<nanoseconds>(4096ms).count()};
    static inline constexpr auto RETRANSMISSION_COUNT                   = uint8_t{30};

    static inline constexpr auto OUTBOUND_SCHEDULE_MIN_FREQUENCY        = double{10000};
    static inline constexpr auto OUTBOUND_SCHEDULE_MAX_FREQUENCY        = double{10000000};
    static inline constexpr auto OUTBOUND_SCHEDULE_LEARNING_RATE        = double{0.20}; //anomaly dampener
    static inline constexpr auto OUTBOUND_SCHEDULE_MAX_Q_TIME           = uint64_t{duration_cast<nanoseconds>(1s).count()};

    static inline constexpr auto WORKER_BREAK_INTERVAL                  = 128ms;
}

namespace dg::network_kernel_mailbox_ack::memory{

    class Allocatable{

        public:

            virtual ~Allocatable() noexcept = default;
            virtual char * malloc(size_t) noexcept = 0; //extend responsibility of malloc here - force noexcept - recoverability is component's optional responsibility
            virtual void free(void *) noexcept = 0;
    };
} 

namespace dg::network_kernel_mailbox_ack::packet_controller{

    using namespace dg::network_kernel_mailbox_ack::types;
    using namespace dg::network_kernel_mailbox_ack::model;
    
    //this is too tedious 
    //for now - assume that global memory exhaustion is not a recoverable error
    //assume local memory exhaustion is temporary - recoverable
    //noexcept all of the function that does not return an error 

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
            virtual void add_retriable(Packet) noexcept = 0; //should have an extension to solve memory_exhaustion error (spin until pushable or drop) - assume that the container is leaky (that at least one worker is popping the package_center)
            virtual void ack(global_packet_id_t) noexcept = 0;
            virtual auto get_retriables() noexcept -> std::vector<Packet> = 0;
    };

    class PacketCenterInterface{
        
        public:
            
            virtual ~PacketCenterInterface() noexcept = default;
            virtual void push(Packet) noexcept = 0; //should have an extension to solve memory_exhaustion error (spin until pushable or drop) - assume that the container is leaky (that at least one worker is popping the package_center)
            virtual auto pop() noexcept -> std::optional<Packet> = 0;
    };
}

namespace dg::network_kernel_mailbox_ack::worker{

    using namespace dg::network_kernel_mailbox_ack::model;

    class Workable{

        public:
        
            virtual ~Workable() noexcept = default;
    };
}

namespace dg::network_kernel_mailbox_ack::core{

    using namespace dg::network_kernel_mailbox_ack::types;
    using namespace dg::network_kernel_mailbox_ack::model;

    class MailboxInterface{
        
        public: 

            virtual ~MailboxInterface() noexcept = default;
            virtual void send(ip_t, packet_content_t) noexcept = 0;
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

    static auto ipv4toi(std::string ip) noexcept -> unsigned int{

        static_assert(sizeof(unsigned int) == 4u);
        static_assert(CHAR_BIT == 8u);

        const char * frmt = "%u.%u.%u.%u";
        unsigned int q1{};
        unsigned int q2{};
        unsigned int q3{};
        unsigned int q4{};
        
        if (sscanf(ip.c_str(), frmt, &q1, &q2, &q3, &q4) == EOF){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        unsigned int combined = q1 | q2 | q3 | q4;

        if (combined > std::numeric_limits<unsigned char>::max()){
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

    static inline auto ipv4_hostip_val = ipv4_hostip(); 

    static auto get_time_since_epoch() noexcept -> timepoint_t{

        using namespace std::chrono;
        return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
    }

    static auto subtract_timepoint(timepoint_t tp, timelapsed_t dur) noexcept -> timepoint_t{

        return tp - dur;
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
        server.sin_port         = htons(port); //malicious

        if (bind(sock, (struct sockaddr *) &server, sizeof(server)) == -1){
            return dg::network_exception::wrap_kernel_exception(errno);
        }

        return dg::network_exception::SUCCESS;
    }

    static auto nonblocking_send(SocketHandle sock, ip_t to_addr, const void * buf, size_t sz) noexcept -> exception_t{

        struct sockaddr_in server{};
        
        if constexpr(DEBUG_MODE_FLAG){
            if (sz > constants::MAXIMUM_MSG_SIZE){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        auto legacy_to_addr = extract_legacy_addr(to_addr);
        auto to_port        = extract_port(to_addr);

        if (inet_pton(sock.sin_fam, legacy_to_addr.data(), &server.sin_addr) == -1){ //
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

    static auto blocking_recv(SocketHandle sock, std::shared_ptr<memory::Allocatable> allocator) noexcept -> std::expected<std::pair<std::shared_ptr<char[]>, size_t>, exception_t>{

        struct sockaddr_in from{};
        unsigned int length         = sizeof(struct sockaddr_in);
        std::shared_ptr<char[]> buf = utility::malloc(constants::MAXIMUM_MSG_SIZE, allocator); 
        auto n                      = recvfrom(sock, buf.get(), constants::MAXIMUM_MSG_SIZE, 0, (struct sockaddr *) &from, &length);

        // if (n == -1){
        //     throw std::exception{};
        // }
        
        // return std::make_pair(std::move(buf), n);
    }
}

namespace dg::network_kernel_mailbox_ack::packet_service{

    using namespace dg::network_kernel_mailbox_ack::types;

    static auto make_ipv4_request(ip_t to_addr, global_packet_id_t id, types::packet_content_t content) noexcept -> model::Packet{

        model::Packet pkt{};

        pkt.fr_addr     = utility::ipv4_hostip_val;
        pkt.to_addr     = to_addr;
        pkt.id          = id;
        pkt.content     = std::move(content);
        pkt.priority    = 0u;
        pkt.retry_count = 0u;
        pkt.taxonomy    = constants::request;
        pkt.port_stamps = {};

        return pkt;
    } 

    static auto make_ipv6_request(ip_t to_addr, global_packet_id_t id, types::packet_content_t content) noexcept -> model::Packet{

    }

    static auto make_request(ip_t to_addr, global_packet_id_t id, types::packet_content_t content) noexcept -> model::Packet{

        if (utility::is_ipv4_addr(to_addr)){
            return make_ipv4_request(std::move(to_addr), id, std::move(content));
        }

        if (utility::is_ipv6_addr(to_addr)){
            return make_ipv6_request(std::move(to_addr), id, std::move(content));
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }
    
    static auto request_to_ack(model::Packet pkt) noexcept -> model::Packet{

        //assume ack was carried along with request thus properties remain
        std::swap(pkt.fr_addr, pkt.to_addr);
        pkt.content  = {};
        pkt.taxonomy = constants::rts_ack; 
        return pkt;
    } 

    static auto get_transit_time(const model::Packet& pkt) noexcept -> types::timelapsed_t{

        if (pkt.port_stamps.size() % 2 != 0){
            throw std::exception{};
        }

        types::timelapsed_t lapsed{};

        for (size_t i = 1; i < pkt.port_stamps.size(); i += 2){
            lapsed += pkt.port_stamps[i] - pkt.port_stamps[i - 1];
        }

        return lapsed;
    }
}

namespace dg::network_kernel_mailbox_ack::packet_controller{
    
    //literally straight out of google research paper 
    class StdScheduler: public virtual SchedulerInterface{

        private:

            std::unordered_map<ip_t, timepoint_t> last_sched;
            std::unordered_map<ip_t, double> frequency;
            std::unordered_map<ip_t, timelapsed_t> min_rtt;
            std::unordered_map<ip_t, timelapsed_t> total_rtt;
            std::unordered_map<ip_t, size_t> rtt_count;
            std::unique_ptr<std::mutex> mtx;

        public:

            StdScheduler(std::unordered_map<ip_t, timepoint_t> last_sched, 
                         std::unordered_map<ip_t, double> frequency,
                         std::unordered_map<ip_t, timelapsed_t> min_rtt,
                         std::unordered_map<ip_t, timelapsed_t> total_rtt,
                         std::unordered_map<ip_t, size_t> rtt_count,
                         std::unique_ptr<std::mutex> mtx) noexcept: last_sched(std::move(last_sched)),
                                                                    frequency(std::move(frequency)),
                                                                    min_rtt(std::move(min_rtt)),
                                                                    total_rtt(std::move(total_rtt)),
                                                                    rtt_count(std::move(rtt_count)),
                                                                    mtx(std::move(mtx)){}

            auto schedule(ip_t ip) noexcept -> timepoint_t{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                
                if (this->frequency.find(ip) == this->frequency.end()){
                    this->frequency[ip] = constants::OUTBOUND_SCHEDULE_MAX_FREQUENCY;
                }

                if (this->last_sched.find(ip) == this->last_sched.end()){
                    this->last_sched[ip] = utility::get_time_since_epoch();
                }

                auto tentative_sched    = this->last_sched[ip] + utility::frequency_to_period(this->frequency[ip]);
                auto MIN_SCHED          = utility::get_time_since_epoch();
                auto MAX_SCHED          = MIN_SCHED + constants::OUTBOUND_SCHEDULE_MAX_Q_TIME;
                this->last_sched[ip]    = std::clamp(tentative_sched, MIN_SCHED, MAX_SCHED);

                return this->last_sched[ip];
            }

            void feedback(ip_t ip, timelapsed_t lapsed) noexcept{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                const double epsilon = double{0.00001};

                if (this->frequency.find(ip) == this->frequency.end()){
                    this->frequency[ip] = constants::OUTBOUND_SCHEDULE_MAX_FREQUENCY;
                }

                if (this->min_rtt.find(ip) == this->min_rtt.end()){
                    this->min_rtt[ip] = lapsed;
                }
         
                if (lapsed < this->min_rtt[ip]){
                    this->min_rtt[ip] += (lapsed - this->min_rtt[ip]) * constants::OUTBOUND_SCHEDULE_LEARNING_RATE; 
                }

                this->total_rtt[ip] += lapsed;
                this->rtt_count[ip] += 1;

                double avg_rtt      = this->total_rtt[ip] / this->rtt_count[ip]; 
                double dist         = std::max(epsilon, (avg_rtt - this->min_rtt[ip]) * 2);
                double perc         = static_cast<double>(lapsed - this->min_rtt[ip]) / dist; 
                double f            = constants::OUTBOUND_SCHEDULE_MAX_FREQUENCY - (constants::OUTBOUND_SCHEDULE_MAX_FREQUENCY - constants::OUTBOUND_SCHEDULE_MIN_FREQUENCY) * perc;

                this->frequency[ip] += (f - this->frequency[ip]) * constants::OUTBOUND_SCHEDULE_LEARNING_RATE; 
                this->frequency[ip] = std::clamp(this->frequency[ip], constants::OUTBOUND_SCHEDULE_MIN_FREQUENCY, constants::OUTBOUND_SCHEDULE_MAX_FREQUENCY);
            }
        
    };  

    class IPv4IDGenerator: public virtual IDGeneratorInterface{

        private:

            local_packet_id_t pkt_count;
            std::unique_ptr<std::mutex> mtx;

        public:

            IPv4IDGenerator(local_packet_id_t pkt_count,
                            std::unique_ptr<std::mutex> mtx) noexcept: pkt_count(pkt_count),
                                                                       mtx(std::move(mtx)){}

            auto get() noexcept -> global_packet_id_t{

                static_assert(std::is_unsigned_v<global_packet_id_t>);
                static_assert(sizeof(global_packet_id_t) == 8u);
                static_assert(sizeof(local_packet_id_t) == 4u);

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                return (static_cast<global_packet_id_t>(this->pkt_count++) << 32) | static_cast<global_packet_id_t>(utility::ipv4toi(utility::ipv4_hostip_val));
            }
    };

    class ConcurrentIPv4IDGenerator: public virtual IDGeneratorInterface{

        private:

            std::vector<local_packet_id_t> pkt_vec_count;
        
        public:

            ConcurrentIPv4IDGenerator(std::vector<local_packet_id_t> pkt_vec_count) noexcept: pkt_vec_count(std::move(pkt_vec_count)){}

            auto get() noexcept -> global_packet_id_t{
                
                static_assert(std::is_unsigned_v<global_packet_id_t>);
                static_assert(sizeof(global_packet_id_t) == 8u);
                static_assert(sizeof(local_packet_id_t) == 4u);

                size_t thread_idx = dg::network_concurrency::this_thread_idx();
                return (static_cast<global_packet_id_t>(this->pkt_vec_count[thread_idx]++) << 32) | static_cast<global_packet_id_t>(utility::ipv4toi(utility::ipv4_hostip_val));
            }
    };

    class RetransmissionManager: public virtual RetransmissionManagerInterface{

        private:

            std::deque<std::pair<timepoint_t, Packet>> pkt_map;
            std::unordered_set<global_packet_id_t> acked_id;
            std::unique_ptr<std::mutex> mtx;

        public:

            RetransmissionManager(std::deque<std::pair<timepoint_t, Packet>> pkt_map,
                                  std::unordered_set<global_packet_id_t> acked_id,
                                  std::unique_ptr<std::mutex> mtx) noexcept: pkt_map(std::move(pkt_map)),
                                                                             acked_id(std::move(acked_id)),
                                                                             mtx(std::move(mtx)){}

            void add_retriable(Packet pkt) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto ts         = utility::get_time_since_epoch();
                this->pkt_map.push_back(std::make_pair(ts, std::move(pkt))); 
            } 

            void ack(global_packet_id_t pkt_id_t) noexcept{
                
                auto lck_grd    = std::lock_guard<std::mutex>(this->mtx);
                this->acked_id.insert(pkt_id_t);
            }

            auto get_retriables() noexcept -> std::vector<Packet>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto ts         = utility::get_time_since_epoch();
                auto lb_key     = std::make_pair(utility::subtract_timepoint(ts, constants::RETRANSMISSION_DELAY_TIME), Packet{});
                auto last       = std::lower_bound(this->pkt_map.begin(), this->pkt_map.end(), lb_key, [](const auto& lhs, const auto& rhs){return lhs.first < rhs.first;});
                auto rs         = std::vector<Packet>(); 

                for (auto it = this->pkt_map.begin(); it != last; ++it){
                    if (this->acked_id.find(it->second.id) == this->acked_id.end()){
                        rs.push_back(it->second); 
                    }
                }

                this->pkt_map.erase(this->pkt_map.begin(), last);
                return rs;
            }
    };
    
    //TODOs:
    //pkt_ack center should be accumulating to reduce pkt send
    //should have a PacketCenter extension that do in/out byte_sz control to drop packet
    //mutex should be an extension rather than a component's responsibility (single responsibility)
    //add string customized allocator
    //avoid shared_ptr copy assignment
    //improve scheduler by doing direct NIC timestamp
    //avoid mtx by doing affinity assignment
    //doing calibration to find optimal packet size for maximum thruput
    //affinity assignment for network workers
    //customized kernel scheduler implementation

    class PriorityPacketCenter: public virtual PacketCenterInterface{

        private:
            
            std::vector<Packet> pkts;
            std::unique_ptr<std::mutex> mtx;

        public:

            PriorityPacketCenter(std::vector<Packet> pkts,
                                 std::unique_ptr<std::mutex>) noexcept: pkts(std::move(pkts)),
                                                                        mtx(std::move(mtx)){}

            void push(Packet pkt) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto less       = [](const Packet& lhs, const Packet& rhs){return lhs.priority < rhs.priority;};
                this->pkts.push_back(std::move(pkt)); 
                std::push_heap(this->pkts.begin(), this->pkts.end(), less);
            }     

            std::optional<Packet> pop() noexcept{
                
                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);

                if (this->pkts.empty()){
                    return std::nullopt;
                }
                
                auto less = [](const Packet& lhs, const Packet& rhs){return lhs.priority < rhs.priority;};
                std::pop_heap(this->pkts.begin(), this->pkts.end(), less);
                auto rs = std::move(this->pkts.back());
                this->pkts.pop_back();

                return rs;
            }   
    };

    class ScheduledPacketCenter: public virtual PacketCenterInterface{

        private:

            std::vector<ScheduledPacket> pkts;
            std::shared_ptr<SchedulerInterface> scheduler;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            ScheduledPacketCenter(std::vector<ScheduledPacket> pkts, 
                                  std::shared_ptr<SchedulerInterface> scheduler,
                                  std::unique_ptr<std::mutex> mtx) noexcept: pkts(std::move(pkts)),
                                                                             scheduler(std::move(scheduler)),
                                                                             mtx(std::move(mtx)){}
            
            void push(Packet pkt) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto appender   = ScheduledPacket{std::move(pkt), this->scheduler->schedule(pkt.to_addr)};
                auto greater    = [](const auto& lhs, const auto& rhs){return lhs.sched_time > rhs.sched_time;};
                this->pkts.push_back(std::move(appender));
                std::push_heap(this->pkts.begin(), this->pkts.end(), greater);
            }

            auto pop() noexcept -> std::optional<Packet>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);

                if (this->pkts.empty()){
                    return std::nullopt;
                }

                if (this->pkts.front().sched_time > utility::get_time_since_epoch()){
                    return std::nullopt;
                }

                auto greater    = [](const auto& lhs, const auto& rhs){return lhs.sched_time > rhs.sched_time;};
                std::pop_heap(this->pkts.begin(), this->pkts.end(), greater);
                auto rs = std::move(this->pkts.back().pkt);
                this->pkts.pop_back();

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

            std::optional<Packet> pop() noexcept{

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
            std::unordered_set<global_packet_id_t> id_set;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            InboundPacketCenter(std::unique_ptr<PacketCenterInterface> base_pkt_center, 
                                std::unordered_set<global_packet_id_t> id_set,
                                std::unique_ptr<std::mutex> mtx) noexcept: base_pkt_center(std::move(base_pkt_center)),
                                                                           id_set(std::move(id_set)),
                                                                           mtx(std::move(mtx)){}

            void push(Packet pkt) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(this->mtx);

                if (this->id_set.find(pkt.id) != this->id_set.end()){
                    return;
                }

                this->id_set.insert(pkt.id); //memory-exhaustion - require timestamp to clear  
                this->base_pkt_center->push(std::move(pkt));
            }

            std::optional<Packet> pop() noexcept{

                auto lck_grd = dg::network_genult::lock_guard(this->mtx);
                return this->base_pkt_center->pop();
            }
    };

    struct ComponentFactory{

        static auto get_std_scheduler() -> std::unique_ptr<SchedulerInterface>{

            return std::make_unique<StdScheduler>(std::unordered_map<ip_t, timepoint_t>{}, 
                                                  std::unordered_map<ip_t, double>{},
                                                  std::unordered_map<ip_t, timelapsed_t>{},
                                                  std::unordered_map<ip_t, timelapsed_t>{},
                                                  std::unordered_map<ip_t, size_t>{});
        } 

        static auto get_id_generator() -> std::unique_ptr<IDGeneratorInterface>{

            return std::make_unique<IPv4IDGenerator>(0u);
        }

        static auto get_retransmission_manager() -> std::unique_ptr<RetransmissionManagerInterface>{

            return std::make_unique<RetransmissionManager>(std::deque<std::pair<timepoint_t, Packet>>{}, std::unordered_set<global_packet_id_t>{});
        }

        static auto get_priority_packet_center() -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<PriorityPacketCenter>(std::vector<Packet>{});
        }

        static auto get_scheduled_packet_center(std::shared_ptr<SchedulerInterface> scheduler) -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<ScheduledPacketCenter>(std::vector<ScheduledPacket>{}, scheduler);
        }

        static auto get_inbound_packet_center() -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<InboundPacketCenter>(get_priority_packet_center(), std::unordered_set<global_packet_id_t>{});
        }

        static auto get_outbound_packet_center(std::shared_ptr<SchedulerInterface> scheduler) -> std::unique_ptr<PacketCenterInterface>{

            return std::make_unique<OutboundPacketCenter>(get_priority_packet_center(), get_scheduled_packet_center(scheduler));
        }
    };
}

namespace dg::network_kernel_mailbox_ack::worker{

    class OutBoundWorker: public virtual Workable{
        
        private:

            std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center;
            std::shared_ptr<model::SocketHandle> socket;
            std::shared_ptr<std::atomic<bool>> poison_pill;
            std::shared_ptr<memory::Allocatable> allocator;

        public:

            OutBoundWorker(std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center,
                           std::shared_ptr<model::SocketHandle> socket,
                           std::shared_ptr<std::atomic<bool>> poison_pill,
                           std::shared_ptr<memory::Allocatable> allocator): outbound_packet_center(std::move(outbound_packet_center)),
                                                                            socket(std::move(socket)),
                                                                            poison_pill(std::move(poison_pill)),
                                                                            allocator(std::move(allocator)){}

            void operator()() noexcept{

                while (!this->poison_pill_load()){
                    std::optional<Packet> cur = this->outbound_packet_center->pop();

                    if (!static_cast<bool>(cur)){
                        std::this_thread::sleep_for(constants::WORKER_BREAK_INTERVAL); //fine - need fair scheduler from kernel - or reimplementation if necessary - spinlock is bad practice in this case
                        continue;
                    } 

                    cur->port_stamps.push_back(utility::get_time_since_epoch());
                    size_t sz       = dg::compact_serializer::core::integrity_count(cur.value());
                    auto buf        = utility::malloc(sz, this->allocator);
                    dg::compact_serializer::core::integrity_serialize(cur.value(), buf.get());
                    exception_t err = socket_service::nonblocking_send(*this->socket, cur->to_addr, buf.get(), sz); //exception_handling is optional - exception could be solved by using heartbeat broadcast approach (a component is good if all other heartbeats are good) - optional log here
                    dg::network_log_stackdump::error_optional(dg::network_exception::verbose(err));
                }
            }
        
        private:

            bool poison_pill_load() noexcept -> bool{

                //do randomization to reduce atomic read
            }
    };

    class RetryWorker: public virtual Workable{

        private:

            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retry_manager;
            std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center;
            std::shared_ptr<std::atomic<bool>> poison_pill;

        public:

            RetryWorker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retry_manager,
                        std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center,
                        std::shared_ptr<std::atomic<bool>> poison_pill) noexcept: retry_manager(std::move(retry_manager)),
                                                                                  outbound_packet_center(std::move(outbound_packet_center)),
                                                                                  poison_pill(std::move(poison_pill)){}

            void operator()() noexcept{
                
                while (!this->poison_pill_load()){
                    std::vector<Packet> retry_packets = this->retry_manager->get_retriables();

                    if (retry_packets.empty()){
                        std::this_thread::sleep_for(constants::WORKER_BREAK_INTERVAL);
                        continue;
                    }

                    for (auto& e: retry_packets){
                        if (e.retry_count == constants::RETRANSMISSION_COUNT){
                            dg::network_log_stackdump::error_optional(dg::network_exception::verbose(dg::network_exception::LOST_RETRANMISSION));
                        } else{
                            e.retry_count += 1;
                            e.priority = e.retry_count; //priority -
                            this->outbound_packet_center->push(e);
                            this->retry_manager->add_retriable(e);
                        }
                    }
                }
            }
        
        private:

            bool poison_pill_load() noexcept -> bool{

                //do randomization to reduce atomic load
            }
    };

    class InBoundWorker: public virtual Workable{

        private:

            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retry_manager;
            std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center;
            std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center;
            std::shared_ptr<packet_controller::SchedulerInterface> scheduler;
            std::shared_ptr<model::SocketHandle> socket;
            std::shared_ptr<std::atomic<bool>> poison_pill;
            std::shared_ptr<memory::Allocatable> allocator;
        
        public:

            InBoundWorker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retry_manager,
                          std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                          std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center,
                          std::shared_ptr<packet_controller::SchedulerInterface> scheduler,
                          std::shared_ptr<model::SocketHandle> socket,
                          std::shared_ptr<std::atomic<bool>> poison_pill,
                          std::shared_ptr<memory::Allocatable> allocator) noexcept: retry_manager(std::move(retry_manager)),
                                                                                    ob_packet_center(std::move(ob_packet_center)),
                                                                                    ib_packet_center(std::move(ib_packet_center)),
                                                                                    scheduler(std::move(scheduler)),
                                                                                    socket(std::move(socket)),
                                                                                    poison_pill(std::move(poison_pill)),
                                                                                    allocator(std::move(allocator)){}
            
            void operator()() noexcept{
                
                while (!this->poison_pill_load()){
                    auto pkt                = model::Packet{};
                    auto [incoming, len]    = socket_service::blocking_recv(*this->socket, this->allocator);
                    exception_t err         = dg::compact_serializer::core::integrity_deserialize(incoming.get(), len, pkt); 
                    
                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_optional(dg::network_exception::verbose(err));
                        continue;
                    }

                    pkt.port_stamps.push_back(utility::get_time_since_epoch());

                    if (pkt.taxonomy == constants::rts_ack){
                        this->retry_manager->ack(pkt.id);
                        this->scheduler->feedback(pkt.fr_addr, packet_service::get_transit_time(pkt));
                    } else if (pkt.taxonomy == constants::request){
                        this->ib_packet_center->push(pkt);
                        pkt = packet_service::request_to_ack(pkt); 
                        this->ob_packet_center->push(pkt);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            (void) err;
                        }
                    }
                }
            }
        
        private:

            bool poison_pill_load() noexcept -> bool{

                //do randomization to reduce atomic load
            }
    };

    class Supervisor: public virtual Workable{

        private:

            std::unique_ptr<std::thread> thread_ins;
            std::unique_ptr<Workable> worker;    
            std::shared_ptr<std::atomic<bool>> poison_pill;

        public:

            Supervisor(std::unique_ptr<std::thread> thread_ins, 
                       std::unique_ptr<Workable> worker, 
                       std::shared_ptr<std::atomic<bool>> poison_pill) noexcept: thread_ins(std::move(thread_ins)),
                                                                                 worker(std::move(worker)),
                                                                                 poison_pill(std::move(poison_pill)){}

            ~Supervisor() noexcept{

                this->poison_pill->exchange(true, std::memory_order_release);
                this->thread_ins->join();
            }    
    };

    struct ComponentFactory{

        static auto spawn_outbound_worker(std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center, 
                                          std::shared_ptr<int> socket, 
                                          std::shared_ptr<std::atomic<uint8_t>> status,
                                          std::shared_ptr<memory::Allocatable> allocator) -> std::unique_ptr<Workable>{
            
            auto poison_pill    = std::make_shared<std::atomic<bool>>(false);
            auto worker         = std::make_unique<OutBoundWorker>(outbound_packet_center, socket, status, poison_pill, allocator); 
            auto task           = [ins = worker.get()]() noexcept{ins->operator()();};
            auto thread_ins     = std::make_unique<std::thread>(task); 
            auto supervisor     = std::make_unique<Supervisor>(std::move(thread_ins), std::move(worker), poison_pill);

            return supervisor;
        }

        static auto spawn_retry_worker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retry_manager, 
                                       std::shared_ptr<packet_controller::PacketCenterInterface> outbound_packet_center, 
                                       std::shared_ptr<std::atomic<uint8_t>> status) -> std::unique_ptr<Workable>{
            
            auto poison_pill    = std::make_shared<std::atomic<bool>>(false);
            auto worker         = std::make_unique<RetryWorker>(retry_manager, outbound_packet_center, status, poison_pill);
            auto task           = [ins = worker.get()]() noexcept{ins->operator()();};
            auto thread_ins     = std::make_unique<std::thread>(task);
            auto supervisor     = std::make_unique<Supervisor>(std::move(thread_ins), std::move(worker), poison_pill);

            return supervisor;
        }

        static auto spawn_inbound_worker(std::shared_ptr<packet_controller::RetransmissionManagerInterface> retry_manager, 
                                         std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                                         std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center, 
                                         std::shared_ptr<packet_controller::SchedulerInterface> scheduler, 
                                         std::shared_ptr<int> socket,
                                         std::shared_ptr<std::atomic<uint8_t>> status, 
                                         std::shared_ptr<memory::Allocatable> allocator) -> std::unique_ptr<Workable>{
            
            auto poison_pill    = std::make_shared<std::atomic<bool>>(false);
            auto worker         = std::make_unique<InBoundWorker>(retry_manager, ob_packet_center, ib_packet_center, scheduler, socket, status, poison_pill, allocator);
            auto task           = [ins = worker.get()]() noexcept{ins->operator()();};
            auto thread_ins     = std::make_unique<std::thread>(task);
            auto supervisor     = std::make_unique<Supervisor>(std::move(thread_ins), std::move(worker), poison_pill);

            return supervisor;
        }
    };
}

namespace dg::network_kernel_mailbox_ack::core{

    class StdMailBoxController: public virtual MailboxInterface{

        private:

            std::vector<std::unique_ptr<worker::Workable>> workers;
            std::unique_ptr<packet_controller::IDGeneratorInterface> id_gen;
            std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager;
            std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center;
            std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center;

        public:

            StdMailBoxController(std::vector<std::unique_ptr<worker::Workable>> workers, 
                                 std::unique_ptr<packet_controller::IDGeneratorInterface> id_gen,
                                 std::shared_ptr<packet_controller::RetransmissionManagerInterface> retransmission_manager,
                                 std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                                 std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center) noexcept: workers(std::move(workers)),
                                                                                                                       id_gen(std::move(id_gen)),
                                                                                                                       retransmission_manager(std::move(retransmission_manager)),
                                                                                                                       ob_packet_center(std::move(ob_packet_center)),
                                                                                                                       ib_packet_center(std::move(ib_packet_center)){}
            void send(ip_t addr, packet_content_t buf) noexcept{

                global_packet_id_t id   = this->id_gen->get();
                model::Packet pkt       = packet_service::make_request(addr, id, std::move(buf));
                this->retransmission_manager->add_retriable(pkt);
                this->ob_packet_center->push(pkt);
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

        static auto spawn_std_mailbox(std::vector<std::unique_ptr<worker::Workable>> workers, 
                                      std::unique_ptr<packet_controller::IDGeneratorInterface> id_gen, 
                                      std::shared_ptr<packet_controller::RetransmissionManagerInterface> retry_manager, 
                                      std::shared_ptr<packet_controller::PacketCenterInterface> ob_packet_center,
                                      std::shared_ptr<packet_controller::PacketCenterInterface> ib_packet_center,
                                      std::shared_ptr<std::atomic<uint8_t>> status_var) -> std::unique_ptr<MailboxInterface>{
            
            return std::make_unique<StdMailBoxController>(std::move(workers), std::move(id_gen), std::move(retry_manager), 
                                                          std::move(ob_packet_center), std::move(ib_packet_center), std::move(status_var));
        }
    };
}

namespace dg::network_kernel_mailbox_ack{

    struct Config{
        size_t concurrency_count;
        std::shared_ptr<memory::Allocatable> allocator;
    };

    auto spawn(Config config) -> std::unique_ptr<core::MailboxInterface>{

        // std::shared_ptr scheduler{packet_controller::ComponentFactory::get_std_scheduler()};
        // std::shared_ptr retransmission_manager(packet_controller::ComponentFactory::get_retransmission_manager());
        // std::shared_ptr ob_pkt_center(packet_controller::ComponentFactory::get_outbound_packet_center(scheduler));
        // std::shared_ptr ib_pkt_center(packet_controller::ComponentFactory::get_inbound_packet_center()); 

        // auto id_gen     = packet_controller::ComponentFactory::get_id_generator();
        // auto workers    = std::vector<std::unique_ptr<worker::Workable>>();
        // auto sock       = socket_service::open_socket();
        // auto status     = std::make_shared<std::atomic<uint8_t>>(runtime_error::SUCCESS);

        // socket_service::bind_socket_to_port(*sock);

        // for (size_t i = 0; i < config.concurrency_count; ++i){
        //     workers.push_back(worker::ComponentFactory::spawn_inbound_worker(retransmission_manager, ob_pkt_center, ib_pkt_center, scheduler, sock, status, config.allocator));
        //     workers.push_back(worker::ComponentFactory::spawn_outbound_worker(ob_pkt_center, sock, status, config.allocator));
        // }
        // workers.push_back(worker::ComponentFactory::spawn_retry_worker(retransmission_manager, ob_pkt_center, status));

        // return core::ComponentFactory::spawn_std_mailbox(std::move(workers), std::move(id_gen), std::move(retransmission_manager), 
        //                                                  std::move(ob_pkt_center), std::move(ib_pkt_center), std::move(status));
    }
}

#endif