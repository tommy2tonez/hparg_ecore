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

//we'll finalize these within 1-2 days
//we realized that the performance constraint is not from low level optimization but

//(1): memory access pattern
//(2): # of memory orderings
//(3): polymorphic dispatch overhead
//(4): branching overhead
//(5): affinity of tasks

//we attempted to do transmission control, yet it has destructive interference with the memregion frequency feature of extnsrc, which is bad
//so we just generalize this by doing fixed mailchimp transmission_frequency, such is 1GB/s, without loss of generality, outbound to avoid packet losses because of inappropriate frequency 
//we leave the rest of the optimization to the memregion
//this simple technique of packet_id uniqueness + retransmission + UDP + drop ip_tables + affined rx_queues + SO_REUSEPORT + etc. is actually very valuable
//the value would be dramatically decreased if we are to add transmission control protocol, we are breaking single responsibilities + force clients to give up compute power for the tasks that are not quantifiable, or the tasks that could not be optimially solved without being seen from a hollistic view
//we would be able to push the bandwidth -> the saturation limit

//I was running the numbers
//it seems that UDP or every transmission protocol follows the rule of logarit, no matter how many packets we are transmitting
//let's see what we want
//we want for every tree operation, the success rate is 90%
//we want to offset the synchronization overheads by doing concurrent tree operations such is synchronization overhead == last_operation_overhead

//if the success rate is x, we are transmitting 1 million packets (across compute nodes), each 1KB, totalling 1GB
//assume the logarit for 90% percentile is 1.8
//ln(1 << 20) / ln(1.7) ~= 26 retranmissions
//assume the logarit for 50% percentile is 3
//ln(1 << 20) / ln(3) ~= 13 retransmissions

//let's look at it from the individualism point of view
//30% transmission fail rate (70% success rate) for individual packets @ a reasonably load balanced network
//(1 - (30%^n)) ** (1 << 20) = 90%
//what is n?
//ln(1 - 90% ** (1 / (1 << 20))) / ln(30%) == 13 retransmission 
//70% fail -> 45 retransmissions
//we probably want to do only 10 retranmissions, and the other retranmission responsibility goes to the re-request component 

//so it requires 45 retranmissions to have a 90% success rate of 1GB of internal data transferring across computing nodes
//the number does not look very promising
//yet this imposes a problem of multidimensional projections or compression as we call it
//is our 1GB of raw pixels actually 1GB of data?
//how about we map the 1GB of raw pixel -> 1MB of information by using multidimensional projection before transmitting the data to another compute nodes for var_intercourse
//it's the art of projection (we are off topic here, we must be able to push 1GB of data for every tree computation)

//assume that we have a fixed transmission -> kernel, such saturates the outbound of our transmission (we've done our best)
//the approx equation for synchronization time is total_packet_sz / (outbound * compute_node_sz) + log(min(total_packet_sz, MAX_OUTBOUND_SZ), 3) * transmission_delay_time
//we dont want to waste our time for synchronization overheads, how about we increase the number of concurrent operations (rocket launching operations), and reduce the total synchronization -> last_rocket_launch synchronization?

//alright, so this is like a hybrid transmission_controlled protocol, it's a very gay protocol (useless without being used in conjunction with upstream optimizations of frequency, it's been 30 years and we have never gotten the TCP right, when to additive increase, when to multiply decrease, etc.)
//when we thought that we got the protocol right, we came across different patterns for different machines, and this is another machine learning problem
//yet it's designed to do one thing, to saturate the outbound_network_bandwidth + trigger retransmission for every interval

//I think our job is not to ask but to leave room for calibration
//we know the hollistic picture of data transmission, we dont want to do advanced transmission controlled protocol like AIMD or whatever for the reasons being we are wasting compute + the task is not quantifiable + we force the users to waste resource for unnecessary features

//alright fellas, we are not in heat and we are not going to rob banks
//we are going to auction this engine mining, we only get 30% fee for hosting the auction, it's legal fellas, I already asked Mark Zuck to prep a legal team for me on this matter
//when are we publishing the mining engine again? probably a month or a year
//yet we have to detail EVERYTHING in order to make sure that this runs correctly
//code is clear
//(1): exhaustion clear
//(2): finite mempool clear
//(3): consumption_lock clear, released by infretry_device
//(4): memory_order clear
//(5): memory access pattern clear
//(6): bug clear
//(7): leak clear
//(8): packet_id attack patched by using random_id

//I was thinking about shared links vs individual links
//assume optimal conditions, we use shared link, we observe the completed time for each link
//we can replicate the completed time by serving each of the link in the sorted order + pad delays
//proof is conservation of energy, the shared link always uses more energy than the individual links for the <current_task>

//real life is not optimal
//there are cases where shared link would result in better response time for all
//there are two ways to solve the problem:
//use transmission controlled protocol
//link aggregation, the unit is now a bag of links, not individuals
//we aim for simplicity, so we choose the second approach
//its complicated, we choose the path of precomputed frequencies for global optimality, not local optimality
//we realized that the only practical thing that has ever worked is calibration + statistics
//our job is to not tie our hands, and provide enough parameters

//whenever I feel like I might have done something wrong
//let's see what I could do by using the branching technique

//request: wait_request
//         event_driven_request (curious_subcriber or reactive_subcriber pattern, we (observers) assign ids for each of the request, and check every interval for results in constrast to getting notified by the subject)
//              - one_request transmission
//              - multiple_request transmissions: - multiple_request by extending a one-time-transmit protocol
//                                                - multiple_request by extending a multiple-time-transmit protocol
//                                                      - multiple-time (or one_time) transmit protocol by using congestion-controlled algorithm
//                                                          - there is a congestion-controlled component => shared-link is involed, is shared-link runtime-induced or predetermined
//                                                          - shared-link is predetermined => there always exists a not worse solution by using single-links in the optimal condition (proof above)
//                                                          - shared-link is not predetermined => there must exists a congestion-controlled algorithm
//                                                                  - how do we manage connections?
//                                                                  - how do we know the pattern of optimality for each of the connections?
//                                                                  - how expensive is it to know the pattern of optimality for each of the connections?
//                                                      - multiple-time (or one_time) transmission protocol by not using congestion-controlled algorithm
//                                                              - static outbound rate
//                                                              - ASAP outbound rate

//clients are asking for 0.02 ms latency uncertainty from A -> Z, it's hard fellas
//the only way we could achieve such latency is by using reactive_subcriber + mutex (yet abusing mutex will degrade the performance across cores, not only the current operating context)
//we rather think that it's unrealistic for real-life requests, unless we are in a cloud environment (we must consider that our clients might be from clouds) 
//https://github.com/torvalds/linux/blob/a351e9b9fc24e982ec2f0e76379a49826036da12/Documentation/timers/timers-howto.txt

//https://github.com/torvalds/linux/blob/a351e9b9fc24e982ec2f0e76379a49826036da12/Documentation/core-api/atomic_ops.rst
//https://www.kernel.org/doc/Documentation/memory-barriers.txt

//atomic operations, how we would want to use relaxed functions, inference of compiler_code_path serialization (not hardware serialization) by using memory_ordering fences
//what Torvalds, and his friends meant when we wrote the atomic operations are relaxed functions
//if a function is not relaxed, it is not defined, the combination of relaxed functions to create another relaxed function
//  - if one of the relaxed functions is not explicitly inferred in the computation tree of the returning result or its' computation is intended, there must be a fence (we never fence in and out of a void function, because it is the caller responsibility not callee responsibility)
//  - other use-cases of atomic are not defined
//  - after 20 years, we have come to a conclusion fellas, there are two scenerios that we often have:
//      + a concurrent function that is relaxed (all function logics are not reordered, happens in one line of function call) + self-sufficient (push | pop | id | get | get_retriables | etc.)
//      + an open-close pair of concurrent functions
//          + the open-close pair is in the current scope, we would want to use <concurrent_transaction> of std::thread_signal_fence(std::memory_order_seq_cst) in and out of the transaction
//          + the open-close pair is in two different scopes, we would want to std::memory_order_seq_cst post the open and std::memory_order_seq_cst pre the close
//  - this is the formula that is NEVER wrong
//  - we dont really care about the functions that do not deal with atomic variables, compilers are smarter than yall in this case 
//  - this atomic fling is the cause of 99% major software bugs in the world
//  - I know I've spent 90% of my time talking about this topic, because this is the topic that is never outdated 
//  - it's very easy to get right and very easy to get wrong, to the point that I doubt there ever exists a program written in C that is not kernel uses memory orderings correctly (we with the bliss of C++ stack destructor + constructor, xlock_guard serves the purpose of fencing just right)
//  - there are a lot of smart people, think that <concurrent_transaction> and <concurrent_function> are two different things, they're gone fellas ... to the undefined land 

//when we first wrote this, we didnt expect things to be this complicated fellas
//it seems like this is a thin extension on top of the kernel UDP protocol to do retransmission because it really is (we do non-discriminated retransmissions with one goal in mind, to saturate the outbound + inbound)
//real-life transmission protocol is logarit-based, we aren't in a perfect lab of 100% recv
//we'll use simple compression (huffman) if there is a usage 
//we'll run one more code review

//alright Morpheus, we'll wake you up in time, I think you are 100000x smarter than all of us combined so make wise choices
//we have run numbers + calculations, this is probably the best socket path that we could choose

namespace dg::network_kernel_mailbox_impl1::types{

    using factory_id_t          = std::array<char, 32>;
    using local_packet_id_t     = uint64_t;
    using packet_polymorphic_t  = uint8_t;
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
        std::array<char, 16> ip_buf;
        bool flag;

        auto data() const noexcept -> const char *{

            return this->ip_buf.data();
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
            reflector(ip_buf, flag);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ip_buf, flag);
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

    struct PacketBase{
        global_packet_id_t id;
        uint8_t retransmission_count;
        uint8_t priority;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(id, retransmission_count, priority);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(id, retransmission_count, priority);
        }
    };

    struct PacketHeader: PacketBase{
        Address fr_addr;
        Address to_addr;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(static_cast<const PacketBase&>(*this), fr_addr, to_addr);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(static_cast<PacketBase&>(*this), fr_addr, to_addr);
        }
    };

    struct XOnlyRequestPacket{
        dg::string content;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(content);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(content);
        }
    };

    struct RequestPacket: PacketHeader, XOnlyRequestPacket{

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(static_cast<const PacketHeader&>(*this), static_cast<const XOnlyRequestPacket&>(*this));
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(static_cast<PacketHeader&>(*this), static_cast<XOnlyRequestPacket&>(*this));
        }
    };

    struct XOnlyAckPacket{
        dg::vector<PacketBase> ack_vec;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ack_vec);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ack_vec);
        }
    };

    struct AckPacket: PacketHeader, XOnlyAckPacket{

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(static_cast<const PacketHeader&>(*this), static_cast<const XOnlyAckPacket&>(*this));
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(static_cast<PacketHeader&>(*this), static_cast<XOnlyAckPacket&>(*this))
        }
    };

    struct XOnlyKRescuePacket{

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            (void) reflector;
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            (void) reflector;
        }
    };

    struct KRescuePacket: PacketHeader, XOnlyKRescuePacket{

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(static_cast<const PacketHeader&>(*this), static_cast<const XOnlyKRescuePacket&>(*this));
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) noexcept{
            reflector(static_cast<PacketHeader&>(*this), static_cast<XOnlyKRescuePacket&>(*this));
        }
    }; 

    struct Packet: PacketHeader{
        std::variant<XOnlyRequestPacket, XOnlyAckPacket, XOnlyKRescuePacket> xonly_content; //alright, the question is whether this is optional polymorphic, assume non-optimal polymorphic for now (this introduces so many complexities), we'll add measurements
                                                                                            //or we can do it the std_way, the use-case of Packet is not defined if it is to be optional_polymorphic after being <brought_to_life> via <factory_pattern>
    };

    struct ScheduledPacket{
        Packet pkt;
        std::chrono::time_point<std::chrono::utc_clock> sched_time;
    };

    struct QueuedPacket{
        Packet pkt;
        std::chrono::time_point<std::chrono::utc_clock> queued_time;            
    };

    struct MailBoxArgument{
        Address to;
        dg::string content;
    };
}

namespace dg::network_kernel_mailbox_impl1::constants{

    using namespace std::literals::chrono_literals;
    using namespace std::chrono;

    enum packet_kind: dg::network_kernel_mailbox_impl1::types::packet_polymorphic_t{
        ack     = 0u,
        request = 1u,
        krescue = 2u
    };

    static inline constexpr size_t MAXIMUM_MSG_SIZE             = size_t{1} << 10;
    static inline constexpr size_t MAX_PACKET_CONTENT_SIZE      = size_t{1} << 10;
    static inline constexpr size_t MAX_ACK_PER_PACKET           = size_t{1} << 8;
    static inline constexpr size_t DEFAULT_ACCUMULATION_SIZE    = size_t{1} << 4;
}

namespace dg::network_kernel_mailbox_impl1::packet_controller{

    using namespace dg::network_kernel_mailbox_impl1::model;

    class SchedulerInterface{

        public:

            virtual ~SchedulerInterface() noexcept = default;
            virtual auto schedule(Address) noexcept -> std::expected<std::chrono::time_point<std::chrono::utc_clock>, exception_t> = 0;
    };

    class UpdatableInterface{

        public:

            virtual ~UpdatableInterface() noexcept = default;
            virtual void update() noexcept = 0;
    };

    class ExhaustionControllerInterface{

        public:

            virtual ~ExhaustionControllerInterface() noexcept = default;
            virtual auto is_should_wait() noexcept -> bool = 0;
            virtual auto update_waiting_size(size_t) noexcept -> exception_t = 0;
    };

    class KernelOutBoundTransmissionControllerInterface{

        public:

            virtual ~KernelOutBoundTransmissionControllerInterface() noexcept = default;
            virtual auto get_transmit_frequency() noexcept -> uint32_t = 0;
            virtual auto update_waiting_size(size_t) noexcept -> exception_t = 0;
    };

    class IDGeneratorInterface{

        public:

            virtual ~IDGeneratorInterface() noexcept = default;
            virtual auto get() noexcept -> global_packet_id_t = 0;
    };

    class RequestPacketGeneratorInterface{

        public:

            virtual ~PacketGeneratorInterface() noexcept = default;
            virtual auto get(MailBoxArgument&&) noexcept -> std::expected<RequestPacket, exception_t> = 0;
    };

    class KRescuePacketGeneratorInterface{

        public:

            virtual ~KRescuePacketGeneratorInterface() noexcept = default;
            virtual auto get() noexcept -> std::expected<KRescuePacket, exception_t> = 0;
    };

    class KernelRescuePostInterface{

        public:

            virtual ~KernelRescuePostInterface() noexcept = default;
            virtual auto heartbeat() noexcept -> exception_t = 0;
            virtual auto last_heartbeat() noexcept -> std::expected<std::optional<std::chrono::time_point<std::chrono::utc_clock>>, exception_t> = 0;
            virtual void reset() noexcept = 0;
    };

    class AckPacketGeneratorInterface{

        public:

            virtual ~AckPacketGeneratorInterface() noexcept = default;
            virtual auto get(Address, PacketBase *, size_t) noexcept -> std::expected<AckPacket, exception_t> = 0;
    };

    class RetransmissionControllerInterface{

        public:

            virtual ~RetransmissionControllerInterface() noexcept = default;
            virtual void add_retriables(std::move_iterator<Packet *>, size_t, exception_t *) noexcept = 0;
            virtual void ack(global_packet_id_t *, size_t, exception_t *) noexcept = 0;
            virtual void get_retriables(Packet *, size_t&, size_t) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class BufferContainerInterface{

        public:

            virtual ~BufferContainerInterface() noexcept = default;
            virtual void push(std::move_iterator<dg::string *>, size_t, exception_t *) noexcept = 0;
            virtual void pop(dg::string *, size_t&, size_t) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class IPSieverInterface{

        public:

            virtual ~IPSieverInterface() noexcept = default;
            virtual auto thru(Address) noexcept -> std::expected<bool, exception_t> = 0;
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

    class TrafficControllerInterface{

        public:

            virtual ~TrafficControllerInterface() noexcept = default;
            virtual auto thru(Address) noexcept -> std::expected<bool, exception_t> = 0;
            virtual void reset() noexcept = 0;
    };

    class BorderControllerInterface{

        public:

            virtual ~BorderControllerInterface() noexcept = default;
            virtual void thru(Address *, size_t, exception_t *) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class NATIPControllerInterface{

        public:

            virtual ~NATIPControllerInterface() noexcept = default;

            virtual void add_inbound(Address *, size_t, exception_t *) noexcept = 0;
            virtual void add_outbound(Address *, size_t, exception_t *) noexcept = 0;

            virtual void get_inbound_friend_addr(Address *, size_t off, size_t& sz, size_t cap) noexcept = 0; 
            virtual auto get_inbound_friend_addr_size() noexcept -> size_t = 0;

            virtual void get_outbound_friend_addr(Address *, size_t off, size_t& sz, size_t cap) noexcept = 0;
            virtual auto get_outbound_friend_addr_size() noexcept -> size_t = 0;
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

    template <class ...Args, class Iterator>
    static auto finite_set_insert(std::unordered_set<Args...>& container, size_t container_cap, 
                                  Iterator first, Iterator last) noexcept -> std::expected<size_t, exception_t>{

        static_assert(std::is_trivial_v<Iterator>);

        dg::network_stack_allocation::NoExceptAllocation<Iterator[]> rewind_buf(std::distance(first, last));
        size_t rewind_buf_sz = 0u; 

        try{
            for (auto it = first; it != last; ++it){
                if (container.size() == container_cap){
                    break;
                }

                auto [iptr, status] = container.insert(*it);

                if (status){
                    rewind_buf[rewind_buf_sz++] = it;
                }
            }

            return static_cast<size_t>(std::distance(first, it_first));
        } catch (...){
            for (size_t i = 0u; i < rewind_buf_sz; ++i){
                container.erase(*rewind_buf[i]);
            }

            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }
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

        if (bind(sock.kernel_sock_fd, reinterpret_cast<struct sockaddr *>(&server), sizeof(struct sockaddr_in6)) == -1){
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

        if (bind(sock.kernel_sock_fd, reinterpret_cast<struct sockaddr *>(&server), sizeof(struct sockaddr_in)) == -1){
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
        auto n              = sendto(sock.kernel_sock_fd, 
                                     buf, stdx::wrap_safe_integer_cast(sz), 
                                     MSG_DONTWAIT, 
                                     reinterpret_cast<const struct sockaddr *>(&server), sizeof(struct sockaddr_in6));

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
        auto n              = sendto(sock.kernel_sock_fd, 
                                     buf, stdx::wrap_safe_integer_cast(sz), 
                                     MSG_DONTWAIT, 
                                     reinterpret_cast<const struct sockaddr *>(&server), sizeof(struct sockaddr_in)); 

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
        socklen_t from_length           = sizeof(from);
        auto n                          = recvfrom(sock.kernel_sock_fd, 
                                                   dst, stdx::wrap_safe_integer_cast(dst_cap), 
                                                   0,
                                                   reinterpret_cast<struct sockaddr *>(&from), &from_length);

        if (n == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        dst_sz = stdx::safe_integer_cast<size_t>(n);
        return dg::network_exception::SUCCESS;
    }
}

namespace dg::network_kernel_mailbox_impl1::data_structure{

    template <class T>
    class temporal_finite_unordered_set{

        private:

            dg::unordered_set<T> hashset;
            dg::deque<T> entry_deque;
            size_t cap;

        public:

            static_assert(std::is_trivial_v<T>);

            temporal_finite_unordered_set(size_t capacity): hashset(),
                                                            entry_deque(),
                                                            cap(capacity){

                if (capacity == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::BAD_ARGUMENT);
                }

                this->hashset.reserve(capacity);
            }

            inline void insert(T key) noexcept{

                if (this->hashset.contains(key)){
                    return;
                }

                if (this->entry_deque.size() == this->cap) [[unlikely]]{
                    size_t half_cap = std::max(static_cast<size_t>(this->cap >> 1), size_t{1u});

                    for (size_t i = 0u; i < half_cap; ++i){
                        this->hashset.erase(this->entry_deque.front());
                        this->entry_deque.pop_front();
                    }
                }

                this->hashset.insert(key);
                this->entry_deque.push_back(key);
            }

            inline auto contains(const T& key) const noexcept -> bool{

                return this->hashset.contains(key);
            }

            inline auto begin() const noexcept{

                return this->entry_deque.begin();
            }

            inline auto begin() noexcept{

                return this->entry_deque.begin();
            }

            inline auto end() const noexcept{

                return this->entry_deque.end();
            }

            inline auto end() noexcept{

                return this->entry_deque.end();
            }

            inline auto size() const noexcept -> size_t{
                
                return this->entry_deque.size();
            }
    };
}

namespace dg::network_kernel_mailbox_impl1::packet_service{

    using namespace dg::network_kernel_mailbox_impl1::model;

    static auto virtualize_request_packet(RequestPacket pkt) noexcept -> std::expected<Packet, exception_t>{

        Packet rs                       = {};
        static_cast<PacketHeader&>(rs)  = std::move(static_cast<PacketHeader&>(pkt));
        rs.xonly_content                = std::move(static_cast<XOnlyRequestPacket&>(pkt));

        return rs;
    }

    static auto virtualize_ack_packet(AckPacket pkt) noexcept -> std::expected<Packet, exception_t>{

        Packet rs                       = {};
        static_cast<PacketHeader&>(rs)  = std::move(static_cast<PacketHeader&>(pkt));
        rs.xonly_content                = std::move(static_cast<XOnlyAckPacket&>(pkt));

        return rs;
    }

    static auto virtualize_krescue_packet(KRescuePacket pkt) noexcept -> std::expected<Packet, exception_t>{

        Packet rs                       = {};
        static_cast<PacketHeader&>(rs)  = std::move(static_cast<PacketHeader&>(pkt));
        rs.xonly_content                = std::move(static_cast<XOnlyKRescuePacket&>(pkt));

        return rs;
    } 

    static auto devirtualize_request_packet(Packet pkt) noexcept -> std::expected<RequestPacket, exception_t>{

        if (!std::holds_alternative<XOnlyRequestPacket>(pkt.xonly_content)){
            return std::unexpected(dg::network_exception::BAD_POLYMORPHIC_ACCESS);
        }

        RequestPacket rs                        = {};
        static_cast<PacketHeader&>(rs)          = std::move(static_cast<PacketHeader&>(pkt));
        static_cast<XOnlyRequestPacket&>(rs)    = std::move(std::get<XOnlyRequestPacket>(pkt.xonly_content)); 

        return rs;
    }

    static auto devirtualize_ack_packet(Packet pkt) noexcept -> std::expected<AckPacket, exception_t>{

        if (!std::holds_alternative<XOnlyAckPacket>(pkt.xonly_content)){
            return std::unexpected(dg::network_exception::BAD_POLYMORPHIC_ACCESS);
        }

        AckPacket rs                            = {};
        static_cast<PacketHeader&>(rs)          = std::move(static_cast<PacketHeader&>(pkt));
        static_cast<XOnlyAckPacket&>(rs)        = std::move(std::get<XOnlyAckPacket>(pkt.xonly_content));

        return rs;
    }

    static auto devirtualize_krescue_packet(Packet pkt) noexcept -> std::expected<KRescuePacket, exception_t>{

        if (!std::holds_alternative<XOnlyKRescuePacket>(pkt.xonly_content)){
            return std::unexpected(dg::network_exception::BAD_POLYMORPHIC_ACCESS);
        }

        KRescuePacket rs                        = {};
        static_cast<PacketHeader&>(rs)          = std::move(static_cast<PacketHeader&>(pkt));
        static_cast<XOnlyKRescuePacket&>(rs)    = std::move(std::get<XOnlyKRescuePacket>(pkt.xonly_content));

        return rs;
    }

    static auto frequency_to_period(uint32_t frequency) noexcept -> std::chrono::nanoseconds{

        constexpr uint32_t MINIMUM_FREQUENCY    = uint32_t{1};
        constexpr uint32_t MAXIMUM_FREQUENCY    = uint32_t{1} << 30;
        constexpr uint32_t SECOND_METRIC        = uint32_t{1} << 30; 
        uint32_t clamped_frequency              = std::min(std::max(frequency, MINIMUM_FREQUENCY), MAXIMUM_FREQUENCY);
        uint32_t period                         = SECOND_METRIC / clamped_frequency; 

        return std::chrono::nanoseconds{period};
    }

    static inline auto is_request_packet(const Packet& pkt) noexcept -> bool{

        return std::holds_alternative<XOnlyRequestPacket>(pkt.xonly_content);
    }

    static inline auto is_ack_packet(const Packet& pkt) noexcept -> bool{

        return std::holds_alternative<XOnlyAckPacket>(pkt.xonly_content);
    }

    static inline auto is_krescue_packet(const Packet& pkt) noexcept -> bool{

        return std::holds_alternative<XOnlyKRescuePacket>(pkt.xonly_content);
    }

    static inline auto get_packet_polymorphic_type(const Packet& pkt) noexcept -> packet_polymorphic_t{

        if (is_request_packet(pkt)){
            return constants::request;
        } else if (is_ack_packet(pkt)){
            return constants::ack;
        } else if (is_krescue_packet(pkt)){
            return constants::krescue;
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }
    }

    static auto serialize_request_packet(RequestPacket packet) noexcept -> dg::string{

        constexpr size_t header_sz  = dg::network_compact_serializer::integrity_size(PacketHeader{});
        size_t content_sz           = packet.content.size();
        size_t total_sz             = content_sz + header_sz;
        dg::string bstream          = std::move(packet.content);
        bstream.resize(total_sz);
        char * header_ptr           = std::next(bstream.data(), content_sz); //this gives me chill without doing checksum for the buffer (I think this has to be an option), yet we have to steer in the way of <inbound_data_could_be_maliciously_engineered> and we'll come back for statistical chances later
        dg::network_compact_serializer::integrity_serialize_into(header_ptr, static_cast<const PacketHeader&>(packet));

        return bstream;
    }

    static auto serialize_ack_packet(AckPacket packet) noexcept -> dg::string{

        return dg::network_compact_serializer::integrity_serialize<dg::string>(packet);
    }

    static auto serialize_krescue_packet(KRescuePacket packet) noexcept -> dg::string{

        return dg::network_compact_serializer::integrity_serialize<dg::string>(packet);
    }

    static auto deserialize_request_packet(dg::string bstream) noexcept -> std::expected<RequestPacket, exception_t>{

        size_t header_sz    = dg::network_compact_serializer::integrity_size(PacketHeader{});
        RequestPacket rs    = {};
        auto [left, right]  = stdx::backsplit_str(std::move(bstream), header_sz);
        rs.content          = std::move(left);
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<PacketHeader>)(static_cast<PacketHeader&>(rs), right.data(), right.size()); //this would crash, it's this guy responsibility to make sure the integrity of the buffer -> the underlying data (at least there is no serious corruption post the function call)

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    static auto deserialize_ack_packet(dg::string bstream) noexcept -> std::expected<AckPacket, exception_t>{

        return dg::network_compact_serializer::integrity_deserialize<AckPacket>(bstream);
    }

    static auto deserialize_krescue_packet(dg::string bstream) noexcept -> std::expected<KRescuePacket, exception_t>{

        return dg::network_compact_serializer::integrity_deserialize<KRescuePacket>(bstream);
    }

    static auto serialize_packet(Packet packet) noexcept -> dg::string{

        constexpr size_t PACKET_POLYMORPHIC_HEADER_SZ                                   = dg::network_trivial_serializer::size(packet_polymorphic_t{});
        std::array<char, PACKET_POLYMORPHIC_HEADER_SZ> polymorphic_writing_container    = {}; 
        dg::string serialized                                                           = {};

        if (is_request_packet(packet)){
            dg::network_trivial_serializer::serialize_into(polymorphic_writing_container.data(), constants::request);
            serialized = serialize_request_packet(dg::network_exception_handler::nothrow_log(devirtualize_request_packet(std::move(packet))));
        } else if (is_ack_packet(packet)){
            dg::network_trivial_serializer::serialize_into(polymorphic_writing_container.data(), constants::ack);
            serialized = serialize_ack_packet(dg::network_exception_handler::nothrow_log(devirtualize_ack_packet(std::move(packet))));
        } else if (is_krescue_packet(packet)){
            dg::network_trivial_serializer::serialize_into(polymorphic_writing_container.data(), constants::krescue);
            serialized = serialize_krescue_packet(dg::network_exception_handler::nothrow_log(devirtualize_krescue_packet(std::move(packet))));
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort(); //this is not qualified as an exception, we assume global assumption for packet construction
            } else{
                std::unreachable();
            }
        }

        std::copy(polymorphic_writing_container.begin(), polymorphic_writing_container.end(), std::back_inserter(serialized));
        return serialized;
    }

    static auto deserialize_packet(dg::string bstream) noexcept -> std::expected<Packet, exception_t>{

        constexpr size_t PACKET_POLYMORPHIC_HEADER_SZ   = dg::network_trivial_serializer::size(packet_polymorphic_t{});
        auto [left, right]                              = stdx::backsplit_str(std::move(bstream), PACKET_POLYMORPHIC_HEADER_SZ);

        if (right.size() != PACKET_POLYMORPHIC_HEADER_SZ){
            return std::unexpected(dg::network_exception::MALFORMED_SOCKET_PACKET);
        }

        packet_polymorphic_t packet_type = {};
        dg::network_trivial_serializer::deserialize_into(packet_type, right.data());

        if (packet_type == constants::request){
            std::expected<RequestPacket, exception_t> devirtualized_packet = deserialize_request_packet(std::move(left));

            if (!devirtualized_packet.has_value()){
                return std::unexpected(devirtualized_packet.error());
            }

            return virtualize_request_packet(std::move(devirtualized_packet.value()));
        } else if (packet_type == constants::ack){
            std::expected<AckPacket, exception_t> devirtualized_packet = deserialize_ack_packet(std::move(left));

            if (!devirtualized_packet.has_value()){
                return std::unexpected(devirtualized_packet.error());
            }

            return virtualize_ack_packet(std::move(devirtualized_packet.value()));
        } else if (packet_type == constants::krescue){
            std::expected<KRescuePacket, exception_t> devirtualized_packet = deserialize_krescue_packet(std::move(left));

            if (!devirtualized_packet.has_value()){
                return std::unexpected(devirtualized_packet.error());
            }

            return virtualize_krescue_packet(std::move(devirtualized_packet.value()));
        } else{
            return std::unexpected(dg::network_exception::MALFORMED_SOCKET_PACKET);
        }
    }
}

namespace dg::network_kernel_mailbox_impl1::packet_controller{

    class ASAPScheduler: public virtual SchedulerInterface{

        public:

            auto schedule(Address) noexcept -> std::expected<std::chrono::time_point<std::chrono::utc_clock>, exception_t>{

                return std::chrono::utc_clock::now();
            }

            auto feedback(Address, std::chrono::nanoseconds) noexcept -> exception_t{

                return dg::network_exception::SUCCESS;
            }
    };

    class ImmutableKernelOutBoundTransmissionController: public virtual KernelOutBoundTransmissionControllerInterface{

        private:

            uint32_t transmit_frequency;

        public:

            ImmutableKernelOutBoundTransmissionController(uint32_t transmit_frequency) noexcept: transmit_frequency(transmit_frequency){}

            auto get_transmit_frequency() noexcept -> uint32_t{

                return this->transmit_frequency;
            }

            auto update_waiting_size(size_t) noexcept -> exception_t{

                return dg::network_exception::SUCCESS;
            }
    };

    class NoExhaustionController: public virtual ExhaustionControllerInterface{

        public:

            auto is_should_wait() noexcept -> bool{

                return false;
            }

            auto update_waiting_size(size_t) noexcept -> exception_t{

                return dg::network_exception::SUCCESS;
            }
    };

    class DefaultExhaustionController: public virtual ExhaustionControllerInterface{

        public:

            auto is_should_wait() noexcept -> bool{

                return true;
            }

            auto update_waiting_size(size_t) noexcept -> exception_t{

                return dg::network_exception::SUCCESS;
            }
    };

    class IncrementalIDGenerator: public virtual IDGeneratorInterface{

        private:

            std::atomic<local_packet_id_t> last_pkt_id;
            stdx::hdi_container<factory_id_t> factory_id;

        public:

            IncrementalIDGenerator(std::atomic<local_packet_id_t> last_pkt_id,
                                   stdx::hdi_container<factory_id_t> factory_id) noexcept: last_pkt_id(std::move(last_pkt_id)),
                                                                                           factory_id(std::move(factory_id)){}

            auto get() noexcept -> GlobalPacketIdentifier{

                auto rs             = GlobalPacketIdentifier{};
                rs.local_packet_id  = this->last_pkt_id.fetch_add(1u, std::memory_order_relaxed);
                rs.factory_id       = this->factory_id.value;

                return rs;
            }
    };

    //we have to use random_id_generator to avoid coordinated attacks, the chances of packet_id collision are so slim that we dont even care, even if there are collisions, it still follows the rule of 1 request == max_one_receive, ack_sz <= request_sz
    class RandomIDGenerator: public virtual IDGeneratorInterface{

        private:

            stdx::hdi_container<factory_id_t> factory_id;

        public:

            RandomIDGenerator(stdx::hdi_container<factory_id_t> factory_id) noexcept: factory_id(std::move(factory_id)){}

            auto get() noexcept -> GlobalPacketIdentifier{

                auto rs             = GlobalPacketIdentifier{};
                rs.local_packet_id  = dg::network_randomizer::randomize_int<local_packet_id_t>();
                rs.factory_id       = this->factory_id.value;

                return rs;
            }
    };

    class RequestPacketGenerator: public virtual RequestPacketGeneratorInterface{

        private:

            std::unique_ptr<IDGeneratorInterface> id_gen;
            Address host_addr; 

        public:

            RequestPacketGenerator(std::unique_ptr<IDGeneratorInterface> id_gen,
                                   Address host_addr) noexcept: id_gen(std::move(id_gen)),
                                                                host_addr(std::move(host_addr)){}

            auto get(MailBoxArgument&& arg) noexcept -> std::expected<RequestPacket, exception_t>{

                if (arg.content.size() > constants::MAX_PACKET_CONTENT_SIZE){
                    return std::unexpected(dg::network_exception::BAD_SOCKET_BUFFER_LENGTH);
                }

                RequestPacket pkt           = {};
                pkt.fr_addr                 = this->host_addr;
                pkt.to_addr                 = std::move(arg.to);
                pkt.id                      = this->id_gen->get();
                pkt.retransmission_count    = 0u;
                pkt.priority                = 0u;
                pkt.content                 = std::move(arg.content);

                return pkt;
            }
    };

    class AckPacketGenerator: public virtual AckPacketGeneratorInterface{

        private:

            std::unique_ptr<IDGeneratorInterface> id_gen;
            Address host_addr;

        public:

            AckPacketGenerator(std::unique_ptr<IDGeneratorInterface> id_gen,
                               Address host_addr) noexcept: id_gen(std::move(id_gen)),
                                                            host_addr(std::move(host_addr)){}

            auto get(Address to_addr, PacketBase * pkt_base_arr, size_t sz) noexcept -> std::expected<AckPacket, exception_t>{

                auto ack_vec                = dg::network_exception::cstyle_initialize<dg::vector<PacketBase>>(sz);

                if (!ack_vec.has_value()) [[unlikely]]{
                    return std::unexpected(ack_vec.error());
                }

                AckPacket pkt               = {};
                pkt.fr_addr                 = this->host_addr;
                pkt.to_addr                 = to_addr;
                pkt.id                      = this->id_gen->get();
                pkt.retransmission_count    = 0u;
                pkt.priority                = 0u;
                pkt.ack_vec                 = std::move(ack_vec.value());

                std::copy(pkt_base_arr, std::next(pkt_base_arr, sz), pkt.ack_vec.begin());

                return pkt;
            }
    };

    class KRescuePacketGenerator: public virtual KRescuePacketGeneratorInterface{

        private:

            std::unique_ptr<IDGeneratorInterface> id_gen;
            Address host_addr;

        public:

            KRescuePacketGenerator(std::unique_ptr<IDGeneratorInterface> id_gen,
                                   Address host_addr) noexcept: id_gen(std::move(id_gen)),
                                                                host_addr(std::move(host_addr)){}

            auto get() noexcept -> std::expected<KRescuePacket, exception_t>{

                KRescuePacket pkt           = {};
                pkt.fr_addr                 = this->host_addr;
                pkt.to_addr                 = this->host_addr;
                pkt.id                      = this->id_gen->get();
                pkt.retransmission_count    = 0u;
                pkt.priority                = 0u;

                return pkt;
            }
    };

    class KernelRescuePost: public virtual KernelRescuePostInterface{

        private:

            std::atomic<std::chrono::time_point<std::chrono::utc_clock>> ts;

        public:

            using Self = KernelRescuePost;
            static inline constexpr std::chrono::time_point<std::chrono::utc_clock> NULL_TIMEPOINT = std::chrono::time_point<std::chrono::utc_clock>::max(); 

            KernelRescuePost(std::atomic<std::chrono::time_point<std::chrono::utc_clock>> ts) noexcept: ts(std::move(ts)){}

            auto heartbeat() noexcept -> exception_t{

                this->ts.exchange(std::chrono::utc_clock::now(), std::memory_order_relaxed);
                return dg::network_exception::SUCCESS;
            }

            auto last_heartbeat() noexcept -> std::expected<std::optional<std::chrono::time_point<std::chrono::utc_clock>>, exception_t>{

                std::chrono::time_point<std::chrono::utc_clock> rs = this->ts.load(std::memory_order_relaxed);

                if (rs == Self::NULL_TIMEPOINT) [[unlikely]]{
                    return std::optional<std::chrono::time_point<std::chrono::utc_clock>>(std::nullopt);
                } else [[likely]]{
                    return std::optional<std::chrono::time_point<std::chrono::utc_clock>>(rs);
                }
            }

            void reset() noexcept{

                this->ts.exchange(Self::NULL_TIMEPOINT, std::memory_order_relaxed);
            }
    };

    class RetransmissionController: public virtual RetransmissionControllerInterface{

        private:

            dg::deque<QueuedPacket> pkt_deque; //its complicated, we'd fragment memory bad, we will consider a better implementation of deque
            data_structure::temporal_finite_unordered_set<global_packet_id_t> acked_id_hashset;
            std::chrono::nanoseconds transmission_delay_time;
            size_t max_retransmission_sz;
            size_t pkt_deque_capacity;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            RetransmissionController(dg::deque<QueuedPacket> pkt_deque,
                                     data_structure::temporal_finite_unordered_set<global_packet_id_t> acked_id_hashset,
                                     std::chrono::nanoseconds transmission_delay_time,
                                     size_t max_retransmission_sz,
                                     size_t pkt_deque_capacity,
                                     std::unique_ptr<std::mutex> mtx,
                                     stdx::hdi_container<size_t> consume_sz_per_load) noexcept: pkt_deque(std::move(pkt_deque)),
                                                                                                acked_id_hashset(std::move(acked_id_hashset)),
                                                                                                transmission_delay_time(transmission_delay_time),
                                                                                                max_retransmission_sz(max_retransmission_sz),
                                                                                                pkt_deque_capacity(pkt_deque_capacity),
                                                                                                mtx(std::move(mtx)),
                                                                                                consume_sz_per_load(std::move(consume_sz_per_load)){}

            void add_retriables(std::move_iterator<Packet *> pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto now                = std::chrono::utc_clock::now();
                Packet * base_pkt_arr   = pkt_arr.base(); 

                for (size_t i = 0u; i < sz; ++i){
                    if (this->pkt_deque.size() == this->pkt_deque_capacity){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    if (base_pkt_arr[i].retransmission_count >= this->max_retransmission_sz){ //it seems like this is the packet responsibility yet I think this is the retransmission responsibility - to avoid system flooding
                        exception_arr[i] = dg::network_exception::NOT_RETRANSMITTABLE;
                        continue;
                    }

                    QueuedPacket queued_pkt             = {};
                    queued_pkt.pkt                      = std::move(base_pkt_arr[i]);
                    queued_pkt.pkt.retransmission_count += 1;
                    queued_pkt.queued_time              = now;
                    this->pkt_deque.push_back(std::move(queued_pkt));
                    exception_arr[i]                    = dg::network_exception::SUCCESS;
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
                    this->acked_id_hashset.insert(pkt_id_arr[i]);
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            void get_retriables(Packet * output_pkt_arr, size_t& sz, size_t output_pkt_arr_cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                std::chrono::time_point<std::chrono::utc_clock> time_bar = std::chrono::utc_clock::now() - this->transmission_delay_time;

                auto key            = QueuedPacket{};
                key.queued_time     = time_bar;

                auto last           = std::lower_bound(this->pkt_deque.begin(), this->pkt_deque.end(), 
                                                       key, 
                                                       [](const auto& lhs, const auto& rhs){return lhs.queued_time < rhs.queued_time;});

                size_t barred_sz    = std::distance(this->pkt_deque.begin(), last);
                size_t iterable_sz  = std::min(output_pkt_arr_cap, barred_sz);
                auto new_last       = std::next(this->pkt_deque.begin(), iterable_sz);
                sz                  = 0u;

                for (auto it = this->pkt_deque.begin(); it != new_last; ++it){
                    if (this->acked_id_hashset.contains(it->pkt.id)){
                        continue;
                    }

                    output_pkt_arr[sz++] = std::move(it->pkt);
                }

                this->pkt_deque.erase(this->pkt_deque.begin(), new_last);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }
    };

    class ExhaustionControlledRetransmissionController: public virtual RetransmissionControllerInterface{

        private:

            std::unique_ptr<RetransmissionControllerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller;

        public:

            ExhaustionControlledRetransmissionController(std::unique_ptr<RetransmissionControllerInterface> base,
                                                         std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                         std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller) noexcept: base(std::move(base)),
                                                                                                                                                            executor(std::move(executor)),
                                                                                                                                                            exhaustion_controller(std::move(exhaustion_controller)){}

            void add_retriables(std::move_iterator<Packet *> pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                Packet * pkt_arr_base               = pkt_arr.base();
                Packet * pkt_arr_first              = pkt_arr_base;
                Packet * pkt_arr_last               = std::next(pkt_arr_first, sz);
                exception_t * exception_arr_first   = exception_arr;
                exception_t * exception_arr_last    = std::next(exception_arr_first, sz);
                size_t sliding_window_sz            = sz; 

                auto task = [&, this]() noexcept{
                    this->base->add_retriables(std::make_move_iterator(pkt_arr_first), sliding_window_sz, exception_arr_first);

                    size_t waiting_sz                   = std::count(exception_arr_first, exception_arr_last, dg::network_exception::QUEUE_FULL);
                    exception_t err                     = this->exhaustion_controller->update_waiting_size(waiting_sz); 

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    }

                    exception_t * retriable_eptr_first  = std::find(exception_arr_first, exception_arr_last, dg::network_exception::QUEUE_FULL);
                    exception_t * retriable_eptr_last   = std::find_if(retriable_eptr_first, exception_arr_last, [](exception_t err){return e != dg::network_exception::QUEUE_FULL;});
                    size_t relative_offset              = std::distance(exception_arr_first, retriable_eptr_first);
                    sliding_window_sz                   = std::distance(retriable_eptr_first, retriable_eptr_last);

                    std::advance(pkt_arr_first, relative_offset);
                    std::advance(exception_arr_first, relative_offset);

                    return this->exhaustion_controller->is_should_wait() && (pkt_arr_first == pkt_arr_last); //TODOs: we want to subscribe these guys to a load_balancer system
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

    class BufferFIFOContainer: public virtual BufferContainerInterface{

        private:

            dg::deque<dg::string> buffer_vec;
            size_t buffer_vec_capacity;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            BufferFIFOContainer(dg::deque<dg::string> buffer_vec,
                                size_t buffer_vec_capacity,
                                std::unique_ptr<std::mutex> mtx,
                                stdx::hdi_container<size_t> consume_sz_per_load) noexcept: buffer_vec(std::move(buffer_vec)),
                                                                                           buffer_vec_capacity(buffer_vec_capacity),
                                                                                           mtx(std::move(mtx)),
                                                                                           consume_sz_per_load(std::move(consume_sz_per_load)){}

            void push(std::move_iterator<dg::string *> buffer_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t app_cap  = this->buffer_vec_capacity - this->buffer_vec.size();
                size_t app_sz   = std::min(sz, app_cap);
                size_t old_sz   = this->buffer_vec.size();
                size_t new_sz   = old_sz + app_sz;

                this->buffer_vec.resize(new_sz);

                std::copy(buffer_arr, std::next(buffer_arr, app_sz), std::next(this->buffer_vec.begin(), old_sz));
                std::fill(exception_arr, std::next(exception_arr, app_sz), dg::network_exception::SUCCESS);
                std::fill(std::next(exception_arr, app_sz), std::next(exception_arr, sz), dg::network_exception::QUEUE_FULL);
            }

            void pop(dg::string * output_buffer_arr, size_t& sz, size_t output_buffer_arr_cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                sz          = std::min(output_buffer_arr_cap, this->buffer_vec.size());
                auto first  = this->buffer_vec.begin();
                auto last   = std::next(first, sz);

                std::copy(std::make_move_iterator(first), std::make_move_iterator(last), output_buffer_arr);
                this->buffer_vec.erase(first, last);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }
    };

    class ExhaustionControlledBufferContainer: public virtual BufferContainerInterface{

        private:

            std::unique_ptr<BufferContainerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller;

        public:

            ExhaustionControlledBufferContainer(std::unique_ptr<BufferContainerInterface> base,
                                                std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller) noexcept: base(std::move(base)),
                                                                                                                                                   executor(std::move(executor)),
                                                                                                                                                   exhaustion_controller(std::move(exhaustion_controller)){}

            void push(std::move_iterator<dg::string *> buffer_arr, size_t sz, exception_t * exception_arr) noexcept{

                dg::string * buffer_arr_base        = buffer_arr.base();
                dg::string * buffer_arr_first       = buffer_arr_base;
                dg::string * buffer_arr_last        = std::next(buffer_arr_first, sz);
                exception_t * exception_arr_first   = exception_arr;
                exception_t * exception_arr_last    = std::next(exception_arr_first, sz);
                size_t sliding_window_sz            = sz;

                auto task = [&, this]() noexcept{
                    this->base->push(std::make_move_iterator(buffer_arr_first), sliding_window_sz, exception_arr_first);

                    size_t waiting_sz                   = std::count(exception_arr_first, exception_arr_last, dg::network_exception::QUEUE_FULL);
                    exception_t err                     = this->exhaustion_controller->update_waiting_size(waiting_sz);

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    }

                    exception_t * retriable_arr_first   = std::find(exception_arr_first, exception_arr_last, dg::network_exception::QUEUE_FULL);
                    exception_t * retriable_arr_last    = std::find_if(retriable_arr_first, exception_arr_last, [](exception_t err){return err != dg::network_exception::QUEUE_FULL;});

                    size_t relative_offset              = std::distance(exception_arr_first, retriable_arr_first);
                    sliding_window_sz                   = std::distance(retriable_arr_first, retriable_arr_last);

                    std::advance(buffer_arr_first, relative_offset);
                    std::advance(exception_arr_first, relative_offset); 

                    return this->exhaustion_controller->is_should_wait() && (buffer_arr_first == buffer_arr_last); //TODOs: we want to subscribe these guys to a load_balancer system
                };

                auto virtual_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(std::move(task));
                this->executor->exec(virtual_task);
            }

            void pop(dg::string * output_buffer_arr, size_t& sz, size_t output_buffer_arr_cap) noexcept{

                this->base->pop(output_buffer_arr, sz, output_buffer_arr_cap);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    class PacketFIFOContainer: public virtual PacketContainerInterface{

        private:

            dg::deque<Packet> packet_deque;
            size_t packet_deque_capacity;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;
        
        public:

            PacketFIFOContainer(dg::deque<Packet> packet_deque,
                                size_t packet_deque_capacity,
                                std::unique_ptr<std::mutex> mtx,
                                stdx::hdi_container<size_t> consume_sz_per_load) noexcept: packet_deque(std::move(packet_deque)),
                                                                                           packet_deque_capacity(packet_deque_capacity),
                                                                                           mtx(std::move(mtx)),
                                                                                           consume_sz_per_load(std::move(consume_sz_per_load)){}

            void push(std::move_iterator<Packet *> packet_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t app_cap  = this->packet_deque_capacity - this->packet_deque.size();
                size_t app_sz   = std::min(sz, app_cap);
                size_t old_sz   = this->packet_deque.size();
                size_t new_sz   = old_sz + app_sz;

                this->packet_deque.resize(new_sz);

                std::copy(packet_arr, std::next(packet_arr, app_sz), std::next(this->packet_deque.begin(), old_sz));
                std::fill(exception_arr, std::next(exception_arr, app_sz), dg::network_exception::SUCCESS);
                std::fill(std::next(exception_arr, app_sz), std::next(exception_arr, sz), dg::network_exception::QUEUE_FULL);
            }

            void pop(Packet * output_pkt_arr, size_t& sz, size_t output_pkt_arr_cap){

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                sz          = std::min(output_pkt_arr_cap, this->packet_deque.size());
                auto first  = this->packet_deque.begin();
                auto last   = std::next(first, sz);

                std::copy(std::make_move_iterator(first), std::make_move_iterator(last), output_pkt_arr);
                this->packet_deque.erase(first, last);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
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

            void push(std::move_iterator<Packet *> pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto less           = [](const Packet& lhs, const Packet& rhs){return lhs.priority < rhs.priority;};
                auto base_pkt_arr   = pkt_arr.base();

                for (size_t i = 0u; i < sz; ++i){
                    if (this->packet_vec.size() == this->packet_vec_capacity){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    this->packet_vec.push_back(std::move(base_pkt_arr[i]));
                    std::push_heap(this->packet_vec.begin(), this->packet_vec.end(), less); //TODOs: optimizables
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }     

            void pop(Packet * output_pkt_arr, size_t& sz, size_t output_pkt_arr_capacity) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto less       = [](const Packet& lhs, const Packet& rhs){return lhs.priority < rhs.priority;};
                sz              = std::min(this->output_pkt_arr_capacity, this->packet_vec.size());
                Packet * out_it = output_pkt_arr; 

                for (size_t i = 0u; i < sz; ++i){
                    std::pop_heap(this->packet_vec.begin(), this->packet_vec.end(), less); //TODOs: optimizables
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

                auto greater            = [](const ScheduledPacket& lhs, const ScheduledPacket& rhs){return lhs.sched_time > rhs.sched_time;};
                Packet * base_pkt_arr   = pkt_arr.base();

                for (size_t i = 0u; i < sz; ++i){
                    if (this->packet_vec.size() == this->packet_vec_capacity){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    std::expected<std::chrono::time_point<std::chrono::utc_clock>, exception_t> sched_time = this->scheduler->schedule(base_pkt_arr[i].to_addr);

                    if (!sched_time.has_value()){
                        exception_arr[i] = sched_time.error();
                        continue;
                    }

                    auto sched_packet       = ScheduledPacket{};
                    sched_packet.pkt        = std::move(base_pkt_arr[i]);
                    sched_packet.sched_time = sched_time.value();

                    this->packet_vec.push_back(std::move(sched_packet));
                    std::push_heap(this->packet_vec.begin(), this->packet_vec.end(), greater); //TODOs: optimizables
                    exception_arr[i]        = dg::network_exception::SUCCESS;
                }
            }

            void pop(Packet * output_pkt_arr, size_t& sz, size_t output_pkt_arr_capacity) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto greater    = [](const ScheduledPacket& lhs, const ScheduledPacket& rhs){return lhs.sched_time > rhs.sched_time;};
                auto time_bar   = std::chrono::utc_clock::now();
                sz              = 0u;

                for (size_t i = 0u; i < output_pkt_arr_capacity; ++i){
                    if (this->packet_vec.empty()){
                        return;
                    }

                    if (this->packet_vec.front().sched_time > time_bar){
                        return;
                    }

                    std::pop_heap(this->packet_vec.begin(), this->packet_vec.end(), greater); //TODOs: optimizables
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
            std::unique_ptr<PacketContainerInterface> request_container;
            std::unique_ptr<PacketContainerInterface> krescue_container;
            size_t ack_accum_sz;
            size_t request_accum_sz; 
            size_t krescue_accum_sz;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            OutboundPacketContainer(std::unique_ptr<PacketContainerInterface> ack_container,
                                    std::unique_ptr<PacketContainerInterface> request_container,
                                    std::unique_ptr<PacketContainerInterface> krescue_container,
                                    size_t ack_accum_sz,
                                    size_t request_accum_sz,
                                    size_t krescue_accum_sz,
                                    stdx::hdi_container<size_t> consume_sz_per_load) noexcept: ack_container(std::move(ack_container)),
                                                                                               request_container(std::move(request_container)),
                                                                                               krescue_container(std::move(krescue_container)),
                                                                                               ack_accum_sz(ack_accum_sz),
                                                                                               request_accum_sz(request_accum_sz),
                                                                                               krescue_accum_sz(krescue_accum_sz),
                                                                                               consume_sz_per_load(std::move(consume_sz_per_load)){}

            void push(std::move_iterator<Packet *> pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                Packet * base_pkt_arr = pkt_arr.base();

                //
                auto ack_push_resolutor                 = InternalPushResolutor{};
                ack_push_resolutor.dst                  = this->ack_container.get();

                size_t trimmed_ack_accum_sz             = std::min(std::min(this->ack_accum_sz, sz), this->ack_container->max_consume_size());
                size_t ack_accumulator_alloc_sz         = dg::network_producer_consumer::delvrsrv_allocation_cost(&ack_push_resolutor, trimmed_ack_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> ack_accumulator_buf(ack_accumulator_alloc_sz);
                auto ack_accumulator                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ack_push_resolutor, trimmed_ack_accum_sz, ack_accumulator_buf.get()));

                //
                auto request_push_resolutor             = InternalPushResolutor{};
                request_push_resolutor.dst              = this->request_container.get();

                size_t trimmed_request_accum_sz         = std::min(std::min(this->request_accum_sz, sz), this->request_container->max_consume_size());
                size_t request_accumulator_alloc_sz     = dg::network_producer_consumer::delvrsrv_allocation_cost(&request_push_resolutor, trimmed_request_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> request_accumulator_buf(request_accumulator_alloc_sz);
                auto request_accumulator                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&request_push_resolutor, trimmed_request_accum_sz, request_accumulator_buf.get()));

                //

                auto krescue_push_resolutor             = InternalPushResolutor{};
                krescue_push_resolutor.dst              = this->krescue_container.get();

                size_t trimmed_krescue_accum_sz         = std::min(std::min(this->krescue_accum_sz, sz), this->krescue_container->max_consume_size());
                size_t krescue_accumulator_alloc_sz     = dg::network_producer_consumer::delvrsrv_allocation_cost(&krescue_push_resolutor, trimmed_krescue_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> krescue_accumulator_buf(krescue_accumulator_alloc_sz);
                auto krescue_accumulator                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&krescue_push_resolutor, trimmed_krescue_accum_sz, krescue_accumulator_buf.get())); 

                //
                std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

                for (size_t i = 0u; i < sz; ++i){
                    auto delivery_arg           = DeliveryArgument{};
                    uint8_t kind                = packet_service::get_packet_polymorphic_type(base_pkt_arr[i]);
                    delivery_arg.pkt_ptr        = std::next(base_pkt_arr, i);
                    delivery_arg.exception_ptr  = std::next(exception_arr, i);
                    delivery_arg.pkt            = std::move(base_pkt_arr[i]);

                    switch (kind){
                        case constants::ack:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(ack_accumulator.get(), std::move(delivery_arg));
                            break;
                        }
                        case constants::request:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(request_accumulator.get(), std::move(delivery_arg));
                            break;
                        }
                        case constants::krescue:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(krescue_accumulator.get(), std::move(delivery_arg));
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

                //this does not look good yet it's probably a good solution - we cant radix this as a binary tree insertion 
                //as request_container uses schedules
                //ack uses asap priority queues (we rely on request's frequency)
                //and other guys use different kinds of containerization

                constexpr size_t CONTAINER_SZ   = 3u;
                using container_ptr_t           = PacketContainerInterface *;
                container_ptr_t container_arr[CONTAINER_SZ]; 
                container_arr[0]                = this->ack_container.get();
                container_arr[1]                = this->suggest_container.get();
                container_arr[2]                = this->krescue_container.get();

                sz                      = 0u;
                Packet * iter_pkt_arr   = output_pkt_arr;
                size_t iter_pkt_arr_cap = output_pkt_arr_capacity; 

                for (size_t i = 0u; i < CONTAINER_SZ; ++i){
                    if (iter_pkt_arr_cap == 0u){
                        return;
                    }

                    size_t tmp_sz       = {};
                    container_arr[i]->pop(iter_pkt_arr, tmp_sz, iter_pkt_arr_cap);

                    std::advance(iter_pkt_arr, tmp_sz);
                    iter_pkt_arr_cap    -= tmp_sz;
                    sz                  += tmp_sz;
                }
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
                    DeliveryArgument * base_data_arr = data_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        pkt_arr[i] = std::move(base_data_arr[i].pkt);
                    }

                    this->dst->push(std::make_move_iterator(pkt_arr.get()), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            *base_data_arr[i].pkt_ptr       = std::move(pkt_arr[i]);
                            *base_data_arr[i].exception_ptr = exception_arr[i];
                        }
                    }
                }
            };
    };

    class InBoundIDController: public virtual InBoundIDControllerInterface{

        private:

            data_structure::temporal_finite_unordered_set<global_packet_id_t> id_hashset;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            InBoundIDController(data_structure::temporal_finite_unordered_set<global_packet_id_t> id_hashset,
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
                    if (this->id_hashset.contains(packet_id_arr[i])){
                        op[i] = false;
                        continue;
                    }

                    this->id_hashset.insert(packet_id_arr[i]);
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
            std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller;

        public:

            ExhaustionControlledPacketContainer(std::unique_ptr<PacketContainerInterface> base,
                                                std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller) noexcept: base(std::move(base)),
                                                                                                                                                   executor(std::move(executor)),
                                                                                                                                                   exhaustion_controller(std::move(exhaustion_controller)){}

            void push(std::move_iterator<Packet *> pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                Packet * pkt_arr_base               = pkt_arr.base();
                Packet * pkt_arr_first              = pkt_arr_base;
                Packet * pkt_arr_last               = std::next(pkt_arr_first, sz);
                exception_t * exception_arr_first   = exception_arr;
                exception_t * exception_arr_last    = std::next(exception_arr_first, sz);
                size_t sliding_window_sz            = sz;

                auto task = [&, this]() noexcept{
                    this->base->push(std::make_move_iterator(pkt_arr_first), sliding_window_sz, exception_arr_first);

                    size_t waiting_sz                   = std::count(exception_arr_first, exception_arr_last, dg::network_exception::QUEUE_FULL);
                    exception_t err                     = this->exhaustion_controller->update_waiting_size(waiting_sz);

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    } 

                    exception_t * retriable_eptr_first  = std::find(exception_arr_first, exception_arr_last, dg::network_exception::QUEUE_FULL);
                    exception_t * retriable_eptr_last   = std::find_if(retriable_eptr_first, exception_arr_last, [](exception_t err){return err != dg::network_exception::QUEUE_FULL;});
                    size_t relative_offset              = std::distance(exception_arr_first, retriable_eptr_first);
                    sliding_window_sz                   = std::distance(retriable_eptr_first, retriable_eptr_last);

                    std::advance(pkt_arr_first, relative_offset);
                    std::advance(exception_arr_first, relative_offset);

                    return this->exhaustion_controller->is_should_wait() && (pkt_arr_first == pkt_arr_last);  //TODOs: we want to subscribe these guys to a load_balancer system
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

    class TrafficController: public virtual TrafficControllerInterface{

        private:

            dg::unordered_unstable_map<Address, size_t> address_counter_map;
            size_t address_cap;
            size_t global_cap;
            size_t map_cap;
            size_t global_counter;

        public:

            TrafficController(dg::unordered_unstable_map<Address, size_t> address_counter_map,
                              size_t address_cap,
                              size_t global_cap,
                              size_t map_cap,
                              size_t global_counter) noexcept: address_counter_map(std::move(address_counter_map)),
                                                               address_cap(address_cap),
                                                               global_cap(global_cap),
                                                               map_cap(map_cap),
                                                               global_counter(global_counter){}

            auto thru(Address addr) noexcept -> std::expected<bool, exception_t>{

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

            void reset() noexcept{

                this->address_counter_map.clear();
                this->global_counter = 0u;
            }
    };

    class InBoundBorderController: public virtual BorderControllerInterface, 
                                   public virtual UpdatableInterface{

        private:

            std::shared_ptr<packet_controller::NATIPControllerInterface> nat_ip_controller;
            std::unique_ptr<packet_controller::TrafficControllerInterface> traffic_controller;
            dg::unordered_set<Address> thru_ip_set;
            dg::unordered_set<Address> inbound_ip_side_set;
            size_t inbound_ip_side_set_cap; //alright, this is the container's responsibility, we just want to make sure that we provide such exhaustion control by providing the interface
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            InBoundBorderController(std::shared_ptr<packet_controller::NATIPControllerInterface> nat_ip_controller,
                                    std::unique_ptr<packet_controller::TrafficControllerInterface> traffic_controller,
                                    dg::unordered_set<Address> thru_ip_set,
                                    dg::unordered_set<Address> inbound_ip_side_set,
                                    size_t inbound_ip_side_set_cap,
                                    std::unique_ptr<std::mutex> mtx,
                                    stdx::hdi_container<size_t> consume_sz_per_load) noexcept: nat_ip_controller(std::move(nat_ip_controller)),
                                                                                               traffic_controller(std::move(traffic_controller)),
                                                                                               thru_ip_set(std::move(thru_ip_set)),
                                                                                               inbound_ip_side_set(std::move(inbound_ip_side_set)),
                                                                                               inbound_ip_side_set_cap(inbound_ip_side_set_cap),
                                                                                               mtx(std::move(mtx)),
                                                                                               consume_sz_per_load(std::move(consume_sz_per_load)){}

            void thru(Address * addr_arr, size_t sz, exception_t * response_exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                std::expected<size_t, exception_t> insert_sz = utility::finite_set_insert(this->inbound_ip_side_set, this->inbound_ip_side_set_cap, 
                                                                                          addr_arr, std::next(addr_arr, sz)); 
                
                if (!insert_sz.has_value() || insert_sz.value() != sz){
                    if (!insert_sz.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(insert_sz.error()));
                    } else{
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    }
                }

                for (size_t i = 0u; i < sz; ++i){
                    if (!this->thru_ip_set.contains(addr_arr[i])){
                        response_exception_arr[i] = dg::network_exception::BAD_IP_RULE; //alright this might be a bug
                        continue;
                    }

                    std::expected<bool, exception_t> traffic_status = this->traffic_controller->thru(addr_arr[i]);

                    if (!traffic_status.has_value()){
                        response_exception_arr[i] = traffic_status.error();
                        continue;
                    }

                    if (!traffic_status.value()){
                        response_exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    response_exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }

            void update() noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                dg::network_stack_allocation::NoExceptAllocation<exception_t[]> ibapp_exception_arr(this->inbound_ip_side_set.size());
                dg::network_stack_allocation::NoExceptAllocation<Address[]> ibapp_ip_arr(this->inbound_ip_side_set.size());

                std::copy(this->inbound_ip_side_set.begin(), this->inbound_ip_side_set.end(), ibapp_ip_arr.get());
                this->nat_ip_controller->add_inbound(ibapp_ip_arr.get(), this->inbound_ip_side_set.size(), ibapp_exception_arr.get()); //.data() is not defined because it is not <new[] contiguous>, it's funny but it's C++ rule

                for (size_t i = 0u; i < this->inbound_ip_side_set.size(); ++i){
                    if (dg::network_exception::is_failed(ibapp_exception_arr[i])){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(ibapp_exception_arr[i]));
                    }
                }

                this->inbound_ip_side_set.clear();
                this->traffic_controller->reset();
                this->thru_ip_set.clear();

                size_t inbound_addr_cap = this->nat_ip_controller->get_inbound_friend_addr_iteration_size();
                size_t inbound_addr_sz  = {};
                dg::network_stack_allocation::NoExceptAllocation<Address[]> addr_arr(inbound_addr_cap);

                this->nat_ip_controller->get_inbound_friend_addr(addr_arr.get(), 0u, inbound_addr_sz, inbound_addr_cap);

                for (size_t i = 0u; i < inbound_addr_sz; ++i){
                    this->thru_ip_set.insert(addr_arr[i]);
                }
            }
    };

    class OutBoundBorderController: public virtual BorderControllerInterface, 
                                    public virtual UpdatableInterface{

        private:

            std::shared_ptr<packet_controller::NATIPControllerInterface> nat_ip_controller;
            std::unique_ptr<packet_controller::TrafficControllerInterface> traffic_controller;
            dg::unordered_set<Address> outbound_ip_side_set;
            size_t outbound_ip_side_set_cap;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            OutBoundBorderController(std::shared_ptr<packet_controller::NATIPControllerInterface> nat_ip_controller,
                                     std::unique_ptr<packet_controller::TrafficControllerInterface> traffic_controller,
                                     dg::unordered_set<Address> outbound_ip_side_set,
                                     size_t outbound_ip_side_set_cap,
                                     std::unique_ptr<std::mutex> mtx,
                                     stdx::hdi_container<size_t> consume_sz_per_load) noexcept: nat_ip_controller(std::move(nat_ip_controller)),
                                                                                                traffic_controller(std::move(traffic_controller)),
                                                                                                outbound_ip_side_set(std::move(outbound_ip_side_set)),
                                                                                                outbound_ip_side_set_cap(outbound_ip_side_set_cap),
                                                                                                mtx(std::move(mtx)),
                                                                                                consume_sz_per_load(std::move(consume_sz_per_load)){}

            void thru(Address * addr_arr, size_t sz, exception_t * response_exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                std::expected<size_t, exception_t> insert_sz = utility::finite_set_insert(this->outbound_ip_side_set, this->outbound_ip_side_set_cap, 
                                                                                          addr_arr, std::next(addr_arr, sz));

                if (!insert_sz.has_value() || insert_sz.value() != sz){
                    if (!insert_sz.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(insert_sz.error()));
                    } else{
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    }                
                }

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<bool, exception_t> traffic_status = this->traffic_controller->thru(addr_arr[i]);

                    if (!traffic_status.has_value()){
                        response_exception_arr[i] = traffic_status.error();
                        continue;
                    }

                    if (!traffic_status.value()){
                        response_exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    response_exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }

            void update() noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                dg::network_stack_allocation::NoExceptAllocation<exception_t[]> obapp_exception_arr(this->outbound_ip_side_set.size());
                dg::network_stack_allocation::NoExceptAllocation<Address[]> obapp_ip_arr(this->outbound_ip_side_set.size());

                std::copy(this->outbound_ip_side_set.begin(), this->outbound_ip_side_set.end(), obapp_ip_arr.get());
                this->nat_ip_controller->add_outbound(obapp_ip_arr.get(), this->outbound_ip_side_set.size(), obapp_exception_arr.get());

                for (size_t i = 0u; i < this->outbound_ip_side_set.size(); ++i){
                    if (dg::network_exception::is_failed(obapp_exception_arr[i])){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(obapp_exception_arr[i]));
                    }
                }

                this->outbound_ip_side_set.clear();
                this->traffic_controller->reset();
            }
    };

    class NATPunchIPController: public virtual NATIPControllerInterface{

        private:

            data_structure::temporal_finite_unordered_set<Address> inbound_ip_set;
            data_structure::temporal_finite_unordered_set<Address> outbound_ip_set;

        public:

            NATPunchIPController(data_structure::temporal_finite_unordered_set<Address> inbound_ip_set,
                                 data_structure::temporal_finite_unordered_set<Address> outbound_ip_set) noexcept: inbound_ip_set(std::move(inbound_ip_set)),
                                                                                                                   outbound_ip_set(std::move(outbound_ip_set)){}

            void add_inbound(Address * addr_arr, size_t sz, exception_t * exception_arr) noexcept{

                size_t insert_cap   = this->inbound_ip_set.capacity();
                size_t insert_sz    = std::min(sz, insert_cap);

                for (size_t i = 0u; i < insert_sz; ++i){
                    this->inbound_ip_set.insert(addr_arr[i]);
                }

                std::fill(exception_arr, std::next(exception_arr, insert_sz), dg::network_exception::SUCCESS);
                std::fill(std::next(exception_arr, insert_sz), std::next(exception_arr, sz), dg::network_exception::RESOURCE_EXHAUSTION);
            }

            void add_outbound(Address * addr_arr, size_t sz, exception_t * exception_arr) noexcept{

                size_t insert_cap   = this->outbound_ip_set.capacity();
                size_t insert_sz    = std::min(sz, insert_cap);

                for (size_t i = 0u; i < insert_sz; ++i){
                    this->outbound_ip_set.insert(addr_arr[i]);
                }

                std::fill(exception_arr, std::next(exception_arr, insert_sz), dg::network_exception::SUCCESS);
                std::fill(std::next(exception_arr, insert_sz), std::next(exception_arr, sz), dg::network_exception::RESOURCE_EXHAUSTION);
            }

            void get_inbound_friend_addr(Address * out_arr, size_t off, size_t& sz, size_t cap) noexcept{

                sz                  = 0u;
                size_t trimmed_off  = std::min(off, this->inbound_ip_set.size());
                auto first          = std::next(this->inbound_ip_set.begin(), trimmed_off);  
                auto last           = this->inbound_ip_set.end();

                for (auto it = first; it != last; ++it){
                    if (sz == cap){
                        return;
                    }

                    if (this->outbound_ip_set.contains(*it)){
                        out_arr[sz++] = *it; 
                    }
                }
            }

            auto get_inbound_friend_addr_iteration_size() noexcept -> size_t{

                return this->inbound_ip_set.size();
            }

            void get_outbound_friend_addr(Address * out_arr, size_t off, size_t& sz, size_t cap) noexcept{

                size_t trimmed_off  = std::min(off, this->outbound_ip_set.size());
                size_t peek_cap     = this->outbound_ip_set.size() - trimmed_off;
                sz                  = std::min(cap, peek_cap); 
                auto first          = std::next(this->outbound_ip_set.begin(), trimmed_off);
                auto last           = std::next(first, sz); 

                std::copy(first, last, out_arr);
            }

            auto get_outbound_friend_addr_iteration_size() noexcept -> size_t{

                return this->outbound_ip_set.size();
            }
    };

    class NATFriendIPController: public virtual NATIPControllerInterface{

        private:

            std::shared_ptr<IPSieverInterface> inbound_ip_siever;
            std::shared_ptr<IPSieverInterface> outbound_ip_siever;
            data_structure::temporal_finite_unordered_set<Address> inbound_friend_set; //we've yet to know whether add_inbound uniqueness of entries is the caller or callee responsibility - let's make it callee responsibility for now
            data_structure::temporal_finite_unordered_set<Address> outbound_friend_set;
            
        public:

            NATFriendIPController(std::shared_ptr<IPSieverInterface> inbound_ip_siever,
                                  std::shared_ptr<IPSieverInterface> outbound_ip_siever,
                                  data_structure::temporal_finite_unordered_set<Address> inbound_friend_set,
                                  data_structure::temporal_finite_unordered_set<Address> outbound_friend_set) noexcept: inbound_ip_siever(std::move(inbound_ip_siever)),
                                                                                                                        outbound_ip_siever(std::move(outbound_ip_siever)),
                                                                                                                        inbound_friend_set(std::move(inbound_friend_set)),
                                                                                                                        outbound_friend_set(std::move(outbound_friend_set)){}

            void add_inbound(Address * addr_arr, size_t sz, exception_t * exception_arr) noexcept{

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<bool, exception_t> is_thru = this->inbound_ip_siever->thru(addr_arr[i]);

                    if (!is_thru.has_value()){
                        exception_arr[i] = is_thru.error();
                        continue;
                    }

                    if (!is_thru.value()){
                        exception_arr[i] = dg::network_exception::BAD_IP_RULE;
                        continue;
                    }

                    this->inbound_friend_set.insert(addr_arr[i]);
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            void add_outbound(Address * addr_arr, size_t sz, exception_t * exception_arr) noexcept{

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<bool, exception_t> is_thru = this->outbound_ip_siever->thru(addr_arr[i]);

                    if (!is_thru.has_value()){
                        exception_arr[i] = is_thru.error();
                        continue;
                    }

                    if (!is_thru.value()){
                        exception_arr[i] = dg::network_exception::BAD_IP_RULE;
                        continue;
                    }

                    this->outbound_friend_set.insert(addr_arr[i]);
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            void get_inbound_friend_addr(Address * addr_arr, size_t off, size_t& sz, size_t cap) noexcept{

                size_t adjusted_off     = std::min(off, this->inbound_friend_set.size()); 
                size_t peek_cap         = this->inbound_friend_set.size() - adjusted_off;
                sz                      = std::min(cap, peek_cap);  
                auto first              = std::next(this->inbound_friend_set.begin(), adjusted_off);
                auto last               = std::next(first, sz);

                std::copy(first, last, addr_arr);
            }

            auto get_inbound_friend_addr_iteration_size() noexcept -> size_t{

                return this->inbound_friend_deque.size();
            }

            void get_outbound_friend_addr(Address * addr_arr, size_t off, size_t& sz, size_t cap) noexcept{

                size_t adjusted_off     = std::min(off, this->outbound_friend_set.size());
                size_t peek_cap         = this->outbound_friend_set.size() - adjusted_off;
                sz                      = std::min(cap, peek_cap);
                auto first              = std::next(this->outbound_friend_set.begin(), adjusted_off);
                auto last               = std::next(first, sz);

                std::copy(first, last, addr_arr);
            }

            auto get_outbound_friend_addr_iteration_size() noexcept -> size_t{

                return this->outbound_friend_deque.size();
            }
    };

    class NATIPController: public virtual NATIPControllerInterface{

        private:

            std::unique_ptr<NATPunchIPController> punch_controller;
            std::unique_ptr<NATFriendIPController> friend_controller;
            std::unique_ptr<std::mutex> mtx;

        public:

            NATIPController(std::unique_ptr<NATPunchIPController> punch_controller,
                            std::unique_ptr<NATFriendIPController> friend_controller,
                            std::unique_ptr<std::mutex> mtx) noexcept: punch_controller(std::move(punch_controller)),
                                                                       friend_controller(std::move(friend_controller)),
                                                                       mtx(std::move(mtx)){}

            void add_inbound(Address * addr_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                dg::network_stack_allocation::NoExceptAllocation<exception_t[]> punch_exception_arr(sz);
                dg::network_stack_allocation::NoExceptAllocation<exception_t[]> friend_exception_arr(sz);

                this->punch_controller->add_inbound(addr_arr, sz, punch_exception_arr.get());
                this->friend_controller->add_inbound(addr_arr, sz, friend_exception_arr.get());

                for (size_t i = 0u; i < sz; ++i){
                    if (dg::network_exception::is_success(punch_exception_arr[i]) && dg::network_exception::is_success(friend_exception_arr[i])){
                        exception_arr[i] = dg::network_exception::SUCCESS;
                    } else{
                        if (dg::network_exception::is_failed(punch_exception_arr[i])){
                            exception_arr[i] = punch_exception_arr[i];
                        } else{
                            exception_arr[i] = friend_exception_arr[i];
                        }
                    }
                }
            }

            void add_outbound(Address * addr_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                dg::network_stack_allocation::NoExceptAllocation<exception_t[]> punch_exception_arr(sz);
                dg::network_stack_allocation::NoExceptAllocation<exception_t[]> friend_exception_arr(sz);

                this->punch_controller->add_outbound(addr_arr, sz, punch_exception_arr.get());
                this->friend_controller->add_outbound(addr_arr, sz, friend_exception_arr.get());

                for (size_t i = 0u; i < sz; ++i){
                    if (dg::network_exception::is_success(punch_exception_arr[i]) && dg::network_exception::is_success(friend_exception_arr[i])){
                        exception_arr[i] = dg::network_exception::SUCCESS;
                    } else{
                        if (dg::network_exception::is_failed(punch_exception_arr[i])){
                            exception_arr[i] = punch_exception_arr[i];
                        } else{
                            exception_arr[i] = friend_exception_arr[i];
                        }
                    }
                }
            }

            void get_inbound_friend_addr(Address * output, size_t off, size_t& sz, size_t cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                size_t punch_controller_sz = this->punch_controller->get_inbound_friend_addr_iteration_size();

                if (off < punch_controller_sz){                    
                    if (off + cap <= punch_controller_sz){
                        this->punch_controller->get_inbound_friend_addr(output, off, sz, cap);
                    } else{                        
                        size_t tmp_sz           = {};
                        this->punch_controller->get_inbound_friend_addr(output, off, tmp_sz, cap);
                        Address * new_output    = std::next(output, tmp_sz);
                        size_t new_off          = 0u;
                        size_t new_sz           = {};
                        size_t new_cap          = cap - tmp_sz; 
                        this->friend_controller->get_inbound_friend_addr(new_output, new_off, new_sz, new_cap);
                        sz                      = tmp_sz + new_sz; 
                    }
                } else{
                    this->friend_controller->get_inbound_friend_addr(output, off - punch_controller_sz, sz, cap);
                }

                sz = std::distance(output, this->setify(output, std::next(output, sz))); //this might be malicious as we are writing data post the agreed-upon sz
            }

            auto get_inbound_friend_addr_iteration_size() noexcept -> size_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                return this->punch_controller->get_inbound_friend_addr_iteration_size() + this->friend_controller->get_inbound_friend_addr_iteration_size();
            }

            void get_outbound_friend_addr(Address * output, size_t off, size_t& sz, size_t cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                size_t punch_controller_sz = this->punch_controller->get_outbound_friend_addr_iteration_size();

                if (off < punch_controller_sz){                    
                    if (off + cap <= punch_controller_sz){
                        this->punch_controller->get_outbound_friend_addr(output, off, sz, cap);
                    } else{
                        size_t tmp_sz           = {};
                        this->punch_controller->get_outbound_friend_addr(output, off, tmp_sz, cap);
                        Address * new_output    = std::next(output, tmp_sz);
                        size_t new_off          = 0u;
                        size_t new_sz           = {};
                        size_t new_cap          = cap - tmp_sz;
                        this->friend_controller->get_outbound_friend_addr(new_output, new_off, new_sz, new_cap);
                        sz                      = tmp_sz + new_sz;
                    }
                } else{
                    this->friend_controller->get_outbound_friend_addr(output, off - punch_controller_sz, sz, cap);
                }

                sz = std::distance(output, this->setify(output, std::next(output, sz))); //this might be malicious as we are writing data post the agreed-upon sz
            }

            auto get_outbound_friend_addr_iteration_size() noexcept -> size_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                return this->punch_controller->get_outbound_friend_addr_iteration_size() + this->friend_controller->get_outbound_friend_addr_iteration_size();
            }

        private:

            inline auto setify(Address * first, Address * last) noexcept -> Address *{

                dg::unordered_set<Address> rs(first, last);
                return std::copy(rs.begin(), rs.end(), first);
            }
    };

    struct ComponentFactory{

        static auto get_asap_scheduler() -> std::unique_ptr<SchedulerInterface>{

            return std::make_unique<ASAPScheduler>();
        }

        static auto get_kernel_outbound_static_transmission_controller(uint32_t transmit_frequency) -> std::unique_ptr<KernelOutBoundTransmissionControllerInterface>{

            const size_t MIN_TRANSMIT_FREQUENCY = size_t{1};
            const size_t MAX_TRANSMIT_FREQUENCY = size_t{1} << 30; 

            if (std::clamp(transmit_frequency, MIN_TRANSMIT_FREQUENCY, MAX_TRANSMIT_FREQUENCY) != transmit_frequency){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ImmutableKernelOutBoundTransmissionController>(transmit_frequency);
        } 

        static auto get_no_exhaustion_controller() -> std::unique_ptr<ExhaustionControllerInterface>{

            return std::make_unique<NoExhaustionController>();
        }

        static auto get_default_exhaustion_controller() -> std::unique_ptr<ExhaustionControllerInterface>{

            return std::make_unique<DefaultExhaustionController>();
        }

        static auto get_incremental_id_generator(factory_id_t factory_id) -> std::unique_ptr<IDGeneratorInterface>{

            return std::make_unique<IDGenerator>(dg::network_randomizer::randomize_int<local_packet_id_t>(), 
                                                 stdx::hdi_container<factory_id_t>{factory_id});
        }

        static auto get_random_id_generator(factory_id_t factory_id) -> std::unique_ptr<IDGeneratorInterface>{

            return std::make_unique<RandomIDGenerator>(factory_id);
        }

        static auto get_randomid_request_packet_generator(factory_id_t factory_id, Address host_addr) -> std::unique_ptr<RequestPacketGeneratorInterface>{

            return std::make_unique<RequestPacketGenerator>(get_random_id_generator(factory_id), 
                                                            host_addr);
        }

        static auto get_randomid_ack_packet_generator(factory_id_t factory_id, Address host_addr) -> std::unique_ptr<AckPacketGeneratorInterface>{

            return std::make_unique<AckPacketGenerator>(get_random_id_generator(factory_id),
                                                        host_addr);
        } 
        
        static auto get_randomid_krescue_packet_generator(factory_id_t factory_id, Address host_addr) -> std::unique_ptr<KRescuePacketGeneratorInterface>{

            return std::make_unique<KRescuePacketGenerator>(get_random_id_generator(factory_id),
                                                            host_addr);
        }

        static auto get_kernel_rescue_post() -> std::unique_ptr<KernelRescuePostInterface>{

            std::atomic<std::chrono::time_point<std::chrono::utc_clock>> arg{};
            arg.exchange(KernelRescuePost::NULL_TIMEPOINT, std::memory_order_seq_cst);

            return std::make_unique<KernelRescuePost>(std::move(arg));
        } 

        static auto get_retransmission_controller(std::chrono::nanoseconds transmission_delay, 
                                                  size_t max_retransmission_sz, 
                                                  size_t idhashset_cap, 
                                                  size_t retransmission_queue_cap,
                                                  size_t consume_factor = 4u) -> std::unique_ptr<RetransmissionControllerInterface>{

            using namespace std::chrono_literals; 

            const std::chrono::nanoseconds MIN_DELAY    = std::chrono::duration_cast<std::chrono::nanoseconds>(1us);
            const std::chrono::nanoseconds MAX_DELAY    = std::chrono::duration_cast<std::chrono::nanoseconds>(60s);
            const size_t MIN_MAX_RETRANSMISSION         = 0u;
            const size_t MAX_MAX_RETRANSMISSION         = 256u;
            const size_t MIN_IDHASHSET_CAP              = 1u;
            const size_t MAX_IDHASHSET_CAP              = size_t{1} << 25; 
            const size_t MIN_RETRANSMISSION_QUEUE_CAP   = 1u;
            const size_t MAX_RETRANSMISSION_QUEUE_CAP   = size_t{1} << 25; 

            if (std::clamp(transmission_delay, MIN_DELAY, MAX_DELAY) != delay){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(max_retransmission_sz, MIN_MAX_RETRANSMISSION, MAX_MAX_RETRANSMISSION) != max_retransmission_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(idhashset_cap, MIN_IDHASHSET_CAP, MAX_IDHASHSET_CAP) != idhashset_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(retransmission_queue_cap, MIN_RETRANSMISSION_QUEUE_CAP, MAX_RETRANSMISSION_QUEUE_CAP) != retransmission_queue_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_retransmission_consume_sz  = retransmission_queue_cap >> consume_factor;
            size_t tentative_idhashset_consume_sz       = idhashset_cap >> consume_factor; 
            size_t consume_sz                           = std::max(std::min(tentative_retransmission_consume_sz, tentative_idhashset_consume_sz), size_t{1u});

            return std::make_unique<RetransmissionController>(dg::deque<QueuedPacket>{},
                                                              data_structure::temporal_finite_unordered_set<global_packet_id_t>(idhashset_cap),
                                                              transmission_delay,
                                                              max_retransmission_sz,
                                                              retransmission_queue_cap,
                                                              std::make_unique<std::mutex>(),
                                                              stdx::hdi_container<size_t>{consume_sz});
        }

        static auto get_exhaustion_controlled_retransmission_controller(std::unique_ptr<RetransmissionControllerInterface> base,
                                                                        std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                                        std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller) -> std::unique_ptr<RetransmissionControllerInterface>{

            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (exhaustion_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledRetransmissionController>(std::move(base), std::move(executor), std::move(exhaustion_controller));
        } 

        static auto get_buffer_fifo_container(size_t buffer_capacity,
                                              size_t consume_factor = 4u) -> std::unique_ptr<BufferContainerInterface>{
            
            const size_t MIN_BUFFER_CAPACITY    = 1u;
            const size_t MAX_BUFFER_CAPACITY    = size_t{1} << 25;

            if (std::clamp(buffer_capacity, MIN_BUFFER_CAPACITY, MAX_BUFFER_CAPACITY) != buffer_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_consume_sz         = buffer_capacity >> consume_factor;
            size_t consume_sz                   = std::max(tentative_consume_sz, size_t{1u});

            return std::make_unique<BufferFIFOContainer>(dg::deque<dg::string>(),
                                                         buffer_capacity,
                                                         std::make_unique<std::mutex>(),
                                                         stdx::hdi_container<size_t>{consume_sz});
        }

        static auto get_exhaustion_controlled_buffer_container(std::unique_ptr<BufferContainerInterface> base,
                                                               std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                               std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller) -> std::unique_ptr<BufferContainerInterface>{
            
            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (exhaustion_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledBufferContainer>(std::move(base),
                                                                         std::move(executor),
                                                                         std::move(exhaustion_controller));
        }

        static auto get_prioritized_packet_container(size_t heap_capacity,
                                                     size_t consume_factor = 4u) -> std::unique_ptr<PacketContainerInterface>{
            
            const size_t MIN_HEAP_CAPACITY  = size_t{1};
            const size_t MAX_HEAP_CAPACITY  = size_t{1} << 25;

            if (std::clamp(heap_capacity, MIN_HEAP_CAPACITY, MAX_HEAP_CAPACITY) != heap_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_consume_sz     = heap_capacity >> consume_factor;
            size_t consume_sz               = std::max(tentative_consume_sz, size_t{1u});
            auto vec                        = dg::vector<Packet>();
            vec.reserve(heap_capacity);

            return std::make_unique<PrioritizedPacketContainer>(std::move(vec),
                                                                heap_capacity,
                                                                std::make_unique<std::mutex>(),
                                                                stdx::hdi_container<size_t>{consume_sz});
        }

        static auto get_packet_fifo_container(size_t packet_vec_capacity,
                                              size_t consume_factor = 4u) -> std::unique_ptr<PacketContainerInterface>{
            
            const size_t MIN_VEC_CAPACITY   = size_t{1};
            const size_t MAX_VEC_CAPACITY   = size_t{1} << 25;

            if (std::clamp(packet_vec_capacity, MIN_VEC_CAPACITY, MAX_VEC_CAPACITY) != packet_vec_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_consume_sz     = packet_vec_capacity >> consume_factor;
            size_t consume_sz               = std::max(tentative_consume_sz, size_t{1u});
            
            return std::make_unique<PacketFIFOContainer>(dg::deque<Packet>(),
                                                         packet_vec_capacity,
                                                         std::make_unique<std::mutex>(),
                                                         stdx::hdi_container<size_t>{consume_sz});
        }

        static auto get_scheduled_packet_container(std::shared_ptr<SchedulerInterface> scheduler,
                                                   size_t packet_vec_capacity,
                                                   size_t consume_factor = 4u) -> std::unique_ptr<PacketContainerInterface>{
            
            const size_t MIN_VEC_CAPACITY   = size_t{1};
            const size_t MAX_VEC_CAPACITY   = size_t{1} << 25;

            if (scheduler == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(packet_vec_capacity, MIN_VEC_CAPACITY, MAX_VEC_CAPACITY) != packet_vec_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_consume_sz     = packet_vec_capacity >> consume_factor;
            size_t consume_sz               = std::max(tentative_consume_sz, size_t{1u});
            auto vec                        = dg::vector<ScheuledPacket>{};
            vec.reserve(packet_vec_capacity); 

            return std::make_unique<ScheduledPacketContainer>(std::move(vec), 
                                                              std::move(scheduler),
                                                              packet_vec_capacity,
                                                              std::make_unique<std::mutex>(),
                                                              stdx::hdi_container<size_t>{consume_sz});
        }

        static auto get_std_outbound_packet_container(size_t ack_capacity,
                                                      size_t request_capacity,
                                                      size_t krescue_capacity,
                                                      size_t accum_sz = constants::DEFAULT_ACCUMULATION_SIZE,
                                                      size_t consume_factor = 4u) -> std::unique_ptr<PacketContainerInterface>{
            
            const size_t MIN_ACCUM_SZ   = 1u;
            const size_t MAX_ACCUM_SZ   = size_t{1} << 25; 

            if (std::clamp(accum_sz, MIN_ACCUM_SZ, MAX_ACCUM_SZ) != accum_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto ack_container  = get_prioritized_packet_container(ack_capacity, consume_factor);
            auto req_container  = get_packet_fifo_container(request_capacity, consume_factor);
            auto rsc_container  = get_packet_fifo_container(krescue_capacity, consume_factor);
            size_t consume_sz   = std::min(std::min(ack_container->max_consume_size(), req_container->max_consume_size()), rsc_container->max_consume_size());

            return std::make_unique<OutBoundPacketContainer>(std::move(ack_container),
                                                             std::move(req_container),
                                                             std::move(rsc_container),
                                                             accum_sz,
                                                             accum_sz,
                                                             accum_sz,
                                                             stdx::hdi_container<size_t>{consume_sz});
        } 

        static auto get_inbound_id_controller(size_t idhashset_cap,
                                              size_t consume_factor = 4u) -> std::unique_ptr<InBoundIDControllerInterface>{
            
            const size_t MIN_IDHASHSET_CAP  = size_t{1};
            const size_t MAX_IDHASHSET_CAP  = size_t{1} << 25;

            if (std::clamp(id_hashset_cap, MIN_IDHASHSET_CAP, MAX_IDHASHSET_CAP) != id_hashset_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_consume_sz     = idhashset_cap >> consume_factor;
            size_t consume_sz               = std::max(tentative_consume_sz, size_t{1u});

            return std::make_unique<InBoundIDController>(data_structure::temporal_finite_unordered_set<global_packet_id_t>(idhashset_cap), 
                                                         std::make_unique<std::mutex>(),
                                                         stdx::hdi_container<size_t>{consume_sz});
        }

        static auto get_synchronous_traffic_controller(size_t peraddr_capacity, 
                                                       size_t global_capacity, 
                                                       size_t addr_capacity) -> std::unique_ptr<TrafficControllerInterface>{

            const size_t MIN_PERADDR_CAPACITY   = 1u;
            const size_t MAX_PERADDR_CAPACITY   = size_t{1} << 25;
            const size_t MIN_GLOBAL_CAPACITY    = 1u;
            const size_t MAX_GLOBAL_CAPACITY    = size_t{1} << 25;
            const size_t MIN_ADDR_CAPACITY      = 1u;
            const size_t MAX_ADDR_CAPACITY      = size_t{1} << 25;

            if (std::clamp(peraddr_capacity, MIN_PERADDR_CAPACITY, MAX_PERADDR_CAPACITY) != peraddr_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(global_capacity, MIN_GLOBAL_CAPACITY, MAX_GLOBAL_CAPACITY) != global_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(addr_capacity, MIN_ADDR_CAPACITY, MAX_ADDR_CAPACITY) != addr_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto address_counter_map    = dg::unordered_unstable_map<Address, size_t>{};
            size_t global_counter       = 0u; 

            address_counter_map.reserve(addr_capacity);

            return std::make_unique<TrafficController>(std::move(address_counter_map), 
                                                       peraddr_capacity, 
                                                       global_capacity, 
                                                       addr_capacity, 
                                                       global_counter);
        }

        static auto get_inbound_border_controller(std::shared_ptr<packet_controller::NATIPControllerInterface> natip_controller,
                                                  size_t peraddr_capacity,
                                                  size_t global_capacity,
                                                  size_t addr_capacity,
                                                  size_t side_update_buf_capacity,
                                                  size_t consume_factor = 4u) -> std::unique_ptr<InBoundBorderController>{ //we cant return interface, this is an implementation that leverages subcribed update to resolve internal states, not solely through the BorderControllerInterface (which is consistent interface of implementations)

            if (natip_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            const size_t MIN_SIDE_UPDATE_BUF_CAPACITY   = size_t{1};
            const size_t MAX_SIDE_UPDATE_BUF_CAPACITY   = size_t{1} << 25;

            if (std::clamp(side_update_buf_capacity, MIN_SIDE_UPDATE_BUF_CAPACITY, MAX_SIDE_UPDATE_BUF_CAPACITY) != side_update_buf_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            dg::unordered_set<Address> thru_ip_set{};
            dg::unordered_set<Address> inbound_ip_side_set{};

            inbound_ip_side_set.reserve(side_update_buf_capacity);

            size_t tentative_consume_sz = side_update_buf_capacity >> consume_factor;
            size_t consume_sz           = std::max(tentative_consume_sz, size_t{1u});

            return std::make_unique<InBoundBorderController>(std::move(natip_controller),
                                                             get_synchronous_traffic_controller(peraddr_capacity, global_capacity, addr_capacity),
                                                             std::move(thru_ip_set),
                                                             thru_ip_set_capacity,
                                                             std::move(inbound_ip_side_set),
                                                             side_update_buf_capacity,
                                                             std::make_unique<std::mutex>(),
                                                             stdx::hdi_container<size_t>{consume_sz});
        }

        static auto get_outbound_border_controller(std::shared_ptr<packet_controller::NATIPControllerInterface> natip_controller,
                                                   size_t peraddr_capacity,
                                                   size_t global_capacity,
                                                   size_t addr_capacity,
                                                   size_t side_update_buf_capacity,
                                                   size_t consume_factor = 4u) -> std::unique_ptr<OutBoundBorderController>{
            
            if (natip_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            const size_t MIN_SIDE_UPDATE_BUF_CAPACITY   = size_t{1};
            const size_t MAX_SIDE_UPDATE_BUF_CAPACITY   = size_t{1} << 25;

            if (std::clamp(side_update_buf_capacity, MIN_SIDE_UPDATE_BUF_CAPACITY, MAX_SIDE_UPDATE_BUF_CAPACITY) != side_update_buf_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            dg::unordered_set<Address> outbound_ip_side_set{};
            outbound_ip_side_set.reserve(side_update_buf_capacity);

            size_t tentative_consume_sz = side_update_buf_capacity >> consume_factor;
            size_t consume_sz           = std::max(tentative_consume_sz, size_t{1});

            return std::make_unique<OutBoundBorderController>(std::move(natip_controller),
                                                              get_synchronous_traffic_controller(peraddr_capacity, global_capacity, addr_capacity),
                                                              std::move(outbound_ip_side_set),
                                                              std::make_unique<std::mutex>(),
                                                              stdx::hdi_container<size_t>{consume_sz});
        }

        static auto get_synchronous_natpunch_ip_controller(size_t inbound_set_capacity,
                                                           size_t outbound_set_capacity) -> std::unique_ptr<NATIPControllerInterface>{
            
            const size_t MIN_INBOUND_SET_CAPACITY   = size_t{1};
            const size_t MAX_INBOUND_SET_CAPACITY   = size_t{1} << 25;
            const size_t MIN_OUTBOUND_SET_CAPACITY  = size_t{1};
            const size_t MAX_OUTBOUND_SET_CAPACITY  = size_t{1} << 25;

            if (std::clamp(inbound_set_capacity, MIN_INBOUND_SET_CAPACITY, MAX_INBOUND_SET_CAPACITY) != inbound_set_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(outbound_set_capacity, MIN_OUTBOUND_SET_CAPACITY, MAX_OUTBOUND_SET_CAPACITY) != outbound_set_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<NATPunchIPController>(data_structure::temporal_finite_unordered_set(inbound_set_capacity),
                                                          data_structure::temporal_finite_unordered_set(outbound_set_capacity));
        }

        static auto get_synchronous_natfriend_ip_controller(std::shared_ptr<IPSieverInterface> inbound_rule,
                                                            std::shared_ptr<IPSieverInterface> outbound_rule,
                                                            size_t inbound_set_capacity,
                                                            size_t outbound_set_capacity) -> std::unique_ptr<NATIPControllerInterface>{

            const size_t MIN_INBOUND_SET_CAPACITY   = size_t{1};
            const size_t MAX_INBOUND_SET_CAPACITY   = size_t{1} << 25;
            const size_t MIN_OUTBOUND_SET_CAPACITY  = size_t{1};
            const size_t MAX_OUTBOUND_SET_CAPACITY  = size_t{1} << 25;

            if (inbound_rule == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (outbound_rule == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(inbound_set_capacity, MIN_INBOUND_SET_CAPACITY, MAX_INBOUND_SET_CAPACITY) != inbound_set_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(outbound_set_capacity, MIN_OUTBOUND_SET_CAPACITY, MAX_OUTBOUND_SET_CAPACITY) != outbound_set_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<NATFriendIPController>(inbound_rule,
                                                           outbound_rule,
                                                           data_structure::temporal_finite_unordered_set(inbound_set_capacity),
                                                           data_structure::temporal_finite_unordered_set(outbound_set_capacity));
        }

        static auto get_nat_ip_controller(std::shared_ptr<IPSieverInterface> inbound_rule,
                                          std::shared_ptr<IPSieverInterface> outbound_rule,
                                          size_t inbound_set_capacity,
                                          size_t outbound_set_capacity) -> std::unique_ptr<NATIPControllerInterface>{
            
            return std::make_unique<NATIPController>(get_synchronous_natpunch_ip_controller(inbound_set_capacity, outbound_set_capacity),
                                                     get_synchronous_natfriend_ip_controller(inbound_rule, outbound_rule, inbound_set_capacity, outbound_set_capacity),
                                                     std::make_unique<std::mutex>());
        }

        static auto get_exhaustion_controlled_packet_container(std::unique_ptr<PacketContainerInterface> base, 
                                                               std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> retry_device, 
                                                               std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller) -> std::unique_ptr<PacketContainerInterface>{

            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (retry_device == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (exhaustion_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledPacketContainer>(std::move(base),
                                                                         std::move(retry_device),
                                                                         std::move(exhaustion_controller));
        }
    };
}

namespace dg::network_kernel_mailbox_impl1::worker{

    using namespace dg::network_kernel_mailbox_impl1::model; 

    class OutBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container;
            std::shared_ptr<packet_controller::BorderControllerInterface> border_controller;
            std::shared_ptr<packet_controller::KernelOutBoundTransmissionControllerInterface> exhaustion_controller;
            std::shared_ptr<model::SocketHandle> socket;
            size_t packet_consumption_cap;
            size_t packet_transmit_cap;
            size_t rest_threshold_sz;

        public:

            OutBoundWorker(std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container,
                           std::shared_ptr<packet_controller::BorderControllerInterface> border_controller,
                           std::shared_ptr<packet_controller::KernelOutBoundTransmissionControllerInterface> exhaustion_controller,
                           std::shared_ptr<model::SocketHandle> socket,
                           size_t packet_consumption_cap,
                           size_t packet_transmit_cap,
                           size_t rest_threshold_sz) noexcept: outbound_packet_container(std::move(outbound_packet_container)),
                                                               border_controller(std::move(border_controller)),
                                                               exhaustion_controller(std::move(exhaustion_controller)),
                                                               socket(std::move(socket)),
                                                               packet_consumption_cap(packet_consumption_cap),
                                                               packet_transmit_cap(packet_transmit_cap),
                                                               rest_threshold_sz(rest_threshold_sz){}

            bool run_one_epoch() noexcept{

                dg::network_stack_allocation::NoExceptAllocation<Packet[]> packet_arr(this->packet_consumption_cap);
                size_t success_sz       = {};
                size_t packet_arr_sz    = {};
                this->outbound_packet_container->pop(packet_arr.get(), packet_arr_sz, this->packet_consumption_cap);

                dg::network_stack_allocation::NoExceptAllocation<Address[]> addr_arr(packet_arr_sz);
                dg::network_stack_allocation::NoExceptAllocation<exception_t[]> traffic_response_arr(packet_arr_sz);

                std::tranform(packet_arr.get(), std::next(packet_arr.get(), packet_arr_sz), addr_arr.get(), [](const Packet& pkt){return pkt.to_addr;});
                this->border_controller->thru(addr_arr.get(), packet_arr_sz, traffic_response_arr.get());

                {
                    auto mailchimp_resolutor                    = InternalMailChimpResolutor{};
                    mailchimp_resolutor.socket                  = this->socket.get();
                    mailchimp_resolutor.exhaustion_controller   = this->exhaustion_controller.get();  
                    mailchimp_resolutor.success_counter         = &success_sz;

                    size_t trimmed_mailchimp_delivery_sz        = std::min(this->packet_transmit_cap, packet_arr_sz);
                    size_t mailchimp_deliverer_alloc_sz         = dg::network_producer_consumer::delvrsrv_allocation_cost(&mailchimp_resolutor, trimmed_mailchimp_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> mailchimp_deliverer_mem(mailchimp_deliverer_alloc_sz);
                    auto mailchimp_deliverer                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&mailchimp_resolutor, trimmed_mailchimp_delivery_sz, mailchimp_deliverer_mem.get())); 

                    for (size_t i = 0u; i < packet_arr_sz; ++i){
                        if (dg::network_exception::is_failed(traffic_response_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(traffic_response_arr[i]));
                            continue;
                        }

                        auto mailchimp_arg      = InternalMailChimpArgument{};
                        mailchimp_arg.dst       = packet_arr[i].to_addr;
                        mailchimp_arg.content   = packet_serivce::serialize_packet(std::move(packet_arr[i]));

                        dg::network_producer_consumer::delvrsrv_deliver(mailchimp_deliverer.get(), std::move(mailchimp_arg));
                    }
                }

                return success_sz >= this->rest_threshold_sz;
            }

        private:

            struct InternalMailChimpArgument{
                Address dst;
                dg::string content;
            };

            struct InternalMailChimpResolutor: dg::network_producer_consumer::ConsumerInterface<InternalMailChimpArgument>{

                model::SocketHandle * socket;
                packet_controller::KernelOutBoundTransmissionControllerInterface * exhaustion_controller;
                size_t * success_counter;

                void push(std::move_iterator<InternalMailChimpArgument *> data_arr, size_t sz) noexcept{

                    exception_t mailchimp_freq_update_err           = this->exhaustion_controller->update_waiting_size(sz);

                    if (dg::network_exception::is_failed(mailchimp_freq_update_err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(mailchimp_freq_update_err));
                    } 

                    InternalMailChimpArgument * base_data_arr       = data_arr.base();
                    uint32_t frequency                              = this->exhaustion_controller->get_transmit_frequency();
                    std::chrono::nanoseconds transmit_period        = packet_service::frequency_to_period(frequency);

                    for (size_t i = 0u; i < sz; ++i){
                        exception_t err = socket_service::send_noblock(*this->socket,
                                                                       base_data_arr[i].dst, 
                                                                       base_data_arr[i].content.data(), base_data_arr[i].content.size());

                        if (dg::network_exception::is_failed(err)){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                        } else{
                            *this->success_counter += 1;
                        }

                        dg::network_asynchronous::hardware_sleep(transmit_period);
                    }
                }
            };
    };

    class RetransmissionWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller;
            std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container;
            size_t retransmission_consumption_cap;
            size_t rest_threshold_sz; 

        public:

            RetransmissionWorker(std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller,
                                 std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container,
                                 size_t retransmission_consumption_cap,
                                 size_t rest_threshold_sz) noexcept: retransmission_controller(std::move(retransmission_controller)),
                                                                     outbound_packet_container(std::move(outbound_packet_container)),
                                                                     retransmission_consumption_cap(retransmission_consumption_cap),
                                                                     rest_threshold_sz(rest_threshold_sz){}

            bool run_one_epoch() noexcept{

                size_t success_counter = {};

                {
                    dg::network_stack_allocation::NoExceptAllocation<Packet[]> packet_arr(this->retransmission_consumption_cap);
                    size_t packet_arr_sz                = {};
                    this->retransmission_controller->get_retriables(packet_arr.get(), packet_arr_sz, this->retransmission_consumption_cap);

                    auto delivery_resolutor             = InternalDeliveryResolutor{};
                    delivery_resolutor.retransmit_dst   = this->retransmission_controller.get();
                    delivery_resolutor.container_dst    = this->outbound_packet_container.get();
                    delivery_resolutor.success_counter  = &success_counter;

                    size_t trimmed_delivery_handle_sz   = std::min(std::min(this->retransmission_controller->max_consume_size(), this->outbound_packet_container->max_consume_size()), packet_arr_sz);
                    size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(&delivery_resolutor, trimmed_delivery_handle_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                    auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&delivery_resolutor, trimmed_delivery_handle_sz, dh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), std::move(packet_arr[i]));
                    }
                }

                return success_counter >= this->rest_threshold_sz;
            }

        private:

            struct InternalDeliveryResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                packet_controller::RetransmissionControllerInterface * retransmit_dst;
                packet_controller::PacketContainerInterface * container_dst;
                size_t * success_counter;

                void push(std::move_iterator<Packet *> packet_arr, size_t sz) noexcept{

                    dg::network_exception::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    dg::network_exception::NoExceptAllocation<Packet[]> cpy_packet_arr(sz);

                    Packet * base_packet_arr = packet_arr.base();
                    std::copy(base_packet_arr, std::next(base_packet_arr, sz), cpy_packet_arr.get());
                    this->container_dst->push(std::make_move_iterator(base_packet_arr), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        } else{
                            *this->success_counter += 1;
                        }
                    }

                    this->retransmit_dst->push(std::make_move_iterator(cpy_packet_arr.get()), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };
    };

    class KernelRescueWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container;
            std::shared_ptr<packet_controller::KernelRescuePostInterface> rescue_post;
            std::unique_ptr<packet_controller::KRescuePacketGeneratorInterface> krescue_gen;
            size_t rescue_packet_sz;
            std::chrono::nanoseconds rescue_threshold;

        public:

            KernelRescueWorker(std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container,
                               std::shared_ptr<packet_controller::KernelRescuePostInterface> rescue_post,
                               std::unique_ptr<packet_controller::KRescuePacketGeneratorInterface> krescue_gen,
                               size_t rescue_packet_sz,
                               std::chrono::nanoseconds rescue_threshold) noexcept: outbound_packet_container(std::move(outbound_packet_container)),
                                                                                    rescue_post(std::move(rescue_post)),
                                                                                    krescue_gen(std::move(krescue_gen)),
                                                                                    rescue_packet_sz(rescue_packet_sz),
                                                                                    rescue_threshold(std::move(rescue_threshold)){}

            bool run_one_epoch() noexcept{

                std::expected<std::optional<std::chrono::time_point<std::chrono::utc_clock>>, exception_t> last_heartbeat = this->rescue_post->last_heartbeat();

                if (!last_heartbeat.has_value()){
                    dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(last_heartbeat.error()));
                    return false;
                }

                if (!last_heartbeat.value().has_value()){
                    return false;
                }

                std::chrono::time_point<std::chrono::utc_clock> now = std::chrono::utc_clock::now();
                std::chrono::nanoseconds lapsed                     = std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_heartbeat.value().value());

                if (lapsed < this->rescue_threshold){
                    return false;
                }

                dg::network_log_stackdump::journal_fast_optional("UDP Rescue Packets enrouting");
                dg::network_stack_allocation::NoExceptAllocation<Packet[]> rescue_packet_arr(this->rescue_packet_sz);
                dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(this->rescue_packet_sz);

                auto gen_func = [this]() noexcept{
                    return dg::network_exception_handler::nothrow_log(packet_service::virtualize_krescue_packet(this->krescue_gen->get()));
                };

                std::generate(rescue_packet_arr.get(), std::next(rescue_packet_arr.get(), this->rescue_packet_sz), gen_func);
                this->outbound_packet_container->push(std::make_move_iterator(rescue_packet_arr.get()), this->rescue_packet_sz, exception_arr.get());

                for (size_t i = 0u; i < this->rescue_packet_sz; ++i){
                    if (dg::network_exception::is_failed(exception_arr[i])){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                    }
                }

                return true;
            }
    };

    class InBoundKernelWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::BufferContainerInterface> buffer_container;
            std::shared_ptr<packet_controller::KernelRescuePostInterface> rescue_post;
            std::shared_ptr<model::SocketHandle> socket;
            size_t buffer_accumulation_sz;
            size_t container_delivery_sz;
            size_t pow2_rescue_heartbeat_interval;

        public:

            InBoundKernelWorker(std::shared_ptr<packet_controller::BufferContainerInterface> buffer_container,
                                std::shared_ptr<packet_controller::KernelRescuePostInterface> rescue_post,
                                std::shared_ptr<model::SocketHandle> socket,
                                size_t buffer_accumulation_sz,
                                size_t container_delivery_sz,
                                size_t pow2_rescue_heartbeat_interval) noexcept: buffer_container(std::move(buffer_container)),
                                                                                 rescue_post(std::move(rescue_post)),
                                                                                 socket(std::move(socket)),
                                                                                 buffer_accumulation_sz(buffer_accumulation_sz),
                                                                                 container_delivery_sz(container_delivery_sz){,
                                                                                 pow2_rescue_heartbeat_interval(pow2_rescue_heartbeat_interval)}

            bool run_one_epoch() noexcept{

                auto buffer_delivery_resolutor  = InternalBufferDeliveryResolutor{};
                buffer_delivery_resolutor.dst   = this->buffer_container.get(); 

                size_t adjusted_delivery_sz     = std::min(this->container_delivery_sz, this->buffer_container->max_consume_size());
                size_t bdh_allocation_cost      = dg::network_producer_consumer::delvrsrv_allocation_cost(&buffer_delivery_resolutor, adjusted_delivery_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> bdh_buf(bdh_allocation_cost);
                auto buffer_delivery_handle     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&buffer_delivery_resolutor, adjusted_delivery_sz, bdh_buf.get()));

                for (size_t i = 0u; i < this->buffer_accumulation_sz; ++i){
                    auto bstream    = dg::string(constants::MAXIMUM_MSG_SIZE, ' '); //TODOs: optimizable
                    size_t sz       = {};
                    exception_t err = socket_service::recv_block(*this->socket, bstream.data(), sz, constants::MAXIMUM_MSG_SIZE); //self-ping to rescue - triggered by an observable relaxed atomic variable (updated every 1024 reads for example) - the rescuer is going to look for the relaxed atomic variable to send rescue packets
                                                                                                                                  //recv block to avoid queuing - kernel optimization reads directly from NIC as soon as possible - there is unfortunately no stable interface except for that for more than 20 years - so let's stick with that for the moment being

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                        return false;
                    }

                    bstream.resize(sz);
                    dg::network_producer_consumer::delvrsrv_deliver(buffer_delivery_handle.get(), std::move(bstream));
                    size_t dice = dg::network_randomizer::randomize_int<size_t>() & (this->pow2_rescue_heartbeat_interval - 1u);

                    if (dice == 0u){
                        exception_t rescue_heartbeat_err = this->rescue_post->heartbeat();

                        if (dg::network_exception::is_failed(rescue_heartbeat_err)){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(rescue_heartbeat_err));
                        }
                    }
                }

                return true;
            }

        private:

            struct InternalBufferDeliveryResolutor: dg::network_producer_consumer::ConsumerInterface<dg::string>{

                packet_controller::BufferContainerInterface * dst;

                void push(std::move_iterator<dg::string *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    this->dst->push(data_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };
    };

    //let's try to grasp what's going on here
    //we are to implement an interface of one_request == max_one_receive
    //there is a problem of retransmission
    //why are we doing retransmission for no-transmission-controlled-protocol again?
    //because the packet lost problem is very popular, we are heading for the 99% use-cases of success, we dont really care about the 1%, it's the <request_implements_socket> responsibility

    //consider the <normal_flow> of request
    //client requests server
    //server responds
    //client waits response or timeout, cancelling the request_id
    //if timeout, client would attempt to re-request

    //consider the <event_driven_flow> of request
    //client requests server with <timeout_window> + <transmission_timestamp> 
    //client detaches
    //server processes
    //server sends another <response_request_with_id> + <the_transmission_timestamp> + <the_timeout_window>
    //client processes the request
    //client wakes up, client gets the response

    //if there was no response, client would wait timeout before sending a duplicate_request or invalidation_request (with the full awareness that server might have already processed the old request without acknowledging it)
    //the reason we would want to wait timeout is because we want synchronization of response, otherwise we dont know the state of the server machine post the request

    //because client <went_to_sleep> before sending a duplicate_request, this is a very time-consuming process, we are talking about 100ms for <timeout> because that's the realistic time for server-client real-life comm
    //we are to wait <100ms> for every re-request, and bookkeeping the numbers, its rather very unconvenient, so we'd try to offset the cost by offloading part of the responsibility to socket, so we could smoothen the curve of overheads  
    //plus, we aren't wasting compute, assume that a very heavy request is already processed by the server but its acknowledgement or response is lost, we are wasting 2x compute or 3x compute (assume 50% packet drop rate, we have 25% chance of getting request responses)
    //plus, if there were packet losses, we don't know what to prioritize if we don't attempt to centralize the requests

    //if things went smoothly, such is the transmission -> dst, dst receives, dst responses, we stop retransmission, diplomacy achieved
    //if things didnt go smoothly, ack packet is lost or retranmission occurred before ack packet being acknowledged, dst is responsible for replying the duplicated requests even though the request_packet has already been processed + received
    //the number of ack packets being sent does not exceed the number of requests being sent

    //https://en.wikipedia.org/wiki/C10k_problem
    //everything we need to know about UDP from kernel dev:
    //RSS (+XPS)
    //affinity hint
    //SO_REUSEPORT
    //SO_ATTACH
    //Pin threads
    //Disable GRO
    //Unload iptables
    //Disable validation
    //Disable audit
    //Skip ID calculation
    //Hyper threading
    //Multiple ports
    //increase rx_queues

    //https://aosabook.org/en/v2/nginx.html
    //we essentially implemented a forked version of this in a modern language 

    //it's more complicated than yall think, we invented this gay transmission protocol to do one thing, event_driven requests + memory_region_frequency
    //it's blazingly fast, if correctly calibrated by using appropriate frequencies
    //I admit I have spent more time to think about socket protocol than actually implementing this, because this is a very important piece of performance, most of the time performance constraints aren't from bandwidth but synchronization overheads
    //we'll run the code tmr

    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller;
            std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container;
            std::shared_ptr<packet_controller::PacketContainerInterface> inbound_packet_container;
            std::shared_ptr<packet_controller::BufferContainerInterface> inbound_buffer_container;
            std::shared_ptr<packet_controller::InBoundIDControllerInterface> inbound_id_controller;
            std::shared_ptr<packet_controller::BorderControllerInterface> inbound_border_controller;
            std::unique_ptr<packet_controller::AckPacketGeneratorInterface> ack_packet_gen;
            Address host_addr;
            size_t ack_vectorization_sz;
            size_t inbound_consumption_sz;
            size_t rest_threshold_sz;

        public:

            InBoundWorker(std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller,
                          std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container,
                          std::shared_ptr<packet_controller::PacketContainerInterface> inbound_packet_container,
                          std::shared_ptr<packet_controller::BufferContainerInterface> inbound_buffer_container,
                          std::shared_ptr<packet_controller::InBoundIDControllerInterface> inbound_id_controller,
                          std::shared_ptr<packet_controller::BorderControllerInterface> inbound_border_controller,
                          std::unique_ptr<packet_controller::AckPacketGeneratorInterface> ack_packet_gen,
                          Address host_addr,
                          size_t ack_vectorization_sz,
                          size_t inbound_consumption_sz,
                          size_t rest_threshold_sz) noexcept: retransmission_controller(std::move(retransmission_controller)),
                                                              outbound_packet_container(std::move(outbound_packet_container)),
                                                              inbound_packet_container(std::move(inbound_packet_container)),
                                                              inbound_buffer_container(std::move(inbound_buffer_container)),
                                                              inbound_id_controller(std::move(inbound_id_controller)),
                                                              inbound_border_controller(std::move(inbound_border_controller)),
                                                              ack_packet_gen(std::move(ack_packet_gen)),
                                                              host_addr(std::move(host_addr)),
                                                              ack_vectorization_sz(ack_vectorization_sz),
                                                              inbound_consumption_sz(inbound_consumption_sz),
                                                              rest_threshold_sz(rest_threshold_sz){}

            bool run_one_epoch() noexcept{

                size_t success_counter = {};

                {
                    dg::network_stack_allocation::NoExceptAllocation<dg::string[]> buf_arr(this->inbound_consumption_sz);
                    size_t buf_arr_sz = {};
                    this->inbound_buffer_container->pop(buf_arr.get(), buf_arr_sz, this->inbound_consumption_sz);

                    //

                    auto ackid_delivery_resolutor                           = InternalRetransmissionAckDeliveryResolutor{};
                    ackid_delivery_resolutor.retransmission_controller      = this->retransmission_controller.get();

                    size_t trimmed_ackid_delivery_sz                        = std::min(this->retransmission_controller->max_consume_size(), buf_arr_sz * constants::MAX_ACK_PER_PACKET); //acked_id_sz <= ack_packet_sz * corresponding_ack_pkt_sz <= ack_packet_sz * MAX_ACK_PER_PACKET <= buf_arr_sz * MAX_ACK_PER_PACKET
                    size_t ackid_deliverer_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(&ackid_delivery_resolutor, trimmed_ackid_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> ackid_deliverer_mem(ackid_deliverer_allocation_cost);
                    auto ackid_deliverer                                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ackid_delivery_resolutor, trimmed_ackid_delivery_sz, ackid_deliverer_mem.get())); 

                    //

                    auto ibpkt_delivery_resolutor                           = InternalPacketDeliveryResolutor{};
                    ibpkt_delivery_resolutor.dst                            = this->inbound_packet_container.get();

                    size_t trimmed_ibpkt_delivery_sz                        = std::min(this->inbound_packet_container->max_consume_size(), buf_arr_sz); //in_bound_sz == req_packet_sz <= buf_arr_sz
                    size_t ibpkt_deliverer_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(&ibpkt_delivery_resolutor, trimmed_ibpkt_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> ibpkt_deliverer_mem(ibpkt_deliverer_allocation_cost);
                    auto ibpkt_deliverer                                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ibpkt_delivery_resolutor, trimmed_ibpkt_delivery_sz, ibpkt_deliverer_mem.get()));

                    //

                    auto obpkt_delivery_resolutor                           = InternalPacketDeliveryResolutor{};
                    obpkt_delivery_resolutor.dst                            = this->outbound_packet_container.get();

                    size_t trimmed_obpkt_delivery_sz                        = std::min(this->outbound_packet_container->max_consume_size(), buf_arr_sz); //outbound_sz == accumulated_ack_packet_sz <= ack_packet_sz <= buf_arr_sz  
                    size_t obpkt_deliverer_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(&obpkt_delivery_resolutor, trimmed_obpkt_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> obpkt_deliverer_mem(obpkt_deliverer_allocation_cost);
                    auto obpkt_deliverer                                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&obpkt_delivery_resolutor, trimmed_obpkt_delivery_sz, obpkt_deliverer_mem.get()));

                    //

                    auto ack_vectorizer_resolutor                           = InternalAckVectorizerResolutor{};
                    ack_vectorizer_resolutor.dst                            = &obpkt_deliverer; 
                    ack_vectorizer_resolutor.ack_packet_gen                 = this->ack_packet_gen.get();

                    size_t trimmed_ack_vectorization_sz                     = std::min(this->ack_vectorization_sz, buf_arr_sz); //ack_vectorization_sz <= ack_pkt_sz <= buf_arr_sz
                    size_t ack_vectorizer_allocation_cost                   = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&ack_vectorizer_resolutor, trimmed_ack_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> ack_vectorizer_mem(ack_vectorizer_allocation_cost);
                    auto ack_vectorizer                                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&ack_vectorizer_resolutor, trimmed_ack_vectorization_sz, ack_vectorizer_mem.get())); 

                    //

                    auto thru_ack_delivery_resolutor                        = InternalThruAckResolutor{};
                    thru_ack_delivery_resolutor.packet_id_deliverer         = &ackid_deliverer;

                    size_t trimmed_thru_ack_delivery_sz                     = std::min(constants::DEFAULT_ACCUMULATION_SIZE, buf_arr_sz); //thru_ack_sz <= buf_arr_sz
                    size_t thru_ack_allocation_cost                         = dg::network_producer_consumer::delvrsrv_allocation_cost(&thru_ack_delivery_resolutor, trimmed_thru_ack_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> thru_ack_mem(thru_ack_allocation_cost);
                    auto thru_ack_deliverer                                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&thru_ack_delivery_resolutor, trimmed_thru_ack_delivery_sz, thru_ack_mem.get())); 

                    //

                    auto thru_request_delivery_resolutor                    = InternalThruRequestResolutor{};
                    thru_request_delivery_resolutor.ack_vectorizer          = &ack_vectorizer;
                    thru_request_delivery_resolutor.inbound_deliverer       = &ibpkt_deliverer;

                    size_t trimmed_thru_request_delivery_sz                 = std::min(constants::DEFAULT_ACCUMULATION_SIZE, buf_arr_sz); //thru_request_sz <= buf_arr_sz
                    size_t thru_request_allocation_cost                     = dg::network_producer_consumer::delvrsrv_allocation_cost(&thru_request_delivery_resolutor, trimmed_thru_request_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> thru_request_mem(thru_request_allocation_cost);
                    auto thru_request_deliverer                             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&thru_request_delivery_resolutor, trimmed_thru_request_delivery_sz, thru_request_mem.get())); 

                    //
                    
                    auto thru_krescue_delivery_resolutor                    = InternalThruKRescueResolutor{};

                    size_t trimmed_thru_krescue_delivery_sz                 = std::min(constants::DEFAULT_ACCUMULATION_SIZE, buf_arr_sz); //thru_rescue_sz <= buf_arr_sz
                    size_t thru_krescue_allocation_cost                     = dg::network_producer_consumer::delvrsrv_allocation_cost(&thru_krescue_delivery_resolutor, trimmed_thru_krescue_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> thru_krescue_mem(thru_krescue_allocation_cost);
                    auto thru_krescue_deliverer                             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&thru_krescue_delivery_resolutor, trimmed_thru_krescue_delivery_sz, thru_krescue_mem.get())); 

                    //

                    auto thru_delivery_resolutor                            = InternalThruResolutor{};
                    thru_delivery_resolutor.ack_thru_deliverer              = &thru_ack_deliverer;
                    thru_delivery_resolutor.request_thru_deliverer          = &thru_request_deliverer;
                    thru_delivery_resolutor.krescue_thru_deliverer          = &thru_krescue_deliverer;

                    size_t trimmed_thru_delivery_sz                         = std::min(constants::DEFAULT_ACCUMULATION_SIZE, buf_arr_sz); //thru_sz <= buf_arr_sz
                    size_t thru_delivery_allocation_cost                    = dg::network_producer_consumer::delvrsrv_allocation_cost(&thru_delivery_resolutor, trimmed_thru_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> thru_delivery_mem(thru_delivery_allocation_cost);
                    auto thru_deliverer                                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&thru_delivery_resolutor, trimmed_thru_delivery_sz, thru_delivery_mem.get())); 

                    //

                    auto nothru_ack_delivery_resolutor                      = InternalNoThruAckResolutor{};
                    nothru_ack_delivery_resolutor.ack_vectorizer            = &ack_vectorizer;

                    size_t trimmed_nothru_ack_delivery_sz                   = std::min(constants::DEFAULT_ACCUMULATION_SIZE, buf_arr_sz); //no_thru_sz <= buf_arr_sz
                    size_t nothru_ack_allocation_cost                       = dg::network_producer_consumer::delvrsrv_allocation_cost(&nothru_ack_delivery_resolutor, trimmed_nothru_ack_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> nothru_ack_delivery_mem(nothru_ack_allocation_cost);
                    auto nothru_ack_deliverer                               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&nothru_ack_delivery_resolutor, trimmed_nothru_ack_delivery_sz, nothru_ack_delivery_mem.get())); 

                    //

                    auto inbound_delivery_resolutor                         = InternalInBoundIDResolutor{};
                    inbound_delivery_resolutor.downstream_dst               = &thru_deliverer;
                    inbound_delivery_resolutor.nothru_ack_dst               = &nothru_ack_deliverer;
                    inbound_delivery_resolutor.inbound_id_controller        = this->inbound_id_controller.get();

                    size_t trimmed_inbound_delivery_sz                      = std::min(this->inbound_id_controller->max_consume_size(), buf_arr_sz); //inbound_sz <= buf_arr_sz
                    size_t inbound_allocation_cost                          = dg::network_producer_consumer::delvsrv_allocation_cost(&inbound_delivery_resolutor, trimmed_inbound_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> inbound_mem(inbound_allocation_cost);
                    auto inbound_deliverer                                  = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&inbound_delivery_resolutor, trimmed_inbound_delivery_sz, inbound_mem.get())); 

                    //

                    auto traffic_resolutor                                  = InternalTrafficResolutor{};
                    traffic_resolutor.downstream_dst                        = &inbound_deliverer;
                    traffic_resolutor.border_controller                     = this->inbound_border_controller.get();

                    size_t trimmed_traffic_resolutor_delivery_sz            = std::min(this->inbound_border_controller->max_consume_size(), buf_arr_sz); //traffic_stop_sz <= buf_arr_sz
                    size_t traffic_resolutor_allocation_cost                = dg::network_producer_consumer::delvrsrv_allocation_cost(&traffic_resolutor, trimmed_traffic_resolutor_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> traffic_resolutor_mem(traffic_resolutor_allocation_cost);
                    auto traffic_resolutor_deliverer                        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&traffic_resolutor, trimmed_traffic_resolutor_delivery_sz, traffic_resolutor_mem.get())); 

                    for (size_t i = 0u; i < buf_arr_sz; ++i){
                        std::expected<Packet, exception_t> pkt = packet_service::deserialize_packet(std::move(buf_arr[i]));

                        if (!pkt.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(pkt.error()));
                            continue;
                        }

                        if (pkt->to_addr != this->host_addr){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(dg::network_exception::MALFORMED_PACKET)); //
                            continue;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(traffic_resolutor_deliverer.get(), std::move(pkt.value()));
                    }
                }

                return success_counter >= this->rest_threshold_sz; //this is absurb
            }

        private:

            struct InternalRetransmissionAckDeliveryResolutor: dg::network_producer_consumer::ConsumerInterface<global_packet_id_t>{

                packet_controller::RetransmissionControllerInterface * retransmission_controller;

                void push(std::move_iterator<global_packet_id_t *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    global_packet_id_t * base_data_arr = data_arr.base();
                    this->retransmission_controller->ack(base_data_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

            struct InternalPacketDeliveryResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                packet_controller::PacketContainerInterface * dst;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    this->dst->push(data_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

            struct InternalAckVectorizerResolutor: dg::network_producer_consumer::KVConsumerInterface<Address, PacketBase>{

                dg::network_producer_consumer::DeliveryHandle<Packet> * dst;
                packet_controller::AckPacketGeneratorInterface * ack_packet_gen;

                void push(Address fr_addr, std::move_iterator<PacketBase *> data_arr, size_t sz) noexcept{

                    PacketBase * base_data_arr = data_arr.base();
                    std::expected<AckPacket, exception_t> ack_pkt = this->ack_packet_gen->get(fr_addr, base_data_arr, sz);

                    if (!ack_pkt.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(ack_pkt.error()));
                        return;
                    }

                    Packet virtualized_pkt = dg::network_exception_handler::nothrow_log(packet_service::virtualize_ack_packet(std::move(ack_pkt.value())));
                    dg::network_producer_consumer::delvrsrv_deliver(this->dst, std::move(virtualized_pkt));
                }
            };

            struct InternalTrafficResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::DeliveryHandle<Packet> * downstream_dst;
                packet_controller::BorderControllerInterface * border_controller;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<Address[]> addr_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> response_arr(sz);

                    Packet * base_data_arr = data_arr.base();
                    std::transform(base_data_arr, std::next(base_data_arr, sz), addr_arr.get(), [](const Packet& packet){return packet.fr_addr;});
                    this->border_controller->thru(addr_arr.get(), sz, response_arr.get()); //we dont want to include packet_sz because we always assume packet as a unit, like persons, we dont say that a midget Mexican != an oversized Mexican

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(response_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(response_arr[i]));
                            continue;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(this->downstream_dst, std::move(base_data_arr[i]));
                    }
                }
            };

            struct InternalInBoundIDResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::DeliveryHandle<Packet> * downstream_dst;
                dg::network_producer_consumer::DeliveryHandle<Packet> * nothru_ack_dst;
                packet_controller::InBoundIDControllerInterface * inbound_id_controller;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<global_packet_id_t[]> id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> response_arr(sz);

                    Packet * base_data_arr = data_arr.base();
                    std::transform(base_data_arr, std::next(base_data_arr, sz), id_arr.get(), [](const Packet& packet){return packet.id;});
                    this->inbound_id_controller->thru(id_arr.get(), sz, response_arr.get());

                    using radix_t   = dg::network_producer_consumer::DeliveryHandle<Packet> *;
                    radix_t radix_table[2];
                    radix_table[0]  = this->nothru_ack_dst;
                    radix_table[1]  = this->downstream_dst;

                    for (size_t i = 0u; i < sz; ++i){
                        if (!response_arr[i].has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(response_arr[i].error()));
                            continue;
                        }

                        if (!response_arr[i].value() && packet_service::is_request_packet(base_data_arr[i]) || response_arr[i].has_value()){
                            dg::network_producer_consumer::delvrsrv_deliver(radix_table[static_cast<int>(response_arr[i].has_value())], std::move(base_data_arr[i]));
                        }
                    }
                }
            };

            struct InternalNoThruAckResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::KVDeliveryHandle<Address, PacketBase> * ack_vectorizer;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    Packet * base_data_arr  = data_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_producer_consumer::delvrsrv_deliver(this->ack_vectorizer, base_data_arr[i].fr, static_cast<const PacketBase&>(base_data_arr[i]));
                    }
                }
            };

            struct InternalThruResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::DeliveryHandle<Packet> * ack_thru_deliverer;
                dg::network_producer_consumer::DeliveryHandle<Packet> * request_thru_deliverer;
                dg::network_producer_consumer::DeliveryHandle<Packet> * krescue_thru_deliverer;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    Packet * base_data_arr  = data_arr.base(); 

                    for (size_t i = 0u; i < sz; ++i){                        
                        //this is meaningless without triviality

                        if (packet_service::is_ack_packet(base_data_arr[i])){
                            dg::network_producer_consumer::delvrsrv_deliver(this->ack_thru_deliverer, std::move(base_data_arr[i]));
                        } else if (packet_service::is_request_packet(base_data_arr[i])){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_thru_deliverer, std::move(base_data_arr[i]));
                        } else if (packet_service::is_krescue_packet(base_data_arr[i])){
                            dg::network_producer_consumer::delvrsrv_deliver(this->krescue_thru_deliverer, std::move(base_data_arr[i]));
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    }
                }
            };

            struct InternalThruAckResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::DeliveryHandle<global_packet_id_t> * packet_id_deliverer;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    Packet * base_data_arr = data_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        for (const PacketBase& e: std::get<XOnlyAckPacket>(base_data_arr[i].xonly_content).ack_vec){
                            dg::network_producer_consumer::delvrsrv_deliver(this->packet_id_deliverer, e.id);
                        }
                    }
                }
            };

            struct InternalThruRequestResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::KVDeliveryHandle<Address, PacketBase> * ack_vectorizer;
                dg::network_producer_consumer::DeliveryHandle<Packet> * inbound_deliverer; 

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    Packet * base_data_arr = data_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_producer_consumer::delvrsrv_deliver(this->ack_vectorizer, base_data_arr[i].fr, static_cast<const PacketBase&>(base_data_arr[i]));
                        dg::network_producer_consumer::delvrsrv_deliver(this->inbound_deliverer, std::move(base_data_arr[i]));
                    }
                }
            };

            struct InternalThruKRescueResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    (void) data_arr;
                }
            };
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

        static auto spawn_retransmission_worker(std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller, 
                                                std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            if (retransmission_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (outbound_packet_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<RetransmissionWorker>(std::move(retransmission_controller), std::move(outbound_packet_container));
        }

        static auto spawn_inbound_worker(std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller, 
                                         std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container,
                                         std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container,
                                         std::shared_ptr<packet_controller::InBoundIDControllerInterface> ib_id_controller,
                                         std::shared_ptr<packet_controller::BorderControllerInterface> ib_traffic_controller,
                                         std::shared_ptr<packet_controller::SchedulerInterface> scheduler, 
                                         std::shared_ptr<SocketHandle> socket) -> std::unique_ptr<dg::network_concurrency::WorkerInterface>{
            
            if (retransmission_controller == nullptr){
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

            return std::make_unique<InBoundWorker>(std::move(retransmission_controller), std::move(ob_packet_container), 
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
            std::unique_ptr<packet_controller::RequestPacketGeneratorInterface> packet_gen;
            std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller;
            std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container;
            std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container;
            size_t consume_sz_per_load;

        public:

            RetransmittableMailBoxController(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec, 
                                             std::unique_ptr<packet_controller::RequestPacketGeneratorInterface> packet_gen,
                                             std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller,
                                             std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container,
                                             std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container,
                                             size_t consume_sz_per_load) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                                   packet_gen(std::move(packet_gen)),
                                                                                   retransmission_controller(std::move(retransmission_controller)),
                                                                                   ob_packet_container(std::move(ob_packet_container)),
                                                                                   ib_packet_container(std::move(ib_packet_container)),
                                                                                   consume_sz_per_load(consume_sz_per_load){}

            void send(std::move_iterator<MailBoxArgument *> data_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                MailBoxArgument * base_data_arr                 = data_arr.base();

                auto internal_deliverer                         = InternalOBDeliverer{};
                internal_deliverer.ob_packet_container          = this->ob_packet_container.get();
                internal_deliverer.retransmission_controller    = this->retransmission_controller.get();

                size_t trimmed_ob_delivery_sz                   = std::min(std::min(this->ob_packet_container->max_consume_size(), this->retransmission_controller->max_consume_size()), sz);
                size_t ob_deliverer_allocation_cost             = dg::network_producer_consumer::delvrsrv_allocation_cost(&internal_deliverer, trimmed_ob_delivery_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> ob_deliverer_mem(ob_deliverer_allocation_cost);
                auto ob_deliverer                               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&internal_deliverer, trimmed_ob_delivery_sz, ob_deliverer_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<Packet, exception_t> pkt = this->packet_gen->get(std::move(base_data_arr[i]));

                    if (!pkt.has_value()){
                        exception_arr[i] = pkt.error();
                        continue;
                    }

                    exception_arr[i] = dg::network_exception::SUCCESS;
                    dg::network_producer_consumer::delvrsrv_deliver(ob_deliverer.get(), std::move(pkt.value()));
                }
            }

            void recv(dg::string * output_arr, size_t& sz, size_t capacity) noexcept{

                sz                      = 0u;
                size_t pkt_arr_sz       = {};
                size_t pkt_arr_capacity = capacity; 
                dg::network_stack_allocation::NoExceptAllocation<Packet[]> pkt_arr(pkt_arr_capacity);
                this->ib_packet_container->pop(pkt_arr.get(), pkt_arr_sz, pkt_arr_capacity);

                for (size_t i = 0u; i < pkt_arr_sz; ++i){
                    RequestPacket rq_pkt    = dg::network_exception_handler::nothrow_log(packet_service::devirtualize_request_packet(std::move(pkt_arr[i])));
                    output_arr[sz++]        = std::move(rq_pkt.content);
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load;
            }

        private:

            struct InternalOBDeliverer: dg::network_producer_consumer::ConsumerInterface<Packet>{

                packet_controller::PacketContainerInterface * ob_packet_container;
                packet_controller::RetransmissionControllerInterface * retransmission_controller;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<Packet[]> cpy_data_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    Packet * base_data_arr = data_arr.base();
                    std::copy(base_data_arr, std::next(base_data_arr, sz), cpy_data_arr.get());

                    this->ob_packet_container->push(std::make_move_iterator(base_data_arr), sz, exception_arr);

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }

                    this->retransmission_controller->add_retriables(std::make_move_iterator(cpy_data_arr.get()), sz, exception_arr);

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };
    };

    struct ComponentFactory{

        static auto get_retransmittable_mailbox_controller(std::unique_ptr<packet_controller::InBoundIDControllerInterface> ib_id_controller,
                                                           std::unique_ptr<packet_controller::InBoundTrafficController> ib_traffic_controller,
                                                           std::shared_ptr<packet_controller::WAPScheduler> scheduler,
                                                           std::unique_ptr<model::SocketHandle, socket_service::socket_close_t> socket,
                                                           std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen,
                                                           std::unique_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller,
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

            if (retransmission_controller == nullptr){
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
            std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller_sp    = std::move(retransmission_controller);
            std::shared_ptr<packet_controller::PacketContainerInterface> ob_packet_container_sp             = std::move(ob_packet_container);
            std::shared_ptr<packet_controller::PacketContainerInterface> ib_packet_container_sp             = std::move(ib_packet_container);
            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec                            = {};
 
            for (size_t i = 0u; i < num_inbound_worker; ++i){
                auto worker_ins     = worker::ComponentFactory::spawn_inbound_worker(retransmission_controller_sp, ob_packet_container_sp, ib_packet_container_sp, ib_id_controller_sp, ib_traffic_controller_sp, scheduler_sp, socket_sp);
                auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker_ins)));
                daemon_vec.emplace_back(std::move(daemon_handle));
            }

            for (size_t i = 0u; i < num_outbound_worker; ++i){
                auto worker_ins     = worker::ComponentFactory::spawn_outbound_worker(ob_packet_container_sp, socket_sp);
                auto daemon_handle  = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::COMPUTING_DAEMON, std::move(worker_ins)));
                daemon_vec.emplace_back(std::move(daemon_handle));
            }

            for (size_t i = 0u; i < num_retry_worker; ++i){
                auto worker_ins     = worker::ComponentFactory::spawn_retransmission_worker(retransmission_controller_sp, ob_packet_container_sp);
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
                                                                      std::move(retransmission_controller_sp), std::move(ob_packet_container_sp),
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
        std::unique_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller{};
        std::unique_ptr<packet_controller::PacketGeneratorInterface> packet_gen{};
        std::unique_ptr<packet_controller::PacketContainerInterface> ib_packet_container{};
        std::unique_ptr<packet_controller::PacketContainerInterface> ob_packet_container{}; 
        std::unique_ptr<packet_controller::InBoundIDControllerInterface> ib_id_controller{};
        std::unique_ptr<packet_controller::InBoundTrafficController> ib_traffic_controller{};

        scheduler               = packet_controller::ComponentFactory::get_wap_scheduler(config.sched_rtt_minbound, config.sched_rtt_maxbound, config.sched_rtt_discretization_sz,
                                                                                         config.sched_adjecent_minbound, config.sched_adjecent_maxbound, config.sched_adjecent_discretization_sz,
                                                                                         config.sched_outgoing_maxbound, config.sched_update_interval, config.sched_reset_interval,
                                                                                         config.sched_map_cap, config.sched_rtt_vec_cap);

        retransmission_controller  = packet_controller::ComponentFactory::get_retransmission_controller(config.retransmission_delay, config.retransmission_count, config.global_id_flush_cap, config.retransmission_cap);

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
                                                                              std::move(packet_gen), std::move(retransmission_controller),
                                                                              std::move(ob_packet_container), std::move(ib_packet_container),
                                                                              config.traffic_reset_dur, config.sched_update_interval, 
                                                                              config.num_inbound_worker, config.num_outbound_worker, 
                                                                              config.num_retry_worker);
    }
}

#endif
