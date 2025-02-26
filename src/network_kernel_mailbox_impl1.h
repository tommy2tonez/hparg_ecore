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

    struct PacketBase{
        global_packet_id_t id;
        uint8_t retransmission_count;
        uint8_t priority;
        std::array<uint64_t, 16> port_utc_stamps;
        uint8_t port_stamp_sz;
        bool port_stamp_flag;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(id, retransmission_count, priority, port_utc_stamps, port_stamp_sz, port_stamp_flag);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(id, retransmission_count, priority, port_utc_stamps, port_stamp_sz, port_stamp_flag);
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
    };

    struct RequestPacket: PacketHeader, XOnlyRequestPacket{};

    struct XOnlyAckPacket{
        dg::vector<PacketBase> ack_vec;
    };

    struct AckPacket: PacketHeader, XOnlyAckPacket{};

    struct XOnlySuggestionPacket{
        uint32_t suggested_frequency;
    };

    struct SuggestionPacket: PacketHeader, XOnlySuggestionPacket{};

    struct XOnlyInformPacket{
        uint32_t transmit_frequency;
    };

    struct InformPacket: PacketHeader, XOnlyInformPacket{};

    struct XOnlyKRescuePacket{}; 

    struct KRescuePacket: PacketHeader, XOnlyRescuePacket{}; 

    struct Packet: PacketHeader{
        std::variant<XOnlyRequestPacket, XOnlyAckPacket, XOnlySuggestionPacket, XOnlyInformPacket, XOnlyKRescuePacket> xonly_content;
    };

    struct ScheduledPacket{
        Packet pkt;
        std::chrono::time_point<std::chrono::utc_clock> sched_time;
    };

    struct QueuedPacket{
        Packet pkt;
        std::chrono::time_point<std::chrono::utc_clock> queued_time;            
    };

    struct RTTFeedBack{
        Address addr;
        std::optional<std::chrono::nanoseconds> rtt;
    };

    struct OutBoundFeedBack{
        Address addr;
        uint32_t suggested_frequency; //we have discussed a long conversation - it suffices to transmit every information by just transmiting one variable
    };

    struct InBoundFeedFront{
        Address addr;
        uint32_t transmit_frequency; //we have discussed a long conversation - it suffices to transmit every information by just transmiting one variable
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
        request = 1,
        suggest = 2,
        informr = 3,
        krescue = 4
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
    
    //the flow I'm looking for is a component implements FeedBackMachine + SchedMachine (this guy collects data and offload the job to the transformer to run heuristics + etc)
    //a Scheduler that stays affined and trigger an observable update by calling the SchedMachineInterface
    //a TransmitFrequencySuggestor that stays affined and trigger an observable update by calling the SchedMachineInterface 
    //we want to stay affined for as long as possible - before we invoke a payload for every interval - to offset the cost of memory_ordering + mutex acquisition + etc 
    //we'd hope to stay < 10000 lines of code 
    //ETA: next week - or next next week - we'll see how complex this task is
    //we dont really care about the recv_from(wait) - yet it imposes a very significant overhead on the system (memory orderings)

    class FeedBackMachineInterface{

        public:

            virtual ~FeedBackMachineInterface() noexcept = default;
            virtual void rtt_fb(RTTFeedBack *, size_t, exception_t *) noexcept = 0;
            virtual void outbound_frequency_fb(OutBoundFeedBack *, size_t, exception_t *) noexcept = 0; 
            virtual void inbound_frequency_ff(InBoundFeedFront *, size_t, exception_t *) noexcept = 0;
    };

    class TransmitFrequencySuggestorInterface{

        public:

            virtual ~TransmitFrequencySuggestorInterface() noexcept = default;
            virtual auto get_frequency_suggestion(Address) noexcept -> std::expected<uint32_t, exception_t> = 0;
    };

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

    class KernelOutBoundExhaustionControllerInterface{

        public:

            virtual ~KernelOutBoundExhaustionControllerInterface() noexcept = default;
            virtual auto get_transmit_frequency() noexcept -> size_t = 0;
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

    //we must abstract the SuggestionPacket + InformPacket to not compromise future implementations (we have not thought of the expanding direction yet)
    //this means we have to encapsulate the low level details of what SuggestionPacket or InformPacket mean
    //at the high level - we just know that these guys do NAT handshakes - we inform and suggest each other
    //says 1 packet/ second for example - this should suffice in most scenerios

    class SuggestionPacketGetterInterface{

        public:

            virtual ~SuggestionPacketGetterInterface() noexcept = default;
            virtual auto get(Address) noexcept -> std::expected<SuggestionPacket, exception_t> = 0;
    };

    class InformPacketGetterInterface{

        public:

            virtual ~InformPacketGetterInterface() noexcept = default;
            virtual auto get(Address) noexcept -> std::expected<InformPacket, exception_t> = 0;
    };

    class RetransmissionControllerInterface{

        public:

            virtual ~RetransmissionControllerInterface() noexcept = default;
            virtual void add_retriables(std::move_iterator<Packet *>, size_t, exception_t *) noexcept = 0;
            virtual void ack(global_packet_id_t *, size_t, exception_t *) noexcept = 0;
            virtual void get_retriables(Packet *, size_t&, size_t) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    //I admit the container is strange - yet it's hard to split the responsibilities here
    //we'd try to use an intermediate container to avoid recv delays
    //we want to empty the kernel buffer as fast as possible (this really means that we have no overhead except for reading the kernel buffers)
    //how fast? consuming_rate must be == producing rate
    //otherwise we are (1): incorrect flood management (we are processing outdated data - and ingoring new data)
    //                 (2): consuming_rate < producing_rate == flood
    //                 (3): consuming_rate > producing_rate - dropped packets + delays + wasted energy

    //we are internally calibrating this by using a read-ahead meter - somewhat like a branch predictor - if a queue_full is triggered or queue_empty is triggered - the meter is adjusted accordingly
    //this is also a machine learning model - such is the negative label is full and empty - and positive label is no full no empty (equilibrium of producer and consumer has been achieved)
    //for every real-time practical machine learning model - there must be a model, a normalization layer (to clamp into the correct states), and a computation reset of the model

    //let's get through the list of the optimizables
    //(1): affinity_hint + RPS
    //SO_REUSEPORT
    //SO_ATTACH
    //Pin threads
    //affined rx-queues
    //Disable GRO
    //Unload iptables
    //Disable validation
    //Disable audit
    //Skip ID calculation
    //Hyper threading
    //affined ports
    //we'll be back tomorrow to run the benchs
    //hopefully we can sat the bandwidth

    //alright - we'll postpone this to next week - we'll implement the machine learning algorithm of sparse centrality + try to optimize it as far as we could - we'll run some A* algorithm to optimize the convergence of the torch model - that should be our baseline    
    //our goal is to dynamically change the consumption_sz + production_sz - keep the kernel queue sizable (preferably no kernel container - by using recv_block) - and leave the flood management to the kernel
    //we don't really care if our metrics are accurate - like round-trip time - as long as it is predictable (in the sense of metering the meterables) - the sense maker responsibility is the machine learning model responsibility

    //our implementation of the comm protocol is the guarantee of one_send == max one_recv
    //it can mean a lot of things - one_succcesful_send                     = one_recv
    //                            - one_successful_send                     = no_recv
    //                            - one_unsuccessful_send                   = one_recv (we dont know what happened - kernel panic kicked in and the stack unwinds)
    //                            - one_unsuccessful_send                   = no_recv
    //                            - one_partially_successful_send           = one_recv
    //                            - one_partially_successful_send           = no_recv

    //we want to leverage this to implement our request | respond event_driven protocol - one_request == max_one_respond within a validation window - retry after the validation window is expired (we need to take in time-dilation to avoid bugs) to make sure that the request is not overriden
    //its the programmers' responsibility to know that unsuccessful request could mean successful receive - and write code in a progressive way to avoid bugs - take orphan tiles + adopt tiles for example - a lost orphan order is happened post the adopt order and now we are out of sync
    //90% of the web bugs are from overriden requests - it's extremely hard to code web or frontend - I doubt that there even exists a frontend without bugs
    //I dont even know if apache or nginx implement these time-out methods
    //essentially - the practice of request synchronization is a lost practice - and it's extremely dangerous in certain scenerios where attackers want to do timing attack on clients or servers
    //in the world of software engineering - if you dont abstract your components - you wont make it
    //there are literally thousands of updates along the development - and you can't change the calling stack every time you have a new update
    //this is stable engineering - even though there are old dinosaur folks who would disagree with me on this matter
    //it's literally impossible to design things in a hollistic picture - we just care what the component does, what the component needs, and do dependency injection

    //says now I want all exhaustion controlled containers to be subcripted to a machine learning model - input + output
    //what I'd want to do is to "extend" the interface, do dependency injection of the machine learning model and returns the new result    
    //for best practices, we almost always need composition of base_object interface + extend - and never inheritance
    //because the inheritance of a component exposes the protected + private interface which are not supposed to be exposed
    //in other words, if a class is not extensible via public interface - it should not be extended AT ALL

    //alright the exhaustion control pushes the statistics data to the machine learning model
    //how about the traffic controller? we definitely can't extend the interface for every subscriber
    //this is where multiple interface extension shines - we want to inherit the TrafficControllerInterface and the TrafficDataCollectibleInterface
    //the machine learning model now "observes" the traffic_controller intervally and pull the traffic data

    //we don't have such blessing in the world of C - this is precisely why C++ is good if correctly practiced

    //"real life" programming is that - each component is independent of each other - it does not look "dry" but when they fuse together - it just makes sense 
    //we want to be accurate - maybe maintainable - debuggable - and provide an abstraction of initialization + reset + how to use the component
    //we can't aim to have all of those "practices" - it's hard - abstraction is always your friend in terms of software engineering

    //we'll implement these tomorrow:
    //offset shared_ptr<char[]> - to avoid buffer copy + friends - this is very important for retriables
    //and malloc initialization for std::string(sz, ' ') to avoid cpu + memfetch
    //this is the very hard part - the std::memory_order_seq_cst optimization for std::shared_ptr<char[]> destruction
    //memory fragmentation + locality of reference by using node_lifetime cyclic allocations
    //the thing that we worry is whether the memory being fetched is the correct deinitialization memory as if we dont need memory_order_seq_cst or memory_order_acquire for that matter if we dont do std::destroy(static_cast<object *>(buffer))
    //we'll postpone the implementations because there are negative feedbacks - we dont want to compromise the readability for the unquantifiables
    //this further hinders the serialization optimization on the buffer - and the serialization of memory access + etc. -  this is hard

    //recall that std::shared_ptr<> uses std::memory_order_relaxed for the reference of the pointing block
    //the reason being is the pointer value of std::shared_ptr<> is not the responsibility of the acquisition, and the pointing value of std::shared_ptr<> is also not the responsibility of the acquisition
    //remember that copying std::shared_ptr<> without memory ordering or a lock is also considered malicious - this is where most people get it wrong

    //if referencing memory without synchronization is malicious - then who is responsibile for the memory ordering of the buffers - it is the affined allocator's - the allocator at the time of malloc must guarantee that the underlying data of the returning pointer is synchronized
    //we'd want to do affined memory address free (a memory pool) before we trigger free in the global memory pool or another partially affined memory pool

    //well, we are on track fellas - probably a year or 2 years or 5 years - we'll implement all of the mentioned things correctly  

    //this can actually work really well in the mass requests + atomic_flag relaxed implementation - we only need to wait for one or two requests
    //this is partially the nginx implementation
    //yet we have full control over our p2p system, so we can assume that all requests are event_driven
    //we might need to implement a liasion as an abstraction layer to talk to the system

    //we have considered long and hard - recv_block cannot be replaced by recv_noblock
    //                                 - a write from NIC -> kernel queue is expensive
    //                                 - kernel queue size should be kept small to avoid flood (this includes CPU flood + network flood + system flood - it should be the NIC internal responsibility to not contaminate the system)
    //                                 - we want to have full control over the runtime

    //our 10000 lines of network protocol might not be dry - yet we want to establish basic interfaces:
    //send
    //recv
    //inbound_rule (opt - by using dependency injection std::shared_ptr<IPSieverInterface>)
    //outbound_rule (opt - by using dependency injection std::shared_ptr<IPSieverInterface>)
    //the heart of performance is to event-driven everything - so that's where we are heading

    //things are hard to code - we can't really blame anybody because its our job to move forward based on the given arguments
    //there is no better description than just to keep it under 5000 memory orderings/ second + minimize mutex collisions and hope for the best 
    //I dont know what's wrong with people and their BDSM std::memory_order_consume
    //why can't we just be civil and do compiler concurrent transactional payload where EVERYTHING inside the payload is dirty - I guess we'll never know - we are talking hardware instruction - not compiler's reordering yet
    //smart people tend to optimize things maliciously by thinking acquire and release are two different operations - when most of the time - we almost always need a transactional payload 

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
            virtual void thru(Address *, size_t, std::expected<bool, exception_t> *) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    //an address is considered when it appears in inbound and outbound 
    //an address can also be considered in other manners 
    //this is the interface of friend addresses - we do observer pattern to pull this to traffic controller
    //we want to stay affined for as long as possible

    class NATIPControllerInterface{

        public:

            virtual ~NATIPControllerInterface() noexcept = default;

            virtual void add_inbound(Address *, size_t, exception_t *) noexcept = 0;
            virtual void add_outbound(Address *, size_t, exception_t *) noexcept = 0;

            virtual void get_inbound_friend_addr(Address *, size_t off, size_t& sz, size_t cap) noexcept = 0; 
            virtual auto get_inbound_friend_addr_size() noexcept -> size_t = 0;

            virtual void get_outbound_friend_addr(Address *, size_t off, size_t& sz, size_t cap) noexcept = 0;
            virtual auto get_outbound_friend_addr_size() noexcept -> size_t = 0;

            virtual auto last_update() noexcept -> std::chrono::time_point<std::chrono::utc_clock> = 0;
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
        
        //alright - we'll attempt to do exhaustion control at our end - because the kernel does not provide an official interface for dropping packets - I guess we take the matter into our own hands by using frequencies + "wave_lengths"
        //these are subcriptible variables - which we'd want to do "stock prediction" on
        //recall that system optimization is stock prediction - because the stats we optimizing must almost always be at the top - which has clear support lines to backprop 

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
    
    static auto get_transit_time(const model::Packet& pkt) noexcept -> std::expected<std::chrono::nanoseconds, exception_t>{
        
        using namespace std::chrono_literals; 

        // if (pkt.port_stamp_sz % 2 != 0){
        //     return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        // }

        // const std::chrono::nanoseconds MIN_LAPSED = std::chrono::duration_cast<std::chrono::nanoseconds>(1ns);
        // const std::chrono::nanoseconds MAX_LAPSED = std::chrono::duration_cast<std::chrono::nanoseconds>(100s);
        // std::chrono::nanoseconds lapsed{};

        // for (size_t i = 1; i < pkt.port_stamp_sz; i += 2){
        //     std::chrono::nanoseconds cur    = std::chrono::nanoseconds(pkt.port_utc_stamps[i]);
        //     std::chrono::nanoseconds prev   = std::chrono::nanoseconds(pkt.port_utc_stamps[i - 1]);

        //     if (cur < prev){
        //         return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        //     }

        //     std::chrono::nanoseconds diff   = cur - prev; 

        //     if (std::clamp(diff, MIN_LAPSED, MAX_LAPSED) != diff){
        //         return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        //     }

        //     lapsed += diff;
        // }

        // return lapsed;
    }

    static auto port_stamp(model::Packet& pkt) noexcept -> exception_t{

        // if (pkt.port_stamp_sz >= pkt.port_utc_stamps.size()){
        //     return dg::network_exception::RESOURCE_EXHAUSTION;
        // }

        // pkt.port_utc_stamps[pkt.port_stamp_sz++] = stdx::timestamp_conversion_wrap(stdx::utc_timestamp());
        // return dg::network_exception::SUCCESS;
    }

    static auto virtualize_request_packet(RequestPacket) noexcept -> std::expected<Packet, exception_t>{

    }

    static auto virtualize_rts_ack_packet(RTSAckPacket) noexcept -> std::expected<Packet, exception_t>{

    }

    static auto virtualize_suggest_packet(SuggestPacket) noexcept -> std::expected<Packet, exception_t>{

    }

    static auto virtualize_informr_packet(InformrPacket) noexcept -> std::expected<Packet, exception_t>{

    }

    static auto devirtualize_request_packet(Packet pkt) noexcept -> std::expected<RequestPacket, exception_t>{

    }

    static auto devirtualize_rts_ack_packet(Packet pkt) noexcept -> std::expected<RTSAckPacket, exception_t>{

    }

    static auto devirtualize_suggest_packet(Packet pkt) noexcept -> std::expected<SuggestPacket, exception_t>{

    }

    static auto devirtualize_informr_packet(Packet pkt) noexcept -> std::expected<InformrPacket, exception_t>{

    }

    static inline auto is_request_packet(const Packet&) noexcept -> bool{

    }

    static inline auto is_rts_ack_packet(const Packet&) noexcept -> bool{

    }

    static inline auto is_suggest_packet(const Packet&) noexcept -> bool{

    }

    static inline auto is_informr_packet(const Packet&) noexcept -> bool{

    }

    static inline auto is_hrtbeat_packet(const Packet&) noexcept -> bool{

    }

    static inline auto is_krescue_packet(const Packet&) noexcept -> bool{

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

                auto rs             = GlobalPacketIdentifier{};
                rs.local_packet_id  = this->last_pkt_id.value.fetch_add(1u, std::memory_order_relaxed);
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

            auto get(MailBoxArgument arg) noexcept -> std::expected<RequestPacket, exception_t>{

                if (arg.content.size() > constants::MAX_PACKET_CONTENT_SIZE){
                    return std::unexpected(dg::network_exception::SOCKET_BAD_BUFFER_LENGTH);
                }

                RequestPacket pkt           = {};
                pkt.fr_addr                 = this->host_addr;
                pkt.to_addr                 = std::move(arg.to);
                pkt.id                      = this->id_gen->get();
                pkt.retransmission_count    = 0u;
                pkt.priority                = 0u;
                pkt.port_utc_stamps         = {};
                pkt.port_stamp_sz           = 0u;
                pkt.port_stamp_flag         = true;
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

                AckPacket pkt               = {};
                pkt.fr_addr                 = this->host_addr;
                pkt.to_addr                 = to_addr;
                pkt.id                      = this->id_gen->get();
                pkt.retransmission_count    = 0u;
                pkt.priority                = 0u;
                pkt.port_utc_stamps         = {};
                pkt.port_stamp_sz           = 0u;
                pkt.port_stamp_flag         = true;
                pkt.ack_vec                 = dg::vector<PacketBase>(sz);

                std::copy(pkt_base_arr, std::next(pkt_base_arr, sz), pkt.ack_vec.begin());

                return pkt;
            }
    };

    class SuggestionPacketGetter: public virtual SuggestionPacketGetterInterface{

        private:

            std::unique_ptr<IDGeneratorInterface> id_gen;
            Address host_addr;
            std::shared_ptr<TransmitFrequencySuggestorInterface> suggestor;

        public:

            SuggestionPacketGetter(std::unique_ptr<IDGeneratorInterface> id_gen,
                                   Address host_addr,
                                   std::shared_ptr<TransmitFrequencySuggestorInterface> suggestor) noexcept: id_gen(std::move(id_gen)),
                                                                                                             host_addr(std::move(host_addr)),
                                                                                                             suggestor(std::move(suggestor)){}

            auto get(Address to_addr) noexcept -> std::expected<SuggestionPacket, exception_t>{

                std::expected<uint32_t, exception_t> fq = this->suggestor->get_frequency_suggestion(to_addr);

                if (!fq.has_value()){
                    return std::unexpected(fq.error());
                }

                SuggestionPacket pkt            = {};
                pkt.fr_addr                     = this->host_addr;
                pkt.to_addr                     = to_addr;
                pkt.id                          = this->id_gen->get();
                pkt.retransmission_count        = 0u;
                pkt.priority                    = 0u;
                pkt.port_utc_stamps             = {};
                pkt.port_stamp_sz               = 0u;
                pkt.port_stamp_flag             = true;
                pkt.suggested_frequency         = fq.value();

                return pkt;
            }
    };

    class InformPacketGetter: public virtual InformPacketGetterInterface{

        private:

            std::unique_ptr<IDGeneratorInterface> id_gen;
            Address host_addr;
            std::shared_ptr<TransmitFrequencySuggestorInterface> suggestor;

        public:

            InformPacketGetter(std::unique_ptr<IDGeneratorInterface> id_gen,
                               Address host_addr,
                               std::shared_ptr<TransmitFrequencySuggestorInterface> suggestor) noexcept: id_gen(std::move(id_gen)),
                                                                                                         host_addr(std::move(host_addr)),
                                                                                                         suggestor(std::move(suggestor)){}

            auto get(Address to_addr) noexcept -> std::expected<InformPacket, exception_t>{

                std::expected<uint32_t, exception_t> fq = this->suggestor->get_frequency_suggestion(to_addr);

                if (!fq.has_value()){
                    return std::unexpected(fq.error());
                }

                InformPacket pkt                = {};
                pkt.fr_addr                     = this->host_addr;
                pkt.to_addr                     = to_addr;
                pkt.id                          = this->id_gen->get();
                pkt.retransmission_count        = 0u;
                pkt.priority                    = 0u;
                pkt.port_utc_stamps             = {};
                pkt.port_stamp_sz               = 0u;
                pkt.port_stamp_flag             = true;
                pkt.transmit_frequency          = fq.value();

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

                KRescuePacket pkt               = {};
                pkt.fr_addr                     = this->host_addr;
                pkt.to_addr                     = this->host_addr;
                pkt.id                          = this->id_gen->get();
                pkt.retransmission_count        = 0u;
                pkt.priority                    = 0u;
                pkt.port_utc_stamps             = {};
                pkt.port_stamp_sz               = 0u;
                pkt.port_stamp_flag             = true;

                return pkt;
            }
    };

    class KernelRescuePost: public virtual KernelRescuePostInterface{

        private:

            std::atomic<std::chrono::time_point<std::chrono::utc_clock>> ts;
        
        public:

            using self = KernelRescuePost;
            static inline constexpr std::chrono::time_point<std::chrono::utc_clock> NULL_TIMEPOINT = std::chrono::time_point<std::chrono::utc_clock>::max(); 

            KernelRescuePost(std::atomic<std::chrono::time_point<std::chrono::utc_clock>> ts) noexcept: ts(std::move(ts)){}

            auto heartbeat() noexcept -> exception_t{

                this->ts.exchange(std::chrono::utc_clock::now(), std::memory_order_relaxed);
                return dg::network_exception::SUCCESS;
            }

            auto last_heartbeat() noexcept -> std::expected<std::optional<std::chrono::time_point<std::chrono::utc_clock>>, exception_t>{

                std::chrono::time_point<std::chrono::utc_clock> rs = this->ts.load(std::memory_order_relaxed);

                if (rs == self::NULL_TIMEPOINT) [[unlikely]]{
                    return std::optional<std::chrono::time_point<std::chrono::utc_clock>>(std::nullopt);
                } else [[likely]]{
                    return std::optional<std::chrono::time_point<std::chrono::utc_clock>>(rs);
                }
            }

            void reset() noexcept{

                this->ts.exchange(self::NULL_TIMEPOINT, std::memory_order_relaxed);
            }
    };

    class RetransmissionController: public virtual RetransmissionControllerInterface{

        private:

            dg::deque<QueuedPacket> pkt_deque;
            std::unique_ptr<data_structure::unordered_set_interface<global_packet_id_t>> acked_id_hashset;
            std::chrono::nanoseconds transmission_delay_time;
            size_t max_retransmission;
            size_t pkt_deque_capacity;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            RetransmissionController(dg::deque<QueuedPacket> pkt_deque,
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

                    if (base_pkt_arr[i].retransmission_count >= this->max_retransmission){ //it seems like this is the packet responsibility yet I think this is the retransmission responsibility - to avoid system flooding
                        exception_arr[i] = dg::network_exception::NOT_RETRANSMITTABLE;
                        continue;
                    }

                    Packet pkt                  = std::move(base_pkt_arr[i]);
                    pkt.retransmission_count    += 1;
                    QueuedPacket queued_pkt     = {};
                    queued_pkt.pkt              = std::move(pkt);
                    queued_pkt.queued_time      = now;
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
                sz                  = std::min(output_pkt_arr_cap, barred_sz); 
                auto new_last       = std::next(this->pkt_deque.begin(), sz);
                auto out_iter       = output_pkt_arr;

                for (auto it = this->pkt_deque.begin(); it != new_last; ++it){
                    if (this->acked_id_hashset->contains(it->pkt.id)){
                        continue;
                    }

                    *out_iter = std::move(it->pkt);
                    std::advance(out_iter, 1u);
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

            std::unique_ptr<PacketContainerInterface> rts_ack_container;
            std::unique_ptr<PacketContainerInterface> request_container;
            std::unique_ptr<PacketContainerInterface> suggest_container;
            std::unique_ptr<PacketContainerInterface> informr_container;
            std::unique_ptr<PacketContainerInterface> krescue_container;
            size_t rts_ack_accum_sz;
            size_t request_accum_sz; 
            size_t suggest_accum_sz;
            size_t informr_accum_sz;
            size_t krescue_accum_sz;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            OutboundPacketContainer(std::unique_ptr<PacketContainerInterface> rts_ack_container,
                                    std::unique_ptr<PacketContainerInterface> request_container,
                                    std::unique_ptr<PacketContainerInterface> suggest_container,
                                    std::unique_ptr<PacketContainerInterface> informr_container,
                                    std::unique_ptr<PacketContainerInterface> krescue_container,
                                    size_t rts_ack_accum_sz,
                                    size_t request_accum_sz,
                                    size_t suggest_accum_sz,
                                    size_t informr_accum_sz,
                                    size_t krescue_accum_sz,
                                    stdx::hdi_container<size_t> consume_sz_per_load) noexcept: rts_ack_container(std::move(rts_ack_container)),
                                                                                               request_container(std::move(request_container)),
                                                                                               suggest_container(std::move(suggest_container)),
                                                                                               informr_container(std::move(informr_container)),
                                                                                               krescue_container(std::move(krescue_container)),
                                                                                               rts_ack_accum_sz(rts_ack_accum_sz),
                                                                                               request_accum_sz(request_accum_sz),
                                                                                               suggest_accum_sz(suggest_accum_sz),
                                                                                               informr_accum_sz(informr_accum_sz),
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
                auto rts_ack_push_resolutor             = InternalPushResolutor{};
                rts_ack_push_resolutor.dst              = this->rts_ack_container.get();

                size_t trimmed_rts_ack_accum_sz         = std::min(std::min(this->rts_ack_accum_sz, sz), this->rts_ack_container->max_consume_size());
                size_t rts_ack_accumulator_alloc_sz     = dg::network_producer_consumer::delvrsrv_allocation_cost(&rts_ack_push_resolutor, trimmed_rts_ack_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rts_ack_accumulator_buf(rts_ack_accumulator_alloc_sz);
                auto rts_ack_accumulator                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&rts_ack_push_resolutor, trimmed_rts_ack_accum_sz, rts_ack_accumulator_buf.get()));

                //
                auto request_push_resolutor             = InternalPushResolutor{};
                request_push_resolutor.dst              = this->request_container.get();

                size_t trimmed_request_accum_sz         = std::min(std::min(this->request_accum_sz, sz), this->request_container->max_consume_size());
                size_t request_accumulator_alloc_sz     = dg::network_producer_consumer::delvrsrv_allocation_cost(&request_push_resolutor, trimmed_request_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> request_accumulator_buf(request_accumulator_alloc_sz);
                auto request_accumulator                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&request_push_resolutor, trimmed_request_accum_sz, request_accumulator_buf.get()));

                //
                auto suggest_push_resolutor             = InternalPushResolutor{};
                suggest_push_resolutor.dst              = this->suggest_container.get();

                size_t trimmed_suggest_accum_sz         = std::min(std::min(this->suggest_accum_sz, sz), this->suggest_container->max_consume_size());
                size_t suggest_accumulator_alloc_sz     = dg::network_producer_consumer::delvrsrv_allocation_cost(&suggest_push_resolutor, trimmed_suggest_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> suggest_accumulator_buf(suggest_accumulator_alloc_sz);
                auto suggest_accumulator                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&suggest_push_resolutor, trimmed_suggest_accum_sz, suggest_accumulator_buf.get()));

                //
                auto informr_push_resolutor             = InternalPushResolutor{};
                informr_push_resolutor.dst              = this->informr_container.get();

                size_t trimmed_informr_accum_sz         = std::min(std::min(this->informr_accum_sz, sz), this->informr_container->max_consume_size());
                size_t informr_accumulator_alloc_sz     = dg::network_producer_consumer::delvrsrv_allocation_cost(&informr_push_resolutor, trimmed_informr_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> informr_accumulator_buf(informr_accumulator_alloc_sz);
                auto informr_accumulator                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&informr_push_resolutor, trimmed_informr_accum_sz, informr_accumulator_buf.get()));

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
                    uint8_t kind                = base_pkt_arr[i].kind;
                    delivery_arg.pkt_ptr        = std::next(base_pkt_arr, i);
                    delivery_arg.exception_ptr  = std::next(exception_arr, i);
                    delivery_arg.pkt            = std::move(base_pkt_arr[i]);

                    switch (kind){
                        case constants::rts_ack:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(rts_ack_accumulator.get(), std::move(delivery_arg));
                            break;
                        }
                        case constants::request:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(request_accumulator.get(), std::move(delivery_arg));
                            break;
                        }
                        case constants::suggest:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(suggest_accumulator.get(), std::move(delivery_arg));
                            break;
                        }
                        case constants::informr:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(informr_accumulator.get(), std::move(delivery_arg));
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
                //rts_ack uses asap priority queues (we rely on request's frequency)
                //and other guys use different kinds of containerization

                constexpr size_t CONTAINER_SZ   = 5u;
                using container_ptr_t           = PacketContainerInterface *;
                container_ptr_t container_arr[CONTAINER_SZ]; 
                container_arr[0]                = this->rts_ack_container.get();
                container_arr[1]                = this->suggest_container.get();
                container_arr[2]                = this->informr_container.get();
                container_arr[3]                = this->request_container.get();
                container_arr[4]                = this->krescue_container.get();

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
            std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller;

        public:

            ExhaustionControlledPacketContainer(std::unique_ptr<PacketContainerInterface> base,
                                                std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller) noexcept: base(std::move(base)),
                                                                                                                                                   executor(std::move(executor)),
                                                                                                                                                   exhaustion_controller(std::move(exhaustion_controller)){}

            void push(std::move_iterator<Packet *> pkt_arr, size_t sz, exception_t * exception_arr) noexcept{

                Packet * pkt_arr_raw                = pkt_arr.base();
                Packet * pkt_arr_first              = pkt_arr_raw;
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

    class InBoundTrafficController: public virtual TrafficControllerInterface, public virtual UpdatableInterface{

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

    class NATPunchIPController: public virtual NATIPControllerInterface{

        private:

            std::unique_ptr<datastructure::unordered_set_interface<Address>> inbound_ip_set; //
            std::unique_ptr<datastructure::unordered_set_interface<Address>> outbound_ip_set; //
            dg::deque<Address> inbound_addr_deque;
            size_t inbound_addr_deque_cap;
            dg::deque<Address> outbound_addr_deque;
            size_t outbound_addr_deque_cap;

        public:

            NATPunchIPController(std::unique_ptr<datastructure::unordered_set_interface<Address>> inbound_ip_set,
                                 std::unique_ptr<datastructure::unordered_set_interface<Address>> outbound_ip_set,
                                 dg::deque<Address> inbound_addr_deque,
                                 size_t inbound_addr_deque_cap,
                                 dg::deque<Address> outbound_addr_deque,
                                 size_t outbound_addr_deque_cap) noexcept: inbound_ip_set(std::move(inbound_ip_set)),
                                                                           outbound_ip_set(std::move(outbound_ip_set)),
                                                                           inbound_addr_deque(std::move(inbound_addr_deque)),
                                                                           inbound_addr_deque_cap(inbound_addr_deque_cap),
                                                                           outbound_addr_deque(std::move(outbound_addr_deque)),
                                                                           outbound_addr_deque_cap(outbound_addr_deque_cap){}

            void add_inbound(Address *, size_t, exception_t *) noexcept{
                
            }

            void add_outbound(Address *, size_t, exception_t *) noexcept{

            }

            void get_inbound_friend_addr(Address *, size_t off, size_t& sz, size_t cap) noexcept{

            }

            auto get_inbound_friend_addr_size() noexcept -> size_t{

            }

            void get_outbound_friend_addr(Address *, size_t off, size_t& sz, size_t cap) noexcept{

            }

            auto get_outbound_friend_addr_size() noexcept -> size_t{

            }
    };

    class NATFriendIPController: public virtual NATIPControllerInterface{

        private:

            std::shared_ptr<IPSieverInterface> inbound_ip_siever;
            std::shared_ptr<IPSieverInterface> outbound_ip_siever;
            dg::set<Address> inbound_friend_set; //we've yet to know whether add_inbound uniqueness of entries is the caller or callee responsibility - let's make it callee responsibility for now
            size_t inbound_friend_set_cap;
            dg::set<Address> outbound_friend_set;
            size_t outbound_friend_set_cap;

        public:

            NATFriendIPController(std::shared_ptr<IPSieverInterface> inbound_ip_siever,
                                  std::shared_ptr<IPSieverInterface> outbound_ip_siever,
                                  dg::deque<Address> inbound_friend_set,
                                  size_t inbound_friend_set_cap,
                                  dg::deque<Address> outbound_friend_set,
                                  size_t outbound_friend_set_cap) noexcept: inbound_ip_siever(std::move(inbound_ip_siever)),
                                                                            outbound_ip_siever(std::move(outbound_ip_siever)),
                                                                            inbound_friend_set(std::move(inbound_friend_set)),
                                                                            inbound_friend_set_cap(inbound_friend_set_cap),
                                                                            outbound_friend_set(std::move(outbound_friend_set)),
                                                                            outbound_friend_set_cap(outbound_friend_set_cap){}

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

                    this->finite_set_insert(this->inbound_friend_set, this->inbound_friend_set_cap, addr_arr[i]);
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

                    this->finite_set_insert(this->outbound_friend_set, this->outbound_friend_set_cap, addr_arr[i]);
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            void get_inbound_friend_addr(Address * addr_arr, size_t off, size_t& sz, size_t cap) noexcept{

                size_t adjusted_off     = std::min(off, this->inbound_friend_set.size()); 
                size_t max_topping_sz   = this->inbound_friend_set.size() - adjusted_off;
                sz                      = std::min(cap, max_topping_sz);  
                auto first              = std::next(this->inbound_friend_set.begin(), adjusted_off);
                auto last               = std::next(first, sz);

                std::copy(first, last, addr_arr);
            }

            auto get_inbound_friend_addr_size() noexcept -> size_t{

                return this->inbound_friend_deque.size();
            }

            void get_outbound_friend_addr(Address * addr_arr, size_t off, size_t& sz, size_t cap) noexcept{

                size_t adjusted_off     = std::min(off, this->outbound_friend_set.size());
                size_t max_topping_sz   = this->outbound_friend_set.size() - adjusted_off;
                sz                      = std::min(cap, max_topping_sz);
                auto first              = std::next(this->outbound_friend_set.begin(), adjusted_off);
                auto last               = std::next(first, sz);

                std::copy(first, last, addr_arr);
            }

            auto get_outbound_friend_addr_size() noexcept -> size_t{

                return this->outbound_friend_deque.size();
            }

        private:

            template <class T, class U>
            void finite_set_insert(dg::set<T>& set_obj, size_t cap, U value) noexcept{

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
                this->friend_controller->add_inbound(addr_arr, sz, exception_arr);

                for (size_t i = 0u; i < sz; ++i){
                    if (dg::network_exception::is_failed(exception_arr[i])){
                        auto arg            = InBoundPunchResolutorArgument{};
                        arg.exception_ptr   = std::next(exception_arr, i);
                        arg.addr            = addr_arr[i]; 

                        dg::network_producer_consumer::delvrsrv_deliver(&inbound_punch_resolutor, std::move(arg));
                    }
                }
            }

            void add_outbound(Address * addr_arr, size_t sz, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->friend_controller->add_outbound(addr_arr, sz, exception_arr);

                for (size_t i = 0u; i < sz; ++i){
                    if (dg::network_exception::is_failed(exception_arr[i])){
                        auto arg            = OutBoundPunchResolutorArgument{};
                        arg.exception_ptr   = std::next(exception_arr, i);
                        arg.addr            = addr_arr[i];

                        dg::network_producer_consumer::delvrsrv_deliver(&outbound_punch_resolutor, std::move(arg));
                    }
                }
            }

            void get_inbound_friend_addr(Address * output, size_t off, size_t& sz, size_t cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                size_t punch_controller_sz = this->punch_controller->get_inbound_friend_addr_size();

                if (off < punch_controller_sz){                    
                    if (off + cap <= punch_controller_sz){
                        this->punch_controller->get_inbound_friend_addr(output, off, sz, cap); //
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

                    return;
                }

                this->friend_controller->get_inbound_friend_addr(output, off - punch_controller_sz, sz, cap);
            }

            auto get_inbound_friend_addr_size() noexcept -> size_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                return this->punch_controller->get_inbound_friend_addr_size() + this->friend_controller->get_inbound_friend_addr_size();
            }

            void get_outbound_friend_addr(Address * output, size_t off, size_t& sz, size_t cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                size_t punch_controller_sz = this->punch_controller->get_outbound_friend_addr_size();

                if (off < punch_controller_sz){                    
                    if (off + cap <= punch_controller_sz){
                        this->punch_controller->get_outbound_friend_addr(output, off, sz, cap); //
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

                    return;
                }

                this->friend_controller->get_outbound_friend_addr(output, off - punch_controller_sz, sz, cap);
            }

            auto get_outbound_friend_addr_size() noexcept -> size_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                return this->punch_controller->get_outbound_friend_addr_size() + this->friend_controller->get_outbound_friend_addr_size();
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
     
        static auto get_retransmission_controller(std::chrono::nanoseconds delay, size_t max_retransmission, 
                                               size_t idhashset_cap, size_t retransmission_cap) -> std::unique_ptr<RetransmissionControllerInterface>{

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

            return std::make_unique<RetransmissionController>(dg::deque<std::pair<std::chrono::nanoseconds, Packet>>{},
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

    //we've thought long and hard about what ExhaustionControl means and how that would affect system flood
    //system flood happens when the production_sz > consumption_sz at a random given time
    //when that happens - we want to dynamically adjust the warehouse sizes of the supply chain to "adapt" with the flood - we want to process the data and also dump incoming data at an approriate rate
    //we want to think in terms of extremists - think about what happen when the queue size == 0 or queue_size == inf
    //queue_size == 0 means that we are always up-to-date - and the incoming data is always the lastest data
    //queue_size == inf means that we are processing the data from a light year before - and the lastest data is too far away to be responded - which causes a system crash (the data we are processing is no long needed - and the data we need to process is too far away in the FIFO queue)
    //the middle ground is probably to wait out the "sand-storm" and recalibrate the system

    //in that case, we have an invisible extension of the queue which is the data in the infretry device - the question is how long do we want that invisible queue to be?
    //now is the question of the machine learning problem of stock prediction - what is the next best possible move to achieve the defined goals? This is the question we tried to answer in ballinger project (this is 3rd grade kid stuff) - we want a complex model yet the idea remains  
    //now is the question of whether that would affect the upstream optimization - the answer is no - because we define our goals to be as generic as possible
    //we'll be the firsts to implement prediction based on centrality + heuristic to optimize system

    //these optimizables seem simple yet they are very important in real-life scenerios - where the problem of betweenness centrality + maxflow arise
    //we want a model to actually extract pattern + think to make the best possible min-max move
    //

    class OutBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container;
            std::shared_ptr<packet_controller::TrafficControllerInterface> traffic_controller;
            std::shared_ptr<packet_controller::KernelOutBoundExhaustionControllerInterface> exhaustion_controller;
            std::shared_ptr<model::SocketHandle> socket;
            size_t packet_consumption_cap;
            size_t packet_transmit_cap;
            size_t rest_threshold_sz; 

        public:

            OutBoundWorker(std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container,
                           std::shared_ptr<packet_controller::TrafficControllerInterface> traffic_controller,
                           std::shared_ptr<packet_controller::KernelOutBoundExhaustionControllerInterface> exhaustion_controller,
                           std::shared_ptr<model::SocketHandle> socket,
                           size_t packet_consumption_cap,
                           size_t packet_transmit_cap,
                           size_t rest_threshold_sz) noexcept: outbound_packet_container(std::move(outbound_packet_container)),
                                                               traffic_controller(std::move(traffic_controller)),
                                                               exhaustion_controller(std::move(exhaustion_controller)),
                                                               socket(std::move(socket)),
                                                               packet_consumption_cap(packet_consumption_cap),
                                                               packet_transmit_cap(packet_transmit_cap),
                                                               rest_threshold_sz(rest_threshold_sz){}

            bool run_one_epoch() noexcept{

                size_t success_sz = {};

                {
                    dg::network_stack_allocation::NoExceptAllocation<Packet[]> packet_arr(this->packet_consumption_cap);
                    size_t packet_arr_sz = {};
                    this->outbound_packet_container->pop(packet_arr.get(), packet_arr_sz, this->packet_consumption_cap);

                    //
                    dg::network_stack_allocation::NoExceptAllocation<Address[]> addr_arr(packet_arr_sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> traffic_response_arr(packet_arr_sz);

                    std::tranform(packet_arr.get(), std::next(packet_arr.get(), packet_arr_sz), addr_arr.get(), [](const Packet& pkt){return ptk.to_addr;});
                    this->traffic_controller->thru(addr_arr.get(), packet_arr_sz, traffic_response_arr.get());

                    //
                    auto mailchimp_resolutor                    = InternalMailChimpResolutor{};
                    mailchimp_resolutor.socket                  = this->socket.get();
                    mailchimp_resolutor.exhaustion_controller   = this->exhaustion_controller.get();  
                    mailchimp_resolutor.success_counter         = &success_sz;

                    size_t trimmed_mailchimp_deliverer_sz       = std::min(this->packet_transmit_cap, packet_arr_sz);
                    size_t mailchimp_deliverer_alloc_sz         = dg::network_producer_consumer::delvrsrv_allocation_cost(&mailchimp_resolutor, trimmed_mailchimp_deliverer_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> mailchimp_deliverer_mem(mailchimp_deliverer_alloc_sz);
                    auto mailchimp_deliverer                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&mailchimp_resolutor, trimmed_mailchimp_deliverer_sz, mailchimp_deliverer_mem.get())); 

                    for (size_t i = 0u; i < packet_arr_sz; ++i){
                        if (!traffic_response_arr[i].has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(traffic_response_arr[i].error()));
                            continue;
                        }

                        if (!traffic_response_arr[i].value()){
                            dg::network_log_stackdump::error_fast_optional("Outgoing Packets dropped at OutBoundWorker Traffic Stop");
                            continue;
                        }

                        exception_t stamp_err = packet_service::port_stamp(packet_arr[i]); 

                        if (dg::network_exception::is_failed(stamp_err)){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(stamp_err));
                        }

                        auto mailchimp_arg      = InternalMailChimpArgument{};
                        mailchimp_arg.dst       = packet_arr[i].to_addr;
                        mailchimp_arg.content   = utility::serialize_packet(std::move(packet_arr[i]));

                        dg::network_producer_consumer::delvrsrv_deliver(mailchimp_deliverer.get(), std::move(mailchimp_arg));
                    }
                }

                return success_sz > this->rest_threshold_sz;
            }

        private:

            struct InternalMailChimpArgument{
                Address dst;
                dg::string content;
            };

            struct InternalMailChimpResolutor: dg::network_producer_consumer::ConsumerInterface<InternalMailChimpArgument>{

                model::SocketHandle * socket;
                packet_controller::KernelOutBoundExhaustionControllerInterface * exhaustion_controller;
                size_t * success_counter;

                void push(std::move_iterator<InternalMailChimpArgument *> data_arr, size_t sz) noexcept{

                    exception_t mailchimp_freq_update_err           = this->exhaustion_controller->update_waiting_size(sz);

                    if (dg::network_exception::is_failed(mailchimp_freq_update_err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(mailchimp_freq_update_err));
                    } 

                    InternalMailChimpArgument * base_data_arr       = data_arr.base();
                    size_t frequency                                = this->exhaustion_controller->get_transmit_frequency();
                    std::chrono::nanoseconds transmit_wavelength    = this->frequency_to_wavelength(frequency);

                    for (size_t i = 0u; i < sz; ++i){
                        exception_t err = socket_service::send_noblock(*this->socket, base_data_arr[i].dst, base_data_arr[i].content.data(), base_data_arr[i].content.size());

                        if (dg::network_exception::is_failed(err)){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                        } else{
                            *this->success_counter += 1;
                        }

                        dg::network_asynchronous::hardware_sleep(transmit_wavelength);
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

                return success_counter > this->rest_threshold_sz;
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
                    std::copy(base_packet_arr, std::next(base_packet_arr, sz) cpy_packet_arr.get());
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
            std::shared_ptr<packet_controller::KRescuePacketGeneratorInterface> krescue_gen;
            size_t rescue_packet_sz;
            std::chrono::nanoseconds rescue_threshold;

        public:

            KernelRescueWorker(std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container,
                               std::shared_ptr<packet_controller::KernelRescuePostInterface> rescue_post,
                               std::shared_ptr<packet_controller::KRescuePacketGeneratorInterface> krescue_gen,
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
                    return true;
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

    class InBoundInformerWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::NATIPController> nat_ip_controller;
            std::shared_ptr<packet_controller::InformPacketGetterInterface> inbound_suggestor;
            std::shared_ptr<packet_controller::PacketContainerInterface> outbound_container;
            size_t addr_consumption_sz;
            size_t outbound_container_accum_sz;

        public:

            InBoundInformerWorker(std::shared_ptr<packet_controller::NATIPController> nat_ip_controller,
                                  std::shared_ptr<packet_controller::InformPacketGetterInterface> inbound_suggestor,
                                  std::shared_ptr<packet_controller::PacketContainerInterface> outbound_container,
                                  size_t addr_consumption_sz,
                                  size_t outbound_container_accum_sz) noexcept: nat_ip_controller(std::move(nat_ip_controller)),
                                                                                inbound_suggestor(std::move(inbound_suggestor)),
                                                                                outbound_container(std::move(outbound_container)),
                                                                                addr_consumption_sz(addr_consumption_sz),
                                                                                outbound_container_accum_sz(outbound_container_accum_sz){}

            bool run_one_epoch() noexcept{

                dg::network_stack_allocaiton::NoExceptAllocation<Address[]> addr_arr(this->addr_consumption_sz);
                size_t addr_sz  = {};
                size_t offset   = {};

                auto outbound_delivery_resolutor                = InternalOutBoundDeliveryResolutor{};
                outbound_delivery_resolutor.outbound_container  = this->outbound_container.get();

                size_t outbound_deliverer_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&outbound_delivery_resolutor, this->outbound_container_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> outbound_deliverer_mem(outbound_deliverer_allocation_cost);
                auto outbound_deliverer                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&outbound_delivery_resolutor, this->outbound_container_accum_sz, outbound_deliverer_mem.get()));

                while (true){
                    this->nat_ip_controller->get_outbound_addr(addr_arr.get(), offset, addr_sz, this->addr_consumption_sz);

                    for (size_t i = 0u; i < addr_sz; ++i){
                        std::expected<InformrPacket, exception_t> pkt = this->inbound_suggestor->get(addr_arr[i]);

                        if (!pkt.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(pkt.error()));
                            continue;
                        }

                        std::expected<Packet, exception_t> virt_pkt = packet_service::virtualize_informr_packet(std::move(pkt.value()));

                        if (!virt_pkt.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(virt_pkt.error()));
                            continue;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(outbound_deliverer.get(), std::move(virt_pkt.value()));
                    }

                    offset += addr_sz;

                    if (addr_sz != this->addr_consumption_sz){
                        break;
                    }
                }

                return true; 
            }

        private:

            struct InternalOutBoundDeliveryResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                packet_controller::PacketContainerInterface * outbound_container;

                void push(std::move_iterator<Packet *> pkt_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    this->outbound_container->push(pkt_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }  
            };
    };

    class OutBoundSuggestorWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::NATIPController> nat_ip_controller;
            std::shared_ptr<packet_controller::SuggestionPacketGetterInterface> outbound_suggestor;
            std::shared_ptr<packet_controller::PacketContainerInterface> outbound_container;
            size_t addr_consumption_sz;
            size_t outbound_container_accum_sz; 

        public:

            OutBoundSuggestorWorker(std::shared_ptr<packet_controller::NATIPController> nat_ip_controller,
                                    std::shared_ptr<packet_controller::SuggestionPacketGetterInterface> outbound_suggestor,
                                    std::shared_ptr<packet_controller::PacketContainerInterface> outbound_container,
                                    size_t addr_consumption_sz,
                                    size_t outbound_container_accum_sz) noexcept: nat_ip_controller(std::move(nat_ip_controller)),
                                                                                  outbound_suggestor(std::move(outbound_suggestor)),
                                                                                  outbound_container(std::move(outbound_container)),
                                                                                  addr_consumption_sz(addr_consumption_sz),
                                                                                  outbound_container_accum_sz(outbound_container_accum_sz){}

            bool run_one_epoch() noexcept{

                dg::network_stack_allocation::NoExceptAllocation<Address[]> addr_arr(this->addr_consumption_sz);
                size_t addr_sz  = {};
                size_t offset   = 0u;

                auto outbound_delivery_resolutor                = InternalOutBoundDeliveryResolutor{};
                outbound_delivery_resolutor.outbound_container  = this->outbound_container.get();

                size_t outbound_deliverer_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&outbound_delivery_resolutor, this->outbound_container_accum_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> outbound_deliverer_mem(outbound_deliverer_allocation_cost);
                auto outbound_deliverer                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&outbound_delivery_resolutor, this->outbound_container_accum_sz, outbound_deliverer_mem.get()));

                while (true){
                    this->nat_ip_controller->get_inbound_addr(addr_arr.get(), offset, addr_sz, this->addr_consumption_sz);

                    for (size_t i = 0u; i < addr_sz; ++i){
                        std::expected<SuggestionPacket, exception_t> pkt = this->outbound_suggestor->get(addr_arr[i]);

                        if (!pkt.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(pkt.error()));
                            continue;
                        }

                        std::expected<Packet, exception_t> virt_pkt = packet_service::virtualize_suggest_packet(std::move(pkt.value()));

                        if (!virt_pkt.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(virt_pkt.error()));
                            continue;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(outbound_deliverer.get(), std::move(virt_pkt.value()));
                    }

                    offset += addr_sz; 

                    if (addr_sz != this->addr_consumption_sz){
                        break;
                    }
                }

                return true;
            }

        private:

            struct InternalOutBoundDeliveryResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::packet_controller::PacketContainerInterface * outbound_container;

                void push(std::move_iterator<Packet *> pkt_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    this->outbound_container->push(pkt_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };
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

    //I was trying to think of the general solutions for network transportation
    //what precisely are we trying to achieve - are we trying to saturate + uniformly distribute the bandwidth? 
    //or we are trying to transport packets as fast as possible and leave the optimizable to the frequency regions which allow upstream optimizations?
    //I guess we dont know the answer to that yet - so we are in the middle ground - we optimize things that wont compromise | hinder future decisions - and will circle back to dot-the-i-cross-the-t later

    //let's talk in depth about our two very important variables - frequency & rtt
    //those alone could tell everything we need to know about the congestion

    //recall how we did the ballinger stock model prediction, our semantic space of sorted chart and sliding window
    //well it has its use here

    //f(t)  = round-trip time tells us about the state the receipient is in
    //f1(t) = transmit_frequency + f(t) tells us about the client influence on the server
    //f2(t) = suggested_frequency tells us about the normalization data

    //alright - let's talks in terms of actionables and deliverables
    //assume we are in a perfect environment, such is there is no problem of maxflow - we are a <nice balanced distributed Voronoi diagrams>
    //everyone is everyone - there is no one more special than another (we'll patch this later)
    //we communicate by gossiping - there is no communicado in our graph - it's prohibited
    //we want to "observe" the system and do local optimizations in an acceptable window - this is a path search problem 

    //we want to propagate the system stats to the neighbor - and prop that to another neighbor - what is this called? it's called centrality
    //we want to "compress" the system stats by mapping it into another compact semantic space (pretrained)
    //we want to maximize self thru-put - because we are in a uniformly distributed environment - we can do local optimization to approx global optimization
    //we want to re-route the packets - by exploring neighbor nodes (A * search) - recall that local optimization is global optimization in a uniform distribution
    //we probably dont want to re-route - because it would hinder the holistic upstream optimization (using memregion frequency)

    //the things that are in our control, and do not interfere with upstream optimizations, are the obvious optimizables - like maximizing thru-put ack/s - by mutating the adjustables in an acceptable window (like suggested frequency, consume_sz, produce_sz, etc.)
    //we'll advance in the optimization direction - we'll do very simple things like chartifying the acks per IP, chartifying the CPU consumption, chartifying the network consumption, chartifying the consumption_sz, chartifying the production_sz
    //                                                                               chartifying the round-trip time per IP, chartifying the suggested_frequency per IP, etc.

    //                                            - we want to inch forward in the direction (mutating the suggestion, mutating the production_sz, etc.) that approxes local maxima (we dont really care if we inch in the right direction - we actually care, this does create a ripple effect for training - we want to gather data for the stock market prediction)
    //                                            - alright - this is an entire different algorithm - yet the idea is to "observe" the behavior of the system by using "stock prediction" - and inch the system in the direction that is categorized as beneficial
    //                                            - the way that we did stock prediction is actually good - we established the support lines by mapping data into a new semantic space
    //                                            - I was thinking of how that worked so well - to explain it in a nutshell, assume you have an array, you remove the left most element, and you insert a next element, the former is actually 100% predictable - it has clear rules of vec.erase(vec.begin())
    //                                                                                        -                                                                                                                         the latter punishes based on how far down you are in the insertion sort (if your newly inserter moves n - 1 elements to the right - then it has stronger hint of training than moving 1 element to the right which is push_back)
    //                                                                                                                                                                                                                  this is the clear rule of support line in stock market - stock market ALWAYS goes up - and when it doesn't, then we have a problem, we ask how far down it is through the support lines
    
    //                                            - how does this tell about system calibration? system optimization is like stock optimization, we want to maximize everything - it means that the charts we are optimizing must have a clear uptrend reaches the stop and stay there - stock doesn't behave in the way, this is where it differs 
    //                                                                                                                                                                          - system chart also has support lines - says spending 20% of resource on quicksort is not as beneficial as spending 5% of resource on radix sort(in terms of sorted elements/ second)
    //                                            - we want to collect these data to make "better" decisions in terms of future forcast - which creates a loop of <the forcast is based on the data collected from the heuristics and the heuristics is from the output of the forcast>
    //                                                                                                                                  - we dont really care, because the data we collect is always the base source of truth - we want to decrease the bias by adding sliding windows of the mutables  
    //                                                                                                                                  - or we actually care, because we are stuck in the gradient descent of bad heuristics which created bad forcast
    //                                                                                                                                  - or we dont care because the forcast will "learn" the heuristics and still inch into the right direction
    //                                                                                                                                  - I guess we dont know - except for <do your best on the forcast and the heuristics part> - and pray for the best
    //                                                                                                                                  - real time system is actually way more complicated than we think, they are already very heavily optimized - so stepping in the "forcastable beneficial" direction is actually the optimizable that we could safely make without compromising upstream optimizations
    //                                                                                                                                  - I was thinking of the things categorized as forcastable beneficial - it's back to the sliding window problem of 1 day 1 month 1 year or 3 year or 1000 year
    //                                                                                                                                                                                                       - what is the right "unit" sliding window time? I guess it depends - as long as the chart we are observing is somewhat of an avg ack_packet/s vs s
    //                                                                                                                                                                                                       - in the transactional world - we want to approximate the transaction time - and unitize the sliding window based on such
    //

    //                                                  + transformer is a radix of sparse centrality - so the model can actually think in a complex semantic space - can even predict things very accurately in the natural enviroment - not manipulated environment like stock market 
    //                                                  + I think transformer should suffice in this particular use case - we want to twist the model by using simple path search but the optimization stops there 
    //                                                  + Like every machine learning predictions, we want to have 3 things - a base algorithm (also a normalization layer of machine learning output), a machine learning algorithm, and an interval reset of the machine learning algorithm
    //

    //                                            - approxing local maximma is approxing global maxima in a uniform distribution environment, we want to do "good" individually in terms of request_acks/s, and the system will behave in the direction
    //                                            - suggested frequency does more than just suggesting frequency - it's a temporal encoder | decoder by using continuity compression (we must be Cooper writing on the watch)
    //                                            - we map our current buffer to an arbitrary trainable space (which is pretrained or real-time trained) - we convert it to a n! * n! * n! * ... space - and we broadcast it temporally - so that's the plan)
    //                                            - we are the edge of an important history anchor - we hope that humanity could stay human in the near future and understand philosophically that all things are seen relatively

    //we'll implement the first draft - this is a very complicated implementation in terms of performance + accuracy + worst case scenerios + failsafes + compromission of system + etc.
    //                                - yet a very important flood management algorithm - it observes the system holistically to inch the system in the good direction (we define good in the heuristics)

    //thru-put in terms of network is ack/s
    //thru-put in terms of compute is matrix/s
    //thru-put in terms of disk is read_byte/s
    //uniform distribution in terms of network is uniformly distributed ack/s across IP

    //what do we want to do? we want to chartify all of those statistics - sort those statistics (thru-put | unif_dist + etc) - and place them on a buffer - we can actually specify the importance of those guys by allocating more space for one than another  
    //we want to do centrality of system stats - the things that we've just chartified - and broadcast that to the closest neighbors - who would do compression of incoming data + their data and broadcast to another neighbor - it's a radix of dense centrality algorithm - we want something like floyed

    //what's our grand goal again? its to saturate + uniform_distribute the bandwidth?
    //recall that our consumption must be equal production - and we leave the flood management to the kernel (because kernel is good at this)

    //we'll talk about locality of dynamic memory allocations and how that would affect the system
    //we can't spend all RAM bus on the network because that's bad
    //it's the cyclic dynamic memory allocations + allocation_life_time + etc.  

    //we'll try to implement this this week
    //the light of consciousness is the cache L1 L2 L3, RAM
    //it has always been about "attention" - this attention is actually about temporal access of data
    //our attention is about frequency of appearance and temporal access - we'll come up with a way
    //what if we core-dump everything, take snapshots and train our neural network on the buffer?
    //well, it's a very nice topic to explore
    //we'll post our result in a month - it's complicated

    //this is extremely hard to code efficiently - so let's just accept the implementation - we'll be back later to tune kernel - we are expecting to affine 32 cores out of 1024 cores to pull all bandwidth (soon to be scaled to 64GB/s) without contaminating other tasks
    //we dont have time to argue with the non-reasonables - we have a performance constraint to meet and a bandwidth to saturate 
    //we'll rewrite part of the sys if we have to

    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller;
            std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container;
            std::shared_ptr<packet_controller::PacketContainerInterface> inbound_packet_container;
            std::shared_ptr<packet_controller::BufferContainerInterface> inbound_buffer_container;
            std::shared_ptr<packet_controller::InBoundIDControllerInterface> inbound_id_controller;
            std::shared_ptr<packet_controller::TrafficControllerInterface> inbound_traffic_controller;
            std::shared_ptr<packet_controller::FeedBackMachineInterface> feedback_machine;
            std::shared_ptr<packet_controller::AckPacketGeneratorInterface> ack_packet_gen;
            size_t ack_vectorization_sz;
            size_t inbound_consumption_sz;
            size_t rest_threshold_sz;

        public:

            InBoundWorker(std::shared_ptr<packet_controller::RetransmissionControllerInterface> retransmission_controller,
                          std::shared_ptr<packet_controller::PacketContainerInterface> outbound_packet_container,
                          std::shared_ptr<packet_controller::PacketContainerInterface> inbound_packet_container,
                          std::shared_ptr<packet_controller::BufferContainerInterface> inbound_buffer_container,
                          std::shared_ptr<packet_controller::InBoundIDControllerInterface> inbound_id_controller,
                          std::shared_ptr<packet_controller::TrafficControllerInterface> inbound_traffic_controller,
                          std::shared_ptr<packet_controller::FeedBackMachineInterface> feedback_machine,
                          std::shared_ptr<packet_controller::AckPacketGeneratorInterface> ack_packet_gen,
                          size_t ack_vectorization_sz,
                          size_t inbound_consumption_sz,
                          size_t rest_threshold_sz) noexcept: retransmission_controller(std::move(retransmission_controller)),
                                                              outbound_packet_container(std::move(outbound_packet_container)),
                                                              inbound_packet_container(std::move(inbound_packet_container)),
                                                              inbound_buffer_container(std::move(inbound_buffer_container)),
                                                              inbound_id_controller(std::move(inbound_id_controller)),
                                                              inbound_traffic_controller(std::move(inbound_traffic_controller)),
                                                              feedback_machine(std::move(feedback_machine)),
                                                              ack_packet_gen(std::move(ack_packet_gen)),
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
                    ackid_delivery_resolutor.dst                            = this->retransmission_controller.get();

                    size_t trimmed_ackid_delivery_sz                        = std::min(this->retransmission_controller->max_consume_size(), buf_arr_sz * MAX_ACK_PER_PACKET);
                    size_t ackid_deliverer_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(&ackid_delivery_resolutor, trimmed_ackid_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> ackid_deliverer_mem(ackid_deliverer_allocation_cost);
                    auto ackid_deliverer                                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ackid_delivery_resolutor, trimmed_ackid_delivery_sz, ackid_deliverer_mem.get())); 

                    //

                    auto rttfb_delivery_resolutor                           = InternalRTTFeedBackDeliveryResolutor{};
                    rttfb_delivery_resolutor.feedback_machine               = this->feedback_machine.get();

                    size_t trimmed_rttfb_delivery_sz                        = std::min(this->feedback_machine->max_consume_size(), buf_arr_sz * MAX_FEEDBACK_PER_PACKET);
                    size_t rttfb_deliverer_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(&rttfb_delivery_resolutor, trimmed_rttfb_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> rttfb_deliverer_mem(rttfb_deliverer_allocation_cost);
                    auto rttfb_deliverer                                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&rttfb_delivery_resolutor, trimmed_rttfb_delivery_sz, rttfb_deliverer_mem.get())); 

                    //

                    auto obfb_delivery_resolutor                            = InternalOutBoundFeedBackDeliveryResolutor{};
                    obfb_delivery_resolutor.feedback_machine                = this->feedback_machine.get();

                    size_t trimmed_obfb_delivery_sz                         = std::min(this->feedback_machine->max_consume_size(), buf_arr_sz * MAX_FEEDBACK_PER_PACKET);
                    size_t obfb_deliverer_allocation_cost                   = dg::network_producer_consumer::delvrsrv_allocation_cost(&obfb_delivery_resolutor, trimmed_obfb_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> obfb_deliverer_mem(obfb_deliverer_allocation_cost);
                    auto obfb_deliverer                                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&obfb_delivery_resolutor, trimmed_obfb_delivery_sz, obfb_deliverer_mem.get())); 

                    //

                    auto ibff_delivery_resolutor                            = InternalInBoundFeedFrontDeliveryResolutor{};
                    ibff_delivery_resolutor.feedback_machine                = this->feedback_machine.get();

                    size_t trimmed_ibff_delivery_sz                         = std::min(this->feedback_machine->max_consume_size(), buf_arr_sz * MAX_FEEDBACK_PER_PACKET);
                    size_t ibff_deliverer_allocation_cost                   = dg::network_producer_consumer::delvrsrv_allocation_cost(&ibff_delivery_resolutor, trimmed_ibff_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> ibff_deliverer_mem(ibff_deliverer_allocation_cost);
                    auto ibff_deliverer                                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ibff_delivery_resolutor, trimmed_ibff_delivery_sz, ibff_deliverer_mem.get()));

                    //

                    auto ibpkt_delivery_resolutor                           = InternalPacketDeliveryResolutor{};
                    ibpkt_delivery_resolutor.dst                            = this->inbound_packet_container.get();

                    size_t trimmed_ibpkt_delivery_sz                        = std::min(this->inbound_packet_container->max_consume_size(), buf_arr_sz);
                    size_t ibpkt_deliverer_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(&ibpkt_delivery_resolutor, trimmed_ibpkt_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> ibpkt_deliverer_mem(ibpkt_deliverer_allocation_cost);
                    auto ibpkt_deliverer                                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ibpkt_delivery_resolutor, trimmed_ibpkt_delivery_sz, ibpkt_deliverer_mem.get()));

                    //

                    auto obpkt_delivery_resolutor                           = InternalPacketDeliveryResolutor{};
                    obpkt_delivery_resolutor.dst                            = this->outbound_packet_container.get();

                    size_t trimmed_obpkt_delivery_sz                        = std::min(this->outbound_packet_container->max_consume_size(), buf_arr_sz);
                    size_t obpkt_deliverer_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(&obpkt_delivery_resolutor, trimmed_obpkt_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> obpkt_deliverer_mem(obpkt_deliverer_allocation_cost);
                    auto obpkt_deliverer                                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&obpkt_delivery_resolutor, trimmed_obpkt_delivery_sz, obpkt_deliverer_mem.get()));

                    //

                    auto ack_vectorizer_resolutor                           = InternalAckVectorizerResolutor{};
                    ack_vectorizer_resolutor.dst                            = &obpkt_deliverer; 
                    ack_vectorizer_resolutor.ack_packet_gen                 = this->ack_packet_gen.get();

                    size_t trimmed_ack_vectorization_sz                     = std::min(this->ack_vectorization_sz, buf_arr_sz);
                    size_t ack_vectorizer_allocation_cost                   = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&ack_vectorizer_resolutor, trimmed_ack_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> ack_vectorizer_mem(ack_vectorizer_allocation_cost);
                    auto ack_vectorizer                                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&ack_vectorizer_resolutor, trimmed_ack_vectorization_sz, ack_vectorizer_mem.get())); 

                    //

                    auto thru_rts_ack_delivery_resolutor                    = InternalThruRTSAckResolutor{};
                    thru_rts_ack_delivery_resolutor.packet_id_deliverer     = &ackid_deliverer;
                    thru_rts_ack_delivery_resolutor.rtt_fb_deliverer        = &rttfb_deliverer; 

                    size_t trimmed_thru_rts_ack_delivery_sz                 = std::min(DEFAULT_ACCUMULATION_SZ, buf_arr_sz);
                    size_t thru_rts_ack_allocation_cost                     = dg::network_producer_consumer::delvrsrv_allocation_cost(&thru_rts_ack_delivery_resolutor, trimmed_thru_rts_ack_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> thru_rts_ack_mem(thru_rts_ack_allocation_cost);
                    auto thru_rts_ack_deliverer                             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&thru_rts_ack_delivery_resolutor, trimmed_thru_rts_ack_delivery_sz, thru_rts_ack_mem.get())); 

                    //

                    auto thru_request_delivery_resolutor                    = InternalThruRequestResolutor{};
                    thru_request_delivery_resolutor.ack_vectorizer          = &ack_vectorizer;
                    thru_request_delivery_resolutor.inbound_deliverer       = &ibpkt_deliverer;

                    size_t trimmed_thru_request_delivery_sz                 = std::min(DEFAULT_ACCUMULATION_SZ, buf_arr_sz);
                    size_t thru_request_allocation_cost                     = dg::network_producer_consumer::delvrsrv_allocation_cost(&thru_request_delivery_resolutor, trimmed_thru_request_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> thru_request_mem(thru_request_allocation_cost);
                    auto thru_request_deliverer                             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&thru_request_delivery_resolutor, trimmed_thru_request_delivery_sz, thru_request_mem.get())); 

                    //

                    auto thru_suggest_delivery_resolutor                    = InternalThruSuggestResolutor{};
                    thru_suggest_delivery_resolutor.feedback_deliverer      = &obfb_deliverer;

                    size_t trimmed_thru_suggest_delivery_sz                 = std::min(DEFAULT_ACCUMULATION_SZ, buf_arr_sz);
                    size_t thru_suggest_allocation_cost                     = dg::network_producer_consumer::delvrsrv_allocation_cost(&thru_suggest_delivery_resolutor, trimmed_thru_suggest_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> thru_suggest_mem(thru_suggest_allocation_cost);
                    auto thru_suggest_deliverer                             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&thru_suggest_delivery_resolutor, trimmed_thru_suggest_delivery_sz, thru_suggest_mem.get()));

                    //

                    auto thru_informr_delivery_resolutor                    = InternalThruInformrResolutor{};
                    thru_informr_delivery_resolutor.feedfront_deliverer     = &ibff_deliverer;

                    size_t trimmed_thru_informr_delivery_sz                 = std::min(DEFAULT_ACCUMULATION_SZ, buf_arr_sz);
                    size_t thru_informr_allocation_cost                     = dg::network_producer_consumer::delvrsrv_allocation_cost(&thru_informr_delivery_resolutor, trimmed_thru_informr_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> thru_informr_mem(thru_informr_allocation_cost);
                    auto thru_informr_deliverer                             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&thru_informr_delivery_resolutor, trimmed_thru_informr_delivery_sz, thru_informr_mem.get())); 

                    //

                    auto thru_delivery_resolutor                            = InternalThruResolutor{};
                    thru_delivery_resolutor.rts_ack_thru_deliverer          = &thru_rts_ack_deliverer;
                    thru_delivery_resolutor.request_thru_deliverer          = &thru_request_deliverer;
                    thru_delivery_resolutor.suggest_thru_deliverer          = &thru_suggest_deliverer;
                    thru_delivery_resolutor.informr_thru_deliverer          = &thru_informr_deliverer;
                    thru_delivery_resolutor.krescue_thru_deliverer          = &thru_krescue_deliverer;

                    size_t trimmed_thru_delivery_sz                         = std::min(DEFAULT_ACCUMULATION_SZ, buf_arr_sz);
                    size_t thru_delivery_allocation_cost                    = dg::network_producer_consumer::delvrsrv_allocation_cost(&thru_delivery_resolutor, trimmed_thru_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> thru_delivery_mem(thru_delivery_allocation_cost);
                    auto thru_deliverer                                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&thru_delivery_resolutor, trimmed_thru_delivery_sz, thru_delivery_mem.get())); 

                    //

                    auto nothru_ack_delivery_resolutor                      = InternalNoThruAckResolutor{};
                    nothru_ack_delivery_resolutor.ack_vectorizer            = &ack_vectorizer;

                    size_t trimmed_nothru_ack_delivery_sz                   = std::min(DEFAULT_ACCUMULATION_SZ, buf_arr_sz);
                    size_t nothru_ack_allocation_cost                       = dg::network_producer_consumer::delvrsrv_allocation_cost(&nothru_ack_delivery_resolutor, trimmed_nothru_ack_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> nothru_ack_delivery_mem(nothru_ack_allocation_cost);
                    auto nothru_ack_deliverer                               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&nothru_ack_delivery_resolutor, trimmed_nothru_ack_delivery_sz, nothru_ack_delivery_mem.get())); 

                    //

                    auto inbound_delivery_resolutor                         = InternalInBoundIDResolutor{};
                    inbound_delivery_resolutor.downstream_dst               = &thru_deliverer;
                    inbound_delivery_resolutor.nothru_ack_dst               = &nothru_ack_deliverer;

                    size_t trimmed_inbound_delivery_sz                      = std::min(DEFAULT_ACCUMULATION_SZ, buf_arr_sz);
                    size_t inbound_allocation_cost                          = dg::network_producer_consumer::delvsrv_allocation_cost(&inbound_delivery_resolutor, trimmed_inbound_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> inbound_mem(inbound_allocation_cost);
                    auto inbound_deliverer                                  = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&inbound_delivery_resolutor, tirmmed_inbound_delivery_sz, inbound_mem.get())); 

                    //

                    auto traffic_resolutor                                  = InternalTrafficResolutor{};
                    traffic_resolutor.downstream_dst                        = &inbound_deliverer;

                    size_t trimmed_traffic_resolutor_delivery_sz            = std::min(this->inbound_traffic_controller->max_consume_size(), buf_arr_sz);
                    size_t traffic_resolutor_allocation_cost                = dg::network_producer_consumer::delvrsrv_allocation_cost(&traffic_resolutor, trimmed_traffic_resolutor_delivery_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> traffic_resolutor_mem(traffic_resolutor_allocation_cost);
                    auto traffic_resolutor_deliverer                        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&traffic_resolutor, trimmed_traffic_resolutor_delivery_sz, traffic_resolutor_mem.get())); 

                    for (size_t i = 0u; i < buf_arr_sz; ++i){
                        std::expected<Packet, exception_t> pkt = utility::deserialize_packet(std::move(buf_arr[i]));

                        if (!pkt.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(pkt.error()));
                            continue;
                        }

                        exception_t stamp_err = packet_service::port_stamp(pkt.value());

                        if (dg::network_exception::is_failed(stamp_err)){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(stamp_err));
                            continue;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(traffic_resolutor_deliverer.get(), std::move(pkt.value()));
                    }
                }

                return success_counter > this->rest_threshold_sz;
            }

        private:

            //alright - branch prediction pipeline is cured - we will try to minimize the memory footprint by limiting the delivery -> 256 capacity - that's about all that we could do
            //we have more to think than just "that is not fast enough" - it is not fast because the structure is not trivially_copyable_v - which is an optimizable
            //we have to inch the optimization in the direction that is good in terms of branch prediction + memory footprint + memory synchronization (memory orderings) - and we leave the open road for future optimizations
            //we figured that there is no better way than to keep packet size at 4096-8192 bytes - because the kernel does not fragment the transmission at this size - which would heavily determine the thruput
            //recv_block + 8KB packet size are not optional

            struct InternalRetransmissionAckDeliveryResolutor: dg::network_producer_consumer::ConsumerInterface<global_packet_id_t>{

                dg::packet_controller::RetransmissionControllerInterface * dst;

                void push(std::move_iterator<global_packet_id_t> * data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    global_packet_id_t * base_data_arr = data_arr.base();
                    this->dst->ack(base_data_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

            struct InternalRTTFeedBackDeliveryResolutor: dg::network_producer_consumer::ConsumerInterface<RTTFeedBack>{

                dg::packet_controller::FeedBackMachineInterface * feedback_machine;

                void push(std::move_iterator<RTTFeedBack> * data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    RTTFeedBack * base_data_arr = data_arr.base();
                    this->feedback_machine->rtt_fb(base_data_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

            struct InternalOutBoundFeedBackDeliveryResolutor: dg::network_producer_consumer::ConsumerInterface<OutBoundFeedBack>{

                dg::packet_controller::FeedBackMachineInterface * feedback_machine;

                void push(std::move_iterator<OutBoundFeedBack> * data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    OutBoundFeedBack * base_data_arr = data_arr.base();
                    this->feedback_machine->outbound_frequency_fb(base_data_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

            struct InternalInBoundFeedFrontDeliveryResolutor: dg::network_producer_consumer::ConsumerInterface<InBoundFeedFront>{

                dg::packet_controller::FeedBackMachineInterface * feedback_machine;

                void push(std::move_iterator<InBoundFeedFront> * data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    InBoundFeedFront * base_data_arr = data_arr.base();
                    this->feedback_machine->inbound_frequency_ff(base_data_arr, sz, exception_arr.get());

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

                    std::expected<Packet, exception_t> virtualized_pkt = packet_service::virtualize_rts_ack_packet(std::move(ack_pkt.value()));

                    if (!virtualized_pkt.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(virtualized_pkt.error()));
                        return;
                    }

                    //99% branch flows here - no branch overheads
                    dg::network_producer_consumer::delvrsrv_deliver(this->dst, std::move(virtualized_pkt.value()));
                }
            };

            struct InternalTrafficResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::DeliveryHandle<Packet> * downstream_dst;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<Address[]> addr_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> response_arr(sz);

                    Packet * base_data_arr = data_arr.base();
                    std::transform(base_data_arr, std::next(base_data_arr, sz), addr_arr.get(), [](const Packet& packet){return packet.fr_addr;});
                    this->self->inbound_traffic_controller->thru(addr_arr.get(), sz, response_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (!response_arr[i].has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(response_arr[i].error()));
                            continue;
                        }

                        if (!response_arr[i].value()){
                            dg::network_log_stackdump::error_fast_optional("Incoming packet dropped at traffic controller");
                            continue;
                        }

                        //99% branch flows here - no branch overheads
                        dg::network_producer_consumer::delvrsrv_deliver(this->downstream_dst, std::move(base_data_arr[i]));
                    }
                }
            };

            struct InternalInBoundIDResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::DeliveryHandle<Packet> * downstream_dst;
                dg::network_producer_consumer::DeliveryHandle<Packet> * nothru_ack_dst;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<global_packet_id_t[]> id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> response_arr(sz);

                    Packet * base_data_arr = data_arr.base();
                    std::transform(base_data_arr, std::next(base_data_arr, sz), id_arr.get(), [](const Packet& packet){return packet.id;});
                    this->self->inbound_id_controller->thru(id_arr.get(), sz, response_arr.get());

                    using radix_t   = dg::network_producer_consumer::DeliveryHandle<Packet> *;
                    radix_t radix_table[2];
                    radix_table[0]  = this->nothru_ack_dst;
                    radix_table[1]  = this->downstream_dst;

                    for (size_t i = 0u; i < sz; ++i){
                        if (!response_arr[i].has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(response_arr[i].error()));
                            continue;
                        }

                        //this is expensive - there is an optimizable that breaks code readability - if the is_request_packet is likely - we want to abstract the PacketBase and Packet
                        //and do another radix dispatch - we will talk about this later 

                        if (!response_arr[i].value() && packet_service::is_request_packet(base_data_arr[i]) || response_arr[i].has_value()){
                            //99% of branch flows here
                            dg::network_producer_consumer::delvrsrv_deliver(radix_table[static_cast<int>(response_arr[i].has_value())], std::move(base_data_arr[i]));
                        }
                    }
                }
            };

            struct InternalNoThruAckResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::KVDeliveryHandle<Address, PacketBase> * ack_vectorizer;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    Packet * base_data_arr = data_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_producer_consumer::delvrsrv_deliver(this->ack_vectorizer, base_data_arr[i].fr, static_cast<const PacketBase&>(base_data_arr[i]));
                    }
                }
            };

            struct InternalThruResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{
                
                dg::network_producer_consumer::DeliveryHandle<Packet> * rts_ack_thru_deliverer;
                dg::network_producer_consumer::DeliveryHandle<Packet> * request_thru_deliverer;
                dg::network_producer_consumer::DeliveryHandle<Packet> * suggest_thru_deliverer;
                dg::network_producer_consumer::DeliveryHandle<Packet> * informr_thru_deliverer;
                dg::network_producer_consumer::DeliveryHandle<Packet> * krescue_thru_deliverer;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    Packet * base_data_arr = data_arr.base(); 

                    for (size_t i = 0u; i < sz; ++i){                        
                        //no branches - optimized into a radix dispatch

                        if (packet_service::is_rts_ack_packet(base_data_arr[i])){
                            dg::network_producer_consumer::delvrsrv_deliver(this->rts_ack_thru_deliverer, std::move(base_data_arr[i]));
                        } else if (packet_service::is_request_packet(base_data_arr[i])){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_thru_deliverer, std::move(base_data_arr[i]));
                        } else if (packet_service::is_suggest_packet(base_data_arr[i])){
                            dg::network_producer_consumer::delvrsrv_deliver(this->suggest_thru_deliverer, std::move(base_data_arr[i]));
                        } else if (packet_serivce::is_informr_packet(base_data_arr[i])){
                            dg::network_producer_consumer::delvrsrv_deliver(this->informr_thru_deliverer, std::move(base_data_arr[i]));
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

            struct InternalThruRTSAckResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::DeliveryHandle<global_packet_id_t> * packet_id_deliverer;
                dg::network_producer_consumer::DeliveryHandle<RTTFeedBack> * rtt_fb_deliverer;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    Packet * base_data_arr = data_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        std::expected<AckPacket, exception_t> ack_pkt = packet_service::devirtualize_rts_ack_packet(std::move(base_data_arr[i]));  

                        if (!ack_pkt.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(ack_pkt.error()));
                            continue;
                        }

                        for (const PacketBase& e: ack_pkt->ack_vec){
                            exception_t port_stamp_err = packet_service::port_stamp(e);

                            if (dg::network_exception::is_failed(port_stamp_err)){
                                dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(port_stamp_err));
                            }

                            std::expected<std::chrono::nanoseconds, exception_t> lapsed = packet_service::get_transit_time(e);

                            RTTFeedBack fb  = {};
                            fb.addr         = ack_pkt->fr;

                            if (!lapsed.has_value()){
                                fb.rtt = std::nullopt;
                                dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(lapsed.error()));
                            } else{
                                fb.rtt = lapsed.value();
                            }

                            dg::network_producer_consumer::delvrsrv_deliver(this->rtt_fb_deliverer, std::move(fb));
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

            struct InternalThruSuggestResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::DeliveryHandle<OutBoundFeedBack> * feedback_deliverer;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    Packet * base_data_arr = data_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        std::expected<SuggestPacket, exception_t> pkt = packet_service::devirtualize_suggest_packet(std::move(base_data_arr[i]));

                        if (!pkt.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(pkt.error()));
                            continue;
                        }

                        OutBoundFeedBack fb     = {};
                        fb.addr                 = pkt->fr;
                        fb.suggested_frequency  = pkt->suggested_frequency;

                        dg::network_producer_consumer::delvrsrv_deliver(this->feedback_deliverer, std::move(fb));
                    }
                }
            };

            struct InternalThruInformrResolutor: dg::network_producer_consumer::ConsumerInterface<Packet>{

                dg::network_producer_consumer::DeliveryHandle<InBoundFeedFront> * feedfront_deliverer;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    Packet * base_data_arr = data_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        std::expected<InformrPacket, exception_t> pkt = packet_service::devirtualize_informr_packet(std::move(base_data_arr[i]));

                        if (!pkt.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(pkt.error()));
                            continue;
                        }

                        InBoundFeedFront ff     = {};
                        ff.addr                 = pkt->fr;
                        ff.transmit_frequency   = pkt->transmit_frequency;

                        dg::network_producer_consumer::delvrsrv_deliver(this->feedfront_deliverer, std::move(ff));
                    }
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
                                         std::shared_ptr<packet_controller::TrafficControllerInterface> ib_traffic_controller,
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

                //

                auto internal_deliverer                         = InternalIBDeliverer{};
                internal_deliverer.ib_packet_container          = this->ib_packet_container.get();
                internal_deliverer.retransmission_controller    = this->retransmission_controller.get();

                size_t trimmed_ib_delivery_sz                   = std::min(std::min(this->ib_packet_container->max_consume_size(), this->retransmission_controller->max_consume_size()), sz);
                size_t ib_deliverer_allocation_cost             = dg::network_producer_consumer::delvrsrv_allocation_cost(&internal_deliverer, trimmed_ib_delivery_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> ib_deliverer_mem(ib_deliverer_allocation_cost);
                auto ib_deliverer                               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&internal_deliverer, trimmed_ib_delivery_sz, ib_deliverer_mem.get()));

                //

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<Packet, exception_t> pkt = this->packet_gen->get(std::move(base_data_arr[i]));

                    if (!pkt.has_value()){
                        exception_arr[i] = pkt.error();
                        continue;
                    }

                    //we dont want to further complicate the code so everything that is through here is considered "SUCCESS"

                    exception_arr[i] = dg::network_exception::SUCCESS;
                    dg::network_producer_consumer::delvrsrv_deliver(ib_deliverer.get(), std::move(pkt.value()));
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

            struct InternalIBDeliverer: dg::network_producer_consumer::ConsumerInterface<Packet>{

                packet_controller::PacketContainerInterface * ib_packet_container;
                packet_controller::RetransmissionControllerInterface * retransmission_controller;

                void push(std::move_iterator<Packet *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<Packet[]> cpy_data_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    Packet * base_data_arr = data_arr.base();
                    std::copy(base_data_arr, std::next(base_data_arr, sz), cpy_data_arr.get());

                    this->ib_packet_container->push(std::make_move_iterator(base_data_arr), sz, exception_arr);

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