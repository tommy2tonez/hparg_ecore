#ifndef __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__
#define __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__

//define HEADER_CONTROL 10

#include "network_kernel_mailbox_impl1.h"
#include "network_trivial_serializer.h"
#include "network_concurrency.h"
#include "network_std_container.h"
#include <chrono>
#include "network_log.h"
#include "network_concurrency_x.h"
#include "stdx.h"
#include "network_exception_handler.h"

namespace dg::network_kernel_mailbox_impl1_meterlogx{

    using Address = dg::network_kernel_mailbox_impl1::model::Address; 

    struct MeterInterface{
        virtual ~MeterInterface() noexcept = default;
        virtual void tick(size_t) noexcept = 0;
        virtual auto get() noexcept -> std::pair<size_t, std::chrono::nanoseconds> = 0;
    };

    class MtxMeter: public virtual MeterInterface{
        
        private:

            size_t count; 
            std::chrono::nanoseconds then;
            std::unique_ptr<std::mutex> mtx;

        public:

            MtxMeter(size_t coumt,
                     std::chrono::nanoseconds then, 
                     std::unique_ptr<std::mutex> mtx) noexcept: count(count),
                                                                then(then),
                                                                mtx(std::move(mtx)){}
            
            void tick(size_t incoming_sz) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->count += incoming_sz;
            }

            auto get() noexcept -> std::pair<size_t, std::chrono::nanoseconds>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto now        = static_cast<std::chrono::nanoseconds>(stdx::unix_timestamp());
                auto lapsed     = now - this->then;
                auto rs         = std::make_pair(this->count, lapsed);
                this->count     = 0u;
                this->then      = now;

                return rs;
            }
    }; 

    class MeterLogWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            dg::string device_id;
            std::shared_ptr<MeterInterface> send_meter;
            std::shared_ptr<MeterInterface> recv_meter;
        
        public:

            MeterLogWorker(dg::string device_id,
                           std::shared_ptr<MeterInterface> send_meter, 
                           std::shared_ptr<MeterInterface> recv_meter) noexcept: device_id(std::move(device_id)),
                                                                                 send_meter(std::move(send_meter)),
                                                                                 recv_meter(std::move(recv_meter)){}
            
            bool run_one_epoch() noexcept{

                auto [send_bsz, send_dur]   = this->send_meter->get();
                auto [recv_bsz, recv_dur]   = this->recv_meter->get();
                auto send_msg               = this->make_send_meter_msg(send_bsz, send_dur);
                auto recv_msg               = this->make_recv_meter_msg(recv_bsz, recv_dur);

                dg::network_log::journal_fast(send_msg.c_str());
                dg::network_log::journal_fast(recv_msg.c_str());

                return true;
            }
        
        private:

            auto make_send_meter_msg(size_t bsz, std::chrono::nanoseconds dur) noexcept -> dg::string{

                std::chrono::seconds dur_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(dur);
                size_t tick_sz = dur_in_seconds.count();

                if (tick_sz == 0u){
                    return std::format("[METER_REPORT] low meter precision resolution (device_id: {}, part: send_meter)", this->device_id);
                } 

                size_t bsz_per_s = bsz / tick_sz;
                return std::format("[METER_REPORT] {} bytes/s sent to {}", bsz_per_s, this->device_id);
            }

            auto make_recv_meter_msg(size_t bsz, std::chrono::nanoseconds dur) noexcept -> dg::string{

                std::chrono::seconds dur_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(dur);
                size_t tick_sz = dur_in_seconds.count();

                if (tick_sz == 0u){
                    return std::format("[METER_REPORT] low meter precision resolution (device_id: {}, part: recv_meter)", this->device_id);
                }

                size_t bsz_per_s = bsz / tick_sz;
                return std::format("[METER_REPORT] {} bytes/s recv from {}", bsz_per_s, this->device_id);
            }
    };

    class MeteredMailBox: public virtual dg::network_kernel_mailbox_impl1::core::MailboxInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
            std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox;
            std::shared_ptr<MeterInterface> send_meter;
            std::shared_ptr<MeterInterface> recv_meter;
        
        public:

            MeteredMailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemons, 
                           std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox,
                           std::shared_ptr<MeterInterface> send_meter,
                           std::shared_ptr<MeterInterface> recv_meter): daemons(std::move(daemons)),
                                                                        mailbox(std::move(mailbox)),
                                                                        send_meter(std::move(send_meter)),
                                                                        recv_meter(std::move(recv_meter)){}
            
            void send(Address addr, dg::string buf) noexcept{

                this->send_meter->tick(buf.size());
                this->mailbox->send(std::move(addr), std::move(buf));
            }

            auto recv() noexcept -> std::optional<dg::string>{

                std::optional<dg::string> rs = this->mailbox->recv(); 

                if (!rs.has_value()){
                    return std::nullopt;
                }

                this->recv_meter->tick(rs->size());
                return rs;
            }
    };
}

namespace dg::network_kernel_mailbox_impl1_streamx{

    //alright, we are to work on improving the streamx
    //we dont have time to do transmission-controlled protocol, we assume people are well-behaved, planned people, people that know what to do with their life at a random time

    //after a long conversation with my std-friends, they are heavily against the heap_allocation noexcept, even if it is for application purposes
    //I dont really know what precisely their plan is for defrag or extending the heap
    //yet it is said that we can decrease the chance of no-fail allocations -> 0 by controlling the upstream allocation techniques, not here returning dg::string like it is implementation-specific

    //we have come up with two types of std_containers, the std_container that cannot fail because of allocations, and the std_container that can fail because of allocations
    //std_container that cannot fail because of allocations are for preallocated, internal usage of std::unordered_set<> std::unordered_map, std::deque, etc.
    //std_container that can fail because of allocations (we hardly found the reason to fail allocations, yet it's best practice to consider allocations as normal action operations)
    //we are to radix the std_containers

    //alright, let's see what's going on
    //there might be a chance of coordinated streamx attack
    //this is also susceptible to late packets transmissions
    //we want to establish soft_synchronization of packet destruction to avoid statistical chances or engineered chances of incoming packet recv times being evenly spreaded out in an acceptable adjecent destruction window
    //such creates the worst case of n * <waiting_window> -> 60s - 1 hour (which is very bad)
    //we must have an absolute window of transmission right at the moment of <packet_assemble> construction
    //such window is fixed and reasonably fixed, configurable by users (1MB - 16MB, 10 seconds transmisison, for example)
    //we are to not yet worry about the affinity of random_hash_distributed
    //if the lock contention cost (queueing mutex + put thread to sleep + wake thread up == 1024 packets submission, we are to increase the packet size per tx -> 1024, because the affinity problem introduces hand-tune accurate, skewed consumption, we are to avoid that unless we are going to the last of last option for micro optimizations)

    using Address = dg::network_kernel_mailbox_impl1::model::Address; 

    static inline constexpr size_t MAX_STREAM_SIZE                          = size_t{1} << 25;
    static inline constexpr size_t MAX_SEGMENT_SIZE                         = size_t{1} << 10;
    static inline constexpr size_t DEFAULT_KEYVALUE_FEED_SIZE               = size_t{1} << 10; 
    static inline constexpr size_t DEFAULT_KEY_FEED_SIZE                    = size_t{1} << 8;
    static inline constexpr uint32_t PACKET_SEGMENT_SERIALIZATION_SECRET    = 30011; 
    
    struct GlobalIdentifier{
        Address addr;
        uint64_t local_id;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(addr, local_id);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(addr, local_id);
        }
    };

    struct PacketSegment{
        dg::string buf;
        GlobalIdentifier id;
        uint64_t segment_idx;
        uint64_t segment_sz;
    };

    static auto serialize_packet_segment(PacketSegment&& segment) noexcept -> std::expected<dg::string, exception_t>{

        using header_t      = std::tuple<GlobalIdentifier, uint64_t, uint64_t>;
        size_t header_sz    = dg::network_compact_serializer::integrity_size(header_t{});
        size_t old_sz       = segment.buf.size();
        size_t new_sz       = old_sz + header_sz;
        auto header         = header_t{segment.id, segment.idx, segment.sz};

        try{
            segment.buf.resize(new_sz);
            dg::network_compact_serializer::integrity_serialize_into(std::next(segment.buf.data(), old_sz), header, PACKET_SEGMENT_SERIALIZATION_SECRET);
            return std::expected<dg::string, exception_t>(std::move(segment.buf));
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }
    }

    static auto deserialize_packet_segment(dg::string&& buf) noexcept -> std::expected<PacketSegment, exception_t>{

        using header_t      = std::tuple<GlobalIdentifier, uint64_t, uint64_t>;
        size_t header_sz    = dg::network_compact_serializer::integrity_size(header_t{});
        size_t buf_sz       = buf.size();
        auto header         = header_t{}; 

        if (buf_sz < header_sz){
            return std::unexpected(dg::network_exception::SOCKET_STREAM_BAD_SEGMENT);           
        }

        size_t hdr_off      = buf.size() - header_sz;
        char * hdr_buf      = std::next(buf.data(), hdr_off);  

        exception_t err     = dg::network_exception::to_cstyle_function<dg::network_compact_serializer::integrity_deserialize_into<header_t>>(header, hdr_buf, PACKET_SEGMENT_SERIALIZATION_SECRET);

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }
        
        buf.resize(hdr_off);
        PacketSegment rs = {};
        rs.buf = std::move(buf);
        std::tie(rs.id, rs.segment_idx, rs.segment_sz) = header;

        return rs;
    }

    struct AssembledPacket{
        dg::vector<PacketSegment> data;
        size_t collected_segment_sz;
        size_t total_segment_sz;
    };

    static auto assembled_packet_to_buffer(AssembledPacket&& pkt) noexcept -> std::expected<dg::string, exception_t>{

        if (pkt.total_segment_sz != pkt.collected_segment_sz){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (pkt.total_segment_sz != pkt.data.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (pkt.total_segment_sz == 0u){
            return dg::string();
        }

        if (pkt.total_segment_sz == 1u){
            return std::expected<dg::string, exception_t>(std::move(pkt.data.front().buf));
        }

        size_t total_bsz = 0u;

        for (size_t i = 0u; i < pkt.total_segment_sz; ++i){
            total_bsz += pkt.data[i].buf.size();
        }

        std::expected<dg::string, exception_t> rs = dg::network_exception::cstyle_initialize<dg::string>(total_bsz, ' ');

        if (!rs.has_value()){
            return rs;
        }

        char * out_it = rs.value().data(); 

        for (size_t i = 0u; i < pkt.total_segment_sz; ++i){
            out_it = std::copy(pkt.data[i].buf.begin(), pkt.data[i].buf.end(), out_it);
        }

        return rs;
    } 

    struct PacketIDGeneratorInterface{
        virtual ~PacketIDGeneratorInterface() noexcept = default;
        virtual auto get_id() noexcept -> GlobalIdentifier = 0;
    };

    struct PacketizerInterface{
        virtual ~PacketizerInterface() noexcept = default;
        virtual auto packetize(dg::string&&) noexcept -> std::expected<dg::vector<PacketSegment>, exception_t> = 0;
    };

    struct EntranceControllerInterface{
        virtual ~EntranceControllerInterface() noexcept = default;
        virtual void tick(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept = 0; 
        virtual void get_expired_id(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    struct PacketAssemblerInterface{
        virtual ~PacketAssemblerInterface() noexcept = default;
        virtual void assemble(std::move_iterator<PacketSegment *> segment_arr, size_t sz, std::expected<AssembledPacket, exception_t> * assembled_arr) noexcept = 0;
        virtual void destroy(GlobalIdentifier * id_arr, size_t sz) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    struct InBoundContainerInterface{
        virtual ~InBoundContainerInterface() noexcept = default;
        virtual void push(std::move_iterator<dg::string *> arr, size_t sz, exception_t * exception_arr) noexcept = 0;
        virtual void pop(dg::string * output_arr, size_t& sz, size_t cap) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class PacketIDGenerator: public virtual PacketIDGeneratorInterface{

        private:

            Address factory_addr;
            std::atomic<uint64_t> incrementing_id;

        public:

            PacketIDGenerator(Address factory_addr,
                              uint64_t incrementing_id) noexcept: factory_addr(std::move(factory_addr)),
                                                                  incrementing_id(incrementing_id){}

            auto get_id() noexcept -> GlobalIdentifier{

                GlobalIdentifier id = {};
                id.addr             = this->factory_addr;
                id.local_id         = this->incrementing_id.fetch_add(1u, std::memory_order_relaxed);

                return id;
            }
    };

    class Packetizer: public virtual PacketizerInterface{

        private:

            std::unique_ptr<PacketIDGeneratorInterface> packet_id_gen;
            const size_t segment_byte_sz;

        public:

            Packetizer(std::unique_ptr<PacketIDGeneratorInterface> packet_id_gen,
                       size_t segment_byte_sz) noexcept: packet_id_gen(std::move(packet_id_gen)),
                                                         segment_byte_sz(segment_byte_sz){}

            auto packetize(dg::string&& buf) noexcept -> std::expected<dg::vector<PacketSegment>, exception_t>{

                if (buf.size() > MAX_STREAM_SIZE){
                    return std::unexpected(dg::network_exception::SOCKET_BAD_BUFFER_LENGTH);
                }

                size_t segment_even_sz  = buf.size() / this->segment_byte_sz;
                size_t segment_odd_sz   = size_t{buf.size() % this->segment_byte_sz != 0u};
                size_t segment_sz       = segment_even_sz + segment_odd_sz;

                if (segment_sz > MAX_SEGMENT_SIZE){
                    return std::unexpected(dg::network_exception::SOCKET_BAD_BUFFER_LENGTH);
                }

                std::expected<dg::vector<PacketSegment>, exception_t> rs = dg::network_exception::cstyle_initialize<dg::vector<PacketSegment>>(std::max(size_t{1}, segment_sz));

                if (!rs.has_value()){
                    return std::unexpected(rs.error());
                }

                if (segment_sz == 0u){
                    PacketSegment segment   = {};
                    segment.buf             = {};
                    segment.id              = this->packet_id_gen->get_id();
                    segment.segment_idx     = 0u; //this is logically incorrect, we are not protected by range bro anymore
                    segment.segment_sz      = 0u;
                    rs.value()[0]           = std::move(segment);

                    return rs;
                }

                if (segment_sz == 1u){ //premature very useful optimization, effectively skip 1 buffer iteration
                    PacketSegment segment   = {};
                    segment.buf             = std::move(buf);
                    segment.id              = this->packet_id_gen->get_id();
                    segment.segment_idx     = 0u;
                    segment.segment_sz      = 1u;
                    rs.value()[0]           = std::move(segment);

                    return rs;
                }

                for (size_t i = 0u; i < segment_sz; ++i){
                    size_t first                                    = this->segment_byte_sz * i;
                    size_t last                                     = std::min(buf.size(), this->segment_byte_sz * (i + 1)); 
                    PacketSegment segment                           = {};
                    segment.id                                      = this->packet_id_gen->get();
                    segment.segment_idx                             = i;
                    segment.segment_sz                              = segment_sz; 
                    std::expected<dg::string, exception_t> app_buf  = dg::network_exception::cstyle_initialize<dg::string>((last - first), ' '); 

                    if (!app_buf.has_value()){
                        return std::unexpected(app_buf.error());
                    }

                    std::copy(std::next(buf.begin(), first), std::next(buf.begin(), last), app_buf.value().begin());
                    rs.value()[i] = std::move(app_buf.value());
                }

                return rs;
            }
    };

    struct EntranceEntry{
        std::chrono::time_point<std::chrono::utc_clock, std::chrono::nanoseconds> timestamp;
        GlobalIdentifier key;
        __uint128_t entry_id;
    };

    class EntranceController: public virtual EntranceControllerInterface{

        private:

            dg::vector<EntranceEntry> entrance_entry_pq; //no exhausted container
            const size_t entrance_entry_pq_cap; 
            dg::unordered_map<GlobalIdentifier, size_t> key_id_map; //no exhausted container
            const size_t key_id_map_cap;
            __uint128_t id_ticker;
            const std::chrono::nanoseconds expiry_period;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> tick_sz_per_load;

        public:

            EntranceController(dg::vector<EntranceEntry> entrance_entry_pq,
                               size_t entrance_entry_pq_cap,
                               dg::unordered_map<GlobalIdentifier, size_t> key_id_map,
                               szie_t key_id_map_cap,
                               __uint128_t id_ticker,
                               std::chrono::nanoseconds expiry_period,
                               std::unique_ptr<std::mutex> mtx,
                               stdx::hdi_container<size_t> tick_sz_per_load) noexcept: entrance_entry_pq(std::move(entrance_entry_pq)),
                                                                                       entrance_entry_pq_cap(entrance_entry_pq_cap),
                                                                                       key_id_map(std::move(key_id_map)),
                                                                                       key_id_map_cap(key_id_map_cap),
                                                                                       id_ticker(id_ticker),
                                                                                       expiry_period(std::move(expiry_period)),
                                                                                       mtx(std::move(mtx)),
                                                                                       tick_sz_per_load(std::move(tick_sz_per_load)){}

            void tick(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION)); //exceeding clock + queue accuracy
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto now        = std::chrono::utc_clock::now();
                auto greater    = [](const EntranceEntry& lhs, const EntranceEntry& rhs){return lhs.timestamp > rhs.timestamp || (lhs.timestamp == rhs.timestamp && lhs.entry_id > rhs.entry_id);};

                for (size_t i = 0u; i < sz; ++i){
                    if (this->entrance_entry_pq.size() == this->entrace_entry_pq_cap){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    if (this->key_id_map.size() == this->key_id_map_cap){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    auto entry              = EntranceEntry{};
                    entry.timestamp         = now;
                    entry.key               = global_id_arr[i];
                    entry.entry_id          = this->id_ticker++;

                    this->entrance_entry_pq.push_back(entry);
                    std::push_heap(this->entrance_entry_pq.begin(), this->entrance_entry_pq.end(), greater);
                    this->key_id_map[global_id_arr[i]] = entry.entry_id;
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            //assume finite sorted queue of entrance_entry_pq
            //assume key_id_map points to the lastest GlobalIdentifier guy in the entrance_entry_pq if there exists an equal GlobalIdentifier in the queue

            auto get_expired_id(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto bar_time   = std::chrono::utc_clock::now() - this->expiry_period;
                sz              = 0u;
                auto greater    = [](const EntranceEntry& lhs, const EntranceEntry& rhs){return lhs.timestamp > rhs.timestamp || (lhs.timestamp == rhs.timestamp && lhs.entry_id > rhs.entry_id);};

                for (size_t i = 0u; i < cap; ++i){
                    if (this->entrace_entry_pq.empty()){
                        return;
                    }

                    if (this->entrance_entry_pq.front().timestamp > bar_time){
                        return; 
                    }

                    std::pop_heap(this->entrace_entry_pq.begin(), this->entrace_entry_pq.end(), greater);
                    EntranceEntry entry = std::move(this->entrace_entry_pq.back());
                    this->entrace_entry_pq.pop_back();
                    auto map_ptr        = this->key_id_map.find(entry.key); 

                    if constexpr(DEBUG_MODE_FLAG){
                        if (map_ptr == this->key_id_map.end()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    if (map_ptr->second == entry.entry_id){
                        output_arr[sz++] = entry.key;
                        this->key_id_map.erase(map_ptr);
                    }
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->tick_sz_per_load.value();
            }
    };

    class RandomHashDistributedEntranceController: public virtual EntranceControllerInterface{

        private:

            std::unique_ptr<std::unique_ptr<EntranceControllerInterface>[]> base_arr;
            const size_t keyvalue_feed_cap;
            const size_t pow2_base_arr_sz;
            const size_t zero_get_expired_id_bounce_sz;
            const size_t max_tick_per_load; 

        public:

            RandomHashDistributedEntranceController(std::unique_ptr<std::unique_ptr<EntranceControllerInterface>[]> base_arr,
                                                    size_t keyvalue_feed_cap,
                                                    size_t pow2_base_arr_sz,
                                                    size_t zero_get_expired_id_bounce_sz,
                                                    size_t max_tick_per_load) noexcept: base_arr(std::move(base_arr)),
                                                                                        keyvalue_feed_cap(keyvalue_feed_cap),
                                                                                        pow2_base_arr_sz(pow2_base_arr_sz),
                                                                                        zero_get_expired_id_bounce_sz(zero_get_expired_id_bounce_sz),
                                                                                        max_tick_per_load(max_tick_per_load){}

            void tick(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{
                
                auto feed_resolutor                 = InternalFeedResolutor{};
                feed_resolutor.dst                  = this->base_arr.get(); 

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feed_allocation_cost         = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feed_mem(feed_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feed_mem.get()));

                std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

                for (size_t i = 0u; i < sz; ++i){
                    size_t hashed_value     = dg::network_hash::hash_reflectible(global_id_arr[i]);
                    size_t partitioned_idx  = hased_value & (this->pow2_base_arr_sz - 1u);
                    auto feed_arg           = InternalFeedArgument{};
                    feed_arg.id             = global_id_arr[i];
                    feed_arg.failed_err_ptr = std::next(exception_arr, i);

                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }
            }

            void get_expired_id(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

                sz = 0u;

                for (size_t i = 0u; i < this->zero_get_expired_id_bounce_sz; ++i){
                    size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                    size_t idx          = random_value & (this->pow2_base_arr_sz - 1u);
                    this->base_arr[idx]->get_expired_id(output_arr, sz, cap);
                    
                    if (sz != 0u){
                        return;
                    }
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_tick_per_load;
            }
        
        private:

            struct InternalFeedArgument{
                GlobalIdentifier id;
                exception_t * failed_err_ptr;
            };

            struct InternalFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalFeedArgument>{

                std::unique_ptr<EntranceControllerInterface> * dst;

                void push(const size_t& idx, std::move_iterator<InternalFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> global_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        global_id_arr[i] = base_data_arr[i].id;
                    }

                    this->dst[idx]->tick(global_id_arr.get(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            base_data_arr[i].failed_err_ptr = exception_arr[i];
                        }
                    }
                }
            };
    };

    class ExhaustionControlledEntranceController: public virtual EntranceControllerInterface{

        private:

            std::unique_ptr<EntranceControllerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller;

        public:

            ExhaustionControlledEntranceController(std::unique_ptr<EntranceControllerInterface> base,
                                                   std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                   std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller) noexcept: base(std::move(base)),
                                                                                                                                   executor(std::move(executor)),
                                                                                                                                   exhaustion_controller(std::move(exhaustion_controller)){}

            void tick(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                GlobalIdentifier * first_id_ptr     = global_id_arr;
                GlobalIdentifier * last_id_ptr      = std::next(first_id_ptr, sz);
                exception_t * first_exception_ptr   = exception_arr;
                exception_t * last_exception_ptr    = std::next(exception_arr, sz);
                size_t sliding_window_sz            = sz;

                //let's try to grasp what's going on here
                //we are protected by range bro
                //assume we are to resolve every QUEUE_FULL, our state is correct at all time
                //we are to maintain the state of peeking the next sliding_window of QUEUE_FULL
                //such is we are to decay one state -> another state of lesser std::distance(first_id_ptr, last_id_ptr)

                auto task = [&, this]() noexcept{
                    this->base->tick(first_id_ptr, sliding_window_sz, first_exception_ptr);
                    size_t queueing_sz                          = std::count(first_exception_ptr, last_exception_ptr, dg::network_exception::QUEUE_FULL);
                    exception_t err                             = this->exhaustion_controller->update_waiting_size(queueing_sz);

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    }

                    exception_t * first_retriable_exception_ptr = std::find(first_exception_ptr, last_exception_ptr, dg::network_exception::QUEUE_FULL);
                    exception_t * last_retriable_exception_ptr  = std::find_if(first_retriable_exception_ptr, last_exception_ptr, [](exception_t err){return err != dg::network_exception::QUEUE_FULL;});
                    sliding_window_sz                           = std::distance(first_retriable_exception_ptr, last_retriable_exception_ptr);
                    size_t relative_offset                      = std::distance(first_exception_ptr, first_retriable_exception_ptr);

                    std::advance(first_id_ptr, relative_offset);
                    std::advance(first_exception_ptr, relative_offset);

                    return !this->exhaustion_controller->is_should_wait() || first_id_ptr == last_id_ptr;
                };

                auto virtual_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(std::move(task));
                this->executor->exec(virtual_task);
            }

            void get_expired_id(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

                this->base->get_expired_id(output_arr, sz, cap);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    class PacketAssembler: public virtual PacketAssemblerInterface{

        private:

            dg::unordered_map<GlobalIdentifier, AssembledPacket> packet_map;
            size_t packet_map_cap;
            size_t global_packet_segment_cap;
            size_t global_packet_segment_counter;
            size_t max_segment_sz_per_assembly; 
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            PacketAssembler(dg::unordered_map<GlobalIdentifier, AssembledPacket> packet_map,
                            size_t packet_map_cap,
                            size_t global_packet_segment_cap,
                            size_t global_packet_segment_counter,
                            size_t max_segment_sz_per_assembly,
                            std::unique_ptr<std::mutex> mtx,
                            stdx::hdi_container<size_t> consume_sz_per_load) noexcept: packet_map(std::move(packet_map)),
                                                                                       packet_map_cap(packet_map_cap),
                                                                                       global_packet_segment_cap(global_packet_segment_cap),
                                                                                       global_packet_segment_counter(global_packet_segment_counter),
                                                                                       max_segment_sz_per_assembly(max_segment_sz_per_assembly),
                                                                                       mtx(std::move(mtx)),
                                                                                       consume_sz_per_load(std::move(consume_sz_per_load)){}

            void assemble(std::move_iterator<PacketSegment *> segment_arr, size_t sz, std::expected<AssembledPacket, exception_t> * assembled_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                PacketSegment * segment_arr_base = segment_arr.base();

                for (size_t i = 0u; i < sz; ++i){
                    if (segment_arr_base[i].segment_sz == 0u){
                        assembled_arr[i] = this->make_empty_assembled_packet(0u);
                        continue;
                    }

                    auto map_ptr = this->packet_map.find(segment_arr_base[i].id);

                    if (map_ptr == this->packet_map.end()){
                        if (segment_arr_base[i].segment_sz > this->max_segment_sz_per_assembly){
                            assembled_arr[i] = std::unexpected(dg::network_exception::SOCKET_STREAM_BAD_SEGMENT_SIZE);
                            continue;
                        }

                        if (this->packet_map.size() == this->packet_map_cap){
                            assembled_arr[i] = std::unexpected(dg::network_exception::QUEUE_FULL);
                            continue;
                        }

                        if (this->global_packet_segment_counter + segment_arr_base[i].segment_sz > this->global_packet_segment_cap){
                            assembled_arr[i] = std::unexpected(dg::network_exception::QUEUE_FULL);
                            continue;
                        }

                        std::expected<AssembledPacket, exception_t> waiting_pkt = this->make_empty_assembled_packet(segment_arr_base[i].segment_sz);

                        if (!waiting_pkt.has_value()){
                            assembled_arr[i] = std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                            continue;
                        }

                        auto [emplace_ptr, status]          = this->packet_map.try_emplace(segment_arr_base[i].id, std::move(waiting_pkt.value()));
                        dg::network_exception_handler::dg_assert(status);
                        map_ptr                             = emplace_ptr;
                        this->global_packet_segment_counter += segment_arr_base[i].segment_sz; 
                    }

                    //this is fine, because it is object <dig_in> operator = (object&&), there wont be issues, yet we want to make this explicit

                    size_t segment_idx                      = segment_arr_base[i].segment_idx;
                    map_ptr->second.data[segment_idx]       = std::move(segment_arr_base[i]);
                    map_ptr->second.collected_segment_sz    += 1;

                    if (map_ptr->second.collected_segment_sz == map_ptr->second.total_segment_sz){
                        this->global_packet_segment_counter -= map_ptr->second.total_segment_sz; 
                        assembled_arr[i]                    = std::move(map_ptr->second);
                        this->packet_map.erase(map_ptr);
                    } else{
                        assembled_arr[i] = std::unexpected(dg::network_exception::SOCKET_STREAM_QUEUING);
                    }
                }
            }

            void destroy(GlobalIdentifier * id_arr, size_t sz) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    auto map_ptr = this->packet_map.find(id_arr[i]);

                    if (map_ptr == this->packet_map.end()){
                        continue;
                    }

                    this->global_packet_segment_counter -= map_ptr->second.total_segment_sz;
                    this->packet_map.erase(map_ptr);
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load.value;
            }

        private:

            auto make_empty_assembled_packet(size_t segment_sz) noexcept -> std::expected<AssembledPacket, exception_t>{

                auto vec            = dg::network_exception::cstyle_initialize<dg::vector<PacketSegment>>(segment_sz);

                if (!vec.has_value()){
                    return std::unexpected(vec.error());
                }

                size_t collected    = 0u;
                size_t total        = segment_sz;

                return AssembledPacket{std::move(vec.value()), collected, total};
            }
    };

    class RandomHashDistributedPacketAssembler: public virtual PacketAssemblerInterface{

        private:

            std::unique_ptr<std::unique_ptr<PacketAssemblerInterface>[]> base_arr;
            const size_t keyvalue_feed_cap;
            const size_t pow2_base_arr_sz;
            const size_t consume_sz_per_load;

        public:

            RandomHashDistributedPacketAssembler(std::unique_ptr<std::unique_ptr<PacketAssemblerInterface>[]> base_arr,
                                                 size_t keyvalue_feed_cap,
                                                 size_t pow2_base_arr_sz,
                                                 size_t consume_sz_per_load) noexcept: base_arr(std::move(base_arr)),
                                                                                       keyvalue_feed_cap(keyvalue_feed_cap),
                                                                                       pow2_base_arr_sz(pow2_base_arr_sz),
                                                                                       consume_sz_per_load(consume_sz_per_load){}

            void assemble(std::move_iterator<PacketSegment *> segment_arr, size_t sz, std::expected<AssembledPacket, exception_t> * assembled_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                PacketSegment * base_segment_arr    = segment_arr.base();

                auto feed_resolutor                 = InternalAssembleFeedResolutor{};
                feed_resolutor.dst                  = this->base_arr.get(); 

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feed_allocation_cost         = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feed_mem(feed_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feed_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    size_t hashed_value             = dg::network_hash::hash_reflectible(base_segment_arr[i].id);
                    size_t partitioned_idx          = hashed_value & (this->pow2_base_arr_sz - 1u);

                    auto feed_arg                   = InternalAssembleFeedArgument{}:
                    feed_arg.segment                = std::move(base_segment_arr[i]);
                    feed_arg.fallback_segment_ptr   = std::next(base_segment_arr, i);
                    feed_arg.rs                     = std::next(std::next(assembled_arr, i)); 

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, std::move(feed_arg));
                }
            }

            void destroy(GlobalIdentifier * id_arr, size_t sz) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }
                
                auto feed_resolutor                 = InternalDestroyFeedResolutor{};
                feed_resolutor.dst                  = this->base_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feed_allocation_cost         = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feed_mem(feed_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feed_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    size_t hashed_value             = dg::network_hash::hash_reflectible(id_arr[i]);
                    size_t partitioned_idx          = hashed_value & (this->pow2_base_arr_sz - 1u);

                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), partitioned_idx, id_arr[i]);
                } 
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load;
            }
        
        private:

            struct InternalAssembleFeedArgument{
                PacketSegment segment;
                PacketSegment * fallback_segment_ptr;
                std::expected<AssembledPacket, exception_t> * rs;
            };

            struct InternalAssembleFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalAssembleFeedArgument>{
                std::unique_ptr<PacketAssemblerInterface> * dst;

                void push(const size_t& idx, std::move_iterator<InternalAssembleFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalAssembleFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<PacketSegment[]> segment_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<AssembledPacket, exception_t>[]> rs_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        segment_arr[i] = std::move(base_data_arr[i].segment);
                    }

                    this->dst[idx]->assemble(std::make_move_iterator(segment_arr.get()), sz, rs_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (rs_arr[i].has_value()){
                            *base_data_arr[i].rs = std::move(rs_arr[i]); //thru
                            continue;   
                        }

                        if (rs_arr[i].error() == dg::network_exception::SOCKET_STREAM_QUEUING){
                            *base_data_arr[i].rs = std::move(rs_arr[i]); //thru
                            continue;
                        }

                        //fallback, move semantic does not apply to failed exception_t
                        *base_data_arr[i].rs                    = std::move(rs_arr[i]);
                        *base_data_arr[i].fallback_segment_ptr  = std::move(segment_arr[i]);
                    }
                }
            };

            struct InternalDestroyFeedResolutor: dg::network_producer_consumer::KVConsuemrInterface<size_t, GlobalIdentifier>{

                std::unique_ptr<PacketAssemblerInterface> * dst;

                void push(const size_t& idx, std::move_iterator<GlobalIdentifier *> data_arr, size_t sz) noexcept{

                    this->dst[idx]->destroy(data_arr.base(), sz);
                }
            };
    };

    class ExhaustionControlledPacketAssembler: public virtual PacketAssemblerInterface{

        private:

            std::unique_ptr<PacketAssemblerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller;

        public:

            ExhaustionControlledPacketAssembler(std::unique_ptr<PacketAssemblerInterface> base,
                                                std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller) noexcept: base(std::move(base)),
                                                                                                                                executor(std::move(executor)),
                                                                                                                                exhaustion_controller(std::move(exhaustion_controller)){}

            void assemble(std::move_iterator<PacketSegment *> segment_arr, size_t sz, std::expected<AssembledPacket, exception_t> * assembled_arr) noexcept{

                PacketSegment * base_segment_arr                                    = segment_arr.base();
                PacketSegment * first_segment_ptr                                   = base_segment_arr;
                PacketSegment * last_segment_ptr                                    = std::next(first_segment_ptr, sz);
                std::expected<AssembledPacket, exception_t> * first_assembled_ptr   = assembled_arr;
                std::expected<AssembledPacket, exception_t> * last_assembled_ptr    = std::next(first_assembled_ptr, sz);
                size_t sliding_window_sz                                            = sz;
                auto full_signal                                                    = std::expected<AssembledPacket, exception_t>(std::unexpected(dg::network_exception::QUEUE_FULL)); 

                auto task = [&, this]() noexcept{
                    this->base->assemble(std::make_move_iterator(first_segment_ptr), sliding_window_sz, first_assembled_ptr);
                    size_t queueing_sz  = std::count(first_assembled_ptr, last_assembled_ptr, full_signal);
                    exception_t err     = this->exhaustion_controller->update_waiting_size(queueing_sz);

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    }

                    std::expected<AssembledPacket, exception_t> * first_retriable_ptr   = std::find(first_assembled_ptr, last_assembled_ptr, full_signal);
                    std::expected<AssembledPacket, exception_t> * last_retriable_ptr    = std::find_if(first_retriable_ptr, last_assembled_ptr, [](const auto& err){return err != full_signal;});
                    sliding_window_sz                                                   = std::distance(first_retriable_ptr, last_retriable_ptr);
                    size_t relative_offset                                              = std::distance(first_assembled_ptr, first_retriable_ptr);

                    std::advance(first_segment_ptr, relative_offset);
                    std::advance(first_assembled_ptr, relative_offset);

                    return !this->exhaustion_controller->is_should_wait() || first_segment_ptr == last_segment_ptr;
                }
            }

            void destroy(GlobalIdentifier * id_arr, size_t sz) noexcept{

                this->base->destroy(id_arr, sz);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    //OK, dg::deque -> cyclic queue
    class BufferFIFOContainer: public virtual InBoundContainerInterface{

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
                std::fill(std::next(exception_arr, app_sz), std::next(exception_arr, sz), dg::network_exception::SOCKET_QUEUE_FULL);
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

    //OK
    class ExhaustionControlledBufferContainer: public virtual InBoundContainerInterface{

        private:

            std::unique_ptr<InBoundContainerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller;

        public:

            ExhaustionControlledBufferContainer(std::unique_ptr<InBoundContainerInterface> base,
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

                    size_t waiting_sz                   = std::count(exception_arr_first, exception_arr_last, dg::network_exception::SOCKET_QUEUE_FULL);
                    exception_t err                     = this->exhaustion_controller->update_waiting_size(waiting_sz);

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(err));
                    }

                    exception_t * retriable_arr_first   = std::find(exception_arr_first, exception_arr_last, dg::network_exception::SOCKET_QUEUE_FULL);
                    exception_t * retriable_arr_last    = std::find_if(retriable_arr_first, exception_arr_last, [](exception_t err){return err != dg::network_exception::SOCKET_QUEUE_FULL;});

                    size_t relative_offset              = std::distance(exception_arr_first, retriable_arr_first);
                    sliding_window_sz                   = std::distance(retriable_arr_first, retriable_arr_last);

                    std::advance(buffer_arr_first, relative_offset);
                    std::advance(exception_arr_first, relative_offset); 

                    return !this->exhaustion_controller->is_should_wait() || (buffer_arr_first == buffer_arr_last); //TODOs: we want to subscribe these guys to a load_balancer system
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

    //OK
    class HashDistributedBufferContainer: public virtual InBoundContainerInterface{

        private:

            std::unique_ptr<std::unique_ptr<InBoundContainerInterface>[]> buffer_container_vec;
            size_t pow2_buffer_container_vec_sz;
            size_t zero_buffer_retry_sz;
            size_t consume_sz_per_load; 

        public:

            HashDistributedBufferContainer(std::unique_ptr<std::unique_ptr<InBoundContainerInterface>[]> buffer_container_vec,
                                           size_t pow2_buffer_container_vec_sz,
                                           size_t zero_buffer_retry_sz,
                                           size_t consume_sz_per_load) noexcept: buffer_container_vec(std::move(buffer_container_vec)),
                                                                                 pow2_buffer_container_vec_sz(pow2_buffer_container_vec_sz),
                                                                                 zero_buffer_retry_sz(zero_buffer_retry_sz),
                                                                                 consume_sz_per_load(consume_sz_per_load){}

            void push(std::move_iterator<dg::string *> buffer_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                size_t random_idx   = random_value & (this->pow2_buffer_container_vec_sz - 1u);

                this->buffer_container_vec[random_idx]->push(buffer_arr, sz, exception_arr);
            }

            void pop(dg::string * output_buffer_arr, size_t& sz, size_t output_buffer_arr_cap) noexcept{

                sz = 0u;

                for (size_t i = 0u; i < this->zero_buffer_retry_sz; ++i){
                    size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                    size_t random_idx   = random_value & (this->pow2_buffer_container_vec_sz - 1u);

                    this->buffer_container_vec[random_idx]->pop(output_buffer_arr, sz, output_buffer_arr_cap);

                    if (sz != 0u){
                        return;
                    }
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load;
            }
    };

    //OK
    class ReactingBufferContainer: public virtual InBoundContainerInterface{

        private:

            std::unique_ptr<InBoundContainerInterface> base;
            std::unique_ptr<ComplexReactor> reactor;
            std::chrono::nanoseconds max_wait_time;

        public:

            ReactingBufferContainer(std::unique_ptr<InBoundContainerInterface> base,
                                    std::unique_ptr<ComplexReactor> reactor,
                                    std::chrono::nanoseconds max_wait_time) noexcept: base(std::move(base)),
                                                                                      reactor(std::move(reactor)),
                                                                                      max_wait_time(max_wait_time){}

            void push(std::move_iterator<dg::string *> buffer_arr, size_t sz, exception_t * exception_arr) noexcept{

                this->base->push(buffer_arr, sz, exception_arr);
                size_t thru_sz = std::count(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);
                this->reactor->increment(thru_sz);
            }

            void pop(dg::string * output_buffer_arr, size_t& sz, size_t output_buffer_arr_cap) noexcept{

                std::atomic_signal_fence(std::memory_order_seq_cst);
                this->reactor->subscribe(this->max_wait_time);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                this->base->pop(output_buffer_arr, sz, output_buffer_arr_cap);
                this->reactor->decrement(sz);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    //OK
    class ExpiryWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<PacketAssemblerInterface> packet_assembler;
            std::shared_ptr<EntranceControllerInterface> entrance_controller;
            size_t expired_id_consume_sz;
            size_t busy_consume_sz;

        public:

            ExpiryWorker(std::shared_ptr<PacketAssemblerInterface> packet_assembler,
                         std::shared_ptr<EntranceControllerInterface> entrance_controller,
                         size_t expired_id_consume_sz,
                         size_t busy_consume_sz) noexcept: packet_assembler(std::move(packet_assembler)),
                                                           entrance_controller(std::move(entrance_controller)),
                                                           expired_id_consume_sz(expired_id_consume_sz),
                                                           busy_consume_sz(busy_consume_sz){}
            
            bool run_one_epoch() noexcept{

                size_t id_arr_cap   = this->expired_id_consume_sz;
                size_t id_arr_sz    = {};
                dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> id_arr(id_arr_cap);
                this->entrance_controller->get_expired_id(id_arr.get(), id_arr_sz, id_arr_cap);

                //I admit things could be done more optimally, don't even think about that, because we would definitely change things in the future, at which point we will circle back to feed

                auto pa_feed_resolutor              = InternalPacketAssemblerDestroyFeedResolutor{};
                pa_feed_resolutor.dst               = this->packet_assembler.get(); 

                size_t trimmed_pa_feed_sz           = std::min(std::min(DEFAULT_KEY_FEED_SIZE, this->packet_assembler->max_consume_size()), id_arr_sz);
                size_t pa_feeder_allocation_cost    = dg::network_producer_consumer::delvrsrv_allocation_cost(&pa_feed_resolutor, trimmed_pa_feed_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> pa_feeder_mem(pa_feeder_allocation_cost);
                auto pa_feeder                      = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&pa_feed_resolutor, trimmed_pa_feed_sz, pa_feeder_mem.get()));

                for (size_t i = 0u; i < id_arr_sz; ++i){
                    dg::network_producer_consumer::delvsrv_deliver(pa_feeder.get(), id_arr[i]);
                }

                return id_arr_sz >= this->busy_consume_sz;
            }

        private:

            struct InternalPacketAssemblerDestroyFeedResolutor{

                PacketAssemblerInterface * dst;

                void push(std::move_iterator<GlobalIdentifier *> id_arr, size_t id_arr_sz) noexcept{

                    this->dst->destroy(id_arr.base(), id_arr_sz);
                }
            };     
    };

    //OK
    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<PacketAssemblerInterface> packet_assembler;
            std::shared_ptr<InBoundContainerInterface> inbound_container;
            std::shared_ptr<EntranceControllerInterface> entrance_controller;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
            size_t upstream_consume_sz;
            size_t busy_consume_sz;

        public:

            InBoundWorker(std::shared_ptr<PacketAssemblerInterface> packet_assembler,
                          std::shared_ptr<InBoundContainerInterface> inbound_container,
                          std::shared_ptr<EntranceControllerInterface> entrance_controller,
                          std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                          size_t upstream_consume_sz,
                          size_t busy_consume_sz) noexcept: packet_assembler(std::move(packet_assembler)),
                                                            inbound_container(std::move(inbound_container)),
                                                            entrance_controller(std::move(entrance_controller)),
                                                            base(std::move(base)),
                                                            upstream_consume_sz(upstream_consume_sz),
                                                            busy_consume_sz(busy_consume_sz){}

            bool run_one_epoch() noexcept{
                
                size_t consuming_sz     = 0u;
                size_t consuming_cap    = this->upstream_consume_sz;
                dg::network_stack_allocation::NoExceptAllocation<dg::string[]> buf_arr(consuming_cap);

                this->base->recv(buf_arr.get(), consuming_sz, consuming_cap); 

                auto et_feed_resolutor                      = InternalEntranceFeedResolutor{};
                et_feed_resolutor.entrance_controller       = this->entrance_controller.get();

                size_t trimmed_et_feed_cap                  = std::min(std::min(DEFAULT_KEY_FEED_SIZE, consuming_sz), this->entrance_controller->max_consume_size());
                size_t et_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&et_feed_resolutor, trimmed_et_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> et_feeder_mem(et_feeder_allocation_cost);
                auto et_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&et_feed_resolutor, trimmed_et_feed_cap, et_feeder_mem.get()));

                size_t ib_feed_resolutor                    = InternalInBoundFeedResolutor{};
                ib_feed_resolutor.inbound_container         = this->inbound_container.get();

                size_t trimmed_ib_feed_cap                  = std::min(std::min(DEFAULT_KEY_FEED_SIZE, consuming_sz), this->inbound_container->max_consume_size());
                size_t ib_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&ib_feed_resolutor, trimmed_ib_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> ib_feeder_mem(ib_feeder_allocation_cost);
                auto ib_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ib_feed_resolutor, trimmed_ib_feed_cap, ib_feeder_mem.get())); 

                auto ps_feed_resolutor                      = InternalPacketSegmentFeedResolutor{};
                ps_feed_resolutor.packet_assembler          = this->packet_assembler.get();
                ps_feed_resolutor.entrance_feeder           = et_feeder.get();
                ps_feed_resolutor.inbound_feeder            = ib_feeder.get();

                size_t trimmed_ps_feed_cap                  = std::min(DEFAULT_KEY_FEED_SIZE, consuming_sz);
                size_t ps_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&ps_feed_resolutor, trimmed_ps_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> ps_feeder_mem(ps_feeder_allocation_cost);
                auto ps_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ps_feed_resolutor, trimmed_ps_feed_cap, ps_feeder_mem.get()));  

                for (size_t i = 0u; i < consuming_sz; ++i){
                    std::expected<PacketSegment, exception_t> pkt = deserialize_packet_segment(std::move(buf_arr[i]));

                    if (!pkt.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(pkt.error()));
                        continue;
                    }

                    dg::network_producer_consumer::delvrsrv_deliver(ps_feeder.get(), std::move(pkt.value()));
                }

                return consuming_sz >= this->busy_consume_sz;
            }
        
        private:

            struct InternalEntranceFeedResolutor: dg::network_producer_consumer::ConsumerInterface<GlobalIdentifier>{

                EntranceControllerInterface * entrance_controller;

                void push(std::move_iterator<GlobalIdentifier *> id_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    this->entrance_controller->tick(id_arr.base(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::SOCKET_STREAM_LEAK));
                        }
                    }
                }
            };

            struct InternalInBoundFeedResolutor: dg::network_producer_consumer::ConsumerInterface<dg::string>{

                InBoundContainerInterface * inbound_container;

                void push(std::move_iterator<dg::string *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    this->inbound_container->push(data_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

            struct InternalPacketSegmentFeedResolutor: dg::network_producer_consumer::ConsumerInterface<PacketSegment>{

                PacketAssemblerInterface * packet_assembler;
                dg::network_producer_consumer::DeliveryHandle<GlobalIdentifier> * entrance_feeder;
                dg::network_producer_consumer::DeliveryHandle<dg::string> * inbound_feeder;

                void push(std::move_iterator<PacketSegment *> data_arr, size_t sz) noexcept{
                    
                    PacketSegment * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> global_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<AssembledPacket, exception_t>[]> assembled_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        global_id_arr[i] = base_data_arr[i].id; 
                    }

                    this->packet_assembler->assemble(std::make_move_iterator(base_data_arr), sz, assembled_arr.get());
                    
                    std::atomic_signal_fence(std::memory_order_seq_cst);

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_producer_consumer::delvrsrv_deliver(this->entrance_feeder, global_id_arr[i]);
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        if (assembled_arr[i].has_value()){
                            std::expected<dg::string, exception_t> buf = assembled_packet_to_buffer(static_cast<AssembledPacket&&>(assembled_arr[i].value())); 

                            if (!buf.has_value()){
                                dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(buf.error()));
                            } else{
                                dg::network_producer_consumer::delvrsrv_deliver(this->inbound_feeder, std::move(buf.value()));
                            }
                        } else{
                            if (assembled_arr[i].error() != dg::network_exception::SOCKET_STREAM_QUEUING){
                                dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(assembled_arr[i].error()));
                            }
                        }
                    }
                }
            };
    };

    //OK
    class MailBox: public virtual dg::network_kernel_mailbox_impl1::core::MailboxInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::unique_ptr<PacketizerInterface> packetizer;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
            std::shared_ptr<InBoundContainerInterface> inbound_container;
            size_t transmission_vectorization_sz;

        public:

            MailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                    std::unique_ptr<PacketizerInterface> packetizer,
                    std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                    std::shared_ptr<InBoundContainerInterface> inbound_container,
                    size_t transmission_vectorization_sz) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                    packetizer(std::move(packetizer)),
                                                                    base(std::move(base)),
                                                                    inbound_container(std::move(inbound_container)),
                                                                    transmission_vectorization_sz(transmission_vectorization_sz){}

            void send(std::move_iterator<MailBoxArgument *> data_arr, size_t sz, exception_t * exception_arr) noexcept{
                
                MailBoxArgument * base_data_arr = data_arr.base();

                auto feed_resolutor             = InternalFeedResolutor{};
                feed_resolutor.dst              = this->base.get(); 

                size_t trimmed_mailbox_feed_sz  = std::min(std::min(this->transmission_vectorization_sz, sz * MAX_SEGMENT_SIZE), this->base->max_consume_size());
                size_t feeder_allocation_cost   = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_mailbox_feed_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_mailbox_feed_sz, feed_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<dg::vector<PacketSegment>, exception_t> segment_vec = this->packetizer->packetize(static_cast<dg::string&&>(base_data_arr[i].content));

                    if (!segment_vec.has_value()){
                        exception_arr[i] = segment_vec.error();
                        continue;
                    }

                    for (size_t j = 0u; j < segment_vec.value().size(); ++j){
                        dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), base_data_arr[i].to, std::move(segment_vec.value()[j])); //we are to attempt to temporally group the <to> transmission, to increase ack vectorization chances
                    }

                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            void recv(dg::string * output_arr, size_t& output_arr_sz, size_t output_arr_cap) noexcept{

                return this->inbound_container->pop(output_arr, output_arr_sz, output_arr_cap);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }

        private:

            struct InternalFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<Address, PacketSegment>{

                dg::network_kernel_mailbox_impl1::core::MailBoxInterface * dst;

                void push(const Address& to, std::move_iterator<PacketSegment *> data_arr, size_t sz) noexcept{

                    PacketSegment * base_data_arr   = data_arr.base();
                    size_t arr_cap                  = sz;
                    size_t arr_sz                   = 0u;

                    dg::network_stack_allocation::NoExceptAllocation<MailBoxArgument[]> mailbox_arg_arr(arr_cap);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(arr_cap);

                    for (size_t i = 0u; i < sz; ++i){
                        std::expected<dg::string, exception_t> serialized = serialize_packet_segment(static_cast<PacketSegment&&>(base_data_arr[i]));

                        if (!serialized.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(serialized.error()));
                            continue;
                        }

                        mailbox_arg_arr[arr_sz].content = std::move(serialized.value());
                        mailbox_arg_arr[arr_sz].to      = to;
                        arr_sz                          += 1;
                    }

                    this->dst->send(std::make_move_iterator(mailbox_arg_arr.get()), arr_sz, exception_arr.get());

                    for (size_t i = 0u; i < arr_sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

    };

    struct Factory{

        static auto spawn_packetizer(size_t segment_byte_sz, Address factory_addr) -> std::unique_ptr<PacketizerInterface>{
            
            const size_t MIN_SEGMENT_SIZE = size_t{1} << 8;
            const size_t MAX_SEGMENT_SIZE = size_t{1} << 13; 

            if (std::clamp(segment_byte_sz, MIN_SEGMENT_SIZE, MAX_SEGMENT_SIZE) != segment_byte_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t random_seed = dg::network_randomizer::randomize_int<size_t>(); //this is to avoid reset + id collision
            return std::make_unique<Packetizer>(segment_byte_sz, factory_addr, random_seed);
        }

        //this needs exhaustion controlled - program-defined
        static auto spawn_entrance_controller(std::chrono::nanoseconds expiry_period) -> std::unique_ptr<EntranceControllerInterface>{
            
            const std::chrono::nanoseconds MIN_EXPIRY_PERIOD = std::chrono::milliseconds(1); 
            const std::chrono::nanoseconds MAX_EXPIRY_PERIOD = std::chrono::seconds(20);

            if (std::clamp(expiry_period.count(), MIN_EXPIRY_PERIOD.count(), MAX_EXPIRY_PERIOD.count()) != expiry_period.count()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }
 
            auto entrance_entry_pq  = dg::vector<EntranceEntry>{};
            auto key_id_map         = dg::unordered_map<GlobalIdentifier, size_t>{};
            size_t random_seed      = dg::network_randomizer::randomize_int<size_t>();
            auto mtx                = std::make_unique<std::mutex>();

            return std::make_unique<EntranceController>(std::move(entrance_entry_pq), std::move(key_id_map), random_seed, expiry_period, std::move(mtx));
        }

        static auto spawn_packet_assembler() -> std::unique_ptr<PacketAssemblerInterface>{

            auto packet_map = dg::unordered_map<GlobalIdentifier, AssembledPacket>{};
            auto mtx        = std::make_unique<std::mutex>();

            return std::make_unique<PacketAssembler>(std::move(packet_map), std::move(mtx));
        }

        static auto spawn_exhaustion_controlled_packet_assembler(std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor, size_t capacity) -> std::unique_ptr<PacketAssemblerInterface>{
            
            const size_t MIN_CAPACITY = size_t{1} << 8;
            const size_t MAX_CAPACITY = size_t{1} << 20;

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(capacity, MIN_CAPACITY, MAX_CAPACITY) != capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto base           = spawn_packet_assembler();
            auto counter_map    = dg::unordered_map<GlobalIdentifier, size_t>{};
            size_t size         = 0u;
            auto mtx            = std::make_unique<std::mutex>(); 

            return std::make_unique<ExhaustionControlledPacketAssembler>(std::move(base), std::move(executor), std::move(counter_map), capacity, size, std::move(mtx));
        }

        static auto spawn_inbound_container() -> std::unique_ptr<InBoundContainerInterface>{

            auto vec    = dg::deque<dg::string>{};
            auto mtx    = std::make_unique<std::mutex>();

            return std::make_unique<InBoundContainer>(std::move(vec), std::move(mtx));
        }

        static auto spawn_exhaustion_controlled_inbound_container(std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor, size_t capacity) -> std::unique_ptr<InBoundContainerInterface>{

            const size_t MIN_CAPACITY = size_t{1} << 8;
            const size_t MAX_CAPACITY = size_t{1} << 20;

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(capacity, MIN_CAPACITY, MAX_CAPACITY) != capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto base   = spawn_inbound_container();
            size_t size = 0u;
            auto mtx    = std::make_unique<std::mutex>(); 

            return std::make_unique<ExhaustionControlledInBoundContainer>(std::move(base), std::move(executor), capacity, size, std::move(mtx));
        }

        static auto spawn_mailbox_streamx(std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                                          std::unique_ptr<InBoundContainerInterface> inbound_container,
                                          std::unique_ptr<PacketAssemblerInterface> packet_assembler,
                                          std::unique_ptr<EntranceControllerInterface> entrance_controller,
                                          std::unique_ptr<PacketizerInterface> packetizer,
                                          size_t num_inbound_worker,
                                          size_t num_expiry_worker) -> std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>{
            
            const size_t MIN_INBOUND_WORKER     = 1u;
            const size_t MAX_INBOUND_WORKER     = 1024u;
            const size_t MIN_EXPIRY_WORKER      = 1u;
            const size_t MAX_EXPIRY_WORKER      = 1024u; 

            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (inbound_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (packet_assembler == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (entrance_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (packetizer == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(num_inbound_worker, MIN_INBOUND_WORKER, MAX_INBOUND_WORKER) != num_inbound_worker){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(num_expiry_worker, MIN_EXPIRY_WORKER, MAX_EXPIRY_WORKER) != num_expiry_worker){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base_sp   = std::move(base);
            std::shared_ptr<InBoundContainerInterface> inbound_container_sp                     = std::move(inbound_container);
            std::shared_ptr<PacketAssemblerInterface> packet_assembler_sp                       = std::move(packet_assembler);
            std::shared_ptr<EntranceControllerInterface> entrance_controller_sp                 = std::move(entrance_controller);
            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec{};

            //TODOs: this is buggy - this will be DL if exception raises - not to worry about this now
            for (size_t i = 0u; i < num_inbound_worker; ++i){
                auto worker = std::make_unique<InBoundWorker>(packet_assembler_sp, inbound_container_sp, entrance_controller_sp, base_sp);
                auto handle = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker)));
                daemon_vec.push_back(std::move(handle));
            }

            //TODOs: this is buggy - this will be DL if exception raises - not to worry about this now
            for (size_t i = 0u; i < num_expiry_worker; ++i){
                auto worker = std::make_unique<ExpiryWorker>(packet_assembler_sp, entrance_controller_sp);
                auto handle = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::TRANSPORTATION_DAEMON, std::move(worker)));
                daemon_vec.push_back(std::move(handle));
            }

            return std::make_unique<MailBox>(std::move(daemon_vec), std::move(packetizer), base_sp, inbound_container_sp);
        }
    };

    struct Config{
        Address factory_addr;
        size_t expiry_worker_count;
        size_t inbound_worker_count;
        size_t packet_assembler_capacity; 
        size_t inbound_container_capacity;
        size_t segment_byte_sz;
        std::chrono::nanoseconds packet_expiry;
        std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device;
        std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
    };

    auto spawn(Config config) -> std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>{

        std::unique_ptr<InBoundContainerInterface> inbound_container        = Factory::spawn_exhaustion_controlled_inbound_container(config.infretry_device, config.inbound_container_capacity);
        std::unique_ptr<PacketAssemblerInterface> packet_assembler          = Factory::spawn_exhaustion_controlled_packet_assembler(config.infretry_device, config.packet_assembler_capacity);
        std::unique_ptr<EntranceControllerInterface> entrance_controller    = Factory::spawn_entrance_controller(config.packet_expiry);
        std::unique_ptr<PacketizerInterface> packetizer                     = Factory::spawn_packetizer(config.segment_byte_sz, config.factory_addr);
        
        return Factory::spawn_mailbox_streamx(std::move(config.base), std::move(inbound_container), std::move(packet_assembler), 
                                              std::move(entrance_controller), std::move(packetizer), config.inbound_worker_count, 
                                              config.expiry_worker_count);
    }
}

namespace dg::network_kernel_mailbox_impl1_radixx{

    using Address = dg::network_kernel_mailbox_impl1::model::Address; 
    using radix_t = uint32_t; 

    struct RadixMessage{
        radix_t radix;
        dg::string content;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(radix, content);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(radix, content);
        }
    };

    static auto serialize_radixmsg(RadixMessage inp) noexcept -> dg::string{

        constexpr size_t HEADER_SZ  = dg::network_trivial_serializer::size(radix_t{});
        size_t content_sz           = inp.content.size();
        size_t total_sz             = content_sz + HEADER_SZ;
        auto rs                     = std::move(inp.content);
        rs.resize(total_sz);
        char * header_ptr           = rs.data() + content_sz;
        dg::network_trivial_serializer::serialize_into(header_ptr, inp.radix);

        return rs;
    }

    static auto deserialize_radixmsg(dg::string inp) noexcept -> RadixMessage{

        constexpr size_t HEADER_SZ  = dg::network_trivial_serializer::size(radix_t{});
        auto [left, right]          = stdx::backsplit_str(std::move(inp), HEADER_SZ);
        radix_t radix               = {};
        dg::network_trivial_serializer::deserialize_into(radix, right.data());
        
        return RadixMessage(radix, std::move(left));
    }

    struct ExhaustionControllerInterface{
        virtual ~ExhaustionControllerInterface() noexcept = default;
        virtual auto thru_one() noexcept -> bool = 0;
        virtual void exit_one() noexcept = 0;
    };

    struct RadixMailboxInterface{
        virtual ~RadixMailboxInterface() noexcept = default;
        virtual void send(Address addr, dg::string buf, radix_t radix) noexcept = 0;
        virtual auto recv(radix_t radix) noexcept -> std::optional<dg::string> = 0;
    };

    struct InBoundContainerInterface{
        virtual ~InBoundContainerInterface() noexcept = default;
        virtual auto pop(radix_t) noexcept -> std::optional<dg::string> = 0;
        virtual void push(radix_t, dg::string) noexcept = 0;
    };

    class StdExhaustionController: public virtual ExhaustionControllerInterface{

        private:

            size_t cur_sz;
            size_t capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            StdExhaustionController(size_t cur_sz, 
                                    size_t capacity,
                                    std::unique_ptr<std::mutex> mtx) noexcept: cur_sz(cur_sz),
                                                                               capacity(capacity),
                                                                               mtx(std::move(mtx)){}
            
            auto thru_one() noexcept -> bool{
                
                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (this->cur_sz == this->capacity){
                    return false;
                }

                this->cur_sz += 1;
                return true;
            }

            void exit_one() noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->cur_sz -= 1;
            }
    };

    class InBoundContainer: public virtual InBoundContainerInterface{

        private:

            dg::unordered_map<radix_t, dg::deque<dg::string>> map;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            InBoundContainer(dg::unordered_map<radix_t, dg::deque<dg::string>> map,
                             std::unique_ptr<std::mutex> mtx) noexcept: map(std::move(map)),
                                                                        mtx(std::move(mtx)){}
            
            auto pop(radix_t radix) noexcept -> std::optional<dg::string>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto ptr        = this->map.find(radix);

                if (ptr == this->map.end()){
                    return std::nullopt;
                }

                if (ptr->second.empty()){
                    return std::nullopt;
                }

                dg::string rs = std::move(ptr->second.front());
                ptr->second.pop_front();

                return rs;
            }

            void push(radix_t radix, dg::string content) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->map[radix].push_back(std::move(content));
            }
    };

    class ExhaustionControlledInBoundContainer: public virtual InBoundContainerInterface{

        private:

            std::unique_ptr<InBoundContainerInterface> base;
            dg::unordered_map<radix_t, std::unique_ptr<ExhaustionControllerInterface>> exhaustion_controller_map;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledInBoundContainer(std::unique_ptr<InBoundContainerInterface> base, 
                                                 dg::unordered_map<radix_t, std::unique_ptr<ExhaustionControllerInterface>> exhaustion_controller_map,
                                                 std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                 std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                            exhaustion_controller_map(std::move(exhaustion_controller_map)),
                                                                                            executor(std::move(executor)),
                                                                                            mtx(std::move(mtx)){}
            
            auto pop(radix_t radix) noexcept -> std::optional<dg::string>{

                return this->internal_pop(radix);
            }
            
            void push(radix_t radix, dg::string content) noexcept{

                auto lambda = [&]() noexcept{return this->internal_push(radix, content);};
                auto exe    = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(lambda)>(std::move(lambda));
                this->executor->exec(exe);
            }
        
        private:

            auto internal_push(radix_t& radix, dg::string& content) noexcept -> bool{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto ec_ptr     = this->exhaustion_controller_map.find(radix);

                if constexpr(DEBUG_MODE_FLAG){
                    if (ec_ptr == this->exhaustion_controller_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                if (!ec_ptr->second->thru_one()){
                    return false;
                }

                this->base->push(radix, std::move(content));
                return true;
            }
            
            auto internal_pop(radix_t radix) noexcept -> std::optional<dg::string>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto rs         = this->base->pop(radix);

                if (!rs.has_value()){
                    return std::nullopt;
                } 

                auto ec_ptr     = this->exhaustion_controller_map.find(radix);

                if constexpr(DEBUG_MODE_FLAG){
                    if (ec_ptr == this->exhaustion_controller_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                ec_ptr->second->exit_one();
                return rs;
            }
    };

    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox;
            std::shared_ptr<InBoundContainerInterface> inbound_container;
        
        public:

            InBoundWorker(std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox,
                          std::shared_ptr<InBoundContainerInterface> inbound_container) noexcept: mailbox(std::move(mailbox)),
                                                                                                  inbound_container(std::move(inbound_container)){}
            
            bool run_one_epoch() noexcept{

                std::optional<dg::string> recv_data = this->mailbox->recv();

                if (!recv_data.has_value()){
                    return false;
                }
                
                RadixMessage msg = deserialize_radixmsg(std::move(recv_data.value()));
                this->inbound_container->push(msg.radix, std::move(msg.content));

                return true;
            }
    };

    class RadixMailBox: public virtual RadixMailboxInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec; 
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
            std::shared_ptr<InBoundContainerInterface> inbound_container;

        public:

            RadixMailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec, 
                         std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                         std::shared_ptr<InBoundContainerInterface> inbound_container) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                                                 base(std::move(base)),
                                                                                                 inbound_container(std::move(inbound_container)){}
            
            void send(Address addr, dg::string buf, radix_t radix) noexcept{

                RadixMessage msg{radix, std::move(buf)};
                dg::string bstream = serialize_radixmsg(std::move(msg));
                this->base->send(addr, std::move(bstream));
            }

            auto recv(radix_t radix) noexcept -> std::optional<dg::string>{

                return this->inbound_container->pop(radix);
            }
    };

    struct Factory{

        static auto spawn_exhaustion_controller(size_t capacity) -> std::unique_ptr<ExhaustionControllerInterface>{

            const size_t MIN_CAPACITY   = 1u;
            const size_t MAX_CAPACITY   = size_t{1} << 20;

            if (std::clamp(capacity, MIN_CAPACITY, MAX_CAPACITY) != capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto mtx = std::make_unique<std::mutex>(); 
            return std::make_unique<StdExhaustionController>(0u, capacity, std::move(mtx));
        }

        static auto spawn_inbound_container() -> std::unique_ptr<InBoundContainerInterface>{

            auto map    = dg::unordered_map<radix_t, dg::deque<dg::string>>{};
            auto mtx    = std::make_unique<std::mutex>(); 

            return std::make_unique<InBoundContainer>(std::move(map), std::move(mtx));
        }

        static auto spawn_exhaustion_controlled_inbound_container(dg::unordered_map<radix_t, size_t> exhaustion_map,
                                                                  std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) -> std::unique_ptr<InBoundContainerInterface>{
            
            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            dg::unordered_map<radix_t, std::unique_ptr<ExhaustionControllerInterface>> exhaustion_controller_map{};

            for (const auto& pair: exhaustion_map){
                exhaustion_controller_map.insert(std::make_pair(std::get<0>(pair), spawn_exhaustion_controller(std::get<1>(pair))));
            }

            std::unique_ptr<InBoundContainerInterface> base = spawn_inbound_container();
            std::unique_ptr<std::mutex> mtx = std::make_unique<std::mutex>(); 

            return std::make_unique<ExhaustionControlledInBoundContainer>(std::move(base), std::move(exhaustion_controller_map), std::move(executor), std::move(mtx));
        }

        static auto spawn_mailbox_radixx(std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                                         std::unique_ptr<InBoundContainerInterface> inbound_container,
                                         size_t num_inbound_worker) -> std::unique_ptr<RadixMailboxInterface>{
            
            const size_t MIN_INBOUND_WORKER = 1u;
            const size_t MAX_INBOUND_WORKER = 1024u; 

            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (inbound_container == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(num_inbound_worker, MIN_INBOUND_WORKER, MAX_INBOUND_WORKER) != num_inbound_worker){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec{};
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base_sp = std::move(base);
            std::shared_ptr<InBoundContainerInterface> inbound_container_sp = std::move(inbound_container);

            //TODO: this is buggy - this will be DL if exception raises - not to worry about this now
            for (size_t i = 0u; i < num_inbound_worker; ++i){
                auto worker = std::make_unique<InBoundWorker>(base_sp, inbound_container_sp);
                auto handle = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker)));
                daemon_vec.push_back(std::move(handle));
            }

            return std::make_unique<RadixMailBox>(std::move(daemon_vec), base_sp, inbound_container_sp);
        }
    };

    struct Config{
        dg::unordered_map<radix_t, size_t> exhaustion_capacity_map;
        std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
        std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> retry_device;
        size_t num_inbound_worker;
    };

    auto spawn(Config config) -> std::unique_ptr<RadixMailboxInterface>{

        auto inbound_container = Factory::spawn_exhaustion_controlled_inbound_container(config.exhaustion_capacity_map, config.retry_device);
        auto rs = Factory::spawn_mailbox_radixx(std::move(config.base), std::move(inbound_container), config.num_inbound_worker); 

        return rs;
    }
};

/*
namespace dg::network_kernel_mailbox_impl1_heartbeatx{

    using radix_t = network_kernel_mailbox_impl1_radixx::radix_t; 

    struct ObserverInterface{
        virtual ~ObserverInterface() noexcept = default;
        virtual void notify() noexcept = 0; //this has to be an exitable-in-all-scenerios-invoke - to enforce this - return a std::atomic<bool>& for notifier to set - it's the observer's worker responsibility to observe the std::atomic<bool>& intervally  
    };

    struct HeartBeatMonitorInterface{
        virtual ~HeartBeatMonitorInterface() noexcept = default;
        virtual void recv_signal(const Address&) noexcept = 0;
        virtual bool check() noexcept = 0;
    };

    class HeartBeatMonitor: public virtual HeartBeatMonitorInterface{

        private:

            dg::unordered_map<Address, std::chrono::nanoseconds> address_ts_dict;
            std::chrono::nanoseconds error_threshold;
            std::chrono::nanoseconds termination_threshold;
            std::unique_ptr<std::mutex> mtx;

        public:

            HeartBeatMonitor(dg::unordered_map<Address, std::chrono::nanoseconds> address_ts_dict,
                             std::chrono::nanoseconds error_threshold,
                             std::chrono::nanoseconds termination_threshold,
                             std::unique_ptr<std::mutex> mtx) noexcept: address_ts_dict(std::move(address_ts_dict)),
                                                                        error_threshold(std::move(error_threshold)),
                                                                        termination_threshold(std::move(termination_threshold)),
                                                                        mtx(std::move(mtx)){}

            void recv_signal(const Address& addr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto ptr = this->address_ts_dict.find(addr);

                if (ptr == this->address_ts_dict.end()){
                    auto err_msg = this->make_foreign_heartbeat_error_msg(addr);
                    dg::network_log::error_fast(err_msg.c_str());
                    return;
                }

                ptr->second = dg::network_genult::unix_timestamp();
            }

            bool check() noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                std::chrono::nanoseconds now = dg::network_genult::unix_timestamp(); 
                bool status = true; 

                for (const auto& pair: this->address_ts_dict){
                    if (dg::network_genult::timelapsed(pair.second, now) > this->error_threshold){
                        auto err_msg = this->make_missing_heartbeat_error_msg(pair.first); 
                        dg::network_log::error_fast(err_msg.c_str());
                        status = false;
                    }

                    if (dg::network_genult::timelapsed(pair.second, now) > this->termination_threshold){
                        auto err_msg = this->make_missing_heartbeat_error_msg(pair.first);
                        dg::network_log::critical(err_msg.c_str());
                        std::abort();
                    }
                }

                return status;
            }
        
        private:

            auto make_missing_heartbeat_error_msg(const Address& addr) const noexcept -> dg::string{ //global memory pool - better to be noexcept here

                const char * fmt = "[NETWORKSTACK_HEARTBEAT] heartbeat not detected from {}:{}"; //ip-resolve is done externally - via log_reading - virtual ip is required to spawn proxy (if a node is not responding)
                return std::format(fmt, addr.ip, size_t{addr.port});
            }

            auto make_foreign_heartbeat_error_msg(const Address& addr) const noexcept -> dg::string{

                const char * fmt = "[NETWORKSTACK_HEARTBEAT] foreign heartbeat from {}:{}";
                return std::format(fmt, addr.ip, size_t{addr.port});
            }
    };

    class HeartBeatBroadcaster: public virtual dg::network_concurrency::WorkerInterface{

        private:

            dg::vector<Address> addr_table;
            Address host_addr;
            std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox;
            radix_t heartbeat_channel; 

        public:

            HeartBeatBroadcaster(dg::vector<Address> addr_table,
                                 Address host_addr,
                                 std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox,
                                 radix_t heartbeat_channel) noexcept addr_table(std::move(addr_table)),
                                                                     host_addr(std::move(host_addr)),
                                                                     mailbox(std::move(mailbox)),
                                                                     heartbeat_channel(std::move(heartbeat_channel)){}

            bool run_one_epoch() noexcept{
                
                size_t host_addr_sz = dg::network_compact_serializer::size(this->host_addr);
                dg::string serialized_host_addr(host_addr_sz);
                dg::network_compact_serializer::serialize_into(serialized_host_addr.data(), this->host_addr); 

                for (const auto& addr: this->addr_table){
                     this->mailbox->send(addr, serialized_host_addr, this->heartbeat_channel);
                }

                return true;
            }
    };

    class HeartBeatReceiver: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::unique_ptr<HeartBeatMonitorInterface> heartbeat_monitor;
            std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox;
            std::shared_ptr<ObserverInterface> observer;
            radix_t heartbeat_channel; 

        public:

            HeartBeatReceiver(std::unique_ptr<HeartBeatMonitorInterface> heartbeat_monitor,
                              std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox,
                              std::shared_ptr<ObserverInterface> observer,
                              radix_t heartbeat_channel) noexcept: heartbeat_monitor(std::move(heartbeat_monitor)),
                                                                   mailbox(std::move(mailbox)),
                                                                   observer(std::move(observer)),
                                                                   heartbeat_channel(std::move(heartbeat_channel)){}
            
            bool run_one_epoch() noexcept{

                if (!this->heartbeat_monitor->check()){
                    this->observer->notify();
                }

                std::optional<dg::string> buf = this->mailbox->recv(this->heartbeat_channel);
                
                if (!static_cast<bool>(buf)){
                    return false;
                }

                Address heartbeat_addr{};
                dg::network_compact_serializer::deserialize_into(hearbeat_addr, buf->data()); 
                this->heartbeat_monitor->recv_signal(heartbeat_addr);

                return true;
            }
    };

    class MailBox: public virtual network_kernel_mailbox_impl1_radixx::MailBoxInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
            std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox;

        public:

            MailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemons,
                    std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox) noexcept: daemons(std::move(daemons)),
                                                                                                                 mailbox(std::move(mailbox)){}


            void send(Address addr, dg::string buf, radix_t radix) noexcept{
                
                this->mailbox->send(std::move(addr), std::move(buf), radix);
            }

            auto recv(radix_t radix) noexcept -> std::optional<dg::string>{

                return this->mailbox->recv(radix);
            }
    };
}

namespace dg::network_kernel_mailbox_impl1_concurrentx{


    using radix_t = dg::network_kernel_mailbox_impl1_radixx::radix_t; 

    template <size_t CONCURRENCY_SZ>
    class ConcurrentMailBox: public virtual dg::network_kernel_mailbox_impl1_radixx::MailBoxInterface{

        private:

            dg::vector<std::unique_ptr<dg::network_kernel_mailbox_impl1_radixx::MailBoxInterface>> mailbox_vec;
        
        public:

            ConcurrentMailBox(dg::vector<std::unique_ptr<dg::network_kernel_mailbox_impl1_radixx::MailBoxInterface>> mailbox_vec,
                              std::integral_constant<size_t, CONCURRENCY_SZ>) noexcept: mailbox_vec(std::move(mailbox_vec)){}

            void send(Address addr, dg::string buf, radix_t radix) noexcept{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                this->mailbox_vec[idx]->send(std::move(addr), std::move(buf), radix);
            }

            auto recv(radix_t radix) noexcept -> std::optional<std::network_std_container::string>{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                return this->mailbox_vec[idx]->recv(radix);
            }
    };
}
*/

#endif