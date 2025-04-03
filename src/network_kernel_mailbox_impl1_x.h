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
    //yet it is said that we can decrease the chance of no-fail allocations -> 0 by controlling the upstream allocation techniques, not here return dg::string like it is implementation-specific

    //we have come up with two types of std_containers, the std_container that cannot fail because of allocations, and the std_container that can fail because of allocations
    //std_container that cannot fail because of allocations are for preallocated, internal usage of std::unordered_set<> std::unordered_map, std::deque, etc.
    //std_container that can fail because of allocations (we hardly found the reason to fail allocations, yet it's best practice to consider allocations as normal action operations)

    //we are to radix the std_containers

    using Address = dg::network_kernel_mailbox_impl1::model::Address; 

    static inline constexpr size_t MAX_STREAM_SIZE  = size_t{1} << 25;
    static inline constexpr size_t MAX_SEGMENT_SIZE = size_t{1} << 10;

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

    static auto serialize_packet_segment(PacketSegment segment) noexcept -> std::expected<dg::string, exception_t>{

        // constexpr size_t HEADER_SZ  = dg::network_trivial_serializer::size(std::make_tuple(GlobalIdentifier{}, uint64_t{}, uint64_t{}));
        // std::array<char, HEADER_SZ> buf{};
        // auto rs = std::move(segment.buf);
        // dg::network_trivial_serializer::serialize_into(buf.data(), std::make_tuple(segment.id, segment.segment_idx, segment.segment_sz));
        // std::copy(buf.begin(), buf.end(), std::back_inserter(rs));

        // return rs;
    }

    static auto deserialize_packet_segment(dg::string buf) noexcept -> std::expected<PacketSegment, exception_t>{

        // constexpr size_t HEADER_SZ  = dg::network_trivial_serializer::size(std::make_tuple(GlobalIdentifier{}, uint64_t{}, uint64_t{}));
        // auto [lhs_str, rhs_str]     = stdx::backsplit_str(std::move(buf), HEADER_SZ);
        // auto rs                     = PacketSegment{};
        // auto data_tup               = std::make_tuple(GlobalIdentifier{}, uint64_t{}, uint64_t{}); 

        // dg::network_trivial_serializer::deserialize_into(data_tup, rhs_str.data());
        // std::tie(rs.id, rs.segment_idx, rs.segment_sz) = data_tup;

        // rs.buf = std::move(lhs_str);

        // return rs;
    }

    struct AssembledPacket{
        dg::vector<PacketSegment> data;
        size_t collected_segment_sz;
        size_t total_segment_sz;
    };

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
        virtual void get_expired(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    struct PacketAssemblerInterface{
        virtual ~PacketAssemblerInterface() noexcept = default;
        virtual void assemble(std::move_iterator<PacketSegment *> segment_arr, size_t sz, std::expected<AssembledPacket, exception_t> * assembled_arr) noexcept = 0;        
        virtual void destroy(GlobalIdentifier * id, size_t sz) noexcept = 0;
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

            auto get_expired(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

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
            const size_t zero_get_expired_bounce_sz;
            const size_t max_tick_per_load; 

        public:

            RandomHashDistributedEntranceController(std::unique_ptr<std::unique_ptr<EntranceControllerInterface>[]> base_arr,
                                                    size_t keyvalue_feed_cap,
                                                    size_t pow2_base_arr_sz,
                                                    size_t zero_get_expired_bounce_sz,
                                                    size_t max_tick_per_load) noexcept: base_arr(std::move(base_arr)),
                                                                                        keyvalue_feed_cap(keyvalue_feed_cap),
                                                                                        pow2_base_arr_sz(pow2_base_arr_sz),
                                                                                        zero_get_expired_bounce_sz(zero_get_expired_bounce_sz),
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

            void get_expired(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

                sz = 0u;

                for (size_t i = 0u; i < this->zero_get_expired_bounce_sz; ++i){
                    size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                    size_t idx          = random_value & (this->pow2_base_arr_sz - 1u);
                    this->base_arr[idx]->get_expired(output_arr, sz, cap);
                    
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

            }

            void get_expired(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

                this->base->get_expired(output_arr, sz, cap);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    class PacketAssembler: public virtual PacketAssemblerInterface{

        private:

            dg::unordered_map<GlobalIdentifier, AssembledPacket> packet_map;
            size_t packet_map_cap;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            PacketAssembler(dg::unordered_map<GlobalIdentifier, AssembledPacket> packet_map,
                            size_t packet_map_cap, 
                            std::unique_ptr<std::mutex> mtx,
                            stdx::hdi_container<size_t> consume_sz_per_load) noexcept: packet_map(std::move(packet_map)),
                                                                                       packet_map_cap(packet_map_cap),
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
                        if (this->packet_map.size() == this->packet_map_cap){
                            assembled_arr[i] = std::unexpected(dg::network_exception::QUEUE_FULL);
                            continue;
                        }

                        std::expected<AssembledPacket, exception_t> waiting_pkt = this->make_empty_assembled_packet(segment_arr_base[i].segment_sz);

                        if (!waiting_pkt.has_value()){
                            assembled_arr[i] = std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                            continue;
                        } 

                        auto [emplace_ptr, status] = this->packet_map.try_emplace(segment_arr_base[i].id, std::move(waiting_pkt.value()));
                        dg::network_exception_handler::dg_assert(status);
                        map_ptr = emplace_ptr;
                    }

                    map_ptr->second.data[segment_arr_base[i].segment_idx]   = std::move(segment_arr_base[i]);
                    map_ptr->second.collected_segment_sz                    += 1;

                    if (map_ptr->second.collected_segment_sz == map_ptr->second.total_segment_sz){
                        assembled_arr[i] = std::move(map_ptr->second);
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
                    this->packet_map.erase(id_arr[i]);
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

                return AssembledPacket{std::move(vec), collected, total};
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

                std::fill(assembled_arr, std::next(assembled_arr, sz), std::expected<AssembledPacket, exception_t>(std::unexpected(dg::network_exception::SOCKET_STREAM_QUEUING)));

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
    };

    class ExhaustionControlledPacketAssembler: public virtual PacketAssemblerInterface{

        private:

            std::unique_ptr<PacketAssemblerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            dg::unordered_map<GlobalIdentifier, size_t> counter_map;
            size_t capacity;
            size_t size;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledPacketAssembler(std::unique_ptr<PacketAssemblerInterface> base,
                                                std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                dg::unordered_map<GlobalIdentifier, size_t> counter_map,
                                                size_t capacity,
                                                size_t size,
                                                std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                           executor(std::move(executor)),
                                                                                           counter_map(std::move(counter_map)),
                                                                                           capacity(capacity),
                                                                                           size(size),
                                                                                           mtx(std::move(mtx)){}
            
            auto assemble(PacketSegment segment) noexcept -> std::optional<AssembledPacket>{
                
                std::optional<AssembledPacket> rs{};
                auto task = [&]() noexcept{
                    return this->internal_assemble(segment, rs);
                };
                auto virt_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(task);
                this->executor->exec(virt_task);

                return rs; 
            }

            void destroy(const GlobalIdentifier& id) noexcept{

                this->internal_destroy(id);
            }
        
        private:

            auto internal_assemble(PacketSegment& segment, std::optional<AssembledPacket>& rs) noexcept -> bool{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                
                if (this->size == this->capacity){
                    return false;
                }

                auto map_ptr = this->counter_map.find(segment.id);

                if (map_ptr == this->counter_map.end()){
                    auto [emplace_ptr, status] = this->counter_map.emplace(std::make_pair(segment.id, 1u));
                    dg::network_exception_handler::dg_assert(status);
                    map_ptr = emplace_ptr;
                } else{
                    map_ptr->second += 1;
                }

                this->size += 1;
                rs = this->base->assemble(std::move(segment));

                if (rs.has_value()){
                    this->size -= map_ptr->second;
                    this->counter_map.erase(map_ptr);
                }

                return true;
            }

            void internal_destroy(const GlobalIdentifier& id) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr    = this->counter_map.find(id);

                if (map_ptr != this->counter_map.end()){
                    this->size -= map_ptr->second;
                    this->counter_map.erase(map_ptr);
                }

                this->base->destroy(id);
            }
    };

    class InBoundContainer: public virtual InBoundContainerInterface{

        private:

            dg::deque<dg::string> vec;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            InBoundContainer(dg::deque<dg::string> vec,
                             std::unique_ptr<std::mutex> mtx) noexcept: vec(std::move(vec)),
                                                                        mtx(std::move(mtx)){}
            
            void push(dg::string buf) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->vec.push_back(std::move(buf));
            }

            auto pop() noexcept -> std::optional<dg::string>{
                
                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                
                if (this->vec.empty()){
                    return std::nullopt;
                }

                auto rs = std::move(this->vec.front());
                this->vec.pop_front();

                return rs;
            }
    };

    class DistributedInBoundContainer: public virtual InBoundContainerInterface{

    };

    class ExhaustionControlledInBoundContainer: public virtual InBoundContainerInterface{

        private:

            std::unique_ptr<InBoundContainerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            size_t capacity;
            size_t size; 
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledInBoundContainer(std::unique_ptr<InBoundContainerInterface> base,
                                                 std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                 size_t capacity,
                                                 size_t size,
                                                 std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                            executor(std::move(executor)),
                                                                                            capacity(capacity),
                                                                                            size(size),
                                                                                            mtx(std::move(mtx)){}

            void push(dg::string data) noexcept{

                auto task = [&]() noexcept{
                    return this->internal_push(data);
                };
                auto virt_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(task);
                this->executor->exec(virt_task);
            }

            auto pop() noexcept -> std::optional<dg::string>{

                return this->internal_pop();
            }
        
        private:

            auto internal_push(dg::string& data) noexcept -> bool{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (this->size == this->capacity){
                    return false;
                }

                this->base->push(std::move(data));
                this->size += 1;

                return true;
            }

            auto internal_pop() noexcept -> std::optional<dg::string>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                std::optional<dg::string> rs = this->base->pop();

                if (rs.has_value()){
                    this->size -= 1;
                }

                return rs;
            }
    };

    class ReactingInBoundContainer: public virtual InBoundContainerInterface{

    };

    class ExpiryWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<PacketAssemblerInterface> packet_assembler;
            std::shared_ptr<EntranceControllerInterface> entrance_controller;
        
        public:

            ExpiryWorker(std::shared_ptr<PacketAssemblerInterface> packet_assembler,
                         std::shared_ptr<EntranceControllerInterface> entrance_controller) noexcept: packet_assembler(std::move(packet_assembler)),
                                                                                                     entrance_controller(std::move(entrance_controller)){}
            
            bool run_one_epoch() noexcept{

                // dg::vector<GlobalIdentifier> expired_id_vec = this->entrance_controller->get_expired();

                // if (expired_id_vec.empty()){
                //     return false;
                // }

                // for (const auto& expired_id: expired_id_vec){
                //     this->packet_assembler->destroy(expired_id);
                // }

                return true;
            }
    };

    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<PacketAssemblerInterface> packet_assembler;
            std::shared_ptr<InBoundContainerInterface> inbound_container;
            std::shared_ptr<EntranceControllerInterface> entrance_controller;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;

        public:

            InBoundWorker(std::shared_ptr<PacketAssemblerInterface> packet_assembler,
                          std::shared_ptr<InBoundContainerInterface> inbound_container,
                          std::shared_ptr<EntranceControllerInterface> entrance_controller,
                          std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base) noexcept: packet_assembler(std::move(packet_assembler)),
                                                                                                                    inbound_container(std::move(inbound_container)),
                                                                                                                    entrance_controller(std::move(entrance_controller)),
                                                                                                                    base(std::move(base)){}

            bool run_one_epoch() noexcept{

                std::optional<dg::string> data = this->base->recv();

                if (!data.has_value()){
                    return false;
                }

                PacketSegment pkt   = deserialize_packet_segment(std::move(data.value()));
                GlobalIdentifier id = pkt.id;
                std::optional<AssembledPacket> assembled_packet = this->packet_assembler->assemble(std::move(pkt));

                std::atomic_signal_fence(std::memory_order_seq_cst);
                this->entrance_controller->tick(id); //important that tick is here - otherwise leak
                std::atomic_signal_fence(std::memory_order_seq_cst);

                if (assembled_packet.has_value()){
                    this->inbound_container->push(this->to_bstream(std::move(assembled_packet.value())));
                }

                return true;
            }
        
        private:

            auto to_bstream(AssembledPacket assembled_packet) noexcept -> dg::string{

                dg::network_exception_handler::dg_assert(assembled_packet.total_segment_sz != 0u);
                dg::string rs = std::move(assembled_packet.data[0].buf); 
                size_t extra_sz = 0u; 

                for (size_t i = 1u; i < assembled_packet.data.size(); ++i){
                    extra_sz += assembled_packet.data[i].buf.size();
                }

                size_t old_sz   = rs.size();
                size_t new_sz   = old_sz + extra_sz;
                rs.resize(new_sz);
                char * last     = rs.data() + old_sz;

                for (size_t i = 1u; i < assembled_packet.data.size(); ++i){
                    last = std::copy(assembled_packet.data[i].buf.begin(), assembled_packet.data[i].buf.end(), last);
                }

                return rs;
            }
    };

    class MailBox: public virtual dg::network_kernel_mailbox_impl1::core::MailboxInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
            std::unique_ptr<PacketizerInterface> packetizer;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
            std::shared_ptr<InBoundContainerInterface> inbound_container;

        public:

            MailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemons,
                    std::unique_ptr<PacketizerInterface> packetizer,
                    std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                    std::shared_ptr<InBoundContainerInterface> inbound_container) noexcept: daemons(std::move(daemons)),
                                                                                            packetizer(std::move(packetizer)),
                                                                                            base(std::move(base)),
                                                                                            inbound_container(std::move(inbound_container)){}

            void send(Address addr, dg::string arg) noexcept{
                
                //this is an abortable error - because it's no_response | no_transmit deterministic - implies program errors

                if (arg.size() > MAX_STREAM_SIZE){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                    std::abort();
                }

                dg::vector<PacketSegment> segment_vec = this->packetizer->packetize(arg);

                for (PacketSegment& segment: segment_vec){
                    this->base->send(addr, serialize_packet_segment(std::move(segment)));
                }
            }

            auto recv() noexcept -> std::optional<dg::string>{

                return this->inbound_container->pop();
            }
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