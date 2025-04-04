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
        virtual auto get_count() noexcept -> size_t = 0;
        virtual auto get_count_since() noexcept -> std::chrono::time_point<std::chrono::steady_clock> = 0;  
        virtual void reset() noexcept = 0;
    };

    class MtxMeter: public virtual MeterInterface{

        private:

            std::atomic<size_t> count;
            std::atomic<std::chrono::time_point<std::chrono::steady_clock>> since;

        public:

            MtxMeter() = default;

            void tick(size_t incoming_sz) noexcept{
                
                this->count.fetch_add(incoming_sz, std::memory_order_relaxed);
            }

            auto get_count() noexcept -> size_t{

                return this->count.load(std::memory_order_relaxed);
            }

            auto get_count_since() noexcept -> std::chrono::time_point<std::chrono::steady_clock>{

                return this->since.load(std::memory_order_relaxed);
            }

            void reset() noexcept{

                stdx::seq_cst_guard seqcst_tx;

                this->count.exchange(0u, std::memory_order_relaxed);
                this->since.exchange(std::chrono::steady_clock::now(), std::memory_order_relaxed);
            }
    };

    class RandomDistributedMtxMeter: public virtual MeterInterface{

        private:

            std::unique_ptr<std::unique_ptr<MeterInterface>[]> base_arr;
            size_t pow2_base_arr_sz;
        
        public:

            RandomDistributedMtxMeter(std::unique_ptr<std::unique_ptr<MeterInterface>[]> base_arr,
                                      size_t pow2_base_arr_sz) noexcept: base_arr(std::move(base_arr)),
                                                                         pow2_base_arr_sz(pow2_base_arr_sz){}

            void tick(size_t incoming_sz) noexcept{

                size_t random_value = dg::network_randomizer::randomize_int<size_t>();
                size_t random_idx   = random_value & (this->pow2_base_arr_sz - 1u);

                this->base_arr[random_idx]->tick(incoming_sz);
            }

            auto get_count() noexcept -> size_t{

                size_t total_sz = 0u;

                for (size_t i = 0u; i < this->pow2_base_arr_sz; ++i){
                    total_sz += this->base_arr[i]->get_count();
                }

                return total_sz;
            }

            auto get_count_since() noexcept -> std::chrono::time_point<std::chrono::steady_clock>{

                std::chrono::time_point<std::chrono::steady_clock> since = this->base_arr[0]->get_count_since();

                for (size_t i = 1u; i < this->pow2_base_arr_sz; ++i){
                    since = std::min(since, this->base_arr[i]->get_count_since());
                }

                return since;
            }

            void reset() noexcept{

                for (size_t i = 0u; i < this->pow2_base_arr_sz; ++i){
                    this->base_arr[i]->reset();
                }
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
 
                dg::string send_msg = this->make_send_meter_msg(this->send_meter->get_count(), this->send_meter->get_count_since());
                dg::string recv_msg = this->make_recv_meter_msg(this->recv_meter->get_count(), this->recv_meter->get_count_since());

                dg::network_log::journal_fast(send_msg.c_str());
                dg::network_log::journal_fast(recv_msg.c_str());

                this->send_meter->reset();
                this->recv_meter->reset();

                return true;
            }
        
        private:

            auto make_send_meter_msg(size_t bsz, std::chrono::time_point<std::chrono::steady_clock> dur) noexcept -> dg::string{

                std::chrono::seconds dur_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - dur);
                size_t tick_sz = dur_in_seconds.count();

                if (tick_sz == 0u){
                    return std::format("[METER_REPORT] low meter precision resolution (device_id: {}, part: send_meter)", this->device_id);
                } 

                size_t bsz_per_s = bsz / tick_sz;
                return std::format("[METER_REPORT] {} bytes/s sent to {}", bsz_per_s, this->device_id);
            }

            auto make_recv_meter_msg(size_t bsz, std::chrono::time_point<std::chrono::steady_clock> dur) noexcept -> dg::string{

                std::chrono::seconds dur_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - dur);
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

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
            std::shared_ptr<MeterInterface> send_meter;
            std::shared_ptr<MeterInterface> recv_meter;
        
        public:

            MeteredMailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec, 
                           std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                           std::shared_ptr<MeterInterface> send_meter,
                           std::shared_ptr<MeterInterface> recv_meter): daemon_vec(std::move(daemon_vec)),
                                                                        base(std::move(base)),
                                                                        send_meter(std::move(send_meter)),
                                                                        recv_meter(std::move(recv_meter)){}

            void send(std::move_iterator<MailBoxArgument *> data_arr, size_t sz, exception_t * exception_arr) noexcept{
                
                MailBoxArgument * base_data_arr = data_arr.base();
                size_t total_sz = 0u;

                for (size_t i = 0u; i < sz; ++i){
                    total_sz += base_data_arr[i].content.size();
                }

                this->base->send(std::make_move_iterator(base_data_arr), sz, exception_arr);

                for (size_t i = 0u; i < sz; ++i){
                    if (dg::network_exception::is_failed(exception_arr[i])){
                        total_sz -= base_data_arr[i].content.size();
                    }
                }

                this->send_meter->tick(total_sz);
            }

            void recv(dg::string * output_arr, size_t& output_arr_sz, size_t output_arr_cap) noexcept{

                this->base->recv(output_arr, output_arr_sz, output_arr_cap);
                size_t total_sz = 0u;

                for (size_t i = 0u; i < output_arr_sz; ++i){
                    total_sz += output_arr[i].size();
                }

                this->recv_meter->tick(total_sz);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };
}

namespace dg::network_kernel_mailbox_impl1_flash_streamx{

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

    struct AssembledPacket{
        dg::vector<PacketSegment> data;
        size_t collected_segment_sz;
        size_t total_segment_sz;
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

        std::expected<dg::string, exception_t> rs = dg::network_exception::cstyle_initialize<dg::string>(total_bsz, 0);

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

    struct InBoundGateInterface{
        virtual ~InBoundGateInterface() noexcept = default;
        virtual void thru(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
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

    class TemporalAbsoluteTimeoutInBoundGate: public virtual InBoundGateInterface{

        private:

            data_structure::temporal_finite_unordered_map<GlobalIdentifier, std::chrono::time_point<std::chrono::steady_clock>> abstimeout_map;
            const std::chrono::nanoseconds abs_timeout_dur;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> thru_sz_per_load;

        public:

            TemporalAbsoluteTimeoutInBoundGate(datastructure::temporal_finite_unordered_map<GlobalIdentifier, std::chrono::time_point<std::chrono::utc_clcck>> abstimeout_map,
                                               std::chrono::nanoseconds abs_timeout_dur,
                                               std::unique_ptr<std::mutex> mtx,
                                               stdx::hdi_container<size_t> thru_sz_per_load) noexcept: abstimeout_map(std::move(abstimeout_map)),
                                                                                                       abs_timeout_dur(abs_timeout_dur),
                                                                                                       mtx(std::move(mtx)),
                                                                                                       thru_sz_per_load(std::move(thru_sz_per_load)){}

            void thru(GlobalIdentifier * global_id_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                std::chrono::time_point<std::chrono::steady_clock> now              = std::chrono::steady_clock::now();
                std::chrono::time_point<std::chrono::steady_clock> timeout_value    = now + this->abs_timeout_dur;

                for (size_t i = 0u; i < sz; ++i){
                    auto map_ptr = this->abstimeout_map.find(global_id_arr[i]);

                    if (map_ptr == this->abstimeout_map.end()){
                        auto [emplace_ptr, status] = this->abstimeout_map.try_emplace(global_id_arr[i], timeout_value);
                        dg::network_exception_handler::dg_assert(status);
                        exception_arr[i] = dg::network_exception::SUCCESS; 
                        continue;
                    }

                    if (map_ptr->second < now){
                        exception_arr[i] = dg::network_exception::SOCKET_STREAM_TIMEOUT;
                        continue;                        
                    }

                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->thru_sz_per_load.value;
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
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
        GlobalIdentifier key;
        __uint128_t entry_id;
    };

    class EntranceController: public virtual EntranceControllerInterface{

        private:

            dg::pow2_cyclic_queue<EntranceEntry> entrance_entry_pq; //no exhausted container
            const size_t entrance_entry_pq_cap; 
            dg::unordered_unstable_map<GlobalIdentifier, size_t> key_id_map; //no exhausted container
            const size_t key_id_map_cap;
            __uint128_t id_ticker;
            const std::chrono::nanoseconds expiry_period;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> tick_sz_per_load;

        public:

            EntranceController(dg::pow2_cyclic_queue<EntranceEntry> entrance_entry_pq,
                               size_t entrance_entry_pq_cap,
                               dg::unordered_unstable_map<GlobalIdentifier, size_t> key_id_map,
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

                auto now = std::chrono::steady_clock::now();

                for (size_t i = 0u; i < sz; ++i){
                    if (this->entrance_entry_pq.size() == this->entrace_entry_pq_cap){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    if (this->key_id_map.size() == this->key_id_map_cap){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    auto entry      = EntranceEntry{};
                    entry.timestamp = now;
                    entry.key       = global_id_arr[i];
                    entry.entry_id  = this->id_ticker++;

                    this->entrance_entry_pq.push_back(entry);
                    this->key_id_map[global_id_arr[i]] = entry.entry_id;
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            //assume finite sorted queue of entrance_entry_pq
            //assume key_id_map points to the lastest GlobalIdentifier guy in the entrance_entry_pq

            auto get_expired_id(GlobalIdentifier * output_arr, size_t& sz, size_t cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto bar_time   = std::chrono::steady_clock::now() - this->expiry_period;
                sz              = 0u;

                for (size_t i = 0u; i < cap; ++i){
                    if (this->entrace_entry_pq.empty()){
                        return;
                    }

                    if (this->entrance_entry_pq.front().timestamp > bar_time){
                        return; 
                    }

                    EntranceEntry entry = this->entrace_entry_pq.front();
                    this->entrace_entry_pq.pop_front();
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

            dg::unordered_unstable_map<GlobalIdentifier, AssembledPacket> packet_map;
            size_t packet_map_cap;
            size_t global_packet_segment_cap;
            size_t global_packet_segment_counter;
            size_t max_segment_sz_per_stream; 
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            PacketAssembler(dg::unordered_unstable_map<GlobalIdentifier, AssembledPacket> packet_map,
                            size_t packet_map_cap,
                            size_t global_packet_segment_cap,
                            size_t global_packet_segment_counter,
                            size_t max_segment_sz_per_stream,
                            std::unique_ptr<std::mutex> mtx,
                            stdx::hdi_container<size_t> consume_sz_per_load) noexcept: packet_map(std::move(packet_map)),
                                                                                       packet_map_cap(packet_map_cap),
                                                                                       global_packet_segment_cap(global_packet_segment_cap),
                                                                                       global_packet_segment_counter(global_packet_segment_counter),
                                                                                       max_segment_sz_per_stream(max_segment_sz_per_stream),
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
                PacketSegment * base_segment_arr = segment_arr.base();

                for (size_t i = 0u; i < sz; ++i){
                    if (base_segment_arr[i].segment_sz == 0u){
                        assembled_arr[i] = this->make_empty_assembled_packet(0u);
                        continue;
                    }

                    auto map_ptr = this->packet_map.find(base_segment_arr[i].id);

                    if (map_ptr == this->packet_map.end()){
                        if (base_segment_arr[i].segment_sz > this->max_segment_sz_per_stream){
                            assembled_arr[i] = std::unexpected(dg::network_exception::SOCKET_STREAM_BAD_SEGMENT_SIZE);
                            continue;
                        }

                        if (this->packet_map.size() == this->packet_map_cap){
                            assembled_arr[i] = std::unexpected(dg::network_exception::QUEUE_FULL);
                            continue;
                        }

                        if (this->global_packet_segment_counter + base_segment_arr[i].segment_sz > this->global_packet_segment_cap){
                            assembled_arr[i] = std::unexpected(dg::network_exception::QUEUE_FULL);
                            continue;
                        }

                        std::expected<AssembledPacket, exception_t> waiting_pkt = this->make_empty_assembled_packet(base_segment_arr[i].segment_sz);

                        if (!waiting_pkt.has_value()){
                            assembled_arr[i] = std::unexpected(waiting_pkt.error());
                            continue;
                        }

                        auto [emplace_ptr, status]          = this->packet_map.try_emplace(base_segment_arr[i].id, std::move(waiting_pkt.value()));
                        dg::network_exception_handler::dg_assert(status);
                        map_ptr                             = emplace_ptr;
                        this->global_packet_segment_counter += base_segment_arr[i].segment_sz; 
                    }

                    //this is fine, because it is object <dig_in> operator = (object&&), there wont be issues, yet we want to make this explicit

                    size_t segment_idx                      = base_segment_arr[i].segment_idx;
                    map_ptr->second.data[segment_idx]       = std::move(base_segment_arr[i]);
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
                    feed_arg.rs                     = std::next(assembled_arr, i); 

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
                    std::expected<AssembledPacket, exception_t> * last_retriable_ptr    = std::find_if(first_retriable_ptr, last_assembled_ptr, [&](const auto& err){return err != full_signal;});
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

    //OK
    class BufferFIFOContainer: public virtual InBoundContainerInterface{

        private:

            dg::pow2_cyclic_queue<dg::string> buffer_vec;
            size_t buffer_vec_capacity;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> consume_sz_per_load;

        public:

            BufferFIFOContainer(dg::pow2_cyclic_queue<dg::string> buffer_vec,
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
                this->buffer_vec.erase_front_range(sz);
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
            std::shared_ptr<InBoundGateInterface> inbound_gate;
            std::shared_ptr<InBoundContainerInterface> inbound_container;
            std::shared_ptr<EntranceControllerInterface> entrance_controller;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base;
            size_t upstream_consume_sz;
            size_t busy_consume_sz;

        public:

            InBoundWorker(std::shared_ptr<PacketAssemblerInterface> packet_assembler,
                          std::shared_ptr<InBoundGateInterface> inbound_gate,
                          std::shared_ptr<InBoundContainerInterface> inbound_container,
                          std::shared_ptr<EntranceControllerInterface> entrance_controller,
                          std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base,
                          size_t upstream_consume_sz,
                          size_t busy_consume_sz) noexcept: packet_assembler(std::move(packet_assembler)),
                                                            inbound_gate(std::move(inbound_gate)),
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

                size_t trimmed_ps_feed_cap                  = std::min(std::min(DEFAULT_KEY_FEED_SIZE, consuming_sz), this->packet_assembler->max_consume_size());
                size_t ps_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&ps_feed_resolutor, trimmed_ps_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> ps_feeder_mem(ps_feeder_allocation_cost);
                auto ps_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&ps_feed_resolutor, trimmed_ps_feed_cap, ps_feeder_mem.get()));  

                auto gt_feed_resolutor                      = InternalInBoundGateFeedResolutor{};
                gt_feed_resolutor.inbound_gate              = this->inbound_gate.get();
                gt_feed_resolutor.ps_feeder                 = ps_feeder.get();                    
                
                size_t trimmed_gt_feed_cap                  = std::min(std::min(DEFAULT_KEY_FEED_SIZE, consuming_sz), this->inbound_gate->max_consume_size());
                size_t gt_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&gt_feed_resolutor, trimmed_gt_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> gt_feeder_mem(gt_feeder_allocation_cost);
                auto gt_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&gt_feed_resolutor, trimmed_gt_feed_cap, gt_feeder_mem.get()));

                for (size_t i = 0u; i < consuming_sz; ++i){
                    std::expected<PacketSegment, exception_t> pkt = deserialize_packet_segment(std::move(buf_arr[i]));

                    if (!pkt.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(pkt.error()));
                        continue;
                    }

                    dg::network_producer_consumer::delvrsrv_deliver(gt_feeder.get(), std::move(pkt.value()));
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

            struct InternalInBoundGateFeedResolutor: dg::network_producer_consumer::ConsumerInterface<PacketSegment>{

                InBoundGateInterface * inbound_gate;
                dg::network_producer_consumer::DeliveryHandle<PacketSegment> * ps_feeder;

                void push(std::move_iterator<PacketSegment *> data_arr, size_t sz) noexcept{

                    PacketSegment * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<GlobalIdentifier[]> global_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        global_id_arr[i] = base_data_arr[i].id;
                    }

                    this->inbound_gate->thru(global_id_arr.get(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                            continue;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(this->ps_feeder, std::move(base_data_arr[i]));
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
                auto feeder                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_mailbox_feed_sz, feeder_mem.get()));

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

        static auto get_packet_id_generator(Address factory_addr) -> std::unique_ptr<PacketIDGeneratorInterface>{

            uint64_t random_counter = dg::network_randomizer::randomize_int<uint64_t>();
            return std::make_unique<PacketIDGenerator>(std::move(factory_addr), random_counter);
        } 
        
        static auto get_temporal_absolute_timeout_inbound_gate(size_t map_capacity, 
                                                               std::chrono::nanoseconds abs_timeout_dur,
                                                               size_t max_consume_decay_factor = 2u){
            
            const size_t MIN_MAP_CAPACITY                       = size_t{1};
            const size_t MAX_MAP_CAPACITY                       = size_t{1} << 40;
            const std::chrono::nanoseconds MIN_ABS_TIMEOUT_DUR  = std::chrono::nanoseconds(1u);
            const std::chrono::nanoseconds MAX_ABS_TIMEOUT_DUR  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(1));

            if (std::clamp(map_capacity, MIN_MAP_CAPACITY, MAX_MAP_CAPACITY) != map_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(abs_timeout_dur, MIN_ABS_TIMEOUT_DUR, MAX_ABS_TIMEOUT_DUR) != abs_timeout_dur){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_max_consume_sz = map_capacity >> max_consume_decay_factor;
            size_t max_consume_sz           = std::max(size_t{1}, tentative_max_consume_sz); 

            return std::make_unique<TemporalAbsoluteTimeoutInBoundGate>(datastructure::temporal_finite_unordered_map<>(map_capacity),
                                                                        abs_timeout_dur,
                                                                        std::make_unique<std::mutex>(),
                                                                        stdx::hdi_container<size_t>{max_consume_sz});
        }

        static auto get_packetizer(Address factory_addr, size_t segment_byte_sz) -> std::unique_ptr<PacketizerInterface>{

            const size_t MIN_SEGMENT_BYTE_SZ    = size_t{1};
            const size_t MAX_SEGMENT_BYTE_SZ    = size_t{1} << 30;  

            if (std::clamp(segment_byte_sz, MIN_SEGMENT_BYTE_SZ, MAX_SEGMENT_BYTE_SZ) != segment_byte_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<Packetizer>(get_packet_id_generator(factory_addr),
                                                segment_byte_sz);
        }  

        static auto get_entrance_controller(size_t queue_cap,
                                            size_t unique_id_cap,
                                            std::chrono::nanoseconds expiry_period,
                                            size_t max_consume_decay_factor = 2u) -> std::unique_ptr<EntranceControllerInterface>{
            
            const size_t MIN_QUEUE_CAP                          = size_t{1};
            const size_t MAX_QUEUE_CAP                          = size_t{1} << 30;
            const size_t MIN_UNIQUE_ID_CAP                      = size_t{1};
            const size_t MAX_UNIQUE_ID_CAP                      = size_t{1} << 30;

            const std::chrono::nanoseconds MIN_EXPIRY_PERIOD    = std::chrono::nanoseconds(1);
            const std::chrono::nanoseconds MAX_EXPIRY_PERIOD    = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(1));

            if (std::clamp(queue_cap, MIN_QUEUE_CAP, MAX_QUEUE_CAP) != queue_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(unique_id_cap, MIN_UNIQUE_ID_CAP, MAX_UNIQUE_ID_CAP) != unique_id_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(expiry_period, MIN_EXPIRY_PERIOD, MAX_EXPIRY_PERIOD) != expiry_period){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t upqueue_cap              = stdx::ulog2(stdx::ceil2(queue_cap));
            size_t tentative_max_consume_sz = std::min(upqueue_cap, unique_id_cap) >> max_consume_decay_factor;
            size_t max_consume_sz           = std::max(size_t{1}, tentative_max_consume_sz);
            auto entrance_entry_pq          = dg::pow2_cyclic_queue<EntranceEntry>(upqueue_cap);
            auto key_id_map                 = dg::unordered_unstable_map<GlobalIdentifier, size_t>{};

            key_id_map.reserve(unique_id_cap);

            return std::make_unique<EntranceController>(std::move(entrance_entry_pq),
                                                        upqueue_cap,
                                                        std::move(key_id_map),
                                                        unique_id_cap,
                                                        __uint128_t{0u},
                                                        expiry_period,
                                                        std::make_unique<std::mutex>(),
                                                        stdx::hdi_container<size_t>{max_consume_sz});
        }

        static auto get_random_hash_distributed_entrance_controller(std::vector<std::unique_ptr<EntranceControllerInterface>> base_vec,
                                                                    size_t zero_bounce_sz       = 8u,
                                                                    size_t keyvalue_feed_cap    = DEFAULT_KEYVALUE_FEED_SIZE) -> std::unique_ptr<EntranceControllerInterface>{
            
            const size_t MIN_ZERO_BOUNCE_SZ     = size_t{1};
            const size_t MAX_ZERO_BOUNCE_SZ     = size_t{1} << 20;
            const size_t MIN_KEYVALUE_FEED_CAP  = size_t{1};
            const size_t MAX_KEYVALUE_FEED_CAP  = size_t{1} << 25;

            if (!stdx::is_pow2(base_vec.size())){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(zero_bounce_sz, MIN_ZERO_BOUNCE_SZ, MAX_ZERO_BOUNCE_SZ) != zero_bounce_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(keyvalue_feed_cap, MIN_KEYVALUE_FEED_CAP, MAX_KEYVALUE_FEED_CAP) != keyvalue_feed_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto base_arr           = std::make_unique<std::unique_ptr<EntranceControllerInterface>[]>(base_vec.size());
            size_t base_arr_sz      = base_vec.size();
            size_t max_consume_sz   = std::numeric_limits<size_t>::max(); 

            for (size_t i = 0u; i < base_arr_sz; ++i){
                base_arr[i]     = std::move(base_vec[i]);
                max_consume_sz  = std::min(base_arr[i]->max_consume_size());
            }

            size_t trimmed_feed_cap = std::min(max_consume_sz, keyvalue_feed_cap); 

            return std::make_unique<RandomHashDistributedEntranceController>(std::move(base_arr),
                                                                             trimmed_feed_cap,
                                                                             base_arr_sz,
                                                                             zero_bounce_sz,
                                                                             max_consume_sz);
        }

        static auto get_exhaustion_controlled_entrance_controller(std::unique_ptr<EntranceControllerInterface> base,
                                                                  std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                                  std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller) -> std::unique_ptr<EntranceControllerInterface>{
            
            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (exhaustion_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledEntranceController>(std::move(base),
                                                                            std::move(executor),
                                                                            std::move(exhaustion_controller));
        }

        static auto get_packet_assembler(size_t packet_map_cap,
                                         size_t global_packet_segment_cap,
                                         size_t max_segment_sz_per_stream,
                                         size_t max_consume_decay_factor = 2u) -> std::unique_ptr<PacketAssemblerInterface>{

            const size_t MIN_PACKET_MAP_CAP             = size_t{1};
            const size_t MAX_PACKET_MAP_CAP             = size_t{1} << 30;
            const size_t MIN_GLOBAL_PACKET_SEGMENT_CAP  = size_t{1};
            const size_t MAX_GLOBAL_PACKET_SEGMENT_CAP  = size_t{1} << 30;
            const size_t MIN_MAX_SEGMENT_SZ_PER_STREAM  = size_t{1};
            const size_t MAX_MAX_SEGMENT_SZ_PER_STREAM  = size_t{1} << 30; 

            if (std::clamp(packet_map_cap, MIN_PACKET_MAP_CAP, MAX_PACKET_MAP_CAP) != packet_map_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(global_packet_segment_cap, MIN_GLOBAL_PACKET_SEGMENT_CAP, MAX_GLOBAL_PACKET_SEGMENT_CAP) != global_packet_segment_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(max_segment_sz_per_stream, MIN_MAX_SEGMENT_SZ_PER_STREAM, MAX_MAX_SEGMENT_SZ_PER_STREAM) != max_segment_sz_per_stream){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_max_consume_sz = packet_map_cap >> max_consume_decay_factor;
            size_t max_consume_sz           = std::max(size_t{1}, tentative_max_consume_sz);
            
            auto packet_map                 = dg::unordered_unstable_map<GlobalIdentifier, AssembledPacket>();
            packet_map.reserve(packet_map_cap); 

            return std::make_unique<PacketAssembler>(std::move(packet_map),
                                                    packet_map_cap,
                                                    global_packet_segment_cap,
                                                    size_t{0u},
                                                    max_segment_sz_per_stream,
                                                    std::make_unique<std::mutex>(),
                                                    stdx::hdi_container<size_t>{max_consume_sz});
        }

        static auto get_random_hash_distributed_packet_assembler(std::vector<std::unique_ptr<PacketAssemblerInterface>> base_vec,
                                                                 size_t keyvalue_feed_cap = DEFAULT_KEYVALUE_FEED_SIZE){
            
            const size_t MIN_KEYVALUE_FEED_CAP  = size_t{1};
            const size_t MAX_KEYVALUE_FEED_CAP  = size_t{1} << 25;

            if (!stdx::is_pow2(base_vec.size())){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(keyvalue_feed_cap, MIN_KEYVALUE_FEED_CAP, MAX_KEYVALUE_FEED_CAP) != keyvalue_feed_cap){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t max_consume_sz   = std::numeric_limits<size_t>::max(); 
            auto base_arr           = std::make_unique<std::unique_ptr<PacketAssemblerInterface>[]>(base_vec.size());
            size_t base_arr_sz      = base_vec.size(); 

            for (auto& uptr: base_vec){
                if (uptr == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                max_consume_sz  = std::min(max_consume_sz, uptr->max_consume_size());
                base_arr[i]     = std::move(base_vec[i]);
            }

            size_t trimmed_feed_cap = std::min(keyvalue_feed_cap, max_consume_sz);

            return std::make_unique<RandomHashDistributedPacketAssembler>(std::move(base_arr),
                                                                          trimmed_feed_cap,
                                                                          base_arr_sz,
                                                                          max_consume_sz);
        }

        static auto get_exhaustion_controlled_packet_assembler(std::unique_ptr<PacketAssemblerInterface> base,
                                                               std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                               std::shared_ptr<ExhaustionControllerInterface> exhaustion_controller) -> std::unique_ptr<PacketAssemblerInterface>{
            
            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (exhaustion_controller == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ExhaustionControlledPacketAssembler>(std::move(base),
                                                                         std::move(executor),
                                                                         std::move(exhaustion_controller));

        }

        static auto get_buffer_fifo_container(size_t buffer_capacity,
                                              size_t consume_factor = 4u) -> std::unique_ptr<InBoundContainerInterface>{
            
            const size_t MIN_BUFFER_CAPACITY    = 1u;
            const size_t MAX_BUFFER_CAPACITY    = size_t{1} << 25;

            if (std::clamp(buffer_capacity, MIN_BUFFER_CAPACITY, MAX_BUFFER_CAPACITY) != buffer_capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t tentative_consume_sz         = buffer_capacity >> consume_factor;
            size_t consume_sz                   = std::max(tentative_consume_sz, size_t{1u});

            return std::make_unique<BufferFIFOContainer>(dg::pow2_cyclic_queue<dg::string>(stdx::ulog2(stdx::ceil2(buffer_capacity))),
                                                         buffer_capacity,
                                                         std::make_unique<std::mutex>(),
                                                         stdx::hdi_container<size_t>{consume_sz});
        }

        static auto get_exhaustion_controlled_buffer_container(std::unique_ptr<InBoundContainerInterface> base,
                                                               std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                               std::shared_ptr<packet_controller::ExhaustionControllerInterface> exhaustion_controller) -> std::unique_ptr<InBoundContainerInterface>{
            
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

        static auto get_reacting_buffer_container(std::unique_ptr<InBoundContainerInterface> base,
                                                  size_t reacting_threshold,
                                                  size_t concurrent_subscriber_cap,
                                                  std::chrono::nanoseconds wait_time) -> std::unique_ptr<InBoundContainerInterface>{

            const size_t MIN_REACTING_THRESHOLD             = size_t{1};
            const size_t MAX_REACTING_THRESHOLD             = size_t{1} << 30;
            const std::chrono::nanoseconds MIN_WAIT_TIME    = std::chrono::nanoseconds{1}; 
            const std::chrono::nanoseconds MAX_WAIT_TIME    = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::minutes{1});

            if (base == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(reacting_threshold, MIN_REACTING_THRESHOLD, MAX_REACTING_THRESHOLD) != reacting_threshold){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(wait_time, MIN_WAIT_TIME, MAX_WAIT_TIME) != wait_time){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<ReactingBufferContainer>(std::move(base),
                                                             get_complex_reactor(reacting_threshold, concurrent_subscriber_cap),
                                                             wait_time);

        }

        static auto get_randomhash_distributed_buffer_container(std::vector<std::unique_ptr<InBoundContainerInterface>> base_vec,
                                                                size_t zero_buffer_retry_sz = 8u) -> std::unique_ptr<InBoundContainerInterface>{

            const size_t MIN_BASE_VEC_SZ            = size_t{1};
            const size_t MAX_BASE_VEC_SZ            = size_t{1} << 20;
            const size_t MIN_ZERO_BUFFER_RETRY_SZ   = size_t{1};
            const size_t MAX_ZERO_BUFFER_RETRY_SZ   = size_t{1} << 20;

            if (std::clamp(base_vec.size(), MIN_BASE_VEC_SZ, MAX_BASE_VEC_SZ) != base_vec.size()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!stdx::is_pow2(base_vec.size())){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto base_vec_up        = std::make_unique<std::unique_ptr<InBoundContainerInterface>[]>(base_vec.size());
            size_t consumption_sz   = std::numeric_limits<size_t>::max(); 

            for (size_t i = 0u; i < base_vec.size(); ++i){
                if (base_vec[i] == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                consumption_sz  = std::min(consumption_sz, base_vec[i]->max_consume_size());
                base_vec_up[i]  = std::move(base_vec[i]);
            }

            if (std::clamp(zero_buffer_retry_sz, MIN_ZERO_BUFFER_RETRY_SZ, MAX_ZERO_BUFFER_RETRY_SZ) != zero_buffer_retry_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<HashDistributedBufferContainer>(std::move(base_vec_up),
                                                                    base_vec.size(),
                                                                    zero_buffer_retry_sz,
                                                                    consumption_sz);
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

        // std::unique_ptr<InBoundContainerInterface> inbound_container        = Factory::spawn_exhaustion_controlled_inbound_container(config.infretry_device, config.inbound_container_capacity);
        // std::unique_ptr<PacketAssemblerInterface> packet_assembler          = Factory::spawn_exhaustion_controlled_packet_assembler(config.infretry_device, config.packet_assembler_capacity);
        // std::unique_ptr<EntranceControllerInterface> entrance_controller    = Factory::spawn_entrance_controller(config.packet_expiry);
        // std::unique_ptr<PacketizerInterface> packetizer                     = Factory::spawn_packetizer(config.segment_byte_sz, config.factory_addr);
        
        // return Factory::spawn_mailbox_streamx(std::move(config.base), std::move(inbound_container), std::move(packet_assembler), 
        //                                       std::move(entrance_controller), std::move(packetizer), config.inbound_worker_count, 
        //                                       config.expiry_worker_count);
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