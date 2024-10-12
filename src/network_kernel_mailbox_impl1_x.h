#ifndef __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__
#define __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__

#include "network_kernel_mailbox_impl1.h"
#include "network_trivial_serializer.h"
#include "network_concurrency.h"
#include "network_std_container.h"
#include <chrono>
#include "network_log.h"
#include "network_concurrency.h"
#include "network_concurrency_x.h"

namespace dg::network_kernel_mailbox_impl1_meterlogx{

    struct MeterInterface{
        virtual ~MeterInterface() noexcept = default;
        virtual void tick(size_t) noexcept = 0;
        virtual auto get() noexcept -> std::pair<size_t, std::chrono::nanoseconds> = 0;
    };

    class MtxMeter: public virtual MeterInterface{
        
        private:

            size_t count; 
            std::chrono::nanoseconds unixstamp;
            std::unique_ptr<std::mutex> mtx;

        public:

            MtxMeter(size_t coumt,
                     std::chrono::nanoseconds unixstamp, 
                     st::unique_ptr<std::mutex> mtx) noexcept: count(count),
                                                               unixstamp(unixstamp),
                                                               mtx(std::move(mtx)){}
            
            void tick(size_t incoming_sz) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->count += incoming_sz;
            }

            auto get() noexcept -> std::pair<size_t, std::chrono::nanoseconds>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto curstamp   = static_cast<std::chrono::nanoseconds>(dg::network_genult::unix_timestamp());
                auto rs         = std::make_pair(this->count, dg::network_genult::timelapsed(curstamp, this->unixstamp));
                this->count     = 0u;
                this->unixstamp = curstamp;

                return rs;
            }
    }; 

    class MeterLogWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            dg::std_network_container::string device_id;
            std::shared_ptr<MeterInterface> send_meter;
            std::shared_ptr<MeterInterface> recv_meter;
        
        public:

            MeterLogWorker(dg::std_network_container::string device_id,
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

            auto make_send_meter_msg(size_t bsz, std::chrono::nanoseconds dur) noexcept -> dg::network_std_container::string{

                std::chrono::seconds dur_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(dur);
                size_t tick_sz = dur_in_seconds.count();

                if (tick_sz == 0u){
                    return std::format("[METER_REPORT] low meter precision resolution (device_id: {}, part: send_meter)", this->device_id);
                } 

                size_t bsz_per_s = bsz / tick_sz;
                return std::format("[METER_REPORT] {} bytes/s sent to {}", bsz_per_s, this->device_id);
            }

            auto make_recv_meter_msg(size_t bsz, std::chrono::nanoseconds dur) noexcept -> dg::network_std_container::string{

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

            std::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
            std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox;
            std::shared_ptr<MeterInterface> send_meter;
            std::shared_ptr<MeterInterface> recv_meter;
        
        public:

            MeteredMailBox(std::vector<dg::network_concurrency::daemon_raii_handle_t> daemons, 
                           std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox,
                           std::shared_ptr<MeterInterface> send_meter,
                           std::shared_ptr<MeterInterface> recv_meter): daemons(std::move(daemons)),
                                                                        mailbox(std::move(mailbox)),
                                                                        send_meter(std::move(send_meter)),
                                                                        recv_meter(std::move(recv_meter)){}
            
            void send(Address addr, dg::network_std_container::string buf) noexcept{

                this->send_meter->tick(buf.size());
                this->mailbox->send(std::move(addr), std::move(buf));
            }

            auto recv() -> std::optional<dg::network_std_container::string>{

                std::optional<dg::network_std_container::string> rs = this->mailbox->recv(); 

                if (!static_cast<bool>(rs)){
                    return std::nullopt;
                }

                this->recv_meter->tick(rs->size());
                return rs;
            }
    };
}

namespace dg::network_kernel_mailbox_impl1_streamx{
    
    static inline constexpr size_t MAX_STREAM_SIZE = size_t{1} << 25;

    struct GlobalIdentifier{
        Address addr;
        uint64_t local_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(addr, local_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(addr, local_id);
        }
    };

    struct PacketSegment{
        dg::network_std_container::string buf;
        GlobalIdentifier id;
        uint64_t segment_idx;
        uint64_t segment_sz;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(buf, id, segment_idx, segment_sz);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(buf, id, segment_idx, segment_sz);
        }
    };

    struct AssembledPacket{
        dg::network_std_container::vector<PacketSegment> data;
        size_t collected_segment_sz;
        size_t total_segment_sz;
    };

    struct PacketizerInterface{
        virtual ~PacketizerInterface() noexcept = default;
        virtual auto packetize(const dg::network_std_container::string&) noexcept -> dg::network_std_container::vector<PacketSegment> = 0; //optimizable - worth or not worth it
    };

    struct EntranceControllerInterface{
        virtual ~EntranceControllerInterface() noexcept = default;
        virtual void tick(const GlobalIdentifier&) noexcept = 0; 
        virtual auto get_expired() noexcept -> dg::network_std_container::vector<GlobalIdentifier> = 0;
    };

    struct PacketAssemblerInterface{
        virtual ~PacketAssemblerInterface() noexcept = default;  
        virtual auto assemble(PacketSegment) noexcept -> std::optional<AssembledPacket> = 0;
        virtual void destroy(const GlobalIdentifier& id) noexcept = 0;
    };

    struct InBoundContainerInterface{
        virtual ~InBoundContainerInterface() noexcept = default;
        virtual void push(dg::network_std_container::string) noexcept = 0;
        virtual auto pop() noexcept -> std::optional<dg::network_std_container::string> = 0;
    };

    class Packetizer: public virtual PacketizerInterface{

        private:

            size_t segment_byte_sz;
            Address factory_addr;
            size_t packetized_sz; 

        public:
            
            Packetizer(size_t segment_byte_sz,
                       Address factory_addr,
                       size_t packetized_sz) noexcept: segment_byte_sz(segment_byte_sz),
                                                       factory_addr(std::move(factory_addr)),
                                                       packetized_sz(packetized_sz){}

            auto packetize(const dg::network_std_container::string& buf) noexcept -> dg::network_std_container::vector<PacketSegment>{
                
                if (buf.size() == 0u){
                    return {PacketSegment{{}, this->make_global_packet_id(), 0u, 1u}};
                }

                size_t segment_sz = buf.size() / this->segment_byte_size + size_t{buf.size() % this->segment_byte_size != 0u};
                dg::network_std_container::vector<PacketSegment> rs{};

                for (size_t i = 0u; i < segment_sz; ++i){
                    size_t first            = this->segment_byte_sz * i;
                    size_t last             = std::min(buf.size(), this->segment_byte_sz * (i + 1));
                    PacketSegment packet{};
                    std::copy(buf.data() + first, buf.data() + last, std::back_inserter(packet.buf));
                    packet.id               = this->make_global_packet_id();
                    packet.segment_off      = i;
                    packet.segment_sz       = segment_sz;
                    packet.total_byte_sz    = buf.size();
                    rs.push_back(std::move(packet));
                }

                return rs;
            }
        
        private:

            auto make_global_packet_id() noexcept -> GlobalIdentifier{

                GlobalIdentifier id{};
                id.addr     = this->factory_addr;
                id.local_id = this->packetized_sz;
                this->packetized_sz += 1;

                return id;
            }
    };

    class EntranceController: public virtual EntranceControllerInterface{

        private:

            struct EntranceEntry{
                std::chrono::nanoseconds timestamp;
                GlobalIdentifier key;
                size_t entry_id;
            };

            dg::network_std_container::vector<EntranceEntry> entrace_entry_vec;
            dg::network_std_container::unordered_map<GlobalIdentifier, size_t> key_id_map;
            size_t id_size;
            std::chrono::nanoseconds expiry_period;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            EntranceController(dg::network_std_container::vector<EntranceEntry> entrace_entry_vec,
                               dg::network_std_container::unordered_map<GlobalIdentifier, size_t> key_id_map,
                               size_t id_size,
                               std::chrono::nanoseconds expiry_period,
                               std::unique_ptr<std::mutex> mtx) noexcept: entrace_entry_vec(std::move(entrace_entry_vec)),
                                                                          key_id_map(std::move(key_id_map)),
                                                                          id_size(id_size),
                                                                          expiry_period(std::move(expiry_period)),
                                                                          mtx(std::move(mtx)){}
            
            void tick(const GlobalIdentifier& key) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                size_t entry_id = this->id_size;
                std::chrono::nanoseconds now = dg::network_genult::unix_timestamp();
                EntranceEntry entry{now, key, entry_id};
                this->entrace_entry_vec.push_back(std::move(entry));
                std::push_heap(this->entrace_entry_vec.begin(), this->entrace_entry_vec.end(), [](const EntranceEntry& lhs, const EntranceEntry& rhs){return lhs.timestamp > rhs.timestamp;}); 
                this->key_id_map[key] = entry_id;
                this->id_size += 1;
            }

            auto get_expired() noexcept -> dg::network_std_container::vector<GlobalIdentifier>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto expired    = static_cast<std::chrono::nanoseconds>(dg::network_genult::unix_timestamp()) - this->expiry_period;
                auto rs         = dg::network_std_container::vector<GlobalIdentifier>{};

                while (true){
                    if (this->entrace_entry_vec.empty()){
                        break;
                    }

                    if (this->entrace_entry_vec.front().timestamp > expired){
                        break;
                    }

                    std::pop_heap(this->entrace_entry_vec.begin(), this->entrace_entry_vec.end(), [](const EntranceEntry& lhs, const EntranceEntry& rhs){return lhs.timestamp > rhs.timestamp;});
                    EntranceEntry entry = std::move(this->entrace_entry_vec.back());
                    this->entrace_entry_vec.pop_back();
                    auto map_ptr = this->key_id_map.find(entry.key); 

                    if constexpr(DEBUG_MODE_FLAG){
                        if (map_ptr == this->key_id_map.end()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    if (map_ptr->second == entry.entry_id){
                        rs.push_back(entry.key);
                        this->key_id_map.erase(map_ptr);
                    }
                }

                return rs;
            }
    };

    class PacketAssembler: public virtual PacketAssemblerInterface{

        private:

            dg::network_std_container::unordered_map<GlobalIdentifier, AssembledPacket> packet_map;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            PacketAssembler(dg::network_std_container::unordered_map<GlobalIdentifier, AssembledPacket> packet_map, 
                            std::unique_ptr<std::mutex> mtx) noexcept: packet_map(std::move(packet_map)),
                                                                       mtx(std::move(mtx)){}
            
            auto assemble(PacketSegment segment) noexcept -> std::optional<AssembledPacket>{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr = this->packet_map.find(segment.id);

                if (ptr == this->packet_map.end()){
                    dg::network_genult::assert(segment.segment_sz != 0u);
                    AssembledPacket pkt = this->make_empty_assembled_packet(segment.segment_sz);
                    auto [emplace_ptr, status] = this->packet_map.emplace(std::make_pair(segment.id, std::move(pkt)));
                    dg::network_genult::assert(status);
                    map_ptr = emplace_ptr;
                }

                map_ptr->second.data[segment.segment_idx] = std::move(segment);
                map_ptr->second.collected_segment_sz += 1;

                if (map_ptr->second.collected_segment_sz == map_ptr->second.total_segment_sz){
                    auto rs = std::move(map_ptr->second);
                    this->packet_map.erase(map_ptr);
                    return rs;
                }

                return std::nullopt;
            }

            void destroy(const GlobalIdentifier& id) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->packet_map.erase(id);
            }
        
        private:

            auto make_empty_assembled_packet(size_t segment_sz) noexcept -> AssembledPacket{
                
                auto vec            = dg::network_std_container::vector<PacketSegment>(segment_sz);
                size_t collected    = 0u;
                size_t total        = segment_sz;

                return AssembledPacket{std::move(vec), collected, total};
            }
    };

    class ExhaustionControlledPacketAssembler: public virtual PacketAssemblerInterface{

        private:

            std::unique_ptr<PacketAssemblerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            dg::network_std_container::unordered_map<GlobalIdentifier, size_t> counter_map;
            size_t capacity;
            size_t size;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledPacketAssembler(std::unique_ptr<PacketAssemblerInterface> base, 
                                                std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                dg::network_std_container::unordered_map<GlobalIdentifier, size_t> counter_map,
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

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                
                if (this->size == this->capacity){
                    return false;
                }

                auto map_ptr = this->counter_map.find(segmemt.id);

                if (map_ptr == this->counter_map.end()){
                    auto [emplace_ptr, status] = this->counter_map.emplace(std::make_pair(segment.id, 1u));
                    dg::network_genult::assert(status);
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

            void internal_destroy(const dg::network_std_container::string& id) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
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

            dg::network_std_container::vector<dg::network_std_container::string> vec;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            InBoundContainer(dg::network_std_container::vector<dg::network_std_container::string> vec,
                             std::unique_ptr<std::mutex> mtx) noexcept: vec(std::move(vec)),
                                                                        mtx(std::move(mtx)){}
            
            void push(dg::network_std_container::string buf) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->vec.push_back(std::move(buf));
            }

            auto pop() noexcept -> std::optional<dg::network_std_container::string>{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                
                if (this->vec.empty()){
                    return std::nullopt;
                }

                auto rs = std::move(this->vec.back());
                this->vec.pop_back();

                return rs;
            }
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

            void push(dg::network_std_container::string data) noexcept{

                auto task = [&]() noexcept{
                    return this->internal_push(data);
                };
                auto virt_task = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(task)>(task);
                this->executor->exec(virt_task);
            }

            auto pop() noexcept -> std::optional<dg::network_std_container::string>{

                return this->internal_pop();
            }
        
        private:

            auto internal_push(dg::network_std_container::string& data) noexcept -> bool{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->size == this->capacity){
                    return false;
                }

                this->base->push(std::move(data));
                this->size += 1;

                return true;
            }

            auto internal_pop() noexcept -> std::optional<dg::network_std_container::string>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                std::optional<dg::network_std_container::string> rs = this->base->pop();
                this->size -= static_cast<size_t>(static_cast<bool>(rs));

                return rs;
            }
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

                dg::network_std_container::vector<GlobalIdentifier> expired_id_vec = this->entrance_controller->get_expired();

                if (expired_id_vec.empty()){
                    return false;
                }

                for (const auto& expired_id: expired_id_vec){
                    this->packet_assembler->destroy(expired_id);
                }

                return true;
            }
    };

    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<PacketAssemblerInterface> packet_assembler;
            std::shared_ptr<InBoundContainerInterface> inbound_container;
            std::shared_ptr<EntranceControllerInterface> entrance_controller;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> base;

        public:

            InBoundWorker(std::shared_ptr<PacketAssemblerInterface> packet_assembler,
                          std::shared_ptr<InBoundContainerInterface> inbound_container,
                          std::shared_ptr<EntranceControllerInterface> entrance_controller,
                          std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> base) noexcept: packet_assembler(std::move(packet_assembler)),
                                                                                                                    inbound_container(std::move(inbound_container)),
                                                                                                                    entrance_controller(std::move(entrance_controller)),
                                                                                                                    base(std::move(base)){}

            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::string> data = this->base->recv();

                if (!data.has_value()){
                    return false;
                }

                PacketSegment pkt{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<PacketSegment>)(pkt, data->data(), data->size()); //optimizable - worth or not worth it

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(err));
                    return false;
                }

                GlobalIdentifier id = pkt.id;
                std::optional<AssembledPacket> assembled_packet = this->packet_assembler->assemble(std::move(pkt));
                this->entrance_controller->tick(id); //important that tick is here - otherwise leak

                if (assembled_packet.has_value()){
                    this->inbound_container->push(this->to_bstream(std::move(assembled_packet.value())));
                }

                return true;
            }
        
        private:

            auto to_bstream(AssembledPacket assembled_packet) noexcept -> dg::network_std_container::string{

                dg::network_genult::assert(assembled_packet.total_segment_sz != 0u);
                dg::network_std_container::string rs = std::move(assembled_packet.data[0u].buf); 

                for (size_t i = 1u; i < assembled_packet.data.size(); ++i){
                    std::copy(assembled_packet.data[i].buf.begin(), assembled_packet.data[i].buf.end(), std::back_inserter(rs));
                }

                return rs;
            }
    };

    class MailBox: public virtual dg::network_kernel_mailbox_impl1::core::MailBoxInterface{

        private:

            dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
            std::unique_ptr<PacketizerInterface> packetizer;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> base;
            std::shared_ptr<InBoundContainerInterface> inbound_container;

        public:

            MailBox(dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemons,
                    std::unique_ptr<PacketizerInterface> packetizer,
                    std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> base,
                    std::shared_ptr<InBoundContainerInterface> inbound_container) noexcept: daemons(std::move(daemons)),
                                                                                            packetizer(std::move(packetizer)),
                                                                                            base(std::move(base)),
                                                                                            inbound_container(std::move(inbound_container)){}

            void send(Address addr, dg::network_std_container::string arg) noexcept{
                
                if (arg.size() > MAX_STREAM_SIZE){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                    std::abort();
                }

                dg::network_std_container::vector<PacketSegment> segment_vec = this->packetizer->packetize(arg);

                for (PacketSegment& segment: segment_vec){
                    auto bstream = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(segment), ' '); //optimizable - worth or not worth it
                    dg::network_compact_serializer::integrity_serialize_into(bstream.data(), segment); //optimizable - ...
                    this->base->send(addr, std::move(bstream));
                }
            }

            auto recv() noexcept -> std::optional<dg::network_std_container::string>{

                return this->inbound_container->pop();
            }
    };

    struct Factory{

        static auto spawn_packetizer(size_t segment_byte_sz, Address factory_addr) -> std::unique_ptr<PacketizerInterface>{
            
            const size_t MIN_SEGMENT_SIZE   = size_t{1} << 8;
            const size_t MAX_SEGMENT_SIZE   = size_t{1} << 13; 

            if (std::clamp(segment_byte_sz, MIN_SEGMENT_SIZE, MAX_SEGMENT_SIZE) != segment_byte_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t random_seed = dg::network_randomizer::randomize_int<size_t>(); //this is to avoid reset + id collision
            return std::make_unique<Packetizer>(segment_byte_sz, factory_addr, random_seed);
        }

        static auto spawn_entrance_controller(std::chrono::nanoseconds expiry_period) -> std::unique_ptr<EntranceControllerInterface>{
            
            const std::chrono::nanoseconds MIN_EXPIRY_PERIOD    = std::chrono::milliseconds(1); 
            const std::chrono::nanoseconds MAX_EXPIRY_PERIOD    = std::chrono::seconds(20);

            if (std::clamp(expiry_period.count(), MIN_EXPIRY_PERIOD.count(), MAX_EXPIRY_PERIOD.count()) != expiry_period.count()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto entrance_entry_vec = dg::network_std_container::vector<EntranceEntry>{};
            auto key_id_map         = dg::network_std_container::unordered_map<GlobalIdentifier, size_t>{};
            auto id_size            = dg::network_randomizer::randomize_int<size_t>();             
            auto mtx                = std::make_unique<std::mutex>();

            return std::make_unique<EntranceController>(std::move(entrance_entry_vec), std::move(key_id_map), std::move(id_size), std::move(mtx));
        }

        static auto spawn_packet_assembler() -> std::unique_ptr<PacketAssemblerInterface>{

            auto packet_map = dg::network_std_container::unordered_map<GlobalIdentifier, AssembledPacket>{};
            auto mtx        = std::make_unique<std::mutex>();

            return std::make_unique<PacketAssembler>(std::move(packet_map), std::move(mtx));
        }

        static auto spawn_exhaustion_controlled_packet_assembler(std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor, size_t capacity) -> std::unique_ptr<PacketAssemblerInterface>{
            
            const size_t MIN_CAPACITY = size_t{1} << 10;
            const size_t MAX_CAPACITY = size_t{1} << 20;

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(capacity, MIN_CAPACITY, MAX_CAPACITY) != capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto counter_map    = dg::network_std_container::unordered_map<GlobalIdentifier, size_t>{};
            size_t size         = 0u;
            auto mtx            = std::make_unique<std::mutex>(); 
            auto base           = spawn_packet_assembler();

            return std::make_unique<ExhaustionControlledPacketAssembler>(std::move(base), std::move(executor), std::move(counter_map), capacity, size, std::move(mtx));
        }

        static auto spawn_inbound_container() -> std::unique_ptr<InBoundContainerInterface>{

            auto container  = dg::network_std_container::vector<dg::network_std_container::string>{};
            auto mtx        = std::make_unique<std::mutex>(); 

            return std::make_unique<InBoundContainer>(std::move(container), std::move(mtx));
        }

        static auto spawn_exhaustion_controlled_inbound_container(std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor, size_t capacity) -> std::unique_ptr<InBoundContainerInterface>{

            const size_t MIN_CAPACITY   = size_t{1} << 8;
            const size_t MAX_CAPCITY    = size_t{1} << 20;

            if (executor == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::clamp(capacity, MIN_CAPACITY, MAX_CAPACITY) != capacity){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t size = 0u;
            auto base   = spawn_inbound_container();
            auto mtx    = std::make_unique<std::mutex>(); 

            return std::make_unique<ExhaustionControlledInBoundContainer>(std::move(base), std::move(executor), capacity, size, std::move(mtx));
        }

        static auto spawn_mailbox_streamx(std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> base,
                                          std::unique_ptr<InBoundContainerInterface> inbound_container,
                                          std::unique_ptr<PacketAssemblerInterface> packet_assembler,
                                          std::unique_ptr<EntranceControllerInterface> entrance_controller,
                                          std::unique_ptr<PacketizerInterface> packetizer,
                                          size_t num_inbound_worker,
                                          size_t num_expiry_worker) -> std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface>{
            
            const size_t MIN_INBOUND_WORKER     = 1u;
            const size_t MAX_INBOUND_WORKER     = 1024u;
            const size_t MIN_RETRY_WORKER       = 1u;
            const size_t MAX_RETRY_WORKER       = 1024u; 

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
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUFMENT);
            }

            if (std::clamp(num_expiry_worker, MIN_RETRY_WORKER, MAX_RETRY_WORKER) != num_expiry_worker){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> base_sp = std::move(base);
            std::shared_ptr<InBoundContainerInterface> inbound_container_sp = std::move(inbound_container);
            std::shared_ptr<PacketAssemblerInterface> packet_assembler_sp = std::move(packet_assembler);
            std::shared_ptr<EntranceControllerInterface> entrance_controller_sp = std::move(entrance_controller)
            dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec{}; 

            //this is buggy - this will be DL if exception raises - not to worry about this now
            for (size_t i = 0u; i < num_inbound_worker; ++i){
                std::unique_ptr<dg::network_concurrency::WorkerInterface> worker = std::make_unique<InBoundWorker>(packet_assembler_sp, inbound_container_sp, entrance_controller_sp, base_sp);
                auto handle = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker)));
                daemon_vec.push_back(std::move(handle));
            }

            //this is buggy - this will be DL if exception raises - not to worry about this now
            for (size_t i = 0u; i < num_expiry_worker; ++i){
                std::unique_ptr<dg::network_concurrency::WorkerInterface> worker = std::make_unique<ExpiryWorker>(packet_assembler_sp, entrance_controller_sp);
                auto handle = dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister(dg::network_concurrency::IO_DAEMON, std::move(worker)));
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
        std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> base;
    };

    auto spawn(Config config) -> std::unique_ptr<dg::network_nernel_mailbox_impl1::core::MailBoxInterface>{

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

    using radix_t = uint32_t; 

    static auto serialize_msg(radix_t radix, dg::network_std_container::string content) noexcept -> dg::network_std_container::string{

        constexpr size_t HEADER_SZ      = dg::network_trivial_serializer::size(radix_t{});
        size_t content_sz               = content.size();
        size_t total_sz                 = content_sz + HEADER_SZ;
        content.resize(total_sz);
        char * header_ptr               = content.data() + content_sz;
        dg::network_trivial_serializer::serialize_into(header_ptr, radix);

        return content;
    }

    static auto deserialize_msg(dg::network_std_container::string serialized) noexcept -> std::pair<radix_t, dg::network_std_container::string>{

        constexpr size_t HEADER_SZ      = dg::network_trivial_serializer::size(radix_t{});
        auto [left, right]              = dg::network_genult::backsplit_str(std::move(serialized), HEADER_SZ);
        radix_t radix                   = {};

        dg::network_trivial_serializer::deserialize_into(radix, right.data());

        return std::make_pair(radix, std::move(left));
    }

    struct OutBoundRequest{
        Address dst;
        radix_t radix;
        dg::network_std_container::string content;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(dst, radix, content);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(dst, radix, content);
        }
    };

    struct ExhaustionControllerInterface{
        virtual ~ExhaustionControlInterface() noexcept = default;
        virtual auto inbound(size_t) noexcept -> bool = 0;
        virtual void outbound(size_t) noexcept = 0;
    };

    struct MailBoxInterface{
        virtual ~RadixMailBoxInterface() noexcept = default;
        virtual void send(Address addr, dg::network_std_container::string buf, radix_t radix) noexcept = 0;
        virtual auto recv(radix_t radix) noexcept -> std::optional<dg::network_std_container::string> = 0;
    };

    struct InBoundContainerInterface{
        virtual ~InBoundContainerInterface() noexcept = default;
        virtual auto get(radix_t) noexcept -> std::optional<dg::network_std_container::string> = 0;
        virtual void push(radix_t, dg::network_std_container::string) noexcept = 0;
    };

    struct OutBoundContainerInterface{
        virtual ~OutBoundContainerInterface() noexcept = default;
        virtual void push(OutBoundRequest) noexcept = 0;
        virtual auto pop() noexcept -> std::optional<OutBoundRequest> = 0;
    };

    class StdExhaustionController: public virtual ExhaustionControllerInterface{

        private:

            size_t cur_sz;
            const size_t capacity;
            const size_t max_unit_sz; 

        public:

            StdExhaustionController(size_t cur_sz, 
                                    size_t capacity,
                                    size_t max_unit_sz) noexcept: cur_sz(cur_sz),
                                                                  capacity(capacity),
                                                                  max_unit_sz(max_unit_sz){}
            
            auto inbound(size_t incoming_sz) noexcept -> bool{

                if (incoming_sz > this->max_unit_sz){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                if (this->cur_sz + incoming_sz > this->capacity){
                    return false;
                }

                this->cur_sz += incoming_sz;
                return true;
            }

            void outbound(size_t outcoming_sz) noexcept{

                this->cur_sz -= outcoming_sz;
            }
    };

    class InBoundContainer: public virtual InBoundContainerInterface{

        private:

            dg::network_std_container::unordered_map<radix_t, dg::network_std_container::vector<dg::network_std_container::string>> map;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            InBoundContainer(dg::network_std_container::unordered_map<radix_t, dg::network_std_container::vector<dg::network_std_container::string>> map,
                             std::unique_ptr<std::mutex> mtx) noexcept: map(std::move(map)),
                                                                        mtx(std::move(mtx)){}
            
            auto get(radix_t radix) noexcept -> std::optional<dg::network_std_container::string>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto ptr        = this->map.find(radix);

                if (ptr == this->map.end()){
                    return std::nullopt;
                }

                if (ptr->second.empty()){
                    return std::nullopt;
                }

                dg::network_std_container::string rs = std::move(ptr->second.back());
                ptr->second.pop_back();

                return std::move(rs);
            }

            void push(radix_t radix, dg::network_std_container::string content) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->map[radix].push_back(std::move(content));
            }
    };

    class ExhaustionControlledInBoundContainer: public virtual InBoundContainerInterface{

        private:

            std::unique_ptr<InBoundContainerInterface> base;
            dg::network_std_container::unordered_map<radix_t, std::unique_ptr<ExhaustionControllerInterface>> exhaustion_controller_map;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledInBoundContainer(std::unique_ptr<InBoundContainerInterface> base, 
                                                 dg::network_std_container::unordered_map<radix_t, std::unique_ptr<ExhaustionControllerInterface>> exhaustion_controller_map,
                                                 std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                 std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                            exhaustion_controller_map(std::move(exhaustion_controller_map)),
                                                                                            executor(std;:move(executor)),
                                                                                            mtx(std::move(mtx)){}
            
            auto get(radix_t radix) noexcept -> std::optional<dg::network_std_container::string>{

                return this->internal_get(radix);
            }
            
            void push(radix_t radix, dg::network_std_container::string content) noexcept{

                // while (!this->internal_push(radix, content)){} //move from wrong -> not yet wrong

                auto lambda = [&]() noexcept{return this->internal_push(radix, content);};
                auto exe    = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(lambda)>(std::move(lambda));
                this->executor->exec(exe);
            }
        
        private:

            auto internal_push(radix_t& radix, dg::network_std_container::string& content) noexcept -> bool{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto ec_ptr     = this->exhaustion_controller_map.find(radix);

                if constexpr(DEBUG_MODE_FLAG){
                    if (ec_ptr == this->exhaustion_controller_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                if (!ec_ptr->second->inbound(content.size())){
                    return false;
                }

                this->base->push(radix, std::move(content));
                return true;
            }
            
            auto internal_get(radix_t radix) noexcept -> std::optional<dg::network_std_container::string>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto rs         = this->base->get(radix);

                if (!static_cast<bool>(rs)){
                    return std::nullopt;
                } 

                auto ec_ptr     = this->exhaustion_controller_map.find(radix);

                if constexpr(DEBUG_MODE_FLAG){
                    if (ec_ptr == this->exhaustion_controller_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                ec_ptr->second->outbound(rs->size());
                return rs;
            }
    };

    class OutBoundContainer: public virtual OutBoundContainerInterface{

        private:

            dg::network_std_container::vector<OutBoundRequest> vec;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            OutBoundContainer(dg::network_std_container::vector<OutBoundRequest> vec,
                              std::unique_ptr<std::mutex> mtx) noexcept: vec(std::move(vec)),
                                                                         mtx(std::move(mtx)){}

            void push(OutBoundRequest request) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->vec.push_back(std::move(request));
            }

            auto pop() noexcept -> std::optional<OutBoundRequest>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->vec.empty()){
                    return std::nullopt;
                }

                OutBoundRequest rs = std::move(this->vec.back());
                this->vec.pop_back();

                return std::move(rs);
            }
    };

    class ExhaustionControlledOutBoundContainer: public virtual OutBoundContainerInterface{

        private:

            std::unique_ptr<OutBoundContainerInterface> base;
            std::unique_ptr<ExhaustionControllerInterface> exhaustion_controller;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledOutBoundContainer(std::unique_ptr<OutBoundContainerInterface> base,
                                                  std::unique_ptr<ExhaustionControllerInterface> exhaustion_controller,
                                                  std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                  std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                             exhaustion_controller(std::move(exhaustion_controller)),
                                                                                             executor(std::move(executor)),
                                                                                             mtx(std::move(mtx)){}

            void push(OutBoundRequest request) noexcept{

                // while (!this->internal_push(request)){} //move from wrong -> not yet wrong

                auto lambda = [&]{return this->internal_push(request);};
                auto exe    = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(lambda)>(std::move(lambda));
                this->executor->exec(exe);
            }

            auto pop() noexcept -> std::optional<OutBoundRequest>{

                return this->internal_pop();
            }
        
        private:

            auto internal_push(OutBoundRequest& request) noexcept -> bool{
                
                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                
                if (!this->exhaustion_controller->inbound(dg::network_compact_serializer::size(request))){
                    return false;
                }

                this->base->push(std::move(request));
                return true;
            }

            auto internal_pop()  noexcept -> std::optional<OutBoundRequest>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto rs         = this->base->pop();

                if (!static_cast<bool>(rs)){
                    return std::nullopt;
                }

                this->exhaustion_controller->outbound(rs->size());
                return rs;
            }
    };

    class AssorterWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> mailbox;
            std::shared_ptr<InBoundContainerInterface> inbound_container;
        
        public:

            AssorterWorker(std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> mailbox,
                           std::shared_ptr<InBoundContainerInterface> inbound_container) noexcept: mailbox(std::move(mailbox)),
                                                                                                   inbound_container(std::move(inbound_container)){}
            
            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::string> recv_data = this->mailbox->recv();

                if (!static_cast<bool>(recv_data)){
                    return false;
                }
                
                auto [radix, msg] = deserialize_msg(std::move(recv_data.value()));
                this->inbound_container->push(radix, std::move(msg));

                return true;
            }
    };
    
    class MailboxDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> mailbox;
            std::shared_ptr<OutBoundContainerInterface> container;
        
        public:

            MailBoxDispatcher(std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> mailbox,
                              std::shared_ptr<OutBoundContainerInterface> container) noexcept: mailbox(std::move(mailbox)),
                                                                                               container(std::move(container)){}
            
            bool run_one_epoch() noexcept{

                std::optional<OutBoundRequest> outbound_data = this->container->pop();

                if (!static_cast<bool>(outbound_data)){
                    return false;
                }

                dg::network_std_container::string bstream = serialize_msg(outbound_data->radix, std::move(outbound_data->content));
                this->mailbox->send(outbound_data->dst, std::move(bstream));
                return true;
            }
    };

    class RadixMailBox: public virtual MailBoxInterface{

        private:

            dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemons; 
            std::shared_ptr<OutBoundContainerInterface> outbound_container;
            std::shared_ptr<InBoundContainerInterface> inbound_container;

        public:

            RadixMailBox(dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemons, 
                         std::shared_ptr<OutBoundContainerInterface> outbound_container,
                         std::shared_ptr<InBoundContainerInterface> inbound_container) noexcept: daemons(std::move(daemons)),
                                                                                                 outbound_container(std::move(outbound_container)),
                                                                                                 inbound_container(std::move(inbound_container)){}
            
            void send(Address addr, dg::network_std_container::string buf, radix_t radix) noexcept{
                
                OutBoundRequest request{std::move(addr), std::move(radix), std::move(buf)};
                this->outbound_container->push(std::move(request));
            }

            auto recv(radix_t radix) noexcept -> std::optional<dg::network_std_container::string>{

                return this->inbound_container->get(radix);
            }
    };
};

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

            dg::network_std_container::unordered_map<Address, std::chrono::nanoseconds> address_ts_dict;
            std::chrono::nanoseconds error_threshold;
            std::chrono::nanoseconds termination_threshold;
            std::unique_ptr<std::mutex> mtx;

        public:

            HeartBeatMonitor(dg::network_std_container::unordered_map<Address, std::chrono::nanoseconds> address_ts_dict,
                             std::chrono::nanoseconds error_threshold,
                             std::chrono::nanoseconds termination_threshold,
                             std::unique_ptr<std::mutex> mtx) noexcept: address_ts_dict(std::move(address_ts_dict)),
                                                                        error_threshold(std::move(error_threshold)),
                                                                        termination_threshold(std::move(termination_threshold)),
                                                                        mtx(std::move(mtx)){}

            void recv_signal(const Address& addr) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                auto ptr = this->address_ts_dict.find(addr);

                if (ptr == this->address_ts_dict.end()){
                    auto err_msg = this->make_foreign_heartbeat_error_msg(addr);
                    dg::network_log::error_fast(err_msg.c_str());
                    return;
                }

                ptr->second = dg::network_genult::unix_timestamp();
            }

            bool check() noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
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

            auto make_missing_heartbeat_error_msg(const Address& addr) const noexcept -> dg::network_std_container::string{ //global memory pool - better to be noexcept here

                const char * fmt = "[NETWORKSTACK_HEARTBEAT] heartbeat not detected from {}:{}"; //ip-resolve is done externally - via log_reading - virtual ip is required to spawn proxy (if a node is not responding)
                return std::format(fmt, addr.ip, size_t{addr.port});
            }

            auto make_foreign_heartbeat_error_msg(const Address& addr) const noexcept -> dg::network_std_container::string{

                const char * fmt = "[NETWORKSTACK_HEARTBEAT] foreign heartbeat from {}:{}";
                return std::format(fmt, addr.ip, size_t{addr.port});
            }
    };

    class HeartBeatBroadcaster: public virtual dg::network_concurrency::WorkerInterface{

        private:

            dg::network_std_container::vector<Address> addr_table;
            Address host_addr;
            std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox;
            radix_t heartbeat_channel; 

        public:

            HeartBeatBroadcaster(dg::network_std_container::vector<Address> addr_table,
                                 Address host_addr,
                                 std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox,
                                 radix_t heartbeat_channel) noexcept addr_table(std::move(addr_table)),
                                                                     host_addr(std::move(host_addr)),
                                                                     mailbox(std::move(mailbox)),
                                                                     heartbeat_channel(std::move(heartbeat_channel)){}

            bool run_one_epoch() noexcept{
                
                size_t host_addr_sz = dg::network_compact_serializer::size(this->host_addr);
                dg::network_std_container::string serialized_host_addr(host_addr_sz);
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

                std::optional<dg::network_std_container::string> buf = this->mailbox->recv(this->heartbeat_channel);
                
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

            dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
            std::shared_ptr<network_kernel_mailbox_impl1_radixx::MailBoxInterface> mailbox;

        public:

            MailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemons,
                    std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox) noexcept: daemons(std::move(daemons)),
                                                                                                                 mailbox(std::move(mailbox)){}


            void send(Address addr, dg::network_std_container::string buf, radix_t radix) noexcept{
                
                this->mailbox->send(std::move(addr), std::move(buf), radix);
            }

            auto recv(radix_t radix) noexcept -> std::optional<dg::network_std_container::string>{

                return this->mailbox->recv(radix);
            }
    };
}

namespace dg::network_kernel_mailbox_impl1_concurrentx{

    using radix_t = dg::network_kernel_mailbox_impl1_radixx::radix_t; 

    template <size_t CONCURRENCY_SZ>
    class ConcurrentMailBox: public virtual dg::network_kernel_mailbox_impl1_radixx::MailBoxInterface{

        private:

            std::vector<std::unique_ptr<dg::network_kernel_mailbox_impl1_radixx::MailBoxInterface>> mailbox_vec;
        
        public:

            ConcurrentMailBox(std::vector<std::unique_ptr<dg::network_kernel_mailbox_impl1_radixx::MailBoxInterface>> mailbox_vec,
                              std::integral_constant<size_t, CONCURRENCY_SZ>) noexcept: mailbox_vec(std::move(mailbox_vec)){}

            void send(Address addr, dg::network_std_container::string buf, radix_t radix) noexcept{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                this->mailbox_vec[idx]->send(std::move(addr), std::move(buf), radix);
            }

            auto recv(radix_t radix) noexcept -> std::optional<std::network_std_container::string>{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                return this->mailbox_vec[idx]->recv(radix);
            }
    };
}