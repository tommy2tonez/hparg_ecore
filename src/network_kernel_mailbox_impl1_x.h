#ifndef __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__
#define __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__

#include "network_kernel_mailbox_impl1.h"
#include "network_trivial_serializer.h"
#include "network_concurrency.h"
#include "network_std_container.h"
#include <chrono>
#include "network_log.h"
#include "network_concurrency.h"

namespace dg::network_kernel_mailbox_impl1_heartbeatx{

    using namespace dg::network_kernel_mailbox_impl1::model; 

    struct HeartBeatMonitorInterface{

        virtual ~HeartBeatMonitorInterface() noexcept = default;
        virtual void recv_signal(const Address&) noexcept = 0;
        virtual void check() noexcept = 0;
    };

    struct BufferCenterInterface{

        virtual ~BufferCenterInterface() noexcept = default;
        virtual void push(dg::network_std_container::string) noexcept = 0;
        virtual auto pop() noexcept -> std::optional<dg::network_std_container::string> = 0;
    };

    struct HeartbeatEncoderInterface{

        virtual ~HeartbeatEncoderInterface() noexcept = default;
        virtual auto encode(Address) noexcept ->  dg::network_std_container::string = 0;
        virtual auto decode(const void *, size_t) noexcept -> std::expected<Address, exception_t> = 0;
    };

    class BufferCenter: public virtual BufferCenterInterface{

        private:

            dg::network_std_container::vector<dg::network_std_container::string> buf_vec;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            BufferCenter(dg::network_std_container::vector<dg::network_std_container::string> buf_vec,
                         std::unique_ptr<std::mutex> mtx) noexcept: buf_vec(std::move(buf_vec)),
                                                                    mtx(std::move(mtx)){}
            
            void push(dg::network_std_container::string data) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->buf_vec.push_back(std::move(data));
            }

            auto pop() noexcept -> std::optional<dg::network_std_container::string>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                
                if (this->buf_vec.size() == 0u){
                    return std::nullopt;
                }

                auto rs = std::move(this->buf_vec.back());
                this->buf_vec.pop_back();

                return rs;
            }
    };

    template <size_t BUFFER_CENTER_SZ>
    class ConcurrentBufferCenter: public virtual BufferCenterInterface{

        private:

            dg::network_std_container::vector<std::unique_ptr<BufferCenterInterface>> buffer_center;
        
        public:

            ConcurrentBufferCenter(dg::network_std_container::vector<std::unique_ptr<BufferCenterInterface>> buffer_center,
                                   const std::integral_constant<size_t, BUFFER_CENTER_SZ>) noexcept: buffer_center(std::move(buffer_center)){}
            
            void push(dg::network_std_container::string data) noexcept{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, BUFFER_CENTER_SZ>{});
                this->buffer_center[idx]->push(std::move(data));
            }

            auto pop() noexcept -> std::optional<dg::network_std_container::string>{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, BUFFER_CENTER_SZ>{});
                return this->buffer_center[idx]->pop();
            }
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

            void check() noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                std::chrono::nanoseconds now = dg::network_genult::unix_timestamp(); 

                for (const auto& pair: this->address_ts_dict){
                    if (dg::network_genult::timelapsed(pair.second, now) > this->error_threshold){
                        auto err_msg = this->make_missing_heartbeat_error_msg(pair.first); 
                        dg::network_log::error_fast(err_msg.c_str());
                    }

                    if (dg::network_genult::timelapsed(pair.second, now) > this->termination_threshold){
                        auto err_msg = this->make_missing_heartbeat_error_msg(pair.first);
                        dg::network_log::critical(err_msg.c_str());
                        std::abort();
                    }
                }
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
            dg::network_std_container::string heartbeat_packet;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> mailbox;
        
        public:

            HeartBeatBroadcaster(dg::network_std_container::vector<Address> addr_table,
                                 dg::network_std_container::string heartbeat_packet,
                                 std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> mailbox) noexcept addr_table(std::move(addr_table)),
                                                                                                                             heartbeat_packet(std::move(heartbeat_packet)),
                                                                                                                             mailbox(std::move(mailbox)){}

            bool run_one_epoch() noexcept{

                for (const auto& addr: this->addr_table){
                     this->mailbox->send(addr, this->heartbeat_packet);
                }

                return true;
            }
    };

    class HeartBeatReceiver: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<HeartBeatMonitorInterface> heartbeat_monitor;
            std::shared_ptr<BufferCenterInterface> ib_buffer_center;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox;
            std::unique_ptr<HeartbeatEncoderInterface> heartbeat_encoder;

        public:

            HeartBeatReceiver(std::shared_ptr<HeartBeatMonitorInterface> heartbeat_monitor,
                              std::shared_ptr<BufferCenterInterface> ib_buffer_center,
                              std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox,
                              std::unique_ptr<HeartbeatEncoderInterface> heartbeat_encoder) noexcept: heartbeat_monitor(std::move(heartbeat_monitor)),
                                                                                                      ib_buffer_center(std::move(ib_buffer_center)),
                                                                                                      mailbox(std::move(mailbox)),
                                                                                                      heartbeat_encoder(std::move(heartbeat_encoder)){}
            
            bool run_one_epoch() noexcept{
                
                std::optional<dg::network_std_container::string> buf = this->mailbox->recv();

                if (!static_cast<bool>(buf)){
                    return false;
                }

                std::expected<Address, exception_t> heartbeat = this->heartbeat_encoder->decode(buf->data(), buf->size());

                if (heartbeat.has_value()){
                    this->heartbeat_monitor->recv_signal(heartbeat.value());
                    return true;
                }
                
                this->ib_buffer_center->push(std::move(buf.value()));
                return true;
            }
    };

    class HeartBeatChecker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<HeartBeatMonitorInterface> monitor;
        
        public:

            HeartBeatChecker(std::shared_ptr<HeartBeatMonitorInterface> monitor) noexcept: monitor(std::move(monitor)){}

            bool run_one_epoch() noexcept{

                this->monitor->check();
                return true;
            }
    };

    class MailBox: public virtual dg::network_kernel_mailbox_impl1::core::MailBoxInterface{

        private:

            dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox;
            std::shared_ptr<BufferCenterInterface> ib_buffer_center;

        public:

            MailBox(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemons,
                    std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox,
                    std::shared_ptr<PacketCenterInterface> ib_buffer_center) noexcept: daemons(std::move(daemons)),
                                                                                       mailbox(std::move(mailbox)),
                                                                                       ib_buffer_center(std::move(ib_buffer_center)){}

            void send(Address addr, dg::network_std_container::string buf) noexcept{
                
                this->mailbox->send(std::move(addr), std::move(buf));
            }

            auto recv() noexcept -> std::optional<dg::network_std_container::string>{

                return this->ib_buffer_center->pop();
            }
    };
}

namespace dg::network_kernel_mailbox_impl1_concurrencyx{

    template <size_t CONCURRENCY_SZ>
    class ConcurrentMailBox: public virtual dg::network_kernel_mailbox_impl1::core::MailBoxInterface{

        private:

            std::vector<std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>> mailbox_vec;
        
        public:

            ConcurrentMailBox(std::vector<std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface>> mailbox_vec,
                              std::integral_constant<size_t, CONCURRENCY_SZ>) noexcept: mailbox_vec(std::move(mailbox_vec)){}

            void send(Address addr, dg::network_std_container::string buf) noexcept{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                this->mailbox_vec[idx]->send(std::move(addr), std::move(buf));
            }

            auto recv() noexcept -> std::optional<std::network_std_container::string>{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                return this->mailbox_vec[idx]->recv();
            }
    };
}

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