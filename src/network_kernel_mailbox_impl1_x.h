#ifndef __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__
#define __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__

#include "network_kernel_mailbox_impl1.h"
#include "network_trivial_serializer.h"
#include "network_concurrency.h"
#include "network_std_container.h" 

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

                const char * fmt = "heartbeat not detected from {}:{}";
                return std::format(fmt, addr.ip, size_t{addr.port});
            }

            auto make_foreign_heartbeat_error_msg(const Address& addr) const noexcept -> dg::network_std_container::string{

                const char * fmt = "foreign heartbeat from {}:{}";
                return std::format(fmt, addr.ip, size_t{addr.port});
            }

    };

    class HeartBeatBroadcaster: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::vector<Address> addr_table;
            dg::network_std_container::string heartbeat_packet;
            std::shared_ptr<dg::network_kernel_mailbox_impl1::core::MailBoxInterface> mailbox;
        
        public:

            HeartBeatBroadcaster(std::vector<Address> addr_table,
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

    class MailBox: public virtual dg::network_kernel_mailbox_impl1_heartbeatx::core::MailBoxInterface{

        private:

            std::vector<dg::network_concurrency::daemon_raii_handle_t> daemons;
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