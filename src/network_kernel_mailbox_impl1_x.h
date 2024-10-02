#ifndef __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__
#define __NETWORK_KERNEL_MAILBOX_IMPL1_X_H__

#include "network_kernel_mailbox_impl1.h"
#include "network_trivial_serializer.h"
#include "network_concurrency.h"
#include "network_std_container.h"
#include <chrono>
#include "network_log.h"
#include "network_concurrency.h"

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

namespace dg::network_kernel_mailbox_impl1_radixx{

    using radix_t = uint8_t; 

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

                return {std::in_place_t{}, std::move(rs)};
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
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledInboundContainer(std::unique_ptr<InBoundContainerInterface> base, 
                                                 dg::network_std_container::unordered_map<radix_t, std::unique_ptr<ExhaustionControllerInterface>> exhaustion_controller_map,
                                                 std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                            exhaustion_controller_map(std::move(exhaustion_controller_map)),
                                                                                            mtx(std::move(mtx)){}
            
            auto get(radix_t radix) noexcept -> std::optional<dg::network_std_container::string>{

                return this->internal_get(radix);
            }
            
            void push(radix_t radix, dg::network_std_container::string content) noexcept{

                while (!this->internal_push(radix, content)){}
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

                return {std::in_place_t{}, std::move(rs)};
            }
    };

    class ExhaustionControlledOutBoundContainer: public virtual OutBoundContainerInterface{

        private:

            std::unique_ptr<OutBoundContainerInterface> base;
            std::unique_ptr<ExhaustionControllerInterface> exhaustion_controller;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledOutBoundContainer(std::unique_ptr<OutBoundContainerInterface> base,
                                                  std::unique_ptr<ExhaustionControllerInterface> exhaustion_controller,
                                                  std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                             exhaustion_controller(std::move(exhaustion_controller)),
                                                                                             mtx(std::move(mtx)){}

            void push(OutBoundRequest request) noexcept{

                while (!this->internal_push(request)){}
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