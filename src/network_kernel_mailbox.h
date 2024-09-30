#ifndef __NETWORK_KERNEL_PRODUCER_H__
#define __NETWORK_KERNEL_PRODUCER_H__

#include <stdint.h>
#include <stddef.h>
#include <array>
#include <string>
#include <memory>
#include "network_std_container.h"
#include <chrono>
#include <vector>
#include <optional>
#include "network_kernel_mailbox_impl1.h"
#include "network_kernel_mailbox_impl1_x.h" 

namespace dg::network_kernel_mailbox{

    struct Config{
        size_t outbound_worker_count;
        size_t inbound_worker_count;
        size_t retransmission_worker_count;
        dg::network_std_container::vector<ip_t> host_ips;
        dg::network_std_container::vector<uint16_t> host_ports;
        std::chrono::nanoseconds retransmission_delay;
        size_t retranmission_count;
        std::optional<size_t> inbound_exhaustion_control_sz;
        std::optional<size_t> outbound_exhaustion_control_sz;
        bool is_meterlog_enabled;
        bool is_heartbeat_enabled;
        bool is_recovery_on_failure_enabled;
    };

    inline std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox;

    void base_init(){

    }

    class RecoveryRunner: public virtual dg::network_recovery_line::RecoveryExecutableInterface{

        private:

            Config config; 

        public:

            RecoveryRunner(Config config) noexcept: config(std::move(config)){}

            void recover() noexcept{

                auto cfunc      = dg::network_exception::to_cstyle_function(base_init); 
                exception_t err = cfunc(this->config);

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(err));
                    std::abort();
                }
            }
    };

    class FailureObserver: public virtual dg::network_kernel_mailbox_impl1_heartbeatx::ObserverInterface{

        private:

            std::unique_ptr<size_t, dg::network_recovery_line::recovery_line_dclose_t> recovery_line;

        public:

            FailureObserver(std::unique_ptr<size_t, dg::network_recovery_line::recovery_line_dclose_t> recovery_line) noexcept: recovery_line(std::move(recovery_line)){}

            void notify() noexcept{

                dg::network_recovery_line::notify(*this->recovery_line);
            }
    };

    //--important to static this - to do socket recovery + friends by respawning component
    //this is probably the most troubling component

    void init(){

    }

    void send(Address addr, dg::network_std_container::string msg) noexcept{

        if (msg.size() > dg::network_kernel_mailbox_impl1::constants::MAXIMUM_MSG_SIZE){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        mailbox->send(std::move(addr), std::move(msg));
    }

    auto recv() noexcept -> std::optional<dg::network_std_container::string>{ //optional string because string.empty() does not mean that it is not a packet
        
        return mailbox->recv();
    }   
}

#endif