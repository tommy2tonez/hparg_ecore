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

    inline std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox; 

    void norm_init(){

    } 

    void heartbeat_init(){

    }

    auto send(Address addr, dg::network_std_container::string msg) noexcept -> exception_t{

        if (msg.size() > dg::network_kernel_mailbox_impl1::constants::MAXIMUM_MSG_SIZE){
            return dg::network_exception::BUFFER_OVERFLOW;
        }

        mailbox->send(std::move(addr), std::move(msg));
        return dg::network_exception::SUCCESS;
    }

    auto recv() noexcept -> std::optional<dg::network_std_container::string>{
        
        return mailbox->recv();
    }   
}

#endif