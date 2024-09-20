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

    //this is probably moderately complicated
    //(1): recovery by using serialized access to init + functors without compromising speed
    //     recovery by compromising program + restart 

    //(2): network stack calibration (single packet center - multiple sockets - multiple ports, 
    //                                multiple packet_centers - single socket - single port, 
    //                                multiple packet centers - multiple sockets - multiple ports 
    //                                network_stack_buffer_sz/ socket,
    //                                number of concurrent workers,
    //                                cpu usage, etc.)
    //Such knowledge is not compile-time deterministic
    
    //(3): program calibration - a maximized network stack calibration might have destructive interference
    //(4): congestion control - this is another tough task - allocation is a natural congestion control - implementing a custom congestion control here might have destructive interference
    //(5): recovery methods: - unresponding IP resolution: program_virtual_ip (program responsibility) or kernel_virtual_ip (kernel responsibility). If former, goto (1)
    //                       - corrupted socket resolution (unlikely): goto (1)

    inline std::unique_ptr<dg::network_kernel_mailbox_impl1::core::MailboxInterface> mailbox;

    void init(){

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