#ifndef __NETWORK_KERNEL_PRODUCER_H__
#define __NETWORK_KERNEL_PRODUCER_H__

#include <stdint.h>
#include <stddef.h>
#include <array>
#include <string>
#include <memory>

namespace dg::network_kernel_mailbox{

    using ip_type = std::string;

    void send_ack(ip_type, const void *, size_t) noexcept{ //relax noexcept next iteration - yet memory exhaustion should be a termination error

    }

    auto recv_ack() noexcept -> std::unique_ptr<char[]>{ //relax noexcept next iteration - yet memory exhaustion should be a termination error

    }

    void send_noack(ip_type, const void *, size_t) noexcept{ //relax noexcept next iteration - yet memory exhaustion should be a termination error

    }

    auto recv_noack() noexcept -> std::unique_ptr<char[]>{ //relax noexcept next iteration - yet memory exhaustion should be a termination error

    }
}

#endif