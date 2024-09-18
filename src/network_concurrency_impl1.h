#ifndef __DG_NETWORK_CONCURRENCY_IMPL1_H__
#define __DG_NETWORK_CONCURRENCY_IMPL1_H__

#ifdef __linux__
#include "network_concurrency_impl1_linux.h"

namespace dg::network_concurrency_impl1::daemon_option_ns{

    using namespace dg::network_concurrency_impl1_linux::daemon_option_ns;
} 

namespace dg::network_concurrency_impl1{

    using namespace dg::network_concurrency_impl1_linux;
} 

#elif _WIN32 
#include "network_concurrency_impl1_window.h" 

namespace dg::network_concurrency_impl1::daemon_option_ns{

    using namespace dg::network_concurrency_impl1_window::daemon_option_ns;
}

namespace dg::network_concurrency_impl1{

    using namespace dg::network_concurrency_impl1_window;
}

#else 
static_assert(false);
#endif

#endif