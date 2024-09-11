#ifndef __NETWORK_CUFSIO_H__
#define __NETWORK_CUFSIO_H__

#ifdef __linux__
#include "network_cufsio_linux.h"

namespace dg::network_cufsio{
    using namespace dg::network_cufsio_linux;
} 

#elif _WIN32
#include "network_cufsio_window.h"

namespace dg::network_cufsio{
    using namespace dg::network_cufsio_window;
}

#else
static_assert(false);
#endif

#endif