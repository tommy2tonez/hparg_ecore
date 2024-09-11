#ifndef __NETWORK_FILEIO_H__
#define __NETWORK_FILEIO_H__

#ifdef __linux__
#include "network_fileio_linux.h"

namespace dg::network_fileio{
    using namespace dg::network_fileio_linux;
}

#elif _WIN32
#include "network_fileio_window.h"

namespace dg::network_fileio{
    using namespace dg::network_fileio_window;
} 

#else
static_assert(false);
#endif

#endif