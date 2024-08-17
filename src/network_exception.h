#ifndef __NETWORK_EXCEPTION_H__
#define __NETWORK_EXCEPTION_H__

#include <exception>
#include <stdint.h>
#include <stdlib.h> 

namespace dg::network_exception{

    struct invalid_arg: std::exception{

        const char * what() const noexcept{

            return "invalid_init";
        }
    };

    static inline constexpr uint8_t CORE_SEGFAULT = 0u;
}

#endif