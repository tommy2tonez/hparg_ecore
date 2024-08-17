
#ifndef __NETWORK_CONCURRENCY__
#define __NETWORK_CONCURRENCY__

#include <stddef.h>
#include <stdint.h>
#include <thread>
#include <vector>

namespace dg::network_concurrency{

    static inline constexpr size_t THREAD_COUNT                     = 30;
    static inline constexpr size_t DAEMON_NETWORK_THREAD_COUNT      = 10;
    static inline constexpr size_t DAEMON_COMPUTE_THREAD_COUNT      = 10; 
    static inline constexpr size_t DAEMON_COLLECTOR_THREAD_COUNT    = 10;

    void init(std::vector<std::thread::id> network_thread_ids, 
              std::vector<std::thread::id> compute_thread_ids,
              std::vector<std::thread::id> daemon_thread_ids){

    }

    inline auto to_thread_idx(std::thread::id) noexcept -> size_t{

    }

    inline auto this_thread_idx() noexcept -> size_t{
        
    }
};

#endif