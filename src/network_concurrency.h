
#ifndef __NETWORK_CONCURRENCY_H__
#define __NETWORK_CONCURRENCY_H__

#include <stddef.h>
#include <stdint.h>
#include <thread>
#include <vector>
#include <memory>
#include "network_concurrency_impl1.h"

namespace dg::network_concurrency{

    using namespace dg::network_concurrency_impl1::daemon_option_ns; 
    using WorkerInterface = dg::network_concurrency_impl1::WorkerInterface; 

    static inline constexpr size_t THREAD_COUNT                     = 30;
    static inline constexpr size_t DAEMON_NETWORK_THREAD_COUNT      = 10;
    static inline constexpr size_t DAEMON_COMPUTE_THREAD_COUNT      = 10; 
    static inline constexpr size_t DAEMON_COLLECTOR_THREAD_COUNT    = 10;

    inline std::unique_ptr<dg::network_concurrency_impl1::DaemonRunnerInterface> daemon_runner{}; 

    void init(std::vector<std::thread::id> network_thread_ids, 
              std::vector<std::thread::id> compute_thread_ids,
              std::vector<std::thread::id> daemon_thread_ids){

    }

    inline auto to_thread_idx(std::thread::id) noexcept -> size_t{

    }

    inline auto this_thread_idx() noexcept -> size_t{
        
    }

    inline auto daemon_register(daemon_t daemon, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<size_t, exception_t>{

        daemon_runner->_register(daemon, std::move(worker));
    }

    inline void daemon_deregister(size_t id) noexcept{

        daemon_runner->deregister(id);
    }

    using daemon_dynamic_deregister_t = void (*)(size_t *) noexcept; 

    inline auto daemon_saferegister(daemon_t daemon, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<std::unique_ptr<size_t, daemon_dynamic_deregister_t>, exception_t>{

        auto destructor = [](size_t * arg_id) noexcept{
            daemon_deregister(*arg_id);
            delete arg_id;
        };

        std::expected<size_t, exception_t> handle = daemon_register(daemon, std::move(worker));
        
        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return {std::in_place_t{}, new size_t{handle.value()}, destructor}; //
    }

    using daemon_raii_handle_t = std::unique_ptr<size_t, daemon_dynamic_deregister_t>;  

};

#endif