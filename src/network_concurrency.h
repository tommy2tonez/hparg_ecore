
#ifndef __NETWORK_CONCURRENCY__
#define __NETWORK_CONCURRENCY__

#include <stddef.h>
#include <stdint.h>
#include <thread>
#include <vector>
#include <memory>

namespace dg::network_concurrency{

    struct WorkerInterface{

        virtual ~WorkerInterface() noexcept = default;
        virtual bool run_one_epoch() noexcept = 0; 
    };


    using daemon_t      = uint8_t; 
    using daemon_id_t   = size_t;

    enum daemon_option: daemon_t{
        COMPUTING_DAEMON        = 0,
        TRANSPORTATION_DAEMON   = 1,
        IO_DAEMON               = 2,
        HEARTBEAT_DAEMON        = 3
    };

    //need kernel magic to make this work
    //if shared thread then risking deadlock
    //if not shared thread then there are thread pollution
    //need to reimplement kernel scheduler - tmr 

    struct DaemonRunnerInterface{

        virtual ~DaemonRunnerInterface() noexcept = default;
        virtual auto next_id(daemon_t) noexcept -> daemon_id_t = 0;
        virtual void register_daemon(daemon_id_t, std::unique_ptr<WorkerInterface>) noexcept  = 0;
        virtual void deregister(size_t) noexcept = 0;
    };

    static inline constexpr size_t THREAD_COUNT                     = 30;
    static inline constexpr size_t DAEMON_NETWORK_THREAD_COUNT      = 10;
    static inline constexpr size_t DAEMON_COMPUTE_THREAD_COUNT      = 10; 
    static inline constexpr size_t DAEMON_COLLECTOR_THREAD_COUNT    = 10;

    inline std::unique_ptr<DaemonRunnerInterface> daemon_runner{}; 

    void init(std::vector<std::thread::id> network_thread_ids, 
              std::vector<std::thread::id> compute_thread_ids,
              std::vector<std::thread::id> daemon_thread_ids){

    }

    inline auto to_thread_idx(std::thread::id) noexcept -> size_t{

    }

    inline auto this_thread_idx() noexcept -> size_t{
        
    }

    inline auto daemon_next_id(daemon_t daemon) noexcept -> daemon_id_t{

        return daemon_runner->next_id(daemon);
    }

    inline void daemon_register(daemon_id_t id, std::unique_ptr<WorkerInterface> worker) noexcept{

        daemon_runner->register_daemon(id, std::move(worker));
    }

    inline void daemon_deregister(daemon_id_t id) noexcept{

        daemon_runner->deregister(id);
    }

    using daemon_dynamic_unregister_t = void (*)(daemon_id_t *) noexcept; 
    using daemon_raii_handle_t = std::unique_ptr<daemon_id_t, daemon_dynamic_unregister_t>;  

    inline auto daemon_saferegister(daemon_id_t id, std::unique_ptr<WorkerInterface> worker) noexcept -> std::unique_ptr<daemon_id_t, daemon_dynamic_unregister_t>{

        auto destructor = [](daemon_id_t * arg_id) noexcept{
            daemon_deregister(*arg_id);
            delete arg_id;
        };

        daemon_register(id, std::move(worker));
        return {new daemon_id_t{id}, destructor};
    }
};

#endif