
#ifndef __NETWORK_CONCURRENCY_H__
#define __NETWORK_CONCURRENCY_H__

#include <stddef.h>
#include <stdint.h>
#include <thread>
#include <vector>
#include <memory>
#include "network_concurrency_impl1.h"
#include <bit>

namespace dg::network_concurrency{

    using namespace dg::network_concurrency_impl1::daemon_option_ns; 
    using WorkerInterface = dg::network_concurrency_impl1::WorkerInterface; 

    using affine_policy_option_t = uint8_t;

    enum affine_policy_option: affine_policy_option_t{
        standard_affine         = 0u, 
        kernel_overwrite_affine = 1u, //this assumes that all cores are isolated for the application -  
        kernel_decide_affine    = 2u;
    };

    struct AffineDaemonPlanMaker{

        auto set_core(int core_id, double speed) -> AffineDaemonPlanMaker&{
 
        }

        auto set_cpu_usage(daemon_kind_t daemon_kind, double cpu_ult_percentage) -> AffineDaemonPlanMaker&{

        }

        auto set_num_worker(daemon_kind_t daemon_kind, size_t sz) -> AffineDaemonPlanMaker&{

        }

        auto get_plan() -> std::vector<std::tuple<daemon_kind_t, std::vector<int>>>{

        }
    };

    struct SleepSchedulePlanMaker{

        auto get_plan() -> std::unordered_map<daemon_kind_t, std::chrono::nanoseconds>{

        }
    };

    struct SleepyDaemonPlanMaker{

        auto set_affine_policy(affine_policy_option_t) -> SleepyDaemonPlanMaker&{

        }

        auto set_computing_thread_count(size_t) -> SleepyDaemonPlanMaker&{

        }

        auto set_io_thread_count(size_t) -> SleepyDaemonPlanMaker&{

        } 

        auto set_transportation_thread_count(size_t) -> SleepyDaemonPlanMaker&{

        }

        auto set_heartbeat_thread_count(size_t) -> SleepyDaemonPlanMaker&{

        }

        auto get_plan() -> std::vector<std::tuple<daemon_kind_t, std::optional<std::vector<int>>, std::chrono::nanoseconds>>{

        }
    };

    static inline constexpr auto AFFINE_POLICY                              = kernel_overwrite_affine;
    static inline constexpr size_t THREAD_COUNT                             = 32;
    static inline constexpr size_t COMPUTING_DAEMON_NETWORK_THREAD_COUNT    = 8;
    static inline constexpr size_t IO_DAEMON_THREAD_COUNT                   = 8;
    static inline constexpr size_t TRANSPORTATION_DAEMON_THREAD_COUNT       = 8; 
    static inline constexpr size_t HEARTBEAT_DAEMON_THREAD_COUNT            = 8;

    struct ConcurrencyResource{
        std::unique_ptr<dg::network_concurrency_impl1::DaemonControllerInterface> daemon_controller; 
        jg::dense_hash_map<std::thread::id, size_t> thrid_to_idx_map;
    };

    inline ConcurrencyResource concurrency_resource{};

    void init(){
        
        auto thrid_to_idx_map   = jg::dense_hash_map<std::thread::id, size_t>{};
        auto daemon_id_map      = std::unordered_map<daemon_kind_t, std::vector<size_t>>{};
        auto id_runner_map      = std::unordered_map<size_t, std::unique_ptr<dg::network_concurrency_impl1::DaemonRunnerInterface>>{};
        
        auto plan = SleepyDaemonPlanMaker().set_affine_policy(AFFINE_POLICY)
                                           .set_computing_thread_count(COMPUTING_DAEMON_NETWORK_THREAD_COUNT)
                                           .set_heartbeat_thread_count(HEARTBEAT_DAEMON_THREAD_COUNT)
                                           .set_io_thread_count(IO_DAEMON_THREAD_COUNT)
                                           .set_transportation_thread_count(TRANSPORTATION_DAEMON_THREAD_COUNT)
                                           .get_plan();
        
        for (const auto& thr_metadata: plan){
            auto [daemon_kind, optional_affine_vector, sleep_dur] = thr_metadata;
            auto [daemon_runner, thr_id] = [&]{
                if (optional_affine_vector){
                    return dg::network_concurrency_impl1::DaemonFactory::spawn_affine_daemon_runner(optional_affine_vector.value(), dg::network_concurrency_impl1::ReschedulerFactory::spawn_sleepy_rescheduler(sleep_dur));
                } else{
                    return dg::network_concurrency_impl1::DaemonFactory::spawn_daemon_runner(dg::network_concurrency_impl1::ReschedulerFactory::spawn_sleepy_rescheduler(sleep_dur));
                }
            }();

            thrid_to_idx_map[thr_id] = thrid_to_idx_map.size();
            daemon_id_map[daemon_kind].push_back(thrid_to_idx_map[thr_id]);
            id_runner_map[thrid_to_idx_map[thr_id]] = std::move(daemon_runner);
        }

        concurrency_resource.thrid_to_idx_map   = std::move(thrid_to_idx_map);
        concurrency_resource.daemon_controller  = dg::network_concurrency_impl1::ControllerFactory::spawn_daemon_controller(std::move(daemon_id_map), std::move(id_runner_map));
    }

    void deinit() noexcept{

        concurrency_resource = {};
    }

    auto to_thread_idx(std::thread::id id) noexcept -> size_t{

        auto ptr = concurrency_resource.thrid_to_idx_map.find(id);

        if constexpr(DEBUG_MODE_FLAG){
            if (ptr == concurrency_resource.thrid_to_idx_map.end()){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        return ptr->second;
    }

    auto this_thread_idx() noexcept -> size_t{
        
        return to_thread_idx(std::this_thread::get_id());
    }

    auto daemon_register(daemon_t daemon, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<size_t, exception_t>{

        daemon_runner->_register(daemon, std::move(worker));
    }

    void daemon_deregister(size_t id) noexcept{

        daemon_runner->deregister(id);
    }

    using daemon_dynamic_deregister_t = void (*)(size_t *) noexcept; 

    auto daemon_saferegister(daemon_t daemon, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<std::unique_ptr<size_t, daemon_dynamic_deregister_t>, exception_t>{

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