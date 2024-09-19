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

namespace dg::network_concurrency_impl1_app{

    using namespace dg::network_concurrency_impl1::daemon_option_ns;

    using affine_policy_option_t = uint8_t;

    enum affine_policy_option: affine_policy_option_t{
        kernel_overwrite_affine_policy  = 0u, //this assumes that all cores are isolated for the application -  
        kernel_decide_affine_policy     = 1u;
    };

    //this heuristic does not work very well - without a specific implementation of scheduler - this renders useless
    //WLOG assume every affine distribution problem could be viewed as  
    //dict[(core_id, daemon_kind)] = core_usage
    //dict[[core_id, daemon_kind]] = list(threads)
    //len(list(threads)) / total_thread(core_id) = core_usage
    //the number of threads then - is dynamic (runtime determined) - not static - yet one could argue that it's runtime-determined-compile-time-deterministic
    
    struct MonoAffineDaemonPlanMaker{

        private:

            std::unordered_map<int, double> core_speed_map;
            std::unordered_map<daemon_kind_t, double> daemon_usage_map;
            size_t thr_count_per_core;
            bool is_threadpercore_initialized; 
            bool is_core_speed_initialized;
            bool is_daemon_usage_initialized;

        public:

            MonoAffineDaemonPlanMaker(): core_speed_map(),
                                         daemon_usage_map(),
                                         thr_count_per_core(), 
                                         is_threadpercore_initialized(false), 
                                         is_core_speed_initialized(false),
                                         is_daemon_usage_initialized(false){}

            auto set_thread_per_core(size_t thr_count_per_core) -> MonoAffineDaemonPlanMaker&{

                if (thr_count_per_core == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->thr_count_per_core = thr_count_per_core;
                this->is_threadpercore_initialized = true;
                return *this;
            }

            auto set_core(int core_id, double speed) -> MonoAffineDaemonPlanMaker&{

                if (speed <= 0){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                core_speed_map[core_id] = speed;
                this->is_core_speed_initialized = true;
                return *this;
            }

            auto set_cpu_usage(daemon_kind_t daemon_kind, double cpu_usage) -> MonoAffineDaemonPlanMaker&{

                if (cpu_usage <= 0){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                daemon_usage_map[daemon_kind] = cpu_usage;
                this->is_core_speed_initialized = true;
                return *this;
            }

            auto make_plan() -> std::unordered_map<daemon_kind_t, std::vector<std::vector<int>>>{

                if (!this->is_core_speed_initialized){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!this->is_threadpercore_initialized){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!this->is_daemon_usage_initialized){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                std::unordered_map<int, double> normalized_core_speed_map                                   = this->normalize_core_speed(this->core_speed_map); 
                std::unordered_map<daemon_kind_t, double> normalized_daemon_usage_map                       = this->normalize_daemon_usage(this->daemon_usage_map);
                std::unordered_map<daemon_kind_t, std::unordered_map<int, double>> daemon_affineusage_map   = this->make_daemon_affineusage_map(normalized_core_speed_map, normalized_daemon_usage_map); 

                return this->internal_make_plan(daemon_affineusage_map);
            }
        
        private:

            auto normalize_core_speed(std::unordered_map<int, double> core_speed_map) -> std::unordered_map<int, double>{

                double total = {};

                for (const auto& pair_iter: core_speed_map){
                    total += pair_iter->second;
                }

                for (auto& pair_iter: core_speed_map){
                    pair_iter->second /= total;
                }

                return core_speed_map;
            }

            auto normalize_daemon_usage(std::unordered_map<daemon_kind_t, double> daemon_usage_map) -> std::unordered_map<daemon_kind_t, double>{

                double total = {};

                for (const auto& pair_iter: daemon_usage_map){
                    total += pair_iter->second;
                }

                for (auto& pair_iter: daemon_usage_map){
                    pair_iter->second /= total;
                }

                return daemon_usage_map;
            }

            auto make_daemon_affineusage_map(std::unordered_map<int, double> core_speed_map, 
                                             std::unordered_map<daemon_kind_t, double> daemon_usage_distribution) -> std::unordered_map<daemon_kind_t, std::unordered_map<int, double>>{
                
                std::vector<std::pair<int, double>> core_speed_vec(core_speed_map.begin(), core_speed_map.end());
                std::unordered_map<daemon_kind_t, std::unordered_map<int, double>> rs{};

                for (const auto& pair_iter: daemon_usage_distribution){
                    auto [daemon_kind, demanding_usage] = pair_iter;
                    std::unordered_map<int, double> affine_dist{};

                    while (true){
                        if (core_speed_vec.size() == 0u){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }

                        if (core_speed_vec.back().second > demanding_usage){
                            affine_dist[core_speed_vec.back().first] = demanding_usage;
                            core_speed_vec.back().second -= demanding_usage;
                            demanding_usage = 0u;
                            break;
                        }

                        affine_dist[core_speed_vec.back().first] = core_speed_vec.back().second;
                        demanding_usage -= core_speed_vec.back().second;
                        core_speed_vec.pop_back(); 
                    }

                    rs[daemon_kind] = std::move(affine_dist);
                }

                return rs;
            }

            auto to_thr_group_cpu_set(int core_id, double normalized_core_usage) -> std::vector<std::vector<int>>{

                size_t thr_count = std::max(size_t{1}, static_cast<size_t>(normalized_core_usage * this->thr_count_per_core)); 
                std::vector<std::vector<int>> rs{};

                for (size_t i = 0u; i < thr_count; ++i){
                    rs.push_back(std::vector<int>{core_id});
                }

                return rs;
            } 

            auto internal_make_plan(std::unordered_map<daemon_kind_t, std::unordered_map<int, double>> daemon_affineusage_map) -> std::unordered_map<daemon_kind_t, std::vector<std::vector<int>>>{

                std::unordered_map<daemon_kind_t, std::vector<std::vector<int>>> rs{};

                for (const auto& outer_iter: daemon_affineusage_map){
                    for (const auto& inner_iter: outer_iter->second){
                        std::vector<std::vector<int>> thr_group_cpuset = to_thr_group_cpu_set(inner_iter->first, inner_iter->second); 
                        rs[outer_iter->first].insert(rs[outer_iter->first].end(), thr_group_cpuset.begin(), thr_group_cpuset.end());
                    }
                }

                return rs;
            }
    };

    struct UniformDaemonPlanMaker{ //uniform == core_speed is uniform - this could be achieved externally by kernel spawning virtual core or internally by program wrapping logics around virtual core

        private:

            std::optional<affine_policy_option_t> policy;
            std::optional<double> computing_cpu_usage;
            std::optional<double> kernel_io_cpu_usage;
            std::optional<double> transportation_cpu_usage;
            std::optional<double> heartbeat_cpu_usage; 
            std::optional<size_t> high_parallel_low_compute_thr_per_core;
            std::optional<size_t> low_parallel_high_compute_thr_per_core;

        public:

            auto set_affine_policy(affine_policy_option_t policy) -> UniformDaemonPlanMaker&{

                this->policy = policy;
                return *this;
            }

            auto set_high_parallel_low_compute_thr_per_core(size_t sz) -> UniformDaemonPlanMaker&{

                if (sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->high_parallel_low_compute_thr_per_core = sz;
                return *this;
            }

            auto set_low_parallel_high_compute_thr_per_core(size_t sz) -> UniformDaemonPlanMaker&{
                
                if (sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->low_parallel_high_compute_thr_per_core = sz;
                return *this;
            }

            auto set_computing_cpu_usage(double cpu_usage) -> UniformDaemonPlanMaker&{

                if (cpu_usage <= 0){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->computing_cpu_usage = cpu_usage;
                return *this;
            }

            auto set_kernel_io_cpu_usage(double cpu_usage) -> UniformDaemonPlanMaker&{ //io is high_parallel - to consume kernel packet before it disappears (this is low_compute if use inplace_buffer + one time allocation from kernel) + block thread for network response - 

                if (cpu_usage <= 0){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->kernel_io_cpu_usage = cpu_usage;
                return *this;
            }

            auto set_transportation_cpu_usage(double cpu_usage) -> UniformDaemonPlanMaker&{ //transporation == moving buffer from one place -> another - without actually directly copying the buffer

                if (cpu_usage <= 0){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->transportation_cpu_usage = cpu_usage;
                return *this;
            }

            auto set_heartbeat_cpu_usage(double cpu_usage) -> UniformDaemonPlanMaker&{

                if (cpu_usage <= 0){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->heartbeat_cpu_usage = cpu_usage;
                return *this;
            }

            auto make_plan() -> std::vector<std::tuple<daemon_kind_t, std::optional<std::vector<int>>>>{

                if (!static_cast<bool>(policy)){
                    this->set_affine_policy(kernel_decide_affine_policy);
                    return this->make_plan();
                }

                if (!static_cast<bool>(computing_cpu_usage)){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!static_cast<bool>(kernel_io_cpu_usage)){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!static_cast<bool>(transportation_cpu_usage)){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!static_cast<bool>(heartbeat_cpu_usage)){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!static_cast<bool>(high_parallel_low_compute_thr_per_core)){
                    this->set_high_parallel_low_compute_thr_per_core(max_parallel_performance_hyperthread_count_per_core());
                    return this->make_plan();
                }

                if (!static_cast<bool>(low_parallel_high_compute_thr_per_core)){
                    this->set_low_parallel_high_compute_thr_per_core(max_compute_performance_hyperthread_count_per_core());
                    return this->make_plan();
                }

                if (this->policy == kernel_overwrite_affine_policy){
                    return this->internal_make_overwrite_affine_plan();
                }

                if (this->policy == kernel_decide_affine_policy){
                    return this->internal_make_kerneldecide_affine_plan();
                }

                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
                return {};
            }
        
        private:

            auto internal_make_overwrite_affine_plan() -> std::vector<std::tuple<daemon_kind_t, std::optional<std::vector<int>>>>{

                size_t core_count = std::thread::hardware_concurrency();

                if (core_count == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::UNDEFINED_HARDWARE_CONCURRENCY); //better to throw exception here 
                }
                
                auto rs                         = std::vector<std::tuple<daemon_kind_t, std::optional<std::vector<int>>>>();
                double high_compute_sum         = this->computing_cpu_usage.value() + this->kernel_io_cpu_usage.value();
                double high_parallel_sum        = this->transportation_cpu_usage.value() + this->heartbeat_cpu_usage.value(); 
                double total_sum                = high_compute_sum + high_parallel_sum;
                size_t high_compute_core_count  = core_count * (high_compute_sum / total_sum);
                size_t high_parallel_core_count = core_count - high_compute_core_count;
                
                auto high_compute_plan_maker    = MonoAffineDaemonPlanMaker();
                auto high_parallel_plan_maker   = MonoAffineDaemonPlanMaker();

                for (size_t i = 0u; i < high_compute_core_count; ++i){
                    high_compute_plan_maker.set_core(i, 1);
                }

                for (size_t i = 0u; i < high_parallel_core_count; ++i){
                    high_parallel_plan_maker.set_core(i + high_compute_core_count, 1);
                }

                high_compute_plan_maker.set_thread_per_core(this->low_parallel_high_compute_thr_per_core.value())
                                       .set_cpu_usage(COMPUTING_DAEMON, this->computing_cpu_usage.value())
                                       .set_cpu_usage(IO_DAEMON, this->kernel_io_cpu_usage.value());
                
                high_parallel_plan_maker.set_thread_per_core(this->high_parallel_low_compute_thr_per_core.value())
                                        .set_cpu_usage(TRANSPORATION_DAEMON, this->transportation_cpu_usage.value())
                                        .set_cpu_usage(HEARTBEAT_DAEMON, this->heartbeat_cpu_usage.value());

                for (const auto& outter_iter: high_compute_plan_maker.make_plan()){
                    for (const auto& inner_iter: outer_iter->second){
                        rs.push_back(std::make_tuple(outter_iter->first, inner_iter));
                    }
                }

                for (const auto& outer_iter: high_paralell_plan_maker.make_plan()){
                    for (const auto& inner_iter: outer_iter->second){
                        rs.push_back(std::make_tuple(outter_iter->first, inner_iter));
                    }
                }

                return rs;
            }

            auto internal_make_kerneldecide_affine_plan() -> std::vector<std::tuple<daemon_kind_t, std::optional<std::vector<int>>>>{

            }
    };

    struct Config{
        affine_policy_option_t policy;
        double computing_cpu_usage;
        double io_cpu_usage;
        double transportation_cpu_usage;
        double heartbeat_cpu_usage;
    };

    auto spawn(Config config) -> std::pair<std::unique_ptr<dg::network_concurrency_impl1::DaemonControllerInterface>, std::vector<std::thread::id>>{

        auto thr_vec        = std::vector<std::thread::id>();
        auto daemon_id_map  = std::unordered_map<daemon_kind_t, std::vector<size_t>>{};
        auto id_runner_map  = std::unordered_map<size_t, std::unique_ptr<dg::network_concurrency_impl1::DaemonRunnerInterface>>{};
        auto plan           = UniformDaemonPlanMaker().set_affine_policy(config.policy)
                                                      .set_computing_cpu_usage(config.computing_cpu_usage)
                                                      .set_io_cpu_usage(config.io_cpu_usage)
                                                      .set_transportation_cpu_usage(config.transportation_cpu_usage)
                                                      .set_heartbeat_cpu_usage(config.heartbeat_cpu_usage)
                                                      .make_plan();
        
        for (const auto& thr_metadata: plan){
            auto [daemon_kind, optional_cpuset] = thr_metadata;
            auto [daemon_runner, thr_id] = [&]{
                if (optional_cpuset){
                    return dg::network_concurrency_impl1::DaemonFactory::spawn_affine_daemon_runner(optional_cpuset.value());
                } else{
                    return dg::network_concurrency_impl1::DaemonFactory::spawn_daemon_runner();
                }
            }();

            static_assert(std::has_unique_object_representations_v<std::thread::id>);
            size_t thr_numerical_rep = std::bit_cast<size_t>(thr_id);
            daemon_id_map[daemon_kind].push_back(thr_numerical_rep); 
            id_runner_map[thr_numerical_rep] = std::move(daemon_runner);
            thr_vec.push_back(thr_id);
        }

        auto controller = dg::network_concurrency_impl1::ControllerFactory::spawn_daemon_controller(std::move(daemon_id_map), std::move(id_runner_map));
        return {std::move(controller), std::move(thr_vec)};
    }
}

namespace dg::network_concurrency_impl1_manual{

} 

#endif