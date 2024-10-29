#ifndef __DG_NETWORK_CONCURRENCY_IMPL1_H__
#define __DG_NETWORK_CONCURRENCY_IMPL1_H__

#include "stdx.h"
#include <optional>

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

namespace dg::network_concurrency_impl1::planner{

    using namespace dg::network_concurrency_impl1::daemon_option_ns;

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

                this->thr_count_per_core            = thr_count_per_core;
                this->is_threadpercore_initialized  = true;
                
                return *this;
            }

            auto set_core(int core_id, double speed) -> MonoAffineDaemonPlanMaker&{
                
                const double MIN_SPEED  = double{0.001f};
                const double MAX_SPEED  = double{1};

                if (std::clamp(speed, MIN_SPEED, MAX_SPEED) != speed){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }
                
                this->core_speed_map[core_id]   = speed;
                this->is_core_speed_initialized = true;
                
                return *this;
            }

            auto set_cpu_usage(daemon_kind_t daemon_kind, double cpu_usage) -> MonoAffineDaemonPlanMaker&{

                const double MIN_CPU_USAGE  = double{0.001f};
                const double MAX_CPU_USAGE  = double{1};

                if (std::clamp(cpu_usage, MIN_CPU_USAGE, MAX_CPU_USAGE) != cpu_usage){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->daemon_usage_map[daemon_kind] = cpu_usage;
                this->is_daemon_usage_initialized   = true;
                
                return *this;
            }

            auto make_plan() -> std::unordered_map<daemon_kind_t, std::vector<std::vector<int>>>{

                if (!this->is_threadpercore_initialized){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!this->is_core_speed_initialized){
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

                for (const auto& map_pair: core_speed_map){
                    total += map_pair.second;
                }

                for (auto& map_pair: core_speed_map){
                    map_pair.second /= total;
                }

                return core_speed_map;
            }

            auto normalize_daemon_usage(std::unordered_map<daemon_kind_t, double> daemon_usage_map) -> std::unordered_map<daemon_kind_t, double>{

                double total = {};

                for (const auto& map_pair: daemon_usage_map){
                    total += map_pair.second;
                }

                for (auto& map_pair: daemon_usage_map){
                    map_pair.second /= total;
                }

                return daemon_usage_map;
            }

            auto make_daemon_affineusage_map(std::unordered_map<int, double> core_speed_map, std::unordered_map<daemon_kind_t, double> daemon_usage_map) -> std::unordered_map<daemon_kind_t, std::unordered_map<int, double>>{
                
                auto core_speed_vec = std::vector<std::pair<int, double>>(core_speed_map.begin(), core_speed_map.end());
                auto rs             = std::unordered_map<daemon_kind_t, std::unordered_map<int, double>>{};

                for (const auto& map_pair: daemon_usage_map){
                    auto [daemon_kind, demanding_usage] = map_pair;
                    auto affine_dist                    = std::unordered_map<int, double>{};

                    while (true){
                        if (demanding_usage == 0.f){
                            break;
                        }

                        if (core_speed_vec.empty()){
                            break;
                        }

                        if (core_speed_vec.back().second > demanding_usage){
                            affine_dist[core_speed_vec.back().first] = demanding_usage;
                            core_speed_vec.back().second -= demanding_usage;
                            demanding_usage = 0.f;
                            break;
                        }

                        affine_dist[core_speed_vec.back().first] = core_speed_vec.back().second;
                        demanding_usage -= core_speed_vec.back().second;
                        core_speed_vec.pop_back(); 
                    }

                    if (affine_dist.empty()){
                        dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                    }
                    
                    rs[daemon_kind] = std::move(affine_dist);
                }

                for (auto& map_pair: rs){
                    for (auto& usage_pair: map_pair.second){
                        usage_pair.second /= core_speed_map[usage_pair.first];
                    }
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
                    for (const auto& inner_iter: outer_iter.second){
                        std::vector<std::vector<int>> thr_group_cpuset = to_thr_group_cpu_set(inner_iter.first, inner_iter.second); 
                        rs[outer_iter.first].insert(rs[outer_iter.first].end(), thr_group_cpuset.begin(), thr_group_cpuset.end());
                    }
                }

                return rs;
            }
    };

    struct UniformDaemonPlanMaker{ //uniform == core_speed is uniform - this could be achieved externally by kernel spawning virtual core or internally by program wrapping logics around virtual core

        private:

            std::optional<std::vector<int>> core_group;
            std::optional<double> computing_cpu_usage;
            std::optional<double> kernel_io_cpu_usage;
            std::optional<double> transportation_cpu_usage;
            std::optional<double> heartbeat_cpu_usage; 
            std::optional<size_t> high_parallel_thr_per_core;
            std::optional<size_t> high_compute_thr_per_core;

        public:

            auto set_core_group(const std::vector<int>& core_group) -> UniformDaemonPlanMaker&{

                if (core_group.empty()){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->core_group = core_group;
                return *this;
            } 

            auto set_high_parallel_hyperthread_per_core(size_t sz) -> UniformDaemonPlanMaker&{

                if (sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->high_parallel_thr_per_core = sz;
                return *this;
            }

            auto set_high_compute_hyperthread_per_core(size_t sz) -> UniformDaemonPlanMaker&{
                
                if (sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->high_compute_thr_per_core = sz;
                return *this;
            }

            auto set_computing_cpu_usage(double cpu_usage) -> UniformDaemonPlanMaker&{

                if (cpu_usage <= 0.f){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->computing_cpu_usage = cpu_usage;
                return *this;
            }

            auto set_kernel_io_cpu_usage(double cpu_usage) -> UniformDaemonPlanMaker&{ //io is high_parallel - to consume kernel packet before it disappears (this is low_compute if use inplace_buffer + one time allocation from kernel) + block thread for network response - 

                if (cpu_usage <= 0.f){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->kernel_io_cpu_usage = cpu_usage;
                return *this;
            }

            auto set_transportation_cpu_usage(double cpu_usage) -> UniformDaemonPlanMaker&{ //transporation == moving buffer from one place -> another - without actually directly copying the buffer

                if (cpu_usage <= 0.f){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->transportation_cpu_usage = cpu_usage;
                return *this;
            }

            auto set_heartbeat_cpu_usage(double cpu_usage) -> UniformDaemonPlanMaker&{

                if (cpu_usage <= 0.f){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                this->heartbeat_cpu_usage = cpu_usage;
                return *this;
            }

            auto make_plan() -> std::unordered_map<daemon_kind_t, std::vector<std::vector<int>>>{

                if (!this->core_group.has_value()){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!this->computing_cpu_usage.has_value()){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!this->kernel_io_cpu_usage.has_value()){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!this->transportation_cpu_usage.has_value()){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!this->heartbeat_cpu_usage.has_value()){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!this->high_parallel_thr_per_core.has_value()){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                if (!this->high_compute_thr_per_core.has_value()){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                return this->internal_make_plan();
            }
        
        private:

            auto internal_make_plan() -> std::unordered_map<daemon_kind_t, std::vector<std::vector<int>>>{

                size_t core_count               = this->core_group->size();
                auto rs                         = std::unordered_map<daemon_kind_t, std::vector<std::vector<int>>>();;
                double high_compute_sum         = this->computing_cpu_usage.value() + this->kernel_io_cpu_usage.value();
                double high_parallel_sum        = this->transportation_cpu_usage.value() + this->heartbeat_cpu_usage.value(); 
                double total_sum                = high_compute_sum + high_parallel_sum;
                size_t high_compute_core_count  = core_count * (high_compute_sum / total_sum);
                size_t high_parallel_core_count = core_count - high_compute_core_count;
                auto high_compute_plan_maker    = MonoAffineDaemonPlanMaker();
                auto high_parallel_plan_maker   = MonoAffineDaemonPlanMaker();

                if (high_compute_core_count + high_parallel_core_count != core_count){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                for (size_t i = 0u; i < high_compute_core_count; ++i){
                    high_compute_plan_maker.set_core(this->core_group->operator[](i), 1);
                }

                for (size_t i = 0u; i < high_parallel_core_count; ++i){
                    high_parallel_plan_maker.set_core(this->core_group->operator[](i + high_compute_core_count), 1);
                }

                auto high_compute_plan  = high_compute_plan_maker.set_thread_per_core(this->high_compute_thr_per_core.value())
                                                                 .set_cpu_usage(COMPUTING_DAEMON, this->computing_cpu_usage.value())
                                                                 .set_cpu_usage(IO_DAEMON, this->kernel_io_cpu_usage.value())
                                                                 .make_plan();
                
                auto high_parallel_plan = high_parallel_plan_maker.set_thread_per_core(this->high_parallel_thr_per_core.value())
                                                                  .set_cpu_usage(TRANSPORTATION_DAEMON, this->transportation_cpu_usage.value())
                                                                  .set_cpu_usage(HEARTBEAT_DAEMON, this->heartbeat_cpu_usage.value())
                                                                  .make_plan();
                
                rs.insert(high_compute_plan.begin(), high_compute_plan.end());
                rs.insert(high_parallel_plan.begin(), high_parallel_plan.end());

                return rs;
            }
    };
     
    static auto vectorize_plan(std::unordered_map<daemon_kind_t, std::vector<std::vector<int>>> plan) -> std::vector<std::pair<daemon_kind_t, std::vector<int>>>{

        std::vector<std::pair<daemon_kind_t, std::vector<int>>> rs{};

        for (const auto& pair_iter: plan){
            for (const std::vector<int>& cpuset: pair_iter.second){
                rs.push_back(std::make_pair(pair_iter.first, cpuset));
            }
        }

        return rs;
    }

    struct Config{
        double computing_cpu_usage;
        double io_cpu_usage;
        double transportation_cpu_usage;
        double heartbeat_cpu_usage;
        size_t high_parallel_hyperthread_per_core;
        size_t high_compute_hyperthread_per_core;
        std::vector<int> uniform_affine_group;
    };

    auto spawn(Config config) -> std::pair<std::unique_ptr<dg::network_concurrency_impl1::DaemonControllerInterface>, std::vector<std::thread::id>>{

        auto thr_vec        = std::vector<std::thread::id>();
        auto daemon_vec     = std::vector<std::pair<std::unique_ptr<dg::network_concurrency_impl1::DaemonRunnerInterface>, daemon_kind_t>>{};
        auto plan           = UniformDaemonPlanMaker().set_computing_cpu_usage(config.computing_cpu_usage)
                                                      .set_kernel_io_cpu_usage(config.io_cpu_usage)
                                                      .set_transportation_cpu_usage(config.transportation_cpu_usage)
                                                      .set_heartbeat_cpu_usage(config.heartbeat_cpu_usage)
                                                      .set_high_parallel_hyperthread_per_core(config.high_parallel_hyperthread_per_core)
                                                      .set_high_compute_hyperthread_per_core(config.high_compute_hyperthread_per_core)
                                                      .set_core_group(config.uniform_affine_group)
                                                      .make_plan();
        
        for (auto thr_metadata: vectorize_plan(plan)){
            auto [daemon_kind, cpuset] = thr_metadata;
            std::unique_ptr<dg::network_concurrency_impl1::DaemonDedicatedRunnerInterface> runner{};

            if (cpuset.empty()){
                runner = dg::network_concurrency_impl1::DaemonRunnerFactory::spawn_std_daemon_runner();
            } else{
                runner = dg::network_concurrency_impl1::DaemonRunnerFactory::spawn_std_daemon_affine_runner(cpuset);
            }

            std::thread::id thr_id = runner->id();
            thr_vec.push_back(thr_id);
            daemon_vec.push_back(std::make_pair(std::move(runner), daemon_kind));
        }

        auto controller = dg::network_concurrency_impl1::ControllerFactory::spawn_daemon_controller(std::move(daemon_vec));
        return {std::move(controller), std::move(thr_vec)};
    }
}

#endif