#ifndef __DG_NETWORK_CONCURRENCY_IMPL1_LINUX_H__
#define __DG_NETWORK_CONCURRENCY_IMPL1_LINUX_H__

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <vector>
#include "network_exception.h"
#include "network_log.h"

namespace dg::network_concurrency_impl1_linux::daemon_option_ns{

    using daemon_kind_t = uint8_t; 

    enum daemon_option: daemon_kind_t{
        COMPUTING_DAEMON        = 0u,
        TRANSPORTATION_DAEMON   = 1u,
        IO_DAEMON               = 2u,
        HEARTBEAT_DAEMON        = 3u
    };
} 

namespace dg::network_concurrency_impl1_linux{

    using namespace daemon_option_ns;

    struct WorkerInterface{
        virtual ~WorkerInterface() noexcept = default;
        virtual bool run_one_epoch() noexcept = 0; 
    };
    
    struct ReschedulerInterface{
        virtual ~ReschedulerInterface() noexcept = default;
        virtual void reschedule() noexcept = 0;
    };

    struct DaemonRunnerInterface{
        virtual ~DaemonRunnerInterface() noexcept = default;
        virtual void set_worker(std::unique_ptr<WorkerInterface>) noexcept = 0;
    };

    struct DaemonControllerInterface{
        virtual ~DaemonControllerInterface() noexcept = default;
        virtual auto _register(daemon_kind_t, std::unique_ptr<WorkerInterface>) noexcept -> std::expected<size_t, exception_t> = 0;
        virtual void deregister(size_t) noexcept = 0;
    };

    class RestWorker: public virtual WorkerInterface{

        public:

            bool run_one_epoch() noexcept{
                
                return false;
            }
    };

    struct WorkerFactory{

        static auto spawn_rest() -> std::unique_ptr<WorkerInterface>{

            return std::make_unique<RestWorker>();
        }
    };

    class SleepyRescheduler: public virtual ReschedulerInterface{

        private:

            std::chrono::nanoseconds sleep_dur;
        
        public: 

            SleepyRescheduler(std::chrono::nanoseconds sleep_dur) noexcept: sleep_dur(std::move(sleep_dur)){}

            void reschedule() noexcept{

                std::this_thread::sleep_for(this->sleep_dur);
            }
    };

    class SleepyYieldRescheduler: public virtual ReschedulerInterface{

        private:

            std::chrono::nanoseconds sleep_dur;
        
        public:

            SleepyYieldRescheduler(std::chrono::nanoseconds sleep_dur) noexcept: sleep_dur(std::move(sleep_dur)){}

            void reschedule() noexcept{

                std::this_thread::sleep_for(this->sleep_dur);
                std::this_thread::yield();
            }
    };

    class StdDaemonRunner: public virtual DaemonRunnerInterface{

        private:

            std::shared_ptr<std::atomic<bool>> poison_pill; //whatever - this is hard to implement correctly - ask std - not me
            std::unique_ptr<std::atomic_flag> mtx;
            std::unique_ptr<WorkerInterface> worker;
            std::unique_ptr<ReschedulerInterface> rescheduler; 

        public:

            StdDaemonRunner(std::shared_ptr<std::atomic<bool>> poison_pill,
                            std::unique_ptr<std::atomic_flag> mtx,
                            std::unique_ptr<WorkerInterface> worker,
                            std::unique_ptr<ReschedulerInterface> rescheduler) noexcept: poison_pill(std::move(poison_pill)),
                                                                                         mtx(std::move(mtx)),
                                                                                         worker(std::move(worker)),
                                                                                         rescheduler(std::move(rescheduler)){}

            void set_worker(std::unique_ptr<WorkerInterface> worker) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->worker = std::move(worker);
            }

            void run() noexcept{

                while (!this->load_poison_pill()){
                    auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                    bool run_flag   = this->worker->run_one_epoch();

                    if (!run_flag){
                        this->rescheduler->reschedule();
                    }
                }
            }
        
        private:

            auto load_poison_pill() const noexcept -> bool{

                constexpr size_t DICE_SZ = 8;

                if (dg::network_randomizer::randomize_range(std::integral_constant<size_t, DICE_SZ>{} == 0u)){
                    return this->poison_pill->load(std::memory_order_acq_rel);
                }

                return false;
            }
    };

    class StdRaiiDaemonRunner: public virtual DaemonRunnerInterface{

        private:

            std::shared_ptr<DaemonRunnerInterface> daemon_runner; //daemon_runner is referenced by both std::thread and raii - 
            std::unique_ptr<std::thread> thread;
            std::shared_ptr<std::atomic<bool>> runner_poison; //fine - not a good practice here - yet this is interface implementation - usable for a specific usecase only - the art of engineering is actually being specific - keep it simple stupid - don't over complicate things
        
        public:

            StdRaiiDaemonRunner(std::shared_ptr<DaemonRunnerInterface> daemon_runner, 
                                std::unique_ptr<std::thread> thread,
                                std::shared_ptr<std::atomic<bool>> runner_poison) noexcept: daemon_runner(std::move(daemon_runner)),
                                                                                            thread(std::move(thread)),
                                                                                            runner_poison(std::move(runner_poison)){}

            ~StdRaiiDaemonRunner() noexcept{
                
                this->runner_poison->exchange(true, std::memory_order_acq_rel);
                this->thread->join();
            }

            void set_worker(std::unique_ptr<WorkerInterface> worker) noexcept{

                this->daemon_runner->set_worker(std::move(worker));
            }
    };

    class DaemonController: public virtual DaemonControllerInterface{

        private:

            std::unordered_map<daemon_kind_t, std::vector<size_t>> daemon_id_map;
            std::unordered_map<size_t, std::unique_ptr<DaemonRunnerInterface>> id_runner_map;
            std::unique_ptr<std::mutex> mtx;

        public:

            DaemonController(std::unordered_map<daemon_kind_t, std::vector<size_t>> daemon_id_map,
                             std::unordered_map<size_t, std::unique_ptr<DaemonRunnerInterface>> id_runner_map,
                             std::unique_ptr<std::mutex> mtx) noexcept: daemon_id_map(std::move(daemon_id_map)),
                                                                        id_runner_map(std::move(id_runner_map)),
                                                                        mtx(std::move(mtx)){}

            auto _register(daemon_kind_t daemon, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<size_t, exception_t>{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr = this->daemon_id_map.find(daemon);

                if (map_ptr == this->daemon_id_map.end()){
                    return std::unexpected(dg::network_exception::UNSUPPORTED_DAEMON_MODE);
                }

                if (map_ptr->second.size() == 0u){
                    return std::unexpected(dg::network_exception::NO_DAEMON_EXECUTOR_AVAILABLE);
                }

                size_t id = map_ptr->second.back();
                map_ptr->second.pop_back(); 
                this->id_runner_map[id]->set_worker(std::move(worker));

                return id;
            }

            auto deregister(size_t id) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto worker     = dg::network_exception_handler::nothrow_log(dg::network_exception::to_cstyle_function(WorkerFactory::spawn_rest)());

                this->id_runner_map[id]->set_worker(std::move(worker));
            }
    };

    static auto internal_make_cpuset(std::vector<int> cpu_set) noexcept -> cpu_set_t{ //linux (posix) internal interface 

        //preconds cpu_set - might overflow

        cpu_set_t rs{};
        CPU_ZERO(&rs);

        for (int cpu: cpu_set){
            CPU_SET(cpu, &rs);
        }

        return rs;
    }

    struct ReschedulerFactory{

        static auto spawn_sleepy_yield_rescheduler(std::chrono::nanoseconds sleep_dur) -> std::unique_ptr<ReschedulerInterface>{

            return std::make_unique<SleepyYieldRescheduler>(sleep_dur);
        }

        static auto spawn_sleepy_rescheduler(std::chrono::nanoseconds sleep_dur) -> std::unique_ptr<ReschedulerInterface>{

            return std::make_unique<SleepyRescheduler>(sleep_dur);
        }
    }; 

    struct DaemonFactory{

        static auto spawn_affine_daemon_runner(std::vector<int> cpu_set, std::unique_ptr<ReschedulerInterface> rescheduler) -> std::pair<std::unique_ptr<DaemonRunnerInterface>, std::thread::id>{ //i think its fine to abstractize std::thread::id - and return it here - one could argue to include std::thread::id as part of the interface 
            
            auto px_cpu_set     = internal_make_cpuset(cpu_set);
            auto mtx            = std::make_unique<std::atomic_flag>();
            mtx->clear();
            auto poison_pill    = std::make_shared<std::atomic<bool>>(bool{false});
            auto worker         = WorkerFactory::spawn_rest();
            auto daemon_runner  = std::make_shared<StdDaemonRunner>(poison_pill, std::move(mtx), std::move(worker), std::move(rescheduler));
            auto executable     = [=] noexcept{
                daemon_runner->run();
            };
            auto thr_resource   = std::make_unique<std::thread>(std::move(executable));
            auto thr_id         = thr_resource->get_id();
            int err             = pthread_setaffinity_np(thr_resource->native_handle(), sizeof(cpu_set_t), &px_cpu_set);
            
            if (err != 0){
                if (err == EFAULT){
                    dg::network_exception::throw_exception(dg::network_exception::PTHREAD_EFAULT);
                }
                if (err == EINVAL){
                    dg::network_exception::throw_exception(dg::network_exception::PTHREAD_EINVAL);
                }
                if (err == ESRCH){
                    dg::network_exception::throw_exception(dg::network_exception::PTHREAD_ESRCH);
                }

                dg::network_exception::throw_exception(dg::network_exception::UNIDENTIFIED_EXCEPTION);
            }

            return {std::make_unique<StdRaiiDaemonRunner>(daemon_runner, std::move(thr_resource), poison_pill), std::move(thr_id)};
        }

        static auto spawn_daemon_runner(std::unique_ptr<ReschedulerInterface> rescheduler) -> std::pair<std::unique_ptr<DaemonRunnerInterface>, std::thread::id>{

            auto mtx            = std::make_unique<std::atomic_flag>();
            mtx->clear();
            auto poison_pill    = std::make_shared<std::atomic<bool>>(bool{false});
            auto worker         = WorkerFactory::spawn_rest();
            auto daemon_runner  = std::make_shared<StdDaemonRunner>(poison_pill, std::move(mtx), std::move(worker), std::move(rescheduler));
            auto executable     = [=] noexcept{
                daemon_runner->run();
            };
            auto thr_resource   = std::make_unique<std::thread>(std::move(executable));
            auto thr_id         = thr_resource->get_id();

            return {std::make_unique<StdRaiiDaemonRunner>(daemon_runner, std::move(thr_resource), poison_pill), std::move(thr_id)};
        }
    };

    struct ControllerFactory{

        static auto spawn_daemon_controller(std::unordered_map<daemon_kind_t, std::vector<size_t>> daemon_id_map,
                                            std::unordered_map<size_t, std::unique_ptr<DaemonRunnerInterface>> id_runner_map) -> std::unique_ptr<DaemonControllerInterface>{

            return std::make_unique<DaemonController>(std::move(daemon_id_map), std::move(id_runner_map), std::make_unique<std::mutex>());
        }
    };
} 

#endif