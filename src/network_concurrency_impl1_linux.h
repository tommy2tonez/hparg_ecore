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
#include "network_utility.h"

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

    struct StdDaemonRunnableInterface{
        virtual ~StdDaemonRunnableInterface() noexcept = default;
        virtual void run() noexcept = 0; 
    };

    struct WorkerInterface{
        virtual ~WorkerInterface() noexcept = default;
        virtual bool run_one_epoch() noexcept = 0; //precond - exitable-in-all-scenerios execution
    };
    
    struct ReschedulerInterface{
        virtual ~ReschedulerInterface() noexcept = default;
        virtual void reschedule() noexcept = 0;
    };

    struct DaemonRunnerInterface{
        virtual ~DaemonRunnerInterface() noexcept = default;
        virtual void set_worker(std::unique_ptr<WorkerInterface>) noexcept = 0; //precond - valid_unique_ptr<> - need to enforce this by using function signature
    };

    struct DaemonDedicatedRunnerInterface: DaemonRunnerInterface{
        virtual ~DaemonDedicatedRunnerInterface() noexcept = default;
        virtual auto id() noexcept -> std::thread::id = 0;
    };

    struct DaemonControllerInterface{
        virtual ~DaemonControllerInterface() noexcept = default;
        virtual auto _register(daemon_kind_t, std::unique_ptr<WorkerInterface>) noexcept -> std::expected<size_t, exception_t> = 0; //precond - valid_unique_ptr<> - need to enforce this by using function signature
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

    class StdDaemonRunner: public virtual DaemonRunnerInterface,
                           public virtual StdDaemonRunnableInterface{

        private:

            std::shared_ptr<std::atomic<bool>> poison_pill; //whatever - this is extremely hard to implement correctly - ask std - not me
            std::unique_ptr<std::atomic_flag> mtx;
            std::shared_ptr<WorkerInterface> worker;
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

                this->internal_set_worker(std::move(worker));
            }

            void run() noexcept{

                while (!this->load_poison_pill()){
                    bool run_flag = this->internal_get_worker()->run_one_epoch();

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

            void internal_set_worker(std::shared_ptr<WorkerInterface> worker) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->worker = std::move(worker);
            }

            void internal_get_worker() noexcept -> std::shared_ptr<WorkerInterface>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                return this->worker;
            }
    };

    class StdRaiiDaemonRunner: public virtual DaemonDedicatedRunnerInterface{

        private:

            std::shared_ptr<DaemonRunnerInterface> daemon_runner;
            std::unique_ptr<std::thread> thread;
            std::shared_ptr<std::atomic<bool>> runner_poison;
        
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

            auto id() noexcept -> std::thread::id{
 
                return this->thread->get_id();
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

            auto _register(daemon_kind_t daemon_kind, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<size_t, exception_t>{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr = this->daemon_id_map.find(daemon_kind);

                if (map_ptr == this->daemon_id_map.end()){
                    return std::unexpected(dg::network_exception::UNSUPPORTED_DAEMON_KIND);
                }

                if (map_ptr->second.size() == 0u){
                    return std::unexpected(dg::network_exception::NO_DAEMON_RUNNER_AVAILABLE);
                }

                size_t id = map_ptr->second.back();
                map_ptr->second.pop_back(); 
                this->id_runner_map[id]->set_worker(std::move(worker));

                return this->encode(id, daemon_kind);
            }

            auto deregister(size_t encoded) noexcept{

                auto lck_grd            = dg::network_genult::lock_guard(*this->mtx);
                auto [id, daemon_kind]  = this->decode(encoded);
                auto worker             = dg::network_exception_handler::nothrow_log(dg::network_exception::to_cstyle_function(WorkerFactory::spawn_rest)());
                
                this->daemon_id_map[daemon_kind].push_back(id);
                this->id_runner_map[id]->set_worker(std::move(worker));
            }
        
        private:

            auto encode(size_t id, daemon_kind_t daemon_kind) noexcept -> size_t{

                static_assert(sizeof(size_t) + sizeof(daemon_kind_t) <= sizeof(dg::max_unsigned_t));
                static_assert(std::is_unsigned_v<daemon_kind_t>);
                dg::max_unsigned_t promoted_encoded = (static_cast<dg::max_unsigned_t>(id) << (sizeof(daemon_kind_t) * CHAR_BIT)) | static_cast<dg::max_unsigned_t>(daemon_kind); 

                return dg::network_genult::wrap_safe_integer_cast(promoted_encoded);
            }

            auto decode(size_t encoded) noexcept -> std::pair<size_t, daemon_kind_t>{
                
                static_assert(std::is_unsigned_v<daemon_kind_t>);
                size_t id                   = encoded >> (sizeof(daemon_kind_t) * CHAR_BIT);   
                daemon_kind_t daemon_kind   = encoded & low<size_t>(std::integral_constant<size_t, (sizeof(daemon_kind_t) * CHAR_BIT)>{});

                return {id, daemon_kind};
            }
    };

    struct ReschedulerFactory{

        static auto spawn_sleepy_yield_rescheduler(std::chrono::nanoseconds sleep_dur) -> std::unique_ptr<ReschedulerInterface>{

            return std::make_unique<SleepyYieldRescheduler>(sleep_dur);
        }

        static auto spawn_sleepy_rescheduler(std::chrono::nanoseconds sleep_dur) -> std::unique_ptr<ReschedulerInterface>{

            return std::make_unique<SleepyRescheduler>(sleep_dur);
        }
    };

    static void dg_legacy_cpuset_free(cpu_set_t * cpu_set) noexcept{

        CPU_FREE(cpu_set);
    } 

    using dg_legacy_cpuset_free_t = void (*)(cpu_set_t *) noexcept; 

    struct NonLegacyPosixCpuSet{
        std::unique_ptr<cpu_set_t, dg_legacy_cpuset_free_t> legacy_cpusetup;
        size_t alloc_sz;
    };

    struct NonLegacyPosixCPUSetController{

        static auto make_cpuset(size_t cpu_sz) -> std::unique_ptr<NonLegacyPosixCpuSet>{

            std::unique_ptr<cpu_set_t, dg_legacy_cpuset_free_t> legacy_cpusetup = {CPU_ALLOC(cpu_sz), dg_legacy_cpuset_free};

            if (!legacy_cpusetup){
                dg::network_exception::throw_exception(dg::network_exception::OUT_OF_MEMORY);
            }
            
            size_t alloc_sz = CPU_ALLOC_SIZE(cpu_sz);
            CPU_ZERO_S(alloc_sz, legacy_cpusetup.get()); 

            return std::make_unique<NonLegacyPosixCpuSet>(NonLegacyPosixCpuSet{std::move(legacy_cpusetup), alloc_sz});
        }

        static auto add_cpu_to_cpuset(NonLegacyPosixCpuSet * dst, int cpu_id){

            CPU_SET_S(cpu_id, dst->alloc_sz, dst->legacy_cpusetup.get());
        }
    };

    struct StdThreadFactory{

        template <class T>
        static void internal_pthread_setaffinity_np(T&& thr_handle, NonLegacyPosixCpuSet * cpusetp){

            int err = pthread_setaffinity_np(std::forward<T>(thr_handle), cpusetp->alloc_sz, cpusetp->legacy_cpusetup.get());
            
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
        }

        static auto spawn_thread(std::shared_ptr<StdDaemonRunnableInterface> runnable, std::vector<int> cpu_vec) -> std::unique_ptr<std::thread>{

            auto executable = [=]() noexcept{
                runnable->run();
            };

            std::unique_ptr<std::thread> thr_instance       = std::make_unique<std::thread>(std::move(executable));
            std::unique_ptr<NonLegacyPosixCpuSet> cpu_set   = NonLegacyPosixCPUSetController::make_cpuset(cpu_vec.size()); 

            for (int cpu_id: cpu_vec){
                NonLegacyPosixCPUSetController::add_cpu_to_cpuset(cpu_set.get(), cpu_id);
            }
            
            internal_pthread_setaffinity_np(thr_instance->native_handle(), cpu_set.get());
            return thr_instance;
        }

        static auto spawn_thread(std::shared_ptr<StdDaemonRunnableInterface> runnable) -> std::unique_ptr<std::thread>{

            auto executable = [=]() noexcept{
                runnable->run();
            };

            return std::make_unique<std::thread>(std::move(executable));
        }
    };

    struct DaemonRunnerFactory{

        static auto spawn_std_daemon_affine_runner(std::vector<int> cpu_set) -> std::unique_ptr<DaemonDedicatedRunnerInterface>{

            using namespace std::chrono_literals;
             
            auto rescheduler    = ReschedulerFactory::spawn_sleepy_rescheduler(150ms);
            auto mtx            = std::make_unique<std::atomic_flag>();
            mtx->clear();
            auto poison_pill    = std::make_shared<std::atomic<bool>>(bool{false});
            auto worker         = WorkerFactory::spawn_rest();
            auto daemon_runner  = std::make_shared<StdDaemonRunner>(poison_pill, std::move(mtx), std::move(worker), std::move(rescheduler));
            auto thr_instance   = StdThreadFactory::spawn_thread(daemon_runner, cpu_set); 
            auto raii_runner    = std::make_unique<StdRaiiDaemonRunner>(daemon_runner, std::move(thr_instance), poison_pill); 

            return raii_runner;
        } 

        static auto spawn_std_daemon_runner() -> std::unique_ptr<DaemonDedicatedRunnerInterface>{

            using namespace std::chrono_literals;
             
            auto rescheduler    = ReschedulerFactory::spawn_sleepy_rescheduler(150ms);
            auto mtx            = std::make_unique<std::atomic_flag>();
            mtx->clear();
            auto poison_pill    = std::make_shared<std::atomic<bool>>(bool{false});
            auto worker         = WorkerFactory::spawn_rest();
            auto daemon_runner  = std::make_shared<StdDaemonRunner>(poison_pill, std::move(mtx), std::move(worker), std::move(rescheduler));
            auto thr_instance   = StdThreadFactory::spawn_thread(daemon_runner);
            auto raii_runner    = std::make_unique<StdRaiiDaemonRunner>(daemon_runner, std::move(thr_instance), poison_pill); 

            return raii_runner;
        }
    };

    struct ControllerFactory{

        //need to abstractize this
        static auto spawn_daemon_controller(std::unordered_map<daemon_kind_t, std::vector<size_t>> daemon_id_map,
                                            std::unordered_map<size_t, std::unique_ptr<DaemonRunnerInterface>> id_runner_map) -> std::unique_ptr<DaemonControllerInterface>{

            return std::make_unique<DaemonController>(std::move(daemon_id_map), std::move(id_runner_map), std::make_unique<std::mutex>());
        }
    };
} 

#endif