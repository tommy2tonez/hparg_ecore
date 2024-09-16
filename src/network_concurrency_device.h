#ifndef __DG_CONCURRENCY_DEVICE_H__
#define __DG_CONCURRENCY_DEVICE_H__

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <vector>

namespace dg::network_concurrency_device{

    using daemon_t = uint8_t; 

    enum daemon_option: daemon_t{
        COMPUTING_DAEMON        = 0,
        TRANSPORTATION_DAEMON   = 1,
        IO_DAEMON               = 2,
        HEARTBEAT_DAEMON        = 3
    };

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
        virtual auto _register(daemon_t, std::unique_ptr<WorkerInterface>) noexcept -> std::expected<size_t, exception_t> = 0;
        virtual void deregister(size_t) noexcept = 0;
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

    class StdDaemonRunner: public virtual DaemonRunnerInterface{

        private:

            std::unique_ptr<std::atomic_flag> mtx;
            std::shared_ptr<std::atomic<bool>> poison_pill;
            std::unique_ptr<WorkerInterface> worker;
            std::unique_ptr<ReschedulerInterface> rescheduler; 

        public:

            StdDaemonRunner(std::unique_ptr<std::atomic_flag> mtx,
                            std::shared_ptr<std::atomic<bool>> poison_pill, 
                            std::unique_ptr<WorkerInterface> worker,
                            std::unique_ptr<ReschedulerInterface> rescheduler) noexcept: mtx(std::move(mtx)),
                                                                                         poison_pill(std::move(poison_pill)),
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
                    return this->poison_pill->load(std::memory_order_acq_rel); //relaxed qualified - 
                }

                return false;
            }
    };

    class StdDaemonRunnerRaii: public virtual DaemonRunnerInterface{

        private:

            std::unique_ptr<DaemonRunnerInterface> daemon_runner;
            std::unique_ptr<std::thread> thread;
            std::shared_ptr<std::atomic<bool>> daemon_poison;
        
        public:

            StdDaemonRunnerRaii(std::unique_ptr<DaemonRunnerInterface> daemon_runner, 
                                std::unique_ptr<std::thread> thread,
                                std::shared_ptr<std::atomic<bool>> daemon_poison) noexcept: daemon_runner(std::move(daemon_runner)),
                                                                                            thread(std::move(thread)),
                                                                                            daemon_poison(std::move(daemon_poison)){}

            ~StdDaemonRunnerRaii() noexcept{

                this->daemon_poison->exchange(true, std::memory_order_acq_rel); //relaxed qualified - the variable is not detached to any memory-related problem - yet I would like to promote this to release
                this->thread->join();
            }

            void set_worker(std::unique_ptr<WorkerInterface> worker) noexcept{

                this->daemon_runner->set_worker(std::move(worker));
            }
    };

    class DaemonController: public virtual DaemonControllerInterface{

        private:

            std::unordered_map<daemon_t, std::vector<size_t>> daemon_id_map;
            std::vector<std::unique_ptr<DaemonRunnerInterface>> daemon_runner_table;
            std::unique_ptr<std::mutex> mtx;

        public:

            DaemonController(std::unordered_map<daemon_t, std::vector<size_t>> daemon_id_map,
                             std::vector<std::unique_ptr<DaemonRunnerInterface>> daemon_runner_table,
                             std::unique_ptr<std::mutex> mtx) noexcept: daemon_id_map(std::move(daemon_id_map)),
                                                                        daemon_runner_table(std::move(daemon_runner_table)),
                                                                        mtx(std::move(mtx)){}

            auto _register(daemon_t daemon, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<size_t, exception_t>{
                
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
                this->daemon_runner_table[id].set_worker(std::move(worker));

                return id;
            }

            auto deregister(size_t id) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->daemon_runner_table[id].set_worker(WorkerFactory::spawn_rest());
            }
    };
} 

#endif