
namespace dg::network_cuda_asynchronous::exception{

    struct LaunchException{
        exception_t sys_err;
        bool is_completed;
    };

    using launch_exception_t = LaunchException;

    auto make_from_syserr(exception_t sys_err) noexcept -> LaunchException{

        return LaunchException{sys_err, false};
    }

    auto get_syserr(LaunchException err) noexcept -> exception_t{

        return err.sys_err;
    }

    auto mark_completed(LaunchException err) noexcept -> LaunchException{

        return LaunchException{err.sys_err, true};
    }

    auto is_completed(LaunchException err) noexcept -> bool{

        return err.is_completed;
    }
}

namespace dg::network_cuda_asynchronous{

    using wo_ticketid_t         = uint64_t; 
    using launch_exception_t    = exception::launch_exception_t; 

    struct VirtualExecutableInterface{
        virtual ~VirtualExecutableInterface() noexcept = default; 
        virtual auto run() noexcept -> exception_t = 0;
    };

    struct WorkOrder{
        wo_ticketid_t ticket_id;
        dg::vector<int> env; //this is very futuristic - because usually operations can only be operated in the same environment
        std::unique_ptr<VirtualExecutableInterface> executable;
        size_t runtime_complexity;
    };

    struct WorkOrderContainerInterface{
        virtual ~WorkOrderContainerInterface() noexcept = default;
        virtual void push(WorkOrder) noexcept -> exception_t = 0;
        virtual auto pop() noexcept -> WorkOrder = 0;
    };

    struct WorkTicketControllerInterface{
        virtual ~WorkTicketControllerInterface() noexcept = default;
        virtual auto next_ticket() noexcept -> std::expected<ticket_id_t, exception_t> = 0;
        virtual void mark_completed(ticket_id_t, launch_exception_t) noexcept = 0;
        virtual void add_observer(ticket_id_t, std::shared_ptr<std::mutex>) noexcept = 0;
        virtual void close_ticket(ticket_id_t) noexcept = 0;
    };

    struct AsynchronousDeviceInterface{
        virtual ~AsynchronousDeviceInterface() noexcept = default;
        virtual auto exec(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<void *, exception_t> = 0;
        virtual auto sync(void *) noexcept -> launch_exception_t = 0;
        virtual void close_handle(void *) noexcept = 0;
    };

    template <class Executable>
    class CudaKernelVirtualExecutable: public virtual VirtualExecutableInterface{

        private:

            Executable executable;

        public:

            static_assert(std::is_nothrow_destructible_v<Executable>);
            static_assert(noexcept(std::declval<Executable>()()));
            static_assert(std::is_same_v<void, decltype(std::declval<Executable>()())>);

            CudaKernelVirtualExecutable(Executable executable) noexcept(std::is_nothrow_move_constructible_v<Executable>): executable(std::move(executable)){}

            auto run() noexcept -> exception_t{

                cudaGetLastError(); //flush error here
                this->executable();
                return dg::network_exception::wrap_cuda_exception(cudaGetLastError());
            }
    };

    //or we can use a timed mutex - and a raii container here - whichever ways

    class WorkOrderContainer: public virtual WorkOrderContainerInterface{

        private:

            dg::deque<WorkOrder> workorder_vec;
            dg::deque<std::pair<std::shared_ptr<std::mutex>, WorkOrder *>> waiting_queue; //we can still make this - if dg::vector<WorkOrder> is shared_ptr - I feel like this is not quantifiable task - we could have just aggregated the order somewhere else and turns it into one big asynchronous dispatch
            size_t workorder_vec_capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            WorkOrderContainer(dg::deque<WorkOrder> workorder_vec,
                               dg::deque<std::pair<std::shared_ptr<std::mutex>, WorkOrder *>> waiting_queue,
                               size_t workorder_vec_capacity,
                               std::unique_ptr<std::mutex> mtx) noexcept: workorder_vec(std::move(workorder_vec)),
                                                                          waiting_queue(std::move(waiting_queue)),
                                                                          workorder_vec_capacity(workorder_vec_capacity),
                                                                          mtx(std::move(mtx)){}

            auto push(WorkOrder wo) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (!this->waiting_queue.empty()){
                    auto [pending_mtx, fetch_addr] = std::move(this->waiting_queue.front());
                    this->waiting_queue.pop_front();
                    *fetching_addr = std::move(wo);
                    std::atomic_thread_fence(std::memory_order_release);
                    pending_mtx->unlock();
                    return dg::network_exception::SUCCESS;
                }

                if (this->workorder_vec.size() == this->workorder_vec_capacity){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                this->workorder_vec.push_back(std::move(wo));
                return dg::network_exception::SUCCESS;
            }

            auto pop() noexcept -> WorkOrder{

                std::shared_ptr<std::mutex> pending_mtx = {};
                WorkOrder workorder = {};

                {
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->workorder_vec.empty()){
                        auto rs = std::move(this->workorder_vec.front());
                        this->workorder_vec.pop_front();
                        return rs;
                    }

                    pending_mtx = std::make_shared<std::mutex>();
                    this->waiting_queue.push_back(std::make_pair(std::move(pending_mtx), &workorder));
                }

                stdx::xlock_guard<std::mutex> lck_grd(*pending_mtx);
                return workorder;
            }
    };

    class WorkTicketController: public virtual WorkTicketControllerInterface{

        private:

            size_t wo_sz;
            dg::unordered_map<wo_ticketid_t, launch_exception_t> wo_status_map;
            std::unique_ptr<std::mutex> mtx;

        public:
            
            WorkTicketController(size_t wo_sz, 
                                 dg::unordered_map<wo_ticketid_t, launch_exception_t> wo_status_map,
                                 std::unique_ptr<std::mutex> mtx) noexcept: wo_sz(wo_sz),
                                                                            wo_status_map(std::move(wo_status_map)),
                                                                            mtx(std::move(mtx)){}

            auto next_ticket() noexcept -> std::expected<wo_ticketid_t, exception_t>{

                stdx::xlock_guard<stdx::mutex> lck_grd(*this->mtx);
                wo_ticketid_t nxt_id    = dg::network_genult::safe_integer_cast<wo_ticketid_t>(this->wo_sz);
                this->wo_sz             += 1;

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->wo_status_map.find(nxt_id) != this->wo_status_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->wo_status_map[nxt_id] = exception::make_from_syserr(dg::network_exception::SUCCESS);
                return nxt_id;
            }

            void set_status(wo_ticketid_t id, launch_exception_t err) noexcept{

                stdx::xlock_guard<stdx::mutex> lck_grd(*this->mtx);
                auto map_ptr    = this->wo_status_map.find(id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->wo_status_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                ptr->second = err;
            }

            auto get_status(wo_ticketid_t id) noexcept -> launch_exception_t{

                stdx::xlock_guard<stdx::mutex> lck_grd(*this->mtx);
                auto map_ptr    = this->wo_status_map.find(id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->wo_status_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return map_ptr->second;
            }

            void close_ticket(wo_ticketid_t id) noexcept{

                stdx::xlock_guard<stdx::mutex> lck_grd(*this->mtx);
                auto map_ptr    = this->wo_status_map.find(id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->wo_status_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->wo_status_map.erase(map_ptr);
            }
    };

    class KernelLaunchController: public virtual KernelLaunchControllerInterface{

        private:

            std::shared_ptr<WorkOrderContainerInterface> wo_container;
            std::shared_ptr<WorkTicketControllerInterface> ticket_controller;
        
        public:

            KernelLaunchController(std::shared_ptr<WorkOrderContainerInterface> wo_container,
                                   std::shared_ptr<WorkTicketControllerInterface> ticket_controller) noexcept: wo_container(std::move(wo_container)),
                                                                                                               ticket_controller(std::move(ticket_controller)){}
            
            auto launch(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<wo_ticketid_t, exception_t>{

                std::expected<bool, exception_t> env_status = dg::network_cuda_controller::cuda_is_valid_device(env, env_sz);

                if (!env_status.has_value()){
                    return std::unexpected(env_status.error());
                }

                if (!env_status.value()){
                    return std::unexpected(dg::network_exception::UNSUPPORTED_CUDA_DEVICE);
                }

                std::expected<wo_ticketid_t, exception_t> ticket_id = this->ticket_controller->next_ticket();
                
                if (!ticket_id.has_value()){
                    return std::unexpected(ticket_id.error());
                }

                auto wo = WorkOrder{ticket_id.value(), std::move(executable), dg::vector<int>(env, env + env_sz), runtime_complexity};
                this->wo_container->push(std::move(wo));

                return ticket_id.value();
            }

            auto status(wo_ticketid_t id) noexcept -> launch_exception_t{

                return this->ticket_controller->get_status(id);
            }

            void close(wo_ticketid_t id) noexcept{

                this->ticket_controller->close_ticket(id);
            }
    };

    // template <size_t CONCURRENCY_SZ> //deprecate next iteration
    // class ConcurrentKernelLaunchController: public virtual KernelLaunchControllerInterface{

    //     private:

    //         dg::vector<std::unique_ptr<KernelLaunchControllerInterface>> controller_vec;

    //     public:

    //         static_assert(CONCURRENCY_SZ != 0u);
    //         static_assert(CONCURRENCY_SZ <= std::numeric_limits<uint8_t>::max());

    //         ConcurrentKernelLaunchController(dg::vector<std::unique_ptr<KernelLaunchControllerInterface>> controller_vec, 
    //                                          std::integral_constant<size_t, CONCURRENCY_SZ>) noexcept: controller_vec(std::move(controller_vec)){}


    //         auto launch(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<wo_ticketid_t, exception_t>{

    //             size_t thr_idx = dg::network_concurrency::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
    //             std::expected<wo_ticketid_t, exception_t> rs = this->controller_vec[thr_idx]->launch(std::move(executable), env, env_sz, runtime_complexity);

    //             if (!rs.has_value()){
    //                 return std::unexpected(rs.error());
    //             }

    //             return this->encode(rs.value(), thr_idx);
    //         }

    //         auto status(wo_ticketid_t encoded_id) noexcept -> launch_exception_t{

    //             auto [id, thr_id] = this->decode(encoded_id);
    //             return this->controller_vec[thr_id]->status(id);
    //         }

    //         void close(wo_ticketid_t encoded_id) noexcept{

    //             auto [id, thr_id] = this->decode(encoded_id);
    //             this->controller_vec[thr_id]->close(encoded_id);
    //         }
        
    //     private:

    //         auto encode(wo_ticketid_t id, uint8_t thr_id) noexcept -> wo_ticketid_t{

    //             static_assert(std::is_unsigned_v<wo_ticketid_t>);
    //             using promoted_t = dg::max_unsigned_t;
    //             static_assert(sizeof(wo_ticketid_t) + sizeof(uint8_t) <= sizeof(promoted_t));
    //             promoted_t encoded = (static_cast<promoted_t>(id) << (sizeof(uint8_t) * CHAR_BIT)) | static_cast<promoted_t>(thr_id);

    //             return dg::network_genult::safe_integer_cast<wo_ticketid_t>(encoded);
    //         }

    //         auto decode(wo_ticketid_t encoded_id) noexcept -> std::pair<wo_ticketid_t, uint8_t>{

    //             wo_ticketid_t id    = encoded_id >> (sizeof(uint8_t) * CHAR_BIT);
    //             uint8_t thr_id      = encoded_id & low<wo_ticketid_t>(std::integral_constant<size_t, (sizeof(uint8_t) * CHAR_BIT)>{});

    //             return {id, thr_id};
    //         }
    // };

    //alrights - we still think aggregating the orders is this guy responsibility to reduce cuda synchronization overhead - so we'll build a component called leaky aggregator
    //workers will try to pop() values of certain size and push data into this leaky containers - leaky containers is subcripted to a drainer - drainer after timeout will drain the leaky container and move it to another dispatcher of smaller vectorization_sz
    //so we have vectorization_sz of 256 128 64 32 16 8 4 2 1 etc. 
    //we try to vectorize 256 within 50ms - after 50ms the leaky container will be drained into worker of vectorization_sz 128, etc.
    //we'll fill these today
    //we are moving towards low latency dispatches - for most of the cases where the workload inbound is reasonable
    //when the workload inbound is not reasonable - we move to high latency but still processes the orders
    //we also want to aggregate the synchronizables into one sync() order - we'll be back

    class CudaDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WorkOrderContainerInterface> wo_container;
            std::shared_ptr<WorkTicketControllerInterface> ticket_controller;
        
        public:

            CudaDispatcher(std::shared_ptr<WorkOrderContainerInterface> wo_container,
                           std::shared_ptr<WorkTicketControllerInterface> ticket_controller) noexcept: wo_container(std::move(wo_container)),
                                                                                                       ticket_controller(std::move(ticket_controller)){}
            
            bool run_one_epoch() noexcept{

                dg::vector<WorkOrder> wo_vec = this->wo_container->pop();

                if (wo_vec.empty()){
                    return false;
                }

                auto env = this->extract_environment(wo_vec);
                auto grd = dg::network_cuda_controller::lock_env_guard(env.data(), env.size());
                dg::network_cuda_controller::cuda_synchronize();

                for (auto& wo: wo_vec){
                    exception_t err = wo.executable->run();

                    if (dg::network_exception::is_failed(err)){
                        this->ticket_controller->set_status(wo.ticket_id, exception::mark_completed(exception::make_from_syserr(err)));
                    }
                }

                exception_t err = dg::network_cuda_controller::cuda_synchronize();

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(err)); //don't know what kind of error is returned/ to return here - rather abort - to avoid data races + undefined behaviors  
                    std::abort();
                }

                for (const auto& wo: wo_vec){
                    this->ticket_controller->set_status(wo.ticket_id, exception::mark_completed(exception::make_from_syserr(dg::network_exception::SUCCESS)));
                }

                return true;
            }
        
        private:

            auto extract_environment(const dg::vector<WorkOrder>& wo_vec) noexcept -> dg::vector<int>{

                auto env_set = dg::unordered_set<int>{};

                for (const auto& wo: wo_vec){
                    env_set.insert(wo.env.begin(), wo.env.end());
                }

                return dg::vector<int>(env_set.begin(), env_set.end());
            }
    };

    inline std::unique_ptr<KernelLaunchControllerInterface> kernel_launcher{}; 

    template <class Executable>
    auto make_kernel_launch_task(Executable executable) noexcept -> std::unique_ptr<VirtualExecutableInterface>{

        static_assert(std::is_nothrow_move_constructible<Executable>);
        return std::make_unique<CudaKernelVirtualExecutable<Executable>>(std::move(executable));
    }

    auto cuda_launch(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> exception_t{

        std::expected<wo_ticketid_t, exception_t> launch_id = kernel_launcher->launch(std::move(executable), env, env_sz, runtime_complexity);
        
        if (!launch_id.has_value()){
            return launch_id.error();
        }

        launch_exception_t err = {};
        auto synchronizable = [&err, id = launch_id.value()]() noexcept{
            err = kernel_launcher->status(id);
            return exception::is_completed(err);
        };

        dg::network_asynchronous::wait(synchronizable);
        return exception::get_syserr(err);
    }
}