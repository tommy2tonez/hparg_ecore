
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

    //we'll write a bad sketch first then we'll be back for tuning

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
    };

    struct WorkOrderContainerInterface{
        virtual ~WorkOrderContainerInterface() noexcept = default;
        virtual void push(WorkOrder) noexcept -> exception_t = 0;
        virtual auto pop(size_t sz) noexcept -> dg::vector<WorkOrder> = 0;
    };

    //we are ditching ticket tmr
    struct WorkTicketControllerInterface{
        virtual ~WorkTicketControllerInterface() noexcept = default;
        virtual auto next_ticket() noexcept -> std::expected<wo_ticketid_t, exception_t> = 0;
        virtual auto mark_completed(wo_ticketid_t, launch_exception_t) noexcept -> exception_t = 0;
        virtual auto add_observer(wo_ticketid_t, std::shared_ptr<std::mutex>) noexcept -> std::expected<launch_exception_t, exception_t> = 0;
        virtual void close_ticket(wo_ticketid_t) noexcept = 0;
    };

    struct AsynchronousDeviceInterface{
        virtual ~AsynchronousDeviceInterface() noexcept = default;
        virtual auto exec(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz) noexcept -> std::expected<void *, exception_t> = 0;
        virtual auto sync(void *) noexcept -> launch_exception_t = 0; //sync must be a nothrow ops - in the sense that it syncs upon task completion to avoid memory corruption - it's supposed to be std::expected<launch_exception_t, exception_t> but because it is a must-sync - it returns the results POST the launch - which is launch_exception_t
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

    class WorkOrderContainer: public virtual WorkOrderContainerInterface{

        private:

            dg::deque<WorkOrder> workorder_vec;
            dg::deque<std::pair<std::shared_ptr<std::mutex>, WorkOrder *>> waiting_queue;
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

                if (this->workorder_vec.size() < this->workorder_vec_capacity){
                    this->workorder_vec.push_back(std::move(wo));
                    return dg::network_exception::SUCCESS;
                }

                return dg::network_exception::RESOURCE_EXHAUSTION;
            }

            auto pop(size_t sz) noexcept -> dg::vector<WorkOrder>{

                std::shared_ptr<std::mutex> pending_mtx = {};
                WorkOrder pending_wo                    = {};
                dg::vector<WorkOrder> rs                = {};
                
                {
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->workorder_vec.empty()){
                        size_t popping_sz   = std::min(sz, this->workorder_vec.size()); 
                        auto first          = this->workorder_vec.begin();
                        auto last           = std::next(first, popping_sz);
                        std::copy(std::make_move_iterator(first), std::make_move_iterator(last), std::back_inserter(rs));
                        this->workorder_vec.erase(first, last);

                        return rs;
                    }

                    pending_mtx = std::make_shared<std::mutex>();
                    this->waiting_queue.push_back(std::make_pair(std::move(pending_mtx), &pending_wo));
                }

                stdx::xlock_guard<std::mutex> lck_grd(*pending_mtx);
                rs.push_back(std::move(pending_wo));

                return rs;
            }
    };

    class WorkTicketController: public virtual WorkTicketControllerInterface{

        private:

            struct TicketResource{
                dg::vector<std::shared_ptr<std::mutex>> observer_vec;
                launch_exception_t launch_err;
                bool is_completed;
            };

            dg::unordered_map<wo_ticketid_t, TicketResource> ticket_resource_map;
            dg::deque<wo_ticketid_t> available_ticket_vec;
            std::unique_ptr<std::mutex> mtx;

        public:

            WorkTicketController(dg::unordered_map<wo_ticketid_t, TicketResource> ticket_resource_map,
                                 dg::deque<wo_ticketid_t> available_ticket_vec,
                                 std::unique_ptr<std::mutex> mtx) noexcept: ticket_resource_map(std::move(ticket_resource_map)),
                                                                            available_ticket_vec(std::move(available_ticket_vec)),
                                                                            mtx(std::move(mtx)){}

            auto next_ticket() noexcept -> std::expected<wo_ticketid_t, exception_t>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (this->available_ticket_vec.empty()){
                    return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                }

                ticket_id_t next_ticket_id = this->available_ticket_vec.front();

                if constexpr(DEBUG_MODE_FLAG){
                    auto map_ptr = this->ticket_resource_map.find(next_ticket_id);

                    if (map_ptr != this->ticket_resource_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->ticket_resource_map.insert(std::make_pair(next_ticket_id, TicketResource{{}, QUEUED_ASYNC_ORDER, false}));
                this->available_ticket_vec.pop_front();

                return next_ticket_id;
            }

            auto mark_completed(wo_ticketid_t ticket_id, launch_exception_t launch_err) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->ticket_resource_map.find(ticket_id); 

                if (map_ptr == this->ticket_resource_map.end()){
                    return dg::network_exception::RESOURCE_ABSENT;
                }

                if (map_ptr->second.is_completed){
                    return dg::network_exception::ASYNC_DOUBLE_COMPLETION;
                }

                map_ptr->second.launch_err      = launch_err;
                map_ptr->second.is_completed    = true;

                for (const auto& mtx_sptr: map_ptr->second.observer_vec){
                    mtx_sptr->unlock();
                }

                map_ptr->second.observer_vec.clear();
                return dg::network_exception::SUCCESS;
            }

            void add_observer(wo_ticketid_t ticket_id, std::shared_ptr<std::mutex> pending_mtx) noexcept -> exception_t{

                if (pending_mtx == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->ticket_resource_map.find(ticket_id);

                if (map_ptr == this->ticket_resource_map.end()){
                    return dg::network_exception::RESOURCE_ABSENT;
                }

                if (map_ptr->second.is_completed){
                    pending_mtx->unlock();
                    return dg::network_exception::SUCCESS;
                }

                map_ptr->second.observer_vec.push_back(std::move(pending_mtx));
                return dg::network_exception::SUCCESS;
            }

            auto launch_status(wo_ticketid_t ticket_id) noexcept -> std::expected<launch_exception_t, exception_t>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->ticket_resource_map.find(ticket_id);

                if (map_ptr == this->ticket_resource_map.end()){
                    return std::unexpected(dg::network_exception::RESOURCE_ABSENT);
                }

                return map_ptr->second.launch_err;
            }

            void close_ticket(wo_ticketid_t ticket_id) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->ticket_resource_map.find(ticket_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->ticket_resource_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort(); 
                    }
                }

                if (!map_ptr->second.is_completed){
                    for (const auto& mtx_sptr: map_ptr->second.observer_vec){
                        mtx_sptr->unlock();
                    }
                }

                this->ticket_resource_map.erase(map_ptr);
                this->available_ticket_vec.push_back(ticket_id);
            }
    };

    //we are aiming to reduce latency of dispatches by removing cuda synchronization overheads
    //we aggregate the orders and dispatch + guarantee consistency of interfaces for future maintaince

    class CudaAsynchronousDevice: public virtual AsynchronousDeviceInterface{

        private:

            struct InternalHandle{
                wo_ticketid_t ticket_id;
            };

            std::shared_ptr<WorkOrderContainerInterface> wo_container;
            std::shared_ptr<WorkTicketControllerInterface> ticket_controller;
        
        public:

            CudaAsynchronousDevice(std::shared_ptr<WorkOrderContainerInterface> wo_container,
                                   std::shared_ptr<WorkTicketControllerInterface> ticket_controller) noexcept: wo_container(std::move(wo_container)),
                                                                                                               ticket_controller(std::move(ticket_controller)){}

            auto exec(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz) noexcept -> std::expected<void *, exception_t>{

                if (executable == nullptr){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                std::expected<bool, exception_t> cuda_status    = dg::network_cuda_controller::ping_cuda();

                if (!cuda_status.has_value()){
                    return std::unexpected(cuda_status.error())
                }

                if (!cuda_status.value()){
                    return std::unexpected(dg::network_exception::CUDA_UNAVAILABLE);
                }

                std::expected<bool, exception_t> env_status     = dg::network_cuda_controller::cuda_is_valid_device(env, env_sz);

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

                auto wo = WorkOrder{ticket_id.value(), std::move(executable), dg::vector<int>(env, env + env_sz)};
                exception_t container_err = this->wo_container->push(std::move(wo));

                if (dg::network_exception::is_failed(container_err)){
                    this->ticket_controller->close_ticket(ticket_id.value());
                    return std::unexpected(container_err);
                }

                InternalHandle * internal_handle    = new InternalHandle{ticket_id.value()}; //TODOs: internalize allocations
                void * void_internal_handle         = internal_handle;

                return void_internal_handle;
            }

            auto sync(void * handle) noexcept -> launch_exception_t{

                InternalHandle * internal_handle        = static_cast<InternalHandle *>(handle);
                std::shared_ptr<std::mutex> thread_mtx  = std::make_shared<std::mutex>();
                thread_mtx->lock();
                this->ticket_controller->add_observer(internal_handle->ticket_id, thread_mtx);
                thread_mtx->lock();

                return this->ticket_controller->launch_status(internal_handle->ticket_id);
            }

            void close_handle(void * handle) noexcept{

                InternalHandle * internal_handle        = static_cast<InternalHandle *>(handle);
                this->ticket_controller->close_ticket(internal_handle->ticket_id);
                delete internal_handle;
            }
    };

    //we are easing the access synchronization + cache invalidating of mutex by using "concurrent_async_device"
    class ConcurrentCudaAsyncDevice: public virtual AsynchronousDeviceInterface{

        private:

            struct InternalHandle{
                void * wo_handle;
                size_t async_device_idx;
            };

            const std::vector<std::unique_ptr<AsynchronousDeviceInterface>> async_device_vec; //we add constness because we are potentially in unprotected concurrent context

        public:

            ConcurrentCudaAsyncDevice(std::vector<std::unique_ptr<AsynchronousDeviceInterface>> async_device_vec) noexcept: async_device_vec(std::move(async_device_vec)){}

            auto exec(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz) noexcept -> std::expected<void *, exception_t>{
                
                assert(stdx::is_pow2(this->async_device_vec.size()));

                size_t random_clue                              = dg::network_randomizer::randomize_int<size_t>();
                size_t async_device_idx                         = random_clue & (this->async_device_vec.size() - 1u);
                std::expected<void *, exception_t> wo_handle    = this->async_device_vec[async_device_idx]->exec(std::move(executable), env, env_sz);

                if (!wo_handle.has_value()){
                    return std::unexpected(wo_handle.error());
                }

                InternalHandle * internal_handle                = new InternalHandle{wo_handle.value(), async_device_idx}; //TODOs: internalize allocations
                void * void_internal_handle                     = internal_handle;

                return void_internal_handle;
            }

            auto sync(void * handle) noexcept -> launch_exception_t{

                InternalHandle * internal_handle = static_cast<InternalHandle *>(handle);
                return this->async_device_vec[internal_handle->async_device_idx]->sync(internal_handle->wo_handle);
            }

            void close_handle(void * handle) noexcept{

                InternalHandle * internal_handle = static_cast<InternalHandle *>(handle);
                this->async_device_vec[internal_handle->async_device_idx]->close_handle(internal_handle->wo_handle);
                delete internal_handle;
            }
    };

    class CudaDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WorkOrderContainerInterface> wo_container;
            std::shared_ptr<WorkTicketControllerInterface> ticket_controller;
            size_t dispatch_sz;

        public:

            CudaDispatcher(std::shared_ptr<WorkOrderContainerInterface> wo_container,
                           std::shared_ptr<WorkTicketControllerInterface> ticket_controller,
                           size_t dispatch_sz) noexcept: wo_container(std::move(wo_container)),
                                                         ticket_controller(std::move(ticket_controller)),
                                                         dispatch_sz(dispatch_sz){}

            bool run_one_epoch() noexcept{

                dg::vector<WorkOrder> wo_vec = this->wo_container->pop(this->dispatch_sz);

                if (wo_vec.empty()){
                    return false;
                }

                auto env = this->extract_environment(wo_vec);
                auto grd = dg::network_cuda_controller::lock_env_guard(env.data(), env.size()); //cuda left a very very few design choices, we do mutex for now

                for (auto& wo: wo_vec){
                    exception_t err = wo.executable->run();

                    if (dg::network_exception::is_failed(err)){
                        this->ticket_controller->mark_completed(wo.ticket_id, exception::mark_completed(exception::make_from_syserr(err)));
                    }
                }

                dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::cuda_synchronize()); //don't know what kind of error is returned/ to return here - must abort - to avoid data races + undefined behaviors  

                for (const auto& wo: wo_vec){
                    this->ticket_controller->mark_completed(wo.ticket_id, exception::mark_completed(exception::make_from_syserr(dg::network_exception::SUCCESS)));
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