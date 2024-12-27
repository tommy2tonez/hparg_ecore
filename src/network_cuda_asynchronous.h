
#ifndef __CUDA_ASYNCHRONOUS_H__
#define __CUDA_ASYNCHRONOUS_H__

namespace dg::network_cuda_asynchronous{

    class VirtualExecutableInterface{
        
        public:

            virtual ~VirtualExecutableInterface() noexcept = default; 
            virtual auto run() noexcept -> exception_t = 0;
    };

    class Synchronizable{

        public:

            virtual ~Synchronizable() noexcept = default;
            virtual auto sync() noexcept -> exception_t = 0; //sync() guarantees that the operation completes - exception_t, if there is, is about error at launching - cuda_sync() might be corrupted so we aren't catching runtime errors - it is in the precond
    };

    struct SynchronizationStatus{
        exception_t launch_exception;
        std::mutex sync_mtx;
    };

    struct WorkOrder{
        dg::vector<int> env; //this is very futuristic - because usually operations can only be operated in the same environment
        std::unique_ptr<VirtualExecutableInterface> executable;
        std::shared_ptr<SynchronizationStatus> sync_status;
    };

    class WorkOrderContainerInterface{
        
        public:

            virtual ~WorkOrderContainerInterface() noexcept = default;
            virtual void push(WorkOrder) noexcept -> exception_t = 0;
            virtual auto pop(size_t sz) noexcept -> dg::vector<WorkOrder> = 0;
    };

    class AsyncDeviceInterface{
        
        public:

            virtual ~AsyncDeviceInterface() noexcept = default;
            virtual auto exec(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz) noexcept -> std::expected<std::unique_ptr<Synchronizable>, exception_t> = 0;
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

    class TaskSynchronizer: public virtual Synchronizable{

        private:

            std::shared_ptr<SynchronizationStatus> sync_status;
            bool is_synced;

        public:

            TaskSynchronizer(std::shared_ptr<SynchronizationStatus> sync_status) noexcept: sync_status(std::move(sync_status)),
                                                                                           is_synced(false){

                assert(this->sync_status != nullptr);
            }

            TaskSynchronizer(const TaskSynchronizer&) = delete;
            TaskSynchronizer& operator =(const TaskSynchronizer&) = delete;

            TaskSynchronizer(TaskSynchronizer&& other) noexcept: sync_status(std::move(other.sync_status)),
                                                                 is_synced(other.is_synced){

                other.is_synced = true;
            }

            TaskSynchronizer& operator =(TaskSynchronizer&& other) noexcept{

                if (this != std::addressof(other)){
                    this->sync_status   = std::move(other.sync_status);
                    this->is_synced     = other.is_synced;
                    other.is_synced     = true;
                }

                return *this;
            }

            ~TaskSynchronizer() noexcept{

                if (!this->is_synced){
                    this->sync_status->sync_mtx->lock();
                }
            }

            auto sync() noexcept -> exception_t{

                if (!this->is_synced){
                    this->sync_status->sync_mtx->lock();
                    this->is_synced = true;
                }

                return this->sync_status->launch_exception;
            }
    };

    class WorkOrderContainer: public virtual WorkOrderContainerInterface{

        private:

            dg::deque<WorkOrder> workorder_vec;
            dg::deque<std::pair<std::mutex *, WorkOrder *>> waiting_queue;
            size_t workorder_vec_capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            WorkOrderContainer(dg::deque<WorkOrder> workorder_vec,
                               dg::deque<std::pair<std::mutex *, WorkOrder *>> waiting_queue,
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

                std::mutex pending_mtx{};
                WorkOrder pending_wo        = {};
                dg::vector<WorkOrder> rs    = {};

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

                    pending_mtx.lock();
                    this->waiting_queue.push_back(std::make_pair(&pending_mtx, &pending_wo));
                }

                stdx::xlock_guard<std::mutex> lck_grd(pending_mtx);
                rs.push_back(std::move(pending_wo));

                return rs;
            }
    };

    class CudaAsyncDevice: public virtual AsyncDeviceInterface{

        private:

            const std::shared_ptr<WorkOrderContainerInterface> wo_container;
        
        public:

            CudaAsyncDevice(std::shared_ptr<WorkOrderContainerInterface> wo_container) noexcept: wo_container(std::move(wo_container)){}

            auto exec(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz) noexcept -> std::expected<std::unique_ptr<Synchronizable>, exception_t>{

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

                auto sync_status                = std::make_shared<SynchronizationStatus>();
                sync_status->launch_exception   = QUEUED_ASYNC;
                sync_status->sync_mtx->lock();
                auto wo                         = WorkOrder{std::move(executable), dg::vector<int>(env, env + env_sz), sync_status};
                exception_t err                 = this->wo_container->push(std::move(wo));

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                }

                return std::unique_ptr<Synchronizable>(std::make_unique<TaskSynchronizer>(std::move(sync_status)));
            }
    };

    //we are easing the access synchronization + cache invalidating of mutex by using "concurrent_async_device"
    class ConcurrentCudaAsyncDevice: public virtual AsyncDeviceInterface{

        private:

            const std::vector<std::unique_ptr<AsyncDeviceInterface>> async_device_vec; //we add constness because we are potentially in unprotected concurrent context

        public:

            ConcurrentCudaAsyncDevice(std::vector<std::unique_ptr<AsyncDeviceInterface>> async_device_vec) noexcept: async_device_vec(std::move(async_device_vec)){}

            auto exec(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz) noexcept -> std::expected<std::unique_ptr<Synchronizable>, exception_t>{

                assert(stdx::is_pow2(this->async_device_vec.size()));

                size_t random_clue          = dg::network_randomizer::randomize_int<size_t>();
                size_t async_device_idx     = random_clue & (this->async_device_vec.size() - 1u);

                return this->async_device_vec[async_device_idx]->exec(std::move(executable), env, env_sz);
            }
    };

    class CudaDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WorkOrderContainerInterface> wo_container;
            size_t dispatch_sz;

        public:

            CudaDispatcher(std::shared_ptr<WorkOrderContainerInterface> wo_container,
                           size_t dispatch_sz) noexcept: wo_container(std::move(wo_container)),
                                                         dispatch_sz(dispatch_sz){}

            bool run_one_epoch() noexcept{

                dg::vector<WorkOrder> wo_vec    = this->wo_container->pop(this->dispatch_sz);

                if (wo_vec.empty()){
                    return false;
                }

                dg::vector<WorkOrder> sync_vec  = {};
                auto env                        = this->extract_environment(wo_vec);
                auto grd                        = dg::network_cuda_controller::lock_env_guard(env.data(), env.size()); //cuda left a very very few design choices, we do mutex for now

                for (auto& wo: wo_vec){
                    exception_t err = wo.executable->run();

                    if (dg::network_exception::is_failed(err)){
                        wo.sync_status->launch_exception = err;
                        
                        if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                            std::atomic_thread_fence(std::memory_order_release);
                        }

                        wo.sync_status->sync_mtx->unlock();
                    } else{
                        sync_vec.push_back(std::move(wo));
                    }
                }

                dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::cuda_synchronize()); //don't know what kind of error is returned/ to return here - must abort - to avoid data races + undefined behaviors  

                for (auto& wo: sync_vec){
                    wo.sync_status->launch_exception = dg::network_exception::SUCCESS;

                    if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                        std::atomic_thread_fence(std::memory_order_release);
                    }

                    wo.sync_status->sync_mtx->unlock();
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
}

#endif