#ifndef __DG_NETWORK_HOST_ASYNCHRONOUS_H__
#define __DG_NETWORK_HOST_ASYNCHRONOUS_H__

#include <memory>
#include <unordered_map>
#include <memory>

namespace dg::network_host_asynchronous{

    //we'll add the std::memory_order_relaxed synchronization - this is a very important optimizable 
    //alright, this is a very hard task
    //we need to rebuild this to take batches of WorkOrder, we need to be considerate about the memory orderings, we'll revisit this tomorrow
    //we'll stick with unique_ptr<> for best practices, yet we need to build a better affined allocator, such is like the stack_allocation yet does malloc + free on sz_cap or interval cap, or timeout, says 1MB/ free, or 1MB/ malloc, we have good partial deallocator (fragmentation management) internally, we just need to keep the allocation lifetimes under control 
    //note that we can allocate from one affined allocator, and deallocate on another
    //this is expected, yet we attempt to further affine things by doing aggregations

    using async_device_id_t = size_t;

    class WorkOrder{

        public:

            virtual ~WorkOrder() noexcept = default;
            virtual void run() noexcept = 0;
    };

    class Synchronizable{

        public:

            virtual ~Synchronizable() noexcept = default;
            virtual void sync() noexcept = 0;
    };

    class WorkOrderContainerInterface{

        public:

            virtual ~WorkOrderContainerInterface() noexcept = default;
            virtual auto push(std::unique_ptr<WorkOrder>) noexcept -> exception_t = 0;
            virtual auto pop() noexcept -> std::unique_ptr<WorkOrder> = 0; 
    };

    class AsyncDeviceInterface{

        public:

            virtual ~AsyncDeviceInterface() noexcept = default;
            virtual auto exec(std::unique_ptr<WorkOrder>) noexcept -> exception_t = 0;
    };

    class AsyncDeviceXInterface{

        public:

            virtual ~AsyncDeviceXInterface() noexcept = default;
            virtual auto exec(std::unique_ptr<WorkOrder>) noexcept -> std::expected<std::unique_ptr<Synchronizable>, exception_t> = 0;
    };

    class WorkLoadEstimatorInterface{

        public:

            virtual ~WorkLoadEstimatorInterface() noexcept = default;
            virtual auto estimate(size_t est_flops) noexcept -> std::expected<size_t, exception_t> = 0;
    };

    class LoadBalancerInterface{

        public:

            virtual ~LoadBalancerInterface() noexcept = default;
            virtual auto open_handle(size_t est_flops) noexcept -> std::expected<void *, exception_t> = 0;
            virtual auto get_async_device_id(void *) noexcept -> async_device_id_t = 0;
            virtual void close_handle(void *) noexcept = 0;
    };

    class LoadBalancedAsyncDeviceXInterface{

        public:

            virtual ~LoadBalancedAsyncDeviceXInterface() noexcept = default;
            virtual auto exec(std::unique_ptr<WorkOrder>, size_t est_flops) noexcept -> std::expected<std::unique_ptr<Synchronizable>, exception_t> = 0;
    };

    template <class Lambda>
    class LambdaWrappedWorkOrder: public virtual WorkOrder{

        private:

            Lambda lambda;

        public:

            static_assert(std::is_nothrow_destructible_v<Lambda>);
            // static_assert(std::is_nothrow_invocable_v<Lambda>);

            LambdaWrappedWorkOrder(Lambda lambda) noexcept(noexcept(std::is_nothrow_move_constructible_v<Lambda>)): lambda(std::move(lambda)){}

            void run() noexcept(noexcept(std::is_nothrow_invocable_v<Lambda>)){ //we let the compiler to solve the polymorphism overriding issue

                this->lambda();
            }
    };

    template <class Lambda>
    auto make_virtual_work_order(Lambda lambda) noexcept(noexcept(std::is_nothrow_move_constructible_v<Lambda>)) -> std::unique_ptr<WorkOrder>{

        return std::make_unique<LambdaWrappedWorkOrder<Lambda>>(std::move(lambda)); //TODOs: internalize allocations - we don't accept memory exhaustion because that's a major source of bugs - and we can't be too cautious catching every memory allocations - it clutters the code
    }

    class TaskSynchronizer: public virtual Synchronizable{

        private:

            std::shared_ptr<std::mutex> mtx;
            bool is_synced; 

        public:

            TaskSynchronizer(std::shared_ptr<std::mutex> mtx) noexcept: mtx(std::move(mtx)),
                                                                        is_synced(false){
                assert(this->mtx != nullptr);
            }

            TaskSynchronizer(const TaskSynchronizer&) = delete;
            TaskSynchronizer& operator =(const TaskSynchronizer&) = delete;

            TaskSynchronizer(TaskSynchronizer&& other) noexcept: mtx(std::move(other.mtx)),
                                                                 is_synced(other.is_synced){

                other.is_synced = true;
            }

            TaskSynchronizer& operator =(TaskSynchronizer&& other) noexcept{

                if (this != std::addressof(other)){
                    this->mtx       = std::move(other.mtx);
                    this->is_synced = other.is_synced;
                    other.is_synced = true;
                }

                return *this;
            }

            ~TaskSynchronizer() noexcept{

                if (!this->is_synced){
                    this->mtx->lock();
                }
            }

            void sync() noexcept{

                if (!this->is_synced){
                    this->mtx->lock();
                    this->is_synced = true;
                }
            }
    };

    class WorkOrderContainer: public virtual WorkOrderContainerInterface{

        private:

            dg::deque<std::unique_ptr<WorkOrder>> wo_vec;
            dg::deque<std::pair<std::mutex *, std::unique_ptr<WorkOrder> *>> waiting_vec;
            size_t wo_vec_capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            WorkOrderContainer(dg::deque<std::unique_ptr<WorkOrder>> wo_vec,
                               dg::deque<std::pair<std::mutex *, std::unique_ptr<WorkOrder> *>> waiting_vec,
                               size_t wo_vec_capacity,
                               std::unique_ptr<std::mutex> mtx) noexcept: wo_vec(std::move(wo_vec)),
                                                                          waiting_vec(std::move(waiting_vec)),
                                                                          wo_vec_capacity(wo_vec_capacity),
                                                                          mtx(std::move(mtx)){}

            auto push(std::unique_ptr<WorkOrder> wo) noexcept -> exception_t{

                if (wo == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (!this->waiting_vec.empty()){
                    auto [pending_mtx, fetching_addr] = std::move(this->waiting_vec.front());
                    this->waiting_vec.pop_front();
                    *fetching_addr = std::move(wo);
                    std::atomic_thread_fence(std::memory_order_release);
                    pending_mtx->unlock();
                    return dg::network_exception::SUCCESS;
                }

                if (this->wo_vec.size() < this->wo_vec_capacity){
                    this->wo_vec.push_back(std::move(wo));
                    return dg::network_exception::SUCCESS;
                }

                return dg::network_exception::RESOURCE_EXHAUSTION;
            }

            auto pop() noexcept -> std::unique_ptr<WorkOrder>{

                std::mutex pending_mtx{};
                std::unique_ptr<WorkOrder> wo = {};

                {
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->wo_vec.empty()){
                        auto rs = std::move(this->wo_vec.front());
                        this->wo_vec.pop_front();
                        return rs;
                    }

                    pending_mtx.lock();
                    this->waiting_vec.push_back(std::make_pair(&pending_mtx, &wo));
                }

                stdx::xlock_guard<std::mutex> lck_grd(pending_mtx);
                return wo;
            }
    };

    class AsyncOrderExecutor: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WorkOrderContainerInterface> wo_container;

        public:

            AsyncOrderExecutor(std::shared_ptr<WorkOrderContainerInterface> wo_container) noexcept: wo_container(std::move(wo_container)){}

            bool run_one_epoch() noexcept{

                std::unique_ptr<WorkOrder> wo = this->wo_container->pop();

                if (wo == nullptr){
                    return false;
                }

                wo->run();
                return true;
            }
    };

    class AsyncHeartBeatWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WorkOrderContainerInterface> wo_container;
            std::chrono::nanoseconds max_timeout;
            std::shared_ptr<std::atomic<intmax_t>> last_heartbeat_in_utc_nanoseconds;

        public:

            AsyncHeartBeatWorker(std::shared_ptr<WorkOrderContainerInterface> wo_container,
                                 std::chrono::nanoseconds max_timeout,
                                 std::shared_ptr<std::atomic<intmax_t>> last_heartbeat_in_utc_nanoseconds) noexcept: wo_container(std::move(wo_container)),
                                                                                                                     max_timeout(std::move(max_timeout)),
                                                                                                                     last_heartbeat_in_utc_nanoseconds(std::move(last_heartbeat_in_utc_nanoseconds)){}

            bool run_one_epoch() noexcept{

                //we assume that the concurrency is not corrupted but the usage of asynchronous device is corrupted
                //such can be bad asynchronous work_order (deadlocks or too compute heavy) and renders the asynchronous device useless - we avoid that by having a heartbeat worker to declare a certain latency

                std::chrono::nanoseconds expiry = std::chrono::nanoseconds(last_heartbeat_in_utc_nanoseconds->load(std::memory_order_relaxed)) + this->max_timeout;
                std::chrono::nanoseconds now    = stdx::utc_timestamp();

                if (expiry < now){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort(); //there is no better resolution than to abort in this case - this is a severe error
                }

                auto task = [ticker = this->last_heartbeat_in_utc_nanoseconds]() noexcept{
                    ticker->exchange(static_cast<intmax_t>(stdx::utc_timestamp().count()), std::memory_order_relaxed); //TODOs: assume intmax_t safe_cast
                };
                auto virtual_task = make_virtual_work_order(std::move(task));
                this->wo_container->push(std::move(virtual_task)); //

                return true;
            }
    };

    class AsyncDevice: public virtual AsyncDeviceInterface{

        private:

            const std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            const std::shared_ptr<WorkOrderContainerInterface> wo_container;

        public:

            AsyncDevice(std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                        std::shared_ptr<WorkOrderContainerInterface> wo_container) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                                             wo_container(std::move(wo_container)){}


            auto exec(std::unique_ptr<WorkOrder> wo) noexcept -> exception_t{

                return this->wo_container->push(std::move(wo));
            }
    };

    class AsyncDeviceX: public virtual AsyncDeviceXInterface{

        private:

            const std::unique_ptr<AsyncDeviceInterface> async_device;

        public:

            AsyncDeviceX(std::unique_ptr<AsyncDeviceInterface> async_device) noexcept: async_device(std::move(async_device)){}

            auto exec(std::unique_ptr<WorkOrder> work_order) noexcept -> std::expected<std::unique_ptr<Synchronizable>, exception_t>{

                if (work_order == nullptr){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                auto mtx_sptr           = std::make_shared<std::mutex>(); //TODOs: internalize allocations
                mtx_sptr->lock();
                auto task               = [mtx_sptr, work_order_arg = std::move(work_order)]() noexcept{
                    work_order_arg->run();

                    if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                    } else{
                        std::atomic_thread_fence(std::memory_order_release);
                    }
       
                    mtx_sptr->unlock();
                };
                auto virtual_task       = make_virtual_work_order(std::move(task));
                exception_t async_err   = this->async_device->exec(std::move(virtual_task));

                if (dg::network_exception::is_failed(async_err)){
                    return std::unexpected(async_err);
                }

                return std::unique_ptr<Synchronizable>(std::make_unique<TaskSynchronizer>(std::move(mtx_sptr))); //TODOs: internalize allocations
            }
    };

    struct UniformLoadBalancerHeapNode{
        async_device_id_t async_device_id;
        size_t current_load;
        size_t max_load;
        size_t heap_idx;
    };

    class UniformLoadBalancer: public virtual LoadBalancerInterface{

        private:

            struct InternalHandle{
                async_device_id_t async_device_id;
                UniformLoadBalancerHeapNode * heap_node;
                size_t task_load;
            };

            std::vector<std::unique_ptr<UniformLoadBalancerHeapNode>> load_balance_heap;
            std::unique_ptr<WorkLoadEstimatorInterface> estimator;
            std::unique_ptr<std::mutex> mtx;

        public:

            UniformLoadBalancer(std::vector<std::unique_ptr<UniformLoadBalancerHeapNode>> load_balance_heap,
                                std::unique_ptr<WorkLoadEstimatorInterface> estimator,
                                std::unique_ptr<std::mutex> mtx) noexcept: load_balance_heap(std::move(load_balance_heap)),
                                                                           estimator(std::move(estimator)),
                                                                           mtx(std::move(mtx)){}

            auto open_handle(size_t est_flops) noexcept -> std::expected<void *, exception_t>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                std::expected<size_t, exception_t> est_workload = this->estimator->estimate(est_flops); 

                if (!est_workload.has_value()){
                    return std::unexpected(est_workload.error());
                }

                UniformLoadBalancerHeapNode * front_node = this->load_balance_heap.front().get();

                if (front_node->current_load + est_workload.value() > front_node->max_load){
                    return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                }

                front_node->current_load       += est_workload.value();
                this->push_down_at(0u);
                InternalHandle * dynamic_handle = new InternalHandle{front_node->async_device_id, front_node, est_workload.value()}; //TODOs: internalize allocation
                void * void_dynamic_handle      = dynamic_handle;

                return void_dynamic_handle;
            }

            auto get_async_device_id(void * handle) noexcept{

                return static_cast<InternalHandle *>(stdx::safe_ptr_access(handle))->async_device_id;
            } 

            void close_handle(void * handle) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                InternalHandle * internal_handle        = static_cast<InternalHandle *>(stdx::safe_ptr_access(handle));
                UniformLoadBalancerHeapNode * cur_node  = internal_handle->heap_node;
                cur_node->current_load                 -= internal_handle->task_load;
                this->push_up_at(cur_node->heap_idx);

                delete internal_handle;
            }

        private:

            void push_down_at(size_t idx) noexcept{

                size_t c = idx * 2 + 1;

                if (c >= this->load_balance_heap.size()){
                    return;
                }

                if (c + 1 < this->load_balance_heap.size() && this->load_balance_heap[c + 1]->current_load < this->load_balance_heap[c]->current_load){
                    c += 1;
                }

                if (this->load_balance_heap[idx]->current_load <= this->load_balance_heap[c]->current_load){
                    return;
                }

                std::swap(this->load_balance_heap[idx]->heap_idx, this->load_balance_heap[c]->heap_idx);
                std::swap(this->load_balance_heap[idx], this->load_balance_heap[c]);
                this->push_down_at(c);
            }

            void push_up_at(size_t idx) noexcept{

                if (idx == 0u){
                    return;
                }

                size_t c = (idx - 1) >> 1;

                if (this->load_balance_heap[c]->current_load <= this->load_balance_heap[idx]->current_load){
                    return;
                }

                std::swap(this->load_balance_heap[c]->heap_idx, this->load_balance_heap[idx]->heap_idx);
                std::swap(this->load_balance_heap[c], this->load_balance_heap[idx]);
                this->push_up_at(c);
            }
    };

    class DistributedLoadBalancer: public virtual LoadBalancerInterface{

        private:

            struct InternalHandle{
                void * load_balancer_handle;
                size_t load_balancer_idx;
            };

            const std::vector<std::unique_ptr<LoadBalancerInterface>> load_balancer_vec;

        public:

            DistributedLoadBalancer(std::vector<std::unique_ptr<LoadBalancerInterface>> load_balancer_vec) noexcept: load_balancer_vec(std::move(load_balancer_vec)){}

            auto open_handle(size_t est_flops) noexcept -> std::expected<void *, exception_t>{

                assert(stdx::is_pow2(this->load_balancer_vec.size()));

                size_t random_clue                                      = dg::network_randomizer::randomize_int<size_t>();
                size_t balancer_idx                                     = random_clue & (this->load_balancer_vec.size() - 1u);
                std::expected<void *, exception_t> load_balancer_handle = this->load_balancer_vec[balancer_idx]->open_handle(est_flops);

                if (!load_balancer_handle.has_value()){
                    return std::unexpected(load_balancer_handle.error());
                }

                InternalHandle * internal_handle    = new InternalHandle{load_balancer_handle.value(), balancer_idx}; //TODOs: internalize allocations
                void * void_internal_handle         = internal_handle;

                return void_internal_handle;
            }

            auto get_async_device_id(void * handle) noexcept -> async_device_id_t{

                InternalHandle * internal_handle = static_cast<InternalHandle *>(stdx::safe_ptr_access(handle));
                return this->load_balancer_vec[internal_handle->load_balancer_idx]->get_async_device_id(internal_handle->load_balancer_handle);
            }

            void close_handle(void * handle) noexcept{

                InternalHandle * internal_handle = static_cast<InternalHandle *>(stdx::safe_ptr_access(handle));
                this->load_balancer_vec[internal_handle->load_balancer_idx]->close_handle(internal_handle->load_balancer_handle);
                delete internal_handle;
            }
    };

    class LoadBalancedAsyncDeviceX: public virtual LoadBalancedAsyncDeviceXInterface{

        private:

            const std::unordered_map<async_device_id_t, std::unique_ptr<AsyncDeviceXInterface>> async_device_map;
            const std::shared_ptr<LoadBalancerInterface> load_balancer;

        public:

            LoadBalancedAsyncDeviceX(std::unordered_map<async_device_id_t, std::unique_ptr<AsyncDeviceXInterface>> async_device_map,
                                     std::unique_ptr<LoadBalancerInterface> load_balancer) noexcept: async_device_map(std::move(async_device_map)),
                                                                                                     load_balancer(std::move(load_balancer)){}

            auto exec(std::unique_ptr<WorkOrder> work_order, size_t est_flops) noexcept -> std::expected<std::unique_ptr<Synchronizable>, exception_t>{

                if (work_order == nullptr){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                std::expected<void *, exception_t> load_balance_handle = this->load_balancer->open_handle(est_flops);

                if (!load_balance_handle.has_value()){
                    return std::unexpected(load_balance_handle.error());
                }

                async_device_id_t async_device_id = this->load_balancer->get_async_device_id(load_balance_handle.value());
                auto map_ptr = this->async_device_map.find(async_device_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->async_device_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto task           = [load_balancer_arg = this->load_balancer, work_order_arg = std::move(work_order), load_balance_handle_arg = load_balance_handle.value()]() noexcept{
                    work_order_arg();
                    std::atomic_signal_fence(std::memory_order_release); //alright - this is very important - otherwise we are languagely incorrect
                    load_balancer_arg->close_handle(load_balance_handle_arg);
                };
                auto virtual_task   = make_virtual_work_order(std::move(task));
                std::expected<std::unique_ptr<Synchronizable>, exception_t> syncer = map_ptr->second->exec(std::move(virtual_task));

                if (!syncer.has_value()){
                    this->load_balancer->close_handle(load_balance_handle.value());
                    return std::unexpected(syncer.error());
                }

                return syncer;
            }
    };

    class MemoryUnsafeSynchronizer{

        private:

            dg::vector<std::unique_ptr<Synchronizable>> synchronizable_vec;
        
        public:

            auto add(std::unique_ptr<Synchronizable> syncable) noexcept -> exception_t{

                if (syncable == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                this->synchronizable_vec.push_back(std::move(syncable));
                return dg::network_exception::SUCCESS;
            }

            void sync() noexcept{

                for (auto& synchronizable: this->synchronizable_vec){
                    synchronizable->sync();
                }

                this->synchronizable_vec.clear();
            }
    };

    //for the sake of simplicity, we want to declare 3 types of variables
    //local variables: the normal variables that referenced by the current_thread, it's prefetch, postfetch, etc. fetch aren't dependent on another variables - WLOG, a = foo(); b = bar(); c = foobar(); bar() may or may not invoke concurrent transaction, foo() and bar() are normal functions then a, b, c are local variables
    //atomic variables
    //concurrent variables: the variables whose states are dependent on atomic variables (usually for serialized accesses)
    //memory ordering is about concurrent variables, atomic variables and their relationships - we don't care about local variables

    //this is actually hard
    //assume two scenerios 
    //first  - we are dispatching local variables -> asynchronous device
    //second - we are dispatching concurrent variables -> asynchronous device
    //we want to prove that the memory ordering stategy of this works - such is there is no mis-prefetch - if sync() is in the same scope of the local_variables

    //(1): if we reference the local variable in the same scope of sync() - we are protected by std::memory_order_acquire
    //     if we reference the local_variable in the outer scope w.r.t sync() - if the memory_ordering is seen by the compiler - compiler is responsible for flushing the assumptions of related local_variables - and the immediate subsequent acquisitions of those must issue brand new instructions
    //                                                                        - if the memory_ordering is not seen by the compiler => function is not inlinable => compiler lost track of variables => compiler is responsible for equivalent action of std::atomic_signal_fence(memory_order_consume) of related local_variables post the function call  
    //(2): if we want to access concurrent variables - we need to use concurrency precautions - such is lock_guard and std::memory_order_acquire and release in and out of the concurrent transaction
    //we dont care about hardware because std::atomic_thread_fence(std::memory_order_etc) is a sufficient hardware instruction but not compiling instruction
    //shit's hard fellas, don't come up with your own concurrent routine - even std makes mistakes 

    class MemorySafeSynchronizer{

        private:

            dg::vector<std::unique_ptr<Synchronizable>> synchronizable_vec;

        public:

            MemorySafeSynchronizer(): synchronizable_vec(){}

            inline __attribute__((always_inline)) ~MemorySafeSynchronizer() noexcept{

                this->synchronizable_vec.clear();

                if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } else{
                    std::atomic_thread_fence(std::memory_order_acquire);
                }
            }

            auto add(std::unique_ptr<Synchronizable> syncable) noexcept -> exception_t{

                if (syncable == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                this->synchronizable_vec.push_back(std::move(syncable));
                return dg::network_exception::SUCCESS;
            }

            inline __attribute__((always_inline)) void sync() noexcept{

                for (auto& synchronizable: this->synchronizable_vec){
                    synchronizable->sync();
                }

                this->synchronizable_vec.clear();

                if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } else{
                    std::atomic_thread_fence(std::memory_order_acquire);
                }
            }
    };

    template <class SyncObject, class ptr_t>
    class RestrictPointerSynchronizer{

        private:

            SyncObject * synchronizable;
            dg::unordered_set<ptr_t> pointer_set;

        public:

            //let's assume people are rational
            RestrictPointerSynchronizer(SyncObject& synchronizable) noexcept: synchronizable(&synchronizable),
                                                                              pointer_set(){}

            template <class Iterator>
            auto add_range(Iterator first, Iterator last) noexcept -> exception_t{

                for (auto it = first; it != last; ++it){
                    if (this->pointer_set.contains(*it)){
                        this->pointer_set.clear();
                        this->synchronizable->sync();
                        break;
                    }
                }

                this->pointer_set.insert(first, last);
                return dg::network_exception::SUCCESS;
            }

            template <class ...Args>
            auto add(Args ...args) noexcept -> exception_t{

                bool flag = (this->pointer_set.contains(args) || ...);

                if (flag){
                    this->pointer_set.clear();
                    this->synchronizable->sync();
                }

                (this->pointer_set.insert(args), ...);
                return dg::network_exception::SUCCESS;
            }

            void clear() noexcept{

                this->pointer_set.clear();
            }
    };
}

#endif