#ifndef __DG_NETWORK_HOST_ASYNCHRONOUS_H__
#define __DG_NETWORK_HOST_ASYNCHRONOUS_H__

#include <memory>
#include <unordered_map>
#include <memory>

namespace dg::network_host_asynchronous{

    //alright fellas - code is clear - we'll move on

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

    class SynchronizationControllerInterface{

        public:

            virtual ~SynchronizationControllerInterface() noexcept = default;
            virtual auto open_ticket() noexcept -> std::expected<ticket_id_t, exception_t> = 0;
            virtual auto mark_completed(ticket_id_t) noexcept -> exception_t = 0;
            virtual auto add_observer(ticket_id_t, std::shared_ptr<std::mutex>) noexcept -> exception_t = 0;
            virtual void close_ticket(ticket_id_t) noexcept = 0; 
    };

    class AsyncDeviceXInterface{

        public:

            virtual ~AsyncDeviceXInterface() noexcept = default;
            virtual auto exec(std::unique_ptr<WorkOrder>) noexcept -> std::expected<void *, exception_t> = 0; //we might want to return std::unique_ptr<Synchronizable> here - we'll stick with C approach for now
            virtual void sync(void *) noexcept = 0; //synchronization must be noexcept here
            virtual void close_handle(void *) noexcept = 0;
    };

    using load_balance_handle_t = LoadBalanceHandle;

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
            virtual auto exec(std::unique_ptr<WorkOrder>, size_t est_flops) noexcept -> std::expected<void *, exception_t> = 0;
            virtual void sync(void *) noexcept = 0;
            virtual void close_handle(void *) noexcept = 0; 
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

    class WorkOrderContainer: public virtual WorkOrderContainerInterface{

        private:

            dg::deque<std::unique_ptr<WorkOrder>> wo_vec;
            dg::vector<std::pair<std::shared_ptr<std::mutex>, std::unique_ptr<WorkOrder> *>> waiting_vec;
            size_t wo_vec_capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            WorkOrderContainer(dg::deque<std::unique_ptr<WorkOrder>> wo_vec,
                               dg::vector<std::pair<std::shared_ptr<std::mutex>, std::unique_ptr<WorkOrder> *>> waiting_vec,
                               size_t wo_vec_capacity,
                               std::unique_ptr<std::mutex> mtx) noexcept: wo_vec(std::move(wo_vec)),
                                                                          waiting_vec(std::move(waiting_vec)),
                                                                          wo_vec_capacity(wo_vec_capacity),
                                                                          mtx(std::move(mtx)){}

            auto push(std::unique_ptr<WorkOrder> wo) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (!this->waiting_vec.empty()){
                    auto [pending_mtx, fetching_addr] = std::move(this->waiting_vec.front());
                    this->waiting_vec.pop_front();
                    *fetching_addr = std::move(wo);
                    std::atomic_thread_fence(std::memory_order_release);
                    pending_mtx->unlock();
                    return dg::network_exception::SUCCESS;
                }

                if (this->wo_vec.size() == this->wo_vec_capacity){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                this->wo_vec.push_back(std::move(wo));
                return dg::network_exception::SUCCESS;
            }

            auto pop() noexcept -> std::unique_ptr<WorkOrder>{

                std::shared_ptr<std::mutex> pending_mtx = {};
                std::unique_ptr<WorkOrder> wo = {};

                {
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->wo_vec.empty()){
                        auto rs = std::move(this->wo_vec.front());
                        this->wo_vec.pop_front();
                        return rs;
                    }

                    pending_mtx = std::make_shared<std::mutex>(); //TODOs: internalize allocations
                    pending_mtx->lock();

                    this->waiting_vec.push_back(std::make_pair(std::move(pending_mtx), &wo));
                }

                stdx::xlock_guard<std::mutex> lck_grd(*pending_mtx);
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
            std::shared_ptr<std::atomic<intmax_t>> last_heartbeat_in_utc_nanoseconds; //relaxed is sufficient

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

            std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::shared_ptr<WorkOrderContainerInterface> wo_container;
        
        public:

            AsyncDevice(std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                        std::shared_ptr<WorkOrderContainerInterface> wo_container) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                                             wo_container(std::move(wo_container)){}


            auto exec(std::unique_ptr<WorkOrder> wo) noexcept -> exception_t{

                //this is responsible for making sure a through operation to be through - such is there is no SUCCESS and stuck - there must be an observer to observe wo_container

                return this->wo_container->push(std::move(wo));
            }
    };

    // auto spawn_async_device(size_t concurrent_worker, size_t work_order_cap) -> std::unique_ptr<AsyncDeviceInterface>{

    // }

    class SynchronizationController: public virtual SynchronizationControllerInterface{

        private:

            struct TicketResource{
                dg::vector<std::shared_ptr<std::mutex>> observer_vec;
                bool is_completed;
            };

            dg::unordered_map<ticket_id_t, TicketResource> ticket_resource_map;
            dg::deque<ticket_id_t> available_ticket_vec; //we might want incrementors
            std::unique_ptr<std::mutex> mtx;

        public:

            SynchronizationController(dg::unordered_map<ticket_id_t, TicketResource> ticket_resource_map,
                                      dg::deque<ticket_id_t> available_ticket_vec,
                                      std::unique_ptr<std::mutex> mtx) noexcept: ticket_resource_map(std::move(ticket_resource_map)),
                                                                                 available_ticket_vec(std::move(available_ticket_vec)),
                                                                                 mtx(std::move(mtx)){}

            auto open_ticket() noexcept -> std::expected<ticket_id_t, exception_t>{

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

                this->ticket_resource_map.insert(std::make_pair(next_ticket_id, TicketResource{{}, false}));
                this->available_ticket_vec.pop_front();

                return next_ticket_id;
            }

            auto mark_completed(ticket_id_t ticket_id) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->ticket_resource_map.find(ticket_id);

                if (map_ptr == this->ticket_resource_map.end()){
                    return dg::network_exception::RESOURCE_ABSENT;
                }

                if (map_ptr->second.is_completed){
                    return dg::network_exception::SUCCESS;
                }

                for (const auto& mtx_sptr: map_ptr->second.observer_vec){
                    mtx_sptr->unlock();
                }

                map_ptr->second.observer_vec.clear();
                map_ptr->second.is_completed = true;

                return dg::network_exception::SUCCESS;
            }

            auto add_observer(ticket_id_t ticket_id, std::shared_ptr<std::mutex> mtx) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->ticket_resource_map.find(ticket_id);

                if (map_ptr == this->ticket_resource_map.end()){
                    return dg::network_exception::RESOURCE_ABSENT;
                }

                if (map_ptr->second.is_completed){
                    mtx->unlock();
                    return dg::network_exception::SUCCESS;
                }

                map_ptr->second.observer_vec.push_back(std::move(mtx));
                return dg::network_exception::SUCCESS;
            }

            void close_ticket(ticket_id_t ticket_id) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->ticket_resource_map.find(ticket_id);

                if (map_ptr == this->ticket_resource_map.end()){
                    return;
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

    class AsyncDeviceX: public virtual AsyncDeviceXInterface{

        private:

            struct InternalHandle{
                ticket_id_t ticket_id;
            };

            const std::unique_ptr<AsyncDeviceInterface> async_device;
            const std::shared_ptr<SynchronizationControllerInterface> sync_controller;

        public:

            AsyncDeviceX(std::unique_ptr<AsyncDeviceInterface> async_device,
                         std::unique_ptr<SynchronizationControllerInterface> sync_controller) noexcept: async_device(std::move(async_device)),
                                                                                                        sync_controller(std::move(sync_controller)){}

            auto exec(std::unique_ptr<WorkOrder> work_order) noexcept -> std::expected<void *, exception_t>{

                std::expected<ticket_id_t, exception_t> ticket_id = this->sync_controller->open_ticket();

                if (!ticket_id.has_value()){
                    return std::unexpected(ticket_id.error());
                }

                auto task = [sync_controller_arg = this->sync_controller, ticket_id_arg = ticket_id.value(), work_order_arg = std::move(work_order)]() noexcept{
                    work_order_arg->run();
                    dg::network_exception_handler::nothrow_log(sync_controller_arg->mark_completed(ticket_id_arg)); //we actually need to force noexcept here - close_ticket before synchronization is prohibited
                };
                auto virtual_task       = make_virtual_work_order(std::move(task));
                exception_t async_err   = this->async_device->exec(std::move(virtual_task));

                if (dg::network_exception::is_failed(async_err)){
                    this->sync_controller->close_ticket(ticket_id.value());
                    return std::unexpected(async_err);
                }

                InternalHandle * dynamic_handle     = new InternalHandle{ticket_id.value()}; //TODOs: internalize allocations 
                void * void_dynamic_handle          = dynamic_handle; 

                return void_dynamic_handle;
            }

            void sync(void * handle) noexcept{

                InternalHandle * internal_handle    = static_cast<InternalHandle *>(stdx::safe_ptr_access(handle));
                std::shared_ptr<std::mutex> mtx     = std::make_shared<std::mutex>(); //TODOs: internalize allocations
                mtx->lock(); //relaxed is sufficient
                dg::network_exception_handler::nothrow_log(this->sync_controller->add_observer(internal_handle->ticket_id, mtx)); 
                mtx->lock(); //relaxed is sufficient
            }

            void close_handle(void * handle) noexcept{

                //closing ticket before synchronization is undefined - we must check that
                InternalHandle * internal_handle = static_cast<InternalHandle *>(stdx::safe_ptr_access(handle));
                this->sync_controller->close_ticket(internal_handle->ticket_id);
                delete internal_handle;
            }
    };

    // auto spawn_async_devicex(std::unique_ptr<AsyncDeviceInterface>) -> std::unique_ptr<AsyncDeviceXInterface>{

    //     //we encapsulate the std::unique_ptr<SynchronizationControllerInterface> here
    // }

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

                front_node->current_load += est_workload.value();
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

                delete internal_handle; //TODOs: internalize allocation
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

                if (this->load_balance_heap[idx]->current_load < this->load_balance_heap[c]->current_load){
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

                if (this->load_balance_heap[c]->current_load < this->load_balance_heap[idx]->current_load){
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

            struct InternalHandle{
                void * async_device_wo_handle;
                void * load_balancer_handle;
            };

            const std::unordered_map<async_device_id_t, std::unique_ptr<AsyncDeviceXInterface>> async_device_map;
            const std::unique_ptr<LoadBalancerInterface> load_balancer;

        public:

            LoadBalancedAsyncDeviceX(std::unordered_map<async_device_id_t, std::unique_ptr<AsyncDeviceXInterface>> async_device_map,
                                     std::unique_ptr<LoadBalancerInterface> load_balancer) noexcept: async_device_map(std::move(async_device_map)),
                                                                                                     load_balancer(std::move(load_balancer)){}

            auto exec(std::unique_ptr<WorkOrder> work_order, size_t est_flops) noexcept -> std::expected<void *, exception_t>{

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

                std::expected<void *, exception_t> wo_handle = map_ptr->second->exec(std::move(work_order));

                if (!wo_handle.has_value()){
                    this->load_balancer->close_load_handle(load_balance_handle.value());
                    return std::unexpected(wo_handle.error());
                }

                InternalHandle * dynamic_handle = new InternalHandle{wo_handle.value(), load_balance_handle.value()};  //TODOs: internalize allocations
                void * void_dynamic_handle      = dynamic_handle;

                return void_dynamic_handle;
            }

            void sync(void * handle) noexcept{

                InternalHandle * internal_handle    = static_cast<InternalHandle *>(stdx::safe_ptr_access(handle));
                async_device_id_t async_device_id   = this->load_balancer->get_async_device_id(internal_handle->load_balancer_handle);
                auto map_ptr                        = this->async_device_map.find(async_device_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->async_device_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                map_ptr->second->sync(internal_handle->async_device_wo_handle);
            }

            void close_handle(void * handle) noexcept{

                InternalHandle * internal_handle    = static_cast<InternalHandle *>(stdx::safe_ptr_access(handle));
                async_device_id_t async_device_id   = this->load_balancer->get_async_device_id(internal_handle->load_balancer_handle);
                auto map_ptr                        = this->async_device_map.find(async_device_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->async_device_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->load_balancer->close_handle(internal_handle->load_balancer_handle);
                map_ptr->second->close_handle(internal_handle->async_device_wo_handle);

                delete internal_handle; //TODOs: internalize allocations
            }
    };

    class TicketSynchronizer: public virtual Synchronizable{

        private:

            std::shared_ptr<AsyncDeviceInterface> async_device;
            void * wo_handle;
            bool sync_flag;

        public:

            TicketSynchronizer(std::shared_ptr<AsyncDeviceInterface> async_device, 
                               void * wo_handle) noexcept: async_device(std::move(async_device)),
                                                           wo_handle(wo_handle),
                                                           sync_flag(false){}

            ~TicketSynchronizer() noexcept{

                if (!this->sync_flag){
                    this->async_device->sync(this->wo_handle);
                }

                this->async_device->close_handle(this->wo_handle);
            }

            void sync() noexcept{

                if (!this->sync_flag){
                    this->async_device->sync(this->wo_handle);
                }

                this->sync_flag = true;
            }
    };

    class TicketXSynchronizer: public virtual Synchronizable{

        private:

            std::shared_ptr<LoadBalancedAsyncDeviceX> async_device;
            void * wo_handle;
            bool sync_flag;

        public:

            TicketXSynchronizer(std::shared_ptr<LoadBalancedAsyncDeviceX> async_device,
                                void * wo_handle) noexcept: async_device(std::move(async_device)),
                                                            wo_handle(wo_handle),
                                                            sync_flag(false){}

            ~TicketXSynchronizer() noexcept{

                if (!this->sync_flag){
                    this->async_device->sync(this->wo_handle);
                }

                this->async_device->close_handle(this->wo_handle);
            }

            void sync() noexcept{

                if (!this->sync_flag){
                    this->async_device->sync(this->wo_handle);
                }

                this->sync_flag = true;
            }
    };

    auto to_unique_synchronizable(std::shared_ptr<AsyncDeviceXInterface> async_device, void * wo_handle) noexcept -> std::unique_ptr<Synchronizable>{ //we use internal allocations so its safe to assume allocations aren't errors

        if constexpr(DEBUG_MODE_FLAG){
            if (async_device == nullptr){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        return std::make_unique<TicketSynchronizer>(std::move(async_device), wo_handle);
    }

    auto to_unique_synchronizable(std::shared_ptr<LoadBalancedAsyncDeviceX> async_device, void * wo_handle) noexcept -> std::unique_ptr<Synchronizable>{

        if constexpr(DEBUG_MODE_FLAG){
            if (async_device == nullptr){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        return std::make_unique<TicketXSynchronizer>(std::move(async_device), wo_handle);
    }

    class BatchSynchronizer: public virtual Synchronizable{

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

    template <class ptr_t>
    class RestrictPointerSynchronizer{

        private:

            Synchronizable * synchronizable;
            dg::unordered_set<ptr_t> pointer_set;

        public:

            //let's assume people are rational
            RestrictPointerSynchronizer(Synchronizable& synchronizable) noexcept: synchronizable(&synchronizable),
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