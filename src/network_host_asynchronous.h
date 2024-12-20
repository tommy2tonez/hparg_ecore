#ifndef __DG_NETWORK_HOST_ASYNCHRONOUS_H__
#define __DG_NETWORK_HOST_ASYNCHRONOUS_H__

#include <memory>
#include <unordered_map>
#include <memory>

namespace dg::network_host_asynchronous{

    //alright fellas - I think the asynchronous device is decently fast
    //goal is to not pollute the L1 cache of the resolutors so we can hyperthread the thing - which is mainly used for dispatching cuda orders 
    //alright - this is, for now, sufficient for asynchronous (in the sense of performance constraints)
    //error communications are within the work_order - the async_device only provides a synchronization interface
    //synchronization must be a through operation otherwise we are risking memory corruption - there is no timeout countdown or whatever, synchronization must happens or abort the program
    //we'll circle back the implementation laters if there are pull requests

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
            virtual void push(std::unique_ptr<WorkOrder>) noexcept -> exception_t = 0;
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

    //we might want to use buffer and trivial serialization here - we'll think about this later 

    class AsyncDeviceXInterface{

        public:

            virtual ~AsyncDeviceXInterface() noexcept = default;
            virtual auto exec(std::unique_ptr<WorkOrder>) noexcept -> std::expected<ticket_id_t, exception_t> = 0; //we might want to return std::unique_ptr<Synchronizable> here - we'll stick with C approach for now
            virtual void sync(ticket_id_t) noexcept = 0; //synchronization must be noexcept here
            virtual void close_ticket(ticket_id_t) noexcept = 0;
    };

    template <class Lambda>
    class LambdaWrappedWorkOrder: public virtual WorkOrder{

        private:

            Lambda lambda;

        public:

            static_assert(std::is_nothrow_destructible_v<Lambda>);
            static_assert(std::is_nothrow_invocable_v<Lambda>);

            LambdaWrappedWorkOrder(Lambda lambda) noexcept(noexcept(std::is_nothrow_move_constructible_v<Lambda>)): lambda(std::move(lambda)){}

            void run() noexcept{

                this->lambda();
            }
    };

    template <class Lambda>
    auto make_virtual_work_order(Lambda lambda) noexcept(noexcept(std::is_nothrow_move_constructible<Lambda>)) -> std::unique_ptr<WorkOrder>{

        return std::make_unique<LambdaWrappedWorkOrder<Lambda>>(std::move(lambda)); //TODOs: internal allocations - we don't accept memory exhaustion because that's a major source of bugs - and we can't be too cautious catching every memory allocations - it clutters the code
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
                    *fetching_addr = std::move(wo);
                    this->waiting_vec.pop_front();
                    std::atomic_thread_fence(std::memory_order_seq_cst);
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

                    pending_mtx = std::make_shared<std::mutex>();
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

    class AsyncDevice: public virtual AsyncDeviceInterface{

        private:

            std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::shared_ptr<WorkOrderContainerInterface> wo_container;
        
        public:

            AsyncDevice(std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                        std::shared_ptr<WorkOrderContainerInterface> wo_container) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                                             wo_container(std::move(wo_container)){}


            auto exec(std::unique_ptr<WorkOrder> wo) noexcept -> exception_t{

                return this->wo_container->push(std::move(wo));
            }
    };

    auto spawn_async_device(size_t concurrent_worker, size_t work_order_cap) -> std::unique_ptr<AsyncDeviceInterface>{

    }

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

                ticket_id_t next_ticket_id = this->available_ticket_vec.back();

                if constexpr(DEBUG_MODE_FLAG){
                    auto map_ptr = this->ticket_resource_map.find(next_ticket_id);

                    if (map_ptr != this->ticket_resource_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->ticket_resource_map.insert(std::make_pair(next_ticket_id, TicketResource{{}, false}));
                this->available_ticket_vec.pop_back();

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
                this->available_ticket_vec.push_front(ticket_id);
            }
    };

    class AsyncDeviceX: public virtual AsyncDeviceXInterface{

        private:

            const std::unique_ptr<AsyncDeviceInterface> async_device;
            const std::unique_ptr<SynchronizationControllerInterface> sync_controller;

        public:

            AsyncDeviceX(std::unique_ptr<AsyncDeviceInterface> async_device,
                         std::unique_ptr<SynchronizationControllerInterface> sync_controller) noexcept: async_device(std::move(async_device)),
                                                                                                        sync_controller(std::move(sync_controller)){}

            auto exec(std::unique_ptr<WorkOrder> work_order) noexcept -> std::expected<ticket_id_t, exception_t>{

                std::expected<ticket_id_t, exception_t> ticket_id = this->sync_controller->open_ticket();

                if (!ticket_id.has_value()){
                    return std::unexpected(ticket_id.error());
                }

                auto task = [this, arg_ticket_id = ticket_id.value(), arg_work_order = std::move(work_order)]() noexcept{
                    arg_work_order->run(); //actually the memory ordering responsibility is the work order responsibility - if the work order is a concurrent transaction - it must take concurrency precautions
                    dg::network_exception_handler::nothrow_log(this->sync_controller->mark_completed(arg_ticket_id)); //we actually need to force noexcept here - close_ticket before synchronization is prohibited
                };

                auto virtual_task       = make_virtual_work_order(std::move(task));
                exception_t async_err   = this->async_device->exec(std::move(virtual_task));

                if (dg::network_exception::is_failed(async_err)){
                    this->sync_controller->close_ticket(ticket_id.value());
                    return std::unexpected(async_err);
                }

                return ticket_id.value();
            }

            void sync(ticket_id_t ticket_id) noexcept{

                std::shared_ptr<std::mutex> mtx = std::make_shared<std::mutex>();
                mtx->lock(); //relaxed is sufficient
                dg::network_exception_handler::nothrow_log(this->sync_controller->add_observer(ticket_id, mtx)); 
                mtx->lock(); //relaxed is sufficient
            }

            void close_ticket(ticket_id_t ticket_id) noexcept{

                //closing ticket before synchronization is undefined - we must check that
                this->sync_controller->close_ticket(ticket_id);
            }
    };
    
    auto spawn_async_devicex(std::unique_ptr<AsyncDeviceInterface>) -> std::unique_ptr<AsyncDeviceXInterface>{

        //we encapsulate the std::unique_ptr<SynchronizationControllerInterface> here
    }

    class DistributedAsyncDeviceX: public virtual AsyncDeviceXInterface{

        private:

            const std::vector<std::unique_ptr<AsyncDeviceXInterface>> async_device_vec;

        public:

            DistributedAsyncDeviceX(std::vector<std::unique_ptr<AsyncDeviceXInterface>> async_device_vec) noexcept: async_device_vec(std::move(async_device_vec)){}

            auto exec(std::unique_ptr<WorkOrder> work_order) noexcept -> std::expected<distributed_ticket_id_t, exception_t>{

                assert(stdx::is_pow2(async_device_vec.size()));

                size_t random_clue = dg::network_randomizer::randomize_uint<size_t>(); 
                size_t idx = random_clue & (async_device_vec.size() - 1u);
                std::expected<ticket_id_t, exception_t> ticket_id = this->async_device_vec[idx]->exec(std::move(work_order));

                if (!ticket_id.has_value()){
                    return std::unexpected(ticket_id.error());
                }

                return DistributedTicketId{idx, ticket_id.value()};
            }

            void sync(distributed_ticket_id_t ticket_id) noexcept{

                this->async_device_vec[ticket_id.async_device_idx]->sync(ticket_id.ticket_id);
            }

            void close_ticket(ticket_id_t ticket_id) noexcept{

                this->async_device_vec[ticket_id.async_device_idx]->close_ticket(ticket_id.ticket_id);
            }
    };

    auto spawn_distributed_async_devicex(std::vector<std::unique_ptr<AsyncDeviceXInterface>> async_device_vec) -> std::unique_ptr<AsyncDeviceXInterface>{

        if (!stdx::is_pow2(async_device_vec.size())){
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }
        
        auto iter = std::find(async_device_vec.begin(), async_device_vec.end(), nullptr);

        if (iter != async_device_vec.end()){
            dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
        }

        return std::make_unique<DistributedAsyncDeviceX>(std::move(async_device_vec));
    }

    class TicketSynchronizer: public virtual Synchronizable{

        private:

            std::shared_ptr<AsyncDeviceInterface> async_device;
            ticket_id_t ticket_id;
        
        public:

            TicketSynchronizer(std::shared_ptr<AsyncDeviceInterface> async_device, 
                               ticket_id_t ticket_id) noexcept: async_device(std::move(async_device)),
                                                                ticket_id(ticket_id){
                
                assert(this->async_device != nullptr);
            }

            ~TicketSynchronizer() noexcept{

                this->async_device->sync(this->ticket_id);
                this->async_device->close_ticket(this->ticket_id);
            }

            void sync() noexcept{

                this->async_device->sync(this->ticket_id);
            }
    };

    auto to_unique_synchronizable(std::shared_ptr<AsyncDeviceXInterface> async_device, ticket_id_t ticket_id) noexcept -> std::unique_ptr<Synchronizable>{ //we use internal allocations so its safe to assume allocations aren't errors

        //this is a precond
        if constexpr(DEBUG_MODE_FLAG){
            if (async_device == nullptr){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        return std::make_unique<TicketSynchronizer>(std::move(async_device), ticket_id);
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

                this->synchronizable_vec.clear();
            }
    };

    template <class ptr_t>
    class RestrictPointerSynchronizer{

        private:

            Synchronizable * synchronizable;
            dg::unordered_set<ptr_t> pointer_set;

        public:

            RestrictPointerSynchronizer(Synchronizable& synchronizable) noexcept: synchronizable(&synchronizable),
                                                                                  pointer_set(pointer_set){}

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
    };
}

#endif