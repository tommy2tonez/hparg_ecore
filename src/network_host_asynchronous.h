#ifndef __DG_NETWORK_HOST_ASYNCHRONOUS_H__
#define __DG_NETWORK_HOST_ASYNCHRONOUS_H__

#include <memory>
#include <unordered_map>
#include <memory>
#include "network_std_container.h"
#include "network_concurrency.h"
#include "network_exception.h"
#include <semaphore>
#include "stdx.h"
#include "network_log.h"
#include "network_randomizer.h"
#include "network_exception_handler.h"

namespace dg::network_host_asynchronous{

    //we'll add the std::memory_order_relaxed synchronization - this is a very important optimizable 
    //alright, this is a very hard task
    //we need to rebuild this to take batches of WorkOrder, we need to be considerate about the memory orderings, we'll revisit this tomorrow
    //we'll stick with unique_ptr<> for best practices, yet we need to build a better affined allocator, such is like the stack_allocation yet does malloc + free on sz_cap or interval cap, or timeout, says 1MB/ free, or 1MB/ malloc, we have good partial deallocator (fragmentation management) internally, we just need to keep the allocation lifetimes under control 
    //note that we can allocate from one affined allocator, and deallocate on another
    //this is expected, yet we attempt to further affine things by doing aggregations

    //let's see, we are to dispatch 32MB of linear complexity to asynchronous device per WorkOrder, this is our unit
    //assume that we have 1024 cores, each could crunch 15GB of linear/ s
    //we are expecting 1TB of linear crunch per second

    //32MB == 1024*32 ~= 32000 memory orderings/ second, this is acceptable
    //the overhead factors are of 10x 
    //we are expecting 320000 memory orderings/ second, this is not acceptable
    //we'll try to bring the overhead -> not more than the actual have to incur cost

    using async_device_id_t = size_t;

    class WorkOrder{

        public:

            virtual ~WorkOrder() noexcept = default;
            virtual void run() noexcept = 0; //this is syntactically incorrect, I long for const noexcept, because we can literally store size_t * const and mutating the size_t instead of the pointer
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

    class RadixerInterface{

        public:

            static inline constexpr uint8_t RAM_HEAVY   = 0u;
            static inline constexpr uint8_t CPU_HEAVY   = 1u;

            virtual ~RadixerInterface() noexcept = default;
            virtual auto radix(size_t est_cpu_flops, size_t est_ram_flops) noexcept ->  uint8_t = 0;
    };

    class XLoadBalancedAsyncDeviceXInterface{

        public:

            virtual ~XLoadBalancedAsyncDeviceXInterface() noexcept = default;
            virtual auto exec(std::unique_ptr<WorkOrder>, size_t est_cpu_flops, size_t est_ram_flops) noexcept -> std::expected<std::unique_ptr<Synchronizable>, exception_t> = 0; //10MB/ dispatch or 1MB/ dispatch (this unit is very important, that's why we need the help of tiles)
    };

    template <class Lambda>
    class LambdaWrappedWorkOrder: public virtual WorkOrder{

        private:

            Lambda lambda;

        public:

            static_assert(std::is_nothrow_destructible_v<Lambda>);
            // static_assert(std::is_nothrow_invocable_v<Lambda>);

            LambdaWrappedWorkOrder(Lambda lambda) noexcept(std::is_nothrow_move_constructible_v<Lambda>): lambda(std::move(lambda)){}

            void run() noexcept{

                static_assert(std::is_nothrow_invocable_v<Lambda>);
                this->lambda();
            }
    };

    class SharedWrappedWorkOrder: public virtual WorkOrder{

        private:

            std::shared_ptr<WorkOrder> base;

        public:

            SharedWrappedWorkOrder(std::shared_ptr<WorkOrder> base): base(std::move(base)){

                if (this->base == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }
            }

            void run() noexcept{

                this->base->run();
            }
    };

    class FinitePoolLinearWorkOrderContainer: public virtual WorkOrder{

        private:

            dg::vector<std::unique_ptr<WorkOrder>> wo_container; 

        public:

            FinitePoolLinearWorkOrderContainer(size_t cap): wo_container(){

                this->wo_container.reserve(cap);
            }

            auto add(std::unique_ptr<WorkOrder>&& wo) noexcept -> exception_t{

                if (this->wo_container.size() == this->wo_container.capacity()){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                this->wo_container.push_back(std::move(wo));
                return dg::network_exception::SUCCESS;
            }
            
            auto size() const noexcept -> size_t{

                return this->wo_container.size();
            }

            auto get(size_t idx) noexcept -> std::unique_ptr<WorkOrder>&{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->wo_container.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return this->wo_container[idx];
            }

            void clear() noexcept{

                this->wo_container.clear();
            }

            void run() noexcept{

                for (auto& wo: this->wo_container){
                    wo->run();
                }
            }            
    };

    auto make_virtual_workorder_sequential_container(size_t sz) -> std::expected<std::unique_ptr<FinitePoolLinearWorkOrderContainer>, exception_t>{

        try{
            return dg::network_allocation::make_unique<FinitePoolLinearWorkOrderContainer>(sz);
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }
    }

    template <class Lambda>
    auto make_virtual_work_order(Lambda lambda) -> std::unique_ptr<WorkOrder>{

        return dg::network_allocation::make_unique<LambdaWrappedWorkOrder<Lambda>>(std::move(lambda));
    }

    auto to_unique_workorder(std::shared_ptr<WorkOrder> base) -> std::unique_ptr<WorkOrder>{

        return dg::network_allocation::make_unique<SharedWrappedWorkOrder>(std::move(base)); //unique because the usage scope is that it is only referenced by the async_consumer, shared_ptr<> to avoid that we lose data in case of exceptions, we arent doing std::unique_ptr<>&&, there are certain language limitations  
    }

    class TaskSynchronizer: public virtual Synchronizable{

        private:

            std::binary_semaphore smp;
            bool synchronization_responsibility_flag;

        public:

            //this is bad practice

            TaskSynchronizer() noexcept: smp(0),
                                         synchronization_responsibility_flag(true){}

            TaskSynchronizer(const TaskSynchronizer&) = delete;
            TaskSynchronizer& operator =(const TaskSynchronizer&) = delete;
            TaskSynchronizer(TaskSynchronizer&& other) = delete;
            TaskSynchronizer& operator =(TaskSynchronizer&& other) = delete;

            ~TaskSynchronizer() noexcept{

                this->sync();
            }

            void sync() noexcept{

                bool responsibility_flag = std::exchange(this->synchronization_responsibility_flag, false);

                if (responsibility_flag){
                    this->smp.acquire();
                }
            }

            auto get_observer() noexcept -> std::binary_semaphore *{

                return &this->smp;
            } 

            void release() noexcept{

                this->synchronization_responsibility_flag = false;
            }
    };

    class WorkOrderContainer: public virtual WorkOrderContainerInterface{

        private:

            dg::pow2_cyclic_queue<std::unique_ptr<WorkOrder>> wo_vec;
            dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, std::unique_ptr<WorkOrder> *>> waiting_vec;
            std::unique_ptr<std::mutex> mtx;

        public:

            WorkOrderContainer(dg::pow2_cyclic_queue<std::unique_ptr<WorkOrder>> wo_vec,
                               dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, std::unique_ptr<WorkOrder> *>> waiting_vec,
                               std::unique_ptr<std::mutex> mtx) noexcept: wo_vec(std::move(wo_vec)),
                                                                          waiting_vec(std::move(waiting_vec)),
                                                                          mtx(std::move(mtx)){}

            ~WorkOrderContainer() noexcept{

                if (!this->wo_vec.empty()){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }
            }

            auto push(std::unique_ptr<WorkOrder> wo) noexcept -> exception_t{

                if (wo == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                std::binary_semaphore * releasing_smp = nullptr; 

                exception_t rs = [&, this]{
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->waiting_vec.empty()){
                        auto [pending_smp, fetching_addr] = this->waiting_vec.front();
                        this->waiting_vec.pop_front();
                        *fetching_addr = std::move(wo);
                        std::atomic_signal_fence(std::memory_order_seq_cst); //semaphore has their virtues, we just need a signal fence
                        // pending_smp->release(); //this could hinder the lockability of lck_grd
                        releasing_smp = pending_smp;
                        return dg::network_exception::SUCCESS;
                    }

                    if (this->wo_vec.size() == this->wo_vec.capacity()){
                        return dg::network_exception::RESOURCE_EXHAUSTION;
                    }

                    dg::network_exception_handler::nothrow_log(this->wo_vec.push_back(std::move(wo)));
                    return dg::network_exception::SUCCESS;
                }();

                if (releasing_smp != nullptr){
                    releasing_smp->release();
                }                

                return rs;
            }

            auto pop() noexcept -> std::unique_ptr<WorkOrder>{

                std::binary_semaphore pending_smp(0);
                std::unique_ptr<WorkOrder> wo = {};

                while (true){
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->wo_vec.empty()){
                        auto rs = std::move(this->wo_vec.front());
                        this->wo_vec.pop_front();

                        return rs;
                    }

                    if (this->waiting_vec.size() == this->waiting_vec.capacity()){
                        //something is very wrong
                        continue;
                    }

                    dg::network_exception_handler::nothrow_log(this->waiting_vec.push_back(std::make_pair(&pending_smp, &wo)));
                    break;
                }

                stdx::volatile_access(&pending_smp)->acquire();

                return std::unique_ptr<WorkOrder>(std::move(*stdx::volatile_access(&wo, pending_smp)));
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

    class AsyncDeviceX: public virtual AsyncDeviceXInterface{

        private:

            std::unique_ptr<AsyncDeviceInterface> async_device;

        public:

            AsyncDeviceX(std::unique_ptr<AsyncDeviceInterface> async_device) noexcept: async_device(std::move(async_device)){}

            auto exec(std::unique_ptr<WorkOrder> work_order) noexcept -> std::expected<std::unique_ptr<Synchronizable>, exception_t>{

                if (work_order == nullptr){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                auto rs = dg::network_exception::to_cstyle_function(dg::network_allocation::make_unique<TaskSynchronizer>)(); //

                if (!rs.has_value()){
                    return std::unexpected(rs.error());
                }

                auto rs_grd             = stdx::resource_guard([&]() noexcept{
                    rs.value()->release();
                });

                auto smp_reference      = rs.value()->get_observer();  //
                auto task               = [smp_reference, work_order_arg = std::move(work_order)]() noexcept{ //
                    work_order_arg->run();
                    //better safe than sorry
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    smp_reference->release();
                };

                auto virtual_task       = dg::network_exception::to_cstyle_function(make_virtual_work_order<decltype(task)>)(std::move(task)); //

                if (!virtual_task.has_value()){
                    return std::unexpected(virtual_task.error());
                }

                exception_t async_err   = this->async_device->exec(std::move(virtual_task.value())); //

                if (dg::network_exception::is_failed(async_err)){
                    return std::unexpected(async_err);
                }

                rs_grd.release();

                return std::unique_ptr<Synchronizable>(std::move(rs.value())); //
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
            size_t fixed_size_overhead;
            size_t max_unit_load;
            std::unique_ptr<std::mutex> mtx;

        public:

            UniformLoadBalancer(std::vector<std::unique_ptr<UniformLoadBalancerHeapNode>> load_balance_heap,
                                size_t fixed_size_overhead,
                                size_t max_unit_load,
                                std::unique_ptr<std::mutex> mtx) noexcept: load_balance_heap(std::move(load_balance_heap)),
                                                                           fixed_size_overhead(fixed_size_overhead),
                                                                           max_unit_load(max_unit_load),
                                                                           mtx(std::move(mtx)){}

            auto open_handle(size_t est_flops) noexcept -> std::expected<void *, exception_t>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                size_t est_workload = est_flops + this->fixed_size_overhead;

                if (est_workload > this->max_unit_load){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                if (this->load_balance_heap.front()->current_load + est_workload > this->load_balance_heap.front()->max_load){
                    return std::unexpected(dg::network_exception::QUEUE_FULL);
                }

                UniformLoadBalancerHeapNode * front_node                    = this->load_balance_heap.front().get();
                std::expected<InternalHandle *, exception_t> dynamic_handle = dg::network_exception::to_cstyle_function(dg::network_allocation::std_new_object<InternalHandle, InternalHandle>)(InternalHandle{front_node->async_device_id, front_node, est_workload});

                if (!dynamic_handle.has_value()){
                    return std::unexpected(dynamic_handle.error());
                }

                front_node->current_load += est_workload;
                this->push_down_at(0u);

                return static_cast<void *>(dynamic_handle.value());
            }

            auto get_async_device_id(void * handle) noexcept -> async_device_id_t{

                return static_cast<InternalHandle *>(stdx::safe_ptr_access(handle))->async_device_id;
            } 

            void close_handle(void * handle) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                InternalHandle * internal_handle        = static_cast<InternalHandle *>(stdx::safe_ptr_access(handle));
                UniformLoadBalancerHeapNode * cur_node  = internal_handle->heap_node;
                cur_node->current_load                  -= internal_handle->task_load;
                this->push_up_at(cur_node->heap_idx);

                dg::network_allocation::std_delete_object<InternalHandle>(internal_handle);
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

            std::vector<std::unique_ptr<LoadBalancerInterface>> load_balancer_vec;
            size_t overload_bounce_sz;

        public:

            DistributedLoadBalancer(std::vector<std::unique_ptr<LoadBalancerInterface>> load_balancer_vec,
                                    size_t overload_bounce_sz) noexcept: load_balancer_vec(std::move(load_balancer_vec)),
                                                                         overload_bounce_sz(overload_bounce_sz){}

            auto open_handle(size_t est_flops) noexcept -> std::expected<void *, exception_t>{

                for (size_t i = 0u; i < this->overload_bounce_sz; ++i){
                    std::expected<void *, exception_t> handle = this->internal_open_handle(est_flops);

                    if (handle.has_value()){
                        return handle; 
                    }

                    if (handle.error() != dg::network_exception::QUEUE_FULL){
                        return std::unexpected(handle.error());
                    }
                }

                return this->internal_open_handle(est_flops);
            }

            auto get_async_device_id(void * handle) noexcept -> async_device_id_t{

                InternalHandle * internal_handle = static_cast<InternalHandle *>(stdx::safe_ptr_access(handle));
                return this->load_balancer_vec[internal_handle->load_balancer_idx]->get_async_device_id(internal_handle->load_balancer_handle);
            }

            void close_handle(void * handle) noexcept{

                InternalHandle * internal_handle = static_cast<InternalHandle *>(stdx::safe_ptr_access(handle));
                this->load_balancer_vec[internal_handle->load_balancer_idx]->close_handle(internal_handle->load_balancer_handle);
                dg::network_allocation::std_delete_object<InternalHandle>(internal_handle);
            }

        private:

            auto internal_open_handle(size_t est_flops) noexcept -> std::expected<void *, exception_t>{

                assert(stdx::is_pow2(this->load_balancer_vec.size()));

                size_t random_clue                                      = dg::network_randomizer::randomize_int<size_t>();
                size_t balancer_idx                                     = random_clue & (this->load_balancer_vec.size() - 1u);
                std::expected<void *, exception_t> load_balancer_handle = this->load_balancer_vec[balancer_idx]->open_handle(est_flops);

                if (!load_balancer_handle.has_value()){
                    return std::unexpected(load_balancer_handle.error());
                }

                std::expected<InternalHandle *, exception_t> internal_handle = dg::network_exception::to_cstyle_function(dg::network_allocation::std_new_object<InternalHandle, InternalHandle>)(InternalHandle{load_balancer_handle.value(), balancer_idx});

                if (!internal_handle.has_value()){
                    this->load_balancer_vec[balancer_idx]->close_handle(load_balancer_handle.value());
                    return std::unexpected(internal_handle.error());
                }

                return static_cast<void *>(internal_handle.value());
            }
    };

    class LoadBalancedAsyncDeviceX: public virtual LoadBalancedAsyncDeviceXInterface{

        private:

            dg::unordered_unstable_map<async_device_id_t, std::unique_ptr<AsyncDeviceXInterface>> async_device_map;
            std::shared_ptr<LoadBalancerInterface> load_balancer;

        public:

            LoadBalancedAsyncDeviceX(dg::unordered_unstable_map<async_device_id_t, std::unique_ptr<AsyncDeviceXInterface>> async_device_map,
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

                auto load_balance_handle_grd        = stdx::resource_guard([&]() noexcept{
                    this->load_balancer->close_handle(load_balance_handle.value());
                });

                async_device_id_t async_device_id   = this->load_balancer->get_async_device_id(load_balance_handle.value());
                auto map_ptr                        = this->async_device_map.find(async_device_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->async_device_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto task = [load_balancer_arg          = this->load_balancer,
                             work_order_arg             = std::move(work_order), 
                             load_balance_handle_arg    = load_balance_handle.value()]() noexcept{

                    work_order_arg->run();
                    std::atomic_signal_fence(std::memory_order_seq_cst); //alright - this is very important - otherwise we are languagely incorrect
                    load_balancer_arg->close_handle(load_balance_handle_arg);
                };

                auto virtual_task = dg::network_exception::to_cstyle_function(make_virtual_work_order<decltype(task)>)(std::move(task));

                if (!virtual_task.has_value()){
                    return std::unexpected(virtual_task.error());                    
                }

                std::expected<std::unique_ptr<Synchronizable>, exception_t> syncer = map_ptr->second->exec(std::move(virtual_task.value()));

                if (!syncer.has_value()){
                    return std::unexpected(syncer.error());
                }

                load_balance_handle_grd.release();

                return syncer;
            }
    };

    //100%

    //the problem is that we want to use this asynchronous device as much as possible (literally)
    //because we could abstract the compute or the RAM fetch without being afraid of affecting the system
    //we have a "proven" component to maximize certain things 
    //we have thought very long and hard, the est_cpu_flops + est_ram_flops are the minimum yet sufficient variables to dispatch asynchronous workorders

    //cpu flops is somewhat an affined variable (if we are not talking about inter-comm like relaxed cmpexch + memory orderings + friends)
    //ram flops is somewhat a global variable

    //we admit that this is an acceptable approach, not an all round approach
    //because most of the time, we are either dispatching O(n) cpu_flops + O(n) ram flops
    //                          or we are dispatching O(n^2) cpu flops  + O(n) ram flops

    //the latter is CPU bound, the former is RAM bound
    //we'd want to have a RAM boundaries for the two type of workers

    //RAM_HEAVY pulling 50% and CPU_HEAVY pulling another 50%
    //we'd attempt to bottleneck that by using hyperthreading (to limit compute) or the # of working cores 
    //this is very important

    class XLoadBalancedAsyncDeviceX: public virtual XLoadBalancedAsyncDeviceXInterface{

        private:

            std::unique_ptr<RadixerInterface> radixer;
            std::unique_ptr<LoadBalancedAsyncDeviceXInterface> ram_heavy_async_device;
            std::unique_ptr<LoadBalancedAsyncDeviceXInterface> cpu_heavy_async_device;

        public:

            XLoadBalancedAsyncDeviceX(std::unique_ptr<RadixerInterface> radixer,
                                      std::unique_ptr<LoadBalancedAsyncDeviceXInterface> ram_heavy_async_device,
                                      std::unique_ptr<LoadBalancedAsyncDeviceXInterface> cpu_heavy_async_device) noexcept: radixer(std::move(radixer)),
                                                                                                                           ram_heavy_async_device(std::move(ram_heavy_async_device)),
                                                                                                                           cpu_heavy_async_device(std::move(cpu_heavy_async_device)){}

            auto exec(std::unique_ptr<WorkOrder> work_order, size_t est_cpu_flops, size_t est_ram_flops) noexcept -> std::expected<std::unique_ptr<Synchronizable>, exception_t>{

                if (work_order == nullptr){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                uint8_t dispatch_code = this->radixer->radix(est_cpu_flops, est_ram_flops);

                if (dispatch_code == RadixerInterface::RAM_HEAVY){
                    return this->ram_heavy_async_device->exec(std::move(work_order), est_ram_flops); //I was thinking for ram heavy async device to pull 80% RAM (by limiting the number of concurrent workers), leave the 20% to the cpu_heavy_async_device, to buffer the RAM -> CPU caches to do linear or combinatorial heavy operations
                                                                                                     //this is the acceptable approach, we dont really have other ways to manage a global variable such as RAM
                } else if (dispatch_code == RadixerInterface::CPU_HEAVY){
                    return this->cpu_heavy_async_device->exec(std::move(work_order), est_cpu_flops);
                } else {
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }
            }
    };

    class MemorySafeSynchronizer{

        private:

            dg::vector<std::unique_ptr<Synchronizable>> synchronizable_vec;

        public:
            
            static inline constexpr size_t MIN_CAP = 8u;

            MemorySafeSynchronizer(): MemorySafeSynchronizer(MIN_CAP){} 

            MemorySafeSynchronizer(size_t cap): synchronizable_vec(std::max(cap, MIN_CAP)){}

            inline __attribute__((always_inline)) ~MemorySafeSynchronizer() noexcept{

                this->sync();
            }

            auto add(std::unique_ptr<Synchronizable> syncable) noexcept -> exception_t{

                if (syncable == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                if (this->synchronizable_vec.size() == this->synchronizable_vec.capacity()){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                this->synchronizable_vec.push_back(std::move(syncable));
                return dg::network_exception::SUCCESS;
            }

            inline __attribute__((always_inline)) auto addsync(std::unique_ptr<Synchronizable> syncable) noexcept -> exception_t{

                stdx::seq_cst_guard seqcst_tx; //making sure addsync is not reordered if compiler is to see addsync, make this a compiler thread fence for all the results that this addsync could affect, alright, the results that addsync could affect is limited to the immediate calling function due to limitations

                if (syncable == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                if (this->synchronizable_vec.size() == this->synchronizable_vec.capacity()){
                    this->sync();
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
                    std::atomic_signal_fence(std::memory_order_seq_cst);
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

    //this is a small component < 1000 lines yet a very crucial backbone component for our forwarding + backwarding tiles
    //we are relying on this to be consistent, do hot + cold dispatch, keep all CPUs running

    //we haven't considered RAM flops which is a fairly important piece of performance constraint
    //the time it takes for a worker to complete is
    //total_in_queue_ram / RAM_BUS_RATE * ram_usage(worker) 
    //this is actually a leetcode hard problem

    //let's see what we are trying to optimize
    //we want to complete a task ASAP
    //we want to maximize CPU thruput + RAM thruput

    //is completing a task ASAP == maximizing CPU thruput + RAM thruput? 

    //it is a two very different thing

    //the problem is that we are missing a variable, which would heavily hinder our future architect if we are to walk down this road
    //the CPU flops variable is an affined, detached variable
    //the RAM flops is a global, unified variable

    //this is hard!
    //the problem is that when we are reaching the RAM saturations (or it is considered, RAM_HEAVY, there should be dedicated workers to exactly pull the RAM)
    //and the other tasks CPU_HEAVY should be distributed like how we are doing things now  

    //we dont really want to change the interface
    //so we'd do an extension
    //to add another variable, called ram_flops
    //and we would want to radix a task as CPU heavy or RAM_HEAVY based on the cpu_flops/ ram_flops ratio
    //and we'd move from there

    //the heap order is maintained (is it???) for every inbound work order

    //in order for that to happen, we can control our "virtuous" dispatches, keep the compute size to be < certain sizes
    //we aren't really waiting, so there is no synchronization overheads
    //we are in the nosync business of logit density mining
}

#endif