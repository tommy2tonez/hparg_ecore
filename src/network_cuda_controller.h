#ifndef __DG_NETWORK_CUDA_CONTROLLER_H__
#define __DG_NETWORK_CUDA_CONTROLLER_H__

#include "network_utility.h"
#include <atomic>
#include <mutex>
#include "network_std_container.h"
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <deque>

namespace dg::network_cuda_controller{
    
    //this is the sole interface to communicate with cuda_runtime lib - to allow synchronize accesses to cuda_runtime lib 

    struct ControllerResource{
        dg::network_std_container::vector<int> device;
        std::unique_ptr<std::recursive_mutex> mtx;
    };

    inline std::unique_ptr<ControllerResource> controller_resource{};

    auto init() noexcept -> exception_t{

        int count{};
        int master_gpu_id = 0;

        exception_t err = dg::network_exception::wrap_cuda_exception(cudaGetDeviceCount(&count));

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        if (count == 0){
            return dg::network_exception::CUDA_NOT_SUPPORTED;
        }

        exception_t err = dg::network_exception::wrap_cuda_exception(cudaSetDevice(master_gpu_id)); //0 has to be the master gpu - if not then a wrapper function is required to make this happen - 

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        controller_resource = std::make_unique<ControllerResource>(ControllerResource{{master_gpu_id}, std::make_unique<std::recursive_mutex>()}); //resource allocation error need to be isolated and made noexcept - instead of relying on init noexcept (bad practice)
        return dg::network_exception::SUCCESS;
    }

    void deinit() noexcept{

        exception_t err = dg::network_exception::wrap_cuda_exception(cudaDeviceReset());

        if (dg::network_exception::is_failed(err)){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(err));
            std::abort();
        }

        controller_resource = {};
    }

    auto cuda_is_valid_device(int * device, size_t sz) noexcept -> std::expected<bool, exception_t>{
        
        int count{};
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaGetDeviceCount(&count)); 

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }
        
        if constexpr(DEBUG_MODE_FLAG){
            if (count == 0){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        sz = dg::network_genult::inplace_make_set(device, sz);

        if (sz == 0u){
            return false;
        }

        int * first             = device;
        int * last              = first + sz; 
        const int MIN_DEVICE_ID = 0;
        const int MAX_DEVICE_ID = count - 1; 

        return std::find_if(first, last, [=](int cur){return std::clamp(cur, MIN_DEVICE_ID, MAX_DEVICE_ID) != cur;}) != last;
    }

    auto cuda_set_device(int * device, size_t sz) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(device, sz));

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        controller_resource->device = dg::network_std_container::vector<int>(device, device + sz);
        return dg::network_exception::SUCCESS;
    }

    auto cuda_malloc(void ** ptr, size_t blk_sz) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMalloc(ptr, blk_sz));

        return err;
    }

    auto cuda_free(void * ptr) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaFree(ptr));

        return err;
    }

    auto cuda_memset(void * ptr, int value, size_t sz) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMemset(ptr, value, sz));

        return err;
    } 

    auto cuda_memcpy(void * dst, const void * src, size_t sz, cudaMemcpyKind kind) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMemcpy(dst, src, sz, kind));

        return err;
    } 

    auto cuda_memcpy_peer(void * dst, int dst_id, const void * src, size_t src_id, size_t sz) noexcept -> exception_t{

        // auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMemcpyPeer(dst, dst_id, src, src_id, sz));
        
        return err;
    }
   
    auto cuda_synchronize() noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaDeviceSynchronize());

        return err;
    }

    //this is protected interface - don't invoke if cannot guarantee lock hierarchical order - risking deadlock if done otherwise - bad practice

    auto cuda_env_lock_guard(int * device, size_t sz) noexcept{

        controller_resource->mtx->lock();
        dg::network_std_container::vector<int> old_device = controller_resource->device;

        auto resource_backout = [old_device]() noexcept{
            controller_resource->device = old_device;
            exception_t err = dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(controller_resource->device.data(), controller_resource->device.size())); 
            dg::network_exception_handler::nothrow_log(err);
            controller_resource->mtx->unlock();
        };

        exception_t err = dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(device, sz)); 
        dg::network_exception_handler::nothrow_log(err);

        return dg::network_genult::resource_guard(resource_backout);
    }

    //----
}

namespace dg::network_cuda_kernel_launcher{

    //-reimplement round robin today + tmr - 
    //minimal viable products: (1): temporal locality of memory operation
    //                         (2): device locality
    //                         (3): fair scheduler of kernel launches
    
    //an attempt to solve locality here is premature - the sole goal of this component is to reduce synchronization calls to legacy api (saturate GPU flops)
    //what is the goal of this component? - saturate GPU consumption
    //where is the bottleneck ?
    //if bottleneck is not enough asynchronous calls being launched - then it's the concurrent kernel_launcher + launch balancer problem - only works for parallel tasks
    //what's an ideal kernel launch size? 256x256 or 64x64?
    //I think for compression its 256x256
    //for AGI its 64x64 - 64x64 is A LOT of contexts
    //user inputs are tokenized into byte_stream, batched -> grid  
    //massive processing of user inputs - instead of individual tokens

    using wo_ticketid_t = uint64_t; 

    struct ExecutableInterface{
        virtual ~ExecutableInterface() noexcept = default; 
        virtual void run() noexcept = 0;
    };

    struct WorkOrder{
        wo_ticketid_t ticket_id;
        dg::network_std_container::vector<int> env;
        std::unique_ptr<ExecutableInterface> executable;
        size_t runtime_complexity;
    };

    struct WorkOrderContainerInterface{
        virtual ~WorkOrderContainerInterface() noexcept = default;
        virtual void push(WorkOrder) noexcept = 0; //since this is an application - I don't think propagate error code is necessary here - since memory exhaustion could be solved by abstraction
        virtual auto pop() noexcept -> dg::network_std_container::vector<WorkOrder> = 0;
    };

    struct WorkTicketControllerInterface{
        virtual ~WorkTicketControllerInterface() noexcept = default;
        virtual auto next_ticket() noexcept -> std::expected<wo_ticketid_t, exception_t> = 0;
        virtual void set_status(wo_ticketid_t, exception_t) noexcept = 0; 
        virtual auto get_status(wo_ticketid_t) noexcept -> exception_t = 0;
        virtual void close_ticket(wo_ticketid_t) noexcept = 0;
    };

    struct KernelLauncherControllerInterface{
        virtual ~KernelLauncherControllerInterface() noexcept = default;
        virtual auto launch(std::unique_ptr<ExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<wo_ticketid_t, exception_t>;
        virtual auto status(wo_ticketid_t) noexcept -> exception_t = 0;
        virtual void close(wo_ticketid_t) noexcept = 0;
    };

    template <class Executable>
    class DynamicExecutable: public virtual ExecutableInterface{

        private:

            Executable executable;
        
        public:

            static_assert(std::is_nothrow_destructible_v<Executable>);
            static_assert(std::is_nothrow_invokable_v<Executable>);

            DynamicExecutable(Executable executable): executable(std::move(executable)){}

            void run() noexcept{

                this->executable();
            }
    };
     
    class LoadBalancedWorkOrderContainer: public virtual WorkOrderContainerInterface{

        private:

            dg::network_std_container::deque<WorkOrder> work_order_vec;
            size_t min_complexity_thrhold;
            size_t suggested_max_complexity_thrhold;
            std::chrono::nanoseconds last_consumed;
            std::chrono::nanoseconds max_waittime_thrhold; 
            std::unique_ptr<std::mutex> mtx;

        public:

            LoadBalancedWorkOrderContainer(dg::network_std_container::deque<WorkOrder> work_order_vec,
                                           size_t min_complexity_thrhold,
                                           size_t suggested_max_complexity_thrhold,
                                           std::chrono::nanoseconds last_consumed,
                                           std::chrono::nanoseconds max_waittime_thrhold,
                                           std::unique_ptr<std::mutex> mtx) noexcept: work_order_vec(std::move(work_order_vec)),
                                                                                      min_complexity_thrhold(min_complexity_thrhold),
                                                                                      suggested_max_complexity_thrhold(suggested_max_complexity_thrhold),
                                                                                      last_consumed(last_consumed),
                                                                                      max_waittime_thrhold(max_waittime_thrhold),
                                                                                      mtx(std::move(mtx)){}
            
            void push(WorkOrder wo) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->work_order_vec.push_back(std::move(wo));
            }

            auto pop() noexcept -> dg::network_std_container::vector<WorkOrder>{ //even though I think std::optional<std::vector<WorkOrder>> is way more performant than this - I think that's the vector container's responsibility than an optimization to make

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (!this->internal_is_due()){
                    return {};
                } 

                auto rs = this->internal_pop();  
                this->last_consumed = dg::network_genult::unix_timestamp();
                return rs;
            }
        
        private:

            auto internal_pop() noexcept -> dg::network_std_container::vector<WorkOrder>{

                size_t peeking_sz = 0u; 
                dg::network_std_container::vector<WorkOrder> rs{};

                while (!this->work_order_vec.empty()){
                    rs.push_back(std::move(this->work_order_vec.front()));
                    this->work_order_vec.pop_front();
                    peeking_sz += this->work_order_vec.back().runtime_complexity;

                    if (peeking_sz > this->suggested_max_complexity_thrhold){
                        return rs;
                    }
                }

                return rs;
            }

            auto internal_is_due() const noexcept -> bool{
                
                std::chrono::nanoseconds now    = dg::network_genult::unix_timestamp();
                std::chrono::nanoseconds lapsed = dg::network_genult::timelapsed(this->last_consumed, now); 

                if (lapsed > this->max_waittime_thrhold){
                    return true;
                }

                size_t peeking_sz = 0u; 

                for (const auto& wo: this->work_order_vec){
                    peeking_sz += wo.runtime_complexity;

                    if (peeking_sz >= this->min_complexity_thrhold){
                        return true;
                    }
                }

                return false;
            }
    };

    class WorkTicketController: public virtual WorkTicketControllerInterface{

        private:

            size_t wo_sz;
            std::unordered_map<wo_ticketid_t, exception_t> wo_status_map;
            std::unique_ptr<std::mutex> mtx;

        public:
            
            WorkTicketController(size_t wo_sz, 
                                 std::unordered_map<wo_ticketid_t, exception_t> wo_status_map,
                                 std::unique_ptr<std::mutex> mtx) noexcept: wo_sz(wo_sz),
                                                                            wo_status_map(std::move(wo_status_map)),
                                                                            mtx(std::move(mtx)){}

            auto next_ticket() noexcept -> std::expected<wo_ticketid_t, exception_t>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx); 
                size_t nxt_id   = this->wo_sz;
                this->wo_sz     += 1;

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->wo_status_map.find(id) != this->wo_status_map.end()){
                        dg::network_log_stackdump::crticial(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->wo_status_map[id] = CUDA_EXECUTABLE_WAITING_DISPATCH;

                return nxt_id;
            }

            void set_status(wo_ticketid_t id, exception_t err) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto ptr        = this->wo_status_map.find(id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->wo_status_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                ptr->second = err;
            }

            auto get_status(wo_ticketid_t id) noexcept -> exception_t{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
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

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
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

    class KernelLauncherController: public virtual KernelLauncherControllerInterface{

        private:

            std::shared_ptr<WorkOrderContainerInterface> wo_container;
            std::shared_ptr<WorkTicketControllerInterface> id_controller;
        
        public:

            KernelLauncherController(std::shared_ptr<WorkOrderContainerInterface> wo_container,
                                     std::shared_ptr<WorkTicketControllerInterface> id_controller) noexcept: wo_container(std::move(wo_container)),
                                                                                                             id_controller(std::move(id_controller)){}
            
            auto launch(std::unique_ptr<ExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<wo_ticketid_t, exception_t>{

                std::expected<bool, exception_t> env_status = dg::network_cuda_controller::cuda_is_valid_device(env, env_sz);

                if (!env_status.has_value()){
                    return std::unexpected(env_status.error());
                }

                if (!env_status.value()){
                    return std::unexpected(dg::network_exception::UNSUPPORTED_CUDA_DEVICE);
                }

                std::expected<wo_ticketid_t, exception_t> wo_id = this->id_controller->next_ticket();
                
                if (!wo_id.has_value()){
                    return std::unexpected(wo_id.error());
                }

                auto wo = WorkOrder{wo_id.value(), std::move(executable), std::move(env), runtime_complexity};
                this->wo_container->push(std::move(wo));

                return wo_id.value();
            }

            auto status(wo_ticketid_t id) noexcept -> exception_t{

                return this->id_controller->get_status(id);
            }

            void close(wo_ticketid_t id) noexcept{

                this->id_controller->close_ticket(id);
            }
    };

    template <size_t CONCURRENCY_SZ> //deprecate next iteration
    class ConcurrentKernelLauncherController: public virtual KernelLauncherControllerInterface{

        private:

            dg::network_std_container::vector<std::unique_ptr<KernelLauncherControllerInterface>> controllers;

        public:

            static_assert(CONCURRENCY_SZ <= std::numeric_limits<uint8_t>::max());

            ConcurrentKernelLauncherController(dg::network_std_container<std::unique_ptr<KernelLauncherControllerInterface>> controllers, 
                                               std::integral_constant<size_t, CONCURRENCY_SZ>) noexcept: controllers(std::move(controllers)){}


            auto launch(std::unique_ptr<ExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<wo_ticketid_t, exception_t>{

                size_t thr_idx = dg::network_concurrency::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                std::expected<wo_ticketid_t, exception_t> rs = this->controllers[thr_idx]->launch(std::move(executable), env, env_sz, runtime_complexity);

                if (!rs.has_value()){
                    return std::unexpected(rs.error());
                }

                return this->encode(rs.value(), thr_idx);
            }

            auto status(wo_ticketid_t encoded_id) noexcept -> exception_t{

                auto [id, thr_id] = this->decode(encoded_id);
                return this->controllers[thr_id]->status(id);
            }

            void close(wo_ticketid_t encoded_id) noexcept{

                auto [id, thr_id] = this->decode(encoded_id);
                this->controllers[thr_id]->close(encoded_id);
            }
        
        private:

            auto encode(wo_ticketid_t id, uint8_t thr_id) noexcept -> wo_ticketid_t{

                static_assert(std::is_unsigned_v<wo_ticketid_t>);
                using promoted_t = dg::max_unsigned_t;
                static_assert(sizeof(wo_ticketid_t) + sizeof(uint8_t) <= promoted_t);
                promoted_t encoded = (static_cast<promoted_t>(id) << (sizeof(uint8_t) * CHAR_BIT)) | static_cast<promoted_t>(thr_id);

                return dg::network_genult::safe_integer_cast<wo_ticketid_t>(encoded);
            }

            auto decode(wo_ticketid_t encoded_id) noexcept -> std::pair<wo_ticketid_t, uint8_t>{

                wo_ticketid_t id    = encoded_id >> (sizeof(uint8_t) * CHAR_BIT);
                uint8_t thr_id      = encoded_id & low<wo_ticketid_t>(std::integral_constant<size_t, sizeof(uint8_t) * CHAR_BIT>{});

                return {id, thr_id};
            }
    };

    class CudaDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WorkOrderContainerInterface> wo_container;
            std::shared_ptr<WorkTicketControllerInterface> wo_controller;
        
        public:

            CudaDispatcher(std::shared_ptr<WorkOrderContainerInterface> wo_container,
                           std::shared_ptr<WorkTicketControllerInterface> wo_controller) noexcept: wo_container(std::move(wo_container)),
                                                                                                   wo_controller(std::move(wo_controller)){}
            
            bool run_one_epoch() noexcept{

                dg::network_std_container::vector<WorkOrder> wos = this->wo_container->pop();

                if (wos.empty()){
                    return false;
                }

                dg::network_std_container::vector<int> env = this->extract_environment(wos);
                auto grd = dg::network_cuda_controller::cuda_env_lock_guard(env.data(), env.size());
                dg::network_exception::flush_cuda_exception();

                for (const auto& wo: wos){
                    wo.executable->run();
                    exception_t err = dg::network_exception::get_last_cuda_exception();

                    if (dg::network_exception::is_failed(err)){
                        wo_container->set_status(wo.ticket_id, dg::network_exception::join_exception(dg::network_exception::CUDA_LAUNCH_COMPLETED, err));
                    }
                }

                exception_t err = dg::network_cuda_controller::cuda_synchronize();

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(err)); //don't know what kind of error is returned here - rather abort - 
                    std::abort();
                }

                for (const auto& wo: wos){
                    wo_controller->set_status(wo.ticket_id, dg::network_exception::CUDA_LAUNCH_COMPLETED);
                }

                return true;
            }
        
        private:

            auto extract_environment(const dg::network_std_container::vector<WorkOrder>& wos) noexcept -> dg::network_std_container::vector<int>{

                dg::network_std_container::vector<int> env = {};

                for (const auto& wo: wos){
                    env.insert(env.end(), wo.env.begin(), wo.env.end());
                }

                dg::network_std_container::unordered_set<int> env_set(env.begin(), env.end(), env.size());
                return dg::network_std_container::vector<int>(env_set.begin(), env_set.end());
            }
    };

    inline std::unique_ptr<KernelLauncherControllerInterface> kernel_launcher{}; 

    using dispatch_t = uint8_t; 

    enum dispatch_option: dispatch_t{
        par_launch  = 0u,
        seq_launch = 1u
    };

    //this is important - allow grouping of tasks - reduce synchronization overhead - increase locality 
    template <class Executable>
    auto make_sequential_task(Executable executable) noexcept -> std::unique_ptr<ExecutableInterface>{

        //need to define responsibility + error communication here
        static_assert(std::is_nothrow_move_constructible<Executable>);

        auto lambda = [exec = std::move(executable)]() noexcept{
            static_assert(noexcept(exec())); 
            // static_assert(std::is_same_v<exception_t, decltype(exec())>);
            // exec();
            // dg::network_exception::set_last_cuda_exception(dg::network_exception::wrap_cuda_exception(cudaGetLastError()));
            // exception_t err = dg::network_cuda_controller::cuda_synchronize();
            // dg::network_exception::set_last_cuda_exception();
            // dg::network_exception::set_last_cuda_exception(err);
        };

        return std::make_unique<DynamicExecutable<decltype(lambda)>>(std::move(lambda));
    }

    //this has to be a thin lambda wrapper to directly invoke cuda kernel_launch <<<>>>
    template <class Executable>
    auto make_parallel_task(Executable executable) noexcept -> std::unique_ptr<ExecutableInterface>{

        //need to define responsibility + error communication here
        static_assert(std::is_nothrow_move_constructible<Executable>);

        auto lambda = [exec = std::move(executable)]() noexcept{
            static_assert(noexcept(exec()));
            exec();
            dg::network_exception::set_last_cuda_exception(dg::network_exception::wrap_cuda_exception(cudaGetLastError()));
        };

        return std::make_unique<DynamicExecutable<decltype(lambda)>>(std::move(lambda));
    }

    template <class Executable>
    auto make_task(Executable executable, dispatch_t dispatch) noexcept -> std::unique_ptr<ExecutableInterface>{

        static_assert(std::is_nothrow_move_constructible<Executable>);

        if (dispatch == par_launch){
            return make_parallel_task(std::move(executable));
        }

        if (dispatch == seq_launch){
            return make_sequential_task(std::move(executable));
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }
        
        return {};
    }

    auto cuda_launch(std::unique_ptr<ExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> exception_t{

        std::expected<wo_ticketid_t, exception_t> launch_id = kernel_launcher->launch(std::move(executable), env, env_sz, runtime_complexity);
        exception_t err = {};
        
        if (!launch_id.has_value()){
            return launch_id.error();
        }

        auto synchronizable = [&err, id = launch_id.value()]() noexcept{
            err = kernel_launcher->status(id);
            return dg::network_exception::has_exception(err, dg::network_exception::CUDA_LAUNCH_COMPLETED);
        };

        dg::network_asynchronous::wait(synchronizable);
        return err;
    }
}

#endif