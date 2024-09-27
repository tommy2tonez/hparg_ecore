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
#include "network_concurrency.h"
#include "network_exception.h"

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

    }

    auto cuda_stream_create() noexcept -> std::expected<cudaStream_t, exception_t>{

        cudaStream_t cuda_stream    = {};
        exception_t err             = dg::network_exception::wrap_cuda_exception(cudaStreamCreate(&cuda_stream));

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return cuda_stream;
    }

    void cuda_stream_close(cudaStream_t cuda_stream) noexcept{ //should be cudaStreamHandle_t then handle synchronization if presents in cuda_stream_create args - yet it violates single responsibility - fine for now 

        dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaStreamDestroy(cuda_stream)));
    }

    void cuda_stream_syncclose(cudaStream_t cuda_stream) noexcept{

        dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaStreamSynchronize(cuda_stream)));
        dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaStreamDestroy(cuda_stream)));
    }

    //rs == raii + sync 
    auto cuda_stream_rscreate() noexcept -> std::expected<dg::network_genult::nothrow_immutable_unique_raii_wrapper<cudaStream_t, decltype(&cuda_stream_syncclose)>, exception_t>{

        std::expected<cudaStream_t, exception_t> stream = cuda_stream_create();

        if (!stream.has_value()){
            return std::unexpected(stream.error());
        }

        return {std::in_place_t{}, stream.value(), cuda_stream_syncclose};
    }

    auto cuda_is_valid_device(int * device, size_t sz) noexcept -> std::expected<bool, exception_t>{

        if (sz == 0u){
            return false;
        }

        dg::network_std_container::unordered_set<int> device_set(device, device + sz, sz);
        
        if (device_set.size() != sz){
            return false;
        }

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

        const int MIN_DEVICE_ID = 0;
        const int MAX_DEVICE_ID = count - 1;  
        auto unmet_cond         = [=](int cur){return std::clamp(cur, MIN_DEVICE_ID, MAX_DEVICE_ID) != cur;};

        return std::find_if(device_set.begin(), device_set.end(), unmet_cond) != device_set.end();
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

        auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaFree(ptr));

        return err;
    }

    auto cuda_memset(void * dst, int c, size_t sz) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        auto stream     = cuda_stream_rscreate();

        if (!stream.has_value()){
            return stream.error();
        }

        return dg::network_exception::wrap_cuda_exception(cudaMemsetAsync(dst, c, sz, stream.value()));
    } 

    auto cuda_memcpy(void * dst, const void * src, size_t sz, cudaMemcpyKind kind) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMemcpy(dst, src, sz, kind));

        return err;
    } 

    auto cuda_memcpy_peer(void * dst, int dst_id, const void * src, size_t src_id, size_t sz) noexcept -> exception_t{

        auto stream = cuda_stream_rscreate();

        if (!stream.has_value()){
            return stream.error();
        } 

        return dg::network_exception::wrap_cuda_exception(cudaMemcpyPeerAsync(dst, dst_id, src, src_id, sz, stream.value()));
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

        return dg::network_genult::resource_guard(resource_backout); //not semantically accurate - yet functionally accurate - improvement required
    }
    //----
}

namespace dg::network_cuda_kernel_par_launcher::global_exception{

    struct signature_dg_network_cuda_kernel_par_launcher_global_exception{}; 
    
    struct LaunchException{
        exception_t sys_err;
        bool is_completed;
    };

    using launch_exception_t            = uint64_t; //
    using exception_container_t         = std::array<LaunchException, dg::network_concurrency::THREAD_COUNT>; 
    using exception_container_object    = dg::network_genult::singleton<signature_dg_network_cuda_kernel_par_launcher_global_exception, exception_container_t>; //important - to avoid overflow and friends - yet I think this is compiler responsibility

    auto make_launch_exception_from_system_exception(exception_t sys_err) noexcept -> LaunchException{

        return {sys_err, false};
    }

    auto get_system_exception(LaunchException err) noexcept -> exception_t{

        return err.sys_err;
    }

    void set_exception(LaunchException err) noexcept{

        size_t thr_idx = dg::network_concurrency::this_thread_idx();
        exception_container_object::get()[thr_idx] = err;
    }

    auto last_exception() noexcept -> LaunchException{

        size_t thr_idx = dg::network_concurrency::this_thread_idx();
        return exception_container_object::get()[thr_idx];
    }

    void flush_exception() noexcept{

        size_t thr_idx = dg;:network_concurrency::this_thread_idx();
        exception_container_object::get()[thr_idx] = make_launch_exception_from_system_exception(dg::network_exception::SUCCESS);
    }

    auto mark_completed(LaunchException err) noexcept -> LaunchException{

        return {err.sys_err, true};
    } 

    auto is_failed(LaunchException err) noexcept -> bool{

        return dg::network_exception::is_failed(err.sys_err);
    }

    auto is_success(LaunchException err) noexcept -> bool{

        return dg::network_exception::is_success(err.sys_err);
    }

    auto is_completed(LaunchException err) noexcept -> bool{

        return err.is_completed;
    }
}

namespace dg::network_cuda_kernel_par_launcher{ //this is only to launch parallel task

    using wo_ticketid_t         = uint64_t; 
    using launch_exception_t    = global_exception::launch_exception_t; 

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
        virtual void set_status(wo_ticketid_t, launch_exception_t) noexcept = 0; 
        virtual auto get_status(wo_ticketid_t) noexcept -> launch_exception_t = 0;
        virtual void close_ticket(wo_ticketid_t) noexcept = 0;
    };

    struct KernelLauncherControllerInterface{
        virtual ~KernelLauncherControllerInterface() noexcept = default;
        virtual auto launch(std::unique_ptr<ExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<wo_ticketid_t, exception_t> = 0;
        virtual auto status(wo_ticketid_t) noexcept -> launch_exception_t = 0;
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

                size_t peek_complexity  = 0u; 
                auto rs                 = g::network_std_container::vector<WorkOrder>{};

                while (!this->work_order_vec.empty()){
                    rs.push_back(std::move(this->work_order_vec.front()));
                    this->work_order_vec.pop_front();
                    peek_complexity += this->work_order_vec.front().runtime_complexity;

                    if (peek_complexity > this->suggested_max_complexity_thrhold){
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

                size_t peek_complexity = 0u; 

                for (const auto& wo: this->work_order_vec){
                    peek_complexity += wo.runtime_complexity;

                    if (peek_complexity >= this->min_complexity_thrhold){
                        return true;
                    }
                }

                return false;
            }
    };

    class WorkTicketController: public virtual WorkTicketControllerInterface{

        private:

            size_t wo_sz;
            std::unordered_map<wo_ticketid_t, launch_exception_t> wo_status_map;
            std::unique_ptr<std::mutex> mtx;

        public:
            
            WorkTicketController(size_t wo_sz, 
                                 std::unordered_map<wo_ticketid_t, launch_exception_t> wo_status_map,
                                 std::unique_ptr<std::mutex> mtx) noexcept: wo_sz(wo_sz),
                                                                            wo_status_map(std::move(wo_status_map)),
                                                                            mtx(std::move(mtx)){}

            auto next_ticket() noexcept -> std::expected<wo_ticketid_t, exception_t>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx); 
                size_t nxt_id   = this->wo_sz;
                this->wo_sz     += 1;

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->wo_status_map.find(nxt_id) != this->wo_status_map.end()){
                        dg::network_log_stackdump::crticial(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->wo_status_map[nxt_id] = global_exception::make_launch_exception_from_system_exception(dg::network_exception::SUCCESS);
                return nxt_id;
            }

            void set_status(wo_ticketid_t id, launch_exception_t err) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
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
            std::shared_ptr<WorkTicketControllerInterface> ticket_controller;
        
        public:

            KernelLauncherController(std::shared_ptr<WorkOrderContainerInterface> wo_container,
                                     std::shared_ptr<WorkTicketControllerInterface> ticket_controller) noexcept: wo_container(std::move(wo_container)),
                                                                                                                 ticket_controller(std::move(ticket_controller)){}
            
            auto launch(std::unique_ptr<ExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<wo_ticketid_t, exception_t>{

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

                auto wo = WorkOrder{ticket_id.value(), std::move(executable), dg::network_std_container::vector<int>(env, env + env_sz), runtime_complexity};
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

    template <size_t CONCURRENCY_SZ> //deprecate next iteration
    class ConcurrentKernelLauncherController: public virtual KernelLauncherControllerInterface{

        private:

            dg::network_std_container::vector<std::unique_ptr<KernelLauncherControllerInterface>> controllers;

        public:

            static_assert(CONCURRENCY_SZ <= std::numeric_limits<uint8_t>::max());

            ConcurrentKernelLauncherController(dg::network_std_container::vector<std::unique_ptr<KernelLauncherControllerInterface>> controllers, 
                                               std::integral_constant<size_t, CONCURRENCY_SZ>) noexcept: controllers(std::move(controllers)){}


            auto launch(std::unique_ptr<ExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<wo_ticketid_t, exception_t>{

                size_t thr_idx = dg::network_concurrency::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                std::expected<wo_ticketid_t, exception_t> rs = this->controllers[thr_idx]->launch(std::move(executable), env, env_sz, runtime_complexity);

                if (!rs.has_value()){
                    return std::unexpected(rs.error());
                }

                return this->encode(rs.value(), thr_idx);
            }

            auto status(wo_ticketid_t encoded_id) noexcept -> launch_exception_t{

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
                uint8_t thr_id      = encoded_id & low<wo_ticketid_t>(std::integral_constant<size_t, (sizeof(uint8_t) * CHAR_BIT)>{});

                return {id, thr_id};
            }
    };

    class CudaDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WorkOrderContainerInterface> wo_container;
            std::shared_ptr<WorkTicketControllerInterface> ticket_controller;
        
        public:

            CudaDispatcher(std::shared_ptr<WorkOrderContainerInterface> wo_container,
                           std::shared_ptr<WorkTicketControllerInterface> ticket_controller) noexcept: wo_container(std::move(wo_container)),
                                                                                                       ticket_controller(std::move(ticket_controller)){}
            
            bool run_one_epoch() noexcept{

                dg::network_std_container::vector<WorkOrder> wos = this->wo_container->pop();

                if (wos.empty()){
                    return false;
                }

                dg::network_std_container::vector<int> env = this->extract_environment(wos);
                auto grd = dg::network_cuda_controller::cuda_env_lock_guard(env.data(), env.size());
                dg::network_cuda_controller::cuda_synchronize(); //flush cuda synchronization err
                global_exception::flush_exception(); //flush exception

                for (const auto& wo: wos){
                    wo.executable->run();
                    launch_exception_t err = global_exception::last_exception();

                    if (global_exception::is_failed(err)){
                        ticket_controller->set_status(wo.ticket_id, global_exception::mark_completed(err));
                    }
                }

                exception_t err = dg::network_cuda_controller::cuda_synchronize();

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(err)); //don't know what kind of error is returned here - rather abort - 
                    std::abort();
                }

                for (const auto& wo: wos){
                    ticket_controller->set_status(wo.ticket_id, global_exception::mark_completed(global_exception::make_launch_exception_from_system_exception(dg::network_exception::SUCCESS)));
                }

                return true;
            }
        
        private:

            auto extract_environment(const dg::network_std_container::vector<WorkOrder>& wos) noexcept -> dg::network_std_container::vector<int>{

                auto env_set = dg::network_std_container::unordered_set<int>{}; 

                for (const auto& wo: wos){
                    env_set.insert(wo.env.begin(), wo.env.end());
                }

                return dg::network_std_container::vector<int>(env_set.begin(), env_set.end());
            }
    };

    inline std::unique_ptr<KernelLauncherControllerInterface> kernel_launcher{}; 

    //this has to be a thin lambda wrapper solely, directly invoking __device__ __host__ <function_name> <<<cuda_config>>> - undefined otherwise - 
    
    template <class Executable>
    auto make_kernel_launch_task(Executable executable) noexcept -> std::unique_ptr<ExecutableInterface>{

        static_assert(std::is_nothrow_move_constructible<Executable>); //

        auto lambda = [exec = std::move(executable)]() noexcept{
            static_assert(noexcept(exec()));
            exec();
            exception_t err = dg::network_exception::wrap_cuda_exception(cudaGetLastError());
            global_exception::set_exception(global_exception::make_launch_exception_from_system_exception(err));
        };

        return std::make_unique<DynamicExecutable<decltype(lambda)>>(std::move(lambda));
    }

    auto cuda_launch(std::unique_ptr<ExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> exception_t{

        std::expected<wo_ticketid_t, exception_t> launch_id = kernel_launcher->launch(std::move(executable), env, env_sz, runtime_complexity);
        
        if (!launch_id.has_value()){
            return launch_id.error();
        }

        launch_exception_t err = {};
        auto synchronizable = [&err, id = launch_id.value()]() noexcept{
            err = kernel_launcher->status(id);
            return global_exception::is_completed(err);
        };

        dg::network_asynchronous::wait(synchronizable);
        return global_exception::get_system_exception(err);
    }
}

#endif