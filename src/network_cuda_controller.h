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

namespace dg::network_cuda_stream{

    static inline constexpr uint8_t SYNC_FLAG = 0b001;

    struct CudaStreamHandle{
        cudaStream_t cuda_stream;
        uint8_t flags;
    };

    auto cuda_stream_create(uint8_t flags) noexcept -> std::expected<CudaStreamHandle, exception_t>{

        cudaStream_t cuda_stream    = {};
        exception_t err             = dg::network_exception::wrap_cuda_exception(cudaStreamCreate(&cuda_stream));

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return CudaStreamHandle{cuda_stream, flags};
    }

    void cuda_stream_close(CudaStreamHandle handle) noexcept{

        bool has_synchronization = (handle.flags & SYNC_FLAG) != 0u;

        if (has_synchronization){
            dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaStreamSynchronize(handle.cuda_stream)));
        }

        dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaStreamDestroy(handle.cuda_stream)));
    }
 
    auto cuda_stream_raiicreate(uint8_t flags) noexcept -> std::expected<dg::network_genult::nothrow_immutable_unique_raii_wrapper<CudaStreamHandle, decltype(&cuda_stream_close)>, exception_t>{

        std::expected<CudaStreamHandle, exception_t> handle = cuda_stream_create(flags);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return {std::in_place_t{}, std::move(handle.value()), cuda_stream_close};
    }

    auto cuda_stream_get_legacy(CudaStreamHandle handle) noexcept -> cudaStream_t{

        return handle.cuda_stream;
    }
} 

namespace dg::network_cuda_controller{
    
    //this is the sole interface to communicate with cuda_runtime lib - to allow synchronized accesses to cuda_runtime lib 
    //this is fine for the first draft - be back for improvement later

    struct ControllerResource{
        dg::network_std_container::vector<int> device;
        size_t total_device_count;
        std::unique_ptr<std::recursive_mutex> mtx;
    };

    inline std::unique_ptr<ControllerResource> controller_resource{};

    auto init() noexcept -> exception_t{

        int count{};

        exception_t err = dg::network_exception::wrap_cuda_exception(cudaGetDeviceCount(&count));

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        if (count <= 0){
            return dg::network_exception::CUDA_NOT_SUPPORTED;
        }

        exception_t err = dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(nullptr, 0u)); 

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        controller_resource = std::make_unique<ControllerResource>(ControllerResource{{}, count, std::make_unique<std::recursive_mutex>()}); //resource allocation error need to be isolated and made noexcept - instead of relying on init noexcept (bad practice)
        return dg::network_exception::SUCCESS;
    }

    void deinit() noexcept{

        //cuda does not build this for deinitialization - sorry but that's the truth - rather this to be program-lifetime than to add redundant logics here
    }

    auto cuda_is_valid_device(int * device, size_t sz) noexcept -> std::expected<bool, exception_t>{ // fine - this is for interface consistency - does not neccessarily need to return error

        if (sz == 0u){
            return true; //is default according to MAN
        }

        dg::network_std_container::unordered_set<int> device_set(device, device + sz, sz);
        
        if (device_set.size() != sz){ //MAN does not specify whether device *, size_t has to be as valid set or not - stricter req
            return false;
        }

        const int MIN_DEVICE_ID = 0;
        const int MAX_DEVICE_ID = controller_resource->total_device_count - 1;
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
        auto stream     = dg::network_cuda_stream::cuda_stream_raiicreate(dg::network_cuda_stream::SYNC_FLAG);

        if (!stream.has_value()){
            return stream.error();
        }

        return dg::network_exception::wrap_cuda_exception(cudaMemsetAsync(dst, c, sz, dg::network_cuda_stream::cuda_stream_get_legacy(stream.value())));
    } 

    auto cuda_memcpy(void * dst, const void * src, size_t sz, cudaMemcpyKind kind) noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaMemcpy(dst, src, sz, kind));

        return err;
    } 

    auto cuda_memcpy_peer(void * dst, int dst_id, const void * src, size_t src_id, size_t sz) noexcept -> exception_t{

        auto stream = network_cuda_stream::cuda_stream_raiicreate(dg::network_cuda_stream::SYNC_FLAG);

        if (!stream.has_value()){
            return stream.error();
        } 

        return dg::network_exception::wrap_cuda_exception(cudaMemcpyPeerAsync(dst, dst_id, src, src_id, sz, dg::network_cuda_stream::cuda_stream_get_legacy(stream.value())));
    }
   
    auto cuda_synchronize() noexcept -> exception_t{

        auto lck_grd    = dg::network_genult::lock_guard(*controller_resource->mtx);
        exception_t err = dg::network_exception::wrap_cuda_exception(cudaDeviceSynchronize());

        return err;
    }

    //this is protected interface - don't invoke if cannot guarantee lock hierarchical order - risking deadlock if done otherwise - bad practice

    auto lock_env_guard(int * device, size_t sz) noexcept{

        controller_resource->mtx->lock();
        auto old_device = controller_resource->device;

        auto resource_backout = [old_device]() noexcept{
            controller_resource->device = old_device;
            dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(controller_resource->device.data(), controller_resource->device.size())));
            controller_resource->mtx->unlock();
        };

        dg::network_exception_handler::nothrow_log(dg::network_exception::wrap_cuda_exception(cudaSetValidDevices(device, sz)));
        return dg::network_genult::resource_guard(std::move(resource_backout)); //not semantically accurate - yet functionally accurate - improvement required
    }
    //----
}

namespace dg::network_cuda_kernel_par_launcher::exception{

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

namespace dg::network_cuda_kernel_par_launcher{

    //though I think concurrency could be/ have to be improved - like lock_guard and friends - I don't think it's going to be a bottleneck (this is a personal perspective - don't decide without proper instruments + usecases)
    //consider that linear operation is very heavy (flops ~= 1 << 20  for tile_dim = 64x64)
    //serialized access could be saturated at 1 << 30 launches/s (CPU flops) - each is at least 1 << 20 flops/launch -  so the total flops would be 1 << 50 per second ~= 1PB/ second - this should be fine for a single computation node (server)
    //this application is multi-threaded (massive decompression - compression of terabytes of data) - so the influx/ outflux is somewhat independent of the synchronization lags
    //this is fine for the first draft - be back for improvement later

    using wo_ticketid_t         = uint64_t; 
    using launch_exception_t    = exception::launch_exception_t; 

    struct VirtualExecutableInterface{
        virtual ~VirtualExecutableInterface() noexcept = default; 
        virtual auto run() noexcept -> exception_t = 0;
    };

    struct WorkOrder{
        wo_ticketid_t ticket_id;
        dg::network_std_container::vector<int> env; //this is very futuristic - because usually operations can only be operated in the same environment
        std::unique_ptr<VirtualExecutableInterface> executable;
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

    struct KernelLaunchControllerInterface{
        virtual ~KernelLaunchControllerInterface() noexcept = default;
        virtual auto launch(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<wo_ticketid_t, exception_t> = 0;
        virtual auto status(wo_ticketid_t) noexcept -> launch_exception_t = 0;
        virtual void close(wo_ticketid_t) noexcept = 0;
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
     
    class LoadBalancedWorkOrderContainer: public virtual WorkOrderContainerInterface{

        private:

            dg::network_std_container::deque<WorkOrder> work_order_vec;
            size_t min_complexity_thrhold;
            size_t suggested_max_complexity_thrhold;
            std::chrono::nanoseconds last_consumed_stamp;
            std::chrono::nanoseconds max_waittime_thrhold; 
            std::unique_ptr<std::mutex> mtx;

        public:

            LoadBalancedWorkOrderContainer(dg::network_std_container::deque<WorkOrder> work_order_vec,
                                           size_t min_complexity_thrhold,
                                           size_t suggested_max_complexity_thrhold,
                                           std::chrono::nanoseconds last_consumed_stamp,
                                           std::chrono::nanoseconds max_waittime_thrhold,
                                           std::unique_ptr<std::mutex> mtx) noexcept: work_order_vec(std::move(work_order_vec)),
                                                                                      min_complexity_thrhold(min_complexity_thrhold),
                                                                                      suggested_max_complexity_thrhold(suggested_max_complexity_thrhold),
                                                                                      last_consumed_stamp(last_consumed_stamp),
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
                this->last_consumed_stamp = dg::network_genult::unix_timestamp();
                return rs;
            }
        
        private:

            auto internal_pop() noexcept -> dg::network_std_container::vector<WorkOrder>{

                size_t peek_complexity  = 0u; 
                auto rs                 = dg::network_std_container::vector<WorkOrder>{};

                while (!this->work_order_vec.empty()){
                    peek_complexity += this->work_order_vec.front().runtime_complexity;
                    rs.push_back(std::move(this->work_order_vec.front()));
                    this->work_order_vec.pop_front();

                    if (peek_complexity > this->suggested_max_complexity_thrhold){
                        return rs;
                    }
                }

                return rs;
            }

            auto internal_is_due() const noexcept -> bool{
                
                std::chrono::nanoseconds now    = dg::network_genult::unix_timestamp();
                std::chrono::nanoseconds lapsed = dg::network_genult::timelapsed(this->last_consumed_stamp, now); 

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

                auto lck_grd            = dg::network_genult::lock_guard(*this->mtx); 
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
    class ConcurrentKernelLaunchController: public virtual KernelLaunchControllerInterface{

        private:

            dg::network_std_container::vector<std::unique_ptr<KernelLaunchControllerInterface>> controller_vec;

        public:

            static_assert(CONCURRENCY_SZ != 0u);
            static_assert(CONCURRENCY_SZ <= std::numeric_limits<uint8_t>::max());

            ConcurrentKernelLaunchController(dg::network_std_container::vector<std::unique_ptr<KernelLaunchControllerInterface>> controller_vec, 
                                             std::integral_constant<size_t, CONCURRENCY_SZ>) noexcept: controller_vec(std::move(controller_vec)){}


            auto launch(std::unique_ptr<VirtualExecutableInterface> executable, int * env, size_t env_sz, size_t runtime_complexity) noexcept -> std::expected<wo_ticketid_t, exception_t>{

                size_t thr_idx = dg::network_concurrency::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{}); 
                std::expected<wo_ticketid_t, exception_t> rs = this->controller_vec[thr_idx]->launch(std::move(executable), env, env_sz, runtime_complexity);

                if (!rs.has_value()){
                    return std::unexpected(rs.error());
                }

                return this->encode(rs.value(), thr_idx);
            }

            auto status(wo_ticketid_t encoded_id) noexcept -> launch_exception_t{

                auto [id, thr_id] = this->decode(encoded_id);
                return this->controller_vec[thr_id]->status(id);
            }

            void close(wo_ticketid_t encoded_id) noexcept{

                auto [id, thr_id] = this->decode(encoded_id);
                this->controller_vec[thr_id]->close(encoded_id);
            }
        
        private:

            auto encode(wo_ticketid_t id, uint8_t thr_id) noexcept -> wo_ticketid_t{

                static_assert(std::is_unsigned_v<wo_ticketid_t>);
                using promoted_t = dg::max_unsigned_t;
                static_assert(sizeof(wo_ticketid_t) + sizeof(uint8_t) <= sizeof(promoted_t));
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

                dg::network_std_container::vector<WorkOrder> wo_vec = this->wo_container->pop();

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

            auto extract_environment(const dg::network_std_container::vector<WorkOrder>& wo_vec) noexcept -> dg::network_std_container::vector<int>{

                auto env_set = dg::network_std_container::unordered_set<int>{};

                for (const auto& wo: wo_vec){
                    env_set.insert(wo.env.begin(), wo.env.end());
                }

                return dg::network_std_container::vector<int>(env_set.begin(), env_set.end());
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

#endif