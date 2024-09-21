
#ifndef __NETWORK_CONCURRENCY_H__
#define __NETWORK_CONCURRENCY_H__

#include <stddef.h>
#include <stdint.h>
#include <thread>
#include <vector>
#include <memory>
#include "network_concurrency_impl1.h"
#include <bit>
#include "network_utility.h"

namespace dg::network_concurrency{

    using namespace dg::network_concurrency_impl1::daemon_option_ns;

    using affine_policy_option_t    = dg::network_concurrency_impl1_app::affine_policy_option_t; 
    using WorkerInterface           = dg::network_concurrency_impl1::WorkerInterface; 

    //fine to not include these - or making these not part of the external interface - user need to rely on exception to spawn workers or abort system
    static inline constexpr size_t THREAD_COUNT = 32; //this is - however - is necessary - 

    struct signature_dg_network_concurrency{}; 

    struct ConcurrencyResource{
        std::unique_ptr<dg::network_concurrency_impl1::DaemonControllerInterface> daemon_controller; 
        jg::dense_hash_map<std::thread::id, size_t> thrid_to_idx_map;
    };

    using concurrency_resource_container = dg::network_genult::singleton<signature_dg_network_concurrency, ConcurrencyResource>;

    void init(){

        auto config = dg::network_concurrency_impl1_app::Config{AFFINE_POLICY, 
                                                                COMPUTING_DAEMON_NETWORK_THREAD_COUNT,
                                                                IO_DAEMON_THREAD_COUNT,
                                                                TRANSPORTATION_DAEMON_THREAD_COUNT,
                                                                HEARTBEAT_DAEMON_THREAD_COUNT};

        auto [controller, thr_vec]                              = dg::network_concurrency_impl1_app::spawn(config); 
        thr_vec                                                 = dg::network_genult::enumerate(std::move(thr_vec));
        concurrency_resource_container::get().daemon_controller = std::move(controller);
        concurrency_resource_container::get().thrid_to_idx_map  = jg::dense_hash_map<std::thread::id, size_t>(thr_vec.begin(), thr_vec.end(), thr_vec.size());
    }

    void deinit() noexcept{

        concurrency_resource_container::get() = {};
    }

    auto this_thread_idx() noexcept -> size_t{
        
        auto ptr = concurrency_resource_container::get().thrid_to_idx_map.find(std::this_thread::get_id());

        if constexpr(DEBUG_MODE_FLAG){
            if (ptr == concurrency_resource_container::get().thrid_to_idx_map.end()){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        return ptr->second;
    }

    auto daemon_register(daemon_kind_t daemon_kind, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<size_t, exception_t>{

        if constexpr(DEBUG_MODE_FLAG){
            auto ptr = concurrency_resource_container::get().thrid_to_idx_map.find(std::this_thread::get_id());
            if (ptr != concurrency_resource_container::get().thrid_to_idx_map.end()){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::PTHREAD_CAUSA_SUI));
                std::abort();
            }
        }

        concurrency_resource_container::get().daemon_controller->_register(daemon_kind, std::move(worker));
    }

    void daemon_deregister(size_t id) noexcept{

        if constexpr(DEBUG_MODE_FLAG){
            auto ptr = concurrency_resource_container::get().thrid_to_idx_map.find(std::this_thread::get_id());
            if (ptr != concurrency_resource_container::get().thrid_to_idx_map.end()){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::PTHREAD_CAUSA_SUI));
                std::abort();
            }
        }

        concurrency_resource_container::get().daemon_controller->deregister(id);
    }

    using daemon_deregister_t = void (size_t) noexcept; 

    auto daemon_saferegister(daemon_kind_t daemon_kind, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<dg::network_genult::nothrow_immutable_unique_raii_wrapper<size_t, daemon_deregister_t>, exception_t>{

        std::expected<size_t, exception_t> handle = daemon_register(daemon_kind, std::move(worker));
        
        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return {handle.value(), daemon_deregister};
    }

    using daemon_raii_handle_t = dg::network_genult::nothrow_immutable_unique_raii_wrapper<size_t, daemon_deregister_t>;
};

#endif