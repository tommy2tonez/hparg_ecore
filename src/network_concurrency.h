
#ifndef __NETWORK_CONCURRENCY_H__
#define __NETWORK_CONCURRENCY_H__

//define HEADER_CONTROL 3

#include <stddef.h>
#include <stdint.h>
#include <thread>
#include <vector>
#include <memory>
#include "network_concurrency_impl1.h"
#include <bit>
#include "network_raii_x.h"
#include "dense_hash_map/dense_hash_map.hpp"

namespace dg::network_concurrency{

    static inline constexpr size_t THREAD_COUNT = 32u;

    using namespace dg::network_concurrency_impl1::daemon_option_ns;
    using WorkerInterface = dg::network_concurrency_impl1::WorkerInterface; 

    struct signature_dg_network_concurrency{}; 

    struct ConcurrencyResource{
        std::unique_ptr<dg::network_concurrency_impl1::DaemonControllerInterface> daemon_controller; 
        jg::dense_hash_map<std::thread::id, size_t> thrid_to_idx_map;
    };

    inline ConcurrencyResource concurrency_resource{};

    void init(){

        // auto config = dg::network_concurrency_impl1_app::Config{AFFINE_POLICY, 
        //                                                         COMPUTING_DAEMON_NETWORK_THREAD_COUNT,
        //                                                         IO_DAEMON_THREAD_COUNT,
        //                                                         TRANSPORTATION_DAEMON_THREAD_COUNT,
        //                                                         HEARTBEAT_DAEMON_THREAD_COUNT};

        // auto [controller, thr_vec]              = dg::network_concurrency_impl1_app::spawn(config); 
        // thr_vec                                 = dg::network_genult::enumerate(std::move(thr_vec));
        // concurrency_resource.daemon_controller  = std::move(controller);
        // concurrency_resource.thrid_to_idx_map   = jg::dense_hash_map<std::thread::id, size_t>(thr_vec.begin(), thr_vec.end(), thr_vec.size());
    }

    void deinit() noexcept{

        concurrency_resource = {};
    }

    auto this_thread_idx() noexcept -> size_t{
        
        auto ptr = concurrency_resource.thrid_to_idx_map.find(std::this_thread::get_id());

        if constexpr(DEBUG_MODE_FLAG){
            if (ptr == concurrency_resource.thrid_to_idx_map.end()){
                std::abort();
            }
        }

        return ptr->second;
    }

    auto daemon_register(daemon_kind_t daemon_kind, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<size_t, exception_t>{

        auto ptr = concurrency_resource.thrid_to_idx_map.find(std::this_thread::get_id());
        
        if (ptr != concurrency_resource.thrid_to_idx_map.end()){
            return std::unexpected(dg::network_exception::PTHREAD_CAUSA_SUI);
        }

        concurrency_resource.daemon_controller->_register(daemon_kind, std::move(worker));
    }

    void daemon_deregister(size_t id) noexcept{

        auto ptr = concurrency_resource.thrid_to_idx_map.find(std::this_thread::get_id());
    
        if (ptr != concurrency_resource.thrid_to_idx_map.end()){
            std::abort();
        }

        concurrency_resource.daemon_controller->deregister(id);
    }

    using daemon_deregister_t = void (*)(size_t) noexcept; 

    auto daemon_saferegister(daemon_kind_t daemon_kind, std::unique_ptr<WorkerInterface> worker) noexcept -> std::expected<dg::unique_resource<size_t, daemon_deregister_t>, exception_t>{

        std::expected<size_t, exception_t> handle = daemon_register(daemon_kind, std::move(worker));
        
        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return dg::unique_resource<size_t, daemon_deregister_t>(handle.value(), daemon_deregister);
    }

    using daemon_raii_handle_t = dg::unique_resource<size_t, daemon_deregister_t>;
};

#endif