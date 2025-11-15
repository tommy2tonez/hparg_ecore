#ifndef __NETWORK_KERNEL_MAP_X_H__
#define __NETWORK_KERNEL_MAP_X_H__

//define HEADER_CONTROL 10

#include <stdint.h>
#include <stddef.h> 
#include <filesystem>
#include "network_exception.h"
#include "network_kernelmap_x_impl1.h" 
#include "network_raii_x.h"
#include "network_type_traits_x.h"
#include "network_std_container.h"

namespace dg::network_kernelmap_x{

    //alright fellas - we have issues from github feedback
    //let's stick to the assumption of atomic + relaxed function (concurrent-sufficient)

    //let's assume relaxed function means that the concurrent state of the program before and after a function invoke (of unspecified order) is correct
    //let's assume we write another function - we want to write a recusrive resolution to maintain the requirements of relaxed

    //let's prove by using induction
    //assume that we have no serialization inference - such is a function is perfectly executed as if the computation flow is purely based on the computation serialization of the arguments    
    //then our recursive resolution maintains correct (induction) - and our function is relaxed

    //assume that we have a serialization inference (which is another word for memory_ordering) - then we place all the relaxed statements inside a concurrent transaction block of [acquire, release]
    //then our recursive resolution maintains correct (induction) - and our function is relaxed
    //assume that we detach the serialization inference - which is a bad practice - we must use std::thread_signal_fence(std::memory_order_release)

    //this is completely valid syntax 

    //std::atomic<int> lck = {};
    //while (!lck.cmp_exchg(EMPTY_STATE, ACQUIRE_STATE, relaxed)){}
    //some_local_variable += lck_value;
    //std::cout << some_local_variable;
    //std::atomic_signal_fence(std::memory_order_release);
    //lck.exchange(EMPTY_STATE, relaxed)

    //this is the equivalence
    //std::atomic<int> lck = {};
    //while (!lck.cmp_exchg(EMPTY_STATE, ACQUIRE_STATE, acquired)){}
    //some_local_variable += lck_value;
    //std::cout << some_local_variable;
    //lck.exchange(EMPTY_STATE, release)

    //alright - I got the message from my kernel brother to do fork - because concurrency in C is - first not implemented correctly by C compilers - second not defined correctly by the std - third very hard to write correctly - fourth no one does it (except the ones that truly understand it - even so they'd often get it wrong)
    //we need to make sure that every real function (we are not talking about always_inline) is self_concurrent_sufficient + relaxed + atomic
    //and use std::atomic_signal_fence(std::memory_order_seq_cst) whenever possible if there is an indirect serialization inference (like kernel_map in and out) - because it's just so
    //alright - std::atomic_signal_fence(std::memory_order_seq_cst) does not guarantee the release of local variables but do guarantee the release of concurrent variables (this is where things get hard and involve the usage of volatiles for special cases)
    //things are tough folks - and I think that my brother does get a point of not using concurrency - it's a topic for the true experts

    using fsys_ptr_t                = dg::network_pointer::fsys_ptr_t;   
    using map_resource_handle_t     = dg::network_kernelmap_x_impl1::model::ConcurrentMapResource;

    inline dg::network_kernelmap_x_impl1::interface::ConcurrentMapInterface * volatile map_instance; //we are seriously reconsidering volatile (according to git pull requests) - because memory ordering might not work as well as we expect

    void init(const dg::unordered_map<fsys_ptr_t, std::filesystem::path>& bijective_alias_map,
              const std::shared_ptr<dg::network_kernelmap_x_impl1::interface::KernelDiskIODeviceInterface>& kernel_io_device,
              size_t memregion_sz,
              double ram_to_disk_ratio,
              size_t distribution_factor){

        stdx::memtransaction_guard transaction_guard;
        auto tmp_map_instance   = dg::network_kernelmap_x_impl1::make(bijective_alias_map,
                                                                      kernel_io_device,
                                                                      memregion_sz,
                                                                      ram_to_disk_ratio,
                                                                      distribution_factor);
        map_instance            = tmp_map_instance.get();
        tmp_map_instance.release();
    }

    void deinit() noexcept{

        stdx::memtransaction_guard transaction_guard;
        delete map_instance;
    }

    auto get_map_instance() noexcept -> dg::network_kernelmap_x_impl1::interface::ConcurrentMapInterface *{

        std::atomic_signal_fence(std::memory_order_acquire); //we are doing double protection
        return map_instance;
    } 

    auto map(fsys_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        return get_map_instance()->map(ptr);
    }

    auto remap_try(map_resource_handle_t map_resource, fsys_ptr_t ptr) noexcept -> std::expected<std::optional<map_resource_handle_t>, exception_t>{

        return get_map_instance()->remap_try(map_resource, ptr);
    }

    auto remap(map_resource_handle_t map_resource, fsys_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        auto _map_instance  = get_map_instance();
        auto remap_rs       = _map_instance->remap_try(map_resource, ptr);

        if (remap_rs.has_value() && remap_rs.value().has_value()){
            return remap_rs.value().value();
        }

        auto newmap_rs      = _map_instance->map(ptr);

        if (newmap_rs.has_value()){
            _map_instance->unmap(map_resource);
        }

        return newmap_rs;
    }

    auto map_nothrow(fsys_ptr_t ptr) noexcept -> map_resource_handle_t{

        return dg::network_exception_handler::nothrow_log(network_kernelmap_x::map(ptr));
    }

    void map_release(map_resource_handle_t map_resource) noexcept{

        get_map_instance()->unmap(map_resource);
    }

    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        network_kernelmap_x::map_release(map_resource);
    };

    auto map_safe(fsys_ptr_t ptr) noexcept -> std::expected<dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        auto map_rs = network_kernelmap_x::map(ptr);

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto map_safe_nothrow(fsys_ptr_t ptr) noexcept -> dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>{

        return dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>(network_kernelmap_x::map_nothrow(ptr), map_release_lambda);
    }

    auto get_host_ptr(map_resource_handle_t map_resource) noexcept -> void *{

        return map_resource.ptr();
    }
}

#endif