#ifndef __NETWORK_UMA_H__
#define __NETWORK_UMA_H__

#include <stdint.h>
#include <stddef.h>
#include "network_uma_definition.h"
#include "network_uma_tlb.h"
#include "network_exception_handler.h"
#include "network_randomizer.h" 

namespace dg::network_uma{
    
    struct signature_dg_network_uma{}; 

    using namespace dg::network_uma_definition; 
    using namespace dg::network_uma_tlb::memqualifier_taxonomy;  
    
    using tlb_factory                   = dg::network_uma_tlb::v1::Factory<signature_dg_network_uma, device_id_t, uma_ptr_t, device_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>, std::integral_constant<size_t, PROXY_COUNT>>; 
    using tlb_instance                  = typename tlb_factory::tlb;
    using tlbdirect_instance            = typename tlb_factory::tlb_direct;
    using uma_ptr_access                = typename tlb_factory::uma_ptr_access;
    using metadata_getter               = typename tlb_factory::uma_metadata; 
    using map_resource_handle_t         = typename tlb_factory::map_resource_handle_t; 

    static_assert(std::is_trivial_v<map_resource_handle_t>);
    static_assert(dg::is_immutable_resource_handle_v<map_resource_handle_t>);

    static inline constexpr size_t MAX_PROXY_PER_REGION = 32;

    //map <arg> -> <result>
    //reachability(result) >= reachability(arg)
    //reachability contract is either fulfilled by allocation or injecting data - up to the user to choose void init() method 
    //<function_name> assumes all inputs
    //<function_name>_nothrow assumes valid inputs, equivalents to __builtin_assume(is_precond_met(args...))
    
    void init(uma_ptr_t * host_region, vma_ptr_t * device_region, device_id_t * device_id, size_t n){

        tlb_factory::init(host_region, device_region, device_id, n);
    }
    
    auto map_direct(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<vma_ptr_t, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(device_id, ptr); 

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        }

        return tlbdirect_instance::map(device_id, ptr);
    }

    auto map_direct_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> vma_ptr_t{

        uma_ptr_access::safe_access(device_id, ptr);
        return tlbdirect_instance::map(device_id, ptr); 
    } 

    auto map_try(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<std::optional<map_resource_handle_t>, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(device_id, ptr);

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        }
        
        return tlb_instance::map_try(device_id, ptr);
    }

    auto map_try_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<map_resource_handle_t, exception_t>{

        uma_ptr_access::safe_access(device_id, ptr); //safe_access here for debuggability (to narrow the internal_error location, map_try has its own internal mechanism - yet the deeper the stack, the harder the debugability) - where the developers set up their bug fence is optional - because the contract of meeting function's precond has been broken
        return tlb_instance::map_try(device_id, ptr);
    }

    auto map_wait(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(device_id, ptr); //this is a feature - not a bug_fence - this feature used in conjunction with invoker_nothrow == bug_fence - this is an engineering practice 

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        }

        return tlb_instance::map_wait(device_id, ptr);
    }

    auto map_wait_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> map_resource_handle_t{

        uma_ptr_access::safe_access(device_id, ptr);
        return tlb_instance::map_wait(device_id, ptr);
    }

    void map_release(map_resource_handle_t map_resource) noexcept{

        tlb_instance::map_release(map_resource);
    }

    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        map_release(map_resource);
    };

    auto mapsafe_wait(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        auto map_rs = map_wait(device_id, ptr); 

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto mapsafe_wait_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>{

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_wait_nothrow(device_id, ptr), map_release_lambda);
    }

    auto mapsafe_try_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>>{

        auto rs = map_try_nothrow(device_id, ptr); 

        if (!static_cast<bool>(rs)){
            return std::nullopt;
        }

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(rs.value(), map_release_lambda);
    } 

    template <size_t SZ>
    auto mapsafe_try_many_nothrow(std::array<std::pair<device_id_t, uma_ptr_t>, SZ> arg) noexcept -> std::optional<std::array<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, SZ>>{

        std::array<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, SZ> rs{};

        for (size_t i = 0; i < SZ; ++i){
            rs[i] = dg::network_genult::tuple_invoke(mapsafe_try_nothrow, arg[i]); //perf-issue - compiler responsibility

            if (!static_cast<bool>(rs[i])){
                return std::nullopt;
            }
        }

        return rs;
    }

    template <size_t SZ>
    auto mapsafe_wait_many_nothrow(std::array<std::pair<device_id_t, uma_ptr_t>, SZ> arg) noexcept -> std::array<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, SZ>{

        while (true){
            if (auto rs = mapsafe_try_many_nothrow(arg); static_cast<bool>(rs)){
                return rs.value();
            }
        }
    }

    //^^^
    //-the interface is consistent yet the implementation of a specific map (memregion) should not be here - for future extension
    
    struct RecusriveMapResource{
        map_resource_handle_t handle;
        uma_ptr_t region_ptr;
        size_t offset;
        bool responsibility_flag;
    };

    using map_recusrive_resource_handle_t = RecusriveMapResource;

    void map_recursive_release(map_recusrive_resource_handle_t map_resource) noexcept{

        if (map_resource.responsibility_flag){
            map_release(map_resource.handle);

            std::unordered_map<uma_ptr_t, std::pair<device_id_t, map_resource_handle_t>>& map_ins = map_recursive_resource_instance::get();
            auto map_ptr = map_ins.find(map_resource.region_ptr);

            if (map_ptr == map_ins.end()){ //DEBUG_FLAG here - all internal error should be disable-able
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_ERROR));
                std::abort();
            }

            map_ins.erase(map_ptr);
        }
    }

    static inline auto map_recursive_release_lambda = [](map_recursive_resource_handle_t map_resource) noexcept{
        map_recursive_release(map_resource);
    };

    auto mapsafe_recusivetry_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>>{
        
        std::unordered_map<uma_ptr_t, std::pair<device_id_t, map_resource_handle_t>>& map_ins = map_recursive_resource_instance::get();
        uma_ptr_t region_ptr        = dg::memult::region(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
        size_t region_off           = dg::memult::region_offset(ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
        auto map_ptr                = map_ins.find(region_ptr);

        if (map_ptr == map_ins.end()){
            auto rs = map_try_nothrow(device_id, region_ptr);

            if (!static_cast<bool>(rs)){
                return std::nullopt;
            }

            map_ins[region_ptr]     = std::make_pair(device_id, rs.value());
            auto handle             = map_recursive_resource_handle_t{rs.value(), region_ptr, region_off, true};
            return dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>(handle, map_recursive_release_lambda);
        }

        auto [bucket_device_id, bucket_resource] = *map_ptr;

        if (bucket_device_id != device_id){ //DEBUG_FLAG here - all internal error should be disable-able
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        auto handle = map_recursive_resource_handle_t{bucket_resource, region_ptr, region_off, false};  //region_ptr should be std::optional - yet I find it more intuitive to have a separate bool flag
        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>(handle, map_recursive_release_lambda);
    }

    auto mapsafe_recursivewait_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>{

        while (true){
            if (auto rs = mapsafe_recusivetry_nothrow(device_id, ptr); static_cast<bool>(rs)){
                return rs.value();
            }
        }
    } 

    template <size_t SZ>
    auto mapsafe_recursivetry_many_nothrow(std::array<std::pair<device_id_t, uma_ptr_t>, SZ> arg) noexcept -> std::optional<std::array<dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>, SZ>>{

        std::array<dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>, SZ> rs{};

        for (size_t i = 0; i < SZ; ++i){
            rs[i] = dg::network_genult::tuple_invoke(mapsafe_recusivetry_nothrow, arg[i]);

            if (!static_cast<bool>(rs[i])){
                return std::nullopt;
            }
        }

        return rs;
    }

    template <size_t SZ>
    auto mapsafe_recursivewait_many_nothrow(std::array<std::pair<device_id_t, uma_ptr_t>, SZ> arg) noexcept -> std::array<dg::genult::nothrow_immutable_unique_raii_wrapper<map_recursive_resource_handle_t, decltype(map_recursive_release_lambda)>, SZ>{

        while (true){
            if (auto rs = mapsafe_recursivetry_many_nothrow(arg); static_cast<bool>(rs)){
                return rs.value();
            }
        }
    }
    //vvv

    //defined for every std::lock_guard<> use cases
    //undefined otherwise
    auto map_relguard(map_resource_handle_t map_resource) noexcept{

        static int i    = 0;
        auto destructor = [=](int *) noexcept{
            map_release(map_resource);
        };

        return std::unique_ptr<int, decltype(destructor)>(&i, destructor);
    } 

    auto get_vma_ptr(map_resource_handle_t map_resource) noexcept -> vma_ptr_t{

        return tlb_instance::get_vma_ptr(map_resource);
    }

    auto get_vma_ptr(map_recursive_resource_handle_t map_resource) noexcept -> vma_ptr_t{

        return dg::memult::advance(get_vma_ptr(map_resource.handle), map_resource.offset);
    } 

    auto device_count(uma_ptr_t ptr) noexcept -> std::expected<size_t, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(ptr);

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        }

        return metadata_getter::device_count(ptr);
    } 

    auto device_count_nothrow(uma_ptr_t ptr) noexcept -> size_t{

        uma_ptr_access::safe_access(ptr);
        return metatdata_getter::device_count(ptr);
    }

    auto device_at(uma_ptr_t ptr, size_t idx) noexcept -> std::expected<device_id_t, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(ptr);

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        }

        return metadata_getter::device_at(ptr, idx);
    } 

    auto device_at_nothrow(uma_ptr_t ptr, size_t idx) noexcept -> device_id_t{

        uma_ptr_access::safe_access(ptr);
        return metadata_getter::device_at(ptr, idx);
    }

    auto internal_device_random_at_nothrow(uma_ptr_t ptr) noexcept -> device_id_t{ //it seems to me that this is a recursive lock problem rather than a randomization problem - yet it's fine to implement it this way - considering that PROXY_PER_PTR should be <= 3 in every scenerio - recursive lock should be a derivative of this interface

        size_t random_value = dg::network_randomizer::randomize_range(std::integral_constant<size_t, MAX_PROXY_PER_REGION>{}); 
        size_t device_sz    = device_count_nothrow(ptr);
        size_t idx          = random_value % device_sz;

        return device_at_nothrow(ptr, idx);
    } 

    auto map_wait(uma_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        exception_t ptrchk = uma_ptr_access::safecthrow_access(ptr);

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        } 

        while (true){
            device_id_t id = internal_device_random_at_nothrow(ptr);
            std::optional<map_resource_handle_t> map_rs = map_try_nothrow(id, ptr);

            if (static_cast<bool>(map_rs)){
                return map_rs.value();
            }
        }
    }

    auto map_wait_nothrow(uma_ptr_t ptr) noexcept -> map_resource_handle_t{

         while (true){
            device_id_t id = internal_device_random_at_nothrow(ptr); 
            std::optional<map_resource_handle_t> map_rs = map_try_nothrow(id, ptr);

            if (static_cast<bool>(map_rs)){
                return map_rs.value();
            }
        }
    }

    auto map_wait_safe(uma_ptr_t ptr) noexcept -> std::expected<dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        auto map_rs = map_wait(ptr);

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto map_wait_safe_nothrow(uma_ptr_t ptr) noexcept -> dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>{

        return dg::genult::nothrow_immutable_unique_raii_wrapper<map_resource_handle_t, decltype(map_release_lambda)>(map_wait_nothrow(ptr), map_release_lambda);
    }
}

#endif