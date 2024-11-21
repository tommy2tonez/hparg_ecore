#ifndef __NETWORK_UMA_H__
#define __NETWORK_UMA_H__

#include <stdint.h>
#include <stddef.h>
#include "network_uma_tlb.h"
#include "network_exception.h"
#include "network_exception_handler.h"
#include "network_randomizer.h" 
#include "stdx.h"
#include "network_raii_x.h"
#include "network_uma_tlb_impl1.h"
#include "network_pointer.h" 
#include <atomic>

namespace dg::network_uma{

    //alright - let's stop treating the compiler like it's regarded - and give some room for optimizations - I think it's decent enough by fencing initializations - unless you are accessing local variables
    //alright guys - after fact-checking - compiler is actually regarded and std::atomic_signal_fence(std::memory_order_acquire) is NOT OPTIONAL to access static inline class members or inline global variables if you are compiling this from different translation units
    //atomic_signal_fence is NOT used for fencing atomic in this case - but to force a read of the global variable and flush any previous assumptions in the local translation unit
    //where to put the atomic_signal_fence is the developer's choice - the atomic_signal_fence is not supposed to be there - because it's the compiler responsibility to volatile the shared_addr inline variables - but we cannot count on the compiler to do that - because that's undefined behavior
    //the ONLY thing that we can count on is the static inline class member and inline global member to have one address across multiple translation units - and their compiler-assumed values during unit compilations are as if we are in a concurrent context - which is precisely why std::atomic_singal_fence(std::memory_order_acquire) is used

    using device_id_t                                   = dg::network_pointer::device_id_t;
    using uma_ptr_t                                     = dg::network_pointer::uma_ptr_t;
    using vma_ptr_t                                     = dg::network_pointer::vma_ptr_t;
    static inline constexpr size_t MEMREGION_SZ         = dg::network_pointer::MEMREGION_SZ;
    static inline constexpr size_t PROXY_COUNT          = dg::network_pointer::MEMORY_PLATFORM_SZ;
    static inline constexpr size_t MAX_PROXY_PER_REGION = dg::network_pointer::MAX_MEMORY_PLATFORM_SZ;

    struct signature_dg_network_uma{}; 

    struct MemoryCopyAbstractDeviceInterface{
        virtual ~MemoryCopyAbstractDeviceInterface() noexcept = default;
        virtual void memcpy(vma_ptr_t, vma_ptr_t, size_t) noexcept = 0;
    };

    class InternalMemoryCopyDevice: public dg::network_uma_tlb_impl1::interface::MemoryCopyDeviceInterface<InternalMemoryCopyDevice>{

        private:

            static inline std::unique_ptr<MemoryCopyAbstractDeviceInterface> abstract_device;

        public:

            using ptr_t = vma_ptr_t; 

            static void init(std::unique_ptr<MemoryCopyAbstractDeviceInterface> arg){

                if (arg == nullptr){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }

                abstract_device = std::move(arg);
            }

            static void deinit() noexcept{

                abstract_device = nullptr;
            }

            static inline void memcpy(ptr_t dst, ptr_t src, size_t n) noexcept{
          
                abstract_device->memcpy(dst, src, n);
            } 
    };

    using direct_tlb_instance               = dg::network_uma_tlb_impl1::generic::DirectTLB<signature_dg_network_uma, device_id_t, uma_ptr_t, vma_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>>;
    using tlb_instance                      = dg::network_uma_tlb_impl1::generic::BiexTLB<signature_dg_network_uma, InternalMemoryCopyDevice::interface_t, device_id_t, uma_ptr_t, vma_ptr_t, std::integral_constant<size_t, MEMREGION_SZ>>;
    using uma_ptr_access                    = dg::network_uma_tlb::access::StdSafeRegionAccess<signature_dg_network_uma, uma_ptr_t, device_id_t, MEMREGION_SZ>; 
    using metadata_getter                   = dg::network_uma_tlb::access::MetadataGetter<signature_dg_network_uma, device_id_t, uma_ptr_t, MEMREGION_SZ>;
    using map_resource_handle_t             = typename tlb_instance::map_resource_handle_t; 
    using recursive_map_resource_handle_t   = decltype(dg::network_uma_tlb::rec_lck::recursive_resource_type(tlb_instance{}));

    void init(uma_ptr_t * uma_region_arr, vma_ptr_t * vma_region_arr, device_id_t * device_id_arr, bool * is_proxy_arr, size_t n, std::unique_ptr<MemoryCopyAbstractDeviceInterface> memory_copy_device){

        stdx::memtransaction_guard transaction_guard;
        InternalMemoryCopyDevice::init(std::move(memory_copy_device));
        direct_tlb_instance::init(uma_region_arr, vma_region_arr, device_id_arr, n);
        tlb_instance::init(uma_region_arr, vma_region_arr, device_id_arr, is_proxy_arr, n);
        uma_ptr_access::init(uma_region_arr, device_id_arr, n);
        metadata_getter::init(uma_region_arr, device_id_arr, n);
    }

    void deinit() noexcept{

        stdx::memtransaction_guard transaction_guard;
        metadata_getter::deinit();
        uma_ptr_access::deinit();
        tlb_instance::deinit();
        direct_tlb_instance::deinit();
        InternalMemoryCopyDevice::deinit();
    }

    auto map_direct(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<vma_ptr_t, exception_t>{
        
        std::atomic_signal_fence(std::memory_order_acquire);
        exception_t ptrchk = uma_ptr_access::safecthrow_access(device_id, ptr); 

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        }

        return direct_tlb_instance::map(device_id, ptr);
    }

    auto map_try(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<std::optional<map_resource_handle_t>, exception_t>{

        std::atomic_signal_fence(std::memory_order_acquire);
        exception_t ptrchk = uma_ptr_access::safecthrow_access(device_id, ptr);

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        }

        return tlb_instance::map_try(device_id, ptr);
    }

    auto map_wait(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        std::atomic_signal_fence(std::memory_order_acquire);
        exception_t ptrchk = uma_ptr_access::safecthrow_access(device_id, ptr);

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        }

        return tlb_instance::map_wait(device_id, ptr);
    }

    auto map_wait(uma_ptr_t ptr) noexcept -> std::expected<map_resource_handle_t, exception_t>{

        std::atomic_signal_fence(std::memory_order_acquire);
        exception_t ptrchk = uma_ptr_access::safecthrow_access(ptr);

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        } 

        while (true){
            size_t random_value     = dg::network_randomizer::randomize_xrange(std::integral_constant<size_t, MAX_PROXY_PER_REGION>{}); 
            size_t device_sz        = metadata_getter::device_count(ptr);
            size_t idx              = random_value % device_sz; //
            device_id_t device_id   = metadata_getter::device_at(ptr, idx);

            if (auto rs = tlb_instance::map_try(device_id, ptr); rs.has_value()){
                return rs.value();
            }
        }
    }

    void map_release(map_resource_handle_t map_resource) noexcept{

        std::atomic_signal_fence(std::memory_order_acquire);
        tlb_instance::map_release(map_resource);
    }

    static inline auto map_release_lambda = [](map_resource_handle_t map_resource) noexcept{
        map_release(map_resource);
    };

    auto get_vma_ptr(map_resource_handle_t map_resource) noexcept -> vma_ptr_t{

        std::atomic_signal_fence(std::memory_order_acquire);
        return tlb_instance::get_vma_ptr(map_resource);
    }

    auto get_vma_ptr(recursive_map_resource_handle_t map_resource) noexcept -> vma_ptr_t{

        std::atomic_signal_fence(std::memory_order_acquire);
        return dg::network_uma_tlb::rec_lck::get_vma_ptr(map_resource); //refactor
    }

    auto map_safewait(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::expected<dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{

        std::atomic_signal_fence(std::memory_order_acquire);
        auto map_rs = network_uma::map_wait(device_id, ptr); 

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    auto map_safewait(uma_ptr_t ptr) noexcept -> std::expected<dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>, exception_t>{
        
        std::atomic_signal_fence(std::memory_order_acquire);
        auto map_rs = network_uma::map_wait(ptr);

        if (!map_rs.has_value()){
            return std::unexpected(map_rs.error());
        }

        return dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>(map_rs.value(), map_release_lambda);
    }

    template <size_t SZ>
    auto lockmap_safetry_many(const std::array<std::pair<device_id_t, uma_ptr_t>, SZ>& args) noexcept /* -> std::expected<std::optional<std::array<unique_resource<recursive_map_resource_handle_t, decltype(destructor)>, SZ>>, exception_t> */ {

        using rec_lockmap_resource_t    = decltype(dg::network_uma_tlb::rec_lck::recursive_lockmap_try_many(tlb_instance{}, args));
        using ret_t                     = std::expected<rec_lockmap_resource_t, exception_t>; 

        std::atomic_signal_fence(std::memory_order_acquire);

        for (size_t i = 0u; i < args.size(); ++i){
            exception_t ptrchk = uma_ptr_access::safecthrow_access(args[i].first, args[i].second);

            if (dg::network_exception::is_failed(ptrchk)){
                return ret_t{std::unexpected(ptrchk)};
            }
        }

        return ret_t{dg::network_uma_tlb::rec_lck::recursive_lockmap_try_many(tlb_instance{}, args)};
    }

    template <size_t SZ>
    auto lockmap_safewait_many(const std::array<std::pair<device_id_t, uma_ptr_t>, SZ>& args) noexcept /* -> std::expected<std::array<unique_resource<recursive_map_resource_handle_t, decltype(destructor)>, SZ>, exception_t> */ {

        using rec_lockmap_resource_t    = decltype(dg::network_uma_tlb::rec_lck::recursive_lockmap_wait_many(tlb_instance{}, args));
        using ret_t                     = std::expected<rec_lockmap_resource_t, exception_t>; 

        std::atomic_signal_fence(std::memory_order_acquire);

        for (size_t i = 0u; i < args.size(); ++i){
            exception_t ptrchk = uma_ptr_access::safecthrow_access(args[i].first, args[i].second);

            if (dg::network_exception::is_failed(ptrchk)){
                return ret_t{std::unexpected(ptrchk)};
            }
        }

        return ret_t{dg::network_uma_tlb::rec_lck::recursive_lockmap_wait_many(tlb_instance{}, args)};
    }
    
    auto device_count(uma_ptr_t ptr) noexcept -> std::expected<size_t, exception_t>{

        std::atomic_signal_fence(std::memory_order_acquire);
        exception_t ptrchk = uma_ptr_access::safecthrow_access(ptr);

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        }

        return metadata_getter::device_count(ptr);
    } 

    auto device_at(uma_ptr_t ptr, size_t idx) noexcept -> std::expected<device_id_t, exception_t>{

        std::atomic_signal_fence(std::memory_order_acquire);
        exception_t ptrchk = uma_ptr_access::safecthrow_access(ptr);

        if (dg::network_exception::is_failed(ptrchk)){
            return std::unexpected(ptrchk);
        }

        return metadata_getter::device_at(ptr, idx);
    }

//---

    auto map_direct_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> vma_ptr_t{

        std::atomic_signal_fence(std::memory_order_acquire);
        uma_ptr_access::safe_access(device_id, ptr);

        return direct_tlb_instance::map(device_id, ptr); 
    } 

    auto map_try_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<map_resource_handle_t>{

        std::atomic_signal_fence(std::memory_order_acquire);
        uma_ptr_access::safe_access(device_id, ptr);

        return tlb_instance::map_try(device_id, ptr);
    }

    auto map_wait_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> map_resource_handle_t{

        std::atomic_signal_fence(std::memory_order_acquire);
        uma_ptr_access::safe_access(device_id, ptr);

        return tlb_instance::map_wait(device_id, ptr);
    }

    auto map_safewait_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>{

        std::atomic_signal_fence(std::memory_order_acquire);
        return dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>(map_wait_nothrow(device_id, ptr), map_release_lambda);
    }

    auto map_safetry_nothrow(device_id_t device_id, uma_ptr_t ptr) noexcept -> std::optional<dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>>{

        std::atomic_signal_fence(std::memory_order_acquire);
        uma_ptr_access::safe_access(device_id, ptr);
        auto rs = tlb_instance::map_try(device_id, ptr);

        if (!rs.has_value()){
            return std::nullopt;
        }

        return dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>(rs.value(), map_release_lambda);
    } 

    template <size_t SZ>
    auto lockmap_safetry_many_nothrow(const std::array<std::pair<device_id_t, uma_ptr_t>, SZ>& args) noexcept /* -> std::optional<std::array<unique_resource<recursive_map_resource_handle_t, decltype(destructor)>, SZ>> */ {

        std::atomic_signal_fence(std::memory_order_acquire);

        for (size_t i = 0u; i < SZ; ++i){
            uma_ptr_access::safe_access(args[i].first, args[i].second);
        }

        return dg::network_uma_tlb::rec_lck::recursive_lockmap_try_many(tlb_instance{}, args);
    }

    template <size_t SZ>
    auto lockmap_safewait_many_nothrow(const std::array<std::pair<device_id_t, uma_ptr_t>, SZ>& args) noexcept /* -> std::array<unique_resource<recursive_map_resource_handle_t, decltype(destructor)>, SZ> */ {

        std::atomic_signal_fence(std::memory_order_acquire);

        for (size_t i = 0u; i < SZ; ++i){
            uma_ptr_access::safe_access(args[i].first, args[i].second);
        }

        return dg::network_uma_tlb::rec_lck::recursive_lockmap_wait_many(tlb_instance{}, args);
    }
    
    auto device_count_nothrow(uma_ptr_t ptr) noexcept -> size_t{

        std::atomic_signal_fence(std::memory_order_acquire);
        uma_ptr_access::safe_access(ptr);

        return metadata_getter::device_count(ptr);
    }

    auto device_at_nothrow(uma_ptr_t ptr, size_t idx) noexcept -> device_id_t{

        std::atomic_signal_fence(std::memory_order_acquire);
        uma_ptr_access::safe_access(ptr);

        return metadata_getter::device_at(ptr, idx);
    }

    auto map_wait_nothrow(uma_ptr_t ptr) noexcept -> map_resource_handle_t{

        std::atomic_signal_fence(std::memory_order_acquire);
        uma_ptr_access::safe_access(ptr);

         while (true){
            size_t random_value     = dg::network_randomizer::randomize_xrange(std::integral_constant<size_t, MAX_PROXY_PER_REGION>{});
            size_t device_sz        = metadata_getter::device_count(ptr);
            size_t idx              = random_value % device_sz;
            device_id_t device_id   = metadata_getter::device_at(ptr, idx);

            if (auto rs = tlb_instance::map_try(device_id, ptr); rs.has_value()){
                return rs.value();
            }
        }
    }

    auto map_wait_safe_nothrow(uma_ptr_t ptr) noexcept -> dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>{

        std::atomic_signal_fence(std::memory_order_acquire);
        return dg::unique_resource<map_resource_handle_t, decltype(map_release_lambda)>(map_wait_nothrow(ptr), map_release_lambda);
    }
}

#endif