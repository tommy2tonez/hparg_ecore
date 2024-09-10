#ifndef __NETWORK_VMAMAP_H__
#define __NETWORK_VMAMAP_H__

#include <variant>
#include "network_kernelmap_x.h"
#include "network_cudafsmap_x.h"
#include "network_log.h"
#include "network_exception.h"

namespace dg::network_vmamap{

    struct HostResource{
        void * value;
    };

    struct FsysResource{
        dg::network_kernelmap_x::map_resource_handle_t value;
    };

    struct CudaResource{
        cuda_ptr_t value;
    };

    struct CuFSResource{
        dg::network_cudafsmap_x::map_resource_handle_t value;
    };

    using vmamap_resource_handle_t = std::variant<HostResource, CudaResource, FsysResource, CudaFsysResource>;

    //fine - defense-line set up 
    //<function_name> takes all input
    //<function_name>_nothrow takes certain input
    //usually latter invokes former 
    //yet this is performance critical - it's reversed
    //optimal security when DEBUG_MODE is enabled
    //optimal speed otherwise

    auto internal_safecthrow_vmaptr_access(vma_ptr_t ptr) noexcept -> exception_t{
        
        auto dict_ptr = vmaptr_hashset->find(ptr);

        if (dict_ptr == vmaptr_hashset->end()){
            return dg::network_exception::BAD_PTR_ACCESS;
        }

        return dg::network_exception::SUCCESS;
    }

    void internal_safe_vmaptr_access(vma_ptr_t ptr) noexcept{

        if constexpr(DEBUG_MODE_FLAG){
            exception_t err = internal_safecthrow_vmaptr_access(ptr);
            if (dg::network_exception::is_failed(err)){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        } else{
            return ptr;
        }
    }

    void unmap(vmamap_resource_handle_t resource) noexcept{

        if (std::holds_alternative<HostResource>(resource)){
            return;
        }

        if (std::holds_alternative<FsysResource>(resource)){
            dg::network_kernelmap_x::map_release(std::get<FsysResource>(resource).value);
            return;
        }

        if (std::holds_alternative<CudaResource>(resource)){
            return;
        }

        if (std::holds_alternative<CuFSResource>(resource)){
            dg::network_cudafsmap_x::map_release(std::get<CuFSResource>(resource).value);
            return;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_lock_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }
    } 

    using unmap_t = void (*)(vmamap_resource_handle_t) noexcept;

    auto map_nothrow(vma_ptr_t ptr) noexcept -> vmamap_resource_handle_t{

        internal_safe_vmaptr_access(ptr);

        if (dg::network_virtual_device::is_host_ptr(ptr)){
            auto host_ptr = dg::network_virtual_device::devirtualize_host_ptr(ptr); //weird
            return HostResource{host_ptr};
        }

        if (dg::network_virtual_device::is_fsys_ptr(ptr)) { 
            auto fsys_ptr = dg::network_virtual_device::devirtualize_fsys_ptr(ptr);
            return FsysResource{dg::network_kernelmap_x::map_nothrow(fsys_ptr)};
        }

        if (dg::network_virtual_device::is_cuda_ptr(ptr)){
            auto cuda_ptr = dg::network_virtual_device::devirtualize_cuda_ptr(ptr);
            return CudaResource{cuda_ptr}; 
        }

        if (dg::network_vitual_device::is_cufs_ptr(ptr)){
            auto cufs_ptr = dg::network_virtual_device::devirtualize_cufs_ptr(ptr);
            return CuFSResource{dg::network_cudafsmap_x::map_nothrow(cufs_ptr)};
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return {};
    }

    auto map(vma_ptr_t ptr) noexcept -> std::expected<vmamap_resource_handle_t, exception_t>{

        exception_t err = internal_safecthrow_vmaptr_access(ptr);

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return map_nothrow(ptr);
    } 

    auto mapsafe_nothrow(vma_ptr_t ptr) noexcept -> dg::network_genult::nothrow_immutable_unique_raii_wrapper<vmamap_resource_handle_t, unmap_t>{

        return {map_nothrow(ptr), unmap};
    }

    auto mapsafe(vma_ptr_t ptr) noexcept -> std::expected<dg::network_genult::nothrow_immutable_unique_raii_wrapper<vmamap_resource_handle_t, unmap_t>, exception_t>{

        exception_t err = safecthrow_vmaptr_access(ptr);

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return mapsafe_nothrow(ptr);
    }

    auto has_host_ptr(vmamap_resource_handle_t resource) noexcept -> bool{

        if (std::holds_alternative<FsysResource>(resource)){
            return true;
        }

        if (std::holds_alternative<HostResource>(resource)){
            return true;
        }

        return false;
    }

    auto has_cuda_ptr(vmamap_resource_handle_t resource) noexcept -> bool{

        if (std::holds_alternative<CuFSResource>(resource)){
            return true;
        }

        if (std::holds_alternative<CudaResource>(resource)){
            return true;
        }
        
        return false;
    }

    auto get_host_ptr(vmamap_resource_handle_t resource) noexcept -> void *{

        if (std::holds_alternative<FsysResource>(resource)){
            return dg::network_kernelmap_x::get_host_ptr(std::get<FsysResource>(resource).value);
        }

        if (std::holds_alternative<HostResource>(resource)){
            return std::get<HostResource>(resource).value;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return {};
    }

    auto get_cuda_ptr(vmamap_resource_handle_t resource) noexcept -> cuda_ptr_t{

        if (std::holds_alternative<CuFSResource>(resource)){
            return dg::network_cudafsmap_x::get_cuda_ptr(std::get<CuFSResource>(resource).value);
        }

        if (std::holds_alternative<CudaResource>(resource)){
            return std::get<CudaResource>(resource).value;
        }

        if constexpr(DEBUG_MODE_FLAG){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
            std::abort();
        }

        return {};
    }
} 

#endif