#ifndef __NETWORK_CUFSIO_LINUX_H__
#define __NETWORK_CUFSIO_LINUX_H__

#include <memory>
#include <cuda_runtime.h>
#include "cufile.h"
#include "network_fileio_linux.h"
#include <unordered_set>
#include "network_utility.h"

namespace dg::network_cufsio_linux{

    struct CudaFileDescriptor{
        dg::network_genult::nothrow_immutable_unique_raii_wrapper<int, dg::network_fileio_linux::kernel_fclose_t> kernel_raii_fd;
        CUfileHandle_t cf_handle;
    };

    using cuda_fclose_t = void (*)(CudaFileDescriptor) noexcept; 

    static inline constexpr size_t DG_CUDIRECT_BLK_SZ       = size_t{1} << 13;
    static inline constexpr auto DG_CU_FILE_HANDLE_OPTION   = CU_FILE_HANDLE_TYPE_OPAQUE_FD;  
    
    inline std::unordered_set<std::pair<cuda_ptr_t, size_t>> cufile_stableptr_hashset{}; 

    void init(cuda_ptr_t * ptr_arr, size_t * sz_arr, size_t n){

        //consider raii - 
        auto logger     = dg::network_log_scope::critical_terminate();
        exception_t err = dg::network_exception::wrap_cuda_exception(cuFileDriverOpen());
        dg::network_exception::throw_exception(err);

        for (size_t i = 0; i < n; ++i){
            cufile_stableptr_hashset.insert(std::make_pair(ptr_arr[i], sz_arr[i]));
            err = dg::network_exception::wrap_cuda_exception(cuFileBufRegister(ptr_arr[i], sz_arr[i], 0u));
            dg::network_exception::throw_exception(err);
        }

        logger.release();
    }

    constexpr auto is_met_cudadirect_dgio_blksz_requirement(size_t sz) noexcept -> bool{

        return sz % DG_CUDIRECT_BLK_SZ == 0u;
    }

    constexpr auto is_met_cudadirect_dgio_ptralignment_requirement(uintptr_t ptr) noexcept -> bool{

        return ptr % DG_CUDIRECT_BLK_SZ = 0u;
    }

    auto is_cufile_buf_registered(cuda_ptr_t dst, size_t sz) noexcept -> bool{

        return cufile_stableptr_hashset.find(std::make_pair(dst, sz)) != cufile_stableptr_hashset.end(); 
    }

    auto dg_cuopen_file(const char * path, int flag) noexcept -> std::expected<dg::network_genult::nothrow_unique_raii_wrapper<CudaFileDescriptor, cuda_fclose_t>, exception_t>{

        auto kfd = dg::network_fileio_linux::dg_open_file(path, flag);
        
        if (!kfd.has_value()){
            return std::unexpected(kfd.error());
        }

        auto destructor = [](CudaFileDescriptor cu_fd) noexcept{
            cuFileHandleDeregister(cu_fd->cf_handle);
        };
    
        CUfileDescr_t cf_descr{};
        CUfileHandle_t cf_handle{};

        cf_descr.handle.fd  = kfd.value();
        cf_descr.type       = DG_CU_FILE_HANDLE_OPTION;
        exception_t err     = dg::network_exception::wrap_cuda_exception(cuFileHandleRegister(&cf_handle, &cf_descr)); 
        
        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return {std::in_place_t{}, CudaFileDescriptor{std::move(kfd.value()), cf_handle}, destructor};
    }

    auto dg_read_binary_direct(const char * fp, cuda_ptr_t dst, size_t dst_cap) noexcept -> exception_t{

        auto raii_fd    = dg_cuopen_file(fp, O_RDONLY | O_DIRECT | O_TRUNC);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        } 

        int kernel_fd   = raii_fd.value()->kernel_raii_fd;
        size_t fsz      = dg::network_fileio_linux::dg_file_size_nothrow(kernel_fd);

        if (!is_met_cudadirect_dgio_blksz_requirement(fsz)){
            return dg::network_exception::BAD_ALIGNMENT;
        }

        if (!is_met_cudadirect_dgio_ptralignment_requirement(pointer_cast<uintptr_t>(dst))){
            return dg::network_exception::BAD_ALIGNMENT;
        }

        if (!is_cufile_buf_registered(dst, dst_cap)){
            return dg::network_exception::UNREGISTERED_CUFILE_PTR;
        }

        if (dst_cap < fsz){
            return dg::network_exception::BUFFER_OVERFLOW;
        }

        if (cuFileRead(raii_fd.value()->cf_handle, dst, fsz, 0u, 0u) != fsz){ //I rather think that recoverablity (offload to kernel_read + friends) - should be cuFileRead's responsibility - if not then an extension is required for such use-case
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::RUNTIME_FILEIO_ERROR)); //this is where the exception + abort line is blurred - yet i think this should be abort (cuopen_file guarantees successful immutable operations - if not then its the cuopen_file problem)
            std::abort();
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_read_binary_direct_nothrow(const char * fp, cuda_ptr_t dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_read_binary_direct(fp, dst, dst_cap));
    }

    auto dg_write_binary_direct(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept -> exception_t{

        auto raii_fd    = dg_cuopen_file(fp, O_WRONLY | O_DIRECT | O_TRUNC);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }
        
        if (!is_met_cudadirect_dgio_blksz_requirement(src_sz)){
            return dg::network_exception::BAD_ALIGNMENT;
        }

        if (!is_met_cudadirect_dgio_ptralignment_requirement(pointer_cast<uintpr_t>(src))){
            return dg::network_exception::BAD_ALIGNMENT;
        }

        if (!is_cufile_buf_registered(src, src_sz)){
            return dg::network_exception::UNREGISTERED_CUFILE_PTR;
        }

        if (cuFileWrite(raii_fd.value()->cf_handle, src, src_sz, 0u, 0u) != src_sz){ //I rather think that recoverability is cuFileWrite's responsibility
            return dg::network_exception::RUNTIME_FILEIO_ERROR; //this should be returning err - because fsys allocation is required for the operation (this cannot be overseen by dg_cuopen_file - unless its writing to the same region - which alters the semantic of the function name) 
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_write_binary_direct_nothrow(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary_direct(fp, src, src_sz));
    }
}

#endif