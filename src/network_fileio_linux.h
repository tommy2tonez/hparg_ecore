#ifndef __NETWORK_FILEIO_LINUX_H__
#define __NETWORK_FILEIO_LINUX_H__

#include <stdint.h>
#include <stddef.h>
#include <memory>
#include "network_exception.h"
#include "network_log.h"
#include <unistd.h>
#include <climits>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <filesystem>

namespace dg::network_fileio_linux{

    static constexpr inline auto DG_FILEIO_MODE                 = S_IRWXU; //user-configurable - compile-payload
    static constexpr inline size_t DG_LEAST_DIRECTIO_BLK_SZ     = size_t{1} << 20; //user-configurable - compile-payload
    static constexpr inline bool NO_KERNEL_FSYS_CACHE_FLAG      = false;

    constexpr auto is_met_direct_dgio_blksz_requirement(size_t blk_sz) noexcept -> bool{

        return blk_sz % DG_LEAST_DIRECTIO_BLK_SZ == 0u;
    }

    constexpr auto is_met_direct_dgio_ptralignment_requirement(uintptr_t ptr) noexcept -> bool{

        return ptr % DG_LEAST_DIRECTIO_BLK_SZ == 0u;
    }  

    using kernel_fclose_t = void (*)(int) noexcept;

    auto dg_open_file(const char * fp, int flag) noexcept -> std::expected<dg::network_genult::nothrow_immutable_unique_raii_wrapper<int, kernel_fclose_t>, exception_t>{

        int fd = open(fp, flag, DG_FILEIO_MODE);

        if (fd == -1){
            return std::unexpected(dg::network_exception::wrap_kernel_exception(errno));
        }

        auto destructor = [](int fd_arg) noexcept{
            if (close(fd_arg) == -1){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_kernel_exception(errno)));
                std::abort();
            }
        };

        return {std::in_place_t{}, fd, destructor}; 
    }
    
    auto dg_file_size(int fd) noexcept -> std::expected<size_t, exception_t>{

        auto rs = lseek64(fd, 0L, SEEK_END);

        if (rs == -1){
            return std::unexpected(dg:network_exception::wrap_kernel_exception(errno)):
        }

        return dg::network_genult::safe_integer_cast<size_t>(rs);
    }

    auto dg_file_size_nothrow(int fd) noexcept -> size_t{

        return dg::network_exception_handler::nothrow_log(dg_file_size(fd));
    } 

    auto dg_fadvise_nocache(int fd) noexcept -> exception_t{

        std::expected<size_t, exception_t> efsz = dg_file_size(fd);

        if (!efsz.has_value()){
            return efsz.error();
        }

        auto rs = posix_fadvise64(fd, 0u, efsz.value(), POSIX_FADV_NOREUSE);

        if (rs != 0){
            return dg::network_exception::wrap_kernel_exception(errno);
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_fadvise_nocache_nothrow(int fd) noexcept{

        dg::network_exception_handler::nothrow_log(dg_fadvise_nocache(fd));
    }

    //user-space-
    //next iteration - include offset -> fp - use directionary to decode <fname, offset> to avoid file pollution 
    //yet its better to do separate file - to guarantee kernel concurrent support on all system

    auto dg_file_exists(const char * fp) noexcept -> std::expected<bool, exception_t>{ //noexcept is important - abort program if bad_alloc

        std::error_code err{};
        bool rs = std::filesystem::exists(fp, err);

        if (static_cast<bool>(err)){
            return std::unexpected(dg::network_exception::wrap_std_errcode(err));
        }

        return rs;
    }

    auto dg_file_exists_nothrow(const char * fp) noexcept -> bool{

        return dg::network_exception_handler::nothrow_log(dg_file_exists(fp));
    } 

    auto dg_file_size(const char * fp) noexcept -> std::expected<size_t, exception_t>{

        auto raii_fd = dg_open_file(fp, O_RDONLY);

        if (!raii_fd.has_value()){
            return std::unexpected(raii_fd.error());
        }

        return dg_file_size_nothrow(raii_fd.value());
    } 

    auto dg_file_size_nothrow(const char * fp) noexcept -> size_t{

        return dg::network_exception_handler::nothrow_log(dg_file_size(fp));
    }

    auto dg_create_binary(const char * fp, size_t fsz) noexcept -> exception_t{

        auto raii_fd = dg_open_file(fp, O_WRONLY | O_CREAT | O_TRUNC);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }

        if (ftruncate64(raii_fd.value(), fsz) == -1){
            return dg::network_exception::wrap_kernel_exception(errno);
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_create_binary_nothrow(const char * fp, size_t fsz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_create_binary(fp, fsz));
    }

    auto dg_create_cbinary(const char * fp, size_t fsz) noexcept -> exception_t{

    }

    void dg_create_cbinary_nothrow(const char * fp, size_t fsz) noexcept{

    }
    
    auto dg_read_binary_direct(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        auto raii_fd = dg_open_file(fp, O_RDONLY | O_DIRECT | O_TRUNC);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }

        int fd      = raii_fd.value();
        size_t fsz  = dg_file_size_nothrow(fd); //this is an error that user should not know of - internal corruption if failed (think of a successful file open guarantees a successful read of metadata)

        if constexpr(NO_KERNEL_FSYS_CACHE_FLAG){
            dg_fadvise_nocache(fd); //this is an error that user should not know of - internal corruption if failed
        }

        if (!is_met_direct_dgio_blksz_requirement(fsz)){
            return dg::network_exception::BAD_ALIGNMENT;
        }

        if (!is_met_direct_dgio_ptralignment_requirement(reinterpret_cast<uintptr_t>(dst))){
            return dg::network_exception::BAD_ALIGNMENT;
        }

        if (dst_cap < fsz){
            return dg::network_exception::BUFFER_OVERFLOW;
        }

        if (read(fd, dst, fsz) != fsz){
            return dg::network_exception::RUNTIME_FILEIO_ERROR; //this is where the exception + abort line is blurred - yet i think this should be abort (open_file guarantees successful immutable operations - if not then its the open_file problem) - global-mtx-lockguard is required internally - external modification is UB 
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_read_binary_direct_nothrow(const char * fp, void * dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_read_binary_direct(fp, dst, dst_cap));
    }

    auto dg_read_binary_indirect(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        auto raii_fd = dg_open_file(fp, O_RDONLY | O_TRUNC);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }        

        int fd      = raii_fd.value();
        size_t fsz  = dg_file_size_nothrow(fd);

        if (dst_cap < fsz){
            return dg::network_exception::BUFFER_OVERFLOW;
        }

        if constexpr(NO_KERNEL_FSYS_CACHE_FLAG){
            dg_fadvise_nocache(fd);
        }

        if (read(fd, dst, fsz) != fsz){
            return dg::network_exception::RUNTIME_FILEIO_ERROR; //this is a very blurred line between exception + abort | yet I choose to not trust sys for now - propagate error to make some root function noexceptable
        }

        return dg::network_exception::SUCCESS;
    } 

    void dg_read_binary_indirect_nothrow(const char * fp, void * dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_read_binary_indirect(fp, dst, dst_cap));
    }

    auto dg_read_binary(const char * fp, void * dst, size_t dst_cap) -> exception_t{

        exception_t err = dg_read_binary_direct(fp, dst, dst_cap);

        if (dg::network_exception::is_success(err)){
            return dg::network_exception::SUCCESS;
        }

        return dg_read_binary_indirect(fp, dst, dst_cap);
    }

    void dg_read_binary_nothrow(const char * fp, void * dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_read_binary(fp, dst, dst_cap));
    }

    auto dg_write_binary_direct(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

        auto raii_fd = dg_open_file(fp, O_WRONLY | O_DIRECT | O_TRUNC);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }
        
        if constexpr(NO_KERNEL_FSYS_CACHE_FLAG){
            dg_fadvise_nocache(raii_fd.value()); //this is an error that user should not know of - internal corruption if failed
        }

        if (!is_met_direct_dgio_blksz_requirement(src_sz)){
            return dg::network_exception::BAD_ALIGNMENT;
        }

        if (!is_met_direct_dgio_ptralignment_requirement(reinterpret_cast<uintptr_t>(src))){
            return dg::network_exception::BAD_ALIGNMENT;
        }

        if (write(raii_fd.value(), src, src_sz) != src_sz){
            return dg::network_exception::RUNTIME_FILEIO_ERROR; //need to be more descriptive
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_write_binary_direct_nothrow(const char * fp, const void * src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary_direct(fp, src, src_sz));
    }

    auto dg_write_binary_indirect(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

        auto raii_fd = dg_open_file(fp, O_WRONLY | O_TRUNC);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }

        if constexpr(NO_KERNEL_FSYS_CACHE_FLAG){
            dg_fadvise_nocache(raii_fd.value());
        }

        if (write(raii_fd.value(), src, src_sz) != src_sz){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_write_binary_indirect_nothrow(const char * fp, const void * src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary_indirect(fp, src, src_sz));
    }

    auto dg_write_binary(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

        exception_t err = dg_write_binary_direct(fp, src, src_sz);

        if (dg::network_exception::is_success(err)){
            return dg::network_exception::SUCCESS;
        }

        return dg_write_binary_indirect(fp, src, src_sz);
    }

    void dg_write_binary_nothrow(const char * fp, const void * src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary(fp, src, src_sz));
    }
}

#endif