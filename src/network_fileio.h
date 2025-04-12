#ifndef __NETWORK_FILEIO_H_
#define __NETWORK_FILEIO_H__

//define HEADER_CONTROL 8

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
#include "network_raii_x.h"
#include "stdx.h" 

namespace dg::network_fileio{

    //alright, we are to implement a virtual file path
    //this introduces so many problems, let's see

    //alright, I dont know about Linux, but Windows does lock the operating folder for concurrent writing | reading
    //we dont really know unless we run some advisory calibration (there is not an official API to deal with this kind of stuff, and you probably dont want to mess with the internal API
    //                                                             the calibration would kind of extract the optimal writing + reading patterns (this is SSDs + kernel specifics) and give the users the data)

    //improve error_code return - convert the errors -> RUNTIME_FILEIO_ERROR for generic purpose 

    static constexpr inline auto DG_FILEIO_MODE                 = S_IRWXU; //user-configurable - compile-payload
    static constexpr inline size_t DG_LEAST_DIRECTIO_BLK_SZ     = size_t{1} << 12; //user-configurable - compile-payload
    static constexpr inline bool NO_KERNEL_FSYS_CACHE_FLAG      = true;

    constexpr auto is_met_direct_dgio_blksz_requirement(size_t blk_sz) noexcept -> bool{

        return blk_sz % DG_LEAST_DIRECTIO_BLK_SZ == 0u;
    }

    constexpr auto is_met_direct_dgio_ptralignment_requirement(uintptr_t ptr) noexcept -> bool{

        return ptr % DG_LEAST_DIRECTIO_BLK_SZ == 0u;
    }

    using kernel_fclose_t = void (*)(int) noexcept;

    auto dg_open_file(const char * fp, int flag) noexcept -> std::expected<dg::unique_resource<int, kernel_fclose_t>, exception_t>{

        int fd = open(fp, flag, DG_FILEIO_MODE);

        if (fd == -1){
            return std::unexpected(dg::network_exception::wrap_kernel_error(errno));
        }

        auto destructor = [](int fd_arg) noexcept{
            if (close(fd_arg) == -1){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_kernel_error(errno)));
                std::abort();
            }
        };

        return dg::unique_resource<int, kernel_fclose_t>{fd, std::move(destructor)}; 
    }
    
    auto dg_file_size(int fd) noexcept -> std::expected<size_t, exception_t>{

        auto rs = lseek64(fd, 0L, SEEK_END);

        if (rs == -1){
            return std::unexpected(dg::network_exception::wrap_kernel_error(errno));
        }

        return stdx::safe_integer_cast<size_t>(rs);
    }

    auto dg_fadvise_nocache(int fd) noexcept -> exception_t{

        std::expected<size_t, exception_t> efsz = dg_file_size(fd);

        if (!efsz.has_value()){
            return efsz.error();
        }

        auto rs = posix_fadvise64(fd, 0u, efsz.value(), POSIX_FADV_NOREUSE);

        if (rs != 0){
            return dg::network_exception::UNIDENTIFIED_EXCEPTION;
        }

        return dg::network_exception::SUCCESS;
    }

    auto dg_file_exists(const char * fp) noexcept -> std::expected<bool, exception_t>{ //noexcept is important - abort program if bad_alloc

        std::error_code err{};
        bool rs = std::filesystem::exists(fp, err);

        if (static_cast<bool>(err)){
            return std::unexpected(dg::network_exception::wrap_std_errcode(err));
        }

        return rs;
    }

    auto dg_file_size(const char * fp) noexcept -> std::expected<size_t, exception_t>{

        auto raii_fd = dg_open_file(fp, O_RDONLY);

        if (!raii_fd.has_value()){
            return std::unexpected(raii_fd.error());
        }

        std::expected<size_t, exception_t> rs = dg_file_size(raii_fd.value());

        if (!rs.has_value()){
            return std::unexpected(rs.error());
        }

        return rs.value();
    }

    auto dg_internal_aggresive_file_size(const char * fp, int flags) noexcept -> std::expected<size_t, exception_t>{

        auto raii_fd = dg_open_file(fp, flags);

        if (!raii_fd.has_value()){
            return std::unexpected(raii_fd.error());
        }

        std::expected<size_t, exception_t> rs = dg_file_size(raii_fd.value());

        if (!rs.has_value()){
            return std::unexpected(rs.error());
        }

        return rs.value();
    }

    auto dg_create_binary(const char * fp, size_t fsz) noexcept -> exception_t{

        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (status.value()){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        exception_t ret_code{};
        bool remove_responsibility{};

        {
            auto raii_fd = dg_open_file(fp, O_WRONLY | O_CREAT | O_TRUNC);

            if (!raii_fd.has_value()){
                remove_responsibility = false;
                ret_code = raii_fd.error();
            } else{
                if (ftruncate64(raii_fd.value(), fsz) == -1){
                    remove_responsibility = true;
                    ret_code = dg::network_exception::wrap_kernel_error(errno);
                } else{
                    remove_responsibility = false;
                    ret_code = dg::network_exception::SUCCESS;
                }
            }
        }

        if (remove_responsibility){
            if (remove(fp) == -1){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_kernel_error(errno)));
                std::abort();
            }
        }

        return ret_code;
    }

    auto dg_ftruncate(const char * fp, size_t fsz) noexcept -> exception_t{

        auto raii_fd = dg_open_file(fp, O_WRONLY);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }

        if (ftruncate64(raii_fd.value(), fsz) == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        return dg::network_exception::SUCCESS;
    }

    auto dg_remove(const char * fp) noexcept -> exception_t{

        if (remove(fp) == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        return dg::network_exception::SUCCESS;
    }

    auto dg_read_binary_direct(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        auto raii_fd = dg_open_file(fp, O_RDONLY | O_DIRECT);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }

        int fd = raii_fd.value();
        std::expected<size_t, exception_t> efsz = dg_file_size(fd);

        if (!efsz.has_value()){
            return efsz.error();
        }

        size_t fsz = efsz.value();

        if constexpr(NO_KERNEL_FSYS_CACHE_FLAG){
            exception_t err = dg_fadvise_nocache(fd);

            if (dg::network_exception::is_failed(err)){
                return err;
            }
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

        if (lseek64(fd, 0L, SEEK_SET) == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        }

        auto read_err = read(fd, dst, fsz); 

        if (read_err < 0){
            return dg::network_exception::wrap_kernel_error(errno);
        } 

        if (fsz != read_err){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        return dg::network_exception::SUCCESS;
    }

    auto dg_read_binary_indirect(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        auto raii_fd = dg_open_file(fp, O_RDONLY);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }        

        int fd = raii_fd.value();
        std::expected<size_t, exception_t> efsz = dg_file_size(fd);

        if (!efsz.has_value()){
            return efsz.error();
        }

        size_t fsz = efsz.value();

        if (dst_cap < fsz){
            return dg::network_exception::BUFFER_OVERFLOW;
        }

        if constexpr(NO_KERNEL_FSYS_CACHE_FLAG){
            exception_t err = dg_fadvise_nocache(fd);

            if (dg::network_exception::is_failed(err)){
                return err;
            }
        }

        if (lseek64(fd, 0L, SEEK_SET) == -1){
            return dg::network_exception::wrap_kernel_error(errno);
        } 

        auto read_err = read(fd, dst, fsz);

        if (read_err < 0){
            return dg::network_exception::wrap_kernel_error(errno);
        } 

        if (fsz != read_err){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        return dg::network_exception::SUCCESS;
    } 

    auto dg_read_binary(const char * fp, void * dst, size_t dst_cap) -> exception_t{

        exception_t err = dg_read_binary_direct(fp, dst, dst_cap);

        if (dg::network_exception::is_success(err)){
            return dg::network_exception::SUCCESS;
        }

        return dg_read_binary_indirect(fp, dst, dst_cap);
    }

    auto dg_write_binary_direct(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

        if (!is_met_direct_dgio_blksz_requirement(src_sz)){
            return dg::network_exception::BAD_ALIGNMENT;
        }

        if (!is_met_direct_dgio_ptralignment_requirement(reinterpret_cast<uintptr_t>(src))){
            return dg::network_exception::BAD_ALIGNMENT;
        }

        std::expected<size_t, exception_t> old_fsz = dg_internal_aggresive_file_size(fp, O_RDWR | O_DIRECT); 
        
        if (!old_fsz.has_value()){
            return old_fsz.error();
        }

        auto resource_grd   = stdx::resource_guard([=]() noexcept{
            dg_ftruncate(fp, old_fsz.value()); //alright this is only a courtesy to snap back to the original data size, which should succeed 99.9999% of the time
        });

        auto raii_fd        = dg_open_file(fp, O_WRONLY | O_DIRECT | O_TRUNC); // alright fellas, I did not know that there exists no way for us to write on a memregion except for using mmap

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }

        if constexpr(NO_KERNEL_FSYS_CACHE_FLAG){
            exception_t err = dg_fadvise_nocache(raii_fd.value());

            if (dg::network_exception::is_failed(err)){
                return err;
            }
        }

        auto write_err  = write(raii_fd.value(), src, src_sz);

        if (write_err < 0){
            return dg::network_exception::wrap_kernel_error(errno);
        } 

        if (write_err != src_sz){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        resource_grd.release();
        return dg::network_exception::SUCCESS;
    }

    auto dg_write_binary_indirect(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

        std::expected<size_t, exception_t> old_fsz = dg_internal_aggresive_file_size(fp, O_RDWR); 

        if (!old_fsz.has_value()){
            return old_fsz.error();
        }

        auto resource_grd   = stdx::resource_guard([=]() noexcept{
            dg_ftruncate(fp, old_fsz.value()); //alright this is only a courtesy to snap back to the original data size, which should succeed 99.9999% of the time
        });

        auto raii_fd        = dg_open_file(fp, O_WRONLY | O_TRUNC);

        if (!raii_fd.has_value()){
            return raii_fd.error();
        }

        if constexpr(NO_KERNEL_FSYS_CACHE_FLAG){
            exception_t err = dg_fadvise_nocache(raii_fd.value());

            if (dg::network_exception::is_failed(err)){
                return err;
            }
        }

        auto write_err = write(raii_fd.value(), src, src_sz);

        if (write_err < 0){
            return dg::network_exception::wrap_kernel_error(errno);
        } 

        if (write_err != src_sz){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        resource_grd.release();
        return dg::network_exception::SUCCESS;
    }

    auto dg_write_binary(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

        exception_t err = dg_write_binary_direct(fp, src, src_sz);

        if (dg::network_exception::is_success(err)){
            return dg::network_exception::SUCCESS;
        }

        return dg_write_binary_indirect(fp, src, src_sz);
    }

    auto dg_create_cbinary(const char * fp, size_t fsz) noexcept -> exception_t{

        exception_t create_err  = dg_create_binary(fp, fsz);

        if (dg::network_exception::is_failed(create_err)){
            return create_err;
        }

        char * buf = static_cast<char *>(std::calloc(fsz, sizeof(char)));

        if (buf == nullptr){
            exception_t rm_err  = dg_remove(fp);

            if (dg::network_exception::is_failed(rm_err)){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(rm_err));
                std::abort();
            }

            return dg::network_exception::RESOURCE_EXHAUSTION;
        }

        exception_t write_err   = dg_write_binary(fp, buf, fsz);
        std::free(buf);

        if (dg::network_exception::is_failed(write_err)){
            exception_t rm_err  = dg_remove(fp);

            if (dg::network_exception::is_failed(rm_err)){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(rm_err));
                std::abort();
            }

            return write_err;
        }

        return dg::network_exception::SUCCESS;
    }
}

#endif