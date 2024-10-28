#ifndef __NETWORK_FILEIO_WINDOW_H__
#define __NETWORK_FILEIO_WINDOW_H__

#include "fileapi.h"
#include "network_exception.h"
#include <filesystem>
#include <memory>
#include "network_utility.h"

namespace dg::network_fileio_window{

    static_assert(std::is_same_v<DWORD, int>);

    using dynamic_fclose_t  = void (*)(HANDLE *) noexcept; //HANDLE type is abstracted here - it's better to do dynamic memory allocation for raii
    
    static inline constexpr int DGIO_SHARED_MODE            = 0x00000000;
    static inline constexpr auto DGIO_SECURITY_ATTRIBUTE    = NULL;
    static inline constexpr int DGIO_FILE_ADVISE            = FILE_FLAG_NO_BUFFERING;

    auto dg_open_file(const char * fp, int flag) noexcept -> std::expected<std::unique_ptr<HANDLE, dynamic_fclose_t>, exception_t>{

        HANDLE handle = CreateFileA(fp, flag, DGIO_SHARED_MODE, DGIO_SECURITY_ATTRIBUTE, TRUNCATE_EXISTING, FILE_ATTRIBUTE_NORMAL, DGIO_FILE_ADVISE); 

        if (handle == INVALID_HANDLE_VALUE){
            return std::unexpected(dg::network_exception::wrap_kernel_error(GetLastError()));
        }

        auto destructor = [](HANDLE * arg) noexcept{
            if (!CloseHandle(*arg)){
                std::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_kernel_error(GetLastError())));
                std::abort();
            }

            delete arg;
        };

        return std::unique_ptr<HANDLE, dynamic_fclose_t>(new HANDLE(std::move(handle)), destructor);
    }

    auto dg_internal_file_size(HANDLE fhandle) noexcept -> std::expected<size_t, exception_t>{

        LARGE_INTERGER fsz = {}; 

        if (!GetFileSizeEx(fhandle, &fsz)){
            return std::unexpected(dg::network_exception::wrap_kernel_error(GetLastError()));
        }

        return dg::network_genult::safe_integer_cast<size_t>(fsz);
    }

    auto dg_internal_file_size_nothrow(HANDLE fhandle) noexcept -> size_t{

        return dg::network_exception_handler::nothrow_log(dg_internal_file_size(fhandle));
    }

    auto dg_file_exists(const char * fp) noexcept -> std::expected<bool, exception_t>{

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

        auto raii_fhandle = dg_open_file(fp, GENERIC_READ);

        if (!raii_fhandle.has_value()){
            return raii_fhandle.error();
        }

        return dg_internal_file_size_nothrow(*(raii_fhandle.value()));
    }

    auto dg_file_size_nothrow(const char * fp) noexcept -> size_t{

        dg::network_exception_handler::nothrow_log(dg_file_size(fp));
    }

    //read_direct is a stricter requirement of read - does not actually guarantee a direct read - just like inlinability
    auto dg_read_binary_direct(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        auto raii_fhandle = dg_open_file(fp, GENERIC_READ);

        if (!raii_fhandle.has_value()){
            return raii_fhandle.error();
        } 

        HANDLE fhandle  = *(raii_fhandle.value());
        size_t fsz      = dg_internal_file_size_nothrow(fhandle);
        int read_bytes  = {};

        if (dst_cap < fsz){
            return dg::network_exception::BUFFER_OVERFLOW;
        } 

        //this is an immutable operation - a successful file open must guarantee a successful read operation

        if (!ReadFile(fhandle, dst, dg::network_genult::safe_integer_cast<int>(fsz), &read_bytes, NULL)){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_kernel_error(GetLastError())));
            std::abort();
        }

        if (dg::network_genult::safe_integer_cast<size_t>(read_bytes) != fsz){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_kernel_error(GetLastError())));
            std::abort();
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_read_binary_direct_nothrow(const char * fp, void * dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_read_binary_direct(fp, dst, dst_cap));
    }

    auto dg_write_binary_direct(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

        auto raii_fhandle = dg_open_file(fp, GENERIC_WRITE);

        if (!raii_fhandle.has_value()){
            return raii_fhandle.error();
        }

        HANDLE fhandle      = *(raii_fhandle.value());
        int written_bytes   = {};

        if (!WriteFile(fhandle, src, dg::network_genult::safe_integer_cast<int>(src_sz), &written_bytes, NULL)){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        if (dg::network_genult::safe_integer_cast<size_t>(written_bytes) != src_sz){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_write_binary_direct_nothrow(const char * fp, const void * src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary_direct(fp, src, src_sz));
    }
} 

#endif