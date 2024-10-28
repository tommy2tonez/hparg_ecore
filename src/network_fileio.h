#ifndef __NETWORK_FILEIO_H__
#define __NETWORK_FILEIO_H__

#include "network_trivial_serializer.h" 
#include "network_hash.h"
#include "stdx.h"

#ifdef __linux__
#include "network_fileio_linux.h"

namespace dg::network_fileio{
    using namespace dg::network_fileio_linux;
}

#elif _WIN32
#include "network_fileio_window.h"

namespace dg::network_fileio{
    using namespace dg::network_fileio_window;
} 

#else
static_assert(false);
#endif

namespace dg::network_fileio_chksum_x{

    struct FileHeader{
        uint64_t chksum;
        uint64_t content_size;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{

            reflector(chksum, content_size);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{

            reflector(chksum, content_size);
        }
    };

    static inline dg::string METADATA_SUFFIX = "DG_NETWORK_CHKSUM_X_METADATA"; 

    auto dg_internal_get_metadata_fp(const char * fp) noexcept -> std::filesystem::path{

        //unlikely yet memory exhaustion should be catched

        try{
            auto ext            = std::filesystem::path(fp).extension();
            auto rawname        = std::filesystem::path(fp).replace_extension("").filename();
            auto new_rawname    = std::format("{}_{}", rawname.native(), METADATA_SUFFIX); 
            auto new_fp         = std::filesystem::path(fp).replace_filename(new_rawname).replace_extension(ext);

            return new_fp;
        } catch (...){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_std_exception(std::current_exception())));
            std::abort();
            return {};
        }
    }

    auto dg_internal_read_metadata(const char * fp) noexcept -> std::expected<FileHeader, exception_t>{

        constexpr size_t HEADER_SZ          = dg::network_trivial_serializer::size(FileHeader{}); 
        std::filesystem::path header_path   = dg_internal_get_metadata_fp(fp);
        auto serialized_header              = std::array<char, HEADER_SZ>{};
        exception_t err                     = dg::network_fileio::dg_read_binary_indirect(header_path.c_str(), serialized_header.data(), HEADER_SZ);

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        FileHeader header{};
        dg::network_trivial_serializer::deserialize_into(header, serialized_header.data());
        
        return header;
    }

    auto dg_internal_write_metadata(const char * fp, FileHeader metadata) -> exception_t{

        constexpr size_t HEADER_SZ          = dg::network_trivial_serializer::size(FileHeader{});
        std::filesystem::path header_path   = dg_internal_get_metadata_fp(fp); 
        auto serialized_header              = std::array<char, HEADER_SZ>{};
        dg::network_trivial_serializer::serialize_into(serialized_header.data(), metadata);

        return dg::network_fileio::dg_write_binary_indirect(header_path.c_str(), serialized_header.data(), HEADER_SZ);
    }

    auto dg_internal_is_file_creatable(const char * fp) noexcept -> std::expected<bool, exception_t>{

        std::filesystem::path metadata_path     = dg_internal_get_metadata_fp(fp);
        std::expected<bool, exception_t> status = dg::network_fileio::dg_file_exists(metadata_path.c_str());

        if (!status.has_value()){
            return std::unexpected(status.error());
        }

        if (status.value()){
            return false;
        }

        status = dg::network_fileio::dg_file_exists(fp);

        if (!status.has_value()){
            return std::unexpected(status.error());
        }

        if (status.value()){
            return false;
        }
        
        return true;
    }

    //--user space--

    auto dg_file_exists(const char * fp) noexcept -> std::expected<bool, exception_t>{

        std::filesystem::path metadata_path     = dg_internal_get_metadata_fp(fp);
        std::expected<bool, exception_t> status = dg::network_fileio::dg_file_exists(metadata_path.c_str());

        if (!status.has_value()){
            return std::unexpected(status.error());
        }
        
        return status.value();
    }

    auto dg_file_exists_nothrow(const char * fp) noexcept -> bool{

        return dg::network_exception_handler::nothrow_log(dg_file_exists(fp));
    }

    auto dg_file_size(const char * fp) noexcept -> std::expected<size_t, exception_t>{

        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return std::unexpected(status.error());
        }

        if (!status.value()){
            return std::unexpected(dg::network_exception::FILE_NOT_FOUND);
        }

        return dg::network_fileio::dg_file_size(fp);
    }

    auto dg_file_size_nothrow(const char * fp) noexcept -> size_t{

        return dg::network_exception_handler::nothrow_log(dg_file_size(fp));
    }

    //if exists - return error
    //if not    - return SUCCESS if successfully created the file and its dependencies
    //          - return error otherwise, atomicity is guaranteed by using abort
    //cbinary = binary + memset(void *, 0, size_t) 

    auto dg_create_cbinary(const char * fp, size_t fsz) noexcept -> exception_t{

        std::expected<bool, exception_t> status = dg_internal_is_file_creatable(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        std::filesystem::path header_path = dg_internal_get_metadata_fp(fp); 
        auto backout_lambda = [=]() noexcept{
            if (dg::network_fileio::dg_file_exists_nothrow(fp)){
                dg::network_fileio::dg_remove_nothrow(fp);
            }

            if (dg::network_fileio::dg_file_exists_nothrow(header_path.c_str())){
                dg::network_fileio::dg_remove_nothrow(header_path.c_str());
            }
        };

        auto backout_guard  = stdx::resource_guard(std::move(backout_lambda));
        exception_t err     = dg::network_fileio::dg_create_cbinary(fp, fsz);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        err = dg::network_fileio::dg_create_binary(header_path.c_str(), 0u);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        //TODOs:
        // auto header = FileHeader{dg::network_hash::cmurmur_hash(fsz), fsz};  //
        // err         = dg_internal_write_metadata(fp, header);

        // if (dg::network_exception::is_failed(err)){
        //     return err;
        // }

        backout_guard.release();
        return dg::network_exception::SUCCESS;
    }

    auto dg_create_cbinary_nothrow(const char * fp, size_t fsz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_create_cbinary(fp, fsz));
    }

    //this should always be noexcept - better to abort the program if unable to delete
    //do not invoke this function to inverse create_binary - invoke nothrow version
    auto dg_remove(const char * fp) noexcept -> exception_t{

        std::filesystem::path header_path = dg_internal_get_metadata_fp(fp);
        exception_t err = dg::network_fileio::dg_remove(fp); 

        if (dg::network_exception::is_failed(err)){
            return err;
        }
        
        return dg::network_fileio::dg_remove(header_path.c_str());
    }

    void dg_remove_nothrow(const char * fp) noexcept{

        dg::network_exception_handler::nothrow_log(dg_remove(fp));
    }

    auto dg_read_binary_direct(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::FILE_NOT_FOUND;
        }

        exception_t err = dg::network_fileio::dg_read_binary_direct(fp, dst, dst_cap);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        std::expected<FileHeader, exception_t> header = dg_internal_read_metadata(fp); 

        if (!header.has_value()){
            return header.error();
        }
        
        if (header.value().content_size > dst_cap){
            return dg::network_exception::CORRUPTED_FILE; //control flows reach here indicate that dst_cap >= file_sz, content_size <= file_sz (contract), dst_cap >= content_size - dst_cap < content_size == corrupted 
        }

        uint64_t file_chksum = dg::network_hash::murmur_hash(reinterpret_cast<const char *>(dst), header.value().content_size); //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 

        if (file_chksum != header.value().chksum){
            return dg::network_exception::CORRUPTED_FILE;
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_read_binary_direct_nothrow(const char * fp, void * dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_read_binary_direct(fp, dst, dst_cap));
    }

    auto dg_read_binary_indirect(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{
        
        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::FILE_NOT_FOUND;
        }

        exception_t err = dg::network_fileio::dg_read_binary_indirect(fp, dst, dst_cap);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        std::expected<FileHeader, exception_t> header = dg_internal_read_metadata(fp);

        if (!header.has_value()){
            return header.error();
        }

        if (header.value().content_size > dst_cap){
            return dg::network_exception::CORRUPTED_FILE; //control flows reach here indicate that dst_cap >= file_sz, content_size <= file_sz (contract), dst_cap >= content_size - dst_cap < content_size == corrupted 
        }

        uint64_t file_chksum = dg::network_hash::murmur_hash(reinterpret_cast<const char *>(dst), header.value().content_size); //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 

        if (file_chksum != header.value().chksum){
            return dg::network_exception::CORRUPTED_FILE;
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

        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::FILE_NOT_FOUND;
        }

        exception_t err = dg::network_fileio::dg_write_binary_direct(fp, src, src_sz);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        FileHeader header{dg::network_hash::murmur_hash(reinterpret_cast<const char *>(src), src_sz), src_sz};  //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 
        err = dg_internal_write_metadata(fp, header);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        return dg::network_exception::SUCCESS;
    }

    void dg_write_binary_direct_nothrow(const char * fp, const void * src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary_direct(fp, src, src_sz));
    }

    auto dg_write_binary_indirect(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::FILE_NOT_FOUND;
        }

        exception_t err = dg::network_fileio::dg_write_binary_indirect(fp, src, src_sz);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        FileHeader header{dg::network_hash::murmur_hash(reinterpret_cast<const char *>(src), src_sz), src_sz};  //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 
        err = dg_internal_write_metadata(fp, header);

        if (dg::network_exception::is_failed(err)){
            return err;
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