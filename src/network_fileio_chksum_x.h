#ifndef __NETWORK_FILEIO_H__
#define __NETWORK_FILEIO_H__

#include "network_trivial_serializer.h" 
#include "network_hash.h"
#include "stdx.h"
#include "network_fileio.h"
#include "network_exception_handler.h"

namespace dg::network_fileio_chksum_x{

    //improve error_code return - convert the errors -> RUNTIME_FILEIO_ERROR for generic purpose 

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

    static inline std::string METADATA_SUFFIX   = "DGFSYS_CHKSUM_X_METADATA"; 
    static inline std::string METADATA_EXT      = "bin";

    auto dg_internal_get_metadata_fp(const char * fp) noexcept -> std::filesystem::path{

        try{
            auto rawname        = std::filesystem::path(fp).replace_extension("").filename();
            auto new_rawname    = std::format("{}_{}", rawname.native(), METADATA_SUFFIX); 
            auto new_fp         = std::filesystem::path(fp).replace_filename(new_rawname).replace_extension(METADATA_EXT);

            return new_fp;
        } catch (...){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_std_exception(std::current_exception())));
            std::abort();
            return {};
        }
    }

    auto dg_internal_create_metadata(const char * fp) noexcept -> exception_t{

        constexpr size_t HEADER_SZ          = dg::network_trivial_serializer::size(FileHeader{}); 
        std::filesystem::path header_path   = dg_internal_get_metadata_fp(fp);
        exception_t err                     = dg::network_fileio::dg_create_binary(header_path.c_str(), HEADER_SZ);

        return err;
    }

    auto dg_internal_remove_metadata(const char * fp) noexcept -> exception_t{

        std::filesystem::path header_path   = dg_internal_get_metadata_fp(fp);
        return dg::network_fileio::dg_remove(header_path.c_str());
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

    auto dg_internal_write_metadata(const char * fp, FileHeader metadata) noexcept -> exception_t{

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

    auto dg_file_exists(const char * fp) noexcept -> std::expected<bool, exception_t>{

        std::filesystem::path metadata_path     = dg_internal_get_metadata_fp(fp);
        std::expected<bool, exception_t> status = dg::network_fileio::dg_file_exists(metadata_path.c_str());

        if (!status.has_value()){
            return std::unexpected(status.error());
        }

        if (!status.value()){
            std::cout << metadata_path << std::endl;
        }        
        
        return status.value();
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

    auto dg_create_cbinary(const char * fp, size_t fsz) noexcept -> exception_t{

        std::expected<bool, exception_t> status = dg_internal_is_file_creatable(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        exception_t file_create_err         = dg::network_fileio::dg_create_cbinary(fp, fsz);

        if (dg::network_exception::is_failed(file_create_err)){
            return file_create_err;
        }

        auto file_create_guard              = stdx::resource_guard([=]() noexcept{
            dg::network_exception_handler::nothrow_log(dg::network_fileio::dg_remove(fp));
        });

        exception_t header_create_err       = dg_internal_create_metadata(fp);

        if (dg::network_exception::is_failed(header_create_err)){
            return header_create_err;
        }

        auto header_create_guard            = stdx::resource_guard([=]() noexcept{
            dg::network_exception_handler::nothrow_log(dg_internal_remove_metadata(fp));
        });

        auto empty_buf                      = std::make_unique<char[]>(fsz);
        auto header                         = FileHeader{dg::network_hash::murmur_hash(empty_buf.get(), fsz), fsz};
        exception_t header_write_err        = dg_internal_write_metadata(fp, header);

        if (dg::network_exception::is_failed(header_write_err)){
            return header_write_err;
        }

        header_create_guard.release();
        file_create_guard.release();

        return dg::network_exception::SUCCESS;
    }

    auto dg_remove(const char * fp) noexcept -> exception_t{

        dg::network_exception_handler::nothrow_log(dg::network_fileio::dg_remove(fp));
        dg::network_exception_handler::nothrow_log(dg_internal_remove_metadata(fp));

        return dg::network_exception::SUCCESS;
    }

    auto dg_read_binary_direct(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::FILE_NOT_FOUND;
        }

        std::expected<FileHeader, exception_t> header = dg_internal_read_metadata(fp); 

        if (!header.has_value()){
            return header.error();
        }
        
        if (header->content_size > dst_cap){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        exception_t err = dg::network_fileio::dg_read_binary_direct(fp, dst, dst_cap);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        uint64_t file_chksum = dg::network_hash::murmur_hash(reinterpret_cast<const char *>(dst), header->content_size); //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 

        if (file_chksum != header->chksum){
            return dg::network_exception::CORRUPTED_FILE;
        }
        
        return dg::network_exception::SUCCESS;
    }

    auto dg_read_binary_indirect(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::FILE_NOT_FOUND;
        }

        std::expected<FileHeader, exception_t> header = dg_internal_read_metadata(fp);

        if (!header.has_value()){
            return header.error();
        }

        if (header->content_size > dst_cap){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        exception_t err = dg::network_fileio::dg_read_binary_indirect(fp, dst, dst_cap);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        uint64_t file_chksum = dg::network_hash::murmur_hash(reinterpret_cast<const char *>(dst), header->content_size); //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 

        if (file_chksum != header->chksum){
            return dg::network_exception::CORRUPTED_FILE;
        }

        return dg::network_exception::SUCCESS;
    }

    auto dg_read_binary(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        exception_t err = dg_read_binary_direct(fp, dst, dst_cap);

        if (dg::network_exception::is_success(err)){
            return dg::network_exception::SUCCESS;
        } 

        return dg_read_binary_indirect(fp, dst, dst_cap);
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

    auto dg_write_binary(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

        exception_t err = dg_write_binary_direct(fp, src, src_sz);

        if (dg::network_exception::is_success(err)){
            return dg::network_exception::SUCCESS;
        }

        return dg_write_binary_indirect(fp, src, src_sz);
    }
}

#endif