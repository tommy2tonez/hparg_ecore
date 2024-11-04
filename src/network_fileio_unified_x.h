#ifndef __DG_NETWORK_UNIFIED_FILEIO_H__
#define __DG_NETWORK_UNIFIED_FILEIO_H__

//define HEADER_CONTROL 10

#include "network_fileio_chksum_x.h"
#include "network_hash.h"
#include <filesystem>
#include "network_exception.h"
#include "stdx.h"
#include <filesystem>
#include "network_compact_serializer.h"

namespace dg::network_fileio_unified_x{
    
    struct Metadata{
        std::vector<std::string> datapath_vec;
        std::vector<bool> path_status_vec;
        uint64_t file_sz;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(datapath_vec, path_status_vec, file_sz);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(datapath_vec, path_status_vec, file_sz);
        }
    };

    static inline std::string METADATA_SUFFIX               = "DGFSYS_UNIFIED_X_METADATA";
    static inline std::string METADATA_EXT                  = "bin";
    static inline constexpr size_t MIN_DATAPATH_SIZE        = 2;
    static inline constexpr size_t MAX_METADATA_SIZE        = size_t{1} << 10;

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

    auto dg_internal_create_metadata(const char * fp, const Metadata& metadata) noexcept -> exception_t{
        
        std::filesystem::path metadata_fp = dg_internal_get_metadata_fp(fp);
        size_t metadata_sz = dg::network_compact_serializer::size(metadata);

        if (metadata_sz > MAX_METADATA_SIZE){
            return dg::network_exception::INVALID_ARGUMENT;
        }
        
        std::string bstream(metadata_sz, ' ');
        dg::network_compact_serializer::serialize_into(bstream.data(), metadata);
        exception_t create_err  = dg::network_fileio_chksum_x::dg_create_cbinary(metadata_fp.c_str(), metadata_sz);

        if (dg::network_exception::is_failed(create_err)){
            return create_err;
        }

        exception_t write_err   = dg::network_fileio_chksum_x::dg_write_binary(metadata_fp.c_str(), bstream.data(), bstream.size());

        if (dg::network_exception::is_failed(write_err)){
            dg::network_exception_handler::nothrow_log(dg::network_fileio_chksum_x::dg_remove(metadata_fp.c_str()));
            return write_err;
        }

        return dg::network_exception::SUCCESS;
    }

    auto dg_internal_remove_metadata(const char * fp) noexcept -> exception_t{

        std::filesystem::path metadata_fp = dg_internal_get_metadata_fp(fp);
        return dg::network_fileio_chksum_x::dg_remove(metadata_fp.c_str());
    }

    auto dg_internal_read_metadata(const char * fp) noexcept -> std::expected<Metadata, exception_t>{
        
        std::filesystem::path metadata_fp = dg_internal_get_metadata_fp(fp);
        std::string bstream(MAX_METADATA_SIZE, ' ');
        exception_t err = dg::network_fileio_chksum_x::dg_read_binary(metadata_fp.c_str(), bstream.data(), bstream.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        Metadata rs{};
        dg::network_compact_serializer::deserialize_into(rs, bstream.data());

        return rs;
    }

    auto dg_internal_write_metadata(const char * fp, const Metadata& metadata) noexcept -> exception_t{

        std::filesystem::path metadata_fp = dg_internal_get_metadata_fp(fp);
        size_t metadata_sz = dg::network_compact_serializer::size(metadata);

        if (metadata_sz > MAX_METADATA_SIZE){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        std::string bstream(metadata_sz, ' ');
        dg::network_compact_serializer::serialize_into(bstream.data(), metadata);

        return dg::network_fileio_chksum_x::dg_write_binary(metadata_fp.c_str(), bstream.data(), bstream.size());
    }

    auto dg_create_cbinary(const char * fp, const std::vector<std::string>& datapath_vec, size_t file_sz) noexcept -> exception_t{

        if (datapath_vec.size() < MIN_DATAPATH_SIZE){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        auto resource_guard_vec = std::vector<std::unique_ptr<stdx::VirtualResourceGuard>>{};
        auto metadata           = Metadata{datapath_vec, std::vector<bool>(datapath_vec.size(), true), file_sz};
        exception_t err         = dg_internal_create_metadata(fp, metadata);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        auto resource_grd       = stdx::vresource_guard([fp]() noexcept{
            dg::network_exception_handler::nothrow_log(dg_internal_remove_metadata(fp));
        });

        resource_guard_vec.push_back(std::move(resource_grd));
        
        for (size_t i = 0u; i < datapath_vec.size(); ++i){
            err = dg::network_fileio_chksum_x::dg_create_cbinary(datapath_vec[i].c_str(), file_sz);

            if (dg::network_exception::is_failed(err)){
                return err;
            }

            resource_grd = stdx::vresource_guard([ffp = datapath_vec[i]]() noexcept{
                dg::network_exception_handler::nothrow_log(dg::network_fileio_chksum_x::dg_remove(ffp.c_str()));
            });

            resource_guard_vec.push_back(std::move(resource_grd));
        }

        for (auto& e: resource_guard_vec){
            e->release();
        }

        return dg::network_exception::SUCCESS;
    }

    auto dg_remove(const char * fp) noexcept -> exception_t{

        Metadata metadata = dg::network_exception_handler::nothrow_log(dg_internal_read_metadata(fp));

        for (const auto& path: metadata.datapath_vec){
            dg::network_exception_handler::nothrow_log(dg::network_fileio_chksum_x::dg_remove(path.c_str()));
        }

        dg::network_exception_handler::nothrow_log(dg_internal_remove_metadata(fp));
        return dg::network_exception::SUCCESS;
    }

    auto dg_file_exists(const char * fp) noexcept -> std::expected<bool, exception_t>{

        std::filesystem::path metadata_fp       = dg_internal_get_metadata_fp(fp); 
        std::expected<bool, exception_t> status = dg::network_fileio_chksum_x::dg_file_exists(metadata_fp.c_str());

        if (!status.has_value()){
            return std::unexpected(status.error());
        }

        return status.value(); 
    }

    auto dg_file_size(const char * fp) noexcept -> std::expected<size_t, exception_t>{
        
        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return std::unexpected(status.error());
        }

        if (!status.value()){
            return std::unexpected(dg::network_exception::RUNTIME_FILEIO_ERROR);
        }

        std::expected<Metadata, exception_t> metadata = dg_internal_read_metadata(fp); 

        if (!metadata.has_value()){
            return std::unexpected(metadata.error());
        }

        return metadata->file_sz;
    } 

    auto dg_read_binary_direct(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }
        
        std::expected<Metadata, exception_t> metadata = dg_internal_read_metadata(fp);

        if (!metadata.has_value()){
            return metadata.error();
        } 
        
        for (size_t i = 0u; i < metadata->datapath_vec.size(); ++i){
            if (metadata->path_status_vec[i]){
                const char * fp = metadata->datapath_vec[i].c_str();
                exception_t err = dg::network_fileio_chksum_x::dg_read_binary_direct(fp, dst, dst_cap);

                if (dg::network_exception::is_success(err)){
                    return err;
                }
            }
        }

        return dg::network_exception::RUNTIME_FILEIO_ERROR; //promote error -> runtime_fileio_error
    }

    auto dg_read_binary_indirect(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        std::expected<Metadata, exception_t> metadata = dg_internal_read_metadata(fp);

        if (!metadata.has_value()){
            return metadata.error();
        }

        for (size_t i = 0u; i < metadata->datapath_vec.size(); ++i){
            if (metadata->path_status_vec[i]){
                const char * fp = metadata->datapath_vec[i].c_str();
                exception_t err = dg::network_fileio_chksum_x::dg_read_binary_indirect(fp, dst, dst_cap);

                if (dg::network_exception::is_success(err)){
                    return err;
                }
            }
        }
        
        return dg::network_exception::RUNTIME_FILEIO_ERROR; //premote error -> runtime_fileio_error
    }

    auto dg_read_binary(const char * fp, void * dst, size_t dst_cap) -> exception_t{

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
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }
        
        std::expected<Metadata, exception_t> metadata = dg_internal_read_metadata(fp);

        if (!metadata.has_value()){
            return metadata.error();
        }

        Metadata new_metadata           = metadata.value();
        new_metadata.file_sz            = src_sz;
        new_metadata.path_status_vec    = std::vector<bool>(metadata->datapath_vec.size(), false);
        bool updated_flag               = false; 

        for (size_t i = 0u; i < metadata->datapath_vec.size(); ++i){
            metadata->path_status_vec[i] = false;

            if (!updated_flag){
                if (i + 1 == metadata->datapath_vec.size()){
                    break;
                }
                exception_t metadata_write_err = dg_internal_write_metadata(fp, metadata.value());

                if (dg::network_exception::is_failed(metadata_write_err)){
                    return metadata_write_err;
                }
            }

            const char * ffp    = metadata->datapath_vec[i].c_str();
            exception_t err     = dg::network_fileio_chksum_x::dg_write_binary_direct(ffp, src, src_sz);

            if (dg::network_exception::is_success(err)){
                new_metadata.path_status_vec[i] = true;
                std::vector<bool>::swap(new_metadata.path_status_vec[i], new_metadata.path_status_vec.back());
                std::swap(new_metadata.datapath_vec[i], new_metadata.datapath_vec.back());
                exception_t metadata_write_err  = dg_internal_write_metadata(fp, new_metadata);

                if (dg::network_exception::is_failed(metadata_write_err)){
                    return metadata_write_err;
                }

                updated_flag = true;
            }
        }

        if (updated_flag){
            return dg::network_exception::SUCCESS;
        }

        return dg::network_exception::RUNTIME_FILEIO_ERROR;
    }

    auto dg_write_binary_indirect(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{
        
        std::expected<bool, exception_t> status = dg_file_exists(fp);

        if (!status.has_value()){
            return status.error();
        }

        if (!status.value()){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }
        
        std::expected<Metadata, exception_t> metadata = dg_internal_read_metadata(fp);

        if (!metadata.has_value()){
            return metadata.error();
        }

        Metadata new_metadata           = metadata.value();
        new_metadata.file_sz            = src_sz;
        new_metadata.path_status_vec    = std::vector<bool>(metadata->datapath_vec.size(), false);
        bool updated_flag               = false; 

        for (size_t i = 0u; i < metadata->datapath_vec.size(); ++i){
            metadata->path_status_vec[i] = false;

            if (!updated_flag){
                if (i + 1 == metadata->datapath_vec.size()){
                    break;
                }
                exception_t metadata_write_err = dg_internal_write_metadata(fp, metadata.value());

                if (dg::network_exception::is_failed(metadata_write_err)){
                    return metadata_write_err;
                }
            }

            const char * ffp    = metadata->datapath_vec[i].c_str();
            exception_t err     = dg::network_fileio_chksum_x::dg_write_binary_indirect(ffp, src, src_sz);

            if (dg::network_exception::is_success(err)){
                new_metadata.path_status_vec[i] = true;
                std::vector<bool>::swap(new_metadata.path_status_vec[i], new_metadata.path_status_vec.back());
                std::swap(new_metadata.datapath_vec[i], new_metadata.datapath_vec.back());
                exception_t metadata_write_err  = dg_internal_write_metadata(fp, new_metadata);

                if (dg::network_exception::is_failed(metadata_write_err)){
                    return metadata_write_err;
                }

                updated_flag = true;
            }
        }

        if (updated_flag){
            return dg::network_exception::SUCCESS;
        }

        return dg::network_exception::RUNTIME_FILEIO_ERROR;
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