#ifndef __DG_NETWORK_UNIFIED_FILEIO_H__
#define __DG_NETWORK_UNIFIED_FILEIO_H__

#include "network_fileio_chksum_x.h"
#include "network_hash.h"
#include <filesystem>
#include "network_exception.h"
#include "stdx.h"
#include <filesystem>
#include "network_compact_serializer.h"
#include <iostream>

namespace dg::network_fileio_unified_x{
    
    //this is very hard to write - because metadata could be compromised - 
    //let's for now - assume that the corruption rate is correlated to the written/read data - such that this unified_x reduces the hardware corruption rate (increase recovery_rate) to header_sz / written_sz | this is an important note
    //WLOG, assume 512 bytes metadata - and 10 MB for each binary file - then the not-recoverable ratio is 1/20000 * corruption_rate/byte - this is reasonably acceptable
    //this is only to solve that very specific problem
    //for now:
    //there are decisions that are hard to make
    //an inverse function of create (remove) is atomic - this requires a few functions to be noexcept - because it would leak very bad otherwise + break logical assumptions across the application - it's better to compromise the logic bug here
    //write partially guarantees atomicity - a failed operation  leave the underlying data in either the original state or the safe-corrupted-state (such that the data is compromised yet an attempt to access the data would result in an error)
    //a SUCCESS write guarantees the next SUCCESS read to read the original data (this is fileio_chksum feature) 
    //std memory exhaustion needs to be handled
    //dg_file_exists could be cached to avoid fsys_call (this assumption assumes that the application has unique reference to the object) - this is not an optimizable by directly calling get_metadata
    //filepath could be a virtual filepath - reinventing the kernel fsystem - this is a nightmare - yet this is the least of our worries in our field of work
    
    struct Metadata{
        std::vector<std::string> datapath_vec;
        std::vector<bool> path_status_vec;
        std::optional<uint64_t> file_sz;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(datapath_vec, path_status_vec, file_sz);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(datapath_vec, path_status_vec, file_sz);
        }
    };

    static inline std::string METADATA_SUFFIX           = "DFSYS_UNIFIED_X_METADATA";
    static inline constexpr size_t MAX_METADATA_SIZE    = size_t{1} << 10; 

    auto dg_internal_get_metadata_fp(const char * fp) noexcept -> std::filesystem::path{

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

        if (!metadata->file_sz.has_value()){
            return std::unexpected(dg::network_exception::RUNTIME_FILEIO_ERROR);
        }

        return metadata->file_sz.value();
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
         
        std::fill(metadata->path_status_vec.begin(), metadata->path_status_vec.end(), false);
        metadata->file_sz = std::nullopt;
        exception_t metadata_write_err = dg_internal_write_metadata(fp, metadata.value());
        
        if (dg::network_exception::is_failed(metadata_write_err)){
            return metadata_write_err;
        }

        auto exception_vec = std::vector<exception_t>{}; 

        for (size_t i = 0u; i < metadata->datapath_vec.size(); ++i){
            const char * fp = metadata->datapath_vec[i].c_str();
            exception_t err = dg::network_fileio_chksum_x::dg_write_binary_direct(fp, src, src_sz);
            exception_vec.push_back(err);
        }

        if (std::find(exception_vec.begin(), exception_vec.end(), dg::network_exception::SUCCESS) == exception_vec.end()){
            return dg::network_exception::RUNTIME_FILEIO_ERROR; //promote error code - runtime_fileio_error
        }

        std::transform(exception_vec.begin(), exception_vec.end(), metadata->path_status_vec.begin(), dg::network_exception::is_success);
        metadata->file_sz = src_sz;
        
        return dg_internal_write_metadata(fp, metadata.value());
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

        std::fill(metadata->path_status_vec.begin(), metadata->path_status_vec.end(), false);
        metadata->file_sz = std::nullopt;
        exception_t metadata_write_err = dg_internal_write_metadata(fp, metadata.value());
        
        if (dg::network_exception::is_failed(metadata_write_err)){
            return metadata_write_err;
        }

        auto exception_vec = std::vector<exception_t>{};

        for (size_t i = 0u; i < metadata->datapath_vec.size(); ++i){
            const char * fp = metadata->datapath_vec[i].c_str();
            exception_t err = dg::network_fileio_chksum_x::dg_write_binary_indirect(fp, src, src_sz);
            exception_vec.push_back(err);
        }

        if (std::find(exception_vec.begin(), exception_vec.end(), dg::network_exception::SUCCESS) == exception_vec.end()){
            return dg::network_exception::RUNTIME_FILEIO_ERROR; //promote error_code -> runtime fileio_error
        }

        std::transform(exception_vec.begin(), exception_vec.end(), metadata->path_status_vec.begin(), dg::network_exception::is_success);
        metadata->file_sz = src_sz;
        
        return dg_internal_write_metadata(fp, metadata.value());
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