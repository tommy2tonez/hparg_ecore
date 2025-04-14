#ifndef __DG_NETWORK_UNIFIED_FILEIO_H__
#define __DG_NETWORK_UNIFIED_FILEIO_H__

//define HEADER_CONTROL 10

#include "network_fileio.h"
#include "network_fileio_chksum_x.h"
#include "network_hash.h"
#include <filesystem>
#include "network_exception.h"
#include "stdx.h"
#include <filesystem>
#include "network_compact_serializer.h"
#include "network_std_container.h"
#include "network_stack_allocation.h"

namespace dg::network_fileio_unified_x{

    struct Metadata{
        dg::vector<dg::string> datapath_vec;
        dg::vector<bool> path_status_vec;
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

    static inline std::string METADATA_SUFFIX                       = "DGFSYS_UNIFIED_X_METADATA";
    static inline std::string METADATA_EXT                          = "data";
    static inline constexpr uint32_t METADATA_SERIALIZATION_SECRET  = 1034526840ULL;
    static inline constexpr size_t MIN_DATAPATH_SIZE                = 2;
    static inline constexpr size_t MAX_DATAPATH_SIZE                = 32u;
    static inline constexpr size_t MAX_METADATA_SIZE                = size_t{1} << 10;
    static inline constexpr size_t DG_LEAST_DIRECTIO_BLK_SZ         = dg::network_fileio_chksum_x::DG_LEAST_DIRECTIO_BLK_SZ; 

    //we'll try to be languagely correct, we'll move on to internal allocations technique later
    //the reference frame was too confusing
    //I mean he is correct

    auto dg_internal_get_metadata_fp(const char * fp) noexcept -> std::expected<std::filesystem::path, exception_t>{

        try{
            auto rawname        = std::filesystem::path(fp).replace_extension("").filename();
            auto new_rawname    = std::format("{}_{}", rawname.native(), METADATA_SUFFIX); 
            auto new_fp         = std::filesystem::path(fp).replace_filename(new_rawname).replace_extension(METADATA_EXT);

            return new_fp;
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }
    }

    auto dg_internal_create_metadata(const char * fp, const Metadata& metadata) noexcept -> exception_t{

        std::expected<std::filesystem::path, exception_t> metadata_fp = dg_internal_get_metadata_fp(fp);

        if (!metadata_fp.has_value()){
            return metadata_fp.error();
        }

        size_t metadata_sz = dg::network_compact_serializer::capintegrity_size(metadata);

        if (metadata_sz > MAX_METADATA_SIZE){
            return dg::network_exception::ALOTTED_BUFFER_EXCEEDED;
        }

        //we'll stick with stack allocations for now
        // dg::network_stack_allocation::NoExceptRawAllocation<char[]> bstream(metadata_sz);
        std::unique_ptr<char[], decltype(&std::free)> bstream(static_cast<char *>(std::malloc(metadata_sz)), std::free);

        if (bstream == nullptr){
            return dg::network_exception::RESOURCE_EXHAUSTION;
        }

        dg::network_compact_serializer::capintegrity_serialize_into(bstream.get(), metadata, METADATA_SERIALIZATION_SECRET);
        exception_t create_err  = dg::network_fileio::dg_create_cbinary(metadata_fp->c_str(), metadata_sz);

        if (dg::network_exception::is_failed(create_err)){
            return create_err;
        }

        exception_t write_err   = dg::network_fileio::dg_write_binary(metadata_fp->c_str(), bstream.get(), metadata_sz);

        if (dg::network_exception::is_failed(write_err)){
            dg::network_exception_handler::nothrow_log(dg::network_fileio::dg_remove(metadata_fp->c_str()));
            return write_err;
        }

        return dg::network_exception::SUCCESS;
    }

    auto dg_internal_exists_metadata(const char * fp) noexcept -> std::expected<bool, exception_t>{

        std::expected<std::filesystem::path, exception_t> metadata_fp = dg_internal_get_metadata_fp(fp);

        if (!metadata_fp.has_value()){
            return std::unexpected(metadata_fp.error());
        }

        std::expected<bool, exception_t> status = dg::network_fileio::dg_file_exists(metadata_fp->c_str());

        if (!status.has_value()){
            return std::unexpected(status.error());
        }

        return status.value();
    } 

    auto dg_internal_remove_metadata(const char * fp) noexcept -> exception_t{

        std::expected<std::filesystem::path, exception_t> metadata_fp = dg_internal_get_metadata_fp(fp);

        if (!metadata_fp.has_value()){
            return metadata_fp.error();
        }

        return dg::network_fileio::dg_remove(metadata_fp->c_str());
    }

    auto dg_internal_read_metadata(const char * fp) noexcept -> std::expected<Metadata, exception_t>{

        std::expected<std::filesystem::path, exception_t> metadata_fp = dg_internal_get_metadata_fp(fp);

        if (!metadata_fp.has_value()){
            return std::unexpected(metadata_fp.error());
        }

        // dg::network_stack_allocation::NoExceptRawAllocation<char[]> bstream(MAX_METADATA_SIZE);
 
         std::unique_ptr<char[], decltype(&std::free)> bstream(static_cast<char *>(std::malloc(MAX_METADATA_SIZE)), std::free);

        if (bstream == nullptr){
            return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
        }

        exception_t err = dg::network_fileio::dg_read_binary(metadata_fp->c_str(), bstream.get(), MAX_METADATA_SIZE);

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        Metadata rs{};
        std::expected<const char *, exception_t> deserialization_err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::capintegrity_deserialize_into<Metadata>)(rs, bstream.get(), MAX_METADATA_SIZE, METADATA_SERIALIZATION_SECRET); //I very forgot

        if (!deserialization_err.has_value()){
            return std::unexpected(deserialization_err.error());
        }

        return rs;
    }

    auto dg_internal_write_metadata(const char * fp, const Metadata& metadata) noexcept -> exception_t{

        std::expected<std::filesystem::path, exception_t> metadata_fp = dg_internal_get_metadata_fp(fp);

        if (!metadata_fp.has_value()){
            return metadata_fp.error();
        }

        size_t metadata_sz = dg::network_compact_serializer::capintegrity_size(metadata);

        if (metadata_sz > MAX_METADATA_SIZE){
            return dg::network_exception::ALOTTED_BUFFER_EXCEEDED;
        }

        // dg::network_stack_allocation::NoExceptRawAllocation<char[]> bstream(metadata_sz);

        std::unique_ptr<char[], decltype(&std::free)> bstream(static_cast<char *>(std::malloc(metadata_sz)), std::free);

        if (bstream == nullptr){
            return dg::network_exception::RESOURCE_EXHAUSTION;
        }

        dg::network_compact_serializer::capintegrity_serialize_into(bstream.get(), metadata, METADATA_SERIALIZATION_SECRET);

        return dg::network_fileio::dg_write_binary(metadata_fp->c_str(), bstream.get(), metadata_sz);
    }

    auto dg_internal_make_metadata(const std::vector<std::string>& datapath_vec, size_t file_sz) noexcept -> std::expected<Metadata, exception_t>{

        try{
            return Metadata{datapath_vec, dg::vector<bool>(datapath_vec.size(), true), file_sz}; //alright, this might not be correct
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }
    }

    auto dg_create_cbinary(const char * fp, const std::vector<std::string>& datapath_vec, size_t file_sz) noexcept -> exception_t{

        if (std::clamp(static_cast<size_t>(datapath_vec.size()), MIN_DATAPATH_SIZE, MAX_DATAPATH_SIZE) != datapath_vec.size()){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        auto resource_guard_vec = dg::network_exception::cstyle_initialize<dg::vector<std::unique_ptr<stdx::VirtualResourceGuard>>>(datapath_vec.size());

        if (!resource_guard_vec.has_value()){
            return resource_guard_vec.error();
        }

        std::expected<Metadata, exception_t> metadata = dg_internal_make_metadata(datapath_vec, file_sz);

        if (!metadata.has_value()){
            return metadata.error();
        }

        exception_t md_create_err = dg_internal_create_metadata(fp, metadata.value());

        if (dg::network_exception::is_failed(md_create_err)){
            return md_create_err;
        }

        auto metadata_grd = stdx::resource_guard([fp]() noexcept{
            dg::network_exception_handler::nothrow_log(dg_internal_remove_metadata(fp));
        });

        for (size_t i = 0u; i < datapath_vec.size(); ++i){
            exception_t bin_create_err = dg::network_fileio_chksum_x::dg_create_cbinary(datapath_vec[i].c_str(), file_sz);

            if (dg::network_exception::is_failed(bin_create_err)){
                return bin_create_err;
            }

            auto bin_resource_task  = [ffp = &datapath_vec[i]]() noexcept{
                dg::network_exception_handler::nothrow_log(dg::network_fileio_chksum_x::dg_remove(ffp->c_str()));
            };

            auto bin_resource_grd   = dg::network_exception::to_cstyle_function(stdx::vresource_guard<decltype(bin_resource_task)>)(bin_resource_task);

            if (!bin_resource_grd.has_value()){
                return bin_resource_grd.error();
            }

            resource_guard_vec.value()[i] = std::move(bin_resource_grd.value());
        }

        for (auto& e: resource_guard_vec.value()){
            e->release();
        }

        metadata_grd.release();

        return dg::network_exception::SUCCESS;
    }

    auto dg_remove(const char * fp) noexcept -> exception_t{

        Metadata metadata = dg::network_exception_handler::nothrow_log(dg_internal_read_metadata(fp));

        for (const auto& path: metadata.datapath_vec){
            exception_t bin_remove_err = dg::network_fileio_chksum_x::dg_remove(path.c_str());

            if (dg::network_exception::is_failed(bin_remove_err)){
                return bin_remove_err;
            }
        }

        return dg_internal_remove_metadata(fp);
    }

    auto dg_file_exists(const char * fp) noexcept -> std::expected<bool, exception_t>{

        return dg_internal_exists_metadata(fp);
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
                    return dg::network_exception::SUCCESS;
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
                    return dg::network_exception::SUCCESS;
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

    //alright, we arent splitting the responsibility correctly
    //this is supposed to be an extension of the chksum_x
    //there is no guarantee of failed write in every scenerio, it only increases the chances of success write + next success read
    //the problem with this is that this is carrying more than one responsibility, thus it voids the value of what this is supposed to do
    //in this case, we must guarantee that the write_metadata is through, the corruption is serious enough to actually be constituted as a panic error
    //we are awared of 1024 other ways to die a program
    //until we've found a sound patch, we'll move on with the solution
    //the sound patch is probably to store the metadata_fp on a RAM virtual disk
    //we have not heard of a RAM virtual disk has a cache flush fail once
    //this implies serious kernel corruption, which is as unlikely as RAM failure or memory corruptions (nasty things could appear)

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

        metadata->file_sz = src_sz; 

        for (size_t i = 0u; i < metadata->datapath_vec.size(); ++i){
            const char * ffp                = metadata->datapath_vec[i].c_str();
            exception_t bin_write_err       = dg::network_fileio_chksum_x::dg_write_binary_direct(ffp, src, src_sz);
            metadata->path_status_vec[i]    = dg::network_exception::is_success(bin_write_err);
        }

        bool has_atleast_one_success = std::find(metadata->path_status_vec.begin(), metadata->path_status_vec.end(), true) != metadata->path_status_vec.end(); 
        dg::network_exception_handler::nothrow_log(dg_internal_write_metadata(fp, metadata.value()));

        if (!has_atleast_one_success){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        return dg::network_exception::SUCCESS;
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

        metadata->file_sz = src_sz; 

        for (size_t i = 0u; i < metadata->datapath_vec.size(); ++i){
            const char * ffp                = metadata->datapath_vec[i].c_str();
            exception_t bin_write_err       = dg::network_fileio_chksum_x::dg_write_binary_indirect(ffp, src, src_sz);
            metadata->path_status_vec[i]    = dg::network_exception::is_success(bin_write_err);
        }

        bool has_atleast_one_success = std::find(metadata->path_status_vec.begin(), metadata->path_status_vec.end(), true) != metadata->path_status_vec.end(); 
        dg::network_exception_handler::nothrow_log(dg_internal_write_metadata(fp, metadata.value()));

        if (!has_atleast_one_success){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;        
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