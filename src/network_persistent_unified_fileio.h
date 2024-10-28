#ifndef __DG_NETWORK_UNIFIED_FILEIO_H__
#define __DG_NETWORK_UNIFIED_FILEIO_H__

#include "network_fileio.h"
#include "network_hash.h"
#include <filesystem>
#include "network_exception.h"
#include "network_atomic_x.h"
#include "stdx.h"
#include <filesystem>

namespace dg::network_persistent_unified_fileio{

    struct Metadata{
        dg::vector<std::string> path_vec;
        dg::vector<bool> path_status_vec;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(path_vec, path_status_vec);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(path_vec, path_status_vec);
        }
    };
    
    auto dg_internal_metadata_filename(const char * fp) noexcept -> std::filesystem::path{

    }

    auto dg_internal_read_metadata(const char * fp) noexcept -> std::expected<Metadata, exception_t>{

    }

    auto dg_file_create(const char * fp, const dg::vector<const char *>& replica_path_vec) noexcept -> exception_t{

    }

    auto dg_file_remove(const char * fp) noexcept -> exception_t{

    }

    auto dg_file_exists(const char * unified_fp) noexcept -> std::expected<bool, exception_t>{

        std::filesystem::path metadata_fp       = dg_internal_metadata_filename(unified_fp); 
        std::expected<bool, exception_t> status = dg::network_fileio::dg_file_exists();

        if (!status.has_value()){
            return std::unexpected(status.error());
        }

        return status.value(); 
    }

    auto dg_file_exists_nothrow(const char * unified_fp) noexcept -> bool{

        return dg::network_exception_handler::nothrow_log(dg_file_exists(unified_fp));
    }

    auto dg_file_size(const char * unified_fp) noexcept -> std::expected<size_t, exception_t>{
        
        std::expected<Metadata, exception_t> metadata = dg_internal_read_metadata(unified_fp); 

        if (!metadata.has_value()){
            return std::unexpected(metadata.error());
        }

        size_t replication_sz = metadata->path_vec.size();

        for (size_t i = 0u; i < replication_sz; ++i){
            if (metadata->path_status_vec[i]){
                const char * fp = metadata->path_vec[i].c_str();
                std::expected<size_t, exception_t> fsz = dg::network_fileio::dg_file_size(fp);
                if (fsz.has_value()){
                    return fsz.value();
                }
            }
        }

        return std::unexpected(dg::network_exception::RUNTIME_FILEIO_ERROR);
    } 

    auto dg_file_size_nothrow(const char * unified_fp) noexcept -> size_t{

        return dg::network_exception_handler::nothrow_log(dg_file_size(unified_fp));
    }

    auto dg_read_binary_direct(const char * unified_fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        std::expected<Metadata, exception_t> metadata = dg_internal_read_metadata(unified_fp);

        if (!metadata.has_value()){
            return metadata.error();
        } 

        size_t replication_sz = metadata->path_vec.size();

        for (size_t i = 0u; i < replication_sz; ++i){
            if (metadata->path_status_vec[i]){
                const char * fp = metadata->path_vec[i].c_str();
                exception_t err = dg::network_fileio::dg_read_binary_direct(fp, dst, dst_cap);

                if (err == dg::network_exception::SUCCESS){
                    return err;
                }
            
                if (err == dg::network_exception::BAD_ALIGNMENT){ //assume alignment error is mono - either all has alignment error or none has alignment error
                    return err;
                }

                //assume buffer_overflow error is mono - either all has buffer_overflow or none has buffer_overflow (replication_is_valid's contract)
                //the existence of this error is mysterious - the original intention was to have everything encoded (including the size) in one buffer - a file_read == a dump to a dedicated buffer
                if (err == dg::network_exception::BUFFER_OVERFLOW){  
                    return err;
                }
            }
        }

        return dg::network_exception::RUNTIME_FILEIO_ERROR; //promote error -> runtime_fileio_error
    }

    void dg_read_binary_direct_nothrow(const char * unified_fp, void * dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_read_binary_direct(unified_fp, dst, dst_cap));
    }

    auto dg_read_binary_indirect(const char * unified_fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        std::expected<Metadata, exception_t> metadata = dg_internal_read_metadata(unified_fp);

        if (!metadata.has_value()){
            return metadata.error();
        }

        size_t replication_sz = metadata->path_vec.size();

        for (size_t i = 0u; i < replication_sz; ++i){
            if (metadata->path_status_vec[i]){
                const char * fp = metadata->path_vec[i].c_str();
                exception_t err = dg::network_fileio::dg_read_binary_indirect(fp, dst, dst_cap);

                if (err == dg::network_exception::SUCCESS){
                    return err;
                }

                if (err == dg::network_exception::BUFFER_OVERFLOW){ //assume buffer_overflow error is mono - either all has buffer_flow or none has buffer_overflow (replication_is_valid's contract)
                    return err;
                }
            }
        }

        return dg::network_exception::RUNTIME_FILEIO_ERROR; //premote error -> runtime_fileio_error
    }

    void dg_read_binary_indirect_nothrow(const char * unified_fp, void * dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_read_binary_indirect(unified_fp, dst, dst_cap));
    }

    auto dg_read_binary(const char * unified_fp, void * dst, size_t dst_cap) -> exception_t{

        exception_t err = dg_read_binary_direct(unified_fp, dst, dst_cap);

        if (err == dg::network_exception::SUCCESS){
            return dg::network_exception::SUCCESS;
        }

        return dg_read_binary_indirect(unified_fp, dst, dst_cap);
    }

    void dg_read_binary_nothrow(const char * unified_fp, void * dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_read_binary(unified_fp, dst, dst_cap));
    }

    auto dg_write_binary_direct(const char * unified_fp, const void * src, size_t src_sz) noexcept -> exception_t{
        
        std::expected<Metadata, exception_t> metadata = dg_internal_read_metadata(unified_fp); 

        if (!metadata.has_value()){
            return metadata.error();
        }

        size_t replication_sz   = metadata->path_vec.size();
        auto exception_vec      = dg::vector<exception_t>(); 

        for (size_t i = 0u; i < replication_sz; ++i){
            const char * fp = metadata->path_vec[i].c_str();
            exception_t err = dg::network_fileio::dg_write_binary_direct(fp, src, src_sz);
            exception_vec.push_back(err);
        }

        if (std::find(exception_vec.begin(), exception_vec.end(), dg::network_exception::SUCCESS) != exception_vec.end()){
            for (size_t i = 0u; i < replication_sz; ++i){
                if (exception_vec[i] == dg::network_exception::SUCCESS){
                    metadata->path_status_vec[i] = true;
                } else{
                    metadata->path_status_vec[i] = false;
                }
            }
            
            dg_internal_write_metadata_nothrow(unified_fp, metadata.value()); //fix later
            return dg::network_exception::SUCCESS;
        }

        for (size_t i = 0u; i < replication_sz; ++i){
            metadata->path_status_vec[i] = false;
        }

        dg_internal_write_metadata_nothrow(unified_fp, metadata.value());
        return dg::network_exception::RUNTIME_FILEIO_ERROR; //promote error code - runtime_fileio_error
    }

    void dg_write_binary_direct_nothrow(const char * unified_fp, const void * src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary_direct(unified_fp, src, src_sz));
    }

    auto dg_write_binary_indirect(const char * unified_fp, const void * src, size_t src_sz) noexcept -> exception_t{

        std::expected<Metadata, exception_t> metadata = dg_internal_read_metadata(unified_fp);

        if (!metadata.has_value()){
            return metadata.error();
        }

        size_t replication_sz   = metadata->path_vec.size();
        auto exception_vec      = dg::vector<exception_t>();

        for (size_t i = 0u; i < replication_sz; ++i){
            const char * fp = metadata->path_vec[i].c_str();
            exception_t err = dg::network_fileio::dg_write_binary_indirect(fp, src, sz);
            exception_vec.push_back(err);
        }

        if (std::find(exception_vec.begin(), exception_vec.end(), dg::network_exception::SUCCESS) != exception_vec.end()){
            for (size_t i = 0u; i < replication_sz; ++i){
                if (exception_vec[i] == dg::network_exception::SUCCESS){
                    metadata->path_status_vec[i] = true;
                } else{
                    metadata->path_stauts_vec[i] = false;
                }
            }

            dg_internal_write_metadata_nothrow(unified_fp, metadata.value()); //fix later
            return dg::network_exception::SUCCESS;
        }
        
        for (size_t i = 0u; i < replication_sz; ++i){
            metadata->path_status_vec[i] = false;
        }

        dg_internal_write_metadata_nothrow(unified_fp, metadata.value()); //fix later
        return dg::network_exception::RUNTIME_FILEIO_ERROR; //promote error_code -> runtime fileio_error
    }

    void dg_write_binary_indirect_nothrow(const char * unified_fp, const void * src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary_indirect(unified_fp, src, src_sz));
    }

    auto dg_write_binary(const char * unified_fp, const void * src, size_t src_sz) noexcept -> exception_t{

        exception_t err = dg_write_binary_direct(unified_fp, src, src_sz); 

        if (err == dg::network_exception::SUCCESS){
            return dg::network_exception::SUCCESS;
        }

        return dg_write_binary_indirect(unified_fp, src, src_sz);
    }

    void dg_write_binary_nothrow(const char * unified_fp, const void * src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary(unified_fp, src, src_sz));
    }
}

#endif