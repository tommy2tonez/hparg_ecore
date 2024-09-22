#ifndef __NETWORK_FILEIO_H__
#define __NETWORK_FILEIO_H__

#include "network_trivial_serializer.h" 
#include "network_hash.h"
#include "network_utility.h"

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

    //separate header file is the right approach - not an intuitive one - yet the "not intuitive" is a problem to solve - not a problem to blame
    //don't argue with me about file and offset - that's also another problem to solve
    //different files guarantee concurrent support from OS - I never get the "offset" thing 

    struct FileHeader{
        uint64_t chksum;
        uint64_t timestamp_in_nanoseconds; //validate with internal in-mem - if last_updated is not as up-to-date as in-mem == corrupted - this is maybe not neccessary 

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const{

            reflector(chksum, timestamp_in_nanoseconds);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector){

            reflector(chksum, timestamp_in_nanoseconds);
        }
    };

    static inline std::string METADATA_SUFFIX = "DG_NETWORK_CHKSUM_X_METADATA"; 

    auto dg_get_metadata_fp(const char * fp) noexcept -> std::filesystem::path{

        auto ext            = std::filesystem::path(fp).extension();
        auto rawname        = std::filesystem::path(fp).replace_extension("").filename();
        auto new_rawname    = std::format("{}_{}", rawname, METADATA_SUFFIX); 
        auto new_fp         = std::filesystem::path(fp).replace_filename(new_rawname).replace_extension(ext);

        return new_fp;
    }

    auto dg_file_exists(const char * fp) noexcept -> std::expected<bool, exception_t>{

        std::filesystem::path metadata_path         = dg_get_metadata_fp(fp);
        std::expected<bool, exception_t> f_status   = dg::network_fileio::dg_file_exists(fp);
        
        if (!status.has_value()){
            return std::unexpected(status.error());
        }

        if (!status.value()){
            return false;
        }

        std::expected<bool, exception_t> m_status   = dg::network_fileio::dg_file_exists(metadata_path.c_str());

        if (!m_status.has_value()){
            return std::unexpected(m_status.error());
        }
        
        if (!m_status.value()){
            return false;
        }

        return true;
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

    auto dg_create_binary(const char * fp, size_t fsz) noexcept -> exception_t{

        std::filesystem::path header_fname = dg_get_metadata_fp(fp); 
        auto backout_lambda = [=]() noexcept{
            if (dg::network_fileio::dg_file_exists_nothrow(fp)){
                dg::network_fileio::remove_nothrow(fp);
            }

            if (dg::network_fileio::dg_file_exists_nothrow(header_fname.c_str())){
                dg::network_fileio::remove_nothrow(header_fname.c_str());
            }
        };

        auto backout_guard  = dg::network_genult::resource_guard(std::move(backout_lambda));
        exception_t err     = dg::network_fileio::dg_create_binary(fp, fsz);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        err = dg::network_fileio::dg_create_binary(header_fname.c_str(), SERIALIZATION_SZ);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        constexpr size_t SERIALIZATION_SZ = dg::network_trivial_serializer::size(FileHeader{});
        auto serialized_header = std::array<char, SERIALIZATION_SZ>{};
        auto header = FileHeader{dg::network_hash::murmur_hash(nullptr, 0u), static_cast<std::chrono::nanoseconds>(dg::network_utility::utc_timestamp()).count()}; 
        dg::network_trivial_serializer::serialize_into(serialized_header.data(), header);
        err = dg::network_fileio::dg_write_binary_indirect(header_fname.c_str(), serialized_header.data(), SERIALIZATION_SZ);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        backout_guard.release();
        return dg::network_exception::SUCCESS;
    }

    auto dg_create_binary_nothrow(const char * fp, size_t fsz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_create_binary(fp, fsz));
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

        constexpr size_t HEADER_SZ          = dg::network_trivial_serializer::size(FileHeader{}); 
        std::filesystem::path header_path   = dg_get_metadata_fp(fp);
        auto serialized_header              = std::array<char, HEADER_SZ>{};
        err = dg::network_fileio::dg_read_binary_indirect(header_path.c_str(), serialized_header.data(), HEADER_SZ);

        if (dg::network_exception::is_failed(err)){
            return err;
        }
        
        std::expected<size_t, exception_t> fsz = dg::network_fileio::dg_file_size(fp);

        if (!fsz.has_value()){
            return fsz.error();
        }

        FileHeader header{};
        dg::network_trivial_serializer::deserialize_into(header, serialized_header.data());
        uint64_t file_chksum = dg::network_hash::murmur_hash(dst, fsz.value()); 

        if (file_chksum != header.chksum){
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

        constexpr size_t HEADER_SZ          = dg::network_trivial_serializer::size(FileHeader{});
        std::filesystem::path header_path   = dg_get_metadata_fp(fp);
        auto serialized_header              = std::array<char, HEADER_SZ>{};
        err = dg::network_fileio::dg_read_binary_indirect(header_path.c_str(), serialized_header.data(), HEADER_SZ);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        std::expected<size_t, exception_t> fsz = dg::network_fileio::dg_file_size(fp);

        if (!fsz.has_value()){
            return fsz.error();
        }

        FileHeader header{};
        dg::network_trivial_serializer::deserialize_into(header, serialized_header.data());
        uint64_t file_chksum = dg::network_hash::murmur_hash(dst, fsz.value());

        if (file_chksum != header.chksum){
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

        constexpr size_t HEADER_SZ          = dg::network_trivial_serializer::size(FileHeader{});    
        std::filesystem::path header_path   = dg_get_metadata_fp(fp);
        auto header                         = FileHeader{dg::network_hash::murmur_hash(src, src_sz), static_cast<std::chrono::nanoseconds>(dg::network_genult::utc_timestamp()).count()}; 
        auto serialized_header              = std::array<char, HEADER_SZ>{};
        dg::network_trivial_serializer::serialize_into(serialized_header.data(), header);
        err = dg::network_fileio::dg_write_binary_indirect(header_path.c_str(), serialized_header.data(), HEADER_SZ);

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

        constexpr size_t HEADER_SZ          = dg::network_trivial_serializer::size(FileHeader{});
        std::filesystem::path header_path   = dg_get_metadata_fp(fp);
        auto header                         = FileHeader{dg::network_hash::murmur_hash(src, src_sz), static_cast<std::chrono::nanoseconds>(dg::network_genult::utc_timestamp()).count()};
        auto serialized_header              = std::array<char, HEADER_SZ>{};
        dg::network_trivial_serializer::serialize_into(serialized_header.data(), header);
        err = dg::network_fileio::dg_write_binary_indirect(header_path.c_str(), serialized_header.data(), HEADER_SZ);

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