#ifndef __NETWORK_FILEIO_CHKSUM_H__
#define __NETWORK_FILEIO_CHKSUM_H__

//define HEADER_CONTROL 9

#include "network_trivial_serializer.h" 
#include "network_hash.h"
#include "stdx.h"
#include "network_fileio.h"
#include "network_exception_handler.h"
#include "network_datastructure.h"
#include <type_traits>

namespace dg::network_fileio_chksum_x{

    //socket is about 1 transmit == max one recv
    //file is about 1 success write == next success read is the inverse operation of the write
    
    //we need to improve the hash_map, it's complicated, the memory_order_acquire + memory_order_release would void the value of our fileio_chksum_x 
    //the hash_map is only good when used with the unified_fileio, and the unified_io does the business of multiple buckets to avoid the false negative filter of the chksum_x
    //remember, the chksum_x responsibility is to filter as many as possible, as long as the true positive rate is high and false positive rate is low
    //we can have false negatives, that's irrelevant
    //alright, this is ..., against our principle of correct is correct, incorrect is incorrect
    //let's improve this
    //we are to design a relaxed hash_map
    //we'll be back
    //I know what you are referring 
    //we chance better than most fellas out there
    //the chance of all bucket collisions for 10 rehashed replicated filepath is lower than that of RAM 
    //as long as we chance better than RAM, I dont even care about the absolute accuracy
    //I mean even the bool cmp can be flipped from false to true or true to false

    struct FileHeader{
        std::pair<uint64_t, uint64_t> chksum;
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

    static inline constexpr size_t MAX_FILE_PATH_SZ                         = 128u;
    static inline std::string METADATA_SUFFIX                               = "DGFSYS_CHKSUM_X_METADATA"; 
    static inline std::string METADATA_EXT                                  = "data";
    static inline constexpr size_t DG_LEAST_DIRECTIO_BLK_SZ                 = dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ;

    static inline constexpr bool HAS_DISTRIBUTED_FILE_HEADER_MAP_SUPPORT    = true; //this is only used as the last resort for increasing true positive or decreasing false positive
    static inline constexpr bool FILE_HEADER_MAP_SZ                         = size_t{1} << 20;  
    static inline constexpr uint32_t HASH_SECRET                            = 4228045292ULL;

    auto raw_fp_to_array(const char * fp) noexcept -> std::array<char, MAX_FILE_PATH_SZ>{

        std::string_view sv(fp);
        size_t new_sz = std::min(MAX_FILE_PATH_SZ, static_cast<size_t>(sv.size()));
        std::array<char, MAX_FILE_PATH_SZ> rs{};
        std::copy(fp, std::next(fp, new_sz), rs.data());

        return rs; 
    }

    struct FileHeaderMapBucket{
        std::array<char, MAX_FILE_PATH_SZ> fp;
        FileHeader file_header;
        bool is_initialized;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(fp, file_header, is_initialized);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(fp, file_header, is_initialized);
        }
    };

    template <class = void>
    class DistributedFileHeaderMap{};

    template <>
    class DistributedFileHeaderMap<std::void_t<std::enable_if_t<HAS_DISTRIBUTED_FILE_HEADER_MAP_SUPPORT>>>{

        private:

            using bucket_loader_t   = dg::network_datastructure::atomic_loader::reflectible_relaxed_loader<FileHeaderMapBucket>;           
            using self              = DistributedFileHeaderMap;

            static inline std::unique_ptr<bucket_loader_t[]> bucket_array = []{
                
                auto rs = std::make_unique<bucket_loader_t[]>(FILE_HEADER_MAP_SZ);

                for (size_t i = 0u; i < FILE_HEADER_MAP_SZ; ++i){
                    rs[i].unload(FileHeaderMapBucket{{}, {}, false});
                }

                return rs;
            }();

        public:

            static auto read(const char * fp) noexcept -> std::optional<FileHeader>{

                stdx::seq_cst_guard seqcst_tx;

                std::array<char, MAX_FILE_PATH_SZ> char_fp  = raw_fp_to_array(fp);
                size_t hashed_value                         = dg::network_hash::hash_reflectible(char_fp);
                size_t idx                                  = hashed_value % FILE_HEADER_MAP_SZ;
                FileHeaderMapBucket bucket                  = self::bucket_array[idx].load();

                if (!bucket.is_initialized){
                    return std::nullopt;
                }

                if (bucket.fp != char_fp){
                    return std::nullopt;
                }

                return bucket.file_header;
            }

            static void push(const char * fp, FileHeader header) noexcept{

                stdx::seq_cst_guard seqcst_tx;

                std::array<char, MAX_FILE_PATH_SZ> char_fp  = raw_fp_to_array(fp);
                size_t hashed_value                         = dg::network_hash::hash_reflectible(char_fp);
                size_t idx                                  = hashed_value % FILE_HEADER_MAP_SZ;

                self::bucket_array[idx].unload(FileHeaderMapBucket{char_fp, header, true});
            }
    };

    template <class T>
    inline auto reflectible_equal(T lhs, T rhs) noexcept -> bool{

        constexpr size_t REFLECTIBLE_SZ = dg::network_trivial_serializer::size(T{});

        std::array<char, REFLECTIBLE_SZ> lhs_byte_representation{};
        std::array<char, REFLECTIBLE_SZ> rhs_byte_representation{};

        dg::network_trivial_serializer::serialize_into(lhs_byte_representation.data(), lhs);
        dg::network_trivial_serializer::serialize_into(rhs_byte_representation.data(), rhs);

        return lhs_byte_representation == rhs_byte_representation;
    } 

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

    auto dg_internal_create_metadata(const char * fp) noexcept -> exception_t{

        std::expected<std::filesystem::path, exception_t> header_path   = dg_internal_get_metadata_fp(fp);
        
        if (!header_path.has_value()){
            return header_path.error();
        }

        constexpr size_t HEADER_SZ          = dg::network_trivial_serializer::size(FileHeader{}); 
        exception_t err                     = dg::network_fileio::dg_create_binary(header_path->c_str(), HEADER_SZ);

        return err;
    }

    auto dg_internal_remove_metadata(const char * fp) noexcept -> exception_t{

        std::expected<std::filesystem::path, exception_t> header_path   = dg_internal_get_metadata_fp(fp);

        if (!header_path.has_value()){
            return header_path.error();
        }

        return dg::network_fileio::dg_remove(header_path->c_str());
    }

    auto dg_internal_read_metadata(const char * fp) noexcept -> std::expected<FileHeader, exception_t>{

        std::expected<std::filesystem::path, exception_t> header_path   = dg_internal_get_metadata_fp(fp);

        if (!header_path.has_value()){
            return std::unexpected(header_path.error());
        }

        constexpr size_t HEADER_SZ          = dg::network_trivial_serializer::size(FileHeader{}); 
        auto serialized_header              = std::array<char, HEADER_SZ>{};
        exception_t err                     = dg::network_fileio::dg_read_binary_indirect(header_path->c_str(), serialized_header.data(), HEADER_SZ);

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        FileHeader header{};
        dg::network_trivial_serializer::deserialize_into(header, serialized_header.data());
        
        return header;
    }

    auto dg_internal_write_metadata(const char * fp, FileHeader metadata) noexcept -> exception_t{

        std::expected<std::filesystem::path, exception_t> header_path   = dg_internal_get_metadata_fp(fp); 

        if (!header_path.has_value()){
            return header_path.error();
        }

        constexpr size_t HEADER_SZ          = dg::network_trivial_serializer::size(FileHeader{});
        auto serialized_header              = std::array<char, HEADER_SZ>{};
        dg::network_trivial_serializer::serialize_into(serialized_header.data(), metadata);

        return dg::network_fileio::dg_write_binary_indirect(header_path->c_str(), serialized_header.data(), HEADER_SZ);
    }

    auto dg_internal_is_file_creatable(const char * fp) noexcept -> std::expected<bool, exception_t>{

        std::expected<std::filesystem::path, exception_t> metadata_path = dg_internal_get_metadata_fp(fp);

        if (!metadata_path.has_value()){
            return std::unexpected(metadata_path.error());
        }

        std::expected<bool, exception_t> status = dg::network_fileio::dg_file_exists(metadata_path->c_str());

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

        std::expected<std::filesystem::path, exception_t> metadata_path = dg_internal_get_metadata_fp(fp);

        if (!metadata_path.has_value()){
            return std::unexpected(metadata_path.error());
        }

        std::expected<bool, exception_t> status = dg::network_fileio::dg_file_exists(metadata_path->c_str());

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
            return std::unexpected(dg::network_exception::FILE_NOT_FOUND);
        }

        return dg::network_fileio::dg_file_size(fp);
    }

    auto dg_create_cbinary(const char * fp, size_t fsz) noexcept -> exception_t{

        if (std::string_view(fp).size() > MAX_FILE_PATH_SZ){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

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

        char * empty_buf                    = static_cast<char *>(std::calloc(fsz, sizeof(char)));
        auto header                         = FileHeader{dg::network_hash::murmur_hash_base(empty_buf, fsz, HASH_SECRET), fsz};
        std::free(empty_buf);
        exception_t header_write_err        = dg_internal_write_metadata(fp, header);

        if (dg::network_exception::is_failed(header_write_err)){
            return header_write_err;
        }

        header_create_guard.release();
        file_create_guard.release();

        if constexpr(HAS_DISTRIBUTED_FILE_HEADER_MAP_SUPPORT){
            DistributedFileHeaderMap<>::push(fp, header);
        }

        return dg::network_exception::SUCCESS;
    }

    auto dg_remove(const char * fp) noexcept -> exception_t{

        //alright, we are being forcy
        //we have to assume that if the user wants to be except-open, noexcept-close, they have to apply their own measurements

        exception_t content_rm_err = dg::network_fileio::dg_remove(fp);

        if (dg::network_exception::is_failed(content_rm_err)){
            return content_rm_err;
        }

        return dg_internal_remove_metadata(fp);
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

        std::pair<uint64_t, uint64_t> file_chksum = dg::network_hash::murmur_hash_base(reinterpret_cast<const char *>(dst), header->content_size, HASH_SECRET); //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 

        if (file_chksum != header->chksum){
            return dg::network_exception::CORRUPTED_FILE;
        }
        
        // if constexpr(HAS_DISTRIBUTED_FILE_HEADER_MAP_SUPPORT){
        //     std::optional<FileHeader> fh = DistributedFileHeaderMap<>::read(fp);

        //     if (!fh.has_value() || !reflectible_equal(fh.value(), header.value())){
        //         return dg::network_exception::CORRUPTED_FILE;
        //     }
        // }

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

        std::pair<uint64_t, uint64_t> file_chksum = dg::network_hash::murmur_hash_base(reinterpret_cast<const char *>(dst), header->content_size, HASH_SECRET); //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 

        if (file_chksum != header->chksum){
            return dg::network_exception::CORRUPTED_FILE;
        }

        // if constexpr(HAS_DISTRIBUTED_FILE_HEADER_MAP_SUPPORT){
        //     std::optional<FileHeader> fh = DistributedFileHeaderMap<>::read(fp);

        //     if (!fh.has_value() || !reflectible_equal(fh.value(), header.value())){
        //         return dg::network_exception::CORRUPTED_FILE;
        //     }
        // }

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

        FileHeader header{dg::network_hash::murmur_hash_base(reinterpret_cast<const char *>(src), src_sz, HASH_SECRET), src_sz};  //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 
        err = dg_internal_write_metadata(fp, header);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        std::expected<FileHeader, exception_t> cmp_metadata = dg_internal_read_metadata(fp);

        if (!cmp_metadata.has_value() || !reflectible_equal(cmp_metadata.value(), header)){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        if constexpr(HAS_DISTRIBUTED_FILE_HEADER_MAP_SUPPORT){
            DistributedFileHeaderMap<>::push(fp, header);
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

        FileHeader header{dg::network_hash::murmur_hash_base(reinterpret_cast<const char *>(src), src_sz, HASH_SECRET), src_sz};  //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 
        err = dg_internal_write_metadata(fp, header);

        if (dg::network_exception::is_failed(err)){
            return err;
        }

        std::expected<FileHeader, exception_t> cmp_metadata = dg_internal_read_metadata(fp);

        if (!cmp_metadata.has_value() || !reflectible_equal(cmp_metadata.value(), header)){
            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        if constexpr(HAS_DISTRIBUTED_FILE_HEADER_MAP_SUPPORT){
            DistributedFileHeaderMap<>::push(fp, header);
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
