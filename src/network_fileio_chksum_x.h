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

    //improve error_code return - convert the errors -> RUNTIME_FILEIO_ERROR for generic purpose 

    //1 success write == guarantee the next success read to read the success write, this is the sole contract that we provide
    //unified_fsys_io would keep track of the contract status, the only thing that is wrong is the metadata header of the unified_fsys_io
    //alright, this should suffice
    
    //let's see what we are trying to do

    //we have a buffer of <char *, size_t>
    //legacy fsys: 1 success write == nasal demon
    //                             == sometimes correct inverse operation read, sometimes incorrect, sometimes modified the file, sometimes the file disappeared out of thin air
    //chksum_x: 1 success write == very likely  (chances better than RAM) good success read, which means the next success read would return the success write data, does not mean that the next success read means nasal demon
    //unified_x: increase the chances of success write and next success read by using multiple data_sources 

    //we dont care about not success write, not success write could mean everything, from modifying the file, to changing the file size -> 0, -> oo
    //we dont care about ACID property either, such is we are assuming that the storage is invalidated if the power is off, or gammar beta ray RAM attacks
    //ACID is storage's business, we aren't doing storage, we are doing forward + backward
    //not a lot of people understand this
    //I mean 99% of people dont understand this, the contract of 1 success write, and the next success read means the inverse operation of the write
    //alright, we need to add support for unordered_map metadata on RAM
    //because the fsys is nasal demon, we have absolutely no guarantee that the read file is on cache, and a cache flush to disk is written later on 
    //disk might block all the cache flush and return the previous tx, which is also considered a valid read

    //we have covered 99.99999999999% of cases
    //our fault tolerance is probably at 10 ** 18 bytes/ error
    //far from our RAM fault
    //its complicated, really is
    //fsys is a very bad place to work on
    //alright, let's strategize 
    //we know for sure that a fsys read would issue at least 1 memory ordering on the kernel side, this is unavoidable
    //so it's fine to actually build a distributed unordered_map to store the metadata_fsys
    //we now face with the problem of how big the map should be?
    //is it a temporal unordered map or replace map, only to reduce the statistical chances, not to bring the fault rate -> RAM rate
    //after all, we are messing with statistics, why use an absolute instrument when there is actually nothing really is absolute  
    //let's stick with the replace map, we are to load std::memory_order_relaxed of std::array<relaxable_sz, char>
    //remember, we are to mess with the statistics, we are to return as many failed read() as possible
    //we load the relaxed in chunks, because we are messing with statistics 
    //the notsynchronizedchance of relaxed is included in the chances of corrupted read()
    //we are using one nasal demon to fight with another nasal demon
    //the two have to agree with each other, this is the punch line

    //this is defined piece of code
    //assume that the fsys nasal demon is incorrect, our map nasal demon is correct (out)
    //                                    incorrect, incorrect (another nasal demon, the chances of this guy to happen is close to 0, matched fp, matched chksum, alright, we need to specialize chksum, we dont go there yet)
    //                                    correct, correct (in)
    //                                    correct, incorrect (out)
    //whatever, we'll move on with the implementation

    //it's complex

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

    static inline constexpr size_t MAX_FILE_PATH_SZ                         = 32u;
    static inline std::string METADATA_SUFFIX                               = "DGFSYS_CHKSUM_X_METADATA"; 
    static inline std::string METADATA_EXT                                  = "data";
    static inline constexpr size_t DG_LEAST_DIRECTIO_BLK_SZ                 = dg::network_fileio::DG_LEAST_DIRECTIO_BLK_SZ;

    static inline constexpr bool HAS_DISTRIBUTED_FILE_HEADER_MAP_SUPPORT    = true;
    static inline constexpr bool FILE_HEADER_MAP_SZ                         = size_t{1} << 20;  

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
        auto header                         = FileHeader{dg::network_hash::murmur_hash_base(empty_buf, fsz), fsz};
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

        std::pair<uint64_t, uint64_t> file_chksum = dg::network_hash::murmur_hash_base(reinterpret_cast<const char *>(dst), header->content_size); //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 

        if (file_chksum != header->chksum){
            return dg::network_exception::CORRUPTED_FILE;
        }
        
        if constexpr(HAS_DISTRIBUTED_FILE_HEADER_MAP_SUPPORT){
            std::optional<FileHeader> fh = DistributedFileHeaderMap<>::read(fp);

            if (fh.has_value()){
                if (!reflectible_equal(fh.value(), header.value())){
                    return dg::network_exception::CORRUPTED_FILE;
                }
            }
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

        std::pair<uint64_t, uint64_t> file_chksum = dg::network_hash::murmur_hash_base(reinterpret_cast<const char *>(dst), header->content_size); //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 

        if (file_chksum != header->chksum){
            return dg::network_exception::CORRUPTED_FILE;
        }

        if constexpr(HAS_DISTRIBUTED_FILE_HEADER_MAP_SUPPORT){
            std::optional<FileHeader> fh = DistributedFileHeaderMap<>::read(fp);

            if (fh.has_value()){
                if (!reflectible_equal(fh.value(), header.value())){
                    return dg::network_exception::CORRUPTED_FILE;
                }
            }
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

        FileHeader header{dg::network_hash::murmur_hash_base(reinterpret_cast<const char *>(src), src_sz), src_sz};  //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 
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

        FileHeader header{dg::network_hash::murmur_hash_base(reinterpret_cast<const char *>(src), src_sz), src_sz};  //UB - keyword reinterpret_cast - resolution: start_lifetime_as_array, or precond void * to be a static_cast<void *>(char *) - this is a bad practice - where precond could be enforced by function signature 
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
