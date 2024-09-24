#ifndef __DG_NETWORK_UNIFIED_FILEIO_H__
#define __DG_NETWORK_UNIFIED_FILEIO_H__

#include "network_fileio.h"
#include "network_hash.h"
#include <filesystem>
#include "network_exception.h"
#include "network_atomic_x.h"

namespace dg::network_persistent_unified_fileio{

    class UnifiedFsysControllerInterface{

        public:

            virtual ~UnifiedFsysControllerInterface() noexcept = default;
            virtual auto exists(const char * unified_fp) const noexcept -> bool = 0;
            virtual auto replication_size(const char * unified_fp) const noexcept -> size_t = 0; 
            virtual auto replication_fpath(const char * unified_fp, size_t idx) const noexcept -> const std::filesystem::path& = 0; //this is equivalent to (-> const char *) - in the sense of storing potential invalid ptr - yet better - force user to do const auto& to store const reference
            virtual auto replication_is_valid(const char * unified_fp, size_t idx) const noexcept -> bool = 0;
            virtual void replication_invalidate(const char * unified_fp, size_t idx) noexcept = 0; 
            virtual void replication_validate(const char * unified_fp, size_t idx) noexcept = 0;
    };

    class UnifiedFsysController: public virtual UnifiedFsysControllerInterface{

        private:

            std::unordered_map<std::string, std::vector<std::filesystem::path>> unified_fsys_map;
            std::unordered_map<std::string, std::vector<bool>> unified_fsys_flag_map;

        public:

            UnifiedFsysController(std::unordered_map<std::string, std::vector<std::filesystem::path>> unified_fsys_map,
                                  std::unordered_map<std::string, std::vector<bool>> unified_fsys_flag_map) noexcept: unified_fsys_map(std::move(unified_fsys_map)),
                                                                                                                      unified_fsys_flag_map(std::move(unified_fsys_flag_map)){}
            
            auto exists(const char * unified_fp) const noexcept -> bool{

                return this->unified_fsys_map.find(unified_fp) != this->unified_fsys_map.end();
            }
            
            auto replication_size(const char * unified_fp) const noexcept -> size_t{

                auto map_ptr = this->unified_fsys_map.find(unified_fp);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->unified_fsys_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return map_ptr->second.size();
            }

            auto replication_fpath(const char * unified_fp, size_t idx) const noexcept -> const std::filesystem::path&{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->replication_size(unified_fp)){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return this->unified_fsys_map.find(unified_fp)->second[idx];
            }

            auto replication_is_valid(const char * unified_fp, size_t idx) const noexcept -> bool{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->replication_size(unified_fp)){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return this->unified_fsys_flag_map.find(unified_fp)->second[idx];
            }

            void replication_invalidate(const char * unified_fp, size_t idx) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->replication_size(unified_fp)){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->unified_fsys_flag_map.find(unified_fp)->second[idx] = false;
            }

            void replication_validate(const char * unified_fp, size_t idx) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->replication_size(unified_fp)){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->unified_fsys_flag_map.find(unified_fp)->second[idx] = true;
            }
    };

    inline std::unique_ptr<UnifiedFsysControllerInterface> controller{};

    //WLOG
    //100GB for each f(x) -> x neural network 
    //estimated compression rate = 1% on all dataset
    //does a 10^4 networks like that

    //let's build a filesystem
    //precisely this
    //the largest distributed filesystem the world has ever seen by using the new compression tech
    //the idea is simple
    //dictionary - middle layers - return true | false for compressible status - if true, route to compressed storage - if false - route to stable storage

    //interface:

    //unstable interface
    //read() - return true if the operation is completed, does not guarantee the integrity of the underlying content
    //write() - return true if the operation is completed, does not guarantee the integrity of the underlying content
    
    //stable interface - inherit from unstable interface
    //read() - return true if the operation is a true reverse operation of the last write push such that g(f(x)) = x - where x is the write's buffer arg
    //write() - return true if the operation is completed, does not guarantee the integrity of the underlying content
    //implementation - use checksum + runlength + timestamp and friends

    //stable noexcept interface - this 
    //ETA: 1 month - sharp
    
    //what does that mean for the future of data-warehousing (the very base of database)?
    //what does that mean for the future of intra-planet communication? 
    //what does that mean for the future of cloud computing (do you actually need that many physical idling servers? - or only a certain uptime servers - or if techno is advanced enough - can users actually work on the virtual buffer without any drawbacks?) 

    //all datalake is stored in the DogeGraph filesystem
    //each transaction is at least 10GB/ operation
    //each transaction invokes an ingestion request from the actual database - if the data is not already in the processing system (this)

    //imagine a system consists of:
    //atomic_buffers + storage engines
    //ingestion accelerators | ingestion gate (guarantee that transaction - one or many commits - is atomic - this is the sole interface to interact with atomic_buffers + storage engines)
    //core - this
    //allocators + decompression plan maker (provide a high level decompression plan + lifetime of decompression (to avoid overhead of moving data from/to storage engine + decompression time - this oversteps into the responsibility of cache))
    //user_cache_system
    //enduser


    void init(std::filesystem::path * main_filepath, std::filesystem::path * replica_filepath, size_t * n, size_t replication_sz){

    }

    auto dg_file_exists(const char * unified_fp) noexcept -> std::expected<bool, exception_t>{

        return controller->exists(unified_fp);
    }

    auto dg_file_exists_nothrow(const char * unified_fp) noexcept -> bool{

        return dg::network_exception_handler::nothrow_log(dg_file_exists(unified_fp));
    }

    auto dg_file_size(const char * unified_fp) noexcept -> std::expected<size_t, exception_t>{
        
        if (!controller->exists(unified_fp)){
            return std::unexpected(dg::network_exception::FILE_NOT_FOUND);
        }
        
        size_t replication_sz = controller->replication_size(unified_fp);

        for (size_t i = 0u; i < replication_sz; ++i){
            if (controller->replication_is_valid(unified_fp, i)){
                const char * fp = controller->replication_fpath(unified_fp, i).c_str();
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

        if (!controller->exists(unified_fp)){
            return dg::network_exception::FILE_NOT_FOUND;
        }

        size_t replication_sz = controller->replication_size(unified_fp); 

        for (size_t i = 0u; i < replication_sz; ++i){
            if (controller->replication_is_valid(unified_fp, i)){
                const char * fp = controller->replication_fpath(unified_fp, i).c_str();
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

        if (!controller->exists(unified_fp)){
            return dg::network_exception::FILE_NOT_FOUND;
        }

        size_t replication_sz = controller->replication_size(unified_fp);

        for (size_t i = 0u; i < replication_sz; ++i){
            if (controller->replication_is_valid(unified_fp, i)){
                const char * fp = controller->replication_fpath(unified_fp, i).c_str();
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

    //this function - if returns error - leaves the underlying content in an undefined state and set all valid_flags -> false
    //              - if returns SUCCESS - at least one of the replications is written - set valid_flags accordingly 

    auto dg_write_binary_direct(const char * unified_fp, const void * src, size_t src_sz) noexcept -> exception_t{
        
        if (!controller->exists(unified_fp)){
            return dg::network_exception::FILE_NOT_FOUND;
        }

        size_t replication_sz   = controller->replication_size(unified_fp);
        auto exception_vec      = std::vector<exception_t>(); 

        for (size_t i = 0u; i < replication_sz; ++i){
            const char * fp = controller->replication_fpath(unified_fp, i).c_str();
            exception_t err = dg::network_fileio::dg_write_binary_direct(fp, src, src_sz);
            exception_vec.push_back(err);
        }

        if (std::find(exception_vec.begin(), exception_vec.end(), dg::network_exception::SUCCESS) != exception_vec.end()){
            for (size_t i = 0u; i < replication_sz; ++i){
                if (exception_vec[i] == dg::network_exception::SUCCESS){
                    controller->replication_validate(unified_fp, i);
                } else{
                    controller->replication_invalidate(unified_fp, i);
                }
            }

            return dg::network_exception::SUCCESS;
        }

        for (size_t i = 0u; i < replication_sz; ++i){
            controller->replication_invalidate(unified_fp, i);
        }

        return dg::network_exception::RUNTIME_FILEIO_ERROR; //promote error code - runtime_fileio_error
    }

    void dg_write_binary_direct_nothrow(const char * unified_fp, const void * src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary_direct(unified_fp, src, src_sz));
    }

    auto dg_write_binary_indirect(const char * unified_fp, const void * src, size_t src_sz) noexcept -> exception_t{

        if (!controller->exists(unified_fp)){
            return dg::network_exception::FILE_NOT_FOUND;
        }

        size_t replication_sz   = controller->replication_size(unified_fp);
        auto exception_vec      = std::vector<exception_t>();

        for (size_t i = 0u; i < replication_sz; ++i){
            const char * fp = controller->replication_fpath(unified_fp, i).c_str();
            exception_t err = dg::network_fileio::dg_write_binary_indirect(fp, src, sz);
            exception_vec.push_back(err);
        }

        if (std::find(exception_vec.begin(), exception_vec.end(), dg::network_exception::SUCCESS) != exception_vec.end()){
            for (size_t i = 0u; i < replication_sz; ++i){
                if (exception_vec[i] == dg::network_exception::SUCCESS){
                    controller->replication_validate(unified_fp, i);
                } else{
                    controller->replication_invalidate(unified_fp, i);
                }
            }

            return dg::network_exception::SUCCESS;
        }
        
        for (size_t i = 0u; i < replication_sz; ++i){
            controller->replication_invalidate(unified_fp, i);
        }

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