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

    //replica is, like concurrency, a very hard problem to define
    //where to compromise the component requires experts' opinion + specific use cases
    //this component is rather a feature than a neccessity

    //this does empty initialization + snaps unified fsys controller to a defined state
    //every external access to the physical files is undefined behavior and not the responsibility of this component - maybe another component inherits this component
    //this component assumes that f(g(x)) = x, for f is read, g is write operation
    //the contract is either guaranteed by the kernel or the network_fileio's internal mechanisms (checksum, timestamp and friends) - not this component responsibility

    void init(std::string * virtual_filepath, std::filesystem::path * physical_filepath, size_t * n, size_t replication_sz){

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

    auto dg_write_binary(const char * unfiied_fp, const void * src, size_t src_sz) noexcept -> exception_t{

        exception_t err = dg_write_binary_direct(unfiied_fp, src, src_sz); 

        if (err == dg::network_exception::SUCCESS){
            return dg::network_exception::SUCCESS;
        }

        return dg_write_binary_indirect(unfiied_fp, src, src_sz);
    }

    void dg_write_binary_nothrow(const char * unfiied_fp, const void * src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_write_binary(unfiied_fp, src, src_sz));
    }
}

#endif