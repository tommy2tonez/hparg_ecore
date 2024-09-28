#ifndef __NETWORK_CUFSIO_LINUX_H__
#define __NETWORK_CUFSIO_LINUX_H__

#include <memory>
#include <cuda_runtime.h>
#include "cufile.h"
#include "network_fileio_linux.h"
#include <unordered_set>
#include "network_utility.h"
#include <mutex>
#include <thread>
#include "network_std_container.h"

namespace dg::network_cufsio_linux::constants{

    static inline constexpr std::optional<size_t> DG_CUFS_POLL_THRESHOLD_SIZE       = std::nullopt;
    static inline constexpr std::optional<size_t> DG_CUFS_MAX_DIRECT_IO_SIZE        = std::nullopt;
    static inline constexpr std::optional<size_t> DG_CUFS_MAX_CACHE_SIZE            = std::nullopt;
    static inline constexpr std::optional<size_t> DG_CUFS_MAX_PINNED_MEMORY_SIZE    = std::nullopt;
    static inline constexpr size_t DG_CUDIRECT_LEAST_POW2_BLK_SZ                    = size_t{1} << 24;
    static inline constexpr size_t DG_CUDIRECT_LEAST_POW2_ALIGNMENT_SZ              = size_t{1} << 24;
    static inline constexpr auto DG_CUFILE_HANDLE_OPTION                            = CU_FILE_HANDLE_TYPE_OPAQUE_FD;  
} 

namespace dg::network_cufsio_linux::utility{

    constexpr auto is_met_cudadirect_dgio_blksz_requirement(size_t sz) noexcept -> bool{

        return sz % constants::DG_CUDIRECT_LEAST_POW2_BLK_SZ == 0u;
    }

    constexpr auto is_met_cudadirect_dgio_ptralignment_requirement(uintptr_t ptr) noexcept -> bool{

        return ptr % constants::DG_CUDIRECT_LEAST_POW2_ALIGNMENT_SZ = 0u;
    }
}

namespace dg::network_cufsio_linux::driver_x{

    struct ObjectInterface{
        virtual ~ObjectInterface() noexcept = default;
    };

    template <class T>
    struct Object: ObjectInterface{
        static_assert(std::is_nothrow_destructible_v<T>);
        T obj; 
    };

    struct CUFSDriverResource{
        size_t reference;
        std::unordered_map<int, std::vector<std::unique_ptr<ObjectInterface>>> rtti_resource; //this is important - acquire all resource then do self-managed allocation in another component if necessary 
        std::vector<int> fd_vec;
        std::mutex mtx;
    };

    inline CUFSDriverResource cufs_driver_resource{}; 

    void dg_cufs_legacy_driver_close() noexcept{

        exception_t err = dg::network_exception::wrap_cuda_exception(cuFileDriverClose());
        
        if (dg::network_exception::is_failed(err)){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(err));
            std::abort();
        }
    }

    auto dg_cufs_legacy_driver_open() noexcept -> exception_t{

        using namespace dg::network_cufsio_linux::constants; 
        
        //TODOs: internalize cuda component exception -> cuda global exception 

        exception_t err = dg::network_exception::wrap_cuda_exception(cuFileDriverOpen());

        if (dg::network_exception::is_failed(err)){
            return err;
        }
        
        auto driver_grd = dg::network_genult::resource_guard(dg_cufs_legacy_driver_close);

        if (static_cast<bool>(DG_CUFS_POLL_THRESHOLD_SIZE)){
            err = dg::network_exception::wrap_cuda_exception(cuFileDriverSetPollMode(true, DG_CUFS_POLL_THRESHOLD_SIZE.value()));

            if (dg::network_exception::is_failed(err)){        
                return err;
            }
        }

        if (static_cast<bool>(DG_CUFS_MAX_DIRECT_IO_SIZE)){
            err = dg::network_exception::wrap_cuda_exception(cuFileDriverSetMaxDirectIOSize(DG_CUFS_MAX_DIRECT_IO_SIZE.value()));

            if (dg::network_exception::is_failed(err)){
                return err;
            }
        }

        if (static_cast<bool>(DG_CUFS_MAX_CACHE_SIZE)){
            err = dg::network_exception::wrap_cuda_exception(cuFileDriverSetMaxCacheSize(DG_CUFS_MAX_CACHE_SIZE.value()));

            if (dg::network_exception::is_failed(err)){
                return err;
            }
        }

        if (static_cast<bool>(DG_CUFS_MAX_PINNED_MEMORY_SIZE)){
            err = dg::network_exception::wrap_cuda_exception(cuFileDriverSetMaxPinnedMemSize(DG_CUFS_MAX_PINNED_MEMORY_SIZE.value()));

            if (dg::network_exception::is_failed(err)){
                return err;
            }
        }

        driver_grd.release();
        return dg::network_exception::SUCCESS;
    }

    auto dg_cufs_driver_open() noexcept -> std::expected<int, exception_t>{

        auto lck_grd = dg::network_genult::lock_guard(cufs_driver_resource.mtx);

        if (cufs_driver_resource.reference == 0u){
            exception_t err = dg_cufs_legacy_driver_open(); 

            if (dg::network_exception::is_failed(err)){
                return std::unexpected(err);
            }
        }

        cufs_driver_resource.reference += 1;

        if constexpr(DEBUG_FLAG_MODE){
            if (cufs_driver_resource.fd_vec.size() == 0u){
                dg::network_exception_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        int fd = cufs_driver_resource.fd_vec.back();
        cufs_driver_resource.fd_vec.pop_back();
        cufs_driver_resource.rtti_resource[fd] = {};

        return fd;
    }

    void dg_cufs_driver_close(int fd) noexcept{

        auto lck_grd    = dg::network_genult::lock_guard(cufs_driver_resource.mtx);
        auto rm_ptr     = cufs_driver_resource.rtti_resource.find(fd);

        if constexpr(DEBUG_FLAG_MODE){
            if (rm_ptr == cufs_driver_resource.rtti_resource.end()){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }

            if (cufs_driver_resource.reference == 0u){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        cufs_driver_resource.rtti_resource.erase(rm_ptr);
        cufs_driver_resource.fd_vec.push_back(fd);
        cufs_driver_resource.reference -= 1;

        if (cufs_driver_resource.reference == 0u){
            dg_cufs_legacy_driver_close();
        }
    }

    template <class T>
    void dg_cufs_driver_register_resource(int fd, T resource) noexcept{
 
        static_assert(std::is_nothrow_move_constructible_v<T>);

        auto lck_grd    = dg::network_genult::lock_guard(cufs_driver_resource.mtx); 
        auto raii_obj   = std::make_unique<Object<T>>(Object<T>{std::move(resource)}); //TODOs: either compromise log at make_unique or macro resolution
        auto dict_ptr   = cufs_driver_resource.rtti_resource.find(fd);

        if constexpr(DEBUG_MODE_FLAG){
            if (dict_ptr == cufs_driver_resource.rtti_resource.end()){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        dict_ptr->second.push_back(std::move(raii_obj));
    }

    auto dg_cufs_driver_dynamic_open() noexcept -> std::expected<int *, exception_t>{

        std::expected<int, exception_t> efd = dg_cufs_driver_open(); 
        
        if (!efd.has_value()){
            return std::unexpected(efd.error());
        }

        return new int(efd.value());  //TODOs: either compromise log at new or macro resolution
    }

    void dg_cufs_driver_dynamic_close(int * fd) noexcept{

        dg_cufs_driver_close(*dg::network_genult::safe_ptr_access(fd));
        delete fd;
    }

    using dg_cufs_driver_dynamic_close_t = void (*)(int *) noexcept;

    auto dg_cufs_driver_safe_open() noexcept -> std::expected<std::unique_ptr<int, dg_cufs_driver_dynamic_close_t>, exception_t>{

        std::expected<int *, exception_t> edfd = dg_cufs_driver_dynamic_open();

        if (!edfd.has_value()){
            return std::unexpected(edfd.error());
        } 

        return {std::in_place_t{}, edfd.value(), dg_cufs_driver_dynamic_close};
    }
} 

namespace dg::network_cufsio_linux::cufs_sptr_controller{

    //TODO
    auto internal_make_from_cudasptr(cuda_ptr_t ptr, size_t sz) noexcept -> cufs_sptr_t{

    }

    auto internal_make_from_hostsptr(cuda_ptr_t ptr, size_t sz) noexcept -> cufs_sptr_t{

    }

    auto get_size(cufs_sptr_t) noexcept -> size_t{

    }

    auto get_cufs_legacy_ptr(cufs_sptr_t ptr) noexcept -> cufs_legacy_ptr_t{

    } 

    auto register_cudasptr(cuda_ptr_t ptr, size_t sz) noexcept -> std::expected<cufs_sptr_t, exception_t>{

        cufs_legacy_ptr_t gptr  = pointer_cast<cufs_legacy_ptr_t>(ptr); 
        exception_t err         = dg::network_exception::wrap_cuda_exception(cuFileBufRegister(gptr, sz, 0));

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return internal_make_from_cudasptr(ptr, sz);
    }
    
    auto register_hostsptr(void * ptr, size_t sz) noexcept -> std::expected<cufs_sptr_t, exception_t>{

        cufs_legacy_ptr_t gptr  = pointer_cast<cufs_legacy_ptr_t>(ptr);
        exception_t err         = dg::network_exception::wrap_cuda_exception(cuFileBufRegister(gptr, sz, 0));

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        } 

        return internal_make_from_hostsptr(ptr, sz);
    } 

    void deregister(cufs_sptr_t ptr) noexcept{

        cufs_legacy_ptr_t gptr  = get_cufs_legacy_ptr(ptr);
        exception_t err         = dg::network_exception::wrap_cuda_exception(cuFileBufDeregister(gptr));

        if constexpr(DEBUG_MODE_FLAG){
            if (dg::network_exception::is_failed(err)){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(err));
                std::abort();
            }
        }
    }

    auto dynamic_register_cudasptr(cuda_ptr_t ptr, size_t sz) noexcept -> std::expected<cufs_sptr_t *, exception_t>{

        std::expected<cufs_sptr_t, exception_t> ecufs_ptr = register_cudasptr(ptr, sz);

        if (!ecufs_ptr.has_value()){
            return std::unexpected(ecufs_ptr.error());
        }

        return new cufs_sptr_t(ecufs_ptr.value());
    }

    auto dynamic_register_hostsptr(void * ptr, size_t sz) noexcept -> std::expected<cufs_sptr_t *, exception_t>{

        std::expected<cufs_sptr_t, exception_t> ecufs_ptr = register_hostsptr(ptr, sz);

        if (!ecufs_ptr.has_value()){
            return std::unexpected(ecufs_ptr.error());
        }

        return new cufs_sptr_t(ecufs_ptr.value());
    } 

    void dynamic_deregister(cufs_sptr_t * ptr) noexcept{

        deregister(*ptr);
        delete ptr;
    }
    
    using dynamic_deregister_t = void (*) (cufs_sptr_t *) noexcept;

    auto safe_register_cudasptr(cuda_ptr_t ptr, size_t sz) noexcept -> std::expected<std::unique_ptr<cufs_sptr_t, dynamic_deregister_t>, exception_t>{

        std::expected<cufs_sptr_t *, exception_t> edcufs_ptr = dynamic_register_cudasptr(ptr, sz); 

        if (!edcufs_ptr.has_value()){
            return std::unexpected(edcufs_ptr.error());
        }

        return {std::in_place_t{}, edcufs_ptr.value(), dynamic_deregister};
    }

    auto safe_register_hostsptr(cuda_ptr_t ptr, size_t sz) noexcept -> std::expected<std::unique_ptr<cufs_sptr_t, dynamic_deregister_t>, exception_t>{

        std::expected<cufs_sptr_t *, exception_t> edcufs_ptr = dynamic_register_hostsptr(ptr, sz);

        if (!edcufs_ptr.has_value()){
            return std::unexpected(edcufs_ptr.error());
        }

        return {std::in_place_t{}, edcufs_ptr.value(), dynamic_deregister};
    }
}

namespace dg::network_cufsio_linux::cufs_io{

    struct CudaFileDescriptor{
        dg::network_genult::nothrow_immutable_unique_raii_wrapper<int, dg::network_fileio_linux::kernel_fclose_t> kernel_raii_fd;
        CUfileHandle_t cf_handle;
    };

    using cuda_fclose_t = void (*)(CudaFileDescriptor *) noexcept; 

    auto dg_cuopen_file(const char * path, int flag) noexcept -> std::expected<std::unique_ptr<CudaFileDescriptor, cuda_fclose_t>, exception_t>{

        auto kfd = dg::network_fileio_linux::dg_open_file(path, flag);
        
        if (!kfd.has_value()){
            return std::unexpected(kfd.error());
        }

        auto destructor = [](CudaFileDescriptor * cu_fd) noexcept{
            cuFileHandleDeregister(cu_fd->cf_handle);
            delete cu_fd;
        };
    
        CUfileDescr_t cf_descr{};
        CUfileHandle_t cf_handle{};

        cf_descr.handle.fd  = kfd.value();
        cf_descr.type       = constants::DG_CUFILE_HANDLE_OPTION;
        exception_t err     = dg::network_exception::wrap_cuda_exception(cuFileHandleRegister(&cf_handle, &cf_descr)); 
        
        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return {std::in_place_t{}, new CudaFileDescriptor{std::move(kfd.value()), cf_handle}, destructor};
    }

    auto dg_curead_file(CudaFileDescriptor& fd, cuda_legacy_ptr_t dst, size_t sz, size_t file_off, size_t dst_off) noexcept -> exception_t{

        auto err = cuFileRead(fd.cf_handle, dst, dg::network_genult::wrap_safe_integer_cast(sz), dg::network_genult::wrap_safe_integer_cast(file_off), dg::network_genult::wrap_safe_integer_cast(dst_off)); 

        if (err != sz){
            if (err < 0){
                if (err == -1){
                    return dg::network_exception::wrap_kernel_exception(errno);
                }
                return dg::network_exception::wrap_cuda_exception(-err);
            }

            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        return dg::network_exception::SUCCESS; 
    }

    auto dg_cuwrite_file(CudaFileDescriptor& fd, cuda_legacy_ptr_t src, size_t sz, size_t file_off, size_t src_off) noexcept -> exception_t{ //TODO: add constness

        auto err = cuFileWrite(fd.cf_handle, src, dg::network_genult::wrap_safe_integer_cast(sz), dg::network_genult::wrap_safe_integer_cast(file_off), dg::network_genult::wrap_safe_integer_cast(src_off));

        if (err != sz){
            if (err < 0){
                if (err == -1){
                    return dg::network_exception::wrap_kernel_exception(errno);
                }
                return dg::network_exception::wrap_cuda_exception(-err);
            }

            return dg::network_exception::RUNTIME_FILEIO_ERROR;
        }

        return dg::network_exception::SUCCESS;
    }
} 

namespace dg::network_cufsio_linux::implementation{
    
    //functionality-wise - fine 
    //naming-wise - not fine

    class CuFSIOInterface{

        public:

            virtual ~CuFSIOInterface() noexcept = default;
            virtual auto read_binary(const char *, cufs_sptr_t) noexcept -> exception_t = 0;
            virtual void write_binary(const char *, cufs_sptr_t) noexcept -> exception_t = 0; 
    };

    class FsysIOInterface{

        public:

            virtual ~FsysIOInterface() noexcept = default;
            virtual auto host_read_binary(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t = 0; //should throw if overcap - this is important in scenerio where fsz is buffer header - assume valid dst, dst_cap, invalid fp
            virtual auto host_write_binary(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t = 0;
            virtual auto cuda_read_binary(const char * fp, cuda_ptr_t  dst, size_t dst_cap) noexcept -> exception_t = 0;
            virtual auto cuda_write_binary(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept -> exception_t = 0; //constness here will be added in the future - to not over-complicate
    };

    class DirectPreAllocatedStableCuFSIO: public virtual CuFSIOInterface{

        private:
            
            std::unique_ptr<int, driver_x::dg_cufs_driver_dynamic_close_t> cufs_driver_fd; //fine - driver fd is unique - a shared_resource == a fork of current fd - also unique_ptr<>
            std::unordered_set<cufs_sptr_t> registered_hashset; 

        public:

            DirectPreAllocatedStableCuFSIO(std::unique_ptr<int, driver_x::dg_cufs_driver_dynamic_close_t> cufs_driver_fd,
                                           std::unordered_set<cufs_sptr_t> registered_hashset) noexcept: cufs_driver_fd(std::move(cufs_driver_fd)),
                                                                                                         registered_hashset(std::move(registered_hashset)){}
            
            auto read_binary(const char * fp, cufs_sptr_t dst) noexcept -> exception_t{

                if (!this->is_registered(dst)){
                    return dg::network_exception::UNREGISTERED_CUFILE_PTR;
                }

                auto raii_fd = cufs_io::dg_cuopen_file(fp, O_RDONLY | O_DIRECT | O_TRUNC);

                if (!raii_fd.has_value()){
                    return raii_fd.error();
                } 

                int kernel_fd                   = raii_fd.value()->kernel_raii_fd;
                size_t fsz                      = dg::network_fileio_linux::dg_file_size_nothrow(kernel_fd);
                size_t dst_cap                  = cufs_sptr_controller::get_size(dst);
                cufs_legacy_ptr_t legacy_dst    = cufs_sptr_controller::get_cufs_legacy_ptr(dst);

                if (dst_cap < fsz){
                    return dg::network_exception::BUFFER_OVERFLOW;
                }

                if (!utility::is_met_cudadirect_dgio_blksz_requirement(fsz)){
                    return dg::network_exception::BAD_ALIGNMENT;
                }

                if constexpr(DEBUG_MODE_FLAG){
                    if (!utility::is_met_cudadirect_dgio_ptralignment_requirement(pointer_cast<uintptr_t>(legacy_dst))){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return cufs_io::dg_curead_file(*(raii_fd.value()), legacy_dst, fsz, 0u, 0u);
            }
        
            auto write_binary(const char * fp, cufs_sptr_t src) noexcept -> exception_t{

                if (!this->is_registered(src)){
                    return dg::network_exception::UNREGISTERED_CUFILE_PTR;
                }

                auto raii_fd = cufs_io::dg_cuopen_file(fp, O_WRONLY | O_DIRECT | O_TRUNC);

                if (!raii_fd.has_value()){
                    return raii_fd.error();
                }

                size_t src_sz = cufs_sptr_controller::get_size(src); 
                cufs_legacy_ptr_t legacy_src = cufs_sptr_controller::get_cufs_legacy_ptr(src);

                if constexpr(DEBUG_MODE_FLAG){
                    if (!utility::is_met_cudadirect_dgio_blksz_requirement(src_sz)){
                        dg::network_exception_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                    
                    if (!utility::is_met_cudadirect_dgio_ptralignment_requirement(pointer_cast<uintptr_t>(legacy_src))){
                        dg::network_exception_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }
                
                return cufs_io::dg_cuwrite_file(*(raii_fd.value()), legacy_src, src_sz, 0u, 0u);
            }
            
        private:

            auto is_registered(cufs_sptr_t key) const noexcept -> bool{

                return this->registered_hashset.find(key) != this->registered_hashset.end();
            }
    };

    class CuPreallocatedStableFsysIO: public virtual FsysIOInterface{

        private:

            std::unique_ptr<CuFSIOInterface> cu_fsio;
            std::unordered_map<std::pair<uintptr_t, size_t>, cufs_sptr_t> kernelptr_to_cufs_dict;
            std::unordered_map<std::pair<uintptr_t, size_t>, cufs_sptr_t> cudaptr_to_cufs_dict;
        
        public:

            CuPreallocatedStableFsysIO(std::unique_ptr<CuFSIOInterface> cu_fsio,
                                       std::unordered_map<std::pair<uintptr_t, size_t>, cufs_sptr_t> kernelptr_to_cufs_dict,
                                       std::unordered_map<std::pair<uintptr_t, size_t>, cufs_sptr_t> cudaptr_to_cufs_dict) noexcept: cu_fsio(std::move(cu_fsio)),
                                                                                                                                     kernelptr_to_cufs_dict(std::move(kernelptr_to_cufs_dict)),
                                                                                                                                     cudaptr_to_cufs_dict(std::move(cudaptr_to_cufs_dict)){}

            auto host_read_binary(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

                auto dict_ptr = this->kernelptr_to_cufs_dict.find(std::make_pair(pointer_cast<uintptr_t>(dst), dst_cap));

                if (dict_ptr == this->kernelptr_to_cufs_dict.end()){
                    return dg::network_exception::UNREGISTERED_CUFILE_PTR;
                }

                cufs_sptr_t cufs_sptr   = dict_ptr->second;
                exception_t err         = this->cu_fsio->read_binary(fp, cufs_sptr);

                if constexpr(DEBUG_MODE_FLAG){
                    if (err == dg::network_exception::UNREGISTERED_CUFILE_PTR){
                        dg::network_exception_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return err;
            }

            auto host_write_binary(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

                auto dict_ptr = this->kernelptr_to_cufs_dict.find(std::make_pair(pointer_cast<uintptr_t>(src), src_sz));

                if (dict_ptr == this->kernelptr_to_cufs_dict.end()){
                    return dg::network_exception::UNREGISTERED_CUFILE_PTR;
                }

                cufs_sptr_t cufs_sptr   = dict_ptr->second;
                exception_t err         = this->cu_fsio->write_binary(fp, cufs_sptr);

                if constexpr(DEBUG_MODE_FLAG){
                    if (err == dg::network_exception::UNREGISTERED_CUFILE_PTR){
                        dg::network_exception_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return err;
            }

            auto cuda_read_binary(const char * fp, cuda_ptr_t dst, size_t dst_cap) noexcept -> exception_t{

                auto dict_ptr = this->cudaptr_to_cufs_dict.find(std::make_pair(pointer_cast<uintptr_t>(dst), dst_cap));

                if (dict_ptr == this->cudaptr_to_cufs_dict.end()){
                    return dg::network_exception::UNREGISTERED_CUFILE_PTR;
                }

                cufs_sptr_t cufs_sptr   = dict_ptr->second;
                exception_t err         = this->cu_fsio->read_binary(fp, cufs_sptr);

                if constexpr(DEBUG_MODE_FLAG){
                    if (err == dg::network_exception::UNREGISTERED_CUFILE_PTR){
                        dg::network_exception_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return err;
            }

            auto cuda_write_binary(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept -> exception_t{

                auto dict_ptr = this->cudaptr_to_cufs_dict.find(std::make_pair(pointer_cast<uintptr_t>(src), src_sz));

                if (dict_ptr == this->cudaptr_to_cufs_dict.end()){
                    return dg::network_exception::UNREGISTERED_CUFILE_PTR;
                }

                cufs_sptr_t cufs_sptr   = dict_ptr->second;
                exception_t err         = this->cu_fsio->write_binary(fp, cufs_sptr);

                if constexpr(DEBUG_MODE_FLAG){
                    if (err == dg::network_exception::UNREGISTERED_CUFILE_PTR){
                        dg::network_exception_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                return err;
            }
    };

    class InDirectFsysIO: public virtual FsysIOInterface{

        public:

            auto host_read_binary(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

                return dg::network_fileio_linux::dg_read_binary(fp, dst, dst_cap);
            }

            auto host_write_binary(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

                return dg::network_fileio_linux::dg_write_binary(fp, src, src_sz);
            }

            auto cuda_read_binary(const char * fp, cuda_ptr_t dst, size_t dst_cap) noexcept -> exception_t{

                auto buf        = dg::network_std_container::string(dst_cap);
                exception_t err = dg::network_fileio_linux::dg_read_binary(fp, buf.data(), dst_cap); 

                if (dg::network_exception::is_failed(err)){
                    return err;
                }

                err = dg::network_exception::wrap_cuda_exception(cudaMemcpy(dst, buf.data(), dst_cap, cudaMemcpyHostToDevice)); //TODO: change -> cuda_controller 
                dg::network_exception_handler::nothrow_log(err);
            }

            auto cuda_write_binary(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept -> exception_t{
 
                auto buf        = dg::network_std_container::string(src_sz);
                exception_t err = dg::network_exception::wrap_cuda_exception(cudaMemcpy(buf.data(), src, src_sz, cudaMemcpyDeviceToHost)); //TODO: change -> cuda_controller
                dg::network_exception_handler::nothrow_log(err);

                return dg::network_fileio_linux::dg_write_binary(fp, buf.data(), src_sz);
            }
    };

    class StdFsysIO: public virtual FsysIOInterface{

        private:

            std::unique_ptr<FsysIOInterface> fast_fsysio;
            std::unique_ptr<FsysIOInterface> slow_fsysio;

        public:

            StdFsysIO(std::unique_ptr<FsysIOInterface> fast_fsysio, 
                      std::unique_ptr<FsysIOInterface> slow_fsysio) noexcept: fast_fsysio(std::move(fast_fsysio)),
                                                                              slow_fsysio(std::move(slow_fsysio)){}

            auto host_read_binary(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

                exception_t err = this->fast_fsysio->host_read_binary(fp, dst, dst_cap);

                if (dg::network_exception::is_success(err)){
                    return dg::network_exception::SUCCESS;
                } 

                return this->slow_fsysio->host_read_binary(fp, dst, dst_cap);
            }

            auto host_write_binary(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

                exception_t err = this->fast_fsysio->host_write_binary(fp, src, src_sz);

                if (dg::network_exception::is_success(err)){
                    return dg::network_exception::SUCCESS;
                }

                return this->slow_fsysio->host_write_binary(fp, src, src_sz);
            }

            auto cuda_read_binary(const char * fp, cuda_ptr_t dst, size_t dst_cap) noexcept -> exception_t{

                exception_t err = this->fast_fsysio->cuda_read_binary(fp, dst, dst_cap);

                if (dg::network_exception::is_success(err)){
                    return dg::network_exception::SUCCESS;
                }

                return this->slow_fsysio->cuda_read_binary(fp, dst, dst_cap);
            }

            auto cuda_write_binary(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept -> exception_t{

                exception_t err = this->fast_fsysio->cuda_write_binary(fp, src, src_sz);

                if (dg::network_exception::is_success(err)){
                    return dg::network_exception::SUCCESS;
                } 

                return this->slow_fsysio->cuda_write_binary(fp, src, src_sz);
            }
    };

    struct FsysIOFactory{

        static auto spawn_preallocated_direct_stable_fsysio(cuda_ptr_t * cuda_ptr, size_t * cuda_ptr_sz, size_t cuda_sz, void ** host_ptr, size_t * host_ptr_sz, size_t host_sz) -> std::unique_ptr<FsysIOInterface>{
            
            auto cudriver_ins       = driver_x::dg_cufs_driver_safe_open();
            auto cuda_to_cufs_dict  = std::unordered_map<std::pair<uintptr_t, size_t>, cufs_sptr_t>{};
            auto host_to_cufs_dict  = std::unordered_map<std::pair<uintptr_t, size_t>, cufs_sptr_t>{};
            auto cufs_sptr_hashset  = std::unordered_set<cufs_sptr_t>{};

            if (!cudriver_ins.has_value()){
                dg::network_exception::throw_exception(cudriver_ins.error());
            }

            int cudriver_fd = *(cudriver_ins.value());

            for (size_t i = 0u; i < cuda_sz; ++i){
                auto cufs_raii_sptr = cufs_sptr_controller::safe_register_cudasptr(cuda_ptr[i], cuda_ptr_sz[i]);
                if (!cufs_raii_sptr.has_value()){
                    dg::network_exception::throw_exception(cufs_raii_sptr.error());
                }
                
                cufs_sptr_t cufs_sptr = *(cufs_raii_sptr.value()); 
                cufs_sptr_hashset.insert(cufs_sptr);
                cuda_to_cufs_dict.insert(std::make_pair(std::make_pair(pointer_cast<uintptr_t>(cuda_ptr[i]), cuda_ptr_sz[i]), cufs_sptr));
                driver_x::dg_cufs_register_resource(cudriver_fd, std::move(cufs_raii_sptr.value()));
            }

            for (size_t i = 0u; i < host_sz; ++i){
                auto cufs_raii_sptr = cufs_sptr_controller::safe_register_hostsptr(host_ptr[i], host_ptr_sz[i]);
                if (!cufs_raii_sptr.has_value()){
                    dg::network_exception::throw_exception(cufs_raii_sptr.error());
                }

                cufs_sptr_t cufs_sptr = *(cufs_raii_sptr.value()); 
                cufs_sptr_hashset.insert(cufs_sptr);
                host_to_cufs_dict.insert(std::make_pair(std::make_pair(pointer_cast<uintptr_t>(host_ptr[i]), host_ptr_sz[i]), cufs_sptr));
                driver_x::dg_cufs_register_resource(cudriver_fd, std::move(cufs_raii_sptr.value()));
            }

            std::unique_ptr<CuFSIOInterface> cufs_io    = std::make_unique<DirectPreAllocatedStableCuFSIO>(std::move(cudriver_ins.value()), std::move(cufs_sptr_hashset));
            std::unique_ptr<FsysIOInterface> rs         = std::make_unique<CuPreallocatedStableFsysIO>(std::move(cufs_io), std::move(host_to_cufs_dict), std::move(cuda_to_cufs_dict));

            return rs;
        }

        static auto spawn_indirect_fsysio() -> std::unique_ptr<FsysIOInterface>{

            return std::make_unique<InDirectFsysIO>();
        }

        static auto spawn_std_fsysio(std::unique_ptr<FsysIOInterface> fast, std::unique_ptr<FsysIOInterface> slow) -> std::unique_ptr<FsysIOInterface>{

            return std::make_unique<StdFsysIO>(std::move(fast), std::move(slow));
        }

        static auto spawn_prereg_direct_or_default_fsysio(cuda_ptr_t * cuda_ptr, size_t * cuda_ptr_sz, size_t cuda_sz,
                                                          void ** host_ptr, size_t * host_ptr_sz, size_t host_sz) -> std::unique_ptr<FsysIOInterface>{
            
            auto zipped_cuda = dg::network_utility::ptrtup_zip(cuda_ptr, cuda_ptr_sz, cuda_sz); 
            auto zipped_host = dg::network_utility::ptrtup_zip(host_ptr, host_ptr_sz, host_sz);
            
            auto cuda_filter = [](std::tuple<cuda_ptr_t, size_t> e){
                return utility::is_met_cudadirect_dgio_ptralignment_requirement(pointer_cast<uintptr_t>(std::get<0>(e))) && utility::is_met_cudadirect_dgio_blksz_requirement(std::get<1>(e));
            };

            auto host_filter = [](std::tuple<void *, size_t> e){
                return utility::is_met_cudadirect_dgio_ptralignment_requirement(pointer_cast<uintptr_t>(std::get<0>(e))) && utility::is_met_cudadirect_dgio_blksz_requirement(std::get<1>(e));
            };

            auto cuda_last  = std::copy_if(zipped_cuda.begin(), zipped_cuda.end(), zipped_cuda.begin(), cuda_filter);
            auto host_last  = std::copy_if(zipped_host.begin(), zipped_host.end(), zipped_host.begin(), host_filter);

            try{
                auto direct_fsysio = spawn_preallocated_direct_stable_fsysio(cuda_ptr, cuda_ptr_sz, std::distance(zipped_cuda.begin(), cuda_last), 
                                                                             host_ptr, host_ptr_sz, std::distance(zipped_host.begin(), host_last)); 

                return spawn_std_fsysio(std::move(direct_fsysio), spawn_indirect_fsysio());
            } catch (...){
                return spawn_indirect_fsysio();
            }
        }
    };
} 

namespace dg::network_cufsio_linux{

    inline std::unique_ptr<FsysIOInterface> cufsio_instance{};

    void init(cuda_ptr_t * cuda_ptr, size_t * cuda_ptr_sz, size_t cuda_sz, void ** host_ptr, size_t * host_ptr_sz, size_t host_sz){
        
        cufsio_instance = implementation::Factory::spawn_prereg_direct_or_default_fsysio(cuda_ptr, cuda_ptr_sz, cuda_sz, host_ptr, host_ptr_sz, host_sz);
    }

    void deinit() noexcept{

        cufsio_instance = {};
    }

    auto dg_cuda_read_binary(const char * fp, cuda_ptr_t dst, size_t dst_cap) noexcept -> exception_t{

        return cufsio_instance->cuda_read_binary(fp, dst, dst_cap);
    }

    void dg_cuda_read_binary_nothrow(const char * fp, cuda_ptr_t dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_cuda_read_binary(fp, dst, dst_cap));
    }

    auto dg_cuda_write_binary(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept -> exception_t{

        return cufsio_instance->cuda_write_binary(fp, src, src_sz);
    } 

    void dg_cuda_write_binary_nothrow(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept{

        dg::network_exception_handler::nothrow_log(dg_cuda_write_binary(fp, src, src_sz));
    }

    auto dg_host_read_binary(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t{

        return cufsio_instance->host_read_binary(fp, dst, dst_cap);
    } 

    void dg_host_read_binary_nothrow(const char * fp, void * dst, size_t dst_cap) noexcept{

        dg::network_exception_handler::nothrow_log(dg_host_read_binary(fp, dst, dst_cap));
    } 

    auto dg_host_write_binary(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t{

        return cufsio_instance->host_write_binary(fp, src, src_sz);
    }

    void dg_host_write_binary_nothrow(const char * fp, const void * src, size_t src_sz){

        dg::network_exception_handler::nothrow_log(dg_host_write_binary(fp, src, src_sz));
    }

    //extend this component to do replicas + chksum - tmr - this should be macro polymorphism - for network_fileio_linux - such that either an application uses this component xor another component for base file writing - chksum extends the base - replica extends chksum
}

#endif