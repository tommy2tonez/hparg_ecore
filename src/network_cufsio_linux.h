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

namespace dg::network_cufsio_linux::driver_x{

    using dg_cufs_dynamic_deregister_t = void (*)(cufs_sptr_t *) noexcept; 

    struct ObjectInterface{
        virtual ~ObjectInterface() noexcept = default;
    };

    template <class T>
    struct Object: ObjectInterface{
        T obj; 
    };

    struct CUFSDriverResource{
        size_t reference;
        std::unordered_map<int, std::vector<std::unique_ptr<ObjectInterface>>> rtti_resource; //this is important - acquire all resource then do self-managed allocation in another component if necessary 
        std::vector<int> fd_vec;
        std::mutex mtx;
    };

    inline CUFSDriverResource cufs_driver_resource{}; 

    auto dg_cufs_driver_open() noexcept -> std::expected<int, exception_t>{

        std::lock_guard<std::mutex> lck_grd(cufs_driver_resource.mtx);

        if (cufs_driver_resource.reference == 0u){
            exception_t err = dg::network_exception::wrap_cuda_exception(cuFileDriverOpen());

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

        return fd;
    }

    void dg_cufs_driver_close(int fd) noexcept{

        std::lock_guard<std::mutex> lck_grd(cufs_driver_resource.mtx);
        auto rm_ptr = cufs_driver_resource.rtti_resource.find(fd);

        if constexpr(DEBUG_FLAG_MODE){
            if (rm_ptr == cufs_driver_resource.rtti_resource.end()){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        cufs_driver_resource.rtti_resource.erase(rm_ptr);
        cufs_driver_resource.fd_vec.push_back(fd);
        cufs_driver_resource.reference -= 1;

        if (cufs_driver_resource.reference == 0u){
            exception_t err = dg::network_exception::wrap_cuda_exception(cuFileDriverClose());

            if constexpr(DEBUG_MODE_FLAG){
                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(err));
                    std::abort();
                }
            }
        }
    }

    template <class T>
    void dg_cufs_driver_register_resource(int fd, T resource) noexcept{

        static_assert(std::is_nothrow_destructible_v<T>);
        static_assert(std::is_nothrow_move_constructible_v<T>);

        std::lock_guard<std::mutex> lck_grd(cufs_driver_resource.mtx); 
        std::unique_ptr<ObjectInterface> obj = std::make_unique<Object<T>>(Object<T>{std::move(resource)});
        auto dict_ptr = cufs_driver_resource.rtti_resource.find(fd);

        if constexpr(DEBUG_MODE_FLAG){
            if (dict_ptr == cufs_driver_resource.rtti_resource.end()){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            }
        }

        dict_ptr->second.push_back(std::move(obj));
    }

    auto dg_cufs_driver_dynamic_open() noexcept -> std::expected<int *, exception_t>{

        std::expected<int, exception_t> efd = dg_cufs_driver_open(); 
        
        if (!efd.has_value()){
            return efd.error();
        }

        return new int(efd.value());
    }

    void dg_cufs_driver_dynamic_close(int * fd) noexcept{

        dg_cufs_driver_close(*dg::network_genult::safe_ptr_access(fd));
        delete fd;
    }

    using dg_cufs_driver_dynamic_close_t = void (*)(int *) noexcept;

    auto dg_cufs_driver_safe_open() noexcept -> std::expected<std::unique_ptr<int, dg_cufs_driver_dynamic_close_t>, exception_t>{

        auto edfd = dg_cufs_driver_dynamic_open();

        if (!edfd.has_value()){
            return std::unexpected(edfd.error());
        } 

        return {std::in_place_t{}, edfd.value(), dg_cufs_driver_dynamic_close};
    }
} 

namespace dg::network_cufsio_linux::cufs_sptr_controller{

    auto get_size(cufs_sptr_t) noexcept -> size_t{

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

        cufs_legacy_ptr_t gptr  = pointer_cast<cufs_legacy_ptr_t>(ptr);
        exception_t err         = dg::network_exception::wrap_cuda_exception(cuFileBufDeregister(gptr));

        if constexpr(DEBUG_MODE_FLAG){
            if (dg::network_exception::is_failed(err)){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(err));
                std::abort();
            }
        }
    }

    //^^^
    //fine - fix later - move into safe_register - no use for these

    auto dynamic_register_cudasptr(cuda_ptr_t ptr, size_t sz) noexcept -> std::expected<cufs_sptr_t *, exception_t>{

        auto ecufs_ptr = register_cudasptr(ptr, sz);

        if (!ecufs_ptr.has_value()){
            return ecufs_ptr.error();
        }

        return new cufs_sptr_t(ecufs_ptr.value());
    }

    auto dynamic_register_hostsptr(void * ptr, size_t sz) noexcept -> std::expected<cufs_sptr_t *, exception_t>{

        auto ecufs_ptr = register_hostsptr(ptr, sz);

        if (!ecufs_ptr.has_value()){
            return ecufs_ptr.error();
        }

        return new cufs_sptr_t(ecufs_ptr.value());
    } 

    void dynamic_deregister(cufs_sptr_t * ptr) noexcept{

        dg_cufs_deregister(*ptr);
        delete ptr;
    }
    
    //vvv

    using dynamic_deregister_t = void (*) (cufs_sptr_t *) noexcept;

    //unique_ptr<> is just a mean to everything raii - not necessarily means that the underlying type has to be the base_type

    auto safe_register_cudasptr(cuda_ptr_t ptr, size_t sz) noexcept -> std::expected<std::unique_ptr<cufs_sptr_t, dynamic_deregister_t>, exception_t>{

        auto edcufs_ptr = dynamic_register_cudasptr(ptr, sz); 

        if (!edcufs_ptr.has_value()){
            return std::unexpected(edcufs_ptr.error());
        }

        return {std::in_place_t{}, edcufs_ptr.value(), dynamic_deregister};
    }

    auto safe_register_hostsptr(cuda_ptr_t ptr, size_t sz) noexcept -> std::expected<std::unique_ptr<cufs_sptr_t, dynamic_deregister_t>, exception_t>{

        auto edcufs_ptr = dynamic_register_hostsptr(ptr, sz);

        if (!edcufs_ptr.has_value()){
            return std::unexpected(edcufs_ptr.error());
        }

        return {std::in_place_t{}, edcufs_ptr.value(), dynamic_deregister};
    }
} 

namespace dg::network_cufsio_linux{

    //--refactor

    struct CudaFileDescriptor{
        dg::network_genult::nothrow_immutable_unique_raii_wrapper<int, dg::network_fileio_linux::kernel_fclose_t> kernel_raii_fd;
        CUfileHandle_t cf_handle;
    };

    using cuda_fclose_t = void (*)(CudaFileDescriptor *) noexcept; 

    static inline constexpr size_t DG_CUDIRECT_BLK_SZ       = size_t{1} << 13;
    static inline constexpr auto DG_CU_FILE_HANDLE_OPTION   = CU_FILE_HANDLE_TYPE_OPAQUE_FD;  
    
    constexpr auto is_met_cudadirect_dgio_blksz_requirement(size_t sz) noexcept -> bool{

        return sz % DG_CUDIRECT_BLK_SZ == 0u;
    }

    constexpr auto is_met_cudadirect_dgio_ptralignment_requirement(uintptr_t ptr) noexcept -> bool{

        return ptr % DG_CUDIRECT_BLK_SZ = 0u;
    }

    auto dg_cuopen_file(const char * path, int flag) noexcept -> std::expected<std::unique_ptr<CudaFileDescriptor, cuda_fclose_t>, exception_t>{ //this is over the line - should be unique_ptr

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
        cf_descr.type       = DG_CU_FILE_HANDLE_OPTION;
        exception_t err     = dg::network_exception::wrap_cuda_exception(cuFileHandleRegister(&cf_handle, &cf_descr)); 
        
        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return {std::in_place_t{}, new CudaFileDescriptor{std::move(kfd.value()), cf_handle}, destructor};
    }

    class CuFSIOInterface{

        public:

            virtual ~CuFSIOInterface() noexcept = default;
            virtual auto read_binary(const char *, cufs_sptr_t) noexcept -> exception_t = 0;
            virtual void write_binary(const char *, cufs_sptr_t) noexcept -> exception_t = 0; 
    };

    class FsysIOInterface{

        public:

            virtual ~FsysIOInterface() noexcept = default;
            virtual auto host_read_binary(const char * fp, void * dst, size_t dst_cap) noexcept -> exception_t = 0;
            virtual auto host_write_binary(const char * fp, const void * src, size_t src_sz) noexcept -> exception_t = 0;
            virtual auto cuda_read_binary(const char * fp, cuda_ptr_t  dst, size_t dst_cap) noexcept -> exception_t = 0;
            virtual auto cuda_write_binary(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept -> exception_t = 0; //constness here will be added in the future - to not over-complicate
    };

    class DirectPreAllocatedStableCuFSIO: public virtual CuFSIOInterface{

        private:
            
            std::unique_ptr<int, driver_x::dg_cufs_driver_dynamic_close_t> cufs_driver_fd;
            std::unordered_set<cufs_sptr_t> registered_hashset; 

        public:

            DirectPreRegisteredStableCUFSIO(std::unique_ptr<int, driver_x::dg_cufs_driver_dynamic_close_t> cufs_driver_fd,
                                            std::unordered_set<cufs_sptr_t> registered_hashset) noexcept: cufs_driver_fd(std::move(cufs_driver_fd)),
                                                                                                          registered_hashset(std::move(registered_hashset)){}
            
            auto read_binary(const char * fp, cufs_sptr_t dst) noexcept -> exception_t{

                if (!is_registered(dst)){
                    return dg::network_exception::UNREGISTERED_CUFILE_PTR;
                }

                auto raii_fd    = dg_cuopen_file(fp, O_RDONLY | O_DIRECT | O_TRUNC);

                if (!raii_fd.has_value()){
                    return raii_fd.error();
                } 

                int kernel_fd   = raii_fd.value()->kernel_raii_fd;
                size_t fsz      = dg::network_fileio_linux::dg_file_size_nothrow(kernel_fd);
                size_t dst_cap  = cufs_sptr_controller::get_size(dst);

                if (dst_cap < fsz){
                    return dg::network_exception::BUFFER_OVERFLOW;
                }

                if (!is_met_cudadirect_dgio_blksz_requirement(fsz)){
                    return dg::network_exception::BAD_ALIGNMENT;
                }

                if constexpr(DEBUG_MODE_FLAG){
                    if (!is_met_cudadirect_dgio_ptralignment_requirement(pointer_cast<uintptr_t>(dst))){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                if (cuFileRead(raii_fd.value()->cf_handle, pointer_cast<cufs_legacy_ptr_t>(dst), fsz, 0u, 0u) != fsz){
                    return dg::network_exception::RUNTIME_FILEIO_ERROR; //this is where the exception + abort line is blurred - I rather think this should throwing exception - cuFileRead is not legacy interface (last line of defense) 
                }

                return dg::network_exception::SUCCESS;
            }
        
            auto write_binary(const char * fp, cufs_sptr_t src) noexcept -> exception_t{

                if (!is_registered(src)){
                    return dg::network_exception::UNREGISTERED_CUFILE_PTR;
                }

                auto raii_fd = dg_cuopen_file(fp, O_WRONLY | O_DIRECT | O_TRUNC);

                if (!raii_fd.has_value()){
                    return raii_fd.error();
                }

                size_t src_sz = cufs_sptr_controller::get_size(src); 

                if constexpr(DEBUG_MODE_FLAG){
                    if (!is_met_cudadirect_dgio_ptralignment_requirement(pointer_cast<uintptr_t>(src))){
                        dg::network_exception_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
            
                    if (!is_met_cudadirect_dgio_blksz_requirement(src_sz)){
                        dg::network_exception_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                if (cuFileWrite(raii_fd.value()->cf_handle, pointer_cast<cufs_legacy_ptr_t>(src), src_sz, 0u, 0u) != src_sz){
                    return dg::network_exception::RUNTIME_FILEIO_ERROR;
                }

                return dg::network_exception::SUCCESS;
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

            CuPreallocatedFsysIO(std::unique_ptr<CuFSIOInterface> cu_fsio,
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

                auto buf        = std::make_unique<char[]>(dst_cap); //dedicated_buffer - or affinity allocator
                exception_t err = dg::network_fileio_linux::dg_read_binary(fp, buf.get(), dst_cap); 

                if (dg::network_exception::is_failed(err)){
                    return err;
                }

                err = dg::network_exception::wrap_cuda_exception(cudaMemcpy(dst, buf.get(), dst_cap, cudaMemcpyHostToDevice)); //change dst_cap -> file_sz
                dg::network_exception_handler::nothrow_log(err);
            }

            auto cuda_write_binary(const char * fp, cuda_ptr_t src, size_t src_sz) noexcept -> exception_t{
 
                auto buf        = std::make_unique<char[]>(src_sz); //dedicated_buffer - or affinity allocator
                exception_t err = dg::network_exception::wrap_cuda_exception(cudaMemcpy(buf.get(), src, src_sz, cudaMemcpyDeviceToHost));
                dg::network_exception_handler::nothrow_log(err);

                return dg::network_fileio_linux::dg_write_binary(fp, buf.get(), src_sz);
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

    inline std::unique_ptr<FsysIOInterface> cufsio_instance{};

    auto spawn_preallocated_direct_stable_fsysio(cuda_ptr_t * cuda_ptr, size_t * cuda_ptr_sz, size_t cuda_sz,
                                                 void ** host_ptr, size_t * host_ptr_sz, size_t host_sz) noexcept -> std::expected<std::unique_ptr<FsysIOInterface>, exception_t>{
        
        auto cudriver_ins       = driver_x::dg_cufs_driver_safe_open();  
        auto cuda_to_cufs_dict  = std::unordered_map<std::pair<uintptr_t, size_t>, cufs_sptr_t>{};
        auto host_to_cufs_dict  = std::unordered_map<std::pair<uintptr_t, size_t>, cufs_sptr_t>{};
        auto cufs_sptr_hashset  = std::unordered_set<cufs_sptr_t>{}; 

        if (!cudriver_ins.has_value()){
            return std::unexpected(cudriver_ins.error());
        }

        int cudriver_fd = *(cudriver_ins.value()) 

        for (size_t i = 0u; i < cuda_sz; ++i){
            auto cufs_raii_sptr = cufs_sptr_controller::safe_register_cudasptr(cuda_ptr[i], cuda_ptr_sz[i]);
            if (!cufs_raii_sptr.has_value()){
                return std::unexpected(cufs_raii_sptr.error());
            }
            
            cufs_sptr_t cufs_sptr = *(cufs_raii_sptr.value()); 
            cufs_sptr_hashset.insert(cufs_sptr);
            cuda_to_cufs_dict.insert(std::make_pair(std::make_pair(pointer_cast<uintptr_t>(cuda_ptr[i]), cuda_ptr_sz[i]), cufs_sptr));
            driver_x::dg_cufs_register_resource(cudriver_fd, std::move(cufs_raii_sptr.value()));
        }

        for (size_t i = 0u; i < host_sz; ++i){
            auto host_raii_sptr = cufs_sptr_controller::safe_register_hostsptr(host_ptr[i], host_ptr_sz[i]);
            if (!host_raii_sptr.has_value()){
                return std::unexpected(host_raii_sptr.error());
            }

            cufs_sptr_t cufs_sptr = *(cufs_raii_sptr.value()); 
            cufs_sptr_hashset.insert(cufs_sptr);
            host_to_cufs_dict.insert(std::make_pair(std::make_pair(pointer_cast<uintptr_t>(host_ptr[i]), host_ptr_sz[i]), cufs_sptr));
            driver_x::dg_cufs_register_resource(cudriver_fd, std::move(host_raii_sptr.value()));
        }

        std::unique_ptr<CuFSIOInterface> cufs_io = std::make_unique<DirectPreAllocatedStableCuFSIO>(std::move(cudriver_ins.value()), std::move(cufs_sptr_hashset));
        return std::make_unique<CuPreallocatedStableFsysIO>(std::move(cufs_io), std::move(host_to_cufs_dict), std::move(cuda_to_cufs_dict));
    }

    auto spawn_indirect_fsysio() noexcept -> std::unique_ptr<FsysIOInterface>{

        return std::make_unique<InDirectFsysIO>();
    }

    auto spawn_std_fsysio(std::unique_ptr<FsysIOInterface> fast, std::unique_ptr<FsysIOInterface> slow) noexcept -> std::unique_ptr<FsysIOInterface>{

        return std::make_unique<StdFsysIO>(std::move(fast), std::move(slow));
    }

    //--user-space--

    void init_preregister(cuda_ptr_t * cuda_ptr, size_t * cuda_ptr_sz, size_t cuda_sz, 
                          void ** host_ptr, size_t * host_ptr_sz, size_t host_sz) noexcept{
        
        //weird - strange - 
        auto [aligned_cuda_uptr, aligned_cuda_uptr_sz, aligned_cuda_sz] = filter_aligned_sptr(cuda_ptr, cuda_ptr_sz, cuda_sz); 
        auto [aligned_host_uptr, aligned_host_uptr_sz, aligned_host_sz] = filter_aligned_sptr(host_ptr, host_ptr_sz, host_sz);

        auto direct_fsysio = spawn_preallocated_direct_stable_fsysio(aligned_cuda_uptr.get(), aligned_cuda_uptr_sz.get(), aligned_cuda_sz, 
                                                                     aligned_host_uptr.get(), aligned_host_uptr_sz.get(), aligned_host_sz); 

        if (!direct_fsysio.has_value()){
            dg::network_log_stackdump::error(dg::network_exception::verbose(direct_fsysio.error()));
            cufosio_instance = spawn_indirect_fsysio(); 
            return;
        }

        cufsio_instance = spawn_std_fsysio(std::move(direct_fsysio.value()), spawn_indirect_fsysio());
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
}

#endif