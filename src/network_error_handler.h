#ifndef __NETWORK_ERROR_HANLDER_H__
#define __NETWORK_ERROR_HANDLER_H__

#include "network_log.h"
#include <stdlib.h>

namespace dg::network_error_handler{

    using cuda_err_t    = int; 
    using kernel_err_t  = int; 
    using core_err_t    = int; 

    static inline void cuda(cuda_err_t err_code, bool is_noexcept, bool has_log_write_on_err){

        if (err_code == CUDA_SUCCESS){
            return;
        }

        if (has_log_write_on_err){
            if (is_noexcept){
                dg::network_log::critical_error("cuda", err_code);
            } else{
                dg::network_log::error("cuda", err_code);
            }
        }
        
        if (is_noexcept){
            std::abort();
        }

        throw cuda_exception_table[err_code];
    }

    static inline void kernel(kernel_err_t err_code, bool is_noexcept, bool has_log_write_on_err){

        if (err_code == KERNEL_SUCCESS){
            return;
        }

        if (has_log_write_on_err){
            if (is_noexcept){
                dg::network_log::critical_error("kernel", err_code);
            } else{
                dg::network_log::error("kernel", err_code);
            }
        }

        if (is_noexcept){
            std::abort();
        }

        throw kernel_exception_table[err_code];
    }

    static inline void core(core_err_t err_code, bool is_noexcept, bool has_log_write_on_err){

        if (err_code == CORE_SUCCESS){
            return;
        }

        if (has_log_write_on_err){
            if (is_noexcept){
                dg::network_log::critical_error("core", err_code);
            } else{
                dg::network_log::error("core", err_code);
            }
        }

        if (is_noexcept){
            std::abort();
        }

        throw core_exception_table[err_code];
    } 

    inline void cuda_noexcept_log(cuda_err_t err_code) noexcept{

        cuda(err_code, true, true);
    }

    inline void cuda_noexcept_nolog(cuda_err_t err_code) noexcept{

        cuda(err_code, true, false);
    }

    inline void cuda_except_log(cuda_err_t err_code){

        cuda(err_code, false, true);
    }

    inline void cuda_except_nolog(cuda_err_t err_code){

        cuda(err_code, false, false);
    }

    inline void kernel_noexcept_log(kernel_err_t err_code) noexcept{

        kernel(err_code, true, true);
    } 

    inline void kernel_noexcept_nolog(kernel_err_t err_code) noexcept{

        kernel(err_code, true, false);
    }
    
    inline void kernel_except_log(kernel_err_t err_code){

        kernel(err_code, false, true);
    }

    inline void kernel_except_nolog(kernel_err_t err_code){

        kernel(err_code, false, false);
    }

    inline void core_noexcept_log(core_err_t err_code) noexcept{

        core(err_code, true, true);
    }

    inline void core_noexcept_nolog(core_err_t err_code) noexcept{

        core(err_code, true, false);
    }

    inline void core_except_log(core_err_t err_code){

        core(err_code, false, true);
    }

    inline void core_except_nolog(core_err_t err_code){

        core(err_code, false, false);
    }
    
} 

#endif