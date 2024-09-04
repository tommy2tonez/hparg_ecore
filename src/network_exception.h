#ifndef __NETWORK_EXCEPTION_H__
#define __NETWORK_EXCEPTION_H__

#include <exception>
#include <stdint.h>
#include <stdlib.h> 
#include <type_traits>

namespace dg::network_exception{

    using exception_t = uint16_t; 

    #ifdef __DG_NETWORK_CUDA_FLAG__
    using cuda_exception_t = cudaError_t;
    #else 
    using cuda_exception_t = uint8_t;
    #endif

    using kernel_exception_t    = int; 
    using core_exception_t      = uint32_t;

    struct base_exception: std::exception{

        const char * c_msg;

        base_exception(const char * c_msg) noexcept: c_msg(c_msg){}

        inline auto what() const noexcept -> const char *{

            return this->c_msg;
        }
    };

    struct invalid_arg: base_exception{

        invalid_arg() noexcept: base_exception("invalid_arg"){}
    };
    
    static inline constexpr exception_t SUCCESS                         = 0u;
    static inline constexpr exception_t CRITICAL_FAILURE                = 0u;
    static inline constexpr exception_t OCCUPIED_MEMREGION              = 0u;
    static inline constexpr exception_t UNWAITABLE_EXCEPTION            = 0u;
    static inline constexpr exception_t SEGFAULT                        = 0u;
    static inline constexpr exception_t INTERNAL_CORRUPTION             = 0u;
    static inline constexpr exception_t CUDA_DEVICE_NOT_FOUND           = 0u;
    static inline constexpr exception_t CUDA_DEVICE_NOT_SUPPORTED       = 0u;
    static inline constexpr exception_t UNMET_PRECOND                   = 0u; 
    static inline constexpr exception_t RECOVERABLE_OUT_OF_MEMORY       = 0u;
    static inline constexpr exception_t OUT_OF_MEMORY                   = 0u;  
    static inline constexpr exception_t INVALID_SERIALIZATION_FORMAT    = 0u;
    static inline constexpr exception_t INVALID_VMAPTR_FORMAT           = 0u;
    static inline constexpr exception_t INVALID_DICTIONARY_KEY          = 0u;
    static inline constexpr exception_t INVALID_TABLE_DISPATCH_CODE     = 0u;
    static inline constexpr exception_t INCOMPATIBLE_OPERATABLE_ID      = 0u;
    static inline constexpr exception_t BAD_TILE_MEMBER_ACCESS          = 0u; 
    static inline constexpr exception_t BAD_PTR_ACCESS                  = 0u;
    static inline constexpr exception_t BAD_OPTIONAL_ACCESS             = 0u; 
    static inline constexpr exception_t BAD_RAII_ACCESS                 = 0u;
    static inline constexpr exception_t BAD_ALIGNMENT                   = 0u;
    static inline constexpr exception_t BUFFER_OVERFLOW                 = 0u;
    static inline constexpr exception_t RUNTIME_FILEIO_ERROR            = 0u; 

    static inline const char * SEGFAULT_CSTR                    = "segmentation_fault";
    static inline const char * UNREACHABLE_CSTR                 = "unreachable_fault"; 
    static inline const char * INVALID_DICTIONARY_KEY_CSTR      = "invalid_dictionary_key";

    inline auto wrap_cuda_exception(cuda_exception_t) noexcept -> exception_t{

    }

    inline auto wrap_kernel_exception(kernel_exception_t) noexcept -> exception_t{

    }

    inline auto wrap_core_exception(core_exception_t) noexcept -> exception_t{

    }

    inline auto is_success(exception_t) noexcept -> bool{

    }

    inline auto is_failed(exception_t) noexcept -> bool{

    }

    inline auto verbose(exception_t) noexcept -> const char *{

    }

    inline void throw_exception(exception_t){
        
    }

    template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_same_v<Args, exception_t>...>, bool> = true>
    inline auto disjunction(Args... args) noexcept -> exception_t{
        
        exception_t rs = SUCCESS; 

        ([&]{
            if (is_failed(args)){
                rs = args;
            }
        }(), ...);

        return rs;
    }

    //WLOG, std::expected<size_t, exception_t>, std::expected<uint32_t, exception_t> -> std::expected<std::tuple<size_t, uint32_t>, exception_t> - to leverage auto [lhs, rhs] = expected.value() technique

    template <class ...Args, std::enable_if_t<std::conjunction_v<is_expected<Args>...>, bool> = true>
    inline auto disjunct_tuplize_expected() noexcept{ //weird semantics - 

    } 

}

#endif