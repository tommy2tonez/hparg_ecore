#ifndef __NETWORK_EXCEPTION_H__
#define __NETWORK_EXCEPTION_H__

#include <exception>
#include <stdint.h>
#include <stdlib.h> 
#include <type_traits>

using exception_t = uint16_t; 

namespace dg::network_exception{

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
    static inline constexpr exception_t UNREGISTERED_CUFILE_PTR         = 0u;
    static inline constexpr exception_t RUNTIME_SOCKETIO_ERROR          = 0u;
    static inline constexpr exception_t BAD_SPIN                        = 0u;
    static inline constexpr exception_t BUFFER_OVERFLOW                 = 0u;
    static inline constexpr exception_t RUNTIME_FILEIO_ERROR            = 0u; 
    static inline constexpr exception_t LOST_RETRANSMISSION             = 0u;
    static inline constexpr exception_t INVALID_INIT_ARG                = 0u;
    static inline constexpr exception_t UNSUPPORTED_DAEMON_MODE         = 0u;
    static inline constexpr exception_t NO_DAEMON_EXECUTOR_AVAILABLE    = 0u;
    static inline constexpr exception_t INVALID_ARGUMENT                = 0u;
    
    static inline const char * SEGFAULT_CSTR                    = "segmentation_fault";
    static inline const char * UNREACHABLE_CSTR                 = "unreachable_fault"; 
    static inline const char * INVALID_DICTIONARY_KEY_CSTR      = "invalid_dictionary_key";

    inline auto wrap_cuda_exception(cuda_exception_t) noexcept -> exception_t{

    }

    inline auto wrap_kernel_exception(kernel_exception_t) noexcept -> exception_t{

    }

    inline auto wrap_core_exception(core_exception_t) noexcept -> exception_t{

    }

    inline auto wrap_std_errcode(...) -> exception_t{ 

    }

    inline auto is_success(exception_t) noexcept -> bool{

    }

    inline auto is_failed(exception_t) noexcept -> bool{

    }

    inline auto verbose(exception_t) noexcept -> const char *{

    }

    inline void throw_exception(exception_t){
        
    }

    template <class T>
    struct base_type{
        using type = T;
    };

    template <class T>
    struct base_type<T&>: base_type<T>{};

    template <class T>
    struct base_type<T&&>: base_type<T>{};

    template <class T>
    struct base_type<const T>: base_type<T>{};

    template <class T>
    struct base_type<volatile T>: base_type<T>{};

    template <class T>
    using base_type_t = typename base_type<T>::type; 

    template <class Functor>
    inline auto to_cstyle_function(Functor functor) noexcept{

        static_assert(std::is_nothrow_move_constructible<Functor>);

        auto rs = [f = std::move(functor)]<class ...Args>(Args&& ...args) noexcept(noexcept(functor(std::forward<Args>(args)...))){
            using ret_t = decltype(f(std::forward<Args>(args)...));

            if constexpr(std::is_same_v<ret_t, void>){
                try{
                    f(std::forward<Args>(args)...);
                    return SUCCESS;
                } catch (...){
                    return wrap_std_exception(std::current_exception());
                }
            } else{
                try{
                    static_assert(std::is_nothrow_move_constructible<ret_t>);
                    static_assert(std::is_same_v<ret_t, base_type_t<ret_t>>);
                    return std::expected<ret_t, exception_t>(f(std::forward<Args>(args)...));
                } catch (...){
                    return std::expected<ret_t, exception_t>(std::unexpected(wrap_std_exception(std::current_exception())));
                }
            }
        };

        return rs;
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

}

#endif