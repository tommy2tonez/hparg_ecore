#ifndef __DEDICATED_BUFFER_H__
#define __DEDICATED_BUFFER_H__

#include <stdint.h>
#include <stddef.h>
#include <type_traits>
#include <thread>
#include <vector>
#include "network_concurrency.h"
#include <cstring>
#include <type_traits>
#include "network_memult.h"
#include "network_log.h" 

namespace dg::network_function_concurrent_buffer{

    template <class ID, size_t BUF_SZ>
    struct DedicatedBuffer{

        private:

            static inline constexpr size_t ALIGNMENT_SZ = dg::memult::simd_align_val_max();
            static inline void ** buf{}; 

        public:

            static_assert(BUF_SZ != 0);

            static void init() noexcept{
                
                auto log_scope = dg::network_log_scope::critical_error_terminate(); 
                buf = new std::add_pointer_t<void>[dg::network_concurrency::THREAD_COUNT]; 

                for (size_t i = 0; i < dg::network_concurrency::THREAD_COUNT; ++i){
                    buf[i] = dg::memult::aligned_alloc_cpp(ALIGNMENT_SZ, BUF_SZ);
                }

                log_scope.release();
            } 

            static inline auto get() noexcept -> void *{

                return std::assume_aligned<ALIGNMENT_SZ>(buf[dg::network_concurrency::this_thread_idx()]);
            }
    };

    template <class FunctionSignature, size_t BUF_SZ, std::enable_if_t<(BUF_SZ > 0), bool> = true>
    struct tag{}; 

    template <class FunctionSignature, size_t BUF_SZ>
    void init(tag<FunctionSignature, BUF_SZ>) noexcept{

        DedicatedBuffer<FunctionSignature, BUF_SZ>::init();
    }

    template <class FunctionSignature, size_t BUF_SZ>
    inline auto get_buf(const tag<FunctionSignature, BUF_SZ>) noexcept -> void *{

        return DedicatedBuffer<FunctionSignature, BUF_SZ>::get();
    }

    template <class FunctionSignature, size_t BUF_SZ>
    inline auto get_cbuf(const tag<FunctionSignature, BUF_SZ>) noexcept -> void *{

        void * rs = get_buf(tag<FunctionSignature, BUF_SZ>{});
        std::memset(rs, 0u, BUF_SZ);

        return rs;
    }
}

namespace dg::network_function_concurrent_local_array{

    template <class LocalVarSignature, class T, size_t SZ, std::enable_if_t<std::is_fundamental_v<T> && (SZ > 0), bool> = true>
    struct fundamental_tag{};

    template <class LocalVarSignature, class T, size_t SZ>
    void init(fundamental_tag<LocalVarSignature, T, SZ>) noexcept{

        constexpr size_t ARRAY_BUF_SZ   = SZ * sizeof(T) + alignof(T); 
        dg::network_function_concurrent_buffer::init(dg::network_function_concurrent_buffer::tag<LocalVarSignature, ARRAY_BUF_SZ>{});
    }

    template <class LocalVarSignature, class T, size_t SZ>
    inline auto get_array(const fundamental_tag<LocalVarSignature, T, SZ>) noexcept -> T *{
        
        constexpr size_t ARRAY_BUF_SZ   = SZ * sizeof(T) + alignof(T); 
        void * buf                      = dg::network_function_concurrent_buffer::get_buf(dg::network_function_concurrent_buffer::tag<LocalVarSignature, ARRAY_BUF_SZ>{}); 
        void * aligned_buf              = dg::memult::align(buf, std::integral_constant<size_t, alignof(T)>{});

        return dg::memult::start_lifetime_as_array<T>(aligned_buf, SZ);
    }

    template <class LocalVarSignature, class T, size_t SZ>
    inline auto get_carray(const fundamental_tag<LocalVarSignature, T, SZ>) noexcept -> T *{

        constexpr size_t ARRAY_BUF_SZ   = SZ * sizeof(T) + alignof(T); 
        void * buf                      = dg::network_function_concurrent_buffer::get_cbuf(dg::network_function_concurrent_buffer::tag<LocalVarSignature, ARRAY_BUF_SZ>{}); 
        void * aligned_buf              = dg::memult::align(buf, std::integral_constant<size_t, alignof(T)>{});

        return dg::memult::start_lifetime_as_array<T>(aligned_buf, SZ);
    }

    template <class LocalVarSignature, class T, size_t SZ>
    inline auto get_default_array(const fundamental_tag<LocalVarSignature, T, SZ>) noexcept -> T *{

        T * arr = get_array(fundamental_tag<LocalVarSignature, T, SZ>{});
        std::fill_n(arr, SZ, T{});
        return arr;
    }
}

#endif