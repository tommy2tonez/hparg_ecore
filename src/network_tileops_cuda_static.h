#ifndef __NETWORK_TILEOPS_CUDA_POLY_H__
#define __NETWORK_TILEOPS_CUDA_POLY_H__ 

#include <stdint.h>
#include <stddef.h>
#include <type_traits>
#include "cublas_x.h"
#include <memory>

namespace network_tileops_cuda_poly{

    auto make_exec_fwd_mono_exp(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

        using namespace dg::cublas_x; 

        auto rhs            = cublas_make_matrix(sq_dim, sq_dim, MATRIX_N, src_type);
        auto rhs_coerced    = cublas_mono_cast(rhs, dst_type);
        auto rs             = cublas_mono_exp(rhs_coerced);
        auto opti_plan      = cublas_optimize_slow(rs);

        return cublas_make_executable(opti_plan);
    }

    auto make_exec_fwd_mono_relu(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_log(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_log2(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_abs(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_cos(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_acos(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_sin(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_asin(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_tan(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_atan(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_sqrt(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_invsqrt(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_negative(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    auto make_exec_fwd_mono_negate(logit_option_t dst_type, logit_option_t src_type, const size_t sq_dim) -> std::unique_ptr<dg::cublas_x::ExecutorInterface>{

    }

    struct Transformer{

        std::unique_ptr<dg::cublas_x::ExecutorInterface> relu_u8_u8;
        std::unique_ptr<dg::cublas_x::ExecutorInterface> relu_u8_u16;
        std::unique_ptr<dg::cublas_x::ExecutorInterface> relu_u8_f8;
        std::unique_ptr<dg::cublas_x::ExecutorInterface> relu_u8_f16;
        std::unique_ptr<dg::cublas_x::ExecutorInterface> relu_f8_u8;
        std::unique_ptr<dg::cublas_x::ExecutorInterface> relu_f8_u16;
        std::unique_ptr<dg::cublas_x::ExecutorInterface> relu_f8_f8;
        std::unique_ptr<dg::cublas_x::ExecutorInterface> relu_f8_f16;
    };

    auto make_transformer() -> std::unique_ptr<Transformer>{

    }

    inline std::unique_ptr<Transformer> transformer_ins = make_transformer();

    void dispatch_forward_mono(...){}
    void dispatch_forward_pair(...){}
    void dispatch_forward_uacm(...){}
    void dispatch_forward_pacm(...){}
    void dispatch_forward_crit(...){}
    void dispatch_forward_msgrfwd(...){}
    void dispatch_forward_msgrbwd(...){}
    void dispatch_backward_mono(...){}
    void dispatch_backward_pair_lhs(...){}
    void dispatch_backward_pair_rhs(...){}
    void dispatch_backward_pacm_lhs(...){}
    void dispatch_backward_pacm_rhs(...){}
    void dispatch_backward_uacm(...){}
}

#endif
