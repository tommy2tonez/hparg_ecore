#ifndef __NETWORK_TILEOPS_CUDA_POLY_H__
#define __NETWORK_TILEOPS_CUDA_POLY_H__

namespace dg::newtork_tileops_cuda_poly{

    auto make_cuda_vptr(cuda_ptr_t, cuda_id_t){

    } 

    void fwd_mono(cuda_vptr_t dst, cuda_vptr_t src, dispatch_t dispatch_code) noexcept{

    }

    void fwd_pair(cuda_vptr_t dst, cuda_vptr_t lhs, cuda_vptr_t rhs, dispatch_t dispatch_code) noexcept{

    }

    void fwd_uacm(cuda_vptr_t dst, std::array<cuda_vptr_t, UACM_COUNT> src, dispatch_t dispatch_code) noexcept{

    }

    void fwd_pacm(cuda_vptr_t dst, std::array<cuda_vptr_t, PACM_COUNT> lhs, std::array<cuda_vptr_t, PACM_COUNT> rhs, dispatch_t dispatch_code) noexcept{

    }

    void fwd_crit(cuda_vptr_t dst, cuda_vptr_t src, dispatch_t dispatch_code) noexcept{

    }

    void fwd_extn(cuda_vptr_t dst, cuda_vptr_t src, dispatch_t dispatch_code) noexcept{

    }

    void fwd_msgrfwd(cuda_vptr_t dst, cuda_vptr_t src, dispatch_t dispatch_code) noexcept{

    }

    void fwd_msgrbwd(cuda_vptr_t dst, cuda_vptr_t src, dispatch_t dispatch_code) noexcept{

    }

    void bwdzr_mono(cuda_vptr_t dst_grad, cuda_vptr_t dst_logit, cuda_vptr_t src_grad, dispatch_t dispatch_code) noexcept{

    }

    void bwdzr_pair(cuda_vptr_t ldst_grad, cuda_vptr_t ldst_logit, 
                    cuda_vptr_t rdst_grad, cuda_vptr_t rdst_logit,
                    cuda_vptr_t src_grad,
                    dispatch_t dispatch_code) noexcept{

    }

    void bwdzr_uacm(std::array<cuda_vptr_t, UACM_COUNT> dst_grad, std::array<cuda_vptr_t, UACM_COUNT> dst_logit,
                    cuda_vptr_t src_grad, 
                    dispatch_t dispatch_code) noexcept{

    }

    void bwdzr_pacm(std::array<cuda_vptr_t, PACM_COUNT> ldst_grad, std::array<cuda_vptr_t, PACM_COUNT> ldst_logit,
                    std::array<cuda_vptr_t, PACM_COUNT> rdst_grad, std::array<cuda_vptr_t, PACM_COUNT> rdst_logit,
                    cuda_vptr_t src_grad, 
                    dispatch_t dispatch_code) noexcept{

    }

    //-- should these be their own type or a part of mono - design question

    void bwdzr_crit(cuda_vptr_t dst_grad, cuda_vptr_t dst_logit, cuda_vptr_t src_grad, dispatch_t dispatch_code) noexcept{

    }

    void bwdzr_extn(cuda_vptr_t dst_grad, cuda_vptr_t dst_logit, cuda_vptr_t src_grad, dispatch_t dispatch_code) noexcept{

    }

    void bwdzr_msgrfwd(cuda_vptr_t dst_grad, cuda_vptr_t dst_logit, cuda_vptr_t src_grad, dispatch_t dispatch_code) noexcept{

    }

    void bwdzr_msgrbwd(cuda_vptr_t dst_grad, cuda_vptr_t dst_logit, cuda_vptr_t src_grad, dispatch_t dispatch_code) noexcept{

    }
} 

#endif