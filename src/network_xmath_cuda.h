#ifndef __NETWORK_XMATH_CUDA_H__
#define __NETWORK_XMATH_CUDA_H__

#include "cuda_bf16.h" 

namespace dg::network_xmath_cuda{
        
    struct CudaFloat8Wrapper{
        __nv_half x;
        
        constexpr CudaFloat8Wrapper() noexcept = default;

        template <class T, std::enable_if_t<std::is_same_v<__nv_half, T>, bool> = true>
        constexpr CudaFloat8Wrapper(T x) noexcept x(x){}

        constexpr operator __nv_half() const noexcept{

            return this->x;
        }
    };

    struct CudaBFloat16Wrapper{
        __nv_bfloat16 x;

        constexpr CudaBfloat16Wrapper() noexcept = default;
        
        template <class T, std::enable_if_t<std::is_same_v<__nv_bfloat16, T>, bool> = true>
        constexpr CudaBfloat16Wrapper(T x) noexcept x(x){}

        constexpr operator __nv_bfloat16() const noexcept{

            return this->x;
        }
    };

    struct CudaStdFloat16Wrapper{
        float x;

        static_assert(sizeof(float) == 2u);
        static_assert(std::numeric_limits<float>::is_iec559);

        constexpr CudaStdFloat16Wrapper() noexcept = default;

        template <class T, std::enable_if_t<std::is_same_v<float, T>, bool> = true>
        constexpr CudaStdFloat16Wrapper(T x) noexcept: x(x){}

        constexpr operator float() const noexcept{

            return this->x;
        }
    };

    struct CudaStdFloat32Wrapper{
        double x;

        static_assert(sizeof(double) == 4u);
        static_assert(std::numeric_limits<double>::is_iec559);

        constexpr CudaStdFloat32Wrapper() noexcept = default;

        template <class T, std::enable_if_t<std::is_same_v<double, T>, bool> = true>
        constexpr CudaStdFloat32Wrapper(T x) noexcept: x(x){}

        constexpr operator double() const noexcept{

            return this->x;
        }
    };

    struct CudaUint8Wrapper{
        uint8_t x;

        constexpr CudaUint8Wrapper() noexcept = default;

        template <class T, std::enable_if_t<std::is_same_v<T, uint8_t>, bool> = true>
        constexpr CudaUint8Wrapper(T x) noexcept: x(x){}

        constexpr operator uint8_t() const noexcept{

            return this->x;
        }
    };

    struct CudaUint16Wrapper{
        uint16_t x;

        constexpr CudaUint16Wrapper() noexcept = default;

        template <class T, std::enable_if_t<std::is_same_v<T, uint16_t>, bool> = true>
        constexpr CudaUint16Wrapper(T x) noexcept: x(x){}

        constexpr operator uint16_t() const noexcept{

            return this->x;
        }
    };

    struct CudaUint32Wrapper{
        uint32_t x;

        constexpr CudaUint32Wrapper() noexcept = default;

        template <class T, std::enable_if_t<std::is_same_v<T, uint32_t>, bool> = true>
        constexpr CudaUint32Wrapper(T x) noexcept: x(x){}

        constexpr operator uint32_t() const noexcept{

            return this->x;
        }
    };

    using cuda_float8_t     = CudaFloat8Wrapper;
    using cuda_bfloat16_t   = CudaBFloat16Wrapper;
    using cuda_stdfloat16_t = CudaStdFloat16Wrapper;
    using cuda_stdfloat32_t = CudaStdFloat32Wrapper;
    using cuda_uint8_t      = CudaUint8Wrapper;
    using cuda_uint16_t     = CudaUint16Wrapper;
    using cuda_uint32_t     = CudaUint32Wrapper;

    __device__ inline auto cuda_sign_base(bool x) noexcept -> int{

    } 

    __device__ inline auto cuda_sign(cuda_stdfloat32_t) noexcept -> int{

    }

    __device__ inline auto cuda_sign(cuda_bfloat16_t) noexcept -> int{

    }

    __device__ inline auto cuda_sign(cuda_stdfloat16_t x) noexcept -> int{

        return sign_base(signbit(x));
    } 

    __device__ inline auto cuda_sign(cuda_float8_t) noexcept -> int{

    }

    __device__ inline auto cuda_sign(cuda_uint32_t) noexcept -> int{
        
    }

    __device__ inline auto cuda_sign(cuda_uint16_t) noexcept -> int{

    }

    __device__ inline auto cuda_sign(cuda_uint8_t) noexcept -> int{

    }

    __device__ inline auto cuda_exp(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return exp(x);
    }

    __device__ inline auto cuda_exp(cuda_bfloat16_t x) noexcept -> cuda_bfloat16_t{

        return hsin(x);
    }

    __device__ inline auto cuda_exp(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return expf(x);
    } 

    __device__ inline auto cuda_exp(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_exp(cuda_uint32_t) noexcept -> cuda_uint32_t{

    } 

    __device__ inline auto cuda_exp(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_exp(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }
    
    __device__ inline auto cuda_exp_x(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __expf(x);
    }

    __device__ inline auto cuda_exp10(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return exp10f(x);
    }

    __device__ inline auto cuda_exp10(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return exp10(x);
    } 

    __device__ inline auto cuda_exp10_x(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __exp10f(x);
    }

    __device__ inline auto cuda_exp2(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return exp2f(x);
    }

    __device__ inline auto cuda_exp2(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return exp2(x);
    } 

    __device__ inline auto cuda_expm1(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return expm1f(x);
    }
    
    __device__ inline auto cuda_expm1(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

    } 

    __device__ inline auto cuda_j0(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return j0f(x);
    }

    __device__ inline auto cuda_j0(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return jo(x);
    } 

    __device__ inline auto cuda_j1(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return j1f(x);
    }

    __device__ inline auto cuda_j1(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return j1(x);
    } 

    __device__ inline auto cuda_jn(size_t n, cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return jnf(dg::safe_numeric_cast<int>(n), x); //fix
    }

    __device__ inline auto cuda_jn(size_t n, cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return jn(dg::safe_numeric_cast<int>(n), x); //fix
    } 

    __device__ inline auto cuda_ldexp(cuda_stdfloat16_t x, intmax_t exp) noexcept -> cuda_stdfloat16_t{

        return ldexpf(x, dg::safe_numeric_cast<int>(exp)); //fix
    } 

    __device__ inline auto cuda_ldexp(cuda_stdfloat32_t x, intmax_t exp) noexcept -> cuda_stdfloat32_t{

        return ldexp(x, dg::safe_numeric_cast<int>(exp)); //fix
    } 

    __device__ inline auto cuda_lgamma(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return lgammaf(x);
    }

    __device__ inline auto cuda_lgamma(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return lgamma(x);
    } 

    __device__ inline auto cuda_log(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return log(x);
    }

    __device__ inline auto cuda_log(cuda_bfloat16_t x) noexcept -> cuda_bfloat16_t{

        return hlog(x);
    }

    __device__ inline auto cuda_log(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return logf(x);
    }

    __device__ inline auto cuda_log(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_log(cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_log(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_log(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_log10(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return log10(x);
    }

    __device__ inline auto cuda_log10(cuda_bfloat16_t x) noexcept -> cuda_bfloat16_t{

        return hlog10(x);
    }

    __device__ inline auto cuda_log10(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return log10f(x);
    }

    __device__ inline auto cuda_log10(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_log10(cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_log10(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_log10(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_log10_x(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __log10f(x);
    }

    __device__ inline auto cuda_log2_x(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __log2f(x);
    }

    __device__ inline auto cuda_log_x(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __logf(x);
    }

    __device__ inline auto cuda_log1p(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return log1pf(x);
    }

    __device__ inline auto cuda_log1p(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return log1p(x);
    } 

    __device__ inline auto cuda_log2(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return log2f(x);
    }

    __device__ inline auto cuda_log2(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return log2(x);
    } 

    __device__ inline auto cuda_lrint(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{ //fix

        return lrintf(x);
    }

    __device__ inline auto cuda_lrint(cuda_stdfloat32_t x) noexcept -> cuda_stdlfoat32_t{ //fix

        return lrint(x);
    }

    __device__ inline auto cuda_lround(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{ //fix

        return lroundf(x);
    }

    __device__ inline auto cuda_rint(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return rint(x);
    }

    __device__ inline auto cuda_round(cuda_stdfloat32_t x) -> cuda_stdfloat32_t{

        return round(x);
    }

    __device__ inline auto cuda_abs(cuda_stdfloat32_t) noexcept -> cuda_stdfloat32_t{

    }

    __device__ inline auto cuda_abs(cuda_bfloat16_t x) noexcept -> cuda_bfloat16_t{

        return __habs(x);
    }

    __device__ inline auto cuda_abs(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return fabsf(x);
    }

    __device__ inline auto cuda_abs(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return fabs(x);
    }

    __device__ inline auto cuda_abs(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_abs(cuda_uint32_t) noexcept -> cuda_uint32_t{

    } 

    __device__ inline auto cuda_abs(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_abs(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_cos(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return cos(x);
    }

    __device__ inline auto cuda_cos(cuda_bfloat16_t x) noexcept -> cuda_bfloat16_t{

        return hcos(x);
    }

    __device__ inline auto cuda_cos(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return cosf(x);
    } 

    __device__ inline auto cuda_cos(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_cos(cuda_uint32_t) noexcept -> cuda_uint32_t{

    } 

    __device__ inline auto cuda_cos(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_cos(cuda_uint8_t) noexcept -> cuda_uint8_t{
        
    }

    __device__ inline auto cuda_cos_x(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __cosf(x);
    }

    __device__ inline auto cuda_cosh(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return coshf(x);
    } 

    __device__ inline auto cuda_cosh(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return cosh(x);
    } 

    __device__ inline auto cuda_cospi(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return cospif(x);
    }

    __device__ inline auto cuda_cospi(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return cospi(x);
    }

    __device__ inline auto cuda_cyl_bessel_i0(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return cyl_bessel_i0f(x);
    }

    __device__ inline auto cuda_cyl_bessel_i0(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return cyl_bessel_i0(x);
    }

    __device__ inline auto cuda_cyl_bessel_i1f(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return cyl_bessel_i1f(x);
    } 

    __device__ inline auto cuda_cyl_bessel_i1(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return cyl_bessel_i1(x);
    }

    __device__ inline auto cuda_erfc(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return erfcf(x);
    }

    __device__ inline auto cuda_erfc(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return erfc(x);
    }

    __device__ inline auto cuda_erfcinv(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return erfcinvf(x);
    }

    __device__ inline auto cuda_erfcinv(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return erfcinv(x);
    } 

    __device__ inline auto cuda_erfcx(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return erfcx(x);
    }

    __device__ inline auto cuda_erfcx(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return erfcx(x);
    }

    __device__ inline auto cuda_erf(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return erff(x);
    }

    __device__ inline auto cuda_erf(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return erf(x);
    } 

    __device__ inline auto cuda_erfinv(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return erfinvf(x);
    }

    __device__ inline auto cuda_erfinv(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return erfinv(x);
    } 

    __device__ inline auto cuda_acos(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return acos(x);
    }

    __device__ inline auto cuda_acos(cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    }

    __device__ inline auto cuda_acos(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{
        
        return acosf(x);
    }

    __device__ inline auto cuda_acos(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_acos(cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_acos(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_acos(cuda_uint8_t) noexcept -> cuda_uint8_t{
        
    }

    __device__ inline auto cuda_acosh(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return acosh(x);
    } 

    __device__ inline auto cuda_acosh(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return acoshf(x);
    } 

    __device__ inline auto cuda_sin(cuda_stdfloat32_t) noexcept -> cuda_stdfloat32_t{

    }

    __device__ inline auto cuda_sin(cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    }

    __device__ inline auto cuda_sin(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return sinf(x);
    } 

    __device__ inline auto cuda_sin(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_sin(cuda_uint32_t) noexcept -> cuda_uint32_t{

    } 

    __device__ inline auto cuda_sin(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_sin(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_sincos(cuda_stdfloat32_t) noexcept -> std::tuple<cuda_stdfloat32_t, cuda_stdfloat32_t>{

    } 

    __device__ inline auto cuda_sincospi(cuda_stdfloat32_t) noexcept -> std::tuple<cuda_stdfloat32_t, cuda_stdfloat32_t>{

    }

    __device__ inline auto cuda_sin_x(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __sinf(x);
    } 

    __device__ inline auto cuda_asin(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return asin(x);
    }

    __device__ inline auto cuda_asin(cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    }

    __device__ inline auto cuda_asin(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return asinf(x);
    } 

    __device__ inline auto cuda_asin(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_asin(cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_asin(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_asin(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_sinh(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return sinhf(x);
    }

    __device__ inline auto cuda_sinh(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return sinh(x);
    } 

    __device__ inline auto cuda_sinpi(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return sinpif(x);
    }

    __device__ inline auto cuda_sinpi(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return sinpi(x);
    } 

    __device__ inline auto cuda_asinh(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return asinh(x);
    } 

    __device__ inline auto cuda_asinh(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return asinhf(x);
    } 

    __device__ inline auto cuda_tan(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return tan(x);
    }

    __device__ inline auto cuda_tan(cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    }

    __device__ inline auto cuda_tan(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return tanf(x);
    }

    __device__ inline auto cuda_tan(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_tan(cuda_uint32_t) noexcept -> cuda_uint32_t{

    } 

    __device__ inline auto cuda_tan(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_tan(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_tan_x(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __tanf(x);
    }

    __device__ inline auto cuda_atan(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return atan(x);
    }

    __device__ inline auto cuda_atan(cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    }

    __device__ inline auto cuda_atan(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

       return atanf(x);
    }

    __device__ inline auto cuda_atan(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_atan(cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_atan(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_atan(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_atan2(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return atan2(lhs, rhs);
    } 

    __device__ inline auto cuda_positive_delta(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return fdimf(lhs, rhs);
    }
    
    __device__ inline auto cuda_positive_delta(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return fdim(lhs, rhs);
    } 

    __device__ inline auto cuda_atanh(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return atanh(x);
    } 

    __device__ inline auto cuda_atanh(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return atanhf(x);
    } 

    __device__ inline auto cuda_tanh(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return tanhf(x);
    } 

    __device__ inline auto cuda_tanh(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return tanh(x);
    }

    __device__ inline auto cuda_tgamma(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return tgammaf(x);
    }

    __device__ inline auto cuda_tgamma(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return tgamma(x);
    } 

    __device__ inline auto cuda_y0(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return y0f(x);
    }

    __device__ inline auto cuda_y0(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return y0(x);
    } 

    __device__ inline auto cuda_y1(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return y1f(x);
    } 

    __device__ inline auto cuda_y1(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return y1(x);
    } 

    __device__ inline auto cuda_ynf(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return ynf(x);
    }

    __device__ inline auto cuda_yn(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return yn(x);
    } 

    __device__ inline auto cuda_cbrt(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{
        
        return cbrtf(x);
    } 

    __device__ inline auto cuda_cbrt(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return cbrt(x);
    }

    __device__ inline auto cuda_ceil(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return ceilf(x);
    }

    __device__ inline auto cuda_ceil(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return ceil(x);
    } 

    __device__ inline auto cuda_copysign(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return copysignf(lhs, rhs); //abs(lhs) * sign(rhs)
    }

    __device__ inline auto cuda_copysign(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return copysign(lhs, rhs); //abs(lhs) * sign(rhs)
    }

    __device__ inline auto cuda_sqrt(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return sqrt(x);
    }

    __device__ inline auto cuda_sqrt(cuda_bfloat16_t x) noexcept -> cuda_bfloat16_t{

        return hsqrt(x);
    }

    __device__ inline auto cuda_sqrt(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return sqrtf(x);
    }

    __device__ inline auto cuda_sqrt(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_sqrt(cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_sqrt(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_sqrt(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_sqrt_rd(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __fsqrt_rd(x);
    }

    __device__ inline auto cuda_sqrt_rd(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return __dsqrt_rd(x);
    }

    __device__ inline auto cuda_sqrt_rn(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __fsqrt_rn(x);
    }

    __device__ inline auto cuda_sqrt_rn(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return __dsqrt_rn(x);
    }

    __device__ inline auto cuda_sqrt_ru(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __fsqrt_ru(x);
    }

    __device__ inline auto cuda_sqrt_ru(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return __dsqrt_ru(x);
    }

    __device__ inline auto cuda_sqrt_rz(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __fsqrt_rz(x);
    }

    __device__ inline auto cuda_sqrt_rz(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return __dsqrt_rz(x);
    }

    __device__ inline auto cuda_invsqrt(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return rsqrt(x);
    }

    __device__ inline auto cuda_invsqrt(cuda_bfloat16_t x) noexcept -> cuda_bfloat16_t{

        return hrsqrt(x);
    }

    __device__ inline auto cuda_invsqrt(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return rsqrtf(x);
    }
    
    __device__ inline auto cuda_invsqrt(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_invsqrt(cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_invsqrt(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_invsqrt(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_invsqrt_rn(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

    }

    __device__ inline auto cuda_negative(cuda_stdfloat32_t) noexcept -> cuda_stdfloat32_t{

    }

    __device__ inline auto cuda_negative(cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    } 

    __device__ inline auto cuda_negative(cuda_stdfloat16_t) noexcept -> cuda_stdfloat16_t{

    }
 
    __device__ inline auto cuda_negative(cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_negative(cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_negative(cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_negative(cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_inverse(){

    }

    __device__ inline auto cuda_inverse_rd(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __frcp_rd(x);
    }

    __device__ inline auto cuda_inverse_rd(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return __drcp_rd(x);
    } 

    __device__ inline auto cuda_inverse_rn(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __frcp_rn(x);
    }

    __device__ inline auto cuda_inverse_rn(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return __drcp_rn(x);
    } 

    __device__ inline auto cuda_inverse_ru(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __frcp_ru(x);
    }

    __device__ inline auto cuda_inverse_ru(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return __drcp_ru(x);
    }

    __device__ inline auto cuda_inverse_rz(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __frcp_rz(x);
    } 

    __device__ inline auto cuda_inverse_rz(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return __drcp_rz(x);
    } 

    __device__ inline auto cuda_negate(cuda_bfloat16_t x) noexcept -> cuda_bfloat16_t{

        return __hneg(x);
    }

    __device__ inline auto cuda_add(cuda_stdfloat32_t, cuda_stdfloat32_t) noexcept -> cuda_stdfloat32_t{

    }

    __device__ inline auto cuda_add(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> cuda_bfloat16_t{

        return __hadd(lhs, rhs);
    }

    __device__ inline auto cuda_add(cuda_stdfloat16_t, cuda_stdfloat16_t) noexcept -> cuda_stdfloat16_t{

    }

    __device__ inline auto cuda_add(cuda_float8_t, cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_add(cuda_uint32_t, cuda_uint32_t) noexcept -> cuda_uint32_t{

    } 

    __device__ inline auto cuda_add(cuda_uint16_t, cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_add(cuda_uint8_t, cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_add_rn(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> cuda_bfloat16_t{

        return __hadd_rn(lhs, rhs);
    } 
    
    __device__ inline auto cuda_add_rn(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fadd_rn(lhs, rhs);
    }

    __device__ inline auto cuda_add_rn(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dadd_rn(lhs, rhs);
    } 

    __device__ inline auto cuda_add_ru(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fadd_ru(lhs, rhs);
    }

    __device__ inline auto cuda_add_ru(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dadd_ru(lhs, rhs);
    }

    __device__ inline auto cuda_add_rz(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fadd_rz(lhs, rhs);
    }

    __device__ inline auto cuda_add_rz(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dadd_rz(lhs, rhs);
    } 

    __device__ inline auto cuda_add_sat(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> cuda_bfloat16_t{

        return __hadd_sat(lhs, rhs);
    } 

    __device__ inline auto cuda_add_rd(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fadd_rd(lhs, rhs);
    } 

    __device__ inline auto cuda_add_rd(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dadd_rd(lhs, rhs);
    } 

    __device__ inline auto cuda_sub(cuda_stdfloat32_t, cuda_stdfloat32_t) noexcept -> cuda_stdfloat32_t{

    }

    __device__ inline auto cuda_sub(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> cuda_bfloat16_t{

        return __hsub(lhs, rhs);
    }

    __device__ inline auto cuda_sub(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

    }
    
    __device__ inline auto cuda_sub(cuda_float8_t, cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_sub(cuda_uint32_t, cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_sub(cuda_uint16_t, cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_sub(cuda_uint8_t, cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_floor(cuda_float16_t x) noexcept -> cuda_float16_t{

        return floorf(x);
    } 

    __device__ inline auto cuda_floor(cuda_float32_t x) noexcept -> cuda_float32_t{

        return floor(x);
    }

    __device__ inline auto cuda_sub_rd(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dsub_rd(lhs, rhs);
    } 

    __device__ inline auto cuda_sub_rn(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> cuda_bfloat16_t{

        return __hsub_rn(lhs, rhs);
    } 

    __device__ inline auto cuda_sub_rn(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fsub_rn(lhs, rhs);
    }

    __device__ inline auto cuda_sub_rn(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dsub_rn(lhs, rhs);
    }

    __device__ inline auto cuda_sub_ru(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fsub_ru(lhs, rhs);
    }

    __device__ inline auto cuda_sub_ru(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dsub_ru(lhs, rhs);
    }

    __device__ inline auto cuda_sub_rz(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fsub_rz(lhs, rhs);
    }

    __device__ inline auto cuda_sub_rz(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dsub_rz(lhs, rhs);
    } 

    __device__ inline auto cuda_sub_sat(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> cuda_bfloat16_t{

        return __hsub_sat(lhs, rhs);    
    }

    __device__ inline auto cuda_mul(cuda_stdfloat32_t, cuda_stdfloat32_t) noexcept -> cuda_stdfloat32_t{

    }

    __device__ inline auto cuda_mul(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> cuda_bfloat16_t{

        return __hmul(lhs, rhs);
    }

    __device__ inline auto cuda_mul(cuda_stdfloat16_t, cuda_stdfloat16_t) noexcept -> cuda_stdfloat16_t{

    }

    __device__ inline auto cuda_mul(cuda_float8_t, cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_mul(cuda_uint32_t, cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_mul(cuda_uint16_t, cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_mul(cuda_uint8_t, cuda_uint8_t) noexcept -> cuda_uint8_t{

    }
    
    __device__ inline auto cuda_mul_rd(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fmul_rd(lhs, rhs);
    }

    __device__ inline auto cuda_mul_rd(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dmul_rd(lhs, rhs);
    }

    __device__ inline auto cuda_mul_rn(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fmul_rn(lhs, rhs);
    }

    __device__ inline auto cuda_mul_rn(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dmul_rn(lhs, rhs);
    }

    __device__ inline auto cuda_mul_rn(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> cuda_bfloat16_t{

        return __hmul_rn(lhs, rhs);
    } 

    __device__ inline auto cuda_mul_ru(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fmul_ru(lhs, rhs);
    }

    __device__ inline auto cuda_mul_ru(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dmul_ru(lhs, rhs);
    } 

    __device__ inline auto cuda_mul_rz(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fmul_rz(lhs, rhs);
    }

    __device__ inline auto cuda_mul_rz(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __dmul_rz(lhs, rhs);
    }

    __device__ inline auto cuda_mul_sat(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> cuda_bfloat16_t{

        return __hmul_sat(lhs, rhs);
    } 

    __device__ inline auto cuda_div(cuda_stdfloat32_t, cuda_stdfloat32_t) noexcept -> cuda_stdfloat32_t{

    }

    __device__ inline auto cuda_div(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> cuda_bfloat16_t{

        return __hdiv(lhs, rhs);
    }

    __device__ inline auto cuda_div(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return fdividef(lhs, rhs);
    }

    __device__ inline auto cuda_div(cuda_float8_t, cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_div(cuda_uint32_t, cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_div(cuda_uint16_t, cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_div(cuda_uint8_t, cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_div_rd(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fdiv_rd(lhs, rhs);
    }

    __device__ inline auto cuda_div_rd(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __ddiv_rd(lhs, rhs);
    } 

    __device__ inline auto cuda_div_rn(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fdiv_rn(lhs, rhs);
    }

    __device__ inline auto cuda_div_rn(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __ddiv_rn(lhs, rhs);
    } 

    __device__ inline auto cuda_div_ru(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fdiv_ru(lhs, rhs);
    }

    __device__ inline auto cuda_div_ru(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __ddiv_ru(lhs, rhs);
    } 

    __device__ inline auto cuda_div_rz(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fdiv_rz(lhs, rhs);
    }

    __device__ inline auto cuda_div_rz(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return __ddiv_rz(lhs, rhs);
    }

    __device__ inline auto cuda_div_x(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __fdividef(lhs, rhs);
    }

    __device__ inline auto cuda_pow(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return pow(lhs, rhs);
    }

    __device__ inline auto cuda_pow(cuda_bfloat16_t, cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    }

    __device__ inline auto cuda_pow(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return powf(lhs, rhs);
    }

    __device__ inline auto cuda_pow(cuda_float8_t, cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_pow(cuda_uint32_t, cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_pow(cuda_uint16_t, cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_pow(cuda_uint8_t, cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_pow_x(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return __powf(lhs, rhs);
    }

    __device__ inline auto cuda_saturate_01(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return __saturatef(x);
    }

    __device__ inline auto cuda_rcbrt(cuda_stdfloat16_t x) noexcept -> cuda_stdfloat16_t{

        return rcbrtf(x);
    }

    __device__ inline auto cuda_rcbrt(cuda_stdfloat32_t x) noexcept -> cuda_stdfloat32_t{

        return rcbrt(x);
    }

    __device__ inline auto cuda_remainder(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return remainderf(lhs, rhs);
    }

    __device__ inline auto cuda_remainder(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return remainder(lhs, rhs);
    }

    __device__ inline auto cuda_rhypot(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return rhypotf(lhs, rhs);
    }

    __device__ inline auto cuda_rhypot(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return rhypot(lhs, rhs);
    }

    __device__ inline auto cuda_norm3d(cuda_stdfloat32_t a, cuda_stdfloat32_t b, cuda_stdfloat32_t c) noexcept -> cuda_stdfloat_32_t{

        return norm3d(a, b, c);
    } 
    
    __device__ inline auto cuda_rnorm3d(cuda_stdfloat16_t a, cuda_stdfloat16_t b, cuda_stdfloat16_t c) noexcept -> cuda_stdfloat16_t{

        return rnorm3df(a, b, c);
    }

    __device__ inline auto cuda_rnorm3d(cuda_stdfloat32_t a, cuda_stdfloat32_t b, cuda_stdfloat32_t c) noexcept -> cuda_stdfloat32_t{

        return rnorm3d(a, b, c);
    }

    __device__ inline auto cuda_norm4d(cuda_stdfloat32_t a, cuda_stdfloat32_t b, cuda_stdfloat32_t c, cuda_stdfloat32_t d) noexcept -> cuda_stdfloat32_t{

        return norm4d(a, b, c, d);
    } 

    __device__ inline auto cuda_rnorm4d(cuda_stdfloat16_t a, cuda_stdfloat16_t b, cuda_stdfloat16_t c, cuda_stdfloat16_t d) noexcept -> cuda_stdfloat16_t{

        return rnorm4df(a, b, c, d);
    }

    __device__ inline auto cuda_rnorm4d(cuda_stdfloat32_t a, cuda_stdfloat32_t b, cuda_stdfloat32_t c, cuda_stdfloat32_t d) noexcept -> cuda_stdfloat32_t{

        return rnorm4d(a, b, c, d);
    } 

    __device__ inline auto cuda_normcdf(cuda_stdfloat32_t x) noexcept{

        return normcdf(x);
    }

    __device__ inline auto cuda_normcdfinv(cuda_stdfloat32_t x) noexcept{

        return normcdfinv(x);
    }

    __device__ inline auto cuda_scalbln(cuda_stdfloat16_t lhs, cuda_int64_t rhs) noexcept -> cuda_stdfloat16_t{ //fix

        static_assert(sizeof(cuda_longint_t) == 8u);
        return scalblnf(lhs, rhs);
    }

    __device__ inline auto cuda_scalbln(cuda_stdfloat32_t lhs, cuda_int64_t rhs) noexcept -> cuda_stdfloat32_t{ //fix

        static_assert(sizeof(cuda_longint_t) == 8u);
        return scalbln(lhs, rhs);
    } 

    __device__ inline auto cuda_scalbn(cuda_stdfloat16_t lhs, cuda_int32_t rhs) noexcept -> cuda_stdfloat16_t{ //fix

        static_assert(sizeof(cuda_int_t) == 4u);
        return scalbnf(lhs, rhs);
    }

    __device__ inline auto cuda_scabn(cuda_stdfloat32_t lhs, cuda_int32_t rhs) noexcept -> cuda_stdfloat32_t{ //fix

        static_assert(sizeof(cuda_int_t) == 4u);
        return scalbn(lhs, rhs);
    } 

    __device__ inline auto cuda_fma(cuda_stdfloat32_t a, cuda_stdfloat32_t b, cuda_stdfloat32_t c) noexcept -> cuda_stdfloat32_t{

        return fma(a, b, c);
    }

    __device__ inline auto cuda_fma(cuda_bfloat16_t a, cuda_bfloat16_t b, cuda_bfloat16_t c) noexcept -> cuda_bfloat16_t{

        return __hfma(a, b, c);
    }

    __device__ inline auto cuda_fma(cuda_stdfloat16_t a, cuda_stdfloat16_t b, cuda_stdfloat16_t c) noexcept -> cuda_stdfloat16_t{

        return fmaf(a, b, c);
    }

    __device__ inline auto cuda_fma(cuda_float8_t, cuda_float8_t, cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_fma(cuda_uint32_t, cuda_uint32_t, cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_fma(cuda_uint16_t, cuda_uint16_t, cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_fma(cuda_uint8_t, cuda_uint8_t, cuda_uint8_t) noexcept -> cuda_uint8_t{

    } 

    __device__ inline auto cuda_fma_rd(cuda_stdfloat16_t a, cuda_stdfloat16_t b, cuda_stdfloat16_t c) noexcept -> cuda_stdfloat16_t{

        return __fmaf_rd(a, b, c);
    }

    __device__ inline auto cuda_fma_rd(cuda_stdfloat32_t a, cuda_stdfloat32_t b, cuda_stdfloat32_t c) noexcept -> cuda_stdfloat32_t{

        return __fma_rd(a, b, c);
    }

    __device__ inline auto cuda_fma_rn(cuda_stdfloat16_t a, cuda_stdfloat16_t b, cuda_stdfloat16_t c) noexcept -> cuda_stdfloat16_t{

        return __fmaf_rn(a, b, c);
    }

    __device__ inline auto cuda_fma_rn(cuda_stdfloat32_t a, cuda_stdfloat32_t b, cuda_stdfloat32_t c) noexcept -> cuda_stdfloat32_t{

        return __fma_rn(a, b, c);
    }

    __device__ inline auto cuda_fma_ru(cuda_stdfloat16_t a, cuda_stdfloat16_t b, cuda_stdfloat16_t c) noexcept -> cuda_stdfloat16_t{

        return __fmaf_ru(a, b, c);
    }

    __device__ inline auto cuda_fma_ru(cuda_stdfloat32_t a, cuda_stdfloat32_t b, cuda_stdfloat32_t c) noexcept -> cuda_stdfloat32_t{

        return __fma_ru(a, b, c);
    }

    __device__ inline auto cuda_fma_rz(cuda_stdfloat16_t a, cuda_stdfloat16_t b, cuda_stdfloat16_t c) noexcept -> cuda_stdfloat16_t{

        return __fmaf_rz(a, b, c);
    }

    __device__ inline auto cuda_fma_rz(cuda_stdfloat32_t a, cuda_stdfloat32_t b, cuda_stdfloat32_t c) noexcept -> cuda_stdfloat32_t{

        return __fma_rz(a, b, c);
    }

    __device__ inline auto cuda_fma_ieee_rd(cuda_stdfloat16_t a, cuda_stdfloat16_t b, cuda_stdfloat16_t c) noexcept -> cuda_stdfloat16_t{

        return __fmaf_ieee_rd(a, b, c);
    } 

    __device__ inline auto cuda_fma_ieee_rn(cuda_stdfloat16_t a, cuda_stdfloat16_t b, cuda_stdfloat16_t c) noexcept -> cuda_stdfloat16_t{

        return __fmaf_ieee_rn(a, b, c);
    }

    __device__ inline auto cuda_fma_ieee_ru(cuda_stdfloat16_t a, cuda_stdfloat16_t b, cuda_stdfloat16_t c) noexcept -> cuda_stdfloat16_t{

        return __fmaf_ieee_ru(a, b, c);
    }

    __device__ inline auto cuda_fma_ieee_rz(cuda_stdfloat16_t a, cuda_stdfloat16_t c, cuda_stdfloat16_t c) noexcept -> cuda_stdfloat16_t{

        return __fmaf_ieee_rz(a, b, c);
    }

    __device__ inline auto cuda_fma_relu(cuda_bfloat16_t a, cuda_bfloat16_t b, cuda_bfloat16_t c) noexcept -> cuda_bfloat16_t{

        return __hfma_relu(a, b, c);
    } 

    __device__ inline auto cuda_fma_sat(cuda_bfloat16_t a, cuda_bfloat16_t b, cuda_bfloat16_t c) noexcept -> cuda_bfloat16_t{

        return __hfma_sat(a, b, c);
    } 

    __device__ inline auto cuda_mod(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return fmodf(lhs, rhs);
    }

    __device__ inline auto cuda_mod(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return fmod(lhs, rhs);
    } 

    __device__ inline auto cuda_hypot(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return hypotf(lhs, rhs);
    }

    __device__ inline auto cuda_hypot(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return hypot(lhs, rhs);
    }

    __device__ inline auto cuda_min(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return min(lhs, rhs);
    }

    __device__ inline auto cuda_min(cuda_stdfloat16_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return min(lhs, rhs);
    }

    __device__ inline auto cuda_min(cuda_stdfloat32_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat32_t{

        return min(lhs, rhs);
    }

    __device__ inline auto cuda_min(cuda_bfloat16_t, cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    }

    __device__ inline auto cuda_min(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return fminf(lhs, rhs);
    }

    __device__ inline auto cuda_min(cuda_float8_t, cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_min(cuda_uint32_t, cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_min(cuda_uint16_t, cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_min(cuda_uint8_t, cuda_uint8_t) noexcept -> cuda_uint8_t{
        
    }

    __device__ inline auto cuda_min_nan(cuda_bfloat16_t, cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    }

    __device__ inline auto cuda_max(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return max(lhs, rhs);
    }

    __device__ inline auto cuda_max(cuda_stdfloat16_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return max(lhs, rhs);
    }

    __device__ inline auto cuda_max(cuda_stdfloat32_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat32_t{

        return max(lhs, rhs);
    }

    __device__ inline auto cuda_max(cuda_bfloat16_t, cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    }

    __device__ inline auto cuda_max(cuda_stdfloat16_t lhs, cuda_stdfloat16_t rhs) noexcept -> cuda_stdfloat16_t{

        return fmaxf(lhs, rhs);
    }

    __device__ inline auto cuda_max(cuda_stdfloat32_t lhs, cuda_stdfloat32_t rhs) noexcept -> cuda_stdfloat32_t{

        return fmax(lhs, rhs);
    } 

    __device__ inline auto cuda_max(cuda_float8_t, cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_max(cuda_uint32_t, cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_max(cuda_uint16_t, cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_max(cuda_uint8_t, cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_max_nan(cuda_bfloat16_t, cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    } 

    __device__ inline auto cuda_eqcmp_mul(cuda_stdfloat32_t, cuda_stdfloat32_t, cuda_stdfloat32_t) noexcept -> cuda_stdfloat32_t{

    }

    __device__ inline auto cuda_eqcmp_mul(cuda_bfloat16_t, cuda_bfloat16_t, cuda_bfloat16_t) noexcept -> cuda_bfloat16_t{

    }

    __device__ inline auto cuda_eqcmp_mul(cuda_stdfloat16_t, cuda_stdfloat16_t, cuda_stdfloat16_t) noexcept -> cuda_stdfloat16_t{

    }

    __device__ inline auto cuda_eqcmp_mul(cuda_float8_t, cuda_float8_t, cuda_float8_t) noexcept -> cuda_float8_t{

    }

    __device__ inline auto cuda_eqcmp_mul(cuda_uint32_t, cuda_uint32_t, cuda_uint32_t) noexcept -> cuda_uint32_t{

    }

    __device__ inline auto cuda_eqcmp_mul(cuda_uint16_t, cuda_uint16_t, cuda_uint16_t) noexcept -> cuda_uint16_t{

    }

    __device__ inline auto cuda_eqcmp_mul(cuda_uint8_t, cuda_uint8_t, cuda_uint8_t) noexcept -> cuda_uint8_t{

    }

    __device__ inline auto cuda_eq(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __heq(lhs, rhs);
    }

    __device__ inline auto cuda_equ(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hequ(lhs, rhs);
    }

    __device__ inline auto cuda_ge(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hge(lhs, rhs);
    }

    __device__ inline auto cuda_geu(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hgeu(lhs, rhs);
    }

    __device__ inline auto cuda_gt(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hgt(lhs, rhs);
    }

    __device__ inline auto cuda_gtu(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hgtu(lhs, rhs);
    }

    __device__ inline auto cuda_isfinite(cuda_stdfloat16_t x) noexcept -> bool{

        return isfinite(x);
    } 

    __device__ inline auto cuda_isfinite(cuda_stdfloat32_t x) noexcept -> bool{

        return isfinite(x);
    }

    __device__ inline auto cuda_isinf(cuda_bfloat16_t x) noexcept -> bool{

        return __hisinf(x);
    }

    __device__ inline auto cuda_isinf(cuda_stdfloat32_t x) noexcept -> bool{

        return isinf(x);
    } 

    __device__ inline auto cuda_isinf(cuda_stdfloat16_t x) noexcept -> bool{

        return isinf(x);
    }

    __device__ inline auto cuda_isnan(cuda_stdfloat16_t x) noexcept -> bool{

        return isnan(x);
    } 

    __device__ inline auto cuda_isnan(cuda_bfloat16_t x) noexcept -> bool{

        return __hisnan(x);
    }

    __device__ inline auto cuda_isnan(cuda_stdfloat32_t x) noexcept -> bool{

        return isnan(x);
    } 

    __device__ inline auto cuda_le(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hle(lhs, rhs);
    }

    __device__ inline auto cuda_leu(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hleu(lhs, rhs);
    }

    __device__ inline auto cuda_lt(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hlt(lhs, rhs);
    }

    __device__ inline auto cuda_ltu(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hltu(lhs, rhs);
    }

    __device__ inline auto cuda_ne(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hne(lsh, rhs);
    }

    __device__ inline auto cuda_neu(cuda_bfloat16_t lhs, cuda_bfloat16_t rhs) noexcept -> bool{

        return __hneu(lhs, rhs);
    }
}

#endif