#ifndef __NETWORK_XMATH_HOST_H__
#define __NETWORK_XMATH_HOST_H__

#include <stddef.h>
#include <stdint.h>
#include <stdfloat>

namespace dg::network_x_math{
     
    inline auto sign(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto exp(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto ln(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto abs(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto cos(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto acos(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto sin(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto asin(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto tan(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto atan(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto sqrt(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    }

    inline auto invsqrt() noexcept{

    }

    inline auto negative(std::bfloat16_t value) noexcept -> std::bfloat16_t{

    } 

    inline auto add(std::bfloat16_t lhs, std::bfloat16_t rhs) noexcept -> std::bfloat16_t{

    }

    inline auto sub(std::bfloat16_t lhs, std::bfloat16_t rhs) noexcept -> std::bfloat16_t{

    }

    inline auto mul(std::bfloat16_t lhs, std::bfloat16_t rhs) noexcept -> std::bfloat16_t{

    }

    inline auto div(std::bfloat16_t lhs, std::bfloat16_t rhs) noexcept -> std::bfloat16_t{

    }

    inline auto pow(std::bfloat16_t lhs, std::bfloat16_t rhs) noexcept -> std::bfloat16_t{

    }

    template <size_t RHS_VALUE>
    inline auto pow(std::bfloat16_t lhs, const std::integral_constant<size_t, RHS_VALUE>) noexcept -> arithmetic_ops_t{

    }

    inline auto fma(std::bfloat16_t first, std::bfloat16_t second, std::bfloat16_t third){

    }

    inline auto min(std::bfloat16_t lhs, std::bfloat16_t rhs) noexcept -> std::bfloat16_t{

    }

    inline auto max(std::bfloat16_t lhs, std::bfloat16_t rhs) noexcept -> std::bfloat16_t{

    }

    inline auto eqcmp_mul(std::bfloat16_t lcmp, std::bfloat16_t rcmp, std::bfloat16_t val) -> std::bfloat16_t{

    }
} 

#endif