#ifndef __DG_NETWORK_BYTECODE_H__
#define __DG_NETWORK_BYTECODE_H__

#include <stdint.h>
#include <stdlib.h>
#include <algorithm>
#include <memory>
#include <array>
#include "network_trivial_serializer.h"

namespace dg::network_bytecode
{
    struct bad_instruction : std::exception{};
    struct outofbound_memaccess : std::exception{};
    struct outofbound_instruction : std::exception{};
    struct context_allocation_overflow : std::exception{}; 
    struct context_allocation_underflow : std::exception{};

    static inline constexpr char ALLOCATE_MEMSET_INSTRUCTION            = std::bit_cast<char>(static_cast<uint8_t>(0u));
    static inline constexpr char DEALLOCATE_RANGE_INSTRUCTION           = std::bit_cast<char>(static_cast<uint8_t>(1));

    static inline constexpr char ASSIGN_CONST_1_INSTRUCTION             = std::bit_cast<char>(static_cast<uint8_t>(2));
    static inline constexpr char ASSIGN_CONST_2_INSTRUCTION             = std::bit_cast<char>(static_cast<uint8_t>(3));
    static inline constexpr char ASSIGN_CONST_4_INSTRUCTION             = std::bit_cast<char>(static_cast<uint8_t>(4));
    static inline constexpr char ASSIGN_CONST_8_INSTRUCTION             = std::bit_cast<char>(static_cast<uint8_t>(5));
    static inline constexpr char ASSIGN_RANGE_INSTRUCTION               = std::bit_cast<char>(static_cast<uint8_t>(6));

    static inline constexpr char GET_ADDR_INSTRUCTION                   = std::bit_cast<char>(static_cast<uint8_t>(7));

    static inline constexpr char ADD_UINT64_INSTRUCTION                 = std::bit_cast<char>(static_cast<uint8_t>(8));
    static inline constexpr char ADD_INT64_INSTRUCTION                  = std::bit_cast<char>(static_cast<uint8_t>(9));
    
    static inline constexpr char ADD_FLOAT_INSTRUCTION                  = std::bit_cast<char>(static_cast<uint8_t>(10));
    static inline constexpr char ADD_DOUBLE_INSTRUCTION                 = std::bit_cast<char>(static_cast<uint8_t>(11));

    static inline constexpr char SUB_UINT64_INSTRUCTION                 = std::bit_cast<char>(static_cast<uint8_t>(12));
    static inline constexpr char SUB_INT64_INSTRUCTION                  = std::bit_cast<char>(static_cast<uint8_t>(13));

    static inline constexpr char SUB_FLOAT_INSTRUCTION                  = std::bit_cast<char>(static_cast<uint8_t>(14));
    static inline constexpr char SUB_DOUBLE_INSTRUCTION                 = std::bit_cast<char>(static_cast<uint8_t>(15));

    static inline constexpr char MUL_UINT64_INSTRUCTION                 = std::bit_cast<char>(static_cast<uint8_t>(16));
    static inline constexpr char MUL_INT64_INSTRUCTION                  = std::bit_cast<char>(static_cast<uint8_t>(17));

    static inline constexpr char MUL_FLOAT_INSTRUCTION                  = std::bit_cast<char>(static_cast<uint8_t>(18));
    static inline constexpr char MUL_DOUBLE_INSTRUCTION                 = std::bit_cast<char>(static_cast<uint8_t>(19));

    static inline constexpr char DIV_UINT64_INSTRUCTION                 = std::bit_cast<char>(static_cast<uint8_t>(20));
    static inline constexpr char DIV_INT64_INSTRUCTION                  = std::bit_cast<char>(static_cast<uint8_t>(21));
    
    static inline constexpr char DIV_FLOAT_INSTRUCTION                  = std::bit_cast<char>(static_cast<uint8_t>(22));
    static inline constexpr char DIV_DOUBLE_INSTRUCTION                 = std::bit_cast<char>(static_cast<uint8_t>(23));

    static inline constexpr char MOD_UINT64_INSTRUCTION                 = std::bit_cast<char>(static_cast<uint8_t>(24));
    static inline constexpr char MOD_INT64_INSTRUCTION                  = std::bit_cast<char>(static_cast<uint8_t>(25));

    static inline constexpr char AND_BOOL_INSTRUCTION                   = std::bit_cast<char>(static_cast<uint8_t>(26));
    static inline constexpr char OR_BOOL_INSTRUCTION                    = std::bit_cast<char>(static_cast<uint8_t>(27));
    static inline constexpr char NOT_BOOL_INSTRUCTION                   = std::bit_cast<char>(static_cast<uint8_t>(28));

    static inline constexpr char BW_AND_UINT64_INSTRUCTION              = std::bit_cast<char>(static_cast<uint8_t>(29));
    static inline constexpr char BW_OR_UINT64_INSTRUCTION               = std::bit_cast<char>(static_cast<uint8_t>(30));
    static inline constexpr char BW_XOR_UINT64_INSTRUCTION              = std::bit_cast<char>(static_cast<uint8_t>(31));
    static inline constexpr char BW_NOT_UINT64_INSTRUCTION              = std::bit_cast<char>(static_cast<uint8_t>(32));

    static inline constexpr char BW_LEFTSHIFT_UINT64_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(33));
    static inline constexpr char BW_RIGHTSHIFT_UINT64_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(34));

    static inline constexpr char CMP_LESS_UINT64_INSTRUCTION            = std::bit_cast<char>(static_cast<uint8_t>(35));
    static inline constexpr char CMP_LESS_INT64_INSTRUCTION             = std::bit_cast<char>(static_cast<uint8_t>(36));
    static inline constexpr char CMP_LESS_FLOAT_INSTRUCTION             = std::bit_cast<char>(static_cast<uint8_t>(37));
    static inline constexpr char CMP_LESS_DOUBLE_INSTRUCTION            = std::bit_cast<char>(static_cast<uint8_t>(38));

    static inline constexpr char CMP_GREATER_UINT64_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(39));
    static inline constexpr char CMP_GREATER_INT64_INSTRUCTION          = std::bit_cast<char>(static_cast<uint8_t>(40));
    static inline constexpr char CMP_GREATER_FLOAT_INSTRUCTION          = std::bit_cast<char>(static_cast<uint8_t>(41));
    static inline constexpr char CMP_GREATER_DOUBLE_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(42));

    static inline constexpr char CMP_EQUAL_UINT64_INSTRUCTION           = std::bit_cast<char>(static_cast<uint8_t>(43));
    static inline constexpr char CMP_EQUAL_INT64_INSTRUCTION            = std::bit_cast<char>(static_cast<uint8_t>(44));
    static inline constexpr char CMP_EQUAL_FLOAT_INSTRUCTION            = std::bit_cast<char>(static_cast<uint8_t>(45));
    static inline constexpr char CMP_EQUAL_DOUBLE_INSTRUCTION           = std::bit_cast<char>(static_cast<uint8_t>(46));
   
    static inline constexpr char CAST_UINT8_TO_BOOL_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(47));
    static inline constexpr char CAST_UINT8_TO_UINT8_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(48));
    static inline constexpr char CAST_UINT8_TO_UINT16_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(49));
    static inline constexpr char CAST_UINT8_TO_UINT32_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(50));
    static inline constexpr char CAST_UINT8_TO_UINT64_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(51));
    static inline constexpr char CAST_UINT8_TO_INT8_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(52));
    static inline constexpr char CAST_UINT8_TO_INT16_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(53));
    static inline constexpr char CAST_UINT8_TO_INT32_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(54));
    static inline constexpr char CAST_UINT8_TO_INT64_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(55));
    static inline constexpr char CAST_UINT8_TO_FLOAT_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(56));
    static inline constexpr char CAST_UINT8_TO_DOUBLE_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(57));

    static inline constexpr char CAST_UINT16_TO_BOOL_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(58));
    static inline constexpr char CAST_UINT16_TO_UINT8_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(59));
    static inline constexpr char CAST_UINT16_TO_UINT16_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(60));
    static inline constexpr char CAST_UINT16_TO_UINT32_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(61));
    static inline constexpr char CAST_UINT16_TO_UINT64_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(62));
    static inline constexpr char CAST_UINT16_TO_INT8_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(63));
    static inline constexpr char CAST_UINT16_TO_INT16_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(64));
    static inline constexpr char CAST_UINT16_TO_INT32_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(65));
    static inline constexpr char CAST_UINT16_TO_INT64_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(66));
    static inline constexpr char CAST_UINT16_TO_FLOAT_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(67));
    static inline constexpr char CAST_UINT16_TO_DOUBLE_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(68));

    static inline constexpr char CAST_UINT32_TO_BOOL_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(69));
    static inline constexpr char CAST_UINT32_TO_UINT8_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(70));
    static inline constexpr char CAST_UINT32_TO_UINT16_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(71));
    static inline constexpr char CAST_UINT32_TO_UINT32_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(72));
    static inline constexpr char CAST_UINT32_TO_UINT64_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(73));
    static inline constexpr char CAST_UINT32_TO_INT8_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(74));
    static inline constexpr char CAST_UINT32_TO_INT16_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(75));
    static inline constexpr char CAST_UINT32_TO_INT32_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(76));
    static inline constexpr char CAST_UINT32_TO_INT64_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(77));
    static inline constexpr char CAST_UINT32_TO_FLOAT_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(78));
    static inline constexpr char CAST_UINT32_TO_DOUBLE_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(79));

    static inline constexpr char CAST_UINT64_TO_BOOL_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(80));
    static inline constexpr char CAST_UINT64_TO_UINT8_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(81));
    static inline constexpr char CAST_UINT64_TO_UINT16_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(82));
    static inline constexpr char CAST_UINT64_TO_UINT32_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(83));
    static inline constexpr char CAST_UINT64_TO_UINT64_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(84));
    static inline constexpr char CAST_UINT64_TO_INT8_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(85));
    static inline constexpr char CAST_UINT64_TO_INT16_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(86));
    static inline constexpr char CAST_UINT64_TO_INT32_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(87));
    static inline constexpr char CAST_UINT64_TO_INT64_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(88));
    static inline constexpr char CAST_UINT64_TO_FLOAT_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(89));
    static inline constexpr char CAST_UINT64_TO_DOUBLE_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(90));

    static inline constexpr char CAST_INT8_TO_BOOL_INSTRUCTION          = std::bit_cast<char>(static_cast<uint8_t>(91));
    static inline constexpr char CAST_INT8_TO_UINT8_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(92));
    static inline constexpr char CAST_INT8_TO_UINT16_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(93));
    static inline constexpr char CAST_INT8_TO_UINT32_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(94));
    static inline constexpr char CAST_INT8_TO_UINT64_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(95));
    static inline constexpr char CAST_INT8_TO_INT8_INSTRUCTION          = std::bit_cast<char>(static_cast<uint8_t>(96));
    static inline constexpr char CAST_INT8_TO_INT16_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(97));
    static inline constexpr char CAST_INT8_TO_INT32_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(98));
    static inline constexpr char CAST_INT8_TO_INT64_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(99));
    static inline constexpr char CAST_INT8_TO_FLOAT_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(100));
    static inline constexpr char CAST_INT8_TO_DOUBLE_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(101));

    static inline constexpr char CAST_INT16_TO_BOOL_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(102));
    static inline constexpr char CAST_INT16_TO_UINT8_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(103));
    static inline constexpr char CAST_INT16_TO_UINT16_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(104));
    static inline constexpr char CAST_INT16_TO_UINT32_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(105));
    static inline constexpr char CAST_INT16_TO_UINT64_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(106));
    static inline constexpr char CAST_INT16_TO_INT8_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(107));
    static inline constexpr char CAST_INT16_TO_INT16_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(108));
    static inline constexpr char CAST_INT16_TO_INT32_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(109));
    static inline constexpr char CAST_INT16_TO_INT64_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(110));
    static inline constexpr char CAST_INT16_TO_FLOAT_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(111));
    static inline constexpr char CAST_INT16_TO_DOUBLE_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(112));

    static inline constexpr char CAST_INT32_TO_BOOL_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(113));
    static inline constexpr char CAST_INT32_TO_UINT8_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(114));
    static inline constexpr char CAST_INT32_TO_UINT16_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(115));
    static inline constexpr char CAST_INT32_TO_UINT32_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(116));
    static inline constexpr char CAST_INT32_TO_UINT64_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(117));
    static inline constexpr char CAST_INT32_TO_INT8_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(118));
    static inline constexpr char CAST_INT32_TO_INT16_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(119));
    static inline constexpr char CAST_INT32_TO_INT32_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(120));
    static inline constexpr char CAST_INT32_TO_INT64_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(121));
    static inline constexpr char CAST_INT32_TO_FLOAT_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(122));
    static inline constexpr char CAST_INT32_TO_DOUBLE_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(123));

    static inline constexpr char CAST_INT64_TO_BOOL_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(124));
    static inline constexpr char CAST_INT64_TO_UINT8_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(125));
    static inline constexpr char CAST_INT64_TO_UINT16_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(126));
    static inline constexpr char CAST_INT64_TO_UINT32_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(127));
    static inline constexpr char CAST_INT64_TO_UINT64_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(128));
    static inline constexpr char CAST_INT64_TO_INT8_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(129));
    static inline constexpr char CAST_INT64_TO_INT16_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(130));
    static inline constexpr char CAST_INT64_TO_INT32_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(131));
    static inline constexpr char CAST_INT64_TO_INT64_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(132));
    static inline constexpr char CAST_INT64_TO_FLOAT_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(133));
    static inline constexpr char CAST_INT64_TO_DOUBLE_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(134));

    static inline constexpr char CAST_FLOAT_TO_BOOL_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(135));
    static inline constexpr char CAST_FLOAT_TO_UINT8_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(136));
    static inline constexpr char CAST_FLOAT_TO_UINT16_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(137));
    static inline constexpr char CAST_FLOAT_TO_UINT32_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(138));
    static inline constexpr char CAST_FLOAT_TO_UINT64_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(139));
    static inline constexpr char CAST_FLOAT_TO_INT8_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(140));
    static inline constexpr char CAST_FLOAT_TO_INT16_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(141));
    static inline constexpr char CAST_FLOAT_TO_INT32_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(142));
    static inline constexpr char CAST_FLOAT_TO_INT64_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(143));
    static inline constexpr char CAST_FLOAT_TO_FLOAT_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(144));
    static inline constexpr char CAST_FLOAT_TO_DOUBLE_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(145));

    static inline constexpr char CAST_DOUBLE_TO_BOOL_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(146));
    static inline constexpr char CAST_DOUBLE_TO_UINT8_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(147));
    static inline constexpr char CAST_DOUBLE_TO_UINT16_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(148));
    static inline constexpr char CAST_DOUBLE_TO_UINT32_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(149));
    static inline constexpr char CAST_DOUBLE_TO_UINT64_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(150));
    static inline constexpr char CAST_DOUBLE_TO_INT8_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(151));
    static inline constexpr char CAST_DOUBLE_TO_INT16_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(152));
    static inline constexpr char CAST_DOUBLE_TO_INT32_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(153));
    static inline constexpr char CAST_DOUBLE_TO_INT64_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(154));
    static inline constexpr char CAST_DOUBLE_TO_FLOAT_INSTRUCTION       = std::bit_cast<char>(static_cast<uint8_t>(155));
    static inline constexpr char CAST_DOUBLE_TO_DOUBLE_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(156));

    static inline constexpr char TEST_THEN_JUMP_INSTRUCTION             = std::bit_cast<char>(static_cast<uint8_t>(157));
    static inline constexpr char TABLE_DISPATCH_INSTRUCTION             = std::bit_cast<char>(static_cast<uint8_t>(158));

    static inline constexpr char ENDIAN_ASSIGN_CONST_1_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(159));
    static inline constexpr char ENDIAN_ASSIGN_CONST_2_INSTRUCITON      = std::bit_cast<char>(static_cast<uint8_t>(160));
    static inline constexpr char ENDIAN_ASSIGN_CONST_4_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(161));
    static inline constexpr char ENDIAN_ASSIGN_CONST_8_INSTRUCTION      = std::bit_cast<char>(static_cast<uint8_t>(162));

    static inline constexpr char CAST_BOOL_TO_BOOL_INSTRUCTION          = std::bit_cast<char>(static_cast<uint8_t>(163));
    static inline constexpr char CAST_BOOL_TO_UINT8_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(164));
    static inline constexpr char CAST_BOOL_TO_UINT16_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(165));
    static inline constexpr char CAST_BOOL_TO_UINT32_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(166));
    static inline constexpr char CAST_BOOL_TO_UINT64_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(167));
    static inline constexpr char CAST_BOOL_TO_INT8_INSTRUCTION          = std::bit_cast<char>(static_cast<uint8_t>(168));
    static inline constexpr char CAST_BOOL_TO_INT16_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(169));
    static inline constexpr char CAST_BOOL_TO_INT32_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(170));
    static inline constexpr char CAST_BOOL_TO_INT64_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(171));
    static inline constexpr char CAST_BOOL_TO_FLOAT_INSTRUCTION         = std::bit_cast<char>(static_cast<uint8_t>(172));
    static inline constexpr char CAST_BOOL_TO_DOUBLE_INSTRUCTION        = std::bit_cast<char>(static_cast<uint8_t>(173));

    static_assert(std::numeric_limits<float>::is_iec559);
    static_assert(std::numeric_limits<double>::is_iec559);
    static_assert(dg::network_trivial_serializer::constants::endianness == std::endian::little);

    static inline constexpr uint64_t MIN_CONTEXT_SIZE                   = uint64_t{0u};
    static inline constexpr uint64_t MAX_CONTEXT_SIZE                   = uint64_t{1} << 30;

    using unsigned_addr_t = uint64_t;
    using range_t = uint32_t; 
    using offset_t = uint32_t;

    struct Context
    {
        char * buffer;
        size_t buffer_sz;
        size_t buffer_offset;
    };

    struct AllocateMemsetInstruction
    {
        range_t allocation_sz;
        char byteset_char;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept
        {
            reflector(allocation_sz, byteset_char);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept
        {
            reflector(allocation_sz, byteset_char);
        }
    };

    struct DeallocateRangeInstruction
    {
        range_t deallocation_sz;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept
        {
            reflector(deallocation_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept
        {
            reflector(deallocation_sz);
        }
    };

    template <size_t DATA_SZ>
    struct AssignConstInstruction
    {
        offset_t lhs_addr_var_back_offset;
        std::array<char, DATA_SZ> data;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept
        {
            reflector(lhs_addr_var_back_offset, data);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept
        {
            reflector(lhs_addr_var_back_offset, data);
        }
    };

    struct AssignRangeInstruction
    {
        offset_t lhs_addr_var_back_offset;
        offset_t rhs_addr_var_back_offset;
        offset_t assign_sz_addr_var_back_offset;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept
        {
            reflector(lhs_addr_var_back_offset, rhs_addr_var_back_offset, assign_sz_addr_var_back_offset);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept
        {
            reflector(lhs_addr_var_back_offset, rhs_addr_var_back_offset, assign_sz_addr_var_back_offset);
        }
    };

    struct GetAddressInstruction
    {
        offset_t dst_var_back_offset;
        offset_t src_var_back_offset;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept
        {
            reflector(dst_var_back_offset, src_var_back_offset);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept
        {
            reflector(dst_var_back_offset, src_var_back_offset);
        }
    };

    struct MonoInstruction
    {
        offset_t dst_addr_var_back_offset;
        offset_t src_addr_var_back_offset;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept
        {
            reflector(dst_addr_var_back_offset, src_addr_var_back_offset);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept
        {
            reflector(dst_addr_var_back_offset, src_addr_var_back_offset);
        }
    };

    struct PairInstruction
    {
        offset_t dst_addr_var_back_offset;
        offset_t lhs_addr_var_back_offset;
        offset_t rhs_addr_var_back_offset;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept
        {
            reflector(dst_addr_var_back_offset, lhs_addr_var_back_offset, rhs_addr_var_back_offset);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept
        {
            reflector(dst_addr_var_back_offset, lhs_addr_var_back_offset, rhs_addr_var_back_offset);
        }
    };

    struct JumpInstruction
    {
        offset_t test_addr_var_back_offset;
        offset_t test_true_instruction_global_offset_var_back_offset;
        offset_t test_false_instruction_global_offset_var_back_offset;

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept
        {
            reflector(test_addr_var_back_offset,
                      test_true_instruction_global_offset_var_back_offset,
                      test_false_instruction_global_offset_var_back_offset);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept
        {
            reflector(test_addr_var_back_offset,
                      test_true_instruction_global_offset_var_back_offset,
                      test_false_instruction_global_offset_var_back_offset);
        }
    };

    template <class T>
    struct EndiannessAwaredRepresentativeType{};

    template <>
    struct EndiannessAwaredRepresentativeType<std::integral_constant<size_t, 1>>
    {
        using type = uint8_t;
    };

    template <>
    struct EndiannessAwaredRepresentativeType<std::integral_constant<size_t, 2>>
    {
        using type = uint16_t;
    };

    template <>
    struct EndiannessAwaredRepresentativeType<std::integral_constant<size_t, 4>>
    {
        using type = uint32_t;
    };

    template <>
    struct EndiannessAwaredRepresentativeType<std::integral_constant<size_t, 8>>
    {
        using type = uint64_t;
    };

    template <class T>
    T * safe_ptr_access(T * ptr) noexcept
    {
        if (ptr == nullptr) [[unlikely]]
        {
            std::abort();
        }
        else [[likely]]
        {
            return ptr;
        }
    }

    auto make_context(size_t context_stack_size) noexcept -> std::expected<Context *, exception_t>
    {
        if (std::clamp(static_cast<uint64_t>(context_stack_size), MIN_CONTEXT_SIZE, MAX_CONTEXT_SIZE) != context_stack_size)
        {
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        char * buffer = static_cast<char *>(std::malloc(context_stack_size));

        if (buffer == nullptr)
        {
            return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
        }

        Context * ctx;

        try
        {
            ctx = new Context(Context{buffer, context_stack_size, 0u});
        }
        catch (...)
        {
            std::free(buffer);
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }

        return ctx;
    }

    void deallocate_context(Context * context) noexcept
    {
        context = safe_ptr_access(context);

        std::free(context->buffer);
        delete context;
    }

    bool is_sub_segment(uintptr_t sub_addr_first, uintptr_t sub_addr_last,
                        uintptr_t addr_first, uintptr_t addr_last)
    {
        if (sub_addr_first < addr_first || sub_addr_last > addr_last) [[unlikely]]
        {
            return false;
        }
        else [[likely]]
        {
            return true;
        }
    } 

    char * context_get_first_addr(Context * context)
    {
        return safe_ptr_access(context)->buffer;
    }

    char * context_get_last_addr(Context * context)
    {
        return std::next(safe_ptr_access(context)->buffer, context->buffer_offset);
    }

    char * context_boundsafe_get_back_offset(Context * context, size_t back_offset)
    {
        if (back_offset + 1u > safe_ptr_access(context)->buffer_offset) [[unlikely]] 
        {
            throw outofbound_memaccess();
        }
        else [[likely]]
        {
            return std::next(context->buffer, context->buffer_offset - 1u - back_offset);
        }
    }

    void throw_bound(Context * context,
                     void * addr, size_t bsz)
    {
        uintptr_t ctx_first     = reinterpret_cast<uintptr_t>(context_get_first_addr(context));
        uintptr_t ctx_last      = reinterpret_cast<uintptr_t>(context_get_last_addr(context));
        uintptr_t addr_first    = reinterpret_cast<uintptr_t>(addr);
        uintptr_t addr_last     = addr_first + bsz;

        if (!is_sub_segment(addr_first, addr_last, ctx_first, ctx_last)) [[unlikely]]
        {
            throw outofbound_memaccess();
        }
    }

    void boundsafe_memcpy(Context * context,
                          void * dst, void * src, size_t bsz)
    {
        throw_bound(context, dst, bsz);

        std::memcpy(dst, safe_ptr_access(src), bsz);
    }

    void bi_boundsafe_memcpy(Context * context,
                             void * dst, void * src, size_t bsz)
    {
        throw_bound(context, dst, bsz);
        throw_bound(context, src, bsz);

        std::memcpy(dst, src, bsz);
    }

    template <class T>
    T boundsafe_load(Context * context, void * src)
    {
        static_assert(std::is_trivially_copyable_v<T>);

        constexpr size_t bsz = sizeof(T);

        throw_bound(context, src, bsz);
        T rs;
        std::memcpy(&rs, src, bsz);

        return rs;
    }

    void exec_allocate_memset(const char * const instruction_ptr,
                              size_t& instruction_offset,
                              const size_t instruction_sz,
                              Context * context)
    {
        AllocateMemsetInstruction instruction;
        constexpr size_t INCREMENTAL_SZ = dg::network_trivial_serializer::size(AllocateMemsetInstruction{});

        if (instruction_offset + INCREMENTAL_SZ > instruction_sz) [[unlikely]]
        {
            throw outofbound_instruction{};
        }
        else [[likely]]
        {
            dg::network_trivial_serializer::deserialize_into(instruction, std::next(safe_ptr_access(instruction_ptr), instruction_offset));
            instruction_offset  += INCREMENTAL_SZ;
            char * write_ptr    = context_get_last_addr(context);
            size_t new_sz       = context->buffer_offset + instruction.allocation_sz;

            if (new_sz > context->buffer_sz) [[unlikely]]
            {
                throw context_allocation_overflow{};
            }
            else [[likely]]
            {
                context->buffer_offset += instruction.allocation_sz;
                std::memset(write_ptr, instruction.byteset_char, instruction.allocation_sz);
            }
        }
    }

    void exec_deallocate_range(const char * const instruction_ptr,
                               size_t& instruction_offset,
                               const size_t instruction_sz,
                               Context * context)
    {
        DeallocateRangeInstruction instruction;
        constexpr size_t INCREMENTAL_SZ = dg::network_trivial_serializer::size(DeallocateRangeInstruction{});

        if (instruction_offset + INCREMENTAL_SZ > instruction_sz) [[unlikely]]
        {
            throw outofbound_instruction{};
        }
        else [[likely]]
        {
            dg::network_trivial_serializer::deserialize_into(instruction, std::next(safe_ptr_access(instruction_ptr), instruction_offset));
            instruction_offset += INCREMENTAL_SZ;
            
            if (instruction.deallocation_sz > context->buffer_offset) [[unlikely]]
            {
                throw context_allocation_underflow{};
            }
            else [[likely]]
            {
                context->buffer_offset -= instruction.deallocation_sz;
            }
        }
    }

    template <size_t SZ>
    void exec_assign_const(const char * const instruction_ptr,
                           size_t& instruction_offset,
                           const size_t instruction_sz,
                           Context * context)
    {
        AssignConstInstruction<SZ> instruction;
        constexpr size_t INCREMENTAL_SZ = dg::network_trivial_serializer::size(AssignConstInstruction<SZ>{});

        if (instruction_offset + INCREMENTAL_SZ > instruction_sz) [[unlikely]]
        {
            throw outofbound_instruction{};
        }
        else [[likely]]
        {
            dg::network_trivial_serializer::deserialize_into(instruction, std::next(safe_ptr_access(instruction_ptr), instruction_offset));
            instruction_offset          += INCREMENTAL_SZ;

            char * lhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.lhs_addr_var_back_offset);
            void * lhs_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(lhs_addr_var_addr));

            boundsafe_memcpy(context, lhs_addr, instruction.data.data(), instruction.data.size());
        }
    } 

    void exec_assign_const_1(const char * const instruction_ptr,
                             size_t& instruction_offset,
                             const size_t instruction_sz,
                             Context * context)
    {
        exec_assign_const<1>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_assign_const_2(const char * const instruction_ptr,
                             size_t& instruction_offset,
                             const size_t instruction_sz,
                             Context * context)
    {
        exec_assign_const<2>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_assign_const_4(const char * const instruction_ptr,
                             size_t& instruction_offset,
                             const size_t instruction_sz,
                             Context * context)
    {
        exec_assign_const<4>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_assign_const_8(const char * const instruction_ptr,
                             size_t& instruction_offset,
                             const size_t instruction_sz,
                             Context * context)
    {
        exec_assign_const<8>(instruction_ptr, instruction_offset, instruction_sz, context);
    }
    
    template <size_t SZ>
    void exec_assign_endian_const(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        AssignConstInstruction<SZ> instruction;
        typename EndiannessAwaredRepresentativeType<SZ>::type pod; 

        constexpr size_t INCREMENTAL_SZ = dg::network_trivial_serializer::size(AssignConstInstruction<SZ>{});

        if (instruction_offset + INCREMENTAL_SZ > instruction_sz) [[unlikely]]
        {
            throw outofbound_instruction{};
        }
        else [[likely]]
        {
            dg::network_trivial_serializer::deserialize_into(instruction, std::next(safe_ptr_access(instruction_ptr), instruction_offset));
            instruction_offset          += INCREMENTAL_SZ;

            static_assert(sizeof(pod) == instruction.data.size());
            dg::network_trivial_serializer::deserialize_into(pod, instruction.data.data());
            
            char * lhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.lhs_addr_var_back_offset);
            char * lhs_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(lhs_addr_var_addr));

            boundsafe_memcpy(context, lhs_addr, &pod, sizeof(pod));
        }
    }

    void exec_assign_endian_const_1(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
    {
        exec_assign_endian_const<1>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_assign_endian_const_2(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
    {
        exec_assign_endian_const<2>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_assign_endian_const_4(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
                                    
    {
        exec_assign_endian_const<4>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_assign_endian_const_8(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
    {
        exec_assign_endian_const<8>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    //we'll make sure there is no undefined behavior by intermediate aliasing, this is an advanced technique that maybe only noipa and careful implementations can avoid 

    void exec_jump(const char * const instruction_ptr,
                   size_t& instruction_offset,
                   const size_t instruction_sz,
                   Context * context)
    {
        JumpInstruction instruction;
        constexpr size_t INCREMENTAL_SZ = dg::network_trivial_serializer::size(JumpInstruction{});
        
        if (instruction_offset + INCREMENTAL_SZ > instruction_sz) [[unlikely]]
        {
            throw outofbound_instruction{};
        }
        else [[likely]]
        {
            dg::network_trivial_serializer::deserialize_into(instruction, std::next(safe_ptr_access(instruction_ptr), instruction_offset));
            instruction_offset          += INCREMENTAL_SZ;

            char * test_addr_var_addr   = context_boundsafe_get_back_offset(context, instruction.test_addr_var_back_offset);
            bool test_value             = boundsafe_load<bool>(context, test_addr_var_addr);
            offset_t tmp_instruction_offset; 

            if (test_value)
            {
                char * var_addr         = context_boundsafe_get_back_offset(context, instruction.test_true_instruction_global_offset_var_back_offset); 
                void * offset_addr      = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, var_addr));
                tmp_instruction_offset  = boundsafe_load<offset_t>(context, offset_addr); 
            }
            else
            {
                char * var_addr         = context_boundsafe_get_back_offset(context, instruction.test_false_instruction_global_offset_var_back_offset);
                void * offset_addr      = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, var_addr));
                tmp_instruction_offset  = boundsafe_load<offset_t>(context, offset_addr);
            }

            if (tmp_instruction_offset > instruction_sz) [[unlikely]]
            {
                throw outofbound_instruction();
            }
            else [[likely]]
            {
                instruction_offset = tmp_instruction_offset;
            }
        }
    } 

    void exec_assign_range(const char * const instruction_ptr,
                           size_t& instruction_offset,
                           const size_t instruction_sz,
                           Context * context)
    {
        AssignRangeInstruction instruction;
        constexpr size_t INCREMENTAL_SZ = dg::network_trivial_serializer::size(AssignRangeInstruction{});
        
        if (instruction_offset + INCREMENTAL_SZ > instruction_sz) [[unlikely]]
        {
            throw outofbound_instruction{};
        }
        else [[likely]]
        {
            dg::network_trivial_serializer::deserialize_into(instruction, std::next(safe_ptr_access(instruction_ptr), instruction_offset));
            instruction_offset          += INCREMENTAL_SZ;

            char * lhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.lhs_addr_var_back_offset);
            char * rhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.rhs_addr_var_back_offset);
            char * sz_addr_var_addr     = context_boundsafe_get_back_offset(context, instruction.assign_sz_addr_var_back_offset); 

            void * lhs_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, lhs_addr_var_addr));
            void * rhs_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, rhs_addr_var_addr));
            void * sz_addr              = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, sz_addr_var_addr));

            uint32_t sz_value           = boundsafe_load<range_t>(context, sz_addr);

            bi_boundsafe_memcpy(context, lhs_addr, rhs_addr, sz_value);
        }
    }

    template <class OperationResolutor>
    void exec_pair(const char * const instruction_ptr,
                   size_t& instruction_offset,
                   const size_t instruction_sz,
                   OperationResolutor&& operation_resolutor,
                   Context * context)
    {
        PairInstruction instruction;
        constexpr size_t INCREMENTAL_SZ = dg::network_trivial_serializer::size(PairInstruction{});

        if (instruction_offset + INCREMENTAL_SZ > instruction_sz) [[unlikely]]
        {
            throw outofbound_instruction{};
        }
        else [[likely]]
        {
            dg::network_trivial_serializer::deserialize_into(instruction, std::next(safe_ptr_access(instruction_ptr), instruction_offset));
            instruction_offset          += INCREMENTAL_SZ;

            char * dst_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.dst_addr_var_back_offset);
            char * lhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.lhs_addr_var_back_offset);
            char * rhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.rhs_addr_var_back_offset);

            void * dst_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, dst_addr_var_addr));
            void * lhs_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, lhs_addr_var_addr));
            void * rhs_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, rhs_addr_var_addr));

            operation_resolutor(dst_addr, lhs_addr, rhs_addr);
        }
    }

    template <class OperationResolutor>
    void exec_mono(const char * const instruction_ptr,
                   size_t& instruction_offset,
                   const size_t instruction_sz,
                   OperationResolutor&& operation_resolutor,
                   Context * context)
    {
        MonoInstruction instruction;
        constexpr size_t INCREMENTAL_SZ = dg::network_trivial_serializer::size(MonoInstruction{});

        if (instruction_offset + INCREMENTAL_SZ > instruction_sz) [[unlikely]]
        {
            throw outofbound_instruction{};
        } 
        else [[likely]]
        {
            dg::network_trivial_serializer::deserialize_into(instruction, std::next(safe_ptr_access(instruction_ptr), instruction_offset));
            instruction_offset          += INCREMENTAL_SZ;

            char * dst_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.dst_addr_var_back_offset);
            char * src_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.src_addr_var_back_offset);

            void * dst_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, dst_addr_var_addr));
            void * src_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, src_addr_var_addr));

            operation_resolutor(dst_addr, src_addr);
        }
    }

    template <class T, class PairwiseOperation>
    void exec_pair_store(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         PairwiseOperation&& pairwise_operation,
                         Context * context)
    {
        auto resolutor = [context, &pairwise_operation](void * dst, void * lhs, void * rhs)
        {
            T lhs_value = boundsafe_load<T>(context, lhs);
            T rhs_value = boundsafe_load<T>(context, rhs);
            auto result_value = pairwise_operation(lhs_value, rhs_value);

            boundsafe_memcpy(context, dst, &result_value, sizeof(result_value));
        };

        exec_pair(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    template <class T, class MonowiseOperation>
    void exec_mono_store(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         MonowiseOperation&& monowise_operation,
                         Context * context)
    {
        auto resolutor = [context, &monowise_operation](void * dst, void * src)
        {
            T src_value = boundsafe_load<T>(context, src);
            auto dst_value = monowise_operation(src_value);

            boundsafe_memcpy(context, dst, &dst_value, sizeof(dst_value));
        };

        exec_mono(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_getaddr(const char * const instruction_ptr,
                      size_t& instruction_offset,
                      const size_t instruction_sz,
                      Context * context)
    {
        GetAddressInstruction instruction;
        constexpr size_t INCREMENTAL_SZ = dg::network_trivial_serializer::size(GetAddressInstruction{});

        if (instruction_offset + INCREMENTAL_SZ > instruction_sz) [[unlikely]]
        {
            throw outofbound_instruction{};
        } 
        else [[likely]]
        {
            dg::network_trivial_serializer::deserialize_into(instruction, std::next(safe_ptr_access(instruction_ptr), instruction_offset));
            instruction_offset          += INCREMENTAL_SZ;

            char * dst_var              = context_boundsafe_get_back_offset(context, instruction.dst_var_back_offset);
            char * src_var              = context_boundsafe_get_back_offset(context, instruction.src_var_back_offset);

            unsigned_addr_t src_addr    = reinterpret_cast<unsigned_addr_t>(src_var);

            boundsafe_memcpy(context, dst_var, &src_addr, sizeof(unsigned_addr_t));
        }
    }

    template <class T>
    void exec_add(const char * const instruction_ptr,
                  size_t& instruction_offset,
                  const size_t instruction_sz,
                  Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<T>(lhs + rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_add_uint64(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         Context * context)
    {
        exec_add<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_add_int64(const char * const instruction_ptr,
                        size_t& instruction_offset,
                        const size_t instruction_sz,
                        Context * context)
    {
        exec_add<int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_add_float(const char * const instruction_ptr,
                        size_t& instruction_offset,
                        const size_t instruction_sz,
                        Context * context)
    {
        exec_add<float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_add_double(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         Context * context)
    {
        exec_add<double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_sub(const char * const instruction_ptr,
                  size_t& instruction_offset,
                  const size_t instruction_sz,
                  Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<T>(lhs - rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_sub_uint64(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         Context * context)
    {
        exec_sub<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_sub_int64(const char * const instruction_ptr,
                        size_t& instruction_offset,
                        const size_t instruction_sz,
                        Context * context)
    {
        exec_sub<int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_sub_float(const char * const instruction_ptr,
                        size_t& instruction_offset,
                        const size_t instruction_sz,
                        Context * context)
    {
        exec_sub<float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_sub_double(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         Context * context)
    {
        exec_sub<double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }
    
    template <class T>
    void exec_mul(const char * const instruction_ptr,
                  size_t& instruction_offset,
                  const size_t instruction_sz,
                  Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<T>(lhs * rhs);
        };

        exec_pair_store<T>(instruction_ptr,instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_mul_uint64(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         Context * context)
    {
        exec_mul<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_mul_int64(const char * const instruction_ptr,
                        size_t& instruction_offset,
                        const size_t instruction_sz,
                        Context * context)
    {
        exec_mul<int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_mul_float(const char * const instruction_ptr,
                        size_t& instruction_offset,
                        const size_t instruction_sz,
                        Context * context)
    {
        exec_mul<float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_mul_double(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         Context * context)
    {
        exec_mul<double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_div(const char * const instruction_ptr,
                  size_t& instruction_offset,
                  const size_t instruction_sz,
                  Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<T>(lhs / rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_div_uint64(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         Context * context)
    {
        exec_div<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_div_int64(const char * const instruction_ptr,
                        size_t& instruction_offset,
                        const size_t instruction_sz,
                        Context * context)
    {
        exec_div<int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_div_float(const char * const instruction_ptr,
                        size_t& instruction_offset,
                        const size_t instruction_sz,
                        Context * context)
    {
        exec_div<float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_div_double(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         Context * context)
                         
    {
        exec_div<double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_mod(const char * const instruction_ptr,
                  size_t& instruction_offset,
                  const size_t instruction_sz,
                  Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<T>(lhs % rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_mod_uint64(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         Context * context)
    {
        exec_mod<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_mod_int64(const char * const instruction_ptr,
                        size_t& instruction_offset,
                        const size_t instruction_sz,
                        Context * context)
    {
        exec_mod<int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_and(const char * const instruction_ptr,
                  size_t& instruction_offset,
                  const size_t instruction_sz,
                  Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<bool>(lhs && rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_and_bool(const char * const instruction_ptr,
                       size_t& instruction_offset,
                       const size_t instruction_sz,
                       Context * context)
    {
        exec_and<bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_or(const char * const instruction_ptr,
                 size_t& instruction_offset,
                 const size_t instruction_sz,
                 Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<bool>(lhs || rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_or_bool(const char * const instruction_ptr,
                      size_t& instruction_offset,
                      const size_t instruction_sz,
                      Context * context)
    {
        exec_or<bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_not(const char * const instruction_ptr,
                  size_t& instruction_offset,
                  const size_t instruction_sz,
                  Context * context)
    {
        auto resolutor = [](T lhs)
        {
            return static_cast<bool>(!lhs);
        };

        exec_mono_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_not_bool(const char * const instruction_ptr,
                       size_t& instruction_offset,
                       const size_t instruction_sz,
                       Context * context)
    {
        exec_not<bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_bw_and(const char * const instruction_ptr,
                     size_t& instruction_offset,
                     const size_t instruction_sz,
                     Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<T>(lhs & rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_bw_and_uint64(const char * const instruction_ptr,
                            size_t& instruction_offset,
                            const size_t instruction_sz,
                            Context * context)
    {
        exec_bw_and<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_bw_or(const char * const instruction_ptr,
                    size_t& instruction_offset,
                    const size_t instruction_sz,
                    Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<T>(lhs | rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_bw_or_uint64(const char * const instruction_ptr,
                           size_t& instruction_offset,
                           const size_t instruction_sz,
                           Context * context)
    {
        exec_bw_or<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_bw_xor(const char * const instruction_ptr,
                     size_t& instruction_offset,
                     const size_t instruction_sz,
                     Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<T>(lhs ^ rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_bw_xor_uint64(const char * const instruction_ptr,
                            size_t& instruction_offset,
                            const size_t instruction_sz,
                            Context * context)
    {
        exec_bw_xor<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_bw_not(const char * const instruction_ptr,
                     size_t& instruction_offset,
                     const size_t instruction_sz,
                     Context * context)
    {
        auto resolutor = [](T lhs)
        {
            return static_cast<T>(~lhs);
        };

        exec_mono_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_bw_not_uint64(const char * const instruction_ptr,
                            size_t& instruction_offset,
                            const size_t instruction_sz,
                            Context * context)
    {
        exec_bw_not<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }
    
    template <class T>
    void exec_bw_leftshift(const char * const instruction_ptr,
                           size_t& instruction_offset,
                           const size_t instruction_sz,
                           Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<T>(lhs << rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_bw_leftshift_uint64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_bw_leftshift<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    } 

    template <class T>
    void exec_bw_rightshift(const char * const instruction_ptr,
                            size_t& instruction_offset,
                            const size_t instruction_sz,
                            Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<T>(lhs >> rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_bw_rightshift_uint64(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_bw_rightshift<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_cmp_less(const char * const instruction_ptr,
                       size_t& instruction_offset,
                       const size_t instruction_sz,
                       Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<bool>(lhs < rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_cmp_less_uint64(const char * const instruction_ptr,
                              size_t& instruction_offset,
                              const size_t instruction_sz,
                              Context * context)
    {
        exec_cmp_less<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cmp_less_int64(const char * const instruction_ptr,
                             size_t& instruction_offset,
                             const size_t instruction_sz,
                             Context * context)
    {
        exec_cmp_less<int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cmp_less_float(const char * const instruction_ptr,
                             size_t& instruction_offset,
                             const size_t instruction_sz,
                             Context * context)
    {
        exec_cmp_less<float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cmp_less_double(const char * const instruction_ptr,
                              size_t& instruction_offset,
                              const size_t instruction_sz,
                              Context * context)
    {
        exec_cmp_less<double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_cmp_greater(const char * const instruction_ptr,
                          size_t& instruction_offset,
                          const size_t instruction_sz,
                          Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<bool>(lhs > rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_cmp_greater_uint64(const char * const instruction_ptr,
                                 size_t& instruction_offset,
                                 const size_t instruction_sz,
                                 Context * context)
    {
        exec_cmp_greater<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cmp_greater_int64(const char * const instruction_ptr,
                                size_t& instruction_offset,
                                const size_t instruction_sz,
                                Context * context)
    {
        exec_cmp_greater<int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cmp_greater_float(const char * const instruction_ptr,
                                size_t& instruction_offset,
                                const size_t instruction_sz,
                                Context * context)
    {
        exec_cmp_greater<float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cmp_greater_double(const char * const instruction_ptr,
                                 size_t& instruction_offset,
                                 const size_t instruction_sz,
                                 Context * context)
    {
        exec_cmp_greater<double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class T>
    void exec_cmp_equal(const char * const instruction_ptr,
                        size_t& instruction_offset,
                        const size_t instruction_sz,
                        Context * context)
    {
        auto resolutor = [](T lhs, T rhs)
        {
            return static_cast<bool>(lhs == rhs);
        };

        exec_pair_store<T>(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_cmp_equal_uint64(const char * const instruction_ptr,
                               size_t& instruction_offset,
                               const size_t instruction_sz,
                               Context * context)
    {
        exec_cmp_equal<uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cmp_equal_int64(const char * const instruction_ptr,
                              size_t& instruction_offset,
                              const size_t instruction_sz,
                              Context * context)
    {
        exec_cmp_equal<int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cmp_equal_float(const char * const instruction_ptr,
                              size_t& instruction_offset,
                              const size_t instruction_sz,
                              Context * context)
    {
        exec_cmp_equal<float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cmp_equal_double(const char * const instruction_ptr,
                               size_t& instruction_offset,
                               const size_t instruction_sz,
                               Context * context)
    {
        exec_cmp_equal<double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    template <class FromType, class ToType>
    void exec_cast_store(const char * const instruction_ptr,
                         size_t& instruction_offset,
                         const size_t instruction_sz,
                         Context * context)
    {
        auto resolutor = [context, &monowise_operation](void * dst, void * src)
        {
            FromType src_value = boundsafe_load<T>(context, src);
            ToType dst_value = static_cast<ToType>(src_value);

            boundsafe_memcpy(context, dst, &dst_value, sizeof(ToType));
        };

        exec_mono(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
    }

    void exec_cast_uint8_to_bool(const char * const instruction_ptr,
                                 size_t& instruction_offset,
                                 const size_t instruction_sz,
                                 Context * context)
    {
        exec_cast_store<uint8_t, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint8_to_uint8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint8_t, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint8_to_uint16(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_cast_store<uint8_t, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint8_to_uint32(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_cast_store<uint8_t, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint8_to_uint64(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_cast_store<uint8_t, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint8_to_int8(const char * const instruction_ptr,
                                 size_t& instruction_offset,
                                 const size_t instruction_sz,
                                 Context * context)
    {
        exec_cast_store<uint8_t, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint8_to_int16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint8_t, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint8_to_int32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint8_t, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint8_to_int64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint8_t, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint8_to_float(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint8_t, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint8_to_double(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_cast_store<uint8_t, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_bool(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint16_t, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_uint8(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_cast_store<uint16_t, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_uint16(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
    {
        exec_cast_store<uint16_t, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_uint32(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
    {
        exec_cast_store<uint16_t, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_uint64(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
    {
        exec_cast_store<uint16_t, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_int8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint16_t, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_int16(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_cast_store<uint16_t, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_int32(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_cast_store<uint16_t, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_int64(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_cast_store<uint16_t, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_float(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_cast_store<uint16_t, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint16_to_double(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
    {
        exec_cast_store<uint16_t, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_bool(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint32_t, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_uint8(const char * const instruction_ptr,
                                   size_t& instruction_offset,
                                   const size_t instruction_sz,
                                   Context * context)
    {
        exec_cast_store<uint32_t, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_uint16(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
    {
        exec_cast_store<uint32_t, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_uint32(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
    {
        exec_cast_store<uint32_t, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_uint64(const char * const instruction_ptr,
                                    size_t& instruction_offset,
                                    const size_t instruction_sz,
                                    Context * context)
    {
        exec_cast_store<uint32_t, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_int8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint32_t, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_int16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint32_t, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_int32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint32_t, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_int64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint32_t, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_float(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint32_t, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint32_to_double(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint32_t, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_bool(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_uint8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_uint16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_uint32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_uint64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_int8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_int16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_int32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_int64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_float(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_uint64_to_double(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<uint64_t, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_bool(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_uint8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_uint16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_uint32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_uint64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_int8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_int16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_int32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_int64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_float(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int8_to_double(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int8_t, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_bool(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_uint8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_uint16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_uint32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_uint64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_int8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_int16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_int32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_int64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_float(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int16_to_double(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int16_t, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_bool(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_uint8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_uint16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_uint32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_uint64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_int8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_int16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_int32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_int64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_float(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int32_to_double(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int32_t, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_bool(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_uint8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_uint16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_uint32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_uint64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_int8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_int16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_int32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_int64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_float(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_int64_to_double(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<int64_t, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_bool(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_uint8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_uint16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_uint32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_uint64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_int8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_int16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_int32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_int64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_float(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_float_to_double(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<float, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_bool(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_uint8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_uint16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_uint32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_uint64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_int8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_int16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_int32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_int64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_float(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_double_to_double(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<double, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_bool(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, bool>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_uint8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, uint8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_uint16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, uint16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_uint32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, uint32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_uint64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, uint64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_int8(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, int8_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_int16(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, int16_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_int32(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, int32_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_int64(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, int64_t>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_float(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, float>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    void exec_cast_bool_to_double(const char * const instruction_ptr,
                                  size_t& instruction_offset,
                                  const size_t instruction_sz,
                                  Context * context)
    {
        exec_cast_store<bool, double>(instruction_ptr, instruction_offset, instruction_sz, context);
    }

    const char * run(const char * const bytecode,
                     const size_t bytecode_sz,
                     size_t run_size,
                     Context * context)
    {
        size_t bytecode_offset = 0u; 

        for (size_t i = 0u; i < run_size; ++i)
        {
            if (bytecode_offset == bytecode_sz)
            {
                break;
            }

            switch (bytecode[bytecode_offset])
            {
                case ALLOCATE_MEMSET_INSTRUCTION:
                {
                    exec_allocate_memset(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case DEALLOCATE_RANGE_INSTRUCTION:
                {
                    exec_deallocate_range(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case ASSIGN_CONST_1_INSTRUCTION:
                {
                    exec_assign_const_1(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case ASSIGN_CONST_2_INSTRUCTION:
                {
                    exec_assign_const_2(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case ASSIGN_CONST_4_INSTRUCTION:
                {
                    exec_assign_const_4(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case ASSIGN_CONST_8_INSTRUCTION:
                {
                    exec_assign_const_8(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case ENDIAN_ASSIGN_CONST_1_INSTRUCTION:
                {
                    exec_assign_endian_const_1(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case ENDIAN_ASSIGN_CONST_2_INSTRUCITON:
                {
                    exec_assign_endian_const_2(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case ENDIAN_ASSIGN_CONST_4_INSTRUCTION:
                {
                    exec_assign_endian_const_4(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case ENDIAN_ASSIGN_CONST_8_INSTRUCTION:
                {
                    exec_assign_endian_const_8(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case ASSIGN_RANGE_INSTRUCTION:
                {
                    exec_assign_range(bytecode, bytecode_offset, bytecode_sz, context);
                    break;
                }
                case GET_ADDR_INSTRUCTION:
                {
                    exec_getaddr();
                    break;
                }
                case ADD_UINT64_INSTRUCTION:
                {
                    exec_add_uint64();
                    break;
                }
                case ADD_INT64_INSTRUCTION:
                {
                    exec_add_int64();
                    break;
                }
                case ADD_FLOAT_INSTRUCTION:
                {
                    exec_add_float();
                    break;
                }
                case ADD_DOUBLE_INSTRUCTION:
                {
                    exec_add_double();
                    break;
                }
                case SUB_UINT64_INSTRUCTION:
                {
                    exec_sub_uint64();
                    break;
                }
                case SUB_INT64_INSTRUCTION:
                {
                    exec_sub_int64();
                    break;
                }
                case SUB_FLOAT_INSTRUCTION:
                {
                    exec_sub_float();
                    break;
                }
                case SUB_DOUBLE_INSTRUCTION:
                {
                    exec_sub_double();
                    break;
                }
                case MUL_UINT64_INSTRUCTION:
                {
                    exec_mul_uint64();
                    break;
                }
                case MUL_INT64_INSTRUCTION:
                {
                    exec_mul_int64();
                    break;
                }
                case MUL_FLOAT_INSTRUCTION:
                {
                    exec_mul_float();
                    break;
                }
                case MUL_DOUBLE_INSTRUCTION:
                {
                    exec_mul_double();
                    break;
                }
                case DIV_UINT64_INSTRUCTION:
                {
                    exec_div_uint64();
                    break;
                }
                case DIV_INT64_INSTRUCTION:
                {
                    exec_div_int64();
                    break;
                }
                case DIV_FLOAT_INSTRUCTION:
                {
                    exec_div_float();
                    break;
                }
                case DIV_DOUBLE_INSTRUCTION:
                {
                    exec_div_double();
                    break;
                }
                case MOD_UINT64_INSTRUCTION:
                {
                    exec_mod_uint64();
                    break;
                }
                case MOD_INT64_INSTRUCTION:
                {
                    exec_mod_int64();
                    break;
                }
                case AND_BOOL_INSTRUCTION:
                {
                    exec_and_bool();
                    break;
                }
                case OR_BOOL_INSTRUCTION:
                {
                    exec_or_bool();
                    break;
                }
                case NOT_BOOL_INSTRUCTION:
                {
                    exec_not_bool();
                    break;
                }
                case BW_AND_UINT64_INSTRUCTION:
                {
                    exec_bw_and_uint64();
                    break;
                }
                case BW_OR_UINT64_INSTRUCTION:
                {
                    exec_bw_or_uint64();
                    break;
                }
                case BW_XOR_UINT64_INSTRUCTION:
                {
                    exec_bw_xor_uint64();
                    break;
                }
                case BW_NOT_UINT64_INSTRUCTION:
                {
                    exec_bw_not_uint64();
                    break;
                }
                case BW_LEFTSHIFT_UINT64_INSTRUCTION:
                {
                    exec_bw_leftshift_uint64();
                    break;
                }
                case BW_RIGHTSHIFT_UINT64_INSTRUCTION:
                {
                    exec_bw_rightshift_uint64();
                    break;
                }
                case CMP_LESS_UINT64_INSTRUCTION:
                {
                    exec_cmp_less_uint64();
                    break;
                }
                case CMP_LESS_INT64_INSTRUCTION:
                {
                    exec_cmp_less_int64();
                    break;
                }
                case CMP_LESS_FLOAT_INSTRUCTION:
                {
                    exec_cmp_less_float();
                    break;
                }
                case CMP_LESS_DOUBLE_INSTRUCTION:
                {
                    exec_cmp_less_double();
                    break;
                }
                case CMP_GREATER_UINT64_INSTRUCTION:
                {
                    exec_cmp_greater_uint64();
                    break;
                }
                case CMP_GREATER_INT64_INSTRUCTION:
                {
                    exec_cmp_greater_int64();
                    break;
                }
                case CMP_GREATER_FLOAT_INSTRUCTION:
                {
                    exec_cmp_greater_float();
                    break;
                }
                case CMP_GREATER_DOUBLE_INSTRUCTION:
                {
                    exec_cmp_greater_double();
                    break;
                }
                case CMP_EQUAL_UINT64_INSTRUCTION:
                {
                    exec_cmp_equal_uint64();
                    break;
                }
                case CMP_EQUAL_INT64_INSTRUCTION:
                {
                    exec_cmp_equal_int64();
                    break;
                }
                case CMP_EQUAL_FLOAT_INSTRUCTION:
                {
                    exec_cmp_equal_float();
                    break;
                }
                case CMP_EQUAL_DOUBLE_INSTRUCTION:
                {
                    exec_cmp_equal_double();
                    break;
                }
                case CAST_BOOL_TO_BOOL_INSTRUCTION:
                {
                    break;
                }
                case CAST_BOOL_TO_UINT8_INSTRUCTION:
                {
                    break;
                }
                case CAST_BOOL_TO_UINT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_BOOL_TO_UINT32_INSTRUCTION:
                {
                    break;
                }
                case CAST_BOOL_TO_UINT64_INSTRUCTION:
                {
                    break;
                }
                case CAST_BOOL_TO_INT8_INSTRUCTION:
                {
                    break;
                }
                case CAST_BOOL_TO_INT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_BOOL_TO_INT32_INSTRUCTION:
                {
                    break;
                }
                case CAST_BOOL_TO_INT64_INSTRUCTION:
                {
                    break;
                }
                case CAST_BOOL_TO_FLOAT_INSTRUCTION:
                {
                    break;
                }
                case CAST_BOOL_TO_DOUBLE_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_BOOL_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_UINT8_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_UINT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_UINT32_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_UINT64_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_INT8_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_INT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_INT32_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_INT64_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_FLOAT_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT8_TO_DOUBLE_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_BOOL_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_UINT8_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_UINT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_UINT32_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_UINT64_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_INT8_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_INT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_INT32_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_INT64_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_FLOAT_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT16_TO_DOUBLE_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_BOOL_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_UINT8_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_UINT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_UINT32_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_UINT64_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_INT8_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_INT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_INT32_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_INT64_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_FLOAT_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT32_TO_DOUBLE_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_BOOL_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_UINT8_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_UINT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_UINT32_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_UINT64_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_INT8_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_INT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_INT32_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_INT64_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_FLOAT_INSTRUCTION:
                {
                    break;
                }
                case CAST_UINT64_TO_DOUBLE_INSTRUCTION:
                {
                    break;
                }
                case CAST_INT8_TO_BOOL_INSTRUCTION:
                {
                    break;
                }
                case CAST_INT8_TO_UINT8_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT8_TO_UINT16_INSTRUCTION:
                {
                    break;
                }
                case CAST_INT8_TO_UINT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT8_TO_UINT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT8_TO_INT8_INSTRUCTION:
                {

                    break;
                }
                case CAST_INT8_TO_INT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT8_TO_INT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT8_TO_INT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT8_TO_FLOAT_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT8_TO_DOUBLE_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT16_TO_BOOL_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT16_TO_UINT8_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT16_TO_UINT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT16_TO_UINT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT16_TO_UINT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT16_TO_INT8_INSTRUCTION:
                {

                    break;
                }
                case CAST_INT16_TO_INT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT16_TO_INT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT16_TO_INT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT16_TO_FLOAT_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT16_TO_DOUBLE_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT32_TO_BOOL_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT32_TO_UINT8_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT32_TO_UINT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT32_TO_UINT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT32_TO_UINT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT32_TO_INT8_INSTRUCTION:
                {

                    break;
                }
                case CAST_INT32_TO_INT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT32_TO_INT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT32_TO_INT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT32_TO_FLOAT_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT32_TO_DOUBLE_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT64_TO_BOOL_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT64_TO_UINT8_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT64_TO_UINT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT64_TO_UINT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT64_TO_UINT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT64_TO_INT8_INSTRUCTION:
                {

                    break;
                }
                case CAST_INT64_TO_INT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT64_TO_INT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT64_TO_INT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT64_TO_FLOAT_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_INT64_TO_DOUBLE_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_FLOAT_TO_BOOL_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_FLOAT_TO_UINT8_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_FLOAT_TO_UINT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_FLOAT_TO_UINT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_FLOAT_TO_UINT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_FLOAT_TO_INT8_INSTRUCTION:
                {

                    break;
                }
                case CAST_FLOAT_TO_INT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_FLOAT_TO_INT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_FLOAT_TO_INT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_FLOAT_TO_FLOAT_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_FLOAT_TO_DOUBLE_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_DOUBLE_TO_BOOL_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_DOUBLE_TO_UINT8_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_DOUBLE_TO_UINT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_DOUBLE_TO_UINT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_DOUBLE_TO_UINT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_DOUBLE_TO_INT8_INSTRUCTION:
                {

                    break;
                }
                case CAST_DOUBLE_TO_INT16_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_DOUBLE_TO_INT32_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_DOUBLE_TO_INT64_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_DOUBLE_TO_FLOAT_INSTRUCTION:
                {
                    
                    break;
                }
                case CAST_DOUBLE_TO_DOUBLE_INSTRUCTION:
                {
                    
                    break;
                }
                default:
                {
                    throw bad_instruction{};
                }
            }
        }

        return instruction_ptr;
    }
}

#endif