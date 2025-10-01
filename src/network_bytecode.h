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
    static inline constexpr char BW_NOT_BOOL_INSTRUCTION                = std::bit_cast<char>(static_cast<uint8_t>(32));

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

    using unsigned_addr_t = uint64_t;
    using range_t = uint32_t; 
    using offset_t = uint32_t;

    struct Context
    {
        std::unique_ptr<char[]> buffer;
        size_t buffer_sz;
        size_t buffer_offset;
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

    auto make_context(size_t context_stack_size) noexcept -> std::expected<Context *, exception_t>
    {

    }

    void deallocate_context(Context * context) noexcept
    {

    }

    void exec_allocate_memset()
    {

    } 

    void exec_deallocate_range()
    {

    }

    bool is_sub_segment(uintptr_t sub_addr_first, uintptr_t sub_addr_last,
                        uintptr_t addr_first, uintptr_t addr_last)
    {
        if (sub_addr_first < addr_first || sub_addr_last > addr_last) [[unlikely]]
        {
            return false;
        }

        return true;
    } 

    char * context_get_first_addr(Context * context)
    {
        return context->buffer.get();
    }

    char * context_get_last_addr(Context * context)
    {
        return std::next(context->buffer.get(), context->buffer_offset);
    }

    char * context_boundsafe_get_back_offset(Context * context, size_t back_offset)
    {
        if (back_offset + 1u > context->buffer_offset)
        {
            throw outofbound_memaccess();
        }

        return std::prev(std::next(context->buffer.get(), context->buffer_offset),
                         back_offset + 1u);
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

        instruction_ptr = dg::network_trivial_serializer::deserialize_into(instruction, instruction_ptr);

        char * lhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.lhs_addr_var_back_offset);
        void * lhs_addr             = reinterpret_cast<void>(boundsafe_load<unsigned_addr_t>(lhs_addr_var_addr));

        boundsafe_memcpy(context, lhs_addr, instruction.data.data(), instruction.data.size());
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

        instruction_offset          += std::distance(instruction_ptr, dg::network_trivial_serializer::deserialize_into(instruction, instruction_ptr));

        char * test_addr_var_addr   = context_boundsafe_get_back_offset(context, instruction.test_addr_var_back_offset);
        bool test_value             = boundsafe_load<bool>(context, test_addr_var_addr);
        offset_t tmp_instruction_offset; 

        if (test_value)
        {
            char * var_addr         = context_boundsafe_get_back_offset(context, instruction.test_true_instruction_global_offset_var_back_offset); 
            void * offset_addr      = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, var_addr));
            tmp_instruction_offset  = boundsafe_load<offset_t>(context, offset_addr); 
        } else
        {
            char * var_addr         = context_boundsafe_get_back_offset(context, instruction.test_false_instruction_global_offset_var_back_offset);
            void * offset_addr      = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, var_addr));
            tmp_instruction_offset  = boundsafe_load<offset_t>(context, offset_addr);
        }

        if (tmp_instruction_offset >= instruction_sz)
        {
            throw outofbound_instruction();
        }

        instruction_offset = tmp_instruction_offset;
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

        instruction_offset          += std::distance(instruction_ptr, dg::network_trivial_serializer::deserialize_into(instruction, instruction_ptr));

        char * lhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.lhs_addr_var_back_offset);
        char * rhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.rhs_addr_var_back_offset);
        char * sz_addr_var_addr     = context_boundsafe_get_back_offset(context, instruction.assign_sz_addr_var_back_offset); 

        void * lhs_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, lhs_addr_var_addr));
        void * rhs_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, rhs_addr_var_addr));
        void * sz_addr              = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, sz_addr_var_addr));

        uint32_t sz_value           = boundsafe_load<range_t>(context, sz_addr);

        boundsafe_memcpy(context, lhs_addr, rhs_addr, sz_value);
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

        if (instruction_offset + INCREMENTAL_SZ > instruction_sz)
        {
            throw outofbound_instruction{};
        }

        instruction_offset          += std::distance(instruction_ptr, dg::network_trivial_serializer::deserialize_into(instruction, instruction_ptr));
        
        char * dst_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.dst_addr_var_back_offset);
        char * lhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.lhs_addr_var_back_offset);
        char * rhs_addr_var_addr    = context_boundsafe_get_back_offset(context, instruction.rhs_addr_var_back_offset);

        void * dst_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, dst_addr_var_addr));
        void * lhs_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, lhs_addr_var_addr));
        void * rhs_addr             = reinterpret_cast<void *>(boundsafe_load<unsigned_addr_t>(context, rhs_addr_var_addr));

        operation_resolutor(dst_addr, lhs_addr, rhs_addr);
    }

    template <class T>
    void exec_add(const char * const instruction_ptr,
                  size_t& instruction_offset,
                  const size_t instruction_sz,
                  Context * context)
    {
        auto resolutor = [context](void * dst, void * lhs, void * rhs)
        {
            T lhs_value = boundsafe_load<T>(context, lhs);
            T rhs_value = boundsafe_load<T>(context, rhs);
            T result_value = lhs_value + rhs_value; 

            boundsafe_memcpy(context, dst, &result_value, sizeof(T));
        };

        exec_pair(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
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
        auto resolutor = [context](void * dst, void * lhs, void * rhs)
        {
            T lhs_value = boundsafe_load<T>(context, lhs);
            T rhs_value = boundsafe_load<T>(context, rhs);
            T result_value = lhs_value - rhs_value;

            boundsafe_memcpy(context, dst, &result_value, sizeof(T));
        };

        exec_pair(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
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
        auto resolutor = [context](void * dst, void * lhs, void * rhs)
        {
            T lhs_value = boundsafe_load<T>(context, lhs);
            T rhs_value = boundsafe_load<T>(context, rhs);
            T result_value = lhs_value * rhs_value;

            boundsafe_memcpy(context, dst, &result_value, sizeof(T));
        };

        exec_pair(instruction_ptr, instruction_offset, instruction_sz, resolutor, context);
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

    const char * run(const char * const bytecode,
                     const size_t bytecode_sz,
                     size_t run_size,
                     Context * context)
    {
        const char * const EOR = std::next(bytecode, bytecode_sz);
        const char * instruction_ptr = bytecode;
 
        //I think that mostly we'd want to do branch optimization by targeting a set of frequently used instruction
        //yet this is because we are making assumptions about how branch prediction is implemented

        for (size_t i = 0u; i < run_size; ++i)
        {
            if (instruction_ptr == EOR)
            {
                break;
            }

            switch (*instruction_ptr)
            {
                case ALLOCATE_MEMSET_INSTRUCTION:
                {
                    exec_allocate_memset(instruction_ptr, context);
                    break;
                }
                case DEALLOCATE_RANGE_INSTRUCTION:
                {
                    exec_deallocate_range(instruction_ptr, context);
                    break;
                }
                case ASSIGN_CONST_1_INSTRUCTION:
                {
                    exec_assign_const_1(instruction_ptr, context);
                    break;
                }
                case ASSIGN_CONST_2_INSTRUCTION:
                {
                    exec_assign_const_2(instruction_ptr, context);
                    break;
                }
                case ASSIGN_CONST_4_INSTRUCTION:
                {
                    exec_assign_const_4(instruction_ptr, context);
                    break;
                }
                case ASSIGN_CONST_8_INSTRUCTION:
                {
                    exec_assign_const_8(instruction_ptr, context);
                    break;
                }
                case ASSIGN_RANGE_INSTRUCTION:
                {
                    exec_assign_range(instruction_ptr, context);
                    break;
                }
                case ADD_UINT8_INSTRUCTION:
                {
                    break;
                }
                case ADD_UINT16_INSTRUCTION:
                {
                    break;
                }
                case ADD_UINT32_INSTRUCTION:
                {
                    break;
                }
                case ADD_UINT64_INSTRUCTION:
                {
                    break;
                }
                case SUB_UINT8_INSTRUCTION:
                {
                    break;
                }
                case SUB_UINT16_INSTRUCTION:
                {
                    break;
                }
                case SUB_UINT32_INSTRUCTION:
                {
                    break;
                }
                case SUB_UINT64_INSTRUCTION:
                {
                    break;
                }
                case MUL_UINT8_INSTRUCTION:
                {
                    break;
                }
                case MUL_UINT16_INSTRUCTION:
                {
                    break;
                }
                case MUL_UINT32_INSTRUCTION:
                {
                    break;
                }
                case MUL_UINT64_INSTRUCTION:
                {
                    break;
                }
                case DIV_UINT8_INSTRUCTION:
                {
                    break;
                }
                case DIV_UINT16_INSTRUCTION:
                {
                    break;
                }
                case DIV_UINT32_INSTRUCTION:
                {
                    break;
                }
                case DIV_UINT64_INSTRUCTION:
                {
                    break;
                }
                case MOD_UINT8_INSTRUCTION:
                {
                    break;
                }
                case MOD_UINT16_INSTRUCTION:
                {
                    break;
                }
                case MOD_UINT32_INSTRUCTION:
                {
                    break;
                }
                case MOD_UINT64_INSTRUCTION:
                {
                    break;
                }
                case TEST_N_JUMP_INSTRUCTION:
                {
                    break;
                }
                case SWITCH_N_JUMP_INSTRUCTION:
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