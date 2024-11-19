#ifndef __TILE_METADATA_H__
#define __TILE_METADATA_H__

#include <stdint.h>
#include <stdlib.h>
#include <stdfloat>
#include <array>
#include "network_pointer.h"

namespace dg::network_tile_metadata{

    static_assert(sizeof(char) == 1);
    static_assert(CHAR_BIT == 8);

    //I've been thinking - whether to use correct types here or using polymorphic type (serialization) for (1): decouple dependencies + class member alignment requirement
    //the latter approach seems more intuitive and maintainable - yet introduces a slight difficulty when deserialize the char array to operatable types
    //so I have decided to actually mix the design - things that could be explicitly declared - like operatable_id_t, dispatch_control_t, init_status_t, pong_count_t, crit_kind_t - are explicitly declared
    //things that might be hindering future refactor + introducing dependencies are detached -> compile-time polymorphic types (fixed size char array) which require trivial (de)serialization

    using tile_addr_t           = dg::network_pointer::uma_ptr_t; //these are not for changes
    using polymorphic_header_t  = uint8_t;
    using init_status_t         = uint8_t;
    using observer_t            = dg::network_pointer::uma_ptr_t; //these are not for changes
    using operatable_id_t       = uint64_t;
    using dispatch_control_t    = uint64_t;
    using pong_count_t          = uint64_t;
    using logit_min_t           = std::array<char, 1>;
    using logit_8_t             = std::array<char, 1>;
    using logit_16_t            = std::array<char, 2>;
    using logit_32_t            = std::array<char, 4>;
    using logit_64_t            = std::array<char, 8>;
    using logit_max_t           = std::array<char, 8>;
    using grad_min_t            = std::array<char, 1>;
    using grad_8_t              = std::array<char, 1>;
    using grad_16_t             = std::array<char, 2>;
    using grad_32_t             = std::array<char, 4>;
    using grad_64_t             = std::array<char, 8>;
    using grad_max_t            = std::array<char, 8>;
    using crit_kind_t           = uint8_t;
    using dst_info_t            = std::array<char, 32>;
    using timein_t              = uint64_t;
    using host_u8_t             = uint8_t;
    using host_u16_t            = uint16_t;
    using host_u32_t            = uint32_t;
    using host_u64_t            = uint64_t;
    using host_f8_t             = uint8_t;
    using host_f16_t            = std::bfloat16_t;
    using host_f32_t            = std::float32_t;
    using host_f64_t            = std::float64_t;
    using poly_8_t              = std::array<char, 1>;
    using poly_16_t             = std::array<char, 2>;
    using poly_32_t             = std::array<char, 4>;
    using poly_64_t             = std::array<char, 8>;

    static inline constexpr size_t LOGIT_COUNT_PER_TILE         = size_t{1} << 8;
    static inline constexpr size_t LOGIT_ALIGNMENT_SZ           = size_t{1} << 8;
    static inline constexpr size_t GRAD_ALIGNMENT_SZ            = size_t{1} << 8;
    static inline constexpr size_t MEMREGION_SZ                 = size_t{1} << 20; //MEMREGION requires all tile members to have reachability within its region - such that region(class_member) == region(std::prev(class_member + sizeof(class_member_t)))

    static inline constexpr size_t PACM_ACM_SZ                  = size_t{1} << 5;
    static inline constexpr size_t UACM_ACM_SZ                  = size_t{1} << 5;
    static inline constexpr size_t OBSERVER_ARRAY_SZ            = size_t{1} << 5; 

    static inline constexpr size_t TILE_COUNT_LEAF_8            = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_LEAF_16           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_LEAF_32           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_LEAF_64           = size_t{1} << 20;

    static inline constexpr size_t TILE_COUNT_MONO_8            = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MONO_16           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MONO_32           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MONO_64           = size_t{1} << 20;

    static inline constexpr size_t TILE_COUNT_UACM_8            = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_UACM_16           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_UACM_32           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_UACM_64           = size_t{1} << 20;
    
    static inline constexpr size_t TILE_COUNT_PACM_8            = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PACM_16           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PACM_32           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PACM_64           = size_t{1} << 20;

    static inline constexpr size_t TILE_COUNT_PAIR_8            = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PAIR_16           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PAIR_32           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PAIR_64           = size_t{1} << 20;  
    
    static inline constexpr size_t TILE_COUNT_CRIT_8            = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_CRIT_16           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_CRIT_32           = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_CRIT_64           = size_t{1} << 20;

    static inline constexpr size_t TILE_COUNT_MSGRFWD_8         = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRFWD_16        = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRFWD_32        = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRFWD_64        = size_t{1} << 20;

    static inline constexpr size_t TILE_COUNT_MSGRBWD_8         = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRBWD_16        = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRBWD_32        = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRBWD_64        = size_t{1} << 20;

    //alright - I think enumerated values should not be in this file - I'll refactor later

    // using tile_addr_t           = dg::network_pointer::uma_ptr_t; //these are not for changes
    // using polymorphic_header_t  = uint8_t;
    // using init_status_t         = uint8_t;
    // using observer_t            = dg::network_pointer::uma_ptr_t; //these are not for changes
    // using operatable_id_t       = uint64_t;
    // using dispatch_control_t    = uint64_t;
    // using pong_count_t          = uint64_t;
    // using crit_kind_t           = uint8_t;
    // using dst_info_t            = std::array<char, 32>;
    // using timein_t              = uint64_t;

    static inline constexpr init_status_t TILE_INIT_STATUS_EMPTY                                        = 0u;
    static inline constexpr init_status_t TILE_INIT_STATUS_DECAYED                                      = 1u;
    static inline constexpr init_status_t TILE_INIT_STATUS_INITIALIZED                                  = 2u;
    static inline constexpr init_status_t TILE_INIT_STATUS_DEFAULT                                      = TILE_INIT_STATUS_EMPTY;

    static inline constexpr tile_addr_t TILE_OBSERVER_DEFAULT                                           = {};
    static inline constexpr std::array<tile_addr_t, OBSERVER_ARRAY_SZ> TILE_OBSERVER_ARRAY_DEFAULT      = {};
    static inline constexpr operatable_id_t TILE_OPERATABLE_ID_DEFAULT                                  = {};
    static inline constexpr dispatch_control_t TILE_DISPATCH_CONTROL_DEFAULT                            = {};
    static inline constexpr pong_count_t TILE_PONG_COUNT_DEFAULT                                        = {};
    static inline constexpr crit_kind_t TILE_CRIT_KIND_DEFAULT                                          = {};
    static inline constexpr dst_info_t TILE_DST_INFO_DEFAULT                                            = {};
    static inline constexpr tile_addr_t TILE_DESCENDANT_DEFAULT                                         = {};
    
}

#endif