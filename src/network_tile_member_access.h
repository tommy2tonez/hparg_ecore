#ifndef __NETWORK_TILE_MEMBER_ACCESS_H__
#define __NETWORK_TILE_MEMBER_ACCESS_H__

#include <stdint.h>
#include <stdlib.h>
#include <array>
#include <limits.h>
#include <cstring>
#include "network_segcheck_bound.h"
#include "network_memult.h"
#include "network_exception_handler.h"
#include "network_pointer.h"
#include <array>
#include "assert.h"
#include "network_std_container.h"

namespace dg::network_tile_member_access::implementation{
    
    using uma_ptr_t = dg::network_pointer::uma_ptr_t;

    static constexpr auto dg_align(size_t alignment_sz, size_t blk_sz) noexcept -> size_t{

        assert(stdx::is_pow2(alignment_sz));

        size_t bit_mask     = ~static_cast<size_t>(alignment_sz - 1);  
        size_t fwd_blk_sz   = blk_sz + alignment_sz - 1;

        return fwd_blk_sz & bit_mask;
    }

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t ALIGNMENT_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OBSERVER_ARRAY_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ>
    struct LeafAddressLookup{

        private:

            using self          = LeafAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline volatile uma_ptr_t head{}; 

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::head, ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            }

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + dg_align(ALIGNMENT_SZ, self::offset_id(TILE_COUNT) + PADDING_SZ);
            } 

            template <size_t ARR_IDX>
            static constexpr auto offset_observer_addr(size_t idx, const std::integral_constant<size_t, ARR_IDX>) noexcept -> size_t{
                
                return idx * (OBSERVER_VALUE_SZ * OBSERVER_ARRAY_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ) + OBSERVER_VALUE_SZ * ARR_IDX); 
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_observer_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + dg_align(ALIGNMENT_SZ, self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            } 

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + dg_align(ALIGNMENT_SZ, self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static_assert(stdx::is_pow2(ALIGNMENT_SZ));

            static void init(uma_ptr_t buf){

                self::head = dg::pointer_cast<uma_ptr_t>(dg_align(ALIGNMENT_SZ, dg::pointer_cast<typename dg::ptr_info<uma_ptr_t>::max_unsigned_t>(buf)));
                access_ins::init(self::head, dg::memult::advance(self::head, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto buf_size() -> size_t{

                return self::offset_pong_count_addr(TILE_COUNT) + ALIGNMENT_SZ - 1;
            } 

            static consteval auto tile_size() -> size_t{

                return TILE_COUNT;
            }

            static consteval auto observer_arr_size() -> size_t{

                return OBSERVER_ARRAY_SZ;
            }

            static consteval auto logit_group_size() -> size_t{

                return LOGIT_VALUE_SZ;
            }

            static consteval auto grad_group_size() -> size_t{

                return GRAD_VALUE_SZ;
            }

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            } 

            template <size_t ARR_IDX>
            static inline auto observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

                static_assert(ARR_IDX < OBSERVER_ARRAY_SZ);
                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr)), std::integral_constant<size_t, ARR_IDX>{}));
            }

            static inline auto tile_logit_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_logit_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto tile_grad_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_grad_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto operatable_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_operatable_id_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto dispatch_control_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_dispatch_control_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto pong_count_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_pong_count_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto notification_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }

            static inline auto rcu_lock_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }
    };

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t ALIGNMENT_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OBSERVER_ARRAY_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ>
    struct MonoAddressLookup{

        private:

            using self          = MonoAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>; 

            static inline volatile uma_ptr_t head{};

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::head, ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{
                
                return idx;
            }

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + dg_align(ALIGNMENT_SZ, self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            template <size_t ARR_IDX>
            static constexpr auto offset_observer_addr(size_t idx, const std::integral_constant<size_t, ARR_IDX>) noexcept -> size_t{

                return idx * (OBSERVER_VALUE_SZ * OBSERVER_ARRAY_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ) + OBSERVER_VALUE_SZ * ARR_IDX);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_observer_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + dg_align(ALIGNMENT_SZ, self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + dg_align(ALIGNMENT_SZ, self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }
            
            static constexpr auto offset_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + dg_align(ALIGNMENT_SZ, self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static_assert(stdx::is_pow2(ALIGNMENT_SZ));

            static void init(uma_ptr_t buf){

                self::head = dg::pointer_cast<uma_ptr_t>(dg_align(ALIGNMENT_SZ, dg::pointer_cast<typename dg::ptr_info<uma_ptr_t>::max_unsigned_t>(buf)));
                access_ins::init(self::head, dg::memult::advance(self::head, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto buf_size() -> size_t{

                return self::offset_descendant_addr(TILE_COUNT) + ALIGNMENT_SZ - 1;
            }

            static consteval auto tile_size() -> size_t{

                return TILE_COUNT;
            }

            static consteval auto observer_arr_size() -> size_t{

                return OBSERVER_ARRAY_SZ;
            }

            static consteval auto logit_group_size() -> size_t{

                return LOGIT_VALUE_SZ;
            }

            static consteval auto grad_group_size() -> size_t{

                return GRAD_VALUE_SZ;
            }

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            template <size_t ARR_IDX>
            static inline auto observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

                static_assert(ARR_IDX < OBSERVER_ARRAY_SZ);
                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr)), std::integral_constant<size_t, ARR_IDX>{}));
            }

            static inline auto tile_logit_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_logit_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto tile_grad_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_grad_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto operatable_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_operatable_id_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto dispatch_control_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_dispatch_control_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto pong_count_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_pong_count_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto descendant_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_descendant_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto notification_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }

            static inline auto rcu_lock_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }
    };

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t ALIGNMENT_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OBSERVER_ARRAY_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ, size_t ACM_SZ>
    struct UACMAddressLookup{
        
        private:

            using self          = UACMAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline volatile uma_ptr_t head{};

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::head, ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            } 

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + dg_align(ALIGNMENT_SZ, self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            template <size_t ARR_IDX>
            static constexpr auto offset_observer_addr(size_t idx, const std::integral_constant<size_t, ARR_IDX>) noexcept -> size_t{

                return idx * (OBSERVER_VALUE_SZ * OBSERVER_ARRAY_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ) + OBSERVER_VALUE_SZ * ARR_IDX);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_observer_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + dg_align(ALIGNMENT_SZ, self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + dg_align(ALIGNMENT_SZ, self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

            template <size_t ACM_IDX>
            static constexpr auto offset_descendant_addr(size_t idx, const std::integral_constant<size_t, ACM_IDX>) noexcept -> size_t{

                return idx * (DESCENDANT_SZ * ACM_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ) + DESCENDANT_SZ * ACM_IDX);
            }

        public:

            static_assert(stdx::is_pow2(ALIGNMENT_SZ));

            static void init(uma_ptr_t buf){

                self::head = dg::pointer_cast<uma_ptr_t>(dg_align(ALIGNMENT_SZ, dg::pointer_cast<typename dg::ptr_info<uma_ptr_t>::max_unsigned_t>(buf)));
                access_ins::init(self::head, dg::memult::advance(self::head, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto buf_size() -> size_t{

                return self::offset_descendant_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + ALIGNMENT_SZ - 1;
            }

            static consteval auto tile_size() -> size_t{

                return TILE_COUNT;
            }

            static consteval auto observer_arr_size() -> size_t{

                return OBSERVER_ARRAY_SZ;
            }

            static consteval auto accum_size() -> size_t{

                return ACM_SZ;
            }

            static consteval auto logit_group_size() -> size_t{

                return LOGIT_VALUE_SZ;
            }

            static consteval auto grad_group_size() -> size_t{

                return GRAD_VALUE_SZ;
            }

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            template <size_t ARR_IDX>
            static inline auto observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

                static_assert(ARR_IDX < OBSERVER_ARRAY_SZ);
                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr)), std::integral_constant<size_t, ARR_IDX>{}));
            }

            static inline auto tile_logit_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_logit_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto tile_grad_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_grad_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto operatable_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_operatable_id_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto dispatch_control_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_dispatch_control_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto pong_count_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_pong_count_addr(self::index(access_ins::access(ptr))));
            }

            template <size_t IDX>
            static inline auto descendant_addr(uma_ptr_t ptr, const std::integral_constant<size_t, IDX>) noexcept -> uma_ptr_t{

                static_assert(IDX < ACM_SZ);
                return dg::memult::advance(self::get_head(), self::offset_descendant_addr(self::index(access_ins::access(ptr)), std::integral_constant<size_t, IDX>{}));
            }

            static inline auto notification_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }

            static inline auto rcu_lock_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }
    };

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t ALIGNMENT_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OBSERVER_ARRAY_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ, size_t ACM_SZ>
    struct PACMAddressLookup{

        private:

            using self          = PACMAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline volatile uma_ptr_t head{};

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::head, ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            } 

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + dg_align(ALIGNMENT_SZ, self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            template <size_t ARR_IDX>
            static constexpr auto offset_observer_addr(size_t idx, const std::integral_constant<size_t, ARR_IDX>) noexcept -> size_t{

                return idx * (OBSERVER_VALUE_SZ * OBSERVER_ARRAY_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ) + OBSERVER_VALUE_SZ * ARR_IDX);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_observer_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + dg_align(ALIGNMENT_SZ, self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + dg_align(ALIGNMENT_SZ, self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

            template <size_t ACM_IDX>
            static constexpr auto offset_left_descendant_addr(size_t idx, const std::integral_constant<size_t, ACM_IDX>) noexcept -> size_t{

                return idx * (DESCENDANT_SZ * ACM_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ) + DESCENDANT_SZ * ACM_IDX);
            }

            template <size_t ACM_IDX>
            static constexpr auto offset_right_descendant_addr(size_t idx, const std::integral_constant<size_t, ACM_IDX>) noexcept -> size_t{

                return idx * (DESCENDANT_SZ * ACM_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_left_descendant_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ) + DESCENDANT_SZ * ACM_IDX);
            }

        public:

            static_assert(stdx::is_pow2(ALIGNMENT_SZ));

            static void init(uma_ptr_t buf){

                self::head = dg::pointer_cast<uma_ptr_t>(dg_align(ALIGNMENT_SZ, dg::pointer_cast<typename dg::ptr_info<uma_ptr_t>::max_unsigned_t>(buf)));
                access_ins::init(self::head, dg::memult::advance(self::head, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto buf_size() -> size_t{

                return self::offset_right_descendant_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + ALIGNMENT_SZ - 1;
            }

            static consteval auto tile_size() -> size_t{

                return TILE_COUNT;
            }

            static consteval auto observer_arr_size() -> size_t{

                return OBSERVER_ARRAY_SZ;
            }

            static consteval auto accum_size() -> size_t{

                return ACM_SZ;
            }

            static consteval auto logit_group_size() -> size_t{

                return LOGIT_VALUE_SZ;
            }

            static consteval auto grad_group_size() -> size_t{

                return GRAD_VALUE_SZ;
            }

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            template <size_t ARR_IDX>
            static inline auto observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

                static_assert(ARR_IDX < OBSERVER_ARRAY_SZ);
                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr)), std::integral_constant<size_t, ARR_IDX>{}));
            }

            static inline auto tile_logit_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_logit_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto tile_grad_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_grad_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto operatable_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_operatable_id_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto dispatch_control_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_dispatch_control_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto pong_count_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_pong_count_addr(self::index(access_ins::access(ptr))));
            }

            template <size_t IDX>
            static inline auto left_descendant_addr(uma_ptr_t ptr, const std::integral_constant<size_t, IDX>) noexcept -> uma_ptr_t{

                static_assert(IDX < ACM_SZ);
                return dg::memult::advance(self::get_head(), self::offset_left_descendant_addr(self::index(access_ins::access(ptr)), std::integral_constant<size_t, IDX>{}));
            }

            template <size_t IDX>
            static inline auto right_descendant_addr(uma_ptr_t ptr, const std::integral_constant<size_t, IDX>) noexcept -> uma_ptr_t{

                static_assert(IDX < ACM_SZ);
                return dg::memult::advance(self::get_head(), self::offset_right_descendant_addr(self::index(access_ins::access(ptr)), std::integral_constant<size_t, IDX>{}));
            }

            static inline auto notification_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return tile_logit_addr(ptr);
            }

            static inline auto rcu_lock_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return tile_logit_addr(ptr);
            }
    };

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t ALIGNMENT_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OBSERVER_ARRAY_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ>
    struct PairAddressLookup{

        private:

            using self = PairAddressLookup;
            using access_ins = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>; 

            static inline volatile uma_ptr_t head{};

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::head, ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            } 

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + dg_align(ALIGNMENT_SZ, self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            template <size_t ARR_IDX>
            static constexpr auto offset_observer_addr(size_t idx, const std::integral_constant<size_t, ARR_IDX>) noexcept -> size_t{

                return idx * (OBSERVER_VALUE_SZ * OBSERVER_ARRAY_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ) + OBSERVER_VALUE_SZ * ARR_IDX);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_observer_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + dg_align(ALIGNMENT_SZ, self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + dg_align(ALIGNMENT_SZ, self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_left_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + dg_align(ALIGNMENT_SZ, self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_right_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + dg_align(ALIGNMENT_SZ, self::offset_left_descendant_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static_assert(stdx::is_pow2(ALIGNMENT_SZ));

            static void init(uma_ptr_t buf){

                self::head = dg::pointer_cast<uma_ptr_t>(dg_align(ALIGNMENT_SZ, dg::pointer_cast<typename dg::ptr_info<uma_ptr_t>::max_unsigned_t>(buf)));
                access_ins::init(self::head, dg::memult::advance(self::head, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head; 
            }

            static consteval auto buf_size() -> size_t{

                return offset_right_descendant_addr(TILE_COUNT) + ALIGNMENT_SZ - 1;
            } 

            static consteval auto tile_size() -> size_t{

                return TILE_COUNT;
            }

            static consteval auto observer_arr_size() ->  size_t{

                return OBSERVER_ARRAY_SZ;
            }

            static consteval auto logit_group_size() -> size_t{

                return LOGIT_VALUE_SZ;
            }

            static consteval auto grad_group_size() -> size_t{

                return GRAD_VALUE_SZ;
            }

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            template <size_t ARR_IDX>
            static inline auto observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

                static_assert(ARR_IDX < OBSERVER_ARRAY_SZ);
                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr), std::integral_constant<size_t, ARR_IDX>{})));
            }

            static inline auto tile_logit_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_logit_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto tile_grad_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_grad_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto operatable_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_operatable_id_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto dispatch_control_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_dispatch_control_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto pong_count_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_pong_count_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto left_descendant_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_left_descendant_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto right_descendant_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_right_descendant_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto notification_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return tile_logit_addr(ptr);   
            }

            static inline auto rcu_lock_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return tile_logit_addr(ptr);
            }
    };

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t ALIGNMENT_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OBSERVER_ARRAY_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ, size_t CRIT_KIND_SZ>
    struct CritAddressLookup{

        private:

            using self = CritAddressLookup;
            using access_ins = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline volatile uma_ptr_t head{}; 

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::head, ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            }
            
            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + dg_align(ALIGNMENT_SZ, self::offset_id(TILE_COUNT) + PADDING_SZ); 
            } 

            template <size_t ARR_IDX>
            static constexpr auto offset_observer_addr(size_t idx, const std::integral_constant<size_t, ARR_IDX>) noexcept -> size_t{

                return idx * (OBSERVER_VALUE_SZ * OBSERVER_ARRAY_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ) + OBSERVER_VALUE_SZ * ARR_IDX);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_observer_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + dg_align(ALIGNMENT_SZ, self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + dg_align(ALIGNMENT_SZ, self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + dg_align(ALIGNMENT_SZ, self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_crit_kind_addr(size_t idx) noexcept -> size_t{

                return idx * CRIT_KIND_SZ + dg_align(ALIGNMENT_SZ, self::offset_descendant_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static_assert(stdx::is_pow2(ALIGNMENT_SZ));

            static void init(uma_ptr_t buf){

                self::head = dg::pointer_cast<uma_ptr_t>(dg_align(ALIGNMENT_SZ, dg::pointer_cast<typename dg::ptr_info<uma_ptr_t>::max_unsigned_t>(buf)));
                access_ins::init(self::head, dg::memult::advance(self::head, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto buf_size() -> size_t{

                return offset_crit_kind_addr(TILE_COUNT) + ALIGNMENT_SZ - 1;
            } 

            static consteval auto tile_size() -> size_t{

                return TILE_COUNT;
            }

            static consteval auto observer_arr_size() -> size_t{

                return OBSERVER_ARRAY_SZ;
            } 

            static consteval auto logit_group_size() -> size_t{

                return LOGIT_VALUE_SZ;
            }

            static consteval auto grad_group_size() -> size_t{

                return GRAD_VALUE_SZ;
            }

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            } 

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            template <size_t ARR_IDX>
            static inline auto observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

                static_assert(ARR_IDX < OBSERVER_ARRAY_SZ);
                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr)), std::integral_constant<size_t, ARR_IDX>{}));
            }

            static inline auto tile_logit_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_logit_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto tile_grad_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_grad_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto operatable_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_operatable_id_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto dispatch_control_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_dispatch_control_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto pong_count_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_pong_count_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto descendant_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_descendant_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto crit_kind_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_crit_kind_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto notification_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }

            static inline auto rcu_lock_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }
    };

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t ALIGNMENT_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OBSERVER_ARRAY_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ, size_t DST_INFO_SZ>
    struct MsgrFwdAddressLookup{

        private:

            using self          = MsgrFwdAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>; 

            static inline volatile uma_ptr_t head{};

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::get_head(), ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            }

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + dg_align(ALIGNMENT_SZ, self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            template <size_t ARR_IDX>
            static constexpr auto offset_observer_addr(size_t idx, const std::integral_constant<size_t, ARR_IDX>) noexcept -> size_t{

                return idx * (OBSERVER_VALUE_SZ * OBSERVER_ARRAY_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ) + OBSERVER_VALUE_SZ * ARR_IDX);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_observer_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + dg_align(ALIGNMENT_SZ, self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + dg_align(ALIGNMENT_SZ, self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + dg_align(ALIGNMENT_SZ, self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dst_info_addr(size_t idx) noexcept -> size_t{

                return idx * DST_INFO_SZ + dg_align(ALIGNMENT_SZ, self::offset_descendant_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static_assert(stdx::is_pow2(ALIGNMENT_SZ));

            static void init(uma_ptr_t buf){

                self::head = dg::pointer_cast<uma_ptr_t>(dg_align(ALIGNMENT_SZ, dg::pointer_cast<typename dg::ptr_info<uma_ptr_t>::max_unsigned_t>(buf)));
                access_ins::init(self::head, dg::memult::advance(self::head, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto buf_size() -> size_t{

                return self::offset_dst_info_addr(TILE_COUNT) + ALIGNMENT_SZ - 1;
            }

            static consteval auto tile_size() -> size_t{

                return TILE_COUNT;
            }

            static consteval auto observer_arr_size() -> size_t{

                return OBSERVER_ARRAY_SZ;
            } 

            static consteval auto logit_group_size() -> size_t{

                return LOGIT_VALUE_SZ;
            }

            static consteval auto grad_group_size() -> size_t{

                return GRAD_VALUE_SZ;
            }

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            template <size_t ARR_IDX>
            static inline auto observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

                static_assert(ARR_IDX < OBSERVER_ARRAY_SZ);
                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr)), std::integral_constant<size_t, ARR_IDX>{}));
            }

            static inline auto tile_logit_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_logit_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto tile_grad_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_grad_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto operatable_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_operatable_id_addr(self::index(access_ins::access(ptr))));
            } 

            static inline auto dispatch_control_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_dispatch_control_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto pong_count_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_pong_count_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto descendant_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_descendant_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto dst_info_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_dst_info_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto notification_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }

            static inline auto rcu_lock_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }
    };

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t ALIGNMENT_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OBSERVER_ARRAY_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ, size_t DST_INFO_SZ, size_t TIMEIN_SZ>
    struct MsgrBwdAddressLookup{

        private:

            using self          = MsgrBwdAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline volatile uma_ptr_t head{};
            
            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::head, ptr);
            } 

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            } 

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + dg_align(ALIGNMENT_SZ, self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            template <size_t ARR_IDX>
            static constexpr auto offset_observer_addr(size_t idx, const std::integral_constant<size_t, ARR_IDX>) noexcept -> size_t{

                return idx * (OBSERVER_VALUE_SZ * OBSERVER_ARRAY_SZ) + (dg_align(ALIGNMENT_SZ, self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ) + OBSERVER_VALUE_SZ * ARR_IDX);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_observer_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + dg_align(ALIGNMENT_SZ, self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ * dg_align(ALIGNMENT_SZ, self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + dg_align(ALIGNMENT_SZ, self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + dg_align(ALIGNMENT_SZ, self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            } 

            static constexpr auto offset_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + dg_align(ALIGNMENT_SZ, self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dst_info_addr(size_t idx) noexcept -> size_t{

                return idx * DST_INFO_SZ + dg_align(ALIGNMENT_SZ, self::offset_descendant_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_timein_addr(size_t idx) noexcept -> size_t{

                return idx * TIMEIN_SZ + dg_align(ALIGNMENT_SZ, self::offset_dst_info_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static_assert(stdx::is_pow2(ALIGNMENT_SZ));

            static void init(uma_ptr_t buf){
                
                self::head = dg::pointer_cast<uma_ptr_t>(dg_align(ALIGNMENT_SZ, dg::pointer_cast<typename dg::ptr_info<uma_ptr_t>::max_unsigned_t>(buf)));
                access_ins::init(self::head, dg::memult::advance(self::head, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto buf_size() -> size_t{

                return self::offset_timein_addr(TILE_COUNT) + ALIGNMENT_SZ - 1;
            }

            static consteval auto tile_size() -> size_t{

                return TILE_COUNT;
            } 

            static consteval auto observer_arr_size() -> size_t{

                return OBSERVER_ARRAY_SZ;
            }

            static consteval auto logit_group_size() -> size_t{

                return LOGIT_VALUE_SZ;
            }

            static consteval auto grad_group_size() -> size_t{

                return GRAD_VALUE_SZ;
            }

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            } 

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            template <size_t ARR_IDX>
            static inline auto observer_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ARR_IDX>) noexcept -> uma_ptr_t{

                static_assert(ARR_IDX < OBSERVER_ARRAY_SZ);
                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr)), std::integral_constant<size_t, ARR_IDX>{}));
            }

            static inline auto tile_logit_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_logit_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto tile_grad_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_tile_grad_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto operatable_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_operatable_id_addr(self::index(access_ins::access(ptr))));
            } 

            static inline auto dispatch_control_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_dispatch_control_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto pong_count_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_pong_count_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto descendant_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_descendant_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto dst_info_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_dst_info_addr(self::index(access_ins::access(ptr))));
            } 

            static inline auto timein_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_timein_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto notification_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::tile_logit_addr(ptr);
            }

            static inline auto rcu_lock_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return self::rcu_lock_addr(ptr);
            }
    };
}

namespace dg::network_tile_member_access{

    //this implementation is to assume that the total memregion <= 1024 - to fit in L1 | at most L2 cache 
    //assume the mean concurrency factor of 32
    //then there's not a performance constraint
    //otherwise it's better to use unordered_unstable_map for cache efficiency - this is an important note
    //this cannot be told | measured without a hollistic view of the program
    //if the designated core that accesses this is solely responsible for offloading to cuda - and all these (including other tables) fit in L1 cache - then everyone's happy - life moves on
    //if the designated core that accesses this is responsible for doing the actual work - then cache thrashing happens and everyone's not happy - life does not move on

    static_assert(sizeof(char) == 1);
    static_assert(CHAR_BIT == 8);

    static inline constexpr size_t TILE_COUNT_LEAF_8        = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_LEAF_16       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_LEAF_32       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_LEAF_64       = size_t{1} << 20;
    
    static inline constexpr size_t TILE_COUNT_MONO_8        = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MONO_16       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MONO_32       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MONO_64       = size_t{1} << 20;

    static inline constexpr size_t TILE_COUNT_UACM_8        = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_UACM_16       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_UACM_32       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_UACM_64       = size_t{1} << 20;
    
    static inline constexpr size_t TILE_COUNT_PACM_8        = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PACM_16       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PACM_32       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PACM_64       = size_t{1} << 20;

    static inline constexpr size_t TILE_COUNT_PAIR_8        = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PAIR_16       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PAIR_32       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_PAIR_64       = size_t{1} << 20;  
    
    static inline constexpr size_t TILE_COUNT_CRIT_8        = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_CRIT_16       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_CRIT_32       = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_CRIT_64       = size_t{1} << 20;

    static inline constexpr size_t TILE_COUNT_MSGRFWD_8     = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRFWD_16    = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRFWD_32    = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRFWD_64    = size_t{1} << 20;

    static inline constexpr size_t TILE_COUNT_MSGRBWD_8     = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRBWD_16    = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRBWD_32    = size_t{1} << 20;
    static inline constexpr size_t TILE_COUNT_MSGRBWD_64    = size_t{1} << 20; 

    static inline constexpr size_t LOGIT_COUNT_PER_TILE     = size_t{1} << 8;
    static inline constexpr size_t PADDING_SZ               = 0u;
    static inline constexpr size_t MEMREGION_SZ             = size_t{1} << 20;
    static inline constexpr size_t ID_MEMREGION_SZ          = size_t{1} << 10;
    static inline constexpr size_t UACM_ACM_SZ              = size_t{1} << 5;
    static inline constexpr size_t PACM_ACM_SZ              = size_t{1} << 5;
    static inline constexpr size_t OBSERVER_ARRAY_SZ        = size_t{1} << 5;

    static inline constexpr bool IS_SAFE_ACCESS_ENABLED     = true;

    using uma_ptr_t             = dg::network_pointer::uma_ptr_t;
    using tile_polymorphic_t    = uint8_t;
    using init_status_t         = uint8_t;
    using observer_t            = uint64_t;
    using operatable_id_t       = uint64_t;
    using dispatch_control_t    = uint64_t;
    using pong_count_t          = uint64_t;
    using addr_t                = uint64_t; 
    using logit_8_t             = uint8_t;
    using logit_16_t            = uint16_t;
    using logit_32_t            = uint32_t;
    using logit_64_t            = uint64_t;
    using grad_8_t              = uint8_t;
    using grad_16_t             = uint16_t;
    using grad_32_t             = uint32_t;
    using grad_64_t             = uint64_t;
    using crit_kind_t           = uint8_t;
    using dst_info_t            = std::array<char, 32>;
    using timein_t              = uint64_t;

    enum tile_polymorphic_option: tile_polymorphic_t{
        id_leaf_8       = 0u,
        id_leaf_16      = 1u,
        id_leaf_32      = 2u,
        id_leaf_64      = 3u,
        id_mono_8       = 4u,
        id_mono_16      = 5u,
        id_mono_32      = 6u,
        id_mono_64      = 7u,
        id_crit_8       = 8u,
        id_crit_16      = 9u,
        id_crit_32      = 10u,
        id_crit_64      = 11u,
        id_msgrfwd_8    = 12u,
        id_msgrfwd_16   = 13u,
        id_msgrfwd_32   = 14u,
        id_msgrfwd_64   = 15u,
        id_msgrbwd_8    = 16u,
        id_msgrbwd_16   = 17u,
        id_msgrbwd_32   = 18u,
        id_msgrbwd_64   = 19u,
        id_pair_8       = 20u,
        id_pair_16      = 21u,
        id_pair_32      = 22u,
        id_pair_64      = 23u,
        id_uacm_8       = 24u,
        id_uacm_16      = 25u,
        id_uacm_32      = 26u,
        id_uacm_64      = 27u,
        id_pacm_8       = 28u,
        id_pacm_16      = 29u,
        id_pacm_32      = 30u,
        id_pacm_64      = 31u,
    };

    struct network_tile_member_access_signature{}; 

    using leaf8_accessor_t          = dg::network_tile_member_access::implementation::LeafAddressLookup<network_tile_member_access_signature, TILE_COUNT_LEAF_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_8_t),  LOGIT_COUNT_PER_TILE * sizeof(grad_8_t),  sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t)>;
    using leaf16_accessor_t         = dg::network_tile_member_access::implementation::LeafAddressLookup<network_tile_member_access_signature, TILE_COUNT_LEAF_16, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_16_t), LOGIT_COUNT_PER_TILE * sizeof(grad_16_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t)>;
    using leaf32_accessor_t         = dg::network_tile_member_access::implementation::LeafAddressLookup<network_tile_member_access_signature, TILE_COUNT_LEAF_32, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_32_t), LOGIT_COUNT_PER_TILE * sizeof(grad_32_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t)>;
    using leaf64_accessor_t         = dg::network_tile_member_access::implementation::LeafAddressLookup<network_tile_member_access_signature, TILE_COUNT_LEAF_64, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_64_t), LOGIT_COUNT_PER_TILE * sizeof(grad_64_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t)>; 

    using mono8_accessor_t          = dg::network_tile_member_access::implementation::MonoAddressLookup<network_tile_member_access_signature, TILE_COUNT_MONO_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_8_t),  LOGIT_COUNT_PER_TILE * sizeof(grad_8_t),  sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t)>;
    using mono16_accessor_t         = dg::network_tile_member_access::implementation::MonoAddressLookup<network_tile_member_access_signature, TILE_COUNT_MONO_16, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_16_t), LOGIT_COUNT_PER_TILE * sizeof(grad_16_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t)>;
    using mono32_accessor_t         = dg::network_tile_member_access::implementation::MonoAddressLookup<network_tile_member_access_signature, TILE_COUNT_MONO_32, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_32_t), LOGIT_COUNT_PER_TILE * sizeof(grad_32_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t)>;
    using mono64_accessor_t         = dg::network_tile_member_access::implementation::MonoAddressLookup<network_tile_member_access_signature, TILE_COUNT_MONO_64, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_64_t), LOGIT_COUNT_PER_TILE * sizeof(grad_64_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t)>;

    using pair8_accessor_t          = dg::network_tile_member_access::implementation::PairAddressLookup<network_tile_member_access_signature, TILE_COUNT_PAIR_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_8_t) , LOGIT_COUNT_PER_TILE * sizeof(grad_8_t),  sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t)>;
    using pair16_accessor_t         = dg::network_tile_member_access::implementation::PairAddressLookup<network_tile_member_access_signature, TILE_COUNT_PAIR_16, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_16_t), LOGIT_COUNT_PER_TILE * sizeof(grad_16_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t)>;
    using pair32_accessor_t         = dg::network_tile_member_access::implementation::PairAddressLookup<network_tile_member_access_signature, TILE_COUNT_PAIR_32, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_32_t), LOGIT_COUNT_PER_TILE * sizeof(grad_32_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t)>;
    using pair64_accessor_t         = dg::network_tile_member_access::implementation::PairAddressLookup<network_tile_member_access_signature, TILE_COUNT_PAIR_64, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_64_t), LOGIT_COUNT_PER_TILE * sizeof(grad_64_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t)>;

    using uacm8_accessor_t          = dg::network_tile_member_access::implementation::UACMAddressLookup<network_tile_member_access_signature, TILE_COUNT_UACM_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_8_t),  LOGIT_COUNT_PER_TILE * sizeof(grad_8_t),  sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), UACM_ACM_SZ>;
    using uacm16_accessor_t         = dg::network_tile_member_access::implementation::UACMAddressLookup<network_tile_member_access_signature, TILE_COUNT_UACM_16, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_16_t), LOGIT_COUNT_PER_TILE * sizeof(grad_16_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), UACM_ACM_SZ>;
    using uacm32_accessor_t         = dg::network_tile_member_access::implementation::UACMAddressLookup<network_tile_member_access_signature, TILE_COUNT_UACM_32, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_32_t), LOGIT_COUNT_PER_TILE * sizeof(grad_32_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), UACM_ACM_SZ>;
    using uacm64_accessor_t         = dg::network_tile_member_access::implementation::UACMAddressLookup<network_tile_member_access_signature, TILE_COUNT_UACM_64, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_64_t), LOGIT_COUNT_PER_TILE * sizeof(grad_64_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), UACM_ACM_SZ>;

    using pacm8_accessor_t          = dg::network_tile_member_access::implementation::PACMAddressLookup<network_tile_member_access_signature, TILE_COUNT_PACM_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_8_t),  LOGIT_COUNT_PER_TILE * sizeof(grad_8_t),  sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), PACM_ACM_SZ>;
    using pacm16_accessor_t         = dg::network_tile_member_access::implementation::PACMAddressLookup<network_tile_member_access_signature, TILE_COUNT_PACM_16, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_16_t), LOGIT_COUNT_PER_TILE * sizeof(grad_16_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), PACM_ACM_SZ>;
    using pacm32_accessor_t         = dg::network_tile_member_access::implementation::PACMAddressLookup<network_tile_member_access_signature, TILE_COUNT_PACM_32, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_32_t), LOGIT_COUNT_PER_TILE * sizeof(grad_32_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), PACM_ACM_SZ>;
    using pacm64_accessor_t         = dg::network_tile_member_access::implementation::PACMAddressLookup<network_tile_member_access_signature, TILE_COUNT_PACM_64, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_64_t), LOGIT_COUNT_PER_TILE * sizeof(grad_64_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), PACM_ACM_SZ>;

    using crit8_accessor_t          = dg::network_tile_member_access::implementation::CritAddressLookup<network_tile_member_access_signature, TILE_COUNT_CRIT_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_8_t),  LOGIT_COUNT_PER_TILE * sizeof(grad_8_t),  sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(crit_kind_t)>;
    using crit16_accessor_t         = dg::network_tile_member_access::implementation::CritAddressLookup<network_tile_member_access_signature, TILE_COUNT_CRIT_16, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_16_t), LOGIT_COUNT_PER_TILE * sizeof(grad_16_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(crit_kind_t)>;
    using crit32_accessor_t         = dg::network_tile_member_access::implementation::CritAddressLookup<network_tile_member_access_signature, TILE_COUNT_CRIT_32, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_32_t), LOGIT_COUNT_PER_TILE * sizeof(grad_32_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(crit_kind_t)>;
    using crit64_accessor_t         = dg::network_tile_member_access::implementation::CritAddressLookup<network_tile_member_access_signature, TILE_COUNT_CRIT_64, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_64_t), LOGIT_COUNT_PER_TILE * sizeof(grad_64_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(crit_kind_t)>;

    using msgrfwd8_accessor_t       = dg::network_tile_member_access::implementation::MsgrFwdAddressLookup<network_tile_member_access_signature, TILE_COUNT_MSGRFWD_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_8_t),  LOGIT_COUNT_PER_TILE * sizeof(grad_8_t),  sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(dst_info_t)>;
    using msgrfwd16_accessor_t      = dg::network_tile_member_access::implementation::MsgrFwdAddressLookup<network_tile_member_access_signature, TILE_COUNT_MSGRFWD_16, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_16_t), LOGIT_COUNT_PER_TILE * sizeof(grad_16_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(dst_info_t)>;
    using msgrfwd32_accessor_t      = dg::network_tile_member_access::implementation::MsgrFwdAddressLookup<network_tile_member_access_signature, TILE_COUNT_MSGRFWD_32, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_32_t), LOGIT_COUNT_PER_TILE * sizeof(grad_32_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(dst_info_t)>;
    using msgrfwd64_accessor_t      = dg::network_tile_member_access::implementation::MsgrFwdAddressLookup<network_tile_member_access_signature, TILE_COUNT_MSGRFWD_64, PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_64_t), LOGIT_COUNT_PER_TILE * sizeof(grad_64_t), sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(dst_info_t)>;

    using msgrbwd8_accessor_t       = dg::network_tile_member_access::implementation::MsgrBwdAddressLookup<network_tile_member_access_signature, TILE_COUNT_MSGRBWD_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_8_t),   LOGIT_COUNT_PER_TILE * sizeof(grad_8_t),   sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(dst_info_t), sizeof(timein_t)>;
    using msgrbwd16_accessor_t      = dg::network_tile_member_access::implementation::MsgrBwdAddressLookup<network_tile_member_access_signature, TILE_COUNT_MSGRBWD_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_16_t),  LOGIT_COUNT_PER_TILE * sizeof(grad_16_t),  sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(dst_info_t), sizeof(timein_t)>;
    using msgrbwd32_accessor_t      = dg::network_tile_member_access::implementation::MsgrBwdAddressLookup<network_tile_member_access_signature, TILE_COUNT_MSGRBWD_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_32_t),  LOGIT_COUNT_PER_TILE * sizeof(grad_32_t),  sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(dst_info_t), sizeof(timein_t)>;
    using msgrbwd64_accessor_t      = dg::network_tile_member_access::implementation::MsgrBwdAddressLookup<network_tile_member_access_signature, TILE_COUNT_MSGRBWD_8,  PADDING_SZ, MEMREGION_SZ, sizeof(init_status_t), LOGIT_COUNT_PER_TILE * sizeof(logit_64_t),  LOGIT_COUNT_PER_TILE * sizeof(grad_64_t),  sizeof(observer_t), OBSERVER_ARRAY_SZ, sizeof(operatable_id_t), sizeof(dispatch_control_t), sizeof(pong_count_t), sizeof(addr_t), sizeof(dst_info_t), sizeof(timein_t)>;


    struct Resource{
        dg::unordered_unstable_map<uma_ptr_t, tile_polymorphic_t> region_id_map;
        std::unique_ptr<tile_polymorphic_t[]> region_id_table;
        uma_ptr_t region_first;
    };

    inline Resource * volatile resource{};

    //this equivalent to 3 flops per every call if not static - fine - this is better than bugs - volatile is deprecating - which is also a mistake - given the current state of compiler
    //volatile is defined differently by different compilers - yay
    //and the only definition that I liked is by GCC
    //the only scenerio where you use volatile is when the variable is accessed concurrently - and the value of the volatile variable might be wrongly assumed by the compiler (note that this is compiler's internal memory issue, not runtime memory issue) - such that the compiler might not see the initialization of the resource and assume that it is not initialized thus all operations are aborted
    //volatile forces a memory read instruction of the variable (from the compiler's perspective - not hardware perspective)  - such that all assumptions about the pointer must be aborted
    //and volatile only makes sense if std::add_volatile_t<std::add_pointer_t<T>> not std::add_pointer_t<std::add_volatile_t<T>> - this is an important note
    //the pointer is volatile - not the underlying data

    consteval auto get_memory_usage() -> size_t{

        return      leaf8_accessor_t::buf_size() + leaf16_accessor_t::buf_size() + leaf32_accessor_t::buf_size() + leaf64_accessor_t::buf_size()
                +   mono8_accessor_t::buf_size() + mono16_accessor_t::buf_size() + mono32_accessor_t::buf_size() + mono64_accessor_t::buf_size()
                +   pair8_accessor_t::buf_size() + pair16_accessor_t::buf_size() + pair32_accessor_t::buf_size() + pair64_accessor_t::buf_size()
                +   uacm8_accessor_t::buf_size() + uacm16_accessor_t::buf_size() + uacm32_accessor_t::buf_size() + uacm64_accessor_t::buf_size()
                +   pacm8_accessor_t::buf_size() + pacm16_accessor_t::buf_size() + pacm32_accessor_t::buf_size() + pacm64_accessor_t::buf_size()
                +   crit8_accessor_t::buf_size() + crit16_accessor_t::buf_size() + crit32_accessor_t::buf_size() + crit64_accessor_t::buf_size()
                +   msgrfwd8_accessor_t::buf_size() + msgrfwd16_accessor_t::buf_size() + msgrfwd32_accessor_t::buf_size() + msgrfwd64_accessor_t::buf_size()
                +   msgrbwd8_accessor_t::buf_size() + msgrbwd16_accessor_t::buf_size() + msgrbwd32_accessor_t::buf_size() + msgrbwd64_accessor_t::buf_size();
    } 

    void init(uma_ptr_t buf){

        Resource tmp_resource           = {};
        uma_ptr_t cur                   = buf;
        uma_ptr_t first                 = buf;
        uma_ptr_t last                  = dg::memult::advance(first, get_memory_usage()); 
        uma_ptr_t region_first          = dg::memult::region(first, std::integral_constant<size_t, MEMREGION_SZ>{});
        uma_ptr_t region_last           = dg::memult::align(last, std::integral_constant<size_t, MEMREGION_SZ>{});
        size_t table_sz                 = dg::memult::distance(region_first, region_last) / MEMREGION_SZ;
        tmp_resource.region_id_table    = std::make_unique<tile_polymorphic_t[]>(table_sz);
        tmp_resource.region_id_map      = {};
        tmp_resource.region_first       = region_first;

        auto initializer = [&]<class Accessor>(const Accessor, uma_ptr_t cur, uma_ptr_t table_head, tile_polymorphic_t tile_polymorphic){
            Accessor::init(cur);
            uma_ptr_t head = Accessor::get_head();

            for (size_t i = 0u; i < Accessor::tile_size(); ++i){
                uma_ptr_t id_ptr                        = Accessor::id_addr(dg::memult::advance(head, i));
                uma_ptr_t id_region                     = dg::memult::region(id_ptr, std::integral_constant<size_t, MEMREGION_SZ>{});
                uma_ptr_t idd_region                    = dg::memult::region(id_ptr, std::integral_constant<size_t, ID_MEMREGION_SZ>{});
                size_t table_idx                        = dg::memult::distance(table_head, id_region) / MEMREGION_SZ;
                tmp_resource.region_id_map[idd_region]  = tile_polymorphic;
                tmp_resource.region_id_table[table_idx] = tile_polymorphic;
            }

            return dg::memult::advance(cur, Accessor::tile_size());
        };
        
        cur = initializer(leaf8_accessor_t{},  cur, region_first, id_leaf_8);
        cur = initializer(leaf16_accessor_t{}, cur, region_first, id_leaf_16);
        cur = initializer(leaf32_accessor_t{}, cur, region_first, id_leaf_32);
        cur = initializer(leaf64_accessor_t{}, cur, region_first, id_leaf_64);
        
        cur = initializer(mono8_accessor_t{},  cur, region_first, id_mono_8);
        cur = initializer(mono16_accessor_t{}, cur, region_first, id_mono_16);
        cur = initializer(mono32_accessor_t{}, cur, region_first, id_mono_32);
        cur = initializer(mono64_accessor_t{}, cur, region_first, id_mono_64);

        cur = initializer(pair8_accessor_t{},  cur, region_first, id_pair_8);
        cur = initializer(pair16_accessor_t{}, cur, region_first, id_pair_16);
        cur = initializer(pair32_accessor_t{}, cur, region_first, id_pair_32);
        cur = initializer(pair64_accessor_t{}, cur, region_first, id_pair_64);
        
        cur = initializer(uacm8_accessor_t{},  cur, region_first, id_uacm_8);
        cur = initializer(uacm16_accessor_t{}, cur, region_first, id_uacm_16);
        cur = initializer(uacm32_accessor_t{}, cur, region_first, id_uacm_32);
        cur = initializer(uacm64_accessor_t{}, cur, region_first, id_uacm_64);

        cur = initializer(pacm8_accessor_t{},  cur, region_first, id_pacm_8);
        cur = initializer(pacm16_accessor_t{}, cur, region_first, id_pacm_16);
        cur = initializer(pacm32_accessor_t{}, cur, region_first, id_pacm_32);
        cur = initializer(pacm64_accessor_t{}, cur, region_first, id_pacm_64);
        
        cur = initializer(crit8_accessor_t{},  cur, region_first, id_crit_8);
        cur = initializer(crit16_accessor_t{}, cur, region_first, id_crit_16);
        cur = initializer(crit32_accessor_t{}, cur, region_first, id_crit_32);
        cur = initializer(crit64_accessor_t{}, cur, region_first, id_crit_64);
        
        cur = initializer(msgrfwd8_accessor_t{},  cur, region_first, id_msgrfwd_8);
        cur = initializer(msgrfwd16_accessor_t{}, cur, region_first, id_msgrfwd_16);
        cur = initializer(msgrfwd32_accessor_t{}, cur, region_first, id_msgrfwd_32);
        cur = initializer(msgrfwd64_accessor_t{}, cur, region_first, id_msgrfwd_64);

        cur = initializer(msgrbwd8_accessor_t{},  cur, region_first, id_msgrbwd_8);
        cur = initializer(msgrbwd8_accessor_t{},  cur, region_first, id_msgrbwd_16);
        cur = initializer(msgrbwd16_accessor_t{}, cur, region_first, id_msgrbwd_32);
        cur = initializer(msgrbwd32_accessor_t{}, cur, region_first, id_msgrbwd_64);

        resource = new Resource{std::move(tmp_resource)};
    }

    void deinit() noexcept{

        delete resource;
        resource = nullptr;
    }

    constexpr auto is_leaf_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_leaf_8) || (id == id_leaf_16) || (id == id_leaf_32) || (id == id_leaf_64);
    }

    constexpr auto is_mono_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_mono_8) || (id == id_mono_16) || (id == id_mono_32) || (id == id_mono_64);
    }

    constexpr auto is_pair_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_pair_8) || (id == id_pair_16) || (id == id_pair_32) || (id == id_pair_64);
    }

    constexpr auto is_uacm_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_uacm_8) || (id == id_uacm_16) || (id == id_uacm_32) || (id == id_uacm_64);
    } 

    constexpr auto is_pacm_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_pacm_8) || (id == id_pacm_16) || (id == id_pacm_32) || (id == id_pacm_64);
    }

    constexpr auto is_crit_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_crit_8) || (id == id_crit_16) || (id == id_crit_32) || (id == id_crit_64);
    }

    constexpr auto is_msgrfwd_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_msgrfwd_8) || (id == id_msgrfwd_16) || (id == id_msgrfwd_32) || (id == id_msgrfwd_64);
    }

    constexpr auto is_msgrbwd_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_msgrbwd_8) || (id == id_msgrbwd_16) || (id == id_msgrbwd_32) || (id == id_msgrbwd_64);
    }

    constexpr auto is_leaf8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_leaf_8;
    }

    constexpr auto is_mono8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_mono_8;
    }

    constexpr auto is_pair8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pair_8;
    }

    constexpr auto is_uacm8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_uacm_8;
    }

    constexpr auto is_pacm8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pacm_8;
    }

    constexpr auto is_crit8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_crit_8;
    }

    constexpr auto is_msgrfwd8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrfwd_8;
    }

    constexpr auto is_msgrbwd8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrbwd_8;
    } 

    constexpr auto is_leaf16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_leaf_16;
    }

    constexpr auto is_mono16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_mono_16;
    }

    constexpr auto is_pair16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pair_16;
    }

    constexpr auto is_uacm16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_uacm_16;
    }

    constexpr auto is_pacm16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pacm_16;
    }

    constexpr auto is_crit16_tile(tile_polymorphic_t id) noexcept -> bool{
        
        return id == id_crit_16;
    }

    constexpr auto is_msgrfwd16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrfwd_16;
    }

    constexpr auto is_msgrbwd16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrbwd_16;
    }

    constexpr auto is_leaf32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_leaf_32;
    }

    constexpr auto is_mono32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_mono_32;
    }

    constexpr auto is_pair32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pair_32;
    }

    constexpr auto is_uacm32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_uacm_32;
    }

    constexpr auto is_pacm32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pacm_32;
    }

    constexpr auto is_crit32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_crit_32;
    }

    constexpr auto is_msgrfwd32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrfwd_32;
    }

    constexpr auto is_msgrbwd32_tile(tile_polymorphic_t id) noexcept -> bool{
        
        return id == id_msgrbwd_32;
    }

    constexpr auto is_leaf64_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_leaf_64;
    }

    constexpr auto is_mono64_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_mono_64;
    }

    constexpr auto is_pair64_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pair_64;
    }

    constexpr auto is_uacm64_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_uacm_64;
    }

    constexpr auto is_pacm64_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pacm_64;
    }

    constexpr auto is_crit64_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_crit_64;
    }

    constexpr auto is_msgrfwd64_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrfwd_64;
    }

    constexpr auto is_msgrbwd64_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrbwd_64;
    }

    inline auto dg_typeid(uma_ptr_t ptr) noexcept -> tile_polymorphic_t{

        size_t slot = dg::memult::distance(resource->region_first, ptr) / MEMREGION_SZ;
        return resource->region_id_table[slot];
    }

    template <class CallBack>
    constexpr void get_leaf_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr, tile_polymorphic_t id) noexcept{

        if (is_leaf8_tile(id)){
            static_assert(noexcept(cb(leaf8_accessor_t{})));
            cb(leaf8_accessor_t{});
        } else if (is_leaf16_tile(id)){
            static_assert(noexcept(cb(leaf16_accessor_t{})));
            cb(leaf16_accessor_t{});
        } else if (is_leaf32_tile(id)){
            static_assert(noexcept(cb(leaf32_accessor_t{})));
            cb(leaf32_accessor_t{});
        } else if (is_leaf64_tile(id)){
            static_assert(noexcept(cb(leaf64_accessor_t{})));
            cb(leaf64_accessor_t{});
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }
    }

    template <class CallBack>
    constexpr void get_leaf_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr) noexcept{

        return get_leaf_static_polymorphic_accessor(cb, ptr, dg_typeid(ptr));
    }

    template <class CallBack>
    constexpr void get_mono_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr, tile_polymorphic_t id) noexcept{

        if (is_mono8_tile(id)){
            static_assert(noexcept(cb(mono8_accessor_t{})));
            cb(mono8_accessor_t{});
        } else if (is_mono16_tile(id)){
            static_assert(noexcept(cb(mono16_accessor_t{})));
            cb(mono16_accessor_t{});
        } else if (is_mono32_tile(id)){
            static_assert(noexcept(cb(mono32_accessor_t{})));
            cb(mono32_accessor_t{});
        } else if (is_mono64_tile(id)){
            static_assert(noexcept(cb(mono64_accessor_t{})));
            cb(mono64_accessor_t{});
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }
    }

    template <class CallBack>
    constexpr void get_mono_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr) noexcept{

        get_mono_static_polymorphic_accessor(cb, ptr, dg_typeid(ptr));
    }

    template <class CallBack>
    constexpr void get_pair_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr, tile_polymorphic_t id) noexcept{

        if (is_pair8_tile(id)){
            static_assert(noexcept(cb(pair8_accessor_t{})));
            cb(pair8_accessor_t{});
        } else if (is_pair16_tile(id)){
            static_assert(noexcept(cb(pair16_accessor_t{})));
            cb(pair16_accessor_t{});
        } else if (is_pair32_tile(id)){
            static_assert(noexcept(cb(pair32_accessor_t{})));
            cb(pair32_accessor_t{});
        } else if (is_pair64_tile(id)){
            static_assert(noexcept(cb(pair64_accessor_t{})));
            cb(pair64_accessor_t{});
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }
    }

    template <class CallBack>
    constexpr void get_pair_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr) noexcept{

        get_pair_static_polymorphic_accessor(cb, ptr, dg_typeid(ptr));
    }

    template <class CallBack>
    constexpr void get_uacm_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr, tile_polymorphic_t id) noexcept{

        if (is_uacm8_tile(id)){
            static_assert(noexcept(cb(uacm8_accessor_t{})));
            cb(uacm8_accessor_t{});
        } else if (is_uacm16_tile(id)){
            static_assert(noexcept(cb(uacm16_accessor_t{})));
            cb(uacm16_accessor_t{});
        } else if (is_uacm32_tile(id)){
            static_assert(noexcept(cb(uacm32_accessor_t{})));
            cb(uacm32_accessor_t{});
        } else if (is_uacm64_tile(id)){
            static_assert(noexcept(cb(uacm64_accessor_t{})));
            cb(uacm64_accessor_t{});
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }
    }

    template <class CallBack>
    constexpr void get_uacm_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr) noexcept{

        get_uacm_static_polymorphic_accessor(cb, ptr, dg_typeid(ptr));
    }

    template <class CallBack>
    constexpr void get_pacm_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr, tile_polymorphic_t id) noexcept{

        if (is_pacm8_tile(id)){
            static_assert(noexcept(cb(pacm8_accessor_t{})));
            cb(pacm8_accessor_t{});
        } else if (is_pacm16_tile(id)){
            static_assert(noexcept(cb(pacm16_accessor_t{})));
            cb(pacm16_accessor_t{});
        } else if (is_pacm32_tile(id)){
            static_assert(noexcept(cb(pacm32_accessor_t{})));
            cb(pacm32_accessor_t{});
        } else if (is_pacm64_tile(id)){
            static_assert(noexcept(cb(pacm64_accessor_t{})));
            cb(pacm64_accessor_t{});
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }
    }

    template <class CallBack>
    constexpr void get_pacm_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr) noexcept{

        get_pacm_static_polymorphic_accessor(cb, ptr, dg_typeid(ptr));
    }

    template <class CallBack>
    constexpr void get_crit_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr, tile_polymorphic_t id) noexcept{

        if (is_crit8_tile(id)){
            static_assert(noexcept(cb(crit8_accessor_t{})));
            cb(crit8_accessor_t{});
        } else if (is_crit16_tile(id)){
            static_assert(noexcept(cb(crit16_accessor_t{})));
            cb(crit16_accessor_t{});
        } else if (is_crit32_tile(id)){
            static_assert(noexcept(cb(crit32_accessor_t{})));
            cb(crit32_accessor_t{});
        } else if (is_crit64_tile(id)){
            static_assert(noexcept(cb(crit64_accessor_t{})));
            cb(crit64_accessor_t{});
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }
    }

    template <class CallBack>
    constexpr void get_crit_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr) noexcept{

        get_crit_static_polymorphic_accessor(cb, ptr, dg_typeid(ptr));
    } 

    template <class CallBack>
    constexpr void get_msgrfwd_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr, tile_polymorphic_t id) noexcept{

        if (is_msgrfwd8_tile(id)){
            static_assert(noexcept(cb(msgrfwd8_accessor_t{})));
            cb(msgrfwd8_accessor_t{});
        } else if (is_msgrfwd16_tile(id)){
            static_assert(noexcept(cb(msgrfwd16_accessor_t{})));
            cb(msgrfwd16_accessor_t{});
        } else if (is_msgrfwd32_tile(id)){
            static_assert(noexcept(cb(msgrfwd32_accessor_t{})));
            cb(msgrfwd32_accessor_t{});
        } else if (is_msgrfwd64_tile(id)){
            static_assert(noexcept(cb(msgrfwd64_accessor_t{})));
            cb(msgrfwd64_accessor_t{});
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }
    }

    template <class CallBack>
    constexpr void get_msgrfwd_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr) noexcept{

        get_msgrfwd_static_polymorphic_accessor(cb, ptr, dg_typeid(ptr));
    }

    template <class CallBack>
    constexpr void get_msgrbwd_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr, tile_polymorphic_t id) noexcept{

        if (is_msgrbwd8_tile(id)){
            static_assert(noexcept(cb(msgrbwd8_accessor_t{})));
            cb(msgrbwd8_accessor_t{});
        } else if (is_msgrbwd16_tile(id)){
            static_assert(noexcept(cb(msgrbwd16_accessor_t{})));
            cb(msgrbwd16_accessor_t{});
        } else if (is_msgrbwd32_tile(id)){
            static_assert(noexcept(cb(msgrbwd32_accessor_t{})));
            cb(msgrbwd32_accessor_t{});
        } else if (is_msgrbwd64_tile(id)){
            static_assert(noexcept(cb(msgrbwd64_accessor_t{})));
            cb(msgrbwd64_accessor_t{});
        } else{
            if constexpr(DEBUG_MODE_FLAG){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                std::abort();
            } else{
                std::unreachable();
            }
        }
    }

    template <class CallBack>
    constexpr void get_msgrbwd_static_polymorphic_accessor(const CallBack& cb, uma_ptr_t ptr){

        get_msgrbwd_static_polymorphic_accessor(cb, ptr, dg_typeid(ptr));
    }

    inline auto safecthrow_leaf_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        uma_ptr_t id_region     = dg::memult::region(ptr, std::integral_constant<size_t, ID_MEMREGION_SZ>{});
        auto map_ptr            = resource->region_id_map.find(id_region);

        if (map_ptr == resource->region_id_map.end()){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        if (!is_leaf_tile(map_ptr->second)){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }
        
        return ptr;
    }

    inline auto safecthrow_mono_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        uma_ptr_t id_region     = dg::memult::region(ptr, std::integral_constant<size_t, ID_MEMREGION_SZ>{});
        auto map_ptr            = resource->region_id_map.find(id_region);

        if (map_ptr == resource->region_id_map.end()){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        if (!is_mono_tile(map_ptr->second)){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        return ptr;
    }

    auto safecthrow_pair_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        uma_ptr_t id_region     = dg::memult::region(ptr, std::integral_constant<size_t, ID_MEMREGION_SZ>{});
        auto map_ptr            = resource->region_id_map.find(id_region);

        if (map_ptr == resource->region_id_map.end()){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        if (!is_pair_tile(map_ptr->second)){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        return ptr;
    }

    auto safecthrow_uacm_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        uma_ptr_t id_region     = dg::memult::region(ptr, std::integral_constant<size_t, ID_MEMREGION_SZ>{});
        auto map_ptr            = resource->region_id_map.find(id_region);

        if (map_ptr == resource->region_id_map.end()){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        if (!is_uacm_tile(map_ptr->second)){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        return ptr;
    }

    auto safecthrow_pacm_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        uma_ptr_t id_region     = dg::memult::region(ptr, std::integral_constant<size_t, ID_MEMREGION_SZ>{});
        auto map_ptr            = resource->region_id_map.find(id_region);

        if (map_ptr == resource->region_id_map.end()){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        if (!is_pacm_tile(map_ptr->second)){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        return ptr;
    }

    auto safecthrow_crit_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        uma_ptr_t id_region     = dg::memult::region(ptr, std::integral_constant<size_t, ID_MEMREGION_SZ>{});
        auto map_ptr            = resource->region_id_map.find(id_region);

        if (map_ptr == resource->region_id_map.end()){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        if (!is_crit_tile(map_ptr->second)){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        return ptr;
    }

    auto safecthrow_msgrfwd_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        uma_ptr_t id_region     = dg::memult::region(ptr, std::integral_constant<size_t, ID_MEMREGION_SZ>{});
        auto map_ptr            = resource->region_id_map.find(id_region);

        if (map_ptr == resource->region_id_map.end()){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        if (!is_msgrfwd_tile(map_ptr->second)){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        return ptr;
    }

    auto safecthrow_msgrbwd_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        uma_ptr_t id_region     = dg::memult::region(ptr, std::integral_constant<size_t, ID_MEMREGION_SZ>{});
        auto map_ptr            = resource->region_id_map.find(id_region);

        if (map_ptr == resource->region_id_map.end()){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        if (!is_msgrbwd_tile(map_ptr->second)){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        return ptr;
    }

    auto safecthrow_tile_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        uma_ptr_t id_region     = dg::memult::region(ptr, std::integral_constant<size_t, ID_MEMREGION_SZ>{});
        auto map_ptr            = resource->region_id_map.find(id_region);

        if (map_ptr == resource->region_id_map.end()){
            return std::unexpected(dg::network_exception::BAD_ACCESS);
        }

        return ptr;
    }

    auto safe_leaf_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            return dg::network_exception_handler::nothrow_log(safecthrow_leaf_ptr_access(ptr));
        } else{
            return ptr;
        }
    }

    auto safe_mono_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            return dg::network_exception_handler::nothrow_log(safecthrow_mono_ptr_access(ptr));
        } else{
            return ptr;
        }
    }

    auto safe_pair_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            return dg::network_exception_handler::nothrow_log(safecthrow_pair_ptr_access(ptr));
        } else{
            return ptr;
        }
    }

    auto safe_uacm_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            return dg::network_exception_handler::nothrow_log(safecthrow_uacm_ptr_access(ptr));
        } else{
            return ptr;
        }
    }

    auto safe_pacm_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            return dg::network_exception_handler::nothrow_log(safecthrow_pacm_ptr_access(ptr));
        } else{
            return ptr;
        }
    }

    auto safe_crit_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            return dg::network_exception_handler::nothrow_log(safecthrow_crit_ptr_access(ptr));
        } else{
            return ptr;
        }
    }

    auto safe_msgrfwd_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            return dg::network_exception_handler::nothrow_log(safecthrow_msgrfwd_ptr_access(ptr));
        } else{
            return ptr;
        }
    }

    auto safe_msgrbwd_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            return dg::network_exception_handler::nothrow_log(safecthrow_msgrbwd_ptr_access(ptr));
        } else{
            return ptr;
        }
    }

    auto safe_tile_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

        if constexpr(IS_SAFE_ACCESS_ENABLED){
            return dg::network_exception_handler::nothrow_log(safecthrow_tile_ptr_access(ptr));
        } else{
            return ptr;
        }
    }

    auto safethrow_leaf_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safecthrow_leaf_ptr_access(ptr));
    }

    auto safethrow_mono_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safecthrow_mono_ptr_access(ptr));
    }

    auto safethrow_pair_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safecthrow_pair_ptr_access(ptr));
    }

    auto safethrow_uacm_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safecthrow_uacm_ptr_access(ptr));
    }
    
    auto safethrow_pacm_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safecthrow_pacm_ptr_access(ptr));
    }

    auto safethrow_crit_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safecthrow_crit_ptr_access(ptr));
    }

    auto safethrow_msgrfwd_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safecthrow_msgrfwd_ptr_access(ptr));
    }

    auto safethrow_msgrbwd_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safecthrow_msgrbwd_ptr_access(ptr));
    }

    auto safethrow_tile_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safecthrow_tile_ptr_access(ptr));
    }
}

#endif