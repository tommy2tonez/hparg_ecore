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

namespace dg::network_tile_member_access_template{
    
    using uma_ptr_t = dg::network_pointer::uma_ptr_t;

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ>
    struct LeafAddressLookup{

        private:

            using self          = LeafAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline uma_ptr_t head{}; 

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            } 

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::get_head(), ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            }

            static constexpr auto offset_init_status(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + (self::offset_id(TILE_COUNT) + PADDING_SZ);
            } 

            static constexpr auto offset_observer_addr(size_t idx) noexcept -> size_t{
                
                return idx * OBSERVER_VALUE_SZ + (self::offset_init_status(TILE_COUNT) + PADDING_SZ); 
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + (self::offset_observer_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + (self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + (self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + (self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            } 

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + (self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static void init(uma_ptr_t buf){

                self::head = buf;
                access_ins::init(buf, dg::memult::advance(buf, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto size() -> size_t{

                return self::offset_pong_count_addr(TILE_COUNT);
            } 

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status(self::index(access_ins::access(ptr))));
            } 

            static inline auto observer_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr))));
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

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ>
    struct MonoAddressLookup{

        private:

            using self          = MonoAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>; 

            static inline uma_ptr_t head{};

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::get_head(), ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{
                
                return idx;
            }

            static constexpr auto offset_init_status(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + (self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_observer_addr(size_t idx) noexcept -> size_t{

                return idx * OBSERVER_VALUE_SZ + (self::offset_init_status(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + (self::offset_observer_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + (self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + (self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + (self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + (self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }
            
            static constexpr auto offset_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + (self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static void init(uma_ptr_t buf){

                self::head = buf;
                access_ins::init(buf, dg::memult::advance(buf, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto size() -> size_t{

                return self::offset_old_addr(TILE_COUNT);
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status(self::index(access_ins::access(ptr))));
            }

            static inline auto observer_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr))));
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
 
    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ, size_t ACM_SZ>
    struct UACMAddressLookup{
        
        private:

            using self          = UACMAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline uma_ptr_t head{};

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::get_head(), ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            } 

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + (self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_observer_addr(size_t idx) noexcept -> size_t{

                return idx * OBSERVER_VALUE_SZ + (self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + (self::offset_observer_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + (self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + (self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + (self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + (self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

            template <size_t ACM_IDX>
            static constexpr auto offset_descendant_addr(size_t idx, const std::integral_constant<size_t, ACM_IDX>) noexcept -> size_t{

                return idx * (ACM_SZ * DESCENDANT_SZ) + (self::offset_pong_count_addr(TILE_COUNT) + ACM_IDX * DESCENDANT_SZ + PADDING_SZ);
            }

        public:

            static void init(uma_ptr_t buf){

                self::head = buf;
                access_ins::init(buf, dg::memult::advance(buf, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto size() -> size_t{

                return self::offset_descendant_addr(TILE_COUNT, std::integral_constant<size_t, 0>{});
            }

            static consteval auto accum_size() -> size_t{

                return ACM_SZ;
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto observer_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr))));
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

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ, size_t ACM_SZ>
    struct PACMAddressLookup{

        private:

            using self          = PACMAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline uma_ptr_t head{};

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::get_head(), ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            } 

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + (self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_observer_addr(size_t idx) noexcept -> size_t{

                return idx * OBSERVER_VALUE_SZ + (self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + (self::offset_observer_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + (self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + (self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + (self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + (self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

            template <size_t ACM_IDX>
            static constexpr auto offset_left_descendant_addr(size_t idx, const std::integral_constant<size_t, ACM_IDX>) noexcept -> size_t{

                return idx * (DESCENDANT_SZ * ACM_SZ) + (self::offset_pong_count_addr(TILE_COUNT) + ACM_IDX * DESCENDANT_SZ + PADDING_SZ);
            }

            template <size_t ACM_IDX>
            static constexpr auto offset_right_descendant_addr(size_t idx, const std::integral_constant<size_t, ACM_IDX>) noexcept -> size_t{

                return idx * (DESCENDANT_SZ * ACM_SZ) + (self::offset_left_descendant_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + ACM_IDX * DESCENDANT_SZ + PADDING_SZ);
            }

        public:

            static void init(uma_ptr_t buf){

                self::head = buf;
                access_ins::init(buf, dg::memult::advance(buf, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto size() -> size_t{

                return self::offset_right_descendant_addr(TILE_COUNT, std::integral_constant<size_t, 0>{});
            }

            static consteval auto accum_size() -> size_t{

                return ACM_SZ;
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto observer_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr))));
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

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ>
    struct PairAddressLookup{

        private:

            using self = PairAddressLookup;
            using access_ins = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>; 

            static inline uma_ptr_t head{};

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::get_head(), ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            } 

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + (self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_observer_addr(size_t idx) noexcept -> size_t{

                return idx * OBSERVER_VALUE_SZ + (self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + (self::offset_observer_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + (self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + (self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + (self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + (self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_left_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + (self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_right_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + (self::offset_left_descendant_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static void init(uma_ptr_t buf){

                self::head = buf;
                access_ins::init(buf, dg::memult::advance(buf, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head; 
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto observer_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr))));
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

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ, size_t CRIT_KIND_SZ>
    struct CritAddressLookup{

        private:

            using self = CritAddressLookup;
            using access_ins = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline uma_ptr_t head{}; 

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            } 

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::get_head(), ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            }
            
            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + (self::offest_id(TILE_COUNT) + PADDING_SZ); 
            } 

            static constexpr auto offset_observer_addr(size_t idx) noexcept -> size_t{

                return idx * OBSERVER_VALUE_SZ + (self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + (self::offset_observer_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + (self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + (self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + (self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + (self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + (self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_crit_kind_addr(size_t idx) noexcept -> size_t{

                return idx * CRIT_KIND_SZ + (self::offset_descendant_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static void init(uma_ptr_t buf){

                self::head = buf;
                access_ins::init(buf, dg::memult::advance(buf, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto observer_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr))));
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

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ, size_t DST_INFO_SZ>
    struct MsgrFwdAddressLookup{

        private:

            using self          = MsgrFwdAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>; 

            static inline uma_ptr_t head{};

            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            }

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::get_head(), ptr);
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            }

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + (self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_observer_addr(size_t idx) noexcept -> size_t{

                return idx * OBSERVER_VALUE_SZ + (self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + (self::offset_observer_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + (self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ + (self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + (self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + (self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + (self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dst_info_addr(size_t idx) noexcept -> size_t{

                return idx * DST_INFO_SZ + (self::offset_descendant_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static void init(uma_ptr_t buf){

                self::head = buf;
                access_ins::init(buf, dg::memult::advance(buf, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto size() -> size_t{

                return self::offset_dst_info_addr(TILE_COUNT);
            } 

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            }

            static inline auto observer_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr))));
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

            static inline auto rcu_lock_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{ //so it's fine to build hierarchical memory tree - to serialize access at a certain address - but its not fine to just acquire the rcu_lock_addr and operates on other class members without referencing their memory regions -  because memory region might be acquired to do memory_map or other operations at any given time - so it would be undefined semantically -  

                return self::tile_logit_addr(ptr);
            }

    };

    template <class ID, size_t TILE_COUNT, size_t PADDING_SZ, size_t INIT_STATUS_SZ, size_t LOGIT_VALUE_SZ, size_t GRAD_VALUE_SZ, size_t OBSERVER_VALUE_SZ, size_t OPERATABLE_ID_SZ, size_t DISPATCH_CONTROL_SZ, size_t PONG_COUNT_SZ, size_t DESCENDANT_SZ, size_t DST_INFO_SZ, size_t TIMEIN_SZ>
    struct MsgrBwdAddressLookup{

        private:

            using self          = MsgrBwdAddressLookup;
            using access_ins    = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;

            static inline uma_ptr_t head{};
            
            static inline auto get_head() noexcept -> uma_ptr_t{

                return self::head;
            } 

            static inline auto index(uma_ptr_t ptr) noexcept -> size_t{

                return dg::memult::distance(self::head, ptr);
            } 

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            } 

            static constexpr auto offset_init_status_addr(size_t idx) noexcept -> size_t{

                return idx * INIT_STATUS_SZ + (self::offset_id(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_observer_addr(size_t idx) noexcept -> size_t{

                return idx * OBSERVER_VALUE_SZ + (self::offset_init_status_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * LOGIT_VALUE_SZ + (self::offset_observer_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * GRAD_VALUE_SZ + (self::offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_operatable_id_addr(size_t idx) noexcept -> size_t{

                return idx * OPERATABLE_ID_SZ * (self::offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * DISPATCH_CONTROL_SZ + (self::offset_operatable_id_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * PONG_COUNT_SZ + (self::offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            } 

            static constexpr auto offset_descendant_addr(size_t idx) noexcept -> size_t{

                return idx * DESCENDANT_SZ + (self::offset_pong_count_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dst_info_addr(size_t idx) noexcept -> size_t{

                return idx * DST_INFO_SZ + (self::offset_descendant_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_timein_addr(size_t idx) noexcept -> size_t{

                return idx * TIMEIN_SZ + (self::offset_dst_info_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static void init(uma_ptr_t buf){
                
                self::head = buf;
                access_ins::init(buf, dg::memult::advance(buf, TILE_COUNT));
            }

            static void deinit() noexcept{

                (void) self::head;
            }

            static consteval auto size() -> size_t{

                return self::offset_timein_addr(TILE_COUNT);
            }

            static inline auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return access_ins::access(ptr);
            }

            static inline auto init_status_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_init_status_addr(self::index(access_ins::access(ptr))));
            } 

            static inline auto observer_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return dg::memult::advance(self::get_head(), self::offset_observer_addr(self::index(access_ins::access(ptr))));
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

    //the good thing about this is that static inline variables would be optimized -> table_lookup of uma_ptr_t pluses a constant - detected by compiler - if switch case is implemented correctly - std::unreachable() for default:
    //this needs a lot of set up - from uniform size - to inline - to switch case no default - etc. - to the right compiler
    //this is only initialized once - at the beginning at the program - before instantiating concurrency - important - and cannot be deinitialized until all threads are joined - undefined otherwise
    //better to keep the identity buffer internally - not to write it on uma_ptr_t - it's fine - but it's slow - and not worth optimizing
    //the identity buffer has to be allocated by kmalloc + has preciesely 3 flops to read the polymorphic_kind of the tile (assume all those are cached)

    static_assert(sizeof(char) == 1);   
    static_assert(CHAR_BIT == 8);
    
    static inline constexpr size_t TYPE_COUNT               = 15;
    static inline constexpr size_t BUF_SZ                   = size_t{1} << 25;
    static inline constexpr size_t PADDING_SZ               = size_t{1} << 10;
    static inline constexpr size_t ALIGNMENT_SZ             = size_t{1} << 10;
    static inline constexpr size_t LOGIT_COUNT_PER_TILE     = size_t{1} << 10;
    static inline constexpr size_t UNORDERED_ACCUM_SZ       = size_t{1} << 5;
    static inline constexpr size_t LINEAR_GROUP_SZ          = size_t{1} << 5;
    using tile_polymorphic_t = uint8_t; 

    enum object_identifier_option: tile_polymorphic_t{
        id_leaf_8       = 0u,
        id_leaf_16      = 1u,
        id_leaf_32      = 2u,
        id_mono_8       = 3u,
        id_mono_16      = 4u,
        id_mono_32      = 5u,
        id_uacm_8       = 6u,
        id_uacm_16      = 7u,
        id_uacm_32      = 8u,
        id_pacm_8       = 9u,
        id_pacm_16      = 10u,
        id_pacm_32      = 11u,
        id_pair_8       = 12u,
        id_pair_16      = 13u,
        id_pair_32      = 14u,
        id_crit_8       = 15u,
        id_crit_16      = 16u,
        id_crit_32      = 17u,
        id_msgrfwd_8    = 18u,
        id_msgrfwd_16   = 19u,
        id_msgrfwd_32   = 20u,
        id_msgrbwd_8    = 21u,
        id_msgrbwd_16   = 22u,
        id_msgrbwd_32   = 23u
    };

    using identity_t                = uint8_t;
    using observer_value_t          = std::array<char, 256>; //each stable ptr have maximum 256-byte observable registration (backward reference)
    using operatable_id_t           = uint64_t;
    using addr_t                    = uint64_t; 

    using logit_value_8_t           = std::array<char, LOGIT_COUNT_PER_TILE * sizeof(uint8_t)>; //buggy
    using logit_value_16_t          = std::array<char, LOGIT_COUNT_PER_TILE * sizeof(uint16_t)>; //buggy
    using grad_value_8_t            = std::array<char, LOGIT_COUNT_PER_TILE * sizeof(uint8_t)>; //buggy
    using grad_value_16_t           = std::array<char, LOGIT_COUNT_PER_TILE * sizeof(uint16_t)>; //buggy

    using leaf8_accessor_t          = void *;
    using leaf16_accessor_t         = void *;
    using leaf32_accessor_t         = void *;
    using mono8_accessor_t          = void *;
    using mono16_accessor_t         = void *;
    using mono32_accessor_t         = void *;
    using pair8_accessor_t          = void *;
    using pair16_accessor_t         = void *;
    using pair32_accessor_t         = void *;
    using uacm8_accesor_t           = void *;
    using uacm16_accessor_t         = void *;
    using uacm32_accessor_t         = void *;
    using pacm8_accessor_t          = void *;
    using pacm16_accessor_t         = void *;
    using pacm32_accesor_t          = void *;
    using crit8_accessor_t          = void *;
    using crit16_accessor_t         = void *;
    using crit32_accessor_t         = void *;
    using msgrfwd8_accessor_t       = void *;
    using msgrfwd16_accessor_t      = void *;
    using msgrfwd32_accessor_t      = void *;
    using msgrbwd8_accessor_t       = void *;
    using msgrbwd16_accessor_t      = void *;
    using msgrbwd32_accessor_t      = void *;
    using uma_ptr_t                 = dg::network_pointer::uma_ptr_t;

    // auto is_leaf_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return (id == id_leaf_8) || (id == id_leaf_16) || (id == id_leaf_32);
    // }

    // auto is_mono_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return (id == id_mono_8) || (id == id_mono_16) || (id == id_mono_32);
    // }

    // auto is_pair_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return (id == id_pair_8) || (id == id_pair_16) || (id == id_pair_32);
    // }

    // auto is_uacm_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return (id == id_uacm_8) || (id == id_uacm_16) || (id == id_uacm_32);
    // } 

    // auto is_pacm_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return (id == id_pacm_8) || (id == id_pacm_16) || (id == id_pacm_32);
    // }

    // auto is_crit_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return (id == id_crit_8) || (id == id_crit_16) || (id == id_crit_32);
    // }

    // auto is_msgrfwd_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return (id == id_msgrfwd_8) || (id == id_msgrfwd_16) || (id == id_msgrfwd_32);
    // }

    // auto is_msgrbwd_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return (id == id_msgrbwd_8) || (id == id_msgrbwd_16) || (id == id_msgrbwd_32);
    // }

    // auto is_leaf8_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_leaf_8;
    // }

    // auto is_mono8_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_mono_8;
    // }

    // auto is_pair8_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_pair_8;
    // }

    // auto is_uacm8_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_uacm_8;
    // }

    // auto is_pacm8_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_pacm_8;
    // }

    // auto is_crit8_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_crit_8;
    // }

    // auto is_msgrfwd8_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_msgrfwd_8;
    // }

    // auto is_msgrbwd8_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_msgrbwd_8;
    // } 

    // auto is_leaf16_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_leaf_16;
    // }

    // auto is_mono16_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_mono_16;
    // }

    // auto is_pair16_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_pair_16;
    // }

    // auto is_uacm16_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_uacm_16;
    // }

    // auto is_pacm16_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_pacm_16;
    // }

    // auto is_crit16_tile(tile_polymorphic_t id) noexcept -> bool{
        
    //     return id == id_crit_16;
    // }

    // auto is_msgrfwd16_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_msgrfwd_16;
    // }

    // auto is_msgrbwd16_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_msgrbwd_16;
    // }

    // auto is_leaf32_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_leaf_32;
    // }

    // auto is_mono32_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_mono_32;
    // }

    // auto is_pair32_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_pair_32;
    // }

    // auto is_uacm32_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_uacm_32;
    // }

    // auto is_pacm32_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_pacm_32;
    // }

    // auto is_crit32_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_crit_32;
    // }

    // auto is_msgrfwd32_tile(tile_polymorphic_t id) noexcept -> bool{

    //     return id == id_msgrfwd_32;
    // }

    // auto is_msgrbwd32_tile(tile_polymorphic_t id) noexcept -> bool{
        
    //     return id == id_msgrbwd_32;
    // }

    // auto dg_typeid(uma_ptr_t ptr) noexcept -> tile_polymorphic_t{

    //     tile_polymorphic_t id{};
    //     void * dst      = &id;
    //     uma_ptr_t src   = ptr;
    //     // dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(tile_polymorphic_t));

    //     return id;
    // } 

    // template <class CallBack>
    // void get_leaf_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

    //     tile_polymorphic_t id = dg_typeid(ptr);

    //     if (is_leaf8_tile(id)){
    //         cb(leaf8_accessor_t{});
    //         return;
    //     }

    //     if (is_leaf16_tile(id)){
    //         cb(leaf16_accessor_t{});
    //         return;
    //     }

    //     if (is_leaf32_tile(id)){
    //         cb(leaf32_accessor_t{});
    //         return;
    //     }

    //     dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
    //     std::abort();
    // }

    // template <class CallBack>
    // void get_mono_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

    //     tile_polymorphic_t id = dg_typeid(ptr);

    //     if (is_mono8_tile(id)){
    //         cb(mono8_accessor_t{});
    //         return;
    //     }

    //     if (is_mono16_tile(id)){
    //         cb(mono16_accessor_t{});
    //         return;
    //     }

    //     if (is_mono32_tile(id)){
    //         cb(mono32_accessor_t{});
    //         return;
    //     }

    //     dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
    //     std::abort();
    // }

    // template <class CallBack>
    // void get_pair_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

    //     tile_polymorphic_t id = dg_typeid(ptr);

    //     if (is_pair8_tile(id)){
    //         cb(pair8_accessor_t{});
    //         return;
    //     }

    //     if (is_pair16_tile(id)){
    //         cb(pair16_accessor_t{});
    //         return;
    //     }

    //     if (is_pair32_tile(id)){
    //         cb(pair32_accessor_t{});
    //         return;
    //     }

    //     dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
    //     std::abort();
    // }

    // template <class CallBack>
    // void get_uacm_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

    //     tile_polymorphic_t id = dg_typeid(ptr);

    //     if (is_uacm8_tile(id)){
    //         cb(uacm8_accesor_t{});
    //         return;
    //     }

    //     if (is_uacm16_tile(id)){
    //         cb(uacm16_accesor_t{});
    //         return;
    //     }

    //     if (is_uacm_32_tile(id)){
    //         cb(uacm32_accesor_t{});
    //         return;
    //     }

    //     dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
    //     std::abort();
    // }

    // template <class CallBack>
    // void get_pacm_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

    //     tile_polymorphic_t id = dg_typeid(ptr);

    //     if (is_pacm8_tile(id)){
    //         cb(pacm8_accessor_t{});
    //         return;
    //     }

    //     if (is_pacm16_tile(id)){
    //         cb(pacm16_accessor_t{});
    //         return;
    //     }

    //     if (is_pacm32_tile(id)){
    //         cb(pacm32_accessor_t{});
    //         return;
    //     }

    //     dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
    //     std::abort();
    // }

    // template <class CallBack>
    // void get_crit_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

    //     tile_polymorphic_t id = dg_typeid(ptr);

    //     if (is_crit8_tile(id)){
    //         cb(crit8_accessor_t{});
    //         return;
    //     }

    //     if (is_crit16_tile(id)){
    //         cb(crit16_accessor_t{});
    //         return;
    //     }

    //     if (is_crit32_tile(id)){
    //         cb(crit32_accessor_t{});
    //         return;
    //     }

    //     dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
    //     std::abort();
    // }

    // template <class CallBack>
    // void get_msgrfwd_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

    //     tile_polymorphic_t id = dg_typeid(ptr);

    //     if (is_msgrfwd8_tile(id)){
    //         cb(msgrfwd8_accessor_t{});
    //         return;
    //     }

    //     if (is_msgrfwd16_tile(id)){
    //         cb(msgrfwd16_accessor_t{});
    //         return;
    //     }

    //     if (is_msgrfwd32_tile(id)){
    //         cb(msgrfwd32_accessor_t{});
    //         return;
    //     }

    //     dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
    //     std::abort();
    // }

    // template <class CallBack>
    // void get_msgrbwd_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

    //     tile_polymorphic_t id = dg_typeid(ptr);

    //     if (is_msgrbwd8_tile(id)){
    //         cb(msgrbwd8_accessor_t{});
    //         return;
    //     }

    //     if (is_msgrbwd16_tile(id)){
    //         cb(msgrbwd16_accessor_t{});
    //         return;
    //     }

    //     if (is_msgrbwd32_tile(id)){
    //         cb(msgrbwd32_accessor_t{});
    //         return;
    //     }

    //     dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
    //     std::abort();
    // }

    // auto safecthrow_leaf_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

    //     return safe_leaf_ptr_access_instance::access(ptr);
    // }

    // auto safecthrow_mono_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

    //     return safe_mono_ptr_access_instance::access(ptr);
    // }

    // auto safecthrow_pair_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

    //     return safe_pair_ptr_access_instance::access(ptr);
    // }

    // auto safecthrow_uacm_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

    //     return safe_uacm_ptr_access_instance::access(ptr);
    // }

    // auto safecthrow_pacm_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

    //     return safe_pacm_ptr_access_instance::access(ptr);
    // }

    // auto safecthrow_crit_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

    //     return safe_crit_ptr_access_instance::access(ptr);
    // }

    // auto safecthrow_msgrfwd_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

    //     return safe_msgrfwd_ptr_access_instance::access(ptr);
    // }

    // auto safecthrow_msgrbwd_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

    //     return safe_msgrbwd_ptr_access_instance::access(ptr);
    // }

    // auto safecthrow_tile_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

    //     return safe_tile_ptr_access_instance::access(ptr);
    // }

    // auto safe_leaf_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    //     if constexpr(IS_SAFE_ACCESS_ENABLED){
    //         return dg::network_exception_handler::nothrow_log(safecthrow_leaf_ptr_access(ptr));
    //     } else{
    //         return ptr;
    //     }
    // }

    // auto safe_mono_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    //     if constexpr(IS_SAFE_ACCESS_ENABLED){
    //         return dg::network_exception_handler::nothrow_log(safecthrow_mono_ptr_access(ptr));
    //     } else{
    //         return ptr;
    //     }
    // }

    // auto safe_pair_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    //     if constexpr(IS_SAFE_ACCESS_ENABLED){
    //         return dg::network_exception_handler::nothrow_log(safecthrow_pair_ptr_access(ptr));
    //     } else{
    //         return ptr;
    //     }
    // }

    // auto safe_uacm_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    //     if constexpr(IS_SAFE_ACCESS_ENABLED){
    //         return dg::network_exception_handler::nothrow_log(safecthrow_uacm_ptr_access(ptr));
    //     } else{
    //         return ptr;
    //     }
    // }

    // auto safe_pacm_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    //     if constexpr(IS_SAFE_ACCESS_ENABLED){
    //         return dg::network_exception_handler::nothrow_log(safecthrow_pacm_ptr_access(ptr));
    //     } else{
    //         return ptr;
    //     }
    // }

    // auto safe_crit_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    //     if constexpr(IS_SAFE_ACCESS_ENABLED){
    //         return dg::network_exception_handler::nothrow_log(safecthrow_crit_ptr_access(ptr));
    //     } else{
    //         return ptr;
    //     }
    // }

    // auto safe_msgrfwd_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    //     if constexpr(IS_SAFE_ACCESS_ENABLED){
    //         return dg::network_exception_handler::nothrow_log(safecthrow_msgrfwd_ptr_access(ptr));
    //     } else{
    //         return ptr;
    //     }
    // }

    // auto safe_msgrbwd_ptr_access(uma_ptr_t ptr) noexcept -> uma_ptr_t{

    //     if constexpr(IS_SAFE_ACCESS_ENABLED){
    //         return dg::network_exception_handler::nothrow_log(safecthrow_msgrbwd_ptr_access(ptr));
    //     } else{
    //         return ptr;
    //     }
    // }

    // auto safethrow_leaf_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

    //     return dg::network_exception_handler::throw_log(safecthrow_leaf_ptr_access(ptr));
    // }

    // auto safethrow_mono_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

    //     return dg::network_exception_handler::throw_log(safecthrow_mono_ptr_access(ptr));
    // }

    // auto safethrow_pair_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

    //     return dg::network_exception_handler::throw_log(safecthrow_pair_ptr_access(ptr));
    // }

    // auto safethrow_uacm_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

    //     return dg::network_exception_handler::throw_log(safecthrow_uacm_ptr_access(ptr));
    // }
    
    // auto safethrow_pacm_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

    //     return dg::network_exception_handler::throw_log(safecthrow_pacm_ptr_access(ptr));
    // }

    // auto safethrow_crit_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

    //     return dg::network_exception_handler::throw_log(safecthrow_crit_ptr_access(ptr));
    // }

    // auto safethrow_msgrfwd_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

    //     return dg::network_exception_handler::throw_log(safecthrow_msgrfwd_ptr_access(ptr));
    // }

    // auto safethrow_msgrbwd_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

    //     return dg::network_exception_handler::throw_log(safecthrow_msgrbwd_ptr_access(ptr));
    // }
}

#endif