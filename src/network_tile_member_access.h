#include <stdint.h>
#include <stdlib.h>
#include <array>
#include <limits.h>
#include <cstring>
#include "network_segcheck_bound.h"
#include "network_uma.h"
#include "network_memult.h"
#include "network_exception_handler.h"
#include "network_memops_uma.h"

namespace dg::network_tile_member_access_template{
    
    using uma_ptr_t = dg::network_uma::uma_ptr_t;
    static_assert(dg::is_ptr_v<uma_ptr_t>);

    template <class ...Args>
    struct tags{};

    template <uint8_t ID>
    struct PtrAccess{

        using self  = PtrAccess;
        using base  = dg::network_segcheck_bound::StdAccess<self, uma_ptr_t>;  

        static void init(uma_ptr_t buf, size_t buf_sz) noexcept{

            base::init(buf, memult::forward(buf, buf_sz));
        }

        static constexpr auto access(uma_ptr_t buf) noexcept -> uma_ptr_t{

            return base::access(buf);
        }
    };

    template <size_t TILE_COUNT, size_t PADDING_SZ, class identity_t, class logit_value_t, class grad_value_t, class observing_value_t, class bit_control_t, class dispatch_control_t, class pong_count_t, uint8_t ID>
    struct LeafAddressLookup: protected PtrAccess<ID>{
        
        private:

            static inline uma_ptr_t head{}; 

        protected:

            static inline auto get_head() noexcept -> uma_ptr_t{

                return head;
            } 

            static constexpr auto index(uma_ptr_t ptr) noexcept -> size_t{

                return memult::distance(head, access(ptr));
            }

            static constexpr auto offset_id(size_t idx) noexcept -> size_t{

                return idx;
            }

            static constexpr auto offset_observing_addr(size_t idx) noexcept -> size_t{
                
                return idx * sizeof(observing_value_t) + (offset_id(TILE_COUNT) + PADDING_SZ); 
            }

            static constexpr auto offset_tile_logit_addr(size_t idx) noexcept -> size_t{

                return idx * sizeof(logit_value_t) + (offset_observing_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_tile_grad_addr(size_t idx) noexcept -> size_t{

                return idx * sizeof(grad_value_t)  + (offset_tile_logit_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_bit_control_addr(size_t idx) noexcept -> size_t{

                return idx * sizeof(bit_control_t) + (offset_tile_grad_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_dispatch_control_addr(size_t idx) noexcept -> size_t{

                return idx * sizeof(dispatch_control_t) + (offset_bit_control_addr(TILE_COUNT) + PADDING_SZ);
            } 

            static constexpr auto offset_pong_count_addr(size_t idx) noexcept -> size_t{

                return idx * sizeof(pong_count_t) + (offset_dispatch_control_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static_assert(sizeof(identity_t) == sizeof(char));

            static void init(uma_ptr_t buf) noexcept{

                head = buf;
                dg::network_uma::memset_synchronous_alldevice_bypass_qualifier(buf, ID, TILE_COUNT);
                PtrAccess<ID>::init(buf, TILE_COUNT);
            }

            static constexpr auto size() noexcept -> size_t{

                return offset_pong_count_addr(TILE_COUNT);
            } 

            static constexpr auto id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{
                
                return access(ptr);
            }

            static constexpr auto observing_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_observing_addr(index(ptr)));
            }

            static constexpr auto tile_logit_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_tile_logit_addr(index(ptr)));
            }

            static constexpr auto tile_grad_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_tile_grad_addr(index(ptr)));
            }

            static constexpr auto bit_control_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_bit_control_addr(index(ptr)));
            }

            static constexpr auto dispatch_control_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_dispatch_control_addr(index(ptr)));
            }

            static constexpr auto pong_count_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_pong_count_addr(index(ptr)));
            }

            static constexpr auto notification_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return tile_logit_addr(ptr);
            }

            static constexpr auto rcu_lock_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return tile_logit_addr(ptr);
            }
    };

    template <size_t TILE_COUNT, size_t PADDING_SZ, class identity_t, class logit_value_t, class grad_value_t, class observing_value_t, class bit_control_t, class dispatch_control_t, class pong_count_t, class grad_acm_id_t, class grad_ver_id_t, class old_addr_t, uint8_t ID>
    struct MonoAddressLookup: LeafAddressLookup<TILE_COUNT, PADDING_SZ, identity_t, logit_value_t, grad_value_t, observing_value_t, bit_control_t, dispatch_control_t, pong_count_t, ID>{

        private:

            using base = LeafAddressLookup<TILE_COUNT, PADDING_SZ, identity_t, logit_value_t, grad_value_t, observing_value_t, bit_control_t, ID>;

            static constexpr auto offset_old_addr(size_t idx) noexcept -> size_t{

                return idx * sizeof(old_addr_t) + (base::size() + PADDING_SZ);
            }

            static constexpr auto offset_grad_acm_id_addr(size_t idx) noexcept -> size_t{

                return idx * sizeof(grad_acm_id_t) + (offset_old_addr(TILE_COUNT) + PADDING_SZ);
            }

            static constexpr auto offset_child_grad_ver_id_addr(size_t idx) noexcept -> size_t{

                return idx * sizeof(grad_ver_id_t) + (offset_grad_acm_id_addr(TILE_COUNT) + PADDING_SZ);
            }

        public:

            static constexpr auto size() noexcept -> size_t{

                return offset_child_grad_ver_id_addr(TILE_COUNT);
            }

            static constexpr auto old_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_old_addr(index(ptr)));
            }

            static constexpr auto grad_acm_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_grad_acm_id_addr(index(ptr)));
            } 

            static constexpr auto child_grad_ver_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_child_grad_ver_id_addr(index(ptr)));
            }
    };

    template <size_t TILE_COUNT, size_t PADDING_SZ, class identity_t, class logit_value_t, class grad_value_t, class observing_value_t, class bit_control_t, class dispatch_control_t, class pong_count_t, class grad_acm_id_t, class grad_ver_id_t, class child_addr_t, size_t ACM_SZ, uint8_t ID>
    struct UACMAddressLookup: LeafAddressLookup<TILE_COUNT, PADDING_SZ, identity_t, logit_value_t, grad_value_t, observing_value_t, bit_control_t, dispatch_control_t, pong_count_t, ID>{

        private:

            using base = LeafAddressLookup<TILE_COUNT, PADDING_SZ, identity_t, logit_value_t, grad_value_t, observing_value_t, bit_control_t, dispatch_control_t, pong_count_t, ID>;

            static constexpr auto offset_grad_acm_id_addr(size_t idx) noexcept -> size_t{

                return idx * sizeof(grad_acm_id_t) + (base::size() + PADDING_SZ);
            }

            template <size_t ACM_IDX>
            static constexpr auto offset_child_grad_ver_id_addr(size_t idx, const std::integral_constant<size_t, ACM_IDX>) noexcept -> size_t{

                static_assert(ACM_IDX < ACM_SZ);
                return idx * (sizeof(grad_ver_id_t) * ACM_SZ) + (offset_grad_acm_id_addr(TILE_COUNT) + PADDING_SZ + sizeof(grad_ver_id_t) * ACM_IDX);
            }

            template <size_t ACM_IDX>
            static constexpr auto offset_child_addr(size_t idx, const std::integral_constant<size_t, ACM_IDX>) noexcept -> size_t{
                
                static_assert(ACM_IDX < ACM_SZ);
                return idx * (sizeof(child_addr_t) * ACM_SZ) + (offset_child_grad_ver_id_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ + sizeof(child_addr_t) * ACM_IDX);
            }

        public:

            static_assert(ACM_SZ != 0u);

            static constexpr auto size() noexcept -> size_t{

                return offset_child_addr(TILE_COUNT, std::integral_constant<size_t, 0>{});
            }

            static constexpr auto grad_acm_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_grad_acm_id_addr(index(ptr)));
            }

            template <size_t ACM_IDX>
            static constexpr auto child_grad_ver_id_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_child_grad_ver_id_addr(index(ptr), std::integral_constant<size_t, ACM_IDX>{}));
            } 

            template <size_t ACM_IDX>
            static constexpr auto child_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_child_addr(index(ptr), std::integral_constant<size_t, ACM_IDX>{}));
            }
    };

    template <size_t TILE_COUNT, size_t PADDING_SZ, class identity_t, class logit_value_t, class grad_value_t, class observing_value_t, class bit_control_t, class dispatch_control_t, class pong_count_t, class grad_acm_id_t, class grad_ver_id_t, class child_addr_t, size_t ACM_SZ, uint8_t ID>
    struct PACMAddressLookup: UACMAddressLookup<TILE_COUNT, PADDING_SZ, identity_t, logit_value_t, grad_value_t, observing_value_t, bit_control_t, dispatch_control_t, pong_count_t, grad_acm_id_t, grad_ver_id_t, child_addr_t, ACM_SZ, ID>{
        
        private:

            using base = UACMAddressLookup<TILE_COUNT, PADDING_SZ, identity_t, logit_value_t, grad_value_t, observing_value_t, bit_control_t, dispatch_control_t, pong_count_t, grad_acm_id_t, grad_ver_id_t, child_addr_t, ACM_SZ, ID>;

            template <size_t ACM_IDX>
            static constexpr auto offset_lhs_child_addr(size_t idx, const std::integral_constant<size_t, ACM_IDX>) noexcept -> size_t{

                static_assert(ACM_IDX < ACM_SZ);
                return idx * (sizeof(child_addr_t) * ACM_SZ) + (base::size() + PADDING_SZ + sizeof(child_addr_t) * ACM_IDX);
            }

            template <size_t ACM_IDX>
            static constexpr auto offset_lhs_grad_ver_id_addr(size_t idx, const std::itnegral_constant<size_t, ACM_IDX>) noexcept -> size_t{

                static_assert(ACM_IDX < ACM_SZ);
                return idx * (sizeof(grad_ver_id_t) * ACM_SZ) + (offset_lhs_child_addr(TILE_COUNT, std::integral_constant<size_t, 0>{}) + PADDING_SZ + sizeof(grad_ver_id_t) * ACM_IDX);
            }

        public:

            static_assert(ACM_SZ != 0);

            static constexpr auto size() noexcept -> size_t{

                return offset_lhs_grad_ver_id_addr(TILE_COUNT, std::integral_constant<size_t, 0>{});
            }

            template <size_t ACM_IDX>
            static constexpr auto lhs_child_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_lhs_child_addr(index(ptr), std::integral_constant<size_t, ACM_IDX>{}));
            }

            template <size_t ACM_IDX>
            static constexpr auto lhs_grad_ver_id_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_lhs_grad_ver_id_addr(index(ptr), std::integral_constant<size_t, ACM_IDX>{}));
            }

            template <size_t ACM_IDX>   
            static constexpr auto rhs_child_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) noexcept -> uma_ptr_t{

                return child_addr(ptr, std::integral_constant<size_t, ACM_IDX>{})
            }

            template <size_t ACM_IDX>   
            static constexpr auto rhs_grad_ver_id_addr(uma_ptr_t ptr, const std::integral_constant<size_t, ACM_IDX>) noexcept -> uma_ptr_t{

                return grad_ver_id_addr(ptr, std::integral_constant<size_t, ACM_IDX>{});
            }
    };

    template <size_t TILE_COUNT, size_t PADDING_SZ, class identity_t, class logit_value_t, class grad_value_t, class observing_value_t, class bit_control_t, class dispatch_control_t, class pong_count_t, class grad_acm_id_t, class grad_ver_id_t, class child_addr_t, uint8_t ID>
    struct PairAddressLookup: MonoAddressLookup<TILE_COUNT, PADDING_SZ, identity_t, logit_value_t, grad_value_t, observing_value_t, bit_control_t, dispatch_control_t, pong_count_t, grad_acm_id_t, grad_ver_id_t, child_addr_t, ID>{

        private:
            
            using base = MonoAddressLookup<TILE_COUNT, PADDING_SZ, identity_t, logit_value_t, grad_value_t, observing_value_t, bit_control_t, dispatch_control_t, pong_count_t, grad_acm_id_t, grad_ver_id_t, child_addr_t, ID>;

            static constexpr auto offset_rhs_child_addr(size_t idx) noexcept -> size_t{

                return idx * sizeof(child_addr_t) + (base::size() + PADDING_SZ);
            }

            static constexpr auto offset_rhs_grad_ver_id_addr(size_t idx) noexcept -> grad_ver_id_t{

                return idx * sizeof(grad_ver_id_t) + (offset_rhs_child_addr(TILE_COUNT) + PADDING_SZ);
            } 

        public:

            static constexpr auto size() noexcept -> size_t{

                return offset_rhs_grad_ver_id_addr(TILE_COUNT);
            }

            static constexpr auto lhs_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{
                
                return old_addr(ptr);
            }

            static constexpr auto lhs_grad_ver_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return child_grad_ver_id_addr(ptr);
            }

            static constexpr auto rhs_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_rhs_child_addr(index(ptr)));
            }

            static constexpr auto rhs_grad_ver_id_addr(uma_ptr_t ptr) noexcept -> uma_ptr_t{

                return memult::advance(get_head(), offset_rhs_grad_ver_id_addr(index(ptr)));
            }
    };

    static inline constexpr size_t MAXIMUM_TILE_LOG2    = 20;  
    
    template <class T>
    struct type_reduction{
        using type = T;
    };

    template <class IntegralType, IntegralType IntegralValue>
    struct type_reduction<std::integral_constant<IntegralType, IntegralValue>>{
        static inline constexpr IntegralType type = IntegralValue;
    };

    template <class ...Args>
    static auto leaf_lookup_type(tags<Args...>){

        auto lambda = []<class Self, size_t IDX>(Self self, std::integral_constant<size_t, IDX>){
            if constexpr(IDX != 0){
                constexpr size_t TILE_COUNT = memult::pow2(IDX);
                using candidate_t           = LeafAddressLookup<type_reduction<Args>::type...>;
                if constexpr(candidate_t::size() <= BUF_SZ){
                    return candidate_t{};
                } else{
                    return self(self, std::integral_constant<size_t, IDX - 1>{});
                }
            } else{
                return std::add_pointer_t<void>{};
            }
        };

        return lambda(lambda, std::integral_constant<size_t, MAXIMUM_TILE_LOG2>{});
    }

    template <class ...Args>
    static auto mono_lookup_type(tags<Args...>){

        auto lambda = []<class Self, size_t IDX>(Self self, std::integral_constant<size_t, IDX>){
            if constexpr(IDX != 0){
                constexpr size_t TILE_COUNT = memult::pow2(IDX);
                using candidate_t           = MonoAddressLookup<type_reduction<Args>::type...>;
                if constexpr(candidate_t::size() <= BUF_SZ){
                    return candidate_t{};
                } else{
                    return self(self, std::integral_constant<size_t, IDX - 1>{});
                }
            } else{
                return std::add_pointer_t<void>{};
            }
        };

        return lambda(lambda, std::integral_constant<size_t, MAXIMUM_TILE_LOG2>{});
    }

    template <class ...Args>
    static auto uacm_lookup_type(tags<Args...>){

        auto lambda = []<class Self, size_t IDX>(Self self, std::integral_constant<size_t, IDX>){
            if constexpr(IDX != 0){
                constexpr size_t TILE_COUNT = memult::pow2(IDX);
                using candidate_t           = UACMAddressLookup<type_reduction<Args>::type...>;
                if constexpr(candidate_t::size() <= BUF_SZ){
                    return candidate_t{};
                } else{
                    return self(self, std::integral_constant<size_t, IDX - 1>{});
                }
            } else{
                return std::add_pointer_t<void>{};
            }
        };

        return lambda(lambda, std::integral_constant<size_t, MAXIMUM_TILE_LOG2>{});
    }

    template <class ...Args>
    static auto pacm_lookup_type(tags<Args...>){

        auto lambda = []<class Self, size_t IDX>(Self self, std::integral_constant<size_t, IDX>){
            if constexpr(IDX != 0){
                constexpr size_t TILE_COUNT = memult::pow2(IDX);
                using candidate_t           = PACMAddressLookup<type_reduction<Args>::type...>;
                if constexpr(candidate_t::size() <= BUF_SZ){
                    return candidate_t{};
                } else{
                    return self(self, std::integral_constant<size_t, IDX - 1>{});
                }
            } else{
                return std::add_pointer_t<void>{};
            }
        };

        return lambda(lambda, std::integral_constant<size_t, MAXIMUM_TILE_LOG2>{});
    }

    template <class ...Args>
    static auto pair_lookup_type(tags<Args...>){

        auto lambda = []<class Self, size_t IDX>(Self self, std::integral_constant<size_t, IDX>){
            if constexpr(IDX != 0){
                constexpr size_t TILE_COUNT = memult::pow2(IDX);
                using candidate_t           = PairAddressLookup<type_reduction<Args>::type...>;
                if constexpr(candidate_t::size() <= BUF_SZ){
                    return candidate_t{};
                } else{
                    return self(self, std::integral_constant<size_t, IDX - 1>{});
                }
            } else{
                return std::add_pointer_t<void>{};
            }
        };

        return lambda(lambda, std::integral_constant<size_t, MAXIMUM_TILE_LOG2>{});
    }

    template <class ...Args>
    static auto crit_lookup_type(tags<Args...>){

    }

    template <class ...Args>
    static auto msgr_lookup_type(tags<Args...>){

    } 
}

namespace dg::network_tile_member_access{

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
    using observing_value_t         = std::array<char, 256>; //each stable ptr have maximum 256-byte observable registration (backward reference)
    using bit_control_t             = uint64_t;
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

    auto is_leaf_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_leaf_8) || (id == id_leaf_16) || (id == id_leaf_32);
    }

    auto is_mono_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_mono_8) || (id == id_mono_16) || (id == id_mono_32);
    }

    auto is_pair_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_pair_8) || (id == id_pair_16) || (id == id_pair_32);
    }

    auto is_uacm_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_uacm_8) || (id == id_uacm_16) || (id == id_uacm_32);
    } 

    auto is_pacm_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_pacm_8) || (id == id_pacm_16) || (id == id_pacm_32);
    }

    auto is_crit_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_crit_8) || (id == id_crit_16) || (id == id_crit_32);
    }

    auto is_msgrfwd_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_msgrfwd_8) || (id == id_msgrfwd_16) || (id == id_msgrfwd_32);
    }

    auto is_msgrbwd_tile(tile_polymorphic_t id) noexcept -> bool{

        return (id == id_msgrbwd_8) || (id == id_msgrbwd_16) || (id == id_msgrbwd_32);
    }

    auto is_leaf8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_leaf_8;
    }

    auto is_mono8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_mono_8;
    }

    auto is_pair8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pair_8;
    }

    auto is_uacm8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_uacm_8;
    }

    auto is_pacm8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pacm_8;
    }

    auto is_crit8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_crit_8;
    }

    auto is_msgrfwd8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrfwd_8;
    }

    auto is_msgrbwd8_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrbwd_8;
    } 

    auto is_leaf16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_leaf_16;
    }

    auto is_mono16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_mono_16;
    }

    auto is_pair16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pair_16;
    }

    auto is_uacm16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_uacm_16;
    }

    auto is_pacm16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pacm_16;
    }

    auto is_crit16_tile(tile_polymorphic_t id) noexcept -> bool{
        
        return id == id_crit_16;
    }

    auto is_msgrfwd16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrfwd_16;
    }

    auto is_msgrbwd16_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrbwd_16;
    }

    auto is_leaf32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_leaf_32;
    }

    auto is_mono32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_mono_32;
    }

    auto is_pair32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pair_32;
    }

    auto is_uacm32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_uacm_32;
    }

    auto is_pacm32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_pacm_32;
    }

    auto is_crit32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_crit_32;
    }

    auto is_msgrfwd32_tile(tile_polymorphic_t id) noexcept -> bool{

        return id == id_msgrfwd_32;
    }

    auto is_msgrbwd32_tile(tile_polymorphic_t id) noexcept -> bool{
        
        return id == id_msgrbwd_32;
    }

    auto dg_typeid(uma_ptr_t ptr) noexcept -> tile_polymorphic_t{

        tile_polymorphic_t id{};
        void * dst      = &id;
        uma_ptr_t src   = ptr;
        dg::network_memops_umax::memcpy_uma_to_host_nothrow(dst, src, sizeof(tile_polymorphic_t));

        return id;
    } 

    template <class CallBack>
    void get_leaf_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

        tile_polymorphic_t id = dg_typeid(ptr);

        if (is_leaf8_tile(id)){
            cb(leaf8_accessor_t{});
            return;
        }

        if (is_leaf16_tile(id)){
            cb(leaf16_accessor_t{});
            return;
        }

        if (is_leaf32_tile(id)){
            cb(leaf32_accessor_t{});
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    template <class CallBack>
    void get_mono_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

        tile_polymorphic_t id = dg_typeid(ptr);

        if (is_mono8_tile(id)){
            cb(mono8_accessor_t{});
            return;
        }

        if (is_mono16_tile(id)){
            cb(mono16_accessor_t{});
            return;
        }

        if (is_mono32_tile(id)){
            cb(mono32_accessor_t{});
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    template <class CallBack>
    void get_pair_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

        tile_polymorphic_t id = dg_typeid(ptr);

        if (is_pair8_tile(id)){
            cb(pair8_accessor_t{});
            return;
        }

        if (is_pair16_tile(id)){
            cb(pair16_accessor_t{});
            return;
        }

        if (is_pair32_tile(id)){
            cb(pair32_accessor_t{});
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    template <class CallBack>
    void get_uacm_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

        tile_polymorphic_t id = dg_typeid(ptr);

        if (is_uacm8_tile(id)){
            cb(uacm8_accesor_t{});
            return;
        }

        if (is_uacm16_tile(id)){
            cb(uacm16_accesor_t{});
            return;
        }

        if (is_uacm_32_tile(id)){
            cb(uacm32_accesor_t{});
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    template <class CallBack>
    void get_pacm_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

        tile_polymorphic_t id = dg_typeid(ptr);

        if (is_pacm8_tile(id)){
            cb(pacm8_accessor_t{});
            return;
        }

        if (is_pacm16_tile(id)){
            cb(pacm16_accessor_t{});
            return;
        }

        if (is_pacm32_tile(id)){
            cb(pacm32_accessor_t{});
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    template <class CallBack>
    void get_crit_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

        tile_polymorphic_t id = dg_typeid(ptr);

        if (is_crit8_tile(id)){
            cb(crit8_accessor_t{});
            return;
        }

        if (is_crit16_tile(id)){
            cb(crit16_accessor_t{});
            return;
        }

        if (is_crit32_tile(id)){
            cb(crit32_accessor_t{});
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    template <class CallBack>
    void get_msgrfwd_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

        tile_polymorphic_t id = dg_typeid(ptr);

        if (is_msgrfwd8_tile(id)){
            cb(msgrfwd8_accessor_t{});
            return;
        }

        if (is_msgrfwd16_tile(id)){
            cb(msgrfwd16_accessor_t{});
            return;
        }

        if (is_msgrfwd32_tile(id)){
            cb(msgrfwd32_accessor_t{});
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    template <class CallBack>
    void get_msgrbwd_static_polymorphic_accessor(CallBack cb, uma_ptr_t ptr) noexcept{

        tile_polymorphic_t id = dg_typeid(ptr);

        if (is_msgrbwd8_tile(id)){
            cb(msgrbwd8_accessor_t{});
            return;
        }

        if (is_msgrbwd16_tile(id)){
            cb(msgrbwd16_accessor_t{});
            return;
        }

        if (is_msgrbwd32_tile(id)){
            cb(msgrbwd32_accessor_t{});
            return;
        }

        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
        std::abort();
    }

    auto safe_leaf_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        return safe_leaf_ptr_access_instance::access(ptr);
    }

    auto safe_mono_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        return safe_mono_ptr_access_instance::access(ptr);
    }

    auto safe_pair_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        return safe_pair_ptr_access_instance::access(ptr);
    }

    auto safe_uacm_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        return safe_uacm_ptr_access_instance::access(ptr);
    }

    auto safe_pacm_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        return safe_pacm_ptr_access_instance::access(ptr);
    }

    auto safe_crit_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        return safe_crit_ptr_access_instance::access(ptr);
    }

    auto safe_msgrfwd_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        return safe_msgrfwd_ptr_access_instance::access(ptr);
    }

    auto safe_msgrbwd_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        return safe_msgrbwd_ptr_access_instance::access(ptr);
    }

    auto safe_tile_ptr_access(uma_ptr_t ptr) noexcept -> std::expected<uma_ptr_t, exception_t>{

        return safe_tile_ptr_access_instance::access(ptr);
    }

    auto throwsafe_leaf_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safe_leaf_ptr_access(ptr));
    }

    auto throwsafe_mono_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safe_mono_ptr_access(ptr));
    }

    auto throwsafe_pair_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safe_pair_ptr_access(ptr));
    }

    auto throwsafe_uacm_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safe_uacm_ptr_access(ptr));
    }
    
    auto throwsafe_pacm_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safe_pacm_ptr_access(ptr));
    }

    auto throwsafe_crit_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safe_crit_ptr_access(ptr));
    }

    auto throwsafe_msgrfwd_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safe_msgrfwd_ptr_access(ptr));
    }

    auto throwsafe_msgrbwd_ptr_access(uma_ptr_t ptr) -> uma_ptr_t{

        return dg::network_exception_handler::throw_log(safe_msgrbwd_ptr_access(ptr));
    }
}