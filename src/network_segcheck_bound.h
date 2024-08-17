#ifndef __NETWORK_SEGCHECK_BOUND_H__
#define __NETWORK_SEGCHECK_BOUND_H__

#include "network_log.h" 

namespace dg::network_segcheck_bound{

    template <class T>
    struct SafeAccessInterface{

        using interface_t   = SafeAccessInterface<T>; 
        using ptr_t         = typename T::ptr_t;
        
        static_assert(dg::is_ptr_v<ptr_t>);

        static inline auto access(ptr_t ptr) noexcept -> ptr_t{

            return T::access(ptr);
        }
    };

    template <class ID, class PtrT>
    struct UnSafeAccess: SafeAccessInterface<UnSafeAccess<ID, PtrT>>{

        using ptr_t = PtrT;

        static void init(ptr_t, ptr_t) noexcept{} 

        static inline auto access(ptr_t ptr) noexcept -> ptr_t{

            return ptr;
        }
    };

    template <class ID, class PtrT>
    struct SafeAlignedAccess: SafeAccessInterface<SafeAlignedAccess<ID, PtrT>>{

        public:

            using ptr_t = PtrT;

        private:

            static inline ptr_t first{};
            static inline ptr_t last{};

        public:

            static void init(ptr_t arg_first, ptr_t arg_last) noexcept{

                first = arg_first;
                last  = arg_last;
            }

            static inline auto access(ptr_t ptr) noexcept -> ptr_t{
                
                if (memult::ptrcmp_aligned(ptr, first) < 0 || memult::ptrcmp_aligned(ptr, last) >= 0){
                    dg::network_log_stackdump::critical_error(dg::network_exception::CORE_SEGFAULT_CSTR);
                    std::abort();
                }

                return ptr;
            }
    }; 

    static inline constexpr bool IS_SAFE_ACCESS_ENABLED = true;

    template <class ID, class ptr_t>
    using StdAccess = std::conditional_t<IS_SAFE_ACCESS_ENABLED, 
                                         SafeAlignedAccess<ID, ptr_t>,
                                         UnSafeAccess<IO, ptr_t>>;

}

#endif