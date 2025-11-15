#ifndef __NETWORK_SEGCHECK_BOUND_H__
#define __NETWORK_SEGCHECK_BOUND_H__

#include "network_log.h" 
#include "network_exception.h"
#include "network_memult.h"

namespace dg::network_segcheck_bound{

    template <class T>
    struct SafeAccessInterface{

        using interface_t   = SafeAccessInterface<T>; 

        template <class T1 = T, std::enable_if_t<dg::is_ptr_v<typename T1::ptr_t>, bool> = true>
        using ptr_t         = typename T1::ptr_t;
        
        template <class T1 = T, std::enable_if_t<std::is_same_v<T, T1>, bool> = true>
        static inline auto access(typename T1::ptr_t ptr) noexcept{

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
                
                if (memult::ptrcmp(ptr, first) < 0 || memult::ptrcmp(ptr, last) >= 0){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::SEGFAULT));
                    std::abort();
                }

                return ptr;
            }
    }; 

    static inline constexpr bool IS_SAFE_ACCESS_ENABLED = true;

    template <class ID, class ptr_t>
    using StdAccess = std::conditional_t<IS_SAFE_ACCESS_ENABLED, 
                                         SafeAlignedAccess<ID, ptr_t>,
                                         UnSafeAccess<ID, ptr_t>>;

}

#endif