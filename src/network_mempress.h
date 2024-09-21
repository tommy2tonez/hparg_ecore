#ifndef __DG_NETWORK_MEMPRESS_H__
#define __DG_NETWORK_MEMPRESS_H__

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <chrono>
#include <array>
#include <memory>
#include <cstring>
#include "assert.h"
#include <type_traits>
#include "network_memlock.h"
#include "network_fundamental_vector.h"
#include "network_producer_consumer.h"
#include "network_exception.h"
#include "network_log.h" 
#include "network_segcheck_bound.h"

namespace dg::network_mempress{

    //I admit bare metal looks strange - I will reconsider this     

    template <class T>
    struct Notifiable{

        using ptr_t = typename T::ptr_t;

        static_assert(dg::is_ptr_v<ptr_t>);

        static void notify(ptr_t region) noexcept{
            
            T::notify(region);
        }
    };

    template <class T>
    struct MemPressInterface{

        using event_t   = typename T::event_t; 
        using ptr_t     = typename T::ptr_t;

        static_assert(dg::is_ptr_v<ptr_t>); 

        static auto first() noexcept -> ptr_t{

            return T::first();
        }

        static auto last() noexcept -> ptr_t{

            return T::last(); 
        }

        static auto capacity() noexcept -> size_t{

            return T::capacity();
        }

        static auto memregion_size() noexcept -> size_t{

            return T::memregion_size();
        }

        static void push(ptr_t ptr, event_t * event, size_t event_sz) noexcept{

            T::push(ptr, event, event_sz);
        } 

        static void collect(ptr_t region, event_t * dst, size_t& event_count, size_t event_cap) noexcept{

            return T::collect(region, dst, event_count, event_cap);
        } 
    };

    template <class ID, class PtrType, class MemRegionSize, class Event, class EventCountPerRegion>
    class EventContainer{}; 

    template <class ID, class PtrType, size_t MEMREGION_SZ, class EventType, size_t EVENT_COUNT_PER_REGION>
    class EventContainer<ID, PtrType, std::integral_constant<size_t, MEMREGION_SZ>, EventType, std::integral_constant<size_t, EVENT_COUNT_PER_REGION>>{

        public:

            using event_t           = EventType;
            using ptr_t             = PtrType;

        private:

            using ptr_arithmetic_t  = typename dg::ptr_info<ptr_t>::max_unsigned_t;
            using memlock           = dg::network_memlock::Lock<ID, std::integral_constant<size_t, MEMREGION_SZ>, ptr_t>;
            using container_t       = dg::network_fundamental_vector::fixed_fundamental_vector<event_t, EVENT_COUNT_PER_REGION>;

            static inline container_t * events{};
            static inline ptr_t first_region{};
            static inline ptr_t last_region{}; 

            static auto memregion_id(ptr_t ptr) noexcept -> size_t{

                return pointer_cast<ptr_arithmetic_t>(ptr) / static_cast<ptr_arithmetic_t>(MEMREGION_SZ);
            } 

        public:

            static_assert(dg::memult::is_pow2(MEMREGION_SZ));
            static_assert(EVENT_COUNT_PER_REGION != 0);

            static void init(ptr_t arg_first, ptr_t arg_last){

                auto logger = dg::network_log_scope::critical_terminate(); 

                if (pointer_cast<ptr_arithmetic_t>(arg_first) == 0u || pointer_cast<ptr_arithmetic_t>(arg_first) % MEMREGION_SZ != 0u || pointer_cast<ptr_arithmetic_t>(arg_last) == 0u || pointer_cast<ptr_arithmetic_t>(arg_last) % MEMREGION_SZ != 0u){
                    throw dg::network_exception::invalid_arg();
                }

                first_region    = arg_first;
                last_region     = arg_last;
                size_t event_sz = pointer_cast<ptr_arithmetic_t>(last_region) / static_cast<ptr_arithmetic_t>(MEMREGION_SZ);
                events          = new container_t[event_sz];
                memlock::init(); //
                logger.release();
            }

            static auto first() noexcept -> ptr_t{

                return first_region;
            }

            static auto last() noexcept -> ptr_t{
                
                return last_region;
            }
    
            static auto capacity() noexcept -> size_t{

                return EVENT_COUNT_PER_REGION;
            }

            static auto memregion_size() noexcept -> size_t{

                return MEMREGION_SZ;
            } 

            static auto try_push(ptr_t ptr, event_t * event, size_t event_sz) noexcept -> bool{

                ptr             = safe_access_instance::access(ptr);
                auto lck_grd    = dg::network_memlock_utility::lock_guard(memlock{}, ptr); //
                size_t id       = memregion_id(ptr); 

                if (events[id].size() + event_sz <= events[id].capacity()){
                    evemts[id].push_back(event, event + event_sz);
                    return true;
                }

                return false;
            } 

            static void collect(ptr_t region, event_t * dst, size_t& dst_count, size_t dst_cap) noexcept{

                region                  = safe_access_instance::access(region);
                auto lck_grd            = dg::network_memlock_utility::lock_guard(memlock{}, region); //
                size_t id               = memregion_id(region);
                size_t cur_sz           = events[id].size();
                size_t peek_sz          = std::min(dst_cap, cur_sz);
                size_t rem_sz           = cur_sz - peek_sz;
                const event_t * src     = dg::memult::advance(events[id].data(), rem_sz);
                dst_count               = peek_sz;

                std::memcpy(dst, src, peek_sz * sizeof(event_t));
                events[id].resize(rem_sz);
            }
    };

    template <class ID, class Container, class Notifier>
    class MemPress{};

    template <class ID, class T, class T1>
    class MemPress<ID, ContainerInterface<T>, Notifiable<T1>>: public MemPressInterface<MemPress<ID, ContainerInterface<T>, Notifiable<T1>>>{

        private:

            using base      = ContainerInterface<T>;
            using notifier  = Notifiable<T1>;

        public:
            
            using ptr_t     = typename base::ptr_t;
            using event_t   = typename base::event_t;

            static auto first() noexcept -> const void *{

                return base::first();
            }

            static auto last() noexcept -> const void *{

                return base::last();
            }

            static auto capacity() noexcept -> size_t{

                return base::capacity();
            }

            static auto memregion_size() noexcept -> size_t{

                return base::memregion_size();
            } 

            static void push(ptr_t ptr, event_t * event, size_t event_sz) noexcept{

                if (event_sz > capacity()){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }

                while (!base::try_push(ptr, event, event_sz)){
                    notifier::notify(region(ptr));
                }
            } 

            static void collect(ptr_t region, event_t * dst, size_t& event_count, size_t event_cap) noexcept{

                base::collect(region, dst, event_count, event_cap);
            }
    };
}

namespace dg::network_mempress_wrapper{
    
    using namespace dg::network_mempress;

    template <class ID, class T>
    struct RetranslatedProducerWrapper{}; 

    template <class ID, class T>
    struct RetranslatedProducerWrapper<ID, MemPressInterface<T>>: dg::network_producer_consumer::DistributedProducerInterface<RetranslatedProducerWrapper<ID, MemPressInterface<T>>>{

        private:

            using self                  = RetranslatedProducerWrapper;
            using base                  = MemPressInterface<T>; 
            using ptr_t                 = typename base::ptr_t; 
            using idx_ptr_t             = typename dg::ptr_info<>::max_unsigned_t; 
            using safe_access_instance  = dg::network_segcheck_bound::StdAccess<self, idx_ptr_t>;

            static inline ptr_t * translation_table{};

        public:

            using event_t = typename base::event_t;

            static auto range() noexcept -> size_t{

                return dg::memult::distance(base::first(), base::last()) / base::memregion_size();
            }

            static void init(ptr_t * original_memregion, ptr_t * translated_memregion, size_t sz){

                auto logger         = dg::network_log_scope::critical_terminate();
                translation_table   = new ptr_t[range()];

                for (size_t i = 0; i < sz; ++i){
                    size_t table_idx                = dg::memult::distance(base::first(), original_memregion[i]) / base::memregion_size();
                    translation_table[table_idx]    = translated_memregion[i]; 
                }

                safe_access_instance::init(static_cast<idx_ptr_t>(0u), static_cast<idx_ptr_t>(range()));
                logger.release();
            }

            static void get(size_t i, event_t * dst, event_t& dst_sz, size_t dst_cap) noexcept{

                idx_ptr_t idx_ptr   = safe_access_instance::access(static_cast<idx_ptr_t>(i)); 
                ptr_t region        = translation_table[idx_ptr];
                base::collect(region, dst, dst_sz, dst_cap);
            }
    };
};


#endif