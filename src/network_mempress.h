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

namespace dg::network_mempress{
    
    template <class T>
    struct Notifiable{

        static inline void notify(const void * memregion) noexcept{
            
            T::notify(memregion);
        }
    };

    template <class T>
    struct EventContainerInterface{

        using event_t = T::event_t; 

        static inline auto first() noexcept -> const void *{

            return T::first();
        }

        static inline auto last() noexcept -> const void *{

            return T::last(); 
        }

        static inline auto capacity() noexcept -> size_t{

            return T::capacity();
        }

        static consteval auto memregion_size() noexcept -> size_t{

            return T::memregion_size();
        }

        static inline auto try_push(const void * region, event_t * event, size_t event_sz) noexcept -> bool{

            return T::try_push(region, event, event_sz);
        } 

        static inline void collect(const void * region, event_t * dst, size_t& event_count, size_t event_cap) noexcept{

            return T::collect(region, dst, event_count, event_cap);
        } 
    };

    template <class T>
    struct UnboundedEventContainerInterface{

        using event_t = T::event_t; 

        static inline auto first() noexcept -> const void *{

            return T::first();
        }

        static inline auto last() noexcept -> const void *{

            return T::last(); 
        }

        static inline auto capacity() noexcept -> size_t{

            return T::capacity();
        }

        static consteval auto memregion_size() noexcept -> size_t{

            return T::memregion_size();
        }

        static inline void push(const void * region, event_t * event, size_t event_sz) noexcept{

            T::push(region, event, event_sz);
        } 

        static inline void collect(const void * region, event_t * dst, size_t& event_count, size_t event_cap) noexcept{

            return T::collect(region, dst, event_count, event_cap);
        } 
    };

    template <class ID, class MemRegionSize, class Event, class EventCountPerRegion>
    class EventContainer{}; 

    template <class ID, size_t MEMREGION_SZ, class EventType, size_t EVENT_COUNT_PER_REGION>
    class EventContainer<ID, std::integral_constant<size_t, MEMREGION_SZ>, EventType, std::integral_constant<size_t, EVENT_COUNT_PER_REGION>>: public EventContainerInterface<EventContainer<ID, std::integral_constant<size_t, MEMREGION_SZ>, EventType, std::integral_constant<size_t, EVENT_COUNT_PER_REGION>>>{

        public:

            using event_t = EventType;

        private:

            using memlock       = dg::network_memlock::CollisionlessLock<ID, std::integral_constant<size_t, MEMREGION_SZ>>;
            using container_t   = dg::network_fundamental_vector::fixed_fundamental_vector<event_t, EVENT_COUNT_PER_REGION>;

            static inline container_t * events{};
            static inline const void * first_region{};
            static inline const void * last_region{}; 

            static inline auto memregion_id(const void * region) noexcept -> size_t{

                return reinterpret_cast<uintptr_t>(region) / MEMREGION_SZ;
            } 

        public:

            static_assert(MEMREGION_SZ != 0);
            static_assert((MEMREGION_SZ & (MEMREGION_SZ - 1 )) == 0)
            static_assert(EVENT_COUNT_PER_REGION != 0);

            static void init(const void * buf, size_t buf_sz) noexcept{

                auto log_scope = dg::network_log_scope::critical_error_catch("dg::network_mempress::EventContainer::init(const void *, size_t)"); 

                if (reinterpret_cast<uintptr_t>(buf) == 0u || reinterpret_cast<uintptr_t>(buf) % MEMREGION_SZ != 0u || buf_sz == 0u || buf_sz % MEMREGION_SZ != 0u){
                    throw dg::network_exception::invalid_init();
                }

                first_region    = buf;
                last_region     = reinterpret_cast<const char *>(buf) + buf_sz;
                size_t event_sz = reinterpret_cast<uintptr_t>(last_region) / MEMREGION_SZ;
                events          = new container_t[event_sz];
                log_scope.release();
            }

            static inline auto first() noexcept -> const void *{

                return first_region;
            }

            static inline auto last() noexcept -> const void *{
                
                return last_region;
            }
    
            static consteval auto capacity() noexcept -> size_t{

                return EVENT_COUNT_PER_REGION;
            }

            static consteval auto memregion_size() noexcept -> size_t{

                return MEMREGION_SZ;
            } 

            static inline auto try_push(const void * region, event_t * event, size_t event_sz) noexcept -> bool{

                if (!memlock::acquire_try(region)){
                    return false;
                }

                size_t id   = memregion_id(region); 
                bool rs     = false;
                if (events[id].size() + event_sz <= events[id].capacity()){
                    evemts[id].push_back(event, event + event_sz);
                    rs = true;
                }
                memlock::acquire_release(region);

                return rs;
            } 

            static inline void collect(const void * region, event_t * dst, size_t& event_count, size_t event_cap) noexcept{

                memlock::acquire_wait(region);
                size_t id       = memregion_id(region);
                size_t cur_sz   = events[id].size();
                size_t peek_sz  = std::min(event_cap, cur_sz);
                size_t rem_sz   = cur_sz - peek_sz;
                event_count     = peek_sz;
                std::memcpy(dst, events[id].data() + rem_sz, peek_sz * sizeof(event_t));
                events[id].resize(rem_sz);
                memlock::acquire_release(region);
            }
    };

    template <class ID, class T, class Notifier>
    class LeakyEventContainer{};

    template <class ID, class T, class T1>
    class LeakyEventContainer<ID, EventContainerInterface<T>, Notifiable<T1>>: public UnboundedEventContainerInterface<LeakyEventContainer<ID, event::EventContainerInterface<T>, Notifiable<T1>>>{

        private:

            using base      = event::EventContainerInterface<T>;
            using notifier  = notifier_interface::Notifiable<T1>;

        public:
            
            using event_t   = typename base::event_t;

            static inline auto first() noexcept -> const void *{

                return base::first();
            }

            static inline auto last() noexcept -> const void *{

                return base::last();
            }

            static inline auto capacity() noexcept -> size_t{

                return base::capacity();
            }

            static consteval auto memregion_size() noexcept -> size_t{

                return base::memregion_size();
            } 

            static inline void push(const void * region, event_t * event, size_t event_sz) noexcept{

                while (!base::try_push(region, event, event_sz)){
                    notifier::notify(region);
                }
            } 

            static inline void collect(const void * region, event_t * dst, size_t& event_count, size_t event_cap) noexcept{

                base::collect(region, dst, event_count, event_cap);
            }
    };
}

namespace dg::network_mempress_wrapper{
    
    using namespace dg::network_mempress;

    template <class ID, class T>
    struct RetranslatedProducerWrapper{}; 

    template <class ID, class T>
    struct RetranslatedProducerWrapper<ID, UnboundedEventContainerInterface<T>>: dg::network_producer_consumer::DistributedProducerInterface<RangeCollectible<ID, UnboundedEventContainerInterface<T>>>{

        private:

            static inline const void ** translation_table{};
            using base = UnboundedEventContainerInterface<T>; 

        public:

            using event_t = typename base::event_t;

            static inline auto range() noexcept -> size_t{

                return std::distance(reinterpret_cast<const char *>(base::first()), reinterpret_cast<const char *>(base::last())) / base::memregion_size();
            }

            static void init(const void ** host_memregion, const void ** translated_memregion, size_t sz) noexcept{

                auto log_scope      = dg::network_log_scope::critical_error_catch("dg::network_mempress_wrapper::RetranslatedProducerWrapper::init(const void **, const void **, size_t)");
                translation_table   = new std::add_pointer_t<const void>[range()];

                for (size_t i = 0; i < sz; ++i){
                    size_t table_idx                = std::distance(reinterpret_cast<const char *>(base::first()), reinterpret_cast<const char *>(host_memregion[i])) / base::memregion_size();
                    translation_table[table_idx]    = translated_memregion[i]; 
                }

                log_scope.release();
            }

            static inline void get(size_t i, event_t * dst, event_t& dst_sz, size_t dst_cap) noexcept{

                const void * memregion = translation_table[i];
                base::collect(memregion, dst, dst_sz, dst_cap);
            }
    };

    template <class ID, class T>
    struct ConsumerWrapper{};

    template <class ID, class T>
    struct ConsumerWrapper<ID, UnboundedEventContainerInterface<T>>: dg::network_producer_consumer::LimitConsumerInterface<ConsumerWrapper<ID, UnboundedEventContainerInterface<T>>>{

        private:

            using base = UnboundedEventContainerInterface<T>; 

        public:

            // using event_t = typename base::event_t; 

            static inline void push(event_t * events, size_t event_sz) noexcept{

                // T::push(events, event_sz);
            }

            static consteval capacity() noexcept -> size_t{

                return base::capacity();
            }
    };

};


#endif