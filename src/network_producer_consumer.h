#ifndef __NETWORK_PRODUCER_CONSUMER_H__
#define __NETWORK_PRODUCER_CONSUMER_H__

#include <stdint.h>
#include <stddef.h>
#include <array>
#include "network_function_concurrent_buffer.h"
#include "network_fundamental_vector.h"

namespace dg::network_producer_consumer{

    template <class T>
    struct ProducerInterface{

        using event_t = typename T::event_t;  

        static inline void get(event_t * events, size_t& event_sz, size_t event_cap) noexcept{

            T::get(events, event_sz, event_cap);
        }
    };

    template <class T>
    struct DistributedProducerInterface{
        
        using event_t = typename T::event_t;

        static inline auto range() noexcept -> size_t{

            return T::range();
        } 

        static inline void get(size_t i, event_t * events, event_t& event_sz, size_t event_cap) noexcept{

            T::collect(i, events, event_sz, event_cap);
        }
    };

    template <class T>
    struct ConsumerInterface{

        using event_t = typename T::event_t;

        static inline void push(event_t * events, size_t event_sz) noexcept{

            T::push(events, event_sz);
        }
    };

    template <class T>
    struct LimitConsumerInterface{

        using event_t = typename T::event_t;

        static inline void push(event_t * events, size_t event_sz) noexcept{

            T::push(events, event_sz);
        }
        
        static inline auto capacity() noexcept -> size_t{

            return T::capacity();
        } 
    };

    template <class ID, class T>
    struct ConsumerWrapper{};

    template <class ID, class T>
    struct ConsumerWrapper<ID, LimitConsumerInterface<T>>: ConsumerInterface<ConsumerWrapper<ID, LimitConsumerInterface<T>>>{

        private:

            using base = LimitConsumerInterface<T>;
        
        public:

            using event_t = typename base::event_t; 

            static inline void push(event_t * events, size_t event_sz) noexcept{
                
                event_t * submit_ptr = events;

                while (event_sz != 0){    
                    size_t submit_sz = std::min(event_sz, base::capacity());  
                    base::push(submit_ptr, submit_sz);
                    submit_ptr  += submit_sz;
                    event_sz    -= submit_sz;
                }
            }
    };

    template <class T>
    struct DeliveryHandlerInterface{

        using event_t = typename T::event_t; 

        static auto get_delivery_handler() noexcept -> void *{
            
            return T::get_delivery_handler();
        }

        static void push_event(void * handler, event_t event) noexcept{

            T::push_event(handler, event);
        }

        static void decommission(void * handler) noexcept{
            
            T::decomission(handler);
        }    
    };

    //singleton w.r.t thread
    template <class ID, class MaxDispatchSize, class Consumer>
    class SingletonDeliveryHandler{}; 

    template <class ID, size_t MAX_DISPATCH_SZ, class T>
    class SingletonDeliveryHandler<ID, std::integral_constant<size_t, MAX_DISPATCH_SZ>, ConsumerInterface<T>>: public DeliveryHandlerInterface<SingletonDeliveryHandler<ID, std::integral_constant<size_t, MAX_DISPATCH_SZ>, ConsumerInterface<T>>>{

        public:

            using event_t       = typename consumer::event_t;
        
        private:

            using container_t   = dg::network_fundamental_vector::fixed_fundamental_vector_view<event_t, MAX_DISPATCH_SZ>;
            using consumer      = ConsumerInterface<T>;  
            
        public: 

            static_assert(MAX_DISPATCH_SZ > 0);

            static auto get_delivery_handler() noexcept -> void *{
                
                struct signature_get_delivery_handler{}; 

                using functor_signature = tags<self, signature_get_delivery_handler>; 
                constexpr size_t BUF_SZ = container_t::flat_byte_size(); 
                char * buf              = dg::network_function_concurrent_local_array::get_buf(dg::network_function_concurrent_local_array::tag<functor_signature, char, BUF_SZ>{});
                container_t container   = container_t{buf, dg::network_fundamental_vector::init_tag{}}; 

                return buf;
            }

            static void push_event(void * handler, event_t event) noexcept{

                container_t container(static_cast<char *>(handler));
                
                if (container.size() == container.capacity()){
                    consumer::push(container.data(), container.size()); 
                    container.clear();
                }

                container.push_back(event);
            }

            static void decommission(void * handler) noexcept{
                
                container_t container{static_cast<char *>(handler)};
                consumer::push(container.data(), container.size());
            }

            static auto get_delivery_handler_safe() noexcept{

                void * handler  = get_delivery_handler();
                auto destructor = [](void * ptr) noexcept{
                    decommission(ptr);
                };

                return std::unique_ptr<void, decltype(destructor)>(handler, destructor);
            }
    };
}

#endif