#ifndef __NETWORK_EVENT_SUBMITTER_H__
#define __NETWORK_EVENT_SUBMITTER_H__

#include "network_function_concurrent_buffer.h" 
#include "network_producer_consumer.h"

namespace dg::network_event_consumer{

    using daemon_submittable_t  = void (*)(void (*)(void) noexcept); 
    
    template <class ...Args>
    struct tags{}; 

    template <class Producer, class EndUser, class MaxDispatchSize>
    class Consumer{};

    template <class T, class T1, size_t MAX_DISPATCH_SIZE>
    class Consumer<dg::network_producer_consumer::ProducerInterface<T>, dg::network_producer_consumer::ConsumerInterface<T1>, std::integral_constant<size_t, MAX_DISPATCH_SIZE>>{

        private:

            using self          = Consumer;
            using producer      = dg::network_producer_consumer::ProducerInterface<T>;
            using consumer      = dg::network_producer_consumer::ConsumerInterface<T1>;
            using event_t       = typename producer::event_t;

        public:

            static_assert(std::is_same_v<typename producer::event_t, typename consumer::event_t>);
            static_assert(MAX_DISPATCH_SIZE > 0);

            static void run() noexcept{
                
                struct signature_dispatch{};

                using functor_signature = tags<self, signature_dispatch>;
                using wo_array_tag      = network_function_concurrent_local_array::Tag<functor_signature, event_t, MAX_DISPATCH_SIZE>; 

                event_t * dispatchables = dg::network_function_concurrent_local_array::get_array(wo_array_tag{});
                size_t sz               = 0u;
                producer::get(dispatchables, sz, MAX_DISPATCH_SIZE);
                consumer::push(dispatchables, sz);
            }

            static void init(daemon_submittable_t daemon_register) noexcept{
                
                daemon_register(run);
            }
    };
}

#endif