#ifndef __NETWORK_EVENT_COLLECTOR_H__
#define __NETWORK_EVENT_COLLECTOR_H__

#include <stddef.h>
#include <stdint.h>
#include <chrono>
#include "network_function_concurrent_buffer.h"
#include <ratio>
#include "network_producer_consumer.h"

namespace dg::network_event_collector{

    using epoch_milli_t                 = size_t;  
    using event_loop_register_t         = void (*)(void (*)(void) noexcept); 
    
    template <class ...Args>
    struct tags{}; 

    template <class ID, class Frequency, class Producer, class Consumer, class CollectCapacity>
    class ScanCollector{}; 

    template <class ID, size_t FREQUENCY, class T, class T1, size_t COLLECT_CAPACITY>
    class ScanCollector<ID, std::integral_constant<size_t, FREQUENCY>, dg::network_producer_consumer::DistributedProducerInterface<T>, dg::network_producer_consumer::ConsumerInterface<T1>, std::integral_constant<size_t, COLLECT_CAPACITY>>{

        private:

            using self              = ScanCollector; 
            using producer          = dg::network_producer_consumer::DistributedProducerInterface<T>;  
            using consumer          = dg::network_producer_consumer::ConsumerInterface<T1>;
            using event_t           = typename producer::event_t; 

            static inline epoch_milli_t last_collected_time{};

            static consteval auto delta_submit_time_in_millisecond() noexcept -> size_t{

                return double{size_t{1} << 20} / FREQUENCY;
            } 
            
        public:

            static_assert(FREQUENCY != 0u);
            static_assert(std::is_same_v<typename producer::event_t, typename consumer::event_t>);
            static_assert(COLLECT_CAPACITY > 0);

            static void run() noexcept{

                using namespace std::chrono;

                struct signature_collect{};

                using functor_signature = tags<self, signature_collect>; 
                using event_array_tag   = dg::network_function_concurrent_local_array::Tag<functor_signature, event_t, COLLECT_CAPACITY>;

                epoch_milli_t cur       = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
                intmax_t delta          = static_cast<intmax_t>(cur) - static_cast<intmax_t>(last_collected_time);
                event_t * events        = dg::network_function_concurrent_local_array::get_array(event_array_tag{});
                size_t sz               = {};

                if (delta < delta_submit_time_in_millisecond()){
                    return;
                }

                for (size_t i = 0; i < producer::range(); ++i){
                    producer::get(i, events, sz, COLLECT_CAPACITY);
                    consumer::push(events, sz);
                }

                last_collected_time = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
            }

            static void init(event_loop_register_t event_loop_register) noexcept{

                using namespace std::chrono;
                last_collected_time = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
                event_loop_register(run);
            }
    };

    template <class ID, class Frequency, class HalfLifeRatio, class Producer, class Consumer, class CollectCapacity>
    class HalfLifeCollector{};

    template <class ID, size_t FREQUENCY, intmax_t HALF_LIFE_DEN, intmax_t HALF_LIFE_NUM, class T, class T1, size_t COLLECT_CAPACITY>
    class HalfLifeCollector<ID, std::integral_constant<size_t, FREQUENCY>, std::ratio<HALF_LIFE_DEN, HALF_LIFE_NUM>, dg::network_producer_consumer::DistributedProducerInterface<T>, dg::network_producer_consumer::ConsumerInterface<T1>, std::integral_constant<size_t, COLLECT_CAPACITY>>{

        private:

            using self              = HalfLifeCollector; 
            using producer          = dg::network_producer_consumer::DistributedProducerInterface<T>;  
            using consumer          = dg::network_producer_consumer::ConsumerInterface<T1>;
            using event_t           = typename producer::event_t; 

            static inline epoch_milli_t last_collected_time{};

            static consteval auto half_life() noexcept -> double{

                return double{HALF_LIFE_DEN} / double{HALF_LIFE_NUM};
            } 

            static consteval auto delta_submit_time_in_millisecond() noexcept -> size_t{

                return double{size_t{1} << 20} / FREQUENCY;
            } 

        public:

            static_assert(FREQUENCY != 0u);
            static_assert(std::is_same_v<typename producer::event_t, typename consumer::event_t>);
            static_assert(COLLECT_CAPACITY > 0);

            static void run() noexcept{

                using namespace std::chrono;

                struct signature_collect{};

                using functor_signature = tags<self, signature_collect>; 
                using event_array_tag   = dg::network_function_concurrent_local_array::Tag<functor_signature, event_t, COLLECT_CAPACITY>;

                epoch_milli_t cur       = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
                intmax_t delta          = static_cast<intmax_t>(cur) - static_cast<intmax_t>(last_collected_time);
                event_t * events        = dg::network_function_concurrent_local_array::get_array(event_array_tag{});
                size_t sz               = {};
                size_t nxt_chk_pt       = producer::range() * half_life();

                if (delta < delta_submit_time_in_millisecond()){
                    return;
                }

                for (size_t i = 0; i < producer::range(); ++i){
                    if (i > nxt_chk_pt){
                        if (dg::network_randomizer::randomize_bool()){
                            return;
                        }
                        nxt_chk_pt += (producer::range() - i) * half_life();
                    }

                    producer::collect(i, events, sz, COLLECT_CAPACITY);
                    consumer::push(events, sz);
                }

                last_collected_time = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
            }

            static void init(event_loop_register_t event_loop_register) noexcept{

                using namespace std::chrono;
                last_collected_time = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
                event_loop_register(run);
            }
    };
}

#endif 