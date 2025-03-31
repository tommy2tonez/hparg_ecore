#ifndef __NETWORK_PRODUCER_CONSUMER_H__
#define __NETWORK_PRODUCER_CONSUMER_H__

//define HEADER_CONTROL 6

#include <stdint.h>
#include <stddef.h>
#include <array>
#include <memory>
#include <atomic> 
#include "network_concurrency_x.h"
#include "stdx.h"
#include <utility>
#include <iterator>
#include "network_std_container.h"
#include "network_log.h"
#include "network_exception.h"
#include "assert.h"

namespace dg::network_producer_consumer{

    template <class EventType>
    static inline constexpr bool event_precond_v    = std::is_same_v<EventType, std::decay_t<EventType>>;

    template <class EventType>
    static inline constexpr bool key_precond_v      = std::is_same_v<EventType, std::decay_t<EventType>>;

    template <class EventType>
    struct ProducerInterface{
        using event_t = EventType;
        
        static_assert(event_precond_v<event_t>);

        virtual ~ProducerInterface() noexcept = default;
        virtual void get(event_t * event_arr, size_t& event_arr_sz, size_t event_arr_cap) noexcept = 0;
    };

    template <class EventType>
    struct ConsumerInterface{
        using event_t = EventType;

        static_assert(event_precond_v<event_t>);        

        virtual ~ConsumerInterface() noexcept = default;
        virtual void push(std::move_iterator<event_t *> event_arr, size_t event_arr_sz) noexcept = 0;
    };

    template <class KeyType, class EventType>
    struct KVConsumerInterface{

        using key_t     = KeyType;
        using event_t   = EventType;

        static_assert(key_precond_v<key_t>);
        static_assert(event_precond_v<event_t>);

        virtual ~KVConsumerInterface() noexcept = default;
        virtual void push(const KeyType& key, std::move_iterator<EventType *> event_arr, size_t event_arr_sz) noexcept = 0;
    };

    template <class EventType, class Lambda>
    class LambdaWrappedConsumer: public virtual ConsumerInterface<EventType>{

        private:

            Lambda lambda;

        public:

            static_assert(std::is_nothrow_destructible_v<Lambda>);

            LambdaWrappedConsumer(Lambda lambda) noexcept(std::is_nothrow_move_constructible_v<Lambda>): lambda(std::move(lambda)){}

            void push(std::move_iterator<EventType *> event_arr, size_t event_arr_sz) noexcept(std::is_nothrow_invocable_v<Lambda, std::move_iterator<EventType *>, size_t>){

                this->lambda(event_arr, event_arr_sz);
            }
    };

    constexpr auto is_pow2(size_t val) noexcept -> bool{

        return val != 0u && (val & (val - 1u)) == 0u; 
    }

    inline auto dg_memalign(char * buf, size_t alignment_sz) noexcept -> char *{

        assert(is_pow2(alignment_sz));

        uintptr_t fwd               = alignment_sz - 1u;
        uintptr_t bitmask           = ~fwd;
        uintptr_t arithmetic_buf    = reinterpret_cast<uintptr_t>(buf); 
        uintptr_t aligned_buf       = (arithmetic_buf + fwd) & bitmask;

        return reinterpret_cast<char *>(aligned_buf);
    }

    template <class T, class ...Args>
    inline auto inplace_construct_object(char * buf, Args&& ...args) -> T *{

        char * aligned_buf = dg_memalign(buf, std::integral_constant<size_t, alignof(T)>{});
        return new (aligned_buf) T(std::forward<Args>(args)...);
    }

    template <class T>
    inline auto inplace_construct_array(char * buf, size_t sz) -> T *{

        char * aligned_buf = dg_memalign(buf, std::integral_constant<size_t, alignof(T)>{});
        return new (aligned_buf) T[sz];
    }

    template <class T, class ...Args>
    inline auto inplace_construct(char * buf, Args&& ...args) -> std::remove_extent_t<T> *{

        if constexpr(std::is_array_v<T>){
            return inplace_construct_array<std::remove_extent_t<T>>(buf, std::forward<Args>(args)...);
        } else{
            return inplace_construct_object<T>(buf, std::forward<Args>(args)...);
        }
    }

    template <class T, class ...Args>
    constexpr auto inplace_construct_object_size(Args&& ...) noexcept -> size_t{

        return sizeof(T) + alignof(T);
    }

    template <class T>
    constexpr auto inplace_construct_array_size(size_t sz) noexcept -> size_t{

        return sz * sizeof(T) + alignof(T);
    }

    template <class T, class ...Args>
    constexpr auto inplace_construct_size(Args&& ...args) noexcept -> size_t{

        if constexpr(std::is_array_v<T>){
            return inplace_construct_array_size<std::remove_extent_t<T>>(std::forward<Args>(args)...);
        } else{
            return inplace_construct_object_size<T>(std::forward<Args>(args)...);
        }
    }

    template <class T>
    inline auto inplace_destruct_object(T * obj) noexcept(std::is_nothrow_destructible_v<T>){

        std::destroy_at(obj);
    }

    template <class T>
    inline auto inplace_destruct_array(T * arr, size_t arr_sz) noexcept(std::is_nothrow_destructible_v<T>){

        std::destroy(arr, std::next(arr, arr_sz));
    }

    //we need to be very cautious of every allocation
    //because that's the 1st cause of death + exploitation
    //std does not guarantee sz accuracy nor fragmentation of your allocations 
    //shared_ptr<> is very expensive because this type-erased is not your every type-erased, it costs 1 std::memory_order_release == std::memory_order_seq_cst to destruct  

    template <class EventType>
    struct DeliveryHandle{
        std::shared_ptr<EventType[]> deliverable_arr; //alright we need to type-erase the responsibility, we cant introduce another DeliveryHandle because it would mess up polymorphic dispatch
        size_t deliverable_sz;
        size_t deliverable_cap;
        ConsumerInterface<EventType> * consumer;
    };

    template<class KeyType, class EventType>
    struct KVDeliveryHandle{
        size_t deliverable_sz;
        size_t deliverable_cap;
        KVConsumerInterface<KeyType, EventType> * consumer;
    };

    template <class event_t>
    auto delvrsrv_open_handle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<DeliveryHandle<event_t> *, exception_t>{

        constexpr size_t MIN_DELIVERABLE_CAP = 0u;
        constexpr size_t MAX_DELIVERABLE_CAP = size_t{1} << 30;

        if (consumer == nullptr){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(deliverable_cap, MIN_DELIVERABLE_CAP, MAX_DELIVERABLE_CAP) != deliverable_cap){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        try{
            auto container              = std::make_unique<event_t[]>(deliverable_cap);
            size_t container_sz         = 0u;
            size_t container_cap        = deliverable_cap;
            auto typeerased_container   = std::shared_ptr<event_t[]>(std::move(container));

            return new DeliveryHandle<event_t>{std::move(typeerased_container), container_sz, container_cap, consumer}; //TODO: global memory exhaustion - internalize new, we have to adhere to the practices
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }
    }

    template <class event_t>
    auto delvrsrv_open_preallocated_handle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap, char * preallocated_buf) noexcept -> std::expected<DeliveryHandle<event_t> *, exception_t>{

        constexpr size_t MIN_DELIVERABLE_CAP = 0u;
        constexpr size_t MAX_DELIVERABLE_CAP = size_t{1} << 30;

        if (consumer == nullptr){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(deliverable_cap, MIN_DELIVERABLE_CAP, MAX_DELIVERABLE_CAP) != deliverable_cap){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (preallocated_buf == nullptr){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        auto destructor = [deliverable_cap](event_t * event_arr) noexcept{
            static_assert(noexcept(network_producer_consumer::inplace_destruct_array(event_arr, deliverable_cap)));
            network_producer_consumer::inplace_destruct_array(event_arr, deliverable_cap);
        };

        try{
            event_t * event_ptr         = network_producer_consumer::inplace_construct<event_t[]>(preallocated_buf, deliverable_cap);
            auto container              = std::unique_ptr<event_t[], decltype(destructor)>(event_ptr, destructor);
            size_t container_sz         = 0u;
            size_t container_cap        = deliverable_cap;
            auto typeerased_container   = std::shared_ptr<event_t[]>(std::move(container)); 

            return new DeliveryHandle<event_t>{std::move(typeerased_container), container_sz, container_cap, consumer}; //TODO: global memory exhaustion - internalize new
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception())); //never gonna happen, C++ ABI or API whatever guarantees that new == exception
        }
    }

    template <class event_t>
    auto delvrsrv_allocation_cost(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> size_t{

        return network_producer_consumer::inplace_construct_size<event_t[]>(deliverable_cap);
    }

    //this is more complex than we think, we are relying on compiler to do their job right
    template <class event_t>
    void delvrsrv_deliver(DeliveryHandle<event_t> * handle, event_t event) noexcept(std::is_nothrow_move_assignable_v<event_t>){

        handle = stdx::safe_ptr_access(handle); 

        if (handle->deliverable_sz == handle->deliverable_cap){            
            if (handle->deliverable_cap == 0u) [[unlikely]]{
                handle->consumer->push(std::make_move_iterator(&event), 1u);
                return;
            } else [[likely]]{
                handle->consumer->push(std::make_move_iterator(handle->deliverable_arr.get()), handle->deliverable_sz);
                handle->deliverable_sz = 0u;
            }
        }

        handle->deliverable_arr[handle->deliverable_sz++] = std::move(event);
    }

    static inline auto delvrsrv_deliver_lambda = []<class event_t>(DeliveryHandle<event_t> * handle, event_t event) noexcept{
        delvrsrv_deliver(handle, std::move(event));
    };

    template <class event_t>
    auto delvrsrv_close_handle(DeliveryHandle<event_t> * handle) noexcept{

        handle = stdx::safe_ptr_access(handle); 

        if (handle->deliverable_sz != 0u){
            handle->consumer->push(std::make_move_iterator(handle->deliverable_arr.get()), handle->deliverable_sz);
        }

        delete handle; //TODO: internalize delete
    }

    static inline auto delvrsrv_close_handle_lambda = []<class event_t>(DeliveryHandle<event_t> * handle) noexcept{
        delvrsrv_close_handle(handle);
    };

    template <class event_t>
    auto delvrsrv_open_raiihandle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<std::unique_ptr<DeliveryHandle<event_t>, decltype(delvrsrv_close_handle_lambda)>, exception_t>{

        std::expected<DeliveryHandle<event_t> *, exception_t> handle = delvrsrv_open_handle(consumer, deliverable_cap);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<DeliveryHandle<event_t>, decltype(delvrsrv_close_handle_lambda)>(handle.value(), delvrsrv_close_handle_lambda);
    }

    template <class event_t>
    auto delvrsrv_open_preallocated_raiihandle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap, char * preallocated_buf) noexcept -> std::expected<std::unique_ptr<DeliveryHandle<event_t>, decltype(delvrsrv_close_handle_lambda)>, exception_t>{

        std::expected<DeliveryHandle<event_t> *, exception_t> handle = delvrsrv_open_preallocated_handle(consumer, deliverable_cap, preallocated_buf);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<DeliveryHandle<event_t>, decltype(delvrsrv_close_handle_lambda)>(handle.value(), delvrsrv_close_handle_lambda);
    }    

    //-----

    //how do we design this? it is std::unordered_map<key_t, std::vector<event_t>> at heart
    //let's see how we could improve this
    //std::unordered_fast_map_insert_only
    
    //std::vector<event_t> -> open_addressing growth
    //how?
    //we are to preallocate the event_t as a vector, and build a mini heap on top of such vector 
    //is that the plan?
    //is it a linked_list problem, we are to extend twice the size whenever the vector<> goes out of space, store the next linked list, <bump_allocate> the next segment of the preallocated buffer
    //if there's no such space, we are to dump the container and the delivery repeats
    //we are to leverage no-extra memory for as long as possible by leveraging the stack_allocation technique
    //we have 1024 cores, only one RAM BUS, so its very important that we stack allocate everything and only heap allocate things that cannot be stack allocated, such heap cannot be fragmennted by using our strategy of cyclic page
    //we are to implement this today + tomorrow, this is important

    template <class key_t, class event_t>
    auto delvrsrv_open_kv_handle(KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<KVDeliveryHandle<key_t, event_t> *, exception_t>{

    }

    template <class key_t, class event_t>
    auto delvrsrv_open_kv_preallocated_handle(KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap, char * preallocated_buf) noexcept -> std::expected<KVDeliveryHandle<key_t, event_t> *, exception_t>{

    }

    template <class key_t, class event_t>
    auto delvrsrv_kv_allocation_cost(KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap) noexcept -> size_t{

    }

    template <class key_t, class event_t>
    void delvrsrv_deliver(KVDeliveryHandle<key_t, event_t> * handle, const key_t& key, event_t event) noexcept{

    }

    template <class key_t, class event_t>
    auto delvrsrv_close_kv_handle(KVDeliveryHandle<key_t, event_t> * handle) noexcept{

    }

    static inline auto delvrsrv_close_kv_handle_lambda = []<class key_t, class event_t>(KVDeliveryHandle<key_t, event_t> * handle) noexcept{
        delvrsrv_close_kv_handle(handle);
    };

    template <class key_t, class event_t>
    auto delvrsrv_open_kv_raiihandle(KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<std::unique_ptr<KVDeliveryHandle<key_t, event_t>, decltype(delvrsrv_close_kv_handle_lambda)>, exception_t>{

        std::expected<KVDeliveryHandle<key_t, event_t> *, exception_t> handle = delvrsrv_open_kv_handle(consumer, deliverable_cap);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<KVDeliveryHandle<key_t, event_t>, decltype(delvrsrv_close_kv_handle_lambda)>(handle.value(), delvrsrv_close_kv_handle_lambda);
    }

    template <class key_t, class event_t>
    auto delvrsrv_open_kv_preallocated_raiihandle(KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap, char * preallocated_buf) noexcept -> std::expected<std::unique_ptr<KVDeliveryHandle<key_t, event_t>, decltype(delvrsrv_close_kv_handle_lambda)>, exception_t>{

        std::expected<KVDeliveryHandle<key_t, event_t> *, exception_t> handle = delvrsrv_open_kv_preallocated_handle(consumer, deliverable_cap, preallocated_buf);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<KVDeliveryHandle<key_t, event_t>, decltype(delvrsrv_close_kv_handle_lambda)>(handle.value(), delvrsrv_close_kv_handle_lambda);
    }
}

#endif