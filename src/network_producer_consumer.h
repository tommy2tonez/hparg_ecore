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
#include "dg_map_variants.h"

namespace dg::network_producer_consumer{

    template <class EventType>
    static inline constexpr bool meets_event_precond_v  = std::is_same_v<EventType, std::decay_t<EventType>>;

    template <class EventType>
    static inline constexpr bool meets_key_precond_v    = std::is_same_v<EventType, std::decay_t<EventType>>;

    template <class EventType>
    struct ProducerInterface{
        using event_t = EventType;
        
        static_assert(meets_event_precond_v<event_t>);

        virtual ~ProducerInterface() noexcept = default;
        virtual void get(event_t * event_arr, size_t& event_arr_sz, size_t event_arr_cap) noexcept = 0;
    };

    template <class EventType>
    struct ConsumerInterface{
        using event_t = EventType;

        static_assert(meets_event_precond_v<event_t>);        

        virtual ~ConsumerInterface() noexcept = default;
        virtual void push(std::move_iterator<event_t *> event_arr, size_t event_arr_sz) noexcept = 0;
    };

    template <class KeyType, class EventType>
    struct KVConsumerInterface{

        using key_t     = KeyType;
        using event_t   = EventType;

        static_assert(meets_key_precond_v<key_t>);
        static_assert(meets_event_precond_v<event_t>);

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


    //^^^
    //fine, pass code review 

    template <class EventType>
    struct DeliveryHandle{
        EventType * deliverable_arr;
        size_t deliverable_sz;
        size_t deliverable_cap;
        ConsumerInterface<EventType> * consumer;
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
            auto container              = std::unique_ptr<event_t[]>(new event_t[deliverable_cap]);
            size_t container_sz         = 0u;
            size_t container_cap        = deliverable_cap;
            auto delivery_handle        = std::unique_ptr<DeliveryHandle<event_t>>(new DeliveryHandle<event_t>{container.get(), container_sz, container_cap, consumer});
            auto rs                     = delivery_handle.get();

            container.release();
            delivery_handle.release();

            return rs;
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

        try{
            auto raii_event_ptr     = network_producer_consumer::raii_inplace_construct<event_t[]>(preallocated_buf, deliverable_cap);
            size_t container_sz     = 0u;
            size_t container_cap    = deliverable_cap;
            auto raii_handle_ptr    = network_producer_consumer::raii_inplace_construct<DeliveryHandle<event_t>>(std::next(preallocated_buf, network_producer_consumer::inplace_construct_size<event_t[]>(deliverable_cap)),
                                                                                                                 DeliveryHandle<event_t>{raii_event_ptr.value(), container_sz, container_cap, consumer});
            auto rs                 = raii_handle_ptr.value();

            raii_event_ptr.release();
            raii_handle_ptr.release();

            return rs;
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }
    }

    template <class event_t>
    auto delvrsrv_allocation_cost(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> size_t{

        return network_producer_consumer::inplace_construct_size<event_t[]>(deliverable_cap) 
               + network_producer_consumer::inplace_construct_size<DeliveryHandle<event_t>>(DeliveryHandle<event_t>{});
    }

    template <class event_t>
    void delvrsrv_deliver(DeliveryHandle<event_t> * handle, event_t event) noexcept{

        static_assert(std::is_nothrow_move_assignable_v<event_t>);
        handle = stdx::safe_ptr_access(handle); 

        if (handle->deliverable_sz == handle->deliverable_cap){            
            if (handle->deliverable_cap == 0u) [[unlikely]]{
                handle->consumer->push(std::make_move_iterator(&event), 1u);
                return;
            } else [[likely]]{
                handle->consumer->push(std::make_move_iterator(handle->deliverable_arr), handle->deliverable_sz);
                handle->deliverable_sz = 0u;
            }
        }

        handle->deliverable_arr[handle->deliverable_sz++] = std::move(event);
    }

    static inline auto delvrsrv_deliver_lambda = []<class event_t>(DeliveryHandle<event_t> * handle, event_t event) noexcept{
        delvrsrv_deliver(handle, std::move(event));
    };

    void delvrsrv_clear(DeliveryHandle<event_t> * handle) noexcept{

        handle = stdx::safe_ptr_access(handle);

        if (handle->deliverable_sz != 0u){
            handle->consumer->push(std::make_move_iterator(handle->deliverable_arr), handle->deliverable_sz);
            handle->deliverable_sz = 0u;
        }
    } 

    template <class event_t>
    auto delvrsrv_close_handle(DeliveryHandle<event_t> * handle) noexcept{

        delvrsrv_clear(handle);

        delete[] handle->deliverable_arr;
        delete handle; //TODO: internalize delete
    }

    template <class event_t>
    auto delvrsrv_close_preallocated_handle(DeliveryHandle<event_t> * handle) noexcept{

        delvrsrv_clear(handle);

        network_producer_consumer::inplace_destruct_array(handle->deliverable_arr, handle->deliverable_cap);
        network_producer_consumer::inplace_destruct_object(handle);
    }

    static inline auto delvrsrv_close_handle_lambda = []<class event_t>(DeliveryHandle<event_t> * handle) noexcept{
        delvrsrv_close_handle(handle);
    };

    static inline auto delvrsrv_close_preallocated_handle_lambda = []<class event_t>(DeliveryHandle<event_t> * handle) noexcept{
        delvrsrv_close_preallocated_handle(handle);
    }

    template <class event_t>
    auto delvrsrv_open_raiihandle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<std::unique_ptr<DeliveryHandle<event_t>, decltype(delvrsrv_close_handle_lambda)>, exception_t>{

        std::expected<DeliveryHandle<event_t> *, exception_t> handle = delvrsrv_open_handle(consumer, deliverable_cap);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<DeliveryHandle<event_t>, decltype(delvrsrv_close_handle_lambda)>(handle.value(), delvrsrv_close_handle_lambda);
    }

    template <class event_t>
    auto delvrsrv_open_preallocated_raiihandle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap, char * preallocated_buf) noexcept -> std::expected<std::unique_ptr<DeliveryHandle<event_t>, decltype(delvrsrv_close_preallocated_handle_lambda)>, exception_t>{

        std::expected<DeliveryHandle<event_t> *, exception_t> handle = delvrsrv_open_preallocated_handle(consumer, deliverable_cap, preallocated_buf);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<DeliveryHandle<event_t>, decltype(delvrsrv_close_preallocated_handle_lambda)>(handle.value(), delvrsrv_close_preallocated_handle_lambda);
    }    

    //vvv
    //-----

    //^^^
    //fine, minimum viable product

    struct BumpAllocatorResource{
        char * head;
        size_t sz;
        size_t cap;
    };

    auto bump_allocator_initialize(size_t buf_sz) -> std::expected<BumpAllocatorResource *, exception_t>{

        try{
            size_t sz       = 0u;
            size_t cap      = buf_sz;
            auto safe_head  = std::unique_ptr<char[]>(new char[buf_sz]); //this is harder than expected
            auto resource   = std::unique_ptr<BumpAllocatorResource>(new BumpAllocatorResource{safe_head.get(), sz, cap}); 
            auto rs         = resource.get();
            safe_head.release();
            resource.release();

            return rs;
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }
    }

    auto bump_allocator_preallocated_initialize(size_t buf_sz, char * buf) -> std::expected<BumpAllocatorResource *, exception_t>{

        size_t sz       = 0u;
        size_t cap      = buf_sz;        
        auto resource   = BumpAllocatorResource{buf, sz, cap};

        return network_producer_consumer::inplace_construct<BumpAllocatorResource>(std::next(buf, buf_sz), resource);
    }

    void bump_allocator_deinitialize(BumpAllocatorResource * obj) noexcept{

        delete[] obj->head;
        delete obj;
    }

    void bump_allocator_preallocated_deinitialize(BumpAllocatorResource * obj) noexcept{

        network_producer_consumer::inplace_destruct_object(obj);
    }    

    auto bump_allocator_allocation_cost(size_t buf_sz) noexcept -> size_t{

        return buf_sz + inplace_construct_size<BumpAllocatorResource>(BumpAllocatorResource{});
    }

    void bump_allocator_reset(BumpAllocatorResource& bump_allocator) noexcept{

        bump_allocator.sz = 0u;
    }

    auto bump_allocator_allocate(BumpAllocatorResource& bump_allocator, size_t sz) noexcept -> std::expected<char *, exception_t>{

        size_t nxt_sz = bump_allocator.sz + sz;

        if (nxt_sz > bump_allocator.cap){
            return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
        }

        char * current      = std::next(bump_allocator.head, bump_allocator.sz);
        bump_allocator.sz   += sz; 

        return current;
    }

    //vvv
    //-----------------------

    //it is not undersaid when we say this is probably the most important performance code that we need to get right
    //we'll focus on improving the code quality + bench tomorrow, its gonna be hard, we are to crunch 1 << 28 keyvalue feeds/ second
    //if we can't reach the number, it is not considered a success
    //people reached 1 << 29 keyvalue feeds/ second

    template <class EventType>
    struct KVEventContainer{
        EventType * ptr;
        size_t sz;
        size_t cap;
    };

    template <class EventType>
    void destroy_kv_event_container(KVEventContainer<EventType> event_container) noexcept{

        static_assert(std::is_nothrow_destructible_v<EventType>); //it must be noexcept
        dg::network_producer_consumer::inplace_destruct_array(event_container.ptr, event_container.sz);
    }

    static inline auto destroy_kv_event_container_lambda = []<class EventType>(KVEventContainer<EventType> event_container) noexcept{
        destroy_kv_event_container(event_container);        
    };

    static inline constexpr size_t DELVRSRV_KV_EVENT_CONTAINER_INITIAL_CAP      = size_t{1}; 
    static inline constexpr size_t DELVRSRV_KV_EVENT_CONTAINER_GROWTH_FACTOR    = size_t{2};

    template<class KeyType, class EventType>
    struct KVDeliveryHandle{
        BumpAllocatorResource * bump_allocator;
        dg::map_variants::unordered_unstable_map<KeyType, dg::unique_resource<KVEventContainer<EventType>, decltype(destroy_kv_event_container_lambda)>> key_event_map; //alright, we dont have that destructor, constructor, whatever practices, we are C people, we are to use struct + deinitializer, because it's so convenient, unless this is undefined, because we compiler dont know what to deinitialize first
        size_t deliverable_sz;
        size_t deliverable_cap;
        KVConsumerInterface<KeyType, EventType> * consumer;
    };

    //BumpAllocator has at most 2x the deliverable_cap, is it so
    //x + x1 + x2 + ... + xn = deliverable_cap 
    //capacity <= x * 2, due to insert only growth 

    //we'll figure std::shared_ptr<> out later
    //there are many issues
    //first is allocation, second is shared_ptr<> (std::memory_order_release is bad)
    //third is unordered_map not using preallocated mmeory
    //fourth is branch pipeline + instructions arent good enough
    //we will address these issues tmr


    //
    template <class EventType>
    auto delvrsrv_kv_bump_allocation_cost(size_t deliverable_cap) noexcept -> size_t{

        if (deliverable_cap == 0u){
            return bump_allocator_allocation_cost(0u);
        }

        size_t worst_case_cap   = deliverable_cap * DELVRSRV_KV_EVENT_CONTAINER_GROWTH_FACTOR;
        size_t minimum_cap      = DELVRSRV_KV_EVENT_CONTAINER_INITIAL_CAP;
        size_t bsz              = std::max(worst_case_cap, minimum_cap) * sizeof(EventType) + alignof(EventType)//this is hard, the alignof is hard to code

        return bump_allocator_allocation_cost(bsz);
    }

    template <class EventType>
    auto delvrsrv_kv_get_event_container_bsize(size_t capacity) noexcept -> size_t{

        return inplace_construct_size<EventType[]>(capacity);
    }

    template <class EventType, std::enable_if_t<std::is_nothrow_constructible_v<EventType>, bool> = true>
    auto delvrsrv_kv_get_preallocated_event_container(size_t capacity, char * buf) noexcept -> std::expected<dg::unique_resource<KVEventContainer<EventType>, decltype(destroy_kv_event_container_lambda)>, exception_t>{

        EventType * event_arr   = inplace_construct<EventType[]>(buf, capacity);
        auto container          = KVEventContainer<EventType>{event_arr, 0u, capacity};

        return dg::unique_resource<KVEventContainer<EventType>, decltype(destroy_kv_event_container_lambda)>(container, destroy_kv_event_container_lambda);        
    }

    template <class EventType, std::enable_if_t<!std::is_nothrow_constructible_v<EventType>, bool> = true>
    auto delvrsrv_kv_get_preallocated_event_container(size_t capacity, char * buf) noexcept -> std::expected<dg::unique_resource<KVEventContainer<EventType>, decltype(destroy_kv_event_container_lambda)>, exception_t>{

        try{
            EventType * event_arr   = inplace_construct<EventType[]>(buf, capacity);
            auto container          = KVEventContainer<EventType>(event_arr, 0u, capacity);

            return dg::unique_resource<KVEventContainer<EventType>, decltype(destroy_kv_event_container_lambda)>(container, destroy_kv_event_container_lambda);
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }
    }

    template <class EventLike>
    auto delvrsrv_kv_push_event_container(KVEventContainer<EventType>& container, EventLike&& event) noexcept -> exception_t{

        if constexpr(std::is_nothrow_assignable_v<EventType, EventLike&&>){
            if (container.sz >= container.cap){
                return dg::network_exception::RESOUCE_EXHAUSTION;
            }

            container.ptr[container.sz++] = std::forward<EventLike>(event);
            return dg::network_exception::SUCCESS;
        } else{
            static_assert(FALSE_VAL<>);
        }
    }

    //

    template <class key_t, class event_t>
    auto delvrsrv_kv_open_handle(KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<KVDeliveryHandle<key_t, event_t> *, exception_t>{

        const size_t MIN_DELIVERABLE_CAP    = 0u;
        const size_t MAX_DELIVERABLE_CAP    = size_t{1} << 30; 

        if (consumer == nullptr){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(deliverable_cap, MIN_DELIVERABLE_CAP, MAX_DELIVERABLE_CAP) != deliverable_cap){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        auto bump_allocator     = bump_allocator_initialize(delvrsrv_kv_bump_allocation_cost(deliverable_cap));

        if (!bump_allocator.has_value()){
            return std::unexpected(bump_allocator.error());
        }

        // try{
        //     // std::shared_ptr<BumpAllocatorResource> bump_allocator   = get_bump_allocator(delvrsrv_kv_bump_allocation_cost(deliverable_cap));
        //     auto key_event_map      = dg::map_variants::unordered_unstable_map<key_t, dg::unique_resource<KVEventContainer<EventType, decltype(&free_kv_event_container<event_t>)>>(deliverable_cap);
        //     size_t deliverable_sz   = 0u;

        //     return new KVDeliveryHandle<key_t, event_t>{std::move(bump_allocator), key_event_map, deliverable_sz, deliverable_cap}; //leaks
        // } catch (...){
        //     return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception())); //this is harder than expected
        // }
    }

    template <class key_t, class event_t>
    auto delvrsrv_kv_open_preallocated_handle(KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap, char * preallocated_buf) noexcept -> std::expected<KVDeliveryHandle<key_t, event_t> *, exception_t>{

        // const size_t MIN_DELIVERABLE_CAP    = 0u;
        // const size_t MAX_DELIVERABLE_CAP    = size_t{1} << 30;

        // if (consumer == nullptr){
        //     return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        // }

        // if (std::clamp(deliverable_cap, MIN_DELIVERABLE_CAP, MAX_DELIVERABLE_CAP) != deliverable_cap){
        //     return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        // }

        // try{
        //     std::shared_ptr<BumpAllocatorResource> bump_allocator   = get_preallocated_bump_allocator(delvrsrv_kv_bump_allocation_cost(deliverable_cap), preallocated_buf);
        //     auto key_event_map                                      = dg::map_variants::unordered_unstable_map<key_t, dg::unique_resource<KVEventContainer<EventType, decltype(&free_kv_event_container<event_t>)>>(deliverable_cap);
        //     size_t deliverable_sz                                   = 0u;

        //     return new KVDeliveryHandle<key_t, event_t>{std::move(bump_allocator), key_event_map, deliverable_sz, deliverable_cap};
        // } catch (...){
        //     return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }     
    }

    template <class key_t, class event_t>
    auto delvrsrv_kv_allocation_cost(KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap) noexcept -> size_t{

        return delvrsrv_kv_bump_allocation_cost(deliverable_cap);
    }

    template <class key_t, class event_t>
    void delvrsrv_kv_clear(KVDeliveryHandle<key_t, event_t> * handle) noexcept{

        for (auto it = handle->key_event_map.begin(), it != handle->key_event_map.end(); ++it){
            const auto& key     = it->first;
            auto& value         = it->second;
            handle->consumer->push(key, std::make_move_iterator(value.value().ptr), value.value().sz);
        }

        handle->key_event_map.clear();
        bump_allocator_reset(*handle->bump_allocator);
        handle->deliverable_sz = 0u;
    }

    //let's see
    //alright fellas, a lot of people think they are smart fellas, how about we except everything and throw everything, no you will have very BAD LEAK doing things like that
    //the idea originally is that every thing that is exceptable is thrown upon construction, and delivery (feed) must not throw for whatever reason, the things that do not adhere to the rules are not defined and pruned during the static_assert()
    //this is not an implementation defect, because we have experienced so many std defects (insert(first, last) mid way OOM, now we have bad leaks)
    //we are destruction noexcept, construction except, capacity, delivery noexcept people
    //we are C people, that live in the C++ world
    //we hope that things are simple and there aint thrown destruction + thrown move + thrown copy + etc.

    template <class key_t, class event_t>
    void delvrsrv_kv_deliver(KVDeliveryHandle<key_t, event_t> * handle, const key_t& key, event_t event) noexcept{

        static_assert(std::is_nothrow_default_constructible_v<key_t>); //...
        static_assert(std::is_nothrow_default_constructible_v<event_t>);
        static_assert(std::is_nothrow_destructible_v<event_t>);
        static_assert(std::is_nothrow_move_constructible_v<event_t>);

        handle = stdx::safe_ptr_access(handle);

        if (handle->deliverable_sz == handle->deliverable_cap){
            if (handle->deliverable_cap == 0u) [[unlikely]]{
                handle->consumer->push(key, std::make_move_iterator(&event), 1u);
                return;
            } else [[likely]]{
                delvrsrv_kv_clear(handle);
            }
        }

        auto map_ptr = handle->key_event_map.find(key);  

        //resolving map_ptr, return valid map_ptr if the code path is to through the if
        if (map_ptr == handle->key_event_map.end()){
            std::expected<char *, exception_t> buf = bump_allocator_allocate(*handle->bump_allocator, delvrsrv_kv_get_event_container_bsize<event_t>(DELVRSRV_KV_EVENT_CONTAINER_INITIAL_CAP));

            if (!buf.has_value()) [[unlikely]]{
                if (buf.error() == dg::network_exception::RESOURCE_EXHAUSTION){
                    delvrsrv_kv_clear(handle);
                    buf = dg::network_exception_handler::nothrow_log(bump_allocator_allocate(*handle->bump_allocator, delvrsrv_kv_get_event_container_bsize<event_t>(DELVRSRV_KV_EVENT_CONTAINER_INITIAL_CAP)));
                } else{
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }
            }

            auto req_container                  = dg::network_exception_handler::nothrow_log(delvrsrv_kv_get_preallocated_event_container<event_t>(DELVRSRV_KV_EVENT_CONTAINER_INITIAL_CAP, buf.value()));
            auto [emplace_ptr, emplace_status]  = handle->key_event_map.try_emplace(key, std::move(req_container));  
            dg::network_exception::dg_assert(emplace_status);
            map_ptr                             = emplace_ptr;
        }

        //resolving map_ptr, return valid_map_ptr if the code path is to through the if
        if (map_ptr->second.value().sz == map_ptr->second.value().cap){
            size_t new_sz = map_ptr->second.value().cap * DELVRSRV_KV_EVENT_CONTAINER_GROWTH_FACTOR;
            std::expected<char *, exception_t> buf = bump_allocator_allocate(*handle->bump_allocator, delvrsrv_kv_get_event_container_bsize<event_t>(new_sz));

            if (!buf.has_value()) [[unlikely]]{
                if (buf.error() == dg::network_exception::RESOURCE_EXHAUSTION){
                    delvrsrv_kv_clear(handle);
                    buf                                 = dg::network_exception_handler::nothrow_log(bump_allocator_allocate(*handle->bump_allocator, delvrsrv_kv_get_event_container_bsize<event_t>(DELVRSRV_KV_EVENT_CONTAINER_INITIAL_CAP)));
                    auto req_container                  = dg::network_exception_handler::nothrow_log(delvrsrv_kv_get_preallocated_event_container<event_t>(DELVRSRV_KV_EVENT_CONTAINER_INITIAL_CAP, buf.value()));
                    auto [emplace_ptr, emplace_status]  = handle->key_event_map.try_emplace(key, std::move(req_container));
                    dg::network_exception::dg_assert(emplace_status);
                    map_ptr                             = emplace_ptr;
                } else{
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }
            } else{
                auto req_container  = dg::network_exception_handler::nothrow_log(delvrsrv_kv_get_preallocated_event_container<event_t>(new_sz, buf.value()));
                std::copy(std::make_move_iterator(map_ptr->second.value().ptr), std::make_move_iterator(std::next(map_ptr->second.value().ptr, map_ptr->second.value().sz)), req_container.value().ptr);
                map_ptr->second     = std::move(req_container);
            }
        }

        dg::network_exception_handler::nothrow_log(delvrsrv_kv_push_event_container(map_ptr->second.value(), std::move(event))); //valid map_ptr push_back
        handle->deliverable_sz += 1;
    }

    template <class key_t, class event_t>
    void delvrsrv_kv_close_handle(KVDeliveryHandle<key_t, event_t> * handle) noexcept{

        handle = stdx::safe_ptr_access(handle);
        delvrsrv_kv_clear(handle);

        // delete handle;
    }

    template <class key_t, class event_t>
    void delvrsrv_kv_close_preallocated_handle(KVDeliveryHandle<key_t, event_t> * handle) noexcept{

        handle = stdx::safe_ptr_access(handle);
        delvrsrv_kv_clear(handle);

        //
    }

    static inline auto delvrsrv_kv_close_handle_lambda = []<class key_t, class event_t>(KVDeliveryHandle<key_t, event_t> * handle) noexcept{
        delvrsrv_kv_close_handle(handle);
    };

    template <class key_t, class event_t>
    auto delvrsrv_kv_open_raiihandle(KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<std::unique_ptr<KVDeliveryHandle<key_t, event_t>, decltype(delvrsrv_kv_close_handle_lambda)>, exception_t>{

        std::expected<KVDeliveryHandle<key_t, event_t> *, exception_t> handle = delvrsrv_kv_open_handle(consumer, deliverable_cap);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<KVDeliveryHandle<key_t, event_t>, decltype(delvrsrv_kv_close_handle_lambda)>(handle.value(), delvrsrv_kv_close_handle_lambda);
    }

    template <class key_t, class event_t>
    auto delvrsrv_kv_open_preallocated_raiihandle(KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap, char * preallocated_buf) noexcept -> std::expected<std::unique_ptr<KVDeliveryHandle<key_t, event_t>, decltype(delvrsrv_kv_close_handle_lambda)>, exception_t>{

        std::expected<KVDeliveryHandle<key_t, event_t> *, exception_t> handle = delvrsrv_kv_open_preallocated_handle(consumer, deliverable_cap, preallocated_buf);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<KVDeliveryHandle<key_t, event_t>, decltype(delvrsrv_kv_close_handle_lambda)>(handle.value(), delvrsrv_kv_close_handle_lambda);
    }
}

#endif