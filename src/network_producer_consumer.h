#ifndef __NETWORK_PRODUCER_CONSUMER_H__
#define __NETWORK_PRODUCER_CONSUMER_H__

#include <stdint.h>
#include <stddef.h>
#include <array>
#include <memory>

namespace dg::network_producer_consumer{

    template <class EventType>
    struct ProducerInterface{
        using event_t = EventType;
        static_assert(std::is_trivial_v<event_t>);

        virtual ~ProducerInterface() noexcept = default;
        virtual void get(event_t * events, size_t& event_sz, size_t event_cap) noexcept = 0;
    };

    template <class EventType>
    struct DistributedProducerInterface{
        using event_t = EventType;
        static_assert(std::is_trivial_v<event_t>);
        
        virtual ~DistributedProducerInterface() noexcept = default; 
        virtual auto range() const noexcept -> size_t = 0;
        virtual void get(size_t i, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept = 0;  
    };

    template <class EventType>
    struct ConsumerInterface{
        using event_t = EventType;
        static_assert(std::is_trivial_v<event_t>);
        
        virtual ~ConsumerInterface() noexcept = default;
        virtual void push(event_t * src, size_t src_sz) noexcept = 0;
    };

    template <class EventType>
    struct LimitConsumerInterface{
        using event_t = EventType;
        static_assert(std::is_trivial_v<event_t>);
        
        virtual ~LimitConsumerInterface() noexcept = default;
        virtual void push(event_t * src, size_t src_sz) noexcept = 0;  
        virtual auto capacity() const noexcept -> size_t = 0;
    };

    template <class EventType>
    class LimitConsumerToConsumerWrapper: public virtual ConsumerInterface<EventType>{

        private:

            std::shared_ptr<LimitConsumerInterface<EventType>> base;
        
        public:

            LimitConsumerToConsumerWrapper(std::shared_ptr<LimitConsumerInterface<EventType>> base) noexcept: base(std::move(base)){}

            void push(event_t * events, size_t event_sz) noexcept{
                
                event_t * cur   = events;
                size_t rem_sz   = event_sz; 

                while (rem_sz != 0u){
                    size_t submitting_sz = dg::network_genult::safe_posint_access(std::min(rem_sz, this->base->capacity()));
                    this->base->push(cur, submitting_sz);
                    std::advance(cur, submitting_sz);
                    rem_sz -= submitting_sz;
                }
            }
    };

    //don't think that this is some kind of invention - it's hard (if not impossibile) to do std-container-compatibility + c-style error throw for instantiation
    //I rather let compiler does all the magics for constructor, destructor, move, copy, who-knows-what-in-the-future (which is a mess), and use std_compatible way - unique_ptr<> to stay in the "holy grail" of std_container_compatibility

    template <class event_t>
    struct DeliveryHandle{
        static_assert(std::is_trivial_v<event_t>);

        std::unique_ptr<event_t[]> deliverable_arr;
        size_t deliverable_sz;
        size_t deliverable_cap;
        ConsumerInterface<event_t> * consumer;
    };

    template <class event_t>
    auto delvrsrv_open_handle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<DeliveryHandle<event_t> *, exception_t>{

        if (!consumer){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (deliverable_cap == 0u){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return new DeliveryHandle<event_t>{std::make_unique<event_t[]>(deliverable_cap), 0u, deliverable_cap, consumer}; //TODO: global memory exhaustion -
    }

    template <class event_t>
    void delvrsrv_deliver(DeliveryHandle<event_t> * handle, event_t event) noexcept{

        handle = dg::network_genult::safe_ptr_access(handle); 

        if (handle->deliverable_sz == handle->deliverable_cap){
            handle->consumer->push(handle->deliverable_arr.get(), handle->deliverable_sz);
            handle->deliverable->sz = 0u;
        }

        handle->deliverable_arr[handle->deliverable_sz++] = std::move(event);
    }

    template <class event_t>
    auto delvrsrv_close_handle(DeliveryHandle<event_t> * handle) noexcept{

        handle = dg::network_genult::safe_ptr_access(handle);
        handle->consumer->push(handle->deliverable_arr.get(), handle->deliverable_sz);
        delete handle;
    }

    template <class event_t>
    auto delvrsrv_open_raiihandle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<std::unique_ptr<DeliveryHandle<event_t>, decltype(&delvrsrv_close_handle<event_t>)>, exception_t>{

        std::expected<DeliveryHandle<event_t> *, exception_t> handle = delvrsrv_open_handle(consumer, deliverable_cap);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return {std::in_place_t{}, handle.value(), delvrsrv_close_handle<event_t>};
    }

    template <class event_t>
    struct XDeliveryHandle{
        std::unique_ptr<event_t[]> deliverable_arr;
        size_t deliverable_sz;
        size_t deliverable_cap;
        size_t meter_sz;
        size_t meter_cap;
        size_t unit_meter_cap;
        ConsumerInterface<event_t> * consumer;
    };

    template <class event_t>
    auto xdelvsrv_open_handle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap, size_t meter_cap, size_t unit_meter_cap) -> std::expected<XDeliveryHandle<event_t> *, exception_t>{

        static_assert(std::is_trivial_v<event_t>);

        if (!consumer){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (deliverable_cap == 0u){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }
        
        if (unit_meter_cap > meter_cap){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return new XDeliveryHandle<event_t>{std::make_unique<event_t[]>(deliverable_cap), 0u, deliverable_cap, 0u, meter_cap, unit_meter_cap, consumer}; 
    }

    template <class event_t>
    auto xdelvsrv_deliver(XDeliveryHandle<event_t> * handle, event_t event, size_t meter_sz) noexcept -> exception_t{

        handle = dg::network_genult::safe_ptr_access(handle); 

        if (meter_sz > handle->unit_meter_cap){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        bool flush_cond1    = handle->deliverable_cap == handle->deliverable_vec.size();
        bool flush_cond2    = handle->meter_sz + meter_sz > handle->meter_cap;

        if (flush_cond1 || flush_cond2){
            handle->consumer->push(handle->deliverable_arr.get(), handle->deliverable_sz);
            handle->deliverable_sz = 0u;
            handle->meter_sz = 0u;
        }

        handle->deliverable_arr[handle->deliverable_sz++] = std::move(event);
        handle->meter_sz += meter_sz;

        return dg::network_exception::SUCCESS;
    }

    template <class event_t>
    auto xdelvsrv_close_handle(XDeliveryHandle<event_t> * handle) noexcept{

        handle = dg::network_genult::safe_ptr_access(handle); //avoid closing nullptr
        handle->consumer->push(handle->deliverable_arr.get(), handle->deliverable_sz);
        delete handle; //internal memory - solve later
    }

    template <class event_t>
    auto xdelvsrv_open_raiihandle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap, size_t meter_cap, size_t unit_meter_sz) noexcept -> std::expected<std::unique_ptr<XDeliveryHandle<event_t>, decltype(&xdelvsrv_close_handle<event_t>)>, exception_t>{

        std::expected<XDeliveryHandle<event_t> *, exception_t> handle = xdelvsrv_open_handle(consumer, deliverable_cap, meter_cap, unit_meter_sz);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return {std::in_place_t{}, handle.value(), xdelvsrv_close_handle<event_t>};
    }
}

namespace dg::network_raii_producer_consumer{

    //it's hard to code this correctly so its better to be correct for a very minute subset of usecases than not being correct at all 

    template <class EventType>
    struct ProducerInterface{
        using event_t = EventType;
        static_assert(std::is_nothrow_destructible_v<event_t>);
        static_assert(std::is_nothrow_move_constructible_v<event_t>);
        static_assert(std::is_nothrow_move_assignable_v<event_t>);

        virtual ~ProducerInterface() noexcept = default;
        virtual auto get(size_t) noexcept -> dg::network_std_container::vector<event_t> = 0; //yeah - optional is werid -
    };

    template <class EventType>
    struct ConsumerInterface{
        using event_t = EventType;
        static_assert(std::is_nothrow_destructible_v<event_t>);
        static_assert(std::is_nothrow_move_constructible_v<event_t>); 
        static_assert(std::is_nothrow_move_assignable_v<event_t>);

        virtual ~ConsumerInterface() noexcept = default;
        virtual void push(dg::network_std_container::vector<event_t>) noexcept = 0;
    };

    template <class EventType>
    struct LimitConsumerInterface{
        using event_t = EventType;
        static_assert(std::is_nothrow_destructible_v<event_t>);
        static_assert(std::is_nothrow_move_constructible_v<event_t>);
        static_assert(std::is_nothrow_move_assignable_v<event_t>);

        virtual ~LimitConsumerInterface() noexcept = default;
        virtual void push(dg::network_std_container::vector<event_t>) noexcept = 0;
        virtual auto capcity() const noexcept -> size_t = 0; 
    };

    template <class EventType>
    class LimitConsumerToConsumerWrapper: public virtual ConsumerInterface<EventType>{

        private:

            std::shared_ptr<LimitConsumerInterface<EventType>> base;
        
        public:

            LimitConsumerToConsumerWrapper(std::shared_ptr<LimitConsumerInterface<EventType>> base) noexcept: base(std::move(base)){}

            void push(dg::network_std_container::vector<EventType> vec) noexcept{

                while (!vec.empty()){ 
                    size_t extracting_sz = dg::network_genult::safe_posint_access(std::min(vec.size(), this->base->capacity()));
                    this->base->push(this->extract_back(vec, extracting_sz));
                }
            }
        
        private:

            auto extract_back(dg::network_std_container::vector<EventType>& vec, size_t extracting_sz) noexcept -> dg::network_std_container::vector<EventType>{
                
                if constexpr(DEBUG_MODE_FLAG){
                    if (extracting_sz > vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                dg::network_std_container::vector<EventType> rs{};
                rs.reserve(extracting_sz);

                for (size_t i = 0u; i < extracting_sz; ++i){
                    rs.push_back(std::move(vec.back()));
                    vec.pop_back();
                }

                return rs;
            }
    };

    template <class event_t>
    struct DeliveryHandle{
        static_assert(std::is_nothrow_destructible_v<event_t>);
        static_assert(std::is_nothrow_move_constructible_v<event_t>);
        static_assert(std::is_nothrow_move_assignable_v<event_t>);

        dg::network_std_container::vector<event_t> deliverable_vec;
        size_t deliverable_cap;
        ConsumerInterface<event_t> * consumer;
    };

    template <class event_t>
    auto delvsrv_open_handle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<DeliveryHandle<event_t> *, exception_t>{

        if (!consumer){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (deliverable_cap == 0u){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        dg::network_std_container::vector<event_t> deliverable_vec{};
        deliverable_vec.reserve(deliverable_cap);

        return new DeliveryHandle<event_t>{std::move(deliverable_vec), deliverable_cap, consumer};
    }

    template <class event_t>
    void delvsrv_deliver(DeliveryHandle<event_t> * handle, event_t event) noexcept{

        handle = dg::network_genult::safe_ptr_access(handle);

        if (handle->deliverable_vec.size() == handle->deliverable_cap){
            handle->consumer->push(std::move(handle->deliverable_vec));
        }

        handle->deliverable_vec.push_back(std::move(event));
    }

    template <class event_t>
    void delvsrv_close_handle(DeliveryHandle<event_t> * handle) noexcept{

        handle = dg::network_genult::safe_ptr_access(handle);
        handle->consumer->push(std::move(handle->deliverable_vec));
        delete handle;
    }

    template <class event_t>
    auto delvsrv_open_raiihandle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<std::unique_ptr<DeliveryHandle<event_t>, decltype(&delvsrv_close_handle<event_t>)>, exception_t>{

        std::expected<DeliveryHandle<event_t *>, exception_t> handle = delvsrv_open_handle(consumer, deliverable_cap);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return {std::in_place_t{}, handle.value(), delvsrv_close_handle<event_t>};
    }

    template <class event_t>
    struct XDeliveryHandle{
        static_assert(std::is_nothrow_destructible_v<event_t>);
        static_assert(std::is_nothrow_move_constructible_v<event_t>);
        static_assert(std::is_nothrow_move_assignable_v<event_t>);

        dg::network_std_container::vector<event_t> deliverable_vec;
        size_t deliverable_cap; //don't know if this is necessary - vector provides strong guarantee for this
        size_t meter_sz;
        size_t meter_cap;
        size_t meter_unit_cap;
        ConsumerInterface<event_t> * consumer;
    };

    template <class event_t>
    auto xdelvsrv_open_handle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap, size_t meter_cap, size_t meter_unit_cap) noexcept -> std::expected<XDeliveryHandle<event_t> *, exception_t>{
        
        if (!consumer){
            return std::unexpected(dg::network_exception::INVALID_AGRUMENT);
        }

        if (deliverable_cap == 0u){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (meter_unit_cap > meter_cap){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        dg::network_std_container::vector<event_t> deliverable_vec{};
        deliverable_vec.reserve(deliverable_cap);

        return new XDeliveryHandle<event_t>{std::move(deliverable_vec), deliverable_cap, 0u, meter_cap, meter_unit_cap, consumer};
    }

    template <class event_t>
    auto xdelvsrv_deliver(XDeliveryHandle<event_t> * handle, event_t event, size_t meter_sz) noexcept -> exception_t{

        handle = dg::network_genult::safe_ptr_access(handle);

        if (meter_sz > handle->meter_unit_cap){
            return dg::network_exception::INVALID_ARGUMENT;    
        }

        bool flush_cond_1   = handle->deliverable_vec.size() == handle->deliverable_cap;
        bool flush_cond_2   = handle->meter_sz + meter_sz > handle->meter_cap;

        if (flush_cond_1 || flush_cond_2){
            handle->consumer->push(std::move(handle->deliverable_vec));
            handle->meter_sz = 0u;
        }

        handle->deliverable_vec.push_back(std::move(event));
        handle->meter_sz += meter_sz;

        return dg::network_exception::SUCCESS;
    }

    template <class event_t>
    void xdelvsrv_close_handle(XDeliveryHandle<event_t> * handle) noexcept{

        handle = dg::network_genult::safe_ptr_access(handle);
        handle->consumer->push(std::move(handle->deliverable_vec));
        delete handle;
    }

    template <class evnet_t>
    auto xdelvsrv_open_raiihandle(ConsumerInterface<event_t> * consuemr, size_t deliverable_cap, size_t meter_cap, size_t meter_unit_cap) noexcept -> std::expected<std::unique_ptr<XDeliveryHandle<event_t>, decltype(&xdelvsrv_close_handle<event_t>)>, exception_t>{

        std::expected<XDeliveryHandle<event_t> *, exception_t> handle = xdelvsrv_open_handle(consumer, deliverable_cap, meter_cap, meter_unit_cap);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return {std::in_place_t{}, handle.value(0, xdelvsrv_clost_handle<event_t>)};
    }
}

#endif