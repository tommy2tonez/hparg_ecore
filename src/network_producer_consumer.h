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
        
        virtual ~ProducerInterface() noexcept = default;
        virtual void get(event_t * events, size_t& event_sz, size_t event_cap) noexcept = 0;
    };

    template <class EventType>
    struct DistributedProducerInterface{
        using event_t = EventType;

        virtual ~DistributedProducerInterface() noexcept = default; 
        virtual auto range() const noexcept -> size_t = 0;
        virtual void get(size_t i, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept = 0;  
    };

    template <class EventType>
    struct ConsumerInterface{
        using event_t = EventType;

        virtual ~ConsumerInterface() noexcept = default;
        virtual void push(event_t * src, size_t src_sz) noexcept = 0;
    };

    template <class EventType>
    struct LimitConsumerInterface{
        using event_t = EventType;

        virtual ~LimitConsumerInterface() noexcept = default;
        virtual void push(event_t * src, size_t src_sz) noexcept = 0;  
        virtual auto capacity() const noexcept -> size_t = 0;
    };

    template <class EventType>
    class LimitConsumerToConsumerWrapper: public ConsumerInterface<EventType>{

        private:

            std::shared_ptr<LimitConsumerInterface<EventType>> base;
        
        public:

            LimitConsumerToConsumerWrapper(std::shared_ptr<LimitConsumerInterface<EventType>> base) noexcept: base(std::move(base)){}

            void push(event_t * events, size_t event_sz) noexcept{
                
                event_t * cur   = events;
                size_t rem_sz   = event_sz; 

                while (rem_sz != 0u){
                    size_t submitting_sz = std::min(rem_sz, this->base->capacity());
                    if constexpr(DEBUG_MODE_FLAG){
                        if (submitting_sz == 0u){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }
                    this->base->push(cur, submitting_sz);
                    std::advance(cur, submitting_sz);
                    rem_sz -= submitting_sz;
                }
            }
    };

    template <class event_t>
    struct DeliveryHandle{
        dg::network_std_container::vector<event_t> delivery_items;
        size_t delivery_thrhold;
        ConsumerInterface<event_t> * consumer;
    };

    //I don't want to use raw_ptr here - yet this is a neccessity for most of the use cases where ConsumerInterface scope is not the same as delivery_handle scope
    template <class event_t>
    auto delvrsrv_open_handle(ConsumerInterface<event_t> * consumer, size_t delivery_thrhold) noexcept -> std::expected<DeliveryHandle<event_t> *, exception_t>{ //I think it's always a good practice to open -> expected // this is for interface consistency - future refactoring

        if (delivery_thrhold == 0u){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }
        
        dg::network_std_container::vector<event_t> vec{}; //yeah - std does not have constructor taking in reservation sz
        vec.reserve(delivery_thrhold);
        return new DeliveryHandle<event_t>{std::move(vec), delivery_thrhold, consumer}; //TODO: global memory exhaustion -
    }

    template <class event_t>
    void delvrsrv_deliver(DeliveryHandle<event_t> * handle, event_t event) noexcept{

        handle->delivery_items.push_back(event);

        if (handle->delivery_items.size() == handle->delivery_thrhold){
            handle->consumer->push(handle->delivery_items.data(), handle->delivery_items.size());
            handle->delivery_items.clear();
        }
    }

    template <class event_t>
    auto delvrsrv_close_handle(DeliveryHandle<event_t> * handle) noexcept{

        handle = dg::network_genult::safe_ptr_access(handle);
        handle->consumer->push(handle->delivery_items.data(), handle->delivery_items.size());
        delete handle;
    }

    template <class event_t>
    auto delvrsrv_open_raiihandle(ConsumerInterface<event_t> * consumer, size_t delivery_thrhold) noexcept -> std::expected<std::unique_ptr<DeliveryHandle<event_t>, decltype(&delvrsrv_close_handle<event_t>)>, exception_t>{

        std::expected<DeliveryHandle<event_t> *, exception_t> handle = delvrsrv_open_handle(consumer, delivery_thrhold);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return {std::in_place_t{}, handle.value(), delvrsrv_close_handle<event_t>};
    }
}

#endif