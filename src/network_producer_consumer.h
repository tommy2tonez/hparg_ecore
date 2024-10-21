#ifndef __NETWORK_PRODUCER_CONSUMER_H__
#define __NETWORK_PRODUCER_CONSUMER_H__

#include <stdint.h>
#include <stddef.h>
#include <array>
#include <memory>
#include <atomic> 
#include "network_concurrency_x.h"

namespace dg::network_producer_consumer{

    template <class EventType>
    struct ProducerInterface{
        using event_t = EventType;
        static_assert(std::is_trivial_v<event_t>);

        virtual ~ProducerInterface() noexcept = default;
        virtual void get(event_t * events, size_t& event_sz, size_t event_cap) noexcept = 0;
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
        virtual auto push(event_t * src, size_t src_sz) noexcept -> bool = 0;  
        virtual auto capacity() const noexcept -> size_t = 0;
    };

    template <class EventType>
    class LimitConsumerToConsumerWrapper: public virtual ConsumerInterface<EventType>{

        private:

            std::shared_ptr<LimitConsumerInterface<EventType>> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;

        public:

            LimitConsumerToConsumerWrapper(std::shared_ptr<LimitConsumerInterface<EventType>> base,
                                           std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) noexcept: base(std::move(base)),
                                                                                                                                      executor(std::move(executor)){}

            void push(event_t * events, size_t event_sz) noexcept{
                
                event_t * cur   = events;
                size_t rem_sz   = event_sz; 

                while (rem_sz != 0u){
                    size_t submitting_sz = dg::network_genult::safe_posint_access(std::min(rem_sz, this->base->capacity()));
                    dg::network_concurrency_infretry_x::ExecutableWrapper exe([&]() noexcept{return this->base->push(cur, submitting_sz)});
                    this->executor->exec(exe);
                    std::advance(cur, submitting_sz);
                    rem_sz -= submitting_sz;
                }
            }
    };

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

        const size_t MIN_DELIVERABLE_CAP = size_t{1};
        const size_t MAX_DELIVERABLE_CAP = size_t{1} << 20;

        if (!consumer){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(deliverable_cap, MIN_DELIVERABLE_CAP, MAX_DELIVERABLE_CAP) != deliverable_cap){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return new DeliveryHandle<event_t>{std::make_unique<event_t[]>(deliverable_cap), 0u, deliverable_cap, consumer}; //TODO: global memory exhaustion - internalize new
    }

    template <class event_t>
    void delvrsrv_deliver(DeliveryHandle<event_t> * handle, event_t event) noexcept{

        handle = dg::network_genult::safe_ptr_access(handle); 

        if (handle->deliverable_sz == handle->deliverable_cap){
            handle->consumer->push(handle->deliverable_arr.get(), handle->deliverable_sz);
            handle->deliverable_sz = 0u;
        }

        handle->deliverable_arr[handle->deliverable_sz++] = std::move(event);
    }

    template <class event_t>
    auto delvrsrv_close_handle(DeliveryHandle<event_t> * handle) noexcept{

        handle = dg::network_genult::safe_ptr_access(handle);
        handle->consumer->push(handle->deliverable_arr.get(), handle->deliverable_sz);
        delete handle; //TODO: internalize delete
    }

    template <class event_t>
    auto delvrsrv_open_raiihandle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<std::unique_ptr<DeliveryHandle<event_t>, decltype(&delvrsrv_close_handle<event_t>)>, exception_t>{

        std::expected<DeliveryHandle<event_t> *, exception_t> handle = delvrsrv_open_handle(consumer, deliverable_cap);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<DeliveryHandle<event_t>, decltype(&delvrsrv_close_handle<event_t>)>(handle.value(), delvrsrv_close_handle<event_t>);
    }

    template <class EventType>
    struct WareHouseInterface: virtual ProducerInterface<EventType>,
                               virtual ConsumerInterface<EventType>{};

    template <class EventType>
    class LckWareHouse: public virtual WareHouseInterface<EventType>{

        private:

            dg::vector<EventType> vec;
            const size_t digest_cap;
            std::unique_ptr<std::mutex> mtx;

        public:

            LckWareHouse(dg::vector<EventType> vec,
                         size_t digest_cap,
                         std::unique_ptr<std::mutex> mtx) noexcept: vec(std::move(vec)),
                                                                    digest_cap(digest_cap),
                                                                    mtx(std::move(mtx)){}
         
            auto push(EventType * ingestible_arr, size_t sz) noexcept -> bool{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->digest_cap){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->vec.size() + sz > this->vec.capacity()){
                    return false;
                }

                this->vec.insert(this->vec.end(), ingestible_arr, ingestible_arr + sz);
                return true;
            }

            void get(EventType * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                dst_sz          = std::min(dst_cap, this->vec.size());
                size_t new_sz   = this->vec.size() - dst_sz;
                std::copy(this->vec.begin() + new_sz, this->vec.end(), dst);
                this->vec.resize(new_sz);
            }

            auto capacity() const noexcept -> size_t{

                return this->digest_cap;
            }
    };
}

namespace dg::network_raii_producer_consumer{
    
    template <class EventType>
    struct ProducerInterface{
        using event_t = EventType;
        static_assert(std::is_nothrow_destructible_v<event_t>);
        static_assert(std::is_nothrow_move_constructible_v<event_t>);
        static_assert(std::is_nothrow_move_assignable_v<event_t>);

        virtual ~ProducerInterface() noexcept = default;
        virtual auto get(size_t) noexcept -> dg::vector<event_t> = 0;
    };

    template <class EventType>
    struct ConsumerInterface{
        using event_t = EventType;
        static_assert(std::is_nothrow_destructible_v<event_t>);
        static_assert(std::is_nothrow_move_constructible_v<event_t>); 
        static_assert(std::is_nothrow_move_assignable_v<event_t>);

        virtual ~ConsumerInterface() noexcept = default;
        virtual void push(dg::vector<event_t>) noexcept = 0;
    };

    template <class EventType>
    struct LimitConsumerInterface{
        using event_t = EventType;
        static_assert(std::is_nothrow_destructible_v<event_t>);
        static_assert(std::is_nothrow_move_constructible_v<event_t>);
        static_assert(std::is_nothrow_move_assignable_v<event_t>);

        virtual ~LimitConsumerInterface() noexcept = default;
        virtual auto push(dg::vector<event_t>) noexcept -> dg::vector<event_t> = 0; //this is b way to do this - better off with internalized meter
        virtual auto capcity() const noexcept -> size_t = 0; 
    };

    template <class EventType>
    class LimitConsumerToConsumerWrapper: public virtual ConsumerInterface<EventType>{

        private:

            std::shared_ptr<LimitConsumerInterface<EventType>> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;

        public:

            LimitConsumerToConsumerWrapper(std::shared_ptr<LimitConsumerInterface<EventType>> base,
                                           std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor) noexcept: base(std::move(base)),
                                                                                                                                      executor(std::move(executor)){}

            void push(dg::vector<EventType> vec) noexcept{

                while (!vec.empty()){ 
                    size_t extracting_sz = dg::network_genult::safe_posint_access(std::min(vec.size(), this->base->capacity()));
                    dg::vector<EventType> ingestible_vec = this->extract_back(vec, extracting_sz);
                    auto lambda = [&]() noexcept{
                        ingestible_vec = this->base->push(std::move(ingestible_vec));
                        return ingestible_vec.empty();
                    };
                    dg::network_concurrency_infretry_x::ExecutableWrapper exe(lambda); 
                    this->executor->exec(exe);
                }
            }
        
        private:

            auto extract_back(dg::vector<EventType>& vec, size_t extracting_sz) noexcept -> dg::vector<EventType>{
                
                if constexpr(DEBUG_MODE_FLAG){
                    if (extracting_sz > vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                dg::vector<EventType> rs{};
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

        dg::vector<event_t> deliverable_vec;
        size_t deliverable_cap;
        ConsumerInterface<event_t> * consumer;
    };

    template <class event_t>
    auto delvsrv_open_handle(ConsumerInterface<event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<DeliveryHandle<event_t> *, exception_t>{

        const size_t MIN_DELIVERABLE_CAP    = size_t{1};
        const size_t MAX_DELIVERABLE_CAP    = size_t{1} << 20;

        if (!consumer){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(deliverable_cap, MIN_DELIVERABLE_CAP, MAX_DELIVERABLE_CAP) != deliverable_cap){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        dg::vector<event_t> deliverable_vec{};
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

        return std::unique_ptr<DeliveryHandle<event_t>, decltype(&delvsrv_close_handle<event_t>)>{handle.value(), delvsrv_close_handle<event_t>};
    }

    template <class EventType>
    struct WareHouseInterface: virtual ProducerInterface<EventType>,
                               virtual LimitConsumerInterface<EventType>{};

    template <class EventType>
    class LckWareHouse: public virtual WareHouseInterface<EventType>{

        private:

            dg::vector<EventType> vec;
            const size_t ingest_cap;
            std::unique_ptr<std::mutex> mtx;

        public:

            LckWareHouse(dg::vector<EventType> vec, 
                         size_t ingest_cap,
                         std::unique_ptr<std::mutex> mtx) noexcept: vec(std::move(vec)),
                                                                    ingest_cap(ingest_cap),
                                                                    mtx(std::move(mtx)){}
            
            auto push(dg::vector<EventType> ingesting_vec) noexcept -> dg::vector<EventType>{

                if constexpr(DEBUG_MODE_FLAG){
                    if (ingesting_vec.size() > this->ingest_cap){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->vec.size() + ingesting_vec.size() > this->vec.capacity()){
                    return ingesting_vec;
                }

                for (auto& ingestible: ingesting_vec){
                    this->vec.push_back(std::move(ingestible));
                }

                return {};
            }

            auto get(size_t cap) noexcept -> dg::vector<event_t>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                dg::vector<EventType> rs{};

                for (size_t i = 0u; i < cap; ++i){
                    if (this->vec.empty()){
                        return rs;
                    } 

                    rs.push_back(std::move(this->vec.back()));
                    this->vec.pop_back();
                }

                return rs;            
            } 

            auto capacity() const noexcept -> size_t{

                return this->ingest_cap;
            }
    };

    template <size_t CONCURRENCY_SZ, class EventType> //deprecate next iteration
    class ConcurrentWarehouse: public virtual DropBoxInterface{

        private:

            dg::vector<std::unique_ptr<WareHouseInterface<EventType>>> warehouse_vec;
            const size_t warehouse_cap;

        public:

            ConcurrentWarehouse(dg::vector<std::unique_ptr<WareHouseInterface<EventType>>> warehouse_vec,
                                size_t warehouse_cap, 
                                std::integral_constant<size_t, CONCURRENCY_SZ>) noexcept: warehouse_vec(std::move(warehouse_vec)),
                                                                                          warehouse_cap(warehouse_cap){}

            auto push(dg::vector<EventType> vec) noexcept -> std::optional<dg::vector<EventType>>{

                size_t thr_idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{});
                return this->warehouse_vec[thr_id]->push(std::move(vec));
            }

            auto capacity() const noexcept -> size_t{

                return this->warehouse_cap;
            }

            auto get(size_t cap) noexcept -> dg::vector<EventType>{

                auto thr_idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{});
                return this->warehouse_vec[thr_idx]->get(cap);
            }
    };
}

#endif