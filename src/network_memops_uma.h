#ifndef __NETWORK_MEMOPS_UMA_H__
#define __NETWORK_MEMOPS_UMA_H__

#include "network_memlock.h"
#include "network_uma.h"
#include "network_memops.h"
#include "network_exception_handler.h"
#include <atomic>
#include "stdx.h"
#include "network_pointer.h"

namespace dg::network_memops_uma{

    struct signature_dg_network_memops_uma; 

    using uma_ptr_t         = dg::network_pointer::uma_ptr_t;
    using uma_lock_instance = dg::network_memlock_impl1::Lock<signature_dg_network_memops_uma, std::integral_constant<size_t, dg::network_pointer::MEMREGION_SZ>, uma_ptr_t>; 

    void init(uma_ptr_t first, uma_ptr_t last){

        stdx::memtransaction_guard grd;
        uma_lock_instance::init(first, last);
    }

    void deinit() noexcept{

        stdx::memtransaction_guard grd;
        uma_lock_instance::deinit();
    }

    template <class ...Args>
    class memlock_guard{

        private:

            decltype(dg::network_memlock::recursive_lock_guard_many(uma_lock_instance{}, std::declval<Args>()...)) resource;
        
        public:

            using self = memlock_guard;

            inline __attribute__((always_inline)) memlock_guard(Args ...args) noexcept{

                std::atomic_signal_fence(std::memory_order_seq_cst);
                this->resource = dg::network_memlock::recursive_lock_guard_many(uma_lock_instance{}, args...);
                std::atomic_signal_fence(std::memory_order_seq_cst);

                if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } else{
                    std::atomic_thread_fence(std::memory_order_acquire);
                }
            }

            inline __attribute__((always_inline)) ~memlock_guard() noexcept{

                if constexpr(STRONG_MEMORY_ORDERING_FLAG){
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } else{
                    std::atomic_thread_fence(std::memory_order_release);
                }

                std::atomic_signal_fence(std::memory_order_seq_cst);
                this->resource = std::nullopt;
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }

            memlock_guard(const self&) = delete;
            memlock_guard(self&&) = delete;

            memlock_guard& operator =(const self&) = delete;
            memlock_guard& operator =(self&&) = delete;
    };

    auto memcpy_uma_to_vma(vma_ptr_t dst, uma_ptr_t src, size_t n) noexcept -> exception_t{

        auto src_map_rs = dg::network_uma::map_wait_safe(src);

        if (!src_map_rs.has_value()){
            return src_map_rs.error();
        }

        vma_ptr_t src_vptr  = dg::network_uma::get_vma_const_ptr(src_map_rs.value().value());        
        return dg::network_memops_virt::memcpy(dst, src_vptr, n);
    }

    void memcpy_uma_to_vma_nothrow(vma_ptr_t dst, uma_ptr_t src, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_uma_to_vma(dst, src, n));
    }

    auto memcpy_vma_to_uma(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept -> exception_t{

        auto dst_map_rs = dg::network_uma::map_wait_safe(dst);

        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        vma_ptr_t dst_vptr  = dg::network_uma::get_vma_ptr(dst_map_rs.value().value());
        return dg::network_memops_virt::memcpy(dst_vptr, src, n);
    }

    void memcpy_vma_to_uma_nothrow(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memcpy_vma_to_uma(dst, src, n));
    }

    //this function does not guarantee atomicity - such that a failed operation leave the writing uma_ptr_t in an undefined state 
    auto memcpy_vma_to_uma_directall(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept -> exception_t{

        std::expected<size_t, exception_t> device_range = dg::network_uma::device_count(dst);

        if (!device_range.has_value()){
            return device_range.error();
        } 

        for (size_t i = 0u; i < device_range.value(); ++i){
            std::expected<device_id_t, exception_t> device_id = dg::network_uma::device_at(dst, i);

            if (!device_id.has_value()){
                return device_id.error();
            }

            std::expected<vma_ptr_t, exception_t> dst_vptr = dg::network_uma::map_direct(device_id.value(), dst);

            if (!dst_vptr.has_value()){
                return dst_vptr.error();
            }

            exception_t memcpy_err = dg::network_memops_virt::memcpy(dst_vptr.value(), src, n);

            if (dg::network_exception::is_failed(memcpy_err)){
                return memcpy_err;
            }
        }

        return dg::network_exception::SUCCESS;
    }

    void memcpy_vma_to_uma_directall_nothrow(uma_ptr_t dst, vma_ptr_t src, size_t n) noexcept{

        size_t device_range = dg::network_uma::device_count_nothrow(dst);

        for (size_t i = 0u; i < device_range; ++i){
            device_id_t id      = dg::network_uma::device_at_nothrow(dst, i);
            vma_ptr_t dst_vptr  = dg::network_uma::map_direct_nothrow(id, dst);
            dg::network_memops_virt::memcpy_nothrow(dst_vptr, src, n);
        }
    } 

    auto memset(uma_ptr_t dst, int c, size_t n) noexcept -> exception_t{

        auto dst_map_rs = dg::network_uma::map_wait_safe(dst);
        
        if (!dst_map_rs.has_value()){
            return dst_map_rs.error();
        }

        vma_ptr_t dst_vptr  = dg::network_uma::get_vma_ptr(dst_map_rs.value().value());
        return dg::network_memops_virt::memset(dst_vptr, c, n);
    }

    void memset_nothrow(uma_ptr_t dst, int c, size_t n) noexcept{

        dg::network_exception_handler::nothrow_log(memset(dst, c, n));
    }

    //this function does not guarantee atomicity - such that a failed operation leave the writing uma_ptr_t in an undefined state 
    auto memset_directall(uma_ptr_t dst, int c, size_t n) noexcept -> exception_t{
        
        std::expected<size_t, exception_t> device_range = dg::network_uma::device_count(dst);

        if (!device_range.has_value()){
            return device_range.error();
        }

        for (size_t i = 0u; i < device_range.value(); ++i){
            std::expected<device_id_t, exception_t> device_id = dg::network_uma::device_at(dst, i);

            if (!device_id.has_value()){
                return device_id.error();
            }

            std::expected<vma_ptr_t, exception_t> dst_vptr = dg::network_uma::map_direct(device_id.value(), dst);

            if (!dst_vptr.has_value()){
                return dst_vptr.error();
            } 

            exception_t memset_err = dg::network_memops_virt::memset(dst_vptr.value(), c, n);

            if (dg::network_exception::is_failed(memset_err)){
                return memset_err;
            }
        }

        return dg::network_exception::SUCCESS;
    }

    void memset_directall_nothrow(uma_ptr_t dst, int c, size_t n) noexcept{

        size_t device_range = dg::network_uma::device_count_nothrow(dst);

        for (size_t i = 0u; i < device_range; ++i){
            device_id_t id      = dg::network_uma::device_at_nothrow(dst, i);
            vma_ptr_t dst_vptr  = dg::network_uma::map_direct_nothrow(id, dst);
            dg::network_memops_virt::memset_nothrow(dst_vptr, c, n);
        }
    }

    //we dont really have time to do fancy stuff fellas
    //we are very glad that this actually works at all and debuggable
    //

    template <class key_t, class event_t>
    struct RegionRetryEntry{
        key_t key;
        event_t * event_ptr;
        size_t event_ptr_sz;
    };

    template <class key_t, class event_t>
    struct RegionKVDeliveryHandle{
        event_t * callback_event_container; //we'll fix this later
        size_t callback_event_container_sz;
        size_t callback_event_container_cap;

        RegionRetryEntry<key_t, event_t> * retry_entry_arr;

        size_t retry_entry_arr_sz;
        size_t retry_entry_arr_cap;

        bool was_callbacked;

        dg::network_producer_consumer::KVConsumerInterface<key_t, event_t> * event_container_callback_handler;
        dg::network_producer_consumer::KVConsumerInterface<key_t, event_t> * consumer_callback_handler;
        dg::network_producer_consumer::KVDeliveryHandle<key_t, event_t> * base;
    };

    template <class T, class = void>
    struct is_region_reflectible: std::false_type{};

    template <class T>
    struct is_region_reflectible<T, std::void_t<decltype(std::declval<const T&>().region_reflect([](...){}))>>: std::true_type{};

    template <class T>
    struct is_uma_ptr: std::false_type{};

    template <>
    struct is_uma_ptr<uma_ptr_t>: std::true_type{};

    template <class T>
    static inline constexpr bool is_region_reflectible_v                    = is_region_reflectible<T>::value;

    template <class T>
    static inline constexpr bool is_uma_ptr_v                               = is_uma_ptr<T>::value;

    template <class T>
    static inline constexpr bool is_met_region_reflection_requirements_v    = is_region_reflectible_v<T> || is_uma_ptr_v<T>;//

    template <class T>
    static inline constexpr bool is_met_regionkv_key_requirements_v         = std::is_trivial_v<T> && is_met_region_reflection_requirements_v<T>; 

    template <class T>
    static inline constexpr bool is_met_regionkv_value_requirements_v       = std::is_nothrow_destructible_v<T> && std::is_nothrow_default_constructible_v<T> && std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>;

    template <class T>
    static consteval auto region_reflection_size() -> size_t{

        size_t rs = {};
        T obj;

        obj.region_reflect([&]<class ...Args>(Args&& ...args){
            rs = sizeof...(Args);
        });

        return rs;
    }

    template <class T>
    static constexpr auto reflect_region(const T& obj) noexcept{

        using decay_t_t = std::decay_t<const T&>; 

        static_assert(is_met_region_reflection_requirements_v<decay_t_t>);

        if constexpr(is_uma_ptr_v<decay_t_t>){
            return std::array<uma_ptr_t, 1u>{obj};
        } else if constexpr(is_region_reflectible_v<decay_t_t>){
            std::array<uma_ptr_t, region_reflection_size<T>()> rs;

            auto reflector = [&]<class ...Args>(Args&& ...args){
                static_assert(std::conjunction_v<std::is_nothrow_constructible<uma_ptr_t, Args&&>...>);
                rs = std::array<uma_ptr_t, sizeof...(Args)>{std::forward<Args>(args)...};
            };

            return rs;
        } else{
            static_assert(FALSE_VAL<>);
        }
    } 

    template <class key_t, class event_t>
    class RegionKVDeliveryHandleInternalCallBackHandler: public virtual dg::network_producer_consumer::KVConsumerInterface<key_t, event_t>{

        private:

            RegionKVDeliveryHandle<key_t, event_t> * base; //the interface of the event_container_callback_handler is actually the magic to allow this

        public:

            RegionKVDeliveryHandleInternalCallBackHandler() = default;

            void set_base(RegionKVDeliveryHandle<key_t, event_t> * base) noexcept{

                this->base = base;
            }

            auto get_base() const noexcept -> RegionKVDeliveryHandle<key_t, event_t> *{

                return this->base;
            }

            void push(const key_t& key, std::move_iterator<event_t *> event_arr, size_t event_arr_sz) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->base->callback_event_container_sz + event_arr_sz > this->base->callback_event_container_cap){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }

                    if (this->base->retry_entry_arr_sz == this->base->retry_entry_arr_cap){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }

                    if (this->base == nullptr){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                event_t * load_addr = std::next(this->base->callback_event_container, this->base->callback_event_container_sz); 
                std::copy(event_arr, std::next(event_arr, event_arr_sz), load_addr);
                this->base->retry_entry_arr[this->base->retry_entry_arr_sz] = RegionRetryEntry<key_t, event_t>{.key           = key,
                                                                                                               .event_ptr     = load_addr,
                                                                                                               .event_ptr_sz  = event_arr_sz,
                                                                                                               .retry_count   = 0u};

                this->base->callback_event_container_sz += event_arr_sz;
                this->base->retry_entry_arr_sz          += 1u;
                this->base->was_callbacked              = true;
            }
    };

    template <class key_t, class event_t>
    auto delvrsrv_regionkv_open_handle(dg::network_producer_consumer::KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<RegionKVDeliveryHandle<key_t, event_t> *, exception_t>{

        static_assert(is_met_regionkv_key_requirements_v<key_t>);
        static_assert(is_met_regionkv_value_requirements_v<event_t>);

        if (consumer == nullptr){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        //we need to buff the deliverable_cap -> 1u, we'll figure the way

        if (deliverable_cap == 0u){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        std::unique_ptr<event_t[]> event_container = {};

        try{
            event_container = std::make_unique<event_t[]>(deliverable_cap);
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }

        std::unique_ptr<RegionRetryEntry<key_t, event_t>[]> retry_entry_arr = {};

        try {
            retry_entry_arr = std::make_unique<RegionRetryEntry<key_t, event_t>[]>(deliverable_cap);
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }

        std::unique_ptr<RegionKVDeliveryHandleInternalCallBackHandler<key_t, event_t>> internal_callback_handler = {};

        try{
            internal_callback_handler = std::make_unique<RegionKVDeliveryHandleInternalCallBackHandler<key_t, event_t>>();
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }

        std::unique_ptr<dg::network_producer_consumer::KVDeliveryHandle<key_t, event_t>, &dg::network_producer_consumer::delvrsrv_kv_close_handle> internal_delivery_handle = {};
        std::expected<dg::network_producer_consumer::KVDeliveryHandle<key_t, event_t> *, exception_t> raw_internal_delivery_handle = dg::network_producer_consumer::delvrsrv_kv_open_handle(internal_callback_handler.get(), deliverable_cap);

        if (!raw_internal_delivery_handle.has_value()){
            return std::unexpected(raw_internal_delivery_handle.error());
        }

        internal_delivery_handle = std::unique_ptr<dg::network_producer_consumer::KVDeliveryHandle<key_t, event_t>, &dg::network_producer_consumer::delvrsrv_kv_close_handle>(raw_internal_delivery_handle.get(), dg::network_producer_consumer::delvrsrv_kv_close_handle);

        std::unique_ptr<RegionKVDeliveryHandle<key_t, event_t>> rs = {};

        try{
            rs = std::make_unique<RegionKVDeliveryHandle<key_t, event_t>>();
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }

        internal_callback_handler->set_base(rs.get());

        *rs = RegionKVDeliveryHandle{.callback_event_container          = event_container.get(),
                                     .callback_event_container_sz       = 0u,
                                     .callback_event_container_cap      = deliverable_cap,
                                     .retry_entry_arr                   = retry_entry_arr.get(),
                                     .retry_entry_arr_sz                = 0u,
                                     .retry_entry_arr_cap               = deliverable_cap,
                                     .was_callbacked                    = false,
                                     .event_container_callback_handler  = internal_callback_handler.get(),
                                     .consumer_callback_handler         = consumer,
                                     .base                              = internal_delivery_handle.get()}; 

        RegionKVDeliveryHandle<key_t, event_t> * rs_ptr = rs.get();

        event_container.release();
        retry_entry_arr.release();
        internal_callback_handler.release();
        internal_delivery_handle.release();        
        rs.release();

        return rs_ptr;
    }

    template <class key_t, class event_t>
    auto delvrsrv_regionkv_open_preallocated_handle(dg::network_producer_consumer::KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap, char * inplace_mem) noexcept -> std::expected<RegionKVDeliveryHandle<key_t, event_t> *, exception_t>{

        static_assert(is_met_regionkv_key_requirements_v<key_t>);
        static_assert(is_met_regionkv_value_requirements_v<event_t>);

        if (consumer == nullptr){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        //we need to buff the deliverable_cap -> 1u, we'll figure the way

        if (deliverable_cap == 0u){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        char * current_memory_pointer   = inplace_mem;
        event_t * event_container       = {};

        try{
            current_memory_pointer  = stdx::align_ptr(current_memory_pointer, alignof(event_t));
            event_container         = new (current_memory_pointer) event_t[deliverable_cap];
            std::advance(current_memory_pointer, deliverable_cap * sizeof(event_t)); 
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }

        auto event_container_guard      = stdx::resource_guard([&]() noexcept{
            std::destroy(event_container, std::next(event_container, deliverable_cap));
        });

        RegionRetryEntry<key_t, event_t> * retry_entry_container = {};

        try{
            current_memory_pointer  = stdx::align_ptr(current_memory_pointer, alignof(RegionRetryEntry<key_t, event_t>));
            retry_entry_container   = new (current_memory_pointer) RegionRetryEntry<key_t, event_t>[deliverable_cap];
            std::advance(current_memory_pointer, deliverable_cap * sizeof(RegionRetryEntry<key_t, event_t>));
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }

        auto retry_entry_container_guard    = stdx::resource_guard([&]() noexcept{
            std::destroy(retry_entry_container, std::next(retry_entry_container, deliverable_cap));
        });

        RegionKVDeliveryHandleInternalCallBackHandler<key_t, event_t> * internal_callback_handler = {};
        
        try {
            current_memory_pointer      = stdx::align_ptr(current_memory_pointer, alignof(RegionKVDeliveryHandleInternalCallBackHandler<key_t, event_t>));
            internal_callback_handler   = new (current_memory_pointer) RegionKVDeliveryHandleInternalCallBackHandler<key_t, event_t>;
            std::advance(current_memory_pointer, sizeof(RegionKVDeliveryHandleInternalCallBackHandler<key_t, event_t>));
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }

        auto internal_callback_handler_guard    = stdx::resource_guard([&]() noexcept{
            std::destroy_at(internal_callback_handler);
        });

        std::expected<dg::network_producer_consumer::KVDeliveryHandle<key_t, event_t> *, exception_t> internal_delivery_handle = dg::network_producer_consumer::delvrsrv_kv_open_preallocated_handle(internal_callback_handler, deliverable_cap);

        if (!internal_delivery_handle.has_value()){
            return std::unexpected(internal_delivery_handle.error());
        }

        std::advance(current_memory_pointer, dg::network_producer_consumer::delvrsrv_kv_allocation_cost(internal_callback_handler, deliverable_cap));

        auto internal_delivery_handle_guard     = stdx::resource_guard([&]() noexcept{
            dg::network_producer_consumer::delvrsrv_kv_close_preallocated_handle(internal_delivery_handle.value());
        });

        RegionKVDeliveryHandle<key_t, event_t> * rs = {};

        try{
            current_memory_pointer  = stdx::align_ptr(current_memory_pointer, alignof(RegionKVDeliveryHandle<key_t, event_t>));
            rs                      = new (current_memory_pointer) RegionKVDeliveryHandle<key_t, event_t>;
            std::advance(current_memory_pointer, sizeof(RegionKVDeliveryHandle<key_t, event_t>));
        } catch (...){
            return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
        }

        auto rs_guard = stdx::resource_guard([&]() noexcept{
            std::destroy_at(rs);
        });

        event_container_guard.release();
        retry_entry_container_guard.release();
        internal_callback_handler_guard.release();
        internal_delivery_handle_guard.release();
        rs_guard.release();

        internal_callback_handler->set_base(rs);

        *rs = RegionKVDeliveryHandle{.callback_event_container          = event_container,
                                     .callback_event_container_sz       = 0u,
                                     .callback_event_container_cap      = deliverable_cap,
                                     .retry_entry_arr                   = retry_entry_container,
                                     .retry_entry_arr_sz                = 0u,
                                     .retry_entry_arr_cap               = deliverable_cap,
                                     .was_callbacked                    = false,
                                     .event_container_callback_handler  = internal_callback_handler,
                                     .consumer_callback_handler         = consumer,
                                     .base                              = internal_delivery_handle.value()};

        return rs;
    }

    template <class key_t, class event_t>
    constexpr auto delvrsrv_regionkv_allocation_cost(dg::network_producer_consumer::KVConsumerInterface<key_t, event_t> *, size_t deliverable_cap){

        return (alignof(event_t) + sizeof(event_t) * deliverable_cap) 
                + (alignof(RegionRetryEntry<key_t, event_t>) + sizeof(RegionRetryEntry<key_t, event_t>) * deliverable_cap) 
                + (alignof(RegionKVDeliveryHandleInternalCallBackHandler<key_t, event_t>) + sizeof(RegionKVDeliveryHandleInternalCallBackHandler<key_t, event_t>))
                + dg::network_producer_consumer::delvrsrv_kv_allocation_cost(std::add_pointer_t<dg::network_producer_consumer::KVConsumerInterface<key_t, event_t>>(nullptr), deliverable_cap)
                + (alignof(RegionKVDeliveryHandle<key_t, event_t>) * sizeof(RegionKVDeliveryHandle<key_t, event_t>));
    }

    template <class T>
    void nothrow_defaultize(T * arr, size_t arr_sz) noexcept{

        static_assert(std::is_nothrow_default_constructible_v<T>);
        static_assert(std::is_nothrow_move_assignable_v<T>);

        for (size_t i = 0u; i < arr_sz; ++i){
            arr[i] = T{};            
        }
    }

    template <class key_t, class event_t>
    __attribute__((noinline)) void delvrsrv_regionkv_internal_clear_callback(RegionKVDeliveryHandle<key_t, event_t> * handle) noexcept{

        handle = stdx::safe_ptr_access(handle); 

        if (!handle->was_callbacked){
            return;
        }

        size_t bad_trylock_sz = 0u; 

        for (size_t i = 0u; i < handle->retry_entry_arr_sz; ++i){
            auto reflected_region   = reflect_region(handle->retry_entry_arr[i].key);
            auto trylock_resource   = decltype(dg::network_memlock::recursive_trylock_guard_array(uma_lock_instance{}, reflected_region))();
            auto trylock_eventloop  = [&]() noexcept{
                trylock_resource = dg::network_memlock::recursive_trylock_guard_array(uma_lock_instance{}, reflected_region);
                return trylock_resource.has_value();
            };

            stdx::eventloop_expbackoff_spin(trylock_eventloop, stdx::SPINLOCK_SIZE_MAGIC_VALUE);

            if (!trylock_resource.has_value()){
                std::swap(handle->retry_entry_arr[bad_trylock_sz++], handle->retry_entry_arr[i]);
                continue;
            }

            stdx::memtransaction_guard memtx_guard;
            handle->consumer_callback_handler->push(handle->retry_entry_arr[i].key, std::make_move_iterator(handle->retry_entry_arr[i].event_ptr), handle->retry_entry_arr[i].event_ptr_sz);
            nothrow_defaultize(handle->retry_entry_arr[i].event_ptr, handle->retry_entry_arr[i].event_ptr_sz);
        }

        //the wait implementation is hard, we'll tackle the problem later, it involves a lot of stuff
        //the wait implementation could be implemented by (1): swapping the failed retry first, release all the successfully acquired regions
        //wait the front retry + acquire the other guys, rinse and repeat
        //

        for (size_t i = 0u; i < bad_trylock_sz; ++i){
            auto reflected_region   = reflect_region(handle->retry_entry_arr[i].key);
            auto lock_resource      = dg::network_memlock::recursive_lock_guard_array(uma_lock_instance{}, reflected_region);

            stdx::memtransaction_guard memtx_guard;
            handle->consumer_callback_handler->push(handle->retry_entry_arr[i].key, std::make_move_iterator(handle->retry_entry_arr[i].event_ptr), handle->retry_entry_arr[i].event_ptr_sz);
            nothrow_defaultize(handle->retry_entry_arr[i].event_ptr, handle->retry_entry_arr[i].event_ptr_sz);
        }

        handle->callback_event_container_sz = 0u;
        nothrow_defaultize(handle->retry_entry_arr, handle->retry_entry_arr_sz);
        handle->retry_entry_arr_sz          = 0u;
        handle->was_callbacked              = false;
    }

    template <class key_t, class event_t>
    inline void delvrsrv_regionkv_deliver(RegionKVDeliveryHandle<key_t, event_t> * handle, const key_t& key, event_t event) noexcept{

        handle = stdx::safe_ptr_access(handle);

        static_assert(std::is_nothrow_destructible_v<event_t>);
        static_assert(std::is_nothrow_move_constructible_v<event_t>);

        dg::network_producer_consumer::delvrsrv_kv_deliver(handle->base, key, std::move(event));

        if (handle->was_callbacked) [[unlikely]]{
            delvrsrv_regionkv_internal_clear_callback(handle);
        }
    }

    template <class key_t, class event_t>
    void delvrsrv_regionkv_clear(RegionKVDeliveryHandle<key_t, event_t> * handle) noexcept{

        handle = stdx::safe_ptr_access(handle);

        dg::network_producer_consumer::delvrsrv_kv_clear(handle);

        if (handle->was_callbacked){
            delvrsrv_regionkv_internal_clear_callback(handle);
        }
    }

    template <class key_t, class event_t>
    void delvrsrv_regionkv_close_handle(RegionKVDeliveryHandle<key_t, event_t> * handle) noexcept{

        delvrsrv_regionkv_clear(handle);

        static_assert(std::is_nothrow_destructible_v<event_t>); //not necessary
        static_assert(std::is_nothrow_destructible_v<RegionRetryEntry<key_t, event_t>>);

        delete[] handle->callback_event_container;
        delete[] handle->retry_entry_arr;
        delete handle->event_container_callback_handler; 

        dg::network_producer_consumer::delvrsrv_kv_close_handle(handle->base);

        delete handle;
    }

    template <class key_t, class event_t>
    void delvrsrv_regionkv_close_preallocated_handle(RegionKVDeliveryHandle<key_t, event_t> * handle) noexcept{

        delvrsrv_regionkv_clear(handle);

        static_assert(std::is_nothrow_destructible_v<event_t>); //not necessary
        static_assert(std::is_nothrow_destructible_v<RegionRetryEntry<key_t, event_t>>);

        std::destroy(handle->callback_event_container, std::next(handle->callback_event_container, handle->callback_event_container_cap));
        std::destroy(handle->retry_entry_arr, std::next(handle->retry_entry_arr, handle->retry_entry_arr_cap));
        std::destroy_at(handle->event_container_callback_handler);

        dg::network_producer_consumer::delvrsrv_kv_close_preallocated_handle(handle->base);

        std::destroy_at(handle);
    }

    template <class key_t, class event_t>
    auto delvrsrv_regionkv_open_raiihandle(dg::network_producer_consumer::KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<std::unique_ptr<RegionKVDeliveryHandle<key_t, event_t>, decltype(&delvrsrv_regionkv_close_handle)>, exception_t>{

        std::expected<RegionKVDeliveryHandle<key_t, event_t> *, exception_t> handle = delvrsrv_regionkv_open_handle(consumer, deliverable_cap);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<RegionKVDeliveryHandle<key_t, event_t>, decltype(&delvrsrv_regionkv_close_handle)>(handle.value(), delvrsrv_regionkv_close_handle);
    }

    template <class key_t, class event_t>
    auto delvrsrv_regionkv_open_preallocated_raiihandle(dg::network_producer_consumer::KVConsumerInterface<key_t, event_t> * consumer, size_t deliverable_cap) noexcept -> std::expected<std::unique_ptr<RegionKVDeliveryHandle<key_t, event_t>, decltype(&delvrsrv_regionkv_close_preallocated_handle)>, exception_t>{

        std::expected<RegionKVDeliveryHandle<key_t, event_t> *, exception_t> handle = delvrsrv_regionkv_open_preallocated_handle(consumer, deliverable_cap);

        if (!handle.has_value()){
            return std::unexpected(handle.error());
        }

        return std::unique_ptr<RegionKVDeliveryHandle<key_t, event_t>, decltype(&delvrsrv_regionkv_close_preallocated_handle)>(handle.value(), delvrsrv_regionkv_close_preallocated_handle);
    }
}

namespace dg::network_memops_umax{

    auto memcpy_uma_to_host(void * dst, uma_ptr_t src, size_t n) noexcept -> exception_t{

        vma_ptr_t dst_vptr = dg::network_virtual_device::virtualize_host_ptr(dst); 
        return dg::network_memops_uma::memcpy_uma_to_vma(dst_vptr, src, n);
    }

    void memcpy_uma_to_host_nothrow(void * dst, uma_ptr_t src, size_t n) noexcept{

        vma_ptr_t dst_vptr = dg::network_virtual_device::virtualize_host_ptr(dst); 
        dg::network_memops_uma::memcpy_uma_to_vma_nothrow(dst_vptr, src, n);
    }

    auto memcpy_host_to_uma(uma_ptr_t dst, void * src, size_t n) noexcept -> exception_t{ //remove constness for now - next iteration 

        vma_ptr_t src_vptr = dg::network_virtual_device::virtualize_host_ptr(src);
        return dg::network_memops_uma::memcpy_vma_to_uma(dst, src_vptr, n);
    }

    void memcpy_host_to_uma_nothrow(uma_ptr_t dst, void * src, size_t n) noexcept{

        vma_ptr_t src_vptr = dg::network_virtual_device::virtualize_host_ptr(src);
        dg::network_memops_uma::memcpy_vma_to_uma_nothrow(dst, src_vptr, n);
    }

    auto memcpy_host_to_uma_directall(uma_ptr_t dst, void * src, size_t n) noexcept -> exception_t{ //remove constness for now - next iteration

        vma_ptr_t src_vptr = dg::network_virtual_device::virtualize_host_ptr(src);
        return dg::network_memops_uma::memcpy_vma_to_uma_directall(dst, src_vptr, n);
    }

    void memcpy_host_to_uma_directall_nothrow(uma_ptr_t dst, void * src, size_t n) noexcept{

        vma_ptr_t src_vptr = dg::network_virtual_device::virtualize_host_ptr(src);
        dg::network_memops_uma::memcpy_vma_to_uma_directall_nothrow(dst, src_vptr, n);
    }
}

#endif