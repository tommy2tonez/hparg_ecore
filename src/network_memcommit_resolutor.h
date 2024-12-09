#ifndef __EVENT_DISPATCHER_H__
#define __EVENT_DISPATCHER_H__

#include <stdint.h>
#include <stddef.h>
#include <network_addr_lookup.h>
#include "network_tile_member_getsetter.h"
#include "network_memcommit_factory.h"
#include "network_producer_consumer.h"

#ifndef __NETWORK_TILEOPS_H__
#define __NETWORK_TILEOPS_H__

#include "network_tileops.h"
// #include "network_tileops_16_32.h"
// #include "network_tileops_32_16.h"
// #include "network_tileops_32_32.h"
// #include "network_memregion_lock.h"
#include "network_memory_utility.h"
#include "network_function_concurrent_buffer.h"
#include "network_tileops_poly.h"
#include "network_tile_member_getsetter.h" 
#include "network_memops_uma.h"
#include "network_vmamap.h"

namespace dg::network_memcommit_resolutor{

    //alright guys
    //goal of this week is to split 1024 concurrent memregion, each concurrent memregion has 1024 subregions for mmap, totaling 1 << 20 regions, each 1 << 23 in size (8MB) - so we are aiming at 8TBs of RAID SSDs - 128GBs of cuda - and 512 GBs of RAM - this is one node configuration
    //we want to dispatch the ping/pong + init by concurrent region radix - we offset the overhead of 1024 by using no-collision hashmap with replace + deliver on collision + open_addressing like map_variants - we'll see
    //gonna be lots of coding - we dont worry about the compiler for now
    //we need to actually consider cold-instruction-cache - which potentially reduces performance in critical paths - we'll increase delivery_size to offset the cost 

    //alrights guys - 5 flops ping_pong + init + orphan/ tile today - we want to reduce the tile size -> 1024 - because it's a necessity - otherwise too big uacm would not extract any meaningful result   
    //let's do this
    //gonna be bumpy road - we need to change all the codes

    struct UnifiedMemoryIPRetrieverInterface{
        virtual ~UnifiedMemoryIPRetrieverInterface() noexcept = default;
        virtual auto ip(uma_ptr_t) noexcept -> Address = 0;
    };

    struct HostIPRetrieverInterface{
        virtual ~HostIPRetrieverInterface() noexcept = default;
        virtual auto ip() noexcept -> Address = 0;
    };

    struct ForeignTileAliasGetterInterface{
        virtual ~ForeignTileAliasGetterInterface() noexcept = default;
        virtual auto alias(uma_ptr_t) noexcept -> std::optional<uma_ptr_t> = 0; //reduce lock_collisions by using distributed hash_map
    };

    template <class T>
    struct Request{
        Address requestee;
        Address requestor;
        T content;
    };

    class ForwardPingLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        public:

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                (void) ptr_arr;
            }
    };

    class ForwardPingPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPairSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity,
                                           size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity),
                                                                              vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                {
                    //we need the scope here to hinder compiler reordering of stack unwinding

                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle->get();
                    auto vectorization_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz);

                    //we want to deliver key on collisions - to reduce hash_map size
                    //because lock acquisition is expensive - spin + memflush + do_transaction + memflush - we want to vectorize the dispatches that happen to share the same rcu_memregion enough to offset the cost of lock_acquistion + cache fetching - yet allow room for other tasks to lock_compete
                    //delvrsrv memory locality is another task we need to talk about - we want to reuse the allocations for maximum locality
                    //worst-case dispatch is one per bucket - and it should not be 5% more expensive than the no-vectorization approach - in order for that to happen - we need to devirtualize by adding ID to KVConsumerInterface<ID, uma_ptr_t, uma_ptr_t> or using crtp

                    if (!vectorization_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorization_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr      = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_region    = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorization_delivery_handle->get(), lck_region, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr = ptr_arr[i];
                        init_status_t init_status = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(ptr);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t left_descendant   = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(ptr);
                                uma_ptr_t right_descendant  = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(ptr);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant, ptr));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant, ptr));
                                dg::network_tile_member_getsetter::set_pair_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort(); 
                                    break;
                                } else{
                                    std::unreachable();
                                    break;
                                }
                        }
                    }
                }
            };
    };

    class ForwardPingUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity,
                                           size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity),
                                                                              vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle->get();
                    auto vectorization_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz);

                    if (!vectorization_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorization_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr      = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr      = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorization_delivery_handle->get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        ptr = ptr_arr[i];
                        init_status_t init_status = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(ptr);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                std::array<uma_ptr_t, UACM_ACM_SZ> descendant_arr = dg::network_tile_member_getsetter::get_uacm_descendant_nothrow(ptr);

                                for (size_t i = 0u; i < UACM_ACM_SZ; ++i){
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant_arr[i], ptr));
                                }

                                dg::network_tile_member_getsetter::set_uacm_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                    break;
                                } else{
                                    std::unreachable();
                                    break;
                                }
                        }
                    }
                }
            };
    };

    class ForwardPingPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity,
                                           size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity),
                                                                              vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle->get();
                    auto vectorization_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->delivery_capacity);

                    if (!vectorization_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorization_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorization_delivery_handle->get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{

                dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr = ptr_arr[i];
                        init_status_t init_status = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(ptr);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                std::array<uma_ptr_t, PACM_ACM_SZ> left_descendant_arr  = dg::network_tile_member_getsetter::get_pacm_left_descendant_nothrow(ptr);
                                std::array<uma_ptr_t, PACM_ACM_SZ> right_descendant_arr = dg::network_tile_member_getsetter::get_pacm_right_descendant_nothrow(ptr);

                                for (size_t i = 0u; i < PACM_ACM_SZ; ++i){
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant_arr[i], ptr));
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant_arr[i], ptr));
                                }

                                dg::network_tile_member_getsetter::set_pacm_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                    break;
                                } else{
                                    std::unreachable();
                                    break;
                                }
                        }    
                    }
                }
            };
    };

    class ForwardPingExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingExtnSrcSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacity,
                                              size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                 delivery_capacity(delivery_capacity),
                                                                                 vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle->get();
                    auto vectorization_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz);

                    if (!vectorization_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorization_delivery_handle->get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{
               
                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr = ptr_arr[i];
                        init_status_t init_status = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(ptr);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(ptr);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr));
                                dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                    break;
                                } else{
                                    std::unreachable();
                                    break;
                                }
                        }    
                    }
                }
            };
    };

    class ForwardPingExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            DstExternalForwardPingSignalResolutor(std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                  std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                  std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                                  size_t delivery_capacity,
                                                  size_t vectorization_sz): uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                            host_ip_retriever(std::move(host_ip_retriever)),
                                                                            request_box(std::move(request_box)),
                                                                            delivery_capacity(delivery_capacity),
                                                                            vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.uma_ip_retriever          = this->uma_ip_retriever->get();
                    internal_resolver.host_ip_retriever         = this->host_ip_retriever->get();
                    internal_resolver.request_delivery_handle   = delivery_handle->get();
                    auto vectorization_delivery_handle          = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz);

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorization_delivery_handle->get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{
                
                UnifiedMemoryIPRetrieverInterface * uma_ip_retriever;
                HostIPRetrieverInterface * host_ip_retriever;
                dg::network_producer_consumer::DeliveryHandle<external_virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr = ptr_arr[i];
                        init_status_t init_status = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(ptr);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t counterpart   = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(ptr);
                                Address requestee_ip    = this->uma_ip_retriever->ip(counterpart);
                                Address requestor_ip    = this->host_ip_retriever->ip();
                                auto ping_request       = Request<external_virtual_memory_event_t>{requestee_ip, requestor_ip, dg::network_external_memcommit_factory::make_event_forward_ping_signal(counterpart)};

                                dg::network_producer_conumser::delvrsrv_deliver(this->request_delivery_handle, std::move(ping_request));
                                dg::network_tile_member_getsetter::set_extndst_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                    break;
                                } else{
                                    std::unreachable();
                                    break;
                                }
                        }
                    }
                }
            };
    };

    class ForwardPingCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingCritSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity,
                                           size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity),
                                                                              vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle->get();
                    auto vectorization_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz);

                    if (!vectorization_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorization_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorization_delivery_handle->get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                    
                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr = ptr_arr[i];
                        init_status_t init_status = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(ptr);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(ptr);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr));
                                dg::network_tile_member_getsetter::set_crit_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                    break;
                                } else{
                                    std::unreachable();
                                    break;
                                }
                        }
                    }
                }
            };
    };

    class ForwardPingMsgrFwdResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_mmeory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingMsgrFwdResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                        size_t delivery_capacity,
                                        size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                           delivery_capacity(delivery_capacity),
                                                                           vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle->get();
                    auto vectorization_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz);

                    if (!vectorization_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorization_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorization_delivery_handle->get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{
                
                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                    
                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr = ptr_arr[i];
                        init_status_t init_status = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(ptr);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(ptr);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr));
                                dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                    break;
                                } else{
                                    std::unreachable();
                                    break;
                                }
                        }
                    }
                }
            };
    };

    class ForwardPingMsgrBwdResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingMsgrBwdResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                        size_t delivery_capacity,
                                        size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                           delivery_capacity(delivery_capacity),
                                                                           vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle->get();
                    auto vectorization_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz);

                    if (!vectorization_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorization_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorization_delivery_handle->get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                    
                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr = ptr_arr[i];
                        init_status_t init_status = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(ptr);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(ptr);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr));
                                dg::network_tile_member_getsetter::set_msgrbwd_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                    break;
                                } else{
                                    std::unreachable();
                                    break;
                                }
                        }
                    }
                }
            };
    };

    class ForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;

        public:

            ForwardPingSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor,
                                       size_t leaf_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor,
                                       size_t pair_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor,
                                       size_t uacm_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor,
                                       size_t pacm_dispatch_sz,
                                       std::unique_Ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor,
                                       size_t extnsrc_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor,
                                       size_t extndst_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor,
                                       size_t crit_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor,
                                       size_t msgrfwd_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor,
                                       size_t msgrbwd_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
                                                                             leaf_dispatch_sz(leaf_dispatch_sz),
                                                                             pair_resolutor(std::move(pair_resolutor)),
                                                                             pair_dispatch_sz(pair_dispatch_sz),
                                                                             uacm_resolutor(std::move(uacm_resolutor)),
                                                                             uacm_dispatch_sz(uacm_dispatch_sz),
                                                                             pacm_resolutor(std::move(pacm_resolutor)),
                                                                             pacm_dispatch_sz(pacm_dispatch_sz),
                                                                             extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                             extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                             extndst_resolutor(std::move(extndst_resolutor)),
                                                                             extndst_dispatch_sz(extndst_dispatch_sz),
                                                                             crit_resolutor(std::move(crit_resolutor)),
                                                                             crit_dispatch_sz(crit_dispatch_sz),
                                                                             msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                             msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                             msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                             msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto leaf_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_resolutor.get(), this->leaf_dispatch_sz);
                auto pair_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz);
                auto uacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz);
                auto pacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz);
                auto extnsrc_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz);
                auto extndst_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz);
                auto crit_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz);
                auto msgrfwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz);
                auto msgrbwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz);

                if (!dg::network_exception::conjunc_expect_has_value(leaf_delivery_handle, pair_delivery_handle, uacm_delivery_handle,
                                                                     pacm_delivery_handle, extnsrc_delivery_handle, extndst_delivery_handle,
                                                                     crit_delivery_handle, msgrfwd_delivery_handle, msgrbwd_delivery_handle)){

                    dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(ptr_arr[i]); 

                    if (!tile_kind.has_value()){ //this branch is never taken - so we don't worry - this takes at most 2-3 CPU cycle
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(leaf_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PAIR:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pair_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_UACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(uacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNSRC:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extnsrc_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNDST:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extndst_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_CRIT:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(crit_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRFWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrfwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRBWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrbwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        default:
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                                break;
                            } else{
                                std::unreachable();
                                break;
                            }
                    }
                }
            }
    };

    //---

    class ForwardPongLeafRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPongLeafRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_leaf_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: //TODOs: add-user-designated logs, soft-err - no-warning for now
                        break; 
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongPairRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPongPairRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_pair_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongUACMRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongUACMRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_uacm_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongPACMRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongPACMRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_pacm_init_status_nohtorw(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_pacm_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongExtnSrcRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        public:

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                (void) ptr_arr; //no-ops + error_log
            }
    };

    class ForwardPongExtnDstRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPongExtnDstRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacty) noexcept: request_box(std::move(request_box)),
                                                                                 delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_extndst_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongCritRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongCritRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_crit_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {   
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongMsgrFwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPongMsgrFwdRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{
                
                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_msgrfwd_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongMsgrBwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongMsgrBwdRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_msgrbwd_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> pair_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> immu_resolutor;
            const size_t immu_dispatch_sz;

        public:

            ForwardPongRequestResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> leaf_resolutor,
                                        size_t leaf_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> pair_resolutor,
                                        size_t pair_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> uacm_resolutor,
                                        size_t uacm_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> pacm_resolutor,
                                        size_t pacm_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> extnsrc_resolutor,
                                        size_t extnsrc_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> extndst_resolutor,
                                        size_t extndst_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> crit_resolutor,
                                        size_t crit_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> msgrfwd_resolutor,
                                        size_t msgrfwd_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> msgrbwd_resolutor,
                                        size_t msgrbwd_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> immu_resolutor,
                                        size_t immu_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
                                                                           leaf_dispatch_sz(leaf_dispatch_sz),
                                                                           pair_resolutor(std::move(pair_resolutor)),
                                                                           pair_dispatch_sz(pair_dispatch_sz),
                                                                           uacm_resolutor(std::move(uacm_resolutor)),
                                                                           uacm_dispatch_sz(uacm_dispatch_sz),
                                                                           pacm_resolutor(std::move(pacm_resolutor)),
                                                                           pacm_dispatch_sz(pacm_dispatch_sz),
                                                                           extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                           extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                           extndst_resolutor(std::move(extndst_resolutor)),
                                                                           extndst_dispatch_sz(extndst_dispatch_sz),
                                                                           crit_resolutor(std::move(crit_resolutor)),
                                                                           crit_dispatch_sz(crit_dispatch_sz),
                                                                           msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                           msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                           msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                           msgrbwd_dispatch_sz(msgrbwd_dispatch_sz),
                                                                           immu_resolutor(std::move(immu_resolutor)),
                                                                           immu_dispatch_sz(immu_dispatch_sz){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto leaf_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_resolutor.get(), this->leaf_dispatch_sz);
                auto pair_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz);
                auto uacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz);
                auto pacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz);
                auto extnsrc_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz);
                auto extndst_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz);
                auto crit_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz);
                auto msgrfwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz);
                auto msgrbwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz);
                auto immu_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->immu_resolutor.get(), this->immu_dispatch_sz); 

                if (!dg::network_exception::conjunc_expect_has_value(leaf_delivery_handle, pair_delivery_handle, uacm_delivery_handle,
                                                                     pacm_delivery_handle, extnsrc_delivery_handle, extndst_delivery_handle,
                                                                     crit_delivery_handle, msgrfwd_delivery_handle, msgrbwd_delivery_handle,
                                                                     immu_delivery_handle)){
                    
                    dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(std::get<0>(ptr_arr[i]));

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(leaf_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PAIR:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pair_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_UACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(uacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNSRC:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extnsrc_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNDST:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extndst_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_CRIT:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(crit_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRFWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrfwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRBWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrbwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_IMMU:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(immu_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        default:
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                                break;
                            } else{
                                std::unreachable();
                                break;
                            }
                    }
                }
            }
    };

    //

    class ForwardPongLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

    };

    class ForwardPongMonoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongMonoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error())):
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(dst);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_do_signal(dst));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongPairSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(dst);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED: [[fallthrough]]
                        break;
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        pong_count_t pong_count = dg::network_tile_member_getsetter::get_pair_pong_count_nothrow(dst);
                        pong_count += 1u;

                        if (pong_count >= dg::network_tile_metadata::PAIR_DESCENDANT_SIZE){
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_do_signal(dst));
                        }

                        dg::network_tile_member_getsetter::set_pair_pong_count_nothrow(pong_count);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(dst);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY:
                    case TILE_INIT_STATUS_ORPHANED:
                    case TILE_INIT_STATUS_ADOPTED:
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        pong_count_t pong_count = dg::network_tile_member_getsetter::get_uacm_pong_count_nothrow(dst);
                        pong_count += 1u;

                        if (pong_count >= dg::network_tile_metadata::UACM_DESCENDANT_SIZE){
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_do_signal(dst));
                        }

                        dg::network_tile_member_getsetter::set_uacm_pong_count_nothrow(dst, pong_count);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(dst);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY:
                    case TILE_INIT_STATUS_ORPHANED:
                    case TILE_INIT_STATUS_ADOPTED:
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        pong_count_t pong_count = dg::network_tile_member_getsetter::get_pacm_pong_count_nothrow(dst);
                        pong_count += 1u;

                        if (pong_count >= dg::network_tile_metadata::PACM_DESCENDANT_SIZE){
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_do_signal(dst));
                        }

                        dg::network_tile_member_getsetter::set_pacm_pong_count_nothrow(dst, pong_count);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongExtnSrcSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(dst);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY:
                    case TILE_INIT_STATUS_ORPHANED:
                    case TILE_INIT_STATUS_ADOPTED:
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_do_signal(dst));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }

    };

    class ForwardPongExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPongExtnDstSignalResolutor()
    };

    class ForwardPongCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongCritSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(dst);
                
                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(dst);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY:
                    case TILE_INIT_STATUS_ORPHANED:
                    case TILE_INIT_STATUS_ADOPTED:
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_do_signal(dst));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPongMsgrFwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwadPongMsgrFwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                             size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                 delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(dst);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY:
                    case TILE_INIT_STATUS_ORPHANED:
                    case TILE_INIT_STATUS_ADOPTED:
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_do_signal(dst));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else[
                            std::unreachable();
                            break;
                        ]
                }
            }
    };

    class ForwardPongMsgrBwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongMsgrBwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(dst);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY:
                    case TILE_INIT_STATUS_ORPHANED:
                    case TILE_INIT_STATUS_ADOPTED:
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_do_signal(dst));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
        
    };

    //let's add constness because we are in a concurrent context

    class ForwardPongSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consuemr::ConsumerInterface<uma_ptr_t>> mono_resolutor;
            const size_t mono_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz; 
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> immu_resolutor;
            const size_t immu_dispatch_sz; 

        public:

            ForwardPongSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor,
                                       size_t leaf_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> mono_resolutor,
                                       size_t mono_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor,
                                       size_t pair_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor,
                                       size_t uacm_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor,
                                       size_t pacm_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor,
                                       size_t extnsrc_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor,
                                       size_t extndst_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor,
                                       size_t crit_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor,
                                       size_t msgrfwd_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor,
                                       size_t msgrbwd_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> immu_resolutor,
                                       size_t immu_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
                                                                          leaf_dispatch_sz(std::move(leaf_dispatch_sz)),
                                                                          mono_resolutor(std::move(mono_resolutor)),
                                                                          mono_dispatch_sz(std::move(mono_dispatch_sz)),
                                                                          pair_resolutor(std::move(pair_resolutor)),
                                                                          pair_dispatch_sz(std::move(pair_dispatch_sz)),
                                                                          uacm_resolutor(std::move(uacm_resolutor)),
                                                                          uacm_dispatch_sz(std::move(uacm_dispatch_sz)),
                                                                          pacm_resolutor(std::move(pacm_resolutor)),
                                                                          pacm_dispatch_sz(std::move(pacm_dispatch_sz)),
                                                                          extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                          extnsrc_dispatch_sz(std::move(extnsrc_dispatch_sz)),
                                                                          extndst_resolutor(std::move(extndst_resolutor)),
                                                                          extndst_dispatch_sz(std::move(extndst_dispatch_sz)),
                                                                          crit_resolutor(std::move(crit_resolutor)),
                                                                          crit_dispatch_sz(std::move(crit_dispatch_sz)),
                                                                          msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                          msgrfwd_dispatch_sz(std::move(msgrfwd_dispatch_sz)),
                                                                          msgrbwd_dispatch_sz(std::move(msgrbwd_dispatch_sz)),
                                                                          msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                          immu_resolutor(std::move(immu_resolutor)),
                                                                          immu_dispatch_sz(std::move(immu_dispatch_sz)){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{
                
                auto leaf_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_resolutor.get(), this->leaf_dispatch_sz);
                auto mono_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->mono_resolutor.get(), this->mono_dispatch_sz);
                auto pair_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz);
                auto uacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz);
                auto pacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz);
                auto extnsrc_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz);
                auto extndst_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz);
                auto crit_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz);
                auto msgrfwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz);
                auto msgrbwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz);
                auto immu_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->immu_resolutor.get(), this->immu_dispatch_sz);

                if (!dg::network_exception::conjunc_expect_has_value(leaf_delivery_handle, mono_delivery_handle, pair_delivery_handle,
                                                                     uacm_delivery_handle, pacm_delivery_handle, extnsrc_delivery_handle,
                                                                     extndst_delivery_handle, crit_delivery_handle, msgrfwd_delivery_handle,
                                                                     msgrbwd_delivery_handle, immu_delivery_handle)){
                    
                    dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(ptr_arr[i]);

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(leaf_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MONO:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(mono_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PAIR:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pair_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_UACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(uacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNSRC:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extnsrc_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNDST:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extndst_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_CRIT:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(crit_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRFWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrfwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRBWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrbwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_IMMU:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(immu_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        default:
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                                break;
                            } else[
                                std::unreachable();
                                break;
                            ]
                    }
                }
            }

    };

    //

    class ForwardPingPongLeafRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPingPongLeafRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                    delivery_capacity(delivery_capacity){}
            
            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_leaf_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        //leaf is not decayed - either empty or initialized - so this is an error
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }

            }
    };

    class ForwardPingPongPairRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPingPongPairRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                    delivery_capacity(delivery_capacity){}
            
            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_pair_push_observer(requestee, requestor); //I'll implement this later - placeholder
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t left_descendant   = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(ptr);
                        uma_ptr_t right_descendant  = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(ptr);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant, requestee));
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant, requestee));
                        dg::network_tile_member_getsetter::set_pair_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPingPongUACMRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
    
        public:

            ForwardPingPongUACMRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                    delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_uacm_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED: [[fallthrough]]
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        std::array<uma_ptr_t, UACM_ACM_SZ> descendant_arr = dg::network_tile_member_getsetter::get_uacm_descendant_nothrow(ptr); //alrights - I'll do the array_views

                        for (size_t i = 0u; i < UACM_ACM_SZ; ++i){
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant_arr[i], requestee));
                        }

                        dg::network_tile_member_getsetter::set_uacm_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPingPongPACMRequstResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPingPongPACMRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                    delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(requestee);

                switch (init_stauts){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_pacm_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        std::array<uma_ptr_t, PACM_ACM_SZ> left_descendant_arr  = dg::network_tile_member_getsetter::get_pacm_left_descendant_nothrow(requestee);
                        std::array<uam_ptr_t, PACM_ACM_SZ> right_descendant_arr = dg::network_tile_member_getsetter::get_pacm_right_descendant_nothrow(requestee);

                        for (size_t i = 0u; i < PACM_ACM_SZ; ++i){
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant_arr[i], requestee));
                            dg::network_producer_consumer::delvrrsv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant_arr[i], requestee));
                        }

                        dg::network_tile_member_getsetter::set_pacm_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPingPongExtnSrcRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            FowrardPingPongExtnSrcRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                   size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                       delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_extnsrc_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t descendant = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(requestee);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant));
                        dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPingPongExtnDstRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t request_delivery_capacity;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> outbound_request_box;
            const size_t outbound_delivery_capacity;

        public:

            ForwardPingPongExtnDstRequestResolutor(std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                   std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                   std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                   size_t request_delivery_capacity,
                                                   std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> outbound_request_box,
                                                   size_t outbound_delivery_capacity) noexcept: uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                                host_ip_retriever(std::move(host_ip_retriever)),
                                                                                                request_box(std::move(request_box)),
                                                                                                request_delivery_capacity(request_delivery_capacity),
                                                                                                outbound_request_box(std::move(outbound_request_box)),
                                                                                                outbound_delivery_capacity(outbound_delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle            = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->request_delivery_capacity);
                auto outbound_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->outbound_request_box.get(), this->outbound_delivery_capacity);s

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                if (!outbound_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(outbound_delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get(), outbound_delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, 
                         dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle,
                         dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * outbound_delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        dg::network_tile_member_getsetter::controller_extndst_push_observer(requestee, requestor);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t counterpart   = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(requestee);
                        Address requestee_ip    = this->uma_ip_retriever->ip(counterpart);
                        Address requestor_ip    = this->host_ip_retriever->ip();
                        auto ping_request       = Request<external_virtual_memory_event_t>{requestee_ip, requestor_ip, dg::network_external_memcommit_factory::make_event_forward_ping_signal(counterpart)};

                        dg::network_producer_consumer::delvrsrv_deliver(outbound_delivery_handle, std::move(ping_request));
                        dg::network_tile_member_getsetter::set_extndst_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPingPongCritRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPingPongCritRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                    delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                        dg::network_tile_member_getsetter::controller_crit_push_observer(requestee, requestor);
                        break;
                    case TILE_INIT_STATUS_INITIALIZED:
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED; [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t descendant = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(requestee);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant));
                        dg::network_tile_member_getsetter::set_crit_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPingPongMsgrFwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPingPongMsgrFwdRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                   size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                       delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                        dg::network_tile_member_getsetter::controller_msgrfwd_push_observer(requestee, requestor);
                        break;
                    case TILE_INIT_STATUS_INITIALIZED:
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestor));
                        break;
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(requestee);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant));
                        dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }

                }
            }
    };

    class ForwardPingPongMsgrBwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:    

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPingPongMsgrBwdRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                   size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                       delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(requestee);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED:
                        dg::network_tile_member_getsetter::controller_msgrbwd_push_observer(requestee, requestor);
                        break;
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(requestee));
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }

                switch (init_status){
                    case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                    case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                    case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                    case TILE_INIT_STATUS_INITIALIZED:
                        break;
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(requestee);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant));
                        dg::network_tile_member_getsetter::set_msgrbwd_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                        break;
                    }
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
    };

    class ForwardPingPongRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> leaf_pingpong_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> mono_pingpong_resolutor;
            const size_t mono_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> pair_pingpong_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> uacm_pingpong_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> pacm_pingpong_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> extnsrc_pingpong_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> extndst_pingpong_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> crit_pingpong_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> msgrfwd_pingpong_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> msgrbwd_pingpong_resolutor;
            const size_t msgrbwd_dispatch_sz;

        public:

            ForwardPingPongRequestResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> leaf_pingpong_resolutor,
                                            size_t leaf_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> mono_pingpong_resolutor,
                                            size_t mono_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> pair_pingpong_resolutor,
                                            size_t pair_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> uacm_pingpong_resolutor,
                                            size_t uacm_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> pacm_pingpong_resolutor,
                                            size_t pacm_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> extnsrc_pingpong_resolutor,
                                            size_t extnsrc_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> extndst_pingpong_resolutor,
                                            size_t extndst_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> crit_pingpong_resolutor,
                                            size_t crit_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> msgrfwd_pingpong_resolutor,
                                            size_t msgrfwd_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> msgrbwd_pingpong_resolutor,
                                            size_t msgrbwd_dispatch_sz) noexcept: leaf_pingpong_resolutor(std::move(leaf_pingpong_resolutor)),
                                                                                  leaf_dispatch_sz(leaf_dispatch_sz),
                                                                                  mono_pingpong_resolutor(std::move(mono_pingpong_resolutor)),
                                                                                  mono_dispatch_sz(mono_dispatch_sz),
                                                                                  pair_pingpong_resolutor(std::move(pair_pingpong_resolutor)),
                                                                                  pair_dispatch_sz(pair_dispatch_sz),
                                                                                  uacm_pingpong_resolutor(std::move(uacm_pingpong_resolutor)),
                                                                                  uacm_dispatch_sz(uacm_dispatch_sz),
                                                                                  pacm_pingpong_resolutor(std::move(pacm_pingpong_resolutor)),
                                                                                  pacm_dispatch_sz(pacm_dispatch_sz),
                                                                                  extnsrc_pingpong_resolutor(std::move(extnsrc_pingpong_resolutor)),
                                                                                  extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                                  extndst_pingpong_resolutor(std::move(extndst_pingpong_resolutor)),
                                                                                  extndst_dispatch_sz(extndst_dispatch_sz),
                                                                                  crit_pingpong_resolutor(std::move(crit_pingpong_resolutor)),
                                                                                  crit_dispatch_sz(crit_dispatch_sz),
                                                                                  msgrfwd_pingpong_resolutor(std::move(msgrfwd_pingpong_resolutor)),
                                                                                  msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                                  msgrbwd_pingpong_resolutor(std::move(msgrbwd_pingpong_resolutor)),
                                                                                  msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

        void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

            auto leaf_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_pingpong_resolutor.get(), this->leaf_dispatch_sz);
            auto mono_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->mono_pingpong_resolutor.get(), this->mono_dispatch_sz);
            auto pair_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_pingpong_resolutor.get(), this->pair_dispatch_sz);
            auto uacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_pingpong_resolutor.get(), this->uacm_dispatch_sz);
            auto pacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_pingpong_resolutor.get(), this->pacm_dispatch_sz);
            auto extnsrc_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_pingpong_resolutor.get(), this->extnsrc_dispatch_sz);
            auto extndst_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_pingpong_resolutor.get(), this->extndst_dispatch_sz);
            auto crit_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_pinpong_resolutor.get(), this->crit_dispatch_sz);
            auto msgrfwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_pingpong_resolutor.get(), this->msgrfwd_dispatch_sz);
            auto msgrbwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_pingpong_resolutor.get(), this->msgrbwd_dispatch_sz);

            if (!dg::network_exception::conjunc_expect_has_value(leaf_delivery_handle, mono_delivery_handle, pair_delivery_handle,
                                                                 uacm_delivery_handle, pacm_delivery_handle, extnsrc_delivery_handle,
                                                                 extndst_delivery_handle, crit_delivery_handle, msgrfwd_delivery_handle,
                                                                 msgrbwd_delivery_handle)){
                
                dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                return;
            }

            for (size_t i = 0u; i < sz; ++i){
                std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(std::get<0>(ptr_arr[i]));

                if (!tile_kind.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                    continue;
                }

                switch (tile_kind.value()){
                    case TILE_KIND_LEAF:
                        dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(leaf_delivery_handle)->get(), ptr_arr[i]);
                        break;
                    case TILE_KIND_MONO:
                        dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(mono_delivery_handle)->get(), ptr_arr[i]);
                        break;
                    case TILE_KIND_PAIR:
                        dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pair_delivery_handle)->get(), ptr_arr[i]);
                        break;
                    case TILE_KIND_UACM:
                        dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(uacm_delivery_handle)->get(), ptr_arr[i]);
                        break;
                    case TILE_KIND_PACM:
                        dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pacm_delivery_handle)->get(), ptr_arr[i]);
                        break;
                    case TILE_KIND_EXTNSRC:
                        dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extnsrc_delivery_handle)->get(), ptr_arr[i]);
                        break;
                    case TILE_KIND_EXTNDST:
                        dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extndst_delivery_handle)->get(), ptr_arr[i]);
                        break;
                    case TILE_KIND_CRIT:
                        dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(crit_delivery_handle)->get(), ptr_arr[i]);
                        break;
                    case TILE_KIND_MSGRFWD:
                        dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrfwd_delivery_handle)->get(), ptr_arr[i]);
                        break;
                    case TILE_KIND_MSGRBWD:
                        dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrbwd_delivery_handle)->get(), ptr_arr[i]);
                        break;
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                            break;
                        } else{
                            std::unreachable();
                            break;
                        }
                }
            }
        }
    };

    //

    class ForwardInitMonoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardInitMonoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            //alright guys - things are complicated - we want to see if init_status_t == DECAYED - we then want to see if src is initialized - then we forward - then we decay init_signal -> pong_signal
            //5 flops/ dispatch is prolly a dream - we tried our best to reduce as many polymorphism overhead as possible - I think we better vectorize uma_ptr_t * dispatch to reduce cuda synchronization overheads here - rather than using "array" approaches - this is a bad approach as we already talked about this being not a quantifiable thing
            //think of the vectorizations as delvsrv_open_raiihandle - the only diff is we stop when std::vector<std::tuple<void * __restrict__, const void * __restrict__, const void * __restrict__>> contract is broken

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_lck_addr  = get_mono_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {}; 

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr); 
                    src = get_mono_src_nothrow(dst);
                }

                uma_ptr_t src_lck_addr = get_tile_rcu_addr_nothrow(src); //access_err
                dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr, src_lck_addr); //we dont want to use try_lock because it's not a good practice here - so let's actually do twice lock_guards - lock_guard does mmeory flush + everything - which is good

                //we want to combine some of these guys to avoid too many cache reads - we'll do that after implementing this - we cant rely on uma_ptr_t * being adjecent to offset the costs

                uma_ptr_t new_src                                           = get_mono_src_nothrow(dst);
                init_status_t dst_init_status                               = get_mono_init_status_nothrow(dst);
                operatable_id_t dst_operatable_id                           = get_mono_operatable_id_nothrow(dst);
                std::array<uma_ptr_t, OBSERVER_ARRAY_CAP> dst_observer_arr  = get_mono_observer_array(dst);
                size_t dst_observer_arr_sz                                  = get_mono_observer_array_size(dst);
                uma_ptr_t dst_logit_umaptr                                  = get_mono_logit_addr_nothrow(dst);
                dispatch_control_t dispatch_control                         = get_mono_dispatch_control_nothrow(dst);
                init_status_t src_init_status                               = get_tile_init_status_nothrow(src);
                operatable_id_t src_operatable_id                           = get_tile_operatable_id_nothrow(src);
                uma_ptr_t src_logit_umaptr                                  = get_tile_logit_addr_nothrow(src);

                if (new_src != src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return;
                }

                auto [dst_vd_id, src_vd_id, dp_device, tileops_dp]  = dg::network_dispatch_control::decode_mono(dispatch_control);
                auto [dst_map_resource, src_map_resource]           = dg::network_uma::lockmap_safewait_many<2u>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}}); //weird - I mean this could combined with vmamap - yet I have yet wanted to complicate this further
                auto dst_logit_vmaptr                               = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto src_logit_vmaptr                               = dg::network_uma::get_vma_ptr(src_map_resource); 
                auto dst_logit_vmamap                               = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
                auto src_logit_vmamap                               = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);
                
                //no-ops on errors

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                    dg::network_tileops_cuda_poly::fwd_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), tileops_dp);
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                    dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), tileops_dp);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                set_mono_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                for (size_t i = 0u; i < dst_observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(dst_observer_arr[i]));
                }
            }
    };

    class ForwardInitPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardInitPairSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                using namespace dg::network_tile_member_getsetter;
                auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error())); //print statemnt needs to be more descriptive
                    return;
                } 

                uma_ptr_t dst_lck_addr  = get_pair_rcu_addr_nothrow(dst);
                uma_ptr_t lhs           = {};
                uma_ptr_t rhs           = {};

                //fine - refactor later
                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr);
                    lhs = get_pair_left_descendant_nothrow(dst);
                    rhs = get_pair_right_descendant_nothrow(dst);
                }

                uma_ptr_t lhs_lck_addr  = get_tile_rcu_addr_nothrow(lhs);
                uma_ptr_t rhs_lck_addr  = get_tile_rcu_addr_nothrow(rhs);
                dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr, lhs_lck_addr, rhs_lck_addr); //access err

                uma_ptr_t new_lhs                                       = get_pair_left_descendant_nothrow(dst);
                uma_ptr_t new_rhs                                       = get_pair_right_descendant_nothrow(dst);
                uma_ptr_t dst_logit_umaptr                              = get_pair_logit_addr_nothrow(dst);
                operatable_id_t dst_operatable_id                       = get_pair_operatable_id_nothrow(dst);
                init_status_t dst_init_status                           = get_pair_init_status_nothrow(dst);
                dispatch_control_t dispatch_control                     = get_pair_dispatch_control_nothrow(dst);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = get_pair_observer_arr_nothrow(dst);
                size_t observer_arr_sz                                  = get_pair_observer_arr_size_nothrow(dst);
                operatable_id_t lhs_operatable_id                       = get_tile_operatable_id_nothrow(lhs);
                init_status_t lhs_init_status                           = get_tile_init_status_nothrow(lhs);
                uma_ptr_t lhs_logit_umaptr                              = get_tile_logit_addr_nothrow(lhs);
                operatable_id_t rhs_operatable_id                       = get_tile_operatable_id_nothrow(rhs);
                init_status_t rhs_init_status                           = get_tile_init_status_nothrow(rhs);
                uma_ptr_t rhs_logit_umaptr                              = get_tile_logit_addr_nothrow(rhs);

                if (lhs != new_lhs){
                    return;
                }

                if (rhs != new_rhs){
                    return;
                }

                if (dst_operatable_id != lhs_operatable_id || lhs_operatable_id != rhs_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return;
                }

                if (lhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (rhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                auto [dst_vd_id, lhs_vd_id, rhs_vd_id, dp_device_kind, tileops_dp_id] = dg::network_dispatch_control::decode_pair(dispatch_control);

                auto [dst_map_resource, lhs_map_resource, rhs_map_resource] = dg::network_uma::lockmap_safewait_many<3u>({{dst_logit_umaptr, dst_vd_id}, {lhs_logit_umaptr, lhs_vd_id}, {rhs_logit_umaptr, rhs_vd_id}});
                auto dst_logit_vmaptr   = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto lhs_logit_vmaptr   = dg::network_uma::get_vma_ptr(lhs_map_resource);
                auto rhs_logit_vmaptr   = dg::network_uma::get_vma_ptr(rhs_map_resource); 
                auto dst_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
                auto lhs_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(lhs_logit_vmaptr);
                auto rhs_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(rhs_logit_vmaptr); 

                //dispatch errs 
                //no-ops on errors

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_cuda_poly::fwd_pair(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(lhs_logit_vmamap), dg::network_vmamap::get_cuda_ptr(rhs_logit_vmamap), tileops_dp_id);
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    dg::network_tileops_host_poly::fwd_pair(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(lhs_logit_vmamap), dg::network_vmamap::get_host_ptr(rhs_logit_vmamap), tileops_dp_id);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                for (size_t i = 0u; i < observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i], dst));
                }

                set_pair_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
            }
    };

    class ForwardInitUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardInitUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

            }
    };

    class ForwardInitPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerIntterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardInitPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{
                
            }
    };

    class ForwardInitExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<Request<external_virtual_memory_event_t>>>> request_box;
            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const size_t delivery_capacity;

        public:

            SrcExternalForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                                  std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                  std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                  size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                      uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                      host_ip_retriever(std::move(host_ip_retriever)),
                                                                                      delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_lck_addr = get_extnsrc_rcu_addr_nothrow(dst);
                uma_ptr_t src = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr);
                    src = get_extnsrc_descendant_nothrow(src);
                }

                uma_ptr_t src_lck_addr = get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_lck_addr, src_lck_addr);
             
                uma_ptr_t new_src                       = get_extnsrc_descendant_nothrow(dst);
                init_status_t dst_init_status           = get_extnsrc_init_status_nothrow(dst);
                operatable_id_t dst_operatable_id       = get_extnsrc_operatable_id_nothrow(dst);
                uma_ptr_t dst_logit_umaptr              = get_extnsrc_logit_addr_nothrow(dst); 
                dispatch_control_t dst_dispatch_control = get_extnsrc_dispatch_control_nothrow(dst);
                uma_ptr_t counterpart                   = get_extnsrc_counterpart_nothrow(dst);
                init_status_t src_init_status           = get_tile_init_status_nothrow(src);
                operatble_id_t src_operatable_id        = get_tile_operatable_id_nothrow(src);
                uma_ptr_t src_logit_umaptr              = get_tile_logit_addr_nothrow(src); 

                if (new_src != src){
                    return;
                }

                if (src_operatable_id != dst_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                auto [dst_vd_id, src_vd_id, dp_device_kind, tileops_dp_id]  = dg::network_dispatch_control::decode_extnsrc(dispatch_control);
                auto [dst_map_resource, src_map_resource]                   = dg::network_uma::lockmap_safewait_many<2u>({dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id});
                auto dst_logit_vmaptr   = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto src_logit_vmaptr   = dg::network_uma::get_vma_ptr(src_map_resource);
                auto dst_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
                auto src_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);

                //dispatch errs
                //no-ops on errors

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_cuda_poly::fwd_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), tileops_dp_id);
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), tileops_dp_id);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                set_extnsrc_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                dg::string serialized                           = serialize_extnsrc(dst);
                Address fr_addr                                 = this->host_ip_retriever->ip();
                Address to_addr                                 = this->uma_ip_retriever->ip(counterpart);
                external_virtual_memory_event_t inject_event    = dg::network_external_memcommit_factory::make_event_shadow_injection(dst, TILE_KIND_EXTNSRC, std::move(serialized));
                external_virtual_memory_event_t notify_event    = dg::network_external_memcommit_factory::make_event_forward_init_signal(counterpart);
                external_virtual_memory_event_t event           = dg::network_external_memcommit_factory::make_event_sequential(std::move(inject_event), std::move(notify_event));
                Request<external_virtual_memory_event_t> rq     = {to_addr, fr_addr, std::move(event)};

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(rq));
            }
    };

    class ForwardInitExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter;
            const size_t delivery_capacity;

        public:

            DstExternalForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                  std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter,
                                                  size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                      alias_getter(std::move(alias_getter)),
                                                                                      delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error())); //be more descriptive
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_extndst_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = get_extndst_descendant_nothrow(dst);
                }

                std::optional<uma_ptr_t> local_src = this->alias_getter->alias(src);

                if (!local_src.has_value()){
                    return;
                }

                uma_ptr_t local_src_rcu_addr                                = get_extnsrc_rcu_addr_nothrow(local_src.value()); //access_err
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, local_src_rcu_addr);
                uma_ptr_t new_src                                           = get_extndst_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id                           = get_extndst_operatable_id_nothrow(dst);
                init_status_t dst_init_status                               = get_extndst_init_status_nothrow(dst);
                dispatch_control_t dispatch_control                         = get_extndst_dispatch_control_nothrow(dst);
                uma_ptr_t dst_logit_umaptr                                  = get_extndst_logit_addr_nothrow(dst); 
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> dst_observer_arr    = get_extndst_observer_array_nothrow(dst);
                size_t dst_observer_arr_sz                                  = get_extndst_observer_array_size_nothrow(dst);
                operatable_id_t src_operatable_id                           = get_extnsrc_operatable_id_nothrow(local_src.value());
                init_status_t src_init_status                               = get_extnsrc_init_status_nothrow(local_src.value());
                uma_ptr_t src_logit_umaptr                                  = get_extnsrc_logit_addr_nothrow(local_src.value());

                if (new_src != src){
                    return; //no-ops no-error
                }

                if (dst_operatable_id != src_operatable_id){
                    return; //no-ops no-error
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return; //no-ops no-error
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return; //no-ops no-error
                }

                auto [dst_vd_id, src_vd_id, dp_device_kind, tileops_dp_code]    = dg::network_dispatch_control::decode_extndst(dispatch_control);
                auto [dst_map_resource, src_map_resource]                       = dg::network_uma::lockmap_safewait_many<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                auto dst_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto src_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(src_map_resource);
                auto dst_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr); //weird invention
                auto src_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);

                //dispatch device confirms pre dispatch - no-ops no-error

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_cuda_poly::fwd_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmammap::get_cuda_ptr(src_logit_vmamap), tileops_dp_code);
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmammap::get_host_ptr(src_logit_vmamap), tileops_dp_code);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                set_extndst_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                for (size_t i = 0u; i < dst_observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(dst_observer_arr[i]));
                }
            }
    };

    class ForwardInitCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            CritForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{
                
                auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error())); //be more descriptive
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_crit_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = get_crit_descendant_nothrow(dst);
                }

                uma_ptr_t src_rcu_addr                                  = get_tile_rcu_addr_nothrow(src);
                dg::network_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                                       = get_crit_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id                       = get_crit_operatable_id_nothrow(dst);
                init_status_t dst_init_status                           = get_crit_init_status_nothrow(dst);
                dispatch_control_t dispatch_control                     = get_crit_dispatch_control_nothrow(dst);
                std::array<uma_ptr_t, OBSERVER_ARRAY_CAP> observer_arr  = get_crit_observer_array_nothrow(dst);
                size_t observer_arr_sz                                  = get_crit_observer_array_size_nothrow(dst);
                uma_ptr_t dst_logit_umaptr                              = get_crit_logit_addr_nothrow(dst);
                uma_ptr_t dst_grad_umaptr                               = get_crit_grad_addr_nothrow(dst); 
                uma_ptr_t dst_clogit_umaptr                             = get_crit_clogit_addr_nothrow(dst);
                crit_kind_t crit_kind                                   = get_crit_crit_kind(dst);
                crit_ratio_t crit_ratio                                 = get_crit_crit_ratio(dst);
                init_status_t src_init_status                           = get_tile_init_status_nothrow(src);
                operatable_id_t src_operatable_id                       = get_tile_operatable_id_nothrow(src);
                uma_ptr_t src_logit_umaptr                              = get_tile_logit_addr_nothrow(src);
                uma_ptr_t src_grad_umaptr                               = get_tile_grad_addr_nothrow(src); //access compromise - no-ops on errors 

                if (new_src != src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return;
                }

                auto dispatch_info = dg::network_dispatch_control::decode_crit(dispatch_control);

                {
                    auto [dst_logit_map_resource, src_logit_map_resource] = dg::network_uma::lockmap_safewait_many<2>({{dst_logit_umaptr, dispatch_info.fwd_dst_logit_vd_id}, {src_logit_umaptr, dispatch_info.fwd_src_logit_vd_id}});
                    auto dst_logit_vmaptr   = dg::network_uma::get_vma_ptr(dst_logit_map_resource);
                    auto src_logit_vmaptr   = dg::network_uma::get_vma_ptr(src_logit_map_resource);
                    auto dst_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr); //
                    auto src_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr); //

                    if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.fwd_dp_device_kind)){
                        dg::network_tileops_cuda_poly::fwd_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), dispatch_info.fwd_tileops_dp_code);
                    } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.fwd_dp_device_kind)){
                        dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), dispathc_info.fwd_tileops_dp_code);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                }

                {
                    auto [dst_logit_map_resource, dst_clogit_map_resource, dst_grad_map_resource] = dg::network_uma::lockmap_safewait_many<3>({{dst_logit_umaptr, dispatch_info.crit_dst_logit_vd_id}, {dst_clogit_umaptr, dispatch_info.crit_dst_clogit_vd_id}, {dst_grad_umaptr, dispatch_info.crit_dst_grad_vd_id}});
                    auto dst_logit_vmaptr   = dg::network_uma::get_vma_ptr(dst_logit_map_resource);
                    auto dst_clogit_vmaptr  = dg::network_uma::get_vma_ptr(dst_clogit_map_resource);
                    auto dst_grad_vmaptr    = dg::network_uma::get_vma_ptr(dst_grad_map_resource);

                    if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.crit_dp_device_kind)){

                    } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.crit_dp_device_kind)){

                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                }

                {
                    auto [src_grad_map_resource, src_logit_map_resource, dst_grad_map_resource] = dg::network_uma::lockmap_safewait_many<3>({{src_grad_umaptr, dispatch_info.bwd_src_grad_vd_id}, {src_logit_umaptr, dispatch_info.bwd_src_logit_vd_id}, {dst_grad_umaptr, dispatch_info.bwd_dst_grad_vd_id}});
                    auto src_grad_vmaptr    = dg::network_uma::get_vma_ptr(src_grad_map_resource);
                    auto src_logit_vmaptr   = dg::network_uma::get_vma_ptr(src_logit_map_resource);
                    auto dst_grad_vmaptr    = dg::network_uma::get_vma_ptr(dst_grad_map_resource);
                    auto src_grad_vmamap    = dg::network_vmamap::mapsafe_nothrow(src_grad_vmaptr);
                    auto src_logit_vmamap   = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);
                    auto dst_grad_vmamap    = dg::network_vmamap::mapsafe_nothrow(dst_grad_vmaptr);

                    if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.bwd_dp_device_kind)){
                        dg::network_tileops_cuda_poly::bwd_mono(dg::network_vmamap::get_cuda_ptr(src_grad_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap), dispatch_info.bwd_tileops_dp_code);
                    } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.bwd_dp_device_kind)){
                        dg::network_tileops_host_poly::bwd_mono(dg::network_vmamap::get_host_ptr(src_grad_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), dg::network_vmamap::get_host_ptr(dst_grad_vmamap), dispatch_info.bwd_tileops_dp_code);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }

                }

                set_crit_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                for (size_t i = 0u; i < observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i]));
                }

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class ForwardInitMsgrFwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t request_box_delivery_capacity;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box;
            const size_t eu_packet_box_delivery_capacity;

        public:

            MsgrFwdForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserInterface>> eu_packet_box,
                                              size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                  eu_packet_box(std::move(eu_packet_box)),
                                                                                  delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto request_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);
                auto eu_packet_delivery_handle  = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->eu_packet_box.get(), this->eu_packet_box_delivery_capacity); 

                if (!request_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(request_delivery_handle.error()));
                    return;
                }

                if (!eu_packet_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(eu_packet_delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], request_delivery_handle->get(), eu_packet_delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, 
                         dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle,
                         dg::network_producer_consumer::DeliveryHandle<EndUserPacket> * eu_packet_delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_msgrfwd_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = get_msgrfwd_descendant_nothrow(dst);
                }

                uma_ptr_t src_rcu_addr                                      = get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                                           = get_msgrfwd_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id                           = get_msgrfwd_operatable_id_nothrow(dst);
                init_status_t dst_init_status                               = get_msgrfwd_init_status_nothrow(dst);
                dispatch_control_t dispatch_control                         = get_msgrfwd_dispatch_control_nothrow(dst);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> dst_observer_arr    = get_msgrfwd_observer_array_nothrow(dst);
                size_t dst_observer_arr_size                                = get_msgrfwd_observer_array_size_nothrow(dst);
                uma_ptr_t dst_logit_umaptr                                  = get_msgrfwd_logit_addr_nothrow(dst); 
                size_t msgr_retry_count                                     = get_msgrfwd_retry_count_nothrow(dst);
                dst_info_t msgr_dst                                         = get_msgrfwd_dst_info_nothrow(dst);
                logit_id_t msgr_logit_id                                    = get_msgrfwd_logit_id_nothrow(dst); 
                transmit_urgency_t msgr_transmit_urgency                    = get_msgrfwd_transmit_urgency_nothrow(dst);
                transmit_comm_t msgr_transmit_comm                          = get_msgrfwd_transmit_comm_nothrow(dst);
                operatable_id_t src_operatable_id                           = get_tile_operatable_id_nothrow(src);
                init_status_t src_init_status                               = get_tile_init_status_nothrow(src);
                uma_ptr_t src_logit_umaptr                                  = get_tile_logit_addr_nothrow(src); 

                if (new_src != src){
                    return; //no-ops no-err
                }

                if (dst_operatable_id != src_operatable_id){
                    return; //no-ops no-err
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return; //no-ops no-err
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return; //no-ops no-err
                }

                auto [dst_vd_id, src_vd_id, dp_device_kind, tileops_dp_code]    = dg::network_dispatch_control::decode_msgrfwd(dispatch_control);
                auto [dst_map_resource, src_map_resource]                       = dg::network_uma::lockmap_safewait_many<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                auto dst_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto src_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(src_map_resource);
                auto dst_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
                auto src_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);

                //no-ops on errors

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_cuda_poly::forward_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), tileops_dp_code);
                }  else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    dg::network_tileops_host_poly::forward_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), tileops_dp_code);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                set_msgrfwd_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                for (size_t i = 0u; i < dst_observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(dst_observer_arr[i]));
                }

                EndUserPacket eu_packet{};

                eu_packet.kind          = EUPACKET_MSGRFWD_TRANSMIT;
                eu_packet.content       = dg::network_compact_serializer::serialize<dg::string>(LogitData{logit_id, get_msgrfwd_logit_nothrow(dst)});
                eu_packet.dst           = msgr_dst;
                eu_packet.retry_count   = msgr_retry_count;
                eu_packet.urgency       = msgr_transmit_urgency;
                eu_packet.comm          = msgr_transmit_comm;

                dg::network_producer_consumer::delvrsrv_deliver(eu_packet_delivery_handle, std::move(eu_packet));
            }
    };

    class ForwardInitMsgrBwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardInitMsgrBwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_msgrbwd_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = get_msgrbwd_descendant_nothrow(dst);
                }

                uma_ptr_t src_rcu_addr                                      = get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                                           = get_msgrbwd_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id                           = get_msgrbwd_operatable_id_nothrow(dst);
                dispatch_control_t dispatch_control                         = get_msgrbwd_dispatch_control_nothrow(dst);
                init_status_t dst_init_status                               = get_msgrbwd_init_status_nothrow(dst);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> dst_observer_arr    = get_msgrbwd_observer_array_nothrow(dst);
                size_t dst_observer_arr_sz                                  = get_msgrbwd_observer_array_size_nothrow(dst);
                uma_ptr_t dst_logit_umaptr                                  = get_msgrbwd_loigt_addr_nothrow(dst);
                init_status_t src_init_status                               = get_tile_init_status_nothrow(src);
                operatable_id_t src_operatable_id                           = get_tile_operatable_id_nothrow(src);
                uma_ptr_t src_logit_umaptr                                  = get_tile_logit_addr_nothrow(src);

                if (new_src != src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                auto [dst_vd_id, src_vd_id, dp_device_kind, tileops_dp_code]    = dg::network_dispatch_control::decode_msgrbwd(dispatch_control);
                auto [dst_map_resource, src_map_resource]                       = dg::network_uma::lockmap_safewait_many<2>({{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                auto dst_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(dst_map_resource);
                auto src_logit_vmaptr                                           = dg::network_uma::get_vma_ptr(src_map_resource);
                auto dst_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(dst_logit_vmaptr);
                auto src_logit_vmamap                                           = dg::network_vmamap::mapsafe_nothrow(src_logit_vmaptr);

                if (dg::network_dispatch_control::is_cuda_dispatch(tileops_dp_code)){
                    dg::network_tileops_cuda_poly::forward_mono(dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap), dg::network_vmamap::get_cuda_ptr(src_logit_vmamap), tileops_dp_code);
                } else if (dg::network_dispatch_control::is_host_dispatch(tileops_dp_code)){
                    dg::network_tileops_host_poly::forward_mono(dg::network_vmamap::get_host_ptr(dst_logit_vmamap), dg::network_vmamap::get_host_ptr(src_logit_vmamap), tileops_dp_code);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                set_msgrbwd_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                for (size_t i = 0u; i < dst_observer_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_pong_signal(dst_observer_arr[i]));
                }
            }
    };

    class ForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> mono_resolutor;
            const size_t mono_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;


        public:

            ForwardInitSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> mono_resolutor,
                                       size_t mono_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor,
                                       size_t pair_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor,
                                       size_t uacm_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor,
                                       size_t pacm_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor,
                                       size_t extnsrc_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor,
                                       size_t extndst_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor,
                                       size_t crit_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor,
                                       size_t msgrfwd_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor,
                                       size_t msgrbwd_dispatch_sz): mono_resolutor(std::move(mono_resolutor)),
                                                                    mono_dispatch_sz(mono_dispatch_sz),
                                                                    pair_resolutor(std::move(pair_resolutor)),
                                                                    pair_dispatch_sz(pair_dispatch_sz),
                                                                    uacm_resolutor(std::move(uacm_resolutor)),
                                                                    uacm_dispatch_sz(uacm_dispatch_sz),
                                                                    pacm_resolutor(std::move(pacm_resolutor)),
                                                                    pacm_dispatch_sz(pacm_dispatch_sz),
                                                                    extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                    extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                    extndst_resolutor(std::move(extndst_resolutor)),
                                                                    extndst_dispatch_sz(extndst_dispatch_sz),
                                                                    crit_resolutor(std::move(crit_resolutor)),
                                                                    crit_dispatch_sz(crit_dispatch_sz),
                                                                    msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                    msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                    msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                    msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{
                
                auto mono_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->mono_resolutor.get(), this->mono_dispatch_sz);
                auto pair_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz);
                auto uacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz);
                auto pacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz);
                auto extnsrc_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz);
                auto extndst_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz);
                auto crit_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz);
                auto msgrfwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz);
                auto msgrbwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz); 

                //resource leak is expected here - we don't want to communicate the error because it would mess up the code very badly - let's assume that everything is "might", "maybe", and set a timeout for that - like network packet
                //users of the engines wait for msgrfwds to be intiialized within a certain windows - or deallocate and move on
                //error handlings in producer-consumer situation is not encouraged - and it is not a good practice also - this is an error that should be controlled in users' code flow

                if (!dg::network_exception::conjunc_expect_has_value(mono_delivery_handle, pair_delivery_handle, uacm_delivery_handle,
                                                                     pacm_delivery_handle, extnsrc_delivery_handle, extndst_delivery_handle,
                                                                     crit_delivery_handle, msgrfwd_delivery_handle, msgrbwd_delivery_handle)){

                    dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION)); //usually I would abort and throw kernel an exception - to restart the program but this is better to be a no-ops
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(ptr_arr[i]); //better to chk for error here - it is not expensive in terms of branching - because that branch is mostly not taken

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_MONO:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(mono_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PAIR:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pair_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_UACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(uacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNSRC:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extnsrc_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNDST:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extndst_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_CRIT:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(crit_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRFWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrfwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRBWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrbwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        default:
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                                break;
                            } else{
                                std::unreachable();
                                break;
                            }
                    };
                }
            }
    };

    //

    class BackwardDoLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoLeafSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr                  = get_leaf_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                init_status_t init_status               = get_leaf_init_status_nothrow(dst);
                grad_status_t grad_status               = get_leaf_grad_status_nothrow(dst);
                uma_ptr_t logit_umaptr                  = get_leaf_logit_addr_nothrow(dst);
                uma_ptr_t grad_umaptr                   = get_leaf_grad_addr_nothrow(dst);
                dispatch_control_t dispatch_control     = get_leaf_grad_dispatch_control_nothrow(dst);

                if (init_status != TILE_INIT_STATUS_INITIALIZED){
                    return; //no-ops no-err
                }

                if (grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }      
                
                auto [logit_vd_id, grad_vd_id, dp_device_kind, tileops_dp_id]   = dg::network_dispatch_control::decode_gradupdate_leaf(dispatch_control);
                auto [logit_map_resource, grad_map_resource]                    = dg::network_uma::lockmap_safewait_many_nothrow<2u>({{logit_umaptr, logit_vd_id}, {grad_umaptr, grad_vd_id}});
                auto logit_vmamap                                               = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(logit_map_resource));
                auto grad_vmamap                                                = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(grad_map_resource));

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_cuda_poly::grad_update_n_zero(dg::network_vmamap::get_cuda_ptr(logit_vmamap), dg::network_vmamap::get_cuda_ptr(grad_vmamap), tileops_dp_id);
                } else if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    dg::network_tileops_host_poly::grad_update_n_zero(dg::network_vmamap::get_host_ptr(logit_vmamap), dg::network_vmamap::get_host_ptr(grad_vmamap), tileops_dp_id);
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

            }
    };

    class BackwardDoMonoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            BackwardDoMonoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle){

                auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_mono_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = get_mono_descendant_nothrow(dst);
                }        

                uma_ptr_t src_rcu_addr  = get_tile_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(src_rcu_addr, dst_rcu_addr); 

                uma_ptr_t new_src                   = get_mono_descendant_nothrow(dst);
                init_status_t dst_init_status       = get_mono_init_status_nothrow(dst);
                grad_status_t dst_grad_status       = get_mono_grad_status_nothrow(dst);
                operatable_id_t dst_operatable_id   = get_mono_operatable_id_nothrow(dst); //operatable_id seems like an individual identifier - use operatable_group_id instead
                uma_ptr_t dst_grad_umaptr           = get_mono_grad_addr_nothrow(dst);
                dispatch_control_t dispatch_control = get_mono_backprop_dispatch_control_nothrow(dst);
                init_status_t src_init_status       = get_tile_init_status_nothrow(src); //ptr-access
                operatable_id_t src_operatable_id   = get_tile_operatable_id_nothrow(src);
                grad_status_t src_grad_status       = get_tile_grad_status_nothrow(src);
                uma_ptr_t src_grad_umaptr           = get_tile_grad_addr_nothrow(src);
                uma_ptr_t src_logit_umaptr          = get_tile_logit_addr_nothrow(src);

                if (src != new_src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                //refactor later
                auto [backwardee_grad_vd_id, backwardee_logit_vd_id, backwarder_grad_vd_id, dp_device_kind, tileops_dp_code]    = dg::network_dispatch_control::decode_bwd_mono(dispatch_control);
                auto [backwardee_grad_map_resource, backwardee_logit_map_resource, backwarder_grad_map_resouce]                 = dg::network_uma::lockmap_safewait_many_nothrow<3u>({{src_grad_umaptr, backwardee_grad_vd_id}, {src_logit_umaptr, backwardee_logit_vd_id}, {dst_grad_umaptr, backwarder_grad_vd_id}});

                auto backwardee_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_grad_map_resource));
                auto backwardee_logit_vmamap    = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_logit_map_resource));
                auto backwarder_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resouce));

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_cuda_poly::bwd_mono_zero_n_add(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_code);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_cuda_poly::bwd_mono_zero_n_assign(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_code);
                        set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_INITIALIZED);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_host_poly::bwd_mono_zero_n_add(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_code);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_host_poly::bwd_mono_zero_n_assign(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_code);
                        set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_INITIALIZED);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class BackwardDoPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            BackwardDoPairSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle){

                auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_pair_rcu_addr_nothrow(ptr);
                uma_ptr_t lhs           = {};
                uma_ptr_t rhs           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    lhs = get_pair_left_descendant_nothrow(dst);
                    rhs = get_pair_right_descendant_nothrow(dst);
                }

                uma_ptr_t lhs_rcu_addr              = get_tile_rcu_addr_nothrow(lhs);
                uma_ptr_t rhs_rcu_addr              = get_tile_rcu_addr_nothrow(rhs);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, lhs_rcu_addr, rhs_rcu_addr);

                uma_ptr_t new_lhs                   = get_pair_left_descendant_nothrow(dst);
                uma_ptr_t new_rhs                   = get_pair_right_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id   = get_pair_operatable_id_nothrow(dst);
                init_status_t dst_init_status       = get_pair_init_status_nothrow(dst);
                grad_status_t dst_grad_status       = get_pair_grad_status_nothrow(dst);
                dispatch_control_t dispatch_control = get_pair_bwd_dispatch_control_nothrow(dst);
                uma_ptr_t dst_grad_umaptr           = get_pair_grad_addr_nothrow(dst);

                operatable_id_t lhs_operatable_id   = get_tile_operatable_id_nothrow(lhs);
                init_status_t lhs_init_status       = get_tile_init_status_nothrow(lhs);
                grad_status_t lhs_grad_status       = get_tile_grad_status_nothrow(lhs);
                uma_ptr_t lhs_grad_umaptr           = get_tile_grad_addr_nothrow(lhs);
                uma_ptr_t lhs_logit_umaptr          = get_tile_logit_addr_nothrow(lhs);
                operatable_id_t rhs_operatable_id   = get_tile_operatable_id_nothrow(rhs);
                init_status_t rhs_init_status       = get_tile_init_status_nothrow(rhs);
                grad_status_t rhs_grad_status       = get_tile_grad_status_nothrow(rhs);
                uma_ptr_t rhs_grad_umaptr           = get_tile_grad_addr_nothrow(rhs);
                uma_ptr_t rhs_logit_umaptr          = get_tile_logit_addr_nothrow(rhs);

                if (lhs != new_lhs){
                    return;
                }

                if (rhs != new_rhs){
                    return;
                }

                if (lhs_operatable_id != rhs_operatable_id || lhs_operatable_id != dst_operatable_id){
                    return;
                }

                if (lhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (rhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                auto dispatch_info = dg::network_dispatch_control::decode_bwd_pair(dispatch_control);

                {
                    auto [lhs_grad_map_resource, lhs_logit_map_resource, rhs_logit_map_resource, backwarder_grad_map_resource] = dg::network_uma::lockmap_safewait_many_nothrow<4>({{lhs_grad_umaptr, dispatch_info.lhs_grad_vd_id}, {lhs_logit_umaptr, dispatch_info.lhs_logit_vd_id},
                                                                                                                                                                                    {rhs_logit_umaptr, dispatch_info.rhs_logit_vd_id}, {dst_grad_umaptr, dispatch_info.backwarder_grad_vd_id}});

                    auto lhs_grad_vmamap        = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(lhs_grad_map_resource));
                    auto lhs_logit_vmamap       = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(lhs_logit_map_resource));
                    auto rhs_logit_vmamap       = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(rhs_logit_map_resource));
                    auto backwarder_grad_vmamap = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resource));

                    if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.lhs_dp_device_kind)){
                        if (lhs_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                            dg::network_tileops_cuda_poly::bwd_pair_lhs_keep_n_add(dg::network_vmamap::get_cuda_ptr(lhs_grad_vmamap), dg::network_vmamap::get_cuda_ptr(lhs_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), dg::network_vmamap::get_cuda_ptr(rhs_logit_vmamap), dispatch_info.lhs_bwd_tileops_dp_code);
                        } else if (lhs_grad_status == TILE_GRAD_STATUS_EMPTY){
                            dg::network_tileops_cuda_poly::bwd_pair_lhs_keep_n_assign(dg::network_vmamap::get_cuda_ptr(lhs_grad_vmamap), dg::network_vmamap::get_cuda_ptr(lhs_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), dg::network_vmamap::get_cuda_ptr(rhs_logit_vmamap), dispatch_info.lhs_bwd_tileops_dp_code); //prolly zero_grad to avoid excessive code gen - zero_grad and bwd_pair is always accm - because zero_grad == prefetch - it's actually not very expensive - according to my knowledge about bignum, 2 iterations == 2x time - so... 
                            set_tile_grad_status_nothrow(lhs, TILE_GRAD_STATUS_INITIALIZED);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.lhs_dp_device_kind)){
                        if (lhs_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                            dg::network_tileops_host_poly::bwd_pair_lhs_keep_n_add(dg::network_vmamap::get_host_ptr(lhs_grad_vmamap), dg::network_vmamap::get_host_ptr(lhs_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), dg::network_vmamap::get_host_ptr(rhs_logit_vmamap), dispatch_info.lhs_bwd_tileops_dp_code); //combine these - to unclutter UI
                        } else if (lhs_grad_status == TILE_GRAD_STATUS_EMPTY){
                            dg::network_tileops_host_poly::bwd_pair_lhs_keep_n_assign(dg::network_vmamap::get_host_ptr(lhs_grad_vmamap), dg::network_vmamap::get_host_ptr(lhs_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), dg::network_vmamap::get_host_ptr(rhs_logit_vmamap), dispatch_info.lhs_bwd_tileops_dp_code);
                            set_tile_grad_status_nothrow(lhs, TILE_GRAD_STATUS_INITIALIZED);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }

                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(lhs));
                }

                {
                    auto [rhs_grad_map_resource, rhs_logit_map_resource, lhs_logit_map_resource, backwarder_grad_map_resource] = dg::network_uma::lockmap_safewait_many_nothrow<4>({{rhs_grad_umaptr, dispatch_info.rhs_grad_vd_id}, {rhs_logit_umaptr, dispatch_info.rhs_logit_vd_id}, 
                                                                                                                                                                                    {lhs_logit_umaptr, dispatch_info.lhs_logit_vd_id}, {dst_grad_umaptr, dispatch_info.backwarder_grad_vd_id}});

                    auto rhs_grad_vmamap        = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(rhs_grad_map_resource));
                    auto rhs_logit_vmamap       = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(rhs_logit_map_resource));
                    auto lhs_logit_vmamap       = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(lhs_logit_map_resource));
                    auto backwarder_grad_vmamap = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resource));

                    if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.rhs_dp_device_kind)){
                        if (rhs_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                            dg::network_tileops_cuda_poly::bwd_pair_rhs_zero_n_add(dg::network_vmamap::get_cuda_ptr(rhs_grad_vmamap), dg::network_vmamap::get_cuda_ptr(rhs_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), dg::network_vmamap::get_cuda_ptr(lhs_logit_vmamap), dispatch_info.rhs_bwd_tileops_dp_code); //maybe dispatch option instead of enumerating the function name - I think it's fine either way
                        } else if (rhs_grad_status == TILE_GRAD_STATUS_EMPTY){
                            dg::network_tileops_cuda_poly::bwd_pair_rhs_zero_n_assign(dg::network_vmamap::get_cuda_ptr(rhs_grad_vmamap), dg::network_vmamap::get_cuda_ptr(rhs_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), dg::network_vmamap::get_cuda_ptr(lhs_logit_vmamap), dispatch_info.rhs_bwd_tileops_dp_code);
                            set_tile_grad_status_nothrow(rhs, TILE_GRAD_STATUS_INITIALIZED);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.rhs_dp_device_kind)){
                        if (rhs_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                            dg::network_tileops_host_poly::bwd_pair_rhs_zero_n_add(dg::network_vmamap::get_host_ptr(rhs_grad_vmamap), dg::network_vmamap::get_host_ptr(rhs_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), dg::network_vmamap::get_host_ptr(lhs_logit_vmamap), dispatch_info.rhs_bwd_tileops_dp_code);
                        } else if (rhs_grad_status == TILE_GRAD_STATUS_EMPTY){
                            dg::network_tileops_host_poly::bwd_pair_rhs_zero_n_assign(dg::network_vmamap::get_host_ptr(rhs_grad_vmamap), dg::network_vmamap::get_host_ptr(rhs_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), dg::network_vmamap::get_host_ptr(lhs_logit_vmamap), dispatch_info.rhs_bwd_tileops_dp_code);
                            set_tile_grad_status_nothrow(rhs, TILE_GRAD_STATUS_INITIALIZED);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }

                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(rhs));
                }
            }
    };

    class BackwardDoUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

            }
    };

    class BackwardDoPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

            }
    };

    class BackwardDoExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter;
            const size_t delivery_capacity;
        
        public:

            BackwardDoExtnSrcSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                             std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter, //
                                             size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                 alias_getter(std::move(alias_getter)),
                                                                                 delivery_capacity(delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{
                
                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get());

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = get_extnsrc_rcu_addr_nothrow(dst);
                uma_ptr_t counterpart   = {};
                uma_ptr_t src           = {}; 

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    uma_ptr_t counterpart   = get_extnsrc_counterpart_nothrow(dst);
                    uma_ptr_t src           = get_extnsrc_descendant_nothrow(dst);
                }

                std::optional<uma_ptr_t> local_counterpart = this->alias_getter->alias(counterpart);

                if (!local_counterpart.has_value()){
                    return;
                }

                uma_ptr_t local_counterpart_rcu_addr    = get_extndst_rcu_addr_nothrow(local_counterpart.value());
                uma_ptr_t src_rcu_addr                  = get_tile_rcu_addr_nothrow(src);
                dg::network_uma_memops::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr, local_counterpart_rcu_addr);

                uma_ptr_t new_src                       = get_extnsrc_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id       = get_extnsrc_operatable_id_nothrow(dst);
                init_status_t init_status               = get_extnsrc_init_status_nothrow(dst);
                uma_ptr_t new_counterpart               = get_extnsrc_counterpart_nothrow(dst); 
                dispatch_control_t dispatch_control     = get_extnsrc_dispatch_control_nothrow(dst);

                uma_ptr_t dst_grad_umaptr               = get_extndst_logit_addr_nothrow(local_counterpart.value());
                uma_ptr_t external_counterpart          = get_extndst_counterpart_nothrow(local_counterpart.value());
                uma_ptr_t external_selfaddr             = get_extndst_selfaddr_nothrow(local_counterpart.value());
                uma_ptr_t external_grad_status          = get_extndst_grad_status_nothrow(local_counterpart.value());
                uma_ptr_t src_grad_umaptr               = get_tile_logit_addr_nothrow(src);
                uma_ptr_t src_logit_umaptr              = get_tile_grad_addr_nothrow(src);
                operatable_id_t src_operatable_id       = get_tile_operatable_id_nothrow(src);

                if (new_src != src){
                    return;
                }

                if (new_counterpart != external_selfaddr){
                    return;
                }

                if (external_counterpart != dst){
                    return;
                }

                if (src_operatable_id != dst_operatable_id){
                    return;
                }

                if (dst_operatable_id != external_operatable_id){
                    return;
                }

                if (external_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (external_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                auto [backwardee_grad_vd_id, backwardee_logit_vd_id, backwarder_grad_vd_id, dp_device_kind, tileops_dp_code]    = dg::network_dispatch_control::decode_bwd_extnsrc(dispatch_control);
                auto [backwardee_grad_map_resource, backwardee_logit_map_resource, backwarder_grad_map_resource]                = dg::network_uma::lockmap_safewait_many_nothrow<3>({{src_grad_umaptr, backwardee_grad_vd_id}, {src_logit_umaptr, backwardee_logit_vd_id}, {dst_grad_umaptr, backwarder_grad_vd_id}}); 
                
                auto backwardee_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_grad_map_resource)); 
                auto backwardee_logit_vmamap    = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_logit_map_resource));
                auto backwarder_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resource));

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_cuda_poly::bwd_mono_zero_n_add(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_code);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_cuda_poly::bwd_mono_zero_n_assign(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_code);
                        set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_INITIALIZED);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_host_poly::bwd_mono_zero_n_add(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_code);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_host_poly::bwd_mono_zero_n_assign(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_code);
                        set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_INITIALIZED);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class BackwardDoExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> requst_box;
            const size_t delivery_capacity;
            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
        
        public:

            BackwardDoExtnDstSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                             size_t delivery_capacity,
                                             std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever) noexcept: request_box(std::move(request_box)),
                                                                                                                            delivery_capacity(delivery_capacity),
                                                                                                                            uma_ip_retriever(std::move(uma_ip_retriever)){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<Request<Request<external_virtual_memory_event_t>>> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()))
                    return;
                }

                uma_ptr_t dst_rcu_addr      = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(dst);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                uma_ptr_t counterpart       = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(dst);
                init_status_t init_status   = dg::network_tile_member_getsetter::get_init_status_nothrow(dst);
                grad_status_t grad_status   = dg::network_tile_member_getsetter::get_grad_status_nothrow(dst);
                dg::string serialized       = dg::network_tile_member_getsetter::serialize_extndst(dst); 

                if (init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                external_virtual_memory_event_t injection_event     = dg::network_external_memcommit_factory::make_event_foreign_injection(dst, std::move(serialized));
                external_virtual_memory_event_t signal_event        = dg::network_external_memcommit_factory::make_event_backward_do_signal(counterpart);
                external_virtual_memory_event_t event               = dg::network_external_memcommit_factory::make_sequential_event(std::move(injection_event), std::move(signal_event));
                
                Request<external_virtual_memory_event_t> request{};
                request.requestor   = this->host_ip_retriever->ip();
                request.requestee   = this->uma_ip_retriever->ip(counterpart);
                request.content     = std::move(event);

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(request));
            }
    };

    class BackwardDoCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            BackwardDoCritSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }

        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(src);
                }

                uma_ptr_t src_rcu_addr              = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                   = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id   = dg::network_tile_member_getsetter::get_crit_operatable_id_nothrow(dst);
                init_status_t dst_init_status       = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(dst);
                dispatch_control_t dispatch_control = dg::network_tile_member_getsetter::get_crit_dispatch_control_nothrow(dst);
                uma_ptr_t dst_grad_umaptr           = dg::network_tile_member_getsetter::get_crit_grad_addr_nothrow(dst);
                grad_status_t dst_grad_status       = dg::network_tile_member_getsetter::get_crit_grad_status_nothrow(dst);

                operatable_id_t src_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_id_nothrow(src);
                init_status_t src_init_status       = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(src);
                grad_status_t src_grad_status       = dg::network_tile_member_getsetter::get_tile_grad_status_nothrow(src);
                uma_ptr_t src_grad_umaptr           = dg::network_tile_member_getsetter::get_tile_grad_addr_nothrow(src);
                uma_ptr_t src_logit_umaptr          = dg::network_tile_member_getsetter::get_tile_logit_addr_nothrow(src);

                if (src != new_src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                auto [backwardee_grad_vd_id, backwardee_logit_vd_id, backwarder_grad_vd_id, dp_device_kind, tileops_dp_code]    = dg::network_dispatch_control::decode_bwd_crit(dispatch_control);
                auto [backwardee_grad_map_resource, backwardee_logit_map_resource, backwarder_grad_map_resource]                = dg::network_uma::lockmap_safewait_many_nothrow<3u>({{src_grad_umaptr, backwardee_grad_vd_id}, {src_logit_umaptr, backwardee_logit_vd_id}, {dst_grad_umaptr, backwarder_grad_vd_id}});

                auto backwardee_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_grad_map_resource));
                auto backwardee_logit_vmamap    = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_logit_map_resource));
                auto backwarder_grad_vmamap     = dg::networK_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resource));

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_cuda_poly::bwd_mono_zero_n_add(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_code);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_cuda_poly::bwd_mono_zero_n_assign(dg::networK_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_code);
                        set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_INITIALIZED);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_host_poly::bwd_mono_zero_n_add(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_code);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_host_poly::bwd_mono_zero_n_assign(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_code);
                        set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_INITIALIZED);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class BackwardDoMsgrFwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoMsgrFwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                             size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                 delivery_capacity(delivery_capacity){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(dst);
                }

                uma_ptr_t src_rcu_addr                  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                       = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_msgrfwd_operatable_id_nothrow(dst);
                dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_msgrfwd_dispatch_control_nothrow(dst);
                init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(dst);
                grad_status_t dst_grad_status           = dg::network_tile_member_getsetter::get_msgrfwd_grad_status_nothrow(dst);
                uma_ptr_t dst_grad_umaptr               = dg::network_tile_member_getsetter::get_msgrfwd_grad_addr_nothrow(dst);

                uma_ptr_t src_grad_umaptr               = dg::network_tile_member_getsetter::get_tile_grad_addr_nothrow(src);
                uma_ptr_t src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr_nothrow(src);
                init_stauts_t src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(src);
                operatable_id_t src_operatable_id       = dg::network_tile_member_getsetter::get_tile_operatable_id_nothrow(src);

                if (new_src != src){
                    return;
                }

                if (dst_operatable_id != src_operatable_id){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                auto [backwardee_grad_vd_id, backwardee_logit_vd_id, backwarder_grad_vd_id, dp_device_kind, tileops_dp_code]    = dg::network_dispatch_control::decode_bwd_msgrfwd(dispatch_control);
                auto [backwardee_grad_map_resource, backwardee_logit_map_resource, backwarder_grad_map_resource]                = dg::network_uma::lockmap_safewait_many_nothrow<3>({{src_grad_umaptr, backwardee_grad_vd_id}, {src_logit_umaptr, backwardee_logit_vd_id}, {dst_grad_umaptr, backwarder_grad_vd_id}});

                auto backwardee_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_grad_map_resource));
                auto backwardee_logit_vmamap    = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_logit_map_resource));
                auto backwarder_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resource));

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_cuda_poly::bwd_mono_zero_n_add(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_code);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_cuda_poly::bwd_mono_zero_n_assign(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_code);
                        set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_INITIALIZED);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_host_poly::bwd_mono_zero_n_add(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_code);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_host_poly::bwd_mono_zero_n_assign(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_code);
                        set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_INITIALIZED);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class BackwardDoMsgrBwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t request_delivery_capacity;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box;
            const size_t eu_delivery_capacity; 

        public:

            BackwardDoMsgrBwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                             size_t request_delivery_capacity,
                                             std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box,
                                             size_t eu_delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                    request_delivery_capacity(request_delivery_capacity),
                                                                                    eu_packet_box(std::move(eu_packet_box)),
                                                                                    eu_delivery_capacity(eu_delivery_capacity){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->request_delivery_capacity);
                auto eu_delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->eu_packet_box.get(), this->eu_delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                if (!eu_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(eu_delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(ptr_arr[i], delivery_handle->get(), eu_delivery_handle->get());
                }
            }
        
        private:

            void resolve(uma_ptr_t dst, 
                         dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle,
                         dg::network_producer_consumer::DeliveryHandle<EndUserPacket> * eu_packet_delivery_handle) noexcept{

                auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(dst);

                if (!ptrchk.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                    return;
                }

                uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(dst);
                uma_ptr_t src           = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr);
                    src = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(dst);
                }

                uma_ptr_t src_rcu_addr                      = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(src);
                dg::network_memops_uma::memlock_guard mem_grd(dst_rcu_addr, src_rcu_addr);
                uma_ptr_t new_src                           = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(dst);
                operatable_id_t dst_operatable_id           = dg::network_tile_member_getsetter::get_msgrbwd_operatable_id_nothrow(dst);
                dispatch_control_t dispatch_control         = dg::network_tile_member_getsetter::get_msgrbwd_dispatch_control_nothrow(dst);
                init_status_t dst_init_status               = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(dst);
                bool dst_transmit_toggle                    = dg::network_tile_member_getsetter::get_msgrbwd_transmit_toggle_nothrow(dst);
                uma_ptr_t dst_grad_umaptr                   = dg::network_tile_member_getsetter::get_msgrbwd_grad_addr_nothrow(dst);
                uma_ptr_t tmp_grad_umaptr                   = dg::network_tile_member_getsetter::get_msgrbwd_tmp_grad_addr_nothrow(dst);
                grad_status_t dst_grad_status               = dg::network_tile_member_getsetter::get_msgrbwd_grad_status_nothrow(dst);
                timein_t dst_timein                         = dg::network_tile_member_getsetter::get_msgrbwd_timein_nothrow(dst);
                dst_info_t msgr_dst                         = dg::network_tile_member_getsetter::get_msgrbwd_dst_info_nothrow(dst);
                retry_count_t msgr_retry_count              = dg::network_tile_member_getsetter::get_msgrbwd_retry_count_nothrow(dst);
                transmit_urgency_t msgr_transmit_urgency    = dg::network_tile_member_getsetter::get_msgrbwd_transmit_urgency_nothrow(dst);
                transmit_comm_t msgr_transmit_comm          = dg::network_tile_member_getsetter::get_msgrbwd_transmit_comm_nothrow(dst);
                uma_ptr_t src_grad_umaptr                   = dg::network_tile_member_getsetter::get_tile_grad_addr_nothrow(src);
                uma_ptr_t src_logit_umaptr                  = dg::network_tile_member_getsetter::get_tile_logit_addr_nothrow(src);
                operatable_id_t src_operatable_id           = dg::network_tile_member_getsetter::get_tile_operatable_id_nothrow(src);
                init_status_t src_init_status               = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(src);

                if (new_src != src){
                    return;
                }

                if (src_operatable_id != dst_operatable_id){
                    return;
                }

                if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                    return;
                }

                if (dst_grad_status != TILE_GRAD_STATUS_INITIALIZED){
                    return;
                }

                auto [backwardee_grad_map_resource, backwardee_logit_map_resource, backwarder_grad_map_resource, tmp_grad_map_resource] = dg::network_uma::lockmap_safewait_many_nothrow<4>({{src_grad_umaptr, backwardee_grad_vd_id}, {src_logit_umaptr, backwardee_logit_vd_id}, 
                                                                                                                                                                                             {dst_grad_umaptr, backwarder_grad_vd_id}, {tmp_grad_umaptr, tmp_grad_vd_id}});

                auto backwardee_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_grad_map_resource));
                auto backwardee_logit_vmamap    = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwardee_logit_map_resource));
                auto backwarder_grad_vmamap     = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(backwarder_grad_map_resource));
                auto tmp_grad_vmamap            = dg::network_vmamap::mapsafe_nothrow(dg::network_uma::get_vma_ptr(tmp_grad_map_resource));

                if (dg::network_dispatch_control::is_cuda_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_cuda_poly::bwd_mono_keep_n_add(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileop_dp_code);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_cuda_poly::bwd_mono_keep_n_assign(dg::network_vmamap::get_cuda_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), tileops_dp_code);
                        set_msgrbwd_grad_status_nothrow(dst, TILE_GRAD_STATUS_INITIALIZED);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else if (dg::network_dispatch_control::is_host_dispatch(dp_device_kind)){
                    if (src_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                        dg::network_tileops_host_poly::bwd_mono_keep_n_add(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_code);
                    } else if (src_grad_status == TILE_GRAD_STATUS_EMPTY){
                        dg::network_tileops_host_poly::bwd_mono_keep_n_assign(dg::network_vmamap::get_host_ptr(backwardee_grad_vmamap), dg::network_vmamap::get_host_ptr(backwardee_logit_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), tileops_dp_code);
                        set_msgrbwd_grad_status_nothrow(dst, TILE_GRAD_STATUS_INITIALIZED);
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else{
                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    } else{
                        std::unreachable();
                    }
                }

                if (dst_timein < stdx::utc_timestamp()){
                    if (dg::network_dispatch_control::is_cuda_dispatch(gradaccum_dp_device_kind)){
                        if (tmp_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                            dg::network_tileops_cuda_poly::grad_accum_zero_n_add(dg::network_vmamap::get_cuda_ptr(tmp_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), gradaccum_tileops_dp_code);
                        } else if (tmp_grad_status == TILE_GRAD_STATUS_EMPTY){
                            dg::network_tileops_cuda_poly::grad_accum_zero_n_assign(dg::network_vmamap::get_cuda_ptr(tmp_grad_vmamap), dg::network_vmamap::get_cuda_ptr(backwarder_grad_vmamap), gradaccum_tileops_dp_code);
                            set_msgrbwd_tmp_grad_status_nothrow(dst, TILE_GRAD_STATUS_INITIALIZED);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    } else if (dg::network_dispatch_control::is_host_dispatch(gradaccum_dp_device_kind)){
                        if (tmp_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                            dg::network_tileops_host_poly::grad_accum_zero_n_add(dg::network_vmamap::get_host_ptr(tmp_grad_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), gradaccum_tileops_dp_code);
                        } else if (tmp_grad_status == TILE_GRAD_STATUS_EMPTY){
                            dg::network_tileops_host_poly::grad_accum_zero_n_assign(dg::network_vmamap::get_host_ptr(tmp_grad_vmamap), dg::network_vmamap::get_host_ptr(backwarder_grad_vmamap), gradaccum_tileops_dp_code);
                            set_msgrbwd_tmp_grad_status_nothrow(dst, TILE_GRAD_STATUS_INITIALIZED);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_logt_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            } else{
                                std::unreachable();
                            }
                        }
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                } else{
                    if (dst_transmit_toggle){
                        (void) dst_transmit_toggle;
                    } else{
                        EndUserPacket eu_packet{};
                        eu_packet.kind          = EUPACKET_MSGRBWD_TRANSMIT;
                        eu_packet.dst           = msgr_dst;
                        eu_packet.retry_count   = msgr_retry_count;
                        eu_packet.urgency       = msgr_transmit_urgency;
                        eu_packet.comm          = msgr_transmit_comm;

                        if (tmp_grad_status == TILE_GRAD_STATUS_INITIALIZED){
                            eu_packet.content   = dg::network_compact_serializer::serialize<dg::string>(GradientData{logit_id, get_msgrbwd_grad_nothrow(dst)});
                        } else if (tmp_grad_status == TILE_GRAD_STATUS_EMPTY){
                            eu_packet.content   = dg::network_compact_serializer::serialize<dg::string>(GradientData{logit_id, std::nullopt});
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(eu_packet_delivery_handle, std::move(eu_packet));
                        set_msgrbwd_transmit_toggle_nothrow(dst, true);
                    }
                }

                dg::network_producer_consumer::delvrsrv_deliver(request_delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class BackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> mono_resolutor;
            const size_t mono_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;

        public:

            BackwardDoSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor,
                                      size_t leaf_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> mono_resolutor,
                                      size_t mono_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pair_resolutor,
                                      size_t pair_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> uacm_resolutor,
                                      size_t uacm_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> pacm_resolutor,
                                      size_t pacm_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extnsrc_resolutor,
                                      size_t extnsrc_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> extndst_resolutor,
                                      size_t extndst_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> crit_resolutor,
                                      size_t crit_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrfwd_resolutor,
                                      size_t msgrfwd_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> msgrbwd_resolutor,
                                      size_t msgrbwd_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
                                                                            leaf_dispatch_sz(leaf_dispatch_sz),
                                                                            mono_resolutor(std;:move(mono_resolutor)),
                                                                            mono_dispatch_sz(mono_dispatch_sz),
                                                                            pair_resolutor(std::move(pair_resolutor)),
                                                                            pair_dispatch_sz(pair_dispatch_sz),
                                                                            uacm_resolutor(std::move(uacm_resolutor)),
                                                                            uacm_dispatch_sz(uacm_dispatch_sz),
                                                                            pacm_resolutor(std::move(pacm_resolutor)),
                                                                            pacm_dispatch_sz(pacm_dispatch_sz),
                                                                            extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                            extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                            extndst_resolutor(std::move(extndst_resolutor)),
                                                                            extndst_dispatch_sz(extndst_dispatch_sz),
                                                                            crit_resolutor(std::move(crit_resolutor)),
                                                                            crit_dispatch_sz(crit_dispatch_sz),
                                                                            msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                            msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                            msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                            msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto leaf_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_resolutor.get(), this->leaf_dispatch_sz);
                auto mono_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->mono_resolutor.get(), this->mono_dispatch_sz);
                auto pair_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz);
                auto uacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz);
                auto pacm_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz);
                auto extnsrc_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz);
                auto extndst_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz);
                auto crit_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz);
                auto msgrfwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz);
                auto msgrbwd_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz);

                if (!dg::network_exception::conjunc_expect_has_value(leaf_delivery_handle, mono_delivery_handle, pair_delivery_handle, 
                                                                     uacm_delivery_handle, pacm_delivery_handle, extnsrc_delivery_handle, 
                                                                     extndst_delivery_handle, crit_delivery_handle, msgrfwd_delivery_handle, 
                                                                     msgrbwd_delivery_handle)){

                    dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(ptr_arr[i]);

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(leaf_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MONO:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(mono_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PAIR:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pair_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_UACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(uacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_PACM:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNSRC:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extnsrc_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_EXTNDST:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extndst_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_CRIT:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(crit_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRFWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrfwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        case TILE_KIND_MSGRBWD:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrbwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        default:
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                                break;
                            } else{
                                std::unreachable();
                                break;
                            }
                    }
                }
            }
    };

    //

    class MemCommitResolutor: public virtual dg::network_concurrency::WorkerInterface{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>> producer;
            const size_t producer_consume_capacity;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_ping_signal_resolutor;
            const size_t fwd_ping_delivery_capacity; 
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pong_request_resolutor;
            const size_t fwd_pong_request_delivery_capacity;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_pong_signal_resolutor;
            const size_t fwd_pong_signal_delivery_capacity;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_do_resolutor;
            const size_t fwd_do_delivery_capacity;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> bwd_do_resolutor;
            const size_t bwd_do_delivery_capacity;

        public:

            MemCommitResolutor(std::shared_ptr<dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>> producer,
                               size_t producer_consume_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_ping_signal_resolutor,
                               size_t fwd_ping_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pong_request_resolutor,
                               size_t fwd_pong_request_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_pong_signal_resolutor,
                               size_t fwd_pong_signal_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> fwd_do_resolutor,
                               size_t fwd_do_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> bwd_do_resolutor,
                               size_t bwd_do_delivery_capacity) noexcept: producer(std::move(producer)),
                                                                          producer_consume_capacity(producer_consume_capacity),
                                                                          fwd_ping_signal_resolutor(std::move(fwd_ping_signal_resolutor)),
                                                                          fwd_ping_delivery_capacity(fwd_ping_delivery_capacity),
                                                                          fwd_pong_request_resolutor(std::move(fwd_pong_request_resolutor)),
                                                                          fwd_pong_request_delivery_capacity(fwd_pong_request_delivery_capacity),
                                                                          fwd_pong_signal_resolutor(std::move(fwd_pong_signal_resolutor)),
                                                                          fwd_pong_signal_delivery_capacity(fwd_pong_signal_delivery_capacity),
                                                                          fwd_do_resolutor(std::move(fwd_do_resolutor)),
                                                                          fwd_do_delivery_capacity(fwd_do_delivery_capacity),
                                                                          bwd_do_resolutor(std::move(bwd_do_resolutor)),
                                                                          bwd_do_delivery_capacity(bwd_do_delivery_capacity){}

            bool run_one_epoch() noexcept{

                auto virtual_memory_event_arr   = std::make_unique<virtual_memory_event_t>(this->producer_consume_capacity); //TODOs: internalize allocations
                size_t virtual_memory_event_sz  = {};
                this->producer->get(virtual_memory_event_arr.get(), virtual_memory_event_sz, this->producer_consume_capacity);

                if (virtual_memory_event_sz == 0u){
                    return false;
                }

                //refactor - this logic is transform consumers

                auto fwd_ping_signal_resolution_array   = std::make_unique<uma_ptr_t[]>(this->fwd_ping_delivery_capacity);
                auto fwd_pong_request_resolution_array  = std::make_unique<std::tuple<uma_ptr_t, uma_ptr_t>[]>(this->fwd_pong_request_delivery_capacity);
                auto fwd_pong_signal_resolution_array   = std::make_unique<uma_ptr_t[]>(this->fwd_pong_signal_delivery_capacity);
                auto fwd_do_resolution_array            = std::make_unique<uma_ptr_t[]>(this->fwd_do_delivery_capacity);
                auto bwd_do_resolution_array            = std::make_unique<uma_ptr_t[]>(this->bwd_do_delivery_capacity);

                auto fwd_ping_signal_lambda_consumer    = [this, &](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    for (size_t i = 0u; i < arr_sz; ++i){
                        fwd_ping_signal_resolution_array[i] = dg::network_memcommit_factory::read_event_forward_ping_signal(event_arr[i]);
                    }

                    this->fwd_ping_signal_resolutor->push(fwd_ping_signal_resolution_array.get(), arr_sz);
                };

                auto fwd_pong_request_lambda_consumer   = [this, &](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    for (size_t i = 0u; i < arr_sz; ++i){
                        fwd_pong_request_resolution_array[i] = dg::network_memcommit_factory::read_event_forward_pong_request(event_arr[i]);
                    }

                    this->fwd_pong_request_resolutor->push(fwd_pong_request_resolution_array.get(), arr_sz);
                };

                auto fwd_pong_signal_lambda_consumer    = [this, &](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    for (size_t i = 0u; i < arr_sz; ++i){
                        fwd_pong_signal_resolution_array[i] = dg::network_memcommit_factory::read_event_forward_pong_signal(event_arr[i]);
                    }
                    
                    this->fwd_pong_signal_resolutor->push(fwd_pong_signal_resolution_array.get(), arr_sz);
                };

                auto fwd_do_lambda_consumer             = [this, &](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    for (size_t i = 0u; i < arr_sz; ++i){
                        fwd_do_resolution_array[i] = dg::network_memcommit_factory::read_event_forward_do_signal(event_arr[i]);
                    }

                    this->fwd_do_resolutor->push(fwd_do_resolution_array.get(), arr_sz);
                };

                auto bwd_do_lambda_consumer             = [this, &](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    for (size_t i = 0u; i < arr_sz; ++i){
                        bwd_do_resolution_array[i] = dg::network_memcommit_factory::read_event_backward_do_signal(event_arr[i]);
                    }

                    this->bwd_do_resolutor->push(bwd_do_resolution_array.get(), arr_sz);
                };

                auto fwd_ping_signal_virtual_consumer   = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_ping_signal_lambda_consumer)>(fwd_ping_signal_lambda_consumer);
                auto fwd_pong_request_virtual_consumer  = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_pong_request_lambda_consumer)>(fwd_pong_request_lambda_consumer);
                auto fwd_pong_signal_virtual_consumer   = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_pong_signal_lambda_consumer)>(fwd_pong_signal_lambda_consumer);
                auto fwd_do_virtual_consumer            = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_do_lambda_consumer)>(fwd_do_lambda_consumer);
                auto bwd_do_virtual_consumer            = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(bwd_do_lambda_consumer)>(bwd_do_lambda_consumer);

                stdx::seq_cst_guard seqcst_guard;

                auto fwd_ping_signal_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_ping_signal_virtual_consumer, this->fwd_ping_delivery_capacity);
                auto fwd_pong_request_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_pong_request_virtual_consumer, this->fwd_pong_request_delivery_capacity);
                auto fwd_pong_signal_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_pong_signal_virtual_consumer, this->fwd_pong_signal_delivery_capacity);
                auto fwd_do_delivery_handle             = dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_do_virtual_consumer, this->fwd_do_delivery_capacity);
                auto bwd_do_delivery_handle             = dg::network_producer_consumer::delvrsrv_open_raiihandle(&bwd_do_virtual_consumer, this->bwd_do_delivery_capacity);

                if (!dg::network_exception::conjunc_expect_has_value(fwd_ping_signal_delivery_handle, fwd_pong_request_delivery_handle,
                                                                     fwd_pong_signal_delivery_handle, fwd_do_delivery_handle,
                                                                     bwd_do_delivery_handle)){

                    dg::network_log_stackdump::error(dg::network_exception::verbose(dg::network_exception::RESOURCE_EXHAUSTION));
                    return false;
                }

                for (size_t i = 0u; i < virtual_memory_event_sz; ++i){
                    memory_event_kind_t event_kind = dg::network_memcommit_factory::read_event_kind(virtual_memory_event_arr[i]);

                    switch (event_kind){
                        case dg::network_memcommit_factory::event_kind_forward_ping_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_ping_signal_delivery_handle)->get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_forward_pong_request:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_pong_request_delivery_handle)->get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_forward_pong_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_pong_signal_delivery_handle)->get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_forward_do_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_do_delivery_handle)->get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_backward_do_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(bwd_do_delivery_handle)->get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        default:
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                                break;
                            } else{
                                std::unreachable();
                                break;
                            }
                    }
                }

                return true;
            }
    };
}

#endif