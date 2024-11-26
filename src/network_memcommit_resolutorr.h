#ifndef __EVENT_DISPATCHER_H__
#define __EVENT_DISPATCHER_H__

#include <stdint.h>
#include <stddef.h>
#include <network_addr_lookup.h>
#include "network_tile_member_getsetter.h"
#include "network_memcommit_factory.h"
#include "network_producer_consumer.h"

namespace dg::network_memcommit_resolutor{

    struct UnifiedMemoryIPRetrieverInterface{
        virtual ~UnifiedMemoryIPRetrieverInterface() noexcept = default;
        virtual auto ip(uma_ptr_t) const noexcept -> Address = 0; 
    };

    struct HostIPRetrieverInterface{
        virtual ~HostIPRetrieverInterface() noexcept = default;
        virtual auto ip() const noexcept -> Address = 0;
    };

    struct ExternalAddressAliasGetterInterface{ //
        virtual ~ExternalAddressAliasGetterInterface() noexcept = default;
        virtual auto alias(uma_ptr_t) noexcept -> std::optional<uma_ptr_t> = 0; //change semantics
    };

    template <class T>
    struct Request{
        Address requestee;
        Address requestor;
        T content;
    };

    class OrdinaryForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            OrdinaryForwardPingSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                uma_ptr_t rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(ptr);
                dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(ptr);

                switch (init_status){
                    case TILE_INIT_STATUS_ORPHANED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        std::array<uma_ptr_t, MAX_DESCENDANT_SIZE> descendants{};
                        size_t descendant_size{};
                        std::tie(descendants, descendant_size) = dg::network_tile_member_getsetter::get_tile_descendants_nothrow(ptr); 

                        for (size_t i = 0u; i < descendant_size; ++i){
                            virtual_memory_event_t ping_request = dg::network_memcommit_factory::make_event_forward_ping_request(descendants[i], ptr);
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(ping_request));
                            virtual_memory_event_t pong_request = dg::network_memcommit_factory::make_event_forward_pong_request(descendants[i], ptr);
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(pong_request));
                        }

                        dg::network_tile_member_getsetter::set_initialization_status_nothrow(ptr, INIT_STATUS_DECAYED);
                        break;
                    }
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        break;
                    }
                    default:
                    {
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

    class DstExternalForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::unique_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            std::unique_ptr<HostIPRetrieverInterface> host_ip_retriever;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            const size_t delivery_capacity;

        public:

            DstExternalForwardPingSignalResolutor(std::unique_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                  std::unique_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                  std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<external_virtual_memory_event_t>> request_box,
                                                  size_t delivery_capacity): uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                             host_ip_retriever(std::move(host_ip_retriever)),
                                                                             request_box(std::move(request_box)),
                                                                             delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t requestee, dg::network_producer_consumer::DeliveryHandle<external_virtual_memory_event_t> * handle) noexcept{

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_dstextclone_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_guard(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_dstextclone_init_status_nothrow(requestee);

                switch (init_status){
                    case TILE_INIT_STATUS_ORPHANED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        uma_ptr_t requestor     = dg::network_tile_member_getsetter::get_dstextclone_src_addr_nothrow(requestee);
                        Address requestor_ip    = this->uma_ip_retriever->ip(src_addr);
                        Address requestee_ip    = this->host_ip_retriever->ip();
                        Request<external_virtual_memory_event_t> ping_request{requestor_ip, requestee_ip, dg::network_external_memcommit_factory::make_event_external_forward_ping_signal(requestor)};
                        Request<external_virtual_memory_event_t> pong_request{requestor_ip, requestee_ip, dg::network_external_memcommit_factory::make_event_external_forward_pong_request(requestor, requestee)};
                        dg::network_producer_conumser::delvrsrv_deliver(handle, std::move(ping_request));
                        dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(pong_request));
                        dg::network_tile_member_getsetter::set_dstextclone_init_status_nothrow(requestee, INIT_STATUS_DECAYED);
                        break;
                    }
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        break;
                    }
                    default:
                    {
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

    class ForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ordinary_ping_resolutor;
            const size_t ordinary_ping_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> dst_external_ping_resolutor;
            const size_t dst_external_ping_dispatch_sz;

        public:

            ForwardPingSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ordinary_ping_resolutor,
                                       size_t ordinary_ping_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> dst_external_ping_resolutor,
                                       size_t dst_external_ping_dispatch_sz) noexcept: ordinary_ping_resolutor(std::move(ordinary_ping_resolutor)),
                                                                                       ordinary_ping_dispatch_sz(ordinary_ping_dispatch_sz),
                                                                                       dst_external_ping_resolutor(std::move(dst_external_ping_resolutor)),
                                                                                       dst_external_ping_dispatch_sz(dst_external_ping_dispatch_sz){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto ordinary_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->ordinary_ping_resolutor.get(), this->ordinary_ping_dispatch_sz);
                auto dst_external_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->dst_external_ping_resolutor.get(), this->dst_external_ping_dispatch_sz);

                if (!ordinary_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(ordinary_delivery_handle.error()));
                    return;
                }

                if (!dst_external_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(dst_external_delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    tile_kind_t tile_kind = dg::network_tile_member_getsetter::get_tile_kind_nothrow(std::get<0>(ptr_arr[i])); 

                    if (tile_kind == TILE_KIND_DSTEXTCLONE){
                        dg::network_producer_consumer::delvrsrv_deliver(dst_external_delivery_handle.get(), ptr_arr[i]);
                    } else{
                        dg::network_producer_consumer::delvrsrv_deliver(ordinary_delivery_handle.get(), ptr_arr[i]);
                    }
                }
            }
    };

    class ForwardPongRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPongRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                        size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                            delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                uma_ptr_t requestee_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(requestee);
                dg::network_memops_uma::memlock_guard mem_grd(requestee_rcu_addr);
                init_status_t init_status = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(requestee_rcu_addr);

                switch (init_status){
                    case TILE_INIT_STATUS_ORPHANED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_ADOPTED:
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        size_t observer_arr_sz                                  = dg::network_tile_member_getsetter::get_tile_observer_size_nothrow(requestee);
                        size_t observer_arr_idx                                 = observer_arr_sz % OBSERVER_ARRAY_SZ;
                        std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = dg::network_tile_member_getsetter::get_tile_observer_nothrow(requestee);
                        observer_arr[observer_arr_idx]                          = requestor;
                        size_t new_observer_arr_sz                              = observer_arr_idx + 1;

                        dg::network_tile_member_getsetter::set_tile_observer_nothrow(requestee, observer_arr);
                        dg::network_tile_member_getsetter::set_tile_observer_size_nothrow(requestee, new_observer_arr_sz);
                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_pong_signal(requestor, requestee);
                        dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(request));
                        break;
                    }
                    default:
                    {
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

    class ForwardPongSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPongSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                       size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                           delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle.get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t signalee, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                uma_ptr_t signalee_rcu_addr         = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(signalee);
                dg::network_memops_uma::memlock_guard mem_grd(signalee_rcu_addr);
                init_status_t init_status           = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(signalee);

                switch (init_status){
                    case TILE_INIT_STATUS_ORPHANED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_ADOPTED:
                    {
                        break;
                    }
                    case TILE_INIT_STATUS_DECAYED:
                    {
                        pong_count_t pong_count         = dg::network_tile_member_getsetter::get_tile_pong_count_nothrow(signalee);
                        descendant_size_t descendant_sz = dg::network_tile_member_getsetter::get_tile_descendant_size_nothrow(signalee);

                        if (descendant_sz == pong_count){
                            virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_do_signal(signalee);
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(request));
                        } else{
                            pong_count += 1;
                            dg::network_tile_member_getsetter::set_tile_pong_count_nothrow(signalee, pong_count);
                            if (pong_count == descendant_sz){
                                virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_do_signal(signalee);
                                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(request));
                            }
                        }

                        break;
                    }
                    case TILE_INIT_STATUS_INITIALIZED:
                    {
                        break;
                    }
                    default:
                    {
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

    class OrdinaryForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            OrdinaryForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }
                
                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                bool forward_status = dg::network_tileops_host_handler::forward(ptr);

                if (!forward_status){
                    return;
                }

                uma_ptr_t ptr_rcu_addr                                  = dg::entwork_tile_member_getsetter::get_tile_rcu_addr_nothrow(ptr);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = {};
                size_t observer_arr_sz                                  = {};
                init_status_t init_status                               = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(ptr_rcu_addr);
                    init_status = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(ptr);

                    if (init_status != TILE_INIT_STATUS_INITIALIZED){ //this is fine - because there status post_forward is expected to be init_status_initialized - so dispatch_table_code guard is not a neccesity here
                        return;
                    }

                    observer_arr    = dg::network_tile_member_getsetter::get_tile_observer_nothrow(ptr);
                    observer_arr_sz = dg::network_tile_member_getsetter::get_tile_observer_size_nothrow(ptr);

                    for (size_t i = 0u; i < observer_arr_sz; ++i){
                        virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i]);
                        dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
                    }
                }
            }
    };

    class CritForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            CritForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                bool forward_status = dg::network_tileops_host_handler::forward_crit(ptr);

                if (!forward_status){
                    return;
                }

                uma_ptr_t ptr_lock_addr                                 = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(ptr);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = {};
                size_t observer_arr_sz                                  = {};
                init_status_t init_status                               = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(ptr_lock_addr);
                    init_status = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(ptr);

                    if (init_status != TILE_INIT_STATUS_INITIALIZED){
                        return;
                    }

                    observer_arr        = dg::network_tile_member_getsetter::get_crit_observer_nothrow(ptr);
                    observer_arr_sz     = dg::network_tile_member_getsetter::get_crit_observer_size_nothrow(ptr);

                    for (size_t i = 0u; i < observer_arr_sz; ++i){
                        virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i]);
                        dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
                    }

                    virtual_memory_event_t request = dg::network_memcommit_factory::make_event_backward_do_signal(ptr);
                    dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
                }
            }
    };

    class SrcExternalForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            std::unique_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            std::unique_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const size_t delivery_capacity;

        public:

            SrcExternalForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                  std::unique_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                  std::unique_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                  size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                      uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                      host_ip_retriever(std::move(host_ip_retriever)),
                                                                                      delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                bool forward_status = dg::network_tileops_host_handler::forward_srcextclone(ptr);

                if (!forward_status){
                    return;
                }

                uma_ptr_t ptr_rcu_addr      = dg::network_tile_member_getsetter::get_srcextclone_rcu_addr_nothrow(ptr);
                init_status_t init_status   = {};
                uma_ptr_t dst_addr          = {}; 

                {
                    dg::network_memops_uma::memlock_guard mem_grd(ptr_rcu_addr);
                    init_status = dg::network_tile_member_getsetter::get_srcexclone_init_status_nothrow(ptr);

                    if (init_status != TILE_INIT_STATUS_INITIALIZED){
                        return;
                    }

                    dst_addr = dg::network_tile_member_getsetter::get_srcexclone_dst_addr_nothrow(ptr);
                    //
                }
            }
    };

    class DstExternalForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            std::shared_ptr<ExternalAddressAliasGetterInterface> alias_getter;
            const size_t delivery_capacity;

        public:

            DstExternalForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                  std::shared_ptr<ExternalAddressAliasGetterInterface> alias_getter,
                                                  size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                      alias_getter(std::move(alias_getter)),
                                                                                      delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                std::optional<uma_ptr_t> alias = this->alias_getter->alias(ptr);
                
                if (!alias.has_value()){
                    return;
                }

                bool forward_status = dg::network_tileops_host_handler::forward_dstextclone(ptr, alias.value());

                if (!forward_status){
                    return;
                }

                uma_ptr_t ptr_lock_addr                                 = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(ptr);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = {};
                size_t observer_arr_sz                                  = {};
                init_status_t init_status                               = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(ptr_lock_addr);
                    init_status = dg::network_tile_member_getsetter::get_dstextclone_init_status_nothrow(ptr);

                    if (init_status != TILE_INIT_STATUS_INITIALIZED){
                        return;
                    }

                    observer_arr    = dg::network_tile_member_getsetter::get_dstextclone_observer_nothrow(ptr);
                    observer_arr_sz = dg::network_tile_member_getsetter::get_dstextclone_observer_size_nothrow(ptr);

                    for (size_t i = 0u; i < observer_arr_sz; ++i){
                        virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i]);
                        dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
                    }
                }
            }
    };

    class MsgrFwdForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            MsgrFwdForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle->get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                bool forward_status = dg::network_tileops_host_handler::forward_msgrfwd(ptr);

                if (!forward_status){
                    return;
                }

                uma_ptr_t ptr_lock_addr                                 = dg::network_tile_member_get_setter::get_tile_rcu_addr_nothrow(ptr);
                std::array<uma_ptr_t, OBSERVER_ARR_CAP> observer_arr    = {};
                size_t observer_arr_sz                                  = {};
                init_status_t init_status                               = {};
                dst_info_t dst_info                                     = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(ptr_lock_addr);
                    init_status = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(ptr);

                    if (init_status != TILE_INIT_STATUS_INITIALIZED){
                        return;
                    }

                    observer_arr    = dg::network_tile_member_getsetter::get_msgrfwd_observer_nothrow(ptr);
                    observer_arr_sz = dg::network_tile_member_getsetter::get_msgrfwd_observer_size_nothrow(ptr);
                    dst_info        = dg::network_tile_member_getsetter::get_msgrfwd_dst_info_nothrow(ptr);

                    for (size_t i = 0u; i < observer_arr_sz; ++i){
                        virtual_memory_event_t request = dg::network_memcommit_factory::make_event_forward_pong_signal(observer_arr[i]);
                        dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
                    }


                } 
            }
    };

    class ForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ordinary_forwardinit_resolutor;
            const size_t ordinary_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> crit_forwardinit_resolutor;
            const size_t crit_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> src_external_forwardinit_resolutor;
            const size_t src_external_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> dst_external_forwardinit_resolutor;
            const size_t dst_external_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> msgrfwd_forwardinit_resolutor;
            const size_t msgrfwd_dispatch_sz;

        public:

            ForwardInitSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ordinary_forwardinit_resolutor,
                                       size_t ordinary_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> crit_forwardinit_resolutor,
                                       size_t crit_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> src_external_forwardinit_resolutor,
                                       size_t src_external_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> dst_external_forwardinit_resolutor,
                                       size_t dst_external_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> msgrfwd_forwardinit_resolutor,
                                       size_t msgrfwd_dispatch_sz) noexcept: ordinary_forwardinit_resolutor(std::move(ordinary_forwardinit_resolutor)),
                                                                             ordinary_dispatch_sz(ordinary_dispatch_sz),
                                                                             crit_forwardinit_resolutor(std::move(crit_forwardinit_resolutor)),
                                                                             crit_dispatch_sz(crit_dispatch_sz), 
                                                                             src_external_forwardinit_resolutor(std::move(src_external_forwardinit_resolutor)),
                                                                             src_external_dispatch_sz(src_external_dispatch_sz),
                                                                             dst_external_forwardinit_resolutor(std::move(dst_external_forwardinit_resolutor)),
                                                                             dst_external_dispatch_sz(dst_external_dispatch_sz),
                                                                             msgrfwd_forwardinit_resolutor(std::move(msgrfwd_forwardinit_resolutor)),
                                                                             msgrfwd_dispatch_sz(msgrfwd_dispatch_sz){}
            
            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto ordinary_fwdinit_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->ordinary_forwardinit_resolutor.get(), this->ordinary_dispatch_sz);
                auto crit_fwdinit_delivery_handle           = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_forwardinit_resolutor.get(), this->crit_dispatch_sz);
                auto src_external_fwdinit_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->src_external_forwardinit_resolutor.get(), this->src_external_dispatch_sz);
                auto dst_external_fwdinit_delivery_handle   = dg::entwork_producer_consumer::delvrsrv_open_raiihandle(this->dst_external_forwardinit_resolutor.get(), this->dst_external_dispatch_sz);
                auto msgrfwd_fwdinit_delivery_handle        = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_forwardinit_resolutor.get(), this->msgrfwd_dispatch_sz);

                if (!ordinary_fwdinit_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(ordinary_fwdinit_delivery_handle.error()));
                    return;
                }

                if (!crit_fwdinit_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(crit_fwdinit_delivery_handle.error()));
                    return;
                }

                if (!src_external_fwdinit_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(src_external_fwdinit_delivery_handle.error()));
                    return;
                }

                if (!dst_external_fwdinit_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(dst_external_fwdinit_delivery_handle.error()));
                    return;
                }

                if (!msgrfwd_fwdinit_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(msgrfwd_fwdinit_delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    tile_kind_t tile_kind = dg::network_tile_member_getsetter::get_tile_kind_nothrow(std::get<0>(ptr_arr[i]));

                    if (tile_kind == TILE_KIND_SRCEXTCLONE){
                        dg::network_producer_consumer::delvrsrv_deliver(src_external_fwdinit_delivery_handle->get(), ptr_arr[i]);
                    } else if (tile_kind == TILE_KIND_DSTEXTCLONE){
                        dg::network_producer_consumer::delvrsrv_deliver(dst_external_fwdinit_delivery_handle->get(), ptr_arr[i]);
                    } else if (tile_kind == TILE_KIND_CRIT){
                        dg::network_producer_consumer::delvrsrv_deliver(crit_fwdinit_delivery_handle->get(), ptr_arr[i]);
                    } else if (tile_kind == TILE_KIND_MSGRFWD){
                        dg::network_producer_consumer::delvrsrv_deliver(msgrfwd_fwdinit_delivery_handle->get(), ptr_arr[i]);
                    } else{
                        dg::network_producer_consumer::delvrsrv_deliver(ordinary_fwdinit_delivery_handle->get(), ptr_arr[i]);
                    }
                }
            }
    };

    class DstExternalBackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<virtual_memory_event_t>>> request_box;
            const size_t delivery_capacity;
            std::shared_ptr<UnifiedMemoryIPRetrieverInterface>  uma_ip_retriever;
        
        public:

            DstExternalBackwardDoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<virtual_memory_event_t>>> request_box,
                                                 size_t delivery_capacity,
                                                 std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever) noexcept: request_box(std::move(request_box)),
                                                                                                                                delivery_capacity(delivery_capacity),
                                                                                                                                uma_ip_retriever(std::move(uma_ip_retriever)){}
            

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * handle) noexcept{

            }
    };      

    class SrcExternalBackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            std::shared_ptr<ExternalAddressAliasGetterInterface> alias_getter;
            const size_t delivery_capacity;

        public:

            SrcExternalBackwardDoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                 std::shared_ptr<ExternalAddressAliasGetterInterface> alias_getter,
                                                 const size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                           alias_getter(std::move(alias_getter)),
                                                                                           delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t backwarder, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                std::optional<uma_ptr_t> alias = this->alias_getter->alias(backwarder);

                if (!alias.has_value()){
                    return;
                }

                bool backward_status = dg::network_tileops_host_handler::backward(backwarder, alias.value());

                if (!backward_status){
                    return;
                }

                uma_ptr_t backwarder_rcu_addr                               = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(backwarder);
                std::array<uma_ptr_t, MAX_DESCENDANT_SIZE> descendant_arr   = {};
                size_t descendant_arr_sz                                    = {};
                init_status_t init_status                                   = {};

                {
                    dg::network_memops_uma::memlock_guard mem_grd(backwarder_rcu_addr);
                    init_status = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(backwarder);

                    switch (init_status){
                        case TILE_INIT_STATUS_ORPHANED:
                            break;
                        case TILE_INIT_STATUS_ADOPTED:
                            break;
                        case ITLE_INIT_STATUS_DECAYED:
                            break;
                        case TILE_INIT_STATUS_INITIALIZED:
                            descendant_arr      = dg::network_tile_member_getsetter::get_srcextclone_descendant_nothrow(backwarder);
                            descendant_arr_sz   = dg::network_tile_member_getsetter::get_srcextclone_descendant_size_nothrow(backwarder);
                            
                            for (size_t i = 0u; i < descendant_arr_sz; ++i){
                                virtual_memory_event_t request = dg::network_memcommit_factory::make_event_backward_do_request(descendant_arr[i]);
                                dg::network_producer_consumer::delvrsrv_deliver(handle, std::move(request));
                            }
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

    class OrdinaryBackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            OrdinaryBackwardDoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle->get());
                }
            }

        private:

            void internal_resolve(uma_ptr_t backwarder, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * delivery_handle) noexcept{

                bool backward_status = dg::network_tileops_host_handler::backward(backwarder);

                if (!backward_status){
                    return;
                }

                uma_ptr_t backwarder_rcu_addr                               = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(backwarder);
                std::array<uma_ptr_t, MAX_DESCENDANT_SIZE> descendant_arr   = {};
                size_t descendant_arr_sz                                    = {};
                init_status_t init_status                                   = {}; 

                {
                    dg::network_memops_uma::memlock_guard mem_grd(backwarder_rcu_addr);
                    init_status = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(backwarder);

                    switch (init_status){
                        case TILE_INIT_STATUS_ORPHANED:
                            break;
                        case TILE_INIT_STATUS_ADOPTED:
                            break;
                        case TILE_INIT_STATUS_INITIALIZED:
                            descendant_arr      = dg::network_tile_member_getsetter::get_tile_descendant_nothrow(backwarder);
                            descendant_arr_sz   = dg::network_tile_member_getsetter::get_tile_descendant_size_nothrow(backwarder);

                            for (size_t i = 0u; i < descendant_arr_sz; ++i){
                                virtual_memory_event_t request = dg::network_memcommit_factory::make_event_backward_do_request(descendant_arr[i]);
                                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, std::move(request));
                            }

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
    
    class BackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> src_external_backward_resolutor;
            const size_t src_external_backward_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> dst_external_backward_resolutor;
            const size_t dst_external_backward_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ordinary_backward_resolutor;
            const size_t ordinary_backward_dispatch_sz;
        
        public:

            BackwardDoSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> src_external_backward_resolutor,
                                      size_t src_external_backward_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> dst_external_backward_resolutor,
                                      size_t dst_external_backward_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ordinary_backward_resolutor,
                                      size_t ordinary_backward_dispatch_sz) noexcept: src_external_backward_resolutor(std::move(src_external_backward_resolutor)),
                                                                                      src_external_backward_dispatch_sz(src_external_backward_dispatch_sz),
                                                                                      dst_external_backward_resolutor(std::move(dst_external_backward_resolutor)),
                                                                                      dst_external_backward_dispatch_sz(dst_external_backward_dispatch_sz),
                                                                                      ordinary_backward_resolutor(std::move(ordinary_backward_resolutor)),
                                                                                      ordinary_backward_dispatch_sz(ordinary_backward_dispatch_sz){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto src_external_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->src_external_backward_resolutor.get(), this->src_external_backward_dispatch_sz);
                auto dst_external_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->dst_external_backward_resolutor.get(), this->dst_external_backward_dispatch_sz);
                auto ordinary_delivery_handle       = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->ordinary_backward_resolutor.get(), this->ordinary_backward_dispatch_sz);

                if (!src_external_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(src_external_delivery_handle.error()));
                    return;
                }

                if (!dst_external_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(dst_external_delivery_handle.error()));
                    return;
                }

                if (!ordinary_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(ordinary_delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    tile_kind_t tile_kind = dg::network_tile_member_getsetter::get_tile_kind_nothrow(std::get<0>(ptr_arr[i]));

                    if (tile_kind == TILE_KIND_SRCEXTCLONE){
                        dg::network_producer_consumer::delvrsrv_deliver(src_external_delivery_handle->get(), std::get<0>(ptr_arr[i]));
                    } else if (tile_kind == TILE_KIND_DSTEXTCLONE){
                        dg::network_producer_consumer::delvrsrv_deliver(dst_external_delivery_handle->get(), std::get<0>(ptr_arr[i]));
                    } else{
                        dg::network_producer_consumer::delvrsrv_deliver(ordinary_delivery_handle->get(), std::get<0>(ptr_arr[i]));
                    }
                }
            }
    };

    class MemCommitResolutor: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>> producer;
            const size_t producer_consume_capacity;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> fwd_ping_signal_resolutor;
            const size_t fwd_ping_delivery_capacity; 
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pong_request_resolutor;
            const size_t fwd_pong_request_delivery_capacity;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> fwd_pong_signal_resolutor;
            const size_t fwd_pong_signal_delivery_capacity;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> fwd_do_signal_resolutor;
            const size_t fwd_do_delivery_capacity;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> bwd_do_signal_resolutor;
            const size_t bwd_do_delivery_capacity;

        public:

            MemCommitResolutor(std::shared_ptr<dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>> producer,
                               size_t producer_consume_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> fwd_ping_signal_resolutor,
                               size_t fwd_ping_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pong_request_resolutor,
                               size_t fwd_pong_request_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> fwd_pong_signal_resolutor,
                               size_t fwd_pong_signal_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> fwd_do_signal_resolutor,
                               size_t fwd_do_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> bwd_do_signal_resolutor,
                               size_t bwd_do_delivery_capacity) noexcept: producer(std::move(producer)),
                                                                          producer_consume_capacity(producer_consume_capacity),
                                                                          fwd_ping_signal_resolutor(std::move(fwd_ping_signal_resolutor)),
                                                                          fwd_ping_delivery_capacity(fwd_ping_delivery_capacity),
                                                                          fwd_pong_request_resolutor(std::move(fwd_pong_request_resolutor)),
                                                                          fwd_pong_request_delivery_capacity(fwd_pong_request_delivery_capacity),
                                                                          fwd_pong_signal_resolutor(std::move(fwd_pong_signal_resolutor)),
                                                                          fwd_pong_signal_delivery_capacity(fwd_pong_signal_delivery_capacity),
                                                                          fwd_do_signal_resolutor(std::move(fwd_do_signal_resolutor)),
                                                                          fwd_do_delivery_capacity(fwd_do_delivery_capacity),
                                                                          bwd_do_signal_resolutor(std::move(bwd_do_signal_resolutor)),
                                                                          bwd_do_delivery_capacity(bwd_do_delivery_capacity){}

            bool run_one_epoch() noexcept{

                using namespace dg::network_memcommit_factory;

                auto fwd_ping_signal_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_ping_signal_resolutor.get(), this->fwd_ping_delivery_capacity);
                auto fwd_pong_request_delivery_handle   = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_pong_request_resolutor.get(), this->fwd_pong_request_delivery_capacity);
                auto fwd_pong_signal_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_pong_signal_resolutor.get(), this->fwd_pong_signal_delivery_capacity);
                auto fwd_do_delivery_handle             = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_do_signal_resolutor.get(), this->fwd_do_delivery_capacity);
                auto bwd_do_delivery_handle             = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->bwd_do_signal_resolutor.get(), this->bwd_do_delivery_capacity);

                if (!fwd_ping_signal_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(fwd_ping_signal_delivery_handle.error()));
                    return false;
                }

                if (!fwd_pong_request_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(fwd_pong_request_delivery_handle.error()));
                    return false;
                }

                if (!fwd_do_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(fwd_do_delivery_handle.error()));
                    return false;
                }

                if (!bwd_do_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(bwd_do_delivery_handle.error()));
                    return false;
                }

                auto virtual_memory_event_arr   = std::make_unique<virtual_memory_event_t>(this->producer_consumer_capacity);
                size_t virtual_memory_event_sz  = {};
                this->producer->get(virtual_memory_event_arr.get(), virtual_memory_event_sz, this->producer_consumer_capacity);

                for (size_t i = 0u; i < virtual_memory_event_sz; ++i){
                    memory_event_kind_t event_kind = read_event_kind(virtual_memory_event_arr[i]);

                    switch (event_kind){
                        case event_kind_forward_ping_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_ping_signal_delivery_handle)->get(), read_event_forward_ping_signal(virtual_memory_event_arr[i]));
                            break;
                        case event_kind_forward_pong_request:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_pong_request_delivery_handle)->get(), read_event_forward_pong_request(virtual_memory_event_arr[i]));
                            break;
                        case event_kind_forward_pong_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_pong_signal_delivery_handle)->get(), read_event_forward_pong_signal(virtual_memory_event_arr[i]));
                            break;
                        case event_kind_forward_do_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(fwd_do_delivery_handle)->get(), read_event_forward_do_signal(virtual_memory_event_arr[i]));
                            break;
                        case event_kind_backward_do_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(bwd_do_delivery_handle)->get(), read_event_backward_do_signal(virtual_memory_event_arr[i]));
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