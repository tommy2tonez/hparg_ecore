#ifndef __EVENT_DISPATCHER_H__
#define __EVENT_DISPATCHER_H__

#include <stdint.h>
#include <stddef.h>
#include <network_addr_lookup.h>
#include "network_tile_member_access.h"   
#include "network_memcommit_factory.h" 
#include "network_producer_consumer.h"

namespace dg::network_memcommit_resolutor{

    //this is actually where the bottleneck is
    //these workorders will be dispatched -> asynchronous device in batches or individually (what asynchronous devices decide to do for optimizations is none of these guys' responsibilities)

    //ideally - in a perfectly implemented scheduler for high parallel application - latter
    //        - in a less perfectly implemented scheduler - former (probably does something like sequential lock acquiring of non-overlapping memory_regions - all lock acquires should be serialized at these resolutors (and batched as a single commit lock_try) - to assure of lock hierarchical ordering)
    //        - GlobalResolutor does not necessarily deliver directly to these resolutors - that's wrong 
    //        - GlobalResolutor might deliver to individual centers which then delivered to these resolutors - this is the beauty of abstraction
    //        - reconsider if too many memory_events should incur lags for fwd + bwd - this cannot be told without proper instruments - because if GPU is saturated then it's hard to tell the ends 


    //responsibility: request initialization - if already initialized - abort
    //                                       - if not initialized: - if descendants pings already sent - abort
    //                                                             - if descendants pings not already sent - decay ping signal -> descendants ping_pong signals -> deliver to request_box 

    //                                       - optimizables: reduce lck_request by leveraging sequential locality (this is actually hard to achieve)
    //                                                       reduce lck_request by leveraging temporal locality (this is easier to achieve) - says keep an unordered_set until a specific capacity or a repeated address is found

    //define ordinary - do precond in internal_resolve
    class OrdinaryForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box;
            const size_t delivery_capacity; //this should be kept internally - 

        public:

            OrdinaryForwardPingSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box,
                                               size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{
                
                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle.get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<virtual_memory_commit_t> * delivery_handle) noexcept{

                uma_ptr_t rcu_addr          = dg::network_tile_member_safe_getsetter::get_rcu_addr_nothrow(ptr);
                auto lck_grd                = dg::network_memops_uma::memlock_guard(rcu_addr);
                init_status_t init_status   = dg::network_tile_member_getsetter::get_initialization_status_nothrow(ptr);

                switch (init_status){
                    case INIT_STATUS_INITIALIZED:
                        return;
                    case INIT_STATUS_DECAYED:
                        return;
                    case INIT_STATUS_EMPTY:
                        std::array<uma_ptr_t, MAX_DESCENDANT_SIZE> descendants{};
                        size_t descendant_size{};
                        std::tie(descendants, descendant_size) = dg::network_tile_member_getsetter::get_descendants_nothrow(ptr); 

                        for (size_t i = 0u; i < descendant_size; ++i){
                            virtual_memory_commit_t ping_request = dg::network_memcommit_factory::make_event_forward_ping_request(descendants[i], ptr);
                            dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, ping_request);
                            virtual_memory_commit_t pong_request = dg::network_memcommit_factory::make_event_forward_pong_request(descendants[i], ptr);
                            dg::network_producer_consumer::delvserv_deliver(delivery_handle, pong_request);
                        }

                        dg::network_tile_member_getsetter::set_initialization_status_nothrow(ptr, INIT_STATUS_DECAYED);
                        return;
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort(); 
                        }
                        return;
                }
            }
    };

    struct UnifiedMemoryIPRetrieverInterface{
        virtual ~UnifiedMemoryIPRetrieverInterface() noexcept = default;
        virtual auto ip(uma_ptr_t) const noexcept -> Address = 0; 
    };

    struct HostIPRetrieverInterface{
        virtual ~HostIPRetrieverInterface() noexcept = default;
        virtual auto ip() const noexcept -> Address = 0;
    };

    template <class T>
    struct Request{
        Address requestee;
        Address requestor;
        T content;
    };

    class ExternalForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_commit_t>>> request_box;
            const size_t delivery_capacity;
        
        public:

            ExternalForwardPingSignalResolutor(std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                               std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                               std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<external_virtual_memory_commit_t>> request_box,
                                               size_t delivery_capacity): uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                          host_ip_retriever(std::move(host_ip_retriever)),
                                                                          request_box(std::move(request_box)),
                                                                          delivery_capacity(delivery_capacity){}
            
            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle.get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t requestee, dg::network_producer_consumer::DeliveryHandle<external_virtual_memory_commit_t> * handle) noexcept{

                uma_ptr_t rcu_addr          = dg::network_tile_member_safe_getsetter::get_extclone_rcu_addr_nothrow(requestee);
                auto lck_grd                = dg::network_memops_uma::memlock_guard(rcu_addr);
                init_status_t init_status   = dg::network_tile_member_getsetter::get_extclone_initialization_status_nothrow(requestee);

                switch (init_status){
                    case INIT_STATUS_INITIALIZED:
                        return;
                    case INIT_STATUS_DECAYED:
                        return;
                    case INIT_STATUS_EMPTY:
                        uma_ptr_t requestor     = dg::network_tile_member_getsetter::get_extclone_src_addr_nothrow(requestee); 
                        Address requestor_ip    = this->uma_ip_retriever->ip(src_addr);
                        Address requestee_ip    = this->host_ip_retriever->ip();
                        Request<external_virtual_memory_commit_t> ping_request{requestor_ip, requestee_ip, dg::network_external_memcommit_factory::make_event_external_forward_ping_signal(requestor)};
                        Request<external_virtual_memory_commit_t> pong_request{requestor_ip, requestee_ip, dg::network_external_memcommit_factory::make_event_external_forward_pong_request(requestor, requestee)};
                        dg::network_producer_conumser::delvsrv_deliver(handle, std::move(ping_request));
                        dg::network_producer_consumer::delvsrv_deliver(handle, std::move(pong_request));
                        return;
                    default:
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                        return;
                }
            }
    };

    class ForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ordinary_ping_resolutor;
            const size_t ordinary_ping_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> external_ping_resolutor;
            const size_t external_ping_dispatch_sz;

        public:

            ForwardPingSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ordinary_ping_resolutor,
                                       size_t ordinary_ping_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> external_ping_resolutor,
                                       size_t external_ping_dispatch_sz) noexcept: ordinary_ping_resolutor(std::move(ordinary_ping_resolutor)),
                                                                                   ordinary_ping_dispatch_sz(ordinary_ping_dispatch_sz),
                                                                                   external_ping_resolutor(std::move(external_ping_resolutor)),
                                                                                   external_ping_dispatch_sz(external_ping_dispatch_sz){}
            
            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto ordinary_delivery_handle   = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->ordinary_ping_resolutor.get(), this->ordinary_ping_dispatch_sz));
                auto external_delivery_handle   = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->external_ping_resolutor.get(), this->external_ping_dispatch_sz));

                for (size_t i = 0u; i < sz; ++i){
                    tile_kind_t tile_kind = dg::network_tile_member_safe_getsetter::get_tile_kind_nothrow(std::get<0>(ptr_arr[i])); 

                    if (tile_kind == TILE_KIND_EXTERNAL_CLONE){
                        dg::network_producer_consumer::delvsrv_deliver(external_delivery_handle.get(), ptr_arr[i]);
                    } else{
                        dg::network_producer_consumer::delvsrv_deliver(ordinary_delivery_handle.get(), ptr_arr[i]);
                    }
                }
            }
    };

    //responsibility: request pong: - if already initialized: signal pong
    //                              - if not initialized: push -> registered_init_observing_addr 

    class ForwardPongRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardPongRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box,
                                        size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                            delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle.get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_commit_t> * delivery_handle) noexcept{

                uma_ptr_t requestee_rcu_addr    = dg::network_tile_member_safe_getsetter::get_rcu_addr_nothrow(requestee);
                uma_ptr_t requestor_rcu_addr    = dg::network_tile_member_safe_getsetter::get_rcu_addr_nothrow(requestor);
                auto lck_grd                    = dg::network_memops_uma::memlock_many_guard(requestee_rcu_addr, requestor_rcu_addr);
                init_status_t init_status       = dg::network_tile_member_getsetter::get_initialization_status_nothrow(requestee_rcu_addr);

                if (init_status == INIT_STATUS_INITIALIZED){
                    virtual_memory_commit_t commit = dg::network_memcommit_factory::make_event_forward_pong_signal(requestee, requestor);
                    dg::network_producer_consumer::delvsrv_deliver(delivery_handle, commit);
                    return;
                }

                //this is actually hard - this has to be fixed_size_array - circular observing addresses - remember - that accuracy is not important - fixed buffer is 
                //- two reasons:
                //first - memory region + cache + locality, 
                //second - serializable uma_buffer - such that an uma_buffer ptr equals to bit_torrent transfer + metadata without complicated serialization protocol 

                if (init_status == INIT_STATUS_EMPTY || init_status == INIT_STATUS_DECAYED){
                    std::array<uma_ptr_t, MAX_INIT_OBSERVING_ADDR_COUNT> init_observing_addr = dg::network_tile_member_getsetter::get_init_observing_addr_nothrow(requestee);
                    size_t init_observing_addr_sz = dg::network_tile_member_getsetter::get_init_observing_addr_sz_nothrow(requestee);
                    init_observing_addr[init_observing_addr_sz] = requestor;
                    init_observing_addr_sz = (init_observing_addr_sz + 1) % MAX_INIT_OBSERVING_ADDR_COUNT; 

                    dg::network_tile_member_getsetter::set_init_observing_addr_nothrow(requestee, init_observing_addr);
                    dg::network_tile_member_getsetter::set_init_observing_addr_sz_nothrow(requestee, init_observing_addr_sz);
                    return;
                }

                if constexpr(DEBUG_MODE_FLAG){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION)); //extension bug_guard - initialization status is not considered 
                    std::abort();
                }
            }
    };

    //responsibility: signal_pong: - if already initialized: abort
    //                             - if not initialized: if decayed: check if same operatable_id + is_src_initialized + saturate pong_signal_countdown at 0 - first 0 will send init_signal
    //                                                   if not decayed: - abort
    //

    class ForwardPongSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardPongSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box,
                                       size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                           delivery_capacity(delivery_capacity){}
            
            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), std::get<1>(ptr_arr[i]), delivery_handle.get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t requestee, uma_ptr_t requestor, dg::network_producer_consumer::DeliveryHandle<virtual_memory_commit_t> * delivery_handle) noexcept{

                uma_ptr_t requestee_rcu_addr        = dg::network_tile_member_safe_getsetter::get_rcu_addr_nothrow(requestee);
                uma_ptr_t requestor_rcu_addr        = dg::network_tile_member_safe_getsetter::get_rcu_addr_nothrow(requestor);
                auto lck_grd                        = dg::network_memops_uma::memlock_many_guard(requestee_rcu_addr, requestor_rcu_addr);
                init_status_t requestee_init_status = dg::network_tile_member_getsetter::get_initialization_status_nothrow(requestee); 
                init_status_t requestor_init_status = dg::network_tile_member_getsetter::get_initialization_status_nothrow(requestor);
                operatable_id_t requestee_ops_id    = dg::network_tile_member_getsetter::get_operatable_id_nothrow(requestee);
                operatable_id_t requestor_ops_id    = dg::network_tile_member_getsetter::get_operatable_id_nothrow(requestor); 
                pong_count_t requestee_pong_count   = dg::network_tile_member_getsetter::get_pong_count_nothrow(requestee); 

                if (requestor_init_status != INIT_STATUS_INITIALIZED){
                    return;
                }

                if (requestee_init_status != INIT_STATUS_DECAYED){
                    return;
                }

                if (requestee_ops_id != requestor_ops_id){
                    return;
                }
                
                static_assert(std::is_unsigned_v<pong_count_t>);

                if (requestee_pong_count == 0u){
                    return;
                }

                requestee_pong_count -= 1;
                dg::network_tile_member_getsetter::set_pong_count_nothrow(requestee, requestee_pong_count);

                if (requestee_pong_count == 0u){
                    virtual_memory_commit_t init_request = dg::network_memcommit_factory::make_event_forward_do_signal(requestee);
                    dg::network_producer_consumer::delvsrv_deliver(delivery_handle, init_request);
                }
            }

    };

    //responsibility: do_initialization: - if already initialized: abort
    //                                   - if not initialized: - if decayed: - if pong_signal_countdount == 0: - if operatable_id is the same: - invoke void forward_poly(uma_ptr_t) noexcept, decay -> pong signal + set init_status -> INIT_STATUS_INITIALIZED
    //                                                                                                         - if operatable_id is not the same: abort
    //                                                                       - if pong_signal_countdount != 0: - abort
    //                                                         - if not decayed: abort
    //this is incredibly slow - look optimizables
    
    //define ordinary - do precond in internal_resolve
    class OrdinaryForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            OrdinaryForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box,
                                               size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity){}

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                for (size_t i = 0u; i < sz; ++i){
                    auto [descendants, sz] = this->get_descendants(std::get<0>(ptr_arr[i]));
                    this->internal_resolve(std::get<0>(ptr_arr[i]), descendants.data(), sz, delivery_handle.get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t root, uma_ptr_t * descendant, size_t descendant_size, dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * handle) noexcept{

                //this is incredibly hard to do performantly - I rather sleep on this than to implement incorrectly here - remember that CPU/GPU ratio should be 1 / (1 << 20)  - for efficient computing
                //first - need to have the temporal locality right - that ptr_arr, sz has to batch scheduled - by collectors or whatever - this depends heavily on the use cases - like compression - tons of data go through the same dictionary - scheduling is important for saturating GPUs 
                //second - need to get the memory_lock right 
                //third - need to get the asynchronous device scheduling right
                //fourth - need to implement the asynchronos::wait(vector<work_ticketid_t>) right
                //fifth - need to implement asynchronous machine right
            }

            auto get_descendants(uma_ptr_t ptr) noexcept -> std::pair<std::array<uma_ptr_t, MAX_DESCENDANT_SIZE>, size_t>{

                auto rcu_addr   = dg::network_tile_member_getsetter::get_rcu_addr_nothrow(ptr);
                auto lck_grd    = dg::network_memops_uma::memlock_guard(rcu_addr);

                return dg::network_tile_member_getsetter::get_descendants_nothrow(ptr);
            }

    };

    //ordinary + inplace decay -> crit, crit has two logit_value array - 1 is expected, 1 is ordinary, a crit signal will do bce + ce + friends then decay -> backward signals  

    class CritForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

    };

    //do clone then do external_memcommit_t -> observing addr
    //a tile_kind is external if it is a mono_clone operation of an external tile or both a mono_clone operation a mono_clone operationed of one or many external tiles 
    
    class ExternalForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

    };

    class ForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ordinary_forwardinit_resolutor;
            const size_t ordinary_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> crit_forwardinit_resolutor;
            const size_t crit_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> external_forwardinit_resolutor;
            const size_t external_dispatch_sz;

        public:

            ForwardInitSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ordinary_forwardinit_resolutor,
                                       size_t ordinary_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> crit_forwardinit_resolutor,
                                       size_t crit_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> external_forwardinit_resolutor),
                                       size_t external_dispatch_sz noexcept: ordinary_forwardinit_resolutor(std::move(ordinary_forwardinit_resolutor)),
                                                                             ordinary_dispatch_sz(ordinary_dispatch_sz),
                                                                             crit_forwardinit_resolutor(std::move(crit_forwardinit_resolutor)),
                                                                             crit_dispatch_sz(crit_dispatch_sz), 
                                                                             external_forwardinit_resolutor(std::move(external_forwardinit_resolutor)),
                                                                             external_dispatch_sz(external_dispatch_sz){}
            
            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{
                
                auto ordinary_fwdinit_delivery_handle   = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->ordinary_forwardinit_resolutor.get(), this->ordinary_dispatch_sz));
                auto crit_fwdinit_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->crit_forwardinit_resolutor.get(), this->crit_dispatch_sz));
                auto external_fwdinit_delivery_handle   = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->external_forwardinit_resolutor.get(), this->external_dispatch_sz)); 

                for (size_t i = 0u; i < sz; ++i){
                    tile_kind_t tile_kind = dg::network_tile_member_safe_getsetter::get_tile_kind_nothrow(std::get<0>(ptr_arr[i]));

                    if (tile_kind == TILE_KIND_EXTERNAL_CLONE){
                        dg::network_producer_consumer::delvsrv_deliver(external_fwdinit_delivery_handle.get(), ptr_arr[i]);
                    } else if (tile_kind == TILE_KIND_CRIT){
                        dg::network_producer_consumer::delvsrv_deliver(crit_fwdinit_delivery_handle.get(), ptr_arr[i]);
                    } else{
                        dg::network_producer_consumer::delvsrv_deliver(ordinary_fwdinit_delivery_handle.get(), ptr_arr[i]);
                    }
                }
            }
    };

    class ExternalBackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<virtual_memory_commit_t>>> request_box;
            const size_t delivery_capacity;
            std::shared_ptr<UnifiedMemoryIPRetrieverInterface>  uma_ip_retriever;
        
        public:

            ExternalBackwardDoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<virtual_memory_commit_t>>> request_box,
                                              size_t delivery_capacity,
                                              std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever) noexcept: request_box(std::move(request_box)),
                                                                                                                             delivery_capacity(delivery_capacity),
                                                                                                                             uma_ip_retriever(std::move(uma_ip_retriever)){}
            

            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                for (size_t i = 0u; i < sz; ++i){
                    this->internal_resolve(std::get<0>(ptr_arr[i]), delivery_handle.get());
                }
            }
        
        private:

            void internal_resolve(uma_ptr_t ptr, dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_commit_t>> * handle) noexcept{

                uma_ptr_t rcu_addr          = dg::network_tile_member_safe_getsetter::get_extclone_rcu_addr_nothrow(ptr);
                auto lck_grd                = dg::network_memops_uma::memlock_guard(rcu_addr);
                size_t grad_buf_sz          = dg::network_tile_member_getsetter::get_extclone_grad_buf_sz_nothrow(ptr);
                uma_ptr_t grad_addr         = dg::network_tile_member_getsetter::get_extclone_grad_addr_nothrow(ptr);  
                uma_ptr_t requesting_addr   = dg::network_tile_member_getsetter::get_extclone_src_addr_nothrow(ptr);
                operatable_id_t ops_id      = dg::network_tile_member_getsetter::get_extclone_operatable_id_nothrow(ptr);
                Address requesting_ip       = this->uma_ip_retriever->ip(ptr);

                dg::network_std_container::string tmp_buf(dg::network_tile_member_getsetter::grad_buf_sz);
                dg::network_memops_uma::memcpy_uma_to_host(grad_add, tmp_buf.data(), grad_buf_sz);
                Request<external_virtual_memory_commit_t> request{requesting_ip, dg::network_external_memcommit_factory::make_event_external_backward_do_signal(requesting_addr, ops_id, std::move(tmp_buf))};
                dg::network_producer_consumer::delvsrv_deliver(handle, std::move(request));
                dg::network_tileops_handler::zero_grad(ptr);
            }
    };      

    //define ordinary - do precond in internal_resolve
    class OrdinaryBackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsuemrInterface<std::tuple<uma_ptr_t>>{

    };

    class BackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumterInterface<std::tuple<uma_ptr_t>>{

        private:

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ext_backward_resolutor;
            const size_t ext_bwd_dispatch_sz;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ord_backward_resolutor;
            const size_t ord_bwd_dispatch_sz;
        
        public:

            BackwardDoSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ext_backward_resolutor,
                                      size_t ext_bwd_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> ord_backward_resolutor,
                                      size_t ord_bwd_dispatch_sz) noexcept: ext_backward_resolutor(std::move(ext_backward_resolutor)),
                                                                            ext_bwd_dispatch_sz(ext_bwd_dispatch_sz),
                                                                            ord_backward_resolutor(std::move(ord_backward_resolutor)),
                                                                            ord_bwd_dispatch_sz(ord_bwd_dispatch_sz){}
            
            void push(std::tuple<uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto ext_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->ext_backward_resolutor.get(), this->ext_bwd_dispatch_sz));
                auto ord_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_raiihandle(this->ord_backward_resolutor.get(), this->ord_bwd_dispatch_sz));

                for (size_t i = 0u; i < sz; ++i){
                    tile_kind_t tile_kind = dg::network_tile_member_safe_getsetter::get_tile_kind_nothrow(std::get<0>(ptr_arr[i]));

                    if (tile_kind == TILE_KIND_EXTERNAL_CLONE){
                        dg::network_producer_consumer::delvsrv_deliver(ext_delivery_handle.get(), std::get<0>(ptr_arr[i]));
                    }

                    dg::network_producer_consumer::delvsrv_deliver(ord_delivery_handle.get(), std::get<0>(ptr_arr[i]));
                }
            }
    };
}

#endif