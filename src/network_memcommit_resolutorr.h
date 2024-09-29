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

    class ForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box;
            const size_t delivery_capacity; //this should be kept internally - 

        public:

            ForwardPingSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box,
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

                uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_rcu_addr_nothrow(ptr);
                auto lck_grd                = dg::network_memops_uma::memlock_guard(rcu_addr);
                init_status_t init_status   = dg::network_tile_member_getsetter::get_initialization_status_nothrow(ptr);

                if (init_status != INIT_STATUS_EMPTY){
                    return;
                }

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

                uma_ptr_t requestee_rcu_addr    = dg::network_tile_member_getsetter::get_rcu_addr_nothrow(requestee);
                uma_ptr_t requestor_rcu_addr    = dg::network_tile_member_getsetter::get_rcu_addr_nothrow(requestor);
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

                uma_ptr_t requestee_rcu_addr        = dg::network_tile_member_getsetter::get_rcu_addr_nothrow(requestee);
                uma_ptr_t requestor_rcu_addr        = dg::network_tile_member_getsetter::get_rcu_addr_nothrow(requestor);
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
                    virtual_memory_commit_t init_request = dg::network_memcommit_factory::make_event_forward_init_signal(requestee);
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
    
    class ForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            ForwardInitSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box,
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

    //

    class ForwardLoadRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

    };

    class ForwardLoadResponseResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:


    };

    class ForwardLoadRedirectResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

    };

    class BackwardReadySignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virutal_memory_commit_t>> request_box;

    };

    class BackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{ //this might not be necessary - optimizable 

        private:

            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_commit_t>> request_box;
    };

    class BackwardLoadRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

    };

    class BackwardLoadResponseResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:


    };

    class BackwardLoadRedirectRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

    };

}

#endif