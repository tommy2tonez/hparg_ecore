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

    //alrights - been doing technical analysis - and I think this would work out well - we are expecting 100TB cuda flops/core*s - with proper tuning + allocations + locality + memregion_frequency - we'll achieve the goal
    //code-wise - hard to read but clear and efficient - need to do branch profile guided optimizations - expecting at least 1 << 28 mem_events/core * s - we are dispatching 256x256 tiles
    //we solved 5 major problems:
    //backward prop synchronization
    //efficient cluster computation (forwards and backwards) (1 << 20 computation nodes) + data transfer by using memregion frequencies + cutf_ptr_t
    //memory problems - we aren't doing rumtime memories - we allocate things once and leave things be
    //concurrent training ingestion problems - we aren't waiting - we ingest things and invoke forward_init and leave things be 
    //box problem - this is the problem that most machine learning framework has - we do unit
    //diamond hands guys - we aint selling - we'll be rich

    //consider these scenrios:
    //(1): users ingest leafs - define forward_operatable_id, backward_operatable_id and super_memevent_operatable_id
    //(2): users allocate tiles: - define forward_operatable_id, backward_operatable_id and memevent_operatable_id 
    //(3): users ingest input and expected: immu and crit
    //(4): users want to train logits: - signal forward_inits at base (with memevent_operatable_id) or forward_pings at crit
    //(5): users reallocate tiles to train logits
    //(6): users extract leafs by using msgrfwd
        //- trainning successful in expected way - forward -> crit -> backward -> leaf update
        //- training not successful: things dont crash
        //- we need to time the update for all crit tiles to backprop to leafs - then define a circular training ingestion - says - tile0 -> tile1 -> tile2 -> tile1 -> tile0
        //                                                                                                                          tile0 -> tile11 -> tile12 -> tile11 -> tile0
        //                                                                                                                          tile0 -> tile1 -> tile2 -> tile1 -> tile0
        //pros: we dont wait for training to complete - we ingest the database at once to train - on 1 << 20 computation nodes

    //-------------------------
    //(7): users reingest leafs
    //(8): rinse and repeat
    //(9): users want to train groups of logits by using blkr or backward_operatable_id

    //-------------------------
    //(10): users want to alter path - user branch the current tree and do their operations

    //this prolly takes 20000 lines of raw code - including proper error handing + user logs + efficient tuning + hardware cache tuning + branch tuning
    //this is probably gonna take a week or two - but we'll be on time

    struct UnifiedMemoryIPRetrieverInterface{
        virtual ~UnifiedMemoryIPRetrieverInterface() noexcept = default;
        virtual auto ip(uma_ptr_t) noexcept -> std::expected<Address, exception_t> = 0;
    };

    struct HostIPRetrieverInterface{
        virtual ~HostIPRetrieverInterface() noexcept = default;
        virtual auto ip() noexcept -> Address = 0;
    };

    struct ForeignTileAliasGetterInterface{
        virtual ~ForeignTileAliasGetterInterface() noexcept = default;
        virtual auto alias(uma_ptr_t) noexcept -> std::expected<std::optional<uma_ptr_t>, exception_t> = 0; //reduce lock_collisions by using distributed hash_map
    };

    template <class T>
    struct Request{
        Address requestee;
        Address requestor;
        T content;
    };

    class ForwardPingLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

        public:

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                (void) event_arr;
            }
    };

    class ForwardPingImmuResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

        public:

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                (void) event_arr;
            }
    };

    class ForwardPingBlkrSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingBlkrSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity,
                                           size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity),
                                                                              vectorization_sz(vectorization_sz){}

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR = 1u;
                size_t max_possible_event_sz    = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_cap     = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_cap, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_blkr_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_blkr_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::reigon(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingSignalEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                   = event_arr[i].dst;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id; 
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_blkr_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_blkr_memevent_operatable_id_nothrow(ptr);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_blkr_descendant_nothrow(ptr);
                                virtual_memory_event_t event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, current_ops_id));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(event));
                                dg::network_tile_member_getsetter::set_blkr_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED); 
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingMonoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingMonoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity,
                                           size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity),
                                                                              vectorization_sz(vectorization_sz){}

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR = 1u;
                size_t max_possible_event_sz    = sz * EVENT_SCALE_FACTOR; 
                size_t trimmed_delivery_cap     = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_cap, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz); 
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingSignalEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                   = event_arr[i].dst;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_mono_memevent_operatable_id_nothrow(ptr);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant    = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(ptr);
                                auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, current_ops_id));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(decay_signal_event));
                                dg::network_tile_member_getsetter::set_mono_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPairSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

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

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 2u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR; 
                size_t trimmed_delivery_cap         = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_cap, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr      = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_region    = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_region, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingSignalEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                   = event_arr[i].dst;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pair_memevent_operatable_id_nothrow(ptr);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t left_descendant   = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(ptr);
                                uma_ptr_t right_descendant  = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(ptr);
                                auto decay_signal_event_1   = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant, ptr, current_ops_id));
                                auto decay_signal_event_2   = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant, ptr, current_ops_id));

                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(decay_signal_event_1));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(decay_signal_event_2));
                                dg::network_tile_member_getsetter::set_pair_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort(); 
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

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

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = UACM_ACM_SZ;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);                
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingSignalEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                       = event_arr[i].dst;
                        operatable_id_t expected_ops_id     = event_arr[i].operatable_id;
                        init_status_t init_status           = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_uacm_memevent_operatable_id_nothrow(ptr);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> descendant_arr(UACM_ACM_SZ);
                                dg::network_tile_member_getsetter::get_uacm_descendant_nothrow(ptr, descendant_arr.get());

                                for (size_t i = 0u; i < UACM_ACM_SZ; ++i){
                                    auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant_arr[i], ptr, current_ops_id));
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(decay_signal_event));
                                }

                                dg::network_tile_member_getsetter::set_uacm_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

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

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{
                
                const size_t EVENT_SCALE_FACTOR     = PACM_ACM_SZ * 2;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingSignalEvent>{

                dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                       = event_arr[i].dst;
                        operatable_id_t expected_ops_id     = event_arr[i].operatable_id;
                        init_status_t init_status           = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_pacm_memevent_operatable_id_nothrow(ptr); //this is memevent_operatable_id_t + forward_operatable_id + backward_operatable_id

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> left_descendant_arr(PACM_ACM_SZ);
                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> right_descendant_arr(PACM_ACM_SZ);
                                dg::network_tile_member_getsetter::get_pacm_left_descendant_nothrow(ptr, left_descendant_arr.get());
                                dg::network_tile_member_getsetter::get_pacm_right_descendant_nothrow(ptr, right_descendant_arr.get());

                                for (size_t i = 0u; i < PACM_ACM_SZ; ++i){
                                    auto decay_signal_event_1 = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant_arr[i], ptr, current_ops_id));
                                    auto decay_signal_event_2 = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant_arr[i], ptr, current_ops_id));
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(decay_signal_event_1));
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(decay_signal_event_2));
                                }

                                dg::network_tile_member_getsetter::set_pacm_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }    
                    }
                }
            };
    };

    class ForwardPingExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

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

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz); 
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingSignalEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                       = event_arr[i].dst;
                        operatable_id_t expected_ops_id     = event_arr[i].operatable_id;
                        init_status_t init_status           = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_extnsrc_memevent_operatable_id_nothrow(ptr);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant    = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(ptr);
                                auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, current_ops_id));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(decay_signal_event));
                                dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

        private:

            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingExtnDstSignalResolutor(std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                              std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                              size_t delivery_capacity,
                                              size_t vectorization_sz) noexcept: uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                 host_ip_retriever(std::move(host_ip_retriever)),
                                                                                 request_box(std::move(request_box)),
                                                                                 delivery_capacity(delivery_capacity),
                                                                                 vectorization_sz(vectorization_sz){}

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.uma_ip_retriever             = this->uma_ip_retriever->get();
                    internal_resolutor.host_ip_retriever            = this->host_ip_retriever->get();
                    internal_resolutor.request_delivery_handle      = delivery_handle.get();
                    size_t trimmed_vectorization_sz                 = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz); 
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get())); //we are risking 0s - we will fix this later

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingSignalEvent>{

                UnifiedMemoryIPRetrieverInterface * uma_ip_retriever;
                HostIPRetrieverInterface * host_ip_retriever;
                dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                       = event_arr[i].dst;
                        operatable_id_t expected_ops_id     = event_arr[i].operatable_id;
                        init_status_t init_status           = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_extndst_memevent_operatable_id_nothrow(ptr);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t counterpart   = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(ptr);
                                auto request            = Request<external_virtual_memory_event_t>{};
                                request.requestee       = this->uma_ip_retriever->ip(counterpart);
                                request.requestor       = this->host_ip_retriever->ip();
                                request.content         = dg::network_external_memcommit_factory::virtualize_event(dg::network_external_memcommit_factory::make_event_forward_pingpong_request(counterpart, ptr, current_ops_id));

                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(request));
                                dg::network_tile_member_getsetter::set_extndst_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

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

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get())); //concurrent memory is sensitive Mom - you code kernel for 40 years

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingSignalEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                       = event_arr[i].dst;
                        operatable_id_t expected_ops_id     = event_arr[i].operatable_id;
                        init_status_t init_status           = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_crit_memevent_operatable_id_nothrow(ptr);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant    = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(ptr);
                                auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, current_ops_id));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(decay_signal_event));
                                dg::network_tile_member_getsetter::set_crit_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingMsgrFwdResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingMsgrFwdResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                        size_t delivery_capacity,
                                        size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                           delivery_capacity(delivery_capacity),
                                                                           vectorization_sz(vectorization_sz){}

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{
                
                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingSignalEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                       = event_arr[i].dst;
                        operatable_id_t expected_ops_id     = event_arr[i].operatable_id;
                        init_status_t init_status           = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_msgrfwd_memevent_operatable_id_nothrow(ptr);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant    = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(ptr);
                                auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, current_ops_id));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(decay_signal_event));
                                dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingMsgrBwdResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

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

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz); 
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingSignalEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                       = event_arr[i].dst;
                        operatable_id_t expected_ops_id     = event_arr[i].operatable_id;
                        init_status_t init_status           = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_msgrbwd_memevent_operatable_id_nothrow(ptr);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant    = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(ptr);
                                auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, current_ops_id));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(decay_signal_event));
                                dg::network_tile_member_getsetter::set_msgrbwd_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> blkr_resolutor;
            const size_t blkr_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> mono_resolutor;
            const size_t mono_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> pair_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> immu_resolutor;
            const size_t immu_dispatch_sz;

        public:

            ForwardPingSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> leaf_resolutor,
                                       size_t leaf_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> blkr_resolutor,
                                       size_t blkr_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> mono_resolutor,
                                       size_t mono_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> pair_resolutor,
                                       size_t pair_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> uacm_resolutor,
                                       size_t uacm_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> pacm_resolutor,
                                       size_t pacm_dispatch_sz,
                                       std::unique_Ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> extnsrc_resolutor,
                                       size_t extnsrc_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> extndst_resolutor,
                                       size_t extndst_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> crit_resolutor,
                                       size_t crit_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> msgrfwd_resolutor,
                                       size_t msgrfwd_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> msgrbwd_resolutor,
                                       size_t msgrbwd_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> immu_resolutor,
                                       size_t immu_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
                                                                          leaf_dispatch_sz(leaf_dispatch_sz),
                                                                          blkr_resolutor(std::move(blkr_resolutor)),
                                                                          blkr_dispatch_sz(blkr_dispatch_sz),
                                                                          mono_resolutor(std::move(mono_resolutor)),
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
                                                                          msgrbwd_dispatch_sz(msgrbwd_dispatch_sz),
                                                                          immu_resolutor(std::move(immu_resolutor)),
                                                                          immu_dispatch_sz(immu_dispatch_sz){}

            void push(ForwardPingSignalEvent * event_arr, size_t sz) noexcept{

                size_t trimmed_leaf_dispatch_sz = std::min(this->leaf_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> leaf_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->leaf_resolutor.get(), trimmed_leaf_dispatch_sz));

                size_t trimmed_blkr_dispatch_sz = std::min(this->blkr_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> blkr_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->blkr_resolutor.get(), trimmed_blkr_dispatch_sz));

                size_t trimmed_mono_dispatch_sz = std::min(this->mono_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> mono_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->mono_resolutor.get(), trimmed_mono_dispatch_sz));

                size_t trimmed_pair_dispatch_sz = std::min(this->pair_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> pair_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->pair_resolutor.get(), trimmed_pair_dispatch_sz)); 

                size_t trimmed_uacm_dispatch_sz = std::min(this->uacm_dispatch_sz, sz);;
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> uacm_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->uacm_resolutor.get(), trimmed_uacm_dispatch_sz));

                size_t trimmed_pacm_dispatch_sz = std::min(this->pacm_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> pacm_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->pacm_resolutor.get(), trimmed_pacm_dispatch_sz));

                size_t trimmed_crit_dispatch_sz = std::min(this->crit_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> crit_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->crit_resolutor.get(), trimmed_crit_dispatch_sz)); 

                size_t trimmed_immu_dispatch_sz = std::min(this->immu_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> immu_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->immu_resolutor.get(), trimmed_immu_dispatch_sz)); 

                size_t trimmed_extnsrc_dispatch_sz = std::min(this->extnsrc_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extnsrc_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extnsrc_resolutor.get(), trimmed_extnsrc_dispatch_sz));

                size_t trimmed_extndst_dispatch_sz = std::min(this->extndst_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extndst_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extndst_resolutor.get(), trimmed_extndst_dispatch_sz)); 

                size_t trimmed_msgrfwd_dispatch_sz = std::min(this->msgrfwd_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> msgrfwd_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->msgrfwd_resolutor.get(), trimmed_msgrfwd_dispatch_sz));

                size_t trimmed_msgrbwd_dispatch_sz = std::min(this->msgrbwd_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> msgrbwd_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->msgrbwd_resolutor.get(), trimmed_msgrbwd_dispatch_sz));

                auto leaf_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->leaf_resolutor.get(), trimmed_leaf_dispatch_sz, leaf_dh_mem.get()));
                auto blkr_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->blkr_resolutor.get(), trimmed_blkr_dispatch_sz, blkr_dh_mem.get()));
                auto mono_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->mono_resolutor.get(), trimmed_mono_dispatch_sz, mono_dh_mem.get()));
                auto pair_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->pair_resolutor.get(), trimmed_pair_dispatch_sz, pair_dh_mem.get()));
                auto uacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->uacm_resolutor.get(), trimmed_uacm_dispatch_sz, uacm_dh_mem.get()));
                auto pacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->pacm_resolutor.get(), trimmed_pacm_dispatch_sz, pacm_dh_mem.get()));
                auto crit_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->crit_resolutor.get(), trimmed_crit_dispatch_sz, crit_dh_mem.get()));
                auto immu_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->immu_resolutor.get(), trimmed_immu_dispatch_sz, immu_dh_mem.get()));
                auto extnsrc_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extnsrc_resolutor.get(), trimmed_extnsrc_dispatch_sz, extnsrc_dh_mem.get()));
                auto extndst_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extndst_resolutor.get(), trimmed_extndst_dispatch_sz, extndst_dh_mem.get()));
                auto msgrfwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrfwd_resolutor.get(), trimmed_msgrfwd_dispatch_sz, msgrfwd_dh_mem.get()));
                auto msgrbwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrbwd_resolutor.get(), trimmed_msgrbwd_dispatch_sz, msgrbwd_dh_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(event_arr[i].dst); 

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(leaf_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_BLKR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(blkr_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MONO:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(mono_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_PAIR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pair_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_UACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(uacm_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_PACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pacm_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_CRIT:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(crit_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_IMMU:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(immu_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNSRC:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extnsrc_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNDST:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extndst_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRFWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrfwd_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRBWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrbwd_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        default:
                        {
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    }
                }
            }
    };

    //---

    class ForwardPongLeafRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPongLeafRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity,
                                            size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity),
                                                                               vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_leaf_memevent_operatable_id_nothrow(requestee);

                        if (!is_subset_id(expected_ops_id, current_ops_id)){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, expected_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        } 
                    }
                }
            };
    };

    class ForwardPongImmuRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPongImmuRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity,
                                            size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity),
                                                                               vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get())); //

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_immu_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_immu_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_immu_memevent_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){ //I'm tempted to make this an exception like leaf but it's too confusing
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPongBlkrRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;
        
        public:

            ForwardPongBlkrRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity,
                                            size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity),
                                                                               vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++I){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_blkr_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_blkr_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_blkr_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_blkr_memevent_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_blkr_push_observer_nothrow(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPongMonoRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPongMonoRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity,
                                            size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity),
                                                                               vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_mono_memevent_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_mono_push_observer_nothrow(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPongPairRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz; 

        public:

            ForwardPongPairRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity,
                                            size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity),
                                                                               vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR; 
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pair_memevent_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_pair_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPongUACMRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz; 

        public:

            ForwardPongUACMRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity,
                                            size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity),
                                                                               vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{
                
                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get())); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_uacm_memevent_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_uacm_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPongPACMRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPongPACMRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity,
                                            size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity),
                                                                               vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pacm_memevent_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_pacm_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPongExtnSrcRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        public:

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                (void) event_arr;
            }
    };

    class ForwardPongExtnDstRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPongExtnDstRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               size_t delivery_capacity,
                                               size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity),
                                                                                  vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz); 
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_extndst_memevent_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_extndst_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPongCritRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPongCritRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity,
                                            size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity),
                                                                               vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_crit_memevent_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_crit_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPongMsgrFwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPongMsgrFwdRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               size_t delivery_capacity,
                                               size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity),
                                                                                  vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrfwd_memevent_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_msgrfwd_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPongMsgrBwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPongMsgrBwdRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               size_t delivery_capacity,
                                               size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity),
                                                                                  vectorization_sz(vectorization_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz); 
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrbwd_memevent_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_msgrbwd_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPongRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> blkr_resolutor;
            const size_t blkr_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> mono_resolutor;
            const size_t mono_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> pair_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> immu_resolutor;
            const size_t immu_dispatch_sz;

        public:

            ForwardPongRequestResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> leaf_resolutor,
                                        size_t leaf_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::Consumerinterface<ForwardPongRequestEvent>> blkr_resolutor,
                                        size_t blkr_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> mono_resolutor,
                                        size_t mono_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> pair_resolutor,
                                        size_t pair_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> uacm_resolutor,
                                        size_t uacm_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> pacm_resolutor,
                                        size_t pacm_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> extnsrc_resolutor,
                                        size_t extnsrc_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> extndst_resolutor,
                                        size_t extndst_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> crit_resolutor,
                                        size_t crit_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> msgrfwd_resolutor,
                                        size_t msgrfwd_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> msgrbwd_resolutor,
                                        size_t msgrbwd_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> immu_resolutor,
                                        size_t immu_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
                                                                           leaf_dispatch_sz(leaf_dispatch_sz),
                                                                           blkr_resolutor(std::move(blkr_resolutor)),
                                                                           blkr_dispatch_sz(blkr_dispatch_sz),
                                                                           mono_resolutor(std::move(mono_resolutor)),
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
                                                                           msgrbwd_dispatch_sz(msgrbwd_dispatch_sz),
                                                                           immu_resolutor(std::move(immu_resolutor)),
                                                                           immu_dispatch_sz(immu_dispatch_sz){}

            void push(ForwardPongRequestEvent * event_arr, size_t sz) noexcept{

                size_t trimmed_leaf_dispatch_sz = std::min(this->leaf_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> leaf_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->leaf_resolutor.get(), trimmed_leaf_dispatch_sz));

                size_t trimmed_blkr_dispatch_sz = std::min(this->blkr_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> blkr_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->blkr_resolutor.get(), trimmed_blkr_dispatch_sz)); 

                size_t trimmed_mono_dispatch_sz = std::min(this->mono_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> mono_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->mono_resolutor.get(), trimmed_mono_dispatch_sz));

                size_t trimmed_pair_dispatch_sz = std::min(this->pair_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> pair_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->pair_resolutor.get(), trimmed_pair_dispatch_sz));

                size_t trimmed_uacm_dispatch_sz = std::min(this->uacm_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> uacm_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->uacm_resolutor.get(), trimmed_uacm_dispatch_sz));

                size_t trimmed_pacm_dispatch_sz = std::min(this->pacm_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> pacm_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->pacm_resolutor.get(), trimmed_pacm_dispatch_sz));

                size_t trimmed_crit_dispatch_sz = std::min(this->crit_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> crit_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->crit_resolutor.get(), trimmed_crit_dispatch_sz));

                size_t trimmed_immu_dispatch_sz = std::min(this->immu_dispatch_sz, sz); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> immu_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->immu_resolutor.get(), trimmed_immu_dispatch_sz));

                size_t trimmed_extnsrc_dispatch_sz = std::min(this->extnsrc_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extnsrc_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extnsrc_resolutor.get(), trimmed_extnsrc_dispatch_sz));

                size_t trimmed_extndst_dispatch_sz = std::min(this->extndst_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extndst_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extndst_resolutor.get(), trimmed_extndst_dispatch_sz));

                size_t trimmed_msgrfwd_dispatch_sz = std::min(this->msgrfwd_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> msgrfwd_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->msgrfwd_resolutor.get(), trimmed_msgrfwd_dispatch_sz));

                size_t trimmed_msgrbwd_dispatch_sz = std::min(this->msgrbwd_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> msgrbwd_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->msgrbwd_resolutor.get(), trimmed_msgrbwd_dispatch_sz));

                auto leaf_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->leaf_resolutor.get(), trimmed_leaf_dispatch_sz, leaf_dh_mem.get()));
                auto blkr_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->blkr_resolutor.get(), trimmed_blkr_dispatch_sz, blkr_dh_mem.get()));
                auto mono_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->mono_resolutor.get(), trimmed_mono_dispatch_sz, mono_dh_mem.get()));
                auto pair_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->pair_resolutor.get(), trimmed_pair_dispatch_sz, pair_dh_mem.get()));
                auto uacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->uacm_resolutor.get(), trimmed_uacm_dispatch_sz, uacm_dh_mem.get()));
                auto pacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->pacm_resolutor.get(), trimmed_pacm_dispatch_sz, pacm_dh_mem.get()));
                auto crit_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->crit_resolutor.get(), trimmed_crit_dispatch_sz, crit_dh_mem.get()));
                auto immu_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->immu_resolutor.get(), trimmed_immu_dispatch_sz, immu_dh_mem.get()));
                auto extnsrc_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extnsrc_resolutor.get(), trimmed_extnsrc_dispatch_sz, extnsrc_dh_mem.get()));
                auto extndst_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extndst_resolutor.get(), trimmed_extndst_dispatch_sz, extndst_dh_mem.get()));
                auto msgrfwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrfwd_resolutor.get(), trimmed_msgrfwd_dispatch_sz, msgrfwd_dh_mem.get()));
                auto msgrbwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrbwd_resolutor.get(), trimmed_msgrbwd_dispatch_sz, msgrbwd_dh_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(event_arr[i].requestee);

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(leaf_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_BLKR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(blkr_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MONO:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(mono_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_PAIR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pair_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_UACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(uacm_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_PACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pacm_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNSRC:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extnsrc_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNDST:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extndst_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_CRIT:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(crit_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRFWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrfwd_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRBWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrbwd_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_IMMU:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(immu_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        default:
                        {
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    }
                }
            }
    };

    //

    class ForwardPingPongLeafRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz; 

        public:

            ForwardPingPongLeafRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                size_t EVENT_SCALE_FACTOR           = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_leaf_operatable_id_nothrow(requestee);

                        if (!is_subset_id(expected_ops_id, current_ops_id)){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, expected_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongImmuRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongImmuRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR         = 1u;
                size_t max_possible_event_sz            = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity        = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost               = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_immu_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_immu_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_immu_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongBlkrRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;
        
        public:

            ForwardPingPongBlkrRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_blkr_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_blkr_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_blkr_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_blkr_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_blkr_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_blkr_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, current_ops_id)));
                                dg::network_tile_member_getsetter::set_blkr_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongMonoRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongMonoRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR         = 1u;
                size_t max_possible_event_sz            = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity        = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost               = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);   
                auto delivery_handle                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_mono_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_mono_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, current_ops_id)));
                                dg::network_tile_member_getsetter::set_mono_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongPairRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongPairRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 2u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pair_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_pair_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t left_descendant   = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(requestee);
                                uma_ptr_t right_descendant  = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant, requestee, current_ops_id)));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant, requestee, current_ops_id)));
                                dg::network_tile_member_getsetter::set_pair_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongUACMRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongUACMRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = UACM_ACM_SZ;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get())); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_uacm_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_uacm_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> descendant_arr(UACM_ACM_SZ);
                                dg::network_tile_member_getsetter::get_uacm_descendant_nothrow(ptr, descendant_arr.get());

                                for (size_t i = 0u; i < UACM_ACM_SZ; ++i){
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant_arr[i], requestee, current_ops_id)));
                                }

                                dg::network_tile_member_getsetter::set_uacm_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongPACMRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongPACMRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = PACM_ACM_SZ * 2;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pacm_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_pacm_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> left_descendant_arr(PACM_ACM_SZ);
                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> right_descendant_arr(PACM_ACM_SZ);
                                dg::network_tile_member_getsetter::get_pacm_left_descendant_nothrow(requestee, left_descendant_arr.get());
                                dg::network_tile_member_getsetter::get_pacm_right_descendant_nothrow(requestee, right_descendant_arr.get());

                                for (size_t i = 0u; i < PACM_ACM_SZ; ++i){
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant_arr[i], requestee, current_ops_id)));
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant_arr[i], requestee, current_ops_id)));
                                }

                                dg::network_tile_member_getsetter::set_pacm_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongCritRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongCritRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                size_t delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   delivery_capacity(delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_crit_operatable_id_nothrow(requestee); 

                        if (expected_ops_id != current_ops_id){ //this is not clear
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_crit_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED; [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, current_ops_id)));
                                dg::network_tile_member_getsetter::set_crit_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongExtnSrcRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongExtnSrcRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                   size_t delivery_capacity,
                                                   size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                      delivery_capacity(delivery_capacity),
                                                                                      vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR         = 1u;
                size_t max_possible_event_sz            = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity        = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost               = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto delivery_handle                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_extnsrc_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_extnsrc_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, current_ops_id)));
                                dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongExtnDstRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> outbound_request_box;
            const size_t request_delivery_capacity;
            const size_t outbound_delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongExtnDstRequestResolutor(std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                   std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                   std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                   std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> outbound_request_box,
                                                   size_t request_delivery_capacity,
                                                   size_t outbound_delivery_capacity,
                                                   size_t vectorization_sz) noexcept: uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                      host_ip_retriever(std::move(host_ip_retriever)),
                                                                                      request_box(std::move(request_box)),
                                                                                      outbound_request_box(std::move(outbound_request_box)),
                                                                                      request_delivery_capacity(request_delivery_capacity),
                                                                                      outbound_delivery_capacity(outbound_delivery_capacity),
                                                                                      vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t INTERAL_EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_internal_event_sz       = sz * INTERNAL_EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_internal_event_sz); 
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                const size_t EXTERNAL_EVENT_SCALE_FACTOR    = 1u;
                size_t max_possible_external_event_sz       = sz * EXTERNAL_EVENT_SCALE_FACTOR;
                size_t trimmed_outbound_delivery_capacity   = std::min(this->oubound_delivery_capacity, max_possible_event_sz);
                size_t odh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->outbound_request_box.get(), trimmed_outbound_delivery_capacity); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> odh_mem(odh_allocation_cost);
                auto outbound_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->outbound_request_box.get(), trimmed_outbound_delivery_capacity, odh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.uma_ip_retriever         = this->uma_ip_retriever.get();
                    internal_resolutor.host_ip_retriever        = this->host_ip_retriever.get();
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.outbound_delivery_handle = outbound_delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz); 
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                UnifiedMemoryIPRetrieverInterface * uma_ip_retriever;
                HostIPRetrieverInterface * host_ip_retriever;
                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * outbound_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_extndst_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_extndst_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }   
                            }
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t counterpart   = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(requestee);
                                auto ping_request       = Request<external_virtual_memory_event_t>{};
                                ping_request.to         = this->uma_ip_retriever->ip(counterpart);
                                ping_request.fr         = this->host_ip_retriever->ip();
                                ping_request.content    = dg::network_external_memcommit_factory::make_event_forward_ping_signal(counterpart, current_ops_id);

                                dg::network_producer_consumer::delvrsrv_deliver(this->outbound_delivery_handle, std::move(ping_request));
                                dg::network_tile_member_getsetter::set_extndst_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongMsgrFwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongMsgrFwdRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                   size_t delivery_capacity,
                                                   size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                      delivery_capacity(delivery_capacity),
                                                                                      vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrfwd_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_msgrfwd_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, current_ops_id)));
                                dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongMsgrBwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongMsgrBwdRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                   size_t delivery_capacity,
                                                   size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                      delivery_capacity(delivery_capacity),
                                                                                      vectorization_sz(vectorization_sz){}

            void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(event_arr[i].requestee);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(event_arr[i].requestee);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ForwardPingPongRequestEvent>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t requestee             = event_arr[i].requestee;
                        uma_ptr_t requestor             = event_arr[i].requestor;
                        operatable_id_t expected_ops_id = event_arr[i].operatable_id;
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(requestee);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrbwd_operatable_id_nothrow(requestee);

                        if (expected_ops_id != current_ops_id){
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_msgrbwd_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(requestor, current_ops_id)));
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, current_ops_id)));
                                dg::network_tile_member_getsetter::set_msgrbwd_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
    };

    class ForwardPingPongRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> leaf_pingpong_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> blkr_pingpong_resolutor;
            const size_t blkr_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> mono_pingpong_resolutor;
            const size_t mono_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> pair_pingpong_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> uacm_pingpong_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> pacm_pingpong_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> extnsrc_pingpong_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> extndst_pingpong_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> crit_pingpong_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> msgrfwd_pingpong_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> msgrbwd_pingpong_resolutor;
            const size_t msgrbwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> immu_pingpong_resolutor;
            const size_t immu_dispatch_sz;

        public:

            ForwardPingPongRequestResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> leaf_pingpong_resolutor,
                                            size_t leaf_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> blkr_pingpong_resolutor,
                                            size_t blkr_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> mono_pingpong_resolutor,
                                            size_t mono_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> pair_pingpong_resolutor,
                                            size_t pair_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> uacm_pingpong_resolutor,
                                            size_t uacm_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> pacm_pingpong_resolutor,
                                            size_t pacm_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> extnsrc_pingpong_resolutor,
                                            size_t extnsrc_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> extndst_pingpong_resolutor,
                                            size_t extndst_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> crit_pingpong_resolutor,
                                            size_t crit_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> msgrfwd_pingpong_resolutor,
                                            size_t msgrfwd_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> msgrbwd_pingpong_resolutor,
                                            size_t msgrbwd_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> immu_pingpong_resolutor,
                                            size_t immu_dispatch_sz) noexcept: leaf_pingpong_resolutor(std::move(leaf_pingpong_resolutor)),
                                                                               leaf_dispatch_sz(leaf_dispatch_sz),
                                                                               blkr_pingpong_resolutor(std::move(blkr_pingpong_resolutor)),
                                                                               blkr_dispatch_sz(blkr_dispatch_sz),
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
                                                                               msgrbwd_dispatch_sz(msgrbwd_dispatch_sz),
                                                                               immu_pingpong_resolutor(std::move(immu_pingpong_resolutor)),
                                                                               immu_dispatch_sz(immu_dispatch_sz){}

        void push(ForwardPingPongRequestEvent * event_arr, size_t sz) noexcept{

            size_t trimmed_leaf_dispatch_sz     = std::min(this->leaf_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> leaf_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->leaf_pingpong_resolutor.get(), trimmed_leaf_dispatch_sz));

            size_t trimmed_blkr_dispatch_sz     = std::min(this->blkr_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> blkr_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->blkr_pingpong_resolutor.get(), trimmed_blkr_dispatch_sz)); 

            size_t trimmed_mono_dispatch_sz     = std::min(this->mono_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> mono_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->mono_pingpong_resolutor.get(), trimmed_mono_dispatch_sz));

            size_t trimmed_pair_dispatch_sz     = std::min(this->pair_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> pair_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->pair_pingpong_resolutor.get(), trimmed_pair_dispatch_sz));

            size_t trimmed_uacm_dispatch_sz     = std::min(this->uacm_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> uacm_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->uacm_pingpong_resolutor.get(), trimmed_uacm_dispatch_sz));

            size_t trimmed_pacm_dispatch_sz     = std::min(this->pacm_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> pacm_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->pacm_pingpong_resolutor.get(), trimmed_pacm_dispatch_sz));

            size_t trimmed_crit_dispatch_sz     = std::min(this->crit_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> crit_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->crit_pingpong_resolutor.get(), trimmed_crit_dispatch_sz));

            size_t trimmed_immu_dispatch_sz     = std::min(this->immu_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> immu_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->immu_pingpong_resolutor.get(), trimmed_immu_dispatch_sz));

            size_t trimmed_extnsrc_dispatch_sz  = std::min(this->extnsrc_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> extnsrc_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extnsrc_pingpong_resolutor.get(), trimmed_extnsrc_dispatch_sz));

            size_t trimmed_extndst_dispatch_sz  = std::min(this->extndst_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> extndst_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extndst_pingpong_resolutor.get(), trimmed_extndst_dispatch_sz));

            size_t trimmed_msgrfwd_dispatch_sz  = std::min(this->msgrfwd_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> msgrfwd_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->msgrfwd_pingpong_resolutor.get(), trimmed_msgrfwd_dispatch_sz));

            size_t trimmed_msgrbwd_dispatch_sz  = std::min(this->msgrbwd_dispatch_sz, sz);
            dg::network_stack_allocation::NoExceptRawAllocation<char[]> msgrbwd_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->msgrbwd_pingpong_resolutor.get(), trimmed_msgrbwd_dispatch_sz));

            auto leaf_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->leaf_pingpong_resolutor.get(), trimmed_leaf_dispatch_sz, leaf_dh_mem.get()));
            auto mono_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->mono_pingpong_resolutor.get(), trimmed_mono_dispatch_sz, mono_dh_mem.get()));
            auto blkr_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->blkr_pingpong_resolutor.get(), trimmed_blkr_dispatch_sz, blkr_dh_mem.get()));
            auto pair_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->pair_pingpong_resolutor.get(), trimmed_pair_dispatch_sz, pair_dh_mem.get()));
            auto uacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->uacm_pingpong_resolutor.get(), trimmed_uacm_dispatch_sz, uacm_dh_mem.get()));
            auto pacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->pacm_pingpong_resolutor.get(), trimmed_pacm_dispatch_sz, pacm_dh_mem.get()));
            auto crit_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->crit_pingpong_resolutor.get(), trimmed_crit_dispatch_sz, crit_dh_mem.get()));
            auto immu_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->immu_pingpong_resolutor.get(), trimmed_immu_dispatch_sz, immu_dh_mem.get()));
            auto extnsrc_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extnsrc_pingpong_resolutor.get(), this->extnsrc_dispatch_sz));
            auto extndst_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extndst_pingpong_resolutor.get(), this->extndst_dispatch_sz));
            auto msgrfwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrfwd_pingpong_resolutor.get(), this->msgrfwd_dispatch_sz));
            auto msgrbwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrbwd_pingpong_resolutor.get(), this->msgrbwd_dispatch_sz));

            for (size_t i = 0u; i < sz; ++i){
                std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(event_arr[i].requestee);

                if (!tile_kind.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                    continue;
                }

                switch (tile_kind.value()){
                    case TILE_KIND_LEAF:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(leaf_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    case TILE_KIND_BLKR:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(blkr_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    case TILE_KIND_MONO:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(mono_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    case TILE_KIND_PAIR:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(pair_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    case TILE_KIND_UACM:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(uacm_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    case TILE_KIND_PACM:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(pacm_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    case TILE_KIND_EXTNSRC:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(extnsrc_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    case TILE_KIND_EXTNDST:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(extndst_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    case TILE_KIND_CRIT:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(crit_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    case TILE_KIND_MSGRFWD:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(msgrfwd_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    case TILE_KIND_MSGRBWD:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(msgrbwd_delivery_handle.get(), event_arr[i]);
                        break;                        
                    }
                    case TILE_KIND_IMMU:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(immu_delivery_handle.get(), event_arr[i]);
                        break;
                    }
                    default:
                    {
                        if constexpr(DEBUG_MODE_FLAG){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                }
            }
        }
    };

    //

    class ForwardDoLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        public:

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                (void) event_arr;
            }
    };

    class ForwardDoMonoSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t addrfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoMonoSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                           std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                           size_t request_delivery_capacity,
                                           size_t addrfetch_vectorization_sz,
                                           size_t region_vectorization_sz,
                                           size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                      cuda_async_device(std::move(cuda_async_device)),
                                                                                      host_async_device(std::move(host_async_device)),
                                                                                      request_delivery_capacity(request_delivery_capacity),
                                                                                      addrfetch_vectorization_sz(addrfetch_vectorization_sz),
                                                                                      region_vectorization_sz(region_vectorization_sz),
                                                                                      forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<uma_ptr_t>[]> descendant_arr(sz);
                const size_t EVENT_SCALE_FACTOR             = MAX_OBSERVER_ARR_SZ;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    InternalDescendantAddressFetcher fetcher    = {};
                    size_t trimmed_addrfetch_vectorization_sz   = std::min(this->addrfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_addrfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_addrfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(event_arr[i].dst, event_arr[i].operatable_id, std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(descendant_arr[i].value()); //polymorphic access - safeguards

                        if (!src_rcu_addr.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(src_rcu_addr.error()));
                            continue;
                        }

                        size_t lck_region_sz    = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].dst); //descendant_arr[i] has value => event_arr[i].dst is safe access - assumption
                        uma_ptr_t dst_lck_addr  = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        uma_ptr_t src_lck_addr  = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        auto key                = std::make_tuple(dst_lck_addr, src_lck_addr); //we rather use non_unique_representations

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(event_arr[i].dst, descendant_arr[i].value(), event_arr[i].operatable_id));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, operatable_id_t, std::optional<uma_ptr_t> *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, operatable_id_t, std::optional<uma_ptr_t> *> data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [ptr, expected_ops_id, fecthing_ptr]   = data_arr[i];
                        init_status_t init_status                   = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id              = dg::network_tile_member_getsetter::get_mono_operatable_id_nothrow(ptr);

                        if (expected_ops_id != current_ops_id){
                            *fetching_ptr = std::nullopt;
                            continue;
                        } 

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_ptr   = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_ptr   = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(ptr);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::KVConsumerInterface<cuda_ptr_t, std::tuple<cuda_ptr_t, cuda_tileops_dispatch_control_t>>{

                dg::network_cuda_controller::CudaSynchronizer * synchronizer;
                dg::network_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;

                void push(cuda_ptr_t key, std::tuple<cuda_ptr_t, cuda_ptr_t, cuda_tileops_dispatch_control_t> * data_arr, size_t sz) noexcept{
                    
                    size_t dispatching_cudaptr_vec_sz = sz * 2;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> dispatching_cudaptr_vec(dispatching_cudaptr_vec_sz);
                    auto aggregator = dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::aggregator_raiispawn_mono(sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst_logit_cudaptr, src_logit_cudaptr, tileops_dp_code]    = data_arr[i];
                        dispatching_cudaptr_vec[i * 2]                                  = dst_logit_cudaptr;
                        dispatching_cudaptr_vec[i * 2  + 1]                             = src_logit_cudaptr;
                        dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::aggregator_add(aggregator, dst_logit_cudaptr, src_logit_cudaptr, tileops_dp_code));
                    }

                    auto executable = [arg = std::move(aggregator)]() noexcept{
                        dg::network_tileops_cuda_poly::aggregator_exec(arg);
                    };

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(dispatching_cudaptr_vec.get(), std::next(dispatching_cudaptr_vec.get(), dispatching_cudaptr_vec_sz)));
                    auto async_task = dg::network_cuda_controller::virtualize_async_task(std::move(executable));
                    auto async_id   = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(async_task))); //this requires an error issue - we'll work on this later - next sprint
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(async_id));
                }
            };

            struct InternalHostResolutor: dg::network_producer_consumer::KVConsumerInterface<host_ptr_t, std::tuple<host_ptr_t, host_tileops_dispatch_control_t>>{

                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;

                void push(host_ptr_t key, std::tuple<host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t> * data_arr, size_t sz) noexcept{

                    size_t dispatching_hostptr_vec_sz   = sz * 2;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> dispatching_host_ptr_vec(dispatching_hostptr_vec_sz);
                    auto aggregator = dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_raiispawn_mono(sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst_logit_hostptr, src_logit_hostptr, tileops_dp_code]    = data_arr[i];
                        dispatching_hostptr_vec[i * 2]                                  = dst_logit_hostptr;
                        dispatching_hostptr_vec[i * 2 + 1]                              = src_logit_hostptr;
                        dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_add(aggregator, dst_logit_hostptr, src_logit_hostptr, tileops_dp_code));
                    }

                    auto executable = [arg = std::move(aggregator)]() noexcept{
                        dg::network_tileops_host_poly::aggregator_exec(arg);
                    };

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(dispatching_hostptr_vec.get(), std::next(dispatching_hostptr_vec.get(), dispatching_hostptr_vec_sz)));
                    auto async_task     = dg::network_host_asynchronous::virtualize_async_task(std::move(executable));
                    auto async_id       = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(async_task))); //this requires an error issue - we'll work on this later - next sprint
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(async_id));
                }
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t, operatable_id_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_emmory_event_t> * request_delivery_handle;
                size_t vectorization_sz;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));

                    auto umamap_reacquirer                          = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto dst_vmamap_reacquirer                      = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_vmamap_reacquirer                      = dg::network_vmamap::reacquirer_raii_initialize();
                    auto cuda_synchronizer                          = dg::network_cuda_controller::CudaSynchronizer(this->cuda_async_device);
                    auto cuda_restrict_synchronizer                 = dg::network_controller::RestrictPointerSynchronizer(cuda_synchronizer);
                    auto host_synchronizer                          = dg::network_host_asynchronous::Synchronizer(this->host_async_device);
                    auto host_restrict_synchronizer                 = dg::network_controller::RestrictPointerSynchronizer(host_synchronizer);
                    auto internal_cuda_resolutor                    = InternalCudaResolutor();
                    auto internal_host_resolutor                    = InternalHostResolutor();

                    internal_cuda_resolutor.synchronizer            = &cuda_synchronizer;
                    internal_cuda_resolutor.restrict_synchronizer   = &cuda_restrict_synchronizer;
                    internal_cuda_resolutor.async_device            = this->cuda_async_device;

                    internal_host_resolutor.synchronizer            = &host_synchronizer;
                    internal_host_resolutor.restrict_synchronizer   = &host_restrict_synchronizer;
                    internal_host_resolutor.async_device            = this->host_async_device;

                    size_t cdh_trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_allocation_cost(&internal_cuda_resolutor, cdh_trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);  
                    auto cuda_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_cuda_resolutor, cdh_trimmed_vectorization_sz, cdh_mem.get()));

                    size_t hdh_trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_allocation_cost(&internal_host_resolutor, hdh_trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_host_resolutor, hdh_trimmed_vectorization_sz, hdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]    = data_arr[i];

                        //we know that these guys are verified earilier - so it's fine to make such assumption here
                        //for best practices - we do this
                        dg::network_tile_member_access::safe_mono_ptr_access(dst);

                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_mono_operatable_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_mono_forward_operatable_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_mono_dispatch_control_nothrow(dst); 
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_mono_logit_addr_nothrow(dst); 
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_mono_observer_arr_size_nothrow(dst);

                        //we are moving towards our stack
                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(MAX_OBSERVER_ARR_SIZE);
                        dg::network_tile_member_getsetter::get_mono_observer_arr_nothrow(dst, dst_observer_arr.get());

                        //we don't know if these guys have these members (forward requirements for this) - so we must do polymorphic guards here
                        std::expected<operatable_id_t, exception_t> src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_forward_operatable_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);

                        //because this is a feature - initializers intend for this to happen - so we must not issue errors here - this is a soft pass - there are cases where we can forward and not backward - and there are cases we can backward but not forward
                        //we must declare expectations of forward here and do a soft pass

                        if (!src_fwd_operatable_id.has_value() || !src_init_status.has_value() || !src_logit_umaptr.has_value()){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_fwd_operatable_id != src_fwd_operatable_id.value()){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_ADOPTED && dst_init_status != TILE_INIT_STATUS_DECAYED){
                            continue;
                        }

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_mono(dispatch_control);

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr.value(), src_vd_id}})){
                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            cuda_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_synchronizer.sync();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr.value(), src_vd_id}});
                        auto dst_map_vmaptr = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        auto src_map_vmaptr = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_vmamap_reacquirer, dst_map_vmaptr) || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_vmamap_reacquirer, src_map_vmaptr)){       
                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            cuda_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_synchronizer.sync();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_vmamap_reacquirer, dst_map_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(src_vmamap_reacquirer, src_map_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto dst_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(dst_vmamap_reacquirer);
                            auto src_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(src_vmamap_reacquirer);

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), src_logit_cudaptr, std::make_tuple(dst_logit_cudaptr, src_logit_cudaptr, tileops_dp_code));
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_logit_hostptr  = dg::network_vmamap::get_host_ptr(dst_vmamap_reacquirer);
                            auto src_logit_hostptr  = dg::network_vmamap::get_host_ptr(src_vmamap_reacquirer);

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), src_logit_hostptr, std::make_tuple(dst_logit_hostptr, src_logit_hostptr, tileops_dp_code));
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(dst_observer_arr[j], dst_operatable_id)));
                        }

                        dg::network_tile_member_getsetter::set_mono_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                    }
                }
            };
    };

    class ForwardDoPairSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t addrfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoPairSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                           std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                           const size_t request_delivery_capacity,
                                           const size_t addrfetch_vectorization_sz,
                                           const size_t region_vectorization_sz,
                                           const size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                            cuda_async_device(std::move(cuda_async_device)),
                                                                                            host_async_device(std::move(host_async_device)),
                                                                                            request_delivery_capacity(request_delivery_capacity),
                                                                                            addrfetch_vectorization_sz(addrfetch_vectorization_sz),
                                                                                            region_vectorization_sz(region_vectorization_sz),
                                                                                            forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<std::tuple<uma_ptr_t, uma_ptr_t>>[]> descendant_arr(sz);
                const size_t EVENT_SCALE_FACTOR             = MAX_OBSERVER_ARR_SZ;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz); 
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    InternalDescendantAddressFetcher fetcher    = {};
                    size_t trimmed_addrfetch_vectorization_sz   = std::min(this->addrfetch_vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_addrfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_addrfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(event_arr[i].dst, event_arr[i].operatable_id, std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;
                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost); 
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        uma_ptr_t dst                                           = event_arr[i].dst;
                        uma_ptr_t left                                          = std::get<0>(descendant_arr[i].value());
                        uma_ptr_t right                                         = std::get<1>(descendant_arr[i].value());
                        uma_ptr_t dst_rcu_addr                                  = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(dst); //assume that descendant_arr[i].has_value() => safe pair access
                        std::expected<uma_ptr_t, exception_t> left_rcu_addr     = dg::network_tile_member_getsetter::get_tile_rcu_addr(left); //we are doing polymorphic access - it's better to safeguards the assumption here 
                        std::expected<uma_ptr_t, exception_t> right_rcu_addr    = dg::network_tile_member_getsetter::get_tile_rcu_addr(right);

                        if (!left_rcu_addr.has_value() || !right_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz        = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        uma_ptr_t left_lck_addr     = dg::memult::region(left_rcu_addr.value(), lck_region_sz);
                        uma_ptr_t right_lck_addr    = dg::memult::region(right_rcu_addr.value(), lck_region_sz);
                        auto key                    = std::make_tuple(dst_lck_addr, left_lck_addr, right_lck_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(dst, left, right, event_arr[i].operatable_id));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, operatable_id_t, std::add_pointer_t<std::optional<std::tuple<uma_ptr_t, uma_ptr_t>>>>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, operatable_id_t, std::optional<std::tuple<uma_ptr_t, uma_ptr_t>> *> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, expected_ops_id, fetching_addr]  = data_arr[i];
                        init_status_t init_status                   = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(dst);
                        operatable_id_t current_ops_id              = dg::network_tile_member_getsetter::get_pair_operatable_id_nothrow(dst);

                        if (expected_ops_id != current_ops_id){
                            *fetching_addr = std::nullopt;
                            continue;
                        } 

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                pong_count_t pong_count = dg::network_tile_member_getsetter::get_pair_pong_count_nothrow(dst);
                                pong_count += 1u; //has to be unsigned otherwise we risk unsigned wraparound
                                dg::network_tile_member_getsetter::set_pair_pong_count_nothrow(dst, pong_count);

                                if (pong_count >= dg::network_tile_metadata::PAIR_DESCENDANT_COUNT){
                                    *fetching_addr = std::make_tuple(dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(dst), dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(dst));
                                } else{
                                    *fetching_addr = std::nullopt;
                                }

                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::KVConsumerInterface<cuda_ptr_t, std::tuple<cuda_ptr_t, cuda_ptr_t, cuda_tileops_dispatch_control_t>>{

                dg::network_cuda_controller::CudaSynchronizer * synchronizer;
                dg::network_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;

                void push(cuda_ptr_t key, std::tuple<cuda_ptr_t, cuda_ptr_t, cuda_ptr_t, cuda_tileops_dispatch_control_t> * data_arr, size_t sz) noexcept{
                    
                    size_t cuda_ptr_vec_sz  = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_vec(cuda_ptr_vec_sz);
                    auto pair_aggregator = dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::aggregator_raiispawn_pair(sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, lhs, rhs, tileops_dp_code]   = data_arr[i];
                        cuda_ptr_vec[i * 3]                     = dst;
                        cuda_ptr_vec[i * 3 + 1]                 = lhs;
                        cuda_ptr_vec[i * 3 + 2]                 = rhs;
                        dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::aggregator_add(pair_aggregator, dst, lhs, rhs, tileops_dp_code));
                    }

                    auto executable = [arg = std::move(pair_aggregator)]() noexcept{
                        dg::network_tileops_cuda_poly::aggregator_exec(arg); //exceptions -
                    };
                    auto async_task = dg::network_cuda_controller::virtualize_async_task(std::move(executable));

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_vec.get(), std::next(cuda_ptr_vec.get(), cuda_ptr_vec_sz)));
                    auto async_id = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(async_task)));
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(async_id));
                }
            };

            struct InternalHostResolutor: dg::network_producer_consumer::KVConsumerInterface<host_ptr_t, std::tuple<host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t>>{

                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;

                void push(host_ptr_t key, std::tuple<host_ptr_t, host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t> * data_arr, size_t sz) noexcept{

                    size_t host_ptr_vec_sz  = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_vec(host_ptr_vec_sz);
                    auto pair_aggregator    = dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_raiispawn_pair(sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, lhs, rhs, tileops_dp_code]   = data_arr[i];
                        host_ptr_vec[i * 3]                     = dst;
                        host_ptr_vec[i * 3 + 1]                 = lhs;
                        host_ptr_vec[i * 3 + 2]                 = rhs;
                        dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_add(pair_aggregator, dst, lhs, rhs, tileops_dp_code));
                    }

                    auto executable = [arg = std::move(pair_aggregator)]() noexcept{
                        dg::network_tileops_host_poly::aggregator_exec(arg);
                    };
                    auto async_task = dg::network_host_asynchronous::virtualize_async_task(std::move(executable));

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_vec.get(), std::next(host_ptr_vec.get(), host_ptr_vec_sz)));
                    auto async_id = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(async_task)));
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(async_id));
                }
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                size_t vectorization_sz;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;

                void push(std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr), std::get<2>(lck_addr));

                    auto umamap_reacquirer                          = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{});
                    auto dst_vmamap_reacquirer                      = dg::network_vmamap::reacquirer_raii_initialize();
                    auto lhs_vmamap_reacquirer                      = dg::network_vmamap::reacquirer_raii_initialize();
                    auto rhs_vmamap_reacquirer                      = dg::network_vmamap::reacquirer_raii_initialize();
                    
                    auto cuda_synchronizer                          = dg::network_cuda_controller::CudaSynchronizer(this->cuda_async_device);
                    auto host_synchronizer                          = dg::network_host_asynchronous::Synchronizer(this->host_async_device);
                    auto cuda_restrict_synchronizer                 = dg::network_controller::RestrictPointerSynchronizer(cuda_synchronizer);
                    auto host_restrict_synchronizer                 = dg::network_controller::RestrictPointerSynchronizer(host_synchronizer);

                    auto internal_cuda_resolutor                    = InternalCudaResolutor();
                    auto internal_host_resolutor                    = InternalHostResolutor();

                    internal_cuda_resolutor.restrict_synchronizer   = &cuda_restrict_synchronizer;
                    internal_cuda_resolutor.synchronizer            = &cuda_synchronizer;
                    internal_cuda_resolutor.async_device            = this->cuda_async_device;

                    internal_host_resolutor.restrict_synchronizer   = &host_restrict_synchronizer;
                    internal_host_resolutor.synchronizer            = &host_synchronizer;
                    internal_host_resolutor.async_device            = this->host_async_device;

                    size_t trimmed_cuda_vectorization_sz            = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_cuda_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_cuda_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_host_vectorization_sz            = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_host_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_host_resolutor, trimmed_host_vectorization_sz, hdh_mem.get())); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, lhs, rhs, expected_ops_id]   = data_arr[i];
                        dg::network_tile_member_access::safe_pair_ptr_access(dst);
                        uma_ptr_t dst_lhs                       = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(dst);
                        uma_ptr_t dst_rhs                       = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_pair_operatable_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_pair_forward_operatable_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_pair_logit_addr_nothrow(dst);
                        dispatch_major_t dst_dispatch_major     = dg::network_tile_member_getsetter::get_pair_dispatch_major_nothrow(dst);
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_pair_observer_array_size_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_pair_dispatch_control_nothrow(dst);
                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(MAX_OBSERVER_ARR_SZ);
                        dg::network_tile_member_getsetter::get_pair_observer_array_nothrow(dst, dst_observer_arr.get());

                        std::expected<operatable_id_t, exception_t> lhs_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_forward_operatable_id(lhs); //
                        std::expected<uma_ptr_t, exception_t> lhs_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(lhs);
                        std::expected<init_status_t, exception_t> lhs_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(lhs);

                        std::expected<operatable_id_t, exception_t> rhs_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_forward_operatable_id(rhs); //
                        std::expected<uma_ptr_t, exception_t> rhs_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(rhs);
                        std::expected<init_status_t, exception_t> rhs_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(rhs);

                        if (!lhs_fwd_operatable_id.has_value() || !lhs_logit_uma_ptr.has_value() || !lhs_init_status.has_value() || !rhs_fwd_operatable_id.has_value() || !rhs_logit_umaptr.has_value() || !rhs_init_status.has_value()){
                            continue;
                        }

                        if (dst_lhs != lhs){
                            continue;
                        }

                        if (dst_rhs != rhs){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_fwd_operatable_id != lhs_fwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_fwd_operatable_id != rhs_fwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED || dst_init_status != TILE_INIT_STATUS_ADOPTED){
                            continue;
                        }

                        if (lhs_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (rhs_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        auto [dst_vd_id, lhs_vd_id, rhs_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_pair(dispatch_control);

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {lhs_logit_umaptr.value(), lhs_vd_id}, {rhs_logit_umaptr.value(), rhs_vd_id}})){
                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            cuda_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_synchronizer.sync();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {lhs_logit_umaptr.value(), lhs_vd_id}, {rhs_logit_umaptr.value(), rhs_vd_id}});

                        vma_ptr_t dst_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t lhs_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t rhs_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_vmamap_reacquirer, dst_vmaptr) 
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(lhs_vmamap_reacquirer, lhs_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(rhs_vmamap_reacquirer, rhs_vmaptr)){
                            
                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            cuda_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_synchronizer.sync();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_vmamap_reacquirer, dst_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(lhs_vmamap_reacquirer, lhs_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(rhs_vmamap_reacquirer, rhs_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto dst_cudaptr    = dg::network_vmamap::get_cuda_ptr(dst_vmamap_reacquirer);
                            auto lhs_cudaptr    = dg::network_vmamao::get_cuda_ptr(lhs_vmamap_reacquirer);
                            auto rhs_cudaptr    = dg::network_vmamap::get_cuda_ptr(rhs_vmamap_reacquirer);

                            if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_RIGHT){
                                dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), rhs_cudaptr, std::make_tuple(dst_cudaptr, lhs_cudaptr, rhs_cudaptr, tileops_dp_code));
                            } else if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_LEFT){
                                dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), lhs_cudaptr, std::make_tuple(dst_cudaptr, lhs_cudaptr, rhs_cudaptr, tileops_dp_code));
                            } else{
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_hostptr    = dg::network_vmamap::get_host_ptr(dst_vmamap_reacquirer);
                            auto lhs_hostptr    = dg::network_vmamap::get_host_ptr(lhs_vmamap_reacquirer);
                            auto rhs_hostptr    = dg::network_vmamap::get_host_ptr(rhs_vmamap_reacquirer);

                            if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_RIGHT){
                                dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), rhs_hostptr, std::make_tuple(dst_hostptr, lhs_hostptr, rhs_hostptr, tileops_dp_code));
                            } else if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_LEFT){
                                dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), lhs_hostptr, std::make_tuple(dst_hostptr, lhs_hostptr, rhs_hostptr, tileops_dp_code));
                            } else{
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else[
                                    std::unreachable();
                                ]
                            }
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(dst_observer_arr[j], dst_operatable_id)));
                        }

                        dg::network_tile_member_getsetter::set_pair_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED); //bad assumption in asynchronous context
                    }
                }
            };
    };

    class ForwardDoUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardDoUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity,
                                           size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity),
                                                                              vectorization_sz(vectorization_sz){}
            
            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                auto descendant_arr     = std::make_unique<std::optional<dg::vector<uma_ptr_t>>[]>(sz);
                auto delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz);

                    if (!vectorized_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorized_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), lck_addr, std::make_tuple(event_arr[i].dst, std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle->get();
                    auto vectorized_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz);

                    if (!vectorized_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorized_delivery_handle.error()));
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        dg::vector<uma_ptr_t> ptr_vec   = dg::utility::vector_immu_push_back(descendant_arr[i].value(), event_arr[i].dst);
                        dg::vector<uma_ptr_t> rcu_vec   = dg::utility::vector_immu_transform(ptr_vec, [](uma_ptr_t e){return dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(e);});
                        dg::vector<uma_ptr_t> rep_vec   = dg::utility::vector_immu_transform(rcu_vec, [](uma_ptr_t e){return dg::memult::region(e, std::min(dg::network_memops_uma::memlock_region_size(), dg::network_uma::memregion_size()));}); 
                        dg::set<uma_ptr_t> key          = dg::utility::set_make_from_vec(rep_vec);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), key, std::make_tuple(event_arr[i].dst, descendant_arr[i].value()));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, std::optional<dg::vector<uma_ptr_t>> *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, std::optional<dg::vector<uma_ptr_t>> *> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, fetching_addr]   = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(dst);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr  = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_addr  = dg::utility::vector_make_from_array(dg::network_tile_member_getsetter::get_uacm_descendant_nothrow(dst));
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
                }
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<dg::set<uma_ptr_t>, std::tuple<uma_ptr_t, dg::vector<uma_ptr_t>>>{

                void push(dg::set<uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, dg::vector<uma_ptr_t>> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(lck_addr);

                    auto umamap_reacquirer  = dg::network_uma::reacquirer_adaptive_raii_initialize();
                    auto vmamap_reacquirer  = dg::network_vmamap::reacquirer_adaptive_raii_initialize();

                    dg::network_cuda_controller::CudaSynchronizer synchronizer{};
                    dg::network_controller::RestrictPointerSynchronizer restrict_synchronizer(synchronizer);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, descendant_vec]  = data_arr[i];

                        if (dst_descendant_vec.size() != dg::network_tile_metadata::UACM_ACM_SZ){
                            continue;
                        }

                        if (dst_descendant_vec != descendant_vec){
                            continue;
                        }

                        if (dg::utility::set_make_from_vector(descendant_operatable_id_vec).size() != 1u){
                            continue;
                        }

                        if (dst_operatable_id != descendant_operatable_id_vec.front()){
                            continue;
                        }

                        if (std::find_if(descendant_init_status_vec.begin(), descendant_init_status_vec.end(), [](init_status_t status){return status != TILE_INIT_STATUS_INITIALIZED;};) != descendant_init_status_vec.end()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                            continue;
                        }


                    }
                }
            };
    };

    class ForwardDoPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            ForwardDoPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}
            
            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{
                
            }
    };

    class ForwardDoExtnSrcSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t delivery_capacity;
            const size_t addrfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoExtnSrcSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                              std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                              std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                              std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                              size_t delivery_capacity,
                                              size_t addrfetch_vectorization_sz,
                                              size_t region_vectorization_sz,
                                              size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                         uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                         host_ip_retriever(std::move(host_ip_retriever)),
                                                                                         host_async_device(std::move(host_async_device)),
                                                                                         delivery_capacity(delivery_capacity),
                                                                                         addrfetch_vectorization_sz(addrfetch_vectorization_sz),
                                                                                         region_vectorization_sz(region_vectorization_sz),
                                                                                         forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<uma_ptr_t>[]> descendant_arr(sz); //I dont like nullvalues - I feel like its a hack for std::optional<> because it always has been

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalDescendantAddressFetcher fetcher    = {};
                    size_t trimmed_addfetch_vectorization_sz    = std::min(this->addrfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_addfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_addfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(event_arr[i].dst, event_arr[i].operatable_id, std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    internal_resolutor.uma_ip_retriever         = this->uma_ip_retriever.get();
                    internal_resolutor.host_ip_retriever        = this->host_ip_retriever.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.forward_vectorization_sz = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(event_arr[i].dst); //we assume descendant_arr.has_value() => dst is valid extnsrc
                        std::expected<uma_ptr_t, exception_t> src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(descendant_arr[i].value()); //

                        if (!src_rcu_addr.has_value()){
                            continue; //soft-pass - because if this is an error than this should be internal corruption - tiles states must be correct at all times
                        }

                        size_t lck_region_sz    = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_lck_addr  = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        uma_ptr_t src_lck_addr  = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        auto key                = std::make_tuple(dst_rep_addr, src_rep_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(event_arr[i].dst, descendant_arr[i].value(), event_arr[i].operatable_id));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, operatable_id_t, std::optional<uma_ptr_t> *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, operatable_id_t, std::optional<uma_ptr_t> *> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, expected_ops_id_fetching_addr]   = data_arr[i];
                        init_status_t init_status                   = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(dst);
                        operatable_id_t current_ops_id              = dg::network_tile_member_getsetter::get_extnsrc_operatable_id_nothrow(dst);

                        if (expected_ops_id != current_ops_id){
                            *fetching_addr = std::nullopt;
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr  = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_addr  = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(dst);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };

            struct InternalHostResolutor: dg::network_producer_consumer::KVConsumerInterface<host_ptr_t, std::tuple<host_ptr_t, host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t>>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;

                void push(host_ptr_t key, std::tuple<host_ptr_t, host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t> * data_arr, size_t sz) noexcept{

                    size_t host_ptr_vec_sz = sz * 3; //we are assuming unique_restriction for host_ptr_t, host_ptr_t, host_ptr_t
                    dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> host_ptr_vec(host_ptr_vec_sz);
                    auto aggregator = dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_raiispawn(sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, cpy_dst, tileops_dispatch_control] = data_arr[i];
                        host_ptr_vec[i * 3]     = dst;
                        host_ptr_vec[i * 3 + 1] = src;
                        host_ptr_vec[i * 3 + 2] = cpy_dst;

                        auto tile_executable = [dst, src, cpy_dst, counterpart, tileops_dispatch_control]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::forward_mono(dst, src, tileops_dispatch_control));
                            dg::network_exception_handler::nothrow_log(dg::network_memops_clib::memcpy_host_to_host(cpy_dst, dst, dg::network_tileops_host_poly::get_byte_size(tileops_dispatch_control)));
                        };

                        dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_add(aggregator, std::move(tile_executable)));
                    }

                    auto executable = [arg_aggregator = std::move(aggregator)]() noexcept{
                        dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_exec(arg_aggregator));
                    };

                    auto async_task = dg::network_host_asynchronous::virtualize_async_task(std::move(executable));
                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_vec.get(), std::next(host_ptr_vec.get(), host_ptr_vec_sz)));
                    auto async_id   = dg::network_exception_handler::nothrow_log(this->host_async_device->exec(std::move(async_task))); //we'll fix this later 
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(async_id));
                }
            };

            struct OutBoundData{
                Address to;
                Address fr;
                uma_ptr_t counterpart_ptr;
                dg::string buf;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t, operatable_id_t>>{

                dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * request_delivery_handle;
                UnifiedMmeoryIPRetrieverInterface * uma_ip_retriever;
                HostIPRetrieverInterface * host_ip_retriever;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                size_t forward_vectorization_sz;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));

                    auto umamap_reacquirer                  = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto dst_logit_vmamap_reacquirer        = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_logit_vmamap_reacquirer        = dg::network_vmamap::reacquirer_raii_initialize();

                    dg::network_host_asynchronous::Synchronizer synchronizer{};
                    dg::network_host_asynchronous::RestrictPointerSynchronizer restrict_synchronizer(synchronizer);

                    size_t trimmed_forward_vectorization_sz = std::min(this->forward_vectorization_sz, sz);
                    size_t hvdh_allocation_cost             = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_host_resolutor, trimmed_forward_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hvdh_mem(hvdh_allocation_cost);
                    auto host_vectorizer_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_raiihandle(&internal_host_resolutor, trimmed_forward_vectorization_sz, hvdh_mem.get())); 
                    auto outbound_data_vec                  = std::vector<std::optional<OutBoundData>>(sz, std::optional<OutBoundData>(std::nullopt));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = data_arr[i];
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_extnsrc_operatable_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_extnsrc_forward_operatable_id_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_extnsrc_dispatch_control_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_extnsrc_logit_addr_nothrow(dst);
                        uma_ptr_t dst_counterpart               = dg::network_tile_member_getsetter::get_extnsrc_counterpart_nothrow(dst);

                        std::expected<operatable_id_t, exception_t> src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_forward_operatable_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src); 

                        if (!src_fwd_operatable_id.has_value() || !src_init_status.has_value() || !src_logit_umaptr.has_value()){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_fwd_operatable_id != src_fwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED && dst_init_status != TILE_INIT_STATUS_ADOPTED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_extnsrc(dispatch_control);

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr.value(), src_vd_id}})){
                            dg::network_producer_consumer::delvrsrv_clear(host_vectorizer_delivery_handle.get());
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr.value(), src_vd_id}});
                        vma_ptr_t dst_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_logit_vmamap_reacquirer, dst_logit_vmaptr) 
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_umaptr.value())){

                            dg::network_producer_consumer::delvrsrv_clear(host_vectorizer_delivery_handle.get());
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if(dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_logit_hostptr      = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            auto src_logit_hostptr      = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            auto rq                     = OutBoundData{};
                            rq.to                       = this->uma_ip_retriever->ip(dst_counterpart);
                            rq.fr                       = this->host_ip_retriever->ip();
                            rq.counterpart_ptr          = dst_counterpart;
                            rq.buf                      = dg::string(dg::network_tileops_host_poly::get_buffer_size(tileops_dp_code));
                            host_ptr_t rq_buf_data      = rq.buf.data();
                            outbound_data_vec[i]        = std::move(rq);

                            dg::network_producer_consumer::delvrsrv_deliver(host_vectorizer_delivery_handle.get(), src_logit_hostptr, std::make_tuple(dst_logit_hostptr, src_logit_hostptr, rq_buf_data, tileops_dp_code));
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                    }

                    dg::network_producer_consumer::delvrsrv_clear(host_vectorizer_delivery_handle.get());
                    synchronizer.sync();

                    for (size_t i = 0u; i < sz; ++i){
                        if (!outbound_data_vec[i].has_value()){
                            continue;
                        }

                        external_virtual_memory_event_t inject_event    = dg::network_external_memcommit_factory::make_event_shadow_injection(outbound_data_vec[i]->counterpart_ptr, TILE_KIND_EXTNSRC, std::move(outbound_data_vec[i]->buf));
                        external_virtual_memory_event_t notify_event    = dg::network_external_memcommit_factory::make_event_forward_do_signal(outbound_data_vec[i]->counterpart_ptr);
                        external_virtual_memory_event_t event           = dg::network_external_memcommit_factory::make_event_sequential(std::move(inject_event), std::move(notify_event));
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(event));
                    }
                }
            };
    };

    //we do forward cuda + forward host for this
    class ForwardDoExtnDstSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const size_t delivery_capacity;
            const size_t addrfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoExtnDstSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter,
                                              std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                              std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                              size_t delivery_capacity,
                                              size_t addrfetch_vectorization_sz,
                                              size_t region_vectorization_sz,
                                              size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                         alias_getter(std::move(alias_getter)),
                                                                                         host_async_device(std::move(host_async_device)),
                                                                                         cuda_async_device(std::move(cuda_async_device)),
                                                                                         delivery_capacity(delivery_capacity),
                                                                                         addrfetch_vectorization_sz(addrfetch_vectorization_sz),
                                                                                         region_vectorization_sz(region_vectorization_sz),
                                                                                         forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<uma_ptr_t>[]> descendant_arr(sz); 

                const size_t EVENT_SCALE_FACTOR     = MAX_OBSERVER_ARR_SZ;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalDescendantAddressFetcher fetcher    = {};
                    fetcher.alias_getter                        = this->alias_getter.get();

                    size_t trimmed_addrfetch_vectorization_sz   = std::min(this->addrfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_addrfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost); 
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_addrfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memregion_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(event_arr[i].dst, event_arr[i].operatable_id, std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        size_t lck_region_sz    = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size(), static_cast<size_t>(dg::network_uma::memregion_size())));
                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t dst_lck_addr  = dg::memult::region(dst_rcu_addr, lck_region_sz); 

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(descendant_arr[i].value()); 

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        uma_ptr_t src_lck_addr  = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        auto key                = std::make_tuple(dst_lck_addr, src_lck_addr);
                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(event_arr[i].dst, descendant_arr[i].value(), event_arr[i].operatable_id));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::ConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, operatable_id_t, std::optional<uma_ptr_t> *>>{

                ForeignTileAliasGetterInterface * alias_getter;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, operatable_id_t, std::optional<uma_ptr_t> *> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, expected_ops_id, fetching_addr]  = data_arr[i];
                        init_status_t init_status                   = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(dst);
                        operatable_id_t current_ops_id              = dg::network_tile_member_getsetter::get_extndst_operatable_id_nothrow(dst);

                        //branch optimization opportunity - I think clang would do this better

                        if (expected_ops_id != current_ops_id){
                            *fetching_addr = std::nullopt;
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                std::expected<std::optional<uma_ptr_t>, exception_t> alias = this->alias_getter->alias(dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(dst));

                                if (!alias.has_value()){
                                    //
                                    *fetching_addr = std::nullopt;
                                } else{
                                    *fetching_addr = alias.value();
                                }

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

            struct InternalCudaResolutor: dg::network_producer_consumer::KVConsumerInterface<cuda_ptr_t, std::tuple<cuda_ptr_t, cuda_ptr_t, cuda_tileops_dispatch_control_t>>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;

                void push(cuda_ptr_t, std::tuple<cuda_ptr_t, cuda_ptr_t, cuda_tileops_dispatch_control_t> * data_arr, size_t sz) noexcept{

                }
            };

            struct InternalHostResolutor: dg::network_producer_consumer::KVConsumerInterface<host_ptr_t, std::tuple<host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t>>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;

                void push(host_ptr_t, std::tuple<host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t> * data_arr, size_t sz) noexcept{

                }
            };

            struct InternalResolutor: dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t, operatable_id_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                size_t vectorization_sz; 

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr)); //we move the guard post the allocations

                    auto umamap_reacquirer                  = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto dst_logit_vmamap_reacquirer        = dg::network_vmamap::reacquirer_initialize(); 
                    auto src_logit_vmamap_reacquirer        = dg::network_vmamap::reacquirer_initialize();

                    dg::network_cuda_controller::Synchronizer cuda_synchronizer{};
                    dg::network_cuda_controlled::RestrictPointerSynchronizer cuda_restrict_synchronizer(cuda_synchronizer);
                    InternalCudaResolutor cuda_resolutor    = {};
                    cuda_resolutor.async_device             = this->cuda_async_device;
                    cuda_resolutor.synchronizer             = &cuda_synchronizer;
                    cuda_resolutor.restrict_synchronizer    = &cuda_restrict_synchronizer;

                    dg::network_host_asynchronous::Synchronizer host_synchronizer{};
                    dg::network_host_asynchronous::RestrictPointerSynchronizer host_restrict_synchronizer(host_synchronizer);
                    InternalHostResolutor host_resolutor    = {};
                    host_resolutor.async_device             = this->host_async_device;
                    host_resolutor.synchronizer             = &host_synchronizer;
                    host_resolutor.restrict_synchronizer    = &restrict_synchronizer;

                    size_t trimmed_cuda_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&cuda_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&cuda_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_host_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&host_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&host_resolutor, trimmed_host_vectorization_sz, hdh_mem.get()));


                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = ptr_arr[i];
                        uma_ptr_t dst_counterpart               = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_extndst_memevent_operatable_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_extndst_forward_operatable_id_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_extndst_logit_addr_nothrow(dst);
                        
                        std::expected<uma_ptr_t, exception_t> extnsrc_ptr_access = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(src);

                        if (!extnsrc_ptr_access.has_value()){
                            continue;
                        } 

                        uma_ptr_t src_selfaddr                  = dg::network_tile_member_getsetter::get_extnsrc_selfaddr_nothrow(src);
                        operatable_id_t src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_extnsrc_forward_operatable_id_nothrow(src);
                        init_status_t src_init_status           = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(src);
                        uma_ptr_t src_logit_umaptr              = dg::network_tile_member_getsetter::get_extnsrc_logit_addr_nothrow(src);

                        if (dst_counterpart != src_selfaddr){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_fwd_operatable_id != src_fwd_operatable_id){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED && dst_init_status != TILE_INIT_STATUS_ADOPTED){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_extndst(dispatch_control);

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}})){
                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            cuda_synchronizer.sync();
                            host_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                        vma_ptr_t dst_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{}); 

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_logit_vmamap_reacquirer, dst_logit_vmaptr) 
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_vmaptr)){
                            
                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            cuda_synchronizer.sync();
                            host_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto dst_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap_reacquirer);
                            auto src_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), src_logit_cudaptr, std::make_tuple(dst_logit_cudaptr, src_logit_cudaptr, tileops_dp_code)); //we have to vectorize to avoid synchronization overheads - not necessarily because this is a feature
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_logit_hostptr  = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            auto src_logit_hostptr  = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), src_logit_hostptr, std::make_tuple(dst_logit_hostptr, src_logit_hostptr, tileops_dp_code));
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_init_signal(dst_observer_arr[j], dst_operatable_id)));
                        }
                    }
                }
            };
    };

    class ForwardDoCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            CritForwardDoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(event_arr[i].dst, delivery_handle->get());
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
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(observer_arr[i])));
                }

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src)));
            }
    };

    class ForwardDoMsgrFwdSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box;
            const std::shared_ptr<dg::networK_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t eu_delivery_capacity;
            const size_t addrfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoMsgrFwdSingalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box,
                                              std::shared_ptr<dg::network_producer_consumer::AsynchronousDeviceInterface> host_async_device,
                                              size_t request_delivery_capacity,
                                              size_t eu_delivery_capacity,
                                              size_t addrfetch_vectorization_sz,
                                              size_t region_vectorization_sz,
                                              size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                         eu_packet_box(std::move(eu_packet_box)),
                                                                                         host_async_device(std::move(host_async_device)),
                                                                                         request_delivery_capacity(request_delivery_capacity),
                                                                                         eu_delivery_capacity(eu_delivery_capacity),
                                                                                         addrfetch_vectorization_sz(addrfetch_vectorization_sz),
                                                                                         region_vectorization_sz(region_vectorization_sz),
                                                                                         forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<uma_ptr_t>[]> descendant_arr(sz);

                const size_t REQUEST_EVENT_SCALE_FACTOR     = MAX_OBSERVER_ARRAY_SZ;
                size_t max_possible_event_sz                = sz * REQUEST_EVENT_SCALE_FACTOR; 
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost); 
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));
                
                const size_t EU_PACKET_SCALE_FACTOR         = 1u;
                size_t max_possible_eu_packet               = sz * EU_PACKET_SCALE_FACTOR;
                size_t trimmed_eu_packet_delivery_capacity  = std::min(this->eu_packet_delivery_capacity, max_possible_eu_packet); 
                size_t epdh_allocation_cost                 = dg::network_producer_consumer::delvrsrv_allocation_cost(this->eu_packet_box.get(), trimmed_eu_packet_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> epdh_mem(epdh_allocation_cost);
                auto eu_packet_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->eu_packet_box.get(), trimmed_eu_packet_delivery_capacity, epdh_mem.get()));

                {
                    InternalDescendantAddressFetcher fetcher    = {};
                    size_t trimmed_addrfetch_vectorization_sz   = std::min(this->addrfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_addrfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, trimmed_addrfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), lck_addr, std::make_tuple(event_arr[i].dst, event_arr[i].operatable_id, std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor            = {};
                    internal_resolutor.request_delivery_handle      = request_delivery_handle.get();
                    internal_resolutor.eu_packet_delivery_handle    = eu_packet_delivery_handle.get();
                    internal_resolutor.host_async_device            = this->host_async_device.get();
                    internal_resolutor.vectorization_sz             = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz          = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get())); 

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        size_t lck_region_sz    = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size())); 
                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].dst); //assume descendant_arr[i].has_value() => safe_msgrfwd_ptr_access(event_arr[i].dst)

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(descendant_arr[i].value());

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        uma_ptr_t dst_lck_addr  = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        uma_ptr_t src_lck_addr  = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        auto key                = std::make_tuple(dst_lck_addr, src_lck_addr);
                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(event_arr[i].dst, descendant_arr[i].value(), event_arr[i].operatable_id));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, operatable_id_t, std::optional<uma_ptr_t> *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t *> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, expected_ops_id, fetching_addr]  = ptr_arr[i];
                        init_status_t init_status                   = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(dst);
                        operatable_id_t current_ops_id              = dg::network_tile_member_getsetter::get_msgrfwd_memevent_operatable_id_nothrow(dst);

                        if (expected_ops_id != current_ops_id){
                            *fetching_addr = std::nullopt;
                            continue;
                        }

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_addr = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(dst);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };

            struct MsgrFwdData{
                tile_id_t tile_id;
                dg::string logit_value;
                Address dst;
                size_t retry_count;
                eu_packet_urgency_t urgency;
                eu_packet_comm_t comm;  
            };

            struct InternalHostResolutor: dg::network_producer_consumer::KVConsumerInterface<host_ptr_t, std::tuple<host_ptr_t, host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t>>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;

                void push(host_ptr_t, std::tuple<host_ptr_t, host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t> * data_arr, size_t sz) noexcept{ //we are ditching this ambiguity later

                    //we can't be too cautious catching exotic errors or our system wont run - or it gets stuck at a random error that is internally caused by the system (says memory allocations) - we'll try to do designated user logs next iteration because exceptions are detached for memevent 

                    size_t host_ptr_vec_sz = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_vec(host_ptr_vec_sz); //we are assuming unique_restriction for host_ptr_t, host_ptr_t, host_ptr_t
                    auto aggregator = dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_raiispawn(sz)); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, cpy_dst, tileops_dispatch_control]  = data_arr[i];
                        host_ptr_vec[i * 3]                                 = dst;
                        host_ptr_vec[i * 3 + 1]                             = src;
                        host_ptr_vec[i * 3 + 2]                             = cpy_dst;

                        auto tile_executable = [dst, src, cpy_dst, tileops_dispatch_control]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::forward_mono(dst, src, tileops_dispatch_control));
                            dg::network_exception_handler::nothrow_log(dg::network_memops::memcpy_host_to_host(cpy_dst, dst, dg::network_tileops_host_poly::get_byte_size(tileops_dispatch_control)));
                        };

                        dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_add(aggregator, std::move(tile_executable)));
                    }

                    auto executable = [arg_aggregator = std::move(aggregator)]() noexcept{
                        dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_exec(arg_aggregator));
                    };

                    auto async_task = dg::network_host_asynchronous::virtualize_async_task(std::move(executable));
                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_vec.get(), std::next(host_ptr_vec.get(), host_ptr_vec_sz)));
                    auto async_id   = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(async_task)));
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(async_id));
                    // if (!async_id.has_value()){

                    // }
                }
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t, operatable_id_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_producer_consumer::DeliveryHandle<EndUserPacket> * eu_packet_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                size_t vectorization_sz;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t, operatable_id_t> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));

                    auto umamap_reacquirer                  = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto dst_logit_vmamap_reacquirer        = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_logit_vmamap_reacquirer        = dg::network_vmamap::reacquirer_raii_initialize(); 

                    dg::network_host_asynchronous::Synchronizer host_synchronizer{};
                    dg::network_host_asynchronous::RestrictPointerSynchronizer restrict_synchronizer(host_synchronizer);

                    InternalHostResolutor host_resolutor    = {};
                    host_resolutor.async_device             = this->host_async_device;
                    host_resolutor.synchronizer             = &host_synchronizer;
                    host_resolutor.restrict_synchronizer    = &restrict_synchronizer;

                    size_t trimmed_vectorization_sz         = std::min(this->vectorization_sz, sz);
                    size_t hv_allocation_cost               = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&host_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hv_mem(hv_allocation_cost);
                    auto host_vectorizer                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&host_resolutor, trimmed_vectorization_sz, hv_mem.get())); 
                    // dg::network_stack_allocation::NoExceptAllocation<std::optional<MsgrFwdData>[]> msgrfwd_outbound_vec(sz);
                    auto msgrfwd_outbound_vec               = dg::vector<std::optional<MsgrFwdData>>(sz, std::optional<MsgrFwdData>(std::nullopt));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = data_arr[i];
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_msgrfwd_memevent_operatable_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_msgrfwd_forward_operatable_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_msgrfwd_logit_addr_nothrow(dst);
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_msgrfwd_observer_array_size_nothrow(dst);
                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(MAX_OBSERVER_ARR_SZ);
                        dg::network_tile_member_getsetter::get_msgrfwd_observer_array_nothrow(dst, dst_observer_arr.get());
                        dst_info_t dst_msgr_info                = dg::network_tile_member_getsetter::get_msgrfwd_dst_info_nothrow(dst);
                        tile_id_t dst_tile_id                   = dg::network_tile_member_getsetter::get_msgrfwd_tile_id_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_msgrfwd_dispatch_control_nothrow(dst);
                        
                        std::expected<operatable_id_t, exception_t> src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_forward_operatable_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);

                        if (!src_fwd_operatable_id.has_value() || !src_init_status.has_value() || !src_logit_umaptr.has_value()){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_fwd_operatable_id != src_fwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED && dst_init_status != TILE_INIT_STATUS_ADOPTED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        //if these aren't correct - it denotes a system problem - such is incorrect tile states - which is a much more serious problem (unhandled setters/ incorrect state snaps) - so we can safely assume that tile states are correct at all times

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code] = dg::network_exception_handler::nothrow_log(dg::network_dispatch_controller::decode_msgrfwd(dispatch_control));

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr.value(), src_vd_id}})){
                            dg::network_producer_consumer::delvrsrv_clear(host_vectorizer.get());
                            host_synchronizer.sync();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                        vma_ptr_t dst_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_logit_vmamap_reacquirer, dst_logit_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_vmaptr)){

                            dg::network_producer_consumer::delvrsrv_clear(host_vectorizer.get());
                            host_synchronizer.sync();
                            host_restrict_synchronizer.clear();
                        } 

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){ //we are doing cuda_ptr_t (cutf_ptr_t) -> host_ptr_t
                            auto dst_logit_hostptr      = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            auto src_logit_hostptr      = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            MsgrFwdData msgrfwd_data    = {};
                            msgrfwd_data.tile_id        = dst_tile_id;
                            msgrfwd_data.logit_value    = dg::string(dg::network_tileops_host_poly::get_byte_size(tileops_dp_code), ' '); //we want raw strings
                            msgrfwd_data.dst            = dst_msgr_dst;
                            msgrfwd_data.retry_count    = dst_msgr_retry_count;
                            msgrfwd_data.comm           = dst_msgr_comm;
                            host_ptr_t logit_value_data = msgrfwd_data.logit_value.data();
                            msgrfwd_outbound_vec[i]     = std::move(msgrfwd_data);

                            dg::network_producer_consumer::delvrsrv_deliver(host_vectorizer.get(), src_logit_hostptr, std::make_tuple(dst_logit_hostptr, src_logit_hostptr, logit_value_data, tileops_dp_code));
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(dst_observer_arr[j], dst_operatable_id)));
                        }
                    }

                    dg::network_producer_consumer::delvrsrv_clear(host_vectorizer.get());
                    host_synchronizer.sync();

                    for (size_t i = 0u; i < sz; ++i){
                        if (!msgrfwd_outbound_vec[i].has_value()){
                            continue;
                        }

                        EndUserPacket eu_packet = {};
                        eu_packet.kind          = EUPACKET_MSGRFWD; //serialization header
                        eu_packet.content       = dg::network_compact_serializer::serialize<dg::string>(LogitValue{msgrfwd_outbound_vec[i]->tile_id, std::move(msgrfwd_outbound_vec[i]->logit_value)}); //pollute hardware cache
                        eu_packet.dst           = msgrfwd_outbound_vec[i]->dst;
                        eu_packet.retry_count   = msgrfwd_outbound_vec[i]->retry_count;
                        eu_packet.urgency       = msgrfwd_outbound_vec[i]->urgency;
                        eu_packet.comm          = msgrfwd_outbound_vec[i]->comm;

                        dg::network_producer_consumer::delvrsrv_deliver(this->eu_packet_delivery_handle, std::move(eu_packet));
                    }
                }
            };
    };

    class ForwardDoMsgrBwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const size_t delivery_capacity;
            const size_t addrfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoMsgrBwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                            std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                            size_t delivery_capacity,
                                            size_t addrfetch_vectorization_sz,
                                            size_t region_vectorization_sz,
                                            size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                       host_async_device(std::move(host_async_device)),
                                                                                       cuda_async_device(std::move(cuda_async_device)),
                                                                                       delivery_capacity(delivery_capacity),
                                                                                       addrfetch_vectorization_sz(addrfetch_vectorization_sz),
                                                                                       region_vectorization_sz(region_vectorization_sz),
                                                                                       forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<uma_ptr_t>[]> descendant_arr(sz);

                const size_t EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_event_sz        = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_delivery_capacity    = std::min(this->delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost); 
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalDescendantAddressFetcher fetcher    = {};
                    size_t trimmed_addrfetch_vectorization_sz   = std::min(this->addrfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_addrfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_addrfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), lck_addr, std::make_tuple(event_arr[i].dst, event_arr[i].operatable_id, std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz); 
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        size_t lck_region_sz    = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_tile_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t dst_lck_addr  = dg::memult::region(dst_rcu_addr, lck_region_sz);

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(descendant_arr[i].value());
                        
                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        uma_ptr_t src_lck_addr  = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        auto key                = std::make_tuple(dst_lck_addr, src_lck_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(event_arr[i].dst, descendant_arr[i].value(), event_arr[i].operatable_id));
                    }
                }
            }
        
        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, operatable_id_t, std::optional<uma_ptr_t> *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t *> * data, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, expected_ops_id, fetching_addr]  = data[i];
                        init_status_t init_status                   = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(dst);
                        operatable_id_t current_ops_id              = dg::network_tile_member_getsetter::get_msgrbwd_memevent_operatable_id_nothrow(dst);

                        if (expected_ops_id != current_ops_id){
                            *fetching_addr = std::nullopt;
                            continue;
                        } 

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]:
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                *fetching_addr = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(dst);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle; //we dont care about aesthetic that much Mom - it's about code management - things that could be removed are removed in one component
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                size_t vectorization_sz;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * data_arr, size_t sz){

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));

                    auto umamap_reacquirer                  = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto dst_vmamap_reacquirer              = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_vmamap_reacquirer              = dg::network_vmamap::reacquirer_raii_initialize();

                    dg::network_cuda_controller::Synchronizer cuda_synchronizer{};
                    dg::network_cuda_controller::RestrictPointerSynchronizer cuda_restrict_synchronizer(cuda_synchronizer);
                    InternalCudaResolutor cuda_resolutor    = {};
                    cuda_resolutor.async_device             = this->cuda_async_device;
                    cuda_resolutor.synchronizer             = &cuda_synchronizer;
                    cuda_resolutor.restrict_synchronizer    = &cuda_restrict_synchronizer;

                    dg::network_host_controller::Synchronizer host_synchronizer{};
                    dg::network_host_controller::RestrictPointerSynchronizer host_restrict_synchronizer(host_synchronizer);
                    InternalHostResolutor host_resolutor    = {};
                    host_resolutor.async_device             = this->host_async_device;
                    host_resolutor.synchronizer             = &host_synchronizer;
                    host_resolutor.restrict_synchronizer    = &host_restrict_synchronizer;

                    size_t trimmed_host_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&host_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&host_resolutor, trimmed_host_vectorization_sz, hdh_mem.get())); 

                    size_t trimmed_cuda_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&cuda_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&cuda_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = data_arr[i];
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_msgrbwd_memevent_operatable_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_msgrbwd_forward_operatable_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_msgrbwd_logit_addr_nothrow(dst);
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_msgrbwd_observer_array_size_nothrow(dst);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_msgrbwd_dispatch_control_nothrow(dst);
                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(MAX_OBSERVER_ARR_SZ);
                        dg::network_tile_member_getsetter::get_msgrbwd_observer_array_nothrow(dst, dst_observer_arr.get());

                        std::expected<operatable_id_t, exception_t> src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_forward_operatable_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src); 

                        if (!src_fwd_operatable_id.has_value() || !src_init_status.has_value() || !src_logit_umaptr.has_value()){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_fwd_operatable_id != src_fwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED && dst_init_status != TILE_INIT_STATUS_ADOPTED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_msgrbwd(dispatch_control); //we'll convert tuple -> struct later 

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr.value(), src_vd_id}})){ //we assume the semantics of reacquirable == reachable vma_ptr_t(s) are unaffected - this needs to be documented
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            cuda_synchronizer.sync();
                            host_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr.value(), src_vd_id}});
                        vma_ptr_t dst_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_vmamap_reacquirer, dst_vmaptr) || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_vmamap_reacquirer, src_vmaptr)){ //we assume ...
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            cuda_synchronizer.sync();
                            host_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire(dst_vmamap_reacquirer, dst_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(src_vmamap_reacquirer, src_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto dst_cuda_ptr = dg::network_vmamap::get_cuda_ptr(dst_vmamap_reacquirer);
                            auto src_cuda_ptr = dg::network_vmamap::get_cuda_ptr(src_vmamap_reacquirer);

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), src_cuda_ptr, std::make_tuple(dst_cuda_ptr, src_cuda_ptr, tileops_dp_code));
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_host_ptr = dg::network_vmamap::get_host_ptr(dst_vmamap_reacquirer);
                            auto src_host_ptr = dg::network_vmamap::get_host_ptr(src_vmamap_reacquirer); 

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), src_host_ptr, std::make_tuple(dst_host_ptr, src_host_ptr, tileops_dp_code));
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }                        
                        }

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(dst_observer_arr[j], dst_operatable_id)));
                        }

                        set_init_status_msgrbwd_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                    }
                }
            };
    };

    class ForwardDoImmuSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        public:

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                (void) event_arr;
            }
    };

    class ForwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> mono_resolutor;
            const size_t mono_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> pair_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> immu_resolutor;
            const size_t immu_dispatch_sz;

        public:

            ForwardDoSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> leaf_resolutor,
                                     size_t leaf_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> blkr_resolutor,
                                     size_t blkr_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> mono_resolutor,
                                     size_t mono_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> pair_resolutor,
                                     size_t pair_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> uacm_resolutor,
                                     size_t uacm_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> pacm_resolutor,
                                     size_t pacm_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extnsrc_resolutor,
                                     size_t extnsrc_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extndst_resolutor,
                                     size_t extndst_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> crit_resolutor,
                                     size_t crit_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> msgrfwd_resolutor,
                                     size_t msgrfwd_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> msgrbwd_resolutor,
                                     size_t msgrbwd_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> immu_resolutor,
                                     size_t immu_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
                                                                        leaf_dispatch_sz(leaf_dispatch_sz),
                                                                        blkr_resolutor(std::move(blkr_resolutor)),
                                                                        blkr_dispatch_sz(blkr_dispatch_sz),
                                                                        mono_resolutor(std::move(mono_resolutor)),
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
                                                                        msgrbwd_dispatch_sz(msgrbwd_dispatch_sz),
                                                                        immu_resolutor(std::move(immu_resolutor)),
                                                                        immu_dispatch_sz(immu_dispatch_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                size_t trimmed_leaf_dispatch_sz     = std::min(this->leaf_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> leaf_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->leaf_resolutor.get(), trimmed_leaf_dispatch_sz)); 

                size_t trimmed_blkr_dispatch_sz     = std::min(this->blkr_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> blkr_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->blkr_resolutor.get(), trimmed_blkr_dispatch_sz));

                size_t trimmed_mono_dispatch_sz     = std::min(this->mono_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> mono_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->mono_resolutor.get(), trimmed_mono_dispatch_sz));

                size_t trimmed_pair_dispatch_sz     = std::min(this->pair_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> pair_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->pair_resolutor.get(), trimmed_pair_dispatch_sz));

                size_t trimmed_uacm_dispatch_sz     = std::min(this->uacm_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> uacm_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->uacm_resolutor.get(), trimmed_uacm_dispatch_sz));

                size_t trimmed_pacm_dispatch_sz     = std::min(this->pacm_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> pacm_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->pacm_resolutor.get(), trimmed_pacm_dispatch_sz));

                size_t trimmed_crit_dispatch_sz     = std::min(this->crit_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> crit_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->crit_resolutor.get(), trimmed_crit_dispatch_sz));

                size_t trimmed_immu_dispatch_sz     = std::min(this->immu_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> immu_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->immu_resolutor.get(), trimmed_immu_dispatch_sz));

                size_t trimmed_extnsrc_dispatch_sz  = std::min(this->extnsrc_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extnsrc_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extnsrc_resolutor.get(), trimmed_extnsrc_dispatch_sz));

                size_t trimmed_extndst_dispatch_sz  = std::min(this->extndst_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extndst_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extndst_resolutor.get(), trimmed_extndst_dispatch_sz));

                size_t trimmed_msgrfwd_dispatch_sz  = std::min(this->msgrfwd_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> msgrfwd_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->msgrfwd_resolutor.get(), trimmed_msgrfwd_dispatch_sz));

                size_t trimmed_msgrbwd_dispatch_sz  = std::min(this->msgrbwd_dispatch_sz, sz); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> msgrbwd_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->msgrbwd_resolutor.get(), trimmed_msgrbwd_dispatch_sz));

                auto leaf_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->leaf_resolutor.get(), trimmed_leaf_dispatch_sz, leaf_dh_mem.get()));
                auto blkr_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->blkr_resolutor.get(), trimmed_blkr_dispatch_sz, blkr_dh_mem.get()));
                auto mono_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->mono_resolutor.get(), trimmed_mono_dispatch_sz, mono_dh_mem.get()));
                auto pair_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->pair_resolutor.get(), trimmed_pair_dispatch_sz, pair_dh_mem.get()));
                auto uacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->uacm_resolutor.get(), trimmed_uacm_dispatch_sz, uacm_dh_mem.get()));
                auto pacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->pacm_resolutor.get(), trimmed_pacm_dispatch_sz, pacm_dh_mem.get()));
                auto crit_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->crit_resolutor.get(), trimmed_crit_dispatch_sz, crit_dh_mem.get()));
                auto immu_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->immu_resolutor.get(), trimmed_immu_dispatch_sz, immu_dh_mem.get()));
                auto extnsrc_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extnsrc_resolutor.get(), trimmed_extnsrc_dispatch_sz, extnsrc_dh_mem.get()));
                auto extndst_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extndst_resolutor.get(), trimmed_extndst_dispatch_sz, extndst_dh_mem.get()));
                auto msgrfwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrfwd_resolutor.get(), trimmed_msgrfwd_dispatch_sz, msgrfwd_dh_mem.get()));
                auto msgrbwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrbwd_resolutor.get(), trimmed_msgrbwd_dispatch_sz, msgrbwd_dh_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(event_arr[i].dst);

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(leaf_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_BLKR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(blkr_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MONO:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(mono_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_PAIR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pair_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_UACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(uacm_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_PACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pacm_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNSRC:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extnsrc_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNDST:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extndst_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_CRIT:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(crit_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRFWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrfwd_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRBWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrbwd_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_IMMU:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(immu_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        default:
                        {
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    };
                }
            }
    };

    //

    class BackwardDoLeafSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t region_vectorization_sz;

        public:

            BackwardDoLeafSignalResolutorV2(std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                            std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                            size_t region_vectorization_sz) noexcept: cuda_async_device(std::move(cuda_async_device)),
                                                                                      host_async_device(std::move(host_async_device)),
                                                                                      region_vectorization_sz(region_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                InternalResolutor internal_resolutor{};
                internal_resolutor.cuda_async_device    = this->cuda_async_device.get();
                internal_resolutor.host_async_device    = this->host_async_device.get();
                auto delivery_handle                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->region_vectorization_sz));

                for (size_t i = 0u; i < sz; ++i){
                    auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(event_arr[i].dst);

                    if (!ptrchk.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                        continue;
                    }

                    size_t lck_region_sz    = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                    uma_ptr_t rcu_addr      = dg::network_tile_member_getsetter::get_leaf_rcu_addr_nothrow(event_arr[i].dst);
                    uma_ptr_t lck_addr      = dg::memult::region(rcu_addr, lck_region_sz);

                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, event_arr[i].dst);
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    auto umamap_reacquirer          = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();
                    auto logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();
                    auto cuda_synchronizer          = dg::network_cuda_controller::CudaSynchronizer(this->cuda_async_device);
                    auto host_synchronizer          = dg::network_host_asynchronous::Synchronizer(this->host_async_device);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr                       = ptr_arr[i];
                        init_status_t init_status           = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(ptr);
                        grad_status_t grad_status           = dg::network_tile_member_getsetter::get_leaf_grad_status_nothrow(ptr);
                        dispatch_control_t dispatch_control = dg::network_tile_member_getsetter::get_leaf_backward_dispatch_control_nothrow(ptr); //
                        uma_ptr_t logit_umaptr              = dg::network_tile_member_getsetter::get_leaf_logit_addr_nothrow(ptr);
                        uma_ptr_t grad_umaptr               = dg::network_tile_member_getsetter::get_leaf_grad_addr_nothrow(ptr);

                        if (init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (grad_status != TILE_GRAD_STATUS_HAS_VALUE){ //we dont care how, why the gradient is at the leaf - if it is at the leaf - then the leaf is responsible for updating the gradients at backward_do signal
                            continue;
                        }

                        auto [logit_vd_id, grad_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_gradupdate_leaf(dispatch_control); //

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{logit_umaptr, logit_vd_id}, {grad_umaptr, grad_vd_id}})){
                            cuda_synchronizer.sync();
                            host_synchronizer.sync();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{logit_umaptr, logit_vd_id}, {grad_umaptr, grad_vd_id}});
                        vma_ptr_t logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(logit_vmamap_reacquirer, logit_vmaptr) 
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(grad_vmamap_reacquirer, grad_vmaptr)){
                            cuda_synchronizer.sync();
                            host_synchronizer.sync();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(logit_vmamap_reacquirer, logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(grad_vmamap_reacquirer, grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(logit_vmamap_reacquirer);
                            auto grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(grad_vmamap_reacquirer);
                            auto executable     = [=]() noexcept{ //the noexcept here is questionable
                                dg::network_tileops_cuda_poly::grad_update(logit_cudaptr, grad_cudaptr, tileops_dp_code, TILEOPS_POSTOPERATION_ZERO);
                            };
                            auto async_task     = dg::network_cuda_controller::virtualize_async_task(std::move(executable)); 
                            auto async_id       = dg::network_exception_handler::nothrow_log(this->cuda_async_device->exec(std::move(async_task))); //this must be an error in the next sprint

                            dg::network_exception_handler::nothrow_log(cuda_synchronizer.add(async_id));
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto logit_hostptr  = dg::network_vmamap::get_host_ptr(logit_vmamap_reacquirer);
                            auto grad_hostptr   = dg::network_vmamap::get_host_ptr(grad_vmamap_reacquirer);
                            auto executable     = [=]() noexcept{ //the noexcept here is questionable
                                dg::network_tileops_host_poly::grad_update(logit_hostptr, grad_hostptr, tileops_dp_code, TILEOPS_POSTOPERATION_ZERO);
                            };
                            auto async_task     = dg::network_host_asynchronous::virtualize_async_task(std::move(executable));
                            auto async_id       = dg::network_exception_handler::nothrow_log(this->host_async_device->exec(std::move(async_task))); //this must be an error in the next sprint

                            dg::network_exception_handler::nothrow_log(host_synchronizer.add(async_id));
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_leaf_grad_status_nothrow(ptr, TILE_GRAD_STATUS_ZEROED); //this guarantees pointer restriction
                    }
                }
            };
    };

    class BackwardDoMonoSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t addrfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t tileops_vectorization_sz;
        
        public:

            BackwardDoMonoSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                            std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                            size_t request_delivery_capacity,
                                            size_t addrfetch_vectorization_sz,
                                            size_t region_vectorization_sz,
                                            size_t tileops_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                       cuda_async_device(std::move(cuda_async_device)),
                                                                                       host_async_device(std::move(host_async_device)),
                                                                                       request_delivery_capacity(request_delivery_capacity),
                                                                                       addrfetch_vectorization_sz(addrfetch_vectorization_sz),
                                                                                       region_vectorization_sz(region_vectorization_sz),
                                                                                       tileops_vectorization_sz(tileops_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                auto request_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->request_delivery_capacity));
                auto descendant_arr             = std::make_unique<std::optional<uma_ptr_t>[]>(sz);

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->addrfetch_vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(event_arr[i].dst, std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.vectorization_sz         = this->tileops_vectorization_sz;
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->region_vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        size_t lck_region_sz                                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr                              = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].dst); //assumption: descendant_arr[i].has_value() => safe access
                        std::expected<ma_ptr_t, exception_t> src_rcu_addr   = dg::network_tile_member_getsetter::get_tile_rcu_addr(descendant_arr[i].value()); //polymorphic access - safe guard

                        if (!src_rcu_addr.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(src_rcu_addr.error()));
                            continue;
                        }

                        uma_ptr_t dst_lck_addr  = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        uma_ptr_t src_lck_addr  = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        auto key                = dg::utility::to_unique_representation(dst_lck_addr, src_lck_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(descendant_arr[i].value(), event_arr[i].dst));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, std::optional<uma_ptr_t> *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, std::optional<uma_ptr_t> *> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, fetching_addr]   = data_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(dst);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_addr = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(dst);
                                break;
                            }
                            default:
                            {
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        }
                    }
                }
            };
            
            struct InternalCudaResolutor: dg::network_producer_consumer::KVConsumerInterface<cuda_ptr_t, std::tuple<cuda_ptr_t, cuda_ptr_t, cuda_ptr_t, cuda_tileops_dispatch_control_t, grad_status_t>>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::CudaSynchronizer * synchronizer;
                dg::network_controller::RestrictPointerSynchronizer * restrict_synchronizer;

                void push(cuda_ptr_t key, std::tuple<cuda_ptr_t, cuda_ptr_t, cuda_ptr_t, cuda_tileops_dispatch_control_t, grad_status_t> * data_arr, size_t sz) noexcept{

                    auto cuda_ptr_vec   = dg::vector(sz * 3);
                    auto aggregator     = dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::aggregator_raiispawn_backward_mono(sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, lhs, rhs, tileops_dp_code, grad_status]  = data_arr[i];
                        cuda_ptr_vec[i * 3]                                 = dst;
                        cuda_ptr_vec[i * 3 + 1]                             = lhs;
                        cuda_ptr_vec[i * 3 + 2]                             = rhs;
                        dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::aggregator_add(aggregator, dst, lhs, rhs, tileops_dp_code, dg::value_if(grad_status == TILE_GRAD_STATUS_EMPTY, TILEOPS_OPERATION_ASSIGN, TILEOPS_OPERATION_ACCUM), TILEOPS_POSTOPERATION_ZERO));
                    }

                    auto executable     = [arg = std::move(aggregator)]() noexcept{
                        dg::network_tileops_cuda_poly::aggregator_exec(arg);
                    };
                    auto async_task     = dg::network_cuda_controller::virtualize_async_task(std::move(executable));
                    this->restrict_synchronizer->add(cuda_ptr_vec.begin(), cuda_ptr_vec.end());
                    auto async_id       = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(async_task))); //handle error next sprint - we must assume that device does not work 
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(async_id));
                }
            };

            struct InternalHostResolutor: dg::network_producer_consumer::KVConsumerInterface<host_ptr_t, std::tuple<host_ptr_t, host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t, grad_status_t>>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_controller::RestrictPointerSynchronizer * restrict_synchronizer;

                void push(host_ptr_t key, std::tuple<host_ptr_t, host_ptr_t, host_ptr_t, host_tileops_dispatch_control_t, grad_status_t> * data_arr, size_t sz) noexcept{

                    auto host_ptr_vec   = dg::vector(sz * 3);
                    auto aggregator     = dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::aggregator_raiispawn_backward_mono(sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, lhs, rhs, tileops_dp_code, grad_status]  = data_arr[i];
                        host_ptr_vec[i * 3]                                 = dst;
                        host_ptr_vec[i * 3 + 1]                             = lhs;
                        host_ptr_vec[i * 3 + 2]                             = rhs;
                        dg::network_exception_handler::nothrow_log(dg;:network_tileops_host_poly::aggregator_add(aggregator, dst, lhs, rhs, tileops_dp_code, dg::value_if(grad_status == TILE_GRAD_STATUS_EMPTY, TILEOPS_OPERATION_ASSIGN, TILEOPS_OPERATION_ACCUM), TILEOPS_POSTOPERATION_ZERO));
                    }

                    auto executable     = [arg = std::move(aggregator)]() noexcept{
                        dg::network_tileops_host_poly::aggregator_exec(arg);
                    };
                    auto async_task     = dg::network_host_asynchronous::virtualize_async_task(std::move(executable));
                    this->restrict_synchronizer->add(host_ptr_vec.begin(), host_ptr_vec.end());
                    auto async_id       = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(async_task))); //handle error next sprint - we must assume that device does not work 
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(async_id));
                }
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                size_t vectorization_sz;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));

                    auto umamap_reacquirer                          = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{});
                    auto dst_grad_vmamap_reacquirer                 = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_logit_vmamap_reacquirer                = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_grad_vmamap_reacquirer                 = dg::network_vmamap::reacquirer_raii_initialize();
                    auto cuda_synchronizer                          = dg::network_cuda_controller::CudaSynchronizer(this->cuda_async_device);
                    auto cuda_restrict_synchronizer                 = dg::network_controller::RestrictPointerSynchronizer(cuda_synchronizer);
                    auto host_synchronizer                          = dg::network_host_asynchronous::Synchronizer(this->host_async_device);
                    auto host_restrict_synchronizer                 = dg::network_controller::RestrictPointerSynchronizer(host_synchronizer);

                    auto internal_cuda_resolutor                    = InternalCudaResolutor();
                    auto internal_host_resolutor                    = InternalHostResolutor();

                    internal_cuda_resolutor.async_device            = this->cuda_async_device;
                    internal_cuda_resolutor.synchronizer            = &cuda_synchronizer;
                    internal_cuda_resolutor.restrict_synchronizer   = &cuda_restrict_synchronizer;

                    internal_host_resolutor.async_device            = this->host_async_device;
                    internal_host_resolutor.synchronizer            = &host_synchronizer;
                    internal_host_resolutor.restrict_synchronizer   = &host_restrict_synchronizer;

                    auto cuda_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_cuda_resolutor, this->vectorization_sz));
                    auto host_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_host_resolutor, this->vectorization_sz)); 

                    for (size_t i = 0u; i < sz; ++i){
                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != src_operatable_id){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto [src_grad_vd_id, src_logit_vd_id, dst_grad_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_bwd_mono(dispatch_control);

                        if (dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{src_grad_umaptr, src_grad_vd_id}, {src_logit_umaptr, src_logit_vd_id}, {dst_grad_umaptr, dst_grad_vd_id}})){
                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            cuda_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_synchronizer.sync();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_reacquire(umamap_reacquirer, {{src_grad_umaptr, src_grad_vd_id}, {src_logit_umaptr, src_logit_vd_id}, {dst_grad_umaptr, dst_grad_vd_id}});
                        vma_ptr_t src_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{}); 

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(src_grad_vmamap_reacquirer, src_grad_vmaptr) 
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_vmaptr) 
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(dst_grad_vmamap_reacquirer, dst_grad_vmaptr)){

                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            cuda_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_synchronizer.sync();
                            host_restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire(src_grad_vmamap_reacquirer, src_grad_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(src_logit_vmamap_reacquirer, src_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto src_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(src_grad_vmamap_reacquirer);
                            auto src_logit_cudaptr  = dg::networK_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            auto dst_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), src_grad_cudaptr, std::make_tuple(src_grad_cudaptr, src_logit_cudaptr, dst_grad_cudaptr, tileops_dp_code, src_grad_status));
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto src_grad_hostptr   = dg::network_vmamap::get_host_ptr(src_grad_vmamap_reacquirer);
                            auto src_logit_hostptr  = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            auto dst_grad_hostptr   = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), src_grad_hostptr, std::make_tuple(src_grad_hostptr, src_logit_hostptr, dst_grad_hostptr, tileops_dp_code, src_grad_status));
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_mono_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        dg::network_tile_member_getsetter::set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_HAS_VALUE);
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src)));
                    }
                }
            };
    };

    class BackwardDoPairSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t addrfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t tileops_vectorization_sz;
        
        public:

            BackwardDoPairSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> request_box,
                                            std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                            std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                            size_t request_delivery_capacity,
                                            size_t addrfetch_vectorization_sz,
                                            size_t region_vectorization_sz,
                                            size_t tileops_vectorization_sz) noexcept:  request_box(std::move(request_box)),
                                                                                        cuda_async_device(std::move(cuda_async_device)),
                                                                                        host_async_device(std::move(host_async_device)),
                                                                                        request_delivery_capacity(request_delivery_capacity),
                                                                                        addrfetch_vectorization_sz(addrfetch_vectorization_sz),
                                                                                        region_vectorization_sz(region_vectorization_sz),
                                                                                        tileops_vectorization_sz(tileops_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                auto request_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->request_delivery_capacity));
                auto descendant_arr             = std::make_unique<std::optional<std::tuple<uma_ptr_t, uma_ptr_t>>[]>(sz);

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->addrfetch_vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(event_arr[i].dst, std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.vectorization_sz         = this->tileops_vectorization_sz;
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->region_vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        auto [lhs_ptr, rhs_ptr]                             = descendant_arr[i].value();
                        uma_ptr_t dst_rcu_addr                              = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(event_arr[i].dst); //assumption: descendant_arr[i].has_value() => safe access
                        std::expected<uma_ptr_t, exception_t> lhs_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(lhs_ptr);
                        std::expected<uma_ptr_t, exception_t> rhs_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(rhs_ptr); //polymorphic access - every polymorphic access must issue a guard - because correct instantiations guaranteed by setters aren't equivalent to the existence of certain class members

                        if (!lhs_rcu_addr.has_value() || !rhs_rcu_addr.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(dg::network_exception::BAD_ACCESS));
                            continue;
                        }

                        size_t lck_region_sz    = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_lck_addr  = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        uma_ptr_t lhs_lck_addr  = dg::memult::region(lhs_rcu_addr.value(), lck_region_sz);
                        uma_ptr_t rhs_lck_addr  = dg::memult::region(rhs_rcu_addr.value(), lck_region_sz);
                        auto key                = dg::utility::to_unique_representation(dst_lck_addr, lhs_lck_addr, rhs_lck_addr); 

                        //left major (shared left, no shared right)  a + b
                        //right major (shared right, no shared left) a ^ b
                        //left right major (shared some right, shared some left)
                        //worst case scenerio: all shared right or left - synchronization overheads = # of dispatches - this is not expensive if we are dispatching host_asynchronous device - but very expensive if we are dispatching cuda_asynchronous - because the synchronization overheads would slow down the resolutor to do other tasks 
                        //by splitting "major" - we also increase locality of dispatches - not only reducing synchronization overheads

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(event_arr[i].dst, lhs_ptr, rhs_ptr));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, std::optional<std::tuple<uma_ptr_t, uma_ptr_t>> *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, std::optional<std::tuple<uma_ptr_t, uma_ptr_t>> *> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, fetching_addr]   = data_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(dst);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_addr = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr = std::make_tuple(dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(dst), dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(dst));
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
                }
            };

            struct InternalResolutor: dg::network_producer_consumer:KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                size_t vectorization_sz;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;

                void push(std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr), std::get<2>(lck_addr));

                    //alright guys - things is hard - we can assume these guys must have a rcu_lock to proceed - but we cant assume that all descendants have gradients (immu)
                    //we want to safeguard our assumption by doing polymorphic access and a no-err continue - this is defined behavior
                    //we are safe to assume that the tile state is correct at any given time - but we cant assume that certain tiles have certain members
                    //there isn't a better approach than just copy and paste these - it seems funny but this is for the best
                    //thing is actually hard if we cripple a gradient here

                    auto umamap_reacquirer                          = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 5u>{});
                    auto dst_grad_vmamap_reacquirer                 = dg::network_vmamap::reacquirer_raii_initialize();
                    auto lhs_grad_vmamap_reacquirer                 = dg::network_vmamap::reacquirer_raii_initialize();
                    auto lhs_logit_vmamap_reacquirer                = dg::network_vmamap::reacquirer_raii_initialize();
                    auto rhs_grad_vmamap_reacquirer                 = dg::network_vmamap::reacquirer_raii_initialize();
                    auto rhs_logit_vmamap_reacquirer                = dg::network_vmamap::reacquirer_raii_initialize();

                    auto cuda_synchronizer                          = dg::network_cuda_controller::CudaSynchronizer(this->cuda_async_device);
                    auto host_synchronizer                          = dg::network_host_asynchronous::Synchronizer(this->host_async_device);
                    auto cuda_restrict_synchronizer                 = dg::network_controller::RestrictPointerSynchronizer(cuda_synchronizer);
                    auto host_restrict_synchronizer                 = dg::network_controller::RestrictPointerSynchronizer(host_synchronizer);

                    auto cuda_internal_resolutor                    = CudaInternalResolutor();
                    auto host_internal_resolutor                    = HostInternalResolutor();

                    cuda_internal_resolutor.synchronizer            = &cuda_synchronizer;
                    cuda_internal_resolutor.restrict_synchronizer   = &cuda_restrict_synchronizer;
                    cuda_internal_resolutor.async_device            = this->cuda_async_device;

                    host_internal_resolutor.synchronizer            = &host_synchronizer;
                    host_internal_resolutor.restrict_synchronizer   = &host_restrict_synchronizer;
                    host_internal_resolutor.async_device            = this->host_async_device;  

                    auto cuda_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&cuda_internal_resolutor, this->vectorization_sz));
                    auto host_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&host_internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        if (dst_lhs != lhs){
                            continue;
                        }

                        if (dst_rhs != rhs){
                            continue;
                        }

                        if (dst_operatable_id != lhs_operatable_id){
                            continue;
                        }

                        if (dst_operatable_id != rhs_operatable_id){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (lhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (rhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        if (!dg::network_uma::reacquirer_is_reacquirable(umamap_reacquirer, {{lhs_logit_umaptr, dispatch_info.lhs_logit_vd_id}, {lhs_grad_umaptr, dispatch_info.lhs_grad_vd_id}, {rhs_logit_umaptr, dispatch_info.rhs_logit_vd_id}, {rhs_grad_umaptr, dispatch_info.rhs_grad_vd_id}, {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}})){;
                            dg::network_producer_consumer::delvrsrv_clear(cuda_delivery_handle.get());
                            dg::network_producer_consumer::delvrsrv_clear(host_delivery_handle.get());
                            cuda_synchronizer.sync();
                            cuda_restrict_synchronizer.clear();
                            host_synchronizer.sync();
                            host_restrict_synchronizer.clear();
                        }

                        // dg::network_uma::reacquirer_reacquire(umamap_reacquirer, {{lhs_logit_umaptr, dispatch_info.lhs_logit_vd_id}, {lhs_grad_umaptr, dispatch_info.lhs_grad_vd_id}, {rhs_logit_umaptr, dispatch_info.rhs_logit_vd_id}, {rhs_grad_umaptr, dispatch_info.rhs_grad_vd_id}, {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}});
                        // vma_ptr_t lhs_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        // vma_ptr_t lhs_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        // vma_ptr_t rhs_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});
                        // vma_ptr_t rhs_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 3u>{});
                        // vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 4u>{});

                        // if (!dg::network_vmamap::reacquirer_is_region_reacquirable(lhs_logit_vmamap_reacquirer, lhs_logit_vmaptr)
                        //     || !dg::network_vmamap::reacquirer_is_region_reacquirable(lhs_grad_vmamap_reacquirer, lhs_grad_vmaptr)
                        //     || !dg::network_vmamap::reacquirer_is_region_reacquirable(rhs_logit_vmamap_reacquirer, rhs_logit_vmaptr)
                        //     || !dg::network_vmamap::reacquirer_is_region_reacquirable(rhs_grad_vmamap_reacquirer, rhs_grad_vmaptr)
                        //     || !dg::network_vmamap::reacquirer_is_region_reacquirable(dst_grad_vmamap_reacquirer, dst_grad_vmaptr)){
                                
                        //     synchronizer.sync();
                        //     restrict_synchronizer.clear();
                        // }

                        // dg::network_vmamap::reacquirer_reacquire(lhs_logit_vmamap_reacquirer, lhs_logit_vmaptr);
                        // dg::network_vmamap::reacquirer_reacquire(lhs_grad_vmamap_reacquirer, lhs_grad_vmaptr);
                        // dg::network_vmamap::reacquirer_reacquire(rhs_logit_vmamap_reacquirer, rhs_logit_vmaptr);
                        // dg::network_vmamap::reacquirer_reacquire(rhs_grad_vmamap_reacquirer, rhs_grad_vmaptr);
                        // dg::network_vmamap::reacquirer_reacquire(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        // if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dp_device)){  //hmm - this is difficult - let's assume they must be on the same platform for now - to not overcomplicate things
                        //     auto lhs_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(lhs_logit_vmamap_reacquirer);
                        //     auto lhs_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(lhs_grad_vmamap_reacquirer);
                        //     auto rhs_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(rhs_logit_vmamap_reacquirer);
                        //     auto rhs_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(rhs_grad_vmamap_reacquirer);
                        //     auto dst_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);

                        //     restrict_synchronizer.add(lhs_logit_cudaptr, lhs_grad_cudaptr, rhs_logit_cudaptr, rhs_grad_cudaptr, dst_grad_cudaptr);
                        //     auto left_task          = dg::network_tileops_cuda_poly::async_make_task(this->async_device, );

                        //     if (!left_task.has_value()){
                        //         dg::network_log_stackdump::error_fast(dg::network_exception::verbose(left_task.error()));
                        //         continue;
                        //     }

                        //     auto right_task         = dg::network_tileops_cuda_poly::async_make_task(this->async_device, );

                        //     if (!right_task.has_value()){
                        //         dg::network_log_stackdump::error_fast(dg::network_exception::verbose(right_task.error()));
                        //         continue;
                        //     }

                        //     //we must rely on callee's asynchronous atomicity - we offload the bug there 

                        //     auto async_id           = dg::network_tileops_cuda_poly::async_exec(std::move(left_task.value()), std::move(right_task.value()));

                        //     if (!async_id.has_value()){
                        //         dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                        //         continue;
                        //     }

                        //     synchronizer.add(async_id.value());
                        // } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dp_device)){
                        //     auto lhs_logit_hostptr  = dg::network_vmamap::get_host_ptr(lhs_logit_vmamap_reacquirer);
                        //     auto lhs_grad_hostptr   = dg::network_vmamap::get_host_ptr(lhs_grad_vmamap_reacquirer);
                        //     auto rhs_logit_hostptr  = dg::network_vmamap::get_host_ptr(rhs_logit_vmamap_reacquirer);
                        //     auto rhs_grad_hostptr   = dg::network_vmamap::get_host_ptr(rhs_grad_vmamap_reacquirer);
                        //     auto dst_grad_hostptr   = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);

                        //     dg::network_tileops_host_poly::bwd_pair_lhs();
                        //     dg::network_tileops_host_poly::bwd_pair_rhs();
                        // } else{
                        //     if constexpr(DEBUG_MODE_FLAG){
                        //         dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        //         std::abort();
                        //     } else{
                        //         std::unreachable();
                        //     }
                        // }

                        // dg::network_tile_member_getsetter::set_tile_grad_status_nothrow(lhs, TILE_GRAD_STATUS_HAS_VALUE);
                        // dg::network_tile_member_getsetter::set_tile_grad_status_nothrow(rhs, TILE_GRAD_STATUS_HAS_VALUE);
                        // dg::network_tile_member_getsetter::set_pair_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        // dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(lhs)));
                        // dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(rhs)));
                    }
                }
            };         
    };
    
    class BackwardDoUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}
            
            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

            }
    };

    class BackwardDoPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
        
        public:

            BackwardDoPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

            }
    };

    class BackwardDoExtnSrcSignalResolutorV2: public virtual dg::network_produer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter;
            const size_t delivery_capacity;
            const size_t vectorization_sz;
        
        public:

            BackwardDoExtnSrcSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter,
                                               size_t delivery_capacity,
                                               size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                  alias_getter(std::move(alias_getter)),
                                                                                  delivery_capacity(delivery_capacity),
                                                                                  vectorization_sz(vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                auto descendant_localcounterpart_arr    = std::make_unique<std::optional<std::tuple<uma_ptr_t, uma_ptr_t>>[]>(sz);
                auto delivery_handle                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get()));

                {
                    InternalDescendantAddressFetcher fetcher{};
                    fetcher.alias_getter            = this->alias_getter->get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_localcounterpart_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(event_arr[i].dst, std::next(descendant_localcounterpart_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_localcounterpart_arr[i].has_value()){
                            continue;
                        }

                        auto [descendant, localcounterpart] = descendant_localcounterpart_arr[i];
                        uma_ptr_t descendant_rcu_addr       = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant);
                        uma_ptr_t localcounterpart_rcu_addr = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(localcounterpart);
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t descendant_rep_addr       = dg::memult::region(descendant_rcu_addr, std::min(dg::network_memops_uma::memlock_region_size(), dg::network_uma::memregion_size()));
                        uma_ptr_t localcounterpart_rep_addr = dg::memult::region(localcounterpart_rcu_addr, std::min(dg::network_memops_uma::memlock_region_size(), dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rep_addr              = dg::memult::region(dst_rcu_addr, std::min(dg::network_memops_uma::memlock_region_size(), dg::network_uma::memregion_size()));
                        auto key                            = dg::utility::to_unique_representation(descendant_rep_addr, localcounterpart_rep_addr, dst_rep_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(descendant, event_arr[i].dst, local_counterpart));
                    }
                }
            }
        
        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, std::optional<std::tuple<uma_ptr_t, uma_ptr_t>> *>>{ //fix semantics - tuple is loosely defined - not clear

                ForeignTileAliasGetterInterface * alias_getter;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, std::optional<std::tuple<uma_ptr_t, uma_ptr_t>> *> * data_arr, size_t sz){

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);
                    
                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, fetching_addr]   = data_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(dst);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_addr = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                uma_ptr_t descendant                        = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(dst);
                                uma_ptr_t counterpart                       = dg::network_tile_member_getsetter::get_extnsrc_counterpart_nothrow(dst);
                                std::optional<uma_ptr_t> local_counterpart  = this->alias_getter->alias(counterpart);

                                if (!local_counterpart.has_value()){
                                    *fetching_addr = std::nullopt;
                                } else{
                                    *fetching_addr = std::make_tuple(descendant, local_counterpart.value());
                                }

                                break;
                            }
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                        }
                    }
                }
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t> * data_arr, size_t sz) noexcept{

                    //we'll reiterate to get the requirements - and the failsafes - there are tons of bugs right now that I haven't skimmed through just yet

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr), std::get<2>(lck_addr));

                    auto umamap_reacquirer                          = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{});
                    auto localcounterpart_grad_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_logit_vmamap_reacquirer                = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_grad_vmamap_reacquirer                 = dg::network_vmamap::reacquirer_raii_initialize(); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto [src, dst, localcounterpart] = data_arr[i];

                        if (localcounterpart_counterpart != dst){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (localcounterpart_operatable_id != dst_operatable_id){
                            continue;
                        }

                        if (dst_operatable_id != src_operatable_id){
                            continue;
                        }

                        if (localcounterpart_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (localcounterpart_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_bwd_extnsrc(dispatch_control); //we'll work on the dispatch_kind later - extnsrc and extndst must be device_transparent - we'll work through the requirements - 

                        // if (!dg::network_uma::reacquirer_is_reacquirable(umamap_reacquirer, {{dst_grad_umaptr, dst_grad_vd_id}, {src_logit_umaptr, src_logit_vd_id}, {src_grad_umaptr, src_grad_vd_id}})){
                        //     synchronizer.sync();
                        //     restrict_synchronizer.clear();
                        // }

                        dg::network_uma::reacquirer_reacquire(umamap_reacquirer, {{localcounterpart_grad_umaptr, dst_grad_vd_id}, {src_logit_umaptr, src_logit_vd_id}, {src_grad_umaptr, src_grad_vd_id}});
                        vma_ptr_t localcounterpart_grad_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr              = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t src_grad_vmaptr               = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        // if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_grad_vmamap_reacquirer, dst_grad_vmaptr)
                        //     || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_vmaptr)
                        //     || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_grad_vmamap_reacquirer, src_grad_vmaptr)){

                        //     synchronizer.sync();
                        //     restrict_synchronizer.clear();
                        // }

                        dg::network_vmamap::reacquirer_reacquire(localcounterpart_grad_vmamap_reacquirer, dst_grad_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(src_logit_vmamap_reacquirer, src_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(src_grad_vmamap_reacquirer, src_grad_vmaptr);

                        if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_grad_hostptr   = dg::network_vmamap::get_host_ptr(localcounterpart_grad_vmamap_reacquirer);
                            auto src_logit_hostptr  = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            auto src_grad_hostptr   = dg::network_vmamap::get_host_ptr(src_grad_vmamap_reacquirer);

                            dg::network_tileops_host_poly::bwd_mono(src_grad_hostptr, src_logit_hostptr, dst_grad_hostptr, std::value_if(src_grad_status != TILE_GRAD_STATUS_EMPTY, TILEOPS_OPERATION_ACCUM, TILEOPS_OPERATION_ASSIGN), TILEOPS_POSTOPERATION_ZERO);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::INTERNAL_CORRUPTION);
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_extndst_grad_status_nothrow(localcounterpart, TILE_GRAD_STATUS_ZEROED);
                        dg::network_tile_member_getsetter::set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_HAS_VALUE);
                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src)));
                    }
                }
            };
    };

    class BackwardDoExtnDstSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            BackwardDoExtnDstSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                               std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                               std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                               size_t delivery_capacity,
                                               size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                  uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                  host_ip_retriever(std::move(host_ip_retriever)),
                                                                                  delivery_capacity(delivery_capacity),
                                                                                  vectorization_sz(vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    internal_resolutor.uma_ip_retriever         = this->uma_ip_retriever->get();
                    internal_resolutor.host_ip_retriever        = this->host_ip_retriever->get();
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i].dst);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{

                dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * request_delivery_handle;
                UnifiedMemoryIPRetrieverInterface * uma_ip_retriever;
                HostIPRetrieverInterface * host_ip_retriever;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(ptr_arr[i]);

                        if (init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (grad_status == TILE_INIT_STATUS_EMPTY){
                            continue;
                        }

                        uma_ptr_t counterpart   = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(ptr_arr[i]);
                        auto request            = Request<external_virtual_memory_event_t>{};
                        request.fr              = this->host_ip_retriever->ip();
                        request.to              = this->uma_ip_retriever->ip(counterpart);
                        request.content         = dg::network_external_memcommit_factory::make_sequential_event(dg::network_external_memcommit_factory::make_event_shadow_injection(ptr_arr[i], TILE_KIND_EXTNDST, dg::network_tile_member_getsetter::serialize_extndst(ptr_arr[i])),
                                                                                                                dg::network_external_memcommit_factory::make_event_backward_do_signal(counterpart));

                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(request));
                    }
                }
            };
    };

    class BackwardDoCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;

        public:

            BackwardDoCritSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          size_t delivery_capacity) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                for (size_t i = 0u; i < sz; ++i){
                    this->resolve(event_arr[i].dst, delivery_handle->get());
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

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src)));
            }
    };

    class BackwardDoMsgrFwdSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;
        
        public:

            BackwardDoMsgrFwdSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               size_t delivery_capacity,
                                               size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity),
                                                                                  vectorization_sz(vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                auto descendant_arr     = std::make_unique<uma_ptr_t[]>(sz);
                auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, event_arr[i].dst);
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        if (descendant_arr[i] == dg::pointer_limits<uma_ptr_t>::null_value()){
                            continue;
                        }

                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant_arr[i]);
                        uma_ptr_t dst_rep_addr  = dg::memult::region(dst_rcu_addr, std::min(dg::network_memops_uma::memlock_region_size(), dg::network_uma::memregion_size()));
                        uma_ptr_t src_rep_addr  = dg::memult::region(src_rcu_addr, std::min(dg::network_memops_uma::memlock_region_size(), dg::network_uma::memregion_size()));
                        auto key                = dg::utility::to_unique_representation(dst_rep_addr, src_rep_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(event_arr[i].dst, descendant_arr[i]));
                    }
                }
            }
        
        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t *> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, fetching_addr]   = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(dst);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_addr  = dg::pointer_limits<uma_ptr_t>::null_value();
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr  = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(dst); 
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

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));
                    auto umamap_reacquirer              = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{});
                    auto dst_grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src] = data_arr[i];

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != src_operatable_id){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }
                        
                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto [dst_grad_vd_id, src_grad_vd_id, src_logit_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_bwd_msgrfwd(dispatch_control);

                        if (!dg::network_umamap::reacquirer_is_reacquirable(umamap_reacquirer, {{dst_grad_umaptr, dst_grad_vd_id}, {src_grad_umaptr, src_grad_vd_id}, {src_logit_umaptr, src_logit_vd_id}})){
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_umamap::reacquirer_reacquire(umamap_reacquirer, {{dst_grad_umaptr, dst_grad_vd_id}, {src_grad_umaptr, src_grad_vd_id}, {src_logit_umaptr, src_logit_vd_id}});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_grad_vmamap_reacquirer, dst_grad_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_grad_vmamap_reacquirer, src_grad_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_vmaptr)){
                                
                                synchronizer.sync();
                                restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(src_grad_vmamap_reacquirer, src_grad_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if (dg::network_dispatch_controller::is_cuda_dispatch()){
                            auto dst_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);
                            auto src_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(src_grad_vmamap_reacquirer);
                            auto src_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            restrict_synchronizer.add(dst_grad_cudaptr, src_grad_cudaptr, src_logit_cudaptr);
                            auto async_id           = dg::network_tileops_cuda_poly::async_bwd_mono(src_grad_cudaptr, src_logit_cudaptr, dst_grad_cudaptr, tileops_dp_code, dg::value_if(src_grad_status != TILE_GRAD_STATUS_EMPTY, TILEOPS_OPERATION_ACCUM, TILEOPS_OPERATION_ASSIGN), TILEOPS_POSTOPERATION_ZERO);

                            if (!async_id.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                                continue;
                            }

                            synchronizer.add(async_id.value());
                        } else if (dg::network_dispatch_controller::is_host_dispatch()){
                            auto dst_grad_hostptr   = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);
                            auto src_grad_hostptr   = dg::network_vmamap::get_host_ptr(src_grad_vmamap_reacquirer);
                            auto src_logit_hostptr  = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);

                            dg::network_tileops_host_poly::bwd_mono(src_grad_hostptr, src_logit_hostptr, dst_grad_hostptr, tileops_dp_code);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_msgrfwd_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        dg::network_tile_member_getsetter::set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_HAS_VALUE);
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src)));
                    }
                }
            };
    };

    class BackwardDoMsgrBwdSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box;
            const size_t request_delivery_capacity;
            const size_t eu_packet_delivery_capacity;
            const size_t vectorization_sz;
        
        public:

            BackwardDoMsgrBwdSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box,
                                               size_t request_delivery_capacity,
                                               size_t eu_packet_delivery_capacity,
                                               size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                  eu_packet_box(std::move(eu_packet_box)),
                                                                                  request_delivery_capacity(request_delivery_capacity),
                                                                                  eu_packet_delivery_capacity(eu_packet_delivery_capacity),
                                                                                  vectorization_sz(vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                auto descendant_arr             = std::make_unique<uma_ptr_t[]>(sz);
                auto request_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->request_delivery_capacity));
                auto eu_packet_delivery_handle  = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->eu_packet_box.get(), this->eu_packet_delivery_capacity));

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = dg::pointer_limits<uma_ptr_t>::null_value();
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(event_arr[i].dst, std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle     = request_delivery_handle.get();
                    internal_resolutor.eu_packet_delivery_handle    = eu_packet_delivery_handle.get();
                    auto vectorized_delivery_handle                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        if (descendant_arr[i] == dg::pointer_limits<uma_ptr_t>::null_value()){
                            continue;
                        }

                        uma_ptr_t src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant_arr[i]);
                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t src_rep_addr  = dg::memult::region(src_rcu_addr, std::min(dg::network_memops_uma::memlock_region_size(), dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rep_addr  = dg::memult::region(dst_rcu_addr, std::min(dg::network_memops_uma::memlock_region_size(), dg::network_uma::memregion_size()));
                        auto key                = dg::utility::to_unique_representation(src_rep_addr, dst_rep_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(event_arr[i].dst, descendant_arr[i]));
                    }
                }
            }
        
        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVCosnumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t *> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, fetching_addr]   = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(dst);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {   
                                *fetching_addr  = dg::pointer_limits<uma_ptr_t>::null_value();
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr  = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(dst);
                                break;
                            }
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                        }
                    }
                }
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_producer_consumer::DeliveryHandle<EndUserPacket> * eu_packet_delivery_handle;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));

                    auto umamap_reacquirer              = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{});
                    auto dst_grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();

                    dg::network_cuda_controller::CudaSynchronizer synchronizer{};
                    dg::network_controller::RestrictPointerSynchronizer restrict_synchronizer(synchronizer);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src] = data_arr[i];

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != src_operatable_id){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_grad_umaptr, dst_grad_vd_id}, {src_grad_umaptr, src_grad_vd_id}, {src_logit_umaptr, src_logit_vd_id}})){
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_grad_umaptr, dst_grad_vd_id}, {src_grad_umaptr, src_grad_vd_id}, {src_logit_umaptr, src_logit_vd_id}});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{}); 

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_grad_vmamap_reacquirer, dst_grad_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_grad_vmamap_reacquirer, src_grad_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_vmaptr)){

                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(src_grad_vmamap_reacquirer, src_grad_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){

                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){

                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_msgrbwd_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        dg::network_tile_member_getsetter::set_tile_grad_status_nothrow(src, TILE_GRAD_STATUS_HAS_VALUE);
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src)));
                        // dg::network_producer_consumer::delvrsrv_deliver(this->eu_packet_delivery_handle, std::move(eu_packet));
                    }

                }
            };
    };

    class BackwardDoImmuSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        public:

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                (void) event_arr;
            }
    };

    class BackwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

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
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> immu_resolutor;
            const size_t immu_dispatch_sz;

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
                                      size_t msgrbwd_dispatch_sz,
                                      std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> immu_resolutor,
                                      size_t immu_dispatch_sz) noexcept:    leaf_resolutor(std::move(leaf_resolutor)),
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
                                                                            msgrbwd_dispatch_sz(msgrbwd_dispatch_sz),
                                                                            immu_resolutor(std::move(immu_resolutor)),
                                                                            immu_dispatch_sz(immu_dispatch_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                size_t trimmed_leaf_dispatch_sz     = std::min(this->leaf_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> leaf_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->leaf_resolutor.get(), trimmed_leaf_dispatch_sz));

                size_t trimmed_mono_dispatch_sz     = std::min(this->mono_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> mono_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->mono_resolutor.get(), trimmed_mono_dispatch_sz));

                size_t trimmed_pair_dispatch_sz     = std::min(this->pair_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> pair_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->pair_resolutor.get(), trimmed_pair_dispatch_sz));

                size_t trimmed_uacm_dispatch_sz     = std::min(this->uacm_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> uacm_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->uacm_resolutor.get(), trimmed_uacm_dispatch_sz));

                size_t trimmed_pacm_dispatch_sz     = std::min(this->pacm_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> pacm_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->pacm_resolutor.get(), trimmed_pacm_dispatch_sz));

                size_t trimmed_crit_dispatch_sz     = std::min(this->crit_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> crit_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->crit_resolutor.get(), trimmed_crit_dispatch_sz));

                size_t trimmed_immu_dispatch_sz     = std::min(this->immu_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> immu_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->immu_resolutor.get(), trimmed_immu_dispatch_sz));

                size_t trimmed_extnsrc_dispatch_sz  = std::min(this->extnsrc_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extnsrc_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extnsrc_resolutor.get(), trimmed_extnsrc_dispatch_sz));

                size_t trimmed_extndst_dispatch_sz  = std::min(this->extndst_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extndst_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extndst_resolutor.get(), trimmed_extndst_dispatch_sz));
                
                size_t trimmed_msgrfwd_dispatch_sz  = std::min(this->msgrfwd_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> msgrfwd_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->msgrfwd_resolutor.get(), trimmed_msgrfwd_dispatch_sz));

                size_t trimmed_msgrbwd_dispatch_sz  = std::min(this->msgrbwd_dispatch_sz, sz); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> msgrbwd_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->msgrbwd_resolutor.get(), trimmed_msgrbwd_dispatch_sz));

                auto leaf_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->leaf_resolutor.get(), trimmed_leaf_dispatch_sz, leaf_dh_mem.get()));
                auto blkr_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihanlde(this->blkr_resolutor.get(), trimmed_blkr_dispatch_sz, blkr_dh_mem.get()));
                auto mono_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->mono_resolutor.get(), trimmed_mono_dispatch_sz, mono_dh_mem.get()));
                auto pair_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->pair_resolutor.get(), trimmed_pair_dispatch_sz, pair_dh_mem.get()));
                auto uacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->uacm_resolutor.get(), trimmed_uacm_dispatch_sz, uacm_dh_mem.get()));
                auto pacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->pacm_resolutor.get(), trimmed_pacm_dispatch_sz, pacm_dh_mem.get()));
                auto crit_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->crit_resolutor.get(), trimmed_crit_dispatch_sz, crit_dh_mem.get()));
                auto immu_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->immu_resolutor.get(), trimmed_immu_dispatch_sz, immu_dh_mem.get()));
                auto extnsrc_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extnsrc_resolutor.get(), trimmed_extnsrc_dispatch_sz, extnsrc_dh_mem.get()));
                auto extndst_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extndst_resolutor.get(), trimmed_extndst_dispatch_sz, extndst_dh_mem.get()));
                auto msgrfwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrfwd_resolutor.get(), trimmed_msgrfwd_dispatch_sz, msgrfwd_dh_mem.get()));
                auto msgrbwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrbwd_resolutor.get(), trimmed_msgrbwd_dispatch_sz, msgrbwd_dh_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(event_arr[i].dst);

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(leaf_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_BLKR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(blkr_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MONO:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(mono_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_PAIR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pair_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_UACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(uacm_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_PACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pacm_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNSRC:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extnsrc_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNDST:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extndst_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_CRIT:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(crit_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRFWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrfwd_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRBWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrbwd_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_IMMU:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(immu_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        default:
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                    }
                }
            }
    };

    class MemCommitResolutor: public virtual dg::network_concurrency::WorkerInterface{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>> producer;
            const size_t producer_consume_capacity;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> fwd_ping_signal_resolutor;
            const size_t fwd_ping_delivery_capacity; 
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> fwd_pong_request_resolutor;
            const size_t fwd_pong_request_delivery_capacity;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongSignalEvent>> fwd_pong_signal_resolutor;
            const size_t fwd_pong_signal_delivery_capacity;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> fwd_pingpong_request_resolutor;
            const size_t fwd_pingpong_request_delivery_capcity;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> fwd_do_resolutor;
            const size_t fwd_do_delivery_capacity;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>> bwd_do_resolutor;
            const size_t bwd_do_delivery_capacity;

        public:

            MemCommitResolutor(std::shared_ptr<dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>> producer,
                               size_t producer_consume_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> fwd_ping_signal_resolutor,
                               size_t fwd_ping_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> fwd_pong_request_resolutor,
                               size_t fwd_pong_request_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongSignalEvent>> fwd_pong_signal_resolutor,
                               size_t fwd_pong_signal_delivery_capacity, 
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> fwd_pingpong_request_resolutor,
                               size_t fwd_pingpong_request_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> fwd_do_resolutor,
                               size_t fwd_do_delivery_capacity,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>> bwd_do_resolutor,
                               size_t bwd_do_delivery_capacity) noexcept: producer(std::move(producer)),
                                                                          producer_consume_capacity(producer_consume_capacity),
                                                                          fwd_ping_signal_resolutor(std::move(fwd_ping_signal_resolutor)),
                                                                          fwd_ping_delivery_capacity(fwd_ping_delivery_capacity),
                                                                          fwd_pong_request_resolutor(std::move(fwd_pong_request_resolutor)),
                                                                          fwd_pong_request_delivery_capacity(fwd_pong_request_delivery_capacity),
                                                                          fwd_pong_signal_resolutor(std::move(fwd_pong_signal_resolutor)),
                                                                          fwd_pong_signal_delivery_capacity(fwd_pong_signal_delivery_capacity),
                                                                          fwd_pingpong_request_resolutor(std::move(fwd_pingpong_request_resolutor)),
                                                                          fwd_pingpong_request_delivery_capacity(fwd_pingpong_request_delivery_capacity),
                                                                          fwd_do_resolutor(std::move(fwd_do_resolutor)),
                                                                          fwd_do_delivery_capacity(fwd_do_delivery_capacity),
                                                                          bwd_do_resolutor(std::move(bwd_do_resolutor)),
                                                                          bwd_do_delivery_capacity(bwd_do_delivery_capacity){}

            bool run_one_epoch() noexcept{

                //host_concurrency is good at branching + cpu flops but not good at fetching memories - so we must minimize the memory fetching here - and try our best to do branching optimizations (optional) - especially dispatch tables where the branch is not optimized like if else 

                dg::network_stack_allocation::NoExceptRawIfPossibleAllocation<virtual_memory_event_t[]> virtual_memory_event_arr(this->producer_consume_capacity);
                size_t virtual_memory_event_sz = {};
                this->producer->get(virtual_memory_event_arr.get(), virtual_memory_event_sz, this->producer_consume_capacity);

                if (virtual_memory_event_sz == 0u){
                    return false;
                }

                auto fwd_ping_signal_lambda_consumer        = [this](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    dg::network_stack_allocation::NoExceptRawIfPossibleAllocation<ForwardPingSignalEvent[]> tmp_arr(arr_sz);

                    for (size_t i = 0u; i < arr_sz; ++i){
                        tmp_arr[i] = dg::network_memcommit_factory::devirtualize_forward_ping_signal_event(event_arr[i]);
                    }

                    this->fwd_ping_signal_resolutor->push(tmp_arr.get(), arr_sz);
                };

                auto fwd_pong_request_lambda_consumer       = [this](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    dg::network_stack_allocation::NoExceptRawIfPossibleAllocation<ForwardPongRequestEvent[]> tmp_arr(arr_sz);

                    for (size_t i = 0u; i < arr_sz; ++i){
                        tmp_arr[i] = dg::network_memcommit_factory::devirtualize_forward_pong_request_event(event_arr[i]);
                    }

                    this->fwd_pong_request_resolutor->push(tmp_arr.get(), arr_sz);
                };

                auto fwd_pingpong_request_lambda_consumer   = [this](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    dg::network_stack_allocation::NoExceptRawIfPossibleAllocation<ForwardPingPongRequestEvent[]> tmp_arr(arr_sz);

                    for (size_t i = 0u; i < arr_sz; ++i){
                        tmp_arr[i] = dg::network_memcommit_factory::devirtualize_forward_pingpong_request_event(event_arr[i]);
                    }

                    this->fwd_pingpong_request_resolutor->push(tmp_arr.get(), arr_sz);
                };

                auto fwd_pong_signal_lambda_consumer        = [this](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    dg::network_stack_allocation::NoExceptRawIfPossibleAllocation<ForwardPongSignalEvent[]> tmp_arr(arr_sz);

                    for (size_t i = 0u; i < arr_sz; ++i){
                        tmp_arr[i] = dg::network_memcommit_factory::devirtualize_forward_pong_signal_event(event_arr[i]);
                    }

                    this->fwd_pong_signal_resolutor->push(tmp_arr.get(), arr_sz);
                };

                auto fwd_do_lambda_consumer                 = [this](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    dg::network_stack_allocation::NoExceptRawIfPossibleAllocation<ForwardDoSignalEvent[]> tmp_arr(arr_sz);

                    for (size_t i = 0u; i < arr_sz; ++i){
                        tmp_arr[i] = dg::network_memcommit_factory::devirtualize_forward_do_signal_event(event_arr[i]);
                    }

                    this->fwd_do_resolutor->push(tmp_arr.get(), arr_sz);
                };

                auto bwd_do_lambda_consumer                 = [this](virtual_memory_event_t * event_arr, size_t arr_sz) noexcept{
                    dg::network_stack_allocation::NoExceptRawIfPossibleAllocation<BackwardDoSignalEvent[]> tmp_arr(arr_sz);

                    for (size_t i = 0u; i < arr_sz; ++i){
                        tmp_arr[i] = dg::network_memcommit_factory::devirtualize_backward_do_signal_event(event_arr[i]);
                    }

                    this->bwd_do_resolutor->push(tmp_arr.get(), arr_sz);
                };

                auto fwd_ping_signal_virtual_consumer               = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_ping_signal_lambda_consumer)>(fwd_ping_signal_lambda_consumer);
                size_t trimmed_fwd_ping_signal_delivery_capacity    = std::min(this->fwd_ping_signal_delivery_capacity, sz);
                size_t fpsdh_allocation_cost                        = dg::network_producer_consumer::delvrsrv_allocation_cost(&fwd_ping_signal_virtual_consumer, trimmed_fwd_ping_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> fpsdh_mem(fpsdh_allocation_cost);
                auto fwd_ping_signal_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&fwd_ping_signal_virtual_consumer, trimmed_fwd_ping_signal_delivery_capacity, fpsdh_mem.get()));

                auto fwd_pong_request_virtual_consumer              = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_pong_request_lambda_consumer)>(fwd_pong_request_lambda_consumer);
                size_t trimmed_fwd_pong_request_delivery_capacity   = std::min(this->fwd_pong_request_delivery_capacity, sz);
                size_t fpqdh_allocation_cost                        = dg::network_producer_consumer::delvrsrv_allocation_cost(&fwd_pong_request_virtual_consumer, trimmed_fwd_pong_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> fpqdh_mem(fpqdh_allocation_cost);
                auto fwd_pong_request_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&fwd_pong_request_virtual_consumer, trimmed_fwd_pong_request_delivery_capacity, fpqdh_mem.get()));

                auto fwd_pong_signal_virtual_consumer               = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_pong_signal_lambda_consumer)>(fwd_pong_signal_lambda_consumer);
                size_t trimmed_fwd_pong_signal_delivery_capacity    = std::min(this->fwd_pong_signal_delivery_capacity, sz);
                size_t fposdh_allocation_cost                       = dg::network_producer_consumer::delvrsrv_allocation_cost(&fwd_pong_signal_virtual_consuemr, trimmed_fwd_pong_signal_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> fposdh_mem(fposdh_allocation_cost); 
                auto fwd_pong_signal_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&fwd_pong_signal_virtual_consumer, trimmed_fwd_pong_signal_delivery_capacity, fposdh_mem.get()));

                auto fwd_do_virtual_consumer                        = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_do_lambda_consumer)>(fwd_do_lambda_consumer);
                size_t trimmed_fwd_do_delivery_capacity             = std::min(this->fwd_do_delivery_capacity, sz);
                size_t fddh_allocation_cost                         = dg::network_producer_consumer::delvrsrv_allocation_cost(&fwd_do_virtual_consumer, trimmed_fwd_do_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> fddh_mem(fddh_allocation_cost)
                auto fwd_do_delivery_handle                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&fwd_do_virtual_consumer, trimmed_fwd_do_delivery_capacity, fddh_mem.get()));

                auto bwd_do_virtual_consumer                        = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(bwd_do_lambda_consumer)>(bwd_do_lambda_consumer);
                size_t trimmed_bwd_do_delivery_capacity             = std::min(this->bwd_do_delivery_capacity, sz);
                size_t bddh_allocation_cost                         = dg::network_producer_consumer::delvrsrv_allocation_cost(&bwd_do_virtual_consumer, trimmed_bwd_do_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> bddh_mem(bddh_allocation_cost); 
                auto bwd_do_delivery_handle                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&bwd_do_virtual_consumer, trimmed_bwd_do_delivery_capacity, bddh_mem.get()));

                auto fwd_pingpong_request_virtual_consumer          = dg::network_producer_consumer::LambdaWrappedConsumer<virtual_memory_event_t, decltype(fwd_pingpong_request_lambda_consumer)>(fwd_pingpong_request_lambda_consumer);
                size_t trimmed_fwd_pgpg_request_delivery_capacity   = std::min(this->fwd_pingpong_request_delivery_capcity, sz);
                size_t fpprdh_allocation_cost                       = dg::network_producer_consumer::delvrsrv_allocation_cost(&fwd_pingpong_request_virtual_consumer, trimmed_fwd_pgpg_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> fpprdh_mem(fpprdh_allocation_cost);
                auto fwd_pingpong_request_delivery_handle           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&fwd_pingpong_request_virtual_consumer, trimmed_fwd_pgpg_request_delivery_capacity, fpprdh_mem.get()));

                for (size_t i = 0u; i < virtual_memory_event_sz; ++i){
                    memory_event_kind_t event_kind = dg::network_memcommit_factory::read_event_kind(virtual_memory_event_arr[i]);

                    switch (event_kind){
                        case dg::network_memcommit_factory::event_kind_forward_ping_signal:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_ping_signal_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        }
                        case dg::network_memcommit_factory::event_kind_forward_pong_request:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_pong_request_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        }
                        case dg::network_memcommit_factory::event_kind_forward_pingpong_request:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_pingpong_request_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        }
                        case dg::network_memcommit_factory::event_kind_forward_pong_signal:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_pong_signal_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        }
                        case dg::network_memcommit_factory::event_kind_forward_do_signal:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_do_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        }
                        case dg::network_memcommit_factory::event_kind_backward_do_signal:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(bwd_do_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        }
                        default:
                        {
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }
                    }
                }

                return true;
            }
    };
}

#endif