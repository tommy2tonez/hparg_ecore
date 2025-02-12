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

    //alrights - been doing technical analysis
    //assume this successful serialized process:
    //user ingests brain
    //user allocates tiles
    //user signals
    //user invalidates signals
    //user allocates tiles
    //per client requests - we are doing 1024 x 1024 tiles - or 512x512 tiles - we admit that matrix dispatch locality is matrix's responsisiblity - yet we still do vectorized optimizations because that's another radix of optimizations (in particular, synchronization optimizations - we dont want restriction to increase the spinlock overhead which would slowdown the system) (we can't say that fattening the matrix and changing dispatch_control_t would solve everything - it's partially true - but not optimization-wise true) - we'll work on the matrix part later - maybe involving jit and and bytecode or runtime compilations - we'll discuss that part later 
    //we want concurrent training on 100.000 GPU clusters as testing phase
    //we want to leverage memregion frequencies to avoid backward synchro problem + packet backward problems
    //after the testing phase - we'll open source this to be the community version (people sell their hardware to the cause and actually get paid (bid | ask) for something that's actually meaningful - not for mining coins)
    //everyone can have their own super lab model from their devices

    //I was thinking of init - reinit (reusing computation tree) without orphan in between - and the result is permuted computation tree that would dangerously lead to not a number
    //orphan acts as the signal interceptor of the previous computation tree (such does not allow to corrupt the tree - but to stop the computation gracefully)

    //we only care about what happened in between the time slice [entry_of_first_orphan_node, exit_of_last_orphan_node)
    //we dont care about pre-orphan or post-orphan 

    //because pre-orphan is guaranteed to be correct
    //and post-orphan is guaranteed to invalidate the signals

    //let's do code path analysis
    //we have a computation tree - we signal - it forwards - it backwards - exits successfully - everyone's happy

    //we have a computation tree - we signal - it forwards mid way - we change the tree - it does not exit successfuly - everyone's not happy
    //- if we change the tree by using init and the descendant and the root share the same operatable_forward_id but not operatable_memevent_id - we are risking tree corruptions - we have permutated computation tree - such is the old and the new tree are both valid (i'm talking recursively)
    //- if we change the tree by using orphan and the descendant and the root are not the orphaned nodes - it continuing forwards correctly - the computation tree is not altered
    //- if we change the tree by using orphan and one of the descendants is orphaned - the computation tree is not corrupted but stopped
    //- if we change the tree by using orphan and the root is orphaned - the computation tree is not corrupted but stopped   

    //- backward wise:
    //- assume that g(x) = y, where x = root, is a self sufficient gradient in terms of update, every interval I
    //- we would want to orphan the tiles from bottom to top (level-wise)
    //- we want to prove that this does not interfere with training process
    //- let's on focus on the base (not leafs)
    //- assume that we invoke orphan on an arbitrary node
    //- because every leaf update is a sufficient update (atomicity - assumption) - then an orphan of the base nodes at any point in time (we are talking in terms of the alloted time) is a valid operation
    //- this requires the base_node to be a variant of mono tile, says inverse_blkr tile whose descendants are leaf_tiles - and the leaf tile has uniform distribution or detached distribution of influence on the output - this is another research topic
    //- because the update every interval I happens atomically in this sense for the leaf nodes - so an orphan/deinit order at any point in time is a valid operation 

    //- the code looks like it could crash at any moment but actually this is good code
    //- we include fail safes - such is a mis-operation on tiles must be a through operation with no-error
    //- we include polymorphic checks
    //- we do correct state snaps for tiles - such that tiles' states must be correct at all times, guaranteed internally

    //- we might have to support cuda and host dispatches for all tiles - and make platform dispatch an optimizable
    //- alrights - after the program runs smoothly - we want to sandbox the program by using containerization - and allocate finite resources - to make sure the program does not have external interferences and crash
    //- we might have to not abort common errors - and do user-logs - by adding user_id to tiles   
    //- we also can't be too cautious catching every exceptions because it has very little ROI, not readable, and, sometimes, very buggy - says we have a leak internally which leads to OOM (which should have not happened) - and now our program stalls because of the leak instead of aborting
    //- this practice is actually a virtue in software engineering, we care about 99.999999999% runnability - we don't invest in the 0.000000001%
    //- reality shows that programs that abort correctly are preferred over programs that continue on errors
    //- postgres doesnt abort on fsync and they risk silent data corruption
    //- Apple does not abort on memory exhaustion and they have unfixable BIOS error
    //- Microsoft, Google Chrome, etc. 
    //- we aren't following the footsteps - we abort if things go south

    //- as of now - the program is, theoretically, runnable
    //- we'll work on the static operations, we believe that there is no neural network models - only statistics, maths, optimizations and paths - so users should not be the ones that specify the models - only the x and y in f(x) -> y


    //we get the what might've worked - and follow best practices of atomicity + self-concurrent-sufficient + relaxed function
    //except open, except operations, noexcept close - noexcept reverse
    //alright - we must do noexcept for the socket for special implementations - we offload the stack unwinding of exception resolution to the caller
    //then we'd want to look at what might've not worked (things like volatile + std::atomic_signal_fence(std::memory_order_seq_cst) for transactional concurrency (usually open-close)) - things like cuda crash + ssd fails + silent packet lost + etc. (and we implementation hotfixes for this)
    //we'd skip exotic exceptions like memory exhaustion or incorrect internal state snaps or incorrect assumptions because it would expose silent internal corruptions which are very hard to trace + fix 
    //got a feedback to do 0xFF x 0xFF
    //we got an ocean to swim - probably 2 - 3 million lines of code to write
    //the topics we want to cover is low level optimizations
    //regex optimization of machine learning models
    //path optimization of machine learning models
    //guided training
    //efficient distributed computing of logits
    //supervisor controller
    //multi-containerization database
    //ACID property of database - anchor
    //json + efficient communication (somewhat like Flask - everybody loves Python Flask - admit it)
    //load balancer
    //network packet balancer (UDP protocol)
    //etc.

    //we'll talk to apache later

    //^^^

    //we are declaring dependency injection expectations
    //users of the resolutors must coerce (implements) interface by composition of std::shared_ptr<>

    //alright Mom - you say this is B - but I say this is probably the most optimized form this could be programmed - we dont want to tie our hands to limit future optimization efforts
    //we must be able to do 128x128 or 64x64 tiles because it's the most compact form for UDP protocol transfer without risking frictions (our tcp things) 
    //and at that level of 128x128 or 64x64 - we must do pair_accum + unordered_accum for "linear" operations
    //we must use fixed size UACM_ACM_SZ and PACM_ACM_SZ because we want to control the frequency uncertainty - things would get very out of hands if we aren't controlling the accumulation size
    //and I think that fixed size UACM_ACM_SZ should mimic the logic of unordered accum for most of the cases, so does fixed size PACM_ACM_SZ - we dont want to waste storage and memory footprint for "linear" - that's the goal
    //the magic of our engine is the parallel of path optimizations - thing is that traditional training exposes a very clear convergence of loss_rate or validation_rate - and things get very predictable very early on
    //we want to use what we called community detection and centrality algorithm + A * optimization technique, we want to "detect" certain patterns by using traditional methods - like regex cosine similarity and training convergence similarity
    //we want to forge our path through the training - and achieve the "best possible" regex version
    //there is a problem of what trains first, what trains second, etc. - which is also a path problem - we'll talk about how we could generalize these problems
    //alright - after we throw every traditional method at the problem and we can't still solve it - we begin to exponentially discretize the logits - and train it on 1 billion devices - and sell the best possible version for money
    //this is called logit mining

    //thing is the traditional semantic mapping method does not have recursive build up of semantic space - and we are stuck at one semantic layer - things like cosine similarities - or jaccard similarities - but these traditional methods excel at simple training prediction
    //we want to kinda have a "centrality" to talk about the likelihood of path exploration in A* algorithm
    //and we also want to have community detection to see if the situtation is prunable - or worth digging in the direction
    //we want to store the training data points in a global semantic graph - our training optimization engine
    //we'll try to see how things turn out to be

    //but the final boss is to guess the inital logit values - and kinda learn what's the best combo to try - we also want to store this in our training optimization engine
    //we guess the initial logit values exponentially - base 1.2, without loss of generality, try to set random values and store the statistics
    //all of these things are called logit mining - and we store the data points to converge our mining mission faster, more efficient
    //it probably looks like a 1/x graph, no matter if you have 1 billion devices or 10 billion devices - but that's the direction that we are heading
    //people are offering BB $ if we get this to work correctly - so we must be very careful about the agendas and the implementations


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

    //vvv

    template <class T>
    struct Request{
        Address requestee;
        T content;
        uint8_t retry_count;
        std::unique_ptr<dg::network_exception::ExceptionHandlerInterface> exception_handler;
    };

    struct EndUserPacket{
        eu_packet_header_t serialization_header;
        dg::string content;
        Address dst;
        uint8_t retry_count;
        eu_packet_urgency_t urgency;
        eu_packet_comm_t comm;
        std::unique_ptr<dg::network_exception::ExceptionHandlerInterface> exception_handler;
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_blkr_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_blkr_rcu_addr_nothrow(event_arr[i].dst);
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
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_blkr_init_status_nothrow(ptr);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_blkr_operatable_memevent_id_nothrow(ptr);

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_mono_operatable_memevent_id_nothrow(ptr);

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant    = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(ptr);
                                auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, expected_ops_id));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pair_operatable_memevent_id_nothrow(ptr);

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t left_descendant   = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(ptr);
                                uma_ptr_t right_descendant  = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(ptr);
                                auto decay_signal_event_1   = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant, ptr, expected_ops_id));
                                auto decay_signal_event_2   = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant, ptr, expected_ops_id));

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_uacm_operatable_memevent_id_nothrow(ptr);

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> descendant_arr(UACM_ACM_SZ);
                                dg::network_tile_member_getsetter::get_uacm_descendant_nothrow(ptr, descendant_arr.get());

                                for (size_t i = 0u; i < UACM_ACM_SZ; ++i){
                                    auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant_arr[i], ptr, expected_ops_id));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_pacm_operatable_memevent_id_nothrow(ptr);

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> left_descendant_arr(PACM_ACM_SZ);
                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> right_descendant_arr(PACM_ACM_SZ);
                                dg::network_tile_member_getsetter::get_pacm_left_descendant_nothrow(ptr, left_descendant_arr.get());
                                dg::network_tile_member_getsetter::get_pacm_right_descendant_nothrow(ptr, right_descendant_arr.get());

                                for (size_t i = 0u; i < PACM_ACM_SZ; ++i){
                                    auto decay_signal_event_1 = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant_arr[i], ptr, expected_ops_id));
                                    auto decay_signal_event_2 = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant_arr[i], ptr, expected_ops_id));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_extnsrc_operatable_memevent_id_nothrow(ptr);

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant    = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(ptr);
                                auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, expected_ops_id));
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
                    InternalResolutor internal_resolutor            = {};
                    internal_resolutor.uma_ip_retriever             = this->uma_ip_retriever->get();
                    internal_resolutor.host_ip_retriever            = this->host_ip_retriever->get();
                    internal_resolutor.request_delivery_handle      = delivery_handle.get();

                    size_t trimmed_vectorization_sz                 = std::min(this->vectorization_sz, sz);
                    size_t vdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz); 
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get())); //we are risking 0s - we will fix this later

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_extndst_operatable_memevent_id_nothrow(ptr);
                        user_id_t user_id                   = dg::network_tile_member_getsetter::get_extndst_user_id_nothrow(ptr);
                        uint8_t retry_count                 = dg::network_tile_member_getsetter::get_extndst_ping_retry_count_nothrow(ptr); 

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t counterpart       = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(ptr);
                                auto request                = Request<external_virtual_memory_event_t>{};
                                request.requestee           = this->uma_ip_retriever->ip(counterpart);
                                request.requestor           = this->host_ip_retriever->ip();
                                request.content             = dg::network_external_memcommit_factory::virtualize_event(dg::network_external_memcommit_factory::make_event_forward_pingpong_request(counterpart, ptr, expected_ops_id));
                                request.retry_count         = retry_count;
                                request.exception_handler   = dg::network_exception::make_exception_handler_from_lambda([user_id, ptr](exception_t err) noexcept{
                                    if (dg::network_exception::is_failed(err)){
                                        dg::network_log::log_user_tile_error(user_id, ptr, dg::network_exception::verbose(err));
                                    }
                                });
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
                auto delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_delivery_capacity, dh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();

                    size_t trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_crit_operatable_memevent_id_nothrow(ptr);

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant    = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(ptr);
                                auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, expected_ops_id));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_msgrfwd_operatable_memevent_id_nothrow(ptr);

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant    = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(ptr);
                                auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, expected_ops_id));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_msgrbwd_operatable_memevent_id_nothrow(ptr);

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant    = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(ptr);
                                auto decay_signal_event = dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, ptr, expected_ops_id));
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
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> immu_resolutor;
            const size_t immu_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;

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
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> crit_resolutor,
                                       size_t crit_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> immu_resolutor,
                                       size_t immu_dispatch_sz,
                                       std::unique_Ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> extnsrc_resolutor,
                                       size_t extnsrc_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> extndst_resolutor,
                                       size_t extndst_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> msgrfwd_resolutor,
                                       size_t msgrfwd_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingSignalEvent>> msgrbwd_resolutor,
                                       size_t msgrbwd_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
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
                                                                             crit_resolutor(std::move(crit_resolutor)),
                                                                             crit_dispatch_sz(crit_dispatch_sz),
                                                                             immu_resolutor(std::move(immu_resolutor)),
                                                                             immu_dispatch_sz(immu_dispatch_sz),
                                                                             extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                             extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                             extndst_resolutor(std::move(extndst_resolutor)),
                                                                             extndst_dispatch_sz(extndst_dispatch_sz),
                                                                             msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                             msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                             msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                             msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

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

                    if constexpr(DEBUG_MODE_FLAG){
                        if (!tile_kind.has_value()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(tile_kind.error()));
                            std::abort();
                        }
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        uma_ptr_t requestee                     = event_arr[i].requestee;
                        uma_ptr_t requestor                     = event_arr[i].requestor;
                        operatable_id_t expected_ops_id         = event_arr[i].operatable_id;
                        init_status_t init_status               = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(requestee);
                        set_operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_leaf_operatable_memevent_id_set_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (!is_subset_id(expected_ops_id, current_ops_id)){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_immu_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_immu_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_blkr_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_blkr_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_blkr_push_observer_nothrow(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_mono_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_mono_push_observer_nothrow(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pair_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_pair_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_uacm_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_uacm_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pacm_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_pacm_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_extndst_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                   break;
                                }

                                dg::network_tile_member_getsetter::controller_extndst_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_crit_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_crit_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrfwd_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_msgrfwd_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrbwd_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_msgrbwd_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> immu_resolutor;
            const size_t immu_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;

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
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> crit_resolutor,
                                        size_t crit_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> immu_resolutor,
                                        size_t immu_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> extnsrc_resolutor,
                                        size_t extnsrc_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> extndst_resolutor,
                                        size_t extndst_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> msgrfwd_resolutor,
                                        size_t msgrfwd_dispatch_sz,
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPongRequestEvent>> msgrbwd_resolutor,
                                        size_t msgrbwd_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
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
                                                                              crit_resolutor(std::move(crit_resolutor)),
                                                                              crit_dispatch_sz(crit_dispatch_sz),
                                                                              immu_resolutor(std::move(immu_resolutor)),
                                                                              immu_dispatch_sz(immu_dispatch_sz),
                                                                              extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                              extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                              extndst_resolutor(std::move(extndst_resolutor)),
                                                                              extndst_dispatch_sz(extndst_dispatch_sz),
                                                                              msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                              msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                              msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                              msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

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

                    if constexpr(DEBUG_MODE_FLAG){
                        if (!tile_kind.has_value()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(tile_kind.error()));
                            std::abort();
                        }
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        uma_ptr_t requestee                     = event_arr[i].requestee;
                        uma_ptr_t requestor                     = event_arr[i].requestor;
                        operatable_id_t expected_ops_id         = event_arr[i].operatable_id;
                        init_status_t init_status               = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(requestee);
                        set_operatable_id_t current_ops_id      = dg::network_tile_member_getsetter::get_leaf_operatable_memevent_id_set_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (!is_subset_id(expected_ops_id, current_ops_id)){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_immu_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_immu_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_blkr_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_blkr_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_blkr_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_blkr_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, expected_ops_id)));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_mono_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_mono_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, expected_ops_id)));
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

                const size_t EVENT_SCALE_FACTOR     = dg::network_tile_metadata::PAIR_DESCENDANT_COUNT;
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pair_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_pair_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t left_descendant   = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(requestee);
                                uma_ptr_t right_descendant  = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant, requestee, expected_ops_id)));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant, requestee, expected_ops_id)));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_uacm_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_uacm_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> descendant_arr(UACM_ACM_SZ);
                                dg::network_tile_member_getsetter::get_uacm_descendant_nothrow(ptr, descendant_arr.get());

                                for (size_t i = 0u; i < UACM_ACM_SZ; ++i){
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant_arr[i], requestee, expected_ops_id)));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pacm_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_pacm_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> left_descendant_arr(PACM_ACM_SZ);
                                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> right_descendant_arr(PACM_ACM_SZ);
                                dg::network_tile_member_getsetter::get_pacm_left_descendant_nothrow(requestee, left_descendant_arr.get());
                                dg::network_tile_member_getsetter::get_pacm_right_descendant_nothrow(requestee, right_descendant_arr.get());

                                for (size_t i = 0u; i < PACM_ACM_SZ; ++i){
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant_arr[i], requestee, expected_ops_id)));
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant_arr[i], requestee, expected_ops_id)));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_crit_operatable_memevent_id_nothrow(requestee); 

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_crit_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, expected_ops_id)));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_extnsrc_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_extnsrc_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, expected_ops_id)));
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

                const size_t INTERNAL_EVENT_SCALE_FACTOR    = 1u;
                size_t max_possible_internal_event_sz       = sz * INTERNAL_EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_internal_event_sz); 
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                const size_t EXTERNAL_EVENT_SCALE_FACTOR    = 1u;
                size_t max_possible_external_event_sz       = sz * EXTERNAL_EVENT_SCALE_FACTOR;
                size_t trimmed_outbound_delivery_capacity   = std::min(this->oubound_delivery_capacity, max_possible_external_event_sz);
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_extndst_operatable_memevent_id_nothrow(requestee);
                        user_id_t user_id               = dg::network_tile_member_getsetter::get_extndst_user_id_nothrow(requestee);
                        uint8_t retry_count             = dg::network_tile_member_getsetter::get_extndst_ping_retry_count_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_extndst_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t counterpart           = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(requestee);
                                auto ping_request               = Request<external_virtual_memory_event_t>{};
                                ping_request.requestee          = this->uma_ip_retriever->ip(counterpart);
                                ping_request.requestor          = this->host_ip_retriever->ip();
                                ping_request.content            = dg::network_external_memcommit_factory::make_event_forward_pingpong_signal(counterpart, expected_ops_id); //should be pingpong
                                ping_request.retry_count        = retry_count;
                                ping_request.exception_handler  = dg::network_exception::make_exception_handler_from_lambda([user_id, requestee](exception_t err) noexcept{
                                    if (dg::network_exception::is_failed(err)){
                                        dg::network_log::log_user_tile_error(user_id, requestee, dg::network_exception::verbose(err));
                                    }
                                });

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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrfwd_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_msgrfwd_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, expected_ops_id)));
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
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(event_arr[i].requestee);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
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
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrbwd_operatable_memevent_id_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                dg::network_tile_member_getsetter::controller_msgrbwd_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

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
                                if (current_ops_id != expected_ops_id){
                                    break;
                                }

                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee, expected_ops_id)));
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
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> crit_pingpong_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> immu_pingpong_resolutor;
            const size_t immu_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> extnsrc_pingpong_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> extndst_pingpong_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> msgrfwd_pingpong_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> msgrbwd_pingpong_resolutor;
            const size_t msgrbwd_dispatch_sz;

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
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> crit_pingpong_resolutor,
                                            size_t crit_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> immu_pingpong_resolutor,
                                            size_t immu_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> extnsrc_pingpong_resolutor,
                                            size_t extnsrc_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> extndst_pingpong_resolutor,
                                            size_t extndst_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> msgrfwd_pingpong_resolutor,
                                            size_t msgrfwd_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardPingPongRequestEvent>> msgrbwd_pingpong_resolutor,
                                            size_t msgrbwd_dispatch_sz) noexcept: leaf_pingpong_resolutor(std::move(leaf_pingpong_resolutor)),
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
                                                                                  crit_pingpong_resolutor(std::move(crit_pingpong_resolutor)),
                                                                                  crit_dispatch_sz(crit_dispatch_sz),
                                                                                  immu_pingpong_resolutor(std::move(immu_pingpong_resolutor)),
                                                                                  immu_dispatch_sz(immu_dispatch_sz),
                                                                                  extnsrc_pingpong_resolutor(std::move(extnsrc_pingpong_resolutor)),
                                                                                  extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                                  extndst_pingpong_resolutor(std::move(extndst_pingpong_resolutor)),
                                                                                  extndst_dispatch_sz(extndst_dispatch_sz),
                                                                                  msgrfwd_pingpong_resolutor(std::move(msgrfwd_pingpong_resolutor)),
                                                                                  msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                                  msgrbwd_pingpong_resolutor(std::move(msgrbwd_pingpong_resolutor)),
                                                                                  msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

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

                if constexpr(DEBUG_MODE_FLAG){
                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(tile_kind.error()));
                        std::abort();
                    }
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

    //

    //we'll probably dwell on this problem for a week or two to do extreme optimizations - customers are asking petabytes/core*second of 256x256 - this is hard - but its doable
    //I take my brother advice of staying on UDP of size prolly 8K - 16K (this is also tile size) for now (he messaged me the other night) - for various reasons, mostly not flooding the system and offloading the responsibility to the kernel (which is built for this) - this requires extreme optimizations of tile dispatches to not compromise the speed
    //it's important to stay on the unit size of 8KB - because we dont have to deal with network transportation + internal management + flooding + etc - we want raw performance from kernel
    //we are transitioning to all-stack-allocation to avoid affine problems
    //alrights - we are breaking engineering practices to do extreme optimizations here - we'll try to see what the large scale centrality algorithm would look like (it'd probably look like our universe and the spaceship - yet we have more "spaceships" to unhinge + diffract the universe) - we want to keep numerical stability in control - we'll talk about this later

    //I was still trying to prove the completeness of the algorithm 
    //so let's write the proof once and for all
    //(1): the authorization fence of operatable_memevent_id + operatable_forward_id + operatable_backward_id
    //(2): assume that the reference tree is fixed, TILE_INIT_INITIALIZED denotes that the subtree is correctly computed  + intact - prove that a random transaction (a lock of the tree and its descendant) must snap the tree state in a correct state - by using induction, we can prove that after a certain number of totally random operations - the tree is correctly computed
    //                                              we want to have TILE_INIT_INITIALIZED for the pinged logits (this is the grand plan)
    //                                              prove that a regex version of every possible correct state would snap the tree to a correct state (induction) - transactional of resolution payload - correct state consists of (ORPHANED | EMPTY | DECAYED | ADOPTED, or INITIALIZED => logit_value of the subtree is correctly computed)
    //                                              prove that there exists a chance by doing totally random valid operations (snap random node -> orphaned | empty | decayed | adopted or use induction operation to induce the next valid state) - but never false positive
    //                                              prove that the flow of ping + init + decay + forward would converge the tree -> initialized eventually

    //things get very confusing if you aren't calibrating the semantic space (assume the tree is fixed)

    //(3): assume a miscomputation is performed (for whatever reason) - we must be able to prove that the program is not crash (we allow leeway for not-a-number)
    //(4): assume an orphan order (orphan can only be performed on the entire computation tree) is performed during the tree forward + backward computation - the integrity of the leaf, and the not-yet-orphaned must not be compromised if the tree is reused after all orphan orders have been confirmed (post synchronization)
    //(5): assume that no orphan order is called, only adopt order, we must be able to prove that the tree computation is not compromised (as if it's serially computed - there is no diffrerence in the uncertainty of computation)  

    //now we need to calibrate the semantic space and write the algorithm to fulfill the contracts

    //this is actually harder than the allocation tree - that was pretty tough to write also - and a lot more confusing (this took me weeks + months to finalize the designs - it's hard)
    //thing is at this level of programming - we don't actually think things holistically - rather single responsibiltity + atomicity of state snap (what the operation does and what users intend it to do are two very differrent things - linked by probably, maybe, temporally)
    //so it's actually about laying out the basic operations - find the use cases that the users could use - and we are trying to link the basic rules together to establish a contract - it's harder than the traditional approach in the sense of "what user intends the operation to do" and "what the operation actually does" are two different things
    //and I actually couldn't think of other ways for this to work efficiently more than it already does - we must abide to the law of 8K unit tile (because that's the UDP standard for years - I doubt that would be changed soon in a forseeable future)
    //we realized that the thing that slows down the parallel host system, in most scenerios, is the mutex check + the RAM bus + the memory ordering synchronization - not that correctly predicted branch or 16 reduntdant math operations - we'll try our best to work on that later

    //we'll try to aim for 1 PB linear/ host_core*s
    //the thing is that we probably don't need that crazy compute power - rather a guided training (granted, this is another radix of optimization) - think of guided training as unhinging a graph search node - where for each of the nodes, we must dispatch, WLOG, 100GB of data to gather the data and re-evaluate the situation and explore the next hinge
    //we already have our theory, everything that is built on top of a centrality algorithm and produces output based on input must be an intelligent system - this requires the uniform responsibility of centrality nodes (we can't have skewed centrality nodes - it's cheating)
    //the problem with the current training is that the numerical stability is bad - the uncertainty of training exceeds the differential progress
    //my brother took the right approach - yet I aim to generalize that centrality must be done by spaceships operation + message-passing via hardware nearest-neighbors (I dont know if differential is the way - I just know centrality must be done that way - computer scientifically speaking) 
    //imagine this with a one billion handheld devices scale - we'll be there someday fellas - we'll be rich
    //let's see where us lowlifes could get in the capitalism world - but first, we must understand the law of local transportation + frequency (we can't stress this over the internet - it's not feasible - the infrastructure is not ready - we are transporting a massive amount of data by building a voronoi digram) 
    //let's set a goal - we'll do the demo of interstellar in a month or two (for real - by using double semantic space of named entities (training) and centrality (transformer))
    //I ask yall to have faith - I'm not Dr. Brand but I'll for sure can implement what described in the movie  

    //I admit this is very difficult to write - and very difficult to comprehend - there's a saying - now there are God and I who understand the code - soon there's only God
    //yet this speaks to the core of computer science - the flood management (8K UDP offload to kernel - hardly a system would implement this) - multi-users - dynamicity of tile dispatch - routing of tile dispatch - true concurrency - frequency to allow user customized calibration + etc 
    //there are several concepts that we need to talk - first is the concept of unique_ptr and the concept of unique_nobuf_ptr - alright - std always couples the responsibility of buffer and the semantic of the buffer and the lifetime of those (we must decouple the responbiility if we want to live in the low world)

    //we lay the rules for differential neural network std - a semantic calibration layer that is the backbone of every distributed differential neural network implementation - the core
    //we thought long and hard - there isn't really another way that avoids system flood of UDP + backward synchronization overhead
    //the contract that we are heading is the serialized process of ingest leafs -> adopt -> signal -> orphan -> adopt -> signal -> orphan -> etc. (for the same computation tree) (we can run this in parallel)
    //256x256 is probably the best tile size that we can opt for - and we can push the system -> maximum cuda speed of 1PB linear dispatch/ host_core * s
    //this might be an overkill for the current centrality algorithm

    constexpr auto convert_grad_status_to_cuda_write_option(grad_status_t grad_status) noexcept -> cuda_write_option_t{

        if (grad_status == TILE_GRAD_STATUS_HAS_VALUE){
            return CUDA_TILEOPS_OPERATION_ACCUM;
        } else if (grad_status == TILE_GRAD_STATUS_EMPTY){
            return CUDA_TILEOPS_OPERATION_ASSIGN;
        } else{
            std::unreachable();
        }
    }

    constexpr auto convert_grad_status_to_host_write_option(grad_status_t grad_status) noexcept -> host_write_option_t{

        if (grad_status == TILE_GRAD_STATUS_HAS_VALUE){
            return HOST_TILEOPS_OPERATION_ACCUM;
        } else if (grad_status == TILE_GRAD_STATUS_EMPTY){
            return HOST_TILEOPS_OPERATION_ASSIGN;
        } else{
            std::unreachable();
        }
    }

    class ForwardDoLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        public:

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                (void) event_arr;
            }
    };

    //first cut optimization
    //memory fragmentation
    //affined mempool + memfetch
    //branch prediction pipeline
    //instruction cache fetch
    //vectorization + simd
    //memregion acquisition locality of reference
    //affined cache-pollution (operation vs dispatch)
    //fail-safes - we radix fail-safes as misoperations - not system crash - that's mayday) 

    //number one rule of low level optimization - whatever you think of hardware instruction, it's wrong
    //dont try to outsmart compiler - try to give compiler as many environmental arguments as possible - because compiler updates instructions - we dont
    //try to limit memory footprint
    //try to limit instruction cache fetch
    //try to not have holes in table dispatch
    //try to not use division and modulo, except for aligned division and modulo
    //try to not pollute hardware cache
    //try to stay affined - like fork
    //try to not abuse the memory ordering hard sync - it'd crash the system
    //try to do code path analysis of branches - and increase the probality of very skewed distribution - uniform distribution of branches is the worst
    //the one way we can do the very skewed branches is the polymorphic batch dispatch of void * - by using delvrsrv - or multi-containerization
    //try to use unique_nobuf_ptr<> - instead of unique_ptr<> - you would not last in the low-world
    //try to keep everything inside the component and provide a high level abstraction - how to initialize + how to use + etc. - people dont really care about the implementations - if you could try to make it readable - thats a plus
    //things are tough down here - especially in the host asynchronous world + branching world
    //we'll be back to convert device_id_t to uint64_t buffer for fast cmp 
    //thing is we aren't being greedy Mom - it's the god damn job

    //alright - after a long conversation - it's better to have extnsrcsrc + extnsrcdst + extndstsrc + extndstdst 
    //for the reasons being: (1) fixation of the computation tree (we need to pre-initialize the extnsrcdst and the extndstsrc to be able to do foreign injection of pre-assigned addresses)
    //                       (2) user-customized calibration of network packets
    //                       (3) remove the responsibility of internal on-the-fly tiles which lead to system crash
    //                       (4) precond of internal tiles (we dont worry about the device_id_t and device_platform_t now)
    //                       (5) compression of transportation (dimensional reduction -> dimensional expansion of data - unclear if the responsibility is clear here) 256 x 256 x 8 bytes -> 256 x 256 * 1 bit (yet if there is compression it should be this guy's responsibility to do compression - we might use different dispatch codes to do compression - we yet to know - we might use projection to do compression (this only works if we dont have collisions))
    //                       (6) it's just cleaner that way

    //what if we broadcast this to many endpoints? 
    //we still need a preallocated buffer to guarantee compute tree fixation for allocation-duration (this is user's contract) - so this is the approach 
    //256 x 256 is just a generalization of rules and parallelism of rules - because datatype can scale 1 bit 1 byte 2 bytes 4 bytes 8 bytes 16 bytes etc
    //we'll try our best to push the engine to the maximum possible operating speed (this requires alien optimizations of spinlock + memory_ordering + etc.)
    //there is a certain trade off for 1x1 -> 256x256 - we usually find the sweet spot between compute benefits and model convergability
    //we'll be back tomorrow to implement

    //extnsrcsrc -> extnsrcdst (injection) -> extndstdst (forward)
    //extndstdst -> extndstsrc (injection) -> extnsrcsrc  (backward)

    //clear
    class ForwardDoBlkrSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoBlkrSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                         std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                         std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                         size_t request_delivery_capacity,
                                         size_t radxfetch_vectorization_sz,
                                         size_t region_vectorization_sz,
                                         size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                    cuda_async_device(std::move(cuda_async_device)),
                                                                                    host_async_device(std::move(host_async_device)),
                                                                                    request_delivery_capacity(request_delivery_capacity),
                                                                                    radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                    region_vectorization_sz(region_vectorization_sz),
                                                                                    forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                        = 

                const size_t EVENT_SCALE_FACTOR             = dg::network_tile_metadata::MAX_OBSERVER_ARR_SZ;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_blkr_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_blkr_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = DispatchRadixArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;
                    internal_resolutor.allocator                = &arena_allocator;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->src);

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz            = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr          = dg::network_tile_member_getsetter::get_blkr_rcu_addr_nothrow(event_arr[i].dst);

                        auto resolutor_key              = ResolutorKeyArgument{};
                        resolutor_key.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        resolutor_key.dst_vd_id         = dispatch_radix_arg_arr[i]->dst_vd_id;
                        resolutor_key.src_vd_id         = dispatch_radix_arg_arr[i]->src_vd_id;

                        auto resolutor_val              = ResolutorValueArgument{};
                        resolutor_val.dst               = event_arr[i].dst;
                        resolutor_val.src               = dispatch_radix_arg_arr[i]->src;
                        resolutor_val.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key, resolutor_val);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t src;
                device_id_t src_vd_id;
                device_id_t dst_vd_id;
            };

            struct DispatchRadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, DispatchRadixFetcherArgument>{

                void push(uma_ptr_t lck_addr, DispatchRadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(lck_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_blkr_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_blkr_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    auto dispatch_radix         = DispatchRadixArgument{};
                                    auto dispatch_control       = dg::network_tile_member_getsetter::get_blkr_dispatch_control_nothrow(data_arr[i].root);
                                    dispatch_radix.src          = dg::network_tile_member_getsetter::get_blkr_descendant_nothrow(data_arr[i].root);
                                    auto dispatch_info          = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_blkr_forward_dispatch(dispatch_control));
                                    dispatch_radix.src_vd_id    = dispatch_info.src_vd_id;
                                    dispatch_radix.dst_vd_id    = dispatch_info.dst_vd_id;

                                    *data_arr[i].fetching_addr  = dispatch_radix;
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

            struct CudaResolutorArgument{
                cuda_ptr_t dst;
                cuda_ptr_t src;
                cuda_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator; //this is an engineering compromise (we break single responsibility of components - by imposing a lifetime constraint (allocator must outlive synchronizer) - not necessarily best practices - yet we'll find somewhere in the middle ground to meet the performance agendas + the readability agendas

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_vec_sz      = sz * 2;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_vec(cuda_ptr_vec_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_vec[i * 2]     = data_arr[i].dst;
                        cuda_ptr_vec[i * 2 + 1] = data_arr[i].src;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::forward_mono(e.dst, e.src, e.dispatch_control)); //kernel launches might be expensive
                        };

                        size_t async_task_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * async_task_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(async_task_bsz));
                        auto async_task         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, async_task_buf)); 

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(async_task)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_vec.get(), std::next(cuda_ptr_vec.get(), cuda_ptr_vec_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t dst;
                host_ptr_t src;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_vec_sz      = sz * 2;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_vec(host_ptr_vec_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_vec[i * 2]     = data_arr[i].dst;
                        host_ptr_vec[i * 2 + 1] = data_arr[i].src;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::forward_mono(e.dst, e.src, e.dispatch_control));
                        };

                        size_t async_task_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * async_task_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(async_task_bsz));
                        auto async_task         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, async_task_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(async_task)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_vec.get(), std::next(host_ptr_vec.get(), host_ptr_vec_sz)));
                    auto synchroniable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity));
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            //TODOs: optimizables word size memcmp + has_unique_object_representations_v
            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;
                device_id_t dst_vd_id;
                device_id_t src_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                  = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{}));
                    auto dst_vmamap_reacquirer              = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_vmamap_reacquirer              = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                  = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer         = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer));
                    auto cuda_resolutor                     = InternalCudaResolutor{};
                    cuda_resolutor.async_device             = this->cuda_async_device;
                    cuda_resolutor.synchronizer             = &cuda_synchronizer;
                    cuda_resolutor.restrict_synchronizer    = &cuda_restrict_synchronizer;
                    cuda_resolutor.allocator                = this->allocator;

                    auto host_synchronizer                  = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer         = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto host_resolutor                     = InternalHostResolutor{};
                    host_resolutor.async_device             = this->host_async_device;
                    host_resolutor.synchronizer             = &host_synchronizer;
                    host_resolutor.restrict_synchronizer    = &host_restrict_synchronizer;
                    host_resolutor.allocator                = this->allocator;

                    size_t trimmed_cuda_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&cuda_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_host_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_resolutor, trimmed_host_vectorization_sz, hdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_blkr_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_blkr_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_blkr_operatable_forward_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_blkr_init_status_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_blkr_logit_addr_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_blkr_dispatch_control_nothrow(dst);
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_blkr_observer_array_size_nothrow(dst);
                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(dg::network_tile_metadata::MAX_OBSERVER_ARR_SZ);
                        dg::network_tile_member_getsetter::get_blkr_observer_array_nothrow(dst, dst_observer_arr.get());

                        std::expected<operatable_id_t, exception_t> src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_forward_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);

                        if (!src_fwd_operatable_id.has_value() || !src_init_status.has_value() || !src_logit_umaptr.has_value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED && dst_init_status != TILE_INIT_STATUS_ADOPTED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
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

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_blkr_forward_dispatch(dispatch_control));

                        if (dispatch_info.dst_vd_id != key.dst_vd_id){
                            continue;
                        }

                        if (dispatch_info.src_vd_id != key.src_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dispatch_info.dst_vd_id}, 
                                                                                                           {src_logit_umaptr.value(), dispatch_info.src_vd_id}});

                        vma_ptr_t dst_map_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{}); //TODOs: these should be std::expected<> - we'll reconsider whether this is a precond or an exception 
                        vma_ptr_t src_map_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(dst_vmamap_reacquirer, dst_map_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_vmamap_reacquirer, src_map_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_arg               = CudaResolutorArgument{};
                            cuda_arg.dst                = dg::network_vmamap::get_cuda_ptr(dst_vmamap_reacquirer);
                            cuda_arg.src                = dg::network_vmamap::get_cuda_ptr(src_vmamap_reacquirer);
                            cuda_arg.dispatch_control   = dispatch_info.tileops_cuda_dispatch_control; 

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_arg.src, cuda_arg);
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_arg               = HostResolutorArgument{};
                            host_arg.dst                = dg::network_vmamap::get_host_ptr(dst_vmamap_reacquirer);
                            host_arg.src                = dg::network_vmamap::get_host_ptr(src_vmamap_reacquirer);
                            host_arg.dispatch_control   = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_arg.src, host_arg); 
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(dst_observer_arr[j], expected_ops_id)));
                        }

                        dg::network_tile_member_getsetter::set_blkr_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                    }
                }
            };
    };

    //clear
    class ForwardDoMonoSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoMonoSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                           std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                           size_t request_delivery_capacity,
                                           size_t radxfetch_vectorization_sz,
                                           size_t region_vectorization_sz,
                                           size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                      cuda_async_device(std::move(cuda_async_device)),
                                                                                      host_async_device(std::move(host_async_device)),
                                                                                      request_delivery_capacity(request_delivery_capacity),
                                                                                      radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                      region_vectorization_sz(region_vectorization_sz),
                                                                                      forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                        = 

                const size_t EVENT_SCALE_FACTOR             = dg::network_tile_metadata::MAX_OBSERVER_ARR_SZ;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = RadixFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.allocator                = &arena_allocator;
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->src);

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].dst);

                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        resolutor_key_arg.dst_vd_id         = dispatch_radix_arg_arr[i]->dst_vd_id;
                        resolutor_key_arg.src_vd_id         = dispatch_radix_arg_arr[i]->src_vd_id;

                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.src               = dispatch_radix_arg_arr[i]->src;
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t src;
                device_id_t src_vd_id;
                device_id_t dst_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, RadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_mono_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    auto dispatch_radix         = DispatchRadixArgument{};
                                    auto dispatch_control       = dg::network_tile_member_getsetter::get_mono_dispatch_control_nothrow(data_arr[i].root);
                                    dispatch_radix.src          = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(data_arr[i].root);
                                    auto dispatch_info          = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_mono_forward_dispatch(dispatch_control));
                                    dispatch_radix.src_vd_id    = dispatch_info.src_vd_id;
                                    dispatch_radix.dst_vd_id    = dispatch_info.dst_vd_id;

                                    *data_arr[i].fetching_addr  = dispatch_radix;
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

            struct CudaResolutorArgument{
                cuda_ptr_t dst;
                cuda_ptr_t src;
                cuda_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_vec_sz      = sz * 2;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_vec(cuda_ptr_vec_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_vec[i * 2]     = data_arr[i].dst;
                        cuda_ptr_vec[i * 2 + 1] = data_arr[i].src;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::forward_mono(e.dst, e.src, e.dispatch_control));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_vec.get(), std::next(cuda_ptr_vec.get(), cuda_ptr_vec_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t dst;
                host_ptr_t src;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_vec_sz      = sz * 2;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_vec(host_ptr_vec_sz);
                    size_t total_complexity     = {};

                    auto virtual_wo_vec_bsz     = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_vec[i * 2]     = data_arr[i].dst;
                        host_ptr_vec[i * 2 + 1] = data_arr[i].src;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::forward_mono(e.dst, e.src, e.dispatch_control));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_vec.get(), std::next(host_ptr_vec.get(), host_ptr_vec_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            //TODOs: word_size aligned cmp + has_unique_object_representations_v 
            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;
                device_id_t dst_vd_id;
                device_id_t src_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_emmory_event_t> * request_delivery_handle;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                          = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{}));
                    auto dst_vmamap_reacquirer                      = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_vmamap_reacquirer                      = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    
                    auto cuda_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer));
                    auto internal_cuda_resolutor                    = InternalCudaResolutor{};
                    internal_cuda_resolutor.async_device            = this->cuda_async_device;
                    internal_cuda_resolutor.synchronizer            = &cuda_synchronizer;
                    internal_cuda_resolutor.restrict_synchronizer   = &cuda_restrict_synchronizer;
                    internal_cuda_resolutor.allocator               = this->allocator;

                    auto host_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto internal_host_resolutor                    = InternalHostResolutor{};
                    internal_host_resolutor.async_device            = this->host_async_device;
                    internal_host_resolutor.synchronizer            = &host_synchronizer;
                    internal_host_resolutor.restrict_synchronizer   = &host_restrict_synchronizer;
                    internal_host_resolutor.allocator               = this->allocator;

                    size_t cdh_trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&internal_cuda_resolutor, cdh_trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&internal_cuda_resolutor, cdh_trimmed_vectorization_sz, cdh_mem.get()));

                    size_t hdh_trimmed_vectorization_sz             = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&internal_host_resolutor, hdh_trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&internal_host_resolutor, hdh_trimmed_vectorization_sz, hdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_mono_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_mono_operatable_forward_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_mono_dispatch_control_nothrow(dst); 
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_mono_logit_addr_nothrow(dst); 
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_mono_observer_array_size_nothrow(dst);

                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(MAX_OBSERVER_ARR_SZ);
                        dg::network_tile_member_getsetter::get_mono_observer_array_nothrow(dst, dst_observer_arr.get());

                        std::expected<operatable_id_t, exception_t> src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_forward_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);

                        if (!src_fwd_operatable_id.has_value() || !src_init_status.has_value() || !src_logit_umaptr.has_value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_ADOPTED && dst_init_status != TILE_INIT_STATUS_DECAYED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
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

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_mono_forward_dispatch(dispatch_control));

                        if (dispatch_info.dst_vd_id != key.dst_vd_id){
                            continue;
                        }

                        if (dispatch_info.src_vd_id != key.src_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dispatch_info.dst_vd_id}, 
                                                                                                           {src_logit_umaptr.value(), dispatch_info.src_vd_id}});

                        auto dst_map_vmaptr = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        auto src_map_vmaptr = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(dst_vmamap_reacquirer, dst_map_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_vmamap_reacquirer, src_map_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg             = CudaResolutorArgument{};
                            cuda_resolutor_arg.dst              = dg::network_vmamap::get_cuda_ptr(dst_vmamap_reacquirer);
                            cuda_resolutor_arg.src              = dg::network_vmamap::get_cuda_ptr(src_vmamap_reacquirer);
                            cuda_resolutor_arg.dispatch_control = dispatch_info.tileops_cuda_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_resolutor_arg.src, cuda_resolutor_arg);
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.dst              = dg::network_vmamap::get_host_ptr(dst_vmamap_reacquirer);
                            host_resolutor_arg.src              = dg::network_vmamap::get_host_ptr(src_vmamap_reacquirer);
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_resolutor_arg.src, host_resolutor_arg);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(dst_observer_arr[j], expected_ops_id)));
                        }

                        dg::network_tile_member_getsetter::set_mono_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                    }
                }
            };
    };

    //clear
    class ForwardDoPairSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoPairSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                           std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                           size_t request_delivery_capacity,
                                           size_t radxfetch_vectorization_sz,
                                           size_t region_vectorization_sz,
                                           size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                      cuda_async_device(std::move(cuda_async_device)),
                                                                                      host_async_device(std::move(host_async_device)),
                                                                                      request_delivery_capacity(request_delivery_capacity),
                                                                                      radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                      region_vectorization_sz(region_vectorization_sz),
                                                                                      forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                        =; 

                const size_t EVENT_SCALE_FACTOR             = dg::network_tile_metadata::MAX_OBSERVER_ARR_SZ;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz); 
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = RadixFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;
                    internal_resolutor.allocator                = &arena_allocator;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost); 
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> lhs_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->lhs); //we are doing polymorphic access - it's better to safeguards the assumption here 
                        std::expected<uma_ptr_t, exception_t> rhs_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->rhs);

                        if (!lhs_rcu_addr.has_value() || !rhs_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(event_arr[i].dst); //assume that descendant_arr[i].has_value() => safe pair access

                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.lhs_lck_addr      = dg::memult::region(lhs_rcu_addr.value(), lck_region_sz);
                        resolutor_key_arg.rhs_lck_addr      = dg::memult::region(rhs_rcu_addr.value(), lck_region_sz);
                        resolutor_key_arg.dst_vd_id         = dispatch_radix_arg_arr[i]->dst_vd_id;
                        resolutor_key_arg.lhs_vd_id         = dispatch_radix_arg_arr[i]->lhs_vd_id;
                        resolutor_key_arg.rhs_vd_id         = dispatch_radix_arg_arr[i]->rhs_vd_id;

                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.lhs               = dispatch_radix_arg_arr[i]->lhs;
                        resolutor_val_arg.rhs               = dispatch_radix_arg_arr[i]->rhs;
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t lhs;
                uma_ptr_t rhs;
                device_id_t dst_vd_id;
                device_id_t lhs_vd_id;
                device_id_t rhs_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, RadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pair_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != data_arr[i].expected_ops_id){
                                    break;
                                }

                                pong_count_t pong_count = dg::network_tile_member_getsetter::get_pair_pong_count_nothrow(data_arr[i].root);
                                pong_count += 1u; //has to be unsigned otherwise we risk signed wraparound
                                dg::network_tile_member_getsetter::set_pair_pong_count_nothrow(data_arr[i].root, pong_count);

                                if (pong_count >= dg::network_tile_metadata::PAIR_DESCENDANT_COUNT){ //TODOs: optimizables
                                    auto dispatch_radix         = DispatchRadixArgument{};
                                    auto dispatch_control       = dg::network_tile_member_getsetter::get_pair_dispatch_control_nothrow(data_arr[i].root);
                                    dispatch_radix.lhs          = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(data_arr[i].root);
                                    dispatch_radix.rhs          = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(data_arr[i].root);
                                    auto dispatch_info          = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_forward_pair_dispatch(dispatch_control));
                                    dispatch_radix.dst_vd_id    = dispatch_info.dst_vd_id;
                                    dispatch_radix.lhs_vd_id    = dispatch_info.lhs_vd_id;
                                    dispatch_radix.rhs_vd_id    = dispatch_info.rhs_vd_id;

                                    *data_arr[i].fetching_addr  = dispatch_radix;
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

            struct CudaResolutorArgument{
                cuda_ptr_t dst;
                cuda_ptr_t lhs;
                cuda_ptr_t rhs;
                cuda_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_vec_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_vec(cuda_ptr_vec_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz)); 
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_vec[i * 3]     = data_arr[i].dst;
                        cuda_ptr_vec[i * 3 + 1] = data_arr[i].lhs;
                        cuda_ptr_vec[i * 3 + 2] = data_arr[i].rhs;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::decode_pair_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::forward_pair(e.dst, e.lhs, e.rhs, e.dispatch_control));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_vec.get(), std::next(cuda_ptr_vec.get(), cuda_ptr_vec_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t dst;
                host_ptr_t lhs;
                host_ptr_t rhs;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_vec_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_vec(host_ptr_vec_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_vec[i * 3]     = data_arr[i].dst;
                        host_ptr_vec[i * 3 + 1] = data_arr[i].lhs;
                        host_ptr_vec[i * 3 + 2] = data_arr[i].rhs;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_pair_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::forward_pair(e.dst, e.lhs, e.rhs, e.dispatch_control));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_vec.get(), std::next(host_ptr_vec.get(), host_ptr_vec_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            //TODOs: word_size memcmp + has_unique_object_representations_v
            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t lhs_lck_addr;
                uma_ptr_t rhs_lck_addr;
                device_id_t dst_vd_id;
                device_id_t lhs_vd_id;
                device_id_t rhs_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, lhs_lck_addr, rhs_lck_addr, dst_vd_id, lhs_vd_id, rhs_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, lhs_lck_addr, rhs_lck_addr, dst_vd_id, lhs_vd_id, rhs_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t lhs;
                uma_ptr_t rhs;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.lhs_lck_addr, key.rhs_lck_addr);

                    auto umamap_reacquirer                              = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{}));
                    auto dst_vmamap_reacquirer                          = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto lhs_vmamap_reacquirer                          = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto rhs_vmamap_reacquirer                          = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                              = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer                     = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer));
                    auto internal_cuda_resolutor                        = InternalCudaResolutor{};
                    internal_cuda_resolutor.async_device                = this->cuda_async_device;
                    internal_cuda_resolutor.synchronizer                = &cuda_synchronizer;
                    internal_cuda_resolutor.restrict_synchronizer       = &cuda_restrict_synchronizer;
                    internal_cuda_resolutor.allocator                   = this->allocator;

                    auto host_synchronizer                              = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer                     = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto internal_host_resolutor                        = InternalHostResolutor{};
                    internal_host_resolutor.async_device                = this->host_async_device;
                    internal_host_resolutor.synchronizer                = &host_synchronizer;
                    internal_host_resolutor.restrict_synchronizer       = &host_restrict_synchronizer;
                    internal_host_resolutor.allocator                   = this->allocator;

                    size_t trimmed_cuda_vectorization_sz                = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost                          = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&internal_cuda_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle                           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&internal_cuda_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_cuda_mixed_vectorization_sz          = std::min(this->vectorization_sz, sz);
                    size_t cmdh_allocation_cost                         = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&internal_cuda_resolutor, trimmed_cuda_mixed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cmdh_mem(cmdh_allocation_cost);
                    auto cuda_mixed_delivery_handle                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&internal_cuda_resolutor, trimmed_cuda_mixed_vectorization_sz, cmdh_mem.get())); 

                    size_t trimmed_host_vectorization_sz                = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost                          = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&internal_host_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle                           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&internal_host_resolutor, trimmed_host_vectorization_sz, hdh_mem.get())); 

                    size_t trimmed_host_mixed_vectorization_sz          = std::min(this->vectorization_sz, sz);
                    size_t hmdh_allocation_cost                         = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&internal_host_resolutor, trimmed_host_mixed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hmdh_mem(hmdh_allocation_cost);
                    auto host_mixed_delivery_handle                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&internal_host_resolutor, trimmed_host_mixed_vectorization_sz, hmdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, lhs, rhs, expected_ops_id]   = std::make_tuple(data_arr[i].dst, data_arr[i].lhs, data_arr[i].rhs, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_lhs                       = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(dst);
                        uma_ptr_t dst_rhs                       = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_pair_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_pair_operatable_forward_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_pair_logit_addr_nothrow(dst);
                        dispatch_major_t dst_dispatch_major     = dg::network_tile_member_getsetter::get_pair_dispatch_major_nothrow(dst);
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_pair_observer_array_size_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_pair_dispatch_control_nothrow(dst);
                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(dg::network_tile_metadata::MAX_OBSERVER_ARR_SZ);
                        dg::network_tile_member_getsetter::get_pair_observer_array_nothrow(dst, dst_observer_arr.get());

                        std::expected<operatable_id_t, exception_t> lhs_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_forward_id(lhs);
                        std::expected<uma_ptr_t, exception_t> lhs_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(lhs);
                        std::expected<init_status_t, exception_t> lhs_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(lhs);

                        std::expected<operatable_id_t, exception_t> rhs_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_forward_id(rhs);
                        std::expected<uma_ptr_t, exception_t> rhs_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(rhs);
                        std::expected<init_status_t, exception_t> rhs_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(rhs);

                        if (!lhs_fwd_operatable_id.has_value() || !lhs_logit_umaptr.has_value() || !lhs_init_status.has_value() 
                            || !rhs_fwd_operatable_id.has_value() || !rhs_logit_umaptr.has_value() || !rhs_init_status.has_value()){

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

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_pair_forward_dispatch(dispatch_control));

                        if (dispatch_info.dst_vd_id != key.dst_vd_id){
                            continue;
                        }

                        if (dispatch_info.lhs_vd_id != key.lhs_vd_id){
                            continue;
                        }

                        if (dispatch_info.rhs_vd_id != key.rhs_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dispatch_info.dst_vd_id}, 
                                                                                                           {lhs_logit_umaptr.value(), dispatch_info.lhs_vd_id}, 
                                                                                                           {rhs_logit_umaptr.value(), dispatch_info.rhs_vd_id}});

                        vma_ptr_t dst_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t lhs_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t rhs_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(dst_vmamap_reacquirer, dst_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(lhs_vmamap_reacquirer, lhs_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(rhs_vmamap_reacquirer, rhs_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg             = CudaResolutorArgument{};
                            cuda_resolutor_arg.dst              = dg::network_vmamap::get_cuda_ptr(dst_vmamap_reacquirer);
                            cuda_resolutor_arg.lhs              = dg::network_vmamao::get_cuda_ptr(lhs_vmamap_reacquirer);
                            cuda_resolutor_arg.rhs              = dg::network_vmamap::get_cuda_ptr(rhs_vmamap_reacquirer);
                            cuda_resolutor_arg.dispatch_control = dispatch_info.tileops_cuda_dispatch_control;

                            //compiler's hint
                            cuda_ptr_t cuda_dispatch_buf[3];
                            cuda_dispatch_buf[0]                = cuda_resolutor_arg.rhs;
                            cuda_dispatch_buf[1]                = cuda_resolutor_arg.lhs;
                            cuda_dispatch_buf[2]                = dg::pointer_limits<cuda_ptr_t>::null_value();

                            if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_RIGHT){
                                dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_dispatch_buf[0], cuda_resolutor_arg);
                            } else if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_LEFT){
                                dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_dispatch_buf[1], cuda_resolutor_arg);
                            } else if (dst_dispatch_major = PAIR_DISPATCH_MAJOR_MIXED){
                                dg::network_producer_consumer::delvrsrv_deliver(cuda_mixed_delivery_handle.get(), cuda_dispatch_buf[2], cuda_resolutor_arg);
                            } else{
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.dst              = dg::network_vmamap::get_host_ptr(dst_vmamap_reacquirer);
                            host_resolutor_arg.lhs              = dg::network_vmamap::get_host_ptr(lhs_vmamap_reacquirer);
                            host_resolutor_arg.rhs              = dg::network_vmamap::get_host_ptr(rhs_vmamap_reacquirer);
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            //compiler's hint
                            host_ptr_t host_dispatch_buf[3];
                            host_dispatch_buf[0]                = host_resolutor_arg.rhs;
                            host_dispatch_buf[1]                = host_resolutor_arg.lhs;
                            host_dispatch_buf[2]                = dg::pointer_limits<host_ptr_t>::null_value();

                            if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_RIGHT){
                                dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_dispatch_buf[0], host_resolutor_arg);
                            } else if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_LEFT){
                                dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_dispatch_buf[1], host_resolutor_arg);
                            } else if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_MIXED){
                                dg::network_producer_consumer::delvrsrv_deliver(host_mixed_delivery_handle.get(), host_dispatch_buf[2], host_resolutor_arg);
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
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(dst_observer_arr[j], expected_ops_id)));
                        }

                        dg::network_tile_member_getsetter::set_pair_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                    }
                }
            };
    };

    class ForwardDoUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                         std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                         std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                         size_t request_delivery_capacity,
                                         size_t radxfetch_vectorization_sz,
                                         size_t region_vectorization_sz,
                                         size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                    host_async_device(std::move(host_async_device)),
                                                                                    cuda_async_device(std::move(cuda_async_device)),
                                                                                    request_delivery_capacity(request_delivery_capacity),
                                                                                    radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                    region_vectorization_sz(region_vectorization_sz),
                                                                                    forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                constexpr size_t LCK_ADDR_SZ_PER_DISPATCH   = UACM_ACM_SZ + 1u;
                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> descendant_arr(sz * UACM_ACM_SZ);
                dg::network_stack_allocation::NoExceptAllocation<bool[]> validation_arr(sz);
                std::fill(validation_arr.get(), std::next(validation_arr.get(), sz), false);
                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> lck_addr_arr(sz * LCK_ADDR_SZ_PER_DISPATCH);

                const size_t EVENT_SCALE_FACTOR             = MAX_OBSERVER_ARR_SZ;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    InternalDescendantAddressFetcher fetcher    = {};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        uma_ptr_t * radxfetch_addr  = std::next(descendant_arr.get(), i * UACM_ACM_SZ);
                        bool * validation_addr      = std::next(validation_arr.get(), i);  

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(event_arr[i].dst, event_arr[i].operatable_id, radxfetch_addr, validation_addr));
                    }
                }

                {
                    auto internal_resolutor                         = InternalResolutor{};
                    internal_resolutor.request_delivery_handle      = request_delivery_handle.get();
                    internal_resolutor.host_async_device            = this->host_async_device.get();
                    internal_resolutor.cuda_async_device            = this->cuda_async_device.get();
                    internal_resolutor.vectorization_sz             = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz          = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!validation_arr[i]){
                            continue;
                        }

                        size_t lck_region_sz            = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t * cur_descendant_arr  = std::next(descendant_arr.get(), i * UACM_ACM_SZ)
                        uma_ptr_t * cur_lck_addr        = std::next(lck_addr_arr.get(), i * LCK_ADDR_SZ_PER_DISPATCH);
                        bool rcu_addr_flag              = true;
                        cur_lck_addr[0u]                = dg::memult::region(dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(event_arr[i].dst), lck_region_sz);

                        for (size_t j = 0u; j < UACM_ACM_SZ; ++j){
                            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(cur_descendant_arr[j]);

                            if (!rcu_addr.has_value()){
                                rcu_addr_flag = false;
                                break;
                            }

                            cur_lck_addr[j + 1] = dg::memult::region(rcu_addr.value(), lck_region_sz);
                        }

                        if (!rcu_addr_flag){
                            continue;
                        }

                        auto key = dg::vector_view<uma_ptr_t, LCK_ADDR_SZ_PER_DISPATCH>(cur_lck_addr);
                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(event_arr[i].dst, cur_descendant_arr, event_arr[i].operatable_id));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, operatable_id_t, uma_ptr_t *, bool *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, operatable_id_t, uma_ptr_t *, bool *> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, expected_ops_id, fetching_addr, fetching_status] = ptr_arr[i];
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(dst);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_uacm_operatable_memevent_id_nothrow(dst);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id == expected_ops_id){
                                    *fetching_status = true;
                                    dg::network_tile_member_getsetter::get_uacm_descendant_nothrow(dst, fetching_addr);
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

            // struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<dg::set<uma_ptr_t>, std::tuple<uma_ptr_t, dg::vector<uma_ptr_t>>>{

            //     void push(dg::set<uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, dg::vector<uma_ptr_t>> * data_arr, size_t sz) noexcept{

            //         dg::network_memops_uma::memlock_guard mem_grd(lck_addr);

            //         auto umamap_reacquirer  = dg::network_uma::reacquirer_adaptive_raii_initialize();
            //         auto vmamap_reacquirer  = dg::network_vmamap::reacquirer_adaptive_raii_initialize();

            //         dg::network_cuda_controller::Synchronizer synchronizer{};
            //         dg::network_controller::RestrictPointerSynchronizer restrict_synchronizer(synchronizer);

            //         for (size_t i = 0u; i < sz; ++i){
            //             auto [dst, descendant_vec]  = data_arr[i];

            //         }
            //     }
            // };
    };

    class ForwardDoPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                         std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                         std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                         size_t request_delivery_capacity,
                                         size_t radxfetch_vectorization_sz,
                                         size_t region_vectorization_sz
                                         size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                    host_async_device(std::move(host_async_device)),
                                                                                    cuda_async_device(std::move(cuda_async_device)),
                                                                                    request_delivery_capacity(request_delivery_capacity),
                                                                                    radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                    region_vectorization_sz(region_vectorization_sz),
                                                                                    forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{
                
                const size_t LCK_REGION_SZ_PER_DISPATCH     = PACM_ACM_SZ + PACM_ACM_SZ + 1u; 

                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> lck_addr_arr(sz * LCK_REGION_SZ_PER_DISPATCH);
                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> left_descendant_arr(sz * PACM_ACM_SZ);
                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> right_descendant_arr(sz * PACM_ACM_SZ);
                dg::network_stack_allocation::NoExceptAllocation<bool[]> descendant_validation_arr(sz);

                const size_t EVENT_SCALE_FACTOR             = MAX_OBSERVER_ARR_SZ;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    InternalDescendantAddressFetcher fetcher    = {};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        std::expected<uma_ptr_t, exception_t> ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(event_arr[i].dst);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr                      = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr                      = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto radxfetch_arg                      = AddressFetchArgument{}; 
                        radxfetch_arg.root                      = event_arr[i].dst;
                        radxfetch_arg.expected_ops_id           = event_arr[i].operatable_id;
                        radxfetch_arg.fetching_lhs_addr         = std::next(left_descendant_arr.get(), i * PACM_ACM_SZ);
                        radxfetch_arg.fetching_rhs_addr         = std::next(right_descendant_arr.get(), i * PACM_ACM_SZ); 
                        radxfetch_arg.fetching_validation_flag  = std::next(descendant_validation_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, radxfetch_arg);
                    }
                }

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){

                    } 
                }
            }
        
        private:

            struct AddressFetchArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                uma_ptr_t * fetching_lhs_addr;
                uma_ptr_t * fetching_rhs_addr;
                bool * fetching_validation_flag;
            };

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, AddressFetchArgument>{

                void push(uma_ptr_t rcu_addr, AddressFetchArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pacm_operatable_memevent_id_nothrow(data_arr[i].root);
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                data_arr[i].fetching_validation_flag = false;
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id != data_arr[i].expected_ops_id){
                                    *data_arr[i].fetching_validation_flag = false;
                                } else{
                                    *data_arr[i].fetching_validation_flag = true;
                                    dg::network_tile_member_getsetter::get_pacm_left_descendant_nothrow(data_arr[i].root, data_arr[i].fetching_lhs_addr);
                                    dg::network_tile_member_getsetter::get_pacm_right_descendant_nothrow(data_arr[i].root, data_arr[i].fetching_rhs_addr);
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
            }
    };

    //optimizables - not clear
    class ForwardDoExtnSrcSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoExtnSrcSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                            std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                            std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                            std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                            size_t request_delivery_capacity,
                                            size_t radxfetch_vectorization_sz,
                                            size_t region_vectorization_sz,
                                            size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                       uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                       host_ip_retriever(std::move(host_ip_retriever)),
                                                                                       host_async_device(std::move(host_async_device)),
                                                                                       request_delivery_capacity(request_delivery_capacity),
                                                                                       radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                       region_vectorization_sz(region_vectorization_sz),
                                                                                       forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                        =  

                const size_t EVENT_SCALE_FACTOR             = 1u;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz); 
                size_t dh_allocation_cost                   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, dh_mem.get()));

                {
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = RadixFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.uma_ip_retriever         = this->uma_ip_retriever.get();
                    internal_resolutor.host_ip_retriever        = this->host_ip_retriever.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.allocator                = &arena_allocator;
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->src);

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(event_arr[i].dst);

                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        resolutor_key_arg.dst_vd_id         = dispatch_radix_arg_arr[i]->dst_vd_id;
                        resolutor_key_arg.src_vd_id         = dispatch_radix_arg_arr[i]->src_vd_id;

                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.src               = dispatch_radix_arg_arr[i]->src;
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t src;
                device_id_t dst_vd_id;
                device_id_t src_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;   
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, RadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_extnsrc_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    auto dispatch_radix         = DispatchRadixArgument{};
                                    auto dispatch_control       = dg::network_tile_member_getsetter::get_extnsrc_dispatch_control_nothrow(data_arr[i].root);
                                    dispatch_radix.src          = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(data_arr[i].root);
                                    auto dispatch_info          = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_extnsrc_forward_dispatch(dispatch_control));
                                    dispatch_radix.dst_vd_id    = dispatch_info.dst_vd_id;
                                    dispatch_radix.src_vd_id    = dispatch_info.src_vd_id;

                                    *data_arr[i].fetching_addr  = dispatch_radix;
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

            struct HostResolutorArgument{
                host_ptr_t dst;
                host_ptr_t src;
                host_ptr_t cpy_dst;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_vec_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> host_ptr_vec(host_ptr_vec_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_vec[i * 3]     = data_arr[i].dst;
                        host_ptr_vec[i * 3 + 1] = data_arr[i].src;
                        host_ptr_vec[i * 3 + 2] = data_arr[i].cpy_dst;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(MEMCPY)).runtime_complexity;

                        auto work_order         = [e = data_arr[i]]() noexcept{
                            size_t cpy_sz = dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(e.dispatch_control)).dst_byte_size;
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::forward_mono(e.dst, e.src, e.dispatch_control));
                            dg::network_exception_handler::nothrow_log(dg::network_memops_clib::memcpy_host_to_host(e.cpy_dst, e.dst, cpy_sz));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_vec.get(), std::next(host_ptr_vec.get(), host_ptr_vec_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct OutBoundData{
                ExtnSrcMetadataTile extnsrc_metadata_tile;
                uma_ptr_t addr;
                dg::string logit_buf;
            };

            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;
                device_id_t dst_vd_id;
                device_id_t src_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * request_delivery_handle;
                UnifiedMmeoryIPRetrieverInterface * uma_ip_retriever;
                HostIPRetrieverInterface * host_ip_retriever;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                          = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{}));
                    auto dst_logit_vmamap_reacquirer                = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_logit_vmamap_reacquirer                = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto synchronizer                               = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto restrict_synchronizer                      = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&synchronizer));
                    auto host_internal_resolutor                    = InternalHostResolutor{};
                    host_internal_resolutor.async_device            = this->host_async_device;
                    host_internal_resolutor.synchronizer            = &synchronizer;
                    host_internal_resolutor.restrict_synchronizer   = &restrict_synchronizer;
                    host_internal_resolutor.allocator               = this->allocator;

                    size_t trimmed_vectorization_sz                 = std::min(this->vectorization_sz, sz);
                    size_t hvdh_allocation_cost                     = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_internal_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hvdh_mem(hvdh_allocation_cost);
                    auto host_vectorizer_delivery_handle            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_internal_resolutor, trimmed_vectorization_sz, hvdh_mem.get())); 
                    auto outbound_data_vec                          = dg::vector<std::optional<OutBoundData>>(sz, std::optional<OutBoundData>(std::nullopt));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_extnsrc_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_extnsrc_operatable_forward_id_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_extnsrc_dispatch_control_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_extnsrc_logit_addr_nothrow(dst);
                        uma_ptr_t dst_counterpart               = dg::network_tile_member_getsetter::get_extnsrc_counterpart_nothrow(dst);
                        size_t dst_logit_bsz                    = dg::network_tile_member_getsetter::get_extnsrc_logit_byte_size_nothrow(dst);

                        std::expected<operatable_id_t, exception_t> src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_forward_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src); 

                        if (!src_fwd_operatable_id.has_value() || !src_init_status.has_value() || !src_logit_umaptr.has_value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED && dst_init_status != TILE_INIT_STATUS_ADOPTED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
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

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_extnsrc_forward_control(dispatch_control));

                        if (dispatch_info.dst_vd_id != key.dst_vd_id){
                            continue;
                        }

                        if (dispatch_info.src_vd_id != key.src_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dispatch_info.dst_vd_id}, 
                                                                                                           {src_logit_umaptr.value(), dispatch_info.src_vd_id}});

                        vma_ptr_t dst_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if(dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto rq                             = OutBoundData{};
                            dg::network_tile_member_getsetter::burn_extnsrc_metadata_nothrow(dst, rq.extnsrc_metadata_tile);
                            rq.addr                             = dst;
                            rq.logit_buf                        = dg::string(dst_logit_bsz, ' '); //TODOs: optimizables - alright - its hard to optimize this yet - I'll be back
                            host_ptr_t rq_buf_data              = rq.logit_buf.data();
                            outbound_data_vec[i]                = std::move(rq);

                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.dst              = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            host_resolutor_arg.src              = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            host_resolutor_arg.cpy_dst          = rq_buf_data
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_vectorizer_delivery_handle.get(), host_resolutor_arg.src, host_resolutor_arg);
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

                        user_id_t dst_user_id                           = outbound_data_vec[i]->extnsrc_metadata_tile.user_id;
                        uint8_t retry_count                             = outbound_data_vec[i]->extnsrc_metadata_tile.retry_count;
                        uma_ptr_t self_addr                             = outbound_data_vec[i]->addr;
                        ExtnSrcTile extnsrc_tile                        = dg::network_tile_member_getsetter::make_tile_from_metadata(outbound_data_vec[i]->extnsrc_metadata_tile);
                        extnsrc_tile.logit                              = std::move(outbound_data_vec[i]->logit_buf);
                        dg::string serialized                           = dg::network_compact_serializer::serialize<dg::string>(std::move(extnsrc_tile)); //TODOs: optimizables
                        external_virtual_memory_event_t inject_event    = dg::network_external_memcommit_factory::make_event_shadow_injection(outbound_data_vec[i]->extnsrc_metadata_tile.shadow_addr, TILE_KIND_EXTNSRCDST, std::move(serialized));
                        external_virtual_memory_event_t notify_event    = dg::network_external_memcommit_factory::make_event_forward_do_signal(outbound_data_vec[i]->extnsrc_metadata_tile.counterpart);
                        external_virtual_memory_event_t event           = dg::network_external_memcommit_factory::make_event_sequential(std::move(inject_event), std::move(notify_event));

                        auto rq                                         = Request<external_virtual_memory_event_t>{};
                        rq.requestee                                    = this->uma_ip_retriever->ip(outbound_data_vec[i]->extnsrc_metadata_tile.counterpart);
                        rq.requestor                                    = this->uma_ip_retriever->ip();
                        rq.content                                      = std::move(event);
                        rq.retry_count                                  = retry_count;
                        rq.exception_handler                            = dg::network_exception::make_exception_handler_from_lambda([dst_user_id, self_addr](exception_t err) noexcept{
                            if (dg::network_exception::is_failed(err)){
                                dg::network_log::log_user_tile_error(dst_user_id, self_addr, dg::network_exception::verbose(err));
                            }
                        });

                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(rq));
                    }
                }
            };
    };

    //clear
    class ForwardDoExtnSrxSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        public:

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                (void) event_arr;
            }
    };

    //clear
    class ForwardDoExtnDstSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoExtnDstSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                            std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                            size_t request_delivery_capacity,
                                            size_t radxfetch_vectorization_sz,
                                            size_t region_vectorization_sz,
                                            size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                       host_async_device(std::move(host_async_device)),
                                                                                       cuda_async_device(std::move(cuda_async_device)),
                                                                                       request_delivery_capacity(request_delivery_capacity),
                                                                                       radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                       region_vectorization_sz(region_vectorization_sz),
                                                                                       forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz); 

                // auto arena_allocator                        =  

                const size_t EVENT_SCALE_FACTOR             = dg::network_tile_metadata::MAX_OBSERVER_ARR_SZ;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost); 
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost); 
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memregion_size());
                        auto fetch_arg              = RadixFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i); 

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.allocator                = &arena_allocator;
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz); 
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->src); 

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size(), static_cast<size_t>(dg::network_uma::memregion_size())));
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(event_arr[i].dst);

                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        resolutor_key_arg.dst_vd_id         = dispatch_radix_arg_arr[i]->dst_vd_id;
                        resolutor_key_arg.src_vd_id         = dispatch_radix_arg_arr[i]->src_vd_id;

                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.src               = dispatch_radix_arg_arr[i]->src;
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t src;
                device_id_t dst_vd_id;
                device_id_t src_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::ConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, RadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_extndst_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    auto dispatch_radix         = DispatchRadixArgument{};
                                    auto dispatch_control       = dg::network_tile_member_getsetter::get_extndst_dispatch_control_nothrow(data_arr[i].root);
                                    dispatch_radix.src          = dg::network_tile_member_getsetter::get_extndst_forward_shadow_nothrow(data_arr[i].root);
                                    auto dispatch_info          = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_extndst_forward_dispatch(dispatch_control));
                                    dispatch_radix.dst_vd_id    = dispatch_info.dst_vd_id;
                                    dispatch_radix.src_vd_id    = dispatch_info.src_vd_id;

                                    *data_arr[i].fetching_addr  = dispatch_radix;
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

            struct CudaResolutorArgument{
                cuda_ptr_t dst;
                cuda_ptr_t src;
                cuda_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_arr_sz      = sz * 2;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_arr(cuda_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_arr[i * 2]     = data_arr[i].dst;
                        cuda_ptr_arr[i * 2 + 1] = data_arr[i].src;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::forward_mono(e.dst, e.src, e.dispatch_control));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_arr.get(), std::next(cuda_ptr_arr.get(), cuda_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t dst;
                host_ptr_t src;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_arr_sz      = sz * 2;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_arr(host_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_arr[i * 2]     = data_arr[i].dst;
                        host_ptr_arr[i * 2 + 1] = data_arr[i].src;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::forward_mono(e.dst, e.src, e.dispatch_control));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_arr.get(), std::next(host_ptr_arr.get(), host_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            //word_size memcmp + has_unique_object_representations_v 
            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;
                device_id_t dst_vd_id;
                device_id_t src_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::ConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz; 

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                  = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{}));
                    auto dst_logit_vmamap_reacquirer        = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize()); 
                    auto src_logit_vmamap_reacquirer        = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                  = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer         = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer)); 
                    auto cuda_resolutor                     = InternalCudaResolutor{};
                    cuda_resolutor.async_device             = this->cuda_async_device;
                    cuda_resolutor.synchronizer             = &cuda_synchronizer;
                    cuda_resolutor.restrict_synchronizer    = &cuda_restrict_synchronizer;
                    cuda_resolutor.allocator                = this->allocator;

                    auto host_synchronizer                  = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer         = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));                     
                    auto host_resolutor                     = InternalHostResolutor{};
                    host_resolutor.async_device             = this->host_async_device;
                    host_resolutor.synchronizer             = &host_synchronizer;
                    host_resolutor.restrict_synchronizer    = &restrict_synchronizer;
                    host_resolutor.allocator                = this->allocator;

                    size_t trimmed_cuda_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&cuda_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_host_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_resolutor, trimmed_host_vectorization_sz, hdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_extndst_forward_shadow_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_extndst_operatable_memevent_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_extndst_operatable_forward_id_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_extndst_logit_addr_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_extndst_dispatch_control_nothrow(dst);
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_extndst_observer_array_size_nothrow(dst);
                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(dg::network_tile_metadata::MAX_OBSERVER_ARR_SZ);
                        dg::network_tile_member_getsetter::get_extndst_observer_array_nothrow(dst, dst_observer_arr.get());

                        std::expected<uma_ptr_t, exception_t> extnsrx_ptr_access = dg::network_tile_member_access::safecthrow_extnsrx_ptr_access(src);

                        if (!extnsrx_ptr_access.has_value()){
                            continue;
                        } 

                        operatable_id_t src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_extnsrx_operatable_forward_id_nothrow(src);
                        init_status_t src_init_status           = dg::network_tile_member_getsetter::get_extnsrx_init_status_nothrow(src);
                        uma_ptr_t src_logit_umaptr              = dg::network_tile_member_getsetter::get_extnsrx_logit_addr_nothrow(src);

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED && dst_init_status != TILE_INIT_STATUS_ADOPTED){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_fwd_operatable_id != src_fwd_operatable_id){
                            continue;
                        }

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_extndst_forward_dispatch(dispatch_control));

                        if (dispatch_info.dst_vd_id != key.dst_vd_id){
                            continue;
                        }

                        if (dispatch_info.src_vd_id != key.src_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dispatch_info.dst_vd_id}, 
                                                                                                           {src_logit_umaptr, dispatch_info.src_vd_id}});

                        vma_ptr_t dst_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{}); 

                        dg::network_vmamap::region_remapper_remap_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg             = CudaResolutorArgument{};
                            cuda_resolutor_arg.dst              = dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.src              = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.dispatch_control = dispatch_info.tileops_cuda_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_resolutor_arg.src, cuda_resolutor_arg);
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.dst              = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            host_resolutor_arg.src              = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_resolutor_arg.src, host_resolutor_arg);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_extndst_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_init_signal(dst_observer_arr[j], expected_ops_id)));
                        }
                    }
                }
            };
    };

    //clear
    class ForwardDoExtnDsxSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

            (void) event_arr;
        }
    };

    //clear
    class ForwardDoCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz; 

        public:

            ForwardDoCritSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                         std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                         std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                         size_t radxfetch_vectorization_sz,
                                         size_t request_delivery_capacity,
                                         size_t region_vectorization_sz,
                                         size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                    host_async_device(std::move(host_async_device)),
                                                                                    cuda_async_device(std::move(cuda_async_device)),
                                                                                    radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                    request_delivery_capacity(request_delivery_capacity),
                                                                                    region_vectorization_sz(region_vectorization_sz),
                                                                                    forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                        =

                const size_t EVENT_SCALE_FACTOR             = MAX_OBSERVER_ARR_SZ + 1u;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = RadixFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.allocator                = &arena_allocator;
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->src);

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(event_arr[i].dst);

                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        resolutor_key_arg.src_logit_vd_id   = dispatch_radix_arg_arr[i]->src_logit_vd_id;
                        resolutor_key_arg.dst_logit_vd_id   = dispatch_radix_arg_arr[i]->dst_logit_vd_id;
                        resolutor_key_arg.dst_crit_vd_id    = dispatch_radix_arg_arr[i]->dst_crit_vd_id;
                        resolutor_key_arg.dst_grad_vd_id    = dispatch_radix_arg_arr[i]->dst_grad_vd_id;

                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.src               = dispatch_radix_arg_arr[i]->src;
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t src;
                device_id_t src_logit_vd_id;
                device_id_t dst_logit_vd_id;
                device_id_t dst_crit_vd_id;
                device_id_t dst_grad_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, RadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_crit_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    auto dispatch_radix             = DispatchRadixArgument{};
                                    auto dispatch_control           = dg::network_tile_member_getsetter::get_crit_dispatch_control_nothrow(data_arr[i].root);
                                    dispatch_radix.src              = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(data_arr[i].root);
                                    auto dispatch_info              = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_crit_forward_dispatch(dispatch_control));
                                    dispatch_radix.src_logit_vd_id  = dispatch_info.src_logit_vd_id;
                                    dispatch_radix.dst_logit_vd_id  = dispatch_info.dst_logit_vd_id;
                                    dispatch_radix.dst_crit_vd_id   = dispatch_info.dst_crit_vd_id;
                                    dispatch_radix.dst_grad_vd_id   = dispatch_info.dst_grad_vd_id;

                                    *data_arr[i].fetching_addr      = dispatch_radix;
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

            struct CudaResolutorArgument{
                cuda_ptr_t dst_logit_ptr;
                cuda_ptr_t src_logit_ptr;
                cuda_ptr_t dst_crit_ptr;
                cuda_ptr_t dst_grad_ptr;
                cuda_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_arr_sz      = sz * 4;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_arr(cuda_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_arr[i * 4]     = data_arr[i].dst_logit_ptr;
                        cuda_ptr_arr[i * 4 + 1] = data_arr[i].src_logit_ptr;
                        cuda_ptr_arr[i * 4 + 2] = data_arr[i].dst_crit_ptr;
                        cuda_ptr_arr[i * 4 + 3] = data_arr[i].dst_grad_ptr;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::decode_crit_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [arg = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::forward_crit(arg.dst_logit_ptr, arg.src_logit_ptr, 
                                                                                                                   arg.dst_crit_ptr, arg.dst_grad_ptr, 
                                                                                                                   arg.dispatch_control));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        virtual_wo_vec->add(std::move(virtual_wo));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_arr.get(), std::next(cuda_ptr_arr.get(), cuda_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t dst_logit_ptr;
                host_ptr_t src_logit_ptr;
                host_ptr_t dst_crit_ptr;
                host_ptr_t dst_grad_ptr;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_arr_sz      = sz * 4;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_arr(host_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_arr[i * 4]     = data_arr[i].dst_logit_ptr;
                        host_ptr_arr[i * 4 + 1] = data_arr[i].src_logit_ptr;
                        host_ptr_arr[i * 4 + 2] = data_arr[i].dst_crit_ptr;
                        host_ptr_arr[i * 4 + 3] = data_arr[i].dst_grad_ptr;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_crit_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [arg = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::forward_crit(arg.dst_logit_ptr, arg.src_logit_ptr, 
                                                                                                                   arg.dst_crit_ptr, arg.dst_grad_ptr, 
                                                                                                                   arg.dispatch_control));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        virtual_wo_vec->add(std::move(virtual_wo));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_arr.get(), std::next(host_ptr_arr.get(), host_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            //TODOs: word size cmp + has_unique_object_representations_v
            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;
                device_id_t dst_logit_vd_id;
                device_id_t src_logit_vd_id;
                device_id_t dst_crit_vd_id;
                device_id_t dst_grad_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_logit_vd_id, src_logit_vd_id, dst_crit_vd_id, dst_grad_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_logit_vd_id, src_logit_vd_id, dst_crit_vd_id, dst_grad_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                          = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 4u>{}));
                    auto dst_logit_vmamap_reacquirer                = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_logit_vmamap_reacquirer                = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto dst_crit_vmamap_reacquirer                 = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto dst_grad_vmamap_reacquirer                 = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer));
                    auto cuda_internal_resolutor                    = InternalCudaResolutor{};
                    cuda_internal_resolutor.async_device            = this->cuda_async_device;
                    cuda_internal_resolutor.synchronizer            = &cuda_synchronizer;
                    cuda_internal_resolutor.restrict_synchronizer   = &cuda_restrict_synchronizer;
                    cuda_internal_resolutor.allocator               = this->allocator;

                    auto host_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto host_internal_resolutor                    = InternalHostResolutor{};
                    host_internal_resolutor.async_device            = this->host_async_device;
                    host_internal_resolutor.synchronizer            = &host_synchronizer;
                    host_internal_resolutor.restrict_synchronizer   = &host_restrict_synchronizer;
                    host_internal_resolutor.allocator               = this->allocator;

                    size_t trimmed_cuda_vectorization_sz            = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_keyhint_preallocated_raiihandle(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_host_vectorization_sz            = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_internal_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_keyhint_preallocated_raiihandle(&host_internal_resolutor, trimmed_host_vectorization_sz, hdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_crit_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_operatable_fwd_id   = dg::network_tile_member_getsetter::get_crit_operatable_forward_id_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_crit_logit_addr_nothrow(dst);
                        uma_ptr_t dst_crit_umaptr               = dg::network_tile_member_getsetter::get_crit_crit_addr_nothrow(dst);
                        uma_ptr_t dst_grad_umaptr               = dg::network_tile_member_getsetter::get_crit_grad_addr_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_crit_dispatch_control_nothrow(dst);
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_crit_observer_array_size_nothrow(dst);
                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(MAX_OBSERVER_ARR_SZ);
                        dg::network_tile_member_getsetter::get_crit_observer_array_nothrow(dst, dst_observer_arr.get()); 

                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<operatable_id_t, exception_t> src_operatable_fwd_id   = dg::network_tile_member_getsetter::get_tile_operatable_forward_id(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);

                        if (!src_init_status.has_value() || !src_operatable_fwd_id.has_value() || !src_logit_umaptr.has_value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_ADOPTED && dst_init_status != TILE_INIT_STATUS_DECAYED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_operatable_fwd_id != src_operatable_fwd_id.value()){
                            continue;
                        }

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_crit_forward_dispatch(dispatch_control)); 

                        if (dispatch_info.dst_logit_vd_id != key.dst_logit_vd_id){
                            continue;
                        }

                        if (dispatch_info.src_logit_vd_id != key.src_logit_vd_id){
                            continue;
                        }

                        if (dispatch_info.dst_crit_vd_id != key.dst_crit_vd_id){
                            continue;
                        }

                        if (dispatch_info.dst_grad_vd_id != key.dst_grad_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dispatch_info.dst_logit_vd_id},
                                                                                                 {src_logit_umaptr.value(), dispatch_info.src_logit_vd_id},
                                                                                                 {dst_crit_umaptr, dispatch_info.dst_crit_vd_id},
                                                                                                 {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}});

                        vma_ptr_t dst_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t dst_crit_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 3u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(dst_crit_vmamap_reacquirer, dst_crit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg             = CudaResolutorArgument{};
                            cuda_resolutor_arg.dst_logit_ptr    = dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.src_logit_ptr    = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.dst_crit_ptr     = dg::network_vmamap::get_cuda_ptr(dst_crit_vmamap_reacquirer);
                            cuda_resolutor_arg.dst_grad_ptr     = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.dispatch_control = dispatch_info.tileops_cuda_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_resolutor_arg.src_logit_ptr, cuda_resolutor_arg);
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.dst_logit_ptr    = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            host_resolutor_arg.src_logit_ptr    = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            host_resolutor_arg.dst_crit_ptr     = dg::network_vmamap::get_host_ptr(dst_crit_vmamap_reacquirer);
                            host_resolutor_arg.dst_grad_ptr     = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_resolutor_arg.src_logit_ptr, host_resolutor_arg);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_crit_grad_status_nothrow(dst, TILE_GRAD_STATUS_HAS_VALUE);
                        dg::network_tile_member_getsetter::set_crit_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(dst_observer_arr[j], expected_ops_id)));
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(dst, expected_ops_id)));
                    }
                }
            };
    };

    //optimizables
    class ForwardDoMsgrFwdSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t eu_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoMsgrFwdSingalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box,
                                              std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                              size_t request_delivery_capacity,
                                              size_t eu_delivery_capacity,
                                              size_t radxfetch_vectorization_sz,
                                              size_t region_vectorization_sz,
                                              size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                         eu_packet_box(std::move(eu_packet_box)),
                                                                                         host_async_device(std::move(host_async_device)),
                                                                                         request_delivery_capacity(request_delivery_capacity),
                                                                                         eu_delivery_capacity(eu_delivery_capacity),
                                                                                         radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                         region_vectorization_sz(region_vectorization_sz),
                                                                                         forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                        = {};

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
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = RadixFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst; 
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                         = InternalResolutor{};
                    internal_resolutor.request_delivery_handle      = request_delivery_handle.get();
                    internal_resolutor.eu_packet_delivery_handle    = eu_packet_delivery_handle.get();
                    internal_resolutor.host_async_device            = this->host_async_device.get();
                    internal_resolutor.vectorization_sz             = this->forward_vectorization_sz;
                    internal_resolutor.allocator                    = &arena_allocator;

                    size_t trimmed_region_vectorization_sz          = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get())); 

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->src);

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size())); 
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].dst);

                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        resolutor_key_arg.dst_vd_id         = dispatch_radix_arg_arr[i]->dst_vd_id;
                        resolutor_key_arg.src_vd_id         = dispatch_radix_arg_arr[i]->src_vd_id;

                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.src               = dispatch_radix_arg_arr[i]->src;
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t src;
                device_id_t dst_vd_id;
                device_id_t src_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, RadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrfwd_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    auto dispatch_radix         = DispatchRadixArgument{};
                                    auto dispatch_control       = dg::network_tile_member_getsetter::get_msgrfwd_dispatch_control_nothrow(data_arr[i].root);
                                    dispatch_radix.src          = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(data_arr[i].root);
                                    auto dispatch_info          = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_msgrfwd_forward_dispatch());
                                    dispatch_radix.dst_vd_id    = dispatch_info.dst_vd_id;
                                    dispatch_radix.src_vd_id    = dispatch_info.src_vd_id;
                                    *data_arr[i].fetching_addr  = dispatch_radix; 
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

            struct HostResolutorArgument{
                host_ptr_t dst;
                host_ptr_t src;
                host_ptr_t cpy_dst;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_vec_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_vec(host_ptr_vec_sz);
                    size_t total_complexity     = {}; 

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_vec[i * 3]     = data_arr[i].dst;
                        host_ptr_vec[i * 3 + 1] = data_arr[i].src;
                        host_ptr_vec[i * 3 + 2] = data_arr[i].cpy_dst;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(MEMCPY)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            size_t cpy_dst_bsz = dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(e.dispatch_control)).dst_byte_size;
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::forward_mono(e.dst, e.src, e.dispatch_control));
                            dg::network_exception_handler::nothrow_log(dg::network_memops_clib::memcpy_host_to_host(e.cpy_dst, e.dst, cpy_dst_bsz));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_vec.get(), std::next(host_ptr_vec.get(), host_ptr_vec_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct MsgrFwdData{
                user_id_t user_id;
                uma_ptr_t addr;
                dg::string logit_value;
                Address dst;
                eu_packet_urgency_t urgency;
                uint8_t retry_count;
                eu_packet_comm_t comm; 
            };

            //word_size memcmp + has_unique_object_representations_v
            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;
                device_id_t dst_vd_id;
                device_id_t src_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_producer_consumer::DeliveryHandle<EndUserPacket> * eu_packet_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                  = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{}));
                    auto dst_logit_vmamap_reacquirer        = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_logit_vmamap_reacquirer        = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto host_synchronizer                  = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer         = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto host_resolutor                     = InternalHostResolutor{};
                    host_resolutor.async_device             = this->host_async_device;
                    host_resolutor.synchronizer             = &host_synchronizer;
                    host_resolutor.restrict_synchronizer    = &host_restrict_synchronizer;
                    host_resolutor.allocator                = this->allocator;

                    size_t trimmed_vectorization_sz         = std::min(this->vectorization_sz, sz);
                    size_t hv_allocation_cost               = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_resolutor, trimmed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hv_mem(hv_allocation_cost);
                    auto host_vectorizer                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_resolutor, trimmed_vectorization_sz, hv_mem.get())); 

                    auto msgrfwd_outbound_vec               = dg::vector<std::optional<MsgrFwdData>>(sz, std::optional<MsgrFwdData>(std::nullopt));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_msgrfwd_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_msgrfwd_operatable_forward_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_msgrfwd_logit_addr_nothrow(dst);
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_msgrfwd_observer_array_size_nothrow(dst);
                        size_t dst_logit_bsz                    = dg::network_tile_member_getsetter::get_msgrfwd_logit_byte_size_nothrow(dst);
                        dst_info_t dst_msgr_info                = dg::network_tile_member_getsetter::get_msgrfwd_dst_info_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_msgrfwd_dispatch_control_nothrow(dst);
                        user_id_t user_id                       = dg::network_tile_member_getsetter::get_msgrfwd_user_id_nothrow(dst);

                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(MAX_OBSERVER_ARR_SZ);
                        dg::network_tile_member_getsetter::get_msgrfwd_observer_array_nothrow(dst, dst_observer_arr.get());

                        std::expected<operatable_id_t, exception_t> src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_forward_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);

                        if (!src_fwd_operatable_id.has_value() || !src_init_status.has_value() || !src_logit_umaptr.has_value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED && dst_init_status != TILE_INIT_STATUS_ADOPTED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
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

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_msgrfwd_forward_dispatch(dispatch_control));

                        if (dispatch_info.dst_vd_id != key.dst_vd_id){
                            continue;
                        }

                        if (dispatch_info.src_vd_id != key.src_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dispatch_info.dst_vd_id}, 
                                                                                                           {src_logit_umaptr.value(), dispatch_info.src_vd_id}});

                        vma_ptr_t dst_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){ //we are doing cuda_ptr_t (cutf_ptr_t) -> host_ptr_t
                            auto msgrfwd_data                   = MsgrFwdData{};
                            msgrfwd_data.user_id                = user_id;
                            msgrfwd_data.addr                   = dst;
                            msgrfwd_data.logit_value            = dg::string(dst_logit_bsz, ' '); //TODOs: optimizables
                            msgrfwd_data.dst                    = dst_msgr_info.dst;
                            msgrfwd_data.urgency                = dst_msgr_info.urgency;
                            msgrfwd_data.retry_count            = dst_msgr_info.retry_count;
                            msgrfwd_data.comm                   = dst_msgr_info.comm;
                            host_ptr_t cpylogit_value_ptr       = msgrfwd_data.logit_value.data();
                            msgrfwd_outbound_vec[i]             = std::move(msgrfwd_data);

                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.dst              = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            host_resolutor_arg.src              = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            host_resolutor_arg.cpy_dst          = cpylogit_value_ptr;
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_vectorizer.get(), host_resolutor_arg.src, host_resolutor_arg);
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
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(dst_observer_arr[j], expected_ops_id)));
                        }
                    }

                    dg::network_producer_consumer::delvrsrv_clear(host_vectorizer.get());
                    host_synchronizer.sync();

                    for (size_t i = 0u; i < sz; ++i){
                        if (!msgrfwd_outbound_vec[i].has_value()){
                            continue;
                        }

                        user_id_t user_id               = msgrfwd_outbound_vec[i]->user_id
                        uma_ptr_t addr                  = msgrfwd_outbound_vec[i]->addr; 
                        EndUserPacket eu_packet         = {};
                        eu_packet.serialization_header  = EUPACKET_MSGRFWD; //serialization header
                        eu_packet.content               = dg::network_compact_serializer::serialize<dg::string>(LogitValue{msgrfwd_outbound_vec[i]->addr, std::move(msgrfwd_outbound_vec[i]->logit_value)}); //TODOs: optimizables - solve hardware cache pollution 
                        eu_packet.dst                   = msgrfwd_outbound_vec[i]->dst;
                        eu_packet.retry_count           = msgrfwd_outbound_vec[i]->retry_count;
                        eu_packet.urgency               = msgrfwd_outbound_vec[i]->urgency;
                        eu_packet.comm                  = msgrfwd_outbound_vec[i]->comm;
                        eu_packet.exception_handler     = dg::network_exception::make_exception_handler_from_lambda([user_id, addr](exception_t err) noexcept{ //TODOs: optimizables
                            if (dg::network_exception::is_failed(err)){
                                dg::network_log::log_user_tile_error(user_id, addr, dg::network_exception::verbose(err));
                            }
                        });

                        dg::network_producer_consumer::delvrsrv_deliver(this->eu_packet_delivery_handle, std::move(eu_packet));
                    }
                }
            };
    };

    //clear
    class ForwardDoMsgrBwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t forward_vectorization_sz;

        public:

            ForwardDoMsgrBwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                            std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                            size_t request_delivery_capacity,
                                            size_t radxfetch_vectorization_sz,
                                            size_t region_vectorization_sz,
                                            size_t forward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                       host_async_device(std::move(host_async_device)),
                                                                                       cuda_async_device(std::move(cuda_async_device)),
                                                                                       request_delivery_capacity(request_delivery_capacity),
                                                                                       radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                       region_vectorization_sz(region_vectorization_sz),
                                                                                       forward_vectorization_sz(forward_vectorization_sz){}

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                        = ;

                const size_t EVENT_SCALE_FACTOR             = MAX_OBSERVER_ARRAY_SZ;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost); 
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = RadixFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.allocator                = &arena_allocator;
                    internal_resolutor.vectorization_sz         = this->forward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz); 
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->src);

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_msgrbwd_tile_rcu_addr_nothrow(event_arr[i].dst);

                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        resolutor_key_arg.dst_vd_id         = dispatch_radix_arg_arr[i]->dst_vd_id;
                        resolutor_key_arg.src_vd_id         = dispatch_radix_arg_arr[i]->src_vd_id;

                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.src               = dispatch_radix_arg_arr[i]->src;
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t src;
                device_id_t dst_vd_id;
                device_id_t src_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, RadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrbwd_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]:
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    auto dispatch_radix         = DispatchRadixArgument{};
                                    auto dispatch_control       = dg::network_tile_member_getsetter::get_msgrbwd_dispatch_control_nothrow(data_arr[i].root);
                                    dispatch_radix.src          = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(data_arr[i].root);
                                    auto dispatch_info          = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_msgrbwd_forward_dispatch(dispatch_control)); 
                                    dispatch_radix.dst_vd_id    = dispatch_info.dst_vd_id;
                                    dispatch_radix.src_vd_id    = dispatch_info.src_vd_id; 
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

            struct CudaResolutorArgument{
                cuda_ptr_t dst;
                cuda_ptr_t src;
                cuda_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_arr_sz      = sz * 2;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_arr(cuda_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_arr[i * 2]     = data_arr[i].dst;
                        cuda_ptr_arr[i * 2 + 1] = data_arr[i].src;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::forward_mono(e.dst, e.src, e.dispatch_control));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_arr.get(), std::next(cuda_ptr_arr.get(), cuda_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t dst;
                host_ptr_t src;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_arr_sz      = sz * 2;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_arr(host_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_arr[i * 2]     = data_arr[i].dst;
                        host_ptr_arr[i * 2 + 1] = data_arr[i].src;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::forward_mono(e.dst, e.src, e.dispatch_control));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_arr.get(), std::next(host_ptr_arr.get(), host_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            //TODOs: word_size memcmp + has_unique_object_representations_v
            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;
                device_id_t dst_vd_id;
                device_id_t src_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr, dst_vd_id, src_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                  = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{}));
                    auto dst_vmamap_reacquirer              = dg::network_exception_handler::nothrow_log(dg::network_vmamap::remapper_raii_initialize());
                    auto src_vmamap_reacquirer              = dg::network_exception_handler::nothrow_log(dg::network_vmamap::remapper_raii_initialize());

                    auto cuda_synchronizer                  = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer         = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer)); 
                    auto cuda_resolutor                     = InternalCudaResolutor{};
                    cuda_resolutor.async_device             = this->cuda_async_device;
                    cuda_resolutor.synchronizer             = &cuda_synchronizer;
                    cuda_resolutor.restrict_synchronizer    = &cuda_restrict_synchronizer;
                    cuda_resolutor.allocator                = this->allocator;

                    auto host_synchronizer                  = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer         = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer)); 
                    auto host_resolutor                     = InternalHostResolutor{};
                    host_resolutor.async_device             = this->host_async_device;
                    host_resolutor.synchronizer             = &host_synchronizer;
                    host_resolutor.restrict_synchronizer    = &host_restrict_synchronizer;
                    host_resolutor.allocator                = this->allocator;

                    size_t trimmed_host_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_resolutor, trimmed_host_vectorization_sz, hdh_mem.get())); 

                    size_t trimmed_cuda_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&cuda_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_msgrbwd_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_fwd_operatable_id   = dg::network_tile_member_getsetter::get_msgrbwd_operatable_forward_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr              = dg::network_tile_member_getsetter::get_msgrbwd_logit_addr_nothrow(dst);
                        size_t dst_observer_arr_sz              = dg::network_tile_member_getsetter::get_msgrbwd_observer_array_size_nothrow(dst);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_msgrbwd_dispatch_control_nothrow(dst);
                        dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> dst_observer_arr(MAX_OBSERVER_ARR_SZ);
                        dg::network_tile_member_getsetter::get_msgrbwd_observer_array_nothrow(dst, dst_observer_arr.get());

                        std::expected<operatable_id_t, exception_t> src_fwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_forward_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src); 

                        if (!src_fwd_operatable_id.has_value() || !src_init_status.has_value() || !src_logit_umaptr.has_value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED && dst_init_status != TILE_INIT_STATUS_ADOPTED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
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

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_msgrbwd_forward_dispatch(dispatch_control));

                        if (dispatch_info.dst_vd_id != key.dst_vd_id){
                            continue;
                        }

                        if (dispatch_info.src_vd_id != key.src_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dispatch_info.dst_vd_id}, 
                                                                                                           {src_logit_umaptr.value(), dispatch_info.src_vd_id}});

                        vma_ptr_t dst_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(dst_vmamap_reacquirer, dst_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_vmamap_reacquirer, src_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg             = CudaResolutorArgument{};
                            cuda_resolutor_arg.dst              = dg::network_vmamap::get_cuda_ptr(dst_vmamap_reacquirer);
                            cuda_resolutor_arg.src              = dg::network_vmamap::get_cuda_ptr(src_vmamap_reacquirer);
                            cuda_resolutor_arg.dispatch_control = dispatch_info.tileops_cuda_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_resolutor_arg.src, cuda_resolutor_arg);
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.dst              = dg::network_vmamap::get_host_ptr(dst_vmamap_reacquirer);
                            host_resolutor_arg.src              = dg::network_vmamap::get_host_ptr(src_vmamap_reacquirer);
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_resolutor_arg.src, host_resolutor_arg);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }                        
                        }

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_forward_do_signal(dst_observer_arr[j], expected_ops_id)));
                        }

                        dg::network_tile_member_getsetter::set_init_status_msgrbwd_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                    }
                }
            };
    };

    //clear
    class ForwardDoImmuSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        public:

            void push(ForwardDoSignalEvent * event_arr, size_t sz) noexcept{

                (void) event_arr;
            }
    };

    //clear
    class ForwardDoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> blkr_resolutor;
            const size_t blkr_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> mono_resolutor;
            const size_t mono_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> pair_resolutor;
            const size_t pair_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> uacm_resolutor;
            const size_t uacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> pacm_resolutor;
            const size_t pacm_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> crit_resolutor;
            const size_t crit_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> immu_resolutor;
            const size_t immu_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extnsrc_resolutor;
            const size_t extnsrc_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extnsrx_resolutor;
            const size_t extnsrx_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extndst_resolutor;
            const size_t extndst_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extndsx_resolutor;
            const size_t extndsx_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> msgrfwd_resolutor;
            const size_t msgrfwd_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> msgrbwd_resolutor;
            const size_t msgrbwd_dispatch_sz;

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
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> crit_resolutor,
                                     size_t crit_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> immu_resolutor,
                                     size_t immu_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extnsrc_resolutor,
                                     size_t extnsrc_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extnsrx_resolutor,
                                     size_t extnsrx_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extndst_resolutor,
                                     size_t extndst_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> extndsx_resolutor,
                                     size_t extndsx_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> msgrfwd_resolutor,
                                     size_t msgrfwd_dispatch_sz,
                                     std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<ForwardDoSignalEvent>> msgrbwd_resolutor,
                                     size_t msgrbwd_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
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
                                                                           crit_resolutor(std::move(crit_resolutor)),
                                                                           crit_dispatch_sz(crit_dispatch_sz),
                                                                           immu_resolutor(std::move(immu_resolutor)),
                                                                           immu_dispatch_sz(immu_dispatch_sz),
                                                                           extnsrc_resolutor(std::move(extnsrc_resolutor)),
                                                                           extnsrc_dispatch_sz(extnsrc_dispatch_sz),
                                                                           extnsrx_resolutor(std::move(extnsrx_resolutor)),
                                                                           extnsrx_dispatch_sz(extnsrx_dispatch_sz),
                                                                           extndst_resolutor(std::move(extndst_resolutor)),
                                                                           extndst_dispatch_sz(extndst_dispatch_sz),
                                                                           extndsx_resolutor(std::move(extndsx_resolutor)),
                                                                           extndsx_dispatch_sz(extndsx_dispatch_sz),
                                                                           msgrfwd_resolutor(std::move(msgrfwd_resolutor)),
                                                                           msgrfwd_dispatch_sz(msgrfwd_dispatch_sz),
                                                                           msgrbwd_resolutor(std::move(msgrbwd_resolutor)),
                                                                           msgrbwd_dispatch_sz(msgrbwd_dispatch_sz){}

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

                size_t trimmed_extnsrx_dispatch_sz  = std::min(this->extnsrx_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extnsrx_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extnsrx_resolutor.get(), trimmed_extnsrx_dispatch_sz));

                size_t trimmed_extndst_dispatch_sz  = std::min(this->extndst_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extndst_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extndst_resolutor.get(), trimmed_extndst_dispatch_sz));

                size_t trimmed_extndsx_dispatch_sz  = std::min(this->extndsx_dispatch_sz, sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> extndsx_dh_mem(dg::network_producer_consumer::delvrsrv_allocation_cost(this->extndsx_resolutor.get(), trimmed_extndsx_dispatch_sz));
                
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
                auto extnsrx_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extnsrx_resolutor.get(), trimmed_extnsrx_dispatch_sz, extnsrx_dh_mem.get()));
                auto extndst_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extndst_resolutor.get(), trimmed_extndst_dispatch_sz, extndst_dh_mem.get()));
                auto extndsx_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->extndsx_resolutor.get(), trimmed_extndsx_dispatch_sz, extndsx_dh_mem.get()));
                auto msgrfwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrfwd_resolutor.get(), trimmed_msgrfwd_dispatch_sz, msgrfwd_dh_mem.get()));
                auto msgrbwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->msgrbwd_resolutor.get(), trimmed_msgrbwd_dispatch_sz, msgrbwd_dh_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(event_arr[i].dst);

                    if constexpr(DEBUG_MODE_FLAG){
                        if (!tile_kind.has_value()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(tile_kind.error()));
                            std::abort();
                        }
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
                        case TILE_KIND_EXTNSRX:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extnsrx_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNDST:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extndst_delivery_handle.get(), event_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNDSX:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extndsx_delivery_handle.get(), event_arr[i]);
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
                    };
                }
            }
    };

    //

    //clear
    class BackwardDoLeafSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t infofetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t gradupdate_vectorization_sz;

        public:

            BackwardDoLeafSignalResolutorV2(std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                            std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                            size_t infofetch_vectorization_sz,
                                            size_t region_vectorization_sz,
                                            size_t gradupdate_vectorization_sz) noexcept: cuda_async_device(std::move(cuda_async_device)),
                                                                                          host_async_device(std::move(host_async_device)),
                                                                                          infofetch_vectorization_sz(infofetch_vectorization_sz),
                                                                                          region_vectorization_sz(region_vectorization_sz),
                                                                                          gradupdate_vectorization_sz(gradupdate_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                    = {}; 

                {
                    auto dispatch_radix_fetcher                 = InternalDispatchRadixFetcher{};

                    size_t trimmed_infofetch_vectorization_sz   = std::min(this->infofetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&dispatch_radix_fetcher, trimmed_infofetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&dispatch_radix_fetcher, trimmed_infofetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_leaf_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = RadixFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i); 

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                 = InternalResolutor{};
                    internal_resolutor.cuda_async_device    = this->cuda_async_device.get();
                    internal_resolutor.host_async_device    = this->host_async_device.get();
                    internal_resolutor.allocator            = &arena_allocator;
                    internal_resolutor.vectorization_sz     = this->gradupdate_vectorization_sz;

                    size_t trimmed_region_vectorization_sz  = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_leaf_rcu_addr_nothrow(event_arr[i].dst);

                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.dst_logit_vd_id   = dispatch_radix_arg_arr[i]->dst_logit_vd_id;
                        resolutor_key_arg.dst_grad_vd_id    = dispatch_radix_arg_arr[i]->dst_grad_vd_id;

                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id; 

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }   
                }
            }

        private:

            struct DispatchRadixArgument{
                device_id_t dst_logit_vd_id;
                device_id_t dst_grad_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, RadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status               = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(data_arr[i].root);
                        set_operatable_id_t current_ops_id_set  = dg::network_tile_member_getsetter::get_leaf_operatable_memevent_id_set_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (is_subset_id(data_arr[i].expected_ops_id, current_ops_id_set)){
                                    auto dispatch_radix             = DispatchRadixArgument{};
                                    auto dispatch_control           = dg::network_tile_member_getsetter::get_leaf_gradupdate_dispatch_control_nothrow(data_arr[i].root);
                                    auto dispatch_info              = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_leaf_gradupdate_dispatch(dispatch_control));
                                    dispatch_radix.dst_logit_vd_id  = dispatch_info.logit_vd_id;
                                    dispatch_radix.dst_grad_vd_id   = dispatch_info.grad_vd_id;

                                    *data_arr[i].fetching_addr      = dispatch_radix;
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

            struct CudaResolutorArgument{
                cuda_ptr_t logit_ptr;
                cuda_ptr_t grad_ptr;
                cuda_tileops_dispatch_control_t dispatch_control;
                cuda_write_option_t grad_write_option;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t total_complexity     = {}; 

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::decode_grad_update_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::grad_update(e.logit_ptr, e.grad_ptr, e.dispatch_control, e.write_option)); //TODOs: cuda limitation of kernel dispatches
                        };

                        size_t async_task_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * async_task_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(async_task_bsz));
                        auto async_task         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, async_task_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(async_task)));
                    }

                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity));
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t logit_ptr;
                host_ptr_t grad_ptr;
                host_tileops_dispatch_control_t dispatch_control;
                host_write_option_t grad_write_option;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::decode_grad_update_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::grad_update(e.logit_ptr, e.grad_ptr, e.dispatch_control, e.write_option));
                        };

                        size_t async_task_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * async_task_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(async_task_bsz));
                        auto async_task         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, async_task_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(async_task)));
                    }

                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity));
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                device_id_t dst_logit_vd_id;
                device_id_t dst_grad_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, dst_logit_vd_id, dst_grad_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, dst_logit_vd_id, dst_grad_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz; 

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr);

                    auto umamap_reacquirer                  = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{}));
                    auto logit_vmamap_reacquirer            = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto grad_vmamap_reacquirer             = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                  = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_internal_resolutor            = InternalCudaResolutor{};
                    cuda_internal_resolutor.async_device    = this->cuda_async_device;
                    cuda_internal_resolutor.synchronizer    = &cuda_synchronizer;
                    cuda_internal_resolutor.allocator       = this->allocator;

                    auto host_synchronizer                  = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_internal_resolutor            = InternalHostResolutor{};
                    host_internal_resolutor.async_device    = this->host_async_device;
                    host_internal_resolutor.synchronizer    = &host_synchronizer;
                    host_internal_resolutor.allocator       = this->allocator;

                    size_t trimmed_cuda_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_allocation_cost(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_host_vectorization_sz    = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost              = dg::network_producer_consumer::delvrsrv_allocation_cost(&host_internal_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle               = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&host_internal_resolutor, trimmed_host_vectorization_sz, hdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, expected_ops_id]             = std::make_tuple(data_arr[i].dst, data_arr[i].expected_ops_id);
                        operatable_id_set_t operatable_id_set   = dg::network_tile_member_getsetter::get_leaf_operatable_memevent_id_set_nothrow(dst); //we are doing compact set by using interval [first, last)
                        init_status_t init_status               = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(dst);
                        grad_status_t grad_status               = dg::network_tile_member_getsetter::get_leaf_grad_status_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_leaf_gradupdate_dispatch_control_nothrow(dst);
                        uma_ptr_t logit_umaptr                  = dg::network_tile_member_getsetter::get_leaf_logit_addr_nothrow(dst);
                        uma_ptr_t grad_umaptr                   = dg::network_tile_member_getsetter::get_leaf_grad_addr_nothrow(dst);

                        if (init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (!is_subset_id(expected_ops_id, operatable_id_set)){
                            continue;
                        }

                        if (grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_leaf_gradupdate_dispatch(dispatch_control));

                        if (dispatch_info.logit_vd_id != key.dst_logit_vd_id){
                            continue;
                        }

                        if (dispatch_info.grad_vd_id != key.dst_grad_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{logit_umaptr, dispatch_info.logit_vd_id}, 
                                                                                                           {grad_umaptr, dispatch_info.grad_vd_id}});

                        vma_ptr_t logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(logit_vmamap_reacquirer, logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(grad_vmamap_reacquirer, grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg                 = CudaResolutorArgument{};
                            cuda_resolutor_arg.logit_ptr            = dg::network_vmamap::get_cuda_ptr(logit_vmamap_reacquirer);
                            cuda_resolutor_arg.grad_ptr             = dg::network_vmamap::get_cuda_ptr(grad_vmamap_reacquirer);
                            cuda_resolutor_arg.dispatch_control     = dispatch_info.tileops_cuda_dispatch_control;
                            cuda_resolutor_arg.grad_write_option    = CUDA_TILEOPS_OPERATION_ZERO;

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_resolutor_arg);
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg                 = HostResolutorArgument{};
                            host_resolutor_arg.logit_ptr            = dg::network_vmamap::get_host_ptr(logit_vmamap_reacquirer);
                            host_resolutor_arg.grad_ptr             = dg::network_vmamap::get_host_ptr(grad_vmamap_reacquirer);
                            host_resolutor_arg.dispatch_control     = dispatch_info.tileops_host_dispatch_control;
                            host_resolutor_arg.grad_write_option    = HOST_TILEOPS_OPERATION_ZERO;

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_resolutor_arg);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_leaf_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED); //this guarantees pointer restriction
                    }
                }
            };
    };

    //clear
    class BackwardDoMonoSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t backward_vectorization_sz;

        public:

            BackwardDoMonoSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                            std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                            size_t request_delivery_capacity,
                                            size_t radxfetch_vectorization_sz,
                                            size_t region_vectorization_sz,
                                            size_t backward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                        cuda_async_device(std::move(cuda_async_device)),
                                                                                        host_async_device(std::move(host_async_device)),
                                                                                        request_delivery_capacity(request_delivery_capacity),
                                                                                        radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                        region_vectorization_sz(region_vectorization_sz),
                                                                                        backward_vectorization_sz(backward_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                        = 

                const size_t EVENT_SCALE_FACTOR             = 1u;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            std::expected<uma_ptr_t, exception_t> ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = AddressFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.vectorization_sz         = this->backward_vectorization_sz;
                    internal_resolutor.allocator                = &arena_allocator;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> descendant_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->src);

                        if (!descendant_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz            = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr          = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(event_arr[i].dst);

                        auto resolutor_key              = ResolutorKeyArgument{};
                        resolutor_key.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key.src_lck_addr      = dg::memult::region(descendant_rcu_addr.value(), lck_region_sz);
                        resolutor_key.src_grad_vd_id    = dispatch_radix_arg_arr[i]->src_grad_vd_id;
                        resolutor_key.src_logit_vd_id   = dispatch_radix_arg_arr[i]->src_logit_vd_id;
                        resolutor_key.dst_grad_vd_id    = dispatch_radix_arg_arr[i]->dst_grad_vd_id;

                        auto resolutor_val              = ResolutorValueArgument{};
                        resolutor_arg.dst               = event_arr[i].dst;
                        resolutor_arg.src               = dispatch_radix_arg_arr[i]->src;
                        resolutor_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key, resolutor_val);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t src;
                device_id_t src_grad_vd_id;
                device_id_t src_logit_vd_id;
                device_id_t dst_grad_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, RadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_mono_operatable_memevent_id_nothrow(data_arr[i].root);  

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (data_arr[i].expected_ops_id == current_ops_id){
                                    auto dispatch_radix             = DispatchRadixArgument{};
                                    auto dispatch_control           = dg::network_tile_member_getsetter::get_mono_backward_dispatch_control_nothrow(data_arr[i].root);
                                    dispatch_radix.src              = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(data_arr[i].root);
                                    auto dispatch_info              = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_mono_backward_dispatch(dispatch_control));
                                    dispatch_radix.src_grad_vd_id   = dispatch_info.src_grad_vd_id;
                                    dispatch_radix.src_logit_vd_id  = dispatch_info.src_logit_vd_id;
                                    dispatch_radix.dst_grad_vd_id   = dispatch_info.dst_grad_vd_id;

                                    *data_arr[i].fetching_addr      = dispatch_radix;
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

            struct CudaResolutorArgument{
                cuda_ptr_t src_grad_addr;
                cuda_ptr_t src_logit_addr;
                cuda_ptr_t dst_grad_addr;
                grad_status_t src_grad_status;
                cuda_tileops_dispatch_control_t dispatch_control;
            };
            
            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_arr_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_arr(cuda_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_arr[i * 3]     = data_arr[i].src_grad_addr;
                        cuda_ptr_arr[i * 3 + 1] = data_arr[i].src_logit_addr;
                        cuda_ptr_arr[i * 3 + 2] = data_arr[i].dst_grad_addr;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::decode_mono_backward_dispatch(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::backward_mono(e.src_grad_addr, e.src_logit_addr, e.dst_grad_addr, 
                                                                                                                    e.dispatch_control, convert_grad_status_to_cuda_write_option(e.src_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf)); 

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_arr.get(), std::next(cuda_ptr_arr.get(), cuda_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t src_grad_addr;
                host_ptr_t src_logit_addr;
                host_ptr_t dst_grad_addr;
                grad_status_t src_grad_status;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_arr_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_arr(host_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_arr[i * 3]     = data_arr[i].src_grad_addr;
                        host_ptr_arr[i * 3 + 1] = data_arr[i].src_logit_addr;
                        host_ptr_arr[i * 3 + 2] = data_arr[i].dst_grad_addr;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_backward_dispatch(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::backward_mono(e.src_grad_addr, e.src_logit_addr, e.dst_grad_addr, 
                                                                                                                    e.dispatch_control, convert_grad_status_to_host_write_option(e.src_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_arr.get(), std::next(host_ptr_arr.get(), host_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            //TODOs: word_size memcmp + has_unique_object_representations_v
            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;
                device_id_t src_grad_vd_id;
                device_id_t src_logit_vd_id;
                device_id_t dst_grad_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr, src_grad_vd_id, src_logit_vd_id, dst_grad_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr, src_grad_vd_id, src_logit_vd_id, dst_grad_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                              = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{}));
                    auto src_grad_vmamap_reacquirer                     = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_logit_vmamap_reacquirer                    = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto dst_grad_vmamap_reacquirer                     = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                              = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer                     = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer));
                    auto cuda_internal_resolutor                        = InternalCudaResolutor{};
                    cuda_internal_resolutor.async_device                = this->cuda_async_device;
                    cuda_internal_resolutor.synchronizer                = &cuda_synchronizer;
                    cuda_internal_resolutor.restrict_synchronizer       = &cuda_restrict_synchronizer;  
                    cuda_internal_resolutor.allocator                   = this->allocator;

                    auto host_synchronizer                              = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer                     = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto host_internal_resolutor                        = InternalHostResolutor{};
                    host_internal_resolutor.async_device                = this->host_async_device;
                    host_internal_resolutor.synchronizer                = &host_synchronizer;
                    host_internal_resolutor.restrict_synchronizer       = &host_restrict_synchronizer;
                    host_internal_resolutor.allocator                   = this->allocator;

                    size_t trimmed_cuda_vectorization_sz                = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost                          = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle                           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_host_vectorization_sz                = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost                          = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_internal_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle                           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_internal_resolutor, trimmed_host_vectorization_sz, hdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_mono_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_bwd_operatable_id   = dg::network_tile_member_getsetter::get_mono_operatable_backward_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(dst);
                        grad_status_t dst_grad_status           = dg::network_tile_member_getsetter::get_mono_grad_status_nothrow(dst);
                        uma_ptr_t dst_grad_umaptr               = dg::network_tile_member_getsetter::get_mono_grad_addr_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_mono_backward_dispatch_control_nothrow(dst);

                        std::expected<operatable_id_t, exception_t> src_bwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_backward_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_grad_umaptr               = dg::network_tile_member_getsetter::get_tile_grad_addr(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);
                        std::expected<grad_status_t, exception_t> src_grad_status           = dg::network_tile_member_getsetter::get_tile_grad_status(src);

                        if (!src_bwd_operatable_id.has_value() || !src_init_status.has_value() || !src_grad_umaptr.has_value()
                            || !src_logit_umaptr.has_value() || !src_grad_status.has_value()){

                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_bwd_operatable_id != src_bwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_mono_backward_dispatch(dispatch_control));

                        if (dispatch_info.src_grad_vd_id != key.src_grad_vd_id){
                            continue;
                        }

                        if (dispatch_info.src_logit_vd_id != key.src_logit_vd_id){
                            continue;
                        }

                        if (dispatch_info.dst_grad_vd_id != key.dst_grad_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_reacquire_nothrow(umamap_reacquirer, {{src_grad_umaptr.value(), dispatch_info.src_grad_vd_id},
                                                                                                 {src_logit_umaptr.value(), dispatch_info.src_logit_vd_id},
                                                                                                 {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}});

                        vma_ptr_t src_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(src_grad_vmamap_reacquirer, src_grad_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg             = CudaResolutorArgument{};
                            cuda_resolutor_arg.src_grad_addr    = dg::network_vmamap::get_cuda_ptr(src_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.src_logit_addr   = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.dst_grad_addr    = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.src_grad_status  = src_grad_status.value();
                            cuda_resolutor_arg.dispatch_control = dispatch_info.tileops_cuda_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_resolutor_arg.src_logit_addr, cuda_resolutor_arg);
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.src_grad_addr    = dg::network_vmamap::get_host_ptr(src_grad_vmamap_reacquirer);
                            host_resolutor_arg.src_logit_addr   = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            host_resolutor_arg.dst_grad_addr    = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);
                            host_resolutor_arg.src_grad_status  = src_grad_status.value();
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_resolutor_arg.src_logit_addr, host_resolutor_arg); 
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_mono_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        exception_t err = dg::network_tile_member_getsetter::set_tile_grad_status(src, TILE_GRAD_STATUS_HAS_VALUE);

                        if (dg::network_exception::is_failed(err)){
                            (void) err; //
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::vitualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src, expected_ops_id)));
                    }
                }
            };
    };

    //clear
    class BackwardDoBlkrSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        public:

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{
                
                (void) event_arr;
            };
    };

    //clear
    class BackwardDoPairSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t backward_vectorization_sz;

        public:

            BackwardDoPairSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> request_box,
                                            std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                            std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                            size_t request_delivery_capacity,
                                            size_t radxfetch_vectorization_sz,
                                            size_t region_vectorization_sz,
                                            size_t backward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                        cuda_async_device(std::move(cuda_async_device)),
                                                                                        host_async_device(std::move(host_async_device)),
                                                                                        request_delivery_capacity(request_delivery_capacity),
                                                                                        radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                        region_vectorization_sz(region_vectorization_sz),
                                                                                        backward_vectorization_sz(backward_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                        = 

                const size_t EVENT_SCALE_FACTOR             = 2u;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity); 
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr              = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr              = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg                  = AddressFetcherArgument{};
                        fetch_arg.root                  = event_arr[i].dst;
                        fetch_arg.expected_ops_id       = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr         = std::next(dispatch_radix_arg_arr.get(), i); 

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.vectorization_sz         = this->backward_vectorization_sz;
                    internal_resolutor.allocator                = &arena_allocator;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> lhs_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->lhs);
                        std::expected<uma_ptr_t, exception_t> rhs_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->rhs);

                        if (!lhs_rcu_addr.has_value() || !rhs_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz            = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr          = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(event_arr[i].dst); //assumption: descendant_arr[i].has_value() => safe access

                        auto resolutor_key              = ResolutorKeyArgument{};
                        resolutor_key.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key.lhs_lck_addr      = dg::memult::region(lhs_rcu_addr.value(), lck_region_sz);
                        resolutor_key.rhs_lck_addr      = dg::memult::region(rhs_rcu_addr.value(), lck_region_sz);
                        resolutor_key.dst_grad_vd_id    = dispatch_radix_arg_arr[i]->dst_grad_vd_id;
                        resolutor_key.lhs_logit_vd_id   = dispatch_radix_arg_arr[i]->lhs_logit_vd_id;
                        resolutor_key.lhs_grad_vd_id    = dispatch_radix_arg_arr[i]->lhs_grad_vd_id;
                        resolutor_key.rhs_logit_vd_id   = dispatch_radix_arg_arr[i]->rhs_logit_vd_id;
                        resolutor_key.rhs_grad_vd_id    = dispatch_radix_arg_arr[i]->rhs_grad_vd_id;

                        auto resolutor_val              = ResolutorValueArgument{};
                        resolutor_val.dst               = event_arr[i].dst;
                        resolutor_val.lhs               = dispatch_radix_arg_arr[i]->lhs;
                        resolutor_val.rhs               = dispatch_radix_arg_arr[i]->rhs;
                        resolutor_val.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key, resolutor_val);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t lhs;
                uma_ptr_t rhs;
                device_id_t dst_grad_vd_id;
                device_id_t lhs_logit_vd_id;
                device_id_t lhs_grad_vd_id;
                device_id_t rhs_logit_vd_id;
                device_id_t rhs_grad_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDispatchRadixFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, RadixFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pair_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    auto dispatch_radix             = DispatchRadixArgument{};
                                    auto dispatch_control           = dg::network_tile_member_getsetter::get_pair_backward_dispatch_control_nothrow(data_arr[i].root);
                                    dispatch_radix.lhs              = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(data_arr[i].root);
                                    dispatch_radix.rhs              = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(data_arr[i].root);
                                    auto dispatch_info              = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_backward_pair_dispatch(dispatch_control));
                                    dispatch_radix.dst_grad_vd_id   = dispatch_info.dst_grad_vd_id;
                                    dispatch_radix.lhs_logit_vd_id  = dispatch_info.lhs_logit_vd_id;
                                    dispatch_radix.lhs_grad_vd_id   = dispatch_info.lhs_grad_vd_id;
                                    dispatch_radix.rhs_logit_vd_id  = dispatch_info.rhs_logit_vd_id;
                                    dispatch_radix.rhs_grad_vd_id   = dispatch_info.rhs_grad_vd_id;

                                    *data_arr[i].fetching_addr      = dispatch_radix;
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

            struct CudaResolutorArgument{
                cuda_ptr_t lhs_logit_ptr;
                cuda_ptr_t lhs_grad_ptr;
                grad_status_t lhs_grad_status;
                cuda_ptr_t rhs_logit_ptr;
                cuda_ptr_t rhs_grad_ptr;
                grad_status_t rhs_grad_status;
                cuda_ptr_t dst_grad_ptr;
                cuda_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_arr_sz      = sz * 5;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_arr(cuda_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf)); 

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_arr[i * 5]     = data_arr[i].lhs_logit_ptr;
                        cuda_ptr_arr[i * 5 + 1] = data_arr[i].lhs_grad_ptr;
                        cuda_ptr_arr[i * 5 + 2] = data_arr[i].rhs_logit_ptr;
                        cuda_ptr_arr[i * 5 + 3] = data_arr[i].rhs_grad_ptr;
                        cuda_ptr_arr[i * 5 + 4] = data_arr[i].dst_grad_ptr;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::decode_backward_pair_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::backward_pair(e.lhs_logit_ptr, e.lhs_grad_ptr, 
                                                                                                                    e.rhs_logit_ptr, e.rhs_grad_ptr,
                                                                                                                    e.dst_grad_ptr, e.dispatch_control,
                                                                                                                    convert_grad_status_to_cuda_write_option(e.lhs_grad_status),
                                                                                                                    convert_grad_status_to_cuda_write_option(e.rhs_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo))); 
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_arr.get(), std::next(cuda_ptr_arr.get(), cuda_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t lhs_logit_ptr;
                host_ptr_t lhs_grad_ptr;
                grad_status_t lhs_grad_status;
                host_ptr_t rhs_logit_ptr;
                host_ptr_t rhs_grad_ptr;
                grad_status_t rhs_grad_status;
                host_ptr_t dst_grad_ptr;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_arr_sz      = sz * 5;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_arr(host_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_arr[i * 5]     = data_arr[i].lhs_logit_ptr;
                        host_ptr_arr[i * 5 + 1] = data_arr[i].lhs_grad_ptr;
                        host_ptr_arr[i * 5 + 2] = data_arr[i].rhs_logit_ptr;
                        host_ptr_arr[i * 5 + 3] = data_arr[i].rhs_grad_ptr;
                        host_ptr_arr[i * 5 + 4] = data_arr[i].dst_grad_ptr;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_backward_pair_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::backward_pair(data_arr[i].lhs_logit_ptr, data_arr[i].lhs_grad_ptr,
                                                                                                                    data_arr[i].rhs_logit_ptr, data_arr[i].rhs_grad_ptr,
                                                                                                                    data_arr[i].dst_grad_ptr, data_arr[i].dispatch_control,
                                                                                                                    convert_grad_status_to_host_write_option(e.lhs_grad_status),
                                                                                                                    convert_grad_status_to_host_write_option(e.rhs_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_arr.get(), std::next(host_ptr_arr.get(), host_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except + optimizables
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            //TODOs: word_size memcmp + has_unique_object_representations_v 
            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t lhs_lck_addr;
                uma_ptr_t rhs_lck_addr;
                device_id_t dst_grad_vd_id;
                device_id_t lhs_logit_vd_id;
                device_id_t lhs_grad_vd_id;
                device_id_t rhs_logit_vd_id;
                device_id_t rhs_grad_vd_id;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, lhs_lck_addr, rhs_lck_addr, dst_grad_vd_id, lhs_logit_vd_id, lhs_grad_vd_id, rhs_logit_vd_id, rhs_grad_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, lhs_lck_addr, rhs_lck_addr, dst_grad_vd_id, lhs_logit_vd_id, lhs_grad_vd_id, rhs_logit_vd_id, rhs_grad_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t lhs;
                uma_ptr_t rhs;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.lhs_lck_addr, key.rhs_lck_addr);

                    auto umamap_reacquirer                              = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 5u>{}));
                    auto lhs_logit_vmamap_reacquirer                    = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto lhs_grad_vmamap_reacquirer                     = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto rhs_logit_vmamap_reacquirer                    = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto rhs_grad_vmamap_reacquirer                     = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto dst_grad_vmamap_reacquirer                     = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                              = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer                     = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer));
                    auto cuda_internal_resolutor                        = InternalCudaResolutor{};
                    cuda_internal_resolutor.async_device                = this->cuda_async_device;
                    cuda_internal_resolutor.synchronizer                = &cuda_synchronizer;
                    cuda_internal_resolutor.restrict_synchronizer       = &cuda_restrict_synchronizer;
                    cuda_internal_resolutor.allocator                   = this->allocator;

                    auto host_synchronizer                              = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer                     = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto host_internal_resolutor                        = InternalHostResolutor{};
                    host_internal_resolutor.async_device                = this->host_async_device;  
                    host_internal_resolutor.synchronizer                = &host_synchronizer;
                    host_internal_resolutor.restrict_synchronizer       = &host_restrict_synchronizer;
                    host_internal_resolutor.allocator                   = this->allocator;

                    size_t trimmed_cuda_vectorization_sz                = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost                          = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle                           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_host_vectorization_sz                = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost                          = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_internal_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle                           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_internal_resolutor, trimmed_host_vectorization_sz, hdh_mem.get()));

                    size_t trimmed_cuda_mixed_vectorization_sz          = std::min(this->vectorization_sz, sz);
                    size_t vmdh_allocation_cost                         = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_internal_resolutor, trimmed_cuda_mixed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vmdh_mem(vmdh_allocation_cost);
                    auto cuda_mixed_delivery_handle                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&cuda_internal_resolutor, trimmed_cuda_mixed_vectorization_sz, vmdh_mem.get()));

                    size_t trimmed_host_mixed_vectorization_sz          = std::min(this->vectorization_sz, sz);
                    size_t hmdh_allocation_cost                         = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_internal_resolutor, trimmed_host_mixed_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hmdh_mem(hmdh_allocation_cost);
                    auto host_mixed_delivery_handle                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_internal_resolutor, trimmed_host_mixed_vectorization_sz, hmdh_mem.get())); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, lhs, rhs, expected_ops_id]       = std::make_tuple(data_arr[i].dst, data_arr[i].lhs, data_arr[i].rhs, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_lhs                           = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(dst);
                        uma_ptr_t dst_rhs                           = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id           = dg::network_tile_member_getsetter::get_pair_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_bwd_operatable_id       = dg::network_tile_member_getsetter::get_pair_operatable_backward_id_nothrow(dst);
                        dispatch_control_t dispatch_control         = dg::network_tile_member_getsetter::get_pair_backward_dispatch_control_nothrow(dst);
                        init_status_t dst_init_status               = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(dst);
                        grad_status_t dst_grad_status               = dg::network_tile_member_getsetter::get_pair_grad_status_nothrow(dst);
                        uma_ptr_t dst_grad_umaptr                   = dg::network_tile_member_getsetter::get_pair_grad_addr_nothrow(dst);
                        dispatch_major_t dst_dispatch_major         = dg::network_tile_member_getsetter::get_pair_dispatch_major_nothrow(dst);

                        std::expected<operatable_id_t, exception_t> lhs_bwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_backward_id(lhs);
                        std::expected<uma_ptr_t, exception_t> lhs_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(lhs);
                        std::expected<uma_ptr_t, exception_t> lhs_grad_umaptr               = dg::network_tile_member_getsetter::get_tile_grad_addr(lhs);
                        std::expected<init_status_t, exception_t> lhs_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(lhs);
                        std::expected<grad_status_t, exception_t> lhs_grad_status           = dg::network_tile_member_getsetter::get_tile_grad_status(lhs);

                        std::expected<operatable_id_t, exception_t> rhs_bwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_backward_id(rhs);
                        std::expected<uma_ptr_t, exception_t> rhs_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(rhs);
                        std::expected<uma_ptr_t, exception_t> rhs_grad_umaptr               = dg::network_tile_member_getsetter::get_tile_grad_addr(rhs);
                        std::expected<init_status_t, exception_t> rhs_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(rhs);
                        std::expected<grad_status_t, exception_t> rhs_grad_status           = dg::network_tile_member_getsetter::get_tile_grad_status(rhs);

                        if (!lhs_bwd_operatable_id.has_value() || !lhs_logit_umaptr.has_value() || !lhs_grad_umaptr.has_value() || !lhs_init_status.has_value() || !lhs_grad_status.has_value()
                            || rhs_bwd_operatable_id.has_value() || !rhs_logit_umaptr.has_value() || !rhs_grad_umaptr.has_value() || !rhs_init_status.has_value() || !rhs_grad_status.has_value()){

                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (lhs_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (rhs_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_lhs != lhs){
                            continue;
                        }

                        if (dst_rhs != rhs){
                            continue;
                        }

                        if (dst_bwd_operatable_id != lhs_bwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_bwd_operatable_id != rhs_bwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_pair_backward_control(dispatch_control));

                        if (dispatch_info.lhs_logit_vd_id != key.lhs_logit_vd_id){
                            continue;
                        }

                        if (dispatch_info.lhs_grad_vd_id != key.lhs_grad_vd_id){
                            continue;
                        }

                        if (dispatch_info.rhs_logit_vd_id != key.rhs_logit_vd_id){
                            continue;
                        }

                        if (dispatch_info.rhs_grad_vd_id != key.rhs_grad_vd_id){
                            continue;
                        }

                        if (dispatch_info.dst_grad_vd_id != key.dst_grad_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_reacquire_nothrow(umamap_reacquirer, {{lhs_logit_umaptr.value(), dispatch_info.lhs_logit_vd_id}, 
                                                                                                 {lhs_grad_umaptr.value(), dispatch_info.lhs_grad_vd_id}, 
                                                                                                 {rhs_logit_umaptr.value(), dispatch_info.rhs_logit_vd_id}, 
                                                                                                 {rhs_grad_umaptr.value(), dispatch_info.rhs_grad_vd_id}, 
                                                                                                 {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}});

                        vma_ptr_t lhs_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t lhs_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t rhs_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});
                        vma_ptr_t rhs_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 3u>{});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 4u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(lhs_logit_vmamap_reacquirer, lhs_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(lhs_grad_vmamap_reacquirer, lhs_grad_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(rhs_logit_vmamap_reacquirer, rhs_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(rhs_grad_vmamap_reacquirer, rhs_grad_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg             = CudaResolutorArgument{};
                            cuda_resolutor_arg.lhs_logit_ptr    = dg::network_vmamap::get_cuda_ptr(lhs_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.lhs_grad_ptr     = dg::network_vmamap::get_cuda_ptr(lhs_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.lhs_grad_status  = lhs_grad_status.value();
                            cuda_resolutor_arg.rhs_logit_ptr    = dg::network_vmamap::get_cuda_ptr(rhs_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.rhs_grad_ptr     = dg::network_vmamap::get_cuda_ptr(rhs_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.rhs_grad_status  = rhs_grad_status.value();
                            cuda_resolutor_arg.dst_grad_ptr     = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.dispatch_control = dispatch_info.tileops_cuda_dispatch_control;

                            //compiler's hint
                            cuda_ptr_t cuda_dispatch_buf[3];
                            cuda_dispatch_buf[0]                = cuda_resolutor_arg.rhs_logit_ptr;
                            cuda_dispatch_buf[1]                = cuda_resolutor_arg.lhs_logit_ptr;
                            cuda_dispatch_buf[2]                = dg::pointer_limits<cuda_ptr_t>::null_value();

                            if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_RIGHT){
                                dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_dispatch_buf[0], cuda_resolutor_arg);
                            } else if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_LEFT){
                                dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_dispatch_buf[1], cuda_resolutor_arg);
                            } else if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_MIXED){
                                dg::network_producer_consumer::delvrsrv_deliver(cuda_mixed_delivery_handle.get(), cuda_dispatch_buf[2], cuda_resolutor_arg);
                            } else{
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
                                }
                            }
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.lhs_logit_ptr    = dg::network_vmamap::get_host_ptr(lhs_logit_vmamap_reacquirer);
                            host_resolutor_arg.lhs_grad_ptr     = dg::network_vmamap::get_host_ptr(lhs_grad_vmamap_reacquirer);
                            host_resolutor_arg.lhs_grad_status  = lhs_grad_status.value();
                            host_resolutor_arg.rhs_logit_ptr    = dg::network_vmamap::get_host_ptr(rhs_logit_vmamap_reacquirer);
                            host_resolutor_arg.rhs_grad_ptr     = dg::network_vmamap::get_host_ptr(rhs_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.rhs_grad_status  = rhs_grad_status.value();
                            host_resolutor_arg.dst_grad_ptr     = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            //compiler's hint
                            host_ptr_t host_dispatch_buf[3];
                            host_dispatch_buf[0]                = host_resolutor_arg.rhs_logit_ptr;
                            host_dispatch_buf[1]                = host_resolutor_arg.lhs_logit_ptr;
                            host_dispatch_buf[2]                = dg::pointer_limits<host_ptr_t>::null_value(); 

                            if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_RIGHT){
                                dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_dispatch_buf[0], host_resolutor_arg);
                            } else if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_LEFT){
                                dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_dispatch_buf[1], host_resolutor_arg);
                            } else if (dst_dispatch_major == PAIR_DISPATCH_MAJOR_MIXED){
                                dg::network_producer_consumer::delvrsrv_deliver(host_mixed_delivery_handle.get(), host_dispatch_buf[2], host_resolutor_arg);
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

                        dg::network_tile_member_getsetter::set_pair_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        exception_t lhs_grad_set_err = dg::network_tile_member_getsetter::set_tile_grad_status(lhs, TILE_GRAD_STATUS_HAS_VALUE);

                        if (dg::network_exception::is_failed(lhs_grad_set_err)){
                            (void) lhs_grad_set_err;
                            // dg::network_log_stackdump::error_fast(dg::network_exception::verbose(lhs_grad_set_err));
                        }

                        exception_t rhs_grad_set_err = dg::network_tile_member_getsetter::set_tile_grad_status(rhs, TILE_GRAD_STATUS_HAS_VALUE);

                        if (dg::network_exception::is_failed(rhs_grad_set_err)){
                            (void) rhs_grad_set_err;
                            // dg::network_log_stackdump::error_fast(dg::network_exception::verbose(rhs_grad_set_err));
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(lhs, expected_ops_id)));
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(rhs, expected_ops_id)));
                    }
                }
            };         
    };

    class BackwardDoUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t backward_vectorization_sz;

        public:

            BackwardDoUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                          std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                          size_t request_delivery_capacity,
                                          size_t radxfetch_vectorization_sz,
                                          size_t region_vectorization_sz,
                                          size_t backward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                      cuda_async_device(std::move(cuda_async_device)),
                                                                                      host_async_device(std::move(host_async_device)),
                                                                                      request_delivery_capacity(request_delivery_capacity),
                                                                                      radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                      region_vectorization_sz(region_vectorization_sz),
                                                                                      backward_vectorization_sz(backward_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                constexpr size_t LCK_ADDR_SZ_PER_DISPATCH   = UACM_ACM_SZ + 1u;

                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> descendant_arr(sz * UACM_ACM_SZ);
                dg::networK_stack_allocation::NoExceptAllocation<bool[]> validation_arr(sz);
                std::fill(validation_arr.get(), std::next(validation_arr.get(), sz), false);
                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> lck_addr_arr(sz * LCK_ADDR_SZ_PER_DISPATCH); 

                const size_t EVENT_SCALE_FACTOR             = UACM_ACM_SZ;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    InternalDescendantAddressFetcher fetcher    = {};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            std::expected<uma_ptr_t, exception_t> ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr                  = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr                  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg                      = AddressFetchArgument{};
                        fetch_arg.root                      = event_arr[i].dst;
                        fetch_arg.expected_ops_id           = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr             = std::next(descendant_arr.get(), i * UACM_ACM_SZ);
                        fetch_arg.fetching_validation_addr  = std::next(validation_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.vectorization_sz         = this->backward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!validation_arr[i]){
                            continue;
                        }

                        size_t lck_region_sz            = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t * cur_descendant_arr  = std::next(descendant_arr.get(), i * UACM_ACM_SZ);
                        uma_ptr_t * cur_lck_addr        = std::next(lck_addr_arr.get(), i * LCK_ADDR_SZ_PER_DISPATCH);
                        bool validation_flag            = true;
                        cur_lck_addr[0u]                = dg::memult::region(dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(event_arr[i].dst), lck_region_sz);

                        for (size_t j = 0u; j < UACM_ACM_SZ; ++j){
                            std::expected<uma_ptr_t, exception_t> rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(cur_descendant_arr[j]);

                            if (!rcu_addr.has_value()){
                                validation_flag = false;
                                break;
                            }

                            cur_lck_addr[j + 1] = dg::memult::region(rcu_addr.value(), lck_region_sz);
                        }

                        if (!validation_flag){
                            continue;
                        }

                        auto key                        = dg::vector_view<uma_ptr_t, LCK_ADDR_SZ_PER_DISPATCH>(cur_lck_addr);
                        auto resolutor_arg              = ResolutorArgument{};
                        resolutor_arg.root              = event_arr[i].dst;
                        resolutor_arg.expected_ops_id   = event_arr[i].operatable_id;
                        resolutor_arg.descendant        = cur_descendant_arr;  

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, resolutor_arg);
                    }
                }
            }

        private:

            struct AddressFetchArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                uma_ptr_t * fetching_addr;
                bool * fetching_validation_addr;
            };

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, AddressFetchArgument>{

                void push(uma_ptr_t rcu_addr, AddressFetchArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_uacm_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (data_arr[i].expected_ops_id == current_ops_id){
                                    *data_arr[i].fetching_validation_addr = true;
                                    dg::network_tile_member_getsetter::get_uacm_descendant_nothrow(data_arr[i].root, data_arr[i].fetching_addr);
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

            struct ResolutorArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                uma_ptr_t * left_descendant;
                uma_ptr_t * right_descendant;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<, ResolutorArgument>{

            };
    };

    class BackwardDoPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t backward_vectorization_sz;

        public:

            BackwardDoPACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                          std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                          size_t request_delivery_capacity,
                                          size_t radxfetch_vectorization_sz,
                                          size_t region_vectorization_sz,
                                          size_t backward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                      cuda_async_device(std::move(cuda_async_device)),
                                                                                      host_async_device(std::move(host_async_device)),
                                                                                      request_delivery_capacity(request_delivery_capacity),
                                                                                      radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                      region_vectorization_sz(region_vectorization_sz),
                                                                                      backward_vectorization_sz(backward_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                constexpr size_t LCK_ADDR_SZ_PER_DISPATCH       = PACM_ACM_SZ + PACM_ACM_SZ + 1u;

                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> left_descendant_arr(sz * PACM_ACM_SZ);
                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> right_descendant_arr(sz * PACM_ACM_SZ);
                dg::network_stack_allocation::NoExceptAllocation<bool[]> validation_arr(sz);
                std::fill(validation_arr.get(), std::next(validation_arr.get(), sz), false);
                dg::network_stack_allocation::NoExceptAllocation<uma_ptr_t[]> lck_addr_arr(sz * LCK_ADDR_SZ_PER_DISPATCH);

                const size_t EVENT_SCALE_FACTOR                 = PACM_ACM_SZ + PACM_ACM_SZ;
                size_t max_possible_event_sz                    = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity        = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    InternalDescendantAddressFetcher fetcher    = {};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            std::expected<uma_ptr_t, exception_t> ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr                  = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr                  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg                      = AddressFetchArgument{};
                        fetch_arg.root                      = event_arr[i].dst;
                        fetch_arg.expected_ops_id           = event_arr[i].operatable_id;
                        fetch_arg.fetching_lhs_addr         = std::next(left_descendant_arr.get(), i * PACM_ACM_SZ);
                        fetch_arg.fetching_rhs_addr         = std::next(right_descendant_arr.get(), i * PACM_ACM_SZ);
                        fetch_arg.fetching_validation_addr  = std::next(validation_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.vectorization_sz         = this->backward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!validation_arr[i]){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t * lck_addr_ptr            = std::next(lck_addr_arr.get(), i * LCK_ADDR_SZ_PER_DISPATCH);
                        lck_addr_ptr[0u]                    = dg::memult::region(dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(event_arr[i].dst), lck_region_sz); 
                        bool validation_flag                = true;
                        uma_ptr_t * left_descendant_ptr     = std::next(left_descendant_arr.get(), i * PACM_ACM_SZ);
                        uma_ptr_t * right_descendant_ptr    = std::next(right_descendant_arr.get(), i * PACM_ACM_SZ);

                        for (size_t j = 0u; j < PACM_ACM_SZ; ++j){
                            std::expected<uma_ptr_t, exception_t> lhs_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(left_descendant_ptr[j]);
                            std::expected<uma_ptr_t, exception_t> rhs_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(right_descendant_ptr[j]);

                            if (!lhs_rcu_addr.has_value() || !rhs_rcu_addr.has_value()){
                                validation_flag = false;
                                break;
                            }

                            lck_addr_ptr[j * 2 + 1] = dg::memult::region(lhs_rcu_addr.value(), lck_region_sz);
                            lck_addr_ptr[j * 2 + 2] = dg::memult::region(rhs_rcu_addr.value(), lck_region_sz);
                        }

                        if (!validation_flag){
                            continue;
                        }

                        auto key                        = dg::vector_view<uma_ptr_t, LCK_ADDR_SZ_PER_DISPATCH>(lck_addr_ptr);
                        auto resolutor_arg              = ResolutorArgument{};
                        resolutor_arg.root              = event_arr[i].dst;
                        resolutor_arg.expected_ops_id   = event_arr[i].operatable_id;
                        resolutor_arg.left_descendant   = left_descendant_ptr;
                        resolutor_arg.right_descendant  = right_descendant_ptr;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, resolutor_arg);
                    }
                }
            }

        private:

            struct AddressFetchArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                uma_ptr_t * fetching_lhs_addr;
                uma_ptr_t * fetching_rhs_addr;
                bool * fetching_validation_addr;
            };

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, AddressFetchArgument>{

                void push(uma_ptr_t rcu_addr, AddressFetchArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_pacm_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    *data_arr[i].fetching_validation_addr = true;
                                    dg::network_tile_member_getsetter::get_pacm_left_descendant_nothrow(data_arr[i].root, data_arr[i].fetching_lhs_addr);
                                    dg::network_tile_member_getsetter::get_pacm_right_descendant_nothrow(data_arr[i].root, data_arr[i].fetching_rhs_addr);
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

            struct ResolutorArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                uma_ptr_t * left_descendant;
                uma_ptr_t * right_descendant;
            };
    };

    //clear
    class BackwardDoExtnSrcSignalResolutorV2: public virtual dg::network_produer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t backward_vectorization_sz;

        public:

            BackwardDoExtnSrcSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                               std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                               size_t request_delivery_capacity,
                                               size_t radxfetch_vectorization_sz,
                                               size_t region_vectorization_sz,
                                               size_t backward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                           cuda_async_device(std::move(cuda_async_device)),
                                                                                           host_async_device(std::move(host_async_device)),
                                                                                           request_delivery_capacity(request_delivery_capacity),
                                                                                           radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                           region_vectorization_sz(region_vectorization_sz),
                                                                                           backward_vectorization_sz(backward_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<DispatchRadixArgument>[]> dispatch_radix_arg_arr(sz);

                // auto arena_allocator                    = {};

                const size_t EVENT_SCALE_FACTOR             = 1u;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    auto fetcher                                = InternalDispatchRadixFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = AddressFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(dispatch_radix_arg_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.vectorization_sz         = this->backward_vectorization_sz;
                    internal_resolutor.allocator                = &arena_allocator;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!dispatch_radix_arg_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr(dispatch_radix_arg_arr[i]->src);

                        if (!src_rcu_addr.has_value()){
                            continue;                            
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(event_arr[i].dst); 

                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        resolutor_key_arg.src_grad_vd_id    = dispatch_radix_arg_arr[i]->src_grad_vd_id;
                        resolutor_key_arg.src_logit_vd_id   = dispatch_radix_arg_arr[i]->src_logit_vd_id;
                        resolutor_key_arg.dst_grad_vd_id    = dispatch_radix_arg_arr[i]->dst_grad_vd_id;

                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.src               = dispatch_radix_arg_arr[i]->src;
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }
                }
            }

        private:

            struct DispatchRadixArgument{
                uma_ptr_t src;
                device_id_t src_grad_vd_id;
                device_id_t src_logit_vd_id;
                device_id_t dst_grad_vd_id;
            };

            struct RadixFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<DispatchRadixArgument> * fetching_addr;
            };

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, RadixFetcherArgument>{

                void push(uma_ptr_t rcu_addr, AddressFetcherArgument * data_arr, size_t sz){

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_extnsrc_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    auto dispatch_radix             = DispatchRadixArgument{};
                                    auto dispatch_control           = dg::network_tile_member_getsetter::get_extnsrc_backward_dispatch_control_nothrow(data_arr[i].root); 
                                    dispatch_radix.src              = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(data_arr[i].root);
                                    auto dispatch_info              = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_extnsrc_backward_dispatch(dispatch_control));
                                    dispatch_radix.src_grad_vd_id   = dispatch_info.src_grad_vd_id;
                                    dispatch_radix.src_logit_vd_id  = dispatch_info.src_logit_vd_id;
                                    dispatch_radix.dst_grad_vd_id   = dispatch_info.dst_grad_vd_id;

                                    *data_arr[i].fetching_addr      = dispatch_radix;
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

            struct CudaResolutorArgument{
                cuda_ptr_t src_grad_ptr;
                cuda_ptr_t src_logit_ptr;
                cuda_ptr_t dst_grad_ptr;
                cuda_tileops_dispatch_control_t dispatch_control;
                grad_status_t src_grad_status;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_arr_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_arr(cuda_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf)); 

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_arr[i * 3]     = data_arr[i].src_grad_ptr;
                        cuda_ptr_arr[i * 3 + 1] = data_arr[i].src_logit_ptr;
                        cuda_ptr_arr[i * 3 + 2] = data_arr[i].dst_grad_ptr;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::backward_mono(e.src_grad_ptr, e.src_logit_ptr, e.dst_grad_ptr, 
                                                                                                                    e.dispatch_control, convert_grad_status_to_cuda_write_option(e.src_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo))); 
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_arr.get(), std::next(cuda_ptr_arr.get(), cuda_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity));
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t src_grad_ptr;
                host_ptr_t src_logit_ptr;
                host_ptr_t dst_grad_ptr;
                host_tileops_dispatch_control_t dispatch_control;
                grad_status_t src_grad_status;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_arr_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_arr(host_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_arr[i * 3]     = data_arr[i].src_grad_ptr;
                        host_ptr_arr[i * 3 + 1] = data_arr[i].src_logit_ptr;
                        host_ptr_arr[i * 3 + 2] = data_arr[i].dst_grad_ptr;
                        total_complexity        += dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::decode_mono_dispatch_control(data_arr[i].dispatch_control)).runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::backward_mono(e.src_grad_ptr, e.src_logit_ptr, e.dst_grad_ptr, 
                                                                                                                    e.dispatch_control, convert_grad_status_to_host_write_option(e.src_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_arr.get(), std::next(host_ptr_arr.get(), host_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity));
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;
                device_id_t src_grad_vd_id;
                device_id_t src_logit_vd_id;
                device_id_t dst_grad_vd_id; 

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr, src_grad_vd_id, src_logit_vd_id, dst_grad_vd_id);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr, src_grad_vd_id, src_logit_vd_id, dst_grad_vd_id);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_cuda_controler::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr, key.ctrprt_lck_addr);

                    auto umamap_reacquirer                          = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{}));
                    auto src_logit_vmamap_reacquirer                = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_grad_vmamap_reacquirer                 = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize()); 
                    auto dst_grad_vmamap_reacquirer                 = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer)); 
                    auto cuda_internal_resolutor                    = InternalCudaResolutor{};
                    cuda_internal_resolutor.async_device            = this->cuda_async_device;
                    cuda_internal_resolutor.synchronizer            = &cuda_synchronizer;
                    cuda_internal_resolutor.restrict_synchronizer   = &cuda_restrict_synchronizer;
                    cuda_internal_resolutor.allocator               = this->allocator;

                    auto host_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto host_internal_resolutor                    = InternalHostResolutor{};
                    host_internal_resolutor.async_device            = this->host_async_device;
                    host_internal_resolutor.synchronizer            = &host_synchronizer;
                    host_internal_resolutor.restrict_synchronizer   = &host_restrict_synchronizer;
                    host_internal_resolutor.allocator               = this->allocator;

                    size_t trimmed_cuda_vectorization_sz            = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_host_vectorization_sz            = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_internal_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_internal_resolutor, trimmed_host_vectorization_sz, hdh_mem.get())); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src expected_ops_id]         = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);

                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_extnsrc_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_bwd_operatable_id   = dg::network_tile_member_getsetter::get_extnsrc_operatable_backward_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_extnsrc_backward_dispatch_control_nothrow(dst);
                        uma_ptr_t dst_grad_umaptr               = dg::network_tile_member_getsetter::get_extnsrc_grad_addr_nothrow(dst);
                        grad_status_t dst_grad_status           = dg::network_tile_member_getsetter::get_extnsrc_grad_status_nothrow(dst);

                        std::expected<operatable_id_t, exception_t> src_bwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_backward_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_grad_umaptr               = dg::network_tile_member_getsetter::get_tile_grad_addr(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);
                        std::expected<grad_status_t, exception_t> src_grad_status           = dg::network_tile_member_getsetter::get_tile_grad_status(src);

                        if (!src_bwd_operatable_id.has_value() || !src_init_status.has_value() || !src_grad_umaptr.has_value() 
                            || !src_logit_umaptr.has_value() || !src_grad_status.has_value()){

                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_bwd_operatable_id != src_bwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_extnsrc_backward_dispatch(dispatch_control));

                        if (dispatch_info.src_grad_vd_id != key.src_grad_vd_id){
                            continue;
                        }

                        if (dispatch_info.src_logit_vd_id != key.src_logit_vd_id){
                            continue;
                        }

                        if (dispatch_info.dst_grad_vd_id != key.dst_grad_vd_id){
                            continue;
                        }

                        dg::network_uma::region_reacquirer_reacquire_nothrow(umamap_reacquirer, {{src_grad_umaptr.value(), dispatch_info.src_grad_vd_id},
                                                                                                 {src_logit_umaptr.value(), dispatch_info.src_logit_vd_id},
                                                                                                 {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}});

                        vma_ptr_t src_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(src_grad_vmamap_reacquirer, src_grad_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg                 = CudaResolutorArgument{};
                            cuda_resolutor_arg.src_grad_ptr         = dg::network_vmamap::get_cuda_ptr(src_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.src_logit_ptr        = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.dst_grad_ptr         = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.dispatch_control     = dispatch_info.tileops_cuda_dispatch_control;
                            cuda_resolutor_arg.src_grad_status      = src_grad_status.value();

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_resolutor_arg.src_logit_ptr, cuda_resolutor_arg);                
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg                 = HostResolutorArgument{};
                            host_resolutor_arg.src_grad_ptr         = dg::network_vmamap::get_host_ptr(src_grad_vmamap_reacquirer);
                            host_resolutor_arg.src_logit_ptr        = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            host_resolutor_arg.dst_grad_ptr         = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);
                            host_resolutor_arg.dispatch_control     = dispatch_info.tileops_host_dispatch_control;
                            host_resolutor_arg.src_grad_status      = src_grad_status.value();

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_resolutor_arg.src_logit_ptr, host_resolutor_arg);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::INTERNAL_CORRUPTION);
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_extnsrc_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        exception_t src_gradstat_set_err = dg::network_tile_member_getsetter::set_tile_grad_status(src, TILE_GRAD_STATUS_HAS_VALUE);

                        if (dg::network_exception::is_failed(src_gradstat_set_err)){
                            (void) src_gradstat_set_err;
                            //
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src, expected_ops_id)));
                    }
                }
            };
    };

    //clear
    class BackwardDoExtnSrxSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        public:

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                (void) event_arr;
            }
    };

    //optimizables - not clear
    class BackwardDoExtnDstSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box; //alrights - we need to do acked request + log 
            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;

        public:

            BackwardDoExtnDstSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                               std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                               std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                               std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                               size_t request_delivery_capacity,
                                               size_t radxfetch_vectorization_sz,
                                               size_t region_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                         uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                         host_ip_retriever(std::move(host_ip_retriever)),
                                                                                         host_async_device(std::move(host_async_device)),
                                                                                         request_delivery_capacity(request_delivery_capacity),
                                                                                         radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                         region_vectorization_sz(region_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                const size_t EVENT_SCALE_FACTOR             = 1u;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    InternalResolutor internal_resolutor        = {};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.host_ip_retriever        = this->host_ip_retriever.get();
                    internal_resolutor.uma_ip_retriever         = this->uma_ip_retriever.get();

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get())); 

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            std::expected<uma_ptr_t, exception_t> ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size())); 
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t dst_lck_addr              = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), dst_lck_addr, std::make_tuple(event_arr[i].dst, event_arr[i].operatable_id));
                    }
                }
            }

        private:

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * request_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                HostIPRetrieverInterface * host_ip_retriever;
                UnifiedMemoryIPRetrieverInterface * uma_ip_retriever;

                void push(uma_ptr_t lck_addr, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(lck_addr);

                    auto umamap_reacquirer              = dg::network_exception_handler::nothrow_log(dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{}));
                    auto dst_logit_vmamap_reacquirer    = dg::network_exception_handler::nothrow_log(dg::network_vmamap::reacquirer_raii_initialize());
                    auto dst_grad_vmamap_reacquirer     = dg::network_exception_handler::nothrow_log(dg::network_vmamap::reacquirer_raii_initialize());
                    auto host_synchronizer              = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto request_vec                    = dg::vector<std::optional<Request<external_virtual_memory_event_t>>>(sz, std::optional<Request<external_virtual_memory_event_t>>(std::nullopt));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, expected_ops_id]         = std::make_tuple(data_arr[i].dst, data_arr[i].expected_ops_id);
                        init_status_t dst_init_status       = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(dst);
                        operatable_id_t dst_operatable_id   = dg::network_tile_member_getsetter::get_extndst_operatable_memevent_id_nothrow(dst);
                        grad_status_t dst_grad_status       = dg::network_tile_member_getsetter::get_extndst_grad_status_nothrow(dst);
                        uma_ptr_t dst_counterpart           = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr          = dg::network_tile_member_getsetter::get_extndst_logit_addr_nothrow(dst);
                        uma_ptr_t dst_grad_umaptr           = dg::network_tile_member_getsetter::get_extndst_grad_addr_nothrow(dst);
                        dispatch_control_t dispatch_control = dg::network_tile_member_getsetter::get_extndst_backward_dispatch_control_nothrow(dst); 
                        size_t dst_logit_byte_sz            = dg::network_tile_member_getsetter::get_extndst_logit_byte_size_nothrow(dst);
                        size_t dst_grad_byte_sz             = dg::network_tile_member_getsetter::get_extndst_grad_byte_size_nothrow(dst); 

                        std::expected<Address, exception_t> to_addr = this->uma_ip_retriever->ip(dst_counterpart);

                        if (!to_addr.has_value()){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto dispatch_info  = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_extndst_backward_dispatch(dispatch_control)); 

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dispatch_info.dst_logit_vd_id}, 
                                                                                                              {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}})){

                            host_synchronizer.sync();
                        }

                        dg::network_uma::reacquirer_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dispatch_info.dst_logit_vd_id}, 
                                                                                          {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}});

                        vma_ptr_t dst_logit_vmaptr  = dg::network_vmamap::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_vmamap::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_logit_vmamap_reacquirer, dst_logit_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(dst_grad_vmamap_reacquirer, dst_grad_vmaptr)){

                            host_synchronizer.sync();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            host_ptr_t dst_logit_hostptr    = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            host_ptr_t dst_grad_hostptr     = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);
                            size_t total_complexity         = dst_logit_byte_sz + dst_grad_byte_sz + dst_grad_byte_sz; //
                            auto request_ptr                = std::next(request_vec.begin(), i);
                            *request_ptr                    = Request<external_virtual_memory_event_t>{};
                            request_ptr->value().fr         = this->host_ip_retriever->ip();
                            request_ptr->value().to         = to_addr.value();

                            auto executable                 = [dst, dst_logit_hostptr, dst_grad_hostptr, 
                                                               dst_logit_byte_sz, dst_grad_byte_sz, 
                                                               expected_ops_id, request_ptr]() noexcept{

                                ExtnDstTile tile_data           = {};
                                dg::network_tile_member_getsetter::burn_extndst_metadata_nothrow(dst, tile_data);
                                tile_data.logit_value           = dg::string(dst_logit_byte_sz, ' '); //TODOs: optimizables 
                                tile_data.grad_value            = dg::string(dst_grad_byte_sz, ' '); //TODOs: optimizables 
                                host_ptr_t cpydst_logit_hostptr = tile_data.logit_value.data();
                                host_ptr_t cpydst_grad_hostptr  = tile_data.grad_value.data();

                                dg::network_exception_handler::nothrow_log(dg::network_memops_clib::memcpy_host_to_host(cpydst_logit_hostptr, dst_logit_hostptr, dst_logit_byte_sz)); //
                                dg::network_exception_handler::nothrow_log(dg::network_memops_clib::memcpy_host_to_host(cpydst_grad_hostptr, dst_grad_hostptr, dst_grad_byte_sz)); //
                                dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::zero_grad(dst_grad_hostptr, dst_grad_byte_sz));

                                auto inject_event               = dg::network_exception_handler::nothrow_log(dg::network_external_memcommit_factory::make_event_shadow_injection(tile_data.backward_shadow, TILE_KIND_EXTNDSX, dg::network_compact_serializer::serialize<dg::string>(tile_data))); //
                                auto signal_event               = dg::network_exception_handler::nothrow_log(dg::network_external_memcommit_factory::make_event_backward_do_signal(tile_data.backward_shadow, expected_ops_id));
                                auto sequential_event           = dg::network_exception_handler::nothrow_log(dg::network_external_memcommit_factory::make_sequential_event(std::move(inject_event), std::move(signal_event))); 
                                request_ptr->value().content    = std::move(sequential_event); 
                            };

                            auto async_task     = dg::network_host_asynchronous::virtualize_async_task(std::move(executable)); //TODOs: optimizables
                            auto synchronizable = dg::network_exception_handler::nothrow_log(this->host_async_device->exec(std::move(async_task), total_complexity)); //TODOs: except
                            dg::network_exception_handler::nothrow_log(host_synchronizer.add(std::move(synchronizable)));
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else[
                                std::unreachable();
                            ]
                        }

                        dg::network_tile_member_getsetter::set_extndst_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED); //guarantees pointer restriction
                    }

                    host_synchronizer.sync();

                    for (size_t i = 0u; i < sz; ++i){
                        if (!request_vec[i].has_value()){
                            continue;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(request_vec[i].value()));
                    }
                }
            };
    }; 

    class BackwardDoExtnDsxSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t backward_vectorization_sz;
        
        public:

            BackwardDoExtnDsxSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                             std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                             std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                             size_t request_delivery_capacity,
                                             size_t radxfetch_vectorization_sz,
                                             size_t region_vectorization_sz,
                                             size_t backward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                         host_async_device(std::move(host_async_device)),
                                                                                         cuda_async_device(std::move(cuda_async_device)),
                                                                                         request_delivery_capacity(request_delivery_capacity),
                                                                                         radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                         region_vectorization_sz(region_vectorization_sz),
                                                                                         backward_vectorization_sz(backward_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

            }
    };

    class BackwardDoCritSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t backward_vectorization_sz;

        public:

            BackwardDoCritSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                          std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                          std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                          size_t request_delivery_capacity,
                                          size_t radxfetch_vectorization_sz,
                                          size_t region_vectorization_sz,
                                          size_t backward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                      cuda_async_device(std::move(cuda_async_device)),
                                                                                      host_async_device(std::move(host_async_device)),
                                                                                      request_delivery_capacity(request_delivery_capacity),
                                                                                      radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                      region_vectorization_sz(region_vectorization_sz),
                                                                                      backward_vectorization_sz(backward_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<uma_ptr_t>[]> descendant_arr(sz);

                // auto arena_allocator                        = ;

                const size_t EVENT_SCALE_FACTOR             = 1u;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz);
                size_t dh_allocation_cost                   = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> dh_mem(dh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, dh_mem.get()));

                {
                    auto fetcher                                = InternalDescendantAddressFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = AddressFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(descendant_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.vectorization_sz         = this->backward_vectorization_sz;
                    internal_resolutor.allocator                = &arena_allocator;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(descendant_arr[i].value());

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz            = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr          = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(event_arr[i].dst);
                        auto resolutor_key              = ResolutorKeyArgument{};
                        resolutor_key.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        auto resolutor_val              = ResolutorValueArgument{};
                        resolutor_val.dst               = event_arr[i].dst;
                        resolutor_val.src               = descendant_arr[i].value();
                        resolutor_val.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key, resolutor_val);
                    }
                }
            }

        private:

            struct AddressFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<uma_ptr_t> * fetching_addr;
            };

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, AddressFetcherArgument>{

                void push(uma_ptr_t lck_addr, AddressFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(lck_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_crit_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    *data_arr[i].fetching_addr = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(data_arr[i].root);
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

            struct CudaResolutorArgument{
                cuda_ptr_t src_grad_ptr;
                cuda_ptr_t src_logit_ptr;
                cuda_ptr_t dst_grad_ptr;
                grad_status_t src_grad_status;
                cuda_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_arr_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_arr(cuda_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_arr[i * 3]     = data_arr[i].src_grad_ptr;
                        cuda_ptr_arr[i * 3 + 1] = data_arr[i].src_logit_ptr;
                        cuda_ptr_arr[i * 3 + 2] = data_arr[i].dst_grad_ptr;

                        dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::backward_crit_dispatch_control_chk(data_arr[i].dispatch_control));

                        total_complexity        += dg::network_tileops_cuda_poly::decode_backward_crit_dispatch_control(data_arr[i].dispatch_control)->runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::backward_crit(e.src_grad_ptr, e.src_logit_ptr, e.dst_grad_ptr, 
                                                                                                                    e.dispatch_control, convert_grad_status_to_cuda_write_option(e.src_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_arr.get(), std::next(cuda_ptr_arr.get(), cuda_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t src_grad_ptr;
                host_ptr_t src_logit_ptr;
                host_ptr_t dst_grad_ptr;
                grad_status_t src_grad_status;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchrinizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_arr_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_arr(host_ptr_arr_sz);
                    size_t total_complexity     = {}; 

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_buf)); 
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_arr[i * 3]     = data_arr[i].src_grad_ptr;
                        host_ptr_arr[i * 3 + 1] = data_arr[i].src_logit_ptr;
                        host_ptr_arr[i * 3 + 2] = data_arr[i].dst_grad_ptr;

                        dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::backward_crit_dispatch_control_chk(data_arr[i].dispatch_control));

                        total_complexity        += dg::network_tileops_host_poly::decode_backward_crit_dispatch_control(data_arr[i].dispatch_control)->runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::backward_crit(e.src_grad_ptr, e.src_logit_ptr, e.dst_grad_ptr, 
                                                                                                                    e.dispatch_control, convert_grad_status_to_host_write_option(e.src_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_arr.get(), std::next(host_ptr_arr.get(), host_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                          = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{}));
                    auto src_grad_vmamap_reacquirer                 = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_logit_vmamap_reacquirer                = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto dst_grad_vmamap_reacquirer                 = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer));
                    auto cuda_internal_resolutor                    = InternalCudaResolutor{};
                    cuda_internal_resolutor.async_device            = this->cuda_async_device;
                    cuda_internal_resolutor.synchronizer            = &cuda_synchronizer;
                    cuda_internal_resolutor.restrict_synchronizer   = &cuda_restrict_synchronizer;

                    auto host_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto host_internal_resolutor                    = InternalHostResolutor{};
                    host_internal_resolutor.async_device            = this->host_async_device;
                    host_internal_resolutor.synchronizer            = &host_synchronizer;
                    host_internal_resolutor.restrict_synchronizer   = &host_restrict_synchronizer;

                    size_t trimmed_cuda_vectorization_sz            = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_keyhint_preallocated_raiihandle(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz, cdh_mem.get()));

                    size_t trimmed_host_vectorization_sz            = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_internal_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvsrv_open_keyhint_preallocated_raiihandle(&host_internal_resolutor, trimmed_host_vectorization_sz, hdh_mem.get())); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(dst);
                        operatable_id_t dst_bwd_operatable_id   = dg::network_tile_member_getsetter::get_crit_operatable_backward_id_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_crit_operatable_memevent_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(dst);
                        uma_ptr_t dst_grad_umaptr               = dg::network_tile_member_getsetter::get_crit_grad_addr_nothrow(dst);
                        uma_ptr_t dst_grad_status               = dg::network_tile_member_getsetter::get_crit_grad_status_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_crit_backward_dispatch_control_nothrow(dst);
                        
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<operatable_id_t, exception_t> src_bwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_backward_id(src);
                        std::expected<uma_ptr_t, exception_t> src_grad_umaptr               = dg::network_tile_member_getsetter::get_tile_grad_addr(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);
                        std::expected<grad_status_t, exception_t> src_grad_status           = dg::network_tile_member_getsetter::get_tile_grad_status(src);

                        if (!src_init_status.has_value() || !src_bwd_operatable_id.has_value() || !src_grad_umaptr.has_value() 
                            || !src_logit_umaptr.has_value() || !src_grad_status.has_value()){

                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_bwd_operatable_id != src_bwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_crit_backward_dispatch(dispatch_control));

                        dg::network_umamap::region_reacquirer_reacquire_nothrow(umamap_reacquirer, {{src_grad_umaptr.value(), dispatch_info.src_grad_vd_id},
                                                                                                    {src_logit_umaptr.value(), dispatch_info.src_logit_vd_id},
                                                                                                    {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}});

                        uma_ptr_t src_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        uma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        uma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(src_grad_vmamap_reacquirer, src_grad_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg             = CudaResolutorArgument{};
                            cuda_resolutor_arg.src_grad_ptr     = dg::network_vmamap::get_cuda_ptr(src_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.src_logit_ptr    = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.dst_grad_ptr     = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.src_grad_status  = src_grad_status.value();
                            cuda_resolutor_arg.dispatch_control = dispatch_info.tileops_cuda_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_resolutor_arg.src_logit_ptr, cuda_resolutor_arg);
                        } else if (dg::dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.src_grad_ptr     = dg::network_vmamap::get_host_ptr(src_grad_vmamap_reacquirer);
                            host_resolutor_arg.src_logit_ptr    = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            host_resolutor_arg.dst_grad_ptr     = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);
                            host_resolutor_arg.src_grad_status  = src_grad_status.value();
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_resolutor_arg.src_logit_ptr, host_resolutor_arg);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_crit_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        exception_t descendant_set_arr = dg::network_tile_member_getsetter::set_tile_grad_status(src, TILE_GRAD_STATUS_HAS_VALUE);

                        if (dg::network_exception::is_failed(descendant_set_arr)){
                            (void) descendant_set_arr;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src, expected_ops_id)));
                    }
                }
            };
    };

    class BackwardDoMsgrFwdSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device;
            const size_t request_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t backward_vectorization_sz;  

        public:

            BackwardDoMsgrFwdSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                               std::shared_ptr<dg::network_cuda_controller::AsynchronousDeviceInterface> cuda_async_device,
                                               size_t request_delivery_capacity,
                                               size_t radxfetch_vectorization_sz,
                                               size_t region_vectorization_sz,
                                               size_t backward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                           host_async_device(std::move(host_async_device)),
                                                                                           cuda_async_device(std::move(cuda_async_device)),
                                                                                           request_delivery_capacity(request_delivery_capacity),
                                                                                           radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                           region_vectorization_sz(region_vectorization_sz),
                                                                                           backward_vectorization_sz(backward_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<uma_ptr_t>[]> descendant_arr(sz);

                // auto arena_allocator                        =

                const size_t EVENT_SCALE_FACTOR             = 1u;
                size_t max_possible_event_sz                = sz * EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_event_sz); 
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost); 
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                {
                    auto fetcher                                = InternalDescendantAddressFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = AddressFetcherArgument{};
                        fetch_arg.dst               = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(descendant_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                     = InternalResolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.host_async_device        = this->host_async_device.get();
                    internal_resolutor.cuda_async_device        = this->cuda_async_device.get();
                    internal_resolutor.vectorization_sz         = this->backward_vectorization_sz;
                    internal_resolutor.allocator                = &arena_allocator;

                    size_t trimmed_region_vectorization_sz      = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost); 
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(descendant_arr[i].value());

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(event_arr[i].dst);
                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.src               = descendant_arr[i].value();
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }
                }
            }

        private:

            struct AddressFetcherArgument{
                uma_ptr_t dst;
                operatable_id_t expected_ops_id;
                std::optional<uma_ptr_t> * fetching_addr;
            };

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, AddressFetcherArgument>{

                void push(uma_ptr_t rcu_addr, AddressFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrfwd_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    *data_arr[i].fetching_addr = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(data_arr[i].root); 
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

            struct CudaResolutorArgument{
                cuda_ptr_t src_grad_ptr;
                cuda_ptr_t src_logit_ptr;
                cuda_ptr_t dst_grad_ptr;
                grad_status_t src_grad_status;
                cuda_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_arr_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_arr(cuda_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_arr[i * 3]     = data_arr[i].src_grad_ptr;
                        cuda_ptr_arr[i * 3 + 1] = data_arr[i].src_logit_ptr;
                        cuda_ptr_arr[i * 3 + 2] = data_arr[i].dst_grad_ptr;

                        dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::backward_mono_dispatch_control_chk(data_arr[i].dispatch_control));

                        total_complexity        += dg::network_tileops_cuda_poly::decode_backward_mono_dispatch_control(data_arr[i].dispatch_control)->runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::backward_mono(e.src_grad_ptr, e.src_logit_ptr, e.dst_grad_ptr, 
                                                                                                                    e.dispatch_control, convert_grad_status_to_cuda_write_option(e.src_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_arr.get(), std::next(cuda_ptr_arr.get(), cuda_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t src_grad_ptr;
                host_ptr_t src_logit_ptr;
                host_ptr_t dst_grad_ptr;
                grad_status_t src_grad_status;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_arr_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_arr(host_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_arr[i * 3]     = data_arr[i].src_grad_ptr;
                        host_ptr_arr[i * 3 + 1] = data_arr[i].src_logit_ptr;
                        host_ptr_arr[i * 3 + 2] = data_arr[i].dst_grad_ptr;

                        dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::backward_mono_dispatch_control_chk(data_arr[i].dispatch_control));

                        total_complexity        += dg::network_tileops_host_poly::decode_backward_mono_dispatch_control(data_arr[i].dispatch_control)->runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::backward_mono(e.src_grad_ptr, e.src_logit_ptr, e.dst_grad_ptr, 
                                                                                                                    e.dispatch_control, convert_grad_status_to_host_write_option(e.src_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_arr.get(), std::next(host_ptr_arr.get(), host_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                          = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{}));
                    auto src_grad_vmamap_reacquirer                 = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_logit_vmamap_reacquirer                = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto dst_grad_vmamap_reacquirer                 = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer));
                    auto cuda_internal_resolutor                    = InternalCudaResolutor{};
                    cuda_internal_resolutor.async_device            = this->cuda_async_device;
                    cuda_internal_resolutor.synchronizer            = &cuda_synchronizer;
                    cuda_internal_resolutor.restrict_synchronizer   = &cuda_restrict_synchronizer;
                    cuda_internal_resolutor.allocator               = this->allocator;

                    auto host_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto host_internal_resolutor                    = InternalHostResolutor{};
                    host_internal_resolutor.async_device            = this->host_async_device;
                    host_internal_resolutor.synchronizer            = &host_synchronizer;
                    host_internal_resolutor.restrict_synchronizer   = &host_restrict_synchronizer; 
                    host_internal_resolutor.allocator               = this->allocator;

                    size_t trimmed_cuda_vectorization_sz            = std::min(this->vectorization_sz, sz);
                    size_t cdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cdh_mem(cdh_allocation_cost);
                    auto cuda_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&cuda_internal_resolutor, trimmed_cuda_vectorization_sz));

                    size_t trimmed_host_vectorization_sz            = std::min(this->vectorization_sz, sz);
                    size_t hdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_internal_resolutor, trimmed_host_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hdh_mem(hdh_allocation_cost);
                    auto host_delivery_handle                       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_internal_resolutor, trimmed_host_vectorization_sz)); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_msgrfwd_descendant_nothrow(dst);
                        operatable_id_t dst_bwd_operatable_id   = dg::network_tile_member_getsetter::get_msgrfwd_operatable_backward_id_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_msgrfwd_operatable_memevent_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(dst);
                        uma_ptr_t dst_grad_umaptr               = dg::network_tile_member_getsetter::get_msgrfwd_grad_addr_nothrow(dst);
                        grad_status_t dst_grad_status           = dg::network_tile_member_getsetter::get_msgrfwd_grad_status_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_msgrfwd_backward_dispatch_control_nothrow(dst);

                        std::expected<operatable_id_t, exception_t> src_bwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_backward_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<uma_ptr_t, exception_t> src_grad_umaptr               = dg::network_tile_member_getsetter::get_tile_grad_addr(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);
                        std::expected<grad_status_t, exception_t> src_grad_status           = dg::network_tile_member_getsetter::get_tile_grad_status(src);

                        if (!src_bwd_operatable_id.has_value() || !src_init_status.has_value() || !src_grad_umaptr.has_value() 
                            || !src_logit_umaptr.has_value() || !src_grad_status.has_value()){

                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_bwd_operatable_id != src_bwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_msgrfwd_backward_dispatch(dispatch_control));

                        dg::network_umamap::region_reacquirer_reacquire_nothrow(umamap_reacquirer, {{src_grad_umaptr.value(), dispatch_info.src_grad_vd_id}, 
                                                                                                    {src_logit_umaptr.value(), dispatch_info.src_logit_vd_id},
                                                                                                    {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}});

                        vma_ptr_t src_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(src_grad_vmamap_reacquirer, src_grad_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg             = CudaResolutorArgument{};
                            cuda_resolutor_arg.src_grad_ptr     = dg::network_vmamap::get_cuda_ptr(src_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.src_logit_ptr    = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.dst_grad_ptr     = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.src_grad_status  = src_grad_status.value();
                            cuda_resolutor_arg.dispatch_control = dispatch_info.tileops_cuda_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_delivery_handle.get(), cuda_resolutor_arg.src_logit_ptr, cuda_resolutor_arg);
                        } else if (dg::dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.src_grad_ptr     = dg::network_vmamap::get_host_ptr(src_grad_vmamap_reacquirer);
                            host_resolutor_arg.src_logit_ptr    = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            host_resolutor_arg.dst_grad_ptr     = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);
                            host_resolutor_arg.src_grad_status  = src_grad_status.value();
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_delivery_handle.get(), host_resolutor_arg.src_logit_ptr, host_resolutor_arg);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_msgrfwd_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        exception_t descendant_set_err = dg::network_tile_member_getsetter::set_tile_grad_status(src, TILE_GRAD_STATUS_HAS_VALUE);

                        if (dg::network_exception::is_failed(descendant_set_err)){
                            (void) descendant_set_err;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src, expected_ops_id)));
                    }
                }
            };
    };

    class BackwardDoMsgrBwdSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<BackwardDoSignalEvent>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box;
            const std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device;
            const size_t request_delivery_capacity;
            const size_t eu_packet_delivery_capacity;
            const size_t radxfetch_vectorization_sz;
            const size_t region_vectorization_sz;
            const size_t backward_vectorization_sz;

        public:

            BackwardDoMsgrBwdSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box,
                                               std::shared_ptr<dg::network_host_asynchronous::AsynchronousDeviceInterface> host_async_device,
                                               size_t request_delivery_capacity,
                                               size_t eu_packet_delivery_capacity,
                                               size_t radxfetch_vectorization_sz,
                                               size_t region_vectorization_sz,
                                               size_t backward_vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                           eu_packet_box(std::move(eu_packet_box)),
                                                                                           host_async_device(std::move(host_async_device)),
                                                                                           request_delivery_capacity(request_delivery_capacity),
                                                                                           eu_packet_delivery_capacity(eu_packet_delivery_capacity),
                                                                                           radxfetch_vectorization_sz(radxfetch_vectorization_sz),
                                                                                           region_vectorization_sz(region_vectorization_sz),
                                                                                           backward_vectorization_sz(backward_vectorization_sz){}

            void push(BackwardDoSignalEvent * event_arr, size_t sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<std::optional<uma_ptr_t>[]> descendant_arr(sz);

                // auto arena_allocator                        = {};

                const size_t REQUEST_EVENT_SCALE_FACTOR     = 1u;
                size_t max_possible_request_event_sz        = sz * REQUEST_EVENT_SCALE_FACTOR;
                size_t trimmed_request_delivery_capacity    = std::min(this->request_delivery_capacity, max_possible_request_event_sz);
                size_t rdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(this->request_box.get(), trimmed_request_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> rdh_mem(rdh_allocation_cost);
                auto request_delivery_handle                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->request_box.get(), trimmed_request_delivery_capacity, rdh_mem.get()));

                const size_t EUPACKET_EVENT_SCALE_FACTOR    = 1u;
                size_t max_possible_eu_packet_event_sz      = sz * EUPACKET_EVENT_SCALE_FACTOR;
                size_t trimmed_eu_packet_delivery_capacity  = std::min(this->eu_packet_delivery_capacity, max_possible_eu_packet_event_sz);
                size_t epdh_allocation_cost                 = dg::network_producer_consumer::delvrsrv_allocation_cost(this->eu_packet_box.get(), trimmed_eu_packet_delivery_capacity);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> epdh_mem(epdh_allocation_cost); 
                auto eu_packet_delivery_handle              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(this->eu_packet_box.get(), trimmed_eu_packet_delivery_capacity, epdh_mem.get()));

                {
                    auto fetcher                                = InternalDescendantAddressFetcher{};

                    size_t trimmed_radxfetch_vectorization_sz   = std::min(this->radxfetch_vectorization_sz, sz);
                    size_t vdh_allocation_cost                  = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&fetcher, trimmed_radxfetch_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_preallocated_raiihandle(&fetcher, trimmed_radxfetch_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if constexpr(DEBUG_MODE_FLAG){
                            auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(event_arr[i].dst);

                            if (!ptrchk.has_value()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(ptrchk.error()));
                                std::abort();
                            }
                        }

                        uma_ptr_t rcu_addr          = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(event_arr[i].dst);
                        uma_ptr_t lck_addr          = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        auto fetch_arg              = AddressFetcherArgument{};
                        fetch_arg.root              = event_arr[i].dst;
                        fetch_arg.expected_ops_id   = event_arr[i].operatable_id;
                        fetch_arg.fetching_addr     = std::next(descendant_arr.get(), i);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, fetch_arg);
                    }
                }

                {
                    auto internal_resolutor                         = InternalResolutor{};
                    internal_resolutor.request_delivery_handle      = request_delivery_handle.get();
                    internal_resolutor.eu_packet_delivery_handle    = eu_packet_delivery_handle.get();
                    internal_resolutor.host_async_device            = this->host_async_device.get();
                    internal_resolutor.allocator                    = &arena_allocator;
                    internal_resolutor.vectorization_sz             = this->backward_vectorization_sz;

                    size_t trimmed_region_vectorization_sz          = std::min(this->region_vectorization_sz, sz);
                    size_t vdh_allocation_cost                      = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_region_vectorization_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> vdh_mem(vdh_allocation_cost);
                    auto vectorized_delivery_handle                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, trimmed_region_vectorization_sz, vdh_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        std::expected<uma_ptr_t, exception_t> src_rcu_addr = dg::network_tile_member_getsetter::get_tile_rcu_addr(descendant_arr[i].value());

                        if (!src_rcu_addr.has_value()){
                            continue;
                        }

                        size_t lck_region_sz                = std::min(static_cast<size_t>(dg::network_memops_uma::memlock_region_size()), static_cast<size_t>(dg::network_uma::memregion_size()));
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(event_arr[i].dst);
                        auto resolutor_key_arg              = ResolutorKeyArgument{};
                        resolutor_key_arg.dst_lck_addr      = dg::memult::region(dst_rcu_addr, lck_region_sz);
                        resolutor_key_arg.src_lck_addr      = dg::memult::region(src_rcu_addr.value(), lck_region_sz);
                        auto resolutor_val_arg              = ResolutorValueArgument{};
                        resolutor_val_arg.dst               = event_arr[i].dst;
                        resolutor_val_arg.src               = descendant_arr[i].value();
                        resolutor_val_arg.expected_ops_id   = event_arr[i].operatable_id;

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), resolutor_key_arg, resolutor_val_arg);
                    }
                }
            }

        private:

            struct AddressFetcherArgument{
                uma_ptr_t root;
                operatable_id_t expected_ops_id;
                std::optional<uma_ptr_t> * fetching_addr;
            };

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVCosnumerInterface<uma_ptr_t, AddressFetcherArgument>{

                void push(uma_ptr_t rcu_addr, AddressFetcherArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        init_status_t init_status       = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(data_arr[i].root);
                        operatable_id_t current_ops_id  = dg::network_tile_member_getsetter::get_msgrbwd_operatable_memevent_id_nothrow(data_arr[i].root);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {   
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                if (current_ops_id == data_arr[i].expected_ops_id){
                                    *data_arr[i].fetching_addr = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(data_arr[i].root);
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

            struct CudaResolutorArgument{
                cuda_ptr_t src_grad_ptr;
                cuda_ptr_t src_logit_ptr;
                cuda_ptr_t dst_grad_ptr;
                grad_status_t src_grad_status;
                cuda_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalCudaResolutor: dg::network_producer_consumer::ConsumerInterface<CudaResolutorArgument>{

                dg::network_cuda_controller::AsynchronousDeviceInterface * async_device;
                dg::network_cuda_controller::Synchronizer * synchronizer;
                dg::network_cuda_controller::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(CudaResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t cuda_ptr_arr_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<cuda_ptr_t[]> cuda_ptr_arr(cuda_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_cuda_controller::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        cuda_ptr_arr[i * 3]     = data_arr[i].src_grad_ptr;
                        cuda_ptr_arr[i * 3 + 1] = data_arr[i].src_logit_ptr;
                        cuda_ptr_arr[i * 3 + 2] = data_arr[i].dst_grad_ptr;

                        dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::backward_mono_dispatch_control_chk(data_arr[i].dispatch_control));

                        total_complexity        += dg::network_tileops_cuda_poly::decode_backward_mono_dispatch_control(data_arr[i].dispatch_control)->runtime_complexity;
                        auto work_order         = [e = data_arr[i]](){
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_cuda_poly::backward_mono(e.src_grad_ptr, e.src_logit_ptr, e.dst_grad_ptr, 
                                                                                                                    e.dispatch_control, convert_grad_status_to_cuda_write_option(e.src_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_cuda_controller::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_cuda_controller::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(cuda_ptr_arr.get(), std::next(cuda_ptr_arr.get(), cuda_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct HostResolutorArgument{
                host_ptr_t src_grad_ptr;
                host_ptr_t src_logit_ptr;
                host_ptr_t dst_grad_ptr;
                grad_status_t src_grad_status;
                host_tileops_dispatch_control_t dispatch_control;
            };

            struct InternalHostResolutor: dg::network_producer_consumer::ConsumerInterface<HostResolutorArgument>{

                dg::network_host_asynchronous::AsynchronousDeviceInterface * async_device;
                dg::network_host_asynchronous::Synchronizer * synchronizer;
                dg::network_host_asynchronous::RestrictPointerSynchronizer * restrict_synchronizer;
                dg::network_allocation::ArenaAllocatorInterface * allocator;

                void push(HostResolutorArgument * data_arr, size_t sz) noexcept{

                    size_t host_ptr_arr_sz      = sz * 3;
                    dg::network_stack_allocation::NoExceptAllocation<host_ptr_t[]> host_ptr_arr(host_ptr_arr_sz);
                    size_t total_complexity     = {};

                    size_t virtual_wo_vec_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_workorder_sequential_container_size(sz);
                    char * virtual_wo_vec_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virtual_wo_vec_bsz));
                    auto virtual_wo_vec         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_workorder_sequential_container(sz, virtual_wo_vec_buf));

                    for (size_t i = 0u; i < sz; ++i){
                        host_ptr_arr[i * 3]     = data_arr[i].src_grad_ptr;
                        host_ptr_arr[i * 3 + 1] = data_arr[i].src_logit_ptr;
                        host_ptr_arr[i * 3 + 2] = data_arr[i].dst_grad_ptr;

                        dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::backward_mono_dispatch_control_chk(data_arr[i].dispatch_control));

                        total_complexity        += dg::network_tileops_host_poly::decode_backward_mono_dispatch_control(data_arr[i].dispatch_control)->runtime_complexity;
                        auto work_order         = [e = data_arr[i]]() noexcept{
                            dg::network_exception_handler::nothrow_log(dg::network_tileops_host_poly::backward_mono(e.src_grad_ptr, e.src_logit_ptr, e.dst_grad_ptr, 
                                                                                                                    e.dispatch_control, convert_grad_status_to_host_write_option(e.src_grad_status)));
                        };

                        size_t virtual_wo_bsz   = dg::network_host_asynchronous::get_preallocated_virtual_async_task_size(work_order);
                        char * virtual_wo_buf   = dg::network_exception_handler::nothrow_log(this->allocator->malloc(virutal_wo_bsz));
                        auto virtual_wo         = dg::network_exception_handler::nothrow_log(dg::network_host_asynchronous::make_preallocated_virtual_async_task(work_order, virtual_wo_buf));

                        dg::network_exception_handler::nothrow_log(virtual_wo_vec->add(std::move(virtual_wo)));
                    }

                    dg::network_exception_handler::nothrow_log(this->restrict_synchronizer->add(host_ptr_arr.get(), std::next(host_ptr_arr.get(), host_ptr_arr_sz)));
                    auto synchronizable = dg::network_exception_handler::nothrow_log(this->async_device->exec(std::move(virtual_wo_vec), total_complexity)); //TODOs: except
                    dg::network_exception_handler::nothrow_log(this->synchronizer->add(std::move(synchronizable)));
                }
            };

            struct ResolutorKeyArgument{
                uma_ptr_t dst_lck_addr;
                uma_ptr_t src_lck_addr;

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) const noexcept{
                    reflector(dst_lck_addr, src_lck_addr);
                }

                template <class Reflector>
                constexpr void dg_reflect(const Reflector& reflector) noexcept{
                    reflector(dst_lck_addr, src_lck_addr);
                }
            };

            struct ResolutorValueArgument{
                uma_ptr_t dst;
                uma_ptr_t src;
                operatable_id_t expected_ops_id;
            };

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<ResolutorKeyArgument, ResolutorValueArgument>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_producer_consumer::DeliveryHandle<EndUserPacket> * eu_packet_delivery_handle;
                dg::network_host_asynchronous::AsynchronousDeviceInterface * host_async_device;
                dg::network_cuda_controller::AsynchronousDeviceInterface * cuda_async_device;
                dg::network_allocation::ArenaAllocatorInterface * allocator;
                size_t vectorization_sz;

                void push(ResolutorKeyArgument key, ResolutorValueArgument * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(key.dst_lck_addr, key.src_lck_addr);

                    auto umamap_reacquirer                          = dg::network_exception_handler::nothrow_log(dg::network_uma::region_reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{}));
                    auto dst_grad_vmamap_reacquirer                 = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_grad_vmamap_reacquirer                 = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());
                    auto src_logit_vmamap_reacquirer                = dg::network_exception_handler::nothrow_log(dg::network_vmamap::region_remapper_raii_initialize());

                    auto cuda_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_intiialize<dg::network_cuda_controller::Synchronizer>());
                    auto cuda_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_cuda_controller::RestrictPointerSynchronizer>(&cuda_synchronizer));
                    auto cuda_internal_resolutor                    = CudaInternalResolutor{};
                    cuda_internal_resolutor.async_device            = this->cuda_async_device;
                    cuda_internal_resolutor.synchronizer            = &cuda_synchronizer;
                    cuda_internal_resolutor.restrict_synchronizer   = &cuda_restrict_synchronizer;
                    cuda_internal_resolutor.allocator               = this->allocator;

                    auto host_synchronizer                          = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::Synchronizer>());
                    auto host_restrict_synchronizer                 = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_host_asynchronous::RestrictPointerSynchronizer>(&host_synchronizer));
                    auto host_internal_resolutor                    = HostInternalResolutor{};
                    host_internal_resolutor.async_device            = this->host_async_device; 
                    host_internal_resolutor.synchronizer            = &host_synchronizer;
                    host_internal_resolutor.restrict_synchronizer   = &host_restrict_synchronizer;
                    host_internal_resolutor.allocator               = this->allocator;

                    size_t trimmed_cuda_vectorizer_sz               = std::min(this->vectorization_sz, sz);
                    size_t cv_allocation_cost                       = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&cuda_internal_resolutor, trimmed_cuda_vectorizer_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> cv_mem(cv_allocation_cost);
                    auto cuda_vectorizer                            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&cuda_internal_resolutor, trimmed_cuda_vectorizer_sz, cv_mem.get()));

                    size_t trimmed_host_vectorizer_sz               = std::min(this->vectorization_sz, sz);
                    size_t hv_allocation_cost                       = dg::network_producer_consumer::delvrsrv_keyhint_allocation_cost(&host_internal_resolutor, trimmed_host_vectorizer_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> hv_mem(hv_allocation_cost);
                    auto host_vectorizer                            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_keyhint_preallocated_raiihandle(&host_internal_resolutor, trimmed_host_vectorizer_sz, hv_mem.get()));

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src, expected_ops_id]        = std::make_tuple(data_arr[i].dst, data_arr[i].src, data_arr[i].expected_ops_id);
                        uma_ptr_t dst_src                       = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id       = dg::network_tile_member_getsetter::get_msgrbwd_operatable_memevent_id_nothrow(dst);
                        operatable_id_t dst_bwd_operatable_id   = dg::network_tile_member_getsetter::get_msgrbwd_operatable_backward_id_nothrow(dst);
                        init_status_t dst_init_status           = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(dst);
                        grad_status_t dst_grad_status           = dg::network_tile_member_getsetter::get_msgrbwd_grad_status_nothrow(dst);
                        uma_ptr_t dst_grad_umaptr               = dg::network_tile_member_getsetter::get_msgrbwd_grad_addr_nothrow(dst);
                        dispatch_control_t dispatch_control     = dg::network_tile_member_getsetter::get_msgrbwd_backward_dispatch_control_nothrow(dst);

                        std::expected<operatable_id_t, exception_t> src_bwd_operatable_id   = dg::network_tile_member_getsetter::get_tile_operatable_backward_id(src);
                        std::expected<init_status_t, exception_t> src_init_status           = dg::network_tile_member_getsetter::get_tile_init_status(src);
                        std::expected<grad_status_t, exception_t> src_grad_status           = dg::network_tile_member_getsetter::get_tile_grad_status(src);
                        std::expected<uma_ptr_t, exception_t> src_grad_umaptr               = dg::network_tile_member_getsetter::get_tile_grad_addr(src);
                        std::expected<uma_ptr_t, exception_t> src_logit_umaptr              = dg::network_tile_member_getsetter::get_tile_logit_addr(src);
                        
                        if (!src_bwd_operatable_id.has_value() || !src_init_status.has_value() || !src_grad_status.has_value() 
                            || !src_grad_umaptr.has_value() || !src_logit_umaptr.has_value()){

                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (src_init_status.value() != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != expected_ops_id){
                            continue;
                        }

                        if (dst_bwd_operatable_id != src_bwd_operatable_id.value()){
                            continue;
                        }

                        if (dst_grad_status != TILE_GRAD_STATUS_HAS_VALUE){
                            continue;
                        }

                        auto dispatch_info = dg::network_exception_handler::nothrow_log(dg::network_dispatch_control::decode_msgrbwd_backward_dispatch(dispatch_control)); 

                        dg::network_uma::region_reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{src_grad_umaptr.value(), dispatch_info.src_grad_vd_id}, 
                                                                                                           {src_logit_umaptr.value(), dispatch_info.src_logit_vd_id}, 
                                                                                                           {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}});

                        vma_ptr_t src_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{}); 
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});

                        dg::network_vmamap::region_remapper_remap_nothrow(src_grad_vmamap_reacquirer, src_grad_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);
                        dg::network_vmamap::region_remapper_remap_nothrow(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dispatch_platform)){
                            auto cuda_resolutor_arg             = CudaResolutorArgument{};
                            cuda_resolutor_arg.src_grad_ptr     = dg::network_vmamap::get_cuda_ptr(src_grad_vmamap_reacquirer);
                            cuda_resolutor_arg.src_logit_ptr    = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.dst_grad_ptr     = dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap_reacquirer);
                            cuda_resolutor_arg.src_grad_status  = src_grad_status.value();
                            cuda_resolutor_arg.dispatch_control = dispatch_info.tileops_cuda_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(cuda_vectorizer.get(), cuda_resolutor_arg.src_logit_ptr, cuda_resolutor_arg);
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dispatch_platform)){
                            auto host_resolutor_arg             = HostResolutorArgument{};
                            host_resolutor_arg.src_grad_ptr     = dg::network_vmamap::get_host_ptr(src_grad_vmamap_reacquirer);
                            host_resolutor_arg.src_logit_ptr    = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            host_resolutor_arg.dst_grad_ptr     = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);
                            host_resolutor_arg.src_grad_status  = src_grad_status.value();
                            host_resolutor_arg.dispatch_control = dispatch_info.tileops_host_dispatch_control;

                            dg::network_producer_consumer::delvrsrv_deliver(host_vectorizer.get(), host_resolutor_arg.src_logit_ptr, host_resolutor_arg);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_msgrbwd_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        exception_t descendant_set_err = dg::network_tile_member_getsetter::set_tile_grad_status(src, TILE_GRAD_STATUS_HAS_VALUE);

                        if (dg::network_exception::is_failed(descendant_set_err)){
                            (void) descendant_set_err; //
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::virtualize_event(dg::network_memcommit_factory::make_event_backward_do_signal(src, expected_ops_id)));
                    }

                    dg::network_producer_consumer::delvrsrv_clear(cuda_vectorizer.get());
                    dg::network_producer_consumer::delvrsrv_clear(host_vectorizer.get());
                    host_synchronizer.sync();
                    cuda_synchronizer.sync();

                    for (size_t i = 0u; i < sz; ++i){
                        if (!msgrbwd_outbound_vec[i].has_value()){
                            continue;
                        }

                        EndUserPacket eu_packet = {};
                        eu_packet_kind          = EUPACKET_MSGRBWD;
                        eu_packet.content       = dg::network_compact_serializer::serialize<dg::string>(GradValue{msgrbwd_outbound_vec[i]->tile_id, 
                                                                                                                  msgrbwd_outbound_vec[i]->timestamp, 
                                                                                                                  msgrbwd_outbound_vec[i]->grad_accum_sz}); //I dont know what this is for, except for system calibrations - we'll place frequencies on memgions - group those memregions - process them serially to guarantee the advertised frequencies - we'll do our best on the system part - but the true benchmarks must be from the msgrbwds + msgrfwds
                        eu_packet.dst           = msgrbwd_outbound_vec[i]->dst;
                        eu_packet.retry_count   = msgrbwd_outbound_vec[i]->retry_count;
                        eu_packet.urgency       = msgrbwd_outbound_vec[i]->urgency;
                        eu_packet.comm          = msgrbwd_outbound_vec[i]->comm;

                        dg::network_producer_consumer::delvrsrv_deliver(this->eu_packet_delivery_handle, std::move(eu_packet));
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

                    if constexpr(DEBUG_MODE_FLAG){
                        if (!tile_kind.has_value()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(tile_kind.error()));
                            std::abort();
                        }
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