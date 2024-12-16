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

    //the question is to whether devirtualize this at a dispatching center - or devirtualize this at the resolutors
    //the problem with devirtualizing this at a dispatching center is - radix problems + access serialization + locality problems - locality problem is storage problems - we pull virtual_event from a container - we radix it - then we push to another container - the pushing to another container is the expensive phase - where we need to touch at least O(n) space in adjunction to the O(n) pulling space
    //we either need to have dedicated worker on a radix - or ... have a dedicated worker on a number of radixes - which impose (1): code management problem (2): affinity problem, (3): access serialization problem
    //but the trade off is devirtualizing at a dispatching center could achieve better dispatching result - by forcing relevant events together 
    
    //in contrasts, by devirtualizing these guys at the resolutors - we must rely on chances - chances that adjecent virtual_memory_event_t(s) are relevant - and we offload this "semantic relevantness" responsibility to memregion frequency (Hz) + collectors + memory press
    //there is no perfect answer - only use cases, benchmarks, statistics and profile guided optimizations
    //we are moving in the direction of keep it simple, stupid, maintainable for these solutions are only means to an end - which is 5 flops/ virtual_memory_event
    //fast forward could be achieved by pinging more tiles - to reduce the pingpong communication latencies 
    //we will get back to the allocation problem soon - we want to maximize locality for delvrsrv by using heap_stack hybrid allocations

    //alright guys, to summarize:

    //assume 1KB/ tile - we are looking at 1 << 30 tiles for 1TB
    //we want fast bindings + orphans (I'm talking O(1) literally - by vectorization of memregions)
    //we have 1024 concurrent memregions for 32 host cores (we dont care about cuda cores)
    //4096 concurrent memregions for 128 cores
    //the mmmap regions is total_sz/ (tile_sz * 1024) - so we are looking at 1 << 20 mmap regions - and 1 << 10 memlock_regions
    //we want to minimize lock overhead by vectorization - and timing issues by placing frequencies on the memregion press
    //we resolute the virtual_event -> event at the resolutor to avoid access serialization issues + timing issue + management issue + cache issue + etc.
    //we vectorize the memregion by chances (and the worst case scenerio must not be 5% slower than the non-vectorization approach)
    //there is tons of tuning to do - but we'll get there guys 
    //compiler-devirtualization by using ID on interfaces, heap-stack allocations, proper delivery_size, etc, these are calibratables - we have a dedicated program to tune these - we'll get there 
    //there are inputs we can't assume like init_status - we'll work on the asusmptions whether it is at the initializations or the resolutors later - this is not an exclusive statement - because there is a concept of foreign injections

    //---------
    //difficult - because we are expecting 1 billion intiializations/ orphan + pingpong per second
    //every memory operation is expensive - we are talking about queueing the virtual_memory_event_t - deque - dispatch - there is no intermediate steps
    //every misfetch is bad - so we have to actually vectorize by memregions - to avoid misfetches + false sharing between cores
    
    //---------
    //we only store logits + grads on filesystems - other class members are all on RAM - for fast access
    //this is memregion bindings - which we will discuss later
    //we want to group the forwards using frequency - to extract data from filesystem in "one load" - so there is no read from fsys - do ops - evict page - do ops - read from fsys - we want to minimize the fsys read as much as possible - same goes for msgrfwds + msgrbwds reading from cuda_memregion 
    //because of the way we are doing mmap - there is a concurrent limit to how many concurrent workers can mmap at the same time - to avoid resource exhaustion and abort
    //we'll use the memregion radix strategy to do forward and backward - to avoid map acquisition and leverage the transfer function of mapping
    //we also want fair locking - such is there is no particular task that holds the lock for too long - and bottleneck other tasks (we want to solve this by increasing concurrent regions + increase # of concurrent resulutor dispatchers + sleeping mutex to avoid cpu clocks)
    //we also want drainage system - priority system - to avoid flooding + forward of priority orders
    //we want to maximize the usage of heap_stack whenever possible - the one that we used in bignum
    //we want to build heap allocations for different allocation radix - we'll work on this later

    //we are expecting to have these filled in roughly 2-3 weeks

    //---------
    //alright - I think this is an appropriate approach - not necessarily decent - if we are aiming to maximize host_flops
    //sleeping mutex + hyperthreading is a good approach to avoid memregion acquisition collisions (we offload the responsibility to kernel instead of doing internal management) - we'll move in the direction
    //the formula is easy - assume 1024 memlock_regions
    //assume 32 concurrent workers
    //assume each concurrent worker occupies maximum 4 memregions at a time
    //so the collision rate in the worst case scenerio is 128/1024 = 1/8 = 16%
    //so for every 8 task - there is 1 waiting task - since the nature of ping/pong + orphan + inits is not expensive - we want to hyperthread this (to "proceed" to another task while waiting on the "busy" tasks)
    //in another word, we reduce the sleeping overhead by the factor of hyperthreads
    //or we can increase the # of memlock_regions - to reduce the collision rate
    //these are calbratables and it totally depends on the implementation of kernel's completely fair scheduling

    //kv_raiihandle is an insert-only (unordered_unstable_fastinsert_map) map that clears on certain cap - with virtual_addr size of uint8_t - we are not wasting spaces
    //we'll do detail benchmarks later - but these are the optimizables and calibratables - which we don't care about for now

    //--------
    //I've been thinking about what Mom said - that AI is not Machine Learning
    //it's partially true - you want augmented retrieval whatever - graph and discrete logics
    //but I think it's the training that is wrong
    //if you have correct training and perfect logit values to do middle layer compressions - I think we'll get there
    //all the problems in math_approx is what's wrong with Machine Learning
    //the problem is we want to discretize intervals (initial values) to be able to correctly newton_approx
    //without interval discretization - we'll be stuck at suboptimal and local minima and unable to saturate the logit information - this is another problem beside the gear + rotor problem which is very difficult to solve
    //yet I think middle layer compression is a quantifiable task - we have a well defined f(x) -> y, g(f(x)) -> x
    //we want to maximize the compression_rate/ logit_size ratio - we have dedicated miners to work on these problems - running billions of different discretizations to tune these f(x) compression function
    //we want to keep the layer size small enough to be minable - and large enough to achieve the compression rate
    //this is the optimization problem that we'll be working on in the next 2 years

    //-------
    //Mom also complained about the errors - we'll circle back to either abort the program on error or make the error msg more descriptive - this is another task we don't worry about now
    //we'll add journal + important msgs along the way

    //-------
    //alright guys, we implemented locality + vectorization optimizations - this is only a radix of optimization
    //we can reiterate the idea of using array<uma_ptr_t, ARRAY_SZ>, fattening tiles + etc. to saturate GPU powers
    //which are two entire other different radices of optimization
    //we are expecting 1 << 30 /core * s pingpong + init + orphan orders
    //50TBs cuda flops dispatch/core * s
    //and full network bandwidth of msgrfwd + msgrbwd + extnsrc + extndst
    //we'll reiterate the designs

    //-------
    //implementation flaws:
    //assume reference on vma_ptr_t region is a defined operation if vma_ptr_t is not acquired (or invalidated) from uma_ptr_t
    //assume polymorphic getsetters are no_exception operations - fix
    //assume vmamap_nothrow (we can assume this)

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

    class ForwardPingMonoSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box; //it's always good practice to add constness in a concurrent context to hinder compiler's altering the value which works in non-concurrent context but does not work in concurrent context
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingMonoSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity,
                                           size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity),
                                                                              vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());
                        
                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
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
                        init_status_t init_status = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(ptr);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(ptr);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_signal(descendant));
                                dg::network_tile_member_getsetter::set_mono_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
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

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr      = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_region    = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_region, ptr_arr[i]);
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
                                } else{
                                    std::unreachable();
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

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        uma_ptr_t ptr               = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(ptr);

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
                                } else{
                                    std::unreachable();
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

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
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
                                } else{
                                    std::unreachable();
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

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
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
                                } else{
                                    std::unreachable();
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

            ForwardPingExtnDstSignalResolutor(std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                              std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                              size_t delivery_capacity,
                                              size_t vectorization_sz): uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                        host_ip_retriever(std::move(host_ip_retriever)),
                                                                        request_box(std::move(request_box)),
                                                                        delivery_capacity(delivery_capacity),
                                                                        vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.uma_ip_retriever          = this->uma_ip_retriever->get();
                    internal_resolver.host_ip_retriever         = this->host_ip_retriever->get();
                    internal_resolver.request_delivery_handle   = delivery_handle.get();
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
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
                                auto ping_request       = Request<external_virtual_memory_event_t>{};
                                ping_request.requestee  = this->uma_ip_retriever->ip(counterpart);
                                ping_request.requestor  = this->host_ip_retriever->ip();
                                ping_request.content    = dg::network_external_memcommit_factory::make_event_forward_ping_signal(counterpart);

                                dg::network_producer_conumser::delvrsrv_deliver(this->request_delivery_handle, std::move(ping_request));
                                dg::network_tile_member_getsetter::set_extndst_init_status_nothrow(ptr, TILE_INIT_STATUS_DECAYED);
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

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
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
                                } else{
                                    std::unreachable();
                                }
                        }
                    }
                }
            };
    };

    class ForwardPingMsgrFwdResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

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

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
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
                                } else{
                                    std::unreachable();
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

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
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
                                } else{
                                    std::unreachable();
                                }
                        }
                    }
                }
            };
    };

    class ForwadPingImmuResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        public:

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                (void) ptr_arr;
            }
    };

    class ForwardPingSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

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

            //we'll limit the interface casting later -  

            ForwardPingSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor,
                                       size_t leaf_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> mono_resolutor,
                                       size_t mono_dispatch_sz,
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
                                       size_t msgrbwd_dispatch_sz,
                                       std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> immu_resolutor,
                                       size_t immu_dispatch_sz) noexcept: leaf_resolutor(std::move(leaf_resolutor)),
                                                                          leaf_dispatch_sz(leaf_dispatch_sz),
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

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto leaf_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_resolutor.get(), this->leaf_dispatch_sz));
                auto mono_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->mono_resolutor.get(), this->mono_dispatch_sz));
                auto pair_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz));
                auto uacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz));
                auto pacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz));
                auto extnsrc_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz));
                auto extndst_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz));
                auto crit_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz));
                auto msgrfwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz));
                auto msgrbwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz));
                auto immu_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->immu_resolutor.get(), this->immu_dispatch_sz));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(ptr_arr[i]); 

                    if (!tile_kind.has_value()){ //this branch is never taken - so we don't worry - this takes at most 2-3 CPU cycle
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(leaf_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MONO:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(mono_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_PAIR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pair_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_UACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(uacm_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_PACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pacm_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNSRC:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extnsrc_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNDST:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extndst_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_CRIT:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(crit_delivery_handle.get(), ptr_arr[i]);
                            break;                            
                        }
                        case TILE_KIND_MSGRFWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrfwd_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRBWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrbwd_delivery_handle.get(), ptr_arr[i]);
                            break;                        
                        }
                        case TILE_KIND_IMMU:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(immu_delivery_handle.get(), ptr_arr[i]);
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

    //---

    class ForwardPongLeafRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                                break;
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
    };

    class ForwardPongMonoRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_mono_push_observer_nothrow(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
    };

    class ForwardPongPairRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
    };

    class ForwardPongUACMRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz)); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
    };

    class ForwardPongPACMRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
            const size_t vectorization_sz;

        public:

            ForwardPongExtnDstRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                               size_t delivery_capacity,
                                               size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                  delivery_capacity(delivery_capacity),
                                                                                  vectorization_sz(vectorization_sz){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
    };

    class ForwardPongCritRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
    };

    class ForwardPongMsgrFwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolver: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
    };

    class ForwardPongMsgrBwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
    };
    
    class ForwardPongImmuRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_immu_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                
                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_immu_init_status_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                                break;
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
    };

    class ForwardPongRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> leaf_resolutor;
            const size_t leaf_dispatch_sz;
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> mono_resolutor;
            const size_t mono_dispatch_sz;
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
                                        std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> mono_resolutor,
                                        size_t mono_dispatch_sz,
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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto leaf_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_resolutor.get(), this->leaf_dispatch_sz));
                auto mono_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->mono_resolutor.get(), this->mono_dispatch_sz));
                auto pair_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz));
                auto uacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz));
                auto pacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz));
                auto extnsrc_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz));
                auto extndst_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz));
                auto crit_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz));
                auto msgrfwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz));
                auto msgrbwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz));
                auto immu_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->immu_resolutor.get(), this->immu_dispatch_sz));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(std::get<0>(ptr_arr[i]));

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(leaf_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MONO:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(mono_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_PAIR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pair_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_UACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(uacm_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_PACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pacm_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNSRC:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extnsrc_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNDST:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extndst_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_CRIT:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(crit_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRFWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrfwd_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRBWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrbwd_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_IMMU:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(immu_delivery_handle.get(), ptr_arr[i]);
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

    //

    class ForwardPingPongLeafRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status = dg::network_tile_member_getsetter::get_leaf_init_status_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                            // case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            // case TILE_INIT_STATUS_DECAYED:
                            //     //leaf is not decayed - either empty or initialized - so this is an error - we'll reconsider this
                                // dg::network_log_stackdump::error_fast(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                                break;
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
    };

    class ForwardPingPongMonoRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg;:network_tile_member_access::safecthrow_mono_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz){

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                dg::network_tile_member_getsetter::controller_mono_push_observer(requestee, requestor);
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant =  dg::network_tile_member_getsetter::get_mono_left_descendant_nothrow(ptr);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant, requestee));
                                dg::network_tile_member_getsetter::set_mono_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
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
    };

    class ForwardPingPongPairRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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
            
            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant, requestee));
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant, requestee));
                                dg::network_tile_member_getsetter::set_pair_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
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
    };

    class ForwardPingPongUACMRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz)); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_uacm_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant_arr[i], requestee));
                                }

                                dg::network_tile_member_getsetter::set_uacm_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
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
    };

    class ForwardPingPongPACMRequstResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz)); 

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pacm_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pacm_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }
        
        private:
            
            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i]; 
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_pacm_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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
                                    dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(left_descendant_arr[i], requestee));
                                    dg::network_producer_consumer::delvrrsv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(right_descendant_arr[i], requestee));
                                }

                                dg::network_tile_member_getsetter::set_pacm_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
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
    };

    class ForwardPingPongExtnSrcRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            FowrardPingPongExtnSrcRequestResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                   size_t delivery_capacity,
                                                   size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                      delivery_capacity(delivery_capacity),
                                                                                      vectorization_sz(vectorization_sz){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolver internal_resolver{};
                    internal_resolver.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolver, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }
        
        private:
            
            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant));
                                dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
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
    };

    class ForwardPingPongExtnDstRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

        private:

            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t request_delivery_capacity;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> outbound_request_box;
            const size_t outbound_delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardPingPongExtnDstRequestResolutor(std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                   std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                   std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                   size_t request_delivery_capacity,
                                                   std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> outbound_request_box,
                                                   size_t outbound_delivery_capacity,
                                                   size_t vectorization_sz) noexcept: uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                      host_ip_retriever(std::move(host_ip_retriever)),
                                                                                      request_box(std::move(request_box)),
                                                                                      request_delivery_capacity(request_delivery_capacity),
                                                                                      outbound_request_box(std::move(outbound_request_box)),
                                                                                      outbound_delivery_capacity(outbound_delivery_capacity),
                                                                                      vectorization_sz(vectorization_sz){}

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto request_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->request_delivery_capacity));
                auto outbound_delivery_handle   = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->outbound_request_box.get(), this->outbound_delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle  = request_delivery_handle.get();
                    internal_resolutor.outbound_delivery_handle = outbound_delivery_handle.get();

                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;
                dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * outbound_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t counterpart   = dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(requestee);
                                Address requestee_ip    = this->uma_ip_retriever->ip(counterpart);
                                Address requestor_ip    = this->host_ip_retriever->ip();
                                auto ping_request       = Request<external_virtual_memory_event_t>{requestee_ip, requestor_ip, dg::network_external_memcommit_factory::make_event_forward_ping_signal(counterpart)};

                                dg::network_producer_consumer::delvrsrv_deliver(this->outbound_delivery_handle, std::move(ping_request));
                                dg::network_tile_member_getsetter::set_extndst_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
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
    };

    class ForwardPingPongCritRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_crit_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_crit_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i]; 
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_crit_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
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

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED; [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_crit_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant));
                                dg::network_tile_member_getsetter::set_crit_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
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
    };

    class ForwardPingPongMsgrFwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_msgrfwd_init_status_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                                dg::network_tile_member_getsetter::controller_msgrfwd_push_observer(requestee, requestor);
                                break;
                            case TILE_INIT_STATUS_INITIALIZED:
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
                                break;
                            default:
                                if constexpr(DEBUG_MODE_FLAG){
                                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                    std::abort();
                                } else{
                                    std::unreachable();
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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant));
                                dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
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
    };

    class ForwardPingPongMsgrBwdRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(requestee);

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
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestee));
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

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                                break;
                            case TILE_INIT_STATUS_ADOPTED:
                            {
                                uma_ptr_t descendant = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(requestee);
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_pingpong_request(descendant));
                                dg::network_tile_member_getsetter::set_msgrbwd_init_status_nothrow(requestee, TILE_INIT_STATUS_DECAYED);
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
    };

    class ForwardPingPongImmuRequestResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>{

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

            void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_immu_ptr_access(std::get<0>(ptr_arr[i]));

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_immu_rcu_addr_nothrow(std::get<0>(ptr_arr[i]));
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
                    }
                }
            }
        
        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [requestee, requestor] = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_immu_init_status_nothrow(requestee);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]] //this is invalid state
                            case TILE_INIT_STATUS_DECAYED: [[fallthrough]] //this is invalid state
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(requestor));
                                break;
                            }
                            default:
                                //this assumes valid input - we should consider failsafes here
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
            const std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> immu_pingpong_resolutor;
            const size_t immu_dispatch_sz;

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
                                            size_t msgrbwd_dispatch_sz,
                                            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> immu_pingpong_resolutor,
                                            size_t immu_dispatch_sz) noexcept: leaf_pingpong_resolutor(std::move(leaf_pingpong_resolutor)),
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
                                                                               msgrbwd_dispatch_sz(msgrbwd_dispatch_sz),
                                                                               immu_pingpong_resolutor(std::move(immu_pingpong_resolutor)),
                                                                               immu_dispatch_sz(immu_dispatch_sz){}

        void push(std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

            auto leaf_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_pingpong_resolutor.get(), this->leaf_dispatch_sz));
            auto mono_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->mono_pingpong_resolutor.get(), this->mono_dispatch_sz));
            auto pair_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_pingpong_resolutor.get(), this->pair_dispatch_sz));
            auto uacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_pingpong_resolutor.get(), this->uacm_dispatch_sz));
            auto pacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_pingpong_resolutor.get(), this->pacm_dispatch_sz));
            auto extnsrc_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_pingpong_resolutor.get(), this->extnsrc_dispatch_sz));
            auto extndst_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_pingpong_resolutor.get(), this->extndst_dispatch_sz));
            auto crit_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_pinpong_resolutor.get(), this->crit_dispatch_sz));
            auto msgrfwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_pingpong_resolutor.get(), this->msgrfwd_dispatch_sz));
            auto msgrbwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_pingpong_resolutor.get(), this->msgrbwd_dispatch_sz));
            auto immu_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->immu_pingpong_resolutor.get(), this->immu_dispatch_sz));

            for (size_t i = 0u; i < sz; ++i){
                std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(std::get<0>(ptr_arr[i]));

                if (!tile_kind.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                    continue;
                }

                switch (tile_kind.value()){
                    case TILE_KIND_LEAF:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(leaf_delivery_handle.get(), ptr_arr[i]);
                        break;
                    }
                    case TILE_KIND_MONO:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(mono_delivery_handle.get(), ptr_arr[i]);
                        break;
                    }
                    case TILE_KIND_PAIR:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(pair_delivery_handle.get(), ptr_arr[i]);
                        break;
                    }
                    case TILE_KIND_UACM:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(uacm_delivery_handle.get(), ptr_arr[i]);
                        break;
                    }
                    case TILE_KIND_PACM:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(pacm_delivery_handle.get(), ptr_arr[i]);
                        break;
                    }
                    case TILE_KIND_EXTNSRC:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(extnsrc_delivery_handle.get(), ptr_arr[i]);
                        break;
                    }
                    case TILE_KIND_EXTNDST:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(extndst_delivery_handle.get(), ptr_arr[i]);
                        break;
                    }
                    case TILE_KIND_CRIT:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(crit_delivery_handle.get(), ptr_arr[i]);
                        break;
                    }
                    case TILE_KIND_MSGRFWD:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(msgrfwd_delivery_handle.get(), ptr_arr[i]);
                        break;
                    }
                    case TILE_KIND_MSGRBWD:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(msgrbwd_delivery_handle.get(), ptr_arr[i]);
                        break;                        
                    }
                    case TILE_KIND_IMMU:
                    {
                        dg::network_producer_consumer::delvrsrv_deliver(immu_delivery_handle.get(), ptr_arr[i]);
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

    //

    class ForwardInitLeafSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        public:

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                (void) ptr_arr;
            }
    };

    class ForwardInitMonoSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardInitMonoSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                             size_t delivery_capacity,
                                             size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                delivery_capacity(delivery_capacity),
                                                                                vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto descendant_arr     = std::make_unique<uma_ptr_t[]>(sz); //heap-stack
                auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = dg::pointer_limits<uma_ptr_t>::null_value();                        
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(ptr_arr[i], std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz)); //different var

                    for (size_t i = 0u; i < sz; ++i){
                        if (descendant_arr[i] == dg::pointer_limits<uma_ptr_t>::null_value()){
                            continue;
                        }

                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant_arr[i]);
                        uma_ptr_t dst_lck_addr  = dg::memult::region(dst_rcu_addr, dg::network_uma::memregion_size());  //memregion_size to avoid false_sharing - memregion_size must be pow2(x) <= memlock_region_size()
                        uma_ptr_t src_lck_addr  = dg::memult::region(src_rcu_addr, dg::network_uma::memregion_size());  //memregion_size to avoid false_sharing - memregion_size must be pow2(x) <= memlock_region_size()
                        auto key                = dg::utility::to_unique_representation(dst_lck_addr, src_lck_addr);   //consider unique_representation by sorting 

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(ptr_arr[i], descendant_arr[i]));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t *> data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [ptr, fecthing_ptr]    = data_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(ptr);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_ptr   = dg::pointer_limits<uma_ptr_t>::null_value();
                                break;
                            }
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_ptr   = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(ptr);
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
                
                dg::network_producer_consumer::DeliveryHandle<virtual_emmory_event_t> * request_delivery_handle;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * data_arr, size_t sz){

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));
                    
                    auto umamap_reacquirer      = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto dst_vmamap_reacquirer  = dg::network_vmamap::reacquirer_raii_initialize(); 
                    auto src_vmamap_reacquirer  = dg::network_vmamap::reacquirer_raii_initialize();

                    dg::network_cuda_controller::CudaSynchronizer synchronizer{};
                    dg::network_controller::RestrictPointerSynchronizer restrict_synchronizer(synchronizer); //compiler might reorder this and invalidate the pointer reference - guard with a scope

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src]                                             = data_arr[i];
                        init_status_t dst_init_status                               = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(dst);
                        size_t dst_observer_arr_sz                                  = dg::network_tile_member_getsetter::get_mono_observer_array_size_nothrow(dst);
                        std::array<uma_ptr_t, OBSERVER_ARRAY_CAP> dst_observer_arr  = dg::network_tile_member_getsetter::get_mono_observer_array_nothrow(dst);
                        operatable_id_t dst_operatable_id                           = dg::network_tile_member_getsetter::get_mono_operatable_id_nothrow(dst);
                        uma_ptr_t dst_src                                           = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(dst);
                        dispatch_control_t dispatch_control                         = dg::network_tile_member_getsetter::get_mono_dispatch_control_nothrow(dst);
                        // init_status_t src_init_status                               = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(src);
                        // operatable_id_t src_operatable_id                           = dg::network_tile_member_getsetter::get_tile_operatable_id_nothrow(src); 

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != src_operatable_id){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                            continue;
                        }

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_mono(dispatch_control);

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}})){
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        //lets assume valid inputs are enforced by the setters for now - we can't be too cautious because it's bad practice - as long as the program is functional up to reasonable aborts
                        //reasonable aborts are the aborts that could theoretically never happen - by implementations
                        //we'll revisit the implementation design decisions - for now - the program is correct

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                        auto dst_map_vmaptr = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        auto src_map_vmaptr = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_vmamap_reacquirer, dst_map_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_vmamap_reacquirer, src_map_vmaptr)){

                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_vma::reacquirer_reacquire_nothrow(dst_vmamap_reacquirer, dst_map_vmaptr);
                        dg::network_vma::reacquirer_reacquire_nothrow(src_vmamap_reacquirer, src_map_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto dst_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(dst_vmamap_reacquirer);
                            auto src_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(src_vmamap_reacquirer);
                            restrict_synchronizer.add(dst_logit_cuda_ptr, src_logit_cudaptr);
                            auto async_id           = dg::network_tileops_cuda_poly::async_fwd_mono(dst_logit_cudaptr, src_logit_cudaptr, tileops_dp_code);

                            if (!async_id.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                                continue;
                            }

                            synchronizer.add(async_id.value());
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_logit_hostptr  = dg::network_vmamap::get_host_ptr(dst_vmamap_reacquirer);
                            auto src_logit_hostptr  = dg::network_vmamap::get_host_ptr(src_vmamap_reacquirer);
                            dg::network_tileops_host_poly::fwd_mono(dst_logit_hostptr, src_logit_hostptr, tileops_dp_code);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(dst_observer_arr[j]));
                        }

                        set_mono_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                    }
                }
            };
    };

    class ForwardInitPairSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box; //I said that it's hard to split these components babe - I want to virtualize these guys to process in the same logic - but it would be very unmaintainable in the long run - every component is assigned to the tile logic - its detached
            const size_t delivery_capacity;
            const size_t vectorization_sz;
        
        public:

            ForwardInitPairSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                             size_t delivery_capacity,
                                             size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                delivery_capacity(delivery_capacity),
                                                                                vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{
                
                //alright guys - we are rerolling the changes - we abort on errors that could never happen
                //we want program accuracy - if things went south - we throw kernel an exception to restart the program
                //we are not storing data in the core - it's the storage engine responsibility - core should expect crash at any given time

                auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));
                auto descendant_arr     = std::make_unique<std::optional<std::tuple<uma_ptr_t, uma_ptr_t>>[]>(sz);

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        descendant_arr[i]   = {};
                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(ptr_arr[i], std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        uma_ptr_t dst               = ptr_arr[i];
                        uma_ptr_t left              = std::get<0>(descendant_arr[i].value());
                        uma_ptr_t right             = std::get<1>(descendant_arr[i].value());
                        uma_ptr_t dst_lck_addr      = dg::memult::region(dst, dg::network_uma::memregion_size());
                        uma_ptr_t left_lck_addr     = dg::memult::region(left, dg::network_uma::memregion_size());
                        uma_ptr_t right_lck_addr    = dg::memult::region(right, dg::network_uma::memregion_size());
                        auto key                    = dg::utility::to_unique_representation(dst_lck_addr, left_lck_addr, right_lck_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(dst, left, right));
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, std::add_pointer_t<std::optional<std::tuple<uma_ptr_t, uma_ptr_t>>>>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, std::optional<std::tuple<uma_ptr_t, uma_ptr_t>> *> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, fetching_addr]   = ptr_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(dst);

                        //we don't really care about branching here - because it's highly predicted - we'll do optimizations that break readability later if this is the bottleneck - fine - we;ll move on for now

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr = std::nullopt;
                                break;
                            }
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                pong_count_t pong_count = dg::network_tile_member_getsetter::get_pair_pong_count_nothrow(dst);
                                pong_count += 1u; //has to be unsigned otherwise we risk unsigned wraparound
                                dg::network_tile_member_getsetter::set_pair_pong_count_nothrow(dst, pong_count);

                                if (pong_count >= dg::network_tile_metadata::PAIR_DESCENDANT_COUNT){
                                    *fetching_addr = std::make_tuple(dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(dst), dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(dst));
                                } else{
                                    *fetching_addr = std:nullopt;
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

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t>>{
                
                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr), std::get<2>(lck_addr));

                    auto umamap_reacquirer      = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{});
                    auto dst_vmamap_reacquirer  = dg::network_vmamap::reacquirer_raii_initialize();
                    auto lhs_vmamap_reacquirer  = dg::network_vmamap::reacquirer_raii_initialize();
                    auto rhs_vmamap_reacquirer  = dg::network_vmamap::reacquirer_raii_initialize();

                    dg::network_cuda_controller::CudaSynchronizer synchronizer{};
                    dg::network_controller::RestrictPointerSynchronizer restrict_synchronizer(synchronizer);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, lhs, rhs]                                        = data_arr[i];
                        operatable_id_t dst_operatable_id                           = dg::network_tile_member_getsetter::get_pair_operatable_id_nothrow(dst);
                        init_status_t dst_init_status                               = dg::network_tile_member_getsetter::get_pair_init_status_nothrow(dst);
                        dispatch_control_t dispatch_control                         = dg::network_tile_member_getsetter::get_pair_dispatch_control_nothrow(dst);
                        uma_ptr_t dst_logit_umaptr                                  = dg::network_tile_member_getsetter::get_pair_logit_addr_nothrow(dst); 
                        uma_ptr_t dst_lhs                                           = dg::network_tile_member_getsetter::get_pair_left_descendant_nothrow(dst);
                        uma_ptr_t dst_rhs                                           = dg::network_tile_member_getsetter::get_pair_right_descendant_nothrow(dst);
                        std::array<uma_ptr_t, OBSERVER_ARRAY_CAP> dst_observer_arr  = dg::network_tile_member_getsetter::get_pair_observer_array_nothrow(dst);
                        size_t dst_observer_arr_sz                                  = dg::network_tile_member_getsetter::get_pair_observer_array_size_nothrow(dst);
                        // operatable_id_t lhs_operatable_id                           = dg::network_tile_member_getsetter::get_tile_operatable_id_nothrow(lhs);
                        // init_status_t lhs_init_status                               = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(lhs);
                        // uma_ptr_t lhs_logit_umaptr                                  = dg::network_tile_member_getsetter::get_tile_logit_addr_nothrow(lhs);
                        // operatable_id_t rhs_operatable_id                           = dg::network_tile_member_getsetter::get_tile_operatable_id_nothrow(rhs);
                        // init_status_t rhs_init_status                               = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(rhs);
                        // uma_ptr_t rhs_logit_umaptr                                  = dg::network_tile_member_getsetter::get_tile_logit_addr_nothrow(rhs);

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

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                            continue;
                        }

                        if (lhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (rhs_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        auto [dst_vd_id, lhs_vd_id, rhs_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_pair(dispatch_control);

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {lhs_logit_umaptr, lhs_vd_id}, {rhs_logit_umaptr, rhs_vd_id}})){
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        //let's move on to radixing one src multiple dst for now - this is another optimizable

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {lhs_logit_umaptr, lhs_vd_id}, {rhs_logit_umaptr, rhs_vd_id}});

                        vma_ptr_t dst_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t lhs_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t rhs_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_vmamap_reacquirer, dst_vmaptr) 
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(lhs_vmamap_reacquirer, lhs_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(rhs_vmamap_reacquirer, rhs_vmaptr)){

                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_vmamap_reacquirer, dst_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(lhs_vmamap_reacquirer, lhs_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(rhs_vmamap_reacquirer, rhs_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto dst_cudaptr    = dg::network_vmamap::get_cuda_ptr(dst_vmamap_reacquirer);
                            auto lhs_cudaptr    = dg::network_vmamao::get_cuda_ptr(lhs_vmamap_reacquirer);
                            auto rhs_cudaptr    = dg::network_vmamap::get_cuda_ptr(rhs_vmamap_reacquirer);

                            restrict_synchronizer.add(dst_cudaptr, lhs_cudaptr, rhs_cudaptr);
                            auto async_id       = dg::network_tileops_cuda_poly::async_fwd_pair(dst_cudaptr, lhs_cudaptr, rhs_cudaptr, tileops_dp_code);

                            if (!async_id.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                                continue;
                            }

                            synchronizer.add(async_id.value());
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_hostptr    = dg::network_vmamap::get_host_ptr(dst_vmamap_reacquirer);
                            auto lhs_hostptr    = dg::network_vmamap::get_host_ptr(lhs_vmamap_reacquirer);
                            auto rhs_hostptr    = dg::network_vmamap::get_host_ptr(rhs_vmamap_reacquirer);

                            dg::network_tileops_host_poly::fwd_pair(dst_hostptr, lhs_hostptr, rhs_hostptr, tileops_dp_code);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(dst_observer_arr[j]));
                        }

                        dg::network_tile_member_getsetter::set_pair_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                    }
                }
            };
    };

    class ForwardInitUACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardInitUACMSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                           size_t delivery_capacity,
                                           size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                              delivery_capacity(delivery_capacity),
                                                                              vectorization_sz(vectorization_sz){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

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
                        auto ptrchk = dg::network_tile_member_access::safecthrow_uacm_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_uacm_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), lck_addr, std::make_tuple(ptr_arr[i], std::next(descendant_arr.get(), i)));
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

                        dg::vector<uma_ptr_t> ptr_vec   = dg::utility::vector_immu_push_back(descendant_arr[i].value(), ptr_arr[i]);
                        dg::vector<uma_ptr_t> rcu_vec   = dg::utility::vector_immu_transform(ptr_vec, [](uma_ptr_t e){return dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(e);});
                        dg::vector<uma_ptr_t> rep_vec   = dg::utility::vector_immu_transform(rcu_vec, [](uma_ptr_t e){return dg::memult::region(e, dg::network_uma::memregion_size());}); 
                        dg::set<uma_ptr_t> key          = dg::utility::set_make_from_vec(rep_vec);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), key, std::make_tuple(ptr_arr[i], descendant_arr[i].value()));
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

    class ForwardInitPACMSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

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

    class ForwardInitExtnSrcSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box;
            const std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever;
            const std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever;
            const size_t delivery_capacity;
            const size_t vectorization_sz;
        
        public:

            ForwardInitExtnSrcSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<Request<external_virtual_memory_event_t>>> request_box,
                                                std::shared_ptr<UnifiedMemoryIPRetrieverInterface> uma_ip_retriever,
                                                std::shared_ptr<HostIPRetrieverInterface> host_ip_retriever,
                                                size_t delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   uma_ip_retriever(std::move(uma_ip_retriever)),
                                                                                   host_ip_retriever(std::move(host_ip_retriever)),
                                                                                   delivery_capacity(delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto descendant_arr     = std::make_unique<uma_ptr_t[]>(sz);
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
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = dg::pointer_limits<uma_ptr_t>::null_value();
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), lck_addr, std::make_tuple(ptr_arr[i], std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle->get();
                    auto vectorized_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz);

                    if (!vectorized_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorized_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        if (descendant_arr[i] == dg::pointer_limits<uma_ptr_t>::null_value()){
                            continue;
                        }

                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant_arr[i]);
                        uma_ptr_t dst_rep_addr  = dg::memult::region(dst_rcu_addr, dg::network_uma::memregion_size());
                        uma_ptr_t src_rep_addr  = dg::memult::region(src_rcu_addr, dg::network_uma::memregion_size());
                        auto key                = dg::utility::to_unique_representation(dst_rep_addr, src_rep_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), key, std::make_tuple(ptr_arr[i], descendant_arr[ri]));
                    }
                }
            }
        
        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t *> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, fetching_addr]   = data_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_extnsrc_init_status_nothrow(dst);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr  = dg::pointer_limits<uma_ptr_t>::null_value();
                                break;
                            }
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_addr  = dg::network_tile_member_getsetter::get_extnsrc_descendant_nothrow(dst);
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

                dg::network_producer_consumer::DeliveryHandle<Request<external_virtual_memory_event_t>> * request_delivery_handle;
                UnifiedMmeoryIPRetrieverInterface * uma_ip_retriever;
                HostIPRetrieverInterface * host_ip_retriever;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));
                    
                    auto umamap_reacquirer              = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto dst_logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();

                    dg::network_cuda_controller::CudaSynchronizer synchronizer{};
                    dg::network_controller::RestrictPointerSynchronizer restrict_synchronizer(synchronizer);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src] = ptr_arr[i];

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != src_operatable_id){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_extnsrc(dispatch_control);

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}})){
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                        vma_ptr_t dst_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_logit_vmamap_reacquirer, dst_logit_vmaptr) 
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_umaptr)){

                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto dst_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap_reacquirer);
                            auto src_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            restrict_synchronizer.add(dst_logit_cudaptr, src_logit_cudaptr);
                            auto async_id           = dg::network_tileops_cuda_poly::async_fwd_mono(dst_logit_cudaptr, src_logit_cudaptr, tileops_dp_code);

                            if (!async_id.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                                continue;
                            }

                            synchronizer.add(async_id.value());
                        } else if(dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_logit_hostptr  = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            auto src_logit_hostptr  = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            dg::network_tileops_host_poly::fwd_mono(dst_logit_hostptr, src_logit_host_ptr, tileops_dp_code); //exceptions
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_extnsrc_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                        dg::string serialized                           = dg::network_tile_member_getsetter::serialize_extnsrc(dst);
                        Address fr_addr                                 = this->host_ip_retriever->ip();
                        Address to_addr                                 = this->uma_ip_retriever->ip(counterpart);
                        external_virtual_memory_event_t inject_event    = dg::network_external_memcommit_factory::make_event_shadow_injection(dst, TILE_KIND_EXTNSRC, serialized);
                        external_virtual_memory_event_t notify_event    = dg::network_external_memcommit_factory::make_event_forward_init_signal(counterpart);
                        external_virtual_memory_event_t event           = dg::network_external_memcommit_factory::make_event_sequential(inject_event, notify_event);
                        Request<external_virtual_memory_event_t> rq     = {to_addr, fr_addr, std::move(event)};
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(rq));     
                    }
                }
            };
    };

    class ForwardInitExtnDstSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter; //this is hard - we want to abstract the interface - yet provide a foreign injection (this is not recommended due to inputs chks + friends - but for ease of development + future extensions) - essentially we want to keep a deque + normal_hashmap internally, and we concurrent the FIFO hashmap (never exceeds certain cap) by radixing key, then we want a cyclic buffer of free tiles
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardInitExtnDstSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                std::shared_ptr<ForeignTileAliasGetterInterface> alias_getter,
                                                size_t delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   alias_getter(std::move(alias_getter)),
                                                                                   delivery_capacity(delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto descendant_arr     = std::make_unique<uma_ptr_t[]>(sz);
                auto delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity);

                if (!delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(delivery_handle.error()));
                    return;
                }

                {
                    InternalDescendantAddressFetcher fetcher{};
                    fetcher.alias_getter = this->alias_getter->get();
                    auto vectorized_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz);

                    if (!vectorized_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorized_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = dg::pointer_limits<uma_ptr_t>::null_value();
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memregion_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), lck_addr, std::make_pair(ptr_arr[i], std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle->get();
                    auto vectorized_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz);

                    if (!vectorized_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorized_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        if (descendant_arr[i] == dg::pointer_limits<uma_ptr_t>::null_value()){
                            continue;
                        }

                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant_arr[i]); 
                    }
                }
            }

        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::ConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t *>>{

                ForeignTileAliasGetterInterface * alias_getter;

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t *> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, fetching_addr]   = data_arr[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_extndst_init_status_nothrow(dst);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_ADOPTED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr  = dg::pointer_limits<uma_ptr_t>::null_value();
                                break;
                            }
                            case TILE_INIT_STATUS_DECAYED:
                            {   
                                std::optional<uma_ptr_t> alias = this->alias_getter->alias(dg::network_tile_member_getsetter::get_extndst_counterpart_nothrow(dst));

                                if (!alias.has_value()){
                                    *fetching_addr  = dg::pointer_limits<uma_ptr_t>::null_value();
                                } else{
                                    *fetching_addr  = alias.value();
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

            struct InternalResolutor: dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));
                    auto umamap_reacquirer              = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto dst_logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_initialize(); 
                    auto src_logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_initialize();

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src] = ptr_arr[i];

                        if (dst_counterpart != src_selfaddr){
                            continue;
                        }

                        if (dst_operatable_id != src_operatable_id){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_extndst(dispatch_control);

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}})){
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                        vma_ptr_t dst_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{}); 

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_logit_vmamap_reacquirer, dst_logit_vmaptr) 
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_vmaptr)){

                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto dst_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap_reacquirer);
                            auto src_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            restrict_synchronizer.add(dst_logit_cudaptr, src_logit_cudaptr); 
                            auto async_id           = dg::network_tileops_cuda_poly::async_fwd_mono(dst_logit_cudaptr, src_logit_cudaptr, tileops_dp_code);

                            if (!async_id.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                                continue;
                            }

                            synchronizer.add(async_id.value());
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_logit_hostptr  = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            auto src_logit_hostptr  = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            dg::network_tileops_host_poly::fwd_mono(dst_logit_hostptr, src_logit_hostpyr, tileops_dp_code);
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
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_fowrard_init_signal(dst_observer_arr[j]));
                        }
                    }
                }
            };
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
                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(observer_arr[i]));
                }

                dg::network_producer_consumer::delvrsrv_deliver(delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
            }
    };

    class ForwardInitMsgrFwdSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box;

            const size_t request_delivery_capacity;
            const size_t eu_delivery_capacity;
            const size_t vectorization_sz;
        
        public:

            ForwardInitMsgrFwdSingalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                                std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<EndUserPacket>> eu_packet_box,
                                                size_t request_delivery_capacity,
                                                size_t eu_delivery_capacity,
                                                size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                   eu_packet_box(std::move(eu_packet_box)),
                                                                                   request_delivery_capacity(request_delivery_capacity),
                                                                                   eu_delivery_capacity(eu_delivery_capacity),
                                                                                   vectorization_sz(vectorization_sz){}
            
            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto descendant_arr             = std::make_unique<uma_ptr_t[]>(sz);
                auto request_delivery_handle    = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->request_delivery_capacity);
                auto eu_packet_delivery_handle  = dg::network_producer_consumer::delvrsrv_open_raiihandle(this->eu_packet_box.get(), this->eu_delivery_capacity);

                if (!request_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(request_delivery_handle.error()));
                    return;
                }

                if (!eu_packet_delivery_handle.has_value()){
                    dg::network_log_stackdump::error(dg::network_exception::verbose(eu_packet_delivery_handle.error()));
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
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = dg::pointer_limits<uma_ptr_t>::null_value();
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), lck_addr, std::make_tuple(ptr_arr[i], std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle      = request_delivery_handle->get();
                    internal_resolutor.eu_packet_delivery_handle    = eu_packet_delivery_handle->get();
                    auto vectorized_delivery_handle                 = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz); 

                    for (size_t i = 0u; i < sz; ++i){
                        if (descendant_arr[i] == dg::pointer_limits<uma_ptr_t>::null_value()){
                            continue;
                        }

                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant_arr[i]);
                        uma_ptr_t dst_rep_addr  = dg::memult::region(dst_rcu_addr, dg::network_uma::memregion_size());
                        uma_ptr_t src_rep_addr  = dg::memult::region(src_rcu_addr, dg::network_uma::memregion_size()); 
                        auto key                = dg::utility::to_unique_representation(dst_rep_addr, src_rep_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), key, std::make_tuple(ptr_arr[i], descendant_arr[i]));
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
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr = dg::pointer_limits<uma_ptr_t>::null_value();
                                break;
                            }
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_addr = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(dst);
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
                dg::network_producer_consumer::DeliveryHandle<EndUserPacket> * eu_packet_delivery_handle;

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));

                    auto umamap_reacquirer              = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto dst_logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();
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

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_controller::decode_msgrfwd(dispatch_control);

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}})){
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_umaptr, dst_vd_id}, {src_logit_umaptr, src_vd_id}});
                        vma_ptr_t dst_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_logit_vmamap_reacquirer, dst_logit_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_vmaptr)){
                            
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        } 

                        dg::network_vmamap::reacquirer_reacquire_nothrow(dst_logit_vmamap_reacquirer, dst_logit_vmaptr);
                        dg::networK_vmamap::reacquirer_reacquire_nothrow(src_logit_vmamap_reacquirer, src_logit_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto dst_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(dst_logit_vmamap_reacquirer);
                            auto src_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            restrict_synchronizer.add(dst_logit_cudaptr, src_logit_cudaptr);
                            auto async_id           = dg::network_tileops_cuda_poly::async_fwd_mono(dst_logit_cudaptr, src_logit_cudaptr, tileops_dp_code);

                            if (!async_id.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                                continue;
                            }

                            synchronizer.add(async_id.value());
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto dst_logit_hostptr  = dg::network_vmamap::get_host_ptr(dst_logit_vmamap_reacquirer);
                            auto src_logit_hostptr  = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            dg::network_tileops_host_poly::fwd_mono(dst_logit_hostptr, src_logit_hostptr, tileops_dp_code);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_msgrfwd_init_status_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);

                        EndUserPacket eu_packet{};

                        eu_packet.kind          = EUPACKET_MSGRFWD_TRANSMIT;
                        eu_packet.content       = dg::network_compact_serializer::serialize<dg::string>(LogitData{logit_id, dg::network_tile_member_getsetter::get_msgrfwd_logit_nothrow(dst)});
                        eu_packet.dst           = msgr_dst;
                        eu_packet.retry_count   = msgr_retry_count;
                        eu_packet.urgency       = msgr_transmit_urgency;
                        eu_packet.comm          = msgr_transmit_comm;

                        dg::network_producer_consumer::delvrsrv_deliver(this->eu_packet_delivery_handle, eu_packet);

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(dst_observer_arr[j]));
                        }

                    }
                }
            };
    };

    class ForwardInitMsgrBwdSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;

        public:

            ForwardInitMsgrBwdSignalResolutor(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                              size_t delivery_capacity,
                                              size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                                 delivery_capacity(delivery_capacity),
                                                                                 vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto descendant_arr     = std::make_unique<uma_ptr_t[]>(sz); 
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
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = dg::pointer_limits<uma_ptr_t>::null_value();
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), lck_addr, std::make_tuple(ptr_arr[i], std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle = delivery_handle.get();
                    auto vectorized_delivery_handle = dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz);

                    if (!vectorized_delivery_handle.has_value()){
                        dg::network_log_stackdump::error(dg::network_exception::verbose(vectorized_delivery_handle.error()));
                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        if (descendant_arr[i] == dg::pointer_limits<uma_ptr_t>::null_value()){
                            continue;
                        }

                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant_arr[i]);
                        uma_ptr_t dst_lck_addr  = dg::memult::region(dst_rcu_addr, dg::network_uma::memregion_size());
                        uma_ptr_t src_lck_addr  = dg::memult::region(src_rcu_addr, dg::network_uma::memregion_size());
                        auto key                = dg::utility::to_unique_representation(dst_lck_addr, src_lck_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), key, std::make_tuple(ptr_arr[i], descendant_arr[i]));
                    }
                }
            }
        
        private:

            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t *> * data, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, fetching_addr]   = data[i];
                        init_status_t init_status   = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(dst);

                        switch (init_status){
                            case TILE_INIT_STATUS_EMPTY: [[fallthrough]]
                            case TILE_INIT_STATUS_ORPHANED: [[fallthrough]]
                            case TILE_INIT_STATUS_INITIALIZED: [[fallthrough]]
                            case TILE_INIT_STATUS_DECAYED:
                            {
                                *fetching_addr = dg::pointer_limits<uma_ptr_t>::null_value();
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(dst);
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
            
            //OK - tell me the solution for synchronized backwards + minimized fsys read to forward extract then I'll admit I'm wrong

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle; //we dont care about aesthetic that much Mom - it's about code management - things that could be removed are removed in one component

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * data_arr, size_t sz){

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));

                    auto umamap_reacquirer      = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto dst_vmamap_reacquirer  = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_vmamap_reacquirer  = dg::network_vmamap::reacquirer_raii_initialize();

                    dg::network_cuda_controller::CudaSynchronizer synchronizer{};
                    dg::network_controller::RestrictPointerSynchronizer restrict_synchronizer(synchronizer);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, src]                                             = data_arr[i];
                        operatable_id_t dst_operatable_id                           = dg::network_tile_member_getsetter::get_msgrbwd_operatable_id_nothrow(dst);
                        init_status_t dst_init_status                               = dg::network_tile_member_getsetter::get_msgrbwd_init_status_nothrow(dst);
                        uma_ptr_t dst_logit_addr                                    = dg::network_tile_member_getsetter::get_msgrbwd_logit_addr_nothrow(dst);
                        std::array<uma_ptr_t, OBSERVER_ARRAY_CAP> dst_observer_arr  = dg::network_tile_member_getsetter::get_msgrbwd_observer_array_nothrow(dst);
                        size_t dst_observer_arr_sz                                  = dg::network_tile_member_getsetter::get_msgrbwd_observer_array_size_nothrow(dst);
                        uma_ptr_t dst_src                                           = dg::network_tile_member_getsetter::get_msgrbwd_descendant_nothrow(dst);
                        dispatch_control_t dispatch_control                         = dg::network_tile_member_getsetter::get_msgrbwd_dispatch_control_nothrow(dst);
                        operatable_id_t src_operatable_id                           = dg::network_tile_member_getsetter::get_tile_operatable_id_nothrow(src);
                        init_status_t src_init_status                               = dg::network_tile_member_getsetter::get_tile_init_status_nothrow(src);
                        uma_ptr_t src_logit_addr                                    = dg::network_tile_member_getsetter::get_tile_logit_addr_nothrow(src); 

                        if (dst_src != src){
                            continue;
                        }

                        if (dst_operatable_id != src_operatable_id){
                            continue;
                        }

                        if (src_init_status != TILE_INIT_STATUS_INITIALIZED){
                            continue;
                        }

                        if (dst_init_status != TILE_INIT_STATUS_DECAYED){
                            continue;
                        }

                        auto [dst_vd_id, src_vd_id, dp_device, tileops_dp_code]     = dg::network_dispatch_control::decode_msgrbwd(dispatch_control); //we'll convert tuple -> struct later 

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{dst_logit_addr, dst_vd_id}, {src_logit_addr, src_vd_id}})){ //we assume the semantics of reacquirable == reachable vma_ptr_t(s) are unaffected - this needs to be documented
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                            // dg::network_vmamap::reacquirer_clear(dst_vmamap_reacquirer); //this is not mandatory
                            // dg::network_vmamap::reacquirer_clear(src_vmamap_reacquirer); //this is not mandatory
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{dst_logit_addr, dst_vd_id}, {src_logit_addr, src_vd_id}});
                        vma_ptr_t dst_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_vmaptr    = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(dst_vmamap_reacquirer, dst_vmaptr) || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_vmamap_reacquirer, src_vmaptr)){ //we assume ...
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire(dst_vmamap_reacquirer, dst_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(src_vmamap_reacquirer, src_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto dst_cuda_ptr = dg::network_vmamap::get_cuda_ptr(dst_vmamap_reacquirer);
                            auto src_cuda_ptr = dg::network_vmamap::get_cuda_ptr(src_vmamap_reacquirer);
                            restrict_synchronizer.add(dst_cuda_ptr, src_cuda_ptr); 
                            auto async_id = dg::network_tileops_cuda_poly::async_fwd_mono(async_device, dst_cuda_ptr, src_cuda_ptr, tileops_dp_code); //specify async device

                            if (!async_id.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                                continue;
                            }

                            synchronizer.add(async_id.value());
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            dg::network_tileops_host_poly::fwd_mono(dg::network_vmamap::get_host_ptr(dst_vmamap_reacquirer), dg::network_vmamap::get_host_ptr(src_vmamap_reacquirer), tileops_dp_code); //this requires exceptions
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }                        
                        }

                        for (size_t j = 0u; j < dst_observer_arr_sz; ++j){
                            dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_forward_init_signal(dst_observer_arr[j]));
                        }

                        set_init_status_msgrbwd_nothrow(dst, TILE_INIT_STATUS_INITIALIZED);
                    }
                }
            };
    };

    class ForwardInitImmuSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        public:

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                (void) ptr_arr;
            }
    };

    class ForwardInitSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

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

            ForwardInitSignalResolutor(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> leaf_resolutor,
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
                                       size_t immu_dispatch_sz):    leaf_resolutor(std::move(leaf_resolutor)),
                                                                    leaf_dispatch_sz(leaf_dispatch_sz),
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

                //resource leak is expected here - we don't want to communicate the error because it would mess up the code very badly - let's assume that everything is "might", "maybe", and set a timeout for that - like network packet
                //users of the engines wait for msgrfwds to be intiialized within a certain windows - or deallocate and move on
                //error handlings in producer-consumer situation is not encouraged - and it is not a good practice also - this is an error that should be controlled in users' code flow

                if (!dg::network_exception::conjunc_expect_has_value(leaf_delivery_handle, mono_delivery_handle, pair_delivery_handle, 
                                                                     uacm_delivery_handle, pacm_delivery_handle, extnsrc_delivery_handle, 
                                                                     extndst_delivery_handle, crit_delivery_handle, msgrfwd_delivery_handle, 
                                                                     msgrbwd_delivery_handle, immu_delivery_handle)){

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
                        case TILE_KIND_LEAF:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(leaf_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MONO:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(mono_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_PAIR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pair_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_UACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(uacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_PACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(pacm_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNSRC:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extnsrc_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNDST:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(extndst_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_CRIT:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(crit_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRFWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrfwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRBWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(msgrbwd_delivery_handle)->get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_IMMU:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(stdx::to_const_reference(immu_delivery_handle)->get(), ptr_arr[i]);
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
                    };
                }
            }
    };

    //

    class BackwardDoLeafSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const size_t vectorization_sz;
        
        public:

            BackwardDoLeafSignalResolutorV2(size_t vectorization_sz) noexcept: vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                InternalResolutor internal_resolutor{};
                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                for (size_t i = 0u; i < sz; ++i){
                    auto ptrchk = dg::network_tile_member_access::safecthrow_leaf_ptr_access(ptr_arr[i]);

                    if (!ptrchk.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                        continue;
                    }

                    uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_leaf_rcu_addr_nothrow(ptr_arr[i]);
                    uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_uma::memregion_size());

                    dg::network_producer_consumer::delvrsrv_deliver(delivery_handle.get(), lck_addr, ptr_arr[i]);
                }
            }

        private:

            struct InternalResolutor: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, uma_ptr_t>{

                void push(uma_ptr_t rcu_addr, uma_ptr_t * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(rcu_addr);

                    auto umamap_reacquirer          = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 2u>{});
                    auto grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();
                    auto logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();

                    dg::network_cuda_controller::CudaSynchronizer synchronizer{};
                    dg::network_controller::RestrictPointerSynchronizer restrict_synchronizer(synchronizer);

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

                        if (grad_status != TILE_GRAD_STATUS_HAS_VALUE){ //this seems fishy
                            continue;
                        }

                        auto [logit_vd_id, grad_vd_id, dp_device, tileops_dp_code] = dg::network_dispatch_control::decode_gradupdate_leaf(dispatch_control); //

                        if (!dg::network_uma::reacquirer_fixedsize_is_region_reacquirable(umamap_reacquirer, {{logit_umaptr, logit_vd_id}, {grad_umaptr, grad_vd_id}})){
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_fixedsize_reacquire_nothrow(umamap_reacquirer, {{logit_umaptr, logit_vd_id}, {grad_umaptr, grad_vd_id}});
                        vma_ptr_t logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(logit_vmamap_reacquirer, logit_vmaptr) || !dg::network_vmamap::reacquirer_is_region_reacquirable(grad_vmamap_reacquirer, grad_vmaptr)){
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire_nothrow(logit_vmamap_reacquirer, logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire_nothrow(grad_vmamap_reacquirer, grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(logit_vmamap_reacquirer);
                            auto grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(grad_vmamap_reacquirer);
                            restrict_synchronizer.add(logit_cudaptr, grad_cudaptr);
                            auto async_id       = dg::network_tileops_cuda_poly::async_grad_update_n_zero(async_device, logit_cudaptr, grad_cudaptr, tileops_dp_code); //placeholder 

                            if (!async_id.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                                continue;
                            }

                            synchronizer.add(async_id.value());
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            dg::network_tileops_host_poly::grad_update_n_zero(dg::network_vmamap::get_host_ptr(logit_vmamap_reacquirer), dg::network_vmamap::get_host_ptr(grad_vmamap_reacquirer), tileops_dp_code);
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_leaf_grad_status_nothrow(ptr, TILE_GRAD_STATUS_ZEROED);
                    }
                }
            };
    };

    class BackwardDoMonoSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;
        
        public:

            BackwardDoMonoSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<virtual_memory_event_t>> request_box,
                                            size_t delivery_capacity,
                                            size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity),
                                                                               vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));
                auto descendant_arr     = std::make_unique<uma_ptr_t[]>(sz);

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz));
                    
                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_mono_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = dg::pointer_limits<uma_ptr_t>::null_value();
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(ptr_arr[i], std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        if (descendant_arr[i] == dg::pointer_limits<uma_ptr-t>::null_value()){
                            continue;
                        }

                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_mono_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant_arr[i]);
                        uma_ptr_t dst_lck_addr  = dg::memult::region(dst_rcu_addr, dg::network_uma::memregion_size());
                        uma_ptr_t src_lck_addr  = dg::memult::region(src_rcu_addr, dg::network_uma::memregion_size());
                        auto key                = dg::utility::to_unique_representation(dst_lck_addr, src_lck_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(descendant_arr[i], ptr_arr[i]));
                    }
                }
            }
        
        private:
            
            struct InternalDescendantAddressFetcher: dg::network_producer_consumer::KVConsumerInterface<uma_ptr_t, std::tuple<uma_ptr_t, uma_ptr_t *>>{

                void push(uma_ptr_t rcu_addr, std::tuple<uma_ptr_t, uma_ptr_t *> * data_arr, size_t sz) noexcept{

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
                                *fetching_addr = dg::pointer_limits<uma_ptr_t>::null_value();
                                break;
                            }
                            case TILE_INIT_STATUS_INITIALIZED:
                            {
                                *fetching_addr = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(dst);
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

                void push(std::tuple<uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t> * ptr_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr));
                    
                    auto umamap_reacquirer              = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 3u>{});
                    auto dst_grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();
                    auto src_grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();

                    dg::network_cuda_controller::CudaSynchronizer synchronizer{};
                    dg::network_controller::RestrictPointerSynchronizer restrict_synchronizer(synchronizer); //add scope - fix the class - not clear responsibility

                    for (size_t i = 0u; i < sz; ++i){
                        auto [src, dst]                     = ptr_arr[i]; //the src - dst semantics need to be changed - we need to keep it like this for the sake of readability
                        uma_ptr_t dst_src                   = dg::network_tile_member_getsetter::get_mono_descendant_nothrow(dst);
                        operatable_id_t dst_operatable_id   = dg::network_tile_member_getsetter::get_mono_operatable_id_nothrow(dst);
                        init_status_t dst_init_status       = dg::network_tile_member_getsetter::get_mono_init_status_nothrow(dst);
                        // uma_ptr_t dst_logit_umaptr          = dg::network_tile_member_getsetter::get_mono_logit_addr_nothrow(dst);
                        uma_ptr_t dst_grad_umaptr           = dg::network_tile_member_getsetter::get_mono_grad_addr_nothrow(dst);
                        dispatch_control_t dispatch_control = dg::network_tile_member_getsetter::get_mono_bwd_dispatch_control_nothrow(dst);
                        // bool is_backpropable                = dg::network_tile_member_getsetter::has_gradient_nothrow(src); //I feel like this is a duct tape to the get_tile_<whatever> -> std::expected<result, exception_t> - we always should chk the exception_t for polymorphic access - not doing has_gradient_nothrow - this is bad

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
                            synchronizer.sync(); //I'll consider adding exceptions here - 
                            restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_reacquire(umamap_reacquirer, {{src_grad_umaptr, src_grad_vd_id}, {src_logit_umaptr, src_logit_vd_id}, {dst_grad_umaptr, dst_grad_vd_id}});
                        vma_ptr_t src_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t src_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{}); 

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(src_grad_vmamap_reacquirer, src_grad_vmaptr) || !dg::network_vmamap::reacquirer_is_region_reacquirable(src_logit_vmamap_reacquirer, src_logit_vmaptr),
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(dst_grad_vmamap_reacquirer, dst_grad_vmaptr)){
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire(src_grad_vmamap_reacquirer, src_grad_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(src_logit_vmamap_reacquirer, src_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dp_device)){
                            auto src_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(src_grad_vmamap_reacquirer);
                            auto src_logit_cudaptr  = dg::networK_vmamap::get_cuda_ptr(src_logit_vmamap_reacquirer);
                            auto dst_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);
                            restrict_synchronizer.add(src_grad_cudaptr, src_logit_cudaptr, dst_grad_cudaptr); 
                            auto async_id           = dg::network_tileops_cuda_poly::bwd_mono(src_grad_cudaptr, src_logit_cudaptr, dst_grad_cudaptr, tileops_dp_code, dg::value_if(src_grad_status == TILE_GRAD_STATUS_EMPTY, TILEOPS_OPERATION_ASSIGN, TILEOPS_OPERATION_ACCUM), TILEOPS_POSTOPERATION_ZERO);

                            if (!async_id.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                                continue;
                            }

                            synchronizer.add(async_id.value());
                        } else if (dg::network_dispatch_control::is_host_dispatch(dp_device)){
                            auto src_grad_hostptr   = dg::network_vmamap::get_host_ptr(src_grad_vmamap_reacquirer);
                            auto src_logit_hostptr  = dg::network_vmamap::get_host_ptr(src_logit_vmamap_reacquirer);
                            auto dst_grad_hostptr   = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);
                            dg::network_tileops_host_poly::bwd_mono(src_grad_hostptr, src_logit_hostptr, dst_grad_hostptr, tileops_dp_code, dg::value_if(src_grad_status == TILE_GRAD_STATUS_EMPTY, TILEOPS_OPERATION_ASSIGN, TILEOPS_OPERATION_ACCUM), TILEOPS_POSTOPERATION_ZERO); //add exceptions
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
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
                    }
                }
            };
    };

    class BackwardDoPairSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        private:

            const std::shared_ptr<dg::network_producer_consuner::ConsumerInterface<uma_ptr_t>> request_box;
            const size_t delivery_capacity;
            const size_t vectorization_sz;
        
        public:

            BackwardDoPairSignalResolutorV2(std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>> request_box,
                                            size_t delivery_capacity,
                                            size_t vectorization_sz) noexcept: request_box(std::move(request_box)),
                                                                               delivery_capacity(delivery_capacity),
                                                                               vectorization_sz(vectorization_sz){}

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));
                auto descendant_arr     = std::make_unique<std::optional<std::tuple<uma_ptr_t, uma_ptr_t>>[]>(sz);

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_pair_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(ptr_arr[i], std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        if (!descendant_arr[i].has_value()){
                            continue;
                        }

                        auto [lhs_ptr, rhs_ptr] = descendant_arr[i].value();
                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_pair_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lhs_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(lhs_ptr);
                        uma_ptr_t rhs_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(rhs_ptr);
                        uma_ptr_t dst_rep_addr  = dg::memult::region(dst_rcu_addr, dg::network_uma::memregion_size());
                        uma_ptr_t lhs_rep_addr  = dg::memult::region(lhs_rcu_addr, dg::network_uma::memregion_size());
                        uma_ptr_t rhs_rep_addr  = dg::memult::region(rhs_rcu_addr, dg::network_uma::memregion_size());
                        auto key                = dg::utility::to_unique_representation(dst_rep_addr, lhs_rep_addr, rhs_rep_addr); //this might mess up the ordering - this is precisely why I need to weight the pros and cons of this

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(ptr_arr[i], lhs_ptr, rhs_ptr));
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

            struct InternalResolutor: dg::network_producer_consumer:KVConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t>, std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t>>{

                dg::network_producer_consumer::DeliveryHandle<virtual_memory_event_t> * request_delivery_handle;

                void push(std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t> lck_addr, std::tuple<uma_ptr_t, uma_ptr_t, uma_ptr_t> * data_arr, size_t sz) noexcept{

                    dg::network_memops_uma::memlock_guard mem_grd(std::get<0>(lck_addr), std::get<1>(lck_addr), std::get<2>(lck_addr));

                    auto umamap_reacquirer              = dg::network_uma::reacquirer_fixedsize_raii_initialize(std::integral_constant<size_t, 5u>{});
                    auto dst_grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();
                    auto lhs_grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();
                    auto rhs_grad_vmamap_reacquirer     = dg::network_vmamap::reacquirer_raii_initialize();
                    auto lhs_logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();
                    auto rhs_logit_vmamap_reacquirer    = dg::network_vmamap::reacquirer_raii_initialize();

                    dg::network_cuda_controller::CudaSynchronizer synchronizer{};
                    dg::network_controller::RestrictPointerSynchronizer restrict_synchronizer(synchronizer);

                    for (size_t i = 0u; i < sz; ++i){
                        auto [dst, lhs, rhs] = data_arr[i];

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

                        auto dispatch_info = dg::network_dispatch_control::decode_bwd_pair(dispatch_control);

                        if (!dg::network_uma::reacquirer_is_reacquirable(umamap_reacquirer, {{lhs_logit_umaptr, dispatch_info.lhs_logit_vd_id}, {lhs_grad_umaptr, dispatch_info.lhs_grad_vd_id}, {rhs_logit_umaptr, dispatch_info.rhs_logit_vd_id}, {rhs_grad_umaptr, dispatch_info.rhs_grad_vd_id}, {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}})){;
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_uma::reacquirer_reacquire(umamap_reacquirer, {{lhs_logit_umaptr, dispatch_info.lhs_logit_vd_id}, {lhs_grad_umaptr, dispatch_info.lhs_grad_vd_id}, {rhs_logit_umaptr, dispatch_info.rhs_logit_vd_id}, {rhs_grad_umaptr, dispatch_info.rhs_grad_vd_id}, {dst_grad_umaptr, dispatch_info.dst_grad_vd_id}});
                        vma_ptr_t lhs_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 0u>{});
                        vma_ptr_t lhs_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 1u>{});
                        vma_ptr_t rhs_logit_vmaptr  = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 2u>{});
                        vma_ptr_t rhs_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 3u>{});
                        vma_ptr_t dst_grad_vmaptr   = dg::network_uma::get_vma_ptr(umamap_reacquirer, std::integral_constant<size_t, 4u>{});

                        if (!dg::network_vmamap::reacquirer_is_region_reacquirable(lhs_logit_vmamap_reacquirer, lhs_logit_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(lhs_grad_vmamap_reacquirer, lhs_grad_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(rhs_logit_vmamap_reacquirer, rhs_logit_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(rhs_grad_vmamap_reacquirer, rhs_grad_vmaptr)
                            || !dg::network_vmamap::reacquirer_is_region_reacquirable(dst_grad_vmamap_reacquirer, dst_grad_vmaptr)){
                                
                            synchronizer.sync();
                            restrict_synchronizer.clear();
                        }

                        dg::network_vmamap::reacquirer_reacquire(lhs_logit_vmamap_reacquirer, lhs_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(lhs_grad_vmamap_reacquirer, lhs_grad_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(rhs_logit_vmamap_reacquirer, rhs_logit_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(rhs_grad_vmamap_reacquirer, rhs_grad_vmaptr);
                        dg::network_vmamap::reacquirer_reacquire(dst_grad_vmamap_reacquirer, dst_grad_vmaptr);

                        if (dg::network_dispatch_control::is_cuda_dispatch(dispatch_info.dp_device)){  //hmm - this is difficult - let's assume they must be on the same platform for now - to not overcomplicate things
                            auto lhs_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(lhs_logit_vmamap_reacquirer);
                            auto lhs_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(lhs_grad_vmamap_reacquirer);
                            auto rhs_logit_cudaptr  = dg::network_vmamap::get_cuda_ptr(rhs_logit_vmamap_reacquirer);
                            auto rhs_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(rhs_grad_vmamap_reacquirer);
                            auto dst_grad_cudaptr   = dg::network_vmamap::get_cuda_ptr(dst_grad_vmamap_reacquirer);

                            restrict_synchronizer.add(lhs_logit_cudaptr, lhs_grad_cudaptr, rhs_logit_cudaptr, rhs_grad_cudaptr, dst_grad_cudaptr);
                            auto left_task          = dg::network_tileops_cuda_poly::async_make_task(this->async_device, );

                            if (!left_task.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(left_task.error()));
                                continue;
                            }

                            auto right_task         = dg::network_tileops_cuda_poly::async_make_task(this->async_device, );

                            if (!right_task.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(right_task.error()));
                                continue;
                            }

                            //we must rely on callee's asynchronous atomicity - we offload the bug there 

                            auto async_id           = dg::network_tileops_cuda_poly::async_exec(std::move(left_task.value()), std::move(right_task.value()));

                            if (!async_id.has_value()){
                                dg::network_log_stackdump::error_fast(dg::network_exception::verbose(async_id.error()));
                                continue;
                            }

                            synchronizer.add(async_id.value());
                        } else if (dg::network_dispatch_control::is_host_dispatch(dispatch_info.dp_device)){
                            auto lhs_logit_hostptr  = dg::network_vmamap::get_host_ptr(lhs_logit_vmamap_reacquirer);
                            auto lhs_grad_hostptr   = dg::network_vmamap::get_host_ptr(lhs_grad_vmamap_reacquirer);
                            auto rhs_logit_hostptr  = dg::network_vmamap::get_host_ptr(rhs_logit_vmamap_reacquirer);
                            auto rhs_grad_hostptr   = dg::network_vmamap::get_host_ptr(rhs_grad_vmamap_reacquirer);
                            auto dst_grad_hostptr   = dg::network_vmamap::get_host_ptr(dst_grad_vmamap_reacquirer);

                            dg::network_tileops_host_poly::bwd_pair_lhs();
                            dg::network_tileops_host_poly::bwd_pair_rhs();
                        } else{
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                        }

                        dg::network_tile_member_getsetter::set_tile_grad_status_nothrow(lhs, TILE_GRAD_STATUS_HAS_VALUE);
                        dg::network_tile_member_getsetter::set_tile_grad_status_nothrow(rhs, TILE_GRAD_STATUS_HAS_VALUE);
                        dg::network_tile_member_getsetter::set_pair_grad_status_nothrow(dst, TILE_GRAD_STATUS_ZEROED);
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(lhs));
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(rhs));
                    }
                }
            };         
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

    class BackwardDoExtnSrcSignalResolutorV2: public virtual dg::network_produer_consumer::ConsumerInterface<uma_ptr_t>{

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

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto descendant_localcounterpart_arr    = std::make_unique<std::optional<std::tuple<uma_ptr_t, uma_ptr_t>>[]>(sz);
                auto delivery_handle                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get()));

                {
                    InternalDescendantAddressFetcher fetcher{};
                    fetcher.alias_getter            = this->alias_getter->get();
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extnsrc_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_localcounterpart_arr[i] = std::nullopt;
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(ptr_arr[i], std::next(descendant_localcounterpart_arr.get(), i)));
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
                        uma_ptr_t dst_rcu_addr              = dg::network_tile_member_getsetter::get_extnsrc_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t descendant_rep_addr       = dg::memult::region(descendant_rcu_addr, dg::network_uma::memregion_size());
                        uma_ptr_t localcounterpart_rep_addr = dg::memult::region(localcounterpart_rcu_addr, dg::network_uma::memregion_size());
                        uma_ptr_t dst_rep_addr              = dg::memult::region(dst_rcu_addr, dg::network_uma::memregion_size());
                        auto key                            = dg::utility::to_unique_representation(descendant_rep_addr, localcounterpart_rep_addr, dst_rep_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(descendant, ptr_arr[i], local_counterpart));
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
                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle->get(), dg::network_memcommit_factory::make_event_backward_do_signal(src));
                    }
                }
            };
    };

    class BackwardDoExtnDstSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

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

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle  = delivery_handle.get();
                    internal_resolutor.uma_ip_retriever         = this->uma_ip_retriever->get();
                    internal_resolutor.host_ip_retriever        = this->host_ip_retriever->get();
                    auto vectorized_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_extndst_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_extndst_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
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

    class BackwardDoMsgrFwdSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

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

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto descendant_arr     = std::make_unique<uma_ptr_t[]>(sz);
                auto delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->delivery_capacity));

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrfwd_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, ptr_arr[i]);
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

                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_msgrfwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant_arr[i]);
                        uma_ptr_t dst_rep_addr  = dg::memult::region(dst_rcu_addr, dg::network_uma::memregion_size());
                        uma_ptr_t src_rep_addr  = dg::memult::region(src_rcu_addr, dg::network_uma::memregion_size());
                        auto key                = dg::utility::to_unique_representation(dst_rep_addr, src_rep_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(ptr_arr[i], descendant_arr[i]));
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
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
                    }
                }
            };
    };

    class BackwardDoMsgrBwdSignalResolutorV2: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

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

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto descendant_arr             = std::make_unique<uma_ptr_t[]>(sz);
                auto request_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->request_box.get(), this->request_delivery_capacity));
                auto eu_packet_delivery_handle  = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->eu_packet_box.get(), this->eu_packet_delivery_capacity));

                {
                    InternalDescendantAddressFetcher fetcher{};
                    auto vectorized_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&fetcher, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        auto ptrchk = dg::network_tile_member_access::safecthrow_msgrbwd_ptr_access(ptr_arr[i]);

                        if (!ptrchk.has_value()){
                            descendant_arr[i] = dg::pointer_limits<uma_ptr_t>::null_value();
                            dg::network_log_stackdump::error_fast(dg::network_exception::verbose(ptrchk.error()));
                            continue;
                        }

                        uma_ptr_t rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t lck_addr  = dg::memult::region(rcu_addr, dg::network_memops_uma::memlock_region_size());

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), lck_addr, std::make_tuple(ptr_arr[i], std::next(descendant_arr.get(), i)));
                    }
                }

                {
                    InternalResolutor internal_resolutor{};
                    internal_resolutor.request_delivery_handle      = request_delivery_handle.get();
                    internal_resolutor.eu_packet_delivery_handle    = eu_packet_delivery_handle.get();
                    auto vectorized_delivery_handle                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_kv_raiihandle(&internal_resolutor, this->vectorization_sz));

                    for (size_t i = 0u; i < sz; ++i){
                        if (descendant_arr[i] == dg::pointer_limits<uma_ptr_t>::null_value()){
                            continue;
                        }

                        uma_ptr_t src_rcu_addr  = dg::network_tile_member_getsetter::get_tile_rcu_addr_nothrow(descendant_arr[i]);
                        uma_ptr_t dst_rcu_addr  = dg::network_tile_member_getsetter::get_msgrbwd_rcu_addr_nothrow(ptr_arr[i]);
                        uma_ptr_t src_rep_addr  = dg::memult::region(src_rcu_addr, dg::network_uma::memregion_size());
                        uma_ptr_t dst_rep_addr  = dg::memult::region(dst_rcu_addr, dg::network_uma::memregion_size());
                        auto key                = dg::utility::to_unique_representation(src_rep_addr, dst_rep_addr);

                        dg::network_producer_consumer::delvrsrv_deliver(vectorized_delivery_handle.get(), key, std::make_tuple(ptr_arr[i], descendant_arr[i]));
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
                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, dg::network_memcommit_factory::make_event_backward_do_signal(src));
                        // dg::network_producer_consumer::delvrsrv_deliver(this->eu_packet_delivery_handle, std::move(eu_packet));
                    }

                }
            };
    };

    class BackwardDoImmuSignalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<uma_ptr_t>{

        public:

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                (void) ptr_arr;
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

            void push(uma_ptr_t * ptr_arr, size_t sz) noexcept{

                auto leaf_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->leaf_resolutor.get(), this->leaf_dispatch_sz));
                auto mono_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->mono_resolutor.get(), this->mono_dispatch_sz));
                auto pair_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pair_resolutor.get(), this->pair_dispatch_sz));
                auto uacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->uacm_resolutor.get(), this->uacm_dispatch_sz));
                auto pacm_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->pacm_resolutor.get(), this->pacm_dispatch_sz));
                auto extnsrc_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extnsrc_resolutor.get(), this->extnsrc_dispatch_sz));
                auto extndst_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->extndst_resolutor.get(), this->extndst_dispatch_sz));
                auto crit_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->crit_resolutor.get(), this->crit_dispatch_sz));
                auto msgrfwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrfwd_resolutor.get(), this->msgrfwd_dispatch_sz));
                auto msgrbwd_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->msgrbwd_resolutor.get(), this->msgrbwd_dispatch_sz));
                auto immu_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->immu_resolutor.get(), this->immu_dispatch_sz));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<tile_kind_t, exception_t> tile_kind = dg::network_tile_member_getsetter::get_tile_kind(ptr_arr[i]);

                    if (!tile_kind.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(tile_kind.error()));
                        continue;
                    }

                    switch (tile_kind.value()){
                        case TILE_KIND_LEAF:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(leaf_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MONO:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(mono_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_PAIR:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pair_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_UACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(uacm_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_PACM:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(pacm_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNSRC:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extnsrc_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_EXTNDST:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(extndst_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_CRIT:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(crit_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRFWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrfwd_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_MSGRBWD:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(msgrbwd_delivery_handle.get(), ptr_arr[i]);
                            break;
                        }
                        case TILE_KIND_IMMU:
                        {
                            dg::network_producer_consumer::delvrsrv_deliver(immu_delivery_handle.get(), ptr_arr[i]);
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

                auto fwd_ping_signal_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_ping_signal_virtual_consumer, this->fwd_ping_delivery_capacity));
                auto fwd_pong_request_delivery_handle   = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_pong_request_virtual_consumer, this->fwd_pong_request_delivery_capacity));
                auto fwd_pong_signal_delivery_handle    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_pong_signal_virtual_consumer, this->fwd_pong_signal_delivery_capacity));
                auto fwd_do_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(&fwd_do_virtual_consumer, this->fwd_do_delivery_capacity));
                auto bwd_do_delivery_handle             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(&bwd_do_virtual_consumer, this->bwd_do_delivery_capacity));

                for (size_t i = 0u; i < virtual_memory_event_sz; ++i){
                    memory_event_kind_t event_kind = dg::network_memcommit_factory::read_event_kind(virtual_memory_event_arr[i]);

                    switch (event_kind){
                        case dg::network_memcommit_factory::event_kind_forward_ping_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_ping_signal_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_forward_pong_request:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_pong_request_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_forward_pong_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_pong_signal_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_forward_do_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_do_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        case dg::network_memcommit_factory::event_kind_backward_do_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(bwd_do_delivery_handle.get(), std::move(virtual_memory_event_arr[i]));
                            break;
                        default:
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            } else{
                                std::unreachable();
                            }
                    }
                }

                return true;
            }
    };
}

#endif
