#ifndef __DG_NETWORK_ALLOCATION_H__
#define __DG_NETWORK_ALLOCATION_H__

//define HEADER_CONTROL 4

#include "dg_heap/heap.h"
#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <atomic>
#include <thread>
#include "assert.h"
#include <bit>
#include <vector>
#include "stdx.h"
#include "network_memult.h"
#include "network_exception.h"
#include "network_concurrency.h"
#include "network_randomizer.h"
#include "network_producer_consumer.h"
#include <memory>
#include "network_stack_allocation.h"
#include "network_trivial_serializer.h"

namespace dg::network_allocation{
    
    using ptr_type              = uint64_t;
    using alignment_type        = uint16_t;
    using interval_type         = dg::heap::types::interval_type; 
    using heap_sz_type          = dg::heap::types::store_type;

    // static inline constexpr size_t PTROFFS_BSPACE               = sizeof(uint32_t) * CHAR_BIT;
    // static inline constexpr size_t PTRSZ_BSPACE                 = sizeof(uint16_t) * CHAR_BIT;
    // static inline constexpr size_t ALLOCATOR_ID_BSPACE          = sizeof(uint16_t) * CHAR_BIT;
    // static inline constexpr size_t ALIGNMENT_BSPACE             = sizeof(uint16_t) * CHAR_BIT;
    // static inline constexpr ptr_type NETALLOC_NULLPTR           = ptr_type{0u}; 
    // static inline constexpr size_t DEFLT_ALIGNMENT              = alignof(double);
    // static inline constexpr size_t LEAF_SZ                      = 8u;
    // static inline constexpr size_t LEAST_GUARANTEED_ALIGNMENT   = LEAF_SZ;
 
    // static_assert(PTROFFS_BSPACE + PTRSZ_BSPACE + ALLOCATOR_ID_BSPACE + ALIGNMENT_BSPACE <= sizeof(ptr_type) * CHAR_BIT);
    // static_assert(-1 == ~0);
    // static_assert(!NETALLOC_NULLPTR);

    //we would want to batch things, use std::mutex
    //and use another affined allocators to further affine things 
    //this Allocator guarantees no fragmentation if the allocation node lifetime is below the half cap threshold (switchfoot)
    //that Allocator guarantees to free allocations on time, punctually
    //alright, something went wrong

    class GCInterface{

        public:

            virtual ~GCInterface() noexcept = default;
            virtual void gc() noexcept = 0;
    };

    class HeapAllocator{

        private:

            std::unique_ptr<char[], decltype(&std::free)> management_buf;
            std::unique_ptr<dg::heap::core::Allocatable> allocator;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> allocation_base_sz; 

        public:

            HeapAllocator(std::unique_ptr<char[], decltype(&std::free)> management_buf,
                          std::unique_ptr<dg::heap::core::Allocatable> allocator,
                          std::unique_ptr<std::mutex> mtx,
                          stdx::hdi_container<size_t> allocation_base_sz) noexcept: management_buf(std::move(management_buf)),
                                                                                    allocator(std::move(allocator)),
                                                                                    mtx(std::move(mtx)),
                                                                                    allocation_base_sz(allocation_base_sz){}

             void alloc(size_t * blk_arr, size_t blk_arr_sz, std::optional<interval_type> * interval_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < blk_arr_sz; ++i){
                    interval_arr[i] = this->allocator->alloc(stdx::wrap_safe_integer_cast(blk_arr[i]));
                }
             }

             void free(interval_type * interval_arr, size_t sz) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    this->allocator->free(interval_arr[i]);
                }
             }

             void gc() noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                try{
                    this->allocator = dg::heap::user_interface::get_allocator_x(this->management_buf.get());
                } catch (...){
                    std::abort();
                }
             }

             auto base_size() const noexcept -> size_t{

                return this->allocation_base_sz.value;
             }
    };

    //there are too many atomic operations
    //it's fine

    class SemiAutoGCHeapAllocator{

        private:

            std::unique_ptr<HeapAllocator> base;

            stdx::hdi_container<std::atomic<size_t>> allocation_counter;
            stdx::hdi_container<size_t> allocation_counter_exp2_invoke_threshold;

            stdx::hdi_container<std::atomic<std::chrono::time_point<std::chrono::high_resolution_clock>>> last_gc;
            stdx::hdi_container<std::chrono::nanoseconds> gc_dur; 
            stdx::hdi_container<size_t> temporal_gc_pow2_dice_sz; 

        public:

            SemiAutoGCHeapAllocator(std::unique_ptr<HeapAllocator> base,
                                    size_t allocation_counter,
                                    size_t allocation_counter_exp2_invoke_threshold,
                                    std::chrono::time_point<std::chrono::high_resolution_clock> last_gc,
                                    std::chrono::nanoseconds gc_dur,
                                    size_t temporal_gc_pow2_dice_sz) noexcept: base(std::move(base)),
                                                                               allocation_counter(stdx::hdi_container<std::atomic<size_t>>{std::atomic<size_t>(allocation_counter)}),
                                                                               allocation_counter_exp2_invoke_threshold(stdx::hdi_container<size_t>{allocation_counter_exp2_invoke_threshold}),
                                                                               last_gc(stdx::hdi_container<std::atomic<std::chrono::time_point<std::chrono::high_resolution_clock>>>{std::atomic<std::chrono::time_point<std::chrono::high_resolution_clock>>(last_gc)}),
                                                                               gc_dur(stdx::hdi_container<std::chrono::nanoseconds>{gc_dur}),
                                                                               temporal_gc_pow2_dice_sz(stdx::hdi_container<size_t>{temporal_gc_pow2_dice_sz}){}

            void alloc(size_t * blk_arr, size_t blk_arr_sz, std::optional<interval_type> * interval_arr) noexcept{
                
                this->base->alloc(blk_arr, blk_arr_sz, interval_arr);
                size_t new_allocation_blk_sz = 0u;

                for (size_t i = 0u; i < blk_arr_sz; ++i){
                    if (interval_arr[i].has_value()){
                        new_allocation_blk_sz += blk_arr[i];
                    }
                }

                this->update_gc_sensor(new_allocation_blk_sz);
            }

            void free(interval_type * interval_arr, size_t sz) noexcept{

                this->base->free(interval_arr, sz);
            }

            void gc(){

                this->base->gc();
            }

            auto base_size() const noexcept -> size_t{

                return this->base->base_size();
            }

        private:

            void update_gc_sensor(size_t new_blk_sz) noexcept{

                size_t now_allocation_blk_sz    = this->allocation_counter.value.fetch_add(new_blk_sz, std::memory_order_relaxed);
                size_t then_allocation_blk_sz   = now_allocation_blk_sz + new_blk_sz;
                size_t now_idx                  = now_allocation_blk_sz >> this->allocation_counter_exp2_invoke_threshold.value;
                size_t then_idx                 = then_allocation_blk_sz >> this->allocation_counter_exp2_invoke_threshold.value;

                //atomic operation crossed the border
                if (now_idx != then_idx){
                    this->base->gc();
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    this->last_gc.value.exchange(std::chrono::high_resolution_clock::now(), std::memory_order_relaxed);
                    return;
                }

                size_t dice_value               = dg::network_randomizer::randomize_int<size_t>() & (this->temporal_gc_pow2_dice_sz.value - 1u);

                if (dice_value == 0u){
                    auto now        = std::chrono::high_resolution_clock::now();
                    auto expiry     = this->last_gc.value.load(std::memory_order_relaxed) + this->gc_dur.value;

                    if (now >= expiry){
                        this->base->gc();
                        std::atomic_signal_fence(std::memory_order_seq_cst);
                        this->last_gc.value.exchange(now, std::memory_order_relaxed);
                        return;
                    }
                }
            }
    };

    class MultiThreadUniformHeapAllocator{

        private:

            std::vector<std::unique_ptr<SemiAutoGCHeapAllocator>> allocator_vec;
            size_t malloc_vectorization_sz;
            size_t free_vectorization_sz;
            size_t heap_allocator_pow2_exp_base_sz; //let me think of how to split this 

        public:

            MultiThreadUniformHeapAllocator(std::vector<std::unique_ptr<SemiAutoGCHeapAllocator>> allocator_vec,
                                            size_t malloc_vectorization_sz,
                                            size_t free_vectorization_sz,
                                            size_t heap_allocator_pow2_exp_base_sz) noexcept: allocator_vec(std::move(allocator_vec)),
                                                                                              malloc_vectorization_sz(malloc_vectorization_sz),
                                                                                              free_vectorization_sz(free_vectorization_sz),
                                                                                              heap_allocator_pow2_exp_base_sz(heap_allocator_pow2_exp_base_sz){}

            void alloc(size_t * blk_arr, size_t blk_arr_sz, std::optional<interval_type> * rs) noexcept{

                assert(stdx::is_pow2(allocator_vec.size()));

                size_t allocator_idx                = dg::network_concurrency::this_thread_idx() & (this->allocator_vec.size() - 1u);
                auto internal_resolutor             = InternalMallocFeedResolutor{};
                internal_resolutor.dst              = &this->allocator_vec[allocator_idx];
                internal_resolutor.allocation_off   = allocator_idx << this->heap_allocator_pow2_exp_base_sz; 

                size_t trimmed_feed_cap             = std::min(this->malloc_vectorization_sz, blk_arr_sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&internal_resolutor, trimmed_feed_cap); //yeah, this might be problematic, our header control is out of quack
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception::remove_expected(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&internal_resolutor, trimmed_feed_cap, feeder_mem.get()));

                for (size_t i = 0u; i < blk_arr_sz; ++i){
                    auto feed_arg   = InternalMallocFeedArgument{};
                    feed_arg.blk_sz = blk_arr[i];
                    feed_arg.rs     = std::next(rs, i);

                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), feed_arg);
                }
            }

            void free(interval_type * ptr_arr, size_t ptr_arr_sz) noexcept{

                auto internal_resolutor             = InternalFreeFeedResolutor{};
                internal_resolutor.dst              = &this->allocator_vec;

                size_t trimmed_keyvalue_feed_cap    = std::min(this->free_vectorization_sz, ptr_arr_sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception::remove_expected(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&internal_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                for (size_t i = 0u; i < ptr_arr_sz; ++i){
                    auto [base_off, excl_sz]    = ptr_arr[i]; 
                    size_t allocator_idx        = base_off >> this->heap_allocator_pow2_exp_base_sz; //this is confusing
                    size_t allocator_off        = allocator_idx << this->heap_allocator_pow2_exp_base_sz; 
                    auto actual_ptr             = interval_type{std::make_pair(base_off - allocator_off, excl_sz)};

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), allocator_idx, actual_ptr);
                }
            }

            void gc(size_t thr_idx) noexcept{ // this might be a bottleneck if more than 1024 concurrent allocators are in use - this is not likely going to be the case - if a computer has more than 1024 cores - it's something wrong with the computer

                size_t allocator_idx = thr_idx & (this->allocator_vec.size() - 1u);
                this->allocator_vec[allocator_idx]->gc();
            }

            void gc_all() noexcept{

                for (size_t i = 0u; i < this->allocator_vec.size(); ++i){
                    this->allocator_vec[i]->gc();
                }
            }

            auto base_size() const noexcept -> size_t{

                return static_cast<size_t>(this->allocator_vec.size()) << this->heap_allocator_pow2_exp_base_sz;
            }

        private:

            struct InternalMallocFeedArgument{
                size_t blk_sz;
                std::optional<interval_type> * rs;
            };

            struct InternalMallocFeedResolutor: dg::network_producer_consumer::ConsumerInterface<InternalMallocFeedArgument>{

                std::unique_ptr<SemiAutoGCHeapAllocator> * dst;
                size_t allocation_off;

                void push(std::move_iterator<InternalMallocFeedArgument *> data_arr, size_t data_arr_sz) noexcept{

                    auto base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<size_t[]> blk_arr(data_arr_sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::optional<interval_type>[]> rs_arr(data_arr_sz);

                    for (size_t i = 0u; i < data_arr_sz; ++i){
                        blk_arr[i] = base_data_arr[i].blk_sz;
                    }

                    (*this->dst)->alloc(blk_arr.get(), data_arr_sz, rs_arr.get());

                    for (size_t i = 0u; i < data_arr_sz; ++i){
                        if (!rs_arr[i].has_value()){
                            *base_data_arr[i].rs = std::nullopt;
                            continue;
                        }

                        *base_data_arr[i].rs = std::make_pair(rs_arr[i]->first + this->allocation_off, rs_arr[i]->second);
                    }
                }
            };

            struct InternalFreeFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, interval_type>{

                std::vector<std::unique_ptr<SemiAutoGCHeapAllocator>> * dst;

                void push(const size_t& allocator_idx, std::move_iterator<interval_type *> data_arr, size_t sz) noexcept{

                    (*this->dst)[allocator_idx]->free(data_arr.base(), sz);
                }
            };
    };

    class MultiThreadWrappedGarbageCollector: public virtual GCInterface{

        private:

            std::shared_ptr<MultiThreadUniformHeapAllocator> base;
        
        public:

            MultiThreadWrappedGarbageCollector(std::shared_ptr<MultiThreadUniformHeapAllocator> base) noexcept: base(std::move(base)){}

            void gc() noexcept{

                base->gc_all();
            }
    };

    class SemiAutoWrappedGarbageCollector: public virtual GCInterface{

        private:

            std::shared_ptr<SemiAutoGCHeapAllocator> base;
        
        public:

            SemiAutoWrappedGarbageCollector(std::shared_ptr<SemiAutoGCHeapAllocator> base) noexcept: base(std::move(base)){}

            void gc() noexcept{

                this->base->gc();
            }
    };

    class NaiveBumpAllocator{

        private:

            char * buf;
            size_t sz;
        
        public:

            NaiveBumpAllocator() noexcept: buf(nullptr), 
                                      sz(0u){}

            NaiveBumpAllocator(char * buf, size_t sz) noexcept: buf(buf), 
                                                           sz(sz){}

            inline auto malloc(size_t blk_sz) noexcept -> std::expected<void *, exception_t>{

                if (blk_sz > this->sz){
                    return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                }

                char * rs   = this->buf;
                std::advance(this->buf, blk_sz);
                this->sz    -= blk_sz;

                return static_cast<void *>(rs);
            }

            inline auto decommission() noexcept -> std::pair<void *, size_t>{

                auto rs     = std::make_pair(static_cast<void *>(this->buf), this->sz);
                this->buf   = nullptr;
                this->sz    = 0u;

                return rs;
            }
    };

    //we've been working on critical paths for longer than we should have, we'll add compile-time measurements
    //but for now, this allocation allocator is industry-grade
    //no fragmenetation, worst case bound of 5% of total memory
    //maximum overhead from smallbin is 1MB 
    //fast free + fast alloc
    //good temporal locality by using backreferences + window
    //fast malloc
    //good branching
    //good binary size
    //good inlinability of malloc functions 
    //we haven't done the static memory indirections yet (we wont be polluting), we wont go there unless there is a proved performance constraint 
    //we'll move on to implementing other functions, this is giving me headache, the engineering practices were not reflected in the components due to a lot of contraints
    //we'll be adding compile-time measurements, it's been hard because we broke practices for performance
    //we are proud fellas, it's actually hard to write this, I have to admit, it took me months

    template <class BlkSizeOperatableType, size_t HEAP_LEAF_UNIT_ALLOCATION_SZ, size_t POW2_PUNCTUAL_CHECK_INTERVAL_SIZE, size_t MINIMUM_ALLOCATION_BLK_SZ, size_t MAXIMUM_SMALLBIN_BLK_SZ, size_t SMALLBIN_PROBE_SZ>
    class DGStdAllocatorMetadata{

        public:

            //static_asserts();

            using blk_sz_operatable_t   = BlkSizeOperatableType;
            using sz_header_t           = uint16_t;
            using vrs_ctrl_header_t     = uint16_t;
            using largebin_sz_header_t  = uint64_t; 

            // static inline constexpr size_t ALLOCATION_HEADER_SZ             = sizeof(sz_header_t) + sizeof(vrs_ctrl_header_t);  
            // static inline constexpr size_t LARGEBIN_ALLOCATION_HEADER_SZ    = sizeof(largebin_sz_header_t) + ALLOCATION_HEADER_SZ;  //we need to make sure that the LARGEBIN_ALLOCATION_HEADER_SZ is 8 bytes aligned, this is to achieve the all user_ptrs are at least 8-bytes aligned (we trick this by offseting the buf) 

            static consteval auto get_heap_leaf_unit_allocation_size() noexcept -> size_t{

                return HEAP_LEAF_UNIT_ALLOCATION_SZ;
            }

            static consteval auto get_minimum_allocation_blk_size() noexcept -> size_t{

                return MINIMUM_ALLOCATION_BLK_SZ;
            }

            static consteval auto get_maximum_smallbin_blk_size() noexcept -> size_t{

                return MAXIMUM_SMALLBIN_BLK_SZ;
            }

            static consteval auto get_smallbin_probe_size() noexcept -> size_t{

                return SMALLBIN_PROBE_SZ;
            }

            static constexpr auto get_alignment_size() noexcept -> size_t{

                return HEAP_LEAF_UNIT_ALLOCATION_SZ;                
            }

            static consteval auto get_allocation_header_size() noexcept -> size_t{

                return sizeof(sz_header_t) + sizeof(vrs_ctrl_header_t);
            }

            static consteval auto get_largebin_allocation_header_size() noexcept -> size_t{

                return sizeof(largebin_sz_header_t) + get_allocation_header_size(); //+ padding etc.
            }

            static consteval auto get_pow2_punctual_check_interval_size() noexcept -> size_t{

                return POW2_PUNCTUAL_CHECK_INTERVAL_SIZE;
            }

            static consteval auto get_largebin_smallbin_size() noexcept -> size_t{ //I cant come up with a sound version just yet

                return 0u;
            }

            static constexpr auto compile_time_demote_blk_sz(size_t sz) noexcept -> size_t{

                [[assume(sz <= std::numeric_limits<blk_sz_operatable_t>::max())]];
                return sz;
            }
    };

    //we got a pull request for doing sliding window, I've yet to know what that means
    //alright, we'd want to do a cyclic bump queue
    //this is an extension of bump queue to bump_allocate not-reusable segments on L1 cache

    //let's see how we could patch this
    //think in terms of cyclic queue, we have a sliding window, we want to extend the head or tail everytime we dump our bins, and snap back in the opposite direction
    //we would want to move forward and backward in the cyclic queue, what...
    //because forward head is head, backward head is tail
    //the implementation is not easy, we can do this in 2 days
    //until the cyclic bump queue is not extensible, we'll request a new chunk of memory
    //the implementation we've been longing for

    //a dictionary vectorization of head_tail, with the memory footprint overhead of at most the vectorizing freebin, uint16_t bucket_virtual_addr_t, so the total overhead should not exceed 10% of the freebin_vec
    //we compromise the bump_extension at the internal_dump_freebin_vec, 
    //do a dictionary vectorization, find, extend, get head_tail, iterate through the map, find the intervals that do not intersect with the headtail and redump it to the freebin_vec for final dispatch
    //the howtos write this cleanly is another problem
    //when to reverse our cyclic queue, when to forward only, when to etc.
    //the only problem is the in the freebin_vec
    //we'll sleep on this

    //imagine we have an arbitrary cyclic queue
    //this could extend in the head or extend in the tail
    //if more head, then our restart point is the tail
    //if more tail, then our restart point is the head

    //alright, the reverse logic is hard
    //we'll build a forward bump_allocator, and a translator (a virtual layer) to convert from backward -> forward and forward -> backward
    //so a backward allocation == forward_to_backward(backward_to_forward(backward)(mem_blk))
    //we'll keep this our backlogs

    template <class Metadata>
    class DGStdAllocator{

        public:

            struct Allocation{
                void * user_ptr;
                size_t user_ptr_sz;
            };

        private:

            std::shared_ptr<char[]> buf;
            size_t malloc_chk_interval_counter;
            uint64_t smallbin_avail_bitset;
            std::vector<dg::network_datastructure::cyclic_queue::pow2_cyclic_queue<Allocation>> smallbin_reuse_table; //too many indirections

            NaiveBumpAllocator bump_allocator;
            size_t bump_allocator_refill_sz;
            size_t bump_allocator_version_control;

            std::vector<Allocation> freebin_vec;
            size_t freebin_vec_cap;

            std::shared_ptr<MultiThreadUniformHeapAllocator> heap_allocator;

            std::chrono::time_point<std::chrono::high_resolution_clock> last_flush;
            std::chrono::nanoseconds flush_interval;
            size_t allocation_sz_counter;
            size_t allocation_sz_counter_flush_threshold; 

        public:

            using self                  = DGStdAllocator;
            using sz_header_t           = typename Metadata::sz_header_t;
            using vrs_ctrl_header_t     = typename Metadata::vrs_ctrl_header_t;
            using largebin_sz_header_t  = typename Metadata::largebin_sz_header_t;

            static inline constexpr size_t HEAP_LEAF_UNIT_ALLOCATION_SZ     = Metadata::get_heap_leaf_unit_allocation_size();
            static inline constexpr size_t ALLOCATION_HEADER_SZ             = Metadata::get_allocation_header_size();  
            static inline constexpr size_t LARGEBIN_ALLOCATION_HEADER_SZ    = Metadata::get_largebin_allocation_header_size();  //we need to make sure that the LARGEBIN_ALLOCATION_HEADER_SZ is 8 bytes aligned, this is to achieve the all user_ptrs are at least 8-bytes aligned (we trick this by offseting the buf) 
            static inline constexpr size_t SMALLBIN_PROBE_SZ                = Metadata::get_smallbin_probe_size();
            static inline constexpr size_t MINIMUM_ALLOCATION_BLK_SZ        = Metadata::get_minimum_allocation_blk_size();
            static inline constexpr size_t MAXIMUM_SMALLBIN_BLK_SZ          = Metadata::get_maximum_smallbin_blk_size();
            static inline constexpr size_t PUNCTUAL_CHECK_INTERVAL_SZ       = Metadata::get_pow2_punctual_check_interval_size();
            static inline constexpr size_t LARGEBIN_SMALLBIN_SZ             = Metadata::get_largebin_smallbin_size();
            static inline constexpr size_t ALIGNMENT_SZ                     = Metadata::get_alignment_size();

            DGStdAllocator(std::shared_ptr<char[]> buf,
                           size_t malloc_chk_interval_counter,
                           uint64_t smallbin_avail_bitset,
                           std::vector<dg::network_datastructure::cyclic_queue::pow2_cyclic_queue<Allocation>> smallbin_reuse_table,

                           NaiveBumpAllocator bump_allocator,
                           size_t bump_allocator_refill_sz,
                           size_t bump_allocator_version_control,

                           std::vector<Allocation> freebin_vec,
                           size_t freebin_vec_cap, 

                           std::shared_ptr<MultiThreadUniformHeapAllocator> heap_allocator,

                           std::chrono::time_point<std::chrono::high_resolution_clock> last_flush,
                           std::chrono::nanoseconds flush_interval,
                           size_t allocation_sz_counter,
                           size_t allocation_sz_counter_flush_threshold) noexcept: buf(std::move(buf)),
                                                                                   malloc_chk_interval_counter(malloc_chk_interval_counter),
                                                                                   smallbin_avail_bitset(smallbin_avail_bitset),
                                                                                   smallbin_reuse_table(std::move(smallbin_reuse_table)),

                                                                                   bump_allocator(bump_allocator),
                                                                                   bump_allocator_refill_sz(bump_allocator_refill_sz),
                                                                                   bump_allocator_version_control(bump_allocator_version_control),

                                                                                   freebin_vec(std::move(freebin_vec)),
                                                                                   freebin_vec_cap(freebin_vec_cap),

                                                                                   heap_allocator(std::move(heap_allocator)),

                                                                                   last_flush(last_flush),
                                                                                   flush_interval(flush_interval),
                                                                                   allocation_sz_counter(allocation_sz_counter),
                                                                                   allocation_sz_counter_flush_threshold(allocation_sz_counter_flush_threshold){}

            ~DGStdAllocator() noexcept{

                this->internal_commit_waiting_bin();
            }

            inline auto malloc(size_t blk_sz) noexcept -> void *{

                if (blk_sz == 0u){
                    return nullptr;
                }

                this->malloc_chk_interval_counter += 1u;

                //there is no modulo performed if INTERVAL_CHECK_SZ is of uint8_t, uint16_t or uint32_t
                //> MAXIMUM_SMALLBIN_BLK_SZ == > 63 == >= 64
                //== shift 6 bits, 0 cmp
                //or MAXIMUM_SMALLBIN_BLK_SZ == 65535
                //blk_sz <= sizeof(uint32_t) => uint16_t read of latter + 0 cmp, this is the sound solution
                //this means we need to demote blk_sz > uint32_t, we have to specify this in our Metadata

                if (this->malloc_chk_interval_counter % PUNCTUAL_CHECK_INTERVAL_SZ == 0u || Metadata::compile_time_demote_blk_sz(blk_sz) > self::MAXIMUM_SMALLBIN_BLK_SZ) [[unlikely]]{
                    return this->internal_careful_malloc(blk_sz);
                } else [[likely]]{
                    size_t pow2_blk_sz          = stdx::ceil2(std::max(blk_sz, self::MINIMUM_ALLOCATION_BLK_SZ));
                    size_t smallbin_table_idx   = std::countr_zero(pow2_blk_sz); 
                    uint64_t membership_bitset  = (this->smallbin_avail_bitset >> smallbin_table_idx) & stdx::lowones_bitgen<uint64_t>(std::integral_constant<size_t, SMALLBIN_PROBE_SZ>{}); //shift the sz, SMALLBIN_PROBE_SZ if compile-time deterministic would translate to a uint8_t direct read, if probing 8 bits

                    if (membership_bitset == 0u) [[unlikely]]{
                        return this->internal_bump_allocate(blk_sz);
                    } else [[likely]]{                        
                        size_t actual_table_idx     = smallbin_table_idx + std::countr_zero(membership_bitset);
                        auto& smallbin_vec          = this->smallbin_reuse_table[actual_table_idx];
                        void * rs                   = smallbin_vec.back().user_ptr;
                        smallbin_vec.pop_back();
                        this->smallbin_avail_bitset ^= static_cast<uint64_t>(smallbin_vec.empty()) << actual_table_idx;

                        return rs;
                    }
                }
            }

            inline auto realloc(void * user_ptr, size_t blk_sz) noexcept -> void *{

                if (user_ptr == nullptr) [[unlikely]]{
                    return this->malloc(blk_sz);
                }

                size_t user_ptr_sz = this->internal_read_user_ptr_size(user_ptr); 

                if (user_ptr_sz == self::LARGEBIN_SMALLBIN_SZ){
                    user_ptr_sz = this->internal_read_largebin_user_ptr_size(user_ptr); 
                }

                if (blk_sz <= user_ptr_sz){
                    return user_ptr;
                }

                void * return_mem = this->malloc(blk_sz);

                if (return_mem == nullptr){
                    return nullptr;
                }

                std::memcpy(return_mem, user_ptr, user_ptr_sz);
                this->free(user_ptr);

                return return_mem;
            }

            inline void free(void * user_ptr) noexcept{

                if (user_ptr == nullptr){
                    return;
                }

                //uint32_t load

                sz_header_t user_ptr_sz                                 = this->internal_read_user_ptr_size(user_ptr);
                vrs_ctrl_header_t user_ptr_truncated_version_control    = this->internal_read_user_ptr_truncated_version_control(user_ptr);  

                //this is constituted as a shift and a 0 cmp
                //what's better, a direct address read and a cmp
                //this requires user_ptr_sz to be of uint32_t
                //and LARGEBIN_SMALLBIN_SZ to be of uint16_t
                //this is hard
                //or ... we just do a direct read of uint16_t negate cmp, this is the sound solution

                if (user_ptr_sz == self::LARGEBIN_SMALLBIN_SZ) [[unlikely]]{
                    this->internal_large_free(user_ptr);
                } else [[likely]]{
                    vrs_ctrl_header_t current_truncated_version_control = self::internal_get_truncated_version_control(this->bump_allocator_version_control);
                    size_t floor_smallbin_table_idx                     = stdx::ulog2(user_ptr_sz);
                    auto& smallbin_vec                                  = this->smallbin_reuse_table[floor_smallbin_table_idx];

                    //we break practices because smallbin_vec is pow2, < capacity, == shift + 0 cmp, remember 0 cmp is everything, we just hint the compiler to do optimization, we are not allowed to do optimization

                    if (smallbin_vec.size() < smallbin_vec.capacity() && current_truncated_version_control == user_ptr_truncated_version_control) [[likely]]{ //cmp is better if low_resolution, compiler is better than us to decide what instruction to be used
                        //meets the cond of fast free
                        dg::network_exception::dg_noexcept(smallbin_vec.push_back(Allocation{user_ptr, user_ptr_sz}));
                        this->smallbin_avail_bitset |= uint64_t{1} << floor_smallbin_table_idx; //membership registration
                    } else{
                        //does not meet the cond of fast free, either because smallbin_vec.size() == smallbin_vec.capacity() or current_truncated_version_control != user_ptr_truncated_version_control
                        //either way, we have to put at least one allocation -> freebin_vec

                        if (this->freebin_vec.size() == this->freebin_vec_cap) [[unlikely]]{
                            this->internal_dump_freebin_vec();
                        }

                        //freebin_vec room is clear
                        //this implies smallbin_vec.size() == smallbin_vec.capacity(), true | false = true
                        if (current_truncated_version_control == user_ptr_truncated_version_control){
                            //this is still qualified for reusability
                            this->freebin_vec.push_back(smallbin_vec.front()); //make room for smallbin_vec back()
                            smallbin_vec.pop_front();
                            dg::network_exception::dg_noexcept(smallbin_vec.push_back(Allocation{user_ptr, user_ptr_sz}));
                        } else{
                            this->freebin_vec.push_back(Allocation{user_ptr, user_ptr_sz});
                        }
                    }
                }
            }

        private:

            static constexpr auto internal_get_truncated_version_control(size_t version_control) noexcept -> vrs_ctrl_header_t{

                return version_control & static_cast<size_t>(std::numeric_limits<vrs_ctrl_header_t>::max()); //static cast should suffice
            } 

            inline auto internal_interval_to_buf(const interval_type& interval) const noexcept -> std::pair<void *, size_t>{

                const std::pair<heap_sz_type, heap_sz_type>& semantic_representation = interval;

                size_t buf_offset   = static_cast<size_t>(semantic_representation.first) * HEAP_LEAF_UNIT_ALLOCATION_SZ;
                size_t buf_sz       = (static_cast<size_t>(semantic_representation.second) + 1u) * HEAP_LEAF_UNIT_ALLOCATION_SZ;

                return std::make_pair(static_cast<void *>(std::next(this->buf.get(), buf_offset)), buf_sz);
            }

            inline auto internal_aligned_buf_to_interval(const std::pair<void *, size_t>& arg) const noexcept -> interval_type{

                size_t buf_offset   = std::distance(this->buf.get(), static_cast<char *>(arg.first));
                size_t buf_sz       = arg.second;
                size_t heap_offset  = buf_offset / HEAP_LEAF_UNIT_ALLOCATION_SZ;
                size_t heap_excl_sz = (buf_sz / HEAP_LEAF_UNIT_ALLOCATION_SZ) - 1u;

                return std::make_pair(heap_offset, heap_excl_sz);
            }

            constexpr auto internal_get_internal_ptr_head(void * user_ptr) const noexcept -> void *{

                return static_cast<void *>(std::prev(static_cast<char *>(user_ptr), ALLOCATION_HEADER_SZ));
            }

            constexpr auto internal_get_internal_ptr_head(const void * user_ptr) const noexcept -> const void *{

                return static_cast<const void *>(std::prev(static_cast<const char *>(user_ptr), ALLOCATION_HEADER_SZ));
            }

            constexpr auto internal_get_user_ptr_head(void * internal_ptr) const noexcept -> void *{
                
                return std::next(static_cast<char *>(internal_ptr), ALLOCATION_HEADER_SZ);
            }

            constexpr auto internal_get_user_ptr_head(const void * internal_ptr) const noexcept -> const void *{

                return std::next(static_cast<const char *>(internal_ptr), ALLOCATION_HEADER_SZ);
            }

            constexpr auto internal_get_largebin_internal_ptr_head(void * user_ptr) const noexcept -> void *{

                return static_cast<void *>(std::prev(static_cast<char *>(user_ptr), LARGEBIN_ALLOCATION_HEADER_SZ));
            }

            constexpr auto internal_get_largebin_internal_ptr_head(const void * user_ptr) const noexcept -> const void *{

                return static_cast<const void *>(std::prev(static_cast<const char *>(user_ptr), LARGEBIN_ALLOCATION_HEADER_SZ));
            }

            constexpr auto internal_get_largebin_user_ptr_head(void * internal_ptr) const noexcept -> void *{

                return std::next(static_cast<char *>(internal_ptr), LARGEBIN_ALLOCATION_HEADER_SZ);
            }

            constexpr auto internal_get_largebin_user_ptr_head(const void * internal_ptr) const noexcept -> const void *{

                return std::next(static_cast<const char *>(internal_ptr), LARGEBIN_ALLOCATION_HEADER_SZ);
            }

            constexpr void internal_update_allocation_sensor(size_t new_blk_sz) noexcept{

                this->allocation_sz_counter += new_blk_sz;
            }

            constexpr auto internal_read_user_ptr_size(const void * user_ptr) const noexcept -> sz_header_t{

                const void * ptr_head       = this->internal_get_internal_ptr_head(user_ptr);
                sz_header_t allocation_sz   = {};
                std::memcpy(&allocation_sz, ptr_head, sizeof(sz_header_t));

                return allocation_sz; 
            }

            constexpr auto internal_read_user_ptr_truncated_version_control(const void * user_ptr) const noexcept -> vrs_ctrl_header_t{

                const void * ptr_head               = this->internal_get_internal_ptr_head(user_ptr);
                const void * version_control_head   = std::next(static_cast<const char *>(ptr_head), sizeof(sz_header_t)); 
                vrs_ctrl_header_t version_ctrl      = {};
                std::memcpy(&version_ctrl, version_control_head, sizeof(vrs_ctrl_header_t));

                return version_ctrl;
            }

            constexpr auto internal_write_allocation_header(void * internal_ptr, sz_header_t user_ptr_sz, vrs_ctrl_header_t version_control) const noexcept -> void *{

                std::memcpy(internal_ptr, &user_ptr_sz, sizeof(sz_header_t));
                std::memcpy(std::next(static_cast<char *>(internal_ptr), sizeof(sz_header_t)), &version_control, sizeof(vrs_ctrl_header_t));

                return this->internal_get_user_ptr_head(internal_ptr);
            }

            constexpr auto internal_read_largebin_user_ptr_size(const void * user_ptr) const noexcept -> largebin_sz_header_t{

                const void * ptr_head       = this->internal_get_largebin_internal_ptr_head(user_ptr);
                auto rs                     = largebin_sz_header_t{};
                std::memcpy(&rs, ptr_head, sizeof(largebin_sz_header_t)); 

                return rs;
            }

            constexpr auto internal_write_largebin_allocation_header(void * internal_ptr, largebin_sz_header_t user_ptr_sz) const noexcept -> void *{

                assert(user_ptr_sz > self::MAXIMUM_SMALLBIN_BLK_SZ);
                std::memcpy(internal_ptr, &user_ptr_sz, sizeof(largebin_sz_header_t));

                return this->internal_write_allocation_header(std::next(static_cast<char *>(internal_ptr), LARGEBIN_ALLOCATION_HEADER_SZ - ALLOCATION_HEADER_SZ), self::LARGEBIN_SMALLBIN_SZ, 0u); 
            }

            __attribute__((noinline)) void internal_dump_freebin_vec(){

                dg::network_stack_allocation::NoExceptAllocation<interval_type[]> intv_arr(this->freebin_vec.size());

                for (size_t i = 0u; i < this->freebin_vec.size(); ++i){
                    void * internal_ptr     = this->internal_get_internal_ptr_head(this->freebin_vec[i].user_ptr);
                    size_t internal_ptr_sz  = this->freebin_vec[i].user_ptr_sz + ALLOCATION_HEADER_SZ;
                    intv_arr[i]             = this->internal_aligned_buf_to_interval({internal_ptr, internal_ptr_sz});
                }

                this->heap_allocator->free(intv_arr.get(), this->freebin_vec.size());
                this->freebin_vec.clear();
            }

            inline void internal_decommission_bump_allocator() noexcept{

                auto [decom_buf, decom_sz] = this->bump_allocator.decommission();

                if (decom_sz != 0u){
                    //assumption is clear, assume that bump_allocator remaining is aligned, this is achieved by aligned refill_sz + aligned bump_malloc_sz
                    interval_type heap_interval = this->internal_aligned_buf_to_interval({static_cast<void *>(decom_buf), decom_sz});
                    this->heap_allocator->free(&heap_interval, 1u);
                }
            }

            inline auto internal_dispatch_bump_allocator_refill() noexcept -> bool{

                this->internal_decommission_bump_allocator();

                size_t requesting_interval_sz   = this->bump_allocator_refill_sz / HEAP_LEAF_UNIT_ALLOCATION_SZ;
                auto requesting_interval        = std::optional<interval_type>{};

                this->heap_allocator->alloc(&requesting_interval_sz, 1u, &requesting_interval);

                if (!requesting_interval.has_value()){
                    return false;
                }

                std::pair<void *, size_t> requesting_buf    = this->internal_interval_to_buf(requesting_interval.value());
                this->bump_allocator                        = NaiveBumpAllocator(static_cast<char *>(requesting_buf.first), requesting_buf.second);
                this->bump_allocator_version_control        += 1;

                this->internal_update_allocation_sensor(this->bump_allocator_refill_sz);

                return true;
            }

            __attribute__((noinline)) auto internal_bump_allocate(size_t user_blk_sz) noexcept -> void *{

                assert(user_blk_sz != 0u);
                assert(user_blk_sz <= self::MAXIMUM_SMALLBIN_BLK_SZ);

                //now this is hard
                //maximum_smallbin_blk_sz   == 60, header_sz == 4
                //maximum_allocated_sz      == 64
                //user_usable_blk_sz        == 60 <= maximum_smallbin_blk_sz
                //things are clear
                //we dont want to trip the threshold, because that would break the contract of maximum_smallbin_blk_sz by large_free

                user_blk_sz                                     = std::max(user_blk_sz, self::MINIMUM_ALLOCATION_BLK_SZ); //people are requesting user_pow2_blk_sz to be reusable, this is more important than the allocations saved by bump allocator
                size_t pad_blk_sz                               = user_blk_sz + ALLOCATION_HEADER_SZ;
                size_t ceil_blk_sz                              = (((pad_blk_sz - 1u) / HEAP_LEAF_UNIT_ALLOCATION_SZ) + 1u) * HEAP_LEAF_UNIT_ALLOCATION_SZ;
                size_t user_usable_blk_sz                       = ceil_blk_sz - ALLOCATION_HEADER_SZ; 
                std::expected<void *, exception_t> internal_ptr = this->bump_allocator.malloc(ceil_blk_sz);

                if (!internal_ptr.has_value()){
                    if (internal_ptr.error() == dg::network_exception::RESOURCE_EXHAUSTION){
                        bool is_refilled = this->internal_dispatch_bump_allocator_refill();

                        if (!is_refilled){
                            return nullptr;
                        }

                        internal_ptr = dg::network_exception::remove_expected(this->bump_allocator.malloc(ceil_blk_sz));
                    } else{
                        if constexpr(DEBUG_MODE_FLAG){
                            std::abort();
                        } else{
                            std::unreachable();
                        }
                    }
                }

                return this->internal_write_allocation_header(internal_ptr.value(), 
                                                              stdx::wrap_safe_integer_cast(user_usable_blk_sz), 
                                                              self::internal_get_truncated_version_control(this->bump_allocator_version_control));
            }

            void internal_commit_waiting_bin() noexcept{

                this->internal_decommission_bump_allocator();

                for (auto& smallbin_queue: this->smallbin_reuse_table){
                    for (auto& smallbin: smallbin_queue){
                        if (this->freebin_vec.size() == this->freebin_vec_cap){
                            this->internal_dump_freebin_vec();
                        }

                        this->freebin_vec.push_back(smallbin);
                    }

                    smallbin_queue.clear();
                }

                this->internal_dump_freebin_vec();
                this->smallbin_avail_bitset = 0u;
            }

            void internal_reset() noexcept{

                this->internal_commit_waiting_bin();

                this->last_flush            = std::chrono::high_resolution_clock::now();
                this->allocation_sz_counter = 0u;
            }

            void internal_check_for_reset() noexcept{
                
                std::chrono::time_point<std::chrono::high_resolution_clock> now             = std::chrono::high_resolution_clock::now();
                std::chrono::time_point<std::chrono::high_resolution_clock> flush_expiry    = this->last_flush + this->flush_interval;

                bool reset_cond_1   = this->allocation_sz_counter >= this->allocation_sz_counter_flush_threshold;
                bool reset_cond_2   = now >= flush_expiry; 

                if (reset_cond_1 || reset_cond_2){
                    this->internal_reset();
                }
            }

            //this is complicated
            auto internal_large_malloc(size_t user_blk_sz) noexcept -> void *{

                assert(user_blk_sz > self::MAXIMUM_SMALLBIN_BLK_SZ);

                size_t pad_blk_sz                           = user_blk_sz + LARGEBIN_ALLOCATION_HEADER_SZ;
                size_t ceil_blk_sz                          = (((pad_blk_sz - 1u) / HEAP_LEAF_UNIT_ALLOCATION_SZ) + 1u) * HEAP_LEAF_UNIT_ALLOCATION_SZ;
                size_t user_usable_blk_sz                   = ceil_blk_sz - LARGEBIN_ALLOCATION_HEADER_SZ;
                size_t allocating_heap_node_sz              = ceil_blk_sz / HEAP_LEAF_UNIT_ALLOCATION_SZ; 
                std::optional<interval_type> allocated_intv = {};

                this->heap_allocator->alloc(&allocating_heap_node_sz, 1u, &allocated_intv);

                if (!allocated_intv.has_value()){
                    return nullptr;
                }

                std::pair<void *, size_t> requesting_buf    = this->internal_interval_to_buf(allocated_intv.value());
                void * internal_ptr                         = requesting_buf.first;
                void * user_ptr                             = this->internal_write_largebin_allocation_header(internal_ptr, user_usable_blk_sz); //writing maximum_smallbin_blk_sz + 1u for free dispatch code, vrs_ctrl is irrelevant, can be of any values

                this->internal_update_allocation_sensor(ceil_blk_sz);

                return user_ptr;
            }

            __attribute__((noinline)) void internal_large_free(void * user_ptr) noexcept{

                //inverse operation of internal_large_malloc

                void * internal_ptr         = this->internal_get_largebin_internal_ptr_head(user_ptr);
                size_t internal_ptr_sz      = this->internal_read_largebin_user_ptr_size(user_ptr) + LARGEBIN_ALLOCATION_HEADER_SZ; 
                interval_type intv          = this->internal_aligned_buf_to_interval({internal_ptr, internal_ptr_sz});

                this->heap_allocator->free(&intv, 1u);
            }

            //as far as we concerned, this could be one of the public APIs, this function is external sufficient by itself (there are no internal assumptions or asserts)
            __attribute__((noinline)) auto internal_careful_malloc(size_t user_blk_sz) noexcept -> void *{

                if (user_blk_sz == 0u){
                    return nullptr;
                }

                this->internal_check_for_reset();

                if (Metadata::compile_time_demote_blk_sz(user_blk_sz) > self::MAXIMUM_SMALLBIN_BLK_SZ){
                    return this->internal_large_malloc(user_blk_sz);
                }

                size_t pow2_blk_sz          = stdx::ceil2(std::max(user_blk_sz, self::MINIMUM_ALLOCATION_BLK_SZ));
                size_t smallbin_table_idx   = std::countr_zero(pow2_blk_sz);
                uint64_t membership_bitset  = (this->smallbin_avail_bitset >> smallbin_table_idx) & stdx::lowones_bitgen<uint64_t>(std::integral_constant<size_t, SMALLBIN_PROBE_SZ>{}); //shift the sz 

                if (membership_bitset == 0u){
                    return this->internal_bump_allocate(user_blk_sz);
                }

                size_t actual_table_idx     = smallbin_table_idx + std::countr_zero(membership_bitset); 
                auto& smallbin_vec          = this->smallbin_reuse_table[actual_table_idx];
                void * rs                   = smallbin_vec.back().user_ptr;
                smallbin_vec.pop_back();
                this->smallbin_avail_bitset ^= static_cast<uint64_t>(smallbin_vec.empty()) << actual_table_idx;

                return rs;
            }
    };

    template <class AllocatorMetadata>
    class ConcurrentDGStdAllocator{

        private:

            std::vector<DGStdAllocator<AllocatorMetadata>> dg_allocator_vec;

        public:

            static inline constexpr size_t ALIGNMENT_SZ = DGStdAllocator<AllocatorMetadata>::ALIGNMENT_SZ;

            ConcurrentDGStdAllocator(std::vector<DGStdAllocator<AllocatorMetadata>> dg_allocator_vec) noexcept: dg_allocator_vec(std::move(dg_allocator_vec)){}

            inline auto malloc(size_t blk_sz) noexcept -> void *{

                return this->dg_allocator_vec[dg::network_concurrency::this_thread_idx()].malloc(blk_sz);
            }

            inline auto realloc(void * ptr, size_t blk_sz) noexcept -> void *{

                return this->dg_allocator_vec[dg::network_concurrency::this_thread_idx()].realloc(ptr, blk_sz);
            }

            inline void free(void * ptr) noexcept{

                this->dg_allocator_vec[dg::network_concurrency::this_thread_idx()].free(ptr);
            }
    };

    using concurrent_dg_std_allocator_t = ConcurrentDGStdAllocator<DGStdAllocatorMetadata<uint32_t, 8u, 256u, 16u, 63u, 4u>>; //there is a history to aligned cmp, var >> sth != 0u is very fast, 0 cmp is 0 cost, this is due to nullptr + sz optimizations

    class GCWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:
  
            std::shared_ptr<GCInterface> gc_able;

        public:

            GCWorker(std::shared_ptr<GCInterface> gc_able): gc_able(std::move(gc_able)){}

            auto run_one_epoch() noexcept -> bool{

                this->gc_able->gc();
                return false;
            }
    };

    //the problem is probably the aligned alloc, we can't sanitize the address after the address has been tainted by new () inplace construction
    //the only guy that has the ability to do so is the DGStdAllocator without messing with the compiler escape analysis
    //recall that this is a valid address sanitize operation, or it is not...
    //what's the viking way of doing addr_sant, __force__noinline__ + separate make
    //we'll stick with THE WAY to avoid bad practices of new lifetime + friends, it's ... hard
    //because void * user_ptr is originated from the malloc, it is the compiler responsibility to track that, YET... the new case is special, we can't track things post new
    //so we must detach the malloc responsibility there

    // struct Factory{

    //     static auto spawn_heap_allocator(size_t base_sz) -> std::unique_ptr<GCHeapAllocator>{ //devirt here is important

    //         const size_t MIN_BASE_SZ    = 1u;
    //         const size_t MAX_BASE_SZ    = size_t{1} << 40;

    //         if (std::clamp(base_sz, MIN_BASE_SZ, MAX_BASE_SZ) != base_sz){
    //             dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
    //         }

    //         if (!!dg::memult::is_pow2(base_sz)){
    //             dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
    //         }

    //         uint16_t tree_height = stdx::ulog2_aligned(base_sz) + 1u;
    //         size_t buf_sz       = dg::heap::user_interface::get_memory_usage(tree_height);
    //         auto buf            = std::unique_ptr<char[], decltype(&std::free)>(static_cast<char *>(std::malloc(buf_sz)), std::free);

    //         if (!buf){
    //             dg::network_exception::throw_exception(dg::network_exception::OUT_OF_MEMORY);
    //         }

    //         auto allocator  = dg::heap::user_interface::get_allocator_x(buf.get());
    //         auto lck        = std::make_unique<std::atomic_flag>();
    //         auto rs         = std::make_unique<GCHeapAllocator>(std::move(buf), std::move(allocator), std::move(lck));

    //         return rs;
    //     }

    //     static auto spawn_allocator(size_t least_buf_sz) -> std::unique_ptr<Allocator>{ //devirt here is important

    //         const size_t MIN_LEAST_BUF_SZ   = 1u;
    //         const size_t MAX_LEAST_BUF_SZ   = size_t{1} << 40;

    //         if (std::clamp(least_buf_sz, MIN_LEAST_BUF_SZ, MAX_LEAST_BUF_SZ) != least_buf_sz){
    //             dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
    //         }

    //         size_t buf_sz   = stdx::least_pow2_greater_equal_than(std::max(least_buf_sz, LEAF_SZ));
    //         size_t base_sz  = buf_sz / LEAF_SZ;
    //         auto buf        = std::unique_ptr<char[], decltype(&std::free)>(static_cast<char *>(std::aligned_alloc(LEAF_SZ, buf_sz)), std::free);  

    //         if (!buf){
    //             dg::network_exception::throw_exception(dg::network_exception::OUT_OF_MEMORY);
    //         }

    //         std::unique_ptr<GCHeapAllocator> base_allocator = spawn_heap_allocator(base_sz);
    //         return std::make_unique<Allocator>(std::move(buf), std::move(base_allocator));
    //     }

    //     static auto spawn_concurrent_allocator(std::vector<std::unique_ptr<Allocator>> allocator) -> std::unique_ptr<MultiThreadAllocator>{ //devirt here is important - 

    //         const size_t MIN_ALLOCATOR_SZ   = 1u;
    //         const size_t MAX_ALLOCATOR_SZ   = size_t{1} << 8;

    //         if (std::clamp(static_cast<size_t>(allocator.size()), MIN_ALLOCATOR_SZ, MAX_ALLOCATOR_SZ) != allocator.size()){
    //             dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
    //         }

    //         if (!dg::memult::is_pow2(allocator.size())){
    //             dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
    //         }

    //         if (std::find(allocator.begin(), allocator.end(), nullptr) != allocator.end()){
    //             dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
    //         }

    //         return std::make_unique<MultiThreadAllocator>(std::move(allocator));
    //     }

    //     static auto spawn_gc_worker(std::shared_ptr<GCInterface> gc_able) -> dg::network_concurrency::daemon_raii_handle_t{ //this is strange - this overstep into the responsibility - decouple the component

    //         if (gc_able == nullptr){
    //             dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
    //         }

    //         auto worker = std::make_unique<GCWorker>(std::move(gc_able));
    //         auto rs     = dg::network_concurrency::daemon_saferegister(dg::network_concurrency::COMPUTING_DAEMON, std::move(worker));

    //         if (!rs.has_value()){
    //             dg::network_exception::throw_exception(rs.error());
    //         } 

    //         return std::move(rs.value());
    //     }
    // };

    struct AllocationResource{
        std::shared_ptr<concurrent_dg_std_allocator_t> allocator; 
        dg::network_concurrency::daemon_raii_handle_t gc_worker;
    };

    struct AllocationResourceSignature{};

    using allocation_resource_obj = stdx::singleton<AllocationResourceSignature, AllocationResource>; 

    // inline stdx::singleton<AllocationResource> allocation_resource;

    // void init(size_t least_buf_sz, size_t num_allocator){

    //     std::vector<std::unique_ptr<Allocator>> allocator_vec{};

    //     for (size_t i = 0u; i < num_allocator; ++i){
    //         allocator_vec.push_back(Factory::spawn_allocator(least_buf_sz));
    //     }

    //     std::shared_ptr<MultiThreadAllocator> allocator = Factory::spawn_concurrent_allocator(std::move(allocator_vec));
    //     dg::network_concurrency::daemon_raii_handle_t daemon_handle = Factory::spawn_gc_worker(allocator);
    //     allocation_resource = {std::move(allocator), std::move(daemon_handle)};
    // }

    // void deinit() noexcept{

    //     allocation_resource = {};
    // }

    //we cant do aligned alloc for the reason being it's hard, aligned alloc is shared_ptr + type-erased allocation responsibility, we literally can't store the offset for performance + everything constraints
    //DEFAULT_ALIGNMENT_SZ should SUFFICE

    static inline constexpr size_t DEFAULT_ALIGNMENT_SZ = concurrent_dg_std_allocator_t::ALIGNMENT_SZ; 
    using alignment_header_t = uint32_t;

    static inline auto dg_align(void * ptr, uintptr_t alignment_sz) noexcept -> void *{

        assert(stdx::is_pow2(alignment_sz));

        uintptr_t fwd_sz                        = alignment_sz - 1u;
        uintptr_t bit_mask                      = ~fwd_sz;
        uintptr_t aligned_ptr_numerical_addr    = (reinterpret_cast<uintptr_t>(ptr) + fwd_sz) & bit_mask;

        return reinterpret_cast<void *>(aligned_ptr_numerical_addr);
    }

    static inline auto dg_align(const void * ptr, uintptr_t alignment_sz) noexcept -> const void *{

        assert(stdx::is_pow2(alignment_sz));

        uintptr_t fwd_sz                        = alignment_sz - 1u;
        uintptr_t bit_mask                      = ~fwd_sz;
        uintptr_t aligned_ptr_numerical_addr    = (reinterpret_cast<uintptr_t>(ptr) + fwd_sz) & bit_mask;

        return reinterpret_cast<const void *>(aligned_ptr_numerical_addr);
    }

    extern __attribute__((noipa)) auto dg_malloc(size_t blk_sz) noexcept -> void *{

        return allocation_resource_obj::get().allocator->malloc(blk_sz); 
    }

    //alright, this is the headache
    extern __attribute__((noipa)) auto dg_realloc(void * buf, size_t blk_sz) noexcept -> void *{

        allocation_resource_obj::get().allocator->realloc(buf, blk_sz);
    } 

    extern __attribute__((noipa)) void dg_free(void * ptr) noexcept{

        allocation_resource_obj::get().allocator->free(ptr);
    }

    extern __attribute__((noipa)) auto dg_aligned_alloc(size_t alignment, size_t blk_sz) noexcept -> void *{

        if (!stdx::is_pow2(alignment)){
            return nullptr;
        }

        const size_t max_fwd_sz = alignment + (sizeof(alignment_header_t) - 1u);  

        if (max_fwd_sz > std::numeric_limits<alignment_header_t>::max()){
            return nullptr;
        }

        if (blk_sz == 0u){
            return nullptr;
        }

        size_t align_fwd_sz = alignment - 1u;
        size_t adj_blk_sz   = blk_sz + align_fwd_sz + sizeof(alignment_header_t);
        void * ptr          = allocation_resource_obj::get().allocator->malloc(adj_blk_sz);

        if (ptr == nullptr){
            return nullptr;
        }

        void * aligned_ptr              = dg::network_allocation::dg_align(std::next(static_cast<char *>(ptr), sizeof(alignment_header_t)), alignment);
        alignment_header_t difference   = std::distance(static_cast<char *>(ptr), static_cast<char *>(aligned_ptr));
        void * alignment_header_addr    = std::prev(static_cast<char *>(aligned_ptr), sizeof(alignment_header_t));

        std::memcpy(alignment_header_addr, &difference, sizeof(alignment_header_t));

        return aligned_ptr; 
    } 

    extern __attribute__((noipa)) void dg_aligned_free(void * ptr) noexcept{

        if (ptr == nullptr){
            return;
        }

        void * alignment_header_addr    = std::prev(static_cast<char *>(ptr), sizeof(alignment_header_t));
        alignment_header_t difference   = {};
        std::memcpy(&difference, alignment_header_addr, sizeof(alignment_header_t));
        void * org_ptr                  = std::prev(static_cast<char *>(ptr), difference); 

        allocation_resource_obj::get().allocator->free(org_ptr); 
    }

    struct XAlignMetadata{
        alignment_header_t difference;
        size_t blk_sz; 
        
        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(difference, blk_sz);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(difference, blk_sz);
        }
    };

    extern __attribute__((noipa)) auto dg_xaligned_alloc(size_t alignment, size_t blk_sz) noexcept -> void *{

        constexpr size_t METADATA_SZ = dg::network_trivial_serializer::size(XAlignMetadata{});

        if (!stdx::is_pow2(alignment)){
            return nullptr;
        }

        const size_t max_fwd_sz = alignment + (METADATA_SZ - 1u);

        if (max_fwd_sz > std::numeric_limits<alignment_header_t>::max()){
            return nullptr;
        }

        if (blk_sz == 0u){
            return nullptr;
        }

        size_t align_fwd_sz = alignment - 1u;
        size_t adj_blk_sz   = blk_sz + align_fwd_sz + METADATA_SZ;
        void * ptr          = allocation_resource_obj::get().allocator->malloc(adj_blk_sz);

        if (ptr == nullptr){
            return nullptr;
        }

        void * aligned_ptr              = dg::network_allocation::dg_align(std::next(static_cast<char *>(ptr), METADATA_SZ), alignment);
        alignment_header_t difference   = std::distance(static_cast<char *>(ptr), static_cast<char *>(aligned_ptr));
        void * metadata_header_addr     = std::prev(static_cast<char *>(aligned_ptr), METADATA_SZ);

        dg::network_trivial_serializer::serialize_into(static_cast<char *>(metadata_header_addr), XAlignMetadata{difference, blk_sz});

        return aligned_ptr;
    }

    extern __attribute__((noipa)) void dg_xaligned_free(void * ptr) noexcept{

        constexpr size_t METADATA_SZ    = dg::network_trivial_serializer::size(XAlignMetadata{});

        void * metadata_header_addr     = std::prev(static_cast<char *>(ptr), METADATA_SZ);
        auto metadata                   = XAlignMetadata{};
        dg::network_trivial_serializer::deserialize_into(metadata, static_cast<const char *>(metadata_header_addr));
        void * org_ptr                  = std::prev(static_cast<char *>(ptr), metadata.difference); 

        allocation_resource_obj::get().allocator->free(org_ptr); 
    }

    extern __attribute__((noipa)) auto dg_xaligned_blk_size(void * ptr) noexcept -> size_t{

        constexpr size_t METADATA_SZ    = dg::network_trivial_serializer::size(XAlignMetadata{});

        void * metadata_header_addr     = std::prev(static_cast<char *>(ptr), METADATA_SZ);
        auto metadata                   = XAlignMetadata{};
        dg::network_trivial_serializer::deserialize_into(metadata, static_cast<const char *>(metadata_header_addr));

        return metadata.blk_sz;
    }

    //alright now I regretted I just got damn put my code elsewhere
    template <class T>
    using NoExceptAllocator = std::allocator<T>;
    
    // template <class T>
    // struct NoExceptAllocator: std::allocator<T>{
 
    //     // using value_type                                = T;
    //     // using pointer                                   = T *;
    //     // using const_pointer                             = const T *;
    //     // using reference                                 = T&;
    //     // using const_reference                           = const T&;
    //     // using size_type                                 = size_t;
    //     // using difference_type                           = intmax_t;
    //     // using is_always_equal                           = std::true_type;
    //     // using propagate_on_container_move_assignment    = std::true_type;
        
    //     // template <class U>
    //     // struct rebind{
    //     //     using other = NoExceptAllocator<U>;
    //     // };

    //     // auto address(reference x) const noexcept -> pointer{

    //     //     return std::addressof(x);
    //     // }

    //     // auto address(const_reference x) const noexcept -> const_pointer{

    //     //     return std::addressof(x);
    //     // }
        
    //     // constexpr auto allocate(size_t n, const void * hint) -> pointer{ //noexcept is guaranteed internally - this is to comply with std

    //     //     if (n == 0u){
    //     //         return nullptr;
    //     //     }

    //     //     void * buf = cmalloc(n * sizeof(T));

    //     //     if (!buf){
    //     //         dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::OUT_OF_MEMORY));
    //     //         std::abort();
    //     //     }

    //     //     return dg::memult::start_lifetime_as_array<T>(buf, n); //this needs compiler magic to avoid undefined behaviors
    //     // }

    //     // auto allocate(size_t n) -> pointer{
            
    //     //     return allocate(n, std::add_pointer_t<const void>{});
    //     // }
        
    //     // //according to std - deallocate arg is valid ptr - such that allocate -> std::optional<ptr_type>, void deallocate(ptr_type)
    //     // void deallocate(pointer p, size_t n){ //noexcept is guaranteed internally - this is to comply with std

    //     //     if (n == 0u){
    //     //         return;
    //     //     }

    //     //     cfree(static_cast<void *>(p)); //fine - a reverse operation of allocate
    //     // }

    //     // consteval auto max_size() const noexcept -> size_type{

    //     //     return std::numeric_limits<size_type>::max();
    //     // }
        
    //     // template <class U, class... Args>
    //     // void construct(U * p, Args&&... args) noexcept(std::is_nothrow_constructible_v<U, Args...>){

    //     //     new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
    //     // }

    //     // template <class U>
    //     // void destroy(U * p) noexcept(std::is_nothrow_destructible_v<U>){

    //     //     std::destroy_at(p);
    //     // }
    // };

    template <class T, class T1>
    constexpr auto operator==(const NoExceptAllocator<T>&, const NoExceptAllocator<T1>&) noexcept -> bool{

        return true;
    }

    template<class T, class T1>
    constexpr auto operator!=(const NoExceptAllocator<T>&, const NoExceptAllocator<T1>&) noexcept -> bool{

        return false;
    }

    //the case of std object lifetime is a hard case
    //assume that std::malloc() + std::free() works perfectly for segments (as if we are passing reinterpret_cast<uintptr_t>()), not reading the address 

    //what's going on?
    //we create a new object, this object is guaranteed to be __restrict__ * by malloc()
    //this object is tracked by compiler for the restrictness to do pointer optimizations

    //what happens if we step out of the comfort zones, cast to char and read value pre-the object living address?
    //compiler can't see your intentions and mark your program as undefined, what happens next is irrelevant
    //so std_new_object is only defined when post the new () inplace construction operation, the sole memory referenced by the user inferred from the address is the pointer-intercompatible version of the object
    //our program works fine if we only have new operations
    //we actually dont care about delete except for doing cleanup (for the lifetime duration, guaranteed to be read-correct by compiler) and freeing the address, what happens at the address thereafter is irrelevant, yeah, free can actually write things to the address (zero out the bytes) and the std_delete_object or users cannot see (this is defined)
    //this is why my proposal of std::bit_cast<> for std::free is actually a 100% patch
    //we just standardize the free by making it noipa, such is what happens at T * post the free is "seeable" by the std_delete_object, except for std_delete_object cannot see what beyond their scope of seeing, such is the confined place of new[], not the headers

    //who supposed to take in this responsibility, it is the std_new_object + std_delete_object to invoke [[gnu::noinline]] free()
    //this is the std way, because new_object taints the char * lifetime at the address, it must the std_delete_object to untaint the lifetime, we don't yet have std measurements, so its best to extern + noipa the malloc + free for now, actually it's both

    template <class T, class ...Args>
    auto std_new_object(Args&& ...args) -> T *{

        static_assert(sizeof(T) != 0u);
        // char * blk = dg::network_allocation::dg_malloc(sizeof(T) + alignof(T) - 1u); //what's the right way, we can do post write, because we know the object size
        void * blk = nullptr;

        if constexpr(alignof(T) <= dg::network_allocation::DEFAULT_ALIGNMENT_SZ){
            blk = dg::network_allocation::dg_malloc(sizeof(T));
        } else{
            blk = dg::network_allocation::dg_aligned_alloc(alignof(T), sizeof(T));
        }

        if (blk == nullptr){
            throw std::bad_alloc();
        }

        if constexpr(std::is_nothrow_constructible_v<T, Args&&...>){
            return new (blk) T(std::forward<Args>(args)...);
        } else{
            try {
                return new (blk) T(std::forward<Args>(args)...);
            } catch (...){

                if constexpr(alignof(T) <= dg::network_allocation::DEFAULT_ALIGNMENT_SZ){
                    [[gnu::noinline]] dg::network_allocation::dg_free(blk);
                } else{
                    [[gnu::noinline]] dg::network_allocation::dg_aligned_free(blk);
                }

                throw;
            }
        }
    }


    template <class = void>
    static inline constexpr bool FALSE_VAL = false;

    template <class T>
    auto std_delete_object(T * obj) noexcept(std::is_nothrow_destructible_v<T>){ //I dont even know what to do if this throws

        std::destroy_at(obj);

        if constexpr(alignof(T) <= dg::network_allocation::DEFAULT_ALIGNMENT_SZ){
            [[gnu::noinline]] dg::network_allocation::dg_free(static_cast<void *>(obj));
        } else{
            [[gnu::noinline]] dg::network_allocation::dg_aligned_free(static_cast<void *>(obj));
        }
    }

    template <class T>
    auto std_new_array(size_t sz) -> T *{

        static_assert(sizeof(T) != 0u);
        static_assert(std::is_nothrow_default_constructible_v<T>);

        if (sz == 0u){
            return nullptr;
        }

        size_t allocation_blk_sz    = sz * sizeof(T); 
        void * blk                  = dg::network_allocation::dg_xaligned_alloc(alignof(T), allocation_blk_sz);

        //

        if (blk == nullptr){
            throw std::bad_alloc();
        }  

        return new (blk) T[sz];
    }

    template <class T>
    void std_delete_array(T * arr) noexcept{

        if (arr == nullptr){
            return;
        }

        size_t allocation_blk_sz    = dg::network_allocation::dg_xaligned_blk_size(arr);
        size_t sz                   = allocation_blk_sz / sizeof(T);

        std::destroy(arr, std::next(arr, sz));
        [[gnu::noinline]] dg::network_allocation::dg_xaligned_free(static_cast<void *>(arr));
    }

    template <class T, class ...Args>
    auto make_unique(Args&& ...args) -> decltype(auto){

        return std::make_unique<T>(std::forward<Args>(args)...);
    }
}

#endif