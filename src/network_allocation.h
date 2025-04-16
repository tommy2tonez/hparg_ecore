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

namespace dg::network_allocation{
    
    using ptr_type              = uint64_t;
    using alignment_type        = uint16_t;
    using interval_type         = dg::heap::types::interval_type; 
    using heap_sz_type          = dg::heap::types::store_type;

    static inline constexpr size_t PTROFFS_BSPACE               = sizeof(uint32_t) * CHAR_BIT;
    static inline constexpr size_t PTRSZ_BSPACE                 = sizeof(uint16_t) * CHAR_BIT;
    static inline constexpr size_t ALLOCATOR_ID_BSPACE          = sizeof(uint8_t) * CHAR_BIT;
    static inline constexpr size_t ALIGNMENT_BSPACE             = sizeof(uint8_t) * CHAR_BIT;
    static inline constexpr ptr_type NETALLOC_NULLPTR           = ptr_type{0u}; 
    static inline constexpr size_t DEFLT_ALIGNMENT              = alignof(double);
    static inline constexpr size_t LEAF_SZ                      = 8u;
    static inline constexpr size_t LEAST_GUARANTEED_ALIGNMENT   = LEAF_SZ;
 
    static_assert(PTROFFS_BSPACE + PTRSZ_BSPACE + ALLOCATOR_ID_BSPACE + ALIGNMENT_BSPACE <= sizeof(ptr_type) * CHAR_BIT);
    static_assert(-1 == ~0);
    static_assert(!NETALLOC_NULLPTR);

    //we would want to batch things, use std::mutex
    //and use another affined allocators to further affine things 
    //this Allocator guarantees no fragmentation if the allocation node lifetime is below the half cap threshold (switchfoot)
    //that Allocator guarantees to free allocations on time, punctually
    //alright, something went wrong
    //we should expose the GC interface... yet we should not rely on GC to deallocate things correctly
    //our heap allocations do not factor in sz as a clearable threshold, we must attempt to solve that here in this extension
    //such is we are automatically invoking the GC every 5%-10% of total memory being allocated without loss of generality
    //our affined allocator should not use internal metrics such as time because that could be not accurate. we should use absolute metrics such should reflect the low level implementation of fragmentation management
    //as long as every node allocation lifetime does not exceed the time it takes to allocate half of the heap, we should be in the fragmentation free guaranteed zone
    //the problem is that I could not prove that choosing the larger of two intervals is a correct approach. the differences might not converge...
    //this implies that we should force a switch unless the other branch is empty. this is the guaranteed way

    class GCInterface{

        public:

            virtual ~GCInterface() noexcept = default;
            virtual void gc() noexcept = 0;
    };
    
    class HeapAllocatorInterface{

        public:

            virtual ~HeapAllocatorInterface() noexcept = default;
            virtual void alloc(size_t * blk_arr, size_t blk_arr_sz, std::optional<interval_type> * interval_arr) noexcept = 0;
            virtual void free(interval_type * interval_arr, size_t sz) noexcept = 0;
    };

    class AllocatorInterface{

        public:

            virtual ~AllocatorInterface() noexcept = default;
            virtual void malloc(size_t * blk_arr, size_t blk_arr_sz, std::optional<ptr_type> * rs) noexcept = 0; //without optional, I feel very empty, and languagely incorrect
            virtual void free(ptr_type * ptr_arr, size_t ptr_arr_sz) noexcept = 0;
            virtual auto c_addr(ptr_type) noexcept -> std::add_pointer_t<void> = 0;
    };

    class GCHeapAllocatorInterface: public virtual HeapAllocatorInterface,
                                    public virtual GCInterface{};

    class GCAllocatorInterface: public virtual AllocatorInterface,
                                public virtual GCInterface{};

    class GCHeapAllocator: public virtual GCHeapAllocatorInterface{

        private:

            std::unique_ptr<char[], decltype(&std::free)> management_buf;
            std::unique_ptr<dg::heap::core::Allocatable> allocator;
            std::unique_ptr<std::mutex> mtx;

        public:

            GCHeapAllocator(std::unique_ptr<char[], decltype(&std::free)> management_buf,
                            std::unique_ptr<dg::heap::core::Allocatable> allocator,
                            std::unique_ptr<std::mutex> mtx) noexcept: management_buf(std::move(management_buf)),
                                                                       allocator(std::move(allocator)),
                                                                       mtx(std::move(mtx)){}

             void alloc(size_t * blk_arr, size_t blk_arr_sz, std::optional<interval_type> * interval_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < blk_arr_sz; ++i){
                    interval_arr[i] = stdx::wrap_safe_integer_cast(this->allocator->alloc(blk_arr[i]));
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
    };

    class Allocator: public virtual GCAllocatorInterface{

        private:

            std::unique_ptr<char[], decltype(&std::free)> buf;
            std::unique_ptr<GCHeapAllocator> base_allocator;

        public:

            Allocator(std::unique_ptr<char[], decltype(&std::free)> buf,
                      std::unique_ptr<GCHeapAllocator> base_allocator) noexcept: buf(std::move(buf)),
                                                                                 base_allocator(std::move(base_allocator)){}

            void malloc(size_t * blk_arr, size_t blk_arr_sz, std::optional<ptr_type> * rs) noexcept{

                dg::network_stack_allocation::NoExceptRawAllocation<size_t[]> node_blk_arr(blk_arr_sz);
                dg::network_stack_allocation::NoExceptAllocation<std::optional<interval_type>[]> intv_arr(blk_arr_sz);

                for (size_t i = 0u; i < blk_arr_sz; ++i){
                    node_blk_arr[i] = blk_arr[i] / LEAF_SZ + size_t{blk_arr[i] % LEAF_SZ != 0u};
                }

                this->base_allocator->alloc(node_blk_arr.get(), blk_arr_sz, intv_arr.get());

                for (size_t i = 0u; i < blk_arr_sz; ++i){
                    if (!intv_arr[i].has_value()){
                        rs[i] = std::nullopt;
                        continue;
                    }

                    auto [resp_off, resp_sz] = intv_arr[i].value();
                
                    static_assert(std::is_unsigned_v<decltype(resp_off)>);
                    static_assert(std::is_unsigned_v<decltype(resp_sz)>);

                    rs[i] = this->encode_ptr(resp_off, resp_sz);
                }
            }

            void free(ptr_type * ptr_arr, size_t ptr_arr_sz) noexcept{

                dg::network_stack_allocation::NoExceptAllocation<interval_type[]> intv_arr(ptr_arr_sz);

                for (size_t i = 0u; i < ptr_arr_sz; ++i){
                    intv_arr[i] = this->decode_ptr(ptr_arr[i]); 
                }

                this->base_allocator->free(intv_arr.get(), ptr_arr_sz);
            }

            auto c_addr(ptr_type ptr) noexcept -> void *{

                auto [off, _] = this->decode_ptr(ptr);
                char * rs = this->buf.get();
                std::advance(rs, off * LEAF_SZ);

                return rs;
            }

            void gc() noexcept{

                this->base_allocator->gc();
            }
        
        private:

            auto encode_ptr(uint64_t hi, uint64_t lo) const noexcept -> ptr_type{
    
                return (static_cast<ptr_type>(hi) << PTRSZ_BSPACE) | static_cast<ptr_type>(lo + 1);
            }

            auto decode_ptr(ptr_type ptr) const noexcept -> std::pair<uint64_t, uint64_t>{

                ptr_type hi = ptr >> PTRSZ_BSPACE;
                ptr_type lo = stdx::low_bit<PTRSZ_BSPACE>(ptr);

                return {static_cast<uint64_t>(hi), static_cast<uint64_t>(lo) - 1};
            }
    };

    //this is the punchline, the best allocator ever written
    //we'll talk about reusability later

    class MultiThreadAllocator: public virtual GCAllocatorInterface{

        private:

            std::vector<std::unique_ptr<Allocator>> allocator_vec;
            size_t malloc_vectorization_sz;
            size_t free_vectorization_sz;

        public:

            MultiThreadAllocator(std::vector<std::unique_ptr<Allocator>> allocator_vec,
                                 size_t malloc_vectorization_sz,
                                 size_t free_vectorization_sz) noexcept: allocator_vec(std::move(allocator_vec)),
                                                                         malloc_vectorization_sz(malloc_vectorization_sz),
                                                                         free_vectorization_sz(free_vectorization_sz){}

            void malloc(size_t * blk_arr, size_t blk_arr_sz, std::optional<ptr_type> * rs) noexcept{

                assert(stdx::is_pow2(allocator_vec.size()));

                size_t allocator_idx                = dg::network_concurrency::this_thread_idx() & (this->allocator_vec.size() - 1u);
                auto internal_resolutor             = InternalMallocFeedResolutor{}:
                internal_resolutor.dst              = &this->allocator_vec[allocator_idx];

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

            void free(ptr_type * ptr_arr, size_t ptr_arr_sz) noexcept{

                auto internal_resolutor             = InternalFreeFeedResolutor{};
                internal_resolutor.dst              = &this->allocator_vec;

                size_t trimmed_keyvalue_feed_cap    = std::min(this->free_vectorization_sz, ptr_arr_sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&internal_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception::remove_expected(dg::network_producer_consumer::delvsrv_kv_open_preallocated_raiihandle(&internal_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                for (size_t i = 0u; i < ptr_arr_sz; ++i){
                    auto [ptr, allocator_idx] = this->decode_ptr(ptr_arr[i]);
                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), allocator_idx, ptr);
                }
            }

            auto c_addr(ptr_type ptr) noexcept -> void *{

                auto [pptr, allocator_idx] = this->decode_ptr(ptr);
                return this->allocator_vec[allocator_idx]->c_addr(pptr);
            }

            void gc() noexcept{ // this might be a bottleneck if more than 1024 concurrent allocators are in use - this is not likely going to be the case - if a computer has more than 1024 cores - it's something wrong with the computer

                for (size_t i = 0u; i < this->allocator_vec.size(); ++i){
                    this->allocator_vec[i]->gc();
                }
            }

        private:

            static inline auto encode_ptr(ptr_type hi, uint64_t lo) const noexcept -> ptr_type{

                return (hi << ALLOCATOR_ID_BSPACE) | static_cast<ptr_type>(lo);
            }

            static inline auto decode_ptr(ptr_type ptr) const noexcept -> std::pair<ptr_type, uint64_t>{

                ptr_type hi = ptr >> ALLOCATOR_ID_BSPACE;
                ptr_type lo = stdx::low_bit<ALLOCATOR_ID_BSPACE>(ptr);

                return {hi, static_cast<uint64_t>(lo)};
            }

            struct InternalMallocFeedArgument{
                szie_t blk_sz;
                std::optional<ptr_type> * rs;
            };

            struct InternalMallocFeedResolutor: dg::network_producer_consumer::ConsumerInterface<InternalMallocFeedArgument>{

                std::unique_ptr<Allocator> * dst;

                void push(std::move_iterator<InternalMallocFeedArgument*> data_arr, size_t data_arr_sz) noexcept{
                    
                    auto base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptRawAllocation<size_t[]> blk_arr(data_arr_sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::optional<ptr_type>[]> rs_arr(data_arr_sz)

                    for (size_t i = 0u; i < data_arr_sz; ++i){
                        blk_arr[i] = base_data_arr[i].blk_sz;
                    }

                    this->dst->malloc(blk_arr.get(), rs_arr.get(), data_arr_sz);

                    for (size_t i = 0u; i < data_arr_sz; ++i){
                        if (!rs_arr[i].has_value()){
                            *base_data_arr[i].rs = std::nullopt;
                            continue;
                        }

                        *base_data_arr[i].rs = MultiThreadAllocator::encode_ptr(rs_arr[i].value(), allocator_idx);
                    }
                }
            };

            struct InternalFreeFeedArgument: dg::network_producer_consumer::KVConsumerInterface<size_t, ptr_type>{

                std::vector<std::unique_ptr<Allocator>> * dst;

                void push(const size_t& allocator_idx, std::move_iterator<ptr_type *> data_arr, size_t sz) noexcept{

                    (*this->dst)[allocator_idx]->free(data_arr.base(), sz);
                }
            };
    };

    struct AllocationBag{
        std::vector<void *> allocation_stack;
        size_t refill_sz;
    };

    //--- I have run numbers, this will work for most of the cases without fragmentations 
    //the reusability of allocation bag is the novel, we can actually stay on the L1 cache for the duration of max_operation_per_flush or flush_interval, whichever comes first
    //the refill_sz just needs to exceed the maximum heap small individual arrays at a random given time   
    //the MultiThreadAllocator has internal mechanisms (Garbage Collector) to make sure that things are working as expected if the allocation_node lifetime is under certain number
    //if the allocation_node_lifetime is not under a certain number, we have unbalanced left right allocations, as long as the incurring anomalies are acceptable, such would put the left right into an equilibrium state of switchfoot allocations, we won't have fragmentation issues
    //even if we have fragmentation issues, our allocation tree is expected to work in the very worst case scenerio (there is no allocation tree in the market that is capable of doing this)

    //assume we have an allocation tree of 1GB per thread
    //500MB of allocations would be the equivalence of 1 millisecond

    class DGStdAllocator{

        private:

            std::shared_ptr<GCAllocatorInterface> base_allocator;
            std::unordered_map<size_t, AllocationBag> allocation_map;
            std::vector<void *> free_bag;
            size_t free_bag_cap;
            std::chrono::nanoseconds flush_interval;
            std::chrono::time_point<std::chrono::high_resolution_clock> last_flush;
            size_t operation_counter;
            size_t max_operation_per_flush;
            size_t minimum_allocation_blk_sz;
            size_t maximum_smallbin_blk_sz;
            size_t pow2_malloc_chk_interval_sz; 

        public:

            DGStdAllocator(std::shared_ptr<GCAllocatorInterface> base_allocator,
                           std::unordered_map<size_t, AllocationBag> allocation_map,
                           std::vector<void *> free_bag,
                           size_t free_bag_cap,
                           std::chrono::nanoseconds flush_interval,
                           std::chrono::time_point<std::chrono::high_resolution_clock> last_flush,
                           size_t operation_counter,
                           size_t max_operation_per_flush,
                           size_t minimum_allocation_blk_sz,
                           size_t maximum_smallbin_blk_sz,
                           size_t pow2_malloc_chk_interval_sz) noexcept: base_allocator(std::move(base_allocator)),
                                                                    allocation_map(std::move(allocation_map)),
                                                                    free_bag(std::move(free_bag)),
                                                                    free_bag_cap(free_bag_cap),
                                                                    flush_interval(flush_interval),
                                                                    last_flush(last_flush),
                                                                    operation_counter(operation_counter),
                                                                    max_operation_per_flush(max_operation_per_flush),
                                                                    minimum_allocation_blk_sz(minimum_allocation_blk_sz),
                                                                    maximum_smallbin_blk_sz(maximum_smallbin_blk_sz),
                                                                    pow2_malloc_chk_interval_sz(pow2_malloc_chk_interval_sz){}

            ~DGStdAllocator() noexcept{

                this->internal_cleanup();
            }

            auto malloc(size_t blk_sz) noexcept -> void *{

                if (blk_sz == 0u){
                    return nullptr;
                }

                if ((this->operation_counter & (this->pow2_malloc_chk_interval_sz - 1u)) == 0u || blk_sz > this->maximum_smallbin_blk_sz) [[unlikely]]{
                    return this->internal_careful_malloc(blk_sz);
                } else [[likely]]{
                    size_t pow2_blk_sz  = stdx::ceil2(std::max(blk_sz, this->minimum_allocation_blk_sz)); //ceil2 is just a way to describe things, we can do ceil 1.2, 1.3 etc. 
                    auto map_ptr        = this->allocation_map.find(pow2_blk_sz);

                    if (map_ptr->second.allocation_stack.empty()) [[unlikely]]{
                        return this->internal_careful_malloc(blk_sz);
                    } else [[likely]]{
                        void * blk = map_ptr->second.allocation_stack.back();
                        map_ptr->second.allocation_stack.pop_back(); 
                        this->operation_counter += 1;

                        return blk;
                    }
                }
            }

            void free(void * ptr) noexcept{

                //free is a through operation, noaction, because we dont want to trigger things in the free functions, it makes no sense

                // if (this->free_bag.size() == this->free_bag_cap) [[unlikely]]{
                //     this->internal_flush_free_bag();
                // }

                //this is where we want to reuse the freed alloctions
                //we have not come up with a way to reasonably not write header to void * ptr, this is a very expensive operation  

                // this->free_bag.push_back(ptr);
            }

        private:

            __attribute__((noinline)) auto internal_careful_malloc(size_t blk_sz) noexcept -> void *{

            }

            __attribute__((noinline)) void internal_flush_free_bag() noexcept{

            }

            void internal_cleanup(){

            }
    };

    class ConcurrentDGStdAllocator{

        private:

            std::vector<DGStdAllocator> dg_allocator_vec;
        
        public:

            ConcurrentDGStdAllocator(std::vector<DGStdAllocator> dg_allocator_vec) noexcept: dg_allocator_vec(std::move(dg_allocator_vec)){}

            auto malloc(size_t blk_sz) noexcept -> void *{

                return this->dg_allocator_vec[dg::network_concurrency::this_thread_idx()].malloc(blk_sz);
            }

            void free(void * ptr) noexcept{

                this->dg_allocator_vec[dg::network_concurrency::this_thread_idx()].free(ptr);
            }
    };

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

    struct Factory{

        static auto spawn_heap_allocator(size_t base_sz) -> std::unique_ptr<GCHeapAllocator>{ //devirt here is important

            const size_t MIN_BASE_SZ    = 1u;
            const size_t MAX_BASE_SZ    = size_t{1} << 40;

            if (std::clamp(base_sz, MIN_BASE_SZ, MAX_BASE_SZ) != base_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!!dg::memult::is_pow2(base_sz)){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            uint8_t tree_height = stdx::ulog2_aligned(base_sz) + 1u;
            size_t buf_sz       = dg::heap::user_interface::get_memory_usage(tree_height);
            auto buf            = std::unique_ptr<char[], decltype(&std::free)>(static_cast<char *>(std::malloc(buf_sz)), std::free);

            if (!buf){
                dg::network_exception::throw_exception(dg::network_exception::OUT_OF_MEMORY);
            }

            auto allocator  = dg::heap::user_interface::get_allocator_x(buf.get());
            auto lck        = std::make_unique<std::atomic_flag>();
            auto rs         = std::make_unique<GCHeapAllocator>(std::move(buf), std::move(allocator), std::move(lck));

            return rs;
        }

        static auto spawn_allocator(size_t least_buf_sz) -> std::unique_ptr<Allocator>{ //devirt here is important

            const size_t MIN_LEAST_BUF_SZ   = 1u;
            const size_t MAX_LEAST_BUF_SZ   = size_t{1} << 40;

            if (std::clamp(least_buf_sz, MIN_LEAST_BUF_SZ, MAX_LEAST_BUF_SZ) != least_buf_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            size_t buf_sz   = stdx::least_pow2_greater_equal_than(std::max(least_buf_sz, LEAF_SZ));
            size_t base_sz  = buf_sz / LEAF_SZ;
            auto buf        = std::unique_ptr<char[], decltype(&std::free)>(static_cast<char *>(std::aligned_alloc(LEAF_SZ, buf_sz)), std::free);  

            if (!buf){
                dg::network_exception::throw_exception(dg::network_exception::OUT_OF_MEMORY);
            }

            std::unique_ptr<GCHeapAllocator> base_allocator = spawn_heap_allocator(base_sz);
            return std::make_unique<Allocator>(std::move(buf), std::move(base_allocator));
        }

        static auto spawn_concurrent_allocator(std::vector<std::unique_ptr<Allocator>> allocator) -> std::unique_ptr<MultiThreadAllocator>{ //devirt here is important - 

            const size_t MIN_ALLOCATOR_SZ   = 1u;
            const size_t MAX_ALLOCATOR_SZ   = size_t{1} << 8;

            if (std::clamp(static_cast<size_t>(allocator.size()), MIN_ALLOCATOR_SZ, MAX_ALLOCATOR_SZ) != allocator.size()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (!dg::memult::is_pow2(allocator.size())){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (std::find(allocator.begin(), allocator.end(), nullptr) != allocator.end()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return std::make_unique<MultiThreadAllocator>(std::move(allocator));
        }

        static auto spawn_gc_worker(std::shared_ptr<GCInterface> gc_able) -> dg::network_concurrency::daemon_raii_handle_t{ //this is strange - this overstep into the responsibility - decouple the component

            if (gc_able == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            auto worker = std::make_unique<GCWorker>(std::move(gc_able));
            auto rs     = dg::network_concurrency::daemon_saferegister(dg::network_concurrency::COMPUTING_DAEMON, std::move(worker));

            if (!rs.has_value()){
                dg::network_exception::throw_exception(rs.error());
            } 

            return std::move(rs.value());
        }
    };

    struct AllocationResource{
        std::shared_ptr<MultiThreadAllocator> allocator; //devirt here is important - 
        dg::network_concurrency::daemon_raii_handle_t gc_worker;
    };

    inline AllocationResource allocation_resource;

    void init(size_t least_buf_sz, size_t num_allocator){

        std::vector<std::unique_ptr<Allocator>> allocator_vec{};

        for (size_t i = 0u; i < num_allocator; ++i){
            allocator_vec.push_back(Factory::spawn_allocator(least_buf_sz));
        }

        std::shared_ptr<MultiThreadAllocator> allocator = Factory::spawn_concurrent_allocator(std::move(allocator_vec));
        dg::network_concurrency::daemon_raii_handle_t daemon_handle = Factory::spawn_gc_worker(allocator);
        allocation_resource = {std::move(allocator), std::move(daemon_handle)};
    }

    void deinit() noexcept{

        allocation_resource = {};
    }

    auto malloc(size_t blk_sz, size_t alignment) noexcept -> ptr_type{

        assert(dg::memult::is_pow2(alignment));

        if (blk_sz == 0u){
            return NETALLOC_NULLPTR;
        }
        
        size_t fwd_sz       = std::max(alignment, LEAST_GUARANTEED_ALIGNMENT) - LEAST_GUARANTEED_ALIGNMENT;
        size_t adj_blk_sz   = blk_sz + fwd_sz;
        ptr_type ptr        = allocation_resource.allocator->malloc(adj_blk_sz);

        if (!ptr){
            return NETALLOC_NULLPTR;
        }

        alignment_type alignment_log2 = stdx::ulog2_aligned(static_cast<alignment_type>(alignment));
        ptr <<= ALIGNMENT_BSPACE;
        ptr |= static_cast<ptr_type>(alignment_log2);

        return ptr;
    } 

    auto malloc(size_t blk_sz) noexcept -> ptr_type{

        return malloc(blk_sz, DEFLT_ALIGNMENT); 
    }

    auto c_addr(ptr_type ptr) noexcept -> void *{
        
        if (!ptr){
            return nullptr;
        }

        alignment_type alignment_log2   = stdx::low_bit<ALIGNMENT_BSPACE>(ptr); 
        size_t alignment                = size_t{1} << alignment_log2;
        ptr_type pptr                   = ptr >> ALIGNMENT_BSPACE;

        return dg::memult::align(allocation_resource.allocator->c_addr(pptr), alignment);
    }

    void free(ptr_type ptr) noexcept{

        if (!ptr){
            return;
        }

        ptr_type pptr = ptr >> ALIGNMENT_BSPACE;
        allocation_resource.allocator->free(pptr);
    }

    auto cmalloc(size_t blk_sz) noexcept -> void *{

        if (blk_sz == 0u){
            return nullptr;
        }

        constexpr size_t HEADER_SZ  = std::max(stdx::least_pow2_greater_equal_than(sizeof(ptr_type)), DEFLT_ALIGNMENT);
        size_t adj_blk_sz           = blk_sz + HEADER_SZ;
        ptr_type ptr                = malloc(adj_blk_sz);

        if (!ptr){
            return nullptr;
        }

        void * cptr = c_addr(ptr);
        std::memcpy(cptr, &ptr, sizeof(ptr_type));
        char * rs   = static_cast<char *>(cptr);
        std::advance(rs, HEADER_SZ);

        return rs;
    }

    void cfree(void * cptr) noexcept{
        
        constexpr size_t HEADER_SZ = std::max(stdx::least_pow2_greater_equal_than(sizeof(ptr_type)), DEFLT_ALIGNMENT);
        
        if (!cptr){
            return;
        }

        const char * org_cptr = static_cast<const char *>(cptr);
        std::advance(org_cptr, -static_cast<intmax_t>(HEADER_SZ));
        ptr_type ptr    = {};
        std::memcpy(&ptr, org_cptr, sizeof(ptr_type));

        free(ptr);
    }

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

    template <class T, class ...Args>
    auto std_new(Args&& ...args) -> T *{

        return new T(std::forward<Args>(args)...);
    }

    template <class = void>
    static inline constexpr bool FALSE_VAL = false;

    template <class T>
    auto std_delete(std::remove_extent_t<T> * obj) noexcept(std::is_nothrow_destructible_v<std::remove_extent_t<T>>){

        if constexpr(std::is_array_v<T>){
            if constexpr(std::is_unbounded_array_v<T>){
                delete[] obj;
            } else{
                static_assert(FALSE_VAL<>);
            }
        } else{
            delete obj;
        }
    }

    template <class T, class ...Args>
    auto make_unique(Args&& ...args) -> decltype(auto){

        return std::make_unique<T>(std::forward<Args>(args)...);
    }
}

#endif
