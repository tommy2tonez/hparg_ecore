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
    static inline constexpr size_t ALLOCATOR_ID_BSPACE          = sizeof(uint16_t) * CHAR_BIT;
    static inline constexpr size_t ALIGNMENT_BSPACE             = sizeof(uint16_t) * CHAR_BIT;
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

    class HeapAllocator{

        private:

            std::unique_ptr<char[], decltype(&std::free)> management_buf;
            std::unique_ptr<dg::heap::core::Allocatable> allocator;
            std::unique_ptr<std::mutex> mtx;

        public:

            HeapAllocator(std::unique_ptr<char[], decltype(&std::free)> management_buf,
                          std::unique_ptr<dg::heap::core::Allocatable> allocator,
                          std::unique_ptr<std::mutex> mtx) noexcept: management_buf(std::move(management_buf)),
                                                                     allocator(std::move(allocator)),
                                                                     mtx(std::move(mtx)){}

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

             auto base_size() const noexcept{
                
             }
    };

    class MultiThreadUniformHeapAllocator{

        private:

            std::vector<std::unique_ptr<HeapAllocator>> allocator_vec;
            size_t malloc_vectorization_sz;
            size_t free_vectorization_sz;
            size_t heap_allocator_pow2_exp_base_sz; //let me think of how to split this 

        public:

            MultiThreadUniformHeapAllocator(std::vector<std::unique_ptr<HeapAllocator>> allocator_vec,
                                            size_t malloc_vectorization_sz,
                                            size_t free_vectorization_sz,
                                            size_t heap_allocator_pow2_exp_base_sz) noexcept: allocator_vec(std::move(allocator_vec)),
                                                                                              malloc_vectorization_sz(malloc_vectorization_sz),
                                                                                              free_vectorization_sz(free_vectorization_sz),
                                                                                              heap_allocator_pow2_exp_base_sz(heap_allocator_pow2_exp_base_sz){}

            void malloc(size_t * blk_arr, size_t blk_arr_sz, std::optional<interval_type> * rs) noexcept{

                assert(stdx::is_pow2(allocator_vec.size()));

                size_t allocator_idx                = dg::network_concurrency::this_thread_idx() & (this->allocator_vec.size() - 1u);
                auto internal_resolutor             = InternalMallocFeedResolutor{}:
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
                auto feeder                         = dg::network_exception::remove_expected(dg::network_producer_consumer::delvsrv_kv_open_preallocated_raiihandle(&internal_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

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

        private:

            struct InternalMallocFeedArgument{
                szie_t blk_sz;
                std::optional<interval_type> * rs;
            };

            struct InternalMallocFeedResolutor: dg::network_producer_consumer::ConsumerInterface<InternalMallocFeedArgument>{

                std::unique_ptr<Allocator> * dst;
                size_t allocation_off;

                void push(std::move_iterator<InternalMallocFeedArgument *> data_arr, size_t data_arr_sz) noexcept{

                    auto base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptRawAllocation<size_t[]> blk_arr(data_arr_sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::optional<interval_type>[]> rs_arr(data_arr_sz)

                    for (size_t i = 0u; i < data_arr_sz; ++i){
                        blk_arr[i] = base_data_arr[i].blk_sz;
                    }

                    this->dst->malloc(blk_arr.get(), rs_arr.get(), data_arr_sz);

                    for (size_t i = 0u; i < data_arr_sz; ++i){
                        if (!rs_arr[i].has_value()){
                            *base_data_arr[i].rs = std::nullopt;
                            continue;
                        }

                        *base_data_arr[i].rs = std::make_pair(rs_arr[i]->first + this->allocation_off, rs_arr[i]->second);
                    }
                }
            };

            struct InternalFreeFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, ptr_type>{

                std::vector<std::unique_ptr<Allocator>> * dst;

                void push(const size_t& allocator_idx, std::move_iterator<ptr_type *> data_arr, size_t sz) noexcept{

                    (*this->dst)[allocator_idx]->free(data_arr.base(), sz);
                }
            };
    };

    class BumpAllocator{

        private:

            char * buf;
            size_t sz;
        
        public:

            BumpAllocator() noexcept: buf(nullptr), 
                                      sz(0u){}

            BumpAllocator(char * buf, size_t sz) noexcept: buf(buf), 
                                                           sz(sz){}

            auto malloc(size_t blk_sz) noexcept -> std::expected<void *, exception_t>{

                if (blk_sz > this->sz){
                    return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                }

                char * rs   = this->buf;
                std::advance(this->buf, blk_sz);
                this->sz    -= blk_sz;

                return static_cast<void *>(rs);
            }

            auto decommission() noexcept -> std::pair<void *, size_t>{

                auto rs     = std::make_pair(static_cast<void *>(this->buf), this->sz);
                this->buf   = nullptr;
                this->sz    = 0u;

                return rs;
            }
    };

    struct Allocation{
        void * user_ptr;
        size_t user_ptr_sz;
    };

    struct LargeBinMetadata{
        size_t user_ptr_sz;
    };

    //this is harder to write than most people think
    //I couldn't even semanticalize the methods naturally, it's complicated
    //this should be clear

    //let's do one more round of review, let's see our assumption, I'll be back
    //we'll do something called induction programming, as long as we keep all our assumption in all the functions, we should be clear
    //this is actually hard to write
    //we'll do one more review, this time is stack review
    //clear
    //alright, this is the third time I've said this, this has too many components + implementation requirements that are hard to do, yet we've completed the component 

    template <size_t HEAP_LEAF_UNIT_ALLOCATION_SZ>
    class DGStdAllocator{

        private:

            BumpAllocator bump_allocator;
            size_t bump_allocator_refill_sz;
            size_t bump_allocator_version_control; 
            std::shared_ptr<char[]> buf;
            std::shared_ptr<MultiThreadUniformHeapAllocator> heap_allocator;
            std::vector<dg::pow2_cyclic_queue<Allocation>> smallbin_reuse_table;
            size_t minimum_allocation_blk_sz;
            size_t maximum_smallbin_blk_sz;
            std::vector<Allocation> freebin_vec;
            size_t freebin_vec_cap;

            size_t pow2_malloc_chk_interval_sz;
            size_t malloc_chk_interval_counter;
            std::chrono::time_point<std::chrono::high_resolution_clock> last_flush;
            std::chrono::nanoseconds flush_interval;
            size_t allocation_sz_counter;
            size_t allocation_sz_counter_flush_threshold; 

            dg::network_datastructure::unordered_map_variants::unordered_node_map<uintptr_t, LargeBinMetadata> largebin_metadata_dict; 
            size_t largebin_metadata_dict_cap; //we've got bad feedbacks about this, I dont think that's an issue, we are allocating 64KB chunk of memory, 16 bytes of largebin's not gonna dent our overheads

        public:

            using self              = DGStdAllocator;
            using sz_header_t       = uint16_t;
            using vrs_ctrl_header_t = uint16_t;

            static inline constexpr size_t ALLOCATION_HEADER_SZ = sizeof(sz_header_t) + sizeof(vrs_ctrl_header_t);  

            DGStdAllocator(BumpAllocator bump_allocator,
                           size_t bump_allocator_refill_sz,
                           size_t bump_allocator_version_control,
                           std::shared_ptr<char[]> buf,
                           std::shared_ptr<MultiThreadUniformHeapAllocator> heap_allocator,
                           std::vector<dg::pow2_cyclic_queue<Allocation>> smallbin_reuse_table,
                           size_t minimum_allocation_blk_sz,
                           size_t maximum_smallbin_blk_sz,
                           std::vector<Allocation> freebin_vec,
                           size_t freebin_vec_cap, 

                           size_t pow2_malloc_chk_interval_sz,
                           size_t malloc_chk_interval_counter,
                           std::chrono::time_point<std::chrono::high_resolution_clock> last_flush,
                           std::chrono::nanoseconds flush_interval,
                           size_t allocation_sz_counter,
                           size_t allocation_sz_counter_flush_threshold,

                           dg::network_datastructure::unordered_map_variants::unordered_node_map<uintptr_t, LargeBinMetadata> largebin_metadata_dict,
                           size_t largebin_metadata_dict_cap) noexcept: bump_allocator(bump_allocator),
                                                                        bump_allocator_refill_sz(bump_allocator_refill_sz),
                                                                        bump_allocator_version_control(bump_allocator_version_control),
                                                                        buf(std::move(buf)),
                                                                        heap_allocator(std::move(heap_allocator)),
                                                                        smallbin_reuse_table(std::move(smallbin_reuse_table)),
                                                                        minimum_allocation_blk_sz(minimum_allocation_blk_sz),
                                                                        maximum_smallbin_blk_sz(maximum_smallbin_blk_sz),
                                                                        freebin_vec(std::move(freebin_vec)),
                                                                        freebin_vec_cap(freebin_vec_cap),
                                                                        pow2_malloc_chk_interval_sz(pow2_malloc_chk_interval_sz),
                                                                        malloc_chk_interval_counter(malloc_chk_interval_counter),
                                                                        last_flush(last_flush),
                                                                        flush_interval(flush_interval),
                                                                        allocation_sz_counter(allocation_sz_counter),
                                                                        allocation_sz_counter_flush_threshold(allocation_sz_counter_flush_threshold),
                                                                        largebin_metadata_dict(std::move(largebin_metadata_dict)),
                                                                        largebin_metadata_dict_cap(largebin_metadata_dict_cap){}

            ~DGStdAllocator() noexcept{

                this->internal_commit_waiting_bin();
            }

            inline auto malloc(size_t blk_sz) noexcept -> void *{

                //returns nullptr when there are no actions required for resource inverse operation
                //nullptr => blk_sz == 0u or resource exhaustion
 
                if (blk_sz == 0u){
                    return nullptr;
                }

                this->malloc_chk_interval_counter += 1u;

                if ((this->malloc_chk_interval_counter & (this->pow2_malloc_chk_interval_sz - 1u)) == 0u || blk_sz > this->maximum_smallbin_blk_sz) [[unlikely]]{
                    return this->internal_careful_malloc(blk_sz); //assume that internal_careful_malloc allocates largebin_ptr, writes max_sz + 1u to the header for free dispatch code, and keep all smallbins <= maximum_smallbin_blk_sz to be allocated by internal_bump_allocate
                } else [[likely]]{
                    size_t pow2_blk_sz          = stdx::ceil2(std::max(blk_sz, this->minimum_allocation_blk_sz));
                    size_t smallbin_table_idx   = stdx::ulog2(pow2_blk_sz); 
                    auto& smallbin_vec          = this->smallbin_reuse_table[smallbin_table_idx];

                    if (smallbin_vec.empty()) [[unlikely]]{
                        return this->internal_bump_allocate(blk_sz); //assume internal_bump_allocate allocates smallbin user_ptr such thats user_ptr_sz <= maximum_smallbin_blk_sz 
                    } else [[likely]]{
                        void * rs = smallbin_vec.back().user_ptr;
                        smallbin_vec.pop_back();

                        return rs; //assume that smallbin_vec is originated from internal_bump_allocate
                    }
                }
            }

            //even if we are reallocating a valid buffer -> size of 0, std::free has to be invoked for non-null ptrs returned by this function
            //this is not a good assumption...

            inline auto realloc(void * user_ptr, size_t blk_sz) noexcept -> void *{

                //this is clear

                //nullptr special dispatch code, because malloc returns nullptr, realloc has to consider nullptr

                if (user_ptr == nullptr || blk_sz == 0u){
                    this->free(user_ptr);
                    return this->malloc(blk_sz);
                }

                size_t user_ptr_sz = self::internal_read_user_ptr_size(user_ptr);

                if (user_ptr_sz > maximum_smallbin_blk_sz){ //this is large size allocation as specified in malloc()
                    //maximum user_ptr_sz triggered
                    //i have yet to know if to read the large size allocation metadata here
                    size_t user_largeptr_sz = this->internal_read_largemalloc_user_ptr_size(user_ptr); 

                    if (blk_sz <= user_largeptr_sz){
                        return user_ptr;
                    } else{
                        this->free(user_ptr);
                        return this->malloc(blk_sz);
                    }
                } else{
                    //this is small size allocation, user_ptr_sz denotes the exact size of user_ptr

                    if (blk_sz <= user_ptr_sz){ //shrinking the sz, user_ptr_sz should do
                        return user_ptr;
                    } else{
                        this->free(user_ptr);
                        return this->malloc(blk_sz);
                    }
                }
            }

            inline void free(void * user_ptr) noexcept{

                //nullptr special dispatch code, because malloc returns nullptr, free has to consider nullptr
                if (user_ptr == nullptr){
                    return;
                }

                size_t user_ptr_sz                          = self::internal_read_user_ptr_size(user_ptr);
                size_t user_ptr_truncated_version_control   = self::internal_read_user_ptr_truncated_version_control(user_ptr);  

                if (user_ptr_sz > this->maximum_smallbin_blk_sz) [[unlikely]]{ //user_ptr_sz > maximum_smallbin_blk_sz, guaranteed by malloc() to be allocated by largemalloc(), dispatch the inverse operation, internal_large_free()
                    this->internal_large_free(user_ptr);
                } else [[likely]]{
                    //user_ptr_sz <= maximum_smallbin_blk_sz, guaranteed to be allocated by internal_bump_allocate() by malloc() 

                    size_t current_truncated_version_control    = self::internal_get_truncated_version_control(this->bump_allocator_version_control);
                    size_t floor_smallbin_table_idx             = stdx::ulog2(user_ptr_sz); //floor2 the user_ptr_sz, because it is a valid user_ptr_sz (> 0) by internal_bump_malloc()
                    auto& smallbin_vec                          = this->smallbin_reuse_table[floor_smallbin_table_idx]; //getting the bin

                    if (smallbin_vec.size() != smallbin_vec.capacity() && current_truncated_version_control == user_ptr_truncated_version_control) [[likely]]{
                        //meets the cond of fast free
                        dg::network_exception::dg_noexcept(smallbin_vec.push_back(Allocation{user_ptr, user_ptr_sz}));
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

            static constexpr auto internal_get_truncated_version_control(size_t version_control) noexcept -> size_t{

                return version_control & static_cast<size_t>(std::numeric_limits<vrs_ctrl_header_t>::max()); //static cast should suffice
            } 

            inline auto internal_has_largemalloc_dict_room() const noexcept -> bool{

                return this->largebin_metadata_dict.size() != this->largebin_metadata_dict_cap;
            }

            inline void internal_insert_largemalloc_metadata(void * user_ptr, LargeBinMetadata metadata) noexcept{

                auto [_, insert_status] = this->largebin_metadata_dict.insert(std::make_pair(reinterpret_cast<uintptr_t>(user_ptr), std::move(metadata)));
                assert(insert_status); //declare assumptions
            }

            inline void internal_read_largemalloc_user_ptr_size(const void * user_ptr) const noexcept -> size_t{

                auto map_ptr = this->largebin_metadata_dict.find(reinterpret_cast<uintptr_t>(user_ptr));

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->largebin_metadata_dict.end()){
                        std::abort();
                    }

                    return map_ptr->second.user_ptr_sz;
                } else{
                    return map_ptr->second.user_ptr_sz;
                }
            }

            inline void internal_erase_largemalloc_entry(void * user_ptr) noexcept{

                size_t removed_sz = this->largebin_metadata_dict.erase(reinterpret_cast<uintptr_t>(user_ptr));
                assert(removed_sz == 1u);
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

            static constexpr auto internal_get_internal_ptr_head(void * user_ptr) noexcept -> void *{

                return std::prev(static_cast<char *>(user_ptr), ALLOCATION_HEADER_SZ);
            }

            static constexpr auto internal_get_internal_ptr_head(const void * user_ptr) noexcept -> const void *{

                return std::prev(static_cast<const char *>(user_ptr), ALLOCATION_HEADER_SZ);
            }

            static constexpr auto internal_get_user_ptr_head(void * internal_ptr) noexcept -> void *{
                
                return std::next(static_cast<char *>(internal_ptr), ALLOCATION_HEADER_SZ);
            }

            static constexpr auto internal_get_user_ptr_head(const void * internal_ptr) noexcept -> const void *{

                return std::next(static_cast<const char *>(internal_ptr), ALLOCATION_HEADER_SZ);
            }

            inline void internal_update_allocation_sensor(size_t new_blk_sz) noexcept{

                this->allocation_sz_counter += new_blk_sz;
            }

            static constexpr auto internal_read_user_ptr_size(const void * user_ptr) noexcept -> size_t{

                const void * ptr_head       = self::internal_get_internal_ptr_head(user_ptr);
                sz_header_t allocation_sz   = {};
                std::memcpy(&allocation_sz, ptr_head, sizeof(sz_header_t));

                return allocation_sz; 
            }

            static constexpr auto internal_read_user_ptr_truncated_version_control(const void * user_ptr) noexcept -> size_t{

                const void * ptr_head               = self::internal_get_internal_ptr_head(user_ptr);
                const void * version_control_head   = std::next(static_cast<const char *>(ptr_head), sizeof(sz_header_t)); 
                vrs_ctrl_header_t version_ctrl      = {};
                std::memcpy(&version_ctrl, version_control_head, sizeof(vrs_ctrl_header_t));

                return version_ctrl;
            }

            inline auto internal_write_allocation_header(void * internal_ptr, sz_header_t user_ptr_sz, vrs_ctrl_header_t version_control) const noexcept -> void *{

                std::memcpy(internal_ptr, &user_ptr_sz, sizeof(sz_header_t));
                std::memcpy(std::next(static_cast<char *>(internal_ptr), sizeof(sz_header_t)), &version_control, sizeof(vrs_ctrl_header_t));

                return self::internal_get_user_ptr_head(internal_ptr);
            }

            __attribute__((noinline)) void internal_dump_freebin_vec(){

                //assume that freebin is only for smallbins, thus there is no largemalloc dict clear 

                dg::network_stack_allocation::NoExceptAllocation<interval_type[]> intv_arr(this->freebin_vec.size());

                for (size_t i = 0u; i < this->freebin_vec.size(); ++i){
                    auto [user_ptr, user_ptr_sz]    = std::make_pair(this->freebin_vec[i].user_ptr, this->freebin_vec[i].user_ptr_sz);

                    size_t internal_ptr_offset      = std::distance(this->buf.get(), static_cast<char *>(self::internal_get_internal_ptr_head(user_ptr)));
                    size_t internal_ptr_sz          = user_ptr_sz + ALLOCATION_HEADER_SZ;

                    size_t heap_ptr_offset          = internal_ptr_offset / HEAP_LEAF_UNIT_ALLOCATION_SZ;
                    size_t heap_ptr_sz              = internal_ptr_sz / HEAP_LEAF_UNIT_ALLOCATION_SZ;
                    size_t heap_ptr_excl_sz         = heap_ptr_sz - 1u;

                    intv_arr[i]                     = std::make_pair(heap_ptr_offset, heap_ptr_excl_sz);
                }

                this->heap_allocator->free(intv_arr.get(), this->freebin_vec.size());
                this->freebin_vec.clear();
            }

            inline void internal_decommission_bump_allocator() noexcept{

                auto [decom_buf, decom_sz] = this->bump_allocator.decommission();

                if (decom_sz != 0u){
                    //assumption is clear, assume that bump_allocator remaining is aligned, this is achieved by aligned refill_sz + aligned bump_malloc_sz
                    auto heap_interval = this->internal_aligned_buf_to_interval({static_cast<char *>(decom_buf), decom_sz});
                    this->heap_allocator->free(&heap_interval, 1u);
                }
            }

            inline auto internal_dispatch_bump_allocator_refill() noexcept -> bool{

                this->internal_decommission_bump_allocator();

                size_t requesting_interval_sz   = this->bump_allocator_refill_sz / HEAP_LEAF_UNIT_ALLOCATION_SZ;
                auto requesting_interval        = std::optional<interval_type>{};

                this->heap_allocator->malloc(&requesting_interval_sz, 1u, &requesting_interval);

                if (!requesting_interval.has_value()){
                    return false;
                }

                std::pair<char *, size_t> requesting_buf    = this->internal_interval_to_buf(requesting_interval.value());
                this->bump_allocator                        = BumpAllocator(requesting_buf.first, requesting_buf.second);
                this->bump_allocator_version_control        += 1;

                this->internal_update_allocation_sensor(this->bump_allocator_refill_sz);

                return true;
            }

            __attribute__((noinline)) auto internal_bump_allocate(size_t user_blk_sz) noexcept -> void *{

                assert(user_blk_sz != 0u);
                assert(user_blk_sz <= this->maximum_smallbin_blk_sz);

                //now this is hard
                //maximum_smallbin_blk_sz   == 60, header_sz == 4
                //maximum_allocated_sz      == 64
                //user_usable_blk_sz        == 60 <= maximum_smallbin_blk_sz
                //things are clear
                //we dont want to trip the threshold, because that would break the contract of maximum_smallbin_blk_sz by large_free

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

                for (size_t i = 0u; i < this->smallbin_reuse_table.size(); ++i){
                    for (size_t j = 0u; j < this->smallbin_reuse_table[i].size(); ++j){
                        if (this->freebin_vec.size() == this->freebin_vec_cap){
                            this->internal_dump_freebin_vec();
                        }

                        this->freebin_vec.push_back(this->smallbin_reuse_table[i][j]);
                    }

                    this->smallbin_reuse_table[i].clear();
                }

                this->internal_dump_freebin_vec();
            }

            void internal_reset() noexcept{

                this->internal_commit_waiting_bin();

                this->last_flush            = std::chrono::high_resolution_clock::now();
                this->allocation_sz_counter = 0u;
            }

            void internal_check_for_reset() noexcept{
                
                std::chrono::time_point<std::chrono::high_resolution_clock> now             = std::chrono::high_resolution_clock::now();
                std::chrono::time_point<std::chrono::high_resolution_clock> flush_expiry    = this-last_flush + this->flush_interval;

                bool reset_cond_1   = this->allocation_sz_counter >= this->allocation_sz_counter_flush_threshold;
                bool reset_cond_2   = now >= flush_expiry; 

                if (reset_cond_1 || reset_cond_2){
                    this->internal_reset();
                }
            }

            //this is complicated
            auto internal_large_malloc(size_t user_blk_sz) noexcept -> void *{

                assert(user_blk_sz > this->maximum_smallbin_blk_sz);

                if (!this->internal_has_largemalloc_dict_room()){
                    return nullptr;
                }

                size_t pad_blk_sz                           = user_blk_sz + ALLOCATION_HEADER_SZ;
                size_t ceil_blk_sz                          = (((pad_blk_sz - 1u) / HEAP_LEAF_UNIT_ALLOCATION_SZ) + 1u) * HEAP_LEAF_UNIT_ALLOCATION_SZ;
                size_t user_usable_blk_sz                   = ceil_blk_sz - ALLOCATION_HEADER_SZ;
                size_t allocating_heap_node_sz              = ceil_blk_sz / HEAP_LEAF_UNIT_ALLOCATION_SZ; 
                std::optional<interval_type> allocated_intv = {};

                this->heap_allocator->malloc(&allocating_heap_node_sz, 1u, &allocated_intv);

                if (!allocated_intv.has_value()){
                    return nullptr;
                }

                std::pair<char *, size_t> requesting_buf    = this->internal_interval_to_buf(allocated_intv.value());
                void * internal_ptr                         = requesting_buf.first;
                void * user_ptr                             = this->internal_write_allocation_header(internal_ptr, 
                                                                                                     stdx::wrap_safe_integer_cast(this->maximum_smallbin_blk_sz + 1u), 
                                                                                                     0u); //writing maximum_smallbin_blk_sz + 1u for free dispatch code, vrs_ctrl is irrelevant, can be of any values

                this->internal_insert_largemalloc_metadata(user_ptr, LargeBinMetadata{user_usable_blk_sz}); //writing largebin metadata -> user_ptr, not interal ptr, assume rs is not registered, this assumption is not inductable, yet could be proved, assume rs is registered, user_ptr double mallocs ptr (this is internal heap corruption) or rs is not cleared in the last free (internal_corruption)
                this->internal_update_allocation_sensor(ceil_blk_sz);

                return user_ptr;
            }

            __attribute__((noinline)) void internal_large_free(void * user_ptr) noexcept{
                
                //inverse operation of internal_large_malloc

                void * internal_ptr         = self::internal_get_internal_ptr_head(user_ptr);
                size_t user_ptr_sz          = this->internal_read_largemalloc_user_ptr_size(user_ptr); //reading largebin metadata from user_ptr

                size_t internal_ptr_offset  = std::distance(this->buf.get(), static_cast<char *>(internal_ptr));
                size_t internal_ptr_sz      = user_ptr_sz + ALLOCATION_HEADER_SZ; 

                //assume that allocations are HEAP_LEAF_UNIT_ALLOCATION_SZ aligned in terms of offset + sz
                size_t heap_ptr_offset      = internal_ptr_offset / HEAP_LEAF_UNIT_ALLOCATION_SZ;
                size_t heap_ptr_sz          = internal_ptr_sz / HEAP_LEAF_UNIT_ALLOCATION_SZ;
                size_t heap_ptr_excl_sz     = heap_ptr_sz - 1u;

                interval_type intv          = std::make_pair(heap_ptr_offset, heap_ptr_excl_sz);

                this->internal_erase_largemalloc_entry(user_ptr);
                this->heap_allocator->free(&intv, 1u);
            }

            //as far as we concerned, this could be one of the public APIs, this function is external sufficient by itself (there are no internal assumptions or asserts)
            __attribute__((noinline)) auto internal_careful_malloc(size_t user_blk_sz) noexcept -> void *{

                if (user_blk_sz == 0u){
                    return nullptr;
                }

                this->internal_check_for_reset();

                if (user_blk_sz > this->maximum_smallbin_blk_sz){
                    return this->internal_large_malloc(user_blk_sz);
                }

                size_t pow2_blk_sz          = stdx::ceil2(std::max(user_blk_sz, this->minimum_allocation_blk_sz));
                size_t smallbin_table_idx   = stdx::ulog2(pow2_blk_sz);
                auto& smallbin_vec          = this->smallbin_reuse_table[smallbin_table_idx];
                
                if (smallbin_vec.empty()){
                    return this->internal_bump_allocate(user_blk_sz);
                }

                void * rs                   = smallbin_vec.back().user_ptr;
                smallbin_vec.pop_back();

                return rs;
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

            uint16_t tree_height = stdx::ulog2_aligned(base_sz) + 1u;
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
