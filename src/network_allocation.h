#ifndef __DG_NETWORK_ALLOCATION_H__
#define __DG_NETWORK_ALLOCATION_H__

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
#include "network_log.h"
#include "network_exception_handler.h"

namespace dg::network_allocation{

    using ptr_type              = uint64_t;
    using alignment_type        = uint16_t;
    using interval_type         = dg::heap::types::interval_type; 

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

    class GCInterface{

        public:

            virtual ~GCInterface() noexcept = default;
            virtual void gc() noexcept = 0;
    };

    class AllocatorInterface{
        
        public:

            virtual ~AllocatorInterface() noexcept = default;
            virtual auto malloc(size_t) noexcept -> ptr_type = 0;
            virtual void free(ptr_type) noexcept = 0;
            virtual auto c_addr(ptr_type) noexcept -> std::add_pointer_t<void> = 0;
    };

    class GCHeapAllocatorInterface: public virtual dg::heap::core::Allocatable,
                                    public virtual GCInterface{};

    class GCAllocatorInterface: public virtual AllocatorInterface,
                                public virtual GCInterface{};

    class GCHeapAllocator: public virtual GCHeapAllocatorInterface{

        private:

            std::unique_ptr<char[], decltype(&std::free)> management_buf;
            std::unique_ptr<dg::heap::core::Allocatable> allocator;
            std::unique_ptr<std::atomic_flag> lck;

        public:

            GCHeapAllocator(std::unique_ptr<char[], decltype(&std::free)> management_buf,
                            std::unique_ptr<dg::heap::core::Allocatable> allocator,
                            std::unique_ptr<std::atomic_flag> lck) noexcept: management_buf(std::move(management_buf)),
                                                                             allocator(std::move(allocator)),
                                                                             lck(std::move(lck)){}

             std::optional<interval_type> alloc(store_type arg) noexcept{

                auto lck_grd = stdx::lock_guard(*this->lck);
                return this->allocator->alloc(arg);
             }

             void free(const interval_type& arg) noexcept{

                auto lck_grd = stdx::lock_guard(*this->lck);
                this->allocator->free(arg);
             }

             void gc() noexcept{

                auto lck_grd = stdx::lock_guard(*this->lck);

                try{
                    this->allocator = dg::heap::user_interface::get_allocator_x(this->management_buf.get());
                } catch (...){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::wrap_std_exception(std::current_exception())));
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
            
            auto malloc(size_t blk_sz) noexcept -> ptr_type{

                size_t req_node_sz = blk_sz / LEAF_SZ + size_t{blk_sz % LEAF_SZ != 0u};
                std::optional<interval_type> resp = this->base_allocator->alloc(req_node_sz);

                if (!resp.has_value()){
                    return NETALLOC_NULLPTR;
                }

                auto [resp_off, resp_sz] = resp.value();
                
                static_assert(std::is_unsigned_v<decltype(resp_off)>);
                static_assert(std::is_unsigned_v<decltype(resp_sz)>);

                return this->encode_ptr(resp_off, resp_sz); //use 16 bits encoding - 1 bit to denotes log2 roundup and 15 bits to denote the length
            }

            void free(ptr_type ptr) noexcept{
                
                if (!ptr){
                    return;
                }

                auto [off, sz] = this->decode_ptr(ptr);
                this->base_allocator->free({off, sz});
            }

            auto c_addr(ptr_type ptr) noexcept -> void *{

                if (!ptr){
                    return nullptr;
                }

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

    class MultiThreadAllocator: public virtual GCAllocatorInterface{

        private:

            stdx::vector<std::unique_ptr<Allocator>> allocator_vec;

        public:

            MultiThreadAllocator(stdx::vector<std::unique_ptr<Allocator>> allocator_vec) noexcept: allocator_vec(std::move(allocator_vec)){}

            auto malloc(size_t blk_sz) noexcept -> ptr_type{

                size_t allocator_idx    = stdx::pow2mod_unsigned(dg::network_concurrency::this_thread_idx(), this->allocator_vec.size());
                ptr_type rs             = this->allocator_vec[allocator_idx]->malloc(blk_sz);

                if (!rs){
                    return NETALLOC_NULLPTR;
                }

                return this->encode_ptr(rs, allocator_idx);
            }

            void free(ptr_type ptr) noexcept{
                
                if (!ptr){
                    return;
                }

                auto [pptr, allocator_idx] = this->decode_ptr(ptr);
                this->allocator_vec[allocator_idx]->free(pptr);
            }

            auto c_addr(ptr_type ptr) noexcept -> void *{

                if (!ptr){
                    return nullptr;
                }

                auto [pptr, allocator_idx] = this->decode_ptr(ptr);
                return this->allocator_vec[allocator_idx]->c_addr(pptr);
            }

            void gc() noexcept{ // this might be a bottleneck if more than 1024 concurrent allocators are in use - this is not likely going to be the case - if a computer has more than 1024 cores - it's something wrong with the computer

                for (size_t i = 0u; i < this->allocator_vec.size(); ++i){
                    this->allocator_vec[i]->gc();
                }
            }
        
        private:

            auto encode_ptr(ptr_type hi, uint64_t lo) const noexcept -> ptr_type{

                return (hi << ALLOCATOR_ID_BSPACE) | static_cast<ptr_type>(lo);
            }

            auto decode_ptr(ptr_type ptr) const noexcept -> std::pair<ptr_type, uint64_t>{

                ptr_type hi = ptr >> ALLOCATOR_ID_BSPACE;
                ptr_type lo = stdx::low_bit<ALLOCATOR_ID_BSPACE>(ptr);

                return {hi, static_cast<uint64_t>(lo)};
            }
    };

    class GCWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:
  
            std::shared_ptr<GCInterface> gc_able;
        
        public:

            GCWorker(std::shared_ptr<GCInterface> gc_able): gc_able(std::move(gc_able)){}
            
            auto run_one_epoch() noexcept -> bool{

                this->gc_able->gc();
                return true;
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

        static auto spawn_concurrent_allocator(stdx::vector<std::unique_ptr<Allocator>> allocator) -> std::unique_ptr<MultiThreadAllocator>{ //devirt here is important - 

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

        static auto spawn_gc_worker(std::chrono::nanoseconds gc_interval, std::shared_ptr<GCInterface> gc_able) -> dg::network_concurrency::daemon_raii_handle_t{ //this is strange - this overstep into the responsibility - decouple the component

            std::chrono::nanoseconds MIN_GC_INTERVAL    = std::chrono::milliseconds(1);
            std::chrono::nanoseconds MAX_GC_INTERVAL    = std::chrono::seconds(1);

            if (std::clamp(gc_interval.count(), MIN_GC_INTERVAL.count(), MAX_GC_INTERVAL.count()) != gc_interval.count()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            if (gc_able == nullptr){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            return dg::network_exception_handler::throw_nolog(dg::network_concurrency::daemon_saferegister_with_waittime(dg::network_concurrency::COMPUTING_DAEMON, std::make_unique<GCWorker>(std::move(gc_able)), gc_interval)); 
        }
    };

    struct AllocationResource{
        std::shared_ptr<MultiThreadAllocator> allocator; //devirt here is important - 
        dg::network_concurrency::daemon_raii_handle_t gc_worker;
    };

    inline AllocationResource allocation_resource;

    void init(size_t least_buf_sz, size_t num_allocator, std::chrono::nanoseconds gc_interval){

        stdx::vector<std::unique_ptr<Allocator>> allocator_vec{};

        for (size_t i = 0u; i < num_allocator; ++i){
            allocator_vec.push_back(Factory::spawn_allocator(least_buf_sz));
        }

        std::shared_ptr<MultiThreadAllocator> allocator = Factory::spawn_concurrent_allocator(std::move(allocator_vec));
        dg::network_concurrency::daemon_raii_handle_t daemon_handle = Factory::spawn_gc_worker(gc_interval, allocator);
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
    struct NoExceptAllocator{
 
        using value_type                                = T;
        using pointer                                   = T *;
        using const_pointer                             = const T *;
        using reference                                 = T&;
        using const_reference                           = const T&;
        using size_type                                 = size_t;
        using difference_type                           = intmax_t;
        using is_always_equal                           = std::true_type;
        using propagate_on_container_move_assignment    = std::true_type;
        
        template <class U>
        struct rebind{
            using other = NoExceptAllocator<U>;
        };

        auto address(reference x) const noexcept -> pointer{

            return std::addressof(x);
        }

        auto address(const_reference x) const noexcept -> const_pointer{

            return std::addressof(x);
        }
        
        auto allocate(size_t n, const void * hint) -> pointer{ //noexcept is guaranteed internally - this is to comply with std

            if (n == 0u){
                return nullptr;
            }

            void * buf = cmalloc(n * sizeof(T));

            if (!buf){
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::OUT_OF_MEMORY));
                std::abort();
            }

            return dg::memult::start_lifetime_as_array<T>(buf, n); //this needs compiler magic to avoid undefined behaviors
        }

        auto allocate(size_t n) -> pointer{
            
            return allocate(n, std::add_pointer_t<const void>{});
        }
        
        //according to std - deallocate arg is valid ptr - such that allocate -> std::optional<ptr_type>, void deallocate(ptr_type)
        void deallocate(pointer p, size_t n){ //noexcept is guaranteed internally - this is to comply with std

            if (n == 0u){
                return;
            }

            cfree(static_cast<void *>(p)); //fine - a reverse operation of allocate
        }

        consteval auto max_size() const noexcept -> size_type{

            return std::numeric_limits<size_type>::max();
        }
        
        template <class U, class... Args>
        void construct(U * p, Args&&... args) noexcept(std::is_nothrow_constructible_v<U, Args...>){

            new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
        }

        template <class U>
        void destroy(U * p) noexcept(std::is_nothrow_destructible_v<U>){

            std::destroy_at(p);
        }
    };

    template <class T, class T1>
    constexpr auto operator==(const NoExceptAllocator<T>&, const NoExceptAllocator<T1>&) noexcept -> bool{

        return true;
    }

    template<class T, class T1>
    constexpr auto operator!=(const NoExceptAllocator<T>&, const NoExceptAllocator<T1>&) noexcept -> bool{

        return false;
    }
}

#endif