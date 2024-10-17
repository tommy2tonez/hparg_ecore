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
#include "network_utility.h"
#include <vector>
#include "network_memult.h"

namespace dg::network_allocation{

    using ptr_type                                          = uint64_t;
    using alignment_type                                    = uint16_t;
    
    static inline constexpr size_t PTROFFS_BSPACE           = sizeof(uint32_t) * CHAR_BIT;
    static inline constexpr size_t PTRSZ_BSPACE             = sizeof(uint16_t) * CHAR_BIT;
    static inline constexpr size_t ALLOCATOR_ID_BSPACE      = sizeof(uint8_t) * CHAR_BIT;
    static inline constexpr size_t ALIGNMENT_BSPACE         = sizeof(uint8_t) * CHAR_BIT;
    static inline constexpr ptr_type NETALLOC_NULLPTR       = ptr_type{0u}; 
    static inline constexpr size_t ALLOCATOR_COUNT          = dg::network_concurrency::THREAD_COUNT;
    static inline constexpr size_t BINARY_HEIGHT            = 20;
    static inline constexpr size_t LEAF_SZ                  = 32;
    static inline constexpr size_t BUFFER_SZ                = (size_t{1} << (BINARY_HEIGHT - 1)) * LEAF_SZ;
    static inline constexpr size_t DEFLT_ALIGNMENT          = alignof(std::max_align_t);

    static_assert(PTROFFS_BSPACE + PTRSZ_BSPACE + ALLOCATOR_ID_BSPACE + ALIGNMENT_BSPACE <= sizeof(ptr_type) * CHAR_BIT);
    static_assert(-1 == ~0);
    static_assert(!NETALLOC_NULLPTR);
    static_assert(std::add_pointer_t<void>(nullptr) == reinterpret_cast<void *>(0u));

    template <class T, size_t SZ, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr auto low(const std::integral_constant<size_t, SZ>) noexcept -> T{
        
        static_assert(SZ <= sizeof(T) * CHAR_BIT);

        if constexpr(SZ == sizeof(T) * CHAR_BIT){
            return ~T{0u};
        } else{
            return (T{1u} << SZ) - 1; 
        }
    }

    class Allocator{

        private:

            std::shared_ptr<char[]> management_buf; 
            std::shared_ptr<char[]> buf;
            std::unique_ptr<dg::heap::core::Allocatable> allocator; //weird bug (need inspection)
            std::unique_ptr<std::atomic_flag> lck;

        public:
            
            Allocator() = default;

            Allocator(std::shared_ptr<char[]> management_buf,
                      std::shared_ptr<char[]> buf,
                      std::unique_ptr<dg::heap::core::Allocatable> allocator,
                      std::unique_ptr<std::atomic_flag> lck) noexcept: management_buf(std::move(management_buf)),
                                                                       buf(std::move(buf)),
                                                                       allocator(std::move(allocator)),
                                                                       lck(std::move(lck)){}
            
            auto malloc(size_t blk_sz) noexcept -> ptr_type{
                
                size_t req_node_sz = blk_sz / LEAF_SZ + size_t{blk_sz % LEAF_SZ != 0}; 
                return this->malloc_node(req_node_sz);
            }

            void free(ptr_type ptr_addr) noexcept{
                
                if (!ptr_addr){
                    return;
                }

                auto [offs, sz] = decode_ptr(ptr_addr);

                {
                    auto lck_grd = dg::network_genult::lock_guard(*this->lck);
                    this->allocator->free({offs, sz});
                }
            }

            auto c_addr(ptr_type ptr_addr) noexcept -> void *{

                if (!ptr_addr){
                    return nullptr;
                }

                auto [offs, _] = decode_ptr(ptr_addr);
                char * rs = this->buf.get();
                std::advance(rs, offs * LEAF_SZ);

                return rs;
            }
        
        private:

            auto malloc_node(const size_t node_sz) noexcept -> ptr_type{

                auto resp = [&]{
                    auto lck_grd = dg::network_genult(*this->lck);
                    return this->allocator->alloc(node_sz);
                }();

                if (!resp){
                    return NETALLOC_NULLPTR;
                }

                auto [resp_offs, resp_sz] = resp.value();
                return encode_ptr(resp_offs, resp_sz);
            } 

            auto encode_ptr(auto hi, auto lo) const noexcept -> ptr_type{
                
                return (static_cast<ptr_type>(hi) << PTRSZ_BSPACE) | static_cast<ptr_type>(lo + 1);
            }

            auto decode_ptr(ptr_type ptr) const noexcept -> std::pair<ptr_type, ptr_type>{

                ptr_type hi = ptr >> PTRSZ_BSPACE;
                ptr_type lo = ptr & low<ptr_type>(std::integral_constant<size_t, PTRSZ_BSPACE>{});

                return {hi, lo - 1};
            }
    };
    
    class MultiThreadAllocator{

        private:

            std::array<Allocator, ALLOCATOR_COUNT> allocator_vec;

        public:

            MultiThreadAllocator() = default;

            MultiThreadAllocator(std::array<Allocator, ALLOCATOR_COUNT>  allocator_vec) noexcept: allocator_vec(std::move(allocator_vec)){}

            auto malloc(size_t blk_sz) noexcept -> ptr_type{

                size_t thr_id   = dg::network_concurrency::this_thread_idx();
                ptr_type rs     = this->allocator_vec[thr_id].malloc(blk_sz);

                if (!rs){
                    return NETALLOC_NULLPTR;
                }

                return encode_ptr(rs, thr_id);
            }

            void free(ptr_type ptr) noexcept{
                
                if (!ptr){
                    return;
                }

                auto [pptr, thr_id] = decode_ptr(ptr);
                this->allocator_vec[thr_id].free(pptr);
            }

            auto c_addr(ptr_type ptr) noexcept -> void *{

                if (!ptr){
                    return nullptr;
                }

                auto [pptr, thr_id] = decode_ptr(ptr);
                return this->allocator_vec[thr_id].c_addr(pptr);
            }
        
        private:

            auto encode_ptr(auto hi, auto lo) const noexcept -> ptr_type{

                return (static_cast<ptr_type>(hi) << ALLOCATOR_ID_BSPACE) | static_cast<ptr_type>(lo);
            }

            auto decode_ptr(ptr_type ptr) const noexcept -> std::pair<ptr_type, ptr_type>{

                ptr_type hi = ptr >> ALLOCATOR_ID_BSPACE;
                ptr_type lo = ptr & low<ptr_type>(std::integral_constant<size_t, ALLOCATOR_ID_BSPACE>{}); 

                return {hi, lo};
            }
    };
    
    static inline MultiThreadAllocator allocator;

    void init(){

        std::array<Allocator, ALLOCATOR_COUNT> allocator_vec{};

        [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
            (
                [&]{
                    (void) IDX;
                    auto management_buf = dg::heap::user_interface::make(BINARY_HEIGHT);
                    auto manager        = dg::heap::user_interface::get_allocator_x(management_buf.get(), std::integral_constant<size_t, IDX>{});
                    auto bool_flag      = std::make_unique<std::atomic_flag>(0);
                    auto buf            = std::unique_ptr<char[], decltype(&std::free)>(static_cast<char *>(std::aligned_alloc(LEAF_SZ, BUFFER_SZ)), &std::free);
                    if (!buf.get()){
                        throw std::bad_alloc();
                    }
                    allocator_vec[IDX]  = Allocator(std::move(management_buf), std::move(buf), std::move(manager), std::move(bool_flag));
                }(), ...
            );
        }(std::make_index_sequence<ALLOCATOR_COUNT>{});

        allocator = MultiThreadAllocator(std::move(id_to_idx_map), std::move(allocator_vec));
    }

    void deinit() noexcept{

        allocator = {};
    }

    auto malloc(size_t blk_sz, size_t alignment) noexcept -> ptr_type{

        assert(dg::memult::is_pow2(alignment));

        size_t fwd_mul_factor   = std::max(static_cast<size_t>(alignment), static_cast<size_t>(LEAF_SZ)) / LEAF_SZ - 1;
        size_t adj_blk_sz       = blk_sz + fwd_mul_factor * LEAF_SZ;
        ptr_type ptr            = allocator.malloc(adj_blk_sz);

        if (!ptr){
            return NETALLOC_NULLPTR;
        }

        ptr <<= ALIGNMENT_BSPACE;
        ptr |= static_cast<ptr_type>(std::countr_zero(static_cast<alignment_type>(alignment)));

        return ptr;
    } 

    auto malloc(size_t blk_sz) noexcept -> ptr_type{

        return malloc(blk_sz, DEFLT_ALIGNMENT); 
    }

    auto c_addr(ptr_type ptr) noexcept -> void *{
        
        if (!ptr){
            return NETALLOC_NULLPTR;
        }

        size_t alignment_log2   = ptr & low<ptr_type>(std::integral_constant<size_t, ALIGNMENT_BSPACE>{}); 
        size_t alignment        = size_t{1} << alignment_log2; 
        ptr_type pptr           = ptr >> ALIGNMENT_BSPACE; 

        return dg::memult::align(allocator.c_addr(pptr), alignment);
    }

    void free(ptr_type ptr) noexcept{

        if (!ptr){
            return;
        }

        allocator.free(ptr >> ALIGNMENT_BSPACE);
    }

    //-- important for stable system - running for years - this is to avoid fragmentation  - yet there are possible improvements that would require breaking design decisions   
    //-- assume all allocations could be categorized as no_free_allocation and short_free_allocation
    
    //assume heap_size == org_heap_size - size(no_free_allocation)
    //short_free_allocation's lifetime has to be less than heap_size / 2 (without loss of generality) in order for the heap to be no-fragmented guaranteed
    //this implementation is a super logic of circular buffer where circular buffer blocks the allocation | overwrite if head_ptr is not freed
    //wheras this implementation does dynamic head seek

    //the balance between cache-efficiency and fragmentation is tough to find - up to the implementation (number of circular heap_blks) and developer to decide
    //allocator might be global - or local
    //don't optimize this yet before actual profiling
    //requires GC support - a daemon to intervally spawn allocator 

    auto cmalloc(size_t blk_sz) noexcept -> void *{

        constexpr size_t HEADER_SZ  = std::max(dg::memult::least_pow2_greater_eq_than(sizeof(ptr_type)), DEFLT_ALIGNMENT); 
        size_t adj_blk_sz           = blk_sz + HEADER_SZ;  
        ptr_type ptr                = malloc(adj_blk_sz);

        if (!ptr){
            return nullptr;
        }

        void * cptr = c_addr(ptr);
        std::memcpy(cptr, &ptr, sizeof(ptr_type));

        return dg::memult::badvance(cptr, HEADER_SZ);
    }

    void cfree(void * cptr) noexcept{
        
        constexpr size_t HEADER_SZ = std::max(dg::memult::least_pow2_greater_eq_than(sizeof(ptr_type)), DEFLT_ALIGNMENT);
        
        if (!cptr){
            return;
        }

        void * org_cptr = dg::memult::badvance(cptr, -static_cast<intmax_t>(HEADER_SZ));
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
            typedef NoExceptAllocator<U> other;
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

            return dg::memult::start_lifetime_as_array<T>(buf, n);
        }

        auto allocate(size_t n) -> pointer{
            
            return allocate(n, std::add_pointer_t<const void>{});
        }
        
        //according to std - deallocate arg is valid ptr - such that allocate -> std::optional<ptr_type>, void deallocate(ptr_type)
        void deallocate(pointer p, size_t n){ //noexcept is guaranteed internally - this is to comply with std

            //whatever std - 

            if (n == 0u){
                return;
            }

            cfree(static_cast<void *>(p)); //fine - a reverse operation of allocate
        }

        consteval auto max_size() const noexcept -> size_type{

            return BUFFER_SZ / sizeof(T);
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