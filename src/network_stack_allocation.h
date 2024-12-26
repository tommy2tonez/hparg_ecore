#ifndef __DG_NETWORK_STACK_ALLOCATION_H__
#define __DG_NETWORK_STACK_ALLOCATION_H__

#include <stdint.h>
#include <stdlib.h> 
#include <memory>
#include <vector>
#include "network_concurrency.h"
#include "network_exception.h"

namespace dg::network_stack_allocation{

    class StackAllocator{

        private:

            std::vector<size_t> stack_cursor_vec;
            std::unique_ptr<char[]> buf; //we aren't using vector - either string or raw allocation to make sure this is pointer arithmetic compatible
            size_t buf_sz;
            size_t cursor;

        public:

            StackAllocator(std::vector<size_t> stack_cursor_vec,
                           std::unique_ptr<char[]> buf,
                           size_t buf_sz,
                           size_t cursor) noexcept: stack_cursor_vec(std::move(stack_cursor_vec)),
                                                    buf(std::move(buf)),
                                                    buf_sz(buf_sz),
                                                    cursor(cursor){}

            inline auto enter_scope() noexcept -> exception_t{

                if (this->stack_cursor_vec.size() == this->stack_cursor_vec.capacity()){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                this->stack_cursor_vec.push_back(this->cursor);
                return dg::network_exception::SUCCESS;
            }

            inline auto allocate(size_t blk_sz) noexcept -> std::expected<void *, exception_t>{

                if (blk_sz == 0u){
                    return static_cast<void *>(nullptr);
                }

                if (this->cursor + blk_sz > this->buf_sz){
                    return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                }

                void * rs       = std::next(this->buf.get(), this->cursor);
                this->cursor   += blk_sz;

                return rs;
            }

            inline void exit_scope() noexcept{

                this->cursor = this->stack_cursor_vec.back();
                this->stack_cursor_vec.pop_back();
            }
    };

    class ConcurrentAllocator{

        private:

            std::vector<std::unique_ptr<StackAllocator>> allocator_vec;

        public:

            ConcurrentAllocator(std::vector<std::unique_ptr<Allocator> allocator_vec) noexcept: allocator_vec(std::move(allocator_vec)){}

            inline auto enter_scope() noexcept -> exception_t{

                size_t thr_idx = dg::network_concurrency::this_thread_idx();
                return this->allocator_vec[thr_idx]->enter_scope();
            } 

            inline auto allocate(size_t buf_sz) noexcept -> std::expected<void *, exception_t>{

                size_t thr_idx = dg::network_concurrency::this_thread_idx();
                return this->allocator_vec[thr_idx]->allocate(buf_sz);
            }

            inline void exit_scope() noexcept{

                size_t thr_idx = dg::network_concurrency::this_thread_idx();
                this->allocator_vec[thr_idx]->exit_scope();
            }
    };

    struct ComponentFactory{

        static inline auto spawn_stack_allocator(size_t scope_size, size_t buf_sz) -> std::unique_ptr<StackAllocator>{

            const size_t MIN_SCOPE_SIZE = 0u;
            const size_t MAX_SCOPE_SIZE = size_t{1} << 20;
            const size_t MIN_BUF_SIZE   = 0u;
            const size_t MAX_BUF_SIZE   = size_t{1} << 40;

            if (std::clamp(scope_size, MIN_SCOPE_SIZE, MAX_SCOPE_SIZE) != scope_size){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            } 

            if (std::clamp(buf_sz, MIN_BUF_SZ, MAX_BUF_SZ) != buf_sz){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
            }

            std::vector<size_t> stack_cursor_vec{};
            stack_cursor_vec.reserve(scope_size);
            std::unique_ptr<char[]> buf = std::make_unique<char[]>(buf_sz);
            size_t cursor = 0u; 

            return std::make_unique<StackAllocator>(std::move(stack_cursor_vec), std::move(buf), buf_sz, cursor);
        }

        static inline auto spawn_concurrent_allocator(size_t scope_size, size_t buf_sz) -> std::unique_ptr<ConcurrentAllocator>{

            std::vector<std::unique_ptr<StackAllocator>> stack_allocator_vec{};

            for (size_t i = 0u; i < dg::network_concurrency::THREAD_COUNT; ++i){
                stack_allocator_vec.push_back(spawn_stack_allocator(scope_size, buf_sz));
            }

            return std::make_unique<ConcurrentAllocator>(std::move(stack_allocator_vec));
        }
    };

    inline std::unique_ptr<ConcurrentAllocator> allocator;

    void init(size_t scope_size, size_t buf_sz){

        stdx::memtransaction_guard tx_grd;
        allocator = ComponentFactory::spawn_concurrent_allocator(scope_size, buf_sz);
    }

    void deinit() noexcept{

        stdx::memtransaction_guard tx_grd;
        allocator = nullptr;
    }

    auto get_allocator() noexcept -> ConcurrentAllocator *{

        std::atomic_signal_fence(std::memory_order_acquire);
        return allocator.get(); //should've been volatile - but volatile is deprecating - so we must issue a memory flushes for concurrent devices - at the very beginning - and memory_order_acquire signal here 
    }

    struct internal_init_tag{};

    //we assume people are rational and only use stack allocations

    template <class T>
    class Allocation{

        private:

            T * data;

            template <class ...Args>
            Allocation(const internal_init_tag, Args&& ...args){

                ConcurrentAllocator * allocator_ins     = get_allocator();
                exception_t err                         = allocator_ins->enter_scope();

                if (dg::network_exception::is_failed(err)){
                    dg::network_exception::throw_exception(err);
                }

                size_t allocation_sz                    = sizeof(T) + alignof(T) - 1u;
                std::expected<void *, exception_t> buf  = allocator_ins->allocate(allocation_sz);

                if (!buf.has_value()){
                    allocator_ins->exit_scope();
                    dg::network_exception::throw_exception(buf.error());
                }

                void * head = dg::memult::align(buf.value(), std::integral_constant<size_t, alignof(T)>{});

                if constexpr(std::is_nothrow_constructible_v<T, Args&&...>){
                    this->data = new (head) T(std::forward<Args>(args)...);
                } else{
                    try{
                        this->data = new (head) T(std::forward<Args>(args)...);
                    } catch (...){
                        allocator_ins->exit_scope();
                        std::rethrow_exception(std::current_exception());
                    }
                }
            }

        public:

            using self = Allocation;

            static_assert(std::is_nothrow_destructible_v<T>);
            static_assert(sizeof(T) != 0u);
            static_assert(alignof(T) != 0u);

            Allocation(): Allocation(internal_init_tag{}){}

            template <class ...Args>
            Allocation(const std::in_place_t, Args&& ...args): Allocation(internal_init_tag{}, std::forward<Args>(args)...){}

            Allocation(const self&) = delete;
            Allocation(self&&) = delete;
            self& operator =(const self&) = delete;
            self& operator =(self&&) = delete;

            ~Allocation() noexcept{

                std::destroy_at(this->data);
                get_allocator()->exit_scope();
            }
    };

    template <class T>
    class Allocation<T[]>{

        private:

            T * arr;
            size_t arr_sz;

        public:

            static_assert(std::is_nothrow_destructible_v<T>);
            static_assert(sizeof(T) != 0u);
            static_assert(alignof(T) != 0u);

            using self = Allocation;

            Allocation(size_t sz){

                if (sz == 0u){
                    this->arr       = nullptr;
                    this->arr_sz    = 0u;
                    return;
                }

                ConcurrentAllocator * allocator_ins     = get_allocator();
                exception_t err                         = allocator_ins->enter_scope();

                if (dg::network_exception::is_failed(err)){
                    dg::network_exception::throw_exception(err);
                }

                size_t allocation_sz                    = sz * sizeof(T) + (alignof(T) - 1u);
                std::expected<void *, exception_t> buf  = allocator_ins->allocate(allocation_sz);

                if (!buf.has_value()){
                    allocator_ins->exit_scope();
                    dg::network_exception::throw_exception(buf.error());
                }

                void * head = dg::memult::align(buf.value(), std::integral_constant<size_t, alignof(T)>{}); 

                if constexpr(std::is_nothrow_default_constructible_v<T>){
                    this->arr       = new (head) T[sz];
                    this->arr_sz    = sz;
                } else{
                    try{
                        this->arr       = new (head) T[sz];
                        this->arr_sz    = sz;
                    } catch (...){
                        allocator_ins->exit_scope();
                        std::rethrow_exception(std::current_exception());
                    }
                }
            }

            Allocation(const self&) = delete;
            Allocation(self&&) = delete;
            self& operator =(const self&) = delete;
            self& operator =(self&&) = delete;

            ~Allocation() noexcept{

                if (this->arr == nullptr){
                    return;
                }

                std::destroy(this->arr, std::next(this->arr, this->arr_sz));
                get_allocator()->exit_scope();
            }

            auto data() const noexcept -> T *{

                return this->arr;
            }

            auto get() const noexcept -> T *{

                return this->arr;
            }
    };

    template <class T>
    class NoExceptAllocation: public Allocation<T>{

        public:

            NoExceptAllocation() noexcept: Allocation(){}
            
            template <class ...Args>
            NoExceptAllocation(const std::in_place_t, Args&& ...args) noexcept: Allocation(std::in_place_t{}, std::forward<Args>(args)...){}
    };

    template <class T>
    class NoExceptAllocation<T[]>: public Allocation<T>{

        public:

            NoExceptAllocation(size_t sz) noexcept: Allocation(sz){}
    };
} 

#endif