#ifndef __DG_NETWORK_KERNEL_MAILBOX_IMPL1_ALLOCATION_H__
#define __DG_NETWORK_KERNEL_MAILBOX_IMPL1_ALLOCATION_H__

#include "network_kernel_allocator.h"

namespace dg::network_kernel_mailbox_impl1_allocation::allocator_instance
{
    struct Config
    {
        size_t total_mempiece_count;
        size_t mempiece_sz;
        size_t affined_refill_sz;
        size_t affined_mem_vec_capacity;
        size_t affined_free_vec_capacity;
    };
    
    struct namespace_signature{};

    using singleton_object = stdx::singleton<namespace_signature, std::unique_ptr<dg::network_kernel_allocator::AllocatorInterface<dg::network_kernel_allocator::AffinedMapAllocator>>>; 

    void init(Config config)
    {
        using namespace dg::network_kernel_allocator;

        std::unique_ptr<BatchAllocatorInterface> base_allocator = network_kernel_allocator::ComponentFactory::make_batch_allocator(config.total_mempiece_count,
                                                                                                                                   config.mempiece_sz);

        singleton_object::get() = network_kernel_allocator::ComponentFactory::make_affined_map_allocator(std::move(base_allocator),
                                                                                                         config.affined_refill_sz,
                                                                                                         config.affined_mem_vec_capacity,
                                                                                                         config.affined_free_vec_capacity);
    }

    void deinit() noexcept
    {
        singleton_object::get() = nullptr;
    }

    auto dg_malloc(size_t byte_sz) noexcept -> std::expected<void *, exception_t>
    {
        if constexpr(DEBUG_MODE_FLAG)
        {
            if (singleton_object::get() == nullptr)
            {
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                std::abort();
            }            
        }

        if (byte_sz > singleton_object::get()->malloc_size())
        {
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return singleton_object::get()->malloc();
    }

    auto dg_realloc(void * old_ptr, size_t new_sz) noexcept -> std::expected<void *, exception_t>
    {
        if constexpr(DEBUG_MODE_FLAG)
        {
            if (singleton_object::get() == nullptr)
            {
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                std::abort();
            }
        }

        if (new_sz > singleton_object::get()->malloc_size())
        {
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return old_ptr;
    }

    void dg_free(void * mem_ptr) noexcept
    {
        if constexpr(DEBUG_MODE_FLAG)
        {
            if (singleton_object::get() == nullptr)
            {
                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INVALID_ARGUMENT));
                std::abort();
            }
        }

        singleton_object::get()->free(mem_ptr);
    }

    class StaticWrappedAllocator
    {
        public:

            static auto malloc(size_t byte_sz) noexcept -> std::expected<void *, exception_t>
            {
                return allocator_instance::dg_malloc(byte_sz);
            }

            static auto realloc(void * old_ptr, size_t new_sz) noexcept -> std::expected<void *, exception_t>
            {
                return allocator_instance::dg_realloc(old_ptr, new_sz);
            }

            static auto free(void * old_ptr) noexcept
            {
                allocator_instance::dg_free(old_ptr);
            }
    };

    template <typename T>
    struct StdWrappedAllocator
    {
        using value_type = T;

        constexpr StdWrappedAllocator() = default;

        template <typename U>
        constexpr StdWrappedAllocator(const StdWrappedAllocator<U>&) {}

        auto allocate(std::size_t n) -> T *
        {
            std::expected<void *, exception_t> raw_mem = allocator_instance::dg_malloc(n * sizeof(T)); 

            if (!raw_mem.has_value())
            {
                dg::network_exception::throw_exception(raw_mem.error());
            }

            if constexpr(DEBUG_MODE_FLAG)
            {
                if (reinterpret_cast<uintptr_t>(raw_mem.value()) % alignof(std::max_align_t) != 0u)
                {
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                    std::abort();
                }
            }

            return static_cast<T *>(raw_mem.value());
        }

        void deallocate(T* p, std::size_t n) 
        {
            allocator_instance::dg_free(p);
        }

        template <typename U, typename... Args>
        void construct(U* p, Args&&... args)
        {
            new (p) U(std::forward<Args>(args)...);
        }

        template <typename U>
        void destroy(U* p) 
        {
            std::destroy_at(p);
        }
    };

    template <typename T, typename U>
    bool operator==(const StdWrappedAllocator<T>&, const StdWrappedAllocator<U>&)
    {
        return true;
    }

    template <typename T, typename U>
    bool operator!=(const StdWrappedAllocator<T>&, const StdWrappedAllocator<U>&)
    {
        return false;
    }
}

#endif