#ifndef __DG_NETWORK_UTILITY_H__
#define __DG_NETWORK_UTILITY_H__

#include <atomic>
#include <memory>
#include <mutex>
#include "network_log.h"
#include "network_exception.h" 

namespace dg::network_genult{

    template <class T>
    auto safe_ptr_access(T * ptr) noexcept{

        if (ptr == nullptr){
            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::SEGFAULT));
            std::abort();
        }

        return ptr;
    }

    auto lock_guard(std::atomic_flag& lck) noexcept{

        static int i    = 0;
        auto destructor = [&](int *) noexcept{
            lck.clear(std::memory_order_acq_rel);
        };

        while (!lck.test_and_set(std::memory_order_acq_rel)){}
        return std::unique_ptr<int, decltype(destructor)>(&i, std::move(destructor));
    }

    auto lock_guard(std::mutex& lck) noexcept{

        static int i    = 0;
        auto destructor = [&](int *) noexcept{
            lck.unlock();
        };

        lck.lock();
        return std::unique_ptr<int, decltype(destructor)>(&i, std::move(destructor));
    }

    template <class ID, class T>
    class singleton{

        private:

            static inline T obj{};
        
        public:

            static inline auto get() noexcept -> T&{

                return obj;
            }
    }
}

#endif