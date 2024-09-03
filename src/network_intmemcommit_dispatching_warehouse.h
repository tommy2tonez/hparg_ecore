#ifndef __NETWORK_MEMCOMMIT_DISPATCHING_WAREHOUSE_H__
#define __NETWORK_MEMCOMMIT_DISPATCHING_WAREHOUSE_H__

#include <thread>
#include "network_concurrency.h"
#include "network_memory_utility.h"
#include <atomic>
#include "network_fundamental_vector.h"
#include "network_utility.h"
#include "network_log.h"
#include "network_exception.h"

namespace dg::network_intmemcommit_dispatching_warehouse{

    //let's make a bet
    //I will send the challenge to all the compiler out there 
    //I will write optimization O4 in 2 months

    template <class T>
    struct ThreadTableInterface{

        static inline auto map(std::thread::id id) noexcept -> std::thread::id{

            return T::map(id);
        } 
    };

    static inline constexpr bool IS_ATOMIC_OPERATION_PREFERRED = true;
    using Lock = std::conditional_t<IS_ATOMIC_OPERATION_PREFERRED, 
                                    std::atomic_flag,
                                    std::mutex>; 

    template <class ID, class EventT, class T, class Capacity>
    class DispatchingWarehouse{};

    template <class ID, class EventT, class T, size_t CAPACITY>
    class DispatchingWarehouse<ID, EventT, ThreadTableInterface<T>, std::integral_constant<size_t, CAPACITY>>{

        public:

            using event_t = EventT; 

        private:

            alignas(dg::memult::HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE) struct ContainerUnit{
                dg::network_fundamental_vector::fixed_fundamental_vector<event_t, CAPACITY> container;
                Lock lck;
            };

            static inline ContainerUnit * container_unit_table{};
            using thread_table = ThreadTableInterface<T>; 

        public:

            static void init(){
                
                auto logger = dg::network_log_scope::critical_terminate();
                container_unit_table = new ContainerUnit[dg::network_concurrency::THREAD_COUNT];
                logger.release();
            }

            static auto push_try(event_t * events, size_t sz) noexcept -> bool{

                size_t idx              = dg::network_concurrency::this_thread_idx(); 
                auto& container_unit    = container_unit_table[idx];
                auto lck_grd            = dg::network_genult::lock_guard(container_unit.lck);

                if (sz > container_unit.container.capacity()){
                    dg::network_log_stackdump::critical(dg::network_exception::INTERNAL_CORRUPTION);
                    std::abort();
                }

                if (container_unit.container.size() + sz > container_unit.container.capacity()){
                    return false;
                }

                container_unit.container.push_back(events, sz);
                return true;
            }

            static void push(event_t * events, size_t sz) noexcept{

                while (!push_try(events, sz)){}
            }

            static void get(event_t * dst, size_t& sz, size_t dst_cap) noexcept{

                size_t idx              = dg::network_concurrency::to_thread_idx(thread_table::map(std::this_thread::get_id()));
                auto& container_unit    = container_unit_table[idx];
                auto lck_grd            = dg::network_genult::lock_guard(container_unit.lck);
                sz                      = std::min(dst_cap, container_unit.container.size());
                size_t rem_sz           = container_unit.container.size() - sz;
                const event_t * src     = dg::memult::advance(container_unit.container.data(), rem_sz);

                std::memcpy(dst, src, sz * sizeof(event_t));
                container_unit.container.resize(rem_sz);
            }

            static consteval auto capacity() noexcept -> size_t{

                return CAPACITY;
            }
    };

} 

#endif