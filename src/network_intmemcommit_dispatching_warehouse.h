#ifndef __NETWORK_MEMCOMMIT_DISPATCHING_WAREHOUSE_H__
#define __NETWORK_MEMCOMMIT_DISPATCHING_WAREHOUSE_H__

#include <thread>
#include "network_concurrency.h"
#include "network_memory_utility.h"
#include <atomic>
#include "network_fundamental_vector.h"
#include "network_producer_consumer.h" 

namespace dg::network_intmemcommit_dispatching_warehouse{

    template <class T>
    struct ThreadTableInterface{

        static inline auto map(std::thread::id id) noexcept -> std::thread::id{

            return T::map(id);
        } 
    };

    template <class ID, class EventT, class T, class Capacity>
    class DispatchingWarehouse{};

    template <class ID, class EventT, class T, size_t CAPACITY>
    class DispatchingWarehouse<ID, EventT, ThreadTableInterface<T>, std::integral_constant<size_t, CAPACITY>>{

        public:

            using event_t = EventT; 

        private:

            alignas(dg::memult::HARDWARE_DESTRUCTIVE_INTERFERENCE_SIZE) struct ContainerUnit{
                dg::network_fundamental_vector::fixed_fundamental_vector<event_t, CAPACITY> container;
                std::atomic_flag lck;
            };

            static inline ContainerUnit * container_unit_table{};
            using thread_table = ThreadTableInterface<T>; 

        public:

            static void init(){
                
                auto logger = dg::network_log_scope::critical_terminate();
                container_unit_table = new ContainerUnit[dg::network_concurrency::THREAD_COUNT];
                logger.release();
            }

            static inline auto push_try(event_t * events, size_t sz) noexcept -> bool{

                size_t idx                      = dg::network_concurrency::to_thread_idx(std::this_thread::get_id()); 
                ContainerUnit& container_unit   = container_unit_table[idx];

                if (!container_unit.lck.test_and_set(std::memory_order_acq_rel)){
                    return false;
                }

                if (container_unit.container.size() + sz > container_unit.container.capacity()){
                    return false;
                }

                container_unit.container.push_back(events, sz);
                container_unit.lck.clear(std::memory_order_acq_rel);

                return true;
            }

            static inline void push(event_t * events, size_t sz) noexcept{

                while (!push_try(events, sz)){}
            }

            static inline void get(event_t * dst, size_t& sz, size_t dst_cap) noexcept{

                size_t idx                      = dg::network_concurrency::to_thread_idx(thread_table::map(std::this_thread::get_id()));
                ContainerUnit& container_unit   = container_unit_table[idx];

                while (!container_unit.lck.test_and_set(std::memory_order_acq_rel)){}
                sz              = std::min(dst_cap, container_unit.container.size()); 
                size_t rem_sz   = container_unit.container.size() - sz;
                std::memcpy(dst, container_unit.container.data() + rem_sz, sz * sizeof(event_t));
                container_unit.container.resize(rem_sz);
                container_unit.lck.clear(std::memory_order_acq_rel);
            }

            static consteval auto capacity() noexcept -> size_t{

                return CAPACITY;
            }
    };

} 

#endif