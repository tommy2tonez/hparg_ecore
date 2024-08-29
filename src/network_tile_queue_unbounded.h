#ifndef __NETWORK_TILE_UNBOUNDED_QUEUE_H__
#define __NETWORK_TILE_UNBOUNDED_QUEUE_H__

#include <stdint.h>
#include <stdlib.h> 
#include "network_concurrency.h"

namespace dg::network_tile_queue_unbounded{

    //consider moving -> unique_ptr<>
    //move from stricter -> less stricter post profiling
    //1hr
    
    template <class T>
    struct QueueInterface{

        using vma_ptr_t         = typename T::vma_ptr_t;
        using device_id_t       = typename T::device_id_t;
        using tile_taxonomy_t   = typename T::tile_taxonomy_t;

        static inline auto next(device_id_t device_id, tile_taxonomy_t tile_taxonomy) noexcept -> vma_ptr_t{

            return T::next(device_id, tile_taxonomy);
        }
    };

    template <class ID, class vma_ptr_t, class device_id_t, class tile_taxonomy_t>
    struct AtomicQueue: QueueInterface<AtomicQueue<ID, vma_ptr_t, device_id_t, tile_taxonomy_t>>{

        static void init(vma_ptr_t * tile_ptr, device_id_t * device_id, tile_taxonomy_t * tile_taxonomy, size_t n) noexcept{

        }

        static inline auto next(device_id_t device_id, tile_taxonomy_t tile_taxonomy) noexcept -> vma_ptr_t{
            
        }
    };

    template <class ID, class vma_ptr_t, class device_id_t, class tile_taxonomy_t, size_t DEVICE_COUNT, size_t TILE_TAXONOMY_COUNT>
    struct ConcurrentQueue: QueueInterface<ConcurrentQueue<ID, vma_ptr_t, device_id_t, tile_taxonomy_t, DEVICE_COUNT, TILE_TAXONOMY_COUNT>>{
        
        private:

            static inline vma_ptr_t * table{};
            static inline size_t * counter{}; 

        public:

            static void init(vma_ptr_t * tile_ptr, device_id_t * device_id, tile_taxonomy_t * tile_taxonomy, size_t n) noexcept{

            }

            static inline auto next(device_id_t device_id, tile_taxonomy_t tile_taxonomy) noexcept -> vma_ptr_t{
                
                constexpr size_t TABLE_SZ   = DEVICE_COUNT * TILE_TAXONOMY_COUNT * dg::network_concurrency::THREAD_COUNT; //pow2 important
                size_t table_idx            = device_id * (TILE_TAXONOMY_COUNT * dg::network_concurrency::THREAD_COUNT) + (tile_taxonomy * dg::network_concurrency::THREAD_COUNT + dg::network_concurrency::this_thread_idx());
                vma_ptr_t rs                = table[counter[table_idx]];
                counter[table_idx]          += 1;
                counter[table_idx]          %= TABLE_SZ; 

                return rs; 
            }
    };
} 

#endif