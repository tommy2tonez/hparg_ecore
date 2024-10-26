#ifndef __NETWORK_MEMPRESS_COLLECTOR_H__
#define __NETWORK_MEMPRESS_COLLECTOR_H__

#include <stddef.h>
#include <stdint.h>
#include <chrono>
#include <ratio>
#include "network_producer_consumer.h"
#include "network_mempress.h"
#include "network_concurrency.h" 
#include "network_randomizer.h"
#include "network_std_container.h"

namespace dg::network_mempress_collector{

    using event_t   = dg::network_memcommit::virtual_mmeory_event_t;

    struct MempressRetranslatorInterface{
        virtual ~MempressRetranslatorInterface() noexcept = default;
        virtual auto size() noexcept -> size_t = 0;
        virtual auto get(size_t, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept = 0;
    };

    class MempressRetranslator: public virtual MempressRetranslatorInterface{

        private:

            std::vector<uma_ptr_t> idx_map;
            std::shared_ptr<dg::network_mempress::MemoryPressInterface> mempress;
        
        public:

            MempressRetranslator(std::vector<uma_ptr_t> idx_map,
                                 std::shared_ptr<dg::network_mempress::MemoryPressInterface> mempress) noexcept: idx_map(std::move(idx_map)),
                                                                                                                 mempress(std::move(mempress)){}

            auto size() noexcept -> size_t{

                return this->idx_map.size()
            }

            auto get(size_t idx, event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                uma_ptr_t region = this->idx_map[idx];
                this->mempress->collect(region, dst, dst_sz, dst_cap);
            }
    };

    class TemporalCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::unique_ptr<MempressRetranslatorInterface> mempress;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::unique_ptr<event_t[]> event_buf; //fine - refactorables
            size_t event_buf_cap; //fine - refactorables
            std::chrono::nanoseconds last;
            std::chrono::nanoseconds diff;

        public:

            TemporalCollector(std::unique_ptr<MempressRetranslatorInterface>  mempress,
                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                              std::unique_ptr<event_t[]> event_buf,
                              size_t event_buf_cap,
                              std::chrono::nanoseconds last,
                              std::chrono::nanoseconds diff) noexcept: mempress(std::move(mempress)),
                                                                       consumer(std::move(consumer)),
                                                                       event_buf(std::move(event_buf)),
                                                                       event_buf_cap(event_buf_cap),
                                                                       last(last),
                                                                       diff(diff){}
            
            auto run_one_epoch() noexcept -> bool{

                std::chrono::nanoseconds now        = dg::network_genult::unix_timestamp();
                std::chrono::nanoseconds last_diff  = dg::network_genult::timelapsed(this->last, now); 

                if (last_diff < this->diff){
                    return false;
                }

                for (size_t i = 0u; i < this->mempress->size(); ++i){
                    size_t event_buf_sz{};
                    this->mempress->get(i, this->event_buf.get(), event_buf_sz, this->event_buf_cap);
                    this->consumer->push(this->event_buf.get(), event_buf_sz);
                }

                this->last = dg::network_genult::unix_timestamp();
                return true;
            }
    };

    class HalfLifeCollector: public virtual dg::network_concurrency::WorkerInterface{

        private:
            
            std::unique_ptr<MempressRetranslatorInterface> mempress;
            std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer;
            std::unique_ptr<event_t[]> event_buf;
            size_t event_buf_cap;
            std::chrono::nanoseconds last;
            std::chrono::nanoseconds diff;
            double halflife;
        
        public:

            HalfLifeCollector(std::unique_ptr<MempressRetranslatorInterface> mempress,
                              std::shared_ptr<dg::network_producer_consumer::ConsumerInterface<event_t>> consumer,
                              std::unique_ptr<event_t[]> event_buf,
                              size_t event_buf_cap,
                              std::chrono::nanoseconds last,
                              std::chrono::nanoseconds diff,
                              double halflife) noexcept: mempress(std::move(mempress)),
                                                         consumer(std::move(consumer)),
                                                         event_buf(std::move(event_buf)),
                                                         event_buf_cap(event_buf_cap),
                                                         last(last),
                                                         diff(diff),
                                                         halflife(halflife){}

            auto run_one_epoch() noexcept -> bool{

                std::chrono::nanoseconds now        = dg::network_genult::unix_timestamp();
                std::chrono::nanoseconds last_diff  = dg::network_genult::timelapsed(this->last, now);

                if (last_diff < diff){
                    return false;
                }

                size_t nxt_chk_pt = this->mempress->size() * this->halflife;

                for (size_t i = 0u; i < this->mempress->size(); ++i){
                    if (i == nxt_chk_pt){
                        bool coin_flip = dg::network_randomizer::randomize_bool();

                        if (coin_flip){
                            break;
                        }

                        nxt_chk_pt += (this->mempress->size() - i) * this->halflife;
                    }

                    size_t event_buf_sz{};
                    this->mempress->get(i, this->event_buf.get(), event_buf_sz, this->event_buf_cap);
                    this->consumer->push(this->event_buf.get(), event_buf_sz);
                }

                this->last = dg::network_genult::unix_timestamp();
                return true;
            }
    };
}

#endif 