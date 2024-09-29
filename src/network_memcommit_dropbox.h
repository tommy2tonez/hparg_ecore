#ifndef __NETWORK_MEMCOMMIT_DROPBOX_H__
#define __NETWORK_MEMCOMMIT_DROPBOX_H__

#include <stdint.h>
#include <stdlib.h>
#include "network_producer_consumer.h"
#include "network_memcommit_model.h"

namespace dg::network_memcommit_dropbox{

    using virtual_memory_event_t    = network_memcommit_factory::virtual_memory_event_t;  

    struct DropBoxInterface: public virtual dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>,
                             public virtual dg::network_producer_consumer::LimitConsumerInterface<virtual_memory_event_t>{};

    class ExhaustionControlledDropBox: public virtual DropBoxInterface{
        
        private:

            dg::network_std_container::vector<virtual_memory_event_t> event_vec;
            const size_t max_digest_sz; 
            std::unique_ptr<std::mutex> mtx;

        public:
            
            ExhaustionControlledDropBox(dg::network_std_container::vector<virtual_memory_event_t> event_vec,
                                        size_t max_digest_sz,
                                        std::unique_ptr<std::mutex> mtx) noexcept: event_vec(std::move(event_vec)),
                                                                                   max_digest_sz(max_digest_sz),
                                                                                   mtx(std::move(mtx)){}

            void push(virtual_memory_event_t * event, size_t sz) noexcept{

                while (!this->internal_push(event, sz)){}
            }

            auto capacity() const noexcept -> size_t{

                return this->max_digest_sz;
            } 

            void get(virtual_memory_event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                this->internal_get(dst, dst_sz, dst_cap);
            }
        
        private:

            auto internal_push(virtual_memory_event_t * event, size_t sz) noexcept -> bool{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->capacity()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (sz + this->event_vec.size() > this->event_vec.capacity()){
                    return false;
                }

                this->event_vec.insert(this->event_vec.end(), event, event + sz);
                return true;
            }

            void internal_get(virtual_memory_event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                dst_sz          = std::min(this->event_vec.size(), dst_cap);
                size_t new_sz   = this->event_vec.size() - dst_sz;
                auto first      = this->event_vec.begin() + new_sz;
                auto last       = this->event_vec.end();
                std::copy(first, last, dst);
                this->event_vec.resize(new_sz);
            }
    };

    template <size_t CONCURRENCY_SZ> //deprecate next iteration
    class ConcurrentDropBox: public virtual DropBoxInterface{

        private:

            dg::network_std_container::vector<std::unique_ptr<DropBoxInterface>> dropbox_vec;
            const size_t cap; 

        public:

            ConcurrentDropBox(dg::network_std_container::vector<std::unique_ptr<DropBoxInterface>> dropbox_vec,
                              size_t cap, 
                              std::integral_constant<size_t, CONCURRENCY_SZ>) noexcept: dropbox_vec(std::move(dropbox_vec)),
                                                                                        cap(cap){}

            void push(virtual_memory_event_t * src, size_t src_sz) noexcept{

                size_t thr_idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{});
                this->dropbox_vec[thr_id]->push(src, src_sz);
            }

            auto capacity() const noexcept -> size_t{

                return this->cap;
            }

            void get(virtual_memory_event_t * dst, size_t& dst_sz, size_t dst_cap) noexcept{

                auto thr_idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, CONCURRENCY_SZ>{});
                this->dropbox_vec[thr_idx]->get(dst, dst_sz, dst_cap);
            }
    };

    class MemCommitResolutor: public virtual dg::network_concurrency::WorkerInterface{

        private:    

            std::shared_ptr<dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>> producer;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> fwd_ping_signal_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pong_request_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pingpong_request_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pong_signal_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> fwd_init_signal_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_load_request_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_load_response_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_load_redirect_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> bwd_ready_signal_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> bwd_do_signal_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> bwd_load_request_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> bwd_load_response_resolutor;
            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> bwd_load_redirect_resolutor;
            size_t delivery_thrhold; 

        public:


            MemCommitResolutor(std::shared_ptr<dg::network_producer_consumer::ProducerInterface<virtual_memory_event_t>> producer,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> fwd_ping_signal_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pong_request_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pingpong_request_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_pong_signal_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t>>> fwd_init_signal_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_load_request_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_load_response_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> fwd_load_redirect_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> bwd_ready_signal_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> bwd_do_signal_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> bwd_load_request_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> bwd_load_response_resolutor,
                               std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<std::tuple<uma_ptr_t, uma_ptr_t>>> bwd_load_redirect_resolutor) noexcept: producer(std::move(producer)),
                                                                                                                                                                          fwd_ping_signal_resolutor(std::move(fwd_ping_signal_resolutor)),
                                                                                                                                                                          fwd_pong_request_resolutor(std::move(fwd_pong_request_resolutor)),
                                                                                                                                                                          fwd_pingpong_request_resolutor(std::move(fwd_pingpong_request_resolutor)),
                                                                                                                                                                          fwd_pong_signal_resolutor(std::move(fwd_pong_signal_resolutor)),
                                                                                                                                                                          fwd_init_signal_resolutor(std::move(fwd_init_signal_resolutor)),
                                                                                                                                                                          fwd_load_request_resolutor(std::move(fwd_load_request_resolutor)),
                                                                                                                                                                          fwd_load_response_resolutor(std::move(fwd_load_response_resolutor)),
                                                                                                                                                                          fwd_load_redirect_resolutor(std::move(fwd_load_redirect_resolutor)),
                                                                                                                                                                          bwd_ready_signal_resolutor(std::move(bwd_ready_signal_resolutor)),
                                                                                                                                                                          bwd_do_signal_resolutor(std::move(bwd_do_signal_resolutor)),
                                                                                                                                                                          bwd_load_request_resolutor(std::move(bwd_load_request_resolutor)),
                                                                                                                                                                          bwd_load_response_resolutor(std::move(bwd_load_response_resolutor)),
                                                                                                                                                                          bwd_load_redirect_resolutor(std::move(bwd_load_redirect_resolutor)){}
            bool run_one_epoch() noexcept{
                
                using namespace dg::network_memcommit_factory;

                //don't know if an attempt to virtualize here is worth it - 
                //I value the accuracy of the program than exception propagation of the program - because I think throwing every exception without noexcepting it at some root is just pure garbage - wrong - unless you are kernel (again) - yet kernel is just an executable by another kernel - so...
                //the program rather dies but produces the right result than lives and produces the wrong result 
                //this is call program compromision - the user of the program have to expect that the program could crash at any point without notice - yet should not expect that the program produces any wrong result without notice (rule number 1 in programming) 
                //- best yet - the program that allows the user to verify output accuracy outside the program's scope -

                auto fwd_ping_signal_delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_ping_signal_resolutor.get(), this->fwd_ping_signal_resolution_sz));
                auto fwd_pong_request_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_pong_request_resolutor.get(), this->fwd_pong_request_resolution_sz));
                auto fwd_pingpong_request_delivery_handle   = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_pingpong_request_resolutor.get(), this->fwd_pingpong_request_resolution_sz));
                auto fwd_pong_signal_delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_pong_signal_resolutor.get(), this->fwd_pong_signal_resolution_sz));
                auto fwd_init_signal_delivery_handle        = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_init_signal_resolutor.get(), this->fwd_init_signal_resolution_sz));
                auto fwd_load_request_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_load_request_resolutor.get(), this->fwd_load_request_resolution_sz));
                auto fwd_load_response_delivery_handle      = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_load_response_resolutor.get(), this->fwd_load_response_resolution_sz));
                auto fwd_load_redirect_delivery_handle      = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->fwd_load_redirect_resolutor.get(), this->fwd_load_redirect_resolution_sz));
                auto bwd_ready_signal_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->bwd_ready_signal_resolutor.get(), this->bwd_ready_signal_resolution_sz));
                auto bwd_do_signal_delivery_handle          = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->bwd_do_signal_resolutor.get(), this->bwd_do_signal_resolution_sz));
                auto bwd_load_request_delivery_handle       = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->bwd_load_request_resolutor.get(), this->bwd_load_request_resolution_sz));
                auto bwd_load_response_delivery_handle      = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->bwd_load_response_resolutor.get(), this->bwd_load_response_resolution_sz));
                auto bwd_load_redirect_delivery_handle      = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_raiihandle(this->bwd_load_redirect_resolutor.get(), this->bwd_load_redirect_resolution_sz));
                auto retrieving_data                        = dg::network_std_container::vector<virtual_memory_event_t>(this->consuming_sz);
                size_t retrieving_sz                        = {}; 
                this->producer->get(retrieving_data.data(), retrieving_sz, this->consuming_sz);

                if (retrieving_sz == 0u){
                    return false;
                }

                retrieving_data.resize(retrieving_sz);

                for (const virtual_memory_event_t& event: retrieving_data){
                    switch (read_event_taxonomy(event)){
                        case forward_ping_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_ping_signal_delivery_handle.get(), read_event_forward_ping_signal(event));
                            break;
                        case forward_pong_request:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_pong_request_delivery_handle.get(), read_event_forward_pong_request(event));
                            break;
                        case forward_pingpong_request:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_pingpong_request_delivery_handle.get(), read_event_forward_pingpong_request(event));
                            break;
                        case forward_pong_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_pong_signal_delivery_handle.get(), read_event_forward_pong_signal(event));
                            break;
                        case forward_ready_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_ready_signal_delivery_handle.get(), read_event_forward_ready_signal(event))
                            break;
                        case forward_init_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_init_signal_delivery_handle.get(), read_event_forward_init_signal(event));
                            break;
                        case forward_load_request:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_load_request_delivery_handle.get(), read_event_forward_load_request(event));
                            break;
                        case forward_load_response:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_load_response_delivery_handle.get(), read_event_forward_load_response(event));
                            break;
                        case forward_load_redirect_request:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_load_redirect_delivery_handle.get(), read_event_forward_load_redirect_request(event));
                            break;
                        case backward_ready_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(bwd_ready_signal_delivery_handle.get(), read_event_backward_ready_signal(event));
                            break;
                        case backward_do_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(bwd_do_signal_delivery_handle.get(), read_event_backward_do_signal(event));
                            break;
                        case backward_load_request:
                            dg::network_producer_consumer::delvrsrv_deliver(bwd_load_request_delivery_handle.get(), read_event_backward_load_request(event));
                            break;
                        case backward_load_response:
                            dg::network_producer_consumer::delvrsrv_deliver(bwd_load_response_delivery_handle.get(), read_event_backward_load_response(event));
                            break;
                        case backward_load_redirect_request:
                            dg::network_producer_consumer::delvrsrv_deliver(bwd_load_redirect_delivery_handle.get(), read_event_backward_load_redirect_request(event));
                            break;
                        default:
                            if constexpr(DEBUG_MODE_FLAG){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            }
                            break;
                    }
                }

                return true;
            }
    };

}

#endif