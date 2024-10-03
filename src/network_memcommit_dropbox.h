#ifndef __NETWORK_MEMCOMMIT_DROPBOX_H__
#define __NETWORK_MEMCOMMIT_DROPBOX_H__

#include <stdint.h>
#include <stdlib.h>
#include "network_producer_consumer.h"
#include "network_memcommit_model.h"

namespace dg::network_memcommit_dropbox{

    using virtual_memory_event_t    = network_memcommit_factory::virtual_memory_event_t;  

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
                        case forward_pong_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(fwd_pong_signal_delivery_handle.get(), read_event_forward_pong_signal(event));
                            break;
                        case backward_do_signal:
                            dg::network_producer_consumer::delvrsrv_deliver(bwd_do_signal_delivery_handle.get(), read_event_backward_do_signal(event));
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