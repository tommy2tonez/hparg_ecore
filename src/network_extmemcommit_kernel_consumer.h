#ifndef __NETWORK_EXTERNAL_HANDLER_H__
#define __NETWORK_EXTERNAL_HANDLER_H__

#include "network_log.h"
#include "network_concurrency.h"
#include "network_tile_initialization.h"
#include "network_std_container.h"
#include "network_tile_injection.h"
#include "network_tile_signal.h"
#include "network_extmemcommit_model.h"
#include "network_producer_consumer.h"

namespace dg::network_extmemcommit_kernel_handler{

    using poly_event_t  = dg::network_extmemcommit_model::poly_event_t;
    
    template <class T>
    using Request = dg::network_model::Request<T>;

    class RequestConsumer: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<Request<poly_event_t>>> request_producer;
            std::shared_ptr<dg::network_raii_producer_consumer::ConsumerInterface<dg::network_tile_initialization_poly::virtual_payload_t>> initializable_consumer;
            std::shared_ptr<dg::network_raii_producer_consumer::ConsumerInterface<dg::network_tile_signal_poly::virtual_paylaod_t>> signalable_consumer;
            std::shared_ptr<dg::network_raii_producer_consumer::ConsumerInterface<dg::network_tile_injection_poly::virtual_payload_t>> injectible_consumer;
            std::shared_ptr<dg::network_raii_producer_consumer::ConsumerInterface<dg::network_tile_condinjection_poly::virtual_payload_t>> condinjectible_consumer;
            const size_t digest_sz;

        public:

            RequestConsumer(std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<request_t>> request_producer,
                            std::shared_ptr<dg::network_raii_producer_consumer::ConsumerInterface<dg::network_tile_initialization_poly::virtual_payload_t>> initializable_consumer,
                            std::shared_ptr<dg::network_raii_producer_consumer::ConsumerInterface<dg::network_tile_signal_poly::virtual_payload_t>> signalable_consumer,
                            std::shared_ptr<dg::network_raii_producer_consumer::ConsumerInterface<dg::network_tile_injection_poly::virtual_payload_t>> injectible_consumer,
                            std::shared_ptr<dg::network_raii_producer_consumer::ConsumerInterface<dg::network_tile_condinjection_poly::virtual_payload_t>> condinjectible_consumer,
                            size_t digest_sz) noexcept: request_producer(std::move(request_producer)),
                                                        initializable_consumer(std::move(initializable_consumer)),
                                                        signalable_consumer(std::move(signalable_consumer)),
                                                        injectible_consumer(std::move(injectible_consumer)),
                                                        condinjectible_consumer(std::move(condinjectible_consumer)),
                                                        digest_sz(digest_sz){}
            
            bool run_one_epoch() noexcept{

                dg::network_std_container::vector<Request<poly_event_t>> request_vec = this->request_producer->get(this->digest_sz);

                if (request_vec.empty()){
                    return false;
                }

                auto initializable_delivery_handle  = dg::network_exception_handler::nothrow_log(dg::network_raii_producer_consumer::delvsrv_open_raiihandle(this->initializable_consumer.get(), this->initializable_ingest_sz));
                auto signalable_delivery_handle     = dg::network_exception_handler::nothrow_log(dg::network_raii_producer_consumer::delvsrv_open_raiihandle(this->signalable_consumer.get(), this->signalable_ingest_sz)); 
                auto injectible_delivery_handle     = dg::network_exception_handler::nothrow_log(dg::network_raii_producer_consumer::delvsrv_open_raiihandle(this->injectible_consumer.get(), this->injectible_ingest_sz));
                auto condinjectible_delivery_handle = dg::network_exception_handler::nothrow_log(dg::network_raii_producer_consumer::delvsrv_open_raiihandle(this->condinjectible_consumer.get(), this->condinjectible_ingest_sz)); 

                for (Request<poly_event_t>& request: request_vec){
                    poly_event_t event = std::move(request.content); 

                    if (dg::network_extmemcommit_model::is_signal_event(event)){
                        dg::network_raii_producer_consumer::delvsrv_deliver(signalable_delivery_handle.get(), dg::network_extmemcommit_model::get_signal_event(std::move(event)));
                        continue;
                    }

                    if (dg::network_extmemcommit_model::is_inject_event(event)){
                        dg::network_raii_producer_consumer::delvsrv_deliver(injectible_delivery_handle.get(), dg::network_extmemcommit_model::get_inject_event(std::move(event)));
                        continue;
                    }

                    if (dg::network_extmemcommit_model::is_conditional_inject_event(event)){
                        dg::network_raii_producer_consumer::delvsrv_deliver(condinjectible_delivery_handle.get(), dg::network_extmemcommit_model::get_condinject_event(std::move(event)));
                        continue;
                    }

                    if (dg::network_extmemcommit_model::is_init_event(event)){
                        dg::network_raii_producer_consumer::delvsrv_deliver(initializable_delivery_handle.get(), dg::network_extmemcommit_model::get_init_event(std::move(event)));
                        continue;
                    }

                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }
            }
    };

    class InitializableConsumer: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<dg::network_tile_initialization_poly::virtual_payload_t>> producer;
            const size_t digest_sz; 

        public:

            InitializableConsumer(std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<dg::network_tile_initialization_poly::virtual_payload_t>> producer,
                                  size_t digest_sz) noexcept: producer(std::move(producer)),
                                                              digest_sz(digest_sz){}

            bool run_one_epoch() noexcept{

                dg::network_std_container::vector<dg::network_tile_initialization_poly::virtual_payload_t> initializable_vec = this->producer->get(this->digest_sz);

                if (initializable_vec.empty()){
                    return false;
                }

                for (auto& initializable: initializable_vec){
                    exception_t err = dg::network_tile_init_poly::load(std::move(initializable));

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log::error_fast(dg::network_exception::verbose(err));
                    }
                }

                return true;
            }
    };

    class SignalableConsumer: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<dg::network_tile_signal_poly::virtual_payload_t>> producer;
            const size_t digest_sz;

        public:

            SignalableConsumer(std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<dg::network_tile_signal_poly::virtual_payload_t>> producer,
                               size_t digest_sz) noexcept: producer(std::move(producer)),
                                                           digest_sz(digest_sz){}

            bool run_one_epoch() noexcept{

                dg::network_std_container::vector<dg::network_tile_signal_poly::virtual_payload_t> signalable_vec = this->producer->get(this->digest_sz);

                if (signalable_vec.empty()){
                    return false;
                }

                for (auto& signalable: signalable_vec){
                    exception_t err = dg::network_tile_signal_poly::load(std::move(signalable));

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log::error_fast(dg::network_exception::verbose(err));
                    }
                }

                return true;
            }
    };

    class InjectibleConsumer: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<dg::network_tile_injection_poly::virtual_payload_t>> producer;
            const size_t digest_sz;

        public:

            InjectibleConsumer(std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<dg::network_tile_injection_poly::virtual_payload_t>> producer,
                               size_t digest_sz) noexcept: producer(std::move(producer)),
                                                           digest_sz(digest_sz){}

            bool run_one_epoch() noexcept{

                dg::network_std_container::vector<dg::network_tile_injection_poly::virtual_payload_t> injectible_vec = this->producer->get(this->digest_sz);

                if (injectible_vec.empty()){
                    return false;
                }

                for (auto& injectible: injectible_vec){
                    exception_t err = dg::network_tile_inject_poly::load(std::move(injectible));

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log::error_fast(dg::network_exception::verbose(err));
                    }
                }

                return true;
            }
    };

    class ConditionalInjectibleConsumer: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<dg::network_tile_condinjection_poly::virtual_payload_t>> producer;
            const size_t digest_sz;
        
        public:

            ConditionalInjectibleConsumer(std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<dg::network_tile_condinjection_poly::virtual_payload_t>> producer,
                                          size_t digest_sz) noexcept: producer(std::move(producer)),
                                                                      digest_sz(digest_sz){}
            
            bool run_one_epoch() noexcept{

                dg::network_std_container::vector<dg::network_tile_condinjection_poly::virtual_payload_t> injectible_vec = this->producer->get(this->digest_sz);

                if (injectible_vec.empty()){
                    return false;
                }

                for (auto& injectible: injectible_vec){
                    exception_t err = dg::network_tile_condinjection_poly::load(std::move(injectible));

                    if (dg::network_exception::is_failed(err)){
                        dg::network_log::error_fast(dg::network_exception::verbose(err));
                    }
                }

                return true;
            }
    };
}

#endif