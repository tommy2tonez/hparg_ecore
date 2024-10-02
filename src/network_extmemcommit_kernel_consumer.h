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
    using request_t     = dg::network_extmemcommit_dropbox::request_t;
    using event_kind_t  = dg::network_extmemcommit_model::event_kind_t; 

    struct EventBalancerInterface{
        virtual ~EventBalancerInterface() noexcept = default;
        virtual void push(dg::network_std_container::vector<poly_event_t>) noexcept = 0;
        virtual auto pop(event_kind_t) noexcept -> dg::network_std_container::vector<poly_event_t> = 0;
    };

    class SynchronousEventBalancer: public virtual EventBalancerInterface{

        private:

            dg::network_std_container::unordered_map<event_kind_t, dg::network_std_container::vector<poly_event_t>> event_dict;
            dg::network_std_container::unordered_map<event_kind_t, size_t> pop_cap_dict;
            std::unique_ptr<std::mutex> mtx;

        public:

            SynchronousEventBalancer(dg::network_std_container::unordered_map<event_taxo_t, dg::network_std_container::vector<Event>> event_dict, 
                                     dg::network_std_container::unordered_map<event_taxo_t, size_t> pop_cap_dict,
                                     std::unique_ptr<std::mutex> mtx) noexcept: event_dict(std::move(event_dict)),
                                                                                pop_cap_dict(std::move(pop_cap_dict)),
                                                                                mtx(std::move(mtx)){}
            
            void push(dg::network_std_container::vector<poly_event_t> events) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                for (auto& event: events){
                    if (dg::network_extmemcommit_model::is_signal_event(event)){
                        event_dict[signal_event_kind].push_back(std::move(event));
                        continue;
                    }

                    if (dg::network_extmemcommit_model::is_inject_event(event)){
                        event_dict[inject_event_kind].push_back(std::move(event));
                        continue;
                    }

                    if (dg::network_extmemcommit_model::is_init_event(event)){
                        event_dict[init_event_kind].push_back(std::move(event));
                        continue;
                    }

                    if (dg::network_extmemcommit_model::is_conditional_inject_event(event)){
                        event_dict[conditional_inject_event_kind].push_back(std::move(event));
                        continue;
                    }

                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }
            }

            auto pop(event_kind_t kind) noexcept -> std::optional<dg::network_std_container::vector<poly_event_t>>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto dict_ptr   = this->event_dict.find(kind);

                if constexpr(DEBUG_MODE_FLAG){
                    if (dict_ptr == this->event_dict.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                if (dict_ptr->second.size() == 0u){
                    return std::nullopt;
                }

                size_t pop_sz   = std::min(this->pop_cap_dict[kind], dict_ptr->second.size());
                auto rs         = dg::network_std_container::vector<poly_event_t>{};
                rs.reserve(pop_sz);

                for (size_t i = 0u; i < pop_sz; ++i){
                    rs.push_back(std::move(dict_ptr->second.back()));
                    dict_ptr->second.pop_back();
                }

                return rs;
            }
    };
    
    template <size_t BALANCER_SZ>
    class ConcurrentEventBalancer: public virtual EventBalancerInterface{

        private:

            dg::network_std_container::vector<std::unique_ptr<EventBalancerInterface>> event_balancer;

        public:

            ConcurrentEventBalancer(dg::network_std_container::vector<std::unique_ptr<EventBalancerInterface>> event_balancer,
                                    std::integral_constant<size_t, BALANCER_SZ>) noexcept: event_balancer(std::move(event_balancer)){}
            
            void push(dg::network_std_container::vector<poly_event_t> events) noexcept{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, BALANCER_SZ>{});
                this->event_balancer[idx]->push(std::move(events));
            }

            auto pop(event_taxo_t event_taxo) noexcept -> std::optional<dg::network_std_container::vector<poly_event_t>>{

                size_t idx = dg::network_randomizer::randomize_range(std::integral_constant<size_t, BALANCER_SZ>{});
                return this->event_balancer[idx]->pop(event_taxo);
            }
    };

    // class BalancerWrappedInitEventProducer: public virtual InitEventProducerInterface{

    //     private:

    //         std::shared_ptr<EventBalancerInterface> event_balancer;
        
    //     public:

    //         BalancerWrappedInitEventProducer(std::shared_ptr<EventBalancerInterface> event_balancer) noexcept: event_balancer(std::move(event_balancer)){}

    //         auto get() noexcept -> std::optional<dg::network_std_container::vector<InitEvent>>{

    //             std::optional<dg::network_std_container::vector<poly_event_t>> poly_event_arr = this->event_balancer->pop(init_event);

    //             if (!static_cast<bool>(poly_event_arr)){
    //                 return std::nullopt;
    //             }

    //             dg::network_std_container::vector<InitEvent> rs(poly_event_arr->size());

    //             for (size_t i = 0u; i < rs.size(); ++i){
    //                 rs[i] = std::move(std::get<InitEvent>(poly_event_arr->operator[](i)));
    //             }

    //             return rs;
    //         }
    // };

    // class BalancerWrappedSignalEventProducer: public virtual SignalEventProducerInterface{

    //     private:

    //         std::shared_ptr<EventBalancerInterface> event_balancer;
        
    //     public:

    //         BalancerWrappedSignalEventProducer(std::shared_ptr<EventBalancerInterface> event_balancer) noexcept: event_balancer(std::move(event_balancer)){}

    //         auto get() noexcept -> std::optional<dg::network_std_container::vector<SignalEvent>>{

    //             std::optional<dg::network_std_container::vector<poly_event_t>> poly_event_arr = this->event_balancer->pop(signal_event);

    //             if (!static_cast<bool>(poly_event_arr)){
    //                 return std::nullopt;
    //             }

    //             dg::network_std_container::vector<SignalEvent> rs(poly_event_arr->size());

    //             for (size_t i = 0u; i < rs.size(); ++i){
    //                 rs[i] = std::move(std::get<SignalEvent>(poly_event_arr->operator[](i)));
    //             }

    //             return rs;
    //         }
    // };

    // class BalancerWrappedInjectEventProducer: public virtual InjectEventProducerInterface{

    //     private:

    //         std::shared_ptr<EventBalancerInterface> event_balancer;

    //     public:

    //         BalancerWrappedInjectEventProducer(std::shared_ptr<EventBalancerInterface> event_balancer) noexcept: event_balancer(std::move(event_balancer)){}

    //         auto get() noexcept -> std::optional<dg::network_std_container::vector<InjectEvent>>{

    //             std::optional<dg::network_std_container::vector<poly_event_t>> poly_event_arr = this->event_balancer->pop(signal_event);

    //             if (!static_cast<bool>(poly_event_arr)){
    //                 return std::nullopt;
    //             }

    //             dg::network_std_container::vector<InjectEvent> rs(poly_event_arr->size());

    //             for (size_t i = 0u; i < rs.size(); ++i){
    //                 rs[i] = std::move(std::get<InjectEvent>(poly_event_arr->operator[](i)));
    //             }

    //             return rs;
    //         } 
    // };

    class RequestConsumer: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<EventBalancerInterface> event_balancer;
            std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<request_t>> request_producer;
            const size_t request_digest_sz;

        public:

            KernelConsumer(std::shared_ptr<EventBalancerInterface> event_balancer,
                           std::shared_ptr<dg::network_raii_producer_consumer::ProducerInterface<request_t>> request_producer,
                           size_t request_digest_sz) noexcept: event_balancer(std::move(event_balancer)),
                                                               request_producer(std::move(request_producer)),
                                                               request_digest_sz(request_digest_sz){}
            
            bool run_one_epoch() noexcept{

                dg::network_std_container::vector<request_t> request_vec = this->request_producer->get(this->request_digest_sz);

                if (request_vec.empty()){
                    return false;
                }

                dg::network_std_container::vector<poly_event_t> poly_event_vec{};
                poly_event_vec.reserve(request_vec.size());

                for (size_t i = 0u; i < request_vec.size(); ++i){
                    poly_event_vec.push_back(std::move(request_vec[i].event));
                }

                this->event_balancer->push(std::move(poly_event_vec));
                return true;
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
                    dg::network_tile_init_poly::load_nothrow(std::move(initializable));
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
                    dg::network_tile_signal_poly::load_nothrow(std::move(signalable));
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
                    dg::network_tile_inject_poly::load_nothrow(std::move(injectible));
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
                    dg::network_tile_condinjection_poly::load_nothrow(std::move(injectible));
                }

                return true;
            }
    };
}

#endif