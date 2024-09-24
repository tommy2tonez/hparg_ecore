#ifndef __NETWORK_EXTERNAL_HANDLER_H__
#define __NETWORK_EXTERNAL_HANDLER_H__

#include "network_log.h"
#include "network_concurrency.h"
#include "network_tile_queue.h" 
#include "network_tile_initialization.h"
#include "network_std_container.h"
#include "network_tile_injection.h"
#include "network_tile_signal.h"
#include "network_kernel_mailbox.h"

namespace dg::network_extmemcommit_kernel_handler{

    using event_taxo_t = uint8_t;

    enum event_option: event_taxo_t{
        signal_event    = 0u,
        inject_event    = 1u,
        init_event      = 2u
    };

    struct SignalEvent{
        dg::network_tile_signal_poly::virtual_payload_t payload;
    };

    struct InjectEvent{
        dg::network_tile_injection_poly::virtual_payload_t payload;
    };

    struct InitEvent{
        dg::network_tile_initialization_poly::virtual_payload_t payload;
    };

    struct poly_event_t = std::variant<SignalEvent, InjectEvent, InitEvent>; 
    
    struct EventBalancerInterface{
        virtual ~EventBalancerInterface() noexcept = default;
        virtual void push(dg::network_std_container::vector<poly_event_t>) noexcept = 0;
        virtual auto pop(event_taxo_t) noexcept -> std::optional<dg::network_std_container::vector<poly_event_t>> = 0;
    };

    struct KernelProducerInterface{
        virtual ~KernelProducerInterface() noexcept = default;
        virtual auto get() noexcept -> std::optional<dg::network_std_container::string> = 0;    
    };

    struct InitEventProducerInterface{
        virtual ~InitEventProducerInterface() noexcept = default;
        virtual auto get() noexcept -> std::optional<dg::network_std_container::vector<InitEvent>> = 0; 
    };

    struct InjectEventProducerInterface{
        virtual ~InjectEventProducerInterface() noexcept = default;
        virtual auto get() noexcept -> std::optional<dg::network_std_container::vector<InjectEvent>> = 0;
    };

    struct SignalEventProducerInterface{
        virtual ~SignalEventProducerInterface() noexcept = default;
        virtual auto get() noexcept -> std::optional<dg::network_std_container::vector<SignalEvent>> = 0;
    };

    class SynchronousEventBalancer: public virtual EventBalancerInterface{

        private:

            dg::network_std_container::unordered_map<event_taxo_t, dg::network_std_container::vector<poly_event_t>> event_dict;
            dg::network_std_container::unordered_map<event_taxo_t, size_t> pop_cap_dict;
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
                    if (std::holds_alternative<SignalEvent>(event)){
                        event_dict[signal_event].push_back(std::move(event));
                        continue;
                    }

                    if (std::holds_alternative<InjectEvent>(event)){
                        event_dict[inject_event].push_back(std::move(event));
                        continue;
                    }

                    if (std::holds_alternative<InitEvent>(event)){
                        event_dict[init_event].push_back(std::move(event));
                        continue;
                    }

                    if constexpr(DEBUG_MODE_FLAG){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }
            }

            auto pop(event_taxo_t event_taxo) noexcept -> std::optional<dg::network_std_container::vector<poly_event_t>>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto dict_ptr   = this->event_dict.find(event_taxo);

                if constexpr(DEBUG_MODE_FLAG){
                    if (dict_ptr == this->event_dict.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                if (dict_ptr->second.size() == 0u){
                    return std::nullopt;
                }

                size_t pop_sz = std::min(this->pop_cap_dict[event_taxo], dict_ptr->second.size());
                auto rs = dg::network_std_container::vector<poly_event_t>(pop_sz);

                for (size_t i = 0u; i < pop_sz; ++i){
                    rs[i] = std::move(dict_ptr->second.back());
                    dict_ptr->second.pop_back();
                }

                return rs;
            }
    };
    
    class KernelProducer: public virtual KernelProducerInterface{

        public:

            auto get() noexcept -> std::optional<dg::network_std_container::string>{

                return dg::network_kernel_mailbox::recv();
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

    class BalancerWrappedInitEventProducer: public virtual InitEventProducerInterface{

        private:

            std::shared_ptr<EventBalancerInterface> event_balancer;
        
        public:

            BalancerWrappedInitEventProducer(std::shared_ptr<EventBalancerInterface> event_balancer) noexcept: event_balancer(std::move(event_balancer)){}

            auto get() noexcept -> std::optional<dg::network_std_container::vector<InitEvent>>{

                std::optional<dg::network_std_container::vector<poly_event_t>> poly_event_arr = this->event_balancer->pop(init_event);

                if (!static_cast<bool>(poly_event_arr)){
                    return std::nullopt;
                }

                dg::network_std_container::vector<InitEvent> rs(poly_event_arr->size());

                for (size_t i = 0u; i < rs.size(); ++i){
                    rs[i] = std::move(std::get<InitEvent>(poly_event_arr->operator[](i)));
                }

                return rs;
            }
    };

    class BalancerWrappedSignalEventProducer: public virtual SignalEventProducerInterface{

        private:

            std::shared_ptr<EventBalancerInterface> event_balancer;
        
        public:

            BalancerWrappedSignalEventProducer(std::shared_ptr<EventBalancerInterface> event_balancer) noexcept: event_balancer(std::move(event_balancer)){}

            auto get() noexcept -> std::optional<dg::network_std_container::vector<SignalEvent>>{

                std::optional<dg::network_std_container::vector<poly_event_t>> poly_event_arr = this->event_balancer->pop(signal_event);

                if (!static_cast<bool>(poly_event_arr)){
                    return std::nullopt;
                }

                dg::network_std_container::vector<SignalEvent> rs(poly_event_arr->size());

                for (size_t i = 0u; i < rs.size(); ++i){
                    rs[i] = std::move(std::get<SignalEvent>(poly_event_arr->operator[](i)));
                }

                return rs;
            }
    };

    class BalancerWrappedInjectEventProducer: public virtual InjectEventProducerInterface{

        private:

            std::shared_ptr<EventBalancerInterface> event_balancer;

        public:

            BalancerWrappedInjectEventProducer(std::shared_ptr<EventBalancerInterface> event_balancer) noexcept: event_balancer(std::move(event_balancer)){}

            auto get() noexcept -> std::optional<dg::network_std_container::vector<InjectEvent>>{

                std::optional<dg::network_std_container::vector<poly_event_t>> poly_event_arr = this->event_balancer->pop(signal_event);

                if (!static_cast<bool>(poly_event_arr)){
                    return std::nullopt;
                }

                dg::network_std_container::vector<InjectEvent> rs(poly_event_arr->size());

                for (size_t i = 0u; i < rs.size(); ++i){
                    rs[i] = std::move(std::get<InjectEvent>(poly_event_arr->operator[](i)));
                }

                return rs;
            } 
    };

    class KernelConsumer: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<EventBalancerInterface> event_balancer;
            std::unique_ptr<KernelProducerInterface> kernel_producer;
        
        public:

            KernelConsumer(std::shared_ptr<EventBalancerInterface> event_balancer,
                                  std::unique_ptr<KernelProducerInterface> kernel_producer) noexcept: event_balancer(std::move(event_balancer)),
                                                                                                      kernel_producer(std::move(kernel_producer)){}
            
            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::string> kernel_data = this->kernel_producer->get(); 

                if (!static_cast<bool>(kernel_data)){
                    return false;
                }

                dg::network_std_container::vector<poly_event_t> event_payload{};
                std::expected<const char *, exception_t> rs = dg::network_compact_serializer::integrity_deserialize_into(event_payload, kernel_data->data(), kernel_data->size()); //consider convert -> unstable_addr str container that raii stable_addr - somewhat like realloc - to avoid computation

                if (!rs.has_value()){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(rs.error()));
                    std::abort();
                }

                this->event_balancer->push(std::move(event_payload));
                return true;
            }
    };

    class InitEventConsumer: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<InitEventProducerInterface> producer;

        public:

            InitEventConsumer(std::shared_ptr<InitEventProducerInterface> producer) noexcept: producer(std::move(producer)){}

            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::vector<InitEvent>> init_events = this->producer->get();

                if (!static_cast<bool>(init_events)){
                    return false;
                }

                for (auto& init_event: init_events.value()){
                    dg::network_tile_init_poly::load_nothrow(std::move(init_event.payload));
                }

                return true;
            }
    };

    class SignalEventConsumer: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<SignalEventProducerInterface> producer;
        
        public:

            SignalEventConsumer(std::shared_ptr<SignalEventProducerInterface> producer) noexcept: producer(std::move(producer)){}

            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::vector<SignalEvent>> signal_events = this->producer->get();

                if (!static_cast<bool>(signal_events)){
                    return false;
                }

                for (auto& signal_event: signal_events.value()){
                    dg::network_tile_signal_poly::load_nothrow(std::move(signal_event.payload));
                }

                return true;
            }
    };

    class InjectEventConsumer: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<InjectEventProducerInterface> producer;
        
        public:

            InjectEventConsumer(std::shared_ptr<InjectEventProducerInterface> producer) noexcept: producer(std::move(producer)){}

            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::vector<InjectEvent>> inject_events = this->producer->get();

                if (!static_cast<bool>(inject_events)){
                    return false;
                }

                for (auto& inject_event: inject_events.value()){
                    dg::network_tile_inject_poly::load_nothrow(std::move(inject_event.payload));
                }

                return true;
            }
    };
} 

#endif