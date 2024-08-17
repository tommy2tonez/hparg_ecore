#ifndef __NETWORK_EXTERNAL_HANDLER_H__
#define __NETWORK_EXTERNAL_HANDLER_H__

#include "network_log.h"

namespace dg::network_memcommit_kernel_handler{

    using event_loop_register_t = void (*)(void (*)(void) noexcept); 
    using payload_taxonomy_t    = uint8_t; 
    
    enum payload_taxonomy: payload_taxonomy_t{
        payload_tile_signal = 0u,
        payload_tile_inject = 1u,
        payload_tile_init   = 2u
    };

    template <class T>
    struct TileControllerInterface{

        static inline void add_translation_rule(vma_ptr_t new_addr, vma_ptr_t old_addr) noexcept{

            T::add_translation_rule(new_addr, old_addr);
        }
    
        static inline auto locate(vma_ptr_t ptr) noexcept -> virtual_device_id_t{

            return T::locate(ptr);
        }

        static inline void init(vma_ptr_t ptr, tile_init_payload payload) noexcept{

            T::init(ptr, payload);
        }
    
        static inline auto next(virtual_device_id_t device_id, tile_taxonomy_t tile_type) noexcept -> vma_ptr_t{

            return T::next(device_id, tile_type);
        } 
    };

    template <class T>
    struct KernelPayloadDeserializerInterface{

        static inline auto is_correct_format(const char * buf, size_t sz) noexcept -> bool{

            return T::is_correct_format(buf, sz);
        }
   
        static inline auto get_payload_taxonomy(const char * buf, size_t sz) noexcept -> payload_taxonomy_t{

            return T::get_payload_taxonomy(buf, sz);
        }

        static inline auto get_payload(const char * buf, size_t sz) noexcept -> std::pair<const char *, size_t>{

            return T::get_payload(buf, sz);
        }
    };

    template <class T>
    struct MemorySignalEventDeserializerInterface{

        static inline auto get_event(size_t idx, const char * buf, size_t sz) noexcept -> memory_event_t{

            return T::get_event(idx, buf, sz);
        }

        static inline auto get_notifying_addr(size_t idx, const char * buf, size_t sz) noexcept -> vma_ptr_t{
            
            return T::get_notifying_addr(idx, buf, sz);
        }

        static inline auto size(const char * buf, size_t sz) noexcept -> size_t{

            return T::size(buf, sz);
        }
    };

    template <class T>
    struct MemoryInjectionEventDeserializerInterface{

        static inline auto get_event(size_t idx, const char * buf, size_t sz) noexcept -> memory_event_t{

            return T::get_event(idx, buf, sz);
        }

        static inline auto get_notifying_addr(size_t idx, const char * buf, size_t sz) noexcept -> vma_ptr_t{
            
            return T::get_notifying_addr(idx, buf, sz);
        }

        static inline auto get_injector_addr(size_t idx, const char * buf, size_t sz) noexcept -> vma_ptr_t{

            return T::get_injector_addr(idx, buf, sz);
        } 

        static inline auto get_injector_taxonomy(size_t idx, const char * buf, size_t sz) noexcept -> tile_taxonomy_t{

            return T::get_injector_taxonomy(idx, buf, sz);
        }
        
        static inline void get_injector(size_t idx, vma_ptr_t dst, const char * buf, size_t sz) noexcept{ //decouple dependency here

            T::get_injector(idx, dst, buf, sz);
        }

        static inline auto size(const char * buf, size_t sz) noexcept -> size_t{

            return T::size(buf, sz);
        }
    };

    template <class T>
    struct MemoryInitializationEventDeserializerInterface{

        static inline auto get_init_addr(size_t idx, const char * buf, size_t sz) noexcept -> vma_ptr_t{

            return T::get_init_addr(idx, buf, sz);
        } 

        static inline auto get_init_payload(size_t idx, const char * buf, size_t sz) noexcept -> tile_init_payload{

            return T::get_init_payload(idx, buf, sz);
        }

        static inline auto size(const char * buf, size_t sz) noexcept -> size_t{

            return T::size(buf, sz);
        }
    };

    template <class T>
    struct MemoryEventObserverInterface{

        static inline void notify(vma_ptr_t ptr, memory_event_t event) noexcept{

            T::notify(ptr, event);
        }
    };

    template <class T>
    struct KernelProducerInterface: dg::network_producer_consumer::ProducerInterface<KernelProducerInterface<T>>{

        using event_t = char; 

        static inline void get(char * buf, size_t& sz, size_t cap) noexcept{

            T::get(buf, sz, cap);
        }
    };

    template <class ID, class T, class T1, class T2, class T3, class T4, class T5, class T6>
    struct ExternalHandler{};

    template <class ID, class T, class T1, class T2, class T3, class T4, class T5, class T6>
    struct ExternalHandler<ID, TileControllerInterface<T>, KernelPayloadDeserializerInterface<T1>, MemorySignalEventDeserializerInterface<T2>, 
                           MemoryInjectionEventDeserializerInterface<T3>, MemoryInitializationEventDeserializerInterface<T4>,
                           MemoryEventObserverInterface<T5>, KernelProducerInterface<T6>>{
        
        private:    

            using tile_controller                       = TileControllerInterface<T>;
            using kernel_payload_deserializer           = KernelPayloadDeserializerInterface<T4>;
            using memsignal_deserializer                = MemorySignalEventDeserializerInterface<T5>;
            using meminject_deserializer                = MemoryInjectionEventDeserializerInterface<T6>;
            using meminit_deserializer                  = MemoryInitializationEventDeserializerInterface<T7>;
            using memevent_observer                     = MemoryEventObserverInterface<T8>;
            using kernel_producer                       = KernelProducerInterface<T9>;

            static inline void resolve_tile_signal(const char * buf, size_t buf_sz) noexcept{

                size_t sz = memsignal_deserializer::size(buf, buf_sz);

                for (size_t i = 0; i < sz; ++i){
                    auto memevent       = memsignal_deserializer::get_event(i, buf, buf_sz);
                    auto notifying_vma  = memsignal_deserializer::get_notifying_addr(i, buf, buf_sz);
                    memevent_observer::notify(notifying_vma, memevent);
                }
            }

            static inline void resolve_tile_inject(const char * buf, size_t buf_sz) noexcept{

                size_t sz = meminject_deserializer::size(buf, buf_sz);

                for (size_t i = 0; i < sz; ++i){
                    auto memevent       = meminject_deserializer::get_event(i, buf, buf_sz);
                    auto notifying_vma  = meminject_deserializer::get_notifying_addr(i, buf, buf_sz);
                    auto injector_vma   = meminject_deserializer::get_injector_addr(i, buf, buf_sz);
                    auto injector_taxo  = meminject_deserializer::get_injector_taxonomy(i, buf, buf_sz);
                    auto injecting_cand = tile_controller::next(tile_controller::locate(notifying_vma), injector_taxo);
                    meminject_deserializer::get_injector(i, injecting_cand, buf, buf_sz);
                    tile_controller::add_translation_rule(injecting_cand, injector_vma);
                    memevent_observer::notify(notifying_vma, memevent);
                }
            }

            static inline void resolve_tile_init(const char * buf, size_t buf_sz) noexcept{

                size_t sz = meminit_deserializer::size(buf, buf_sz);

                for (size_t i = 0; i < sz; ++i){
                    auto init_vma       = meminit_deserializer::get_init_addr(i, buf, buf_sz);
                    auto init_payload   = meminit_deserializer::get_init_payload(i, buf, buf_sz);
                    tile_controller::init(init_vma, init_payload);
                }
            }

        public:

            static void run() noexcept{

                char * buf = {};
                size_t buf_sz = {};
                size_t BUF_CAP = {};

                kernel_producer::get(buf, buf_sz, BUF_CAP);

                if (!kernel_payload_deserializer::is_correct_format(buf, buf_sz)){
                    dg::network_log_stackdump::error_optional_fast("unrecognized network serialization format");
                    return;
                }
   
                payload_taxonomy_t payload_taxonomy = kernel_payload_deserializer::get_payload_taxonomy(buf, buf_sz);
                std::tie(buf, buf_sz)               = kernel_payload_deserializer::get_payload(buf, buf_sz); 

                switch (payload_taxonomy){
                    case payload_tile_signal:
                        resolve_tile_signal(buf, buf_sz);
                        break;
                    case payload_tile_inject:
                        resolve_tile_inject(buf, buf_sz);
                        break;
                    case payload_tile_init:
                        resolve_tile_init(buf, buf_sz);
                        break;
                    default:
                        std::abort();
                        break;
                }
            }

            static void init(event_loop_register_t event_loop_register) noexcept{

                event_loop_register(run);
            }
    };
} 

#endif