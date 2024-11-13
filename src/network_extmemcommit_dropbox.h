#ifndef __NETWORK_EXTERNAL_MEMCOMMIT_DROPBOX_H__
#define __NETWORK_EXTERNAL_MEMCOMMIT_DROPBOX_H__

#include "network_extmemcommit_model.h"
#include "network_producer_consumer.h"
#include "network_concurrency.h"
#include "network_std_container.h"
#include "network_type_trait_x.h"
#include "stdx.h"

namespace dg::network_extmemcommit_dropbox{

    using event_t = dg::network_external_memcommit_model::poly_event_t; 

    struct Request{
        Address requestor;
        Address requestee;
        event_t event;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(requestor, requestee, event);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(requestor, requestee, event);
        }
    };

    using request_t = Request;
    
    struct RequestContainerInterface{
        virtual ~RequestContainerInterface() noexcept = default;
        virtual void push(Request) noexcept = 0;
        virtual auto pop() noexcept -> std::optional<Request> = 0;
    };

    struct RequestCenterInterface: public virtual RequestContainerInterface{
        virtual ~RequestCenterInterface() noexcept = default;
        virtual void send(Request) noexcept = 0;
        virtual auto recv() noexcept -> std::optional<Request> = 0;
    };

    class LckContainer: public virtual RequestContainerInterface{
        
        private:

            dg::vector<Request> request_vec;
            std::unique_ptr<std::mutex> lck;
        
        public:

            LckContainer(dg::vector<Request> request_vec,
                         std::unique_ptr<std::mutex> lck) noexcept: request_vec(std::move(request_vec)),
                                                                    lck(std::move(lck)){}

            void push(Request request) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->lck);
                this->request_vec.push_back(std::move(request));
            }

            auto pop() noexcept -> std::optional<Request>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->lck);
                
                if (this->request_vec.empty()){
                    return std::nullopt;
                }

                auto rs = std::move(this->request_vec.back());
                this->request_vec.pop_back();

                return {std::in_place_t{}, std::move(rs)};
            }
    };

    class OutBoundDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RequestContainerInterface> outbound_container;
            const size_t vectorization_sz; 
            const size_t addr_vectorization_sz; 

        public:

            OutBoundDispatcher(std::shared_ptr<RequestContainerInterface> outbound_container, 
                               size_t vectorization_sz,
                               size_t addr_vectorization_sz) noexcept: outbound_container(std::move(outbound_container)),
                                                                       vectorization_sz(vectorization_sz),
                                                                       addr_vectorization_sz(addr_vectorization_sz){}

            bool run_one_epoch() noexcept{
                
                const size_t SERIALIZATION_OVERHEAD = dg::network_compact_serializer::size(dg::vector<request_t>{});
                const size_t MAX_DISPATCH_BYTE_SZ   = dg::network_kernel_mailbox::MAX_SUBMIT_SIZE - SERIALIZATION_OVERHEAD;
                HelperClass dispatcher{}; 

                {
                    dg::vector<request_t> recv_request = this->recv();

                    if (recv_request.empty()){
                        return false;
                    }

                    using handle_t      = dg::network_type_traits_x::remove_expected_t<decltype(dg::network_raii_producer_consumer::xdelvsrv_open_raiihandle(&dispatcher, this->addr_vectorization_sz, MAX_DISPATCH_BYTE_SZ, MAX_DISPATCH_BYTE_SZ))>; //interface coersion might not work
                    auto delivery_map   = dg::unordered_map<Address, handle_t>{};

                    for (request_t& request: recv_request){
                        Address dst_ip  = request.requestor;
                        auto map_ptr    = delivery_map.find(dst_ip); 

                        if (map_ptr == delivery_map.end()){
                            auto handle = dg::network_exception_handler::nothrow_log(dg::network_raii_producer_consumer::xdelvsrv_open_raiihandle(&dispatcher, this->addr_vectorization_sz, MAX_DISPATCH_BYTE_SZ, MAX_DISPATCH_BYTE_SZ));
                            auto [emplace_ptr, status] = delivery_map.emplace(std::make_pair(dst_ip, std::move(handle)));
                            map_ptr = emplace_ptr;
                            dg::network_genult::assert(status);
                        }

                        exception_t err = dg::network_producer_consumer::xdelvsrv_deliver(map_ptr->second.get(), std::move(request), dg::network_compact_serializer::size(request)); //dangy
                        dg::network_exception_handler::nothrow_log(err);
                    }
                }

                return true;
            }
        
        private:

            auto recv() noexcept -> dg::vector<Request>{

                dg::vector<Request> rs{};
                rs.reserve(this->vectorization_sz);

                for (size_t i = 0u; i < this->vectorization_sz; ++i){
                    std::optional<Request> request = this->outbound_container->pop();
                    
                    if (!static_cast<bool>(request)){
                        return rs;
                    }

                    rs.push_back(std::move(request.value()));
                }

                return rs;
            }

            struct HelperClass: public virtual dg::network_raii_producer_consumer::ConsumerInterface<request_t>{
            
                void push(dg::vector<request_t> data) noexcept{
                    
                    if (data.size() == 0u){
                        return;
                    }

                    Address dst     = data.front().requestor;
                    size_t bsz      = dg::network_compact_serializer::size(data);
                    auto bstream    = dg::string(bsz);
                    dg::network_compact_serializer::serialize_into(bstream.data(), data);
                    dg::network_kernel_mailbox::send(dst, std::move(bstream), dg::network_kernel_mailbox::CHANNEL_EXTMEMCOMMIT);
                }
            };
    };

    class InBoundDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RequestContainerInterface> inbound_container;
        
        public:

            DropBoxDispatcher(std::shared_ptr<RequestContainerInterface> inbound_container) noexcept: inbound_container(std::move(inbound_container)){}

            bool run_one_epoch() noexcept{

                std::optional<dg::string> bstream = dg::network_kernel_mailbox::recv(dg::network_kernel_mailbox::CHANNEL_EXTMEMCOMMIT);
                
                if (!static_cast<bool>(bstream)){
                    return false;
                }

                dg::vector<Request> recv_data{};
                dg::network_compact_serializer::deserialize_into(recv_data, bstream->data());

                for (Request& request: recv_data){
                    this->inbound_container->push(std::move(request));
                }

                return true;
            }
    };

    //this is probably questionable - yet I think there's more to RequestCenter than just a warehouse
    //split the responsibility here for future extension - the implementation is placeholder only - not necessarily the final implementation
    //the responsibility that could only be extended here is kind request - let's say that signals are compact - injections are expensive - then vectorization sz for each kind should be different 
    //or anonymous request - think onion - request has random routing and a countdown - mask requestor and requestee
    //the implementation should be simple - for countdown != 0, randomize peer, change requestor - current, send request -> the randomized peer, track the request in memory for backprop 
    //not necessary in the next few years - but in the future - where massive P2P network is required - not for computation - but trust issues

    class RequestCenter: public virtual RequestCenterInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> workers;
            std::shared_ptr<RequestContainerInterface> outbound_container;
            std::shared_ptr<RequestContainerInterface> inbound_container;
        
        public:

            RequestCenter(dg::vector<dg::network_concurrency::daemon_raii_handle_t> workers,
                          std::shared_ptr<RequestContainerInterface> outbound_container,
                          std::shared_ptr<RequestContainerInterface> inbound_container) noexcept: workers(std::move(workers)),
                                                                                                  outbound_container(std::move(outbound_container)),
                                                                                                  inbound_container(std::move(inbound_container)){}
            
            void send(Request request) noexcept{

                this->outbound_container->push(std::move(request));
            }

            auto recv() noexcept -> std::optional<Request>{

                return this->inbound_container->pop();
            }
    };

    class RequestDropBox: public virtual dg::network_raii_producer_consumer::ConsumerInterface<Request>{

        private:

            std::shared_ptr<RequestCenterInterface> request_center;
        
        public:

            RequestDropBox(std::shared_ptr<RequestCenterInterface> request_center) noexcept: request_center(std::move(request_center)){}

            void push(dg::vector<Request> request_vec) noexcept{

                for (auto& request: request_vec){
                    this->request_center->send(std::move(request));
                }
            }
    };

    class RequestProducer: public virtual dg::network_raii_producer_consumer::ProducerInterface<Request>{

        private:

            std::shared_ptr<RequestCenterInterface> request_center;
        
        public:

            RequestProducer(std::shared_ptr<RequestCenterInterface> request_center) noexcept: request_center(std::move(request_center)){}

            auto get(size_t capacity) noexcept -> dg::vector<Request>{

                dg::vector<Request> vec{};

                for (size_t i = 0u; i < capacity; ++i){
                    std::optional<Request> request = this->request_center->pop();

                    if (!static_cast<bool>(request)){
                        return vec;
                    }

                    vec.push_back(std::move(request.value()));
                }

                return vec;
            }
    };
}

#endif