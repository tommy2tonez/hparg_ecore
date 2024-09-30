#ifndef __NETWORK_EXTERNAL_MEMCOMMIT_DROPBOX_H__
#define __NETWORK_EXTERNAL_MEMCOMMIT_DROPBOX_H__

#include "network_extmemcommit_model.h"
#include "network_producer_consumer.h"
#include "network_concurrency.h"

namespace dg::network_external_memcommit_dropbox{

    using event_t = dg::network_external_memcommit_model::poly_event_t; 

    struct Request{
        Address requestor;
        Address requestee;
        event_t event;
    };

    using request_t = Request;

    struct DropBoxInterface: public virtual dg::network_producer_consumer::ProducerInterface<request_t>,
                             public virtual dg::network_producer_consumer::ConsumerInterface<request_t>{};


    class LckDropBox: public virtual DropBoxInterface{

        private:
            
            dg::network_std_container::vector<request_t> container;
            std::unique_ptr<std::mutex> mtx;

        public:

            LckDropBox(dg::network_std_container::vector<request_t> container,
                       std::unique_ptr<std::mutex> mtx) noexcept: container(std::move(container)),
                                                                  mtx(std::move(mtx)){}
            
            
    };

    class ConcurrentDropBox: public virtual DropBoxDispatcher{

    };

    //inbound dropbox + outbound dropbox

    class MailBoxDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::unique_ptr<dg::network_producer_consumer::ProducerInterface<request_t> producer;

            struct HelperClass: public virtual dg::network_producer_consumer::ConsumerInterface<request_t>{
            
                void push(request_t * request, size_t sz) noexcept{ //this does not specify ownership - this is bad practice - can do something like std::add_rvalue_reference<request_t> * - yet it looks incredibly dumb
                    
                    if (sz == 0u){
                        return;
                    }

                    Address dst_addr    = request[0].requestor;
                    auto vec_view       = std::span<request_t>{request, sz};
                    size_t bsz          = dg::network_compact_serializer::size(vec_view);
                    auto bstream        = dg::network_std_container::string(bsz);
                    dg::network_compact_serializer::serialize_into(bstream.data(), vec_view);
                    dg::network_kernel_mailbox::send(dst_addr, std::move(bstream), EXTERNAL_MEMCOMMIT_DROPBOX_IDENTIFIER);
                }
            };

        public:

            MailBoxDispatcher(std::unique_ptr<dg::network_producer_consumer::ProducerInterface<request_t> producer) noexcept: producer(std::move(producer)){}

            bool run_one_epoch() noexcept{
                
                const size_t MAX_DISPATCH_BYTE_SZ = dg::kernel_mailbox::UDP_PACKET_SIZE >> 2;
                HelperClass dispatcher{}; 

                {
                    dg::network_std_container::vector<request_t> recv_request{};
                    recv_request.reserve(this->capacity); // 
                    size_t recv_sz{}; 
                    this->producer->get(recv_request.data(), recv_sz, this->capacity);

                    if (recv_sz == 0u){
                        return false;
                    }

                    recv_request.resize(recv_sz);
                    auto delivery_map = dg::network_std_container::unordered_map<Address, decltype(dg::network_producer_consumer::xdelvsrv_open_raiihandle_nothrow(&vetorizer, DELIVERY_THRHOLD, MAX_DISPATCH_BYTE_SZ))>{};

                    for (request_t& req: recv_request){
                        Address dst_ip = req.requestor;

                        if (delivery_map.find(dst_ip) == delivery_map.end()){
                            delivery_map.emplace(std::make_pair(dst_ip, dg::network_producer_consumer::xdelvsrv_open_raiihandle_nothrow(&dispatcher, DELIVERY_THRHOLD, MAX_DISPATCH_BYTE_SZ)));
                        }

                        dg::network_producer_consumer::xdelvsrv_deliver(delivery_map.find(dst_ip)->second.get(), std::move(req));
                    }
                }

                return true;
            }
    };

    class DropBoxDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<request_t>> consumer;
        
        public:

            DropBoxDispatcher(std::unique_ptr<dg::network_producer_consumer::ConsumerInterface<request_t>> consumer) noexcept: consumer(std::move(consumer)){}

            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::string> bstream = dg::network_kernel_mailbox::recv(EXTERNAL_MEMCOMMIT_DROPBOX_IDENTIFIER);
                
                if (!static_cast<bool>(bstream)){
                    return false;
                }

                dg::network_std_container::vector<request_t> recv_data{};
                dg::network_compact_serializer::deserialize_into(recv_data, bstream->data()); 
                this->consumer->push(recv_data.data(), recv_data.size()); //raii transfering should happen here - this is not correct //this does not specify ownership - this is bad practice
            }
    };

}

#endif