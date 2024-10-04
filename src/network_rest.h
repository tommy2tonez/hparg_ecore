#ifndef __DG_NETWORK_REST_H__
#define __DG_NETWORK_REST_H__ 

//this is somewhat still a REST protocol - yet leverage non-blocking communication to avoid thread starvation + friends
//this requires the requestee to be on the same communication platform - an abstraction is required on their side should popularization is required

namespace dg::network_rest{

    struct RestPayload{
        dg::network_std_container::string uri;
         
        Request request;
    };

    struct RequestHandlerInterface{
        virtual ~RequestHandlerInterface() noexcept = default;
        virtual auto handle(Request) noexcept -> Response = 0;
    };


    //on the user side - send via rest_request_channel + register request id -> unordered_map memory (precisely the cuda method)
    //use non-blocking synchronization hook - to wait for request_id to be confirmed by the rest_response_channel worker or timeouted
    //or - register mutex + blocking (not recommended)
    //registering mutex is a method of assigning worker_id -> thread mutex + unblock thread mutex when order is ready - I personally don't like that because its bad

    class RestInBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::unordered_map<dg::network_std_container::string, std::unique_ptr<RequestHandlerInterface>> rest_resolutor;
        
        public:
            
            RestInBoundWorker(std::unordered_map<dg::network_std_container::string, std::unique_ptr<RequestHandlerInterface>> rest_resolutor): rest_resolutor(std::move(rest_resolutor)){}

            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::string> recv_data = dg::network_kernel_mailbox::recv(dg::network_kernel_mailbox::REST_REQUEST_CHANNEL);

                if (!static_cast<bool>(recv_data)){
                    return false;
                }

                RestPayload rest_payload{};
                dg::network_compact_serializer::deserialize_into(rest_payload, recv_data->data());
                Response response   = this->rest_resolutor.find(rest_payload.uri)->second->handle(rest_payload.request);
                size_t response_sz  = dg::network_compact_serializer::size(response);
                auto bstream        = dg::network_std_container::string(response_sz);
                dg::network_compact_serializer::serialize_into(bstream.data(), response);
                dg::network_kernel_mailbox::send(get_requestee(rest_payload), std::move(bstream), dg::network_kernel_mailbox::REST_RESPONSE_CHANNEL);
            }
    };

    class RestManager{

        private:

            dg::network_std_container::vector<dg::network_concurrency::daemon_raiit_handle_t> daemons;
        
        public:

            RestManager(dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemons) noexcept: daemons(std::move(daemons)){}
    };

    void init(){

    }

    void deinit() noexcept{

    }
} 

#endif