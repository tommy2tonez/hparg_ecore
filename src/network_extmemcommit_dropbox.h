#ifndef __NETWORK_EXTERNAL_MEMCOMMIT_DROPBOX_H__
#define __NETWORK_EXTERNAL_MEMCOMMIT_DROPBOX_H__

#include "network_extmemcommit_model.h"
#include "network_producer_consumer.h"
#include "network_concurrency.h"
#include "network_std_container.h"
#include "network_type_trait_x.h"
#include "stdx.h"

namespace dg::network_extmemcommit_dropbox{

    //problem is that we can't use shared_ptr for these requests, so we'd have to use std::move_iterator<> all the times
    //the truth is that 99% of the time, the machine spends time to move large buffer arounds, we dont want to fall into that trap
    //it seems very bad to use the move iterator logics, but trust me, that's our biggest savior, from delvrsrv to these unique_ptr moving to etc. 
    //in this component, we'd want to do a handshake + authentication by using token
    //we are to make sure that the client has approved our request storm before doing actual requests
    //this component looks very minimalistic yet sufficient to do most of the request logics, namely the famous re-request that would be the back back bone of our computation tree

    struct Request{
        Address requestee;
        dg::network_extmemcommit_model::poly_event_t poly_event;
        uint8_t retry_count;
        bool has_unique_request_id;
        std::chrono::nanoseconds timeout;
        std::unique_ptr<dg::network_exception::ExceptionHandlerInterface> exception_handler; //note that exception retuned by this does not guarantee that the server does not commit the request
                                                                                             //this only tells that we have not gotten an explicit response from the server telling that the request was thru
                                                                                             //the has_unique_request_id tells us to only call the application ONCE, to avoid certain overriden requests
    };

    struct AuthorizedRequest{
        Request request;
        std::string token;
    };

    struct DedicatedAuthorizedRequest{
        AuthorizedRequest request;
        std::optional<dg::network_rest::request_id_t> request_id;
    };

    struct Token{
        dg::string token;
        std::chrono::time_point<std::chrono::utc_clock> expiry;
    };

    class WareHouseInterface{

        public:

            virtual ~WareHouseInterface() noexcept = default;
            virtual auto push(dg::vector<Request>&& request_vec) noexcept -> exception_t = 0;
            virtual auto pop() noexcept -> dg::vector<Request> = 0;
    };

    class TokenControllerInterface{

        public:

            virtual ~TokenControllerInterface() noexcept = default;
            virtual void set_token(const Address * const Token *, size_t, exception_t *) noexcept = 0;
            virtual void get_token(const Address *, std::optional<Token> *) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class TokenRequestorInterface{

        public:

            virtual ~TokenRequestorInterface() noexcept = default;
            virtual void request_token(const Address *, std::expected<Token, exception_t> *)  noexcept = 0;
    };

    class RequestAuthorizerInterface{

        public:

            virtual ~RequestAuthorizerInterface() noexcept = default;
            virtual void authorize_request(std::move_iterator<Request *>, size_t, std::expected<AuthorizedRequest, exception_t> *) noexcept = 0;
    };

    class RequestorInterface{

        public:

            virtual ~RequestorInterface() noexcept = default;
            virtual void request(std::move_iterator<AuthorizedRequest *>, size_t) noexcept = 0;
    };

    class WareHouse: public virtual WareHouseInterface{

    };

    class TokenController: public virtual TokenControllerInterface{

        private:

            dg::unordered_unstable_map<Address, Token> token_map;
            size_t map_capacity;

        public:


    };

    class RequestAuthorizer: public virtual RequestAuthorizerInterface{

        private:

            std::unique_ptr<TokenRequestorInterface> token_requestor;
            std::unique_ptr<TokenControllerInterface> token_controller;
            size_t max_retry_count;
            size_t max_timeout;
            std::chrono::nanoseconds token_expiry_window;
        
        public:
            
            RequestAuthorizer(std::unique_ptr<TokenRequestorInterface> token_requestor,
                              std::unique_ptr<TokenControllerInterface> token_controller,
                              size_t max_retry_count,
                              size_t max_timeout,
                              std::chrono::nanoseconds token_expiry_window) noexcept: token_requestor(std::move(token_requestor)),
                                                                                      token_controller(std::move(token_controller)),
                                                                                      max_retry_count(max_retry_count),
                                                                                      max_timeout(max_timeout),
                                                                                      token_expiry_window(token_expiry_window){}

            void authorize_request(std::move_iterator<Request *> request_arr, size_t request_arr_sz, std::expected<AuthorizedRequest, exception_t> * output_arr) noexcept{

            }
    };

    //this component is, contrary to my beliefs, very hard to write
    //we need to be able to do re-request, basic prioritzed re-request by using stack feeders
    //we'd keep the number of retriable -> 3 for now, we can't do a stack guard for vector + friends, it's hard
    //we hardly ever would want to do more than 4 requests, look at the statistical chances, 10 mailbox retry * 4 request retry == 40 transmissions

    //it would take roughly 1 exabyte of data to get 1 lost packet, if we are CRONING it correctly

    //every 1 MB == 1 lost packet
    //the chance of 40 continuous failed transmissions == (1 / 1000) ** 40

    //the success chance of 40 continuous transmissions == 1 - (1 / 1000) ** 40

    //the success chance of 1 million packets = (1 - (1 / 1000) ** 40) ** (10 ** 6) 

    class TrinityRequestor: public virtual RequestorInterface{

        private:

            size_t requestid_feed_vectorization_sz;
            size_t request_feed_vectorization_sz;
            // size_t request2_feed_vectorization_sz; //we probably want to increase the buffer to allow certain leeways, 2 request 1 == 1 request 2, etc.
            // size_t request3_feed_vectorization_sz; 

        public:

            static inline constexpr size_t MAX_REQUEST_RETRY_COUNT = 3u; 

            TrinityRequestor(size_t requestid_feed_vectorization_sz,
                             size_t request_feed_vectorization_sz) noexcept: requestid_feed_vectorization_sz(requestid_feed_vectorization_sz),
                                                                             request_feed_vectorization_sz(request_feed_vectorization_sz){}

            void request(std::move_iterator<AuthorizedRequest *> authorized_request_arr, size_t sz) noexcept{

                AuthorizedRequest * base_authorized_request_arr = authorized_request_arr.base();

                // auto rest_payload = dg::network_rest::ExternalMemcommitRequest{.requestee   = };
                //we'll attempt to get the unique_request_id for the requests that has_unique_request_id
                //we'll try to move the authroized request from one place to another via the * next_retriable_handler
                //because the logic of request is that, we can only do it efficiently by a handful of number, and the first requests should be prioritized over the second or the third requests, etc. 
                //the network_rest responsibility is to make request, timeout, response, get_unique_request_id, we'll tackle the retriable + friends here
                //it's very super complicated to write these guys, so ...

                //it's complicated, we need to keep the order, yet we need to fetch the dedciated ID in a batching fashion
                //we need to move from one feed -> another feed of lower priority by bouncing in the delvrsrv_kv_deliver()

                std::expected<dg::vector<DedicatedAuthorizedRequest>, exception_t> dedicated_authorized_request_vec = dg::network_exception::cstyle_initialize<dg::vector<DedicatedAuthorizedRequest>>(sz);

                if (!dedicated_authorized_request_vec.has_value()){
                    for (size_t i = 0u; i < sz; ++i){
                        if (base_authorized_request_arr[i].request.exception_handler != nullptr){
                            base_authorized_request_arr[i].request.exception_handler->update(dedicated_authorized_request_vec.error());
                        }
                    }

                    return;
                }

                std::expected<dg::vector<exception_t>, exception_t> dedicated_id_err_vec = dg::network_exception::cstyle_initialize<dg::vector<exception_t>>(sz);

                if (!dedicated_id_err_vec.has_value()){
                    for (size_t i = 0u; i < sz; ++i){
                        if (base_authorized_request_arr[i].request.exception_handler != nullptr){
                            base_authorized_request_arr[i].request.exception_handler->update(dedicated_id_err_vec.error());
                        }
                    }

                    return;
                }

                size_t valid_request_sz = 0u;

                for (size_t i = 0u; i < sz; ++i){
                    if (base_authorized_request_arr[i].request.retry_count > MAX_REQUEST_RETRY_COUNT){
                        if (base_authorized_request_arr[i].request.exception_handler != nullptr){
                            base_authorized_request_arr[i].request.exception_handler->update(dg::network_exception::DROPBOX_REQUEST_BAD_RETRY_SIZE);
                        }

                        continue;
                    }

                    dedicated_authorized_request_vec.value()[valid_request_sz].request      = std::move(base_authorized_request_arr[i]);
                    dedicated_authorized_request_vec.value()[valid_request_sz].request_id   = std::nullopt;
                    valid_request_sz                                                        += 1u;
                }

                dedicated_authorized_request_vec->resize(valid_request_sz);
                dedicated_id_err_vec->resize(valid_request_sz);

                {
                    auto feed_resolutor             = DedicatedIDFeedResolutor{};

                    size_t trimmed_feed_sz          = std::min(std::min(this->requestid_feed_vectorization_sz, dg::network_rest::max_request_id_size()), valid_request_sz);
                    size_t feeder_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_feed_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> feed_mem(feeder_allocation_cost);
                    auto feeder                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_feed_sz, feed_mem.get()));

                    std::fill(dedicated_id_err_vec->begin(), dedicated_id_err_vec->end(), dg::network_exception::SUCCESS);

                    for (size_t i = 0u; i < valid_request_sz; ++i){
                        if (dedicated_authorized_request_vec.value()[i].request.request.has_unique_request_id){
                            DedicatedAuthorizedRequest * request_ptr    = std::next(dedicated_authorized_request_vec->data(), i);
                            exception_t * exception_ptr                 = std::next(dedicated_id_err_vec->data(), i);
                            auto feed_arg                               = DedicatedIDFeedResolutorArgument{.fetching_request_ptr    = request_ptr,
                                                                                                           .exception_ptr           = exception_ptr};
                            
                            dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), feed_arg);
                        }
                    }
                }

                {
                    auto feed0_resolutor                    = InternalResolutor{};
                    feed0_resolutor.next_retriable_handler  = nullptr;

                    size_t trimmed_feed0_sz                 = std::min(std::min(this->request_feed_vectorization_sz, dg::network_rest::max_request_size()), valid_request_sz);
                    size_t feeder0_allocation_cost          = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed0_resolutor, trimmed_feed0_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder0_mem(feeder0_allocation_cost);
                    auto feeder0                            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed0_resolutor, trimmed_feed0_sz, feeder0_mem.get()));

                    auto feed1_resolutor                    = InternalResolutor{};
                    feed1_resolutor.next_retriable_handler  = feeder0.get();

                    size_t trimmed_feed1_sz                 = std::min(std::min(this->request_feed_vectorization_sz, dg::network_rest::max_request_size()), valid_request_sz);
                    size_t feeder1_allocation_cost          = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed1_resolutor, trimmed_feed1_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder1_mem(feeder1_allocation_cost);
                    auto feeder1                            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed1_resolutor, trimmed_feed1_sz, feeder1_mem.get()));

                    auto feed2_resolutor                    = InternalResolutor{};
                    feed2_resolutor.next_retriable_handler  = feeder1.get();

                    size_t trimmed_feed2_sz                 = std::min(std::min(this->request_feed_vectorization_sz, dg::network_rest::max_request_size()), valid_request_sz);
                    size_t feeder2_allocation_cost          = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed2_resolutor, trimmed_feed2_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder2_mem(feeder2_allocation_cost);
                    auto feeder2                            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed2_resolutor, trimmed_feed2_sz, feeder2_mem.get()));

                    size_t feed3_resolutor                  = InternalResolutor{};
                    feed3_resolutor.next_retriable_handler  = feeder2.get();

                    size_t trimmed_feed3_sz                 = std::min(std::min(this->request_feed_vectorization_sz, dg::network_rest::max_request_size()), valid_request_sz);
                    size_t feeder3_allocation_cost          = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed3_resolutor, trimmed_feed3_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder3_mem(feeder3_allocation_cost);
                    auto feeder3                            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed3_resolutor, trimmed_feed3_sz, feeder3_mem.get())); 

                    for (size_t i = 0u; i < valid_request_sz; ++i){
                        if (dg::network_exception::is_failed(dedicated_id_err_vec.value()[i])){
                            if (dedicated_authorized_request_vec.value()[i].request.request.exception_handler != nullptr){
                                dedicated_authorized_request_vec.value()[i].request.request.exception_handler->update(dedicated_id_err_vec.value()[i]);
                            }

                            continue;
                        }

                        auto key = dedicated_authorized_request_vec.value()[i].request.request.requestee;
                        dg::network_producer_consumer::delvrsrv_kv_deliver(feeder3.get(), key, std::move(dedicated_authorized_request_vec.value()[i]));
                    }
                }
            }

        private:

            struct DedicatedIDFeedResolutorArgument{
                DedicatedAuthorizedRequest * fetching_request_ptr;
                exception_t * exception_ptr;
            };

            struct DedicatedIDFeedResolutor: dg::network_producer_consumer::ConsumerInterface<DedicatedIDFeedResolutorArgument>{

                void push(std::move_iterator<DedicatedIDFeedResolutorArgument *> data_arr, size_t sz) noexcept{

                    DedicatedIDFeedResolutorArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<dg::network_rest::request_id_t[]> request_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    dg::network_rest::get_dedicated_request_id(request_id_arr.get(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            *base_data_arr[i].exception_ptr = exception_arr[i];
                            continue;
                        }

                        base_data_arr[i].fetching_request_ptr->request_id   = request_id_arr[i];
                        *base_data_arr[i].exception_ptr                     = dg::network_exception::SUCCESS; 
                    }
                }
            };

            struct InternalResolutor: public virtual dg::network_producer_consumer::KVConsumerInterface<Address, DedicatedAuthorizedRequest>{

                dg::network_producer_consumer::KVDeliveryHandle<Address, DedicatedAuthorizedRequest> * next_retriable_handler; 

                void push(const Address& address, std::move_iterator<DedicatedAuthorizedRequest *> auth_request_vec, size_t sz) noexcept{

                    DedicatedAuthorizedRequest * auth_request_vec_base = auth_request_vec.base(); 

                    std::expected<dg::vector<dg::network_rest::ExternalMemcommitRequest>, exception_t> rest_request_vec = dg::network_exception::cstyle_initialize<dg::vector<dg::network_rest::ExternalMemcommitRequest>>(sz);

                    if (!rest_request_vec.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (auth_request_vec_base[i].request.request.exception_handler != nullptr){
                                auth_request_vec_base[i].request.request.exception_handler->update(rest_request_vec.error());
                            }
                        }

                        return;
                    }

                    if (!rest_response_vec.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (auth_request_vec_base[i].request.request.exception_handler != nullptr){
                                auth_request_vec_base[i].request.request.exception_handler->update(rest_response_vec.error());
                            }
                        }

                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_rest::ExternalMemcommitRequest rest_request{.requestee  = auth_request_vec_base[i].request.request.requestee,
                                                                                .requestor  = dg::network_ip_data::host_addr(),
                                                                                .timeout    = auth_request_vec_base[i].request.request.timeout,
                                                                                .payload    = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_extmemcommit_model::poly_event_t>(auth_request_vec_base[i].request.request.poly_event)), //
                                                                                .token      = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::string>(auth_request_vec_base[i].request.token)),
                                                                                .request_id = auth_request_vec_base[i].request_id}; //

                        static_assert(std::is_nothrow_move_assignable_v<dg::network_rest::ExternalMemcommitRequest>);
                        rest_request_vec.value()[i] = std::move(rest_request);
                    }

                    std::expected<dg::vector<dg::network_rest::ExternalMemcommitResponse>, exception_t> rest_response_vec = dg::network_rest::requestmany_extnmemcommit(dg::network_rest::get_normal_rest_controller(), rest_request_vec.value());

                    //we are dealing with three kinds of errors, the client errors, the server errors and the server application errors  
                    //client errors == std::expected<..., exception_t>, not retriable, we are having internal clients flood

                    //server errors == the REST framework of the server is overloaded, bad traffic, bad requests, bad authentication, etc.

                    //server explicit errors (explicit retriable because of etc. reasons, explicit cease to request because of etc. reasons)
                    //server implicit errors (timeout + friends)

                    //the application errors (the request was through, yet the application returns error, this is NOT the cachable properties, cachable means that the application response is unique, if the application has already responded, then we have crossed the protected line of the cachable)

                    if (!rest_response_vec.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (auth_request_vec_base[i].request.exception_handler != nullptr){
                                auth_request_vec_base[i].request.exception_handler->update(rest_response_vec.error());
                            }
                        }

                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(rest_response_vec.value()[i].server_err_code)){
                            if (dg::network_rest::is_retriable_error(rest_response_vec.value()[i].server_err_code)){
                                if (auth_request_vec_base[i].request.request.retry_count == 0u){
                                    if (auth_request_vec_base[i].request.request.exception_handler != nullptr){
                                        auth_request_vec_base[i].request.request.exception_handler->update(dg::network_exception::DROPBOX_REQUEST_MAX_RETRY_REACHED); //buff from server_err_cde -> this code
                                    }
                                } else{
                                    //we'd have to see if this could be through

                                    if (this->next_retriable_handler != nullptr){
                                        auth_request_vec_base[i].request.request.retry_count -= 1u;
                                        auto key = auth_request_vec_base[i].request.request.requestee;
                                        dg::network_producer_consumer::delvrsrv_kv_deliver(this->next_retriable_handler, key, std::move(auth_request_vec_base[i]));
                                    } else {

                                        //we are to notify that this could not be thru, we'd buff that to the REST_REQUEST_MAX_RETRY_REACHED

                                        if (auth_request_vec_base[i].request.request.exception_handler != nullptr){
                                            auth_request_vec_base[i].request.request.exception_handler->update(dg::network_exception::DROPBOX_REQUEST_MAX_RETRY_REACHED);
                                        }
                                    }
                                }
                            } else{
                                if (auth_request_vec_base[i].request.request.exception_handler != nullptr){
                                    auth_request_vec_base[i].request.request.exception_handler->update(rest_response_vec.value()[i].server_err_code);
                                }
                            }

                            continue;
                        }

                        if (dg::network_exception::is_failed(rest_response_vec.value()[i].base_err_code)){
                            if (auth_request_vec_base[i].request.request.exception_handler != nullptr){
                                auth_request_vec_base[i].request.request.exception_handler->update(rest_response_vec.value()[i].base_err_code);
                            }

                            //we got an exception from the base, we know this radixes as not retriable

                            continue;
                        }

                        if (auth_request_vec_base[i].request.request.exception_handler != nullptr){
                            auth_request_vec_base[i].request.request.exception_handler->update(dg::network_exception::SUCCESS);
                        }

                        //we are thru, we are to notify that this was thru
                    }
                }
            };
    };

    class RequestDispatcher: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<WareHouseInterface> warehouse;
            std::shared_ptr<RequestAuthorizerInterface> request_authorizer;
            std::shared_ptr<MemcommitRequestorInterface> memcommit_requestor;

        public:

            RequestDispatcher(std::shared_ptr<WareHouseInterface> warehouse,
                              std::shared_ptr<RequestAuthorizerInterface> request_authorizer,
                              std::shared_ptr<MemcommitRequestorInterface> memcommit_requestor) noexcept: warehouse(std::move(warehouse)),
                                                                                                          request_authorizer(std::move(request_authorizer)),
                                                                                                          memcommit_requestor(std::move(memcommit_requestor)){}

            bool run_one_epoch() noexcept{

                dg::vector<Request> request_vec = this->warehouse->pop();

                std::expected<dg::vector<std::expected<AuthorizedRequest, exception_t>>, exception_t> authorized_request_vec    = dg::network_exception::cstyle_initialize<dg::vector<std::expected<AuthorizedRequest, exception_t>>>(request_vec.size());
                std::expected<dg::vector<AuthorizedRequest>, exception_t> requesting_request_vec                                = dg::network_exception::cstyle_initialize<dg::vector<AuthorizedRequest>>(request_vec.size());
                size_t requesting_request_vec_sz                                                                                = 0u; 

                if (!authorized_request_vec.has_value()){
                    for (const Request& request: request_vec){
                        if (request.exception_handler != nullptr){
                            request.exception_handler->update(authorized_request_vec.error());
                        }
                    }

                    return true;
                }

                if (!requesting_request_vec.has_value()){
                    for (const Request& request: request_vec){
                        if (request.exception_handler != nullptr){
                            request.exception_handler->update(requesting_request_vec.error());
                        }
                    }

                    return true;
                }

                this->request_authroizer->authorize_request(std::make_move_iterator(request_vec.data()), request_vec.size(), authorized_request_vec.data());

                for (size_t i = 0u; i < request_vec.size(); ++i){
                    if (!authorized_request_vec[i].has_value()){
                        if (request_vec[i].exception_handler != nullptr){
                            request_vec[i].exception_handler->update(authorized_request_vec[i].error());
                        }

                        continue;
                    }

                    requesting_request_vec.value()[requesting_request_vec_sz++] = std::move(authorized_request_vec[i].value());
                }

                this->memcommit_requestor->request(std::make_move_iterator(requesting_request_vec.value().data()), requesting_request_vec_sz);

                return true;
            }
    };

    class DropBox: public virtual dg::network_producer_consumer::ConsumerInterface<Request>{

        private:

            std::shared_ptr<WareHouseInterface> warehouse;
            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;

        public:

            DropBox(std::shared_ptr<WareHouseInterface> warehouse,
                    dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec) noexcept: warehouse(std::move(warehouse)),
                                                                                                    daemon_vec(std::move(daemon_vec)){}

            void push(std::move_iterator<Request *> request_arr, size_t sz) noexcept{

                if (sz == 0u){
                    return;
                }

                Request * base_request_arr                                  = request_arr.base();
                std::expected<dg::vector<Request>, exception_t> request_vec = dg::network_exception::cstyle_initialize<dg::vector<Request>>(sz);

                if (!request_vec.has_value()){
                    for (size_t i = 0u; i < sz; ++i){
                        if (base_request_arr[i].exception_handler != nullptr){
                            base_request_arr[i].exception_handler->update(request_vec.error());
                        }
                    }

                    return;
                }

                static_assert(std::is_nothrow_move_assignable_v<Request>);
                std::copy(std::make_move_iterator(base_request_arr), std::make_move_iterator(std::next(base_request_arr, sz)), request_vec->begin());
                exception_t err = this->warehouse->push(std::move(request_vec.value()));

                if (dg::network_exception::is_failed(err)){
                    for (const Request& request: request_vec.value()){
                        if (request.exception_handler != nullptr){
                            request.exception_handler->update(err);
                        }
                    }
                }
            }
    };
}

#endif
