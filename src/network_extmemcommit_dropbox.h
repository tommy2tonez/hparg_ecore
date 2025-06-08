#ifndef __NETWORK_EXTERNAL_MEMCOMMIT_DROPBOX_H__
#define __NETWORK_EXTERNAL_MEMCOMMIT_DROPBOX_H__

#include "network_extmemcommit_model.h"
#include "network_producer_consumer.h"
#include "network_concurrency.h"
#include "network_std_container.h"
#include "network_type_trait_x.h"
#include "stdx.h"

namespace dg::network_extmemcommit_dropbox{

    //alrights, we'll connect this to the network_rest
    //we'll take in Request Order, batch the requests, make the requests, and wait
    //we'll wont be using reactor this time but rather a dg::vector<Request> directly

    //what we'd attempt to do at this component is taking in request
    //authorize the request, keep track of token lifetime + token refresh virtues
    //connect to the network_rest interface

    //keep retrying the request until the timeout
    //if failed invoke the exception_handler

    //very complicated
    //we need to use our own preinstalled symmetric keys
    //such is every connected server on spawn should know each other symmetric encoding method 
    //whenever we are requesting a token, we expose our symmetric key by the token length
    //too complicated, we should have used sessions with asymmetric nonsense

    //the problem is that we'd want to try to hide the destination of the token
    //so the attacker could not use the exposed token along the line to make malicious requests
    //bounce request is a request to another server to do another request
    //it's sort of tors, yet we'd want to support the bounce request to hide our token destination

    //this is complicated to implement, we'll work on this for 2-3 days
    //essentially, we'd allow user to set user|password or persitent token of another p2p credentials via REST
    //we'd kind of register that to the RAM or the persistent database
    //we'd want to peek the value of user|password or persistent token to make that temporary token request
    //that temporary token is exposed to the traffic, and we don't need to worry about that token got into the wrong hand as much as we'd worry about the persistent token
    //we'd get that temp token via TokenControllerInterface

    //the TokenControllerInterface is a finite unordered_map, that use a fifo queue to fit the capacity
    //if the Address * exceeds the TokenControllerCapacity, we'd get the value via the first REST request, and that'd kind of get truncated after pushed to the TokenControllerInterface
    //we have an internal "virtues" of semanticalizing request + response by using struct + compact_serializer, unfortunately, we are not using JSON yet

    //we'd want to retry the request for retry_count time before dropping the request and invoke exception_handler

    //as for WareHouseInterface, we'd need to keep the "workload" under control to make sure that we are not flooding the warehouse
    //we'd return exception immediately to the user if the warehouse is not ExhaustionControlled and the warehouse has reached its designated efficient capacity
    //I think this should suffice for this component
    //we actually made the requests so fast that they are not even requests, but a direct socket communication

    //we are actually very proud of our request protocol, just a few twists here and there for the unordered_map hint + address and we should be good
    //the fact that we have made it this far is actually a miracle

    struct Request{
        Address requestee;
        dg::network_extmemcommit_model::poly_event_t poly_event;
        uint8_t retry_count;
        bool has_unique_request_id;
        std::chrono::nanoseconds timeout;
        std::unique_ptr<dg::network_exception::ExceptionHandlerInterface> exception_handler;
    };

    struct AuthorizedRequest{
        Request request;
        std::string token;
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

    class Requestor: public virtual RequestorInterface{

        private:

            size_t kvfeed_vectorization_sz;

        public:

            Requestor(size_t kvfeed_vectorization_sz) noexcept: kvfeed_vectorization_sz(kvfeed_vectorization_sz){}

            void request(std::move_iterator<AuthorizedRequest *> authorized_request_arr, size_t sz) noexcept{

                // auto rest_payload = dg::network_rest::ExternalMemcommitRequest{.requestee   = };
            }
        
        private:

            struct InternalResolutor: public virtual dg::network_producer_consumer::KVConsumerInterface<Address, AuthorizedRequest>{

                dg::network_producer_consumer::KVDeliveryHandle<Address, AuthorizedRequest> * next_retriable_handler; 

                void push(const Address& address, std::move_iterator<AuthorizedRequest *> auth_request_vec, size_t sz) noexcept{

                    AuthorizedRequest * auth_request_vec_base = auth_request_vec.base(); 

                    std::expected<dg::vector<dg::network_rest::ExternalMemcommitRequest>, exception_t> rest_request_vec = dg::network_exception::cstyle_initialize<dg::vector<dg::network_rest::ExternalMemcommitRequest>>(sz);

                    if (!rest_request_vec.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (auth_request_vec_base[i].request.exception_handler != nullptr){
                                auth_request_vec_base[i].request.exception_handler->update(rest_request_vec.error());
                            }
                        }

                        return;
                    }

                    if (!rest_response_vec.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (auth_request_vec_base[i].request.exception_handler != nullptr){
                                auth_request_vec_base[i].request.exception_handler->update(rest_response_vec.error());
                            }
                        }

                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_rest::ExternalMemcommitRequest rest_request{.requestee  = auth_request_vec_base[i].request.requestee,
                                                                                .requestor  = dg::network_ip_data::host_addr(),
                                                                                .timeout    = auth_request_vec_base[i].request.timeout,
                                                                                .payload    = std::move(auth_request_vec_base[i].request.poly_event), //
                                                                                .token      = std::move(auth_request_vec_base[i].token)}; //

                        static_assert(std::is_nothrow_move_assignable_v<dg::network_rest::ExternalMemcommitRequest>);
                        rest_request_vec.value()[i] = std::move(rest_request);
                    }

                    std::expected<dg::vector<dg::network_rest::ExternalMemcommitResponse>, exception_t> rest_response_vec = dg::network_rest::requestmany_extnmemcommit(dg::network_rest::get_normal_rest_controller(), rest_request_vec.value());

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

                            } else{

                            }

                            continue;
                        }

                        if (dg::network_exception::is_failed(rest_response_vec.value()[i].base_err_code)){
                            //we got an exception from the base, we know this radixes as not retriable

                            continue;
                        }
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