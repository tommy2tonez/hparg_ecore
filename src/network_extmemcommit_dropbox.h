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

    //well we need, we HAVE TO use signal_smph_tile to aggregate roughly 64.000 tiles, worth of at least 1 GB of transfer/ one gatling gun mailchimp
    //we'd deliver that via our wrapped_dropbox which would further split that do the waitable size -> dropbox -> warehouse -> dispatcher -> get authenticated (cache + batch handshake, this is another request, smh) -> trinity requestor

    //this is roughly 10000x faster request, simply by waiting concurrently many guys, we are to hit the worst wait time of max(arr) instead of latency(e) + latency(e1) + ...
    //with all the benefits of requests, literally ...

    //best yet, the chance of request not getting responses is close to 0, one every exabytes to due God glitch

    //what is wrong???
    //our resolutor guy is busing the memory very heavily + waiting the request to be callbacked
    //we are not managing the threads correctly

    //we need to have a guy waiting for the promise, and a guy busing memory to make the request
    //we'd need to offload the responsibility RIGHT THERE
    //so there are guys that are dedicated to waiting and there are guys that are dedicated to busing memory
    //when I'm telling you this is hard, it is hard

    //we need to have somewhat a detached cyclic flow

    //we need to actually build a prioritized synchronization system
    //to actually synchronize in a correct order
    //this is important because we only have 2-3 synchronizer worker, when we are waiting for a guy, we are waiting for ... next guys
    //we need to cap the waiting job size (to not explode the global memory consumption and return error as soon as possible) 

    //

    struct Request{
        Address requestee;
        dg::network_extmemcommit_model::poly_event_t poly_event;
        uint8_t retry_count;
        bool has_request_unique_request_id;
        std::chrono::nanoseconds timeout;
        std::unique_ptr<dg::network_exception::ExceptionHandlerInterface> exception_handler; //note that exception retuned by this does not guarantee that the server does not commit the request
                                                                                             //this only tells that we have not gotten an explicit response from the server telling that the request was thru
                                                                                             //the has_unique_request_id tells us to only call the application ONCE, to avoid certain overriden requests

        std::optional<dg::network_rest::request_id_t> internal_request_id;
    };

    struct AuthorizedRequest{
        Request request;
        std::string token;
    };

    struct Token{
        dg::string token;
        std::chrono::time_point<std::chrono::utc_clock> expiry;
    };

    class DropBoxInterface{

        public:

            virtual ~DropBoxInterface() noexcept = default;
            virtual void drop(std::move_iterator<Request *> request_arr, size_t sz) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class SynchronizableInterface{

        public:

            virtual ~SynchronizableInterface() noexcept = default;
            virtual void sync() noexcept = 0;
    };

    class WareHouseInterface{

        public:

            virtual ~WareHouseInterface() noexcept = default;
            virtual auto push(dg::vector<Request>&& request_vec) noexcept -> exception_t = 0;
            virtual auto pop() noexcept -> dg::vector<Request> = 0;
    };

    class SynchronizableWareHouseInterface{

        public:

            virtual ~SynchronizableWareHouseInterface() noexcept = default;
            virtual auto push(std::unique_ptr<SynchronizableInterface>&&, std::chrono::nanoseconds max_sync_duration) noexcept -> exception_t = 0;
            virtual auto pop() noexcept -> std::unique_ptr<SynchronizableInterface> = 0;
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
    //alright, that was a mistake, I've come to realize that a cyclic request + dedicated thread to wait the synchronizable is actually the best possible approach
    //if we don't cap the request global memory usage, we'll be like in the rocket with no cap in The Martian
    //this is actually hard to implement, very hard

    //

    class TrinityRequestor: public virtual RequestorInterface{

        private:

            size_t requestid_feed_vectorization_sz;
            size_t request_feed_vectorization_sz;
            std::shared_ptr<SynchronizableWareHouseInterface> syncable_warehouse;
            std::shared_ptr<WareHouseInterface> request_warehouse;
            size_t max_retry_count; 

        public:

            static inline constexpr size_t MAX_REQUEST_RETRY_COUNT = 3u; 

            TrinityRequestor(size_t requestid_feed_vectorization_sz,
                             size_t request_feed_vectorization_sz,
                             std::shared_ptr<SynchronizableWareHouseInterface> syncable_warehouse,
                             std::shared_ptr<WareHouseInterface> request_warehouse,
                             size_t max_retry_count) noexcept: requestid_feed_vectorization_sz(requestid_feed_vectorization_sz),
                                                               request_feed_vectorization_sz(request_feed_vectorization_sz),
                                                               syncable_warehouse(std::move(syncable_warehouse)),
                                                               request_warehouse(std::move(request_warehouse)),
                                                               max_retry_count(max_retry_count){}

            void request(std::move_iterator<AuthorizedRequest *> authorized_request_arr, size_t sz) noexcept{

                AuthorizedRequest * base_authorized_request_arr                             = authorized_request_arr.base();
                std::expected<dg::vector<exception_t>, exception_t> dedicated_id_err_vec    = dg::network_exception::cstyle_initialize<dg::vector<exception_t>>(sz);

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
                    if (base_authorized_request_arr[i].request.retry_count > this->max_retry_count){
                        if (base_authorized_request_arr[i].request.exception_handler != nullptr){
                            base_authorized_request_arr[i].request.exception_handler->update(dg::network_exception::DROPBOX_REQUEST_BAD_RETRY_SIZE);
                        }

                        continue;
                    }

                    base_authorized_request_arr[valid_request_sz++] = std::move(base_authorized_request_arr[i]);
                }

                dedicated_id_err_vec->resize(valid_request_sz);

                {
                    auto feed_resolutor             = DedicatedIDFeedResolutor{};

                    size_t trimmed_feed_sz          = std::min(std::min(this->requestid_feed_vectorization_sz, dg::network_rest::max_request_id_size()), valid_request_sz);
                    size_t feeder_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_feed_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> feed_mem(feeder_allocation_cost);
                    auto feeder                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_feed_sz, feed_mem.get()));

                    std::fill(dedicated_id_err_vec->begin(), dedicated_id_err_vec->end(), dg::network_exception::SUCCESS);

                    for (size_t i = 0u; i < valid_request_sz; ++i){
                        if (base_authorized_request_arr[i].request.has_request_unique_request_id && !base_authorized_request_arr[i].request.internal_request_id.has_value()){
                            Request * request_ptr           = &base_authorized_request_arr[i].request;
                            exception_t * exception_ptr     = std::next(dedicated_id_err_vec->data(), i);
                            auto feed_arg                   = DedicatedIDFeedResolutorArgument{.fetching_request_ptr    = request_ptr,
                                                                                               .exception_ptr           = exception_ptr};

                            dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), feed_arg);
                        }
                    }
                }

                {
                    auto feed_resolutor                     = InternalResolutor{};
                    feed_resolutor.synchronizable_warehouse = this->syncable_warehouse;
                    feed_resolutor.request_warehouse        = this->request_warehouse;

                    size_t trimmed_feed_sz                  = std::min(std::min(this->request_feed_vectorization_sz, dg::network_rest::max_request_size()), valid_request_sz);
                    size_t feeder_allocation_cost           = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_feed_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                    auto feeder                             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_feed_sz, feeder_mem.get()));

                    for (size_t i = 0u; i < valid_request_sz; ++i){
                        if (dg::network_exception::is_failed(dedicated_id_err_vec.value()[i])){
                            if (base_authorized_request_arr[i].request.exception_handler != nullptr){
                                base_authorized_request_arr[i].request.exception_handler->update(dedicated_id_err_vec.value()[i]);
                            }

                            continue;
                        }

                        auto key = base_authorized_request_arr[i].request.requestee;
                        dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), key, std::move(base_authorized_request_arr[i]));
                    }
                }
            }

        private:

            struct DedicatedIDFeedResolutorArgument{
                Request * fetching_request_ptr;
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

                        base_data_arr[i].fetching_request_ptr->internal_request_id  = request_id_arr[i];
                        *base_data_arr[i].exception_ptr                             = dg::network_exception::SUCCESS; 
                    }
                }
            };

            struct InternalSynchronizer: virtual SynchronizableInterface{

                private:

                    dg::vector<Request> request_vec;
                    std::unique_ptr<dg::network_rest::Promise<dg::vector<dg::network_rest::ExternalMemcommitResponse>>> promise;
                    std::shared_ptr<WareHouseInterface> request_warehouse;
                    bool was_sync;

                public:

                    InternalSynchronizer(dg::vector<Request> request_vec,
                                         std::unique_ptr<dg::network_rest::Promise<dg::vector<dg::network_rest::ExternalMemcommitResponse>>> promise,
                                         std::shared_ptr<WareHouseInterface> request_warehouse) noexcept: request_vec(std::move(request_vec)),
                                                                                                          promise(std::move(promise)),
                                                                                                          request_warehouse(std::move(request_warehouse)),
                                                                                                          was_sync(false){}
                    
                    ~InternalSynchronizer() noexcept{

                        this->sync();
                    }

                    void sync() noexcept{

                        if constexpr(DEBUG_MODE_FLAG){
                            if (this->was_sync){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            }

                            if (this->promise == nullptr){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            }
                        }

                        dg::vector<dg::network_rest::ExternalMemcommitResponse> response_vec = this->promise->get();

                        if constexpr(DEBUG_MODE_FLAG){
                            if (response_vec.size() != this->request_vec.size()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            }
                        }

                        size_t sz           = this->request_vec.size();
                        size_t retriable_sz = 0u; 

                        for (size_t i = 0u; i < sz; ++i){
                            if (dg::network_exception::is_failed(response_vec[i].server_err_code)){
                                if (dg::network_rest::is_retriable_error(response_vec[i].server_err_code)){
                                    if (this->request_vec[i].retry_count == 0u){
                                        if (this->request_vec[i].exception_handler != nullptr){
                                            this->request_vec[i].exception_handler->update(dg::network_exception::DROPBOX_REQUEST_MAX_RETRY_REACHED); //buff from server_err_cde -> this code
                                        }
                                    } else{
                                        this->request_vec[i].retry_count    -= 1u;
                                        this->request_vec[retriable_sz++]   = std::move(this->request_vec[i]);
                                    }
                                } else{
                                    if (this->request_vec[i].exception_handler != nullptr){
                                        this->request_vec[i].exception_handler->update(response_vec[i].server_err_code);
                                    }
                                }

                                continue;
                            }

                            if (dg::network_exception::is_failed(response_vec[i].base_err_code)){
                                if (this->request_vec[i].exception_handler != nullptr){
                                    this->request_vec[i].exception_handler->update(rest_response_vec.value()[i].base_err_code);
                                }

                                //we got an exception from the base, we know this radixes as not retriable

                                continue;
                            }

                            if (this->request_vec[i].exception_handler != nullptr){
                                this->request_vec[i].exception_handler->update(dg::network_exception::SUCCESS);
                            }

                            //we are thru, we are to notify that this was thru
                        }

                        this->request_vec.resize(retriable_sz);

                        if (!this->request_vec.empty()){
                            exception_t err = this->request_warehouse->push(std::move(this->request_vec));

                            if (dg::network_exception::is_failed(err)){
                                for (size_t i = 0u; i < retriable_sz; ++i){
                                    if (this->request_vec[i].exception_handler != nullptr){
                                        this->request_vec[i].exception_handler->update(err);
                                    }
                                }
                            }
                        }

                        this->was_sync = true;
                    }

                    void release_and_notify(exception_t err) noexcept{

                        for (const Request& request: this->request_vec){
                            request.exception_handler->update(err);
                        }

                        this->was_sync = true;
                        this->request_vec.clear();
                    }
            };

            struct InternalResolutor: public virtual dg::network_producer_consumer::KVConsumerInterface<Address, DedicatedAuthorizedRequest>{

                std::shared_ptr<SynchronizableWareHouseInterface> synchronizable_warehouse;
                std::shared_ptr<WareHouseInterface> request_warehouse;

                //the reason we'd want to do this eventloop is because we cant guarantee that our token duration is good enough
                //and we can't force bussing threads to wait
                //bussing threads should be for bussing memory only

                //dedicated wait threads are very cheap, we can spawn 1024 concurrent waiting threads just to synchronize our requests, that's totally fine

                auto make_response_synchronizable(dg::vector<Request>&& request_vec,
                                                  dg::vector<std::unique_ptr<dg::network_exception::ExceptionHandlerInterface>>&& handler_vec,
                                                  std::unique_ptr<dg::network_rest::Promise<dg::vector<dg::network_rest::ExternalMemcommitResponse>>>&& promise) noexcept -> std::expected<std::unique_ptr<InternalSynchronizer>, exception_t>{

                }

                auto to_base_request_vec(const DedicatedAuthorizedRequest * inp_arr, size_t sz) noexcept -> std::expected<dg::vector<Request>, exception_t>{

                }

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

                    std::expected<dg::vector<Request>, exception_t> base_request_vec = this->to_base_request_vec(auth_request_vec_base, sz);

                    if (!base_request_vec.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (auth_request_vec_base[i].request.request.exception_handler != nullptr){
                                auth_request_vec_base[i].request.request.exception_handler->update(base_request_vec.error());
                            }
                        }

                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_rest::ExternalMemcommitRequest rest_request{.requestee  = auth_request_vec_base[i].request.request.requestee,
                                                                                .requestor  = dg::network_ip_data::host_addr(),
                                                                                .timeout    = auth_request_vec_base[i].request.request.timeout,
                                                                                .payload    = std::move(auth_request_vec_base[i].request.request.poly_event), //
                                                                                .token      = std::move(auth_request_vec_base[i].request.token),
                                                                                .request_id = auth_request_vec_base[i].request_id}; //

                        static_assert(std::is_nothrow_move_assignable_v<dg::network_rest::ExternalMemcommitRequest>);
                        rest_request_vec.value()[i] = std::move(rest_request);
                    }

                    std::expected<std::unique_ptr<dg::network_rest::Promise<dg::vector<dg::network_rest::ExternalMemcommitResponse>>>, exception_t> rest_response_vec = dg::network_rest::requestmany_extnmemcommit(dg::network_rest::get_normal_rest_controller(), rest_request_vec.value());

                    if (!rest_response_vec.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (auth_request_vec_base[i].request.request.exception_handler != nullptr){
                                auth_request_vec_base[i].request.request.exception_handler->update(rest_response_vec.error());
                            }
                        }

                        return;
                    }

                    std::expected<dg::vector<std::unique_ptr<ExceptionHandlerInterface>>> exception_handler_vec = dg::network_exception::cstyle_initialize<dg::vector<std::unique_ptr<ExceptionHandlerInterface>>>(sz);

                    if (!exception_handler_vec.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (auth_request_vec_base[i].request.request.exception_handler != nullptr){
                                auth_request_vec_base[i].request.request.exception_handler->update(exception_handler_vec.error());
                            }
                        }

                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        exception_handler_vec.value()[i] = std::move(auth_request_vec_base[i].request.request.exception_handler);
                    }

                    std::expected<std::unique_ptr<InternalSynchronizer>, exception_t> synchronizable = this->make_response_synchronizable(std::move(base_request_vec.value()),
                                                                                                                                          std::move(exception_handler_vec.value()),
                                                                                                                                          std::move(rest_response_vec.value()));

                    if (!synchronizable.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (exception_handler.value()[i] != nullptr){
                                exception_handler.value()[i]->update(synchronizable.error());
                            }
                        }

                        return;
                    }

                    exception_t err = this->synchronizable_warehouse->push(std::move(synchronizable.value()));

                    if (dg::network_exception::is_failed(err)){
                        synchronizable.value()->release_and_notify(err);
                    }
                }
            };
    };

    class SynchronizerWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<SynchronizableWareHouseInterface> warehouse;
        
        public:

            SynchronizerWorker(std::shared_ptr<SynchronizableWareHouseInterface> warehouse) noexcept: warehouse(std::move(warehouse)){}

            bool run_one_epoch() noexcept{

                std::unique_ptr<SynchronizableInterface> syncable = this->warehouse->pop();
                syncable->sync();

                return true;
            }
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

    class DropBox: public virtual DropBoxInterface{

        private:

            std::shared_ptr<WareHouseInterface> warehouse;
            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            size_t consume_sz_per_load; 

        public:

            DropBox(std::shared_ptr<WareHouseInterface> warehouse,
                    dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                    size_t consume_sz_per_load) noexcept: warehouse(std::move(warehouse)),
                                                          daemon_vec(std::move(daemon_vec)),
                                                          consume_sz_per_load(consume_sz_per_load){}

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

            auto max_consume_size() noexcept -> size_t{

                return this->consume_sz_per_load;
            }
    };

    class WrappedDropBoxConsumer: public virtual dg::network_producer_consumer::ConsumerInterface<Request>{

        private:

            std::shared_ptr<DropBoxInterface> dropbox;
            size_t feed_vectorization_sz; 

        public:

            WrappedDropBoxConsumer(std::shared_ptr<DropBoxInterface> dropbox,
                                   size_t feed_vectorization_sz) noexcept: dropbox(std::move(dropbox)),
                                                                           feed_vectorization_sz(feed_vectorization_sz){}

            void push(std::move_iterator<Request *> request_arr, size_t sz) noexcept{

                Request * base_request_arr      = request_arr.base();

                auto internal_resolutor         = InternalResolutor{};
                internal_resolutor.dropbox      = this->dropbox.get();

                size_t trimmed_feed_sz          = std::min(std::min(sz, this->feed_vectorization_sz), this->dropbox->max_consume_size());
                size_t feeder_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(&internal_resolutor, trimmed_feed_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&internal_resolutor, trimmed_feed_sz, feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), std::move(base_request_arr[i]));
                }
            }
        
        private:    
            
            struct InternalResolutor: dg::network_producer_consumer::ConsumerInterface<Request>{

                DropBoxInterface * dropbox;

                void push(std::move_iterator<Request *> request_arr, size_t request_arr_sz) noexcept{
                    dropbox->drop(request_arr, request_arr_sz);
                }
            };
    };
}

#endif