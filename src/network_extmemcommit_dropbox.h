#ifndef __NETWORK_EXTERNAL_MEMCOMMIT_DROPBOX_H__
#define __NETWORK_EXTERNAL_MEMCOMMIT_DROPBOX_H__

#include "network_extmemcommit_model.h"
#include "network_producer_consumer.h"
#include "network_concurrency.h"
#include "network_std_container.h"
#include "network_type_trait_x.h"
#include "stdx.h"

namespace dg::network_extmemcommit_dropbox{

    //we'll mainly use this to do connection handshakes, our request is actually normal request, very surprising
    //we'll provide a way for user to do NAT Punch + handshake by setting up some forward tiles as forehead before doing any actual heavy forwards + backwards

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
        std::optional<std::chrono::time_point<std::chrono::utc_clock>> expiry;
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
            virtual auto push(std::unique_ptr<SynchronizableInterface>&&, std::chrono::nanoseconds sync_duration) noexcept -> exception_t = 0;
            virtual auto pop() noexcept -> std::unique_ptr<SynchronizableInterface> = 0;
    };

    class TokenCacheControllerInterface{

        public:

            virtual ~TokenCacheControllerInterface() noexcept = default;
            virtual void set_token(const Address * const Token *, size_t, exception_t *) noexcept = 0;
            virtual void get_token(const Address *, size_t sz, std::expected<std::optional<Token>, exception_t> *) noexcept = 0;
            virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    class TokenRequestorInterface{

        public:

            virtual ~TokenRequestorInterface() noexcept = default;
            virtual void request_token(const Address *, size_t, std::expected<Token, exception_t> *)  noexcept = 0;
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

    //

    class WareHouse: public virtual WareHouseInterface{

        private:

            dg::pow2_cyclic_queue<dg::vector<Request>> request_vec_queue;
            dg::pow2_cyclic_queue<std::pair<std::optional<dg::vector<Request>> *, std::binary_semaphore *>> waiting_queue;
            size_t warehouse_population_sz;
            size_t warehouse_population_cap;
            std::unique_ptr<std::mutex> mtx;

        public:

            WareHouse(dg::pow2_cyclic_queue<dg::vector<Request>> request_vec_queue,
                      dg::pow2_cyclic_queue<std::pair<std::optional<dg::vector<Request>> *, std::binary_semaphore *>> waiting_queue,
                      size_t warehouse_population_sz,
                      size_t warehouse_population_cap,
                      std::unique_ptr<std::mutex> mtx) noexcept: request_vec_queue(std::move(request_vec_queue)),
                                                                 waiting_queue(std::move(waiting_queue)),
                                                                 warehouse_population_sz(warehouse_population_sz),
                                                                 warehouse_population_cap(warehouse_population_cap),
                                                                 mtx(std::move(mtx)){}

            auto push(dg::vector<Request>&& request_vec) noexcept -> exception_t{

                if (request_vec.empty()){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                std::binary_semaphore * releasing_smp = nullptr;
                exception_t err = [&]() noexcept{
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->waiting_queue.empty()){
                        auto [fetching_addr, smp]   = this->waiting_queue.front();
                        this->waiting_queue.pop_front();
                        *fetching_addr              = std::move(request_vec);
                        releasing_smp               = smp;

                        return dg::network_exception::SUCCESS;
                    }

                    if (this->request_vec_queue.size() == this->request_vec_queue.capacity()){
                        return dg::network_exception::RESOURCE_EXHAUSTION;
                    }

                    size_t new_warehouse_sz = this->warehouse_population_sz + request_vec.size(); 

                    if (new_warehouse_sz > this->warehouse_population_cap){
                        return dg::network_exception::RESOURCE_EXHAUSTION;
                    }

                    this->request_vec_queue.push_back(std::move(request_vec));
                    this->warehouse_population_sz = new_warehouse_sz;

                    return dg::network_exception::SUCCESS;
                }();

                if (releasing_smp != nullptr){
                    releasing_smp->release();
                }

                return return_err;
            }

            auto pop() noexcept -> dg::vector<Request>{

                std::binary_semaphore smp(0);
                std::optional<dg::vector<Request>> request;

                {
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->request_vec_queue.empty()){
                        auto rs = std::move(this->request_vec_queue.front());
                        this->request_vec_queue.pop_front();
                        return rs;
                    }

                    //we'll see about this

                    if constexpr(DEBUG_MODE_FLAG){
                        if (this->waiting_queue.size() == this->waiting_queue.capacity()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    this->waiting_queue.push_back(std::make_pair(&request, &smp));
                }

                smp.acquire();
                return dg::vector<Request>(std::move(request.value()));
            }
    };

    struct SynchronizableTemporalEntry{
        std::unique_ptr<SynchronizableInterface> synchronizable;
        std::chrono::time_point<std::chrono::steady_clock> abs_timeout;
    };

    class SynchronizableWareHouse: public virtual SynchronizableWareHouseInterface{

        private:

            dg::vector<SynchronizableTemporalEntry> priority_queue;
            dg::pow2_cyclic_queue<std::pair<std::unique_ptr<SynchronizableInterface> *, std::binary_semaphore *>> waiting_queue;
            size_t priority_queue_cap;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<std::chrono::nanoseconds> max_sync_duration;

        public:

            static inline constexpr greater_cmp = [](const SynchronizableTemporalEntry& lhs, const SynchronizableTemporalEntry& rhs) noexcept{
                return lhs.abs_timeout > rhs.abs_timeout;
            };

            SynchronizableWareHouse(dg::vector<SynchronizableTemporalEntry> priority_queue,
                                    dg::pow2_cyclic_queue<std::pair<std::unique_ptr<SynchronizableInterface> *, std::binary_semaphore *>> waiting_queue,
                                    size_t priority_queue_cap,
                                    std::unique_ptr<std::mutex> mtx,
                                    std::chrono::nanoseconds max_sync_duration) noexcept: priority_queue(std::move(priority_queue)),
                                                                                          waiting_queue(std::move(waiting_queue)),
                                                                                          priority_queue_cap(priority_queue_cap),
                                                                                          mtx(std::move(mtx)),
                                                                                          max_sync_duration(stdx::hdi_container<std::chrono::nanoseconds>{max_sync_duration}){}

            auto push(std::unique_ptr<SynchronizableInterface>&& synchronizable, std::chrono::nanoseconds sync_duration) noexcept -> exception_t{

                if (synchronizable == nullptr){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                if (sync_duration > this->max_sync_duration.value){
                    return dg::network_exception::INVALID_ARGUMENT;
                }

                std::binary_semaphore * releasing_smp = nullptr;

                exception_t return_err = [&, this]() noexcept{
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->waiting_queue.empty()){
                        auto [fetching_addr, smp]   = this->waiting_queue.front();
                        this->waiting_queue.pop_front();
                        *fetching_addr              = std::move(synchronizable);
                        releasing_smp               = smp;

                        return dg::network_exception::SUCCESS;
                    }

                    if (this->priority_queue.size() == this->priority_queue_cap){
                        return dg::network_exception::RESOURCE_EXHAUSTION;
                    }

                    auto inserting_entry            = SynchronizableTemporalEntry{.synchronizable   = std::move(synchronizable),
                                                                                  .abs_timeout      = std::chrono::steady_clock::now() + sync_duration};

                    this->priority_queue.push_back(std::move(inserting_entry));
                    std::push_heap(this->priority_queue.begin(), this->priority_queue.end(), greater_cmp);

                    return dg::network_exception::SUCCESS;
                }();

                if (releasing_smp != nullptr){
                    releasing_smp->release();
                }

                return return_err;
            }

            auto pop() noexcept -> std::unique_ptr<SynchronizableInterface>{

                std::binary_semaphore smp(0);
                std::unique_ptr<SynchronizableInterface> syncable; 

                {
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->priority_queue.empty()){
                        std::pop_heap(this->priority_queue.begin(), this->priority_queue.end(), greater_cmp);
                        auto rs = std::move(this->priority_queue.back());
                        this->priority_queue.pop_back();

                        return rs;
                    }

                    if constexpr(DEBUG_MODE_FLAG){
                        if (this->waiting_queue.size() == this->waiting_queue.capacity()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    this->waiting_queue.push_back(std::make_pair(&syncable, &smp));
                }

                smp.acquire();
                return syncable;
            }
    };

    class TokenFetcher: public virtual TokenRequestorInterface{

        private:

            std::chrono::nanoseconds auth2_request_latency;
            size_t feed_vectorization_sz;
            size_t auth2_feed_vectorization_sz;

        public:

            TokenFetcher(std::chrono::nanoseconds auth2_request_latency,
                         size_t feed_vectorization_sz,
                         size_t auth2_feed_vectorization_sz) noexcept: auth2_request_latency(auth2_request_latency),
                                                                       feed_vectorization_sz(feed_vectorization_sz),
                                                                       auth2_feed_vectorization_sz(auth2_feed_vectorization_sz){}

            void request_token(const Address * dst_arr, size_t sz, std::expected<Token, exception_t> * output_arr) noexcept{

                auto auth2_internal_resolutor               = Auth2FeedResolutor{};
                auth2_internal_resolutor.request_latency    = this->auth2_request_latency;

                size_t trimmed_auth2_feed_vectorization_sz  = std::min(std::min(this->auth2_feed_vectorization_sz, dg::network_rest::max_request_size()), sz);
                size_t auth2_feeder_allocation_cost         = dg::network_producer_consumer::delvrsrv_allocation_cost(&auth2_internal_resolutor, trimmed_auth2_feed_vectorization_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> auth2_feeder_mem(auth2_feeder_allocation_cost);
                auto auth2_feeder                           = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&auth2_internal_resolutor, trimmed_auth2_feed_vectorization_sz, auth2_feeder_mem.get()));

                //----

                auto internal_resolutor                     = InternalResolutor{};
                internal_resolutor.auth2_feeder             = auth2_feeder.get();

                size_t trimmed_feed_vectorization_sz        = std::min(std::min(this->feed_vectorization_sz, dg::network_postgres_db::optimal_batch_request_size()), sz);
                size_t feeder_allocation_cost               = dg::network_producer_consumer::delvrsrv_allocation_cost(&internal_resolutor, trimmed_feed_vectorization_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&internal_resolutor, trimmed_feed_vectorization_sz, feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<dg::string, exception_t> generic_address = dg::network_ip_data::to_generic_url_address(dst_arr[i]);

                    if (!generic_addr.has_value()){
                        output_arr[i] = std::unexpected(generic_address.error());
                        continue;
                    }

                    auto feed_arg = InternalFeedArgument{.generic_dst   = std::move(generic_address.value()),
                                                         .token_output  = std::next(output_arr, i)};

                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), std::move(feed_arg));
                }
            }

        private:

            struct Auth2FeedArgument{
                dg::string generic_dst;
                dg::string username;
                dg::string password;
                std::expected<Token, exception_t> * token_output;
            };

            //we'll worry about security later
            //we'll tackle this problem by using a layer of symmetric encoding @ socket level
            //we dont create username, password etc., Client does that, registers that for all of the servers + specifies the p2p auth guide

            struct Auth2FeedResolutor: dg::network_producer_consumer::ConsumerInterface<Auth2FeedArgument>{

                std::chrono::nanoseconds request_latency; 

                void push(std::move_iterator<Auth2FeedArgument *> data_arr, size_t sz) noexcept{

                    Auth2FeedArgument * base_data_arr                                   = data_arr.base();
                    dg::vector<dg::network_rest::Auth2TokenGenerateRequest> request_vec = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::vector<dg::network_rest::Auth2TokenGenerateRequest>>(sz));

                    for (size_t i = 0u; i < sz; ++i){
                        request_vec[i] = dg::network_rest::TokenGenerateRequest{.requestee  = std::move(base_data_arr[i].generic_dst),
                                                                                .requestor  = dg::network_ip_data::host_addr(),
                                                                                .timeout    = this->request_latency,
                                                                                .username   = std::move(base_data_arr[i].username),
                                                                                .password   = std::move(base_data_arr[i].password)};
                    }

                    std::expected<std::unique_ptr<dg::network_rest::BatchPromise<dg::network_rest::Auth2TokenGenerateResponse>>, exception_t> promise = dg::network_rest::requestmany_tokengen(dg::network_rest::get_normal_rest_controller(), request_vec);

                    if (!promise.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            *base_data_arr[i].token_output = std::unexpected(promise.error());
                        }

                        return;
                    }

                    dg::vector<std::expected<dg::network_rest::Auth2TokenGenerateResponse, exception_t>> auth2_token_vec = promise.value()->get();

                    for (size_t i = 0u; i < auth2_token_vec.size(); ++i){
                        if (!auth2_token_vec[i].has_value()){
                            *base_data_arr[i].token_output  = std::unexpected(auth2_token_vec[i].error());
                            continue;
                        }

                        if (dg::network_exception::is_failed(auth2_token_vec[i]->server_err_code)){
                            *base_data_arr[i].token_output  = std::unexpected(auth2_token_vec[i]->server_err_code); 
                            continue;
                        }

                        if (dg::network_exception::is_failed(auth2_token_vec[i]->base_err_code)){
                            *base_data_arr[i].token_output  = std::unexpected(auth2_token_vec[i]->base_err_code);
                            continue;
                        }

                        *base_data_arr[i].token_output  = Token{.token  = std::move(auth2_token_vec[i]->token),
                                                                .expiry = auth2_token_vec[i]->expiry};
                    }
                }
            };

            struct InternalFeedArgument{
                dg::string generic_dst;
                std::expected<Token, exception_t> * token_output;
            };

            struct InternalResolutor: dg::network_producer_consumer::ConsumerInterface<InternalFeedArgument>{

                dg::network_producer_consumer::DeliveryHandle<Auth2FeedArgument> * auth2_feeder;

                void push(std::move_iterator<InternalFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalFeedArgument * base_data_arr    = data_arr.base();
                    dg::vector<dg::string> generic_dst_vec  = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::vector<dg::string>>(sz));

                    for (size_t i = 0u; i < sz; ++i){
                        generic_dst_vec[i] = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::string>(base_data_arr[i].generic_dst));
                    }

                    std::expected<dg::vector<std::expected<dg::network_postgres_db::model::P2PAuthentication, exception_t>>, exception_t> p2p_auth_vec = dg::network_postgres_db::get_authentication_vec_by_id(generic_dst_vec);

                    if (!p2p_auth_vec.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            base_data_arr[i].token_output = std::unexpected(p2p_auth_vec.error());
                        }

                        return;
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        if (!p2p_auth_vec.value()[i].has_value()){
                            *base_data_arr[i].token_output = std::unexpected(p2p_auth_vec.value()[i].error()); 
                            continue;
                        }

                        switch (p2p_auth_vec.value()[i]->authentication_kind){
                            case dg::network_p2p_authentication::Auth2:
                            {
                                std::expected<dg::network_p2p_authentication::Auth2Request, exception_t> auth2_request = dg::network_p2p_authentication::decode_auth2_request_payload(p2p_auth_vec.value()[i]->content); 

                                if (!auth2_request.has_value()){
                                    *base_data_arr[i].token_output  = std::unexpected(auth2_request.error());
                                } else{
                                    auto feed_arg                   = Auth2FeedArgument{.dst            = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::string>(generic_dst_vec[i])),
                                                                                        .username       = std::move(auth2_request->username),
                                                                                        .password       = std::move(auth2_request->password),
                                                                                        .token_output   = base_data_arr[i].token_output};

                                    dg::network_producer_consumer::delvrsrv_deliver(this->auth2_feeder, std::move(feed_arg));
                                }

                                break;
                            }
                            case dg::network_p2p_authentication::DedicatedToken:
                            {
                                std::expected<dg::network_p2p_authentication::DedicatedToken, exception_t> tok = dg::network_p2p_authentication::decode_dedicated_token_payload(p2p_auth_vec.value()[i]->content);

                                if (!tok.has_value()){
                                    *base_data_arr[i].token_output  = std::unexpected(tok.error());
                                } else{
                                    *base_data_arr[i].token_output  = Token{.token   = std::move(tok->token),
                                                                            .expiry  = std::nullopt};
                                }

                                break;
                            }
                            case dg::network_p2p_authentication::Unspecified:
                            {
                                *base_data_arr[i].token_output = std::unexpected(dg::network_exception::REST_P2P_AUTH_UNSPECIFIED);
                                break;
                            }
                            default:
                            {
                                *base_data_arr[i].token_output  = std::unexpected(dg::network_exception::INTERNAL_CORRUPTION);
                                break;
                            }
                        }
                    }
                }
            };

    };

    class TokenController: public virtual TokenCacheControllerInterface{

        private:

            dg::unordered_unstable_map<Address, Token> token_map;
            size_t map_capacity;

        public:


    };

    class RequestAuthorizer: public virtual RequestAuthorizerInterface{

        private:

            std::unique_ptr<TokenRequestorInterface> token_requestor;
            std::unique_ptr<TokenCacheControllerInterface> token_cache_controller;
            std::chrono::nanoseconds token_expiry_window;
            size_t insert_feed_sz;
            size_t tokrequest_feed_sz;
            size_t tokfetch_feed_sz;

        public:

            RequestAuthorizer(std::unique_ptr<TokenRequestorInterface> token_requestor,
                              std::unique_ptr<TokenCacheControllerInterface> token_cache_controller,
                              std::chrono::nanoseconds token_expiry_window,
                              size_t insert_feed_sz,
                              size_t tokrequest_feed_sz,
                              size_t tokfetch_feed_sz) noexcept: token_requestor(std::move(token_requestor)),
                                                                 token_cache_controller(std::move(token_cache_controller)),
                                                                 token_expiry_window(token_expiry_window),
                                                                 insert_feed_sz(insert_feed_sz),
                                                                 tokrequest_feed_sz(tokrequest_feed_sz),
                                                                 tokfetch_feed_sz(tokfetch_feed_sz){}

            void authorize_request(std::move_iterator<Request *> request_arr, size_t request_arr_sz, std::expected<AuthorizedRequest, exception_t> * output_arr) noexcept{

                //this component is actually tough to write
                //we again, would want to stack 3 feeders, we'd fix the problem of requestee by using another map
                //we usually dont ask why this why that, why token this token that, token is not usually batched etc.
                //we just implement, we dont really care if it is only 1 token or 1024 p2p tokens or 1MM tokens 

                Request * base_request_arr  = request_arr.base();
                dg::unordered_unstable_map<Address, std::expected<Token, exception_t>> addr_tok_map = dg::network_exception_handler::nothrow_log(this->make_addr_tok_map(base_request_arr, request_arr_sz));
                size_t actual_addr_sz       = addr_tok_map.size(); 

                {
                    auto insert_feed_resolutor                      = TokenInsertFeedResolutor{};
                    insert_feed_resolutor.cache_controller          = this->token_cache_controller.get();

                    size_t trimmed_insert_feed_sz                   = std::min(std::min(this->insert_feed_sz, this->token_cache_controller->max_consume_size()), actual_addr_sz);
                    size_t insert_feeder_allocation_cost            = dg::network_producer_consumer::delvrsrv_allocation_cost(&insert_feed_resolutor, trimmed_insert_feed_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> insert_feeder_mem(insert_feeder_allocation_cost);
                    auto insert_feeder                              = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&insert_feed_resolutor, trimmed_insert_feed_sz, insert_feeder_mem.get()));

                    //------------------

                    auto request_feed_resolutor                     = TokenRequestFeedResolutor{};
                    request_feed_resolutor.insert_delivery_handle   = insert_feeder.get();
                    request_feed_resolutor.token_requestor          = this->token_requestor.get();

                    size_t trimmed_request_feed_sz                  = std::min(this->tokrequest_feed_sz, actual_addr_sz);
                    size_t request_feeder_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(&request_feed_resolutor, trimmed_request_feed_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> request_feeder_mem(request_feeder_allocation_cost);
                    auto request_feeder                             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&request_feed_resolutor, trimmed_request_feed_sz, request_feeder_mem.get()));

                    //------------------

                    auto tokfetch_feed_resolutor                    = TokenFetcherFeedResolutor{};
                    tokfetch_feed_resolutor.request_delivery_handle = request_feeder.get();
                    tokfetch_feed_resolutor.cache_controller        = this->token_cache_controller.get();
                    tokfetch_feed_resolutor.leeway_latency          = this->token_expiry_window; 

                    size_t trimmed_tokfetch_feed_sz                 = std::min(this->tokfetch_feed_sz, actual_addr_sz);
                    size_t tokfetch_feeder_allocation_cost          = dg::network_producer_consumer::delvrsrv_allocation_cost(&tokfetch_feed_resolutor, trimmed_tokfetch_feed_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> tokfetch_feeder_mem(tokfetch_feeder_allocation_cost);
                    auto tokfetch_feeder                            = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&tokfetch_feed_resolutor, trimmed_tokfetch_feed_sz, tokfetch_feeder_mem.get()));

                    for (auto& map_pair: addr_tok_map){
                        auto fetch_arg  = TokenFetcherArgument{.addr    = map_pair.first,
                                                               .dst     = &map_pair.second};

                        dg::network_producer_consumer::delvrsrv_deliver(tokfetch_feeder.get(), std::move(fetch_arg));
                    }
                }

                for (size_t i = 0u; i < request_arr_sz; ++i){
                    auto map_ptr = addr_tok_map.find(base_request_arr[i].requestee);

                    if constexpr(DEBUG_MODE_FLAG){
                        if (map_ptr == addr_tok_map.end()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    if (!map_ptr->second.has_value()){
                        output_arr[i] = std::unexpected(map_ptr->second.error());
                        continue;
                    }

                    output_arr[i] = AuthorizedRequest{.request  = std::move(base_request_arr[i]),
                                                      .token    = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::string>(map_ptr->second.value()))};
                }
            }

        private:

            static auto make_addr_tok_map(const Request * request_arr, size_t sz) noexcept -> std::expected<dg::unordered_unstable_map<Address, std::expected<Token, exception_t>>, exception_t>{

                try{
                    dg::unordered_unstable_map<Address, std::expected<Token, exception_t>> rs{};

                    for (size_t i = 0u; i < sz; ++i){
                        rs.insert(std::make_pair(request_arr[i].requestee, std::expected<Token, exception_t>(dg::network_exception::EXPECTED_NOT_INITIALIZED)));
                    }

                    return rs;
                } catch (...){
                    return std::unexpected(dg::network_exception::wrap_std_exception(std::current_exception()));
                }
            } 

            struct TokenInsertArgument{
                Address addr;
                Token token;
            };

            struct TokenInsertFeedResolutor: dg::network_producer_consumer::ConsumerInterface<TokenInsertArgument>{

                TokenCacheControllerInterface * cache_controller;

                void push(std::move_iterator<TokenInsertArgument *> data_arr, size_t sz) noexcept{

                    TokenInsertArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<Address[]> addr_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<Token[]> token_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        addr_arr[i]     = std::move(base_data_arr[i].addr);
                        token_arr[i]    = std::move(base_data_arr[i].token);
                    }

                    this->cache_controller->set_token(addr_arr.get(), token_arr.get(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

            struct TokenRequestArgument{
                Address addr;
                std::expected<Token, exception_t> * dst;
            };

            struct TokenRequestFeedResolutor: dg::network_producer_consumer::ConsumerInterface<TokenRequestArgument>{

                dg::network_producer_consumer::DeliveryHandle<TokenInsertArgument> * insert_delivery_handle;
                TokenRequestorInterface * token_requestor;

                void push(std::move_iterator<TokenRequestArgument *> data_arr, size_t sz) noexcept{

                    TokenRequestArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<Address[]> addr_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<Token, exception_t>[]> tok_response_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        addr_arr[i] = base_data_arr[i].addr;
                    }

                    this->token_requestor->request_token(addr_arr.get(), sz, tok_response_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (!tok_response_arr[i].has_value()){
                            *base_data_arr[i].dst = std::unexpected(tok_response_arr[i].error());
                            continue;
                        }

                        *base_data_arr[i].dst   = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::string>(tok_response_arr[i].value()));
                        auto feed_arg           = TokenInsertArgument{.addr     = addr_arr[i],
                                                                      .token    = std::move(tok_response_arr[i].value())};

                        dg::network_producer_consumer::delvrsrv_deliver(this->insert_delivery_handle, std::move(feed_arg));
                    }
                }
            };

            struct TokenFetcherArgument{
                Address addr;
                std::expected<Token, exception_t> * dst;
            };            

            struct TokenFetcherFeedResolutor: dg::network_producer_consumer::ConsumerInterface<TokenFetcherArgument>{

                dg::network_producer_consumer::DeliveryHandle<TokenRequestArgument> * request_delivery_handle;
                TokenCacheControllerInterface * cache_controller;
                std::chrono::nanoseconds leeway_latency;

                void push(std::move_iterator<TokenFetcherArgument *> data_arr, size_t sz) noexcept{

                    TokenFetcherArgument * base_data_arr = data_arr.base();
                    dg::network_stack_allocation::NoExceptAllocation<Address[]> addr_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<std::optional<Token>, exception_t>[]> token_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        addr_arr[i] = base_data_arr[i].addr;
                    }

                    this->cache_controller->get_token(addr_arr.get(), sz, token_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (!token_arr[i].has_value()){
                            *base_data_arr[i].dst   = std::unexpected(token_arr[i].error());
                            continue;
                        }

                        if (token_arr[i].value().has_value()){                            
                            if (token_arr[i].value().value().expiry.has_value()){
                                std::chrono::time_point<std::chrono::utc_clock> now             = std::chrono::utc_clock::now();
                                std::chrono::time_point<std::chrono::utc_clock> token_expiry    = token_arr[i].value().value().expiry.value();
                                std::chrono::time_point<std::chrono::utc_clock> leeway_now      = now + this->leeway_latency;

                                if (leeway_now < token_expiry){
                                    *base_data_arr[i].dst = std::move(token_arr[i].value().value());
                                    continue;
                                }
                            } else{
                                *base_data_arr[i].dst = std::move(token_arr[i].value().value());
                                continue;
                            }
                        }

                        auto fetch_arg = TokenRequestArgument{.addr = base_data_arr[i].addr,
                                                              .dst  = base_data_arr[i].dst};

                        dg::network_producer_consumer::delvrsrv_deliver(this->request_delivery_handle, std::move(fetch_arg));
                    }
                }
            };
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
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                    auto feeder                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_feed_sz, feeder_mem.get()));

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
                    size_t feeder_allocation_cost           = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_feed_sz);
                    dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                    auto feeder                             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_feed_sz, feeder_mem.get()));

                    for (size_t i = 0u; i < valid_request_sz; ++i){
                        if (dg::network_exception::is_failed(dedicated_id_err_vec.value()[i])){
                            if (base_authorized_request_arr[i].request.exception_handler != nullptr){
                                base_authorized_request_arr[i].request.exception_handler->update(dedicated_id_err_vec.value()[i]);
                            }

                            continue;
                        }

                        dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), std::move(base_authorized_request_arr[i]));
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
                    std::unique_ptr<dg::network_rest::BatchPromise<dg::network_rest::ExternalMemcommitResponse>> promise;
                    std::shared_ptr<WareHouseInterface> request_warehouse;
                    bool was_sync;

                public:

                    InternalSynchronizer(dg::vector<Request> request_vec,
                                         std::unique_ptr<dg::network_rest::BatchPromise<dg::network_rest::ExternalMemcommitResponse>> promise,
                                         std::shared_ptr<WareHouseInterface> request_warehouse) noexcept: request_vec(std::move(request_vec)),
                                                                                                          promise(std::move(promise)),
                                                                                                          request_warehouse(std::move(request_warehouse)),
                                                                                                          was_sync(false){

                        if constexpr(DEBUG_MODE_FLAG){
                            if (this->promise == nullptr){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            }
                        }

                        if constexpr(DEBUG_MODE_FLAG){
                            if (this->promise->size() != this->request_vec.size()){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            }
                        }
                    }

                    ~InternalSynchronizer() noexcept{

                        this->sync();
                    }

                    void sync() noexcept{

                        if (this->was_sync){
                            return;
                        }

                        dg::vector<std::expected<dg::network_rest::ExternalMemcommitResponse, exception_t>> response_vec = this->promise->get();

                        size_t sz           = this->request_vec.size();
                        size_t retriable_sz = 0u; 

                        for (size_t i = 0u; i < sz; ++i){
                            if (!response_vec[i].has_value()){
                                if (this->request_vec[i].exception_handler != nullptr){
                                    this->request_vec[i].exception_handler->update(response_vec[i].error());
                                }

                                continue;
                            }

                            if (dg::network_exception::is_failed(response_vec[i]->server_err_code)){
                                if (dg::network_rest::is_retriable_error(response_vec[i]->server_err_code)){
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
                                        this->request_vec[i].exception_handler->update(response_vec[i]->server_err_code);
                                    }
                                }

                                continue;
                            }

                            if (dg::network_exception::is_failed(response_vec[i]->base_err_code)){
                                if (this->request_vec[i].exception_handler != nullptr){
                                    this->request_vec[i].exception_handler->update(response_vec[i]->base_err_code);
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

            struct InternalResolutor: public virtual dg::network_producer_consumer::ConsumerInterface<AuthorizedRequest>{

                std::shared_ptr<SynchronizableWareHouseInterface> synchronizable_warehouse;
                std::shared_ptr<WareHouseInterface> request_warehouse;

                auto make_response_synchronizable(dg::vector<Request>&& request_vec,
                                                  std::unique_ptr<dg::network_rest::BatchPromise<dg::network_rest::ExternalMemcommitResponse>>&& promise) noexcept -> std::expected<std::unique_ptr<InternalSynchronizer>, exception_t>{

                    return dg::network_exception_handler::nothrow_log(dg::network_allocation::cstyle_make_unique<InternalSynchronizer>(std::move(request_vec), 
                                                                                                                                       std::move(promise)),
                                                                                                                                       this->request_warehouse);
                }

                auto to_base_request_vec(std::move_iterator<AuthorizedRequest *> inp_arr, size_t sz) noexcept -> std::expected<dg::vector<Request>, exception_t>{

                    AuthorizedRequest * base_inp_arr                    = inp_arr.base();
                    std::expected<dg::vector<Request>, exception_t> rs  = dg::network_exception::cstyle_initialize<dg::vector<Request>>(sz);

                    if (!rs.has_value()){
                        return std::unexpected(rs.error());
                    }

                    for (size_t i = 0u; i < sz; ++i){
                        rs.value()[i] = std::move(base_inp_arr[i].request);
                    }

                    return rs;
                }

                void push(std::move_iterator<AuthorizedRequest *> auth_request_vec, size_t sz) noexcept{

                    AuthorizedRequest * base_auth_request_vec = auth_request_vec.base(); 
                    dg::network_stack_allocation::NoExceptAllocation<dg::network_rest::ExternalMemcommitRequest[]> rest_request_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        dg::network_rest::ExternalMemcommitRequest rest_request{.requestee  = base_auth_request_vec[i].request.requestee,
                                                                                .requestor  = dg::network_ip_data::host_addr(),
                                                                                .timeout    = base_auth_request_vec[i].request.timeout,
                                                                                .payload    = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::network_extmemcommit_model::poly_event_t>(base_auth_request_vec[i].request.poly_event)), //
                                                                                .token      = dg::network_exception_handler::nothrow_log(dg::network_exception::cstyle_initialize<dg::string>(base_auth_request_vec[i].token)),
                                                                                .request_id = base_auth_request_vec[i].request.request_id};

                        static_assert(std::is_nothrow_move_assignable_v<dg::network_rest::ExternalMemcommitRequest>);
                        rest_request_arr[i] = std::move(rest_request);
                    }

                    std::expected<std::unique_ptr<dg::network_rest::BatchPromise<dg::network_rest::ExternalMemcommitResponse>>, exception_t> promise = dg::network_rest::requestmany_extnmemcommit(dg::network_rest::get_normal_rest_controller(), rest_request_vec.value());

                    if (!promise.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (base_auth_request_vec[i].request.exception_handler != nullptr){
                                base_auth_request_vec[i].request.exception_handler->update(promise.error());
                            }
                        }

                        return;
                    }

                    std::expected<dg::vector<Request>, exception_t> base_request_vec = this->to_base_request_vec(std::make_move_iterator(base_auth_request_vec), sz);

                    if (!base_request_vec.has_value()){
                        for (size_t i = 0u; i < sz; ++i){
                            if (base_auth_request_vec[i].request.exception_handler != nullptr){
                                base_auth_request_vec[i].request.exception_handler->update(base_request_vec.error());
                            }
                        }

                        return;
                    }

                    std::expected<std::unique_ptr<InternalSynchronizer>, exception_t> synchronizable = this->make_response_synchronizable(std::move(base_request_vec.value()),
                                                                                                                                          std::move(promise.value()));

                    if (!synchronizable.has_value()){
                        for (const Request& request: base_request_vec.value()){
                            if (request.exception_handler != nullptr){
                                request.exception_handler->update(synchronizable.error());
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

                this->request_authroizer->authorize_request(std::make_move_iterator(request_vec.data()), request_vec.size(), authorized_request_vec->data());

                for (size_t i = 0u; i < request_vec.size(); ++i){
                    if (!authorized_request_vec.value()[i].has_value()){
                        if (request_vec[i].exception_handler != nullptr){
                            request_vec[i].exception_handler->update(authorized_request_vec.value()[i].error());
                        }

                        continue;
                    }

                    requesting_request_vec.value()[requesting_request_vec_sz++] = std::move(authorized_request_vec.value()[i].value());
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

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
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