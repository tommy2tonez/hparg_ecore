#ifndef __DG_NETWORK_REST_FRAME_H__
#define __DG_NETWORK_REST_FRAME_H__ 

//define HEADER_CONTROL 12

#include <stdint.h>
#include <stdlib.h>
#include "network_std_container.h"
#include <chrono>
#include "network_exception.h"
#include "stdx.h"
#include "network_kernel_mailbox.h"

//alright, we'll stress test this to see if this could reach 1GB of inbound without polluting the internal memory ordering mechanisms
//this component should suffice for extension and enough for single responsibility
//we can't add too many features because it would hinder our future extensibility

namespace dg::network_rest_frame::model{

    using ticket_id_t   = uint64_t;
    using clock_id_t    = uint64_t; 
    using cache_id_t    = uint64_t; 

    static inline constexpr uint32_t INTERNAL_REQUEST_SERIALIZATION_SECRET  = 3312354321ULL;
    static inline constexpr uint32_t INTERNAL_RESPONSE_SERIALIZATION_SECRET = 3554488158ULL;

    struct ClientRequest{
        dg::string requestee_uri;
        dg::string requestor;
        dg::string payload;
        std::chrono::nanoseconds client_timeout_dur;

        std::optional<uint8_t> dual_priority;
        std::optional<ticket_id_t> designated_request_id;
        std::optional<std::chrono::time_point<std::chrono::utc_clock>> server_abs_timeout; //this is hard to solve, we can be stucked in a pipe and actually stay there forever, abs_timeout only works for post the transaction, which is already too late, I dont know of the way to do this correctly

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(requestee_uri, requestor, payload, client_timeout_dur, dual_priority, designated_request_id, server_abs_timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(requestee_uri, requestor, payload, client_timeout_dur, dual_priority, designated_request_id, server_abs_timeout);
        }
    };

    struct Request{
        dg::string requestee_uri;
        dg::string requestor;
        dg::string payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(requestee_uri, requestor, payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(requestee_uri, requestor, payload);
        }
    };

    struct Response{
        dg::string response;
        exception_t err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err_code);
        }
    };

    struct InternalRequest{
        Request request;
        ticket_id_t ticket_id;

        std::optional<uint8_t> dual_priority;
        bool has_unique_response;
        std::optional<cache_id_t> client_request_cache_id;
        std::optional<std::chrono::time_point<std::chrono::utc_clock>> server_abs_timeout;

        //what to add here? security, eligibility, priority, unique request id, cached response??? this is not quantifiable. it seems that those could be solved by using other measurements, we'll circle back to add these measurements
        //we dont know if the security, eligibility is this guy responsibility, yet we know for sure that priority is this guy responsibility, priority needs to be able to be propped down to the mailbox with priority mail, we haven't implemented that yet
        //unique_dedicated_request_id + cached response, such is to avoid duplicate writes from client re-requests  
        //using that is user-optional, we'll implement that 
        //the problem with priority is that we need to implement a binary batched AVL container to do in-order traversal + batch insert
        //heap_pop + heap_push are very expensive in this case
        //it's complicated
        //the re-requests + duplicate writes are very buggy + dangerous
        //we'll try to containerize the bugs at every tranmission level

        //if we look closely, the kernel_mailbox has an unsolvable unique_mailbox_id leak
        //the kernel_mailbox_stream_x has a blacklist leak
        //this dude has a duplicate request problem
        //the duplicate request could be from the user or from one of the leaks
        //we dont really know
        //what we know is that the integrity of the mailbox content must be guaranteed (we need to implement extra measurements in the flash_streamx)
        //duplicate response is actually not a problem, because we are using an ever-increasing request_id, the second time the request_id is referenced, it's gone, either because of the first time releasing the user's smp or got destructed before the first response hit

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(request, ticket_id, dual_priority, has_unique_response, client_request_cache_id, server_abs_timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(request, ticket_id, dual_priority, has_unique_response, client_request_cache_id, server_abs_timeout);
        }
    };

    struct InternalResponse{
        std::expected<Response, exception_t> response;
        ticket_id_t ticket_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, ticket_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, ticket_id);
        }
    };
} 

namespace dg::network_rest_frame::server{

    struct RequestCacheControllerInterface{
        virtual ~RequestCacheControllerInterface() noexcept = 0;
        virtual auto get_cache(cache_id_t * cache_id_arr, size_t sz, std::optional<Response> *) noexcept = 0; //we'll think about making the exclusion responsibility this component responsibility later
        virtual auto insert_cache(cache_id_t * cache_id_arr, std::move_iterator<Response *> response_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    struct CacheWriteExclusionControllerInterface{
        virtual ~RequestHandingExclusionInterface() noexcept = 0;
        virtual auto acquire_cachewrite_exclusion(cache_id_t * cache_id_arr, std::expected<bool, exception_t> * err) = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    struct RequestHandlerInterface{
        using Request   = model::Request;
        using Response  = model::Response; 

        virtual ~RequestHandlerInterface() noexcept = default;
        virtual void handle(std::move_iterator<Request *>, size_t, Response *) noexcept = 0;
    };
}

namespace dg::network_rest_frame::client{

    struct ResponseObserverInterface{
        virtual ~ResponseObserverInterface() noexcept = 0;
        virtual void update(std::expected<Response, exception_t>) noexcept = 0;
        virtual void maybedeferred_memory_ordering_fetch(std::expected<Response, exception_t>) noexcept = 0;
        virtual void maybedeferred_memory_ordering_fetch_close(void * dirty_memory) noexcept = 0;
    };

    struct BatchResponseInterface{
        virtual ~BatchResponseInterface() noexcept = default;
        virtual auto response() noexcept -> std::expected<dg::vector<std::expected<Response, exception_t>>, exception_t> = 0;
    };

    struct ResponseInterface{
        virtual ~ResponseInterface() noexcept = default;
        virtual auto response() noexcept -> std::expected<Response, exception_t> = 0; 
    };

    struct RequestContainerInterface{
        virtual ~RequestContainerInterface() noexcept = default;
        virtual auto push(dg::vector<model::InternalRequest>&&) noexcept -> exception_t = 0;
        virtual auto pop() noexcept -> dg::vector<model::InternalRequest> = 0;
    };

    struct TicketControllerInterface{
        virtual ~TicketControllerInterface() noexcept = default;
        virtual void open_ticket(size_t sz, std::expected<model::ticket_id_t, exception_t> * generated_ticket_arr) noexcept = 0;
        virtual void assign_observer(model::ticket_id_t * ticket_id_arr, size_t sz, ResponseObserverInterface ** assigning_observer_arr, exception_t * exception_arr) noexcept = 0;
        virtual void steal_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::expected<ResponseObserverInterface *, exception_t> * out_observer_arr) noexcept = 0;
        virtual void get_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::expected<ResponseObserverInterface *, exception_t> * out_observer_arr) noexcept = 0;
        virtual void close_ticket(model::ticket_id_t * ticket_id_arr, size_t sz) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0; 
    };

    struct TicketTimeoutManangerInterface{
        virtual ~TicketTimeoutManagerInterface() noexcept = default;
        virtual void clock_in(std::pair<model::ticket_id_t, std::chrono::nanoseconds> * registering_arr, size_t sz, exception_t * exception_arr) noexcept = 0;
        virtual void get_expired_ticket(model::ticket_id_t * output_arr, size_t& sz, size_t cap) noexcept = 0;
        virtual void clear() noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    struct RestControllerInterface{
        virtual ~RestControllerInterface() noexcept = default;
        virtual auto request(model::ClientRequest&&) noexcept -> std::expected<std::unique_ptr<ResponseInterface>, exception_t> = 0;
        virtual auto batch_request(std::move_iterator<model::ClientRequest *>, size_t) noexcept -> std::expected<std::unique_ptr<BatchResponseInterface>, exception_t> = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };
}

namespace dg::network_rest_frame::server_impl1{

    using namespace dg::network_rest_frame::server; 

    class RequestResolverWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            dg::unordered_unstable_map<dg::string, std::unique_ptr<RequestHandlerInterface>> request_handler_map;
            std::shared_ptr<RequestCacheControllerInterface> request_cache_controller;
            std::shared_ptr<CacheWriteExclusionControllerInterface> cachewrite_uex_controller;
            dg::network_kernel_mailbox::transmit_option_t transmit_opt;
            uint32_t resolve_channel;
            size_t resolve_consume_sz;
            uint32_t response_channel;
            size_t mailbox_feed_cap;
            size_t mailbox_prep_feed_cap;
            size_t request_handler_feed_cap;
            size_t cache_controller_feed_cap; 
            size_t busy_consume_sz; 

        public:

            RequestResolverWorker(dg::unordered_unstable_map<dg::string, std::unique_ptr<RequestHandlerInterface>> request_handler_map,
                                  std::shared_ptr<RequestCacheControllerInterface> request_cache_controller,
                                  std::shared_ptr<CacheWriteExclusionControllerInterface> cachewrite_uex_controller,
                                  dg::network_kernel_mailbox::transmit_option_t transmit_opt,
                                  uint32_t resolve_channel,
                                  size_t resolve_consume_sz,
                                  uint32_t response_channel,
                                  size_t mailbox_feed_cap,
                                  size_t mailbox_prep_feed_cap,
                                  size_t request_handler_feed_cap,
                                  size_t cache_controller_feed_cap,
                                  size_t busy_consume_sz) noexcept: request_handler_map(std::move(request_handler_map)),
                                                                    request_cache_controller(std::move(request_cache_controller)),
                                                                    cachewrite_uex_controller(std::move(cachewrite_uex_controller)),
                                                                    transmit_opt(transmit_opt),
                                                                    resolve_channel(resolve_channel),
                                                                    resolve_consume_sz(resolve_consume_sz),
                                                                    response_channel(response_channel),
                                                                    mailbox_feed_cap(mailbox_feed_cap),
                                                                    mailbox_prep_feed_cap(mailbox_prep_feed_cap),
                                                                    request_handler_feed_cap(request_handler_feed_cap),
                                                                    cache_controller_feed_cap(cache_controller_feed_cap),
                                                                    busy_consume_sz(busy_consume_sz){}

            bool run_one_epoch() noexcept{

                size_t recv_buf_cap             = this->resolve_consume_sz;
                size_t recv_buf_sz              = {};
                dg::network_stack_allocation::NoExceptAllocation<dg::string[]> recv_buf_arr(recv_buf_cap);
                dg::network_kernel_mailbox::recv(this->resolve_channel, recv_buf_arr.get(), recv_buf_sz, recv_buf_cap);

                auto feed_resolutor             = InternalResponseFeedResolutor{};
                feed_resolutor.mailbox_channel  = this->response_channel;
                feed_resolutor.transmit_opt     = this->transmit_opt;
            
                size_t trimmed_vectorization_sz = std::min(this->mailbox_feed_cap, recv_buf_sz);
                size_t feeder_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_vectorization_sz);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_vectorization_sz, feeder_mem.get()));

                for (size_t i = 0u; i < recv_buf_sz; ++i){
                    std::expected<model::InternalRequest, exception_t> request = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize<model::InternalRequest, dg::string>)(recv_buf_arr[i], model::INTERNAL_REQUEST_SERIALIZATION_SECRET); 

                    if (!request.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(request.error()));
                        continue;
                    }

                    std::expected<dg::network_kernel_mailbox::Address, exception_t> requestor_addr = dg::network_uri_encoder::extract_mailbox_addr(request->request.requestor);

                    if (!requestor_addr.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(requestor_addr.error()));
                        continue;
                    }

                    auto now = std::chrono::utc_clock::now();

                    //alright, I admit this is hard to write, the feed's gonna drive some foos crazy

                    if (request->server_abs_timeout.has_value() && request->server_abs_timeout.value() <= now){
                        auto response   = model::InternalResponse{.response     = std::unexpected(dg::network_exception::REST_ABSTIMEOUT), 
                                                                  .ticket_id    = request->ticket_id};

                        auto prep_arg   = InternalMailBoxPrepArgument{.to               = requestor_addr.value(),
                                                                      .response         = std::move(response),
                                                                      .dual_priority    = request->dual_priority}; 

                        dg::network_producer_consumer::delvrsrv_deliver(mailbox_prep_feeder.get(), std::move(prep_arg));
                    } else{
                        std::expected<dg::string, exception_t> resource_path = dg::network_uri_encoder::extract_local_path(request->request.requestee_uri);

                        if (!resource_path.has_value()){
                            auto response   = model::InternalResponse{.response     = std::unexpected(resource_path.error()), 
                                                                      .ticket_id    = request->ticket_id};

                            auto prep_arg   = InternalMailBoxPrepArgument{.to               = requestor_addr.value(),
                                                                          .response         = std::move(response),
                                                                          .dual_priority    = request->dual_priority};

                            dg::network_producer_consumer::delvrsrv_deliver(mailbox_prep_feeder.get(), std::move(prep_arg));
                        } else{
                            auto map_ptr = this->request_handler.find(resource_path.value());

                            if (map_ptr == this->request_handle.end()){
                                auto response   = model::InternalResponse{.response   = std::unexpected(dg::network_exception::REST_INVALID_URI), 
                                                                          .ticket_id  = request->ticket_id};

                                auto prep_arg   = InternalMailBoxPrepArgument{.to               = requestor_addr.value(),
                                                                              .response         = std::move(response),
                                                                              .dual_priority    = request->dual_priority};

                                dg::network_producer_consumer::delvrsrv_deliver(mailbox_prep_feeder.get(), std::move(prep_arg));
                            } else{                        
                                if (request->has_unique_response){
                                    if (!request->client_request_cache_id.has_value()){
                                        auto response   = model::InternalResponse{.response     = std::unexpected(dg::network_exception::REST_INVALID_ARGUMENT),
                                                                                  .ticket_id    = request->ticket_id};

                                        auto prep_arg   = InternalMailBoxPrepArgument{.to               = requestor_addr.value(),
                                                                                      .response         = std::move(response),
                                                                                      .dual_priority    = request->dual_priority};

                                        dg::network_producer_consumer::delvrsrv_deliver(mailbox_prep_feeder.get(), std::move(prep_arg)); 
                                    } else{
                                        auto arg   = InternalCacheFeedArgument{.to              = requestor_addr.value(),
                                                                               .local_uri_path  = std::string_view(map_ptr->first),
                                                                               .dual_priority   = request->dual_priority,
                                                                               .cache_id        = request->client_request_cache_id.value(),
                                                                               .ticket_id       = request->ticket_id,
                                                                               .request         = std::move(request->request)};

                                        dg::network_producer_consumer::delvrsrv_deliver(cache_fetch_feeder.get(), std::move(arg));
                                    }
                                } else{
                                    auto key_arg        = std::string_view(map_ptr->first);
                                    auto value_arg      = InternalServerFeedResolutorArgument{.to               = requestor_addr.value(),
                                                                                              .dual_priority    = request->dual_priority,
                                                                                              .cache_write_id   = std::nullopt,
                                                                                              .ticket_id        = request->ticket_id,
                                                                                              .request          = std::move(request->request)};

                                    dg::network_producer_consumer::delvrsrv_kv_deliver(server_resolutor_feeder.get(), key_arg, std::move(value_arg));
                                }
                            }
                        }
                    }
                }

                return recv_buf_sz >= this->busy_consume_sz;
            }
        
        private:

            struct InternalResponseFeedResolutor: dg::network_producer_consumer::ConsumerInterface<dg::network_kernel_mailbox::MailBoxArgument>{

                uint32_t mailbox_channel;
                dg::network_kernel_mailbox::transmit_option_t transmit_opt;

                void push(std::move_iterator<MailBoxArgument *> mailbox_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    dg::network_kernel_mailbox::send(this->mailbox_channel, mailbox_arr, sz, exception_arr.get(), this->transmit_opt);

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };

            struct InternalMailBoxPrepArgument{
                dg::network_kernel_mailbox::Address to;
                Response response;
                std::optional<uint8_t> priority;
            };

            struct InternalMailBoxPrepFeedResolutor: dg::network_producer_consumer::ConsumerInterface<InternalMailBoxPrepArgument>{

                dg::network_producer_consumer::DeliveryHandle<dg::network_kernel_mailbox::MailBoxArgument> * mailbox_feeder;

                void push(std::move_iterator<InternalMailBoxPrepArgument *> data_arr, size_t sz) noexcept{

                    auto base_data_arr = data_arr.base();

                    for (size_t i = 0u; i < sz; ++i){
                        std::expected<dg::string, exception_t> serialized_response = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::serialize<dg::string, Response>)(base_data_arr[i].response, model::INTERNAL_RESPONSE_SERIALIZATION_SECRET);

                        if (!serialized_response.has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(serialized_response.error()));
                            continue;
                        }

                        auto arg = dg::network_kernel_mailbox::MailBoxArgument{.to = base_data_arr[i].to,
                                                                               .content = std::move(serialized_response.value()),
                                                                               .priority = base_data_arr[i].priority};

                        dg::network_producer_consumer::delvrsrv_deliver(this->mailbox_feeder, std::move(arg));
                    }
                }
            };

            struct InternalServerFeedResolutorArgument{
                dg::network_kernel_mailbox::Address to;
                std::optional<uint8_t> dual_priority;
                std::optional<cache_id_t> cache_write_id;
                ticket_id_t ticket_id;
                Request request;
            };

            struct InternalServerFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<std::string_view, InternalServerFeedResolutorArgument>{

                dg::unordered_unstable_map<dg::string, std::unique_ptr<RequestHandlerInterface>> * request_handler_map;
                dg::network_producer_consumer::DeliveryHandle<InternalMailBoxPrepArgument> * mailbox_prep_feeder;
                dg::network_producer_consumer::DeliveryHandle<InternalCacheMapInserterArgument> * cache_map_feeder;

                void push(const std::string_view& local_uri_path, std::move_iterator<InternalServerFeedResolutorArgument *> data_arr, size_t sz) noexcept{

                    auto base_data_arr = data_arr.base();
                    dg::network_stack_allocation::NoExceptAllocation<Request[]> request_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<Response[]> response_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        request_arr[i] = std::move(base_data_arr[i].request);
                    }

                    auto map_ptr = this->request_handler_map->find(local_uri_path);

                    if constexpr(DEBUG_MODE_FLAG){
                        if (map_ptr == this->request_handler_map->end()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    map_ptr->second->handle(std::make_move_iterator(request_arr.get()), sz, response_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (data_arr[i].cache_write_id.has_value()){
                            std::expected<Response, exception_t> cpy_response_arr = dg::network_exception::cstyle_initialize<Response>(response_arr[i]);

                            if (!cpy_response_arr.has_value()){
                                dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(cpy_response_arr.error()));
                            } else{
                                auto cache_mapfeed_arg = InternalCacheMapFeedArgument{.cache_id = base_data_arr[i].cache_write_id,
                                                                                      .response = std::move(cpy_response_arr.value())}; 

                                dg::network_producer_consumer::delvrsrv_deliver(this->cache_map_feeder, std::move(cache_mapfeed_arg));
                            }
                        }

                        auto response   = model::InternalResponse{.response   = std::move(response_arr[i]),
                                                                  .ticket_id  = base_data_arr[i].ticket_id}; 

                        auto prep_arg   = InternalMailBoxPrepArgument{.to               = base_data_arr[i].to,
                                                                      .response         = std::move(response),
                                                                      .dual_priority    = base_data_arr[i].dual_priority};

                        dg::network_producer_consumer::delvrsrv_deliver(this->mailbox_prep_feeder, std::move(prep_arg));
                    }
                } 
            };

            struct InternalCacheServerFeedArgument{
                dg::network_kernel_mailbox::Address to;
                std::string_view local_uri_path;
                std::optional<uint8_t> dual_priority;
                cache_id_t cache_id;
                ticket_id_t ticket_id;
                Request request;
            };

            struct InternalCacheServerFeedResolutor: dg::network_producer_consumer::ConsumerInterface<InternalCacheServerFeedArgument>{

                CacheWriteExclusionControllerInterface * cachewrite_uex_controller;
                dg::network_producer_consumer::KVDeliveryHandle<InternalServerFeedResolutorArgument> * server_resolutor_feeder;
                dg::network_producer_consumer::DeliveryHandle<InternalMailBoxPrepArgument> * mailbox_prep_feeder;

                void push(std::move_iterator<InternalCacheServerFeedArgument *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> cache_write_response_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i] = base_data_arr[i].cache_id;
                    }

                    this->cachewrite_uex_controller->acquire_cachewrite_exclusion(cache_id_arr.get(), sz, cache_write_response_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (!cache_write_response_arr[i].has_value()){
                            auto response = model::InternalResponse{.response   = std::unexpected(cache_write_response_arr[i].error()),
                                                                    .ticket_id  = base_data_arr[i]->ticket_id};
                            
                            auto prep_arg   = InternalMailBoxPrepArgument{.to               = base_data_arr[i].to,
                                                                          .response         = std::move(response),
                                                                          .dual_priority    = base_data_arr[i].dual_priority};

                            dg::network_producer_consumer::delvrsrv_deliver(this->mailbox_prep_feeder, std::move(prep_arg));
                        } else{
                            if (!cache_write_response_arr[i].value()){
                                auto response   = model::InternalResponse{.response   = std::unexpected(dg::network_exception::REST_BAD_UNIQUE_WRITE), //
                                                                          .ticket_id  = base_data_arr[i]->ticket_id};

                                auto prep_arg   = InternalMailBoxPrepArgument{.to               = base_data_arr[i].to,
                                                                              .response         = std::move(response),
                                                                              .dual_priority    = base_data_arr[i].dual_priority};
                                
                                dg::network_producer_consumer::delvrsrv_deliver(this->mailbox_prep_feeder, std::move(prep_arg));
                            } else{
                                auto arg    = InternalServerFeedResolutorArgument{.to               = base_data_arr[i].to,
                                                                                  .dual_priority    = base_data_arr[i].dual_priority,
                                                                                  .cache_write_id   = base_data_arr[i].cache_id,
                                                                                  .ticket_id        = base_data_arr[i].ticket_id,
                                                                                  .request          = std::move(base_data_arr[i].request)};

                                dg::network_producer_consumer::delvrsrv_kv_deliver(this->resolutor_feeder, base_data_arr[i].local_uri_path, std::move(arg));
                            }   
                        }
                    }
                }
            };

            struct InternalCacheFeedArgument{
                dg::network_kernel_mailbox::Address to;
                std::string_view local_uri_path;
                std::optional<uint8_t> dual_priority;
                cache_id_t cache_id;
                ticket_id_t ticket_id;
                Request request;
            };

            struct InternalCacheFeedResolutor: dg::network_producer_consumer::ConsumerInterface<InternalCacheFeedArgument>{

                dg::network_producer_consumer::KVDeliveryHandle<InternalCacheServerFeedArgument> * resolutor_feeder;
                dg::network_producer_consumer::DeliveryHandle<InternalMailBoxPrepArgument> * mailbox_prep_feeder; 

                RequestCacheControllerInterface * cache_controller;

                void push(std::move_iterator<InternalCacheFeedArgument *> data_arr, size_t sz) noexcept{

                    auto base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::optional<Response>[]> response_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i] = base_data_arr[i].cache_id;                        
                    }

                    this->cache_controller->get_cache(cache_id_arr.get(), sz, response_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (response_arr[i].has_value()){
                            auto response = model::InternalResponse{.response   = std::move(response_arr[i].value()),
                                                                    .ticket_id  = base_data_arr[i].ticket_id};

                            auto prep_arg   = InternalMailBoxPrepArgument{.to               = base_data_arr[i].to,
                                                                          .response         = std::move(response),
                                                                          .dual_priority    = base_data_arr[i].dual_priority};
                            
                            dg::network_producer_consumer::delvrsrv_deliver(this->mailbox_prep_feeder, std::move(prep_arg));
                        } else{

                            auto arg    = InternalCacheServerFeedArgument{.to               = base_data_arr[i].to,
                                                                          .local_uri_path   = base_data_arr[i].local_uri_path,
                                                                          .dual_priority    = base_data_arr[i].dual_priority,
                                                                          .cache_id         = base_data_arr[i].cache_id,
                                                                          .ticket_id        = base_data_arr[i].ticket_id,
                                                                          .request          = std::move(base_data_arr[i].request)};

                            dg::network_producer_consumer::delvrsrv_kv_deliver(this->resolutor_feeder, base_data_arr[i].local_uri_path, std::move(arg));
                        }
                    }
                }
            };
    };
} 

namespace dg::network_rest_frame::client_impl1{

    using namespace dg::network_rest_frame::client; 

    //what's the clever way

    class RequestResponseBase{
        
        private:

            std::binary_semaphore smp; //I'm allergic to shared_ptr<>, it costs a memory_order_seq_cst to deallocate the object, we'll do things this way to allow us leeways to do relaxed operations to unlock batches of requests later, thing is that timed_semaphore is not a magic, it requires an entry registration in the operating system, we'll work around things by reinventing the wheel
            std::expected<Response, exception_t> resp;
            std::atomic<bool> is_response_invoked;

        public:

            RequestResponse() noexcept: smp(0u),
                                        resp(std::nullopt),
                                        is_response_invoked(false){}

            void update(std::expected<Response, exception_t> response_arg) noexcept{

                //another fence
                // std::atomic_thread_fence(std::memory_order_acquire); //this is not necessary because RequestResponse * acquisition must involve these mechanisms
                this->resp = std::move(response_arg); //this is the undefined
                this->smp.release();
            }

            void maybedeferred_memory_ordering_fetch(std::expected<Response, exception_t> response_arg) noexcept{

                this->update(std::move(response_arg));
            }

            void maybedeferred_memory_ordering_fetch_close(void * dirty_memory) noexcept{

                (void) dirty_memory;
            }

            auto response() noexcept -> std::expected<Response, exception_t>{

                bool was_invoked = this->is_response_invoked.exchange(true, std::memory_order_relaxed);

                if (was_invoked){
                    return std::unexpected(dg::network_exception::REST_UNIQUE_RESOURCE_ACQUIRED);
                }

                this->smp.acquire();
                return std::expected<Response, exception_t>(std::move(this->resp));
            }
    };

    class RequestResponseBaseObserver: public virtual ResponseObserverInterface{

        private:

            RequestResponseBase * base;
        
        public:

            RequestResponseBaseObserver() = default;

            RequestResponseBaseObserver(RequestResponseBase * base) noexcept: base(base){}

            void update(std::expected<Response, exception_t> resp) noexcept{

                this->base->update(std::move(resp));
            }

            void maybedeferred_memory_ordering_fetch(std::expected<Response, exception_t> resp) noexcept{

                this->base->maybedeferred_memory_ordering_fetch(std::move(resp));
            }

            void maybedeferred_memory_ordering_fetch_close(void * dirty_memory) noexcept{

                this->base->maybedeferred_memory_ordering_fetch_close(dirty_memory);
            }
    };

    class BatchRequestResponseBase{

        private:

            std::atomic<intmax_t> atomic_smp;
            dg::vector<std::expected<Response, exception_t>> resp_vec;
            std::atomic<bool> is_response_invoked;

        public:

            BatchRequestResponseBase(size_t resp_sz): atomic_smp(-static_cast<intmax_t>(resp_sz) + 1),
                                                      resp_vec(resp_sz),
                                                      is_response_invoked(false){

                if (resp_sz == 0u){
                    dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMENT);
                }
            }

            void update(size_t idx, std::expected<Response, exception_t> response) noexcept{

                this->resp_vec[idx] = std::move(response);
                this->atomic_smp.fetch_add(1u, std::memory_order_release);
            }

            void maybedeferred_memory_ordering_fetch(size_t idx, std::expected<Response, exception_t> response) noexcept{
                
                //this is ..., the designated memory area of the producer is clear
                //dg::vector<> size is const, dg::vector<> ptr is const, the logic is sound, but it is not 
                //producer is responsible to defer the memory ordering and do 1 big std::atomic_thread_fence(std::memory_order_release) after the consume

                this->resp_vec[idx] = std::move(response);
            }

            void maybedeferred_memory_ordering_fetch_close(size_t idx, void * dirty_memory) noexcept{

                std::atomic_signal_fence(std::memory_order_seq_cst);
                stdx::empty_noipa(dirty_memory);
                this->atomic_smp.fetch_add(1u, std::memory_order_relaxed);
            }

            auto response() noexcept -> std::expected<dg::vector<std::expected<Response, exception_t>>, exception_t>{

                bool was_invoked = this->is_response_invoked.exchange(true, std::memory_order_relaxed);

                if (was_invoked){
                    return std::unexpected(dg::network_exception::REST_UNIQUE_RESOURCE_ACQUIRED);
                } 

                this->atomic_smp.wait(0, std::memory_order_acquire);
                return dg::vector<std::expected<Response, exception_t>>(std::move(this->resp_vec));
            }
    };

    class BatchRequestResponseBaseDesignatedObserver: public virtual ResponseObserverInterface{

        private:

            BatchRequestResponseBase * base; //we dont care about memory safety, we'll talk about this later, because we are in the dangy phase of performance
                                             //we'll do some very break-the-practice coding
            size_t idx;

        public:

            BatchRequestResponseBaseDesignatedObserver() = default;

            BatchRequestResponseBaseDesignatedObserver(BatchRequestResponseBase * base, 
                                                       size_t idx) noexcept: base(base),
                                                                             idx(idx){}

            void update(std::expected<Response, exception_t> response) noexcept{

                this->base->update(this->idx, std::move(response));
            }

            void maybedeferred_memory_ordering_fetch(std::expected<Response, exception_t> response) noexcept{

                this->base->maybedeferred_memory_ordering_fetch(std::move(response));
            }

            void maybedeferred_memory_ordering_fetch_close(void * dirty_memory) noexcept{

                this->base->maybedeferred_memory_ordering_fetch_close(dirty_memory);
            }
    };

    class BatchRequestResponse: public virtual BatchResponseInterface{

        private:

            dg::vector<BatchRequestResponseBaseDesignatedObserver> observer_arr; 
            BatchRequestResponseBase base;
        
        public:

            BatchRequestResponse(size_t resp_sz): observer_arr(resp_sz),
                                                  base(resp_sz){

                for (size_t i = 0u; i < resp_sz; ++i){
                    this->observer_arr[i] = BatchRequestResponseBaseDesignatedObserver(&this->base, i);
                }
            }

            ~BatchRequestResponse() noexcept{

                //this is harder than expected
                stdx::empty_noipa(this->base->response(), this->observer_arr);
            }

            auto response() noexcept -> std::expected<dg::vector<std::expected<Response, exception_t>>, exception_t>{

                return this->base->response();
            }

            auto response_size() const noexcept -> size_t{

                return this->observer_arr.size();
            }

            auto get_observer(size_t idx) noexcept -> ResponseObserverInterface *{ //response observer is a unique_resource, such is an acquisition of this pointer must involve accurate acquire + release mechanisms like every other dg::string or dg::vector, etc.

                return std::addressof(this->observer_arr[idx]);
            }
    };

    auto make_batch_request_response(size_t resp_sz) noexcept -> std::expected<std::unique_ptr<BatchRequestResponse>, exception_t>{

        return dg::network_exception::to_cstyle_function(dg::network_allocation::std_make_unique<BatchRequestResponse, size_t>)(resp_sz);
    }

    class RequestResponse: public virtual ResponseInterface{

        private:

            RequestResponseBase base;
            RequestResponseBaseObserver observer;

        public:

            RequestResponse(): base(){

                this->observer = RequestResponseBaseObserver(&this->base);
            }

            ~ReleaseSafeRequestResponse() noexcept{

                stdx::empty_noipa(this->base->response(), this->observer);
            }

            auto response() noexcept -> std::expected<Response, exception_t>{

                return this->base->response();
            }

            auto get_observer() noexcept -> ResponseObserverInterface *{ //response observer is a unique_resource, such is an acquisition of this pointer must involve accurate acquire + release mechanisms like every other dg::string or dg::vector, etc.

                return &this->observer;
            }
    };

    auto make_request_response() -> std::expected<std::unique_ptr<RequestResponse>, exception_t>{

        return dg::network_exception::to_cstyle_function(dg::network_allocation::std_make_unique<RequestResponse>)();
    }

    class RequestContainer: public virtual RequestContainerInterface{

        private:

            dg::pow2_cyclic_queue<dg::vector<model::InternalRequest>> producer_queue;
            dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, std::optional<dg::vector<model::InternalRequest>> *>> waiting_queue;
            std::unique_ptr<std::mutex> mtx;

        public:

            RequestContainer(dg::pow2_cyclic_queue<dg::vector<model::InternalRequest>> producer_queue,
                             dg::pow2_cyclic_queue<std::pair<std::binary_semaphore *, dg::vector<model::InternalRequest> *>> waiting_queue,
                             std::unique_ptr<std::mutex> mtx) noexcept: producer_queue(std::move(producer_queue)),
                                                                        waiting_queue(std::move(waiting_queue)),
                                                                        mtx(std::move(mtx)){}

            auto push(dg::vector<model::InternalRequest>&& request) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (!this->waiting_queue.empty()){
                    auto [pending_smp, fetching_addr] = std::move(this->waiting_queue.front());
                    this->waiting_queue.pop_front();
                    *fetching_addr = std::move(request);
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                    pending_smp->release();

                    return dg::network_exception::SUCCESS;
                }

                if (this->producer_queue.size() != this->producer_queue.capacity()){
                    dg::network_exception_handler::nothrow_log(this->producer_queue.push_back(std::move(request)));
                    return dg::network_exception::SUCCESS;
                }

                return dg::network_exception::RESOURCE_EXHAUSTION;
            }

            auto pop() noexcept -> dg::vector<model::InternalRequest>{

                auto pending_smp        = std::binary_semaphore(0);
                auto internal_request   = std::optional<dg::vector<model::InternalRequest>>{};

                while (true){
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->producer_queue.empty()){
                        auto rs = std::move(this->container.front());
                        this->container.pop_front();
                        return rs;
                    }

                    if (this->waiting_queue.size() == this->waiting_queue.capacity()){
                        continue;
                    }

                    this->waiting_queue.push_back(std::make_pair(&pending_smp, &internal_request));
                    break;
                }

                pending_smp->acquire();
                std::atomic_signal_fence(std::memory_order_seq_cst);

                return dg::vector<model::InternalRequest>(std::move(internal_request.value()));
            }
    };

    class TicketController: public virtual TicketControllerInterface{

        private:

            dg::unordered_unstable_map<model::ticket_id_t, std::optional<ResponseObserverInterface *>> ticket_resource_map;
            size_t ticket_resource_map_cap;
            ticket_id_t ticket_id_counter;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> max_consume_per_load;

        public:

            TicketController(dg::unordered_unstable_map<model::ticket_id_t, std::optional<ResponseObserverInterface *>> ticket_resource_map,
                            size_t ticket_resource_map_cap,
                            size_t ticket_id_counter,
                            std::unique_ptr<std::mutex> mtx,
                            stdx::hdi_container<size_t> max_consume_per_load): ticket_resource_map(std::move(ticket_resource_map)),
                                                                               ticket_resource_map_cap(ticket_resource_map_cap),
                                                                               ticket_id_counter(ticket_id_counter),
                                                                               mtx(std::move(mtx)),
                                                                               max_consume_per_load(std::move(max_consume_per_load)){}

            void open_ticket(size_t sz, std::expected<model::ticket_id_t, exception_t> * out_ticket_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    if (this->ticket_resource_map.size() == this->ticket_resource_map_cap){
                        out_ticket_arr[i] = std::unexpected(dg::network_exception::QUEUE_FULL);
                        continue;
                    }

                    model::ticket_id_t new_ticket_id    = this->ticket_id_counter++;
                    auto [map_ptr, status]              = this->ticket_resource_map.insert(std::make_pair(new_ticket_id, std::optional<ResponseObserverInterface *>(std::nullopt)));
                    dg::network_exception_handler::dg_assert(status);
                    out_ticket_arr[i]                   = new_ticket_id;
                }
            }

            void assign_observer(model::ticket_id_t * ticket_id_arr, size_t sz, ResponseObserverInterface ** corresponding_observer_arr, exception_t * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    model::ticket_id_t current_ticket_id = ticket_id_arr[i];
                    auto map_ptr = this->ticket_resource_map.find(current_ticket_id);

                    if (map_ptr == this->ticket_resource_map.end()){
                        exception_arr[i] = dg::network_exception::REST_TICKET_NOT_FOUND;
                        continue;
                    }

                    if (map_ptr->second.has_value()){
                        exception_arr[i] = dg::network_exception::REST_TICKET_OBSERVER_EXISTED;
                        continue;
                    }

                    map_ptr->second     = corresponding_observer_arr[i];
                    exception_arr[i]    = dg::network_exception::SUCCESS;
                }
            }

            void get_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::expected<ResponseObserverInterface *, exception_t> * out_observer_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    model::ticket_id_t current_ticket_id = ticket_id_arr[i];
                    auto map_ptr = this->ticket_resource_map.find(current_ticket_id);

                    if (map_ptr == this->ticket_resource_map.end()){
                        response_arr[i] = std::unexpected(dg::network_exception::REST_TICKET_NOT_FOUND);
                        continue;
                    }

                    if (!map_ptr->second.has_value()){
                        response_arr[i] = std::unexpected(dg::network_exception::REST_TICKET_OBSERVER_NOT_AVAILABLE);
                        continue;
                    }

                    response_arr[i] = map_ptr->second.value();
                }
            }

            void steal_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::expected<ResponseObserverInterface *, exception_t> * response_arr) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    model::ticket_id_t current_ticket_id = ticket_id_arr[i];
                    auto map_ptr = this->ticket_resource_map.find(current_ticket_id);

                    if (map_ptr == this->ticket_resource_map.end()){
                        response_arr[i] = std::unexpected(dg::network_exception::REST_TICKET_NOT_FOUND);
                        continue;
                    }

                    if (!map_ptr->second.has_value()){
                        response_arr[i] = std::unexpected(dg::network_exception::REST_TICKET_OBSERVER_NOT_AVAILABLE);
                        continue;
                    }

                    response_arr[i] = map_ptr->second.value();
                    map_ptr->second = std::nullopt;
                }
            }

            void close_ticket(model::ticket_id_t * ticket_id_arr, size_t sz) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    //we are unforgiving
                    size_t removed_sz = this->ticket_resource_map.erase(ticket_id_arr[i]);

                    if constexpr(DEBUG_MODE_FLAG){
                        if (removed_sz == 0u){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load.value;
            }
    };

    //alright, it is advised that we spawn multiple distributed rest controller + multiple distributed ticket controller
    //RequestContainer is an absolute unit, so we can't really change that
    //we are expecting to do 10GB of REST request payloads per second, it's possible

    //this is to utilize CPU resource, not CPU efficiency, CPU efficiency is already achieved by batching ticket_id_arr, kernel has effective schedulings that render the overhead of mtx -> 1/1000 of the actual tx 
    class DistributedTicketController: public virtual TicketControllerInterface{

        private:

            std::unique_ptr<std::unique_ptr<TicketControllerInterface>[]> base_arr;
            size_t pow2_base_arr_sz;
            size_t max_consume_per_load;
            size_t keyvalue_feed_cap;

        public:

            DistributedTicketController(std::unique_ptr<std::unique_ptr<TicketControllerInterface>[]> base_arr,
                                        size_t pow2_base_arr_sz,
                                        size_t max_consume_per_load,
                                        size_t keyvalue_feed_cap) noexcept: base_arr(std::move(base_arr)),
                                                                            pow2_base_arr_sz(pow2_base_arr_sz),
                                                                            max_consume_per_load(max_consume_per_load),
                                                                            keyvalue_feed_cap(keyvalue_feed_cap){}

            void open_ticket(){

            }

            void assign_observer(){

            }

            void steal_observer(){

            }

            void get_observer(){

            }

            void close_ticket(){

            }

            auto max_consume_size() noexcept -> size_t{

            }
    };

    class TicketTimeoutManager: public virtual TicketTimeoutManagerInterface{

        public:

            struct ExpiryBucket{
                model::ticket_id_t ticket_id;
                std::chrono::time_point<std::chrono::high_resolution_clock> abs_timeout;
            };

        private:

            dg::vector<ExpiryBucket> expiry_bucket_queue; //this is harder than expected, we are afraid of the priority queue, yet we would want to discretize this to avoid priority queues
            size_t expiry_bucket_queue_cap;
            std::chrono::nanoseconds max_dur;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> max_consume_per_load;

        public:

            TicketTimeoutManager(dg::vector<ExpiryBucket> expiry_bucket_queue,
                                 size_t expiry_bucket_queue_cap,
                                 std::chrono::nanoseconds max_dur,
                                 std::unique_ptr<std::mutex> mtx,
                                 stdx::hdi_container<size_t> max_consume_per_load) noexcept: expiry_bucket_queue(std::move(expiry_bucket_queue)),
                                                                                             expiry_bucket_queue_cap(expiry_bucket_queue_cap),
                                                                                             mtx(std::move(mtx)),
                                                                                             max_consume_per_load(std::move(max_consume_per_load)){}

            void clock_in(std::pair<model::ticket_id_t, std::chrono::nanoseconds> * registering_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }
                
                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto now        = std::chrono::high_resolution_clock::now(); 
                auto greater    = [](const ExpiryBucket& lhs, const ExpiryBucket& rhs) noexcept {return lhs.abs_timepout > rhs.abs_timeout;};

                for (size_t i = 0u; i < sz; ++i){
                    auto [ticket_id, current_dur] = registering_arr[i];

                    if (current_dur > this->max_dur){
                        exception_arr[i] = dg::network_exception::INVALID_ARGUMENT;
                        continue;
                    }

                    if (this->expiry_bucket_queue.size() == this->expiry_bucket_queue_cap){
                        exception_arr[i] = dg::network_exception::QUEUE_FULL;
                        continue;
                    }

                    this->expiry_bucket_queue.push_back(ExpiryBucket{ticket_id, now + current_dur});
                    std::push_heap(this->expiry_bucket_queue.begin(), this->expiry_bucket_queue.end(), greater);
                    exception_arr[i] = dg::network_exception::SUCCESS;
                }
            }

            void get_expired_ticket(model::ticket_id_t * ticket_arr, size_t& ticket_arr_sz, size_t ticket_arr_cap) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                ticket_arr_sz   = 0u;
                auto now        = std::chrono::high_resolution_clock::now();
                auto greater    = [](const ExpiryBucket& lhs, const ExpiryBucket& rhs) noexcept {return lhs.abs_timepout > rhs.abs_timeout;};

                while (true){
                    if (ticket_arr_sz == ticket_arr_cap){
                        return;
                    }

                    if (this->expiry_bucket_queue.empty()){
                        return;
                    }

                    if (this->expiry_bucket_queue.front().abs_timeout > now){
                        return;
                    }

                    ticket_arr[ticket_arr_sz++] = this->expiry_bucket_queue.front().ticket_id;
                    std::pop_heap(this->expiry_bucket_queue.begin(), this->expiry_bucket_queue.end(), greater);
                    this->expiry_bucket_queue.pop_back();
                }
            }

            void clear() noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                this->expiry_bucket_queue.clear();
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load.value;
            }
    };

    //this is to utilize CPU resource, not CPU efficiency, CPU efficiency is already achieved by batching ticket_id_arr, kernel has effective schedulings that render the overhead of mtx -> 1/1000 of the actual tx 
    class DistributedTicketTimeoutManager: public virtual TicketTimeoutManagerInterface{

        private:

            std::unique_ptr<std::unique_ptr<TicketTimeoutManagerInterface>[]> base_arr;
            size_t pow2_base_arr_sz;
            size_t keyvalue_feed_cap;
            size_t zero_bounce_sz;
            std::unique_ptr<DrainerPredicateInterface> drainer_predicate;
            size_t drain_peek_cap_per_container;
            size_t max_consume_per_load;

        public:

            DistributedTicketTimeoutManager(std::unique_ptr<std::unique_ptr<TicketTimeoutManagerInterface>[]> base_arr,
                                            size_t pow2_base_arr_sz,
                                            size_t keyvalue_feed_cap,
                                            size_t zero_bounce_sz,
                                            std::unique_ptr<DrainerPredicateInterface> drainer_predicate,
                                            size_t max_consume_per_load) noexcept: base_arr(std::move(base_arr)),
                                                                                   pow2_base_arr_sz(pow2_base_arr_sz),
                                                                                   keyvalue_feed_cap(keyvalue_feed_cap),
                                                                                   zero_bounce_sz(zero_bounce_sz),
                                                                                   drainer_predicate(std::move(drainer_predicate)),
                                                                                   drain_peek_cap_per_container(drain_peek_cap_per_container),
                                                                                   max_consume_per_load(max_consume_per_load){}

            void clock_in(std::pair<model::ticket_id_t, std::chrono::nanoseconds> * registering_arr, size_t sz, exception_t * exception_arr) noexcept{

                auto feed_resolutor                 = InternalClockInFeedResolutor{};
                feed_resolutor.dst                  = this->base_arr.get(); 

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS;);

                for (size_t i = 0u; i < sz; ++i){
                    size_t hashed_value     = dg::network_hash::hash_reflectible(registering_arr[i].first);
                    size_t partitioned_idx  = hashed_value & (this->pow2_base_arr_sz - 1u);
                    auto feed_arg           = InternalClockInFeedArgument{};
                    feed_arg.ticket_id      = registering_arr[i].first;
                    feed_arg.dur            = registering_arr[i].second;
                    feed_arg.exception_ptr  = std::next(exception_arr, i); 

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }
            }

            void get_expired_ticket(model::ticket_id_t * ticket_arr, size_t& ticket_arr_sz, size_t ticket_arr_cap) noexcept{

                if (this->drainer_predicate->is_should_drain()){
                    this->internal_drain(ticket_arr, ticket_arr_sz, ticket_arr_cap);
                    this->drainer_predicate->reset();
                } else{
                    this->internal_curious_pop(ticket_arr, ticket_arr_sz, ticket_arr_cap);
                }
            }

            void clear() noexcept{

                for (size_t i = 0u; i < this->pow2_base_arr_sz; ++i){
                    this->base_arr[i]->clear();
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load;
            }
        
        private:

            struct InternalClockInFeedArgument{
                ticket_id_t ticket_id;
                std::chrono::nanoseconds dur;
                exception_t * exception_ptr;
            };

            struct InternalClockInFeedResolutor{

                std::unique_ptr<TicketTimeoutManagerInterface> * dst;

                void push(const size_t& idx, std::move_iteartor<InternalClockInFeedArgument *> data_arr, size_t sz) noexcept{
                    
                    auto base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<std::pair<model::ticket_id_t, std::chrono::nanoseconds>[]> registering_arr(sz); //pair is not good
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        registering_arr[i] = std::make_pair(base_data_arr[i].ticket_id, base_data_arr[i].dur);
                    }

                    this->dst[idx]->clock_in(registering_arr.get(), sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            *base_data_arr[i].exception_ptr = exception_arr[i];
                        }
                    }
                }
            };

            void internal_drain(model::ticket_id_t * ticket_arr, size_t& ticket_arr_sz, size_t ticket_arr_cap) noexcept{

                size_t random_clue                      = dg::network_randomizer::randomize_int<size_t>();
                ticket_arr_sz                           = 0u;
                model::ticket_id_t * current_ticket_arr = ticket_arr;
                size_t current_arr_cap                  = ticket_arr_cap;

                for (size_t i = 0u; i < this->pow2_base_arr_sz; ++i){
                    if (current_arr_cap == 0u){
                        break;
                    }

                    size_t random_idx   = (random_clue + i) & (this->pow2_base_arr_sz - 1u);
                    size_t current_sz   = {};
                    size_t peeking_cap  = std::min(current_arr_cap, this->drain_peek_cap_per_container); 
                    this->base_arr[random_idx]->get_expired_ticket(current_ticket_arr, current_sz, peeking_cap);
                    std::advance(current_ticket_arr, current_sz);
                    current_arr_cap     -= current_sz;
                    ticket_arr_sz       += current_sz;
                }
            }

            void internal_curious_pop(model::ticket_id_t * ticket_arr, size_t& ticket_arr_sz, size_t ticket_arr_cap) noexcept{

                ticket_arr_sz = 0u;

                for (size_t i = 0u; i < this->zero_bounce_sz; ++i){
                    size_t idx = dg::network_randomizer::randomize_int<size_t>() & (this->pow2_base_arr_sz - 1u);
                    this->base_arr[idx]->get_expired_ticket(ticket_arr, ticket_arr_sz, ticket_arr_cap);

                    if (ticket_arr_sz != 0u){
                        return;
                    }
                }
            }
    };

    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<TicketControllerInterface> ticket_controller;
            uint32_t channel;
            size_t ticket_controller_feed_cap;
            size_t recv_consume_sz;
            size_t busy_consume_sz;
        
        public:

            InBoundWorker(std::shared_ptr<TicketControllerInterface> ticket_controller,
                          uint32_t channel,
                          size_t ticket_controller_feed_cap,
                          size_t recv_consume_sz,
                          size_t busy_consume_sz) noexcept: ticket_controller(std::move(ticket_controller)),
                                                            channel(channel),
                                                            ticket_controller_feed_cap(ticket_controller_feed_cap),
                                                            recv_consume_sz(recv_consume_sz),
                                                            busy_consume_sz(busy_consume_sz){}

            bool run_one_epoch() noexcept{

                size_t buf_arr_cap                          = this->recv_consume_sz;
                size_t buf_arr_sz                           = {};
                dg::network_stack_allocation::NoExceptAllocation<dg::string[]> buf_arr(buf_arr_cap); 
                dg::network_kernel_mailbox::recv(this->channel, buf_arr.get(), buf_arr_sz, buf_arr_cap);

                auto feed_resolutor                         = InternalFeedResolutor{};
                feed_resolutor.ticket_controller            = this->ticket_controller.get();

                size_t trimmed_ticket_controller_feed_cap   = std::min(std::min(this->ticket_controller_feed_cap, this->ticket_controller->max_consume_size()), buf_arr_sz);
                size_t feeder_allocation_cost               = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_ticket_controller_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_ticket_controller_feed_cap, feeder_mem.get())); 

                for (size_t i = 0u; i < buf_arr_sz; ++i){
                    std::expected<model::InternalResponse, exception_t> response = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize<model::InternalResponse, dg::string>(buf_arr[i], model::INTERNAL_RESPONSE_SERIALIZATION_SECRET));

                    if (!response.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(err));
                        continue;
                    }

                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), std::move(response.value()));                    
                }

                return buf_arr_sz >= this->busy_consume_sz;
            }

        private:

            struct InternalFeedResolutor: dg::network_producer_consumer::ConsumerInterface<model::InternalResponse>{

                TicketControllerInterface * ticket_controller;

                void push(std::move_iterator<model::InternalResponse *> response_arr, size_t sz) noexcept{

                    auto base_response_arr = response_arr.base();
                    dg::network_stack_allocation::NoExceptAllocation<model::ticket_id_t[]> ticket_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<ResponseObserverInterface *, exception_t>[]> observer_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        ticket_id_arr[i] = base_response_arr[i].ticket_id;
                    }

                    this->ticket_controller->steal_observer(ticket_id_arr.get(), sz, observer_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (!observer_arr[i].has_value()){
                            continue;
                        }

                        observer_arr[i].value()->maybedeferred_memory_ordering_fetch(std::move(base_response_arr[i].response));
                    }

                    std::atomic_thread_fence(std::memory_order_release);

                    for (size_t i = 0u; i < sz; ++i){
                        if (!observer_arr[i].has_value()){
                            continue;
                        }

                        observer_arr[i].value()->maybedeferred_memory_ordering_fetch_close(static_cast<void *>(&base_response_arr[i].response));
                    }
                }
            };
    };

    class OutBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RequestContainerInterface> request_container;
            dg::network_kernel_mailbox::transmit_option_t transmit_opt;
            uint32_t channel;
            size_t mailbox_feed_cap;

        public:

            OutBoundWorker(std::shared_ptr<RequestContainerInterface> request_container,
                           dg::network_kernel_mailbox::transmit_option_t transmit_opt,
                           uint32_t channel,
                           size_t mailbox_feed_cap) noexcept: request_container(std::move(request_container)),
                                                              transmit_opt(transmit_opt),
                                                              channel(channel),
                                                              mailbox_feed_cap(mailbox_feed_cap){}

            bool run_one_epoch() noexcept{

                dg::vector<model::InternalRequest> request_vec = this->request_container->pop();

                auto feed_resolutor             = InternalFeedResolutor{};
                feed_resolutor.channel          = this->channel;
                feed_resolutor.transmit_opt     = this->transmit_opt;

                size_t trimmed_mailbox_feed_cap = std::min(static_cast<size_t>(request_vec.size()), this->mailbox_feed_cap);
                size_t feeder_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_mailbox_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_mailbox_feed_cap, feeder_mem.get()));

                for (auto& request: request_vec){
                    std::expected<dg::network_kernel_mailbox::Address, exception_t> addr = dg::network_uri_encoder::extract_mailbox_addr(request.request.requestee_uri);

                    if (!addr.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(addr.error()));
                        continue;
                    }

                    std::expected<dg::string, exception_t> bstream = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_serialize<dg::string, model::InternalRequest>)(request, model::INTERNAL_REQUEST_SERIALIZATION_SECRET);

                    if (!bstream.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(bstream.error()));
                        continue;
                    }

                    auto feed_arg       = dg::network_kernel_mailbox::MailBoxArgument{.to       = addr.value(),
                                                                                      .content  = std::move(bstream.value())};

                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), std::move(feed_arg));
                }

                return true;
            }
        
        private:

            struct InternalFeedResolutor{
                uint32_t channel;
                dg::network_kernel_mailbox::transmit_option_t transmit_opt;

                void push(std::move_iterator<dg::network_kernel_mailbox::MailBoxArgument *> mailbox_arg, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    dg::network_kernel_mailbox::send(mailbox_arg, sz, exception_arr.get(), this->transmit_opt);

                    for (size_t i = 0u; i < sz; ++I){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            };
    };

    class ExpiryWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<TicketControllerInterface> ticket_controller;
            std::shared_ptr<TicketTimeoutManagerInterface> ticket_timeout_manager;
            size_t timeout_consume_sz;
            size_t ticketcontroller_observer_steal_cap;
            size_t busy_timeout_consume_sz;  

        public:

            ExpiryWorker(std::shared_ptr<TicketControllerInterface> ticket_controller,
                         std::shared_ptr<TicketTimeoutManagerInterface> ticket_timeout_manager,
                         size_t timeout_consume_sz,
                         size_t ticketcontroller_observer_steal_sz,
                         size_t busy_timeout_consume_sz) noexcept: ticket_controller(std::move(ticket_controller)),
                                                                   ticket_timeout_manager(std::move(ticket_timeout_manager)),
                                                                   timeout_consume_sz(timeout_consume_sz),
                                                                   ticketcontroller_observer_steal_cap(ticketcontroller_observer_steal_cap),
                                                                   busy_timeout_consume_sz(busy_timeout_consume_sz){}

            bool run_one_epoch() noexcept{

                size_t expired_ticket_arr_cap       = this->timeout_consume_sz;
                size_t expired_ticket_arr_sz        = {};
                dg::network_stack_allocation::NoExceptAllocation<model::ticket_id_t[]> expired_ticket_arr(expired_ticket_arr_cap);
                this->ticket_timeout_manager->get_expired_ticket(expired_ticket_arr.get(), expired_ticket_arr_sz, expired_ticket_arr_cap);

                auto feed_resolutor                 = InternalFeedResolutor{};
                feed_resolutor.ticket_controller    = this->ticket_controller.get();

                size_t trimmed_observer_steal_cap   = std::min(this->ticketcontroller_observer_steal_cap, expired_ticket_arr_sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_observer_steal_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_observer_steal_cap, feeder_mem.get()));

                for (size_t i = 0u; i < expired_ticket_arr_sz; ++i){
                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), expired_ticket_arr[i]);
                }

                return expired_ticket_arr_sz >= this->busy_timeout_consume_sz;
            }
        
        private:

            struct InternalFeedResolutor: dg::network_producer_consumer::ConsumerInterface<model::ticket_id_t>{

                TicketControllerInterface * ticket_controller;

                void push(std::move_iterator<model::ticket_id_t *> ticket_id_arr, size_t ticket_id_arr_sz) noexcept{

                    model::ticket_id_t * base_ticket_id_arr = ticket_id_arr.base();
                    dg::network_stack_allocation<std::expected<ResponseObserverInterface *, exception_t>[]> stolen_response_observer_arr(ticket_id_arr_sz);
                    dg::network_stack_allocation<std::expected<Response, exception_t>[]> timeout_response_arr(ticket_id_arr_sz);

                    std::fill(timeout_response_arr.get(), std::next(timeout_response_arr.get(), ticket_id_arr_sz), std::expected<Response, exception_t>(std::unexpected(dg::network_exception::REST_TIMEOUT)));
                    this->ticket_controller->steal_observer(base_ticket_id_arr, ticket_id_arr_sz, stoken_response_observer_arr.get());

                    for (size_t i = 0u; i < ticket_id_arr_sz; ++i){
                        if (!stoken_response_observer_arr[i].has_value()){
                            continue;
                        }

                        stolen_response_observer_arr[i].value()->maybedeferred_memory_ordering_fetch(std::move(timeout_response_arr[i]));
                    }

                    std::atomic_thread_fence(std::memory_order_release);

                    for (size_t i = 0u; i < ticket_id_arr_sz; ++i){
                        if (!stolen_response_observer_arr[i].has_value()){
                            continue;
                        }

                        stolen_response_observer_arr[i].value()->maybedeferred_memory_ordering_fetch_close(static_cast<void *>(&timeout_response_arr[i]));
                    }
                }
            };
    };

    //alright, we'll do a shared_ptr scheme to close_tickets
    //std::unique_ptr<ticket_id_t[]>, an internalized raii_ticket_response holding a reference to a batch_close_ticket shared_ptr
    //this is to guarantee that our ticket close is post the synchronization, thus there are no unsafe memory accesses

    class RestController: public virtual RestControllerInterface{

        private:

            std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::shared_ptr<RequestContainerInterface> request_container;
            std::shared_ptr<TicketControllerInterface> ticket_controller;
            std::shared_ptr<TicketTimeoutManagerInterface> ticket_timeout_manager;
            stdx::hdi_container<size_t> max_consume_per_load;

        public:

            RestController(std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                           std::shared_ptr<RequestContainerInterface> request_container,
                           std::shared_ptr<TicketControllerInterface> ticket_controller,
                           std::shared_ptr<TicketTimeoutManagerInterface> ticket_timeout_manager,
                           stdx::hdi_container<size_t> max_consume_per_load) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                                       request_container(std::move(request_container)),
                                                                                       ticket_controller(std::move(ticket_controller)),
                                                                                       ticket_timeout_manager(std::move(ticket_timeout_manager)),
                                                                                       max_consume_per_load(std::move(max_consume_per_load)){}

            //this is hard to write
            void request(std::move_iterator<model::ClientRequest *> client_request_arr, size_t sz, std::expected<std::unique_ptr<ResponseInterface>, exception_t> * response_arr) noexcept{

                // auto timepoint = this->expiry_factory->get_expiry(rq.timeout);

                // if (!timepoint.has_value()){
                //     return std::unexpected(timepoint.error());
                // }

                // std::shared_ptr<RequestResponse> response                   = std::make_shared<RequestResponse>(timepoint.value()); //internalize allocations
                // std::expected<model::ticket_id_t, exception_t> ticket_id    = this->ticket_controller->get_ticket(response);

                // if (!ticket_id.has_value()){
                //     return std::unexpected(ticket_id.error());
                // }

                // auto internal_request   = InternalRequest{std::move(rq), ticket_id.value()};
                // exception_t err         = this->request_container->push(std::move(internal_request));

                // if (dg::network_exception::is_failed(err)){
                //     this->ticket_controller->close_ticket(ticket_id.value());
                //     return std::unexpected(err);
                // }

                // return std::unique_ptr<ResponseInterface>(std::make_unique<RAIITicketResponse>(std::move(response), make_raii_ticket(ticket_id.value(), this->ticket_controller)));
            }

            auto batch_request(std::move_iterator<model::ClientRequest *> client_request_arr, size_t sz) noexcept -> std::expected<std::unique_ptr<BatchResponseInterface>, exception_t>{

            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load.value;
            }

        private:

            class RAIITicketResponse: public virtual ResponseInterface{

                private:

                    std::unique_ptr<RequestResponse> base;
                    std::shared_ptr<void> ticket_resource;

                public:

                    RaiiTicketResponse(std::unique_ptr<RequestResponse> base, 
                                       std::shared_ptr<void> ticket_resource) noexcept: base(std::move(base)),
                                                                                        ticket(std::move(ticket)){}

                    auto response() noexcept -> std::expected<Response, exception_t>{

                        auto rs                 = this->base->response();
                        this->ticket_resource   = nullptr;
                        return rs;
                    }
            };

            class RAIITicketBatchResponse: public virtual BatchResponseInterface{

                private:

                    std::unique_ptr<BatchRequestResponse> base;
                    std::shared_ptr<void> ticket_resource;
                
                public:

                    RAIITicketBatchResponse(std::unique_ptr<BatchRequestResponse> base,
                                            std::shared_ptr<void> ticket_resource) noexcept: base(std::move(base)),
                                                                                             ticket_resource(std::move(ticket_resource)){}

                    auto response() noexcept -> std::expected<dg::vector<std::expected<Response, exception_t>>, exception_t>{

                        auto rs                 = this->base->response();
                        this->ticket_resource   = nullptr;
                        return rs;
                    }
            };
    };

    //we are reducing the serialization overheads of ticket_center
    //this is to utilize CPU resource + CPU efficiency by running affined task
    //we'll do partial load balancing, Ive yet to know what that is
    //because I think random hash in conjunction with max_consume_size should be representative

    class DistributedRestController: public virtual RestControllerInterface{

        private:

            std::unique_ptr<std::unique_ptr<RestControllerInterface>>[]> rest_controller_arr;
            size_t pow2_rest_controller_arr_sz;
            stdx::hdi_container<size_t> max_consume_per_load;

        public:

            DistributedRestController(std::unique_ptr<std::unique_ptr<RestControllerInterface>[]> rest_controller_arr,
                                      size_t pow2_rest_controller_arr_sz,
                                      stdx::hdi_container<size_t> max_consume_per_load) noexcept: rest_controller_arr(std::move(rest_controller_arr)),
                                                                                                  pow2_rest_controller_arr_sz(pow2_rest_controller_arr_sz),
                                                                                                  max_consume_per_load(std::move(max_consume_per_load)){} 

            void request(std::move_iterator<model::ClientRequest *> request_arr, size_t request_arr_sz, std::expected<std::unique_ptr<ResponseInterface>, exception_t> * response_arr) noexcept{

                size_t random_clue  = dg::network_randomizer::randomize_int<size_t>();
                size_t idx          = random_clue & (this->pow2_rest_controller_arr_sz - 1u);

                this->rest_controller_vec[idx]->request(request_arr, request_arr_sz, response_arr);
            }

            auto batch_request(std::move_iterator<model::ClientRequest *> request_arr, size_t request_arr_sz) -> std::expected<std::unique_ptr<BatchResponseInterface>, exception_t>{

                size_t random_clue  = dg::network_randomizer::randomize_int<size_t>();
                size_t idx          = random_clue & (this->pow2_rest_controller_arr_sz - 1u);

                return this->rest_controller_vec[idx]->request(request_arr, request_arr_sz);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load.value;
            }
    };
}

#endif
