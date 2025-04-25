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

    static inline constexpr uint32_t INTERNAL_REQUEST_SERIALIZATION_SECRET  = 3312354321ULL;
    static inline constexpr uint32_t INTERNAL_RESPONSE_SERIALIZATION_SECRET = 3554488158ULL;

    struct CacheID{
        std::array<char, 8u> ip;
        std::array<char, 8u> native_cache_id;
        std::optional<std::array<char, 8u>> bucket_hint; //this is complicated, we need to increase the number of buckets in order to do bucket_hint, we only fetch 128 bytes of memory for 4 buckets, this is an extremely fast insert
                                                         //find() is another problem, we hardly ever find this thing, it's like 1 of 16384 chance to find a cached response, so ... we only worry about the insert for now 

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(ip, native_cache_id, bucket_hint);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(ip, native_cache_id, bucket_hint);
        }
    };

    using cache_id_t    = CacheID; 

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

    //the reason that we are rewriting these so many times is because each container is different from another
    //it's real
    //every application has to twist their own kind of container, then there is version control problems, so its best to internalize the dependencies, and make it a unit of deliverable 

    struct CacheControllerInterface{
        virtual ~CacheControllerInterface() noexcept = default;
        virtual void get_cache(cache_id_t * cache_id_arr, size_t sz, std::expected<std::optional<Response>, exception_t> *) noexcept = 0; //we'll think about making the exclusion responsibility this component responsibility later
        virtual void insert_cache(cache_id_t * cache_id_arr, std::move_iterator<Response *> response_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept = 0;
        virtual void contains(cache_id_t * cache_id_arr, size_t sz, bool * rs_arr) noexcept = 0;
        virtual void clear() noexcept = 0;
        virtual auto size() const noexcept -> size_t = 0;
        virtual auto capacity() const noexcept -> size_t = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    }; 

    struct InfiniteCacheControllerInterface{
        virtual ~InfiniteCacheControllerInterface() noexcept = default;
        virtual void get_cache(cache_id_t * cache_id_arr, size_t sz, std::expected<std::optional<Response>, exception_t> *) noexcept = 0; //we'll think about making the exclusion responsibility this component responsibility later
        virtual void insert_cache(cache_id_t * cache_id_arr, std::move_iterator<Response *> response_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    struct CacheUniqueWriteControllerInterface{
        virtual ~CacheUniqueWriteControllerInterface() noexcept = default;
        virtual void thru(cache_id_t * cache_id_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept = 0;
        virtual void contains(cache_id_t * cache_id_arr, size_t sz, bool * rs_arr) noexcept = 0;
        virtual void clear() noexcept = 0;
        virtual auto size() const noexcept -> size_t = 0;
        virtual auto capacity() const noexcept -> size_t = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    struct InfiniteCacheUniqueWriteControllerInterface{
        virtual ~InfiniteCacheUniqueWriteControllerInterface() noexcept = default;
        virtual void thru(cache_id_t * cache_id_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept = 0;
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
        virtual ~ResponseObserverInterface() noexcept = default;
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
        virtual auto open_ticket(size_t sz, model::ticket_id_t * rs) noexcept -> exception_t = 0;
        virtual void assign_observer(model::ticket_id_t * ticket_id_arr, size_t sz, ResponseObserverInterface ** assigning_observer_arr, std::expected<bool, exception_t> * exception_arr) noexcept = 0;
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
            std::shared_ptr<InfiniteCacheControllerInterface> request_cache_controller;
            std::shared_ptr<InfiniteCacheUniqueWriteControllerInterface> cachewrite_uex_controller;
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
                                  std::shared_ptr<InfiniteCacheControllerInterface> request_cache_controller,
                                  std::shared_ptr<InfiniteCacheUniqueWriteControllerInterface> cachewrite_uex_controller,
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

                        auto arg = dg::network_kernel_mailbox::MailBoxArgument{.to          = base_data_arr[i].to,
                                                                               .content     = std::move(serialized_response.value()),
                                                                               .priority    = base_data_arr[i].priority};

                        dg::network_producer_consumer::delvrsrv_deliver(this->mailbox_feeder, std::move(arg));
                    }
                }
            };

            struct InternalCacheMapFeedArgument{
                cache_id_t cache_id;
                Response response;
            };

            struct InternalCacheMapFeedResolutor: dg::network_producer_consumer::ConsumerInterface<InternalCacheMapFeedArgument>{

                InfiniteCacheControllerInterface * cache_controller;

                void push(std::move_iterator<InternalCacheMapFeedArgument *> data_arr, size_t sz) noexcept{

                    auto base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<Response[]> response_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> rs_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i]     = base_data_arr[i].cache_id;
                        response_id_arr[i]  = std::move(base_data_arr[i].response);
                    }

                    this->cache_controller->insert_cache(cache_id_arr.get(), std::make_move_iterator(response_id_arr.get()), sz, rs_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (!rs_arr[i].has_value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(rs_arr[i].error()));
                            continue;
                        }

                        if (!rs_arr[i].value()){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(dg::network_exception::REST_CACHEMAP_CORRUPTION));
                            continue;
                        }
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

                InfiniteCacheUniqueWriteControllerInterface * cachewrite_uex_controller;
                dg::network_producer_consumer::KVDeliveryHandle<InternalServerFeedResolutorArgument> * server_resolutor_feeder;
                dg::network_producer_consumer::DeliveryHandle<InternalMailBoxPrepArgument> * mailbox_prep_feeder;

                void push(std::move_iterator<InternalCacheServerFeedArgument *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> cache_write_response_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i] = base_data_arr[i].cache_id;
                    }

                    this->cachewrite_uex_controller->thru(cache_id_arr.get(), sz, cache_write_response_arr.get());

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

                CacheControllerInterface * cache_controller;

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

    struct CacheNormalHasher{

        constexpr auto operator()(const CacheID& cache_id) const noexcept -> size_t{

            return dg::network_hash::hash_reflectible(cache_id);
        }
    };

    struct CacheBucketHintHasher{

        constexpr auto operator()(const CacheID& cache_id) const noexcept -> size_t{

            if (cache_id.bucket_hint.has_value()){
                return dg::network_hash::hash_reflectible(cache_id.bucket_hint.value());
            } else{
                return dg::network_hash::hash_reflectible(cache_id);
            }
        }
    };

    struct CacheIPHasher{

        constexpr auto operator()(const CacheID& cache_id) const noexcept -> size_t{

            return dg::network_hash::hash_reflectible(cache_id.ip);
        }
    };

    template <class Hasher>
    class CacheController: public virtual CacheControllerInterface{

        private:

            dg::unordered_unstable_map<cache_id_t, Response, Hasher> cache_map;
            size_t cache_map_cap;
            size_t max_response_sz;
            size_t max_consume_per_load;

        public:

            CacheController(dg::unordered_unstable_map<cache_id_t, Response, Hasher> cache_map,
                            size_t cache_map_cap,
                            size_t max_response_sz,
                            size_t max_consume_per_load) noexcept: cache_map(std::move(cache_map)),
                                                                   cache_map_cap(cache_map_cap),
                                                                   max_response_sz(max_response_sz),
                                                                   max_consume_per_load(std::move(max_consume_per_load)){}

            void get_cache(cache_id_t * cache_id_arr, size_t sz, std::expected<std::optional<Response>, exception_t> * rs_arr) noexcept{

                for (size_t i = 0u; i < sz; ++i){
                    auto map_ptr = stdx::to_const_reference(this->cache_map).find(cache_id_arr[i]);

                    if (map_ptr == this->cache_map.end()){
                        rs_arr[i] = std::optional<Response>(std::nullopt);
                    } else{
                        std::expected<Response, exception_t> cpy_response = dg::network_exception::cstyle_initialize<Response>(map_ptr->second);
                        
                        if (!cpy_response.has_value()){
                            rs_arr[i] = std::unexpected(cpy_response.error());
                        } else{
                            static_assert(std::is_nothrow_move_constructible_v<Response> && std::is_nothrow_move_assignable_v<Response>);
                            rs_arr[i] = std::optional<Response>(std::move(cpy_resonse.value()));
                        }
                    }
                }
            }

            void insert_cache(cache_id_t * cache_id_arr, std::move_iterator<Response *> response_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto base_response_arr = response_arr.base(); 

                for (size_t i = 0u; i < sz; ++i){
                    if (this->cache_map.size() == this->cache_map_cap){
                        rs_arr[i] = std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                        continue;
                    }

                    if (dg::network_compact_serializer::size(base_response_arr[i]) > this->max_response_sz){
                        rs_arr[i] = std::unexpected(dg::network_exception::REST_CACHE_MAX_RESPONSE_SIZE_REACHED);
                        continue;
                    }

                    static_assert(std::is_nothrow_move_constructible_v<Response> && std::is_nothrow_move_assignable_v<Response>);

                    auto insert_token       = std::make_pair(cache_id_arr[i], std::move(base_response_arr[i]));
                    auto [map_ptr, status]  = this->cache_map.insert(std::move(insert_token));
                    rs_arr[i]               = status;
                }
            }

            void contains(cache_id_t * cache_id_arr, size_t sz, bool * rs_arr) noexcept{

                for (size_t i = 0u; i < sz; ++i){
                    rs_arr[i] = this->cache_map.contains(cache_id_arr[i]);
                }
            }

            void clear() noexcept{

                this->cache_map.clear();
            }

            auto size() const noexcept -> size_t{

                return this->cache_map.size();
            }

            auto capacity() const noexcept -> size_t{

                return this->cache_map_cap;
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load.value;
            }
    };

    class MutexControlledCacheController: public virtual CacheControllerInterface{

        private:

            std::unique_ptr<CacheControllerInterface> base;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            MutexControlledCacheController(std::unique_ptr<CacheControllerInterface> base,
                                           std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                      mtx(std::move(mtx)){}

            void get_cache(cache_id_t * cache_id_arr, size_t sz, std::expected<std::optional<Response, exception_t>> * rs_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->base->get_cache(cache_id_arr, sz, rs_arr);
            }

            void insert_cache(cache_id_t * cache_id_arr, std::move_iterator<Response *> response_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->base->insert_cache(cache_id_arr, response_arr, sz, rs_arr);
            }

            void clear() noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->base->clear();
            }

            void contains(cache_id_t * cache_id_arr, size_t sz, bool * rs_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->base->contains(cache_id_arr, sz, rs_arr);
            }

            auto size() const noexcept -> size_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                return this->base->size();
            }

            auto capacity() const noexcept -> size_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                return this->base->capacity();
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    class SwitchingCacheController: public virtual InfiniteCacheControllerInterface{

        private:

            std::unique_ptr<CacheControllerInterface> left;
            std::unique_ptr<CacheControllerInterface> right;

            bool operating_side;
            size_t switch_population_counter;
            size_t switch_population_threshold; 
            size_t getcache_feed_cap;
            size_t insertcache_feed_cap;

            size_t max_consume_per_load;

        public:

            SwitchingCacheController(std::unique_ptr<CacheControllerInterface> left,
                                     std::unique_ptr<CacheControllerInterface> right,
                                     bool operating_side,
                                     size_t switch_population_counter,
                                     size_t switch_population_threshold,
                                     size_t getcache_feed_cap,
                                     size_t insertcache_feed_cap,
                                     size_t max_consume_per_load) noexcept: left(std::move(left)),
                                                                            right(std::move(right)),
                                                                            operating_side(std::move(operating_side)),
                                                                            switch_population_counter(std::move(switch_population_counter)),
                                                                            switch_population_threshold(std::move(switch_population_threshold)),
                                                                            getcache_feed_cap(getcache_feed_cap),
                                                                            insertcache_feed_cap(insertcache_feed_cap),
                                                                            max_consume_per_load(max_consume_per_load){}

            void get_cache(cache_id_t * cache_id_arr, size_t sz, std::expected<std::optional<Response>, exception_t> * rs_arr) noexcept{

                CacheControllerInterface * major_cache_controller   = nullptr;
                CacheControllerInterface * minor_cache_controller   = nullptr;

                if (this->operating_side == false){
                    major_cache_controller  = this->right.get();
                    minor_cache_controller  = this->left.get();
                }  else{
                    major_cache_controller  = this->left.get();
                    minor_cache_controller  = this->right.get();
                }

                major_cache_controller->get_cache(cache_id_arr, sz, rs_arr);

                auto feed_resolutor                 = InternalGetCacheFeedResolutor{};
                feed_resolutor.dst                  = minor_cache_controller; 

                size_t trimmed_getcache_feed_cap    = std::min(std::min(this->getcache_feed_cap, minor_cache_controller->max_consume_size()), sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_getcache_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_getcache_feed_cap, feeder_mem.get())); 

                for (size_t i = 0u; i < sz; ++i){
                    if (!rs_arr[i].has_value()){
                        continue;
                    }

                    if (rs_arr[i].value().has_value()){
                        continue;
                    }

                    auto feed_arg       = InternalGetCacheFeedArgument{};
                    feed_arg.cache_id   = cache_id_arr[i];
                    feed_arg.rs         = std::next(rs_arr, i);

                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), feed_arg);
                }
            }

            void insert_cache(cache_id_t * cache_id_arr, std::move_iterator<Response *> response_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept{

                //range is the most important thing in this world

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }                    
                }

                Response * base_response_arr = response_arr.base();

                if (this->switch_population_counter + sz >= this->switch_population_threshold){
                    this->internal_dispatch_switch();
                }

                CacheControllerInterface * current_cache_controller = nullptr;
                CacheControllerInterface * other_cache_controller   = nullptr;

                if (this->operating_side == false){
                    current_cache_controller    = this->left.get();
                    other_cache_controller      = this->right.get();
                } else{
                    current_cache_controller    = this->right.get();
                    other_cache_controller      = this->left.get();
                }

                auto feed_resolutor                 = InternalInsertCacheFeedResolutor{};
                feed_resolutor.insert_incrementor   = &this->switch_population_counter;
                feed_resolutor.dst                  = current_cache_controller; 

                size_t trimmed_insertcache_feed_cap = std::min(std::min(this->insertcache_feed_cap, current_cache_controller->max_consume_size()), sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_insertcache_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_insertcache_feed_cap, feeder_mem.get())); 

                dg::network_stack_allocation::NoExceptRawAllocation<bool[]> contain_status_arr(sz);
                other_cache_controller->contains(cache_id_arr, sz, contain_status_arr.get());

                for (size_t i = 0u; i < sz; ++i){
                    if (contain_status_arr[i]){
                        rs_arr[i] = false; 
                        continue;
                    }

                    auto feed_arg   = InternalInsertCacheFeedArgument{.cache_id     = cache_id_arr[i],
                                                                      .rs           = std::next(rs_arr, i),
                                                                      .response     = std::move(base_response_arr[i]),
                                                                      .fallback_ptr = std::next(base_response_arr, i)};

                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), std::move(feed_arg));
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load.value;
            }
        
        private:

            struct InternalInsertCacheFeedArgument{
                cache_id_t cache_id;
                std::expected<bool, exception_t> * rs;
                Response response;
                Response * fallback_ptr;
            };

            struct InternalInsertCacheFeedResolutor: dg::network_producer_consumer::ConsumerInterface<InternalInsertCacheFeedArgument>{

                size_t * insert_incrementor;
                CacheControllerInterface * dst;

                void push(std::move_iterator<InternalInsertCacheFeedArgument *> data_arr, size_t sz) noexcept{

                    auto base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<Response[]> response_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>> rs_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i] = base_data_arr[i].cache_id;
                        response_arr[i] = std::move(base_data_arr[i].response);
                    }

                    this->dst->insert_cache(cache_id_arr.get(), std::make_move_iterator(response_arr.get()), sz, rs_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        *base_data_arr[i].rs = rs_arr[i];

                        if (!rs_arr[i].has_value()){
                            *base_data_arr[i].fallback_ptr = std::move(response_arr[i]);
                            continue;
                        }

                        if (!rs_arr[i].value()){
                            continue;
                        }

                        *this->insert_incrementor += 1u;
                    }     
                }
            };

            struct InternalGetCacheFeedArgument{
                cache_id_t cache_id;
                std::expected<std::optional<Response>, exception_t> * rs;
            };

            struct InternalGetCacheFeedResolutor: dg::network_producer_consumer::ConsumerInterface<InternalGetCacheFeedArgument>{

                CacheControllerInterface * dst;

                void push(std::move_iterator<InternalGetCacheFeedArgument *> data_arr, size_t sz) noexcept{

                    auto base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<std::optional<Response>, exception_t>[]> rs_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i] = base_data_arr[i].cache_id;
                    }

                    this->dst->get_cache(cache_id_arr.get(), sz, rs_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        static_assert(std::is_nothrow_move_assignable_v<std::expected<std::optional<Response>, exception_t>>);
                        *base_data_arr[i].rs = std::move(rs_arr[i]);
                    }
                }
            };

            void internal_dispatch_switch() noexcept{

                if (this->operating_side == false){
                    this->right->clear();
                } else{
                    this->left->clear();
                }

                this->operating_side            = !this->operating_side;
                this->switch_population_counter = 0u;
            }
    }; 

    class MutexControlledInfiniteCacheController: public virtual InfiniteCacheControllerInterface{

        private:

            std::unique_ptr<InifiniteCacheControllerInterface> base;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            MutexControlledInfiniteCacheController(std::unique_ptr<InfiniteCacheControllerInterface> base,
                                                   std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                              mtx(std::move(mtx)){}

            void get_cache(cache_id_t * cache_id_arr, size_t sz, std::expected<std::optional<Response>, exception_t> * rs_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->base->get_cache(cache_id_arr, sz, rs_arr);
            }

            void insert_cache(cache_id_t * cache_id_arr, std::move_iterator<Response *> response_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->base->insert_cache(cache_id_arr, response_arr, sz, rs_arr);
            }            

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    template <class Hasher>
    class DistributedCacheController: public virtual InfiniteCacheControllerInterface{

        private:

            std::unique_ptr<std::unique_ptr<InfiniteCacheControllerInterface>[]> cache_controller_arr;
            size_t pow2_cache_controller_arr_sz;
            size_t getcache_keyvalue_feed_cap;
            size_t insertcache_keyvalue_feed_cap;
            size_t max_consume_per_load; 

        public:

            DistributedCacheController(std::unique_ptr<std::unique_ptr<InfiniteCacheControllerInterface>[]> cache_controller_arr,
                                       size_t pow2_cache_controller_arr_sz,
                                       size_t getcache_keyvalue_feed_cap,
                                       size_t insertcache_keyvalue_feed_cap,
                                       size_t max_consume_per_load) noexcept: cache_controller_arr(std::move(cache_controller_arr)),
                                                                              pow2_cache_controller_arr_sz(pow2_cache_controller_arr_sz),
                                                                              getcache_keyvalue_feed_cap(getcache_keyvalue_feed_cap),
                                                                              insertcache_keyvalue_feed_cap(insertcache_keyvalue_feed_cap),
                                                                              max_consume_per_load(std::move(max_consume_per_load)){}

            void get_cache(cache_id_t * cache_id_arr, size_t sz, std::expected<std::optional<Response>, exception_t> * rs_arr) noexcept{

                auto feed_resolutor                 = InternalGetCacheFeedResolutor{};
                feed_resolutor.cache_controller_arr = this->cache_controller_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->getcache_keyvalue_feed_cap, sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    size_t partitioned_idx  = Hasher{}(cache_id_arr[i]) & (this->pow2_cache_controller_arr_sz - 1u);

                    auto feed_arg           = InternalGetCacheFeedArgument{};
                    feed_arg.cache_id       = cache_id_arr[i];
                    feed_arg.rs_ptr         = std::next(rs_arr, i);

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }
            }

            void insert_cache(cache_id_t * cache_id_arr, std::move_iterator<Response *> response_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                auto base_response_arr              = response_arr.base();

                auto feed_resolutor                 = InternalCacheInsertFeedResolutor{};
                feed_resolutor.cache_controller_arr = this->cache_controller_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->insertcache_keyvalue_feed_cap, sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    size_t partitioned_idx  = Hasher{}(cache_id_arr[i]) & (this->pow2_cache_controller_arr_sz - 1u);
                    auto feed_arg           = InternalCacheInsertFeedArgument{.cache_id     = cache_id_arr[i],
                                                                              .response     = std::move(base_response_arr[i]),
                                                                              .fallback_ptr = std::next(base_response_arr, i),
                                                                              .rs           = std::next(rs_arr, i)};

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, std::move(feed_arg));
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load;
            }

        private:

            struct InternalGetCacheFeedArgument{
                cache_id_t cache_id;
                std::expected<std::optional<Response>, exception_t> * rs_ptr;
            };

            struct InternalGetCacheFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalGetCacheFeedArgument>{

                std::unique_ptr<InfiniteCacheControllerInterface> * cache_controller_arr;

                void push(const size_t& idx, std::move_iterator<InternalGetCacheFeedArgument *> data_arr, size_t sz) noexcept{

                    auto base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<std::optional<Response>, exception_t>[]> rs_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i] = base_data_arr[i].cache_id;
                    }

                    this->cache_controller[idx]->get_cache(cache_id_arr.get(), sz, rs_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        *base_data_arr[i].rs_ptr = std::move(rs_arr[i]);
                    }
                }
            };

            struct InternalCacheInsertFeedArgument{
                cache_id_t cache_id;
                Response response;
                Response * fallback_ptr;
                std::expected<bool, exception_t> * rs;
            };

            struct InternalCacheInsertFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalCacheInsertFeedArgument>{

                std::unique_ptr<InfiniteCacheControllerInterface> * cache_controller_arr;

                void push(const size_t& idx, std::move_iterator<InternalCacheInsertFeedArgument *> data_arr, size_t sz) noexcept{

                    auto base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<Response[]> response_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> rs_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i] = base_data_arr[i].cache_id;
                        response_arr[i] = std::move(base_data_arr[i].response);
                    }

                    this->cache_controller_arr[idx]->insert_cache(cache_id_arr.get(), std::make_move_iterator(response_arr.get()), sz, rs_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        *base_data_arr[i].rs = rs_arr[i];

                        if (!rs_arr[i].has_value()){
                            *base_data_arr[i].fallback_ptr = std::move(response_arr[i]);
                            continue;
                        }
                    }
                }
            };
    };

    template <class Hasher>
    class CacheUniqueWriteController: public virtual CacheUniqueWriteControllerInterface{

        private:

            dg::unordered_unstable_set<cache_id_t, Hasher> cache_id_set;
            size_t cache_id_set_cap;
            size_t max_consume_per_load;

        public:

            CacheUniqueWriteController(dg::unordered_unstable_set<cache_id_t, Hasher> cache_id_set,
                                       size_t cache_id_set_cap,
                                       size_t max_consume_per_load) noexcept: cache_id_set(std::move(cache_id_set)),
                                                                              cache_id_set_cap(cache_id_set_cap),
                                                                              max_consume_per_load(max_consume_per_load){}

            void thru(cache_id_t * cache_id_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                for (size_t i = 0u; i < sz; ++i){
                    auto set_ptr = this->cache_id_set.find(cache_id_arr[i]);

                    if (set_ptr != this->cache_id_set.end()){
                        rs_arr[i] = false; //false, found, already thru
                        continue;
                    }

                    //unique, try to insert

                    if (this->cache_id_set.size() == this->cache_id_set_cap){
                        //cap reached, return cap exception, no_actions 
                        rs_arr[i] = std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                        continue;
                    }

                    auto [_, status]  = this->cache_id_set.insert(cache_id_arr[i]);
                    dg::network_exception_handler::dg_assert(status);
                    rs_arr[i] = true; //thru, uniqueness acknowledged by cache_id_set
                }
            }

            void contains(cache_id_t * cache_id_arr, size_t sz, bool * rs_arr) noexcept{

                for (size_t i = 0u; i < sz; ++i){
                    rs_arr[i] = this->cache_id_set.contains(cache_id_arr[i]);
                }
            }

            void clear() noexcept{

                this->cache_id_set.clear();
            }

            auto size() const noexcept -> size_t{

                return this->cache_id_set.size();
            }

            auto capacity() const noexcept -> size_t{

                return this->cache_id_set_cap;
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load;
            }
    };

    class MutexControlledCacheWriteExclusionController: public virtual CacheUniqueWriteControllerInterface{

        private:

            std::unique_ptr<CacheUniqueWriteControllerInterface> base;
            std::unique_ptr<std::mutex> mtx;

        public:

            MutexControlledCacheWriteExclusionController(std::unique_ptr<CacheUniqueWriteControllerInterface> base,
                                                         std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                                    mtx(std::move(mtx)){}

            void thru(cache_id_t * cache_id_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->base->thru(cache_id_arr, sz, rs_arr);
            }

            void contains(cache_id_t * cache_id_arr, size_t sz, bool * rs_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->base->contains(cache_id_arr, sz, rs_arr);
            }

            void clear() noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->base->clear();
            }

            auto size() const noexcept -> size_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                return this->base->size();
            }

            auto capacity() const noexcept -> size_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                return this->base->capacity();
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    class SwitchingCacheWriteExclusionController: public virtual InfiniteCacheUniqueWriteControllerInterface{

        private:

            std::unique_ptr<CacheUniqueWriteControllerInterface> left_controller;
            std::unique_ptr<CacheUniqueWriteControllerInterface> right_controller;

            bool operating_side;
            size_t switch_population_counter;
            size_t switch_population_threshold;
            size_t thru_feed_cap;
            size_t max_consume_per_load;

        public:

            SwitchingCacheWriteExclusionController(std::unique_ptr<CacheUniqueWriteControllerInterface> left_controller,
                                                   std::unique_ptr<CacheUniqueWriteControllerInterface> right_controller,
                                                   bool operating_side,
                                                   size_t switch_population_counter,
                                                   size_t switch_population_threshold,
                                                   size_t thru_feed_cap,
                                                   size_t max_consume_per_load) noexcept: left_controller(std::move(left_controller)),
                                                                                          right_controller(std::move(right_controller)),
                                                                                          operating_side(operating_side),
                                                                                          switch_population_counter(switch_population_counter),
                                                                                          switch_population_threshold(switch_population_threshold),
                                                                                          thru_feed_cap(thru_feed_cap),
                                                                                          max_consumer_per_load(max_consumer_per_load){}

            void thru(cache_id_t * cache_id_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort():
                    }
                }

                if (this->switch_population_counter + sz >= this->switch_population_threshold){
                    this->internal_dispatch_switch();
                }

                CacheUniqueWriteControllerInterface * current_write_controller  = nullptr;
                CacheUniqueWriteControllerInterface * other_write_controller    = nullptr;

                if (this->operating_side == false){
                    current_write_controller    = this->left_controller.get();
                    other_write_controller      = this->right_controller.get();
                } else{
                    current_write_controller    = this->right_controller.get();
                    other_write_controller      = this->left_controller.get();
                }

                auto feed_resolutor                 = InternalThruWriteFeedResolutor{};
                feed_resolutor.thru_incrementor     = &this->switch_population_counter;
                feed_resolutor.dst                  = current_write_controller;

                size_t trimmed_thru_feed_cap        = std::min(std::min(this->thru_feed_cap, current_write_controller->max_consume_size()), sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_thru_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_thru_feed_cap, feeder_mem.get())); 

                dg::network_stack_allocation::NoExceptRawAllocation<bool[]> contain_status_arr(sz);
                other_write_controller->contains(cache_id_arr, sz, contain_status_arr.get());

                for (size_t i = 0u; i < sz; ++i){
                    if (contain_status_arr[i]){
                        rs_arr[i] = false; //already thru
                        continue;
                    }

                    auto feed_arg   = InternalThruWriteFeedArgument{.cache_id   = cache_id_arr[i],
                                                                    .rs         = std::next(rs_arr, i)};

                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), feed_arg);
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load;
            }
        
        private:

            struct InternalThruWriteFeedArgument{
                cache_id_t cache_id;
                std::expected<bool, exception_t> * rs;
            };

            struct InternalThruWriteFeedResolutor: dg::network_producer_consumer::ConsumerInterface<InternalThruWriteFeedArgument>{
                
                size_t * thru_incrementor;
                CacheUniqueWriteControllerInterface * dst;

                void push(std::move_iterator<InternalThruWriteFeedArgument *> data_arr, size_t sz) noexcept{

                    auto base_data_arr = data_arr.base();
                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> rs_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i] = base_data_arr[i].cache_id;
                    }

                    this->dst->thru(cache_id_arr.get(), sz, rs_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        *base_data_arr[i].rs = rs_arr[i];

                        if (!rs_arr[i].has_value()){
                            continue;
                        }

                        if (!rs_arr[i].value()){
                            continue;
                        }

                        *this->thru_incrementor += 1;
                    }
                }
            };

            void internal_dispatch_switch() noexcept{

                if (this->operating_side == false){
                    this->right_controller->clear();
                } else{
                    this->left_controller->clear();
                }

                this->operating_side            = !this->operating_side;
                this->switch_population_counter = 0u;
            }
    };

    class MutexControlledInfiniteCacheWriteExclusionController: public virtual InfiniteCacheUniqueWriteControllerInterface{

        private:

            std::unique_ptr<InfiniteCacheUniqueWriteControllerInterface> base;
            std::unique_ptr<std::mutex> mtx;

        public:

            MutexControlledInfiniteCacheWriteExclusionController(std::unique_ptr<InfiniteCacheUniqueWriteControllerInterface> base,
                                                                 std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                                            mtx(std::move(mtx)){}

            void thru(cache_id_t * cache_id_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                this->base->thru(cache_id_arr, sz, rs_arr);
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size():
            }
    };

    template <class Hasher>
    class DistributedUniqueCacheWriteController: public virtual InfiniteCacheUniqueWriteControllerInterface{

        private:

            std::unique_ptr<std::unique_ptr<InfiniteCacheUniqueWriteControllerInterface>[]> base_arr;
            size_t pow2_base_arr_sz;
            size_t thru_keyvalue_feed_cap;
            size_t max_consume_per_load;
        
        public:

            DistributedUniqueCacheWriteController(std::unique_ptr<std::unique_ptr<InfiniteCacheWriteControllerInterface>[]> base_arr,
                                                  size_t pow2_base_arr_sz) noexcept: base_arr(std::move(base_arr)),
                                                                                     pow2_base_arr_sz(pow2_base_arr_sz){}

            void thru(cache_id_t * cache_id_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept{

                auto feed_resolutor                     = InternalThruFeedResolutor{};
                feed_resolutor.controller_arr           = this->base_arr.get();

                size_t trimmed_thru_keyvalue_feed_cap   = std::min(this->thru_keyvalue_feed_cap, sz);
                size_t feeder_allocation_cost           = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_thru_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                             = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_thru_keyvalue_feed_cap, feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    size_t partitioned_idx  = Hasher{}(cache_id_arr[i]) & (this->pow2_base_arr_sz - 1u);
                    
                    auto feed_arg           = InternalThruFeedArgument{};
                    feed_arg.cache_id       = cache_id_arr[i];
                    feed_arg.rs_ptr         = std::next(rs_arr, i);

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load;
            }
        
        private:

            struct InternalThruFeedArgument{
                cache_id_t cache_id;
                std::expected<bool, exception_t> * rs_ptr;
            };

            struct InternalThruFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalThruFeedArgument>{

                std::unique_ptr<InfiniteCacheUniqueWriteControllerInterface> * controller_arr;

                void push(const size_t& idx, std::move_iterator<InternalThruFeedArgument *> data_arr, size_t sz) noexcept{

                    auto base_data_arr = data_arr.base();
                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> rs_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i] = base_data_arr[i].cache_id;
                    }

                    this->controller_arr[idx]->thru(cache_id_arr.get(), sz, rs_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        *base_data_arr[i].rs_ptr = rs_arr[i];
                    }
                }
            };
    };

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

                this->resp_vec[idx] = std::move(response);
            }

            void maybedeferred_memory_ordering_fetch_close(size_t idx, void * dirty_memory) noexcept{

                this->internal_maybedeferred_memory_ordering_fetch_close(idx, dirty_memory);
            }

            auto response() noexcept -> std::expected<dg::vector<std::expected<Response, exception_t>>, exception_t>{

                bool was_invoked = this->is_response_invoked.exchange(true, std::memory_order_relaxed);

                if (was_invoked){
                    return std::unexpected(dg::network_exception::REST_UNIQUE_RESOURCE_ACQUIRED);
                } 

                this->atomic_smp.wait(0, std::memory_order_acquire);

                return dg::vector<std::expected<Response, exception_t>>(std::move(this->resp_vec));
            }

        private:

            __attribute__((noipa)) void internal_maybedeferred_memory_ordering_fetch_close(size_t idx, void * dirty_memory) noexcept{

                intmax_t old = this->atomic_smp.fetch_add(1, std::memory_order_relaxed);

                if (old == 0){
                    this->atomic_smp.notify_one():
                }
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
            bool response_wait_responsibility_flag;

        public:

            BatchRequestResponse(size_t resp_sz): observer_arr(resp_sz),
                                                  base(resp_sz),
                                                  response_wait_responsibility_flag(true){

                for (size_t i = 0u; i < resp_sz; ++i){
                    this->observer_arr[i] = BatchRequestResponseBaseDesignatedObserver(&this->base, i);
                }
            }

            ~BatchRequestResponse() noexcept{

                this->wait_response();
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

            void release_response_wait_responsibility() noexcept{

                this->response_wait_responsibility_flag = false;
            }

            void wait_response() noexcept{

                if (this->response_wait_responsibility_flag){
                    stdx::empty_noipa(this->base->response(), this->observer_arr);
                }            
            }
    };

    auto make_batch_request_response(size_t resp_sz) noexcept -> std::expected<std::unique_ptr<BatchRequestResponse>, exception_t>{

        return dg::network_exception::to_cstyle_function(dg::network_allocation::std_make_unique<BatchRequestResponse, size_t>)(resp_sz);
    }

    class RequestResponse: public virtual ResponseInterface{

        private:

            RequestResponseBase base;
            RequestResponseBaseObserver observer;
            bool response_wait_responsibility_flag; 

        public:

            RequestResponse(): base(),
                               response_wait_responsibility_flag(true){

                this->observer = RequestResponseBaseObserver(&this->base);
            }

            ~ReleaseSafeRequestResponse() noexcept{

                this->wait_response();
            }

            auto response() noexcept -> std::expected<Response, exception_t>{

                return this->base->response();
            }

            auto get_observer() noexcept -> ResponseObserverInterface *{ //response observer is a unique_resource, such is an acquisition of this pointer must involve accurate acquire + release mechanisms like every other dg::string or dg::vector, etc.

                return &this->observer;
            }

            void release_response_wait_responsibility() noexcept{

                this->response_wait_responsibility_flag = false;
            }

            void wait_response() noexcept{

                if (this->response_wait_responsibility_flag){
                    stdx::empty_noipa(this->base->response(), this->observer);
                }
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

                std::binary_semaphore * releasing_smp = nullptr;

                exception_t err = [&]() noexcept{
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->waiting_queue.empty()){
                        auto [pending_smp, fetching_addr] = std::move(this->waiting_queue.front());
                        this->waiting_queue.pop_front();
                        *fetching_addr  = std::move(request);
                        std::atomic_signal_fence(std::memory_order_seq_cst);
                        releasing_smp   = pending_smp;

                        return dg::network_exception::SUCCESS;
                    }

                    if (this->producer_queue.size() != this->producer_queue.capacity()){
                        dg::network_exception_handler::nothrow_log(this->producer_queue.push_back(std::move(request)));
                        return dg::network_exception::SUCCESS;
                    }

                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }();

                if (releasing_smp != nullptr){
                    releasing_smp->release();
                }

                return err;
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

            dg::unordered_unstable_map<model::ticket_id_t, std::optional<ResponseObserverInterface *>> ticket_resource_map; //leaks
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

            auto open_ticket(size_t sz, model::ticket_id_t * out_ticket_arr) noexcept -> exception_t{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                size_t new_sz = this->ticket_resource_map.size() + sz;

                if (new_sz > this->ticket_resource_map_cap){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                for (size_t i = 0u; i < sz; ++i){
                    model::ticket_id_t new_ticket_id    = this->ticket_id_counter++;
                    auto [map_ptr, status]              = this->ticket_resource_map.insert(std::make_pair(new_ticket_id, std::optional<ResponseObserverInterface *>(std::nullopt)));
                    dg::network_exception_handler::dg_assert(status);
                    out_ticket_arr[i]                   = new_ticket_id;
                }

                return dg::network_exception::SUCCESS;
            }

            void assign_observer(model::ticket_id_t * ticket_id_arr, size_t sz, ResponseObserverInterface ** corresponding_observer_arr, std::expected<bool, exception_t> * exception_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    model::ticket_id_t current_ticket_id = ticket_id_arr[i];
                    auto map_ptr = this->ticket_resource_map.find(current_ticket_id);

                    if (map_ptr == this->ticket_resource_map.end()){
                        exception_arr[i] = std::unexpected(dg::network_exception::REST_TICKET_NOT_FOUND);
                        continue;
                    }

                    if (map_ptr->second.has_value()){
                        exception_arr[i] = false;
                        continue;
                    }

                    map_ptr->second     = corresponding_observer_arr[i];
                    exception_arr[i]    = true;
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

    //this is about the fastest we could ever write given the instruction set of host memory orderings
    //timed_semaphore registration offloaded -> timeoutmanager
    //relaxed counter -> atomic smp
    //1 memory_order_release/ 1024 workorders
    //no leaks
    //correct codes, we'll talk about the practices later (we are in C fellas, we are extremely greedy people who sincerely want our code to work perfectly)
    //supported re-requests by using designated ids
    //bucket_hint from trusted sources to reduce memory footprint by 10 folds
    //able to saturate 1GB - 5GB of inbound rest_requests/ second if correctly configurated 
    //we'll post the benchs
    //we'll be back tmr

    class RestController: public virtual RestControllerInterface{

        private:

            std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::shared_ptr<RequestContainerInterface> request_container;
            std::shared_ptr<TicketControllerInterface> ticket_controller;
            std::shared_ptr<TicketTimeoutManagerInterface> ticket_timeout_manager;
            stdx::hdi_container<size_t> max_consume_per_load;

        public:

            using self = RestController;

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
                
                //the code is not hard, yet it is extremely easy to leak resources

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                if (sz == 0u){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                model::ClientRequest * base_client_request_arr = client_request_arr.base();
                dg::network_stack_allocation::NoExceptAllocation<model::ticket_id_t[]> ticket_id_arr(sz);
                exception_t err = this->ticket_controller->open_ticket(sz, ticket_id_arr.get());

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                }

                auto ticket_resource_grd = stdx::resource_guard([&]() noexcept{
                    this->ticket_controller->close_ticket(ticket_id_arr.get(), sz);
                });

                std::expected<std::unique_ptr<InternalBatchResponse>, exception_t> response = self::internal_make_batch_request_response(sz, ticket_id_arr.get(), this->ticket_controller);

                if (!response.has_value()){
                    return std::unexpected(response.error());
                }

                ticket_resource_grd.release(); //ticket responsibility tranferred -> internal_batch_response

                auto response_resource_grd = stdx::resource_guard([&]() noexcept{
                    response.value()->release_response_wait_responsibility();
                });

                dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<ResponseObserverInterface>[]> response_observer_arr(sz);
                dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> response_observer_exception_arr(sz);

                for (size_t i = 0u; i < sz; ++i){
                    response_observer_arr[i] = response.value()->get_observer(i);
                }

                this->ticket_controller->assign_observer(ticket_id_arr.get(), sz, response_observer_arr.get(), response_observer_exception_arr.get());

                for (size_t i = 0u; i < sz; ++i){
                    if (!response_observer_exception_arr[i].has_value()){
                        return std::unexpected(response_observer_exception_arr[i].error());
                    }

                    dg::network_exception_handler::dg_assert(response_observer_exception_arr[i].value());
                }

                std::expected<dg::vector<model::InternalRequest>, exception_t> pushing_container = this->internal_make_internal_request(std::make_move_iterator(base_client_request_arr), sz);

                if (!pushing_container.has_value()){
                    return std::unexpected(pushing_container.error());
                }

                exception_t push_err = this->request_container->push(static_cast<dg::vector<model::InternalRequest>&&>(pushing_container.value()));

                if (dg::network_exception::is_failed(push_err)){
                    this->internal_rollback_client_request(base_client_request_arr, std::move(pushing_container.value()));
                    return std::unexpected(push_err);
                }

                dg::network_stack_allocation::NoExceptAllocation<std::pair<model::ticket_id_t, std::chrono::nanoseconds>[]> clockin_arr(sz);
                dg::network_stack_allocation::NoExceptAllocation<exception_t[]> clockin_exception_arr(sz);

                for (size_t i = 0u; i < sz; ++i){
                    clockin_arr[i] = std::make_pair(ticket_id_arr[i], base_client_request_arr[i].client_timeout_dur);
                }

                //clock_in must be thru, this is hard, I dont know why we aren't using shared_ptr<>

                this->ticket_timeout_manager->clock_in(clockin_arr.get(), sz, clockin_exception_arr.get());

                for (size_t i = 0u; i < sz; ++i){
                    if (dg::network_exception::is_failed(clockin_exception_arr[i])){
                        //unable to fail
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(clockin_exception_arr[i]));
                        std::abort();
                    }
                }

                response_resource_grd->release();

                return std::unique_ptr<BatchResponseInterface>(std::move(response.value()));
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load.value;
            }

        private:

            class InternalBatchResponse: public virtual BatchResponseInterface{

                private:

                    std::unique_ptr<BatchRequestResponse> base;
                    std::unique_ptr<model::ticket_id_t[]> ticket_id_arr;
                    size_t ticket_id_arr_sz;
                    std::shared_ptr<TicketControllerInterface> ticket_controller;
                    bool ticket_release_responsibility;

                public:

                    InternalBatchResponse(std::unique_ptr<BatchRequestResponse> base,
                                          std::unique_ptr<model::ticket_id_t[]> ticket_id_arr,
                                          size_t ticket_id_arr_sz,
                                          std::shared_ptr<TicketControllerInterface> ticket_controller,
                                          bool ticket_release_responsibility) noexcept: base(std::move(base)),
                                                                                        ticket_id_arr(std::move(ticket_id_arr)),
                                                                                        ticket_id_arr_sz(ticket_id_arr_sz),
                                                                                        ticket_controller(std::move(ticket_controller)),
                                                                                        ticket_release_responsibility(ticket_release_responsibility){}

                    ~InternalBatchResponse() noexcept{

                        this->base->wait_response();
                        this->release_ticket();
                    }

                    auto response() noexcept -> std::expected<dg::vector<std::expected<Response, exception_t>>, exception_t>{

                        auto rs = this->base->response();
                        this->release_ticket();
                        return rs;
                    }

                    auto response_size() const noexcept -> size_t{

                        return this->base->response_size();
                    }

                    auto get_observer(size_t idx) noexcept -> ResponseObserverInterface *{

                        return this->base->get_observer(idx);
                    }

                    void release_response_wait_responsibility() noexcept{

                        this->base->release_response_wait_responsibility();
                    }

                private:

                    __attribute__((noipa)) void release_ticket() noexcept{

                        if (!this->ticket_release_responsibility){
                            return;
                        }

                        this->ticket_controller->close_ticket(this->ticket_id_arr.get(), ticket_id_arr_sz);                        
                        this->ticket_release_responsibility = false;
                    }
            };

            static auto internal_make_batch_request_response(size_t request_sz, ticket_id_t * ticket_id_arr, std::shared_ptr<TicketControllerInterface> ticket_controller) noexcept -> std::expected<std::unique_ptr<InternalBatchResponse>, exception_t>{
                
                if (request_sz == 0u){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                if (ticket_controller == nullptr){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                std::expected<std::unique_ptr<BatchRequestResponse>, exception_t> base = dg::network_rest_frame::client_impl1::make_batch_request_response(request_sz);

                if (!base.has_value()){
                    return std::unexpected(base.error());
                }

                auto resource_grd = stdx::resource_guard([rptr = base.value().get()]() noexcept{
                    rptr->release_response_wait_responsibility();
                });

                std::expected<std::unique_ptr<model::ticket_id_t[]>, exception_t> cpy_ticket_id_arr = dg::network_exception::to_cstyle_function(dg::network_allocation::make_unique<ticket_id_t[]>)(request_sz);

                if (!cpy_ticket_id_arr.has_value()){
                    return std::unexpected(cpy_ticket_id_arr.error());
                }

                std::copy(ticket_id_arr, std::next(ticket_id_arr, request_sz), cpy_ticket_id_arr.value().get());
                std::expected<std::unique_ptr<InternalBatchResponse>, exception_t> rs = dg::network_allocation::cstyle_make_unique<InternalBatchResponse>(std::move(base.value()), 
                                                                                                                                                          std::move(cpy_ticket_id_arr.value()), 
                                                                                                                                                          request_sz, 
                                                                                                                                                          std::move(ticket_controller), 
                                                                                                                                                          true);

                if (!rs.has_value()){
                    return std::unexpected(rs.error());
                }

                resource_grd->release();

                return rs;
            }

            static auto internal_make_internal_request(std::move_iterator<model::ClientRequest *> request_arr, ticket_id_t * ticket_id_arr, size_t request_arr_sz) noexcept -> std::expected<dg::vector<model::InternalRequest>, exception_t>{

                model::ClientRequest * base_request_arr = request_arr.base();
                std::expected<dg::vector<model::InternalRequest>, exception_t> rs = dg::network_exception::cstyle_initialize<dg::vector<model::InternalRequest>>(request_arr_sz);

                if (!rs.has_value()){
                    return std::unexpected(rs.error());
                }

                for (size_t i = 0u; i < request_arr_sz; ++i){
                    static_assert(std::is_nothrow_move_assignable_v<model::InternalRequest>);
                    static_assert(std::is_nothrow_move_constructible_v<dg::string>);

                    rs.value()[i] = InternalRequest{.request    = Request{.requestee_uri    = std::move(base_request_arr[i].requestee_uri),
                                                                          .requestor        = std::move(base_request_arr[i].requestor),
                                                                          .payload          = std::move(base_request_arr[i].payload)},

                                                    .ticket_id  = ticket_id_arr[i]};
                }

                return rs;
            }

            static void internal_rollback_client_request(model::ClientRequest * client_request_arr, dg::vector<model::InternalRequest> internal_request_arr) noexcept{

                for (size_t i = 0u; i < internal_request_arr.size(); ++i){
                    client_request_arr[i].requestee_uri = std::move(internal_request_arr[i].request.requestee_uri);
                    client_request_arr[i].requestor     = std::move(internal_request_arr[i].request.requestor);
                    client_request_arr[i].payload       = std::move(internal_request_arr[i].request.payload);
                }
            }
    };

    //we are reducing the serialization overheads of ticket_center
    //this is to utilize CPU resource + CPU efficiency by running affined task
    //we'll do partial load balancing, Ive yet to know what that is
    //because I think random hash in conjunction with max_consume_size should be representative
    //we should not be too greedy, and actually implement somewhat a load_balancer

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
