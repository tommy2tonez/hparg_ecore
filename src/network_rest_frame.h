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
//we'll do code review for 1-2 days, we'll move on to implement our beloved Taylor's search

//we'll be back, I just read the proposal for std::hive 
//I dont precisely know what the component does, except for being a vector, a deque son
//vector in the sense of contiguous back insert
//deque in the sense of a discretized range -> a block of memory, such block would refer to another block for some reasons...  
//so basically the std is implementing our std::dense_hash_map<> with their own convention of hashing technique to leverage bucket collisions locality

namespace dg::network_rest_frame::model{

    using ticket_id_t   = __uint128_t; //I've thought long and hard, it's better to do bitshift, because the otherwise would be breaking single responsibilities, breach of extensions
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

    struct CacheControllerInterface{
        virtual ~CacheControllerInterface() noexcept = default;
        virtual void get_cache(cache_id_t * cache_id_arr, size_t sz, std::expected<std::optional<Response>, exception_t> *) noexcept = 0; //we'll think about making the exclusion responsibility this component responsibility later
        virtual void insert_cache(cache_id_t * cache_id_arr, std::move_iterator<Response *> response_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept = 0;
        virtual void contains(cache_id_t * cache_id_arr, size_t sz, bool * rs_arr) noexcept = 0;
        virtual auto max_response_size() const noexcept -> size_t = 0;
        virtual void clear() noexcept = 0;
        virtual auto size() const noexcept -> size_t = 0;
        virtual auto capacity() const noexcept -> size_t = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0; 
    };

    struct InfiniteCacheControllerInterface{
        virtual ~InfiniteCacheControllerInterface() noexcept = default;
        virtual void get_cache(cache_id_t * cache_id_arr, size_t sz, std::expected<std::optional<Response>, exception_t> *) noexcept = 0; //we'll think about making the exclusion responsibility this component responsibility later
        virtual void insert_cache(cache_id_t * cache_id_arr, std::move_iterator<Response *> response_arr, size_t sz, std::expected<bool, exception_t> * rs_arr) noexcept = 0;
        virtual auto max_response_size() const noexcept -> size_t = 0;
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

    struct CacheUniqueWriteTrafficControllerInterface{
        virtual ~CacheUniqueWriteTrafficControllerInterface() noexcept = default;
        virtual auto thru(size_t) noexcept -> exception_t = 0;
        virtual void reset() noexcept = 0;
    };

    struct UpdatableInterface{
        virtual ~UpdatableInterface() noexcept = default;
        virtual void update() noexcept = 0;
    };

    struct RequestHandlerInterface{
        using Request   = model::Request;
        using Response  = model::Response; 

        virtual ~RequestHandlerInterface() noexcept = default;
        virtual void handle(std::move_iterator<Request *>, size_t, Response *) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };
}

namespace dg::network_rest_frame::client{

    struct ResponseObserverInterface{
        virtual ~ResponseObserverInterface() noexcept = default;
        virtual void update(std::expected<Response, exception_t>) noexcept = 0;
        virtual void deferred_memory_ordering_fetch(std::expected<Response, exception_t>) noexcept = 0;
        virtual void deferred_memory_ordering_fetch_close() noexcept = 0;
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
        virtual void assign_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::add_pointer_t<ResponseObserverInterface> * assigning_observer_arr, std::expected<bool, exception_t> * exception_arr) noexcept = 0;
        virtual void steal_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t> * out_observer_arr) noexcept = 0;
        virtual void get_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t> * out_observer_arr) noexcept = 0;
        virtual void close_ticket(model::ticket_id_t * ticket_id_arr, size_t sz) noexcept = 0;
        virtual auto max_consume_size() noexcept -> size_t = 0; 
    };

    struct TicketTimeoutManangerInterface{
        virtual ~TicketTimeoutManagerInterface() noexcept = default;
        virtual void clock_in(std::pair<model::ticket_id_t, std::chrono::nanoseconds> * registering_arr, size_t sz, exception_t * exception_arr) noexcept = 0;
        virtual void get_expired_ticket(model::ticket_id_t * output_arr, size_t& sz, size_t cap) noexcept = 0;
        virtual auto max_clockin_dur() const noexcept -> std::chrono::nanoseconds = 0;
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

//t = sqrt(2h/g), horizontal speed is irrelevant in vaccum, maybe ... not in the realistic scenerio when we need to factor in drag
//s = 1/2at^2
//this is harder to write than you could imagine
//nothing compared to cuda language, we'll be there

namespace dg::network_rest_frame::server_impl1{

    using namespace dg::network_rest_frame::server; 

    class RequestResolverWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            dg::unordered_unstable_map<dg::string, std::unique_ptr<RequestHandlerInterface>> request_handler_map;
            std::shared_ptr<InfiniteCacheControllerInterface> request_cache_controller;
            std::shared_ptr<InfiniteCacheUniqueWriteControllerInterface> cachewrite_uex_controller;
            std::shared_ptr<CacheUniqueWriteTrafficControllerInterface> cachewrite_traffic_controller; //we provide the users a contract, within a certain window to do dup-requests, such cannot be guaranteed if we are not attempting to do traffic control, nasty things could happen 
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
                                  std::shared_ptr<CacheUniqueWriteTrafficControllerInterface> cachewrite_traffic_controller,
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
                                                                    cachewrite_traffic_controller(std::move(cachewrite_traffic_controller)),
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

                size_t recv_buf_cap = this->resolve_consume_sz;
                size_t recv_buf_sz  = {};
                dg::network_stack_allocation::NoExceptAllocation<dg::string[]> recv_buf_arr(recv_buf_cap);
                dg::network_kernel_mailbox::recv(this->resolve_channel, recv_buf_arr.get(), recv_buf_sz, recv_buf_cap);

                auto mailbox_feed_resolutor                                 = InternalResponseFeedResolutor{}; 
                mailbox_feed_resolutor.mailbox_channel                      = this->response_channel;
                mailbox_feed_resolutor.transmit_opt                         = this->transmit_opt;

                size_t trimmed_mailbox_feed_cap                             = std::min(std::min(this->mailbox_feed_cap, dg::network_kernel_mailbox::max_consume_size()), recv_buf_sz);
                size_t mailbox_feeder_allocation_cost                       = dg::network_producer_consumer::delvrsrv_allocation_cost(&mailbox_feed_resolutor, trimmed_mailbox_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> mailbox_feeder_mem(mailbox_feeder_allocation_cost);
                auto mailbox_feeder                                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&mailbox_feed_resolutor, trimmed_mailbox_feed_cap, mailbox_feeder_mem.get())); 

                //---

                auto mailbox_prep_feed_resolutor                            = InternalMailBoxPrepFeedResolutor{};
                mailbox_prep_feed_resolutor.mailbox_feeder                  = mailbox_feeder.get();

                size_t trimmed_mailbox_prep_feed_cap                        = std::min(this->mailbox_prep_feed_cap, recv_buf_sz);
                size_t mailbox_prep_feeder_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(&mailbox_prep_feed_resolutor, trimmed_mailbox_prep_feed_cap);
                dg::network_stack_allocation::NoExceptAllocation<char[]> mailbox_prep_feeder_mem(mailbox_prep_feeder_allocation_cost);
                auto mailbox_prep_feeder                                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&mailbox_prep_feed_resolutor, tirmmed_mailbox_prep_feed_cap, mailbox_prep_feeder_mem.get())); 

                //---

                auto cache_map_insert_feed_resolutor                        = InternalCacheMapFeedResolutor{}; 
                cache_map_insert_feed_resolutor.cache_controller            = this->request_cache_controller.get();

                size_t trimmed_cache_map_insert_feed_cap                    = std::min(std::min(this->cache_map_insert_feed_cap, this->request_cache_controller->max_consume_size()), recv_buf_sz); 
                size_t cache_map_insert_feeder_allocation_cost              = dg::network_producer_consumer::delvrsrv_allocation_cost(&cache_map_insert_feed_resolutor, trimmed_cache_map_insert_feed_cap);
                dg::network_stack_allocation::NoExceptAllocation<char[]> cache_map_insert_feeder_mem(cache_map_insert_feeder_allocation_cost);
                auto cache_map_insert_feeder                                = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&cache_map_insert_feed_resolutor, trimmed_cache_map_insert_feed_cap, cache_map_insert_feeder_mem.get())); 

                //---

                auto server_fetch_feed_resolutor                            = InternalServerFeedResolutor{};
                server_fetch_feed_resolutor.request_handler_map             = &this->request_handler_map;
                server_fetch_feed_resolutor.mailbox_prep_feeder             = mailbox_prep_feeder.get();
                server_fetch_feed_resolutor.cache_map_feeder                = cache_map_insert_feeder.get();

                size_t trimmed_server_feed_cap                              = std::min(this->server_fetch_feed_cap, recv_buf_sz);
                size_t server_feeder_allocation_cost                        = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&server_fetch_feed_resolutor, trimmed_server_fetch_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> server_feeder_mem(server_feeder_allocation_cost);
                auto server_feeder                                          = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&server_fetch_feed_resolutor, trimmed_server_fetch_feed_cap, server_feeder_mem.get()));

                //---

                auto cache_server_fetch_feed_resolutor                      = InternalCacheServerFeedResolutor{}; 
                cache_server_fetch_feed_resolutor.cachewrite_uex_controller = this->cachewrite_uex_controller.get();
                cache_server_fetch_feed_resolutor.cachewrite_tfx_controller = this->cachewrite_traffic_controller.get();
                cache_server_fetch_feed_resolutor.server_feeder             = server_feeder.get();
                cache_server_fetch_feed_resolutor.mailbox_prep_feeder       = mailbox_prep_feeder.get();

                size_t trimmed_cache_server_fetch_feed_cap                  = std::min(std::min(this->cache_server_fetch_cap, this->cachewrite_uex_controller->max_consume_size()), recv_buf_sz);
                size_t cache_server_feeder_allocation_cost                  = dg::network_producer_consumer::delvrsrv_allocation_cost(&cache_server_fetch_feed_resolutor, trimmed_cache_server_fetch_feed_cap);
                dg::network_stack_allocation::NoExceptAllocation<char[]> cache_server_feeder_mem(cache_server_feeder_allocation_cost);
                auto cache_server_feeder                                    = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&cache_server_fetch_feed_resolutor, trimmed_cache_server_fetch_feed_cap, cache_server_feeder_mem.get())); 

                //---

                auto cache_fetch_feed_resolutor                             = InternalCacheFeedResolutor{};
                cache_fetch_feed_resolutor.cache_server_feeder              = cache_server_feeder.get();
                cache_fetch_feed_resolutor.mailbox_prep_feeder              = mailbox_prep_feeder.get();
                cache_fetch_feed_resolutor.cache_controller                 = this->request_cache_controller.get();

                size_t trimmed_cache_fetch_feed_cap                         = std::min(this->cache_fetch_feed_cap, recv_buf_sz);
                size_t cache_fetch_feeder_allocation_cost                   = dg::network_producer_consumer::delvrsrv_allocation_cost(&cache_fetch_feed_resolutor, trimmed_cache_fetch_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> cache_fetch_feeder_mem(cache_fetch_feeder_allocation_cost);
                auto cache_fetch_feeder                                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&cache_fetch_feed_resolutor, trimmed_cache_fetch_feed_Cap, cache_fetch_feeder_mem.get()));

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
                        continue;
                    }

                    std::expected<dg::string, exception_t> resource_path = dg::network_uri_encoder::extract_local_path(request->request.requestee_uri);

                    if (!resource_path.has_value()){
                        auto response   = model::InternalResponse{.response     = std::unexpected(resource_path.error()), 
                                                                  .ticket_id    = request->ticket_id};

                        auto prep_arg   = InternalMailBoxPrepArgument{.to               = requestor_addr.value(),
                                                                      .response         = std::move(response),
                                                                      .dual_priority    = request->dual_priority};

                        dg::network_producer_consumer::delvrsrv_deliver(mailbox_prep_feeder.get(), std::move(prep_arg));
                        continue;
                    }

                    auto map_ptr = this->request_handler.find(resource_path.value());

                    if (map_ptr == this->request_handle.end()){
                        auto response   = model::InternalResponse{.response   = std::unexpected(dg::network_exception::REST_INVALID_URI), 
                                                                  .ticket_id  = request->ticket_id};

                        auto prep_arg   = InternalMailBoxPrepArgument{.to               = requestor_addr.value(),
                                                                      .response         = std::move(response),
                                                                      .dual_priority    = request->dual_priority};

                        dg::network_producer_consumer::delvrsrv_deliver(mailbox_prep_feeder.get(), std::move(prep_arg));
                        continue;
                    }

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

                        dg::network_producer_consumer::delvrsrv_kv_deliver(server_feeder.get(), key_arg, std::move(value_arg));
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
                CacheUniqueWriteTrafficControllerInterface * cachewrite_tfx_controller;

                dg::network_producer_consumer::KVDeliveryHandle<InternalServerFeedResolutorArgument> * server_feeder;
                dg::network_producer_consumer::DeliveryHandle<InternalMailBoxPrepArgument> * mailbox_prep_feeder;

                void push(std::move_iterator<InternalCacheServerFeedArgument *> data_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> cache_write_response_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i] = base_data_arr[i].cache_id;
                    }

                    exception_t thru_status = this->cachewrite_tfx_controller->thru(sz); //if it is already uex_controller->thru, we got a leak, its interval leak, the crack between the read_cache and the thru, if it is not thru, then we are logically correct 

                    if (dg::network_exception::is_failed(thru_status)){
                        for (size_t i = 0u; i < sz; ++I){
                            auto response = model::InternalResponse{.response   = std::unexpected(thru_status),
                                                                    .ticket_id  = base_data_arr[i]->ticket_id};

                            auto prep_arg   = InternalMailBoxPrepArgument{.to               = base_data_arr[i].to,
                                                                          .response         = std::move(response),
                                                                          .dual_priority    = base_data_arr[i].dual_priority};

                            dg::network_producer_consumer::delvrsrv_deliver(this->mailbox_prep_feeder, std::move(prep_arg));
                        }

                        return;
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
                                auto response   = model::InternalResponse{.response   = std::unexpected(dg::network_exception::REST_BAD_CACHE_WRITE), //
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

                                dg::network_producer_consumer::delvrsrv_kv_deliver(this->server_feeder, base_data_arr[i].local_uri_path, std::move(arg));
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

                dg::network_producer_consumer::KVDeliveryHandle<InternalCacheServerFeedArgument> * cache_server_feeder;
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

                            dg::network_producer_consumer::delvrsrv_kv_deliver(this->cache_server_feeder, base_data_arr[i].local_uri_path, std::move(arg));
                        }
                    }
                }
            };
    };
} 

namespace dg::network_rest_frame::client_impl1{

    //we'll implement the Taylor's search, be patient, this is very tough to implement on cuda

    //we are talking about another entire different programming styles + practices
    //one thing we know for sure, we cant trust cuda
    //all important operations must be operated on host, and every backprop + update must have their internal mechanisms of clamping the values
    //we implement our own accelerated linear algebra, we have repeated operations, which we will attempt to find fuzzy representation of projection space by running math_approx + calibrate

    //fine allocations (could be further improved)
    //we have our top-tier filesystem (alright not to be too proud, it's running extremely slow, yet very high accuracy, RAM-accurate)
    //reasonable socket protocol (need to pad requests to actually achieve low latency high thruput)
    //OK unified memory address (we need to improve our locking routines)
    //no synchronization compute
    //OK tiles (memregions + friends)
    //top-tier security line (unif_dist symmetric encoder)

    //what we are missing is ... a backpropagation search (this is still Stone Aged technology, we yet to know a better way than to do search ...)
    //a multi-precision library on cuda (this is very important to achieve 10 ** 18 decimal accuracy of projection space)
    //a Taylor Series patterns database
    //each of those tasks would take 1 year of non-stop working to accurately implement (yeah its hard)
    //we'll see about our timeline later
    //it's gonna be hard to be a trillionaire Mom
    //we'll be there

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
                    auto map_ptr = std::as_const(this->cache_map).find(cache_id_arr[i]);

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

                Response * base_response_arr = response_arr.base(); 

                for (size_t i = 0u; i < sz; ++i){
                    if (this->cache_map.size() == this->cache_map_cap){
                        rs_arr[i] = std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                        continue;
                    }

                    if (base_response_arr[i].response.size() > this->max_response_sz){
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

            auto max_response_size() const noexcept -> size_t{

                return this->max_response_sz;
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

            auto max_response_size() const noexcept -> size_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                return this->base->max_response_size(); 
            } 

            auto max_consume_size() noexcept -> size_t{

                //assumptions
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

                size_t trimmed_getcache_feed_cap    = std::min(this->getcache_feed_cap, sz);
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

            auto max_response_size() const noexcept -> size_t{

                return std::min(this->left->max_response_size(), this->right->max_response_size());                
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

                    InternalInsertCacheFeedArgument * base_data_arr = data_arr.base();

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

            auto max_response_size() const noexcept -> size_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                return this->base->max_response_size();
            }

            auto max_consume_size() noexcept -> size_t{

                //assumptions

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
            size_t max_response_sz;

        public:

            DistributedCacheController(std::unique_ptr<std::unique_ptr<InfiniteCacheControllerInterface>[]> cache_controller_arr,
                                       size_t pow2_cache_controller_arr_sz,
                                       size_t getcache_keyvalue_feed_cap,
                                       size_t insertcache_keyvalue_feed_cap,
                                       size_t max_consume_per_load,
                                       size_t max_response_sz) noexcept: cache_controller_arr(std::move(cache_controller_arr)),
                                                                         pow2_cache_controller_arr_sz(pow2_cache_controller_arr_sz),
                                                                         getcache_keyvalue_feed_cap(getcache_keyvalue_feed_cap),
                                                                         insertcache_keyvalue_feed_cap(insertcache_keyvalue_feed_cap),
                                                                         max_consume_per_load(std::move(max_consume_per_load)),
                                                                         max_response_sz(max_response_sz){}

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
                                                                              .response_ptr = std::make_move_iterator(std::next(base_response_arr, i)),
                                                                              .rs           = std::next(rs_arr, i)};

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, std::move(feed_arg));
                }
            }

            auto max_response_size() const noexcept -> size_t{

                return this->max_response_sz;
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
                std::move_iterator<Response *> response_ptr;
                std::expected<bool, exception_t> * rs;
            };

            struct InternalCacheInsertFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalCacheInsertFeedArgument>{

                std::unique_ptr<InfiniteCacheControllerInterface> * cache_controller_arr;

                void push(const size_t& idx, std::move_iterator<InternalCacheInsertFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalCacheInsertFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<cache_id_t[]> cache_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<Response[]> response_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> rs_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        cache_id_arr[i]                 = base_data_arr[i].cache_id;
                        Response * base_response_ptr    = base_arr[i].response_ptr.base();
                        response_arr[i]                 = std::move(*base_response_ptr);
                    }

                    this->cache_controller_arr[idx]->insert_cache(cache_id_arr.get(), std::make_move_iterator(response_arr.get()), sz, rs_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        *base_data_arr[i].rs = rs_arr[i];

                        if (!rs_arr[i].has_value()){
                            Response * base_response_ptr    = base_data_arr[i].response_ptr.base();
                            *base_response_ptr              = std::move(response_arr[i]);
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

                //I've been thinking long and hard about internal_dispatch_switch() because that seems like the only fuzzy point of logic
                //the problem is that post the switch must guarantee enough room for sz, and switch population threshold is under the capcity of the calling container

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

                    InternalThruFeedArgument * base_data_arr = data_arr.base();

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

    //we'll offload the responsibility of noipa -> stdx
    //we do lambdas for now

    class RequestResponseBase{

        private:

            stdx::inplace_hdi_container<std::atomic_flag> smp; //I'm allergic to shared_ptr<>, it costs a memory_order_seq_cst to deallocate the object, we'll do things this way to allow us leeways to do relaxed operations to unlock batches of requests later, thing is that timed_semaphore is not a magic, it requires an entry registration in the operating system, we'll work around things by reinventing the wheel
            stdx::inplace_hdi_container<std::expected<Response, exception_t>> resp;
            stdx::inplace_hdi_container<bool> is_response_invoked;

        public:

            RequestResponse() noexcept: smp(std::in_place_t{}, false),
                                        resp(std::in_place_t{}, std::unexpected(dg::network_exception::REST_RESPONSE_NOT_INITIALIZED)),
                                        is_response_invoked(std::in_place_t{}, false){}

            void update(std::expected<Response, exception_t> response_arg) noexcept{

                this->resp.value = std::move(response_arg);
                this->smp.value.test_and_set(std::memory_order_release);
            }

            void deferred_memory_ordering_fetch(std::expected<Response, exception_t> response_arg) noexcept{

                this->resp.value = std::move(response_arg);
            }

            void deferred_memory_ordering_fetch_close() noexcept{

                this->internal_deferred_memory_ordering_fetch_close(static_cast<void *>(&this->resp.value)); //not necessary, I'd love to have noipa, I dont know what to do otherwise
            }

            auto response() noexcept -> std::expected<Response, exception_t>{

                bool was_invoked = std::exchange(this->is_response_invoked.value, true);

                if (was_invoked){
                    return std::unexpected(dg::network_exception::REST_RESPONSE_DOUBLE_INVOKE);
                }

                this->smp.value.wait(false, std::memory_order_acquire);

                return std::move(this->resp.value);
            }
        
        private:

            void internal_deferred_memory_ordering_fetch_close(void * dirty_memory) noexcept{

                auto task = [](RequestResponseBase * self_obj, void * dirty_memory_arg) noexcept{
                    (void) dirty_memory_arg;
                    self_obj->smp.value.test_and_set(std::memory_order_relaxed);
                };

                stdx::noipa_do_task(task, this, dirty_memory);
            }
    };

    class BatchRequestResponseBase{

        private:

            stdx::inplace_hdi_container<std::atomic<intmax_t>> atomic_smp;
            dg::vector<std::expected<Response, exception_t>> resp_vec; //alright, there are hardware destructive interference issues, we dont want to talk about that yet
            stdx::inplace_hdi_container<bool> is_response_invoked;

        public:

            BatchRequestResponseBase(size_t resp_sz): atomic_smp(std::in_place_t{}, -static_cast<intmax_t>(stdx::zero_throw(resp_sz)) + 1),
                                                      resp_vec(stdx::zero_throw(resp_sz), std::unexpected(dg::network_exception::REST_RESPONSE_NOT_INITIALIZED)),
                                                      is_response_invoked(std::in_place_t{}, false){}

            void update(size_t idx, std::expected<Response, exception_t> response) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->resp_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION)); //this is vector's business
                        std::abort();
                    }
                }

                this->resp_vec[idx] = std::move(response);
                this->atomic_smp.value.fetch_add(1u, std::memory_order_release);
            }

            void deferred_memory_ordering_fetch(size_t idx, std::expected<Response, exception_t> response) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->resp_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION)); //this is vector's business
                        std::abort();
                    }
                }

                this->resp_vec[idx] = std::move(response);
            }

            void deferred_memory_ordering_fetch_close(size_t idx) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (idx >= this->resp_vec.size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION)); //this is vector's business
                        std::abort();
                    }
                }

                this->internal_deferred_memory_ordering_fetch_close(idx, static_cast<void *>(&this->resp_vec[idx]));
            }

            auto response() noexcept -> std::expected<dg::vector<std::expected<Response, exception_t>>, exception_t>{

                bool was_invoked = std::exchange(this->is_response_invoked.value, true);

                if (was_invoked){
                    return std::unexpected(dg::network_exception::REST_RESPONSE_DOUBLE_INVOKE);
                } 

                this->atomic_smp.value.wait(0, std::memory_order_acquire);

                return dg::vector<std::expected<Response, exception_t>>(std::move(this->resp_vec));
            }

        private:

            void internal_deferred_memory_ordering_fetch_close(size_t idx, void * dirty_memory) noexcept{

                auto task = [](BatchRequestResponseBase * self_obj, size_t idx_arg, void * dirty_memory_arg) noexcept{
                    (void) idx_arg;
                    (void) dirty_memory_arg;

                    intmax_t old = self_obj->atomic_smp.value.fetch_add(1, std::memory_order_relaxed);

                    if (old == 0){
                        self_obj->atomic_smp.value.notify_one():
                    }  
                };

                stdx::noipa_do_task(task, this, idx, dirty_memory);
            }
    };

    //its more complicated than most foos think
    class BatchRequestResponse: public virtual BatchResponseInterface{

        private:

            class BatchRequestResponseBaseDesignatedObserver: public virtual ResponseObserverInterface{

                private:

                    BatchRequestResponseBase * base;
                    size_t idx;

                public:

                    BatchRequestResponseBaseDesignatedObserver() = default;

                    BatchRequestResponseBaseDesignatedObserver(BatchRequestResponseBase * base, 
                                                               size_t idx) noexcept: base(base),
                                                                                     idx(idx){}

                    void update(std::expected<Response, exception_t> response) noexcept{

                        this->base->update(this->idx, std::move(response));
                    }

                    void deferred_memory_ordering_fetch(std::expected<Response, exception_t> response) noexcept{

                        this->base->deferred_memory_ordering_fetch(this->idx, std::move(response));
                    }

                    void deferred_memory_ordering_fetch_close() noexcept{

                        this->base->deferred_memory_ordering_fetch_close(this->idx);
                    }
            };

            dg::vector<BatchRequestResponseBaseDesignatedObserver> observer_arr; 
            BatchRequestResponseBase base;
            bool response_wait_responsibility_flag;

            BatchRequestResponse(size_t resp_sz): observer_arr(resp_sz),
                                                  base(resp_sz),
                                                  response_wait_responsibility_flag(true){

                for (size_t i = 0u; i < resp_sz; ++i){
                    this->observer_arr[i] = BatchRequestResponseBaseDesignatedObserver(&this->base, i);
                }
            }

            friend auto make_batch_request_response(size_t resp_sz) noexcept -> std::expected<std::unique_ptr<BatchRequestResponse>, exception_t>; 

        public:

            BatchRequestResponse(const BatchRequestResponse&) = delete;
            BatchRequestResponse(BatchRequestResponse&&) = delete;
            BatchRequestResponse& operator =(const BatchRequestResponse&) = delete;
            BatchRequestResponse& operator =(BatchRequestResponse&&) = delete;

            ~BatchRequestResponse() noexcept{

                this->wait_response();
            }

            auto response() noexcept -> std::expected<dg::vector<std::expected<Response, exception_t>>, exception_t>{

                return this->base->response();
            }

            auto response_size() const noexcept -> size_t{

                return this->observer_arr.size();
            }

            auto get_observer(size_t idx) noexcept -> std::expected<ResponseObserverInterface *, exception_t>{ //response observer is a unique_resource, such is an acquisition of this pointer must involve accurate acquire + release mechanisms like every other dg::string or dg::vector, etc.

                if (idx >= this->observer_arr.size()){
                    return std::unexpected(dg::network_exception::OUT_OF_RANGE_ACCESS);
                }

                return static_cast<ResponseObserverInterface *>(std::addressof(this->observer_arr[idx]));
            }

            void release_response_wait_responsibility() noexcept{

                this->response_wait_responsibility_flag = false;
            }

            void wait_response() noexcept{

                bool wait_responsibility = std::exchange(this->response_wait_responsibility_flag, false); 

                if (wait_responsibility){
                    stdx::empty_noipa(this->base->response(), this->observer_arr);
                }
            }
    };

    //this is the only defined + stable way of programming, is to use factory std::unique_ptr<> construction
    auto make_batch_request_response(size_t resp_sz) noexcept -> std::expected<std::unique_ptr<BatchRequestResponse>, exception_t>{

        // return dg::network_allocation::cstyle_make_unique<BatchRequestResponse>(resp_sz);
    }

    class RequestResponse: public virtual ResponseInterface{

        private:

            class RequestResponseBaseObserver: public virtual ResponseObserverInterface{

                private:

                    RequestResponseBase * base;

                public:

                    RequestResponseBaseObserver() = default;

                    RequestResponseBaseObserver(RequestResponseBase * base) noexcept: base(base){}

                    void update(std::expected<Response, exception_t> resp) noexcept{

                        this->base->update(std::move(resp));
                    }

                    void deferred_memory_ordering_fetch(std::expected<Response, exception_t> resp) noexcept{

                        this->base->deferred_memory_ordering_fetch(std::move(resp));
                    }

                    void deferred_memory_ordering_fetch_close() noexcept{

                        this->base->deferred_memory_ordering_fetch_close();
                    }
            };

            RequestResponseBase base;
            RequestResponseBaseObserver observer;
            bool response_wait_responsibility_flag; 

            RequestResponse(): base(),
                               response_wait_responsibility_flag(true){

                this->observer = RequestResponseBaseObserver(&this->base);
            }

            friend auto make_request_response() noexcept -> std::expected<std::unique_ptr<RequestResponse>, exception_t>;

        public:

            RequestResponse(const RequestResponse&) = delete;
            RequestResponse(RequestResponse&&) = delete;
            RequestResponse& operator =(const RequestResponse&) = delete;
            RequestResponse& operator =(RequestResponse&&) = delete;

            ~ReleaseSafeRequestResponse() noexcept{

                this->wait_response();
            }

            auto response() noexcept -> std::expected<Response, exception_t>{

                return this->base->response();
            }

            auto get_observer() noexcept -> ResponseObserverInterface *{ //response observer is a unique_resource, such is an acquisition of this pointer must involve accurate acquire + release mechanisms like every other dg::string or dg::vector, etc.

                return static_cast<ResponseObserverInterface *>(&this->observer);
            }

            void release_response_wait_responsibility() noexcept{

                this->response_wait_responsibility_flag = false;
            }

            void wait_response() noexcept{

                bool wait_responsibility = std::exchange(this->response_wait_responsibility_flag, false);

                if (wait_responsibility){
                    stdx::empty_noipa(this->base->response(), this->observer);
                }
            }
    };

    auto make_request_response() noexcept -> std::expected<std::unique_ptr<RequestResponse>, exception_t>{

        // return dg::network_allocation::cstyle_make_unique<RequestResponse>();
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

                    if (this->producer_queue.size() == this->producer_queue.capacity()){
                        return dg::network_exception::RESOURCE_EXHAUSTION;
                    }

                    dg::network_exception_handler::nothrow_log(this->producer_queue.push_back(std::move(request)));
                    return dg::network_exception::SUCCESS;
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
                        auto rs = std::move(this->producer_queue.front());
                        this->producer_queue.pop_front();
                        return rs;
                    }

                    if (this->waiting_queue.size() == this->waiting_queue.capacity()){
                        continue;
                    }

                    this->waiting_queue.push_back(std::make_pair(&pending_smp, &internal_request));
                    break;
                }

                pending_smp.acquire();
                std::atomic_signal_fence(std::memory_order_seq_cst);

                return dg::vector<model::InternalRequest>(std::move(internal_request.value()));
            }
    };

    struct NormalTicketHasher{

        constexpr auto operator()(const ticket_id_t& ticket_id) const noexcept -> size_t{

            return dg::network_hash::hash_reflectible(ticket_id);
        }
    };

    struct IncrementingTicketHasher{

        constexpr auto operator()(const ticket_id_t& ticket_id) const noexcept -> size_t{

            static_assert(std::is_unsigned_v<ticket_id_t>);
            return ticket_id;
        }
    };

    //* is too confusing and bad, we need to make it looks like a pointer container like std::shared_ptr<> or std::unique_ptr<>, we use std::add_pointer_t<>, that looks aesthetically weird yet I would want to differentiate the semantic 

    template <class Hasher>
    class TicketController: public virtual TicketControllerInterface{

        private:

            dg::unordered_unstable_map<model::ticket_id_t, std::optional<std::add_pointer_t<ResponseObserverInterface>>, Hasher> ticket_resource_map; //leaks
            size_t ticket_resource_map_cap;
            ticket_id_t ticket_id_counter;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<size_t> max_consume_per_load;

        public:

            TicketController(dg::unordered_unstable_map<model::ticket_id_t, std::optional<std::add_pointer_t<ResponseObserverInterface>>, Hasher> ticket_resource_map,
                             size_t ticket_resource_map_cap,
                             ticket_id_t ticket_id_counter,
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
                    static_assert(std::is_unsigned_v<ticket_id_t>); //

                    model::ticket_id_t new_ticket_id    = this->ticket_id_counter++;
                    auto [map_ptr, status]              = this->ticket_resource_map.insert(std::make_pair(new_ticket_id, std::optional<std::add_pointer_t<ResponseObserverInterface>>(std::nullopt)));
                    dg::network_exception_handler::dg_assert(status);
                    out_ticket_arr[i]                   = new_ticket_id;
                }

                return dg::network_exception::SUCCESS;
            }

            void assign_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::add_pointer_t<ResponseObserverInterface> * corresponding_observer_arr, std::expected<bool, exception_t> * exception_arr) noexcept{

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

                    if (corresponding_observer_arr[i] == nullptr){
                        exception_arr[i] = std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                        continue;
                    }

                    map_ptr->second     = corresponding_observer_arr[i];
                    exception_arr[i]    = true;
                }
            }

            void get_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t> * out_observer_arr) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    model::ticket_id_t current_ticket_id = ticket_id_arr[i];
                    auto map_ptr = this->ticket_resource_map.find(current_ticket_id);

                    if (map_ptr == this->ticket_resource_map.end()){
                        response_arr[i] = std::unexpected(dg::network_exception::REST_TICKET_NOT_FOUND);
                        continue;
                    }

                    if (!map_ptr->second.has_value()){
                        response_arr[i] = std::unexpected(dg::network_exception::REST_TICKET_OBSERVER_NOT_FOUND);
                        continue;
                    }

                    response_arr[i] = map_ptr->second.value();
                }
            }

            void steal_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t> * response_arr) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                for (size_t i = 0u; i < sz; ++i){
                    model::ticket_id_t current_ticket_id = ticket_id_arr[i];
                    auto map_ptr = this->ticket_resource_map.find(current_ticket_id);

                    if (map_ptr == this->ticket_resource_map.end()){
                        response_arr[i] = std::unexpected(dg::network_exception::REST_TICKET_NOT_FOUND);
                        continue;
                    }

                    if (!map_ptr->second.has_value()){
                        response_arr[i] = std::unexpected(dg::network_exception::REST_TICKET_OBSERVER_NOT_FOUND);
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

    //this is to utilize CPU resource, not CPU efficiency, CPU efficiency is already achieved by batching ticket_id_arr, kernel has effective + efficient schedulings that render the overhead of mtx -> 1/1000 of the actual tx
    class DistributedTicketController: public virtual TicketControllerInterface{

        private:

            std::unique_ptr<std::unique_ptr<TicketControllerInterface>[]> base_arr;
            size_t pow2_base_arr_sz;
            size_t probe_arr_sz;
            size_t keyvalue_feed_cap;
            size_t minimum_discretization_sz;
            size_t max_consume_per_load; 

        public:

            DistributedTicketController(std::unique_ptr<std::unique_ptr<TicketControllerInterface>[]> base_arr,
                                        size_t pow2_base_arr_sz,
                                        size_t probe_arr_sz,
                                        size_t keyvalue_feed_cap,
                                        size_t minimum_discretization_sz,
                                        size_t max_consume_per_load) noexcept: base_arr(std::move(base_arr)),
                                                                               pow2_base_arr_sz(pow2_base_arr_sz),
                                                                               probe_arr_sz(probe_arr_sz),
                                                                               keyvalue_feed_cap(keyvalue_feed_cap),
                                                                               minimum_discretization_sz(minimum_discretization_sz),
                                                                               max_consume_per_load(max_consume_per_load){}

            void open_ticket(size_t sz, model::ticket_id_t * rs) noexcept -> exception_t{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                size_t tentative_discretization_sz  = sz / this->probe_arr_sz;
                size_t discretization_sz            = std::max(tentative_discretization_sz, this->minimum_discretization_sz);
                size_t peeking_base_arr_sz          = sz / discretization_sz + static_cast<size_t>(sz % discretization_sz != 0u); 
                size_t running_sz                   = 0u;

                for (size_t i = 0u; i < peeking_base_arr_sz; ++i){
                    size_t first        = i * discretization_sz;
                    size_t last         = std::max(static_cast<size_t>((i + 1) * discretization_sz), sz);
                    size_t sub_sz       = last - first; 
                    size_t random_clue  = dg::network_randomizer::randomize_int<size_t>(); 
                    size_t base_arr_idx = random_clue & (this->pow2_base_arr_sz - 1u);
                    exception_t err     = this->base_arr[base_arr_idx]->open_ticket(sub_sz, std::next(rs, first));

                    if (dg::network_exception::is_failed(err)){
                        this->close_ticket(rs, running_sz);
                        return err;
                    }

                    for (size_t i = 0u; i < sub_sz; ++i){
                        rs[first + i] = this->internal_encode_ticket_id(rs[first + i], base_arr_idx);                        
                    }

                    running_sz          += sub_sz;
                }

                return dg::network_exception::SUCCESS;
            }

            void assign_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::add_pointer_t<ResponseObserverInterface> * assigning_observer_arr, std::expected<bool, exception_t> * exception_arr) noexcept{

                auto feed_resolutor                 = InternalAssignObserverFeedResolutor{};
                feed_resolutor.controller_arr       = this->base_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    ticket_id_t base_ticket_id                  = {};
                    size_t partitioned_idx                      = {};
                    std::tie(base_ticket_id, partitioned_idx)   = this->internal_decode_ticket_id(ticket_id_arr[i]);

                    if (parititoned_idx >= this->pow2_base_arr_sz){
                        exception_arr[i] = std::unexpected(dg::network_exception::REST_TICKET_NOT_FOUND);
                        continue;
                    }

                    auto feed_arg           = InternalAssignObserverFeedArgument{};
                    feed_arg.base_ticket_id = base_ticket_id;
                    feed_arg.observer       = assigning_observer_arr[i];
                    feed_arg.exception_ptr  = std::next(exception_arr, i);

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }
            }

            void steal_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t> * out_observer_arr) noexcept{

                auto feed_resolutor                 = InternalStealObserverFeedResolutor{};
                feed_resolutor.controller_arr       = this->base_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    ticket_id_t base_ticket_id                  = {};
                    size_t partitioned_idx                      = {};
                    std::tie(base_ticket_id, partitioned_idx)   = this->internal_decode_ticket_id(ticket_id_arr[i]);

                    if (parititoned_idx >= this->pow2_base_arr_sz){
                        out_observer_arr = std::unexpected(dg::network_exception::REST_TICKET_NOT_FOUND);
                        continue;
                    }

                    auto feed_arg               = InternalStealObserverFeedArgument{}:
                    feed_arg.base_ticket_id     = base_ticket_id;
                    feed_arg.out_observer_ptr   = std::next(out_observer_arr, i);

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }
            }

            void get_observer(model::ticket_id_t * ticket_id_arr, size_t sz, std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t> * out_observer_arr) noexcept{

                auto feed_resolutor                 = InternalGetObserverFeedResolutor{};
                feed_resolutor.controller_arr       = this->base_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    ticket_id_t base_ticket_id                  = {};
                    size_t partitioned_idx                      = {};
                    std::tie(base_ticket_id, partitioned_idx)   = this->internal_decode_ticket_id(ticket_id_arr[i]);

                    if (parititoned_idx >= this->pow2_base_arr_sz){
                        out_observer_arr = std::unexpected(dg::network_exception::REST_TICKET_NOT_FOUND);
                        continue;
                    }

                    auto feed_arg               = InternalGetObserverFeedArgument{}:
                    feed_arg.base_ticket_id     = base_ticket_id;
                    feed_arg.out_observer_ptr   = std::next(out_observer_arr, i);

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }
            }

            void close_ticket(model::ticket_id_t * ticket_id_arr, size_t sz) noexcept{

                auto feed_resolutor                 = InternalCloseTicketFeedResolutor{};
                feed_resolutor.controller_arr       = this->base_arr.get();

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                for (size_t i = 0u; i < sz; ++i){
                    ticket_id_t base_ticket_id                  = {};
                    size_t partitioned_idx                      = {};
                    std::tie(base_ticket_id, partitioned_idx)   = this->internal_decode_ticket_id(ticket_id_arr[i]);

                    if (parititoned_idx >= this->pow2_base_arr_sz){
                        out_observer_arr = std::unexpected(dg::network_exception::REST_TICKET_NOT_FOUND);
                        continue;
                    }

                    auto feed_arg           = InternalCloseTicketFeedArgument{};
                    feed_arg.base_ticket_id = base_ticket_id;

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load;
            }
        
        private:

            inline auto internal_encode_ticket_id(ticket_id_t base_ticket_id, size_t base_arr_idx) noexcept -> ticket_id_t{

                static_assert(std::is_unsigned_v<ticket_id_t>);

                size_t popcount = std::countr_zero(this->pow2_base_arr_sz);
                return stdx::safe_unsigned_lshift(base_ticket_id, pop_count) | base_arr_idx;
            }

            inline auto internal_decode_ticket_id(ticket_id_t current_ticket_id) noexcept -> std::pair<ticket_id_t, size_t>{

                static_assert(std::is_unsigned_v<ticket_id_t>);

                size_t popcount             = std::countr_zero(this->pow2_base_arr_sz);
                size_t bitmask              = stdx::lowones_bitgen<size_t>(popcount);
                size_t base_arr_idx         = current_ticket_id & bitmask;
                ticket_id_t base_ticket_id  = current_ticket_id >> popcount;

                return std::make_pair(base_ticket_id, base_arr_idx);
            }

            struct InternalAssignObserverFeedArgument{
                ticket_id_t base_ticket_id;
                std::add_pointer_t<ResponseObserverInterface> observer;
                std::expected<bool, exception_t> * exception_ptr;
            };

            struct InternalAssignObserverFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalAssignObserverFeedArgument>{

                std::unique_ptr<TicketControllerInterface> * controller_arr;

                void push(const size_t& partitioned_idx, std::move_iterator<InternalAssignObserverFeedArgument *> data_arr, size_t sz) noexcept{
                    
                    InternalAssignObserverFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<ticket_id_t[]> ticket_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<ResponseObserverInterface>[]> assigning_observer_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> exception_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        ticket_id_arr[i]            = base_data_arr[i].base_ticket_id;
                        assigning_observer_arr[i]   = base_data_arr[i].observer;
                    }

                    this->controller_arr[partitioned_idx]->assign_observer(ticket_id_arr.get(), sz, assigning_observer_arr.get(), exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        *base_data_arr[i].exception_ptr = exception_arr[i];
                    }
                }
            };

            struct InternalStealObserverFeedArgument{
                ticket_id_t base_ticket_id;
                std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t> * out_observer_ptr;
            };

            struct InternalStealObserverFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalStealObserverFeedArgument>{

                std::unique_ptr<TicketControllerInterface> * controller_arr;

                void push(const size_t& partitioned_idx, std::move_iterator<InternalStealObserverFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalStealObserverFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<ticket_id_t[]> ticket_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t>[]> stealing_observer_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        ticket_id_arr[i] = base_data_arr[i].base_ticket_id;
                    }

                    this->controller_arr[partitioned_idx]->steal_observer(ticket_id_arr.get(), sz, stealing_observer_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        *base_data_arr[i].out_observer_ptr = stealing_observer_arr[i];
                    }
                }
            };

            struct InternalGetObserverFeedArgument{
                ticket_id_t base_ticket_id;
                std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t> * out_observer_ptr;
            };

            struct InternalGetObserverFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, InternalGetObserverFeedArgument>{

                std::unique_ptr<TicketControllerInterface> * controller_arr;

                void push(const size_t& partitioned_idx, std::move_iterator<InternalGetObserverFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalGetObserverFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<ticket_id_t[]> ticket_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t>[]> getting_observer_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        ticket_id_arr[i] = base_data_arr[i].base_ticket_id;
                    }

                    this->controller_arr[partitioned_idx]->get_observer(ticket_id_arr.get(), sz, getting_observer_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        *base_data_arr[i].out_observer_arr = getting_observer_arr[i];
                    }
                }
            };

            struct InternalCloseTicketFeedArgument{
                ticket_id_t base_ticket_id;
            };

            struct InternalCloseTicketFeedResolutor: dg::network_producer_consumer::KVConsumerInterface<size_t, ticket_id_t>{

                std::unique_ptr<TicketControllerInterface> * controller_arr;

                void push(const size_t& partitioned_idx, std::move_iterator<InternalCloseTicketFeedArgument *> data_arr, size_t sz) noexcept{

                    InternalCloseTicketFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<ticket_id_t[]> ticket_id_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        ticket_id_arr[i] = base_data_arr[i].base_ticket_id;
                    }

                    this->controller_arr[partitioned_idx]->close_ticket(ticket_id_arr.get(), sz);
                }
            };
    };

    //we need to implement batch AVL inserts, this is harder than expected
    class TicketTimeoutManager: public virtual TicketTimeoutManagerInterface{

        public:

            struct ExpiryBucket{
                model::ticket_id_t ticket_id;
                std::chrono::time_point<std::chrono::steady_clock> abs_timeout;
            };

        private:

            dg::vector<ExpiryBucket> expiry_bucket_queue; //this is harder than expected, we are afraid of the priority queue, yet we would want to discretize this to avoid priority queues
            size_t expiry_bucket_queue_cap;
            std::unique_ptr<std::mutex> mtx;
            stdx::hdi_container<std::chrono::nanoseconds> max_dur;
            stdx::hdi_container<size_t> max_consume_per_load;

        public:

            TicketTimeoutManager(dg::vector<ExpiryBucket> expiry_bucket_queue,
                                 size_t expiry_bucket_queue_cap,
                                 std::unique_ptr<std::mutex> mtx,
                                 stdx::hdi_container<std::chrono::nanoseconds> max_dur,
                                 stdx::hdi_container<size_t> max_consume_per_load) noexcept: expiry_bucket_queue(std::move(expiry_bucket_queue)),
                                                                                             expiry_bucket_queue_cap(expiry_bucket_queue_cap),
                                                                                             mtx(std::move(mtx)),
                                                                                             max_dur(std::move(max_dur)),
                                                                                             max_consume_per_load(std::move(max_consume_per_load)){}

            void clock_in(std::pair<model::ticket_id_t, std::chrono::nanoseconds> * registering_arr, size_t sz, exception_t * exception_arr) noexcept{

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }
                
                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                auto now        = std::chrono::steady_clock::now(); 
                auto greater    = [](const ExpiryBucket& lhs, const ExpiryBucket& rhs) noexcept {return lhs.abs_timepout > rhs.abs_timeout;};

                for (size_t i = 0u; i < sz; ++i){
                    auto [ticket_id, current_dur] = registering_arr[i];

                    if (current_dur > this->max_clockin_dur()){
                        exception_arr[i] = dg::network_exception::REST_INVALID_TIMEOUT;
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
                auto now        = std::chrono::steady_clock::now();
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

            auto max_clockin_dur() const noexcept -> std::chrono::nanoseconds{

                return this->max_dur.value;
            }

            void clear() noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                this->expiry_bucket_queue.clear();
            }

            auto max_consume_size() noexcept -> size_t{

                return this->max_consume_per_load.value;
            }
    };

    //this is to utilize CPU resource, not CPU efficiency, CPU efficiency is already achieved by batching ticket_id_arr, kernel has effective + efficient schedulings that render the overhead of mtx -> 1/1000 of the actual tx 
    class DistributedTicketTimeoutManager: public virtual TicketTimeoutManagerInterface{

        private:

            std::unique_ptr<std::unique_ptr<TicketTimeoutManagerInterface>[]> base_arr;
            size_t pow2_base_arr_sz;
            size_t keyvalue_feed_cap;
            size_t zero_bounce_sz;
            std::unique_ptr<DrainerPredicateInterface> drainer_predicate;
            std::chrono::nanoseconds max_dur;
            size_t drain_peek_cap_per_container;
            size_t max_consume_per_load;

        public:

            DistributedTicketTimeoutManager(std::unique_ptr<std::unique_ptr<TicketTimeoutManagerInterface>[]> base_arr,
                                            size_t pow2_base_arr_sz,
                                            size_t keyvalue_feed_cap,
                                            size_t zero_bounce_sz,
                                            std::unique_ptr<DrainerPredicateInterface> drainer_predicate,
                                            std::chrono::nanoseconds max_dur,
                                            size_t drain_peek_cap_per_container,
                                            size_t max_consume_per_load) noexcept: base_arr(std::move(base_arr)),
                                                                                   pow2_base_arr_sz(pow2_base_arr_sz),
                                                                                   keyvalue_feed_cap(keyvalue_feed_cap),
                                                                                   zero_bounce_sz(zero_bounce_sz),
                                                                                   drainer_predicate(std::move(drainer_predicate)),
                                                                                   max_dur(max_dur),
                                                                                   drain_peek_cap_per_container(drain_peek_cap_per_container),
                                                                                   max_consume_per_load(max_consume_per_load){}

            void clock_in(std::pair<model::ticket_id_t, std::chrono::nanoseconds> * registering_arr, size_t sz, exception_t * exception_arr) noexcept{

                auto feed_resolutor                 = InternalClockInFeedResolutor{};
                feed_resolutor.manager_arr          = this->base_arr.get(); 

                size_t trimmed_keyvalue_feed_cap    = std::min(this->keyvalue_feed_cap, sz);
                size_t feeder_allocation_cost       = dg::network_producer_consumer::delvrsrv_kv_allocation_cost(&feed_resolutor, trimmed_keyvalue_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                         = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_kv_open_preallocated_raiihandle(&feed_resolutor, trimmed_keyvalue_feed_cap, feeder_mem.get()));

                std::fill(exception_arr, std::next(exception_arr, sz), dg::network_exception::SUCCESS);

                for (size_t i = 0u; i < sz; ++i){
                    if (registering_arr[i].second > this->max_clockin_dur()){
                        exception_arr[i] = dg::network_exception::REST_INVALID_TIMEOUT;
                        continue;
                    }

                    size_t hashed_value     = dg::network_hash::hash_reflectible(registering_arr[i].first);
                    size_t partitioned_idx  = hashed_value & (this->pow2_base_arr_sz - 1u);

                    auto feed_arg           = InternalClockInFeedArgument{};
                    feed_arg.ticket_id      = registering_arr[i].first;
                    feed_arg.dur            = registering_arr[i].second;
                    feed_arg.exception_ptr  = std::next(exception_arr, i); 

                    dg::network_producer_consumer::delvrsrv_kv_deliver(feeder.get(), partitioned_idx, feed_arg);
                }
            }

            auto max_clockin_dur() const noexcept -> std::chrono::nanoseconds{

                return this->max_dur;
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

                std::unique_ptr<TicketTimeoutManagerInterface> * manager_arr;

                void push(const size_t& partitioned_idx, std::move_iterator<InternalClockInFeedArgument *> data_arr, size_t sz) noexcept{
                    
                    InternalClockInFeedArgument * base_data_arr = data_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<std::pair<model::ticket_id_t, std::chrono::nanoseconds>[]> registering_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        registering_arr[i] = std::make_pair(base_data_arr[i].ticket_id, base_data_arr[i].dur);
                    }

                    this->manager_arr[partitioned_idx]->clock_in(registering_arr.get(), sz, exception_arr.get());

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

    class ExhaustionControlledTicketTimeoutManager: public virtual TicketTimeoutManagerInterface{

        private:

            std::unique_ptr<TicketTimeoutManagerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device; //its insanely hard to solve this by using notify + atomic_wait, we'll come up with a patch 

        public:

            ExhaustionControlledTicketTimeoutManager(std::unique_ptr<TicketTimeoutManagerInterface> base,
                                                     std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> infretry_device) noexcept: base(std::move(base)),
                                                                                                                                                       infretry_device(std::move(infretry_device)){}

            void clock_in(std::pair<model::ticket_id_t, std::chrono::nanoseconds> * registering_arr, size_t sz, exception_t * exception_arr) noexcept{

                auto first_registering_ptr          = registering_arr;
                auto last_registering_ptr           = std::next(first_registering_ptr, sz);
                exception_t * first_exception_ptr   = exception_arr;
                exception_t * last_exception_ptr    = std::next(first_exception_ptr, sz);
                size_t sliding_window_sz            = sz;

                auto task = [&, this]() noexcept{
                    this->base->clock_in(first_registering_ptr, sliding_window_sz, first_exception_ptr);

                    exception_t * first_retriable_exception_ptr = std::find(first_exception_ptr, last_exception_ptr, dg::network_exception::QUEUE_FULL);
                    exception_t * last_retriable_exception_ptr  = std::find_if(first_retriable_exception_ptr, last_exception_ptr, [](exception_t err){return err != dg::network_exception::QUEUE_FULL;});

                    size_t relative_offset                      = std::distance(first_exception_ptr, first_retriable_exception_ptr);
                    sliding_window_sz                           = std::distance(first_retriable_exception_ptr, last_retriable_exception_ptr);

                    std::advance(first_registering_ptr, relative_offset);
                    std::advance(first_exception_ptr, relative_offset);

                    return first_exception_ptr == last_exception_ptr;
                };

                dg::network_concurrency_infretry_x::ExecutableWrapper virtual_task(task);
                this->infretry_device->exec(virtual_task);
            }

            void get_expired_ticket(model::ticket_id_t * output_arr, size_t& sz, size_t cap) noexcept{

                this->base->get_expired_ticket(output_arr, sz, cap);
            }

            auto max_clockin_dur() const noexcept -> std::chrono::nanoseconds{

                return this->base->max_clockin_dur();
            }

            void clear() noexcept{

                this->base->clear();
            }

            auto max_consume_size() noexcept -> size_t{

                return this->base->max_consume_size();
            }
    };

    struct AtomicBlock{
        uint32_t thru_counter;
        uint32_t ver_ctrl; //this pattern is called version control update, thru_counter could be a pointer to a new const_read block state, this pattern is equivalent to mutual exclusion stdx::lock_guard<std::mutex> inout transaction, in the sense that every method invoke only is only allowed to hold the mutual exclusion if they read the right ticket number (there is no guy calling at the time ...)

        //assume we are the unique ver_ctrl, nothing happened
        //assume we are sharing the ver_ctrl, the first guy gets the right to update, forfeit all other guys result (as if the other guys never invoke one of the methods by updating the version control, which bring this to we are the unique_ver_ctrl, nothing happened)
        //                                                                                                           proof by contradiction, assume other guy affect the internal state, they must got the ver_ctrl incrementation 

        //we postpone the atomicity of transaction -> compare_exchange_strong
        //and the state propagation -> ver_ctrl
        
        //another pattern is called state propagation by using cmpexch, such is logically harder to program, we bring the state from a random correct state -> another random correct state 
        //imagine malloc + free for example, we dont know if the old -> new value is unique, a decade could happen when we changing the head of the linked list, the comparing result is no longer relevant 

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) const noexcept{
            reflector(thru_counter, ver_ctrl);
        }

        template <class Reflector>
        constexpr void dg_reflect(const Reflector& reflector) noexcept{
            reflector(thru_counter, ver_ctrl);
        }
    };

    using atomic_block_pragma_0_t = std::array<char, sizeof(uint32_t) + sizeof(uint32_t)>;

    class CacheUniqueWriteTrafficController: public virtual CacheUniqueWriteTrafficControllerInterface{

        private:

            stdx::inplace_hdi_container<std::atomic<atomic_block_pragma_0_t>> block_ctrl;
            stdx::hdi_container<size_t> thru_cap;

            static constexpr auto make_pragma_0_block(AtomicBlock arg) noexcept -> atomic_block_pragma_0_t{

                constexpr size_t TRIVIAL_SZ = dg::network_trivial_serializer::size(arg);
                static_assert(TRIVIAL_SZ <= sizeof(atomic_block_pragma_0_t));
                auto rs = atomic_block_pragma_0_t{};
                dg::network_trivial_serializer::serialize_into(rs.data(), arg);

                return rs;
            }

            static constexpr auto read_pragma_0_block(atomic_block_pragma_0_t arg) noexcept -> AtomicBlock{

                auto rs = AtomicBlock{};
                dg::network_trivial_serializer::deserialize_into(rs, arg.data());

                return rs;
            }

        public:

            using self = CacheUniqueWriteTrafficController;

            CacheUniqueWriteTrafficController(size_t thru_cap) noexcept: thru_counter(std::in_place_t{}, self::make_pragma_0_block(AtomicBlock{0u, 0u})),
                                                                         thru_cap(stdx::hdi_containter<size_t>{thru_cap}){}

            auto thru(size_t incoming_sz) noexcept -> exception_t{

                while (true){
                    atomic_block_pragma_0_t then_block  = this->block_ctrl.value.load(std::memory_order_relaxed);
                    AtomicBlock then_semantic_block     = self::read_pragma_0_block(then_block);

                    if (then_semantic_block.thru_counter + incoming_sz > this->thru_cap.value){
                        return dg::network_exception::REST_BAD_CACHE_TRAFFIC;
                    }

                    AtomicBlock now_semantic_block      = AtomicBlock{then_semantic_block.thru_counter + incoming_sz, then_semantic_block.ver_ctrl + 1u};
                    atomic_block_pragma_0_t now_block   = self::make_pragma_0_block(now_semantic_block);

                    bool was_updated                    = this->block_ctrl.value.compare_exchange_strong(then_block, now_block, std::memory_order_relaxed);

                    if (was_updated){
                        return dg::network_exception::SUCCESS;
                    }
                }
            }

            void reset() noexcept{

                while (true){
                    atomic_block_pragma_0_t then_block  = this->block_ctrl.value.load(std::memory_order_relaxed);
                    AtomicBlock then_semantic_block     = self::read_pragma_0_block(then_block); 
                    AtomicBlock now_semantic_block      = AtomicBlock{0u, then_semantic_block.ver_ctrl + 1u};
                    atomic_block_pragma_0_t now_block   = self::make_pragma_0_block(now_semantic_block);

                    bool was_updated                    = this->block_ctrl.value.compare_exchange_strong(then_block, now_block, std::memory_order_relaxed);

                    if (was_updated){
                        return;
                    }
                } 
            }
    };

    class SubscriptibleWrappedResetTrafficController: public virtual UpdatableInterface{

        private:

            stdx::hdi_container<std::chrono::nanoseconds> update_dur;
            stdx::inplace_hdi_container<std::atomic<std::chrono::time_point<std::chrono::steady_clock>>> last_update;
            std::shared_ptr<CacheUniqueWriteTrafficControllerInterface> updating_component;

        public:

            SubscriptibleWrappedResetTrafficController(std::chrono::nanoseconds update_dur,
                                                       std::chrono::time_point<std::chrono::steady_clock> last_update,
                                                       std::shared_ptr<CacheUniqueWriteTrafficControllerInterface> updating_component) noexcept: update_dur(stdx::hdi_container<std::chrono::nanoseconds>{update_dur}),
                                                                                                                                                 last_update(std::in_place_t{}, last_update),
                                                                                                                                                 updating_component(std::move(updating_component)){}

            void update() noexcept{

                //attempt to do atomic_cmpexch to take unique update responsibility, clock always goes forward in time

                std::chrono::time_point<std::chrono::steady_clock> last_update_value    = this->last_update.value.load(std::memory_order_relaxed);
                std::chrono::time_point<std::chrono::steady_clock> now                  = std::chrono::steady_clock::now();
                std::chrono::nanoseconds diff                                           = std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_update_value);

                if (diff < this->update_dur.value){
                    return;
                }

                if (now == last_update_value){
                    return; //bad resolution
                }

                //take update responsibility

                bool has_mtx_ticket = this->last_update.value.compare_exchange_strong(last_update_value, now, std::memory_order_relaxed);

                if (!has_mtx_ticket){
                    return; //other guy got the ticket
                }

                this->updating_component->reset(); //thru
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

                size_t trimmed_ticket_controller_feed_cap   = std::min(this->ticket_controller_feed_cap, buf_arr_sz);
                size_t feeder_allocation_cost               = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_ticket_controller_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                                 = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_ticket_controller_feed_cap, feeder_mem.get())); 

                for (size_t i = 0u; i < buf_arr_sz; ++i){
                    std::expected<model::InternalResponse, exception_t> response = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize<model::InternalResponse, dg::string>(buf_arr[i], model::INTERNAL_RESPONSE_SERIALIZATION_SECRET));

                    if (!response.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(response.error()));
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

                    model::InternalResponse * base_response_arr = response_arr.base();

                    dg::network_stack_allocation::NoExceptAllocation<model::ticket_id_t[]> ticket_id_arr(sz);
                    dg::network_stack_allocation::NoExceptAllocation<std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t>[]> observer_arr(sz);

                    for (size_t i = 0u; i < sz; ++i){
                        ticket_id_arr[i] = base_response_arr[i].ticket_id;
                    }

                    this->ticket_controller->steal_observer(ticket_id_arr.get(), sz, observer_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (!observer_arr[i].has_value()){
                            continue;
                        }

                        observer_arr[i].value()->deferred_memory_ordering_fetch(std::move(base_response_arr[i].response));
                    }

                    std::atomic_thread_fence(std::memory_order_release);
                    //we'd like to have a signal fence std::memory_order_seq_cst just for our sake of safety
                    std::atomic_signal_fence(std::memory_order_seq_cst);

                    for (size_t i = 0u; i < sz; ++i){
                        if (!observer_arr[i].has_value()){
                            continue;
                        }

                        observer_arr[i].value()->deferred_memory_ordering_fetch_close();
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

                size_t trimmed_mailbox_feed_cap = std::min(std::min(this->mailbox_feed_cap, dg::network_kernel_mailbox::max_consume_size()), static_cast<size_t>(request_vec.size()));
                size_t feeder_allocation_cost   = dg::network_producer_consumer::delvrsrv_allocation_cost(&feed_resolutor, trimmed_mailbox_feed_cap);
                dg::network_stack_allocation::NoExceptRawAllocation<char[]> feeder_mem(feeder_allocation_cost);
                auto feeder                     = dg::network_exception_handler::nothrow_log(dg::network_producer_consumer::delvrsrv_open_preallocated_raiihandle(&feed_resolutor, trimmed_mailbox_feed_cap, feeder_mem.get()));

                for (model::InternalRequest& request: request_vec){
                    std::expected<dg::network_kernel_mailbox::Address, exception_t> addr = dg::network_uri_encoder::extract_mailbox_addr(request.request.requestee_uri);

                    if (!addr.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(addr.error()));
                        continue;
                    }

                    std::expected<dg::string, exception_t> bstream = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_serialize<dg::string, model::InternalRequest>)(request, model::INTERNAL_REQUEST_SERIALIZATION_SECRET);

                    if (!bstream.has_value()){
                        dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(bstream.error()));
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

    //this is probably the best part and also the worst part
    //best part in the sense that we dont have real_time scheduling
    //worst part in the sense that we dont have real_time scheduling

    //our uncertainty is probably 5 milliseconds/ check, which is fairly latencied out
    //the other mailbox problem can be solved with large inbounds, probably by streaming padded data to push small data through with low latency
    //we dont have a solution to this problem

    //let's see what we want
    //99% of the time, low latency path is the success path, high latency part is the failed paths (not recving response)
    //for the success path, there is literally no waiting, we push data directly to dg::vector<> -> consumed by worker -> mailbox -> recv_mailbox -> get_data -> fetch

    //for the failed paths
    //usually we set timeout for 10ms - 100ms to recv response
    //the uncertainty of this component decreases in such case
    //so that is no longer an issue

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
                    dg::network_stack_allocation<std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t>[]> stolen_response_observer_arr(ticket_id_arr_sz);

                    this->ticket_controller->steal_observer(base_ticket_id_arr, ticket_id_arr_sz, stolen_response_observer_arr.get());

                    for (size_t i = 0u; i < ticket_id_arr_sz; ++i){
                        if (!stolen_response_observer_arr[i].has_value()){
                            continue;
                        }

                        stolen_response_observer_arr[i].value()->deferred_memory_ordering_fetch(std::unexpected(dg::network_exception::REST_TIMEOUT));
                    }

                    std::atomic_thread_fence(std::memory_order_release);
                    //we'd like to have a signal fence std::memory_order_seq_cst just for our sake of safety
                    std::atomic_signal_fence(std::memory_order_seq_cst);

                    for (size_t i = 0u; i < ticket_id_arr_sz; ++i){
                        if (!stolen_response_observer_arr[i].has_value()){
                            continue;
                        }

                        stolen_response_observer_arr[i].value()->deferred_memory_ordering_fetch_close();
                    }
                }
            };
    };

    class UpdateWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<UpdatableInterface> updatable;
            std::chrono::nanoseconds heartbeat_dur;

        public:

            UpdateWorker(std::shared_ptr<UpdatableInterface> updatable,
                         std::chrono::nanoseconds heartbeat_dur) noexcept: updatable(std::move(updatable)),
                                                                           heartbeat_dur(heartbeat_dur){}

            auto run_one_epoch() noexcept -> bool{

                this->updatable->update();
                std::this_thread::sleep_for(this->heartbeat_dur);

                return true;
            }
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

    //this is the base for our pagination implementation of requests
    //its like a movie player
    //we have a vector of client request (timeranges)
    //client request is running out, we refill our waiting pool of outbound request + inbound response

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

            void request(model::ClientRequest&& client_request) noexcept -> std::expected<std::unique_ptr<ResponseInterface>, exception_t>{

                std::expected<std::unique_ptr<BatchResponseInterface>, exception_t> resp = this->batch_request(std::make_move_iterator(std::addressof(client_request)), 1u); //alright, this might not be defined

                if (!resp.has_value()){
                    return std::unexpected(resp.error());
                }

                return self::internal_make_single_response(static_cast<std::unique_ptr<BatchResponseInterface>&&>(resp.value()));
            }

            //alright, we'll do debugging
            //we'll analyze the success path
            //then we'll analyze the failed paths, their exit points and the effects on the component internal states

            auto batch_request(std::move_iterator<model::ClientRequest *> client_request_arr, size_t sz) noexcept -> std::expected<std::unique_ptr<BatchResponseInterface>, exception_t>{

                //the code is not hard, yet it is extremely easy to leak resources

                if constexpr(DEBUG_MODE_FLAG){
                    if (sz > this->max_consume_size()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                if (sz == 0u){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT); //invalid argument, internal states are not affected, no leaks
                }

                model::ClientRequest * base_client_request_arr = client_request_arr.base();
                dg::network_stack_allocation::NoExceptAllocation<model::ticket_id_t[]> ticket_id_arr(sz);
                exception_t err = this->ticket_controller->open_ticket(sz, ticket_id_arr.get()); //open the ticket

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err); //failed to open tickets, no actions
                }

                auto ticket_resource_grd = stdx::resource_guard([&]() noexcept{
                    this->ticket_controller->close_ticket(ticket_id_arr.get(), sz);
                });

                std::expected<std::unique_ptr<InternalBatchResponse>, exception_t> response = self::internal_make_batch_request_response(sz, ticket_id_arr.get(), this->ticket_controller); //open batch_response associated with the tickets, take ticket responsibility

                if (!response.has_value()){
                    return std::unexpected(response.error()); //failed to open response, close the tickets 
                }

                ticket_resource_grd.release(); //ticket responsibility tranferred -> internal_batch_response

                auto response_resource_grd = stdx::resource_guard([&]() noexcept{
                    response.value()->release_response_wait_responsibility();
                });

                dg::network_stack_allocation::NoExceptAllocation<std::add_pointer_t<ResponseObserverInterface>[]> response_observer_arr(sz);
                dg::network_stack_allocation::NoExceptAllocation<std::expected<bool, exception_t>[]> response_observer_exception_arr(sz);

                for (size_t i = 0u; i < sz; ++i){
                    std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t> observer = response.value()->get_observer(i); 

                    if (!observer.has_value()){
                        return std::unexpected(observer.error()); //failed to get observers, close_tickets by response + release wait responsibility of response + deallocate response resources
                    }

                    response_observer_arr[i] = observer.value(); //get response listening observers
                }

                this->ticket_controller->assign_observer(ticket_id_arr.get(), sz, response_observer_arr.get(), response_observer_exception_arr.get()); //bind observers -> ticket_controller to listen for responses

                for (size_t i = 0u; i < sz; ++i){
                    if (!response_observer_exception_arr[i].has_value()){
                        return std::unexpected(response_observer_exception_arr[i].error()); //failed to bind observers, close tickets by response + release_wait_responsbiility of response + deallocate response resources
                    }

                    dg::network_exception_handler::dg_assert(response_observer_exception_arr[i].value());
                }

                std::chrono::nanoseconds max_timeout_dur = this->ticket_timeout_manager->max_clockin_dur(); 

                dg::network_stack_allocation::NoExceptAllocation<std::pair<model::ticket_id_t, std::chrono::nanoseconds>[]> clockin_arr(sz);
                dg::network_stack_allocation::NoExceptAllocation<exception_t[]> clockin_exception_arr(sz);

                for (size_t i = 0u; i < sz; ++i){
                    clockin_arr[i] = std::make_pair(ticket_id_arr[i], base_client_request_arr[i].client_timeout_dur);

                    if (base_client_request_arr[i].client_timeout_dur > max_timeout_dur){
                        return std::unexpected(dg::network_exception::REST_INVALID_TIMEOUT); //failed to meet timeout preconds, close tickets by response + release_wait_responsbiility of response + deallocate response resources
                    }
                }

                std::expected<dg::vector<model::InternalRequest>, exception_t> pushing_container = this->internal_make_internal_request(std::make_move_iterator(base_client_request_arr), ticket_id_arr.get(), sz);

                if (!pushing_container.has_value()){
                    return std::unexpected(pushing_container.error()); //failed to create a pushed container, client_request_arr remains intact, close tickets by response + release_wait_responsibility of response + deallocate response resources
                }

                exception_t push_err = this->request_container->push(static_cast<dg::vector<model::InternalRequest>&&>(pushing_container.value())); //push the outbound request

                if (dg::network_exception::is_failed(push_err)){
                    this->internal_rollback_client_request(base_client_request_arr, std::move(pushing_container.value()));
                    return std::unexpected(push_err); //failed to push thru the container, ticket_id is not referenced by other components, base_client_request_arr is not intact, reverse the operation, close tickets + release response wait responsibility
                }

                //thru

                this->ticket_timeout_manager->clock_in(clockin_arr.get(), sz, clockin_exception_arr.get()); //clock in the tickets to rescue

                for (size_t i = 0u; i < sz; ++i){
                    if (dg::network_exception::is_failed(clockin_exception_arr[i])){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(clockin_exception_arr[i])); //unable to fail, resource leaks + deadlock otherwise, very dangerous, rather terminate
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

                    //defer construction -> factory, all deferred constructions are marked as noexcept to avoid leaks, such is malloc() -> inplace -> return, fails can only happen at malloc 

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

                    auto get_observer(size_t idx) noexcept -> std::expected<std::add_pointer_t<ResponseObserverInterface>, exception_t>{

                        return this->base->get_observer(idx);
                    }

                    void release_response_wait_responsibility() noexcept{

                        this->base->release_response_wait_responsibility();
                    }

                    void release_ticket_release_responsibility() noexcept{

                        this->ticket_release_responsibility = false;
                    }

                private:

                    void release_ticket() noexcept{

                        auto task = [](InternalBatchResponse * self_obj) noexcept{
                            if (!self_obj->ticket_release_responsibility){
                                return;
                            }

                            self_obj->ticket_controller->close_ticket(self_obj->ticket_id_arr.get(), self_obj->ticket_id_arr_sz);                        
                            self_obj->ticket_release_responsibility = false;
                        };

                        stdx::noipa_do_task(task, this); //this is incredibly hard to get right
                    }
            };

            class InternalSingleResponse: public virtual ResponseInterface{

                private:

                    std::unique_ptr<BatchResponseInterface> base;
                
                public:

                    InternalSingleResponse(std::unique_ptr<BatchResponseInterface> base) noexcept: base(std::move(base)){}

                    auto response() noexcept -> std::expected<Response, exception_t>{

                        auto rs = this->base->response();

                        if (!rs.has_value()){
                            return std::unexpected(rs.error());
                        }

                        if constexpr(DEBUG_MODE_FLAG){
                            if (rs->size() != 1u){
                                dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                                std::abort();
                            }
                        }

                        static_assert(std::is_nothrow_move_constructible_v<Response>);
                        return std::expected<Response, exception_t>(std::move(rs->front()));
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

                std::expected<std::unique_ptr<model::ticket_id_t[]>, exception_t> cpy_ticket_id_arr = dg::network_allocation::cstyle_make_unique<ticket_id_t[]>(request_sz);

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

            static auto internal_make_single_response(std::unique_ptr<BatchResponseInterface>&& base) noexcept -> std::expected<std::unique_ptr<ResponseInterface>, exception_t>{

                return dg::network_exception::cstyle_make_unique<InternalSingleResponse>(static_cast<std::unique_ptr<BatchResponseInterface>&&>(base));
            }

            static auto internal_make_internal_request(std::move_iterator<model::ClientRequest *> request_arr, ticket_id_t * ticket_id_arr, size_t request_arr_sz) noexcept -> std::expected<dg::vector<model::InternalRequest>, exception_t>{

                model::ClientRequest * base_request_arr                             = request_arr.base();
                std::expected<dg::vector<model::InternalRequest>, exception_t> rs   = dg::network_exception::cstyle_initialize<dg::vector<model::InternalRequest>>(request_arr_sz);

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

            static void internal_rollback_client_request(model::ClientRequest * client_request_arr, dg::vector<model::InternalRequest>&& internal_request_arr) noexcept{

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
