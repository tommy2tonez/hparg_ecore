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

//let me think
//request should be an absolute unit, because mailbox can do flash_streamx, aggregation of requests, it makes no sense to unitize things here

//dg::vector<model::InternalRequest> as a unit of consume (1)
//std::unique_ptr<binary_semaphore> to do absolute waiting to avoid shared_ptr<> (2)
//we dispatch the release workorder to a third party, the third party would take release right after a certain period from the ticket_controller(3)
//a hash_distributed ticket_controller (4)
//abs_timeout on server side to do hard synchronization of failed request (no response requests), such is we can do another request without being afraid of overriding (5)
//we need to reduce the memory orderings, binary semaphore for each of the requests in the batch does not sound, does not scale (6)
//we can't factor in time-dilation because there is literally no instruments (7)
//we need to do some sort of encoding + decoding schemes, we can't offload this responsibility to user-space because there are sensitive datas like timeout uri + etc. (8)
//we can skip the encoding + decodings for now
//we'll be working on this component for the next week
//this is a very important component

//our focus would be to not pollute the system memory-orderings-wise
//we have a lot of other jobs to run concurrently, memory-orderings acquire release is a no-no in these situations
//we are hoping we could push 10GB of rest request/ second, it's hard
//this is our main way of doing external comm
//I dont know why our client is being pushy
//we already told them to give us a hard deadline of within 6months - a year to deploy on the mainframe
//I dont know what's so hard with these foos about writing a Taylor Series approximations

namespace dg::network_rest_frame::model{

    //we'll be back tomorrow
    //it's been an exhausting day
    //we are on a rampage of nonstop incoming requests

    using ticket_id_t   = uint64_t;
    using clock_id_t    = uint64_t;
    
    static inline constexpr uint32_t INTERNAL_REQUEST_SERIALIZATION_SECRET  = 3312354321ULL;
    static inline constexpr uint32_t INTERNAL_RESPONSE_SERIALIZATION_SECRET = 3554488158ULL;

    struct ClientRequest{
        dg::string requestee_uri;
        dg::string requestor;
        dg::string payload;
        std::chrono::nanoseconds client_timeout_dur;
        std::optional<std::chrono::time_point<std::chrono::utc_clock>> server_abs_timeout; //this is hard to solve, we can be stucked in a pipe and actually stay there forever, abs_timeout only works for post the transaction, which is already too late, I dont know of the way to do this correctly

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(requestee_uri, requestor, payload, client_timeout_dur, server_abs_timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(requestee_uri, requestor, payload, client_timeout_dur, server_abs_timeout);
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
        std::optional<std::chrono::time_point<std::chrono::utc_clock>> server_abs_timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(request, ticket_id, server_abs_timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(request, ticket_id, server_abs_timeout);
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
        virtual void maybedefer_memory_ordering_fetch(std::expected<Response, exception_t>) noexcept = 0;
        virtual void maybedefer_memory_ordering_fetch_close(void * dirty_memory) noexcept = 0;
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
        virtual void request(std::move_iterator<model::ClientRequest *>, size_t, std::expected<std::unique_ptr<ResponseInterface>, exception_t> *) noexcept = 0; 
        virtual auto max_consume_size() noexcept -> size_t = 0;
    };

    auto make_raii_ticket(model::ticket_id_t ticket_id, std::shared_ptr<TicketControllerInterface> ticket_controller) noexcept -> std::shared_ptr<model::ticket_id_t>{
        
        auto destructor = [controller_arg = std::move(ticket_controller)](model::ticket_id_t * arg) noexcept{
            controller_arg->close_ticket(*arg);
            delete arg;
        };

        return std::unique_ptr<model::ticket_id_t, dectype(destructor)>(new ticket_id_t{ticket_id}, std::move(destructor));
    }
}

namespace dg::network_rest_frame::server_impl1{

    using namespace dg::network_rest_frame::server; 

    class RequestResolverWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            dg::unordered_unstable_map<dg::string, std::unique_ptr<RequestHandlerInterface>> request_handler_map;
            uint32_t resolve_channel;
            size_t resolve_consume_sz;
            uint32_t response_channel;
            size_t response_vectorization_sz;
            size_t busy_consume_sz; 

        public:

            RequestResolverWorker(dg::unordered_unstable_map<dg::string, std::unique_ptr<RequestHandlerInterface>> request_handler_map,
                                  uint32_t resolve_channel,
                                  size_t resolve_consume_sz,
                                  uint32_t response_channel,
                                  size_t response_vectorization_sz,
                                  size_t busy_consume_sz) noexcept: request_handler_map(std::move(request_handler_map)),
                                                                    resolve_channel(resolve_channel),
                                                                    resolve_consume_sz(resolve_consume_sz),
                                                                    response_channel(response_channel),
                                                                    response_vectorization_sz(response_vectorization_sz),
                                                                    busy_consume_sz(busy_consume_sz){}

            bool run_one_epoch() noexcept{

                size_t recv_buf_cap             = this->resolve_consume_sz;
                size_t recv_buf_sz              = {};
                dg::network_stack_allocation::NoExceptAllocation<dg::string[]> recv_buf_arr(recv_buf_cap);
                dg::network_kernel_mailbox::recv(this->resolve_channel, recv_buf_arr.get(), recv_buf_sz, recv_buf_cap);

                auto feed_resolutor             = InternalResponseFeedResolutor{}; 
                feed_resolutor.mailbox_channel  = this->response_channel;
            
                size_t trimmed_vectorization_sz = std::min(this->response_vectorization_sz, recv_buf_sz);
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

                    model::InternalResponse response{};

                    auto now = std::chrono::utc_clock::now();

                    if (request->server_abs_timeout.has_value() && request->server_abs_timeout.value() <= now){
                        response = model::InternalResponse{std::unexpected(dg::network_exception::REST_ABSTIMEOUT), request->ticket_id};
                    } else{
                        std::expected<dg::string, exception_t> resource_path = dg::network_uri_encoder::extract_local_path(request->request.requestee_uri);

                        if (!resource_path.has_value()){
                            response = model::InternalResponse{model::Response{{}, resource_path.error()}, request->ticket_id};
                        } else{
                            auto map_ptr = this->request_handler.find(resource_path.value());

                            if (map_ptr == this->request_handle.end()){
                                response = model::InternalResponse{model::Response{{}, dg::network_exception::REST_INVALID_URI}, request->ticket_id};
                            } else{
                                response = model::InternalResponse{map_ptr->second->handle(std::move(request->request)), request->ticket_id};
                            }
                        }
                    }

                    std::expected<dg::string, exception_t> response_buf = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_serialize<dg::string, model::InternalResponse>)(response, model::INTERNAL_RESPONSE_SERIALIZATION_SECRET);

                    if (!response_buf.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(response_buf.error()));
                        continue;
                    }

                    auto feed_arg       = dg::network_kernel_mailbox::MailBoxArgument{}:
                    feed_arg.to         = requestor_addr.value();
                    feed_arg.content    = std::move(response_buf.value());
 
                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), std::move(feed_arg));
                }

                return recv_buf_sz >= this->busy_consume_sz;
            }
        
        private:

            struct InternalResponseFeedResolutor: dg::network_producer_consumer::ConsumerInterface<dg::network_kernel_mailbox::MailBoxArgument>{

                uint32_t mailbox_channel;

                void push(std::move_iterator<MailBoxArgument *> mailbox_arr, size_t sz) noexcept{

                    dg::network_stack_allocation::NoExceptAllocation<exception_t[]> exception_arr(sz);
                    dg::network_kernel_mailbox::send(this->mailbox_channel, mailbox_arr, sz, exception_arr.get());

                    for (size_t i = 0u; i < sz; ++i){
                        if (dg::network_exception::is_failed(exception_arr[i])){
                            dg::network_log_stackdump::error_fast_optional(dg::network_exception::verbose(exception_arr[i]));
                        }
                    }
                }
            }
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

            void maybedefer_memory_ordering_fetch(std::expected<Response, exception_t> response_arg) noexcept{

                this->update(std::move(response_arg));
            }

            void maybedefer_memory_ordering_fetch_close(void * dirty_memory) noexcept{

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

            void maybedefer_memory_ordering_fetch(std::expected<Response, exception_t> resp) noexcept{

                this->base->maybedefer_memory_ordering_fetch(std::move(resp));
            }

            void maybedefer_memory_ordering_fetch_close(void * dirty_memory) noexcept{

                this->base->maybedefer_memory_ordering_fetch_close(dirty_memory);
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
                std::atomic_thread_fence(std::memory_order_release);
                this->atomic_smp.fetch_add(1u, std::memory_order_relaxed);
            }

            void maybedefer_memory_ordering_fetch(size_t idx, std::expected<Response, exception_t> response) noexcept{
                
                //this is ..., the designated memory area of the producer is clear
                //dg::vector<> size is const, dg::vector<> ptr is const, the logic is sound, but it is not 
                //producer is responsible to defer the memory ordering and do 1 big std::atomic_thread_fence(std::memory_order_release) after the consume

                this->resp_vec[idx] = std::move(response);
            }

            void maybedefer_memory_ordering_fetch_close(size_t idx, void * dirty_memory) noexcept{

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

            void maybedefer_memory_ordering_fetch(std::expected<Response, exception_t> response) noexcept{

                this->base->maybedefer_memory_ordering_fetch(std::move(response));
            }

            void maybedefer_memory_ordering_fetch_close(void * dirty_memory) noexcept{

                this->base->maybedefer_memory_ordering_fetch_close(dirty_memory);
            }
    };

    class BatchRequestResponse: public virtual BatchResponseInterface{

        private:

            std::vector<BatchRequestResponseBaseDesignatedObserver> observer_arr; 
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

                auto pending_smp        = std::binary_semaphore(0u);
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

        private:

        
        public:

            void clock_in(std::pair<model::ticket_id_t, std::chrono::nanoseconds> * registering_arr, size_t sz, exception_t * exception_arr) noexcept{

            }

            void get_expired_ticket(model::ticket_id_t * ticket_arr, size_t& ticket_arr_sz, size_t ticket_arr_cap) noexcept{

            }

            void clear() noexcept{

            }

            auto max_consume_size() noexcept -> size_t{

            }
    };

    class DistributedTicketTimeoutManager: public virtual TicketTimeoutManagerInterface{

        private:

        public:

            void clock_in(std::pair<model::ticket_id_t, std::chrono::nanoseconds> * registering_arr, size_t sz, exception_t * exception_arr) noexcept{

            }

            void get_expired_ticket(model::ticket_id_t * ticket_arr, size_t& ticket_arr_sz, size_t ticket_arr_cap) noexcept{

            }

            void clear() noexcept{

            }

            auto max_consume_size() noexcept -> size_t{

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

                        observer_arr[i].value()->maybedefer_memory_ordering_fetch(std::move(base_response_arr[i].response));
                    }

                    std::atomic_thread_fence(std::memory_order_release);

                    for (size_t i = 0u; i < sz; ++i){
                        if (!observer_arr[i].has_value()){
                            continue;
                        }

                        observer_arr[i].value()->maybedefer_memory_ordering_fetch_close(static_cast<void *>(&base_response_arr[i].response));
                    }
                }
            };
    };

    class OutBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RequestContainerInterface> request_container;
            uint32_t channel;

        public:

            OutBoundWorker(std::shared_ptr<RequestContainerInterface> request_container,
                           uint32_t channel) noexcept: request_container(std::move(request_container)),
                                                       channel(channel){}

            bool run_one_epoch() noexcept{

                dg::vector<model::InternalRequest> request_vec = this->request_container->pop();

                for (auto& request: request_vec){
                    std::expected<dg::network_kernel_mailbox::Address, exception_t> addr = dg::network_uri_encoder::extract_mailbox_addr(request.request.requestee_uri);

                    if (!addr.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(addr.error()));
                        continue;
                    }

                    std::expected<dg::string, exception_t> bstream = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_serialize<dg::string, model::InternalRequest>)(request, model::INTERNAL_REQUEST_SERIALIZATION_SECRET);

                    if (!bstream.has_value()){
                        dg::network_log_stackdump::error_fast(dg::network_exception::verbose(bstream.error()));
                    }

                    auto feed_arg       = dg::network_kernel_mailbox::MailBoxArgument{};
                    feed_arg.to         = addr.value();
                    feed_arg.content    = std::move(bstream.value()); 

                    dg::network_producer_consumer::delvrsrv_deliver(feeder.get(), std::move(feed_arg));
                }

                return true;
            }
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

                        stolen_response_observer_arr[i].value()->maybedefer_memory_ordering_fetch(std::move(timeout_response_arr[i]));
                    }

                    std::atomic_thread_fence(std::memory_order_release);

                    for (size_t i = 0u; i < ticket_id_arr_sz; ++i){
                        if (!stolen_response_observer_arr[i].has_value()){
                            continue;
                        }

                        stolen_response_observer_arr[i].value()->maybedefer_memory_ordering_fetch_close(static_cast<void *>(&timeout_response_arr[i]));
                    }
                }
            };
    };

    class RestController: public virtual RestControllerInterface{

        private:

            std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::shared_ptr<TicketControllerInterface> ticket_controller;
            std::unique_ptr<RequestContainerInterface> request_container;
            std::unique_ptr<ExpiryFactoryInterface> expiry_factory;

        public:

            RestController(std::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                           std::unique_ptr<TicketControllerInterface> ticket_controller,
                           std::unique_ptr<RequestContainerInterface> request_container,
                           std::unique_ptr<ExpiryFactoryInterface> expiry_factory) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                                             ticket_controller(std::move(ticket_controller)),
                                                                                             request_container(std::move(request_container)),
                                                                                             expiry_factory(std::move(expiry_factory)){}

            auto request(model::Request rq) noexcept -> std::expected<std::unique_ptr<ResponseInterface>, exception_t>{

                auto timepoint = this->expiry_factory->get_expiry(rq.timeout);

                if (!timepoint.has_value()){
                    return std::unexpected(timepoint.error());
                }

                std::shared_ptr<RequestResponse> response                   = std::make_shared<RequestResponse>(timepoint.value()); //internalize allocations
                std::expected<model::ticket_id_t, exception_t> ticket_id    = this->ticket_controller->get_ticket(response);

                if (!ticket_id.has_value()){
                    return std::unexpected(ticket_id.error());
                }

                auto internal_request   = InternalRequest{std::move(rq), ticket_id.value()};
                exception_t err         = this->request_container->push(std::move(internal_request));

                if (dg::network_exception::is_failed(err)){
                    this->ticket_controller->close_ticket(ticket_id.value());
                    return std::unexpected(err);
                }

                return std::unique_ptr<ResponseInterface>(std::make_unique<RAIITicketResponse>(std::move(response), make_raii_ticket(ticket_id.value(), this->ticket_controller)));
            }

        private:

            class RAIITicketResponse: public virtual ResponseInterface{

                private:

                    std::shared_ptr<ResponseInterface> base;
                    std::shared_ptr<model::ticket_id_t> ticket;
                
                public:

                    RaiiTicketResponse(std::shared_ptr<ResponseInterface> base, 
                                       std::shared_ptr<model::ticket_id_t> ticket) noexcept: base(std::move(base)),
                                                                                             ticket(std::move(ticket)){}

                    auto response() noexcept -> std::expected<Response, exception_t>{

                        return this->base->response();
                    }
            };
    };

    //we are reducing the serialization overheads of ticket_center
    class DistributedRestController: public virtual RestControllerInterface{

        private:

            std::vector<std::unique_ptr<RestControllerInterface>> rest_controller_vec;

        public:

            DistributedRestController(std::vector<std::unique_ptr<RestControllerInterface>> rest_controller_vec) noexcept: rest_controller_vec(std::move(rest_controller_vec)){} 

            auto request(model::Request rq) noexcept -> std::expected<std::unique_ptr<ResponseInterface>, exception_t>{

                assert(stdx::is_pow2(this->rest_controller_vec.size()));

                size_t random_clue  = dg::network_randomizer::randomize_int<size_t>();
                size_t idx          = random_clue & (this->rest_controller_vec.size() - 1u);

                return this->rest_controller_vec[idx]->request(std::move(rq));          
            }
    };
}

#endif