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

namespace dg::network_rest_frame::model{

    using ticket_id_t   = uint64_t;
    using clock_id_t    = uint64_t;

    struct Request{
        dg::string uri;
        dg::string requestor;
        dg::string payload;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(uri, requestor, payload, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(uri, requestor, payload, timeout);
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

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(request, ticket_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(request, ticket_id);
        }
    };

    struct InternalResponse{
        Response response;
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
        virtual auto handle(Request) noexcept -> Response = 0;
    };
} 

namespace dg::network_rest_frame::client{

    struct ResponseObserverInterface{
        virtual ~ResponseObserverInterface() noexcept = 0;
        virtual void update(Response) noexcept = 0;
    };

    struct ResponseInterface{
        virtual ~ResponseInterface() noexcept = default;
        virtual auto response() noexcept -> std::expected<Response, exception_t> = 0; 
    };

    struct RequestContainerInterface{
        virtual ~RequestContainerInterface() noexcept = default;
        virtual auto push(model::InternalRequest) noexcept -> exception_t = 0;
        virtual auto pop() noexcept -> model::InternalRequest = 0;
    };

    struct TicketControllerInterface{
        virtual ~TicketControllerInterface() noexcept = default;
        virtual auto get_ticket(std::shared_ptr<ResponseObserverInterface>) noexcept -> std::expected<model::ticket_id_t, exception_t> = 0;
        virtual auto set_response(model::ticket_id_t, model::Response) noexcept -> exception_t = 0;
        virtual void close_ticket(model::ticket_id_t) noexcept = 0; 
    };

    struct ExpiryFactoryInterface{
        virtual ~ExpiryFactoryInterface() noexcept = default;
        virtual auto get_expiry(std::chrono::nanoseconds) noexcept -> std::expected<std::chrono::timepoint<std::chrono::system_clock, std::chrono::nanoseconds>, exception_t> = 0;
    };

    struct RestControllerInterface{
        virtual ~RestControllerInterface() noexcept = default;
        virtual auto request(model::Request) noexcept -> std::expected<std::unique_ptr<ResponseInterface>, exception_t> = 0;
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

            dg::unordered_map<dg::string, std::unique_ptr<RequestHandlerInterface>> request_handler;
            const uint32_t resolve_channel;
            const uint32_t response_channel;

        public:

            RequestResolverWorker(dg::unordered_map<dg::string, std::unique_ptr<RequestHandlerInterface>> request_handler,
                                  uint32_t resolve_channel,
                                  uint32_t response_channel) noexcept: request_handler(std::move(request_handler)),
                                                                       resolve_channel(resolve_channel),
                                                                       response_channel(response_channel){}
            
            bool run_one_epoch() noexcept{

                std::optional<dg::string> recv = dg::network_kernel_mailbox::recv(this->resolve_channel);

                if (!recv.has_value()){
                    return false;
                }

                model::InternalRequest request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<model::InternalRequest>)(request, recv->data(), recv->size()); 

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(err));
                    return true;
                }

                std::expected<dg::network_kernel_mailbox::Address, exception_t> requestor_addr = dg::network_uri_encoder::extract_mailbox_addr(request.request.requestor);

                if (!requestor_addr.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(requestor_addr.error()));
                    return true;
                }

                model::InternalResponse response{};
                std::expected<dg::string, exception_t> resource_path = dg::network_uri_encoder::extract_local_path(request.request.uri);

                if (!resource_path.has_value()){
                    response = model::InternalResponse{model::Response{{}, resource_path.error()}, request.ticket_id};
                } else{
                    auto map_ptr = this->request_handler.find(resource_path.value());

                    if (map_ptr == this->request_handle.end()){
                        response = model::InternalResponse{model::Response{{}, dg::network_exception::BAD_REQUEST}, request.ticket_id};
                    } else{
                        response = model::InternalResponse{map_ptr->second->handle(std::move(request.request)), request.ticket_id};
                    }
                }

                auto response_bstream = dg::string(dg::network_compact_serializer::integrity_size(response), ' ');
                dg::network_compact_serializer::integrity_serialize_into(response_bstream.data(), response);
                dg::network_kernel_mailbox::send(requestor_addr.value(), std::move(response_bstream), this->response_channel);
                
                return true;
            }
    };
} 

namespace dg::network_rest_frame::client_impl1{

    using namespace dg::network_rest_frame::client; 

    class RequestResponse: public virtual ResponseObserverInterface,
                           public virtual ResponseInterface{

        private:

            std::timed_mutex mtx;
            Response response;
            std::chrono::timepoint<std::system_clock, std::chrono::nanoseconds> timeout;
            bool is_response_invoked;

        public:

            RequestResponse(std::chrono::timepoint<std::system_clock, std::chrono::nanoseconds> timeout) noexcept: mtx(),
                                                                                                                   response(),
                                                                                                                   timeout(timeout),
                                                                                                                   is_response_invoked(false){
                this->mtx.lock();
            }

            RequestResponse(const RequestResponse&) = delete;
            RequestResponse(RequestResponse&&) = delete;

            RequestResponse& operator =(const RequestResponse&) = delete;
            RequestResponse& operator =(RequestResponse&&) = delete;

            void update(Response response_arg) noexcept{

                this->response = std::move(response_arg);
                std::atomic_thread_fence(std::memory_order_release);
                this->mtx.unlock();
            }

            auto response() noexcept -> std::expected<Response, exception_t>{
                
                if (this->is_response_invoked){
                    return std::unexpected(dg::network_exception::RESOURCE_UNAVAILABLE);
                }

                bool rs = this->mtx.try_lock_until(this->timeout);
                this->is_response_invoked = true;

                if (!rs){
                    return std::unexpected(dg::network_exception::REQUEST_TIMEOUT);
                }

                std::atomic_thread_fence(std::memory_order_acquire);
                return std::move(this->response);
            }
    };

    class RequestContainer: public virtual RequestContainerInterface{

        private:

            dg::deque<model::InternalRequest> container;
            dg::vector<std::pair<std::mutex *, model::InternalRequest *>> waiting_queue; //this is good but we should not be abusing this - this is only for low-latency applications - too many subcriptible mutexes would slow down the system
            size_t capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            RequestContainer(dg::deque<model::InternalRequest> container,
                             dg::vector<std::pair<std::mutex *, model::InternalRequest *>> waiting_queue,
                             size_t capacity,
                             std::unique_ptr<std::mutex> mtx) noexcept: container(std::move(container)),
                                                                        waiting_queue(std::move(waiting_queue)),
                                                                        capacity(capacity),
                                                                        mtx(std::move(mtx)){}

            auto push(model::InternalRequest request) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (!this->waiting_queue.empty()){
                    auto [pending_mtx, fetching_addr] = std::move(this->waiting_queue.front());
                    this->waiting_queue.pop_front();
                    *fetching_addr = std::move(request);
                    std::atomic_thread_fence(std::memory_order_release);
                    pending_mtx->unlock();
                    return dg::network_exception::SUCCESS;
                }

                if (this->container.size() < this->container_capacity){
                    this->container.push_back(std::move(request));
                    return dg::network_exception::SUCCESS;
                }

                return dg::network_exception::RESOURCE_EXHAUSTION;
            }

            auto pop() noexcept -> model::InternalRequest{

                std::mutex pending_mtx{};
                model::InternalRequest internal_request = {};

                {
                    stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                    if (!this->container.empty()){
                        auto rs = std::move(this->container.front());
                        this->container.pop_front();
                        return rs;
                    }

                    pending_mtx.lock();
                    this->waiting_queue.push_back(std::make_pair(&pending_mtx, &internal_request));
                }

                stdx::xlock_guard<std::mutex> lck_grd(pending_mtx);
                return internal_request;
            }
    };

    class TicketController: public virtual TicketControllerInterface{

        private:

            dg::unordered_map<model::ticket_id_t, std::shared_ptr<ResponseObserverInterface>> observer_map;
            ticket_id_t incrementor;
            size_t ticket_cap;
            std::unique_ptr<std::mutex> mtx;

        public:

            TicketController(dg::unordered_map<model::ticket_id_t, std::shared_ptr<ResponseObserverInterface>> observer_map,
                            ticket_id_t incrementor,
                            size_t ticket_cap,
                            std::unique_ptr<std::mutex> mtx): observer_map(std::move(observer_map)),
                                                              incrementor(incrementor),
                                                              ticket_cap(ticket_cap),
                                                              mtx(std::move(mtx)){}

            auto get_ticket(std::shared_ptr<ResponseObserverInterface> response_observer) noexcept -> std::expected<ticket_id_t, exception_t>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (response_observer == nullptr){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

                if (this->observer_map.size() == this->ticket_cap){
                    return std::unexpected(dg::network_exception::RESOURCE_EXHAUSTION);
                }

                ticket_id_t next_ticket_id = this->incrementor;

                if constexpr(DEBUG_MODE_FLAG){
                    if (this->observer_map.contains(next_ticket_id)){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->observer_map.insert(std::make_pair(next_ticket_id, std::move(response_observer)));
                this->incrementor += 1u;

                return next_ticket_id;
            }

            auto set_response(model::ticket_id_t ticket_id, model::Response response) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->observer_map.find(ticket_id);

                if (map_ptr == this->observer_map.end()){
                    return dg::network_exception::RESOURCE_ABSENT;
                }

                map_ptr->second->update(std::move(response));
                return dg::network_exception::SUCCESS;
            }

            void close_ticket(model::ticket_id_t ticket_id) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->observer_map.find(ticket_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->observer_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->observer_map.erase(map_ptr);
            }
    };

    class InBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<TicketControllerInterface> ticket_controller;
            const uint32_t channel;
        
        public:

            InBoundWorker(std::shared_ptr<TicketControllerInterface> ticket_controller,
                          uint32_t channel) noexcept: ticket_controller(std::move(ticket_controller)),
                                                      channel(channel){}

            bool run_one_epoch() noexcept{

                std::optional<dg::string> recv = dg::network_kernel_mailbox::recv(this->channel);

                if (!recv.has_value()){
                    return false;
                }

                model::InternalResponse response{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<model::InternalResponse>)(response, recv->data(), recv->size());

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(err));
                    return true;
                }

                this->ticket_controller->set_response(response.ticket_id, std::move(response.response));
                return true;
            }
    };

    class OutBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RequestContainerInterface> request_container;
            const uint32_t channel;
        
        public:

            OutBoundWorker(std::shared_ptr<RequestContainerInterface> request_container,
                           uint32_t channel) noexcept: request_container(std::move(request_container)),
                                                       channel(channel){}

            bool run_one_epoch() noexcept{

                model::InternalRequest request = this->request_container->pop();
                std::expected<dg::network_kernel_mailbox::Address, exception_t> addr = dg::network_uri_encoder::extract_mailbox_addr(request.request.uri);

                if (!addr.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(addr.error()));
                    return true;
                }

                auto bstream = dg::string(dg::network_compact_serializer::integrity_size(request), ' ');
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
                dg::network_kernel_mailbox::send(addr.value(), std::move(bstream), this->channel);

                return true;
            }
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