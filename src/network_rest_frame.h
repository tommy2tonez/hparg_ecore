#ifndef __DG_NETWORK_REST_FRAME_H__
#define __DG_NETWORK_REST_FRAME_H__ 

#include <stdint.h>
#include <stdlib.h>
#include "network_std_container.h"
#include <chrono>
#include "network_exception.h"
#include "stdx.h"
#include "network_kernel_mailbox.h"

namespace dg::network_post_rest_frame::model{

    using ticket_id_t = uint64_t;

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

namespace dg::network_post_rest_frame::server{

    struct RequestHandlerInterface{
        using Request   = model::Request;
        using Response  = model::Response; 

        virtual ~RequestHandlerInterface() noexcept = default;
        virtual auto handle(Request) noexcept -> Response = 0;
    };
} 

namespace dg::network_post_rest_frame::client{

    struct RequestContainerInterface{
        virtual ~RequestContainerInterface() noexcept = default;
        virtual void push(model::InternalRequest) noexcept = 0;
        virtual auto pop() noexcept -> std::optional<model::InternalRequest> = 0;
    };

    struct TicketControllerInterface{
        virtual ~TicketControllerInterface() noexcept = default;
        virtual auto get_ticket() noexcept -> std::expected<model::ticket_id_t, exception_t> = 0;
        virtual auto set_ticket_resource(model::ticket_id_t, model::Response) noexcept -> exception_t = 0;
        virtual auto is_resource_available(model::ticket_id_t) noexcept -> std::expected<bool, exception_t> = 0;
        virtual auto get_ticket_resource(model::ticket_id_t) noexcept -> std::expected<model::Response, exception_t> = 0;
        virtual void close_ticket(model::ticket_id_t) noexcept = 0; 
    };
    
    struct ClockControllerInterface{
        virtual ~ClockControllerInterface() noexcept = default;
        virtual auto register_clock(std::chrono::nanoseconds) noexcept -> std::expected<size_t, exception_t> = 0;
        virtual auto is_timeout(size_t) noexcept -> std::expected<bool, exception_t> = 0;
        virtual void deregister_clock(size_t) noexcept = 0;
    };

    struct RestControllerInterface{
        virtual ~RestControllerInterface() noexcept = default;
        virtual auto request(model::Request) noexcept -> std::expected<model::ticket_id_t, exception_t> = 0;
        virtual auto is_ready(model::ticket_id_t) noexcept -> std::expected<bool, exception_t> = 0;
        virtual auto response(model::ticket_id_t) noexcept -> std::expected<model::Response, exception_t> = 0;
        virtual void close(model::ticket_id_t) noexcept = 0;
    };

    auto request(RestControllerInterface& controller, model::Request payload) noexcept -> std::expected<model::Response, exception_t>{

        std::expected<model::ticket_id_t, exception_t> ticket_id = controller.request(std::move(payload));

        if (!ticket_id.has_value()){
            return std::unexpected(ticket_id.error());
        }

        std::expected<bool, exception_t> status{}; 

        auto synchronizable = [&]() noexcept{
            status = controller.is_ready(ticket_id.value());

            if (!status.has_value()){
                return true;
            }

            return status.value();
        };

        dg::network_asynchronous::wait(synchronizable);

        if (!status.has_value()){
            controller.close(ticket_id.value());
            return std::unexpected(status.error());
        }

        std::expected<model::Response, exception_t> resp = controller.response(ticket_id.value());
        controller.close(ticket_id.value());

        return resp;
    }

    auto request_many(RestControllerInterface& controller, dg::vector<model::Request> payload_vec) noexcept -> dg::vector<std::expected<model::Response, exception_t>>{
        
        auto ticket_vec = dg::vector<std::expected<model::ticket_id_t, exception_t>>{};
        auto rs_vec     = dg::vector<std::expected<model::Response, exception_t>>{};

        for (model::Request& payload: payload_vec){
            ticket_vec.push_back(controller.request(std::move(payload)));
        }

        auto synchronizable = [&]() noexcept{
            for (std::expected<model::ticket_id_t, exception_t>& ticket_id: ticket_vec){
                if (ticket_id.has_value()){
                    std::expected<bool, exception_t> status = controller.is_ready(ticket_id.value());
                    if (status.has_value()){
                        if (!status.value()){
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        dg::network_asynchronous::wait(synchronizable); //I was thinking if timeout is this component's responsibility then I think that it's not - because RestControllerInterface would be insufficient by itself - that's not a good design

        for (std::expected<model::ticket_id_t, exception_t>& ticket_id: ticket_vec){
            if (ticket_id.has_value()){
                rs_vec.push_back(controller.response(ticket_id.value()));
                controller.close(ticket_id.value());
            } else{
                rs_vec.push_back(std::unexpected(ticket_id.error()));
            }
        }

        return rs_vec;
    }
}

namespace dg::network_post_rest_frame::server_impl1{

    using namespace dg::network_post_rest_frame::server; 

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

namespace dg::network_post_rest_frame::client_impl1{

    using namespace dg::network_post_rest_frame::client; 

    class RequestContainer: public virtual RequestContainerInterface{

        private:

            dg::deque<model::InternalRequest> container;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            RequestContainer(dg::deque<model::InternalRequest> container,
                             std::unique_ptr<std::mutex> mtx) noexcept: container(std::move(container)),
                                                                        mtx(std::move(mtx)){}

            void push(model::InternalRequest request) noexcept{

                auto lck_grd = stdx::lock_guard(*this->mtx);
                this->container.push_back(std::move(request));
            }

            auto pop() noexcept -> std::optional<model::InternalRequest>{

                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (this->container.empty()){
                    return std::nullopt;
                }

                auto rs = std::move(this->container.front());
                this->container.pop_front();

                return rs;
            }
    };

    class ExhaustionControlledRequestContainer: public virtual RequestContainerInterface{

        private:

            std::unique_ptr<RequestContainerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            size_t cur_sz;
            size_t capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControlledRequestContainer(std::unique_ptr<RequestContainerInterface> base,
                                                 std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor,
                                                 size_t cur_sz,
                                                 size_t capacity,
                                                 std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                            executor(std::move(executor)),
                                                                                            cur_sz(cur_sz),
                                                                                            capacity(capacity),
                                                                                            mtx(std::move(mtx)){}

            void push(model::InternalRequest request) noexcept{

                dg::network_concurrency_infretry_x::ExecutableWrapper exe([&]() noexcept{return this->internal_push(request);});
                this->executor->exec(exe);
            }

            auto pop() noexcept -> std::optional<model::InternalRequest>{

                return this->internal_pop();
            }
        
        private:

            auto internal_push(model::InternalRequest& request) noexcept -> bool{
                
                auto lck_grd = stdx::lock_guard(*this->mtx);

                if (this->cur_sz == this->capacity){
                    return false;
                }

                this->base->push(std::move(request));
                this->cur_sz += 1;
                
                return true;
            }

            auto internal_pop() noexcept -> std::optional<model::InternalRequest>{
                
                auto lck_grd = stdx::lock_guard(*this->mtx);
                std::optional<model::InternalRequest> rs = this->base->pop();

                if (rs.has_value()){
                    this->cur_sz -= 1;
                }

                return rs;
            }
    };

    class ClockController: public virtual ClockControllerInterface{

        private:

            dg::unordered_map<size_t, std::chrono::nanoseconds> expiry_map;
            size_t ticket_sz;
            std::chrono::nanoseconds min_dur;
            std::chrono::nanoseconds max_dur;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            ClockController(dg::unordered_map<size_t, std::chrono::nanoseconds> expiry_map,
                            size_t ticket_sz,
                            std::chrono::nanoseconds min_dur,
                            std::chrono::nanoseconds max_dur,
                            std::unique_ptr<std::mutex> mtx) noexcept: expiry_map(std::move(expiry_map)),
                                                                       ticket_sz(ticket_sz),
                                                                       min_dur(min_dur),
                                                                       max_dur(max_dur),
                                                                       mtx(std::move(mtx)){}
            
            auto register_clock(std::chrono::nanoseconds dur) noexcept -> std::expected<size_t, exception_t>{
                
                auto lck_grd = stdx::lock_guard(*this->mtx);
                
                if (std::clamp(dur.count(), this->min_dur.count(), this->max_dur.count()) != dur.count()){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }
                
                size_t nxt_ticket_id    = this->ticket_sz;
                auto then               = static_cast<std::chrono::nanoseconds>(dg::network_genult::unix_timestamp()) + dur;
                auto [map_ptr, status]  = this->expiry_map.emplace(std::make_pair(nxt_ticket_id, then)); 
                
                if (!status){
                    return std::unexpected(dg::network_exception::BAD_INSERT);
                }

                this->ticket_sz += 1;
                return nxt_ticket_id;
            }

            auto is_timeout(size_t id) noexcept -> std::expected<bool, exception_t>{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->expiry_map.find(id);

                if (map_ptr == this->expiry_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                std::chrono::nanoseconds now = dg::network_genult::unix_timestamp();
                bool is_expired = map_ptr->second < now;

                return is_expired;
            }

            void deregister_clock(size_t id) noexcept{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->expiry_map.find(id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->expiry_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->expiry_map.erase(map_ptr);
            }
    };

    //this might need exhaustion control - program defined
    class TicketController: public virtual TicketControllerInterface{

        private:

            dg::unordered_map<model::ticket_id_t, std::optional<model::Response>> response_map;
            size_t ticket_sz;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            TicketController(dg::unordered_map<model::ticket_id_t, std::optional<model::Response>> response_map,
                             size_t ticket_sz,
                             std::unique_ptr<std::mutex> mtx): response_map(std::move(response_map)),
                                                               ticket_sz(ticket_sz),
                                                               mtx(std::move(mtx)){}

            auto get_ticket() noexcept -> std::expected<model::ticket_id_t, exception_t>{

                auto lck_grd                    = stdx::lock_guard(*this->mtx);
                model::ticket_id_t nxt_ticket   = dg::network_genult::wrap_safe_integer_cast(this->ticket_sz);
                auto [map_ptr, status]          = this->response_map.emplace(std::make_pair(nxt_ticket, std::optional<model::Response>{}));

                if (!status){
                    return std::unexpected(dg::network_exception::BAD_INSERT);
                }

                this->ticket_sz += 1;
                return nxt_ticket;
            }

            void close_ticket(model::ticket_id_t ticket_id) noexcept{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->response_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->response_map.erase(map_ptr);
            }

            auto set_ticket_resource(model::ticket_id_t ticket_id, model::Response response) noexcept -> exception_t{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);

                if (map_ptr == this->response_map.end()){
                    return dg::network_exception::BAD_ENTRY;
                }

                map_ptr->second = std::move(response);
                return dg::network_exception::SUCCESS;
            }

            auto is_resource_available(model::ticket_id_t ticket_id) noexcept -> std::expected<bool, exception_t>{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);
                
                if (map_ptr == this->response_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                return map_ptr->second.has_value();
            }

            auto get_ticket_resource(model::ticket_id_t ticket_id) noexcept -> std::expected<model::Response, exception_t>{

                auto lck_grd    = stdx::lock_guard(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);

                if (map_ptr == this->response_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }
                
                if (!map_ptr->second.has_value()){
                    return std::unexpected(dg::network_exception::RESOURCE_NOT_AVAILABLE);
                }

                model::Response response = std::move(map_ptr->second.value());
                map_ptr->second = std::nullopt; 

                return response;
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

                this->ticket_controller->set_ticket_resource(response.ticket_id, std::move(response.response));
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

                std::optional<model::InternalRequest> request = this->request_container->pop();

                if (!request.has_value()){
                    return false;
                }

                std::expected<dg::network_kernel_mailbox::Address, exception_t> addr = dg::network_uri_encoder::extract_mailbox_addr(request->request.uri);

                if (!addr.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(addr.error()));
                    return true;
                }

                auto bstream = dg::string(dg::network_compact_serializer::integrity_size(request.value()), ' ');
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request.value());
                dg::network_kernel_mailbox::send(addr.value(), std::move(bstream), this->channel);

                return true;
            }
    };

    class RestController: public virtual RestControllerInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::unique_ptr<TicketControllerInterface> ticket_controller;
            std::unique_ptr<ClockControllerInterface> clock_controller;
            std::unique_ptr<RequestContainerInterface> request_container;
            dg::unordered_map<model::ticket_id_t, size_t> ticket_clockid_map; 
            std::unique_ptr<std::mutex> map_mtx;

        public:

            RestController(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                           std::unique_ptr<TicketControllerInterface> ticket_controller,
                           std::unique_ptr<ClockControllerInterface> clock_controller,
                           std::unique_ptr<RequestContainerInterface> request_container,
                           dg::unordered_map<model::ticket_id_t, size_t> ticket_clockid_map,
                           std::unique_ptr<std::mutex> map_mtx) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                          ticket_controller(std::move(ticket_controller)),
                                                                          clock_controller(std::move(clock_controller)),
                                                                          request_container(std::move(request_container)),
                                                                          ticket_clockid_map(std::move(ticket_clockid_map)),
                                                                          map_mtx(std::move(map_mtx)){}
            
            auto request(model::Request request) noexcept -> std::expected<model::ticket_id_t, exception_t>{
                
                std::expected<model::ticket_id_t, exception_t> ticket_id = this->ticket_controller->get_ticket();

                if (!ticket_id.has_value()){
                    return std::unexpected(ticket_id.error());
                }

                std::expected<size_t, exception_t> clock_id = this->clock_controller->register_clock(request.timeout);

                if (!clock_id.has_value()){
                    this->ticket_controller->close_ticket(ticket_id.value());
                    return std::unexpected(clock_id.error());
                }

                {
                    auto lck_grd = stdx::lock_guard(*this->map_mtx);
                    auto [map_ptr, status] = this->ticket_clockid_map.emplace(std::make_pair(ticket_id.value(), clock_id.value()));

                    if (!status){
                        this->clock_controller->deregister_clock(clock_id.value());
                        this->ticket_controller->close_ticket(ticket_id.value());
                        return std::unexpected(dg::network_exception::BAD_INSERT);
                    }
                }

                this->request_container->push(model::InternalRequest{std::move(request), ticket_id.value()});
                return ticket_id.value();
            }

            auto is_ready(model::ticket_id_t ticket_id) noexcept -> std::expected<bool, exception_t>{
                
                size_t clock_id = {};

                {
                    auto lck_grd = stdx::lock_guard(*this->map_mtx);
                    auto map_ptr = this->ticket_clockid_map.find(ticket_id);

                    if (map_ptr == this->ticket_clockid_map.end()){
                        return std::unexpected(dg::network_exception::BAD_ENTRY);
                    }

                    clock_id = map_ptr->second;
                }

                std::expected<bool, exception_t> timeout_status = this->clock_controller->is_timeout(clock_id);

                if (!timeout_status.has_value()){
                    return std::unexpected(timeout_status.error());
                }

                if (timeout_status.value()){
                    return true;
                }

                std::expected<bool, exception_t> resource_status = this->ticket_controller->is_resource_available(ticket_id);

                if (!resource_status.has_value()){
                    return std::unexpected(resource_status.error());
                }

                return resource_status.value();
            }

            auto response(model::ticket_id_t ticket_id) noexcept -> std::expected<Response, exception_t>{

                size_t clock_id = {};

                {
                    auto lck_grd = stdx::lock_guard(*this->map_mtx);
                    auto map_ptr = this->ticket_clockid_map.find(ticket_id);

                    if (map_ptr == this->ticket_clockid_map.end()){
                        return std::unexpected(dg::network_exception::BAD_ENTRY);
                    }

                    clock_id = map_ptr->second;
                }

                std::expected<bool, exception_t> timeout_status = this->clock_controller->is_timeout(clock_id);

                if (!timeout_status.has_value()){
                    return std::unexpected(timeout_status.error());
                }

                if (timeout_status.value()){
                    return std::unexpected(dg::network_exception::REQUEST_TIMEOUT);
                }

                return this->ticket_controller->get_ticket_resource(ticket_id);
            }

            void close(model::ticket_id_t ticket_id) noexcept{

                size_t clock_id = {};

                {
                    auto lck_grd = stdx::lock_guard(*this->map_mtx);
                    auto map_ptr = this->ticket_clockid_map.find(ticket_id);

                    if constexpr(DEBUG_MODE_FLAG){
                        if (map_ptr == this->ticket_clockid_map.end()){
                            dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                            std::abort();
                        }
                    }

                    clock_id = map_ptr->second;
                    this->ticket_clockid_map.erase(map_ptr);
                }

                this->clock_controller->deregister_clock(clock_id);
                this->ticket_controller->close_ticket(ticket_id);
            }
    };
}

#endif