#ifndef __DG_NETWORK_REST_H__
#define __DG_NETWORK_REST_H__ 

#include <stdint.h>
#include <stdlib.h>
#include "network_std_container.h"
#include <chrono>
#include "network_exception.h"

namespace dg::network_post_rest::model{

    using ticket_id_t = uint64_t;

    struct Request{
        dg::network_std_container::string uri;
        dg::network_std_container::string requestee;
        dg::network_std_container::string payload;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(uri, requestee, payload, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(uri, requestee, payload, timeout);
        }
    };

    struct Response{
        dg::network_std_container::string response;
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

namespace dg::network_post_rest::server{

    struct RequestHandlerInterface{
        virtual ~RequestHandlerInterface() noexcept = default;
        virtual auto handle(model::Request) noexcept -> model::Response = 0;
    };
} 

namespace dg::network_post_rest::client{

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
        virtual auto register_clock(size_t id, std::chrono::nanoseconds dur) noexcept -> exception_t = 0;
        virtual auto is_timeout(size_t id) noexcept -> std::expected<bool, exception_t> = 0;
        virtual void deregister_clock(size_t id) noexcept = 0;
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

    auto request_many(RestControllerInterface& controller, dg::network_std_container::vector<model::Request> payload_vec) noexcept -> dg::network_std_container::vector<std::expected<model::Response, exception_t>>{
        
        auto ticket_vec = dg::network_std_container::vector<std::expected<model::ticket_id_t, exception_t>>{};
        auto rs_vec     = dg::network_std_container::vector<std::expected<model::Response, exception_t>>{};

        for (model::Request& payload: payload_vec){
            ticket_vec.push_back(controller.request(std::move(payload)));
        }

        auto synchronizable = [&]() noexcept{
            for (std::expected<model::ticket_id_t, exception_t>& ticket: ticket_vec){
                if (ticket.has_value()){
                    std::expected<bool, exception_t> status = controller.is_ready(ticket.value());
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

        for (std::expected<model::ticket_id_t, exception_t>& ticket: ticket_vec){
            if (ticket.has_value()){
                rs_vec.push_back(controller.response(ticket.value()));
                controller.close(ticket.value());
            } else{
                rs_vec.push_back(std::unexpected(ticket.error()));
            }
        }

        return rs_vec;
    }
}

namespace dg::network_post_rest::server_impl1{

    using namespace dg::network_post_rest::server; 

    class RequestResolverWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::unordered_map<std::string, std::unique_ptr<RequestHandlerInterface>> request_handler;
            const uint8_t resolve_channel;
            const uint8_t response_channel 

        public:

            RequestResolverWorker(std::unordered_map<std::string, std::unique_ptr<RequestHandlerInterface>> request_handler,
                                  uint8_t resolve_channel,
                                  uint8_t response_channel) noexcept: request_handler(std::move(request_handler)),
                                                                      resolve_channel(resolve_channel),
                                                                      response_channel(response_channel){}
            
            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::string> recv = dg::network_kernel_mailbox::recv(this->resolve_channel);

                if (!static_cast<bool>(recv)){
                    return false;
                }

                model::InternalRequest request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<model::InternalRequest>)(request, recv->data(), recv->size()); 

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(err));
                    return false;
                }

                std::expected<dg::network_kernel_mailbox::Address, exception_t> requestee_addr = dg::network_uri_encoder::extract_mailbox_addr(request.request.requestee);

                if (!requestee_addr.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(requestee_addr.error()));
                    return false;
                }

                model::InternalResponse response{};
                std::expected<dg::network_std_container::string, exception_t> resource_path = dg::network_uri_encoder::extract_local_path(request.request.uri);

                if (!resource_path.has_value()){
                    response = model::InternalResponse{model::Response{{}, resource_path.error()}, request.ticket_id};
                } else{
                    auto map_ptr = this->request_handler.find(resource_path.value());

                    if (map_ptr == this->request_handle.end()){
                        response = model::InternalResponse{model::Response{{}, dg::network_exception::REQUEST_RESOLUTOR_NOT_FOUND}, request.ticket_id};
                    } else{
                        response = model::InternalResponse{map_ptr->second->handle(request.request), request.ticket_id};
                    }
                }

                auto response_bstream = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(response), ' ');
                dg::network_compact_serializer::integrity_serialize_into(response_bstream.data(), response);
                dg::network_kernel_mailbox::send(requestee_addr.value(), std::move(response_bstream), this->response_channel);
                
                return true;
            }
    };
} 

namespace dg::network_post_rest::client_impl1{

    using namespace dg::network_post_rest::client; 

    class RequestContainer: public virtual RequestContainerInterface{

        private:

            dg::network_std_container::vector<model::InternalRequest> container;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            RequestContainer(dg::network_std_container::vector<model::InternalRequest> container,
                             std::unique_ptr<std::mutex> mtx) noexcept: container(std::move(container)),
                                                                        mtx(std::move(mtx)){}

            void push(model::InternalRequest request) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->container.push_back(std::move(request));
            }

            auto pop() noexcept -> std::optional<model::InternalRequest>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->container.empty()){
                    return std::nullopt;
                }

                auto rs = std::move(this->container.back());
                this->container.pop_back();

                return rs;
            }
    };

    class ExhaustionControllerRequestContainer: public virtual RequestContainerInterface{

        private:

            std::unique_ptr<RequestContainerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> executor;
            size_t cur_sz;
            size_t capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControllerRequestContainer(std::unique_ptr<RequestContainerInterface> base, 
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
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->cur_sz == capacity){
                    return false;
                }

                this->cur_sz += 1;
                this->base->push(std::move(request));
                
                return true;
            }

            auto internal_pop() noexcept -> std::optional<model::InternalRequest>{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                std::optional<model::InternalRequest> rs = this->base->pop();
                this->cur_sz -= static_cast<size_t>(static_cast<bool>(rs));

                return rs;
            }
    };

    class ClockController: public virtual ClockControllerInterface{

        private:

            dg::network_std_container::unordered_map<size_t, std::chrono::nanoseconds> expiry_map;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            ClockController(dg::network_std_container::unordered_map<size_t, std::chrono::nanoseconds> expiry_map,
                            std::unique_ptr<std::mutex> mtx) noexcept: expiry_map(std::move(expiry_map)),
                                                                       mtx(std::move(mtx)){}
            
            auto register_clock(size_t id, std::chrono::nanoseconds dur) noexcept -> exception_t{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr    = this->expiry_map.find(id);

                if (map_ptr != this->expiry_map.end()){
                    return dg::network_exception::ENTRY_EXISTED;
                }

                std::chrono::nanoseconds now    = dg::network_genult::utc_timestamp();
                std::chrono::nanoseconds then   = now + dur; 
                this->expiry_map.insert(std::make_pair(id, then));

                return dg::network_exception::SUCCESS;
            }

            auto is_timeout(size_t id) noexcept -> std::expected<bool, exception_t>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr    = this->expiry_map.find(id);

                if (map_ptr == this->expiry_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                std::chrono::nanoseconds now = dg::network_genult::utc_timestamp();
                bool is_expired = map_ptr->second < now;

                return is_expired;
            }

            void deregister_clock(size_t id) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
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

    class TicketController: public virtual TicketControllerInterface{

        private:

            dg::network_std_container::unordered_map<model::ticket_id_t, std::optional<model::Response>> response_map;
            size_t ticket_sz;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            TicketController(dg::network_std_container::unordered_map<model::ticket_id_t, std::optional<model::Response>> response_map,
                             size_t ticket_sz,
                             std::unique_ptr<std::mutex> mtx): response_map(std::move(response_map)),
                                                               ticket_sz(ticket_sz),
                                                               mtx(std::move(mtx)){}

            auto get_ticket() noexcept -> std::expected<model::ticket_id_t, exception_t>{

                auto lck_grd                    = dg::network_genult::lock_guard(*this->mtx);
                model::ticket_id_t nxt_ticket   = dg::network_genult::wrap_safe_integer_cast(this->ticket_sz);
                auto [map_ptr, status]          = this->response_map.emplace(std::make_pair(nxt_ticket, std::optional<model::Response>{}));
                dg::network_genult::assert(status);
                this->ticket_sz += 1;

                return nxt_ticket;
            }

            void close_ticket(model::ticket_id_t ticket_id) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);in
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

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);

                if (map_ptr == this->response_map.end()){
                    return dg::network_exception::BAD_ENTRY;
                }

                map_ptr->second = std::move(response);
                return dg::network_exception::SUCCESS;
            }

            auto is_resource_available(model::ticket_id_t ticket_id) noexcept -> std::expected<bool, exception_t>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);
                
                if (map_ptr == this->response_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                return static_cast<bool>(map_ptr->second);
            }

            auto get_ticket_resource(model::ticket_id_t ticket_id) noexcept -> std::expected<model::Response, exception_t>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);

                if (map_ptr == this->response_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                return map_ptr->second.value();
            }
    };

    class InboundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<TicketControllerInterface> ticket_controller;
            const uint8_t channel;
        
        public:

            InboundWorker(std::shared_ptr<TicketControllerInterface> ticket_controller,
                          uint8_t channel) noexcept: ticket_controller(std::move(ticket_controller)),
                                                     channel(channel){}

            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::string> recv = dg::network_kernel_mailbox::recv(this->channel);

                if (!static_cast<bool>(recv)){
                    return false;
                }

                model::InternalResponse response{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<model::InternalResponse>)(response, recv->data(), recv->size()); 

                if (dg::network_exception::is_failed(err)){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(err));
                    return false;
                }

                this->ticket_controller->set_ticket_resource(response.ticket_id, std::move(response.response));
                return true;
            }
    };

    class OutBoundWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            std::shared_ptr<RequestContainerInterface> request_container;
            const uint8_t channel;
        
        public:

            OutBoundWorker(std::shared_ptr<RequestContainerInterface> request_container,
                           uint8_t channel) noexcept: request_container(std::move(request_container)),
                                                      channel(channel){}

            bool run_one_epoch() noexcept{

                std::optional<model::InternalRequest> request = this->request_container->pop();

                if (!static_cast<bool>(request)){
                    return false;
                }

                std::expected<dg::network_kernel_mailbox::Address, exception_t> addr = dg::network_uri_encoder::extract_mailbox_addr(request->request.uri);

                if (!addr.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(addr.error()));
                    return false;
                }

                auto bstream = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request.value()), ' ');
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request.value());
                dg::network_kernel_mailbox::send(addr.value(), std::move(bstream), this->channel);

                return true;
            }
    };

    class RestController: public virtual RestControllerInterface{

        private:

            dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> workers;
            std::unique_ptr<TicketControllerInterface> ticket_controller;
            std::unique_ptr<ClockControllerInterface> clock_controller;
            std::unique_ptr<RequestContainerInterface> request_container;

        public:

            RestController(dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> workers,
                           std::unique_ptr<TicketControllerInterface> ticket_controller,
                           std::unique_ptr<ClockControllerInterface> clock_controller,
                           std::unique_ptr<RequestContainerInterface> request_container) noexcept: workers(std::move(workers)),
                                                                                                   ticket_controller(std::move(ticket_controller)),
                                                                                                   clock_controller(std::move(clock_controller)),
                                                                                                   request_container(std::move(request_container)){}
            
            auto request(model::Request request) noexcept -> std::expected<model::ticket_id_t, exception_t>{

                std::expected<model::ticket_id_t, exception_t> ticket_id = this->ticket_controller->get_ticket();

                if (!ticket_id.has_value()){
                    return std::unexpected(ticket_id.error());
                }

                exception_t err = this->clock_controller->register_clock(dg::network_genult::safe_integer_cast<size_t>(ticket_id), request.timeout);

                if (dg::network_exception::is_failed(err)){
                    this->ticket_controller->close_ticket(ticket_id.value());
                    return std::unexpected(err);
                }

                this->request_container->push(model::InternalRequest{std::move(request), ticket_id.value()});
                return ticket_id.value();
            }

            auto is_ready(model::ticket_id_t ticket_id) noexcept -> std::expected<bool, exception_t>{

                std::expected<bool, exception_t> timeout_status = this->clock_controller->is_timeout(dg::network_genult::safe_integer_cast<size_t>(ticket_id));

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

                return this->ticket_controller->get_ticket_resource(ticket_id);
            }

            void close(model::ticket_id_t ticket_id) noexcept{

                this->clock_controller->deregister_clock(ticket_id);
                this->ticket_controller->close_ticket(ticket_id);
            }
    };
    
}

namespace dg::network_post_rest_app{

    struct TokenGenerateRequest{
        dg::network_std_container::string auth_payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestee;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth_payload, uri, requestee, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(auth_payload, uri, requestee, timeout);
        }
    };

    struct TokenGenerateResponse{
        dg::network_std_container::string token;
        exception_t err_code; 

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, err_code);
        }
    };

    struct TokenRefreshRequest{
        dg::network_std_container::string token;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestee;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, uri, requestee, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, uri, requestee, timeout);
        }
    };

    struct TokenRefreshResponse{
        dg::network_std_container::string token;
        exception_t err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, err_code);
        }
    };

    struct TileInitRequest{
        dg::network_std_container::string token;
        dg::network_tile_init_poly::virtual_payload_t payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestee;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload, uri, requestee, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload, uri, requestee, timeout);
        }
    };

    struct TileInitResponse{
        bool is_success;
        exception_t err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(is_success, err_code);
        }

        template <class Reflector>\
        void dg_reflect(const Reflector& reflector){
            reflector(is_success, err_code);
        }
    };

    struct TileInjectRequest{
        dg::network_std_container::string token;
        dg::network_tile_inject_poly::virtual_payload_t payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestee;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload, uri, requestee, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload, uri, requestee, timeout);
        }
    };

    struct TileInjectResponse{
        bool is_sucess;
        exception_t err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(is_success, err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(is_success, err_code);
        }
    };

    struct TileSignalRequest{
        dg::network_std_container::string token;
        dg::network_tile_signal_poly::virtual_payload_t payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestee;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload, uri, requestee, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload, uri, requestee, timeout);
        }
    };

    struct TileSignalResponse{
        bool is_success;
        exception_t err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(is_success, err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(is_success, err_code);
        }
    };

    struct TileCondInjectRequest{
        dg::network_std_container::string token;
        dg::network_tile_condinject_poly::virtual_payload_t payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestee;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload, uri, requestee, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload, uri, requestee, timeout);
        }
    };

    struct TileCondInjectResponse{
        bool is_success;
        exception_t err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(is_success, err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(is_success, err_code);
        }
    }; 

    struct TileSeqMemcommitRequest{
        dg::network_std_container::string token;
        dg::network_tile_seqmemcommit_poly::virtual_payload_t payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestee;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload, uri, requestee, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload, uri, requestee, timeout);
        }
    };

    struct TileSeqMemcommitResponse{
        bool is_success;
        exception_t err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(is_success, err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(is_success, err_code);
        }
    };

    struct SysLogRetrieveRequest{
        dg::network_std_container::string token;
        dg::network_std_container::string kind;
        std::chrono::nanoseconds fr;
        std::chrono::nanoseconds to;
        uint32_t limit;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestee;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, kind, fr, to, limit, uri, requestee, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, kind, fr, to, limit, uri, requestee, timeout);
        }
    };
    
    struct SysLogRetrieveResponse{
        dg::network_std_container::vector<dg::network_postgres_db::model::SystemLogEntry> log_vec;
        exception_t err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(log_vec, err_code);
        }

        template <class Reflector>
        void dg_reflet(const Reflector& reflector){
            reflector(log_vec, err_code);
        }
    };

    struct UserLogRetrieveRequest{
        dg::network_std_container::string token;
        dg::network_std_container::string kind;
        std::chrono::nanoseconds fr;
        std::chrono::nanoseconds to;
        uint32_t limit;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestee;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, kind, fr, to, limit, uri, requestee, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, kind, fr, to, limit, uri, requestee, timeout);
        }
    };

    struct UserLogRetrieveResponse{
        dg::network_std_container::vector<dg::network_postgres_db::model::UserLogEntry> log_vec;
        exception_t err_code;
        
        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(log_vec, err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(log_vec, err_code);
        }
    };

    // class TokenGenerateResolutor{

    //     public:

    //         auto handle(Request request) noexcept -> Response{

    //             std::expected<TokenGenerateRequest, exception_t> tokgen_request = deserialize_token_generate_request(std::move(request.payload));
                
    //             if (!tokgen_request.has_value()){
    //                 return Response{std::nullopt, tokgen_request.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> token = dg::network_auth_usrpwd::token_generate_from_auth_payload(tokgen_request->value());

    //             if (!token.has_value()){
    //                 return Response{std::nullopt, token.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_token_generate_response(TokenGenerateResponse{std::move(token.value())});

    //             if (!response_payload.has_value()){
    //                 return Response{std::nullopt, response_payload.error()};
    //             }

    //             return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
    //         }
    // };

    // class TokenRefreshResolutor{

    //     public:

    //         auto handle(Request request) noexcept -> Response{

    //             std::expected<TokenRefreshRequest, exception_t> tokrefr_request = deserialize_token_refresh_request(std::move(request.payload));

    //             if (!tokrefr_request.has_value()){
    //                 return Response{std::nullopt, tokrefr_request.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> newtok = dg::network_auth_usrpwd::token_refresh(tokrefr_request->token);

    //             if (!newtok.has_value()){
    //                 return Response{std::nullopt, newtok.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_token_refresh_response(TokenRefreshResponse{std::move(newtok.value())});

    //             if (!response_payload.has_value()){
    //                 return Response{std::nullopt, response_payload.error()};
    //             }

    //             return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
    //         }
    // };

    // class TileInitResolutor{

    //     public:

    //         auto handle(Request request) noexcept -> Response{

    //             std::expected<TileInitRequest> tinit_request = deserialize_tile_init_request(std::move(request.payload));

    //             if (!tinit_request.has_value()){
    //                 return Response{std::nullopt, tinit_request.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(tinit_request->token);

    //             if (!usr_id.has_value()){
    //                 return Response{std::nullopt, usr_id.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> clearance = dg::network_user_base::user_get_clearance(usr_id.value());

    //             if (!clearance.has_value()){
    //                 return Response{std::nullopt, clearance.error()};
    //             }

    //             if (!dg::network_user_base::user_clearance_is_tileinit_qualified(clearance.value())){
    //                 return Response{std::nullopt, dg::network_exception::UNMET_CLEARANCE};
    //             }

    //             exception_t err = dg::network_tile_init_poly::load(std::move(tinit_request->payload)); //I'll fix the move and friends later - I prefer immutable datatypes if theres no perf constraints

    //             if (dg::network_exception::is_failed(err)){
    //                 return Response{std::nullopt, err};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_tile_init_response(TileInitResponse{true});

    //             if (!response_payload.has_value()){
    //                 return Response{std::nullopt, response_payload.error()};
    //             }

    //             return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
    //         }
    // };

    // class TileInjectResolutor{

    //     public:

    //         auto handle(Request request) noexcept -> Response{

    //             std::expected<TileInjectRequest, exception_t> tile_inject_request = deserialize_tile_inject_request(std::move(request.payload));

    //             if (!tile_inject_request.has_value()){
    //                 return Response{std::nullopt, tile_inject_request.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(tile_inject_request->token);

    //             if (!usr_id.has_value()){
    //                 return Response{std::nullopt, usr_id.error()}
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> clearance = dg::network_user_base::user_get_clearance(usr_id.value());

    //             if (!clearance.has_value()){
    //                 return Response{std::nullopt, clearance.error()};
    //             }

    //             if (!dg::network_user_base::user_clearance_is_tileinject_qualified(clearance.value())){
    //                 return Response{std::nullopt, dg::network_exception::UNMET_CLEARANCE};
    //             }

    //             exception_t err = dg::network_tile_inject_poly::load(std::move(tile_inject_request->payload));

    //             if (dg::network_exception::is_failed(err)){
    //                 return Response{std::nullopt, err};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_tile_inject_response(TileInjectResponse{true}); //each serialization protocol has a secret - this is to avoid malicious injection + avoid internal_corruption

    //             if (!response_payload.has_value()){
    //                 return Response{std::nullopt, response_payload.error()};
    //             }

    //             return Response{std::move(reponse_payload.value()), dg::network_exception::SUCCESS};
    //         }
    // };

    // class TileSignalResolutor{

    //     public:

    //         auto handle(Request request) noexcept -> Response{

    //             std::expected<TileSingalRequest, exception_t> tile_signal_request = deserialize_tile_signal_request(std::move(request.payload));

    //             if (!tile_signal_request.has_value()){
    //                 return Response{std::nullopt, tile_signal_request.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(tile_signal_request->token);

    //             if (!usr_id.has_value()){
    //                 return Response{std::nullopt, usr_id.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> clearance = dg::network_user_base::user_get_clearance(usr_id.value());

    //             if (!clearance.has_value()){
    //                 return Response{std::nullopt, clearance.error()};
    //             }

    //             if (!dg::network_user_base::user_clearance_is_tilesignal_qualified(clearance.value())){ //
    //                 return Response{std::nullopt, dg::network_exception::UNMET_CLEARANCE};
    //             }

    //             exception_t err = dg::network_tile_signal_poly::load(std::move(tile_signal_request->payload));

    //             if (dg::network_exception::is_failed(err)){
    //                 return Response{std::nullopt, err};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_tile_signal_response(TileSignalResponse{true});

    //             if (!response_payload.has_value()){
    //                 return Response{std::nullopt, response_payload.error()};
    //             }

    //             return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
    //         }
    // };

    // class TileCondInjectResolutor{

    //     public:

    //         auto handle(Request request) noexcept -> Response{

    //             std::expected<TileCondInjectRequest, exception_t> tile_inject_request = deserialize_tile_condinject_request(std::move(request.payload));

    //             if (!tile_inject_request.has_value()){
    //                 return Response{std::nullopt, tile_inject_request.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(tile_inject_request->token);

    //             if (!usr_id.has_value()){
    //                 return Response{std::nullopt, usr_id.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> clearance = dg::network_user_base::user_get_clearance(usr_id.value());

    //             if (!clearance.has_value()){
    //                 return Response{std::nullopt, clearance.error()};
    //             }

    //             if (!dg::network_user_base::user_clearance_is_condinject_qualified(clearance.value())){
    //                 return Response{std::nullopt, dg::network_exception::UNMET_CLEARANCE};
    //             }

    //             exception_t err = dg::network_tile_condinject_poly::load(std::move(tile_inject_request.payload));

    //             if (dg::network_exception::is_failed(err)){
    //                 return Response{std::nullopt, err};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_tile_condinject_response(TileCondInjectResponse{true});

    //             if (!response_payload.has_value()){
    //                 return Response{std::nullopt, response_payload.error()};
    //             }

    //             return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
    //         }
    // };

    // class SysLogRetrieveResolutor{

    //     public:

    //         auto handle(Request request) noexcept -> Response{

    //             std::expected<SysLogRetrieveRequest, exception_t> syslog_get_request = deserialize_getsyslog_request(std::move(request.payload));

    //             if (!syslog_get_request.has_value()){
    //                 return Response{std::nullopt, syslog_get_request.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(syslog_get_request->token);

    //             if (!usr_id.has_value()){
    //                 return Response{std::nullopt, usr_id.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> clearance = dg::network_user_base::user_get_clearance(usr_id.value());

    //             if (!clearance.has_value()){
    //                 return Response{std::nullopt, clearance.error()};
    //             }

    //             if (!dg::network_user_base::user_clearance_is_syslog_qualified(clearance.value())){
    //                 return Response{std::nullopt, dg::network_exception::UNMET_CLEARANCE};
    //             }

    //             std::expected<dg::network_std_container::vector<dg::network_postgres_db::model::SystemLogEntry>, exception_t> syslog_vec = dg::network_postgres_db::get_systemlog(syslog_get_request->kind, syslog_get_request->fr, syslog_get_request->to, syslog_get_request->limit);

    //             if (!syslog_vec.has_value()){
    //                 return Response{std::nullopt, syslog_vec.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_getsyslog_response(SysLogRetrieveResponse{std::move(syslog_vec.value())});
                
    //             if (!response_payload.has_value()){
    //                 return Response{std::nullopt, response_payload.error()};
    //             }

    //             return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
    //         }
    // };

    // class UserLogRetrieveResolutor{

    //     public:

    //         auto handle(Request request) noexcept -> Response{

    //             std::expected<UserLogRetrieveRequest, exception_t> usrlog_get_request = deserialize_getusrlog_request(std::move(request.payload));

    //             if (!usrlog_get_request.has_value()){
    //                 return Response{std::nullopt, usrlog_get_request.error()};
    //             } 

    //             std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(syslog_get_request->token);

    //             if (!usr_id.has_value()){
    //                 return Response{std::nullopt, usr_id.error()};
    //             }

    //             std::expected<dg::network_std_container::vector<dg::network_postgres_db::model::UserLogEntry>, exception_t> usrlog_vec = dg::network_postgres_db::get_userlog(usr_id.value(), usrlog_get_request->kind, usrlog_get_request->fr, usrlog_get_request->to, usrlog_get_request->limit);

    //             if (!usrlog_vec.has_value()){
    //                 return Response{std::nullopt, usrlog_vec.error()};
    //             }

    //             std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_getusrlog_response(UserLogRetrieveResponse{std::move(usrlog_vec.value())});

    //             if (!response_payload.has_value()){
    //                 return Response{std::nullopt, response_payload.error()};
    //             }

    //             return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
    //         }
    // };

    //consumer-producer
    //2 approaches
    
    //1 - chan. luong` hien tai. | doi. toi' khi nao co' du~ lieu, mo? luong - anh khong dong y (kernel linux - windows - friends)
    //2 - inf loop - chay. toi' khi nao co du lieu. | xu ly du lieu
    
    //2 duoc. chon. khi latency (lag) khong quan trong
    //lag = thoi gian du~ lieu. duoc. nhan tu kernel -> thoi` gian du~ lieu. duoc xu ly boi? ung dung.
    //20GB/s du~ lieu. -> ung dung
    //cach 1: 20GB/s - spike (mat data)
    //cach 2: 20GB/s - spike management (load balance of spike)
    //cach 2: khong block luong`, xoa' socket, mo? socket (socket la phan` de~ bi sai nhat cua kernel)
    //socket corrupted - (kernel panic, OOM, ...)
    //peer corrupted - peer not responding (khong tra loi) - doi khi la do socket corrupted
    //-> reset socket - unblock thread - (1) khong hoat dong - compromise program - (bad design)

    //REST request - kernel spawns 1 thread - thread kernel - RB tree, 1 << 20, 1 << 25 - saturated threads - lost compute - DDOS attack
    //nginx (different implementation) - same approach - 10-15 workers (recv REST requests in batch)

    auto request_token_get(dg::network_post_rest::client::RestControllerInterface& controller, TokenGenerateRequest request) noexcept -> std::expected<TokenGenerateResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response; 

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
        auto base_request   = BaseRequest{request.uri, request.requestee, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(controller, std::move(base_request)); 

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TokenGenerateResponse rs{};
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenGenerateResponse>)(rs, base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_token_refresh(dg::network_post_rest::client::RestControllerInterface& controller, TokenRefreshRequest request) noexcept -> std::expected<TokenRefreshResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
        auto base_request   = BaseRequest{request.uri, request.requestee, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(controller, std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        } 

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TokenRefreshResponse rs{};
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenRefreshResponse>)(rs, base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }
        
        return rs;
    }

    auto request_tile_init(dg::network_post_rest::client::RestControllerInterface& controller, TileInitRequest request) noexcept -> std::expected<TileInitResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
        auto base_request   = BaseRequest{request.uri, request.requestee, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(controller, std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TileInitResponse rs{};
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInitResponse>)(rs, base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_tile_inject(dg::network_post_rest::client::RestControllerInterface& controller, TileInjectRequest request) noexcept -> std::expected<TileInjectResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::newtork_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
        auto base_request   = BaseRequest{request.uri, request.requestee, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TileInjectResponse rs{};
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInjectResponse>)(rs, base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_tile_signal(dg::network_post_rest::client::RestControllerInterface& controller, TileSignalRequest request) noexcept -> std::expected<TileSignalResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
        auto base_request   = BaseRequest{request.uri, request.requestee, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TileSignalResponse rs{};
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSignalResponse>)(rs, base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_tile_condinject(dg::network_post_rest::client::RestControllerInterface& controller, TileCondInjectRequest request) noexcept -> std::expected<TileCondInjectResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
        auto base_request   = BaseRequest{request.uri, request.requestee, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TileCondInjectResponse rs{};
        exception_t err     = dg::network_exception::to_csytle_function(dg::network_compact_serializer::integrity_deserialize_into<TileCondInjectResponse>)(rs, base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_tile_seqmemcommit(dg::network_post_rest::client::RestControllerInterface& controller, TileSeqMemcommitRequest request) noexcept -> std::expected<TileSeqMemcommitResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
        auto base_request   = BaseRequest{request.uri, request.requestee, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TileSeqMemcommitResponse rs{};
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSeqMemcommitResponse>)(rs, base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_syslog_get(dg::network_post_rest::client::RestControllerInterface& controller, SysLogRetrieveRequest request) noexcept -> std::expected<SysLogRetrieveResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
        auto base_request   = BaseRequest{request.uri, request.requestee, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        SysLogRetrieveResponse rs{};
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<SysLogRetrieveResponse>)(rs, base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_usrlog_get(dg::network_post_rest::client::RestControllerInterface& controller, UserLogRetrieveRequest request) noexcept -> std::expected<UserLogRetrieveResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
        auto base_request   = BaseRequest{request.uri, request.requestee, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        UserLogRetrieveResponse rs{};
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<UserLogRetrieveResponse>)(rs, base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto requestmany_token_get(dg::network_post_rest::client::RestControllerInterface& controller, dg::network_std_container::vector<TokenGenerateRequest> req_vec) noexcept -> dg::network_std_container::vector<std::expected<TokenGenerateResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TokenGenerateResponse, exception_t>> rs{};

        for (TokenGenerateRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestee, std::move(bstream), req.timestamp};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));
        
        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TokenGenerateResponse appendee{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenGenerateResponse>)(appendee, base_response->response.data(), base_response->response.size());

                if (dg::network_exception::is_failed(err)){
                    rs.push_back(std::unexpected(err))
                } else{
                    rs.push_back(std::move(appendee));
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_token_refresh(dg::network_post_rest::client::RestControllerInterface& controller, dg::network_std_container::vector<TokenRefreshRequest> req_vec) noexcept -> dg::network_std_container::vector<std::expected<TokenRefreshResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TokenRefreshResponse, exception_t>> rs{};

        for (TokenRefreshRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestee, std::move(bstream), req.timestamp};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TokenRefreshResponse appendee{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenRefreshResponse>)(appendee, base_response->response.data(), base_response->response.size());
                
                if (dg::network_exception::is_failed(err)){
                    rs.push_back(std::unexpected(err));
                } else{
                    rs.push_back(std::move(appendee));
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_tile_init(dg::network_post_rest::client::RestControllerInterface& controller, dg::network_std_container::vector<TileInitRequest> req_vec) noexcept -> dg::network_std_container::vector<std::expected<TileInitResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TileInitResponse, exception_t>> rs{};

        for (TileInitRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' '); //optimizable
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req); //optimizable
            auto base_request   = BaseRequest{req.uri, req.requestee, std::move(bstream), req.timestamp};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TileInitResponse appendee{}; //optimizable
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInitResponse>)(appendee, base_response->response.data(), base_response->response.size()); //optimizable
                
                if (dg::network_exception::is_failed(err)){
                    rs.push_back(std::unexpected(err));
                } else{
                    rs.push_back(std::move(appendee));
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_tile_inject(dg::network_post_rest::client::RestControllerInterface& controller, dg::network_std_container::vector<TileInjectRequest> req_vec) noexcept -> dg::network_std_container::vector<std::expected<TileInjectResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TileInjectResponse, exception_t>> rs{};

        for (TileInjectRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestee, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TileInjectResponse appendee{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInjectResponse>)(appendee, base_response->response.data(), base_response->response.size());

                if (dg::network_exception::is_failed(err)){
                    rs.push_back(std::unexpected(err));
                } else{
                    rs.push_back(std::move(appendee));
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_tile_signal(dg::network_post_rest::client::RestControllerInterface& controller, dg::network_std_container::vector<TileSignalRequest> req_vec) noexcept -> dg::network_std_container::vector<std::expected<TileSignalResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TileSignalResponse, exception_t>> rs{};

        for (TileSignalRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestee, std::move(bstream), req.timeout};
            request_vec.push_back(std::move(base_request));
        }
        
        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TileSignalResponse appendee{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSignalResponse>)(appendee, base_response->response.data(), base_response->response.size());

                if (dg::network_exception::is_failed(err)){
                    rs.push_back(std::unexpected(err));
                } else{
                    rs.push_back(std::move(appendee));
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_tile_condinject(dg::network_post_rest::client::RestControllerInterface& controller, dg::network_std_container::vector<TileCondInjectRequest> req_vec) noexcept -> dg::network_std_container::vector<std::expected<TileCondInjectResponse, exception_t>>{
        
        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TileCondInjectResponse, exception_t>> rs{};

        for (TileCondInjectRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestee, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        } 

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TileCondInjectResponse appendee{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileCondInjectResponse>)(appendee, base_response->response.data(), base_response->response.size());

                if (dg::network_exception::is_failed(err)){
                    rs.push_back(std::unexpected(err));
                } else[
                    rs.push_back(std::move(appendee));
                ]
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_tile_seqmemcommit(dg::network_post_rest::client::RestControllerInterface& controller, dg::network_std_container::vector<TileSeqMemcommitRequest> req_vec) noexcept -> dg::network_std_container::vector<std::expected<TileSeqMemcommitResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TileSeqMemcommitResponse, exception_t>> rs{};

        for (TileSeqMemcommitRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestee, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TileSeqMemcommitResponse appendee{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSeqMemcommitResponse>)(appendee, base_response->response.data(), base_response->response.size());

                if (dg::network_exception::is_failed(err)){
                    rs.push_back(std::unexpected(err));
                } else[
                    rs.push_back(std::move(appendee));
                ]
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_syslog_get(dg::network_post_rest::client::RestControllerInterface& controller, dg::network_std_container::vector<SysLogRetrieveRequest> req_vec) noexcept -> dg::network_std_container::vector<std::expected<SysLogRetrieveResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<SysLogRetrieveResponse, exception_t>> rs{};

        for (SysLogRetrieveRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{request.uri, request.requestee, std::move(bstream), request.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                SysLogRetrieveResponse appendee{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<SysLogRetrieveResponse>)(appendee, base_response->response.data(), base_response->response.size());

                if (dg::network_exception::is_failed(err)){
                    rs.push_back(std::unexpected(err));
                } else{
                    rs.push_back(std::move(appendee));
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }   
        }

        return rs;
    }

    auto requestmany_usrlog_get(dg::network_post_rest::client::RestControllerInterface& controller, dg::network_std_container::vector<UserLogRetrieveRequest> req_vec) noexcept -> dg::network_std_container::vector<std::expected<UserLogRetrieveResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<UserLogRetrieveResponse, exception_t>> rs{};

        for (UserLogRetrieveRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestee, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec   = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                UserLogRetrieveResponse appendee{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<UserLogRetrieveResponse>)(appendee, base_response->response.data(), base_response->response.size());

                if (dg::network_exception::is_failed(err)){
                    rs.push_back(std::unexpected(err));
                } else{
                    rs.push_back(std::move(appendee));
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }
};

#endif