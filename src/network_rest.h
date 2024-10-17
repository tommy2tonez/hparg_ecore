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
        dg::network_std_container::string requestor;
        dg::network_std_container::string payload;
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
        using Request   = model::Request;
        using Response  = model::Response; 

        virtual ~RequestHandlerInterface() noexcept = default;
        virtual auto handle(Request) noexcept -> Response = 0;
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

    auto request_many(RestControllerInterface& controller, dg::network_std_container::vector<model::Request> payload_vec) noexcept -> dg::network_std_container::vector<std::expected<model::Response, exception_t>>{
        
        auto ticket_vec = dg::network_std_container::vector<std::expected<model::ticket_id_t, exception_t>>{};
        auto rs_vec     = dg::network_std_container::vector<std::expected<model::Response, exception_t>>{};

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

namespace dg::network_post_rest::server_impl1{

    using namespace dg::network_post_rest::server; 

    class RequestResolverWorker: public virtual dg::network_concurrency::WorkerInterface{

        private:

            dg::network_std_container::unordered_map<dg::network_std_container::string, std::unique_ptr<RequestHandlerInterface>> request_handler;
            const uint32_t resolve_channel;
            const uint32_t response_channel 

        public:

            RequestResolverWorker(dg::network_std_container::unordered_map<dg::network_std_container::string, std::unique_ptr<RequestHandlerInterface>> request_handler,
                                  uint32_t resolve_channel,
                                  uint32_t response_channel) noexcept: request_handler(std::move(request_handler)),
                                                                       resolve_channel(resolve_channel),
                                                                       response_channel(response_channel){}
            
            bool run_one_epoch() noexcept{

                std::optional<dg::network_std_container::string> recv = dg::network_kernel_mailbox::recv(this->resolve_channel);

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
                std::expected<dg::network_std_container::string, exception_t> resource_path = dg::network_uri_encoder::extract_local_path(request.request.uri);

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

                auto response_bstream = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(response), ' ');
                dg::network_compact_serializer::integrity_serialize_into(response_bstream.data(), response);
                dg::network_kernel_mailbox::send(requestor_addr.value(), std::move(response_bstream), this->response_channel);
                
                return true;
            }
    };
} 

namespace dg::network_post_rest::client_impl1{

    using namespace dg::network_post_rest::client; 

    class RequestContainer: public virtual RequestContainerInterface{

        private:

            dg::network_std_container::deque<model::InternalRequest> container;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            RequestContainer(dg::network_std_container::deque<model::InternalRequest> container,
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

                if (this->cur_sz == this->capacity){
                    return false;
                }

                this->base->push(std::move(request));
                this->cur_sz += 1;
                
                return true;
            }

            auto internal_pop() noexcept -> std::optional<model::InternalRequest>{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                std::optional<model::InternalRequest> rs = this->base->pop();

                if (rs.has_value()){
                    this->cur_sz -= 1;
                }

                return rs;
            }
    };

    class ClockController: public virtual ClockControllerInterface{

        private:

            dg::network_std_container::unordered_map<size_t, std::chrono::nanoseconds> expiry_map;
            size_t ticket_sz;
            std::chrono::nanoseconds min_dur;
            std::chrono::nanoseconds max_dur;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            ClockController(dg::network_std_container::unordered_map<size_t, std::chrono::nanoseconds> expiry_map,
                            size_t ticket_sz,
                            std::chrono::nanoseconds min_dur,
                            std::chrono::nanoseconds max_dur,
                            std::unique_ptr<std::mutex> mtx) noexcept: expiry_map(std::move(expiry_map)),
                                                                       ticket_sz(ticket_sz),
                                                                       min_dur(min_dur),
                                                                       max_dur(max_dur),
                                                                       mtx(std::move(mtx)){}
            
            auto register_clock(std::chrono::nanoseconds dur) noexcept -> std::expected<size_t, exception_t>{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                
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

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr    = this->expiry_map.find(id);

                if (map_ptr == this->expiry_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                std::chrono::nanoseconds now = dg::network_genult::unix_timestamp();
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

    //this might need exhaustion control - program defined
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

                if (!status){
                    return std::unexpected(dg::network_exception::BAD_INSERT);
                }

                this->ticket_sz += 1;
                return nxt_ticket;
            }

            void close_ticket(model::ticket_id_t ticket_id) noexcept{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
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

                return map_ptr->second.has_value();
            }

            auto get_ticket_resource(model::ticket_id_t ticket_id) noexcept -> std::expected<model::Response, exception_t>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
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

                std::optional<dg::network_std_container::string> recv = dg::network_kernel_mailbox::recv(this->channel);

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

                auto bstream = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request.value()), ' ');
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request.value());
                dg::network_kernel_mailbox::send(addr.value(), std::move(bstream), this->channel);

                return true;
            }
    };

    class RestController: public virtual RestControllerInterface{

        private:

            dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::unique_ptr<TicketControllerInterface> ticket_controller;
            std::unique_ptr<ClockControllerInterface> clock_controller;
            std::unique_ptr<RequestContainerInterface> request_container;
            dg::network_std_container::unordered_map<model::ticket_id_t, size_t> ticket_clockid_map; 
            std::unique_ptr<std::mutex> map_mtx;

        public:

            RestController(dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                           std::unique_ptr<TicketControllerInterface> ticket_controller,
                           std::unique_ptr<ClockControllerInterface> clock_controller,
                           std::unique_ptr<RequestContainerInterface> request_container,
                           dg::network_std_container::unordered_map<model::ticket_id_t, size_t> ticket_clockid_map,
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
                    auto lck_grd = dg::network_genult::lock_guard(*this->map_mtx);
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
                    auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
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

                return this->ticket_controller->get_ticket_resource(ticket_id);
            }

            void close(model::ticket_id_t ticket_id) noexcept{

                size_t clock_id = {};

                {
                    auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
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

namespace dg::network_post_rest_app{

    //this is fine - since the application is about thruput, lock congestion could be solved by using randomization approach or dedicated affine to avoid lock
    //latency is probably fixed - 100-200ms/request - yet it's hardly an issue because there are so many factors that affect latency 

    struct TokenGenerateRequest{
        dg::network_std_container::string auth_payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth_payload, uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(auth_payload, uri, requestor, timeout);
        }
    };

    struct TokenGenerateResponse{
        dg::network_std_container::string token;
        exception_t err_code; //this is to force generalization of implementation - such that response error code might be from a different dictionary 

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
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, uri, requestor, timeout);
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
        dg::network_tile_init::virtual_payload_t payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload, uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload, uri, requestor, timeout);
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
        dg::network_tile_inject::virtual_payload_t payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload, uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload, uri, requestor, timeout);
        }
    };

    struct TileInjectResponse{
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

    struct TileSignalRequest{
        dg::network_std_container::string token;
        dg::network_tile_signal::virtual_payload_t payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload, uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload, uri, requestor, timeout);
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
        dg::network_tile_condinject::virtual_payload_t payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload, uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload, uri, requestor, timeout);
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
        dg::network_tile_seqmemcommit::virtual_payload_t payload;
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload, uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload, uri, requestor, timeout);
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
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, kind, fr, to, limit, uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, kind, fr, to, limit, uri, requestor, timeout);
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
        void dg_reflect(const Reflector& reflector){
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
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, kind, fr, to, limit, uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, kind, fr, to, limit, uri, requestor, timeout);
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
    
    class TokenGenerateResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                TokenGenerateRequest tokgen_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenGenerateRequest>)(tokgen_request, request.payload.data(), request.payload.size());
                
                if (dg::network_exception::is_failed(err)){
                    TokenGenerateResponse tokgen_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokgen_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> token = dg::network_user::token_generate_from_auth_payload(tokgen_request.auth_payload);

                if (!token.has_value()){
                    TokenGenerateResponse tokgen_response{{}, token.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokgen_response), dg::network_exception::SUCCESS};
                }

                TokenGenerateResponse tokgen_response{std::move(token.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokgen_response), dg::network_exception::SUCCESS};
            }
    };

    class TokenRefreshResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TokenRefreshRequest tokrefr_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenRefreshRequest>)(tokrefr_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TokenRefreshResponse tokrefr_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokrefr_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> newtok = dg::network_user::token_refresh(tokrefr_request.token);

                if (!newtok.has_value()){
                    TokenRefreshResponse tokrefr_response{{}, newtok.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokrefr_response), dg::network_exception::SUCCESS};
                }

                TokenRefreshResponse tokrefr_response{std::move(newtok.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokrefr_response), dg::network_exception::SUCCESS};
            }
    };

    class TileInitResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileInitRequest tileinit_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInitRequest>)(tileinit_request, request.payload.data(), request.payload.size()); 

                if (dg::network_exception::is_failed(err)){
                    TileInitResponse tileinit_response{false, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(tileinit_request.token);
                
                if (!user_id.has_value()){
                    TileInitResponse tileinit_response{false, user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileInitResponse tileinit_response{false, user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileInitResponse tileinit_response{false, dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_init::load(std::move(request.payload)); //it's fine - it's a memory operation - so it's stateless

                if (dg::network_exception::is_failed(load_err)){
                    TileInitResponse tileinit_response{false, load_err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                TileInitResponse tileinit_response{true, dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
            }
    };

    class TileInjectResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                TileInjectRequest tileinject_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInjectRequest>)(tileinject_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TileInjectResponse tileinject_response{false, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(tileinject_request.token);

                if (!user_id.has_value()){
                    TileInjectResponse tileinject_response{false, user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileInjectResponse tileinject_response{false, user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileInjectResponse tileinject_response{false, dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_inject::load(std::move(request.payload));

                if (dg::network_exception::is_failed(load_err)){ //it's fine to not reduce the logic here
                    TileInjectResponse tileinject_response{false, load_err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                TileInjectResponse tileinject_response{true, dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
            }
    };

    class TileSignalResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileSignalRequest tilesignal_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSignalRequest>)(tilesignal_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TileSignalResponse tilesignal_response{false, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(tilesignal_request.token);

                if (!user_id.has_value()){
                    TileSignalResponse tilesignal_response{false, user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileSignalResponse tilesignal_response{false, user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileSignalResponse tilesignal_response{false, dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_signal::load(std::move(tilesignal_request.payload));//this is not stateless - 

                if (dg::network_exception::is_failed(load_err)){
                    TileSignalResponse tilesignal_response{false, load_err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                TileSignalResponse tilesignal_response{true, dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
            }
    };

    class TileCondInjectResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileCondInjectRequest condinject_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileCondInjectRequest>)(condinject_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TileCondInjectResponse condinject_response{false, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(condinject_request.token);

                if (!user_id.has_value()){
                    TileCondInjectResponse condinject_response{false, user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileCondInjectResponse condinject_response{false, user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileCondInjectResponse condinject_response{false, dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_condinject::load(std::move(condinject_request.payload));

                if (dg::network_exception::is_failed(load_err)){
                    TileCondInjectResponse condinject_response{false, load_err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                TileCondInjectResponse condinject_response{true, dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
            }
    };

    class SysLogRetrieveResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                SysLogRetrieveRequest syslog_get_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<SysLogRetrieveRequest>)(syslog_get_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    SysLogRetrieveResponse syslog_get_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(syslog_get_request.token);

                if (!user_id.has_value()){
                    SysLogRetrieveResponse syslog_get_response{{}, user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    SysLogRetrieveResponse syslog_get_response{{}, user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_SYS_READ){
                    SysLogRetrieveResponse syslog_get_response{{}, dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::vector<dg::network_postgres_db::model::SystemLogEntry>, exception_t> syslog_vec = dg::network_postgres_db::get_systemlog(syslog_get_request.kind, syslog_get_request.fr, 
                                                                                                                                                                                  syslog_get_request.to, syslog_get_request.limit);

                if (!syslog_vec.has_value()){
                    SysLogRetrieveResponse syslog_get_response{{}, syslog_vec.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                SysLogRetrieveResponse syslog_get_response{std::move(syslog_vec.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
            }
    };

    class UserLogRetrieveResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                UserLogRetrieveRequest usrlog_get_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<UserLogRetrieveRequest>)(usrlog_get_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    UserLogRetrieveResponse usrlog_get_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(usrlog_get_request.token);

                if (!user_id.has_value()){
                    UserLogRetrieveResponse usrlog_get_response{{}, user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    UserLogRetrieveResponse usrlog_get_response{{}, user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::get_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_SELF_READ){
                    UserLogRetrieveResponse usrlog_get_response{{}, dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::vector<dg::network_postgres_db::model::UserLogEntry>, exception_t> usrlog_vec = dg::network_postgres_db::get_userlog(user_id.value(), usrlog_get_request.kind, 
                                                                                                                                                                              usrlog_get_request.fr, usrlog_get_request.to, 
                                                                                                                                                                              usrlog_get_request.limit);

                if (!usrlog_vec.has_value()){
                    UserLogRetrieveResponse usrlog_get_response{{}, usrlog_vec.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                UserLogRetrieveResponse usrlog_get_response{dg::move(usrlog_vec.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
            }
    };

    auto request_token_get(dg::network_post_rest::client::RestControllerInterface& controller, TokenGenerateRequest request) noexcept -> std::expected<TokenGenerateResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response; 

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), request);
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(controller, std::move(base_request)); 

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TokenGenerateResponse rs{};
        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenGenerateResponse>)(rs, base_response->response.data(), base_response->response.size());

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
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(controller, std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        } 

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TokenRefreshResponse rs{};
        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenRefreshResponse>)(rs, base_response->response.data(), base_response->response.size());

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
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(controller, std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TileInitResponse rs{};
        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInitResponse>)(rs, base_response->response.data(), base_response->response.size());

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
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TileInjectResponse rs{};
        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInjectResponse>)(rs, base_response->response.data(), base_response->response.size());

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
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TileSignalResponse rs{};
        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSignalResponse>)(rs, base_response->response.data(), base_response->response.size());

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
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TileCondInjectResponse rs{};
        exception_t err = dg::network_exception::to_csytle_function(dg::network_compact_serializer::integrity_deserialize_into<TileCondInjectResponse>)(rs, base_response->response.data(), base_response->response.size());

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
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        TileSeqMemcommitResponse rs{};
        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSeqMemcommitResponse>)(rs, base_response->response.data(), base_response->response.size());

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
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        SysLogRetrieveResponse rs{};
        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<SysLogRetrieveResponse>)(rs, base_response->response.data(), base_response->response.size());

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
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        if (dg::network_exception::is_failed(base_response->err_code)){
            return std::unexpected(base_response->err_code);
        }

        UserLogRetrieveResponse rs{};
        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<UserLogRetrieveResponse>)(rs, base_response->response.data(), base_response->response.size());

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
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timestamp};
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
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timestamp};
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
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timestamp};
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
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
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
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
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
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
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
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
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
            auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
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
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
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