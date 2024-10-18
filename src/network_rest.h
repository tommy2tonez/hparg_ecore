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
                    auto lck_grd = dg::network_genult::lock_guard(*this->map_mtx);
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
                    auto lck_grd = dg::network_genult::lock_guard(*this->map_mtx);
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
                    auto lck_grd = dg::network_genult::lock_guard(*this->map_mtx);
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

    struct TokenGenerateBaseRequest{
        dg::network_std_container::string auth_payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth_payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(auth_payload);
        }
    };

    struct TokenGenerateRequest: TokenGenerateBaseRequest{
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TokenGenerateBaseRequest&>(*this), uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TokenGenerateBaseRequest&>(*this), uri, requestor, timeout);
        }
    };

    struct TokenGenerateBaseResponse{
        dg::network_std_container::string token;
        exception_t token_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, token_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, token_err_code);
        }
    };

    struct TokenGenerateResponse: TokenGenerateBaseResponse{
        exception_t server_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TokenGenerateBaseResponse&>(*this), server_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TokenGenerateBaseResponse>(*this), server_err_code);
        }
    };

    struct TokenRefreshBaseRequest{
        dg::network_std_container::string token;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token);
        }
     };

    struct TokenRefreshRequest: TokenRefreshBaseRequest{
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TokenRefreshBaseRequest&>(*this), uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TokenRefreshBaseRequest&>(*this), uri, requestor, timeout);
        }
    };

    struct TokenRefreshBaseResponse{
        dg::network_std_container::string token;
        exception_t token_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, token_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, token_err_code);
        }
    };

    struct TokenRefreshResponse: TokenRefreshBaseResponse{
        exception_t server_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TokenRefreshBaseResponse&>(*this), server_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TokenRefreshBaseResponse&>(*this), server_err_code);
        }
    };

    struct TileInitBaseRequest{
        dg::network_std_container::string token;
        dg::network_tile_init::virtual_payload_t payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload);
        }
    };

    struct TileInitRequest: TileInitBaseRequest{
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TileInitBaseRequest&>(*this), uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TileInitBaseRequest&>(*this), uri, requestor, timeout);
        }
    };

    struct TileInitBaseResponse{
        exception_t init_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(err_code);
        }

        template <class Reflector>\
        void dg_reflect(const Reflector& reflector){
            reflector(err_code);
        }
    };

    struct TileInitResponse: TileInitBaseResponse{
        exception_t server_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TileInitBaseResponse&>(*this), server_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TileInitBaseResponse&>(*this), server_err_code);
        }
    };

    struct TileInjectBaseRequest{
        dg::network_std_container::string token;
        dg::network_tile_inject::virtual_payload_t payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload);_
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload);
        }
    };

    struct TileInjectRequest: TileInjectBaseRequest{
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TileInjectBaseRequest&>(*this), uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TileInjectBaseRequest&>(*this), uri, requestor, timeout);
        }
    };

    struct TileInjectBaseResponse{
        exception_t inject_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(inject_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(inject_err_code);
        }
    };

    struct TileInjectResponse: TileInjectBaseResponse{
        exception_t server_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TileInjectBaseResponse&>(*this), server_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TileInjectBaseResponse&>(*this), server_err_code);
        }
    };

    struct TileSignalBaseRequest{
        dg::network_std_container::string token;
        dg::network_tile_signal::virtual_payload_t payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload);
        }
    };

    struct TileSignalRequest: TileSignalBaseRequest{
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TileSignalBaseRequest&>(*this), uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TileSignalBaseRequest&>(*this), uri, requestor, timeout);
        }
    };

    struct TileSignalBaseResponse{
        exception_t signal_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(signal_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(signal_err_code);
        }
    };

    struct TileSignalResponse: TileSignalBaseResponse{
        exception_t server_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TileSignalBaseResponse&>(*this), server_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TileSignalBaseResponse&>(*this), server_err_code);
        }
    };

    struct TileCondInjectBaseRequest{
        dg::network_std_container::string token;
        dg::network_tile_condinject::virtual_payload_t payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload);
        }
    };

    struct TileCondInjectRequest: TileCondInjectBaseRequest{
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TileCondInjectBaseRequest&>(*this), uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TileCondInjectBaseRequest&>(*this), uri, requestor, timeout);
        }
    };

    struct TileCondInjectBaseResponse{
        exception_t inject_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(inject_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(inject_err_code);
        }
    }; 

    struct TileCondInjectResponse: TileCondInjectBaseResponse{
        exception_t server_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TileCondInjectBaseResponse&>(*this), server_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TileCondInjectBaseResponse&>(*this), server_err_code);
        }
    };

    struct TileMemcommitBaseRequest{
        dg::network_std_container::string token;
        dg::network_tile_seqmemcommit::virtual_payload_t payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, payload);
        }
    };

    struct TileMemcommitRequest: TileMemcommitBaseRequest{
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TileMemcommitBaseRequest&>(*this), uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TileMemcommitBaseRequest&>(*this), uri, requestor, timeout);
        }
    };

    struct TileMemcommitBaseResponse{
        exception_t seqmemcommit_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(seqmemcommit_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(seqmemcommit_err_code);
        }
    };

    struct TileMemcommitResponse: TileMemcommitBaseResponse{
        exception_t server_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const TileMemcommitResponse&>(*this), server_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<TileMemcommitResponse&>(*this), server_err_code);
        }
    };

    struct SysLogRetrieveBaseRequest{
        dg::network_std_container::string token;
        dg::network_std_container::string kind;
        std::chrono::nanoseconds fr;
        std::chrono::nanoseconds to;
        uint32_t limit;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, kind, fr, to, limit);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, kind, fr, to, limit);
        }
    };

    struct SysLogRetrieveRequest: SysLogRetrieveBaseRequest{
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const SysLogRetrieveBaseRequest&>(*this), uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<SysLogRetrieveBaseRequest&>(*this), uri, requestor, timeout);
        }
    };
    
    struct SysLogRetrieveBaseResponse{
        dg::network_std_container::vector<dg::network_postgres_db::model::SystemLogEntry> log_vec;
        exception_t retrieve_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(log_vec, retrieve_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(log_vec, retrieve_err_code);
        }
    };

    struct SysLogRetrieveResponse: SysLogRetrieveBaseResponse{
        exception_t server_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const SysLogRetrieveBaseResponse&>(*this), server_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<SysLogRetrieveBaseResponse&>(*this), server_err_code);
        }
    };

    struct UserLogRetrieveBaseRequest{
        dg::network_std_container::string token;
        dg::network_std_container::string kind;
        std::chrono::nanoseconds fr;
        std::chrono::nanoseconds to;
        uint32_t limit;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, kind, fr, to);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, kind, fr, to);
        }
    };

    struct UserLogRetrieveRequest: UserLogRetrieveBaseRequest{
        dg::network_std_container::string uri;
        dg::network_std_container::string requestor;
        std::chrono::nanoseconds timeout;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const UserLogRetrieveBaseRequest&>(*this), uri, requestor, timeout);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<UserLogRetrieveBaseRequest&>(*this), uri, requestor, timeout);
        }
    };

    struct UserLogRetrieveBaseResponse{
        dg::network_std_container::vector<dg::network_postgres_db::model::UserLogEntry> log_vec;
        exception_t retrieve_err_code;
        
        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(log_vec, retrieve_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(log_vec, retrieve_err_code);
        }
    };

    struct UserLogRetrieveResponse: UserLogRetrieveBaseResponse{
        exception_t server_err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(static_cast<const UserLogRetrieveBaseResponse&>(*this), server_err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(static_cast<UserLogRetrieveBaseResponse&>(*this), server_err_code);
        }
    };
    
    class TokenGenerateResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                TokenGenerateBaseRequest tokgen_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenGenerateBaseRequest>)(tokgen_request, request.payload.data(), request.payload.size());
                
                if (dg::network_exception::is_failed(err)){
                    TokenGenerateBaseResponse tokgen_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokgen_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> token = dg::network_user::token_generate_from_auth_payload(tokgen_request.auth_payload);

                if (!token.has_value()){
                    TokenGenerateBaseResponse tokgen_response{{}, token.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokgen_response), dg::network_exception::SUCCESS};
                }

                TokenGenerateBaseResponse tokgen_response{std::move(token.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokgen_response), dg::network_exception::SUCCESS};
            }
    };

    class TokenRefreshResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TokenRefreshBaseRequest tokrefr_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenRefreshBaseRequest>)(tokrefr_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TokenRefreshBaseResponse tokrefr_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokrefr_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> newtok = dg::network_user::token_refresh(tokrefr_request.token);

                if (!newtok.has_value()){
                    TokenRefreshBaseResponse tokrefr_response{{}, newtok.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokrefr_response), dg::network_exception::SUCCESS}; //fine - 
                }

                TokenRefreshBaseResponse tokrefr_response{std::move(newtok.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tokrefr_response), dg::network_exception::SUCCESS};
            }
    };

    class TileInitResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileInitBaseRequest tileinit_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInitBaseRequest>)(tileinit_request, request.payload.data(), request.payload.size()); 

                if (dg::network_exception::is_failed(err)){
                    TileInitBaseResponse tileinit_response{err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(tileinit_request.token);
                
                if (!user_id.has_value()){
                    TileInitBaseResponse tileinit_response{user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileInitBaseResponse tileinit_response{user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileInitBaseResponse tileinit_response{ dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_init::load(std::move(request.payload));
                TileInitBaseResponse tileinit_response{load_err};
                
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinit_response), dg::network_exception::SUCCESS};
            }
    };

    class TileInjectResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                TileInjectBaseRequest tileinject_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInjectBaseRequest>)(tileinject_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TileInjectBaseResponse tileinject_response{err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(tileinject_request.token);

                if (!user_id.has_value()){
                    TileInjectBaseResponse tileinject_response{user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileInjectBaseResponse tileinject_response{user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileInjectBaseResponse tileinject_response{dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_inject::load(std::move(request.payload));
                TileInjectBaseResponse tileinject_response{load_err};
                
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tileinject_response), dg::network_exception::SUCCESS};
            }
    };

    class TileSignalResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileSignalBaseRequest tilesignal_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSignalBaseRequest>)(tilesignal_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TileSignalBaseResponse tilesignal_response{err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(tilesignal_request.token);

                if (!user_id.has_value()){
                    TileSignalBaseResponse tilesignal_response{user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileSignalBaseResponse tilesignal_response{user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileSignalBaseResponse tilesignal_response{dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_signal::load(std::move(tilesignal_request.payload));//this is not stateless - 
                TileSignalBaseResponse tilesignal_response{load_err};

                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(tilesignal_response), dg::network_exception::SUCCESS};
            }
    };

    class TileCondInjectResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileCondInjectBaseRequest condinject_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileCondInjectBaseRequest>)(condinject_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TileCondInjectBaseResponse condinject_response{err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(condinject_request.token);

                if (!user_id.has_value()){
                    TileCondInjectBaseResponse condinject_response{user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileCondInjectBaseResponse condinject_response{user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileCondInjectBaseResponse condinject_response{dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_condinject::load(std::move(condinject_request.payload));
                TileCondInjectBaseResponse condinject_response{load_err};

                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(condinject_response), dg::network_exception::SUCCESS};
            }
    };

    class TileMemcommitResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileMemcommitBaseRequest memcommit_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileMemcommitBaseRequest>)(memcommit_request, request.payload.data(), request.payload.size()); 

                if (dg::network_exception::is_failed(err)){
                    TileMemcommitBaseResponse memcommit_response{err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(memcommit_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(memcommit_request.token);

                if (!user_id.has_value()){
                    TileMemcommitBaseResponse memcommit_response{user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(memcommit_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileMemcommitBaseResponse memcommit_response{user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(memcommit_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileMemcommitBaseResponse memcommit_response{dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(memcommit_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_memcommit::load(std::move(memcommit_request.payload));
                TileMemcommitBaseResponse memcommit_response{load_err};

                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(memcommit_response), dg::network_exception::SUCCESS};

            }
    };

    class SysLogRetrieveResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                SysLogRetrieveBaseRequest syslog_get_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<SysLogRetrieveBaseRequest>)(syslog_get_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    SysLogRetrieveBaseResponse syslog_get_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(syslog_get_request.token);

                if (!user_id.has_value()){
                    SysLogRetrieveBaseResponse syslog_get_response{{}, user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    SysLogRetrieveBaseResponse syslog_get_response{{}, user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_SYS_READ){
                    SysLogRetrieveBaseResponse syslog_get_response{{}, dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::vector<dg::network_postgres_db::model::SystemLogEntry>, exception_t> syslog_vec = dg::network_postgres_db::get_systemlog(syslog_get_request.kind, syslog_get_request.fr, 
                                                                                                                                                                                  syslog_get_request.to, syslog_get_request.limit);

                if (!syslog_vec.has_value()){
                    SysLogRetrieveBaseResponse syslog_get_response{{}, syslog_vec.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                SysLogRetrieveBaseResponse syslog_get_response{std::move(syslog_vec.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(syslog_get_response), dg::network_exception::SUCCESS};
            }
    };

    class UserLogRetrieveResolutor: public virtual dg::network_post_rest::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                UserLogRetrieveBaseRequest usrlog_get_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<UserLogRetrieveBaseRequest>)(usrlog_get_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    UserLogRetrieveBaseResponse usrlog_get_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_id = dg::network_user::token_extract_userid(usrlog_get_request.token);

                if (!user_id.has_value()){
                    UserLogRetrieveBaseResponse usrlog_get_response{{}, user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    UserLogRetrieveBaseResponse usrlog_get_response{{}, user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::get_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_SELF_READ){
                    UserLogRetrieveBaseResponse usrlog_get_response{{}, dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::network_std_container::vector<dg::network_postgres_db::model::UserLogEntry>, exception_t> usrlog_vec = dg::network_postgres_db::get_userlog(user_id.value(), usrlog_get_request.kind, 
                                                                                                                                                                              usrlog_get_request.fr, usrlog_get_request.to, 
                                                                                                                                                                              usrlog_get_request.limit);

                if (!usrlog_vec.has_value()){
                    UserLogRetrieveBaseResponse usrlog_get_response{{}, usrlog_vec.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                UserLogRetrieveBaseResponse usrlog_get_response{dg::move(usrlog_vec.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::network_std_container::string>(usrlog_get_response), dg::network_exception::SUCCESS};
            }
    };

    auto request_token_get(dg::network_post_rest::client::RestControllerInterface& controller, const TokenGenerateRequest& request) noexcept -> std::expected<TokenGenerateResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response; 

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(static_cast<const TokenGenerateBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TokenGenerateBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(controller, std::move(base_request)); 

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        TokenGenerateResponse rs{};
        rs.server_err_code = base_request->err_code;

        if (dg::network_exception::is_failed(base_response->err_code)){
            return rs;
        }

        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenGenerateBaseResponse>)(static_cast<TokenGenerateBaseResponse&>(rs), 
                                                                                                                                                           base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_token_refresh(dg::network_post_rest::client::RestControllerInterface& controller, const TokenRefreshRequest& request) noexcept -> std::expected<TokenRefreshResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(static_cast<const TokenRefreshBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TokenRefreshBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(controller, std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        } 

        TokenRefreshResponse rs{};
        rs.server_err_code = base_response->err_code;

        if (dg::network_exception::is_failed(base_response->err_code)){
            return rs;
        }

        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenRefreshBaseResponse>)(static_cast<TokenRefreshBaseResponse&>(rs), 
                                                                                                                                                          base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err); //this is unexpected - denotes either misprotocol or miscommunication or corrupted (unlikely) 
        }
        
        return rs;
    }

    auto request_tile_init(dg::network_post_rest::client::RestControllerInterface& controller, const TileInitRequest& request) noexcept -> std::expected<TileInitResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(static_cast<const TileInitBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TileInitBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(controller, std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        TileInitResponse rs{};
        rs.server_err_code = base_response->err_code; 

        if (dg::network_exception::is_failed(base_response->err_code)){
            return rs;
        }

        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInitBaseResponse>)(static_cast<TileInitBaseResponse&>(rs), 
                                                                                                                                                      base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_tile_inject(dg::network_post_rest::client::RestControllerInterface& controller, const TileInjectRequest& request) noexcept -> std::expected<TileInjectResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::newtork_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(static_cast<const TileInjectBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TileInjectBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        TileInjectResponse rs{};
        rs.server_err_code = base_response->err_code; 

        if (dg::network_exception::is_failed(base_response->err_code)){
            return rs;
        }

        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInjectBaseResponse>)(static_cast<TileInjectBaseResponse&>(rs), 
                                                                                                                                                        base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_tile_signal(dg::network_post_rest::client::RestControllerInterface& controller, const TileSignalRequest& request) noexcept -> std::expected<TileSignalResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(static_cast<const TileSignalBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TileSignalBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        TileSignalResponse rs{};
        rs.server_err_code = base_response->err_code; 

        if (dg::network_exception::is_failed(base_response->err_code)){
            return rs;
        }

        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSignalBaseResponse>)(static_cast<TileSignalBaseResponse&>(rs), 
                                                                                                                                                        base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_tile_condinject(dg::network_post_rest::client::RestControllerInterface& controller, const TileCondInjectRequest& request) noexcept -> std::expected<TileCondInjectResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(static_cast<const TileCondInjectBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TileCondInjectBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        TileCondInjectResponse rs{};
        rs.server_err_code = base_response->err_code; 

        if (dg::network_exception::is_failed(base_response->err_code)){
            return rs;
        }

        exception_t err = dg::network_exception::to_csytle_function(dg::network_compact_serializer::integrity_deserialize_into<TileCondInjectBaseResponse>)(static_cast<TileCondInjectBaseResponse&>(rs), 
                                                                                                                                                            base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_tile_seqmemcommit(dg::network_post_rest::client::RestControllerInterface& controller, const TileMemcommitRequest& request) noexcept -> std::expected<TileMemcommitResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(static_cast<const TileMemcommitBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TileMemcommitBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }
        
        TileMemcommitResponse rs{};
        rs.server_err_code = base_response->err_code;

        if (dg::network_exception::is_failed(base_response->err_code)){
            return rs;
        }

        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileMemcommitBaseResponse>)(static_cast<TileMemcommitBaseResponse&>(rs), 
                                                                                                                                                              base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_syslog_get(dg::network_post_rest::client::RestControllerInterface& controller, const SysLogRetrieveRequest& request) noexcept -> std::expected<SysLogRetrieveResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(static_cast<const SysLogRetrieveBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const SysLogRetrieveBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        SysLogRetrieveResponse rs{};
        rs.server_err_code = base_response->err_code;

        if (dg::network_exception::is_failed(base_response->err_code)){
            return rs;
        }

        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<SysLogRetrieveBaseResponse>)(static_cast<SysLogRetrieveBaseResponse&>(rs), 
                                                                                                                                                            base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto request_usrlog_get(dg::network_post_rest::client::RestControllerInterface& controller, const UserLogRetrieveRequest& request) noexcept -> std::expected<UserLogRetrieveResponse, exception_t>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(static_cast<const UserLogRetrieveBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const UserLogRetrieveBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_post_rest::client::request(std::move(base_request));

        if (!base_response.has_value()){
            return std::unexpected(base_response.error());
        }

        UserLogRetrieveResponse rs{};
        rs.server_err_code = base_response->err_code;

        if (dg::network_exception::is_failed(base_response->err_code)){
            return rs;
        }

        exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<UserLogRetrieveBaseResponse>)(static_cast<UserLogRetrieveBaseResponse&>(rs), 
                                                                                                                                                             base_response->response.data(), base_response->response.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return rs;
    }

    auto requestmany_token_get(dg::network_post_rest::client::RestControllerInterface& controller, const dg::network_std_container::vector<TokenGenerateRequest>& req_vec) noexcept -> dg::network_std_container::vector<std::expected<TokenGenerateResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TokenGenerateResponse, exception_t>> rs{};

        for (const TokenGenerateBaseRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timestamp};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));
        
        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TokenGenerateResponse appendee{};
                
                if (dg::network_exception::is_failed(base_repsonse->err_code)){
                    appendee.server_err_code = base_response->err_code;
                    rs.push_back(std::move(appendee)); 
                } else{
                    exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenGenerateBaseResponse>)(static_cast<TokenGenerateBaseResponse&>(appendee), 
                                                                                                                                                                       base_response->response.data(), base_response->response.size());

                    if (dg::network_exception::is_failed(err)){
                        rs.push_back(std::unexpected(err))
                    } else{
                        rs.push_back(std::move(appendee));
                    }
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_token_refresh(dg::network_post_rest::client::RestControllerInterface& controller, const dg::network_std_container::vector<TokenRefreshRequest>& req_vec) noexcept -> dg::network_std_container::vector<std::expected<TokenRefreshResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TokenRefreshResponse, exception_t>> rs{};

        for (const TokenRefreshBaseRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timestamp};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TokenRefreshResponse appendee{};

                if (dg::network_exception::is_failed(base_response->err_code)){
                    appendee.server_err_code = base_response->err_code;
                    rs.push_back(std::move(appendee));
                } else{
                    exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenRefreshBaseResponse>)(static_cast<TokenRefreshBaseResponse&>(appendee), 
                                                                                                                                                                      base_response->response.data(), base_response->response.size());
                
                    if (dg::network_exception::is_failed(err)){
                        rs.push_back(std::unexpected(err));
                    } else{
                        rs.push_back(std::move(appendee));
                    }
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_tile_init(dg::network_post_rest::client::RestControllerInterface& controller, const dg::network_std_container::vector<TileInitRequest>& req_vec) noexcept -> dg::network_std_container::vector<std::expected<TileInitResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TileInitResponse, exception_t>> rs{};

        for (const TileInitBaseRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' '); //optimizable
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req); //optimizable
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timestamp};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TileInitResponse appendee{};

                if (dg::network_exception::is_failed(base_response->err_code)){
                    appendee.server_err_code = base_response->err_code;
                    rs.push_back(std::move(appendee));
                } else{
                    exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInitBaseResponse>)(static_cast<TileInitBaseResponse&>(appendee), 
                                                                                                                                                                  base_response->response.data(), base_response->response.size()); //optimizable
                
                    if (dg::network_exception::is_failed(err)){
                        rs.push_back(std::unexpected(err));
                    } else{
                        rs.push_back(std::move(appendee));
                    }
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_tile_inject(dg::network_post_rest::client::RestControllerInterface& controller, const dg::network_std_container::vector<TileInjectRequest>& req_vec) noexcept -> dg::network_std_container::vector<std::expected<TileInjectResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TileInjectResponse, exception_t>> rs{};

        for (const TileInjectBaseRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TileInjectResponse appendee{};

                if (dg::network_exception::is_failed(base_response->err_code)){
                    appendee.server_err_code = base_response->err_code;
                    rs.push_back(std::move(appendee));
                } else{
                    exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInjectBaseResponse>)(static_cast<TileInjectBaseResponse&>(appendee), 
                                                                                                                                                                    base_response->response.data(), base_response->response.size());

                    if (dg::network_exception::is_failed(err)){
                        rs.push_back(std::unexpected(err));
                    } else{
                        rs.push_back(std::move(appendee));
                    }
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_tile_signal(dg::network_post_rest::client::RestControllerInterface& controller, const dg::network_std_container::vector<TileSignalRequest>& req_vec) noexcept -> dg::network_std_container::vector<std::expected<TileSignalResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TileSignalResponse, exception_t>> rs{};

        for (const TileSignalBaseRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
            request_vec.push_back(std::move(base_request));
        }
        
        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TileSignalResponse appendee{};

                if (dg::network_exception::is_failed(base_response->err_code)){
                    appendee.server_err_code = base_response->err_code;
                    rs.push_back(std::move(appendee));
                } else{
                    exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSignalBaseResponse>)(static_cast<TileSignalBaseResponse&>(appendee), 
                                                                                                                                                                    base_response->response.data(), base_response->response.size());

                    if (dg::network_exception::is_failed(err)){
                        rs.push_back(std::unexpected(err));
                    } else{
                        rs.push_back(std::move(appendee));
                    }
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_tile_condinject(dg::network_post_rest::client::RestControllerInterface& controller, const dg::network_std_container::vector<TileCondInjectRequest>& req_vec) noexcept -> dg::network_std_container::vector<std::expected<TileCondInjectResponse, exception_t>>{
        
        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TileCondInjectResponse, exception_t>> rs{};

        for (const TileCondInjectBaseRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        } 

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TileCondInjectResponse appendee{};

                if (dg::network_exception::is_failed(base_response->err_code)){
                    appendee.server_err_code = base_response->err_code;
                    rs.push_back(std::move(appendee));
                } else{
                    exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileCondInjectBaseResponse>)(static_cast<TileCondInjectBaseResponse&>(appendee), 
                                                                                                                                                                        base_response->response.data(), base_response->response.size());

                    if (dg::network_exception::is_failed(err)){
                        rs.push_back(std::unexpected(err));
                    } else[
                        rs.push_back(std::move(appendee));
                    ]
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_tile_seqmemcommit(dg::network_post_rest::client::RestControllerInterface& controller, const dg::network_std_container::vector<TileMemcommitRequest>& req_vec) noexcept -> dg::network_std_container::vector<std::expected<TileMemcommitResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<TileMemcommitResponse, exception_t>> rs{};

        for (const TileMemcommitBaseRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                TileMemcommitResponse appendee{};

                if (dg::network_exception::is_failed(base_response->err_code)){
                    appendee.server_err_code = base_response->err_code;
                    rs.push_back(std::move(appendee));
                } else{
                    exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileMemcommitBaseResponse>)(static_cast<TileMemcommitBaseResponse&>(appendee), 
                                                                                                                                                                          base_response->response.data(), base_response->response.size());

                    if (dg::network_exception::is_failed(err)){
                        rs.push_back(std::unexpected(err));
                    } else[
                        rs.push_back(std::move(appendee));
                    ]
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }

    auto requestmany_syslog_get(dg::network_post_rest::client::RestControllerInterface& controller, const dg::network_std_container::vector<SysLogRetrieveRequest>& req_vec) noexcept -> dg::network_std_container::vector<std::expected<SysLogRetrieveResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<SysLogRetrieveResponse, exception_t>> rs{};

        for (const SysLogRetrieveBaseRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                SysLogRetrieveResponse appendee{};

                if (dg::network_exception::is_failed(base_response->err_code)){
                    appendee.server_err_code = base_response->err_code;
                    rs.push_back(std::move(appendee));
                } else{
                    exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<SysLogRetrieveBaseResponse>)(static_cast<SysLogRetrieveBaseResponse&>(appendee), 
                                                                                                                                                                        base_response->response.data(), base_response->response.size());

                    if (dg::network_exception::is_failed(err)){
                        rs.push_back(std::unexpected(err));
                    } else{
                        rs.push_back(std::move(appendee));
                    }
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }   
        }

        return rs;
    }

    auto requestmany_usrlog_get(dg::network_post_rest::client::RestControllerInterface& controller, const dg::network_std_container::vector<UserLogRetrieveRequest>& req_vec) noexcept -> dg::network_std_container::vector<std::expected<UserLogRetrieveResponse, exception_t>>{

        using BaseRequest   = dg::network_post_rest::model::Request;
        using BaseResponse  = dg::network_post_rest::model::Response;

        dg::network_std_container::vector<BaseRequest> base_request_vec{};
        dg::network_std_container::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::network_std_container::vector<std::expected<UserLogRetrieveResponse, exception_t>> rs{};

        for (const UserLogRetrieveBaseRequest& req: req_vec){
            auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec   = dg::network_post_rest::client::request_many(std::move(base_request_vec));

        for (std::expected<BaseResponse, exception_t>& base_response: base_response_vec){
            if (base_response.has_value()){
                UserLogRetrieveResponse appendee{};

                if (dg::network_exception::is_failed(base_response->err_code)){
                    appendee.server_err_code = base_response->err_code;
                    rs.push_back(std::move(appendee));
                } else{
                    exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<UserLogRetrieveBaseResponse>)(static_cast<UserLogRetrieveBaseResponse&>(appendee), 
                                                                                                                                                                         base_response->response.data(), base_response->response.size());

                    if (dg::network_exception::is_failed(err)){
                        rs.push_back(std::unexpected(err));
                    } else{
                        rs.push_back(std::move(appendee));
                    }                    
                }
            } else{
                rs.push_back(std::unexpected(base_response.error()));
            }
        }

        return rs;
    }
};

#endif