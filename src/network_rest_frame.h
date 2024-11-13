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

    struct RequestContainerInterface{
        virtual ~RequestContainerInterface() noexcept = default;
        virtual auto push(model::InternalRequest) noexcept -> exception_t = 0;
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
        virtual auto register_clock(std::chrono::nanoseconds) noexcept -> std::expected<model::clock_id_t, exception_t> = 0;
        virtual auto is_timeout(model::clock_id_t) noexcept -> std::expected<bool, exception_t> = 0;
        virtual void deregister_clock(model::clock_id_t) noexcept = 0;
    };

    struct RestControllerInterface{
        virtual ~RestControllerInterface() noexcept = default;
        virtual auto request(model::Request) noexcept -> std::expected<model::ticket_id_t, exception_t> = 0;
        virtual auto is_ready(model::ticket_id_t) noexcept -> std::expected<bool, exception_t> = 0;
        virtual auto response(model::ticket_id_t) noexcept -> std::expected<model::Response, exception_t> = 0;
        virtual void close(model::ticket_id_t) noexcept = 0;
    };

    struct TicketClockMapInterface{
        virtual ~TicketClockMapInterface() noexcept = default;
        virtual auto insert(model::ticket_id_t, model::clock_id_t) noexcept -> exception_t = 0;
        virtual auto map(model::ticket_id_t) noexcept -> std::expected<model::clock_id_t, exception_t> = 0; 
        virtual void erase(model::ticket_id_t) noexcept = 0;
    };

    auto async_request(RestControllerInterface& controller, model::Request payload) noexcept -> std::expected<model::Response, exception_t>{

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
        std::expected<model::Response, exception_t> resp = controller.response(ticket_id.value());
        controller.close(ticket_id.value());

        return resp;
    }

    auto async_request_many(RestControllerInterface& controller, dg::vector<model::Request> payload_vec) noexcept -> dg::vector<std::expected<model::Response, exception_t>>{
        
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

    //its better to do exhaustion control + load balancing at a different site - this is a fast thru operation
    class RequestContainer: public virtual RequestContainerInterface{

        private:

            dg::deque<model::InternalRequest> container;
            size_t capacity;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            RequestContainer(dg::deque<model::InternalRequest> container,
                             size_t capacity,
                             std::unique_ptr<std::mutex> mtx) noexcept: container(std::move(container)),
                                                                        capacity(capacity),
                                                                        mtx(std::move(mtx)){}

            auto push(model::InternalRequest request) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (this->container.size() == this->capacity){
                    return dg::network_exception::RESOURCE_EXHAUSTION;
                }

                this->container.push_back(std::move(request));
                return dg::network_exception::SUCCESS;
            }

            auto pop() noexcept -> std::optional<model::InternalRequest>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);

                if (this->container.empty()){
                    return std::nullopt;
                }

                auto rs = std::move(this->container.front());
                this->container.pop_front();

                return rs;
            }
    };

    class ClockController: public virtual ClockControllerInterface{

        private:

            dg::unordered_map<clock_id_t, std::chrono::nanoseconds> expiry_map;
            size_t ticket_sz;
            std::chrono::nanoseconds min_dur;
            std::chrono::nanoseconds max_dur;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            ClockController(dg::unordered_map<clock_id_t, std::chrono::nanoseconds> expiry_map,
                            size_t ticket_sz,
                            std::chrono::nanoseconds min_dur,
                            std::chrono::nanoseconds max_dur,
                            std::unique_ptr<std::mutex> mtx) noexcept: expiry_map(std::move(expiry_map)),
                                                                       ticket_sz(ticket_sz),
                                                                       min_dur(min_dur),
                                                                       max_dur(max_dur),
                                                                       mtx(std::move(mtx)){}
            
            auto register_clock(std::chrono::nanoseconds dur) noexcept -> std::expected<clock_id_t, exception_t>{
                
                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                
                if (std::clamp(dur.count(), this->min_dur.count(), this->max_dur.count()) != dur.count()){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }
                
                clock_id_t nxt_ticket_id    = this->ticket_sz;
                auto then                   = static_cast<std::chrono::nanoseconds>(dg::network_genult::unix_timestamp()) + dur;
                auto [map_ptr, status]      = this->expiry_map.emplace(std::make_pair(nxt_ticket_id, then)); 
                
                if (!status){
                    return std::unexpected(dg::network_exception::BAD_INSERT);
                }

                this->ticket_sz += 1;
                return nxt_ticket_id;
            }

            auto is_timeout(clock_id_t id) noexcept -> std::expected<bool, exception_t>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr    = this->expiry_map.find(id);

                if (map_ptr == this->expiry_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                std::chrono::nanoseconds now = dg::network_genult::unix_timestamp();
                bool is_expired = map_ptr->second < now;

                return is_expired;
            }

            void deregister_clock(clock_id_t id) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
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

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                model::ticket_id_t nxt_ticket   = dg::network_genult::wrap_safe_integer_cast(this->ticket_sz);
                auto [map_ptr, status]          = this->response_map.emplace(std::make_pair(nxt_ticket, std::optional<model::Response>{}));

                if (!status){
                    return std::unexpected(dg::network_exception::BAD_INSERT);
                }

                this->ticket_sz += 1;
                return nxt_ticket;
            }

            void close_ticket(model::ticket_id_t ticket_id) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
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

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);

                if (map_ptr == this->response_map.end()){
                    return dg::network_exception::BAD_ENTRY;
                }

                map_ptr->second = std::move(response);
                return dg::network_exception::SUCCESS;
            }

            auto is_resource_available(model::ticket_id_t ticket_id) noexcept -> std::expected<bool, exception_t>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);
                
                if (map_ptr == this->response_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                return map_ptr->second.has_value();
            }

            auto get_ticket_resource(model::ticket_id_t ticket_id) noexcept -> std::expected<model::Response, exception_t>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
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

    class TicketClockMap: public virtual TicketClockMapInterface{

        private:

            dg::unordered_map<model::ticket_id_t, model::clock_id_t> ticket_clock_map;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            TicketClockMap(dg::unordered_map<model::ticket_id_t, model::clock_id_t> ticket_clock_map,
                           std::unique_ptr<std::mutex> mtx) noexcept: ticket_clock_map(std::move(ticket_clock_map)),
                                                                      mtx(std::move(mtx)){}
            
            auto insert(model::ticket_id_t ticket_id, model::clock_id_t clock_id) noexcept -> exception_t{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto [map_ptr, status] = this->ticket_clock_map.insert(std::make_pair(ticket_id, clock_id));

                if (!status){
                    return dg::network_exception::BAD_INSERT;
                }

                return dg::network_exception::SUCCESS;
            }

            auto map(model::ticket_id_t ticket_id) noexcept -> std::expected<model::clock_id_t, exception_t>{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->ticket_clock_map.find(ticket_id);

                if (map_ptr == this->ticket_clock_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                return map_ptr->second;
            }

            void erase(model::ticket_id_t ticket_id) noexcept{

                stdx::xlock_guard<std::mutex> lck_grd(*this->mtx);
                auto map_ptr = this->ticket_clock_map.find(ticket_id);

                if constexpr(DEBUG_MODE_FLAG){
                    if (map_ptr == this->ticket_clock_map.end()){
                        dg::network_log_stackdump::critical(dg::network_exception::verbose(dg::network_exception::INTERNAL_CORRUPTION));
                        std::abort();
                    }
                }

                this->ticket_clock_map.erase(map_ptr);
            }
    };

    class RestController: public virtual RestControllerInterface{

        private:

            dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec;
            std::unique_ptr<TicketControllerInterface> ticket_controller;
            std::unique_ptr<ClockControllerInterface> clock_controller;
            std::unique_ptr<RequestContainerInterface> request_container;
            std::unique_ptr<TicketClockMapInterface> ticket_clock_map; 

        public:

            RestController(dg::vector<dg::network_concurrency::daemon_raii_handle_t> daemon_vec,
                           std::unique_ptr<TicketControllerInterface> ticket_controller,
                           std::unique_ptr<ClockControllerInterface> clock_controller,
                           std::unique_ptr<RequestContainerInterface> request_container,
                           std::unique_ptr<TicketClockMapInterface> ticket_clock_map) noexcept: daemon_vec(std::move(daemon_vec)),
                                                                                                ticket_controller(std::move(ticket_controller)),
                                                                                                clock_controller(std::move(clock_controller)),
                                                                                                request_container(std::move(request_container)),
                                                                                                ticket_clock_map(std::move(ticket_clock_map)){}
            
            auto request(model::Request request) noexcept -> std::expected<model::ticket_id_t, exception_t>{
                
                std::expected<model::ticket_id_t, exception_t> ticket_id = this->ticket_controller->get_ticket();

                if (!ticket_id.has_value()){
                    return std::unexpected(ticket_id.error());
                }

                std::expected<model::clock_id_t, exception_t> clock_id = this->clock_controller->register_clock(request.timeout);

                if (!clock_id.has_value()){
                    this->ticket_controller->close_ticket(ticket_id.value());
                    return std::unexpected(clock_id.error());
                }

                exception_t map_err = this->ticket_clock_map->insert(ticket_id.value(), clock_id.value()); 

                if (dg::network_exception::is_failed(map_err)){
                    this->clock_controller->deregister_clock(clock_id.value());
                    this->ticket_controller->close_ticket(ticket_id.value());
                    return std::unexpected(map_err);
                }

                exception_t ins_err = this->request_container->push(model::InternalRequest{std::move(request), ticket_id.value()}); 

                if (dg::network_exception::is_failed(ins_err)){
                    this->ticket_clock_map->erase(ticket_id.value());
                    this->clock_controller->deregister_clock(clock_id.value());
                    this->ticket_controller->close_ticket(ticket_id.value());
                    return std::unexpected(ins_err);
                }

                return ticket_id.value();
            }

            auto is_ready(model::ticket_id_t ticket_id) noexcept -> std::expected<bool, exception_t>{
                
                std::expected<model::clock_id_t, exception_t> clock_id = this->ticket_clock_map->map(ticket_id);

                if (!clock_id.has_value()){
                    return std::unexpected(clock_id.error());
                }

                std::expected<bool, exception_t> timeout_status = this->clock_controller->is_timeout(clock_id.value());

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

                std::expected<model::clock_id_t, exception_t> clock_id = this->ticket_clock_map->map(ticket_id);

                if (!clock_id.has_value()){
                    return std::unexpected(clock_id.error());
                }

                std::expected<bool, exception_t> timeout_status = this->clock_controller->is_timeout(clock_id.value());

                if (!timeout_status.has_value()){
                    return std::unexpected(timeout_status.error());
                }

                if (timeout_status.value()){
                    return std::unexpected(dg::network_exception::TIMEOUT);
                }

                return this->ticket_controller->get_ticket_resource(ticket_id);
            }

            void close(model::ticket_id_t ticket_id) noexcept{
                
                std::expected<model::clock_id_t, exception_t> clock_id = this->ticket_clock_map->map(ticket_id);

                if (!clock_id.has_value()){
                    dg::network_log_stackdump::critical(dg::network_exception::verbose(clock_id.error()));
                    std::abort();
                } 

                this->ticket_clock_map->erase(ticket_id);
                this->clock_controller->deregister_clock(clock_id.value());
                this->ticket_controller->close_ticket(ticket_id);
            }
    };
}

#endif