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
        std::optional<dg::network_std_container::string> response;
        exception_t err_code;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err_code);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err_code);
        }
    };

    struct InternalRequest{
        Request request;
        ticket_id_t ticket_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(request, ticket_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(request, ticket_id);
        }
    };

    struct InternalResponse{
        Response response;
        ticket_id_t ticket_id;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, ticket_id);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
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
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(dg::network_exception::BAD_SERIALIZATION_FORMAT));
                    return false;
                }

                std::expected<dg::network_kernel_mailbox::Address, exception_t> addr = dg::network_uri_encoder::decode_to_mailbox_addr(request.requestee);

                if (!addr.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(dg::network_exception::BAD_SERIALIZATION_FORMAT)); //
                    return false;
                }

                std::expected<dg::network_std_container::string, exception_t> resource_path = dg::network_uri_encoder::extract_local_path(request.uri);

                if (!resource_path.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(dg::network_exception::BAD_SERIALIZATION_FORMAT)); //
                    return false;
                }

                auto map_ptr = this->request_handler.find(resource_path.value());

                if (map_ptr == this->request_handle.end()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(dg::network_exception::BAD_REQUEST));
                    return false;
                }

                Response response       = map_ptr->second->handle(request.request);
                auto response_bstream   = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(response));
                dg::network_compact_serializer::integrity_serialize_into(response_bstream.data(), response);
                dg::network_kernel_mailbox::send(addr.value(), std::move(response_bstream), this->response_channel);
                
                return true;
            }
    };
} 

namespace dg::network_post_rest::client_impl1{

    using namespace dg::network_post_rest::client; 

    class RequestContainer: public virtual RequestContainerInterface{

        private:

            dg::network_std_container::vector<InternalRequest> container;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            RequestContainer(dg::network_std_container::vector<InternalRequest> container,
                             std::unique_ptr<std::mutex> mtx) noexcept: container(std::move(container)),
                                                                        mtx(std::move(mtx)){}

            void push(InternalRequest request) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->container.push_back(std::move(request));
            }

            auto pop() noexcept -> std::optional<InternalRequest>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->container.empty()){
                    return std::nullopt;
                }

                auto rs = std::move(this->container.back());
                this->container.pop_back();

                return {std::in_place_t{}, std::move(rs)};
            }
    };

    class ExhaustionControllerRequestContainer: public virtual RequestContainerInterface{

        private:

            std::unique_ptr<RequestContainerInterface> base;
            std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> exec;
            size_t cur_sz;
            size_t capacity;
            std::unique_ptr<std::mutex> mtx;

        public:

            ExhaustionControllerRequestContainer(std::unique_ptr<RequestContainerInterface> base, 
                                                 std::shared_ptr<dg::network_concurrency_infretry_x::ExecutorInterface> exec,
                                                 size_t cur_sz,
                                                 size_t capacity,
                                                 std::unique_ptr<std::mutex> mtx) noexcept: base(std::move(base)),
                                                                                            exec(std::move(exec)),
                                                                                            cur_sz(cur_sz),
                                                                                            capacity(capacity),
                                                                                            mtx(std::move(mtx)){}

            void push(InternalRequest request) noexcept{

                auto lambda = [&]() noexcept{
                    return this->internal_push(request);
                };

                auto exe = dg::network_concurrency_infretry_x::ExecutableWrapper<decltype(lambda)>(std::move(lambda));
                this->exec->exec(exe);
            }

            auto pop() noexcept -> std::optional<InternalRequest>{

                return this->internal_pop();
            }
        
        private:

            auto internal_push(InternalRequest& request) noexcept -> bool{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);

                if (this->cur_sz == capacity){
                    return false;
                }

                this->cur_sz += 1;
                this->base->push(std::move(request));
            }

            auto internal_pop() noexcept -> std::optional<InternalRequest>{
                
                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                std::optional<InternalRequest> rs = this->base->pop();

                if (rs){
                    this->cur_sz -= 1;
                }

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
                return map_ptr->second < now;
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

            dg::network_std_container::unordered_map<ticket_id_t, std::optional<Response>> response_map;
            size_t ticket_sz;
            std::unique_ptr<std::mutex> mtx;
        
        public:

            TicketController(dg::network_std_container::unordered_map<ticket_id_t, std::optional<Response>> response_map,
                             size_t ticket_sz,
                             std::unique_ptr<std::mutex> mtx): response_map(std::move(response_map)),
                                                               ticket_sz(ticket_sz),
                                                               mtx(std::move(mtx)){}

            auto get_ticket() noexcept -> std::expected<ticket_id_t, exception_t>{

                auto lck_grd            = dg::network_genult::lock_guard(*this->mtx);
                ticket_id_t nxt_ticket  = static_cast<ticket_id_t>(this->ticket_sz); // bad;
                auto [map_ptr, status]  = this->response_map.emplace(std::make_pair(nxt_ticket, std::optional<Response>{}));
                dg::network_genult::assert(status);
                this->ticket_sz += 1;

                return nxt_ticket;
            }

            void close_ticket(ticket_id_t ticket_id) noexcept{

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

            auto set_ticket_resource(ticket_id_t ticket_id, Response response) noexcept -> exception_t{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);

                if (map_ptr == this->response_map.end()){
                    return dg::network_exception::BAD_ENTRY;
                }

                map_ptr->second = std::move(response);
                return dg::network_exception::SUCCESS;
            }

            auto is_resource_available(ticket_id_t ticket_id) noexcept -> std::expected<bool, exception_t>{

                auto lck_grd    = dg::network_genult::lock_guard(*this->mtx);
                auto map_ptr    = this->response_map.find(ticket_id);
                
                if (map_ptr == this->response_map.end()){
                    return std::unexpected(dg::network_exception::BAD_ENTRY);
                }

                return static_cast<bool>(map_ptr->second);
            }

            auto get_ticket_resource(ticket_id_t ticket_id) noexcept -> std::expected<Response, exception_t>{

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

                std::expected<dg::network_kernel_mailbox::Address, exception_t> addr = dg::network_uri_encoder::decode_to_mailbox_addr(request->request.uri);

                if (!addr.has_value()){
                    dg::network_log_stackdump::error_fast(dg::network_exception::verbose(addr.error()));
                    return false;
                }

                auto bstream = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(request.value()));
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
            std::unique_ptr<std::mutex> mtx;

        public:

            RestController(dg::network_std_container::vector<dg::network_concurrency::daemon_raii_handle_t> workers,
                           std::unique_ptr<TicketControllerInterface> ticket_controller,
                           std::unique_ptr<ClockControllerInterface> clock_controller,
                           std::unique_ptr<RequestContainerInterface> request_container,
                           std::unique_ptr<std::mutex> mtx) noexcept: workers(std::move(workers)),
                                                                      ticket_controller(std::move(ticket_controller)),
                                                                      clock_controller(std::move(clock_controller)),
                                                                      request_container(std::move(request_container)),
                                                                      mtx(std::move(mtx)){}
            
            auto request(model::Request request) noexcept -> std::expected<model::ticket_id_t, exception_t>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                std::expected<model::ticket_id_t, exception_t> ticket_id = this->ticket_controller->get_ticket();

                if (!ticket_id.has_value()){
                    return std::unexpected(ticket_id.error());
                }

                exception_t err = this->clock_controller->register_clock(static_cast<size_t>(ticket_id), request.timeout); //bad

                if (dg::network_exception::is_failed(err)){
                    this->ticket_controller->close_ticket(ticket_id.value());
                    return std::unexpected(err);
                }

                this->request_container->push(model::InternalRequest{std::move(request), ticket_id.value()});
                return ticket_id.value();
            }

            auto is_ready(model::ticket_id_t ticket_id) noexcept -> std::expected<bool, exception_t>{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                std::expected<bool, exception_t> timeout_status = this->clock_controller->is_timeout(static_cast<size_t>(ticket_id)); //bad

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

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                return this->ticket_controller->get_ticket_resource(ticket_id);
            }

            void close(model::ticket_id_t ticket_id) noexcept{

                auto lck_grd = dg::network_genult::lock_guard(*this->mtx);
                this->clock_controller->deregister_clock(ticket_id);
                this->ticket_controller->close_ticket(ticket_id);
            }
    };

    auto request(RestControllerInterface& controller, Request payload) noexcept -> std::expected<Response, exception_t>{

        std::expected<ticket_id_t, exception_t> ticket_id = controller.request(payload);

        if (!ticket_id.has_value()){
            return std::unexpected(ticket_id.error());
        }

        auto synchronizable = [&]() noexcept{
            return controller.is_ready(ticket_id.value());
        };

        dg::network_asynchronous::wait(synchronizable);
        std::expected<Response, exception_t> resp = controller.response(ticket_id.value());  

        if (!resp.has_value()){
            return std::unexpected(resp.error());    
        }

        controller.close(ticket_id.value());
        return resp;
    }

    auto request_many(RestControllerInterface& controller, dg::network_std_container::vector<Request> payload_vec) noexcept -> dg::network_std_container::vector<Response>{
        
        // dg::network_std_container::vector<Response> rs{};
        // // dg::network_std_container::vector<std::shared_ptr<ticket_id_t>> ticket_vec{};

        // for (Request& payload: payload_vec){
        //     std::expected<std::shared_ptr<ticket_id_t>, exception_t> ticket = internal_get_request_ticket(std::move(payload));

        //     if (!ticket.has_value()){
        //         ticket_vec.push_back(nullptr);
        //         rs.push_back(Response{{}, ticket.error()});
        //     } else{
        //         ticket_vec.push_back(std::move(ticket.value()));
        //         rs.push_back(Response({}, dg::network_exception::WAITING_RESPONSE));
        //     }
        // }

        // auto synchronizable = [&]() noexcept{
        //     for (std::shared_ptr<ticket_id_t>& ticket: ticket_vec){
        //         if (ticket){
        //             if (!rest_controller->is_ready(*ticket)){
        //                 return false;
        //             }
        //         }
        //     }
        //     return true;
        // };

        // dg::network_asynchronous::wait(synchronizable);

        // for (size_t i = 0u; i < ticket_vec.size(); ++i){
        //     if (ticket_vec[i]){
        //         rs[i] = rest_controller->response(*ticket_vec[i]);
        //     }
        // }

        // return rs;
    }
}

namespace dg::network_post_rest_resolutor{

    using Request           = dg::network_post_rest_frame::Request;
    using Response          = dg::network_post_rest_frame::Response; 

    struct TokenGenerateRequest{
        dg::network_std_container::string auth_payload;
    };

    struct TokenGenerateResponse{
        dg::network_std_container::string token;
    };

    struct TokenRefreshRequest{
        dg::network_std_container::string token;
    };

    struct TokenRefreshResponse{
        dg::network_std_container::string token;
    };

    struct TileInitRequest{
        dg::network_std_container::string token;
        dg::network_tile_init_poly::virtual_payload_t payload;
    };

    struct TileInitResponse{
        bool is_success;
    };

    struct TileInjectRequest{
        dg::network_std_container::string token;
        dg::network_tile_inject_poly::virtual_payload_t payload;
    };

    struct TileInjectResponse{
        bool is_sucess;
    };

    struct TileSignalRequest{
        dg::network_std_container::string token;
        dg::network_tile_signal_poly::virtual_payload_t payload;
    };

    struct TileSignalResponse{
        bool is_success;
    };

    struct SysLogRetrieveRequest{
        dg::network_std_container::string token;
        dg::network_std_container::string kind;
        std::chrono::nanoseconds fr;
        std::chrono::nanoseconds to;
        uint32_t limit;
    };
     
    struct SysLogRetrieveResponse{
        dg::network_std_container::vector<dg::network_postgres_db::model::SystemLogEntry> log_vec; //fine
    };

    struct UserLogRetrieveRequest{
        dg::network_std_container::string token;
        dg::network_std_container::string kind;
        std::chrono::nanoseconds fr;
        std::chrono::nanoseconds to;
        uint32_t limit;
    };

    struct UserLogRetrieveResponse{
        dg::network_std_container::vector<dg::network_postgres_db::model::UserLogEntry> log_vec;
    };

    auto serialize_token_generate_request(TokenGenerateRequest request) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_token_generate_request(dg::network_std_container::string request) noexcept -> std::expected<TokenGenerateRequest, exception_t>{

    }

    auto serialize_token_refresh_request(TokenRefreshRequest request) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    } 

    auto deserialize_token_refresh_request(dg::network_std_container::string request) noexcept -> std::expected<TokenRefreshRequest, exception_t>{

    } 

    auto serialize_tile_init_request(TileInitRequest request) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    } 

    auto deserialize_tile_init_request(dg::network_std_container::string request) noexcept -> std::expected<TileInitRequest, exception_t>{

    }

    auto serialize_tile_inject_request(TileInjectRequest request) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_tile_inject_request(dg::network_std_container::string request) noexcept -> std::expected<TileInjectRequest, exception_t>{

    }

    auto serialize_tile_signal_request(TileSignalRequest request) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_tile_signal_request(dg::network_std_container::string request) noexcept -> std::expected<TileSignalRequest, exception_t>{

    }

    auto serialize_tile_condinject_request(TileCondInjectRequest request) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_tile_condinject_request(dg::network_std_container::string request) noexcept -> std::expected<TileCondInjectRequest, exception_t>{

    }

    auto serialize_getsyslog_request(SysLogRetrieveRequest request) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_getsyslog_request(dg::network_std_container::string) noexcept -> std::expected<SysLogRetrieveRequest, exception_t>{

    }

    auto serialize_getusrlog_request(UsrLogRetrieveRequest request) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_getusrlog_request(dg::network_std_container::string request) noexcept -> std::expected<UsrLogRetrieveRequest, exception_t>{

    }

    auto serialize_token_generate_response(TokenGenerateResponse) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_token_generate_response(dg::network_std_container::string) noexcept -> std::expected<TokenGenerateResponse, exception_t>{

    }

    auto serialize_token_refresh_response(TokenRefreshResponse) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_token_refresh_response(dg::network_std_container::string) noexcept -> std::expected<TokenRefreshResponse, exception_t>{

    }

    auto serialize_tile_init_response(TileInitResponse) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_tile_init_response(dg::network_std_container::string) noexcept -> std::expected<TileInitResponse, exception_t>{

    }

    auto serialize_tile_inject_response(TileInjectResponse) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_tile_inject_response(dg::network_std_container::string) noexcept -> std::expected<TileInjectResponse, exception_t>{

    }

    auto serialize_tile_signal_response(TileSignalResponse) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_tile_signal_response(dg::network_std_container::string) noexcept -> std::expected<TileSignalResponse, exception_t>{

    }

    auto serialize_getsyslog_response(SysLogRetrieveResponse) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_getsyslog_response(dg::network_std_container::string) noexcept -> std::expected<SysLogRetrieveResponse, exception_t>{

    }

    auto serialize_getusrlog_response(UsrLogRetrieveResponse) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

    }

    auto deserialize_getusrlog_response(dg::network_std_container::string) noexcept -> std::expected<UsrLogRetrieveResponse, exception_t>{

    }

    class TokenGenerateResolutor{

        public:

            auto handle(Request request) noexcept -> Response{

                std::expected<TokenGenerateRequest, exception_t> tokgen_request = deserialize_token_generate_request(std::move(request.payload));
                
                if (!tokgen_request.has_value()){
                    return Response{std::nullopt, tokgen_request.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> token = dg::network_auth_usrpwd::token_generate_from_auth_payload(tokgen_request->value());

                if (!token.has_value()){
                    return Response{std::nullopt, token.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_token_generate_response(TokenGenerateResponse{std::move(token.value())});

                if (!response_payload.has_value()){
                    return Response{std::nullopt, response_payload.error()};
                }

                return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
            }
    };

    class TokenRefreshResolutor{

        public:

            auto handle(Request request) noexcept -> Response{

                std::expected<TokenRefreshRequest, exception_t> tokrefr_request = deserialize_token_refresh_request(std::move(request.payload));

                if (!tokrefr_request.has_value()){
                    return Response{std::nullopt, tokrefr_request.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> newtok = dg::network_auth_usrpwd::token_refresh(tokrefr_request->token);

                if (!newtok.has_value()){
                    return Response{std::nullopt, newtok.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_token_refresh_response(TokenRefreshResponse{std::move(newtok.value())});

                if (!response_payload.has_value()){
                    return Response{std::nullopt, response_payload.error()};
                }

                return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
            }
    };

    class TileInitResolutor{

        public:

            auto handle(Request request) noexcept -> Response{

                std::expected<TileInitRequest> tinit_request = deserialize_tile_init_request(std::move(request.payload));

                if (!tinit_request.has_value()){
                    return Response{std::nullopt, tinit_request.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(tinit_request->token);

                if (!usr_id.has_value()){
                    return Response{std::nullopt, usr_id.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> clearance = dg::network_user_base::user_get_clearance(usr_id.value());

                if (!clearance.has_value()){
                    return Response{std::nullopt, clearance.error()};
                }

                if (!dg::network_user_base::user_clearance_is_tileinit_qualified(clearance.value())){
                    return Response{std::nullopt, dg::network_exception::UNMET_CLEARANCE};
                }

                exception_t err = dg::network_tile_init_poly::load(std::move(tinit_request->payload)); //I'll fix the move and friends later - I prefer immutable datatypes if theres no perf constraints

                if (dg::network_exception::is_failed(err)){
                    return Response{std::nullopt, err};
                }

                std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_tile_init_response(TileInitResponse{true});

                if (!response_payload.has_value()){
                    return Response{std::nullopt, response_payload.error()};
                }

                return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
            }
    };

    class TileInjectResolutor{

        public:

            auto handle(Request request) noexcept -> Response{

                std::expected<TileInjectRequest, exception_t> tile_inject_request = deserialize_tile_inject_request(std::move(request.payload));

                if (!tile_inject_request.has_value()){
                    return Response{std::nullopt, tile_inject_request.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(tile_inject_request->token);

                if (!usr_id.has_value()){
                    return Response{std::nullopt, usr_id.error()}
                }

                std::expected<dg::network_std_container::string, exception_t> clearance = dg::network_user_base::user_get_clearance(usr_id.value());

                if (!clearance.has_value()){
                    return Response{std::nullopt, clearance.error()};
                }

                if (!dg::network_user_base::user_clearance_is_tileinject_qualified(clearance.value())){
                    return Response{std::nullopt, dg::network_exception::UNMET_CLEARANCE};
                }

                exception_t err = dg::network_tile_inject_poly::load(std::move(tile_inject_request->payload));

                if (dg::network_exception::is_failed(err)){
                    return Response{std::nullopt, err};
                }

                std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_tile_inject_response(TileInjectResponse{true}); //each serialization protocol has a secret - this is to avoid malicious injection + avoid internal_corruption

                if (!response_payload.has_value()){
                    return Response{std::nullopt, response_payload.error()};
                }

                return Response{std::move(reponse_payload.value()), dg::network_exception::SUCCESS};
            }
    };

    class TileSignalResolutor{

        public:

            auto handle(Request request) noexcept -> Response{

                std::expected<TileSingalRequest, exception_t> tile_signal_request = deserialize_tile_signal_request(std::move(request.payload));

                if (!tile_signal_request.has_value()){
                    return Response{std::nullopt, tile_signal_request.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(tile_signal_request->token);

                if (!usr_id.has_value()){
                    return Response{std::nullopt, usr_id.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> clearance = dg::network_user_base::user_get_clearance(usr_id.value());

                if (!clearance.has_value()){
                    return Response{std::nullopt, clearance.error()};
                }

                if (!dg::network_user_base::user_clearance_is_tilesignal_qualified(clearance.value())){ //
                    return Response{std::nullopt, dg::network_exception::UNMET_CLEARANCE};
                }

                exception_t err = dg::network_tile_signal_poly::load(std::move(tile_signal_request->payload));

                if (dg::network_exception::is_failed(err)){
                    return Response{std::nullopt, err};
                }

                std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_tile_signal_response(TileSignalResponse{true});

                if (!response_payload.has_value()){
                    return Response{std::nullopt, response_payload.error()};
                }

                return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
            }
    };

    class TileCondInjectResolutor{

        public:

            auto handle(Request request) noexcept -> Response{

                std::expected<TileCondInjectRequest, exception_t> tile_inject_request = deserialize_tile_condinject_request(std::move(request.payload));

                if (!tile_inject_request.has_value()){
                    return Response{std::nullopt, tile_inject_request.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(tile_inject_request->token);

                if (!usr_id.has_value()){
                    return Response{std::nullopt, usr_id.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> clearance = dg::network_user_base::user_get_clearance(usr_id.value());

                if (!clearance.has_value()){
                    return Response{std::nullopt, clearance.error()};
                }

                if (!dg::network_user_base::user_clearance_is_condinject_qualified(clearance.value())){
                    return Response{std::nullopt, dg::network_exception::UNMET_CLEARANCE};
                }

                exception_t err = dg::network_tile_condinject_poly::load(std::move(tile_inject_request.payload));

                if (dg::network_exception::is_failed(err)){
                    return Response{std::nullopt, err};
                }

                std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_tile_condinject_response(TileCondInjectResponse{true});

                if (!response_payload.has_value()){
                    return Response{std::nullopt, response_payload.error()};
                }

                return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
            }
    };

    class SysLogRetrieveResolutor{

        public:

            auto handle(Request request) noexcept -> Response{

                std::expected<SysLogRetrieveRequest, exception_t> syslog_get_request = deserialize_getsyslog_request(std::move(request.payload));

                if (!syslog_get_request.has_value()){
                    return Response{std::nullopt, syslog_get_request.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(syslog_get_request->token);

                if (!usr_id.has_value()){
                    return Response{std::nullopt, usr_id.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> clearance = dg::network_user_base::user_get_clearance(usr_id.value());

                if (!clearance.has_value()){
                    return Response{std::nullopt, clearance.error()};
                }

                if (!dg::network_user_base::user_clearance_is_syslog_qualified(clearance.value())){
                    return Response{std::nullopt, dg::network_exception::UNMET_CLEARANCE};
                }

                std::expected<dg::network_std_container::vector<dg::network_postgres_db::model::SystemLogEntry>, exception_t> syslog_vec = dg::network_postgres_db::get_systemlog(syslog_get_request->kind, syslog_get_request->fr, syslog_get_request->to, syslog_get_request->limit);

                if (!syslog_vec.has_value()){
                    return Response{std::nullopt, syslog_vec.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_getsyslog_response(SysLogRetrieveResponse{std::move(syslog_vec.value())});
                
                if (!response_payload.has_value()){
                    return Response{std::nullopt, response_payload.error()};
                }

                return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
            }
    };

    class UserLogRetrieveResolutor{

        public:

            auto handle(Request request) noexcept -> Response{

                std::expected<UserLogRetrieveRequest, exception_t> usrlog_get_request = deserialize_getusrlog_request(std::move(request.payload));

                if (!usrlog_get_request.has_value()){
                    return Response{std::nullopt, usrlog_get_request.error()};
                } 

                std::expected<dg::network_std_container::string, exception_t> usr_id = dg::network_auth_usrpwd::token_extract_userid(syslog_get_request->token);

                if (!usr_id.has_value()){
                    return Response{std::nullopt, usr_id.error()};
                }

                std::expected<dg::network_std_container::vector<dg::network_postgres_db::model::UserLogEntry>, exception_t> usrlog_vec = dg::network_postgres_db::get_userlog(usr_id.value(), usrlog_get_request->kind, usrlog_get_request->fr, usrlog_get_request->to, usrlog_get_request->limit);

                if (!usrlog_vec.has_value()){
                    return Response{std::nullopt, usrlog_vec.error()};
                }

                std::expected<dg::network_std_container::string, exception_t> response_payload = serialize_getusrlog_response(UserLogRetrieveResponse{std::move(usrlog_vec.value())});

                if (!response_payload.has_value()){
                    return Response{std::nullopt, response_payload.error()};
                }

                return Response{std::move(response_payload.value()), dg::network_exception::SUCCESS};
            }
    };
};

#endif