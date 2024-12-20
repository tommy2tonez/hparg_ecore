
#ifndef __DG_NETWORK_REST_H__
#define __DG_NETWORK_REST_H__

#include "network_rest_frame.h"

namespace dg::network_rest{
    
    //Mom said to reconsider the API

    struct TokenGenerateBaseRequest{
        dg::string auth_payload;

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
        dg::string uri;
        dg::string requestor;
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
        dg::string token;
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
        dg::string token;

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
        dg::string uri;
        dg::string requestor;
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
        dg::string token;
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
        dg::string token;
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
        dg::string uri;
        dg::string requestor;
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
        dg::string token;
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
        dg::string uri;
        dg::string requestor;
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
        dg::string token;
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
        dg::string uri;
        dg::string requestor;
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
        dg::string token;
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
        dg::string uri;
        dg::string requestor;
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
        dg::string token;
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
        dg::string uri;
        dg::string requestor;
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
        dg::string token;
        dg::string kind;
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
        dg::string uri;
        dg::string requestor;
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
        dg::vector<dg::network_postgres_db::model::SystemLogEntry> log_vec;
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
        dg::string token;
        dg::string kind;
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
        dg::string uri;
        dg::string requestor;
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
        dg::vector<dg::network_postgres_db::model::UserLogEntry> log_vec;
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
    
    class TokenGenerateResolutor: public virtual dg::network_rest_frame::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                TokenGenerateBaseRequest tokgen_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenGenerateBaseRequest>)(tokgen_request, request.payload.data(), request.payload.size());
                
                if (dg::network_exception::is_failed(err)){
                    TokenGenerateBaseResponse tokgen_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tokgen_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> token = dg::network_user::token_generate_from_auth_payload(tokgen_request.auth_payload);

                if (!token.has_value()){
                    TokenGenerateBaseResponse tokgen_response{{}, token.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tokgen_response), dg::network_exception::SUCCESS};
                }

                TokenGenerateBaseResponse tokgen_response{std::move(token.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tokgen_response), dg::network_exception::SUCCESS};
            }
    };

    class TokenRefreshResolutor: public virtual dg::network_rest_frame::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TokenRefreshBaseRequest tokrefr_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TokenRefreshBaseRequest>)(tokrefr_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TokenRefreshBaseResponse tokrefr_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tokrefr_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> newtok = dg::network_user::token_refresh(tokrefr_request.token);

                if (!newtok.has_value()){
                    TokenRefreshBaseResponse tokrefr_response{{}, newtok.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tokrefr_response), dg::network_exception::SUCCESS}; //fine - 
                }

                TokenRefreshBaseResponse tokrefr_response{std::move(newtok.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tokrefr_response), dg::network_exception::SUCCESS};
            }
    };

    class TileInitResolutor: public virtual dg::network_rest_frame::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileInitBaseRequest tileinit_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInitBaseRequest>)(tileinit_request, request.payload.data(), request.payload.size()); 

                if (dg::network_exception::is_failed(err)){
                    TileInitBaseResponse tileinit_response{err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_id = dg::network_user::token_extract_userid(tileinit_request.token);
                
                if (!user_id.has_value()){
                    TileInitBaseResponse tileinit_response{user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileInitBaseResponse tileinit_response{user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileInitBaseResponse tileinit_response{ dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tileinit_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_init::load(std::move(request.payload));
                TileInitBaseResponse tileinit_response{load_err};
                
                return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tileinit_response), dg::network_exception::SUCCESS};
            }
    };

    class TileInjectResolutor: public virtual dg::network_rest_frame::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                TileInjectBaseRequest tileinject_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileInjectBaseRequest>)(tileinject_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TileInjectBaseResponse tileinject_response{err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_id = dg::network_user::token_extract_userid(tileinject_request.token);

                if (!user_id.has_value()){
                    TileInjectBaseResponse tileinject_response{user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileInjectBaseResponse tileinject_response{user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileInjectBaseResponse tileinject_response{dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tileinject_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_inject::load(std::move(request.payload));
                TileInjectBaseResponse tileinject_response{load_err};
                
                return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tileinject_response), dg::network_exception::SUCCESS};
            }
    };

    class TileSignalResolutor: public virtual dg::network_rest_frame::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileSignalBaseRequest tilesignal_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileSignalBaseRequest>)(tilesignal_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TileSignalBaseResponse tilesignal_response{err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_id = dg::network_user::token_extract_userid(tilesignal_request.token);

                if (!user_id.has_value()){
                    TileSignalBaseResponse tilesignal_response{user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileSignalBaseResponse tilesignal_response{user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileSignalBaseResponse tilesignal_response{dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tilesignal_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_signal::load(std::move(tilesignal_request.payload));//this is not stateless - 
                TileSignalBaseResponse tilesignal_response{load_err};

                return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(tilesignal_response), dg::network_exception::SUCCESS};
            }
    };

    class TileCondInjectResolutor: public virtual dg::network_rest_frame::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileCondInjectBaseRequest condinject_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileCondInjectBaseRequest>)(condinject_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    TileCondInjectBaseResponse condinject_response{err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_id = dg::network_user::token_extract_userid(condinject_request.token);

                if (!user_id.has_value()){
                    TileCondInjectBaseResponse condinject_response{user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileCondInjectBaseResponse condinject_response{user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileCondInjectBaseResponse condinject_response{dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(condinject_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_condinject::load(std::move(condinject_request.payload));
                TileCondInjectBaseResponse condinject_response{load_err};

                return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(condinject_response), dg::network_exception::SUCCESS};
            }
    };

    class TileMemcommitResolutor: public virtual dg::network_rest_frame::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{

                TileMemcommitBaseRequest memcommit_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<TileMemcommitBaseRequest>)(memcommit_request, request.payload.data(), request.payload.size()); 

                if (dg::network_exception::is_failed(err)){
                    TileMemcommitBaseResponse memcommit_response{err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(memcommit_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_id = dg::network_user::token_extract_userid(memcommit_request.token);

                if (!user_id.has_value()){
                    TileMemcommitBaseResponse memcommit_response{user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(memcommit_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    TileMemcommitBaseResponse memcommit_response{user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(memcommit_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_GLOBAL_MODIFY){
                    TileMemcommitBaseResponse memcommit_response{dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(memcommit_response), dg::network_exception::SUCCESS};
                }

                exception_t load_err = dg::network_tile_memcommit::load(std::move(memcommit_request.payload));
                TileMemcommitBaseResponse memcommit_response{load_err};

                return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(memcommit_response), dg::network_exception::SUCCESS};

            }
    };

    class SysLogRetrieveResolutor: public virtual dg::network_rest_frame::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                SysLogRetrieveBaseRequest syslog_get_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<SysLogRetrieveBaseRequest>)(syslog_get_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    SysLogRetrieveBaseResponse syslog_get_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_id = dg::network_user::token_extract_userid(syslog_get_request.token);

                if (!user_id.has_value()){
                    SysLogRetrieveBaseResponse syslog_get_response{{}, user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    SysLogRetrieveBaseResponse syslog_get_response{{}, user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::to_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_SYS_READ){
                    SysLogRetrieveBaseResponse syslog_get_response{{}, dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::vector<dg::network_postgres_db::model::SystemLogEntry>, exception_t> syslog_vec = dg::network_postgres_db::get_systemlog(syslog_get_request.kind, syslog_get_request.fr, 
                                                                                                                                                                                  syslog_get_request.to, syslog_get_request.limit);

                if (!syslog_vec.has_value()){
                    SysLogRetrieveBaseResponse syslog_get_response{{}, syslog_vec.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(syslog_get_response), dg::network_exception::SUCCESS};
                }

                SysLogRetrieveBaseResponse syslog_get_response{std::move(syslog_vec.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(syslog_get_response), dg::network_exception::SUCCESS};
            }
    };

    class UserLogRetrieveResolutor: public virtual dg::network_rest_frame::server::RequestHandlerInterface{

        public:

            auto handle(Request request) noexcept -> Response{
                
                UserLogRetrieveBaseRequest usrlog_get_request{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<UserLogRetrieveBaseRequest>)(usrlog_get_request, request.payload.data(), request.payload.size());

                if (dg::network_exception::is_failed(err)){
                    UserLogRetrieveBaseResponse usrlog_get_response{{}, err};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_id = dg::network_user::token_extract_userid(usrlog_get_request.token);

                if (!user_id.has_value()){
                    UserLogRetrieveBaseResponse usrlog_get_response{{}, user_id.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::string, exception_t> user_clearance = dg::network_user::user_get_clearance(user_id.value());

                if (!user_clearance.has_value()){
                    UserLogRetrieveBaseResponse usrlog_get_response{{}, user_clearance.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                if (dg::network_user::get_clearance_level(user_clearance.value()) < dg::network_user::CLEARANCE_LEVEL_SELF_READ){
                    UserLogRetrieveBaseResponse usrlog_get_response{{}, dg::network_exception::UNMET_CLEARANCE};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                std::expected<dg::vector<dg::network_postgres_db::model::UserLogEntry>, exception_t> usrlog_vec = dg::network_postgres_db::get_userlog(user_id.value(), usrlog_get_request.kind, 
                                                                                                                                                                              usrlog_get_request.fr, usrlog_get_request.to, 
                                                                                                                                                                              usrlog_get_request.limit);

                if (!usrlog_vec.has_value()){
                    UserLogRetrieveBaseResponse usrlog_get_response{{}, usrlog_vec.error()};
                    return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(usrlog_get_response), dg::network_exception::SUCCESS};
                }

                UserLogRetrieveBaseResponse usrlog_get_response{dg::move(usrlog_vec.value()), dg::network_exception::SUCCESS};
                return Response{dg::network_compact_serializer::integrity_serialize<dg::string>(usrlog_get_response), dg::network_exception::SUCCESS};
            }
    };

    auto request_token_get(dg::network_rest_frame::client::RestControllerInterface& controller, const TokenGenerateRequest& request) noexcept -> std::expected<TokenGenerateResponse, exception_t>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response; 

        auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(static_cast<const TokenGenerateBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TokenGenerateBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_rest_frame::client::request(controller, std::move(base_request)); 

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

    auto request_token_refresh(dg::network_rest_frame::client::RestControllerInterface& controller, const TokenRefreshRequest& request) noexcept -> std::expected<TokenRefreshResponse, exception_t>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(static_cast<const TokenRefreshBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TokenRefreshBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_rest_frame::client::request(controller, std::move(base_request));

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

    auto request_tile_init(dg::network_rest_frame::client::RestControllerInterface& controller, const TileInitRequest& request) noexcept -> std::expected<TileInitResponse, exception_t>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(static_cast<const TileInitBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TileInitBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_rest_frame::client::request(controller, std::move(base_request));

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

    auto request_tile_inject(dg::network_rest_frame::client::RestControllerInterface& controller, const TileInjectRequest& request) noexcept -> std::expected<TileInjectResponse, exception_t>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::newtork_post_rest::model::Response;

        auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(static_cast<const TileInjectBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TileInjectBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_rest_frame::client::request(std::move(base_request));

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

    auto request_tile_signal(dg::network_rest_frame::client::RestControllerInterface& controller, const TileSignalRequest& request) noexcept -> std::expected<TileSignalResponse, exception_t>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(static_cast<const TileSignalBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TileSignalBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_rest_frame::client::request(std::move(base_request));

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

    auto request_tile_condinject(dg::network_rest_frame::client::RestControllerInterface& controller, const TileCondInjectRequest& request) noexcept -> std::expected<TileCondInjectResponse, exception_t>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(static_cast<const TileCondInjectBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TileCondInjectBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_rest_frame::client::request(std::move(base_request));

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

    auto request_tile_seqmemcommit(dg::network_rest_frame::client::RestControllerInterface& controller, const TileMemcommitRequest& request) noexcept -> std::expected<TileMemcommitResponse, exception_t>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(static_cast<const TileMemcommitBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const TileMemcommitBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_rest_frame::client::request(std::move(base_request));

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

    auto request_syslog_get(dg::network_rest_frame::client::RestControllerInterface& controller, const SysLogRetrieveRequest& request) noexcept -> std::expected<SysLogRetrieveResponse, exception_t>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(static_cast<const SysLogRetrieveBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const SysLogRetrieveBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_rest_frame::client::request(std::move(base_request));

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

    auto request_usrlog_get(dg::network_rest_frame::client::RestControllerInterface& controller, const UserLogRetrieveRequest& request) noexcept -> std::expected<UserLogRetrieveResponse, exception_t>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(static_cast<const UserLogRetrieveBaseRequest&>(request)), ' ');
        dg::network_compact_serializer::integrity_serialize_into(bstream.data(), static_cast<const UserLogRetrieveBaseRequest&>(request));
        auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
        std::expected<BaseResponse, exception_t> base_response = dg::network_rest_frame::client::request(std::move(base_request));

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

    auto requestmany_token_get(dg::network_rest_frame::client::RestControllerInterface& controller, const dg::vector<TokenGenerateRequest>& req_vec) noexcept -> dg::vector<std::expected<TokenGenerateResponse, exception_t>>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        dg::vector<BaseRequest> base_request_vec{};
        dg::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::vector<std::expected<TokenGenerateResponse, exception_t>> rs{};

        for (const TokenGenerateBaseRequest& req: req_vec){
            auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timestamp};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_rest_frame::client::request_many(std::move(base_request_vec));
        
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

    auto requestmany_token_refresh(dg::network_rest_frame::client::RestControllerInterface& controller, const dg::vector<TokenRefreshRequest>& req_vec) noexcept -> dg::vector<std::expected<TokenRefreshResponse, exception_t>>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        dg::vector<BaseRequest> base_request_vec{};
        dg::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::vector<std::expected<TokenRefreshResponse, exception_t>> rs{};

        for (const TokenRefreshBaseRequest& req: req_vec){
            auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timestamp};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_rest_frame::client::request_many(std::move(base_request_vec));

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

    auto requestmany_tile_init(dg::network_rest_frame::client::RestControllerInterface& controller, const dg::vector<TileInitRequest>& req_vec) noexcept -> dg::vector<std::expected<TileInitResponse, exception_t>>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        dg::vector<BaseRequest> base_request_vec{};
        dg::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::vector<std::expected<TileInitResponse, exception_t>> rs{};

        for (const TileInitBaseRequest& req: req_vec){
            auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(req), ' '); //optimizable
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req); //optimizable
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timestamp};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_rest_frame::client::request_many(std::move(base_request_vec));

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

    auto requestmany_tile_inject(dg::network_rest_frame::client::RestControllerInterface& controller, const dg::vector<TileInjectRequest>& req_vec) noexcept -> dg::vector<std::expected<TileInjectResponse, exception_t>>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        dg::vector<BaseRequest> base_request_vec{};
        dg::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::vector<std::expected<TileInjectResponse, exception_t>> rs{};

        for (const TileInjectBaseRequest& req: req_vec){
            auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_rest_frame::client::request_many(std::move(base_request_vec));

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

    auto requestmany_tile_signal(dg::network_rest_frame::client::RestControllerInterface& controller, const dg::vector<TileSignalRequest>& req_vec) noexcept -> dg::vector<std::expected<TileSignalResponse, exception_t>>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        dg::vector<BaseRequest> base_request_vec{};
        dg::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::vector<std::expected<TileSignalResponse, exception_t>> rs{};

        for (const TileSignalBaseRequest& req: req_vec){
            auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
            request_vec.push_back(std::move(base_request));
        }
        
        base_response_vec = dg::network_rest_frame::client::request_many(std::move(base_request_vec));

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

    auto requestmany_tile_condinject(dg::network_rest_frame::client::RestControllerInterface& controller, const dg::vector<TileCondInjectRequest>& req_vec) noexcept -> dg::vector<std::expected<TileCondInjectResponse, exception_t>>{
        
        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        dg::vector<BaseRequest> base_request_vec{};
        dg::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::vector<std::expected<TileCondInjectResponse, exception_t>> rs{};

        for (const TileCondInjectBaseRequest& req: req_vec){
            auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        } 

        base_response_vec = dg::network_rest_frame::client::request_many(std::move(base_request_vec));

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

    auto requestmany_tile_seqmemcommit(dg::network_rest_frame::client::RestControllerInterface& controller, const dg::vector<TileMemcommitRequest>& req_vec) noexcept -> dg::vector<std::expected<TileMemcommitResponse, exception_t>>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        dg::vector<BaseRequest> base_request_vec{};
        dg::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::vector<std::expected<TileMemcommitResponse, exception_t>> rs{};

        for (const TileMemcommitBaseRequest& req: req_vec){
            auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_rest_frame::client::request_many(std::move(base_request_vec));

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

    auto requestmany_syslog_get(dg::network_rest_frame::client::RestControllerInterface& controller, const dg::vector<SysLogRetrieveRequest>& req_vec) noexcept -> dg::vector<std::expected<SysLogRetrieveResponse, exception_t>>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        dg::vector<BaseRequest> base_request_vec{};
        dg::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::vector<std::expected<SysLogRetrieveResponse, exception_t>> rs{};

        for (const SysLogRetrieveBaseRequest& req: req_vec){
            auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{request.uri, request.requestor, std::move(bstream), request.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec = dg::network_rest_frame::client::request_many(std::move(base_request_vec));

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

    auto requestmany_usrlog_get(dg::network_rest_frame::client::RestControllerInterface& controller, const dg::vector<UserLogRetrieveRequest>& req_vec) noexcept -> dg::vector<std::expected<UserLogRetrieveResponse, exception_t>>{

        using BaseRequest   = dg::network_rest_frame::model::Request;
        using BaseResponse  = dg::network_rest_frame::model::Response;

        dg::vector<BaseRequest> base_request_vec{};
        dg::vector<std::expected<BaseResponse, exception_t>> base_response_vec{};
        dg::vector<std::expected<UserLogRetrieveResponse, exception_t>> rs{};

        for (const UserLogRetrieveBaseRequest& req: req_vec){
            auto bstream        = dg::string(dg::network_compact_serializer::integrity_size(req), ' ');
            dg::network_compact_serializer::integrity_serialize_into(bstream.data(), req);
            auto base_request   = BaseRequest{req.uri, req.requestor, std::move(bstream), req.timeout};
            base_request_vec.push_back(std::move(base_request));
        }

        base_response_vec   = dg::network_rest_frame::client::request_many(std::move(base_request_vec));

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