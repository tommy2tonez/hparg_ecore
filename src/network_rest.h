
#ifndef __DG_NETWORK_REST_H__
#define __DG_NETWORK_REST_H__

#include "network_rest_frame.h"

namespace dg::network_rest{
    
    //Mom said to reconsider the API

    //this is the sole comm -> client -> p2p -> etc., we'll definitely have to implement a liason on client side to talk to this
    //what do we need?

    //log_extraction:
    //  dedicated_log_id extraction
    //  system_log extraction

    //external stable tile operations
    //  public interfaces:
    //      init (in interpretable uint32_t uint64_t)
    //      deinit
    //      orphan
    //      ping tiles

    //system stats:
        //uma memregion size
        //memlock region size
        //uma -> vma_ptr -> fsys -> etc.
        //CPU, GPU, clock, speed, status, etc.
        //memregion frequencies
        //available forward dispatch codes, backward dispatch codes in interpretable strings

    //authentication:
        //register p2p auth
        //register system auth
        //permissions + friends

    //network:
        //blacklist ip
        //greenlist ip
        //set ips max flux
        //set global memregion segments mapping 

    //----------api-version-----------

    //all apis must adhere to the rule of request being semantically sensical (a serialization format that is not JSON), and output being a JSON response (a human readable serialization format), clearly documented in the API explaination section for every server version
    //we'd just have to force the user to use a different serialization for input, for a lot of reasons
    //let's say we have std::chrono::utc_clock, we'd want to catch that error, to be differented from the std::chrono::high_resolution_clock, etc
    //we dont care about output, because it's documented clearly about what that semantically means, while the input we can't enforce such a thing

    struct RESTAPIVersionRequest{

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            (void) reflector;
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            (void) reflector;
        }
    };

    struct RESTAPIVersionResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    struct ExceptionVersionRequest{

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            (void) reflector;    
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            (void) reflector;
        }
    };

    struct ExceptionVersionResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    struct ServerVersionRequest{

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            (void) reflector;
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            (void) reflector;
        }
    };

    struct ServerVersionResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    //--------------------------------

    //---------authentication---------

    struct Auth2UserNamePasswordPayLoad{
        dg::string username;
        dg::string password;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(username, password);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(username, password);
        }
    };

    struct Auth2TokenPayLoad{
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

    struct GenericTokenGenerateAuth2Request{
        std::variant<Auth2UserNamePasswordPayLoad, Auth2TokenPayLoad> payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(payload);
        }
    };

    struct GenericTokenGenerateAuth2Response{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    struct ProtectedTokenGenerateAuth2Request{
        std::variant<Auth2UserNamePasswordPayLoad, Auth2TokenPayLoad> payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(payload);
        }
    };

    struct ProtectedTokenGenerateAuth2Response{
        dg::string token;
        std::chrono::time_point<std::chrono::utc_clock> expiry;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token, expiry, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token, expiry, err);
        }
    };

    struct Auth2UserNamePasswordRegistrationPayLoad{
        dg::string sys_token;
        dg::string username;
        dg::string password;
        dg::string permission_level;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, username, password, permission_level);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, username, password, permission_level);
        }
    };

    struct Auth2DedicatedTokenRegistrationPayLoad{
        dg::string sys_token;
        dg::string token;
        dg::string permission_level;
        std::chrono::time_point<std::chrono::utc_clock> expiry;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, token, permission_level, expiry);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, token, permission_level, expiry);
        }
    };

    struct Auth2RegistrationRequest{
        std::variant<Auth2UserNamePasswordRegistrationPayLoad, Auth2DedicatedTokenRegistrationPayLoad> payload;
        
        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(payload);
        }
    };

    struct Auth2RegistrationResponse{
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(err);
        }
    };

    struct Auth2UserNamePasswordDeregistrationPayLoad{
        dg::string sys_token;
        dg::string username;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, username);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, username);
        }
    };

    struct Auth2TokenDeregistrationPayLoad{
        dg::string sys_token;
        dg::string token;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, token);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, token);
        }
    };

    struct Auth2DeregistrationRequest{
        std::variant<Auth2UserNamePasswordDeregistrationPayLoad, Auth2TokenDeregistrationPayLoad> payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(payload);
        }
    };

    struct Auth2DeregistrationResponse{
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(err);
        }
    };

    //--------------------------------

    //---------log-retrieval----------

    struct SysLogRetrieveRequest{
        dg::string auth2_token;
        dg::string kind;
        std::chrono::time_point<std::chrono::utc_clock> fr;
        std::chrono::time_point<std::chrono::utc_clock> to;
        uint32_t limit;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth2_token, kind, fr, to, limit);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(auth2_token, kind, fr, to, limit);
        }
    };

    struct SysLogRetrieveResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    struct DedicatedLogRetrieveRequest{
        dg::string auth2_token;
        uint64_t dedicated_log_id;
        std::chrono::time_point<std::chrono::utc_clock> fr;
        std::chrono::time_point<std::chrono::utc_clock> to;
        uint32_t limit;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth2_token, dedicated_log_id, fr, to, limit);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(auth2_token, dedicated_log_id, fr, to, limit);
        }
    };

    struct DedicatedLogRetrieveResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    //------------------------------

    //---------tile-actions---------

    struct TileActionRequest{
        dg::string auth2_token;
        dg::string payload;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth2_token, payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(auth2_token, payload);
        }
    };

    struct TileActionResponse{
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(err);
        }
    };

    struct TileActionClientVersionRequest{

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            (void) reflector;
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            (void) reflector;
        }
    };

    struct TileActionClientVersionResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    struct TileActionClientInitPayLoadRequest{
        dg::string auth2_token;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth2_token);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(auth2_token);
        }
    };

    struct TileActionClientInitPayLoadResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    //------------------------------

    //---------system-stats---------

    struct SystemDescriptionRequest{
        dg::string auth2_token;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth2_token);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(auth2_token);
        }
    };

    struct SystemDescriptionResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    struct SystemStatRequest{
        dg::string auth2_token;
        std::chrono::time_point<std::chrono::utc_clock> fr;
        std::chrono::time_point<std::chrono::utc_clock> to;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth2_token, fr, to);
        }
        
        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(auth2_token, fr, to);
        }
    };

    struct SystemStatResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    struct SoftwareConfigurationDescriptionRequest{
        dg::string auth2_token;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth2_token);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(auth2_token);
        }
    };

    struct SoftwareConfigurationDescriptionResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    //------------------------------

    //----------network-------------

    struct NetworkBlackListRequest{
        dg::string sys_token;
        dg::vector<dg::string> ip_vec;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, ip_vec);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, ip_vec);
        }
    };

    struct NetworkBlackListResponse{
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(err);
        }
    };

    struct NetworkGreenListRequest{
        dg::string sys_token;
        dg::vector<dg::string> ip_vec;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, ip_vec);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, ip_vec);
        }
    };

    struct NetworkGreenListResponse{
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(err);
        }
    };

    struct NetworkSetGlobalInBoundCapacityRequest{
        dg::string sys_token;
        uint64_t byte_per_second;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, byte_per_second);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, byte_per_second);
        }
    };

    struct NetworkSetGlobalInBoundCapacityResponse{
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(err);
        }
    };

    struct NetworkSetGlobalOutBoundCapacityRequest{
        dg::string sys_token;
        uint64_t byte_per_second;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, byte_per_second);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, byte_per_second);
        }
    };

    struct NetworkSetGlobalOutBoundCapacityResponse{
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(err);
        }
    };

    struct NetworkSetIndividualInBoundCapacityRequest{
        dg::string sys_token;
        uint64_t byte_per_second;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, byte_per_second);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, byte_per_second);
        }
    };

    struct NetworkSetIndividualInBoundCapacityResponse{
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(err);
        }
    };

    struct NetworkGetConfigurationRequest{
        dg::string sys_token;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token);
        }
    };

    struct NetworkGetConfigurationResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    struct NetworkFluxStatRequest{
        dg::string sys_token;
        std::chrono::time_point<std::chrono::utc_clock> fr;
        std::chrono::time_point<std::chrono::utc_clock> to;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, fr, to);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, fr, to);
        }
    };

    struct NetworkFluxStatResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    struct NetworkIPFluxStatRequest{
        dg::string sys_token;
        dg::string ip;
        std::chrono::time_point<std::chrono::utc_clock> fr;
        std::chrono::time_point<std::chrono::utc_clock> to;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, ip, fr, to);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, ip, fr, to);
        }
    };

    struct NetworkIPFluxStatResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    //------------------------------

    //------memregion-segments------

    struct MemoryMappingDescriptionRequest{
        dg::string sys_token;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token);
        }
    };

    struct MemoryMappingDescriptionResponse{
        dg::string response;
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(response, err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(response, err);
        }
    };

    struct MemoryTranslationUnit{
        uint64_t this_uma_addr;
        uint64_t that_uma_addr;
        dg::string that_ip;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(this_uma_addr, that_uma_addr, that_ip);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(this_uma_addr, that_uma_addr, that_ip);
        }
    };

    struct ExternalMemoryMappingRequest{
        dg::string sys_token;
        dg::vector<MemoryTranslationUnit> memory_mapping_payload; //we'd try to semanticalize these guys whenever possible, we'd implement a liaison to translate this -> Flask
                                                                  //problem is that we can't actually do normal Flask + friends for our system, because it is very error-prone + hard to debug
                                                                  //we'd rather implement a semantic layer to seperate the client + server, we'd rahter implement a liasion, problem is that we'd have to implement version control for mailbox + rest request + etc. 
        
        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(sys_token, memory_mapping_payload);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(sys_token, memory_mapping_payload);
        }
    };

    struct ExternalMemoryMappingResponse{
        exception_t err;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(err);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(err);
        }
    };

    //------------------------------

    static inline constexpr std::string_view REST_API_VERSION_GET_ROUTE                         = std::string_view("/get/basic/rest_api_version");
    static inline constexpr std::string_view REST_API_EXCEPTION_VERSION_GET_ROUTE               = std::string_view("/get/basic/exception_version");
    static inline constexpr std::string_view REST_API_SERVER_VERSION_GET_ROUTE                  = std::string_view("/get/basic/server_version");

    static inline constexpr std::string_view REST_API_AUTH2_GET_ROUTE                           = std::string_view("/get/auth2/token");
    static inline constexpr std::string_view REST_API_AUTH2_REGISTRATION_SET_ROUTE              = std::string_view("/set/auth2/registration");
    static inline constexpr std::string_view REST_API_AUTH2_DEREGISTRATION_SET_ROUTE            = std::string_view("/set/auth2/deregistration");

    static inline constexpr std::string_view REST_API_SYSLOG_GET_ROUTE                          = std::string_view("/get/log/syslog");
    static inline constexpr std::string_view REST_API_DEDICATED_LOG_GET_ROUTE                   = std::string_view("/get/log/dedicated_log");

    static inline constexpr std::string_view REST_API_TILE_ACTION_PAYLOAD_SET_ROUTE             = std::string_view("/set/core/tileaction/payload");
    static inline constexpr std::string_view REST_API_TILE_ACTION_CLIENT_VERSION_GET_ROUTE      = std::string_view("/get/core/tileaction/client_version");
    static inline constexpr std::string_view REST_API_TILE_ACTION_CLIENT_INIT_PAYLOAD_GET_ROUTE = std::string_view("/get/core/tileaction/client_payload");
    // static inline constexpr std::string_view REST_API_SOFTWARE_CONFIG_GET_ROUTE                 = "/get/core/config";

    static inline constexpr std::string_view REST_API_SYSTEM_DESCRIPTION_GET_ROUTE              = std::string_view("/get/sys/description");
    static inline constexpr std::string_view REST_API_SYSTEM_STAT_GET_ROUTE                     = std::string_view("/get/sys/stat");

    static inline constexpr std::string_view REST_API_NETWORK_BLACKLIST_SET_ROUTE               = std::string_view("/set/network/blacklist");
    static inline constexpr std::string_view REST_API_NETWORK_GREENLIST_SET_ROUTE               = std::string_view("/set/network/greenlist");
    static inline constexpr std::string_view REST_API_NETWORK_GLOBAL_INBOUNDCAP_SET_ROUTE       = std::string_view("/set/network/bandwidth/global_inbound");
    static inline constexpr std::string_view REST_API_NETWORK_GLOBAL_OUTBOUNDCAP_SET_ROUTE      = std::string_view("/set/network/bandwidth/global_outbound");
    static inline constexpr std::string_view REST_API_NETWORK_INDIVIDUAL_INBOUNDCAP_SET_ROUTE   = std::string_view("/set/network/bandwidth/individual_inbound");
    static inline constexpr std::string_view REST_API_NETWORK_INDIVIDUAL_OUTBOUNDCAP_SET_ROUTE  = std::string_view("/set/network/bandwidth/individual_outbound");

    static inline constexpr std::string_view REST_API_NETWORK_FLUX_STAT_GET_REQUEST             = std::string_view("/get/network/stat/sys_flux");
    static inline constexpr std::string_view REST_API_NETWORK_IP_FLUX_STAT_GET_REQUEST          = std::string_view("/get/network/stat/ip_flux");

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