#ifndef __DG_NETWORK_AUTH_H__
#define __DG_NETWORK_AUTH_H__

#include "network_std_container.h"

namespace dg::network_token{

    struct Token{
        std::chrono::nanoseconds expiry_time;
        std::chrono::nanoseconds refresh_expiry_time;
        dg::network_std_container::string content;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(expiry_time, refresh_expiry_time, content);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(expiry_time, refresh_expiry_time, content);
        }
    };

    struct EncoderInterface{
        virtual ~EncoderInterface() noexcept = default;
        virtual auto encode(const dg::network_std_container::string&) noexcept -> std::expected<dg::network_std_container::string, exception_t> = 0;
        virtual auto decode(const dg::network_std_container::string&) noexcept -> std::expected<dg::network_std_container::string, exception_t> = 0; 
    };

    struct TokenControllerInterface{
        virtual ~TokenControllerInterface() noexcept = default;
        virtual auto tokenize(const dg::network_std_container::string&) noexcept -> std::expected<dg::network_std_container::string, exception_t> = 0;
        virtual auto detokenize(const dg::network_std_container::string&) noexcept -> std::expected<dg::network_std_container::string, exception_t> = 0;
        virtual auto renew_token(const dg::network_std_container::string&) noexcept -> std::expected<dg::network_std_container::string, exception_t> = 0; 
    };

    struct MurMurMessage{
        uint64_t validation_key;
        dg::network_std_container::string encoded; //whatever - this can be encoded by using dictionaries - let secret be a randomization pivot - generate a random dictionary to map(a) -> b, does N map - then we have an encoding method  

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(validation_key, encoded);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(validation_key, encoded);
        }
    };

    class MurMurEncoder: public virtual EncoderInterface{

        private:

            uint64_t secret;

        public:

            MurMurEncoder(uint64_t secret) noexcept: secret(secret){}

            auto encode(const dg::network_std_container::string& arg) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

                uint64_t key    = dg::network_hash::murmur_hash(arg.data(), arg.size(), this->secret);
                auto msg        = MurMurMessage{key, arg};
                auto bstream    = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(msg));
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), msg);

                return bstream;
            }

            auto decode(const dg::network_std_container::string& arg) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

                MurMurMessage msg{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<MurMurMessage>)(msg, arg.data(), arg.size());

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                }

                uint64_t expected_key = dg::network_hash::murmur_hash(msg.encoded.data(), msg.encoded.size(), this->secret);
                
                if (expected_key != msg.validation_key){
                    return std::unexpected(dg::network_exception::BAD_DECODE);
                }

                return {std::in_place_t{}, std::move(msg.encoded)}
            }
    };
    
    struct Mt19937Message{
        uint64_t seed;
        dg::network_std_container::string encoded;
    };

    class Mt19937Encoder: public virtual EncoderInterface{

        private:

            uint64_t secret;
        
        public:

            Mt19937Encoder(uint64_t secret) noexcept: secret(secret){}

            auto encode(const dg::network_std_container::string& arg) noexcept -> std::expected<dg::network_std_container::string, exception_t>{
                
                uint64_t arg_seed   = dg::network_hash::murmur_hash(arg.data(), arg.size(), this->secret);
                uint64_t seed       = this->hash(std::make_pair(this->secret, this->arg_seed));  
                auto randomizer     = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937(seed)); 
                auto encoded        = dg::network_std_container::string(arg.size());

                for (size_t i = 0u; i < arg.size(); ++i){
                    encoded[i] = this->byte_encode(randomizer, arg[i]);
                }

                auto msg            = Mt19937Message{arg_seed, std::move(encoded)};
                auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(msg));
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), msg);

                return {std::in_place_t{}, std::move(bstream)};
            }

            auto decode(const dg::network_std_container::string& arg) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

                Mt19937Message msg{};
                exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<Mt19937Message>)(msg, arg.data(), arg.size());

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                }

                uint64_t seed       = this->hash(std::make_pair(this->secret, msg.seed));
                auto randomizer     = std::bind(std::uniform_int_distribution<size_t>{}, std::mt19937(seed));
                
                for (size_t i = 0u; i < msg.encoded.size(); ++i){
                    msg.encoded[i] = this->byte_decode(randomizer, msg.encoded[i]);
                }

                return {std::in_place_t{}, std::move(msg.encoded)};
            }
        
        private:
            
            template <class Functor>
            auto get_byte_dict(Functor& functor) noexcept -> dg::network_std_container::vector<uint8_t>{
                
                dg::network_std_container::vector<uint8_t> rs(256);
                std::iota(rs.begin(), rs.end(), 0u);

                for (size_t i = 0u; i < 256; ++i){
                    size_t lhs_idx = static_cast<size_t>(functor()) % 256;
                    size_t rhs_idx = static_cast<size_t>(functor()) % 256;
                    std::swap(rs[lhs_idx], rs[rhs_idx]); 
                }

                return rs;
            } 

            template <class Functor>
            auto byte_encode(Functor& functor, char key) noexcept -> char{

                dg::network_std_container::vector<uint8_t> dict = get_byte_dict(functor);
                return std::bit_cast<char>(dict[std::bit_cast<uint8_t>(key)]);
            }

            template <class Functor>
            auto byte_decode(Functor& functor, char value) noexcept -> char{
                
                dg::network_std_container::vector<uint8_t> dict = get_byte_dict(functor);
                uint8_t key = std::distance(dict.begin(), std::find(dict.begin(), dict.end(), std::bit_cast<uint8_t>(value))); 

                return std::bit_cast<char>(key);
            }
    };

    class DoubleEncoder: public virtual EncoderInterface{

        private:

            std::unique_ptr<EncoderInterface> first_encoder;
            std::unique_ptr<EncoderInterface> second_encoder;
        
        public:

            DoubleEncoder(std::unique_ptr<EncoderInterface> first_encoder,
                          std::unique_ptr<EncoderInterface> second_encoder) noexcept: first_encoder(std::move(first_encoder)),
                                                                                      second_encoder(std::move(second_encoder)){}
            
            auto encode(const dg::network_std_container::string& msg) noexcept -> std::expected<dg::network_std_container::string, exception_t>{
                
                auto first_encoded = this->first_encoder->encode(msg);

                if (!first_encoded.has_value()){
                    return std::unexpected(first_encoded.error());
                } 

                return this->second_encoder->encode(first_encoded.value());
            }

            auto decode(const dg::network_std_container::string& msg) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

                auto second_decoded = this->second_encoder->decode(msg);

                if (!second_decoded.has_value()){
                    return std::unexpected(second_decoded.error());
                }

                return this->first_encoder->decode(second_decoded.value());
            }
    };

    class TokenController: public virtual TokenControllerInterface{

        private:

            std::unique_ptr<EncoderInterface> encoder;
            std::chrono::nanoseconds token_expiry_interval;
            std::chrono::nanoseconds refresh_expiry_interval;
        
        public:

            TokenController(std::unique_ptr<EncoderInterface> encoder,
                            std::chrono::nanoseconds token_expiry_interval,
                            std::chrono::nanoseconds refresh_expiry_interval) noexcept: encoder(std::move(encoder)),
                                                                                        token_expiry_interval(token_expiry_interval),
                                                                                        refresh_expiry_interval(refresh_expiry_interval){}
                                                                                                         
            auto tokenize(const dg::network_std_container::string& content) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

                std::chrono::nanoseconds now                = dg::network_genult::utc_timestamp();
                std::chrono::nanoseconds token_expiry       = now + this->token_expiry_interval;
                std::chrono::nanoseconds refresh_expiry     = now + this->refresh_expiry_interval;
                Token token{token_expiry, refresh_expiry, content};
                
                return this->encoder->encode(this->token_serialize(token));
            }

            auto detokenize(const dg::network_std_container::string& token) noexcept -> std::expected<dg::network_std_container::string, exception_t>{
                
                std::expected<dg::network_std_container::string, exception_t> decoded = this->encoder->decode(token);

                if (!decoded.has_value()){
                    return std::unexpected(decoded.error());
                }

                std::expected<Token, exception_t> deserialized = this->token_deserialize(decoded.value());

                if (!deserialized.has_value()){
                    return std::unexpected(deserialized.error());
                }

                std::chrono::nanoseconds now = dg::network_genult::utc_timestamp();

                if (token.expiry_time < now){
                    return std::unexpected(dg::network_exception::EXPIRED_TOKEN);
                }

                return {std::in_place_t{}, std::move(deserialized.value())}; 
            } 

            auto renew_token(const dg::network_std_container::string& token) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

                std::expected<dg::network_std_container::string, exception_t> decoded = this->encoder->decode(token);

                if (!decoded.has_value()){
                    return std::unexpected(decoded.error());
                }

                std::expected<Token, exception_t> deserialized = this->token_deserialize(decoded.value());

                if (!deserialized.has_value()){
                    return std::unexpected(deserialized.error());
                }
                
                Token token = std::move(deserialized.value());
                std::chrono::nanoseconds now = dg::network_genult::utc_timestamp();

                if (token.refresh_time < now){
                    return std::unexpected(dg::network_exception::EXPIRED_TOKEN);
                }

                token.expiry_time = now + this->token_expiry_interval;
                return this->encoder->encode(this->token_serialize(token));
            }
        
        private:

            auto token_serialize(const Token& token) noexcept -> dg::network_std_container::string{

                size_t token_sz = dg::network_compact_serializer::integrity_size(token);
                auto bstream    = dg::network_std_container::string(token_sz);
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), token);

                return bstream;
            }

            auto token_deserialize(const dg::network_std_container::string& bstream) noexcept -> std::expected<Token, exception_t>{

                Token token{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::deserialize_into<Token>)(token, bstream.data(), bstream.size());

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                } 

                return {std::in_place_t{}, std::move(token)};
            }
    };
} 

namespace dg::network_auth_base{

    static inline constexpr size_t MIN_PWD_SIZE     = 6;
    static inline constexpr size_t MAX_PWD_SIZE     = 32; 

    using user_id_t = std::string; 

    void init(){

    }

    void deinit() noexcept{

    }

    auto account_register_legacy(const dg::network_std_container::string& pwd, User user) noexcept -> exception_t{

        std::expected<UserEntry, exception_t> user = dg::network_postgres_db::get_entry_user_by_externalid(user.id); 

        if (user.has_value()){
            return dg::network_exception::BAD_USER_REGISTRATION;
        }
        
        if (std::clamp(pwd.size(), MIN_PWD_SIZE, MAX_PWD_SIZE) != pwd.size()){
            return dg::network_exception::INVALID_ARGUMENT;
        }

        dg::network_std_container::string salt = dg::network_randomizer::randomize_string(SALT_LENGTH);
        dg::network_std_container::string xpwd = pwd + salt;
        std::expected<dg::network_std_container::string, exception_t> encoded = resource.pw_encoder->encode(xpwd);

        if (!encoded.has_value()){
            return encoded.error();
        }

        auto auth           = LegacyAuth{encoded.value(), salt, user.id};
        auto user_commit_wo = dg::network_postgres_db::create_commit_task(dg::network_postgres_db::create_entry_user, user); 
        auto auth_commit_wo = dg::network_postgres_db::create_commit_task(dg::network_postgres_db::create_entry_legacy_auth, auth); 
        exception_t err     = dg::network_postgres_db::commit({std::move(user_commit_wo), std::move(auth_commit_wo)});
        
        return err;
    }

    auto account_deregister(const user_id_t& id) noexcept -> exception_t{

        return dg::network_postgres_db::delete_user_by_externalid(id);
    }

    auto account_login_by_legacyauth(const dg::network_std_container::string& username, const dg::network_std_container::string& pwd) noexcept -> std::expected<user_id_t, exception_t>{
        
        std::expected<LegacyAuthEntry, exception_t> auth = dg::network_postgres_db::get_entry_legacyauth_by_username(username);

        if (!auth.has_value()){
            return std::unexpected(auth.error());
        }
        
        dg::network_std_container::string expected_verifiable   = auth.value().verifiable;
        dg::network_std_container::string xpwd                  = pwd + auth.value().salt;
        dg::network_std_container::string verifiable            = resource.pw_encoder->encode(xpwd);

        if (verifiable != expected_verifiable){
            return std::unexpected(dg::network_exception::BAD_AUTH);
        }

        return auth.value().user_id;
    }

    auto account_authority(const user_id_t& id) noexcept -> std::expected<authority_kind_t, exception_t>{

        std::expected<UserEntry, exception_t> user = dg::network_postgres_db::get_entry_user_by_externalid(id);

        if (!user.has_value()){
            return std::unexpected(user.error());
        }

        return user.value().authority;
    }

    auto account_exists(const user_id_t& id) noexcept -> std::expected<bool, exception_t>{

        std::expected<UserEntry, exception_t> user = dg::network_postgres_db::get_entry_user_by_externalid(id);

        if (!user.has_value()){
            return std::unexpected(user.error());
        }

        return true;
    }
}

namespace dg::network_auth_x{
    
    using auth_kind_t = uint8_t;

    enum auth_option{
        usridpwd_shared_codec   = 0,
        usridpwd_default_codec  = 1
    };

    struct SecuredAuth{
        auth_kind_t auth_kind;
        dg::network_std_container::string auth_content;
    };

    struct AuthenticationXResource{
        std::unique_ptr<network_token::TokenControllerInterface> token_controller;
        std::unique_ptr<network_token::EncoderInterface> shared_encoder;
        std::unique_ptr<network_token::EncoderInterface> default_encoder;
    };
    
    inline AuthenticationXResource resource{}; 

    void init(){

    }

    void deinit() noexcept{

    }

    auto auth_make_usrpwd_payload_from_shared_codec(const dg::network_std_container::string& id, const dg::network_std_container::string& pwd) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

        auto idpwd_msg      = std::make_pair(id, pwd);
        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::size(idpwd_msg));
        dg::network_compact_serializer::serialize_into(bstream.data(), idpwd_msg);
        auto encoded        = resource.shared_encoder->encode(bstream);

        if (!encoded.has_value()){
            return std::unexpected(encoded.error());
        }

        auto secured_auth   = SecuredAuth{usridpwd_shared_codec, encoded.value()};
        auto bbstream       = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(secured_auth));
        dg::network_compact_serializer::integrity_serialize_into(bbstream.data(), secured_auth);

        return bbstream;
    }

    auto auth_make_usrpwd_payload_from_default_codec(const dg::network_std_container::string& id, const dg::network_std_container::string& pwd) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

        auto idpwd_msg      = std::make_pair(id, pwd);
        auto bstream        = dg::network_std_container::string(dg::network_compact_serializer::size(idpwd_msg));
        dg::network_compact_serializer::serialize_into(bstream.data(), idpwd_msg);
        auto encoded        = resource.default_encoder->encode(bstream);

        if (!encoded.has_value()){
            return std::unexpected(encoded.error());
        }

        auto secured_auth   = SecuredAuth{usridpwd_default_codec, encoded.value()};  
        auto bbstream       = dg::network_std_container::string(dg::network_compact_serializer::integrity_size(secured_auth));
        dg::network_compact_serializer::integrity_serialize_into(bbstream.data(), secured_auth);

        return bbstream;
    }

    auto auth_is_usrpwd_payload(const dg::network_std_container::string& payload) noexcept -> std::expected<bool, exception_t>{

        auto secured_auth   = SecuredAuth{};
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<SecuredAuth>)(secured_auth, payload.data(), payload.size());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        if (secured_auth.auth_kind == usridpwd_shared_codec){
            return true;
        }

        if (secured_auth.auth_kind == usridpwd_default_codec){
            return true;
        }

        return false;
    }

    auto auth_extract_usrpwd_from_payload(const dg::network_std_container::string& payload) noexcept -> std::expected<std::pair<dg::network_std_container::string, dg::network_std_container::string>, exception_t>{

        // std::expected<dg::network_std_container::string, exception_t> decoded = resource.shared_encoder->decode(encoded);

        // if (!decoded.has_value()){
        //     return std::unexpected(decoded.error());
        // }

        // std::pair<dg::network_std_container::string, dg::network_std_container::string> msg{};
        // dg::network_compact_serializer::deserialize_into(msg, decoded.value());

        // return msg;
    }

    auto auth_generate_token_from_usrpwd_payload(const dg::network_std_container::string& auth_payload) noexcept -> std::expected<dg::network_std_container::string, exception_t>{
 
        std::expected<std::pair<dg::network_std_container::string, dg::network_std_container::string>, exception_t> id_pwd = auth_extract_usrpwd_from_payload(auth_payload);

        if (!id_pwd.has_value()){
            return std::unexpected(id_pwd.error());
        }

        auto [id, pwd] = std::move(id_pwd.value());
        std::expected<user_id_t, exception_t> usr_id = dg::network_auth::account_login_by_legacyauth(id, pwd);

        if (!usr_id.has_value()){
            return std::unexpected(usr_id.error());
        }

        auto bstream = dg::network_std_container::string(dg::network_compact_serializer::size(usr_id.value()));
        dg::network_compact_serializer::serialize_into(bstream.data(), usr_id.value());

        return resource.token_controller->tokenize(bstream);
    }

    auto auth_generate_token_from_payload(const dg::network_std_container::string& auth_payload) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

        std::expected<bool, exception_t> is_usrpwd_dispatch = auth_is_usrpwd_payload(auth_payload);

        if (!is_usrpwd_dispatch.has_value()){
            return std::unexpected(is_usrpwd_dispatch.error());
        }

        if (is_usrpwd_dispatch.value()){
            return auth_generate_token_from_usrpwd_payload(auth_payload);
        }

        return std::unexpected(dg::network_exception::INVALID_SERIALIZATION_FORMAT);
    }

    auto auth_refresh_token(const dg::network_std_container::string& token) noexcept -> std::expected<dg::network_std_container::string, exception_t>{

        return resource.token_controller->renew_token(token);
    }

    auto auth_validate_token(const dg::network_std_container::string& token) noexcept -> exception_t{

        std::expected<dg::network_std_container::string, exception_t> serialized_user_id = resource.token_controller->detokenize(token);

        if (!serialized_user_id.has_value()){
            return serialized_user_id.error();
        }
        
        return dg::network_exception::SUCCESS;
    }

    auto auth_extract_userid_from_token(const dg::network_std_container::string& token) noexcept -> std::expected<user_id_t, exception_t>{

        std::expected<dg::network_std_container::string, exception_t> serialized_user_id = resource.token_controller->detokenize(token);

        if (!serialized_user_id.has_value()){
            return std::unexpected(serialized_user_id.error());
        }
        
        user_id_t user_id{};
        dg::network_compact_serializer::deserialize_into(user_id, serialized_user_id.value().data());

        return user_id;
    }
    
    auto account_deregister(const dg::network_std_container::string& auth_payload) noexcept -> exception_t{

    }
}

#endif