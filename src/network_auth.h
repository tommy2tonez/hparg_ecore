#ifndef __DG_NETWORK_AUTH_H__
#define __DG_NETWORK_AUTH_H__

#include "network_std_container.h"

namespace dg::network_auth_utility{

    struct Token{
        std::chrono::nanoseconds expiry_time;
        std::chrono::nanoseconds refresh_expiry_time;
        dg::string content;

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
        virtual auto encode(const dg::string&) noexcept -> std::expected<dg::string, exception_t> = 0;
        virtual auto decode(const dg::string&) noexcept -> std::expected<dg::string, exception_t> = 0; 
    };

    struct TokenControllerInterface{
        virtual ~TokenControllerInterface() noexcept = default;
        virtual auto tokenize(const dg::string&) noexcept -> std::expected<dg::string, exception_t> = 0;
        virtual auto detokenize(const dg::string&) noexcept -> std::expected<dg::string, exception_t> = 0;
        virtual auto renew_token(const dg::string&) noexcept -> std::expected<dg::string, exception_t> = 0;
    };

    struct MurMurMessage{
        uint64_t validation_key;
        dg::string encoded;

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

            auto encode(const dg::string& arg) noexcept -> std::expected<dg::string, exception_t>{

                uint64_t key    = dg::network_hash::murmur_hash(arg.data(), arg.size(), this->secret);
                auto msg        = MurMurMessage{key, arg};
                auto bstream    = dg::string(dg::network_compact_serializer::integrity_size(msg));
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), msg);

                return bstream; //std::move
            }

            auto decode(const dg::string& arg) noexcept -> std::expected<dg::string, exception_t>{

                MurMurMessage msg{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<MurMurMessage>)(msg, arg.data(), arg.size());

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                }

                uint64_t expected_key = dg::network_hash::murmur_hash(msg.encoded.data(), msg.encoded.size(), this->secret);
                
                if (expected_key != msg.validation_key){
                    return std::unexpected(dg::network_exception::BAD_DECODE);
                }

                return msg.encoded; //std::move
            }
    };
    
    struct Mt19937Message{
        uint64_t salt;
        dg::string encoded;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(salt, encoded);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(salt, encoded);
        }
    };

    using mt19937 = std::mersenne_twister_engine<uint64_t, 64, 312, 156, 31,
                                                 0xb5026f5aa96619e9ULL, 29,
                                                 0x5555555555555555ULL, 17,
                                                 0x71d67fffeda60000ULL, 37,
                                                 0xfff7eee000000000ULL, 43,
                                                 6364136223846793005ULL>;

    class Mt19937Encoder: public virtual EncoderInterface{

        private:

            dg::string secret;
            mt19937 salt_randgen;
            
        public:


            Mt19937Encoder(dg::string secret,
                           mt19937 salt_randgen) noexcept: secret(std::move(secret)),
                                                           salt_randgen(std::move(salt_randgen)){}

            auto encode(const dg::string& arg) noexcept -> std::expected<dg::string, exception_t>{

                uint64_t salt       = this->salt_randgen();
                uint64_t seed       = dg::network_hash::murmur_hash(this->secret.data(), this->secret.size(), salt);
                auto randomizer     = mt19937{seed};
                auto encoded        = dg::string(arg.size(), ' ');

                for (size_t i = 0u; i < arg.size(); ++i){
                    encoded[i] = this->byte_encode(arg[i], randomizer);
                }

                return this->serialize(Mt19937Message{salt, std::move(encoded)});
            }

            auto decode(const dg::string& arg) noexcept -> std::expected<dg::string, exception_t>{

                std::expected<Mt19937Message, exception_t> msg = this->deserialize(arg);

                if (!msg.has_value()){
                    return std::unexpected(msg.error());
                }

                uint64_t seed       = dg::network_hash::murmur_hash(this->secret.data(), this->secret.size(), msg->salt);
                auto randomizer     = mt19937{seed};
                auto decoded        = dg::string(msg->encoded.size(), ' ');

                for (size_t i = 0u; i < msg->encoded.size(); ++i){
                    decoded[i] = this->byte_decode(msg->encoded[i], randomizer);
                }

                return decoded;
            }
        
        private:

            template <class Randomizer>
            auto get_byte_dict(Randomizer& randomizer) noexcept -> dg::vector<uint8_t>{
                
                dg::vector<uint8_t> rs(256);
                std::iota(rs.begin(), rs.end(), 0u);

                for (size_t i = 0u; i < 256; ++i){
                    size_t lhs_idx = static_cast<size_t>(randomizer()) % 256;
                    size_t rhs_idx = static_cast<size_t>(randomizer()) % 256;
                    std::swap(rs[lhs_idx], rs[rhs_idx]);
                }

                return rs;
            }

            template <class Randomizer>
            auto byte_encode(char key, Randomizer& randomizer) noexcept -> char{

                dg::vector<uint8_t> dict = get_byte_dict(randomizer);
                return std::bit_cast<char>(dict[std::bit_cast<uint8_t>(key)]);
            }

            template <class Randomizer>
            auto byte_decode(char value, Randomizer& randomizer) noexcept -> char{
                
                dg::vector<uint8_t> dict = get_byte_dict(randomizer);
                uint8_t key = std::distance(dict.begin(), std::find(dict.begin(), dict.end(), std::bit_cast<uint8_t>(value)));

                return std::bit_cast<char>(key);
            }

            auto serialize(const Mt19937Message& msg) noexcept -> dg::string{

                size_t len = dg::network_trivial_serializer::size(uint64_t{}) + msg.encoded.size(); 
                dg::string rs(len, ' ');
                char * last = dg::network_trivial_serializer::serialize_into(rs.data(), msg.salt);
                std::copy(msg.encoded.begin(), msg.encoded.end(), last);

                return rs;
            }

            auto deserialize(const dg::string& bstream) noexcept -> std::expected<Mt19937Message, exception_t>{
                
                if (bstream.size() < dg::network_trivial_serializer::size(uint64_t{})){
                    return std::unexpected(dg::network_exception::BAD_ENCODING_FORMAT);
                }

                Mt19937Message rs   = {};
                const char * last   = dg::network_trivial_serializer::deserialize_into(rs.salt, bstream.data()); 
                std::copy(last, bstream.data() + bstream.size(), std::back_inserter(rs.encoded));

                return rs;
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
            
            auto encode(const dg::string& msg) noexcept -> std::expected<dg::string, exception_t>{
                
                auto first_encoded = this->first_encoder->encode(msg);

                if (!first_encoded.has_value()){
                    return std::unexpected(first_encoded.error());
                } 

                return this->second_encoder->encode(first_encoded.value());
            }

            auto decode(const dg::string& msg) noexcept -> std::expected<dg::string, exception_t>{

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
                                                                                                         
            auto tokenize(const dg::string& content) noexcept -> std::expected<dg::string, exception_t>{

                std::chrono::nanoseconds now                = dg::network_genult::utc_timestamp();
                std::chrono::nanoseconds token_expiry       = now + this->token_expiry_interval;
                std::chrono::nanoseconds refresh_expiry     = now + this->refresh_expiry_interval;
                Token token{token_expiry, refresh_expiry, content};
                
                return this->encoder->encode(this->token_serialize(token));
            }

            auto detokenize(const dg::string& token) noexcept -> std::expected<dg::string, exception_t>{
                
                std::expected<dg::string, exception_t> decoded = this->encoder->decode(token);

                if (!decoded.has_value()){
                    return std::unexpected(decoded.error());
                }

                std::expected<Token, exception_t> deserialized = this->token_deserialize(decoded.value());

                if (!deserialized.has_value()){
                    return std::unexpected(deserialized.error());
                }

                std::chrono::nanoseconds now = dg::network_genult::utc_timestamp();

                if (deserialized.value().expiry_time < now){
                    return std::unexpected(dg::network_exception::EXPIRED_TOKEN);
                }

                return deserialized.value().content; 
            } 

            auto renew_token(const dg::string& token) noexcept -> std::expected<dg::string, exception_t>{

                std::expected<dg::string, exception_t> decoded = this->encoder->decode(token);

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

            auto token_serialize(const Token& token) noexcept -> dg::string{

                auto bstream = dg::string(dg::network_compact_serializer::integrity_size(token));
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), token);

                return bstream;
            }

            auto token_deserialize(const dg::string& bstream) noexcept -> std::expected<Token, exception_t>{

                Token token{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::deserialize_into<Token>)(token, bstream.data(), bstream.size());

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                } 

                return token;
            }
    };
}

namespace dg::network_user{

    struct UserBaseResource{
        std::unique_ptr<network_auth_utility::TokenControllerInterface> token_controller;
        std::unique_ptr<network_auth_utility::EncoderInterface> pw_encoder;
        std::unique_ptr<network_auth_utility::EncoderInterface> auth_shared_encoder;
    };

    inline UserBaseResource resource{};

    void init(){

    }

    void deinit() noexcept{

    }

    auto user_register(const dg::string& user_id, const dg::string& pwd, const dg::string& clearance) noexcept -> exception_t{

        constexpr size_t SALT_FLEX_SZ           = dg::network_postgres_db::model::LEGACYAUTH_SALT_MAX_LENGTH - dg::network_postgres_db_model::LEGACYAUTH_SALT_MIN_LENGTH;
        size_t salt_length                      = dg::network_postgres_db::model::LEGACYAUTH_SALT_MIN_LENGTH + dg::network_randomizer::randomize_range(SALT_FLEX_SZ);
        dg::string salt  = dg::network_randomizer::randomize_string(salt_length);
        dg::string xpwd  = pwd + salt;
        std::expected<dg::string, exception_t> encoded = resource.pw_encoder->encode(xpwd);

        if (!encoded.has_value()){
            return encoded.error();
        }

        auto auth = dg::network_postgres_db::model_factory::make_legacy_auth{salt, encoded.value(), user_id};

        if (!auth.has_value()){
            return auth.error();
        }

        auto user = dg::network_postgres_db::model_factory::make_user(user_id, clearance);

        if (!user.has_value()){
            return user.error();
        }

        auto user_commit_wo = dg::network_postgres_db::make_commitable_create_user(user.value()); 

        if (!user_commit_wo.has_value()){
            return user_commit_wo.error();
        } 

        auto auth_commit_wo = dg::network_postgres_db::make_commitable_create_legacy_auth(auth.value());

        if (!auth_commit_wo.has_value()){
            return auth_commit_wo.error();
        }

        return dg::network_postgres_db::commit({std::move(user_commit_wo.value()), std::move(auth_commit_wo.value())});
    }

    auto user_deregister(const dg::string& user_id) noexcept -> exception_t{

        auto usrdel_commit_wo = dg::network_postgres_db::make_commitable_delete_user_by_id(user_id); //this semantically does not imply that delete the authentication registered with the user - this of course could be enforced at the model design yet it's semantically incorrect 

        if (!usrdel_commit_wo.has_value()){
            return usrdel_commit_wo.error();
        }

        return dg::network_postgres_db::commit({std::move(usrdel_commit_wo.value())});
    }

    auto user_login(const dg::string& user_id, const dg::string& pwd) noexcept -> std::expected<bool, exception_t>{
        
        std::expected<dg::network_postgres_db::model::LegacyAuthEntry, exception_t> auth = dg::network_postgres_db::get_legacyauth_by_userid(user_id);

        if (!auth.has_value()){
            return std::unexpected(auth.error());
        }
        
        dg::string xpwd = pwd + auth->salt;
        std::expected<dg::string, exception_t> verifiable = resource.pw_encoder->encode(xpwd);

        if (!verifiable.has_value()){
            return std::unexpected(verifiable.error());
        }

        return verifiable.value() == auth->verifiable;
    }

    auto user_get_clearance(const dg::string& user_id) noexcept -> std::expected<dg::string, exception_t>{

        std::expected<dg::network_postgres_db::model::UserEntry, exception_t> user = dg::network_postgres_db::get_user_by_id(user_id);

        if (!user.has_value()){
            return std::unexpected(user.error());
        }

        return user->clearance;
    }

    auto user_exists(const dg::string& user_id) noexcept -> std::expected<bool, exception_t>{

        std::expected<dg::network_postgres_db::model::UserEntry, exception_t> user = dg::network_postgres_db::get_user_by_id(user_id);

        if (!user.has_value()){
            if (user.error() == dg::network_exception::RECORD_NOT_FOUND){
                return false;
            }

            return user.error();
        }

        return true;
    }

    auto auth_serialize(const dg::string& id, const dg::string& pwd) noexcept -> std::expected<dg::string, exception_t>{

        auto idpwd_msg      = std::make_pair(id, pwd);
        auto bstream        = dg::string(dg::network_compact_serializer::size(idpwd_msg), ' ');
        dg::network_compact_serializer::serialize_into(bstream.data(), idpwd_msg);

        return resource.auth_shared_encoder->encode(bstream);
    }

    auto auth_deserialize(const dg::string& payload) noexcept -> std::expected<std::pair<dg::string, dg::string>, exception_t>{

        std::expected<dg::string, exception_t> decoded = resource.auth_shared_encoder->decode(payload);

        if (!decoded.has_value()){
            return std::unexpected(decoded.error());
        }

        auto idpwd_pair     = std::pair<dg::string, dg::string>();
        exception_t err     = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::deserialize_into<decltype(idpwd_pair)>)(idpwd_pair, decoded->data());

        if (dg::network_exception::is_failed(err)){
            return std::unexpected(err);
        }

        return idpwd_pair;
    }

    auto token_generate_from_auth_payload(const dg::string& payload) noexcept -> std::expected<dg::string, exception_t>{
 
        std::expected<std::pair<dg::string, dg::string>, exception_t> id_pwd = auth_deserialize(payload);

        if (!id_pwd.has_value()){
            return std::unexpected(id_pwd.error());
        }

        auto [id, pwd] = std::move(id_pwd.value());
        std::expected<bool, exception_t> status = user_login(id, pwd);

        if (!status.has_value()){
            return std::unexpected(status.error());
        }
        
        if (!status.value()){
            return std::unexpected(dg::network_exception::BAD_AUTHENTICATION);
        }

        auto bstream = dg::string(dg::network_compact_serializer::size(id.value()), ' ');
        dg::network_compact_serializer::serialize_into(bstream.data(), id.value());

        return resource.token_controller->tokenize(bstream);
    }

    auto token_refresh(const dg::string& token) noexcept -> std::expected<dg::string, exception_t>{

        return resource.token_controller->renew_token(token);
    }

    auto token_extract_userid(const dg::string& token) noexcept -> std::expected<dg::string, exception_t>{

        return resource.token_controller->detokenize(token);
    }
}

#endif