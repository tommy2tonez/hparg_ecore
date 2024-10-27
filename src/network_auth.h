#ifndef __DG_NETWORK_AUTH_H__
#define __DG_NETWORK_AUTH_H__

#include "network_std_container.h"
#include "stdx.h"
#include "network_postgres_db.h"
#include <chrono>
#include "network_exception.h"

namespace dg::network_auth_utility{

    struct Token{
        std::chrono::nanoseconds expiry_time;
        dg::string content;
    };

    struct EncoderInterface{
        virtual ~EncoderInterface() noexcept = default;
        virtual auto encode(const dg::string&) noexcept -> std::expected<dg::string, exception_t> = 0;
        virtual auto decode(const dg::string&) noexcept -> std::expected<dg::string, exception_t> = 0; 
    };

    struct TokenControllerInterface{
        virtual ~TokenControllerInterface() noexcept = default;
        virtual auto tokenize(const dg::string&, std::chrono::nanoseconds)  noexcept -> std::expected<dg::string, exception_t> = 0; //its look weird that encoder and tokenizer have the same interface
        virtual auto detokenize(const dg::string&) noexcept -> std::expected<dg::string, exception_t> = 0;
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
                auto bstream    = dg::string(dg::network_compact_serializer::integrity_size(msg), ' ');
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), msg);

                return bstream;
            }

            auto decode(const dg::string& arg) noexcept -> std::expected<dg::string, exception_t>{

                MurMurMessage msg{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::integrity_deserialize_into<MurMurMessage>)(msg, arg.data(), arg.size());

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                }

                uint64_t expected_key = dg::network_hash::murmur_hash(msg.encoded.data(), msg.encoded.size(), this->secret);
                
                if (expected_key != msg.validation_key){
                    return std::unexpected(dg::network_exception::BAD_ENCODING_FORMAT);
                }

                return msg.encoded;
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

            static inline constexpr size_t MIN_ENCODING_LENGTH  = size_t{0u};
            static inline constexpr size_t MAX_ENCODING_LENGTH  = size_t{1u} << 15; 
            static inline constexpr size_t MIN_DECODING_LENGTH  = size_t{0u};
            static inline constexpr size_t MAX_DECODING_LENGTH  = size_t{1u} << 16;

            Mt19937Encoder(dg::string secret,
                           mt19937 salt_randgen) noexcept: secret(std::move(secret)),
                                                           salt_randgen(std::move(salt_randgen)){}

            auto encode(const dg::string& arg) noexcept -> std::expected<dg::string, exception_t>{
                
                if (std::clamp(static_cast<size_t>(arg.size()), MIN_ENCODING_LENGTH, MAX_ENCODING_LENGTH) != arg.size()){
                    return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
                }

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
                
                if (std::clamp(static_cast<size_t>(arg.size()), MIN_DECODING_LENGTH, MAX_DECODING_LENGTH) != arg.size()){
                    return std::unexpected(dg::network_exception::BAD_ENCODING_FORMAT);
                }

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
        
        public:

            TokenController(std::unique_ptr<EncoderInterface> encoder) noexcept: encoder(std::move(encoder)){}
                                                                                                         
            auto tokenize(const dg::string& content, std::chrono::nanoseconds expiry_interval) noexcept -> std::expected<dg::string, exception_t>{

                std::chrono::nanoseconds now            = stdx::utc_timestamp();
                std::chrono::nanoseconds token_expiry   = now + expiry_interval;
                Token token                             = Token{token_expiry, content};
                
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

                std::chrono::nanoseconds now = stdx::utc_timestamp();

                if (deserialized.value().expiry_time < now){
                    return std::unexpected(dg::network_exception::EXPIRED_TOKEN);
                }

                return deserialized.value().content;
            } 

        private:

            auto token_serialize(const Token& token) noexcept -> dg::string{

                auto inter_rep  = std::make_tuple(static_cast<uint64_t>(token.expiry_time.count()), token.content);
                auto bstream    = dg::string(dg::network_compact_serializer::integrity_size(inter_rep), ' ');
                dg::network_compact_serializer::integrity_serialize_into(bstream.data(), inter_rep);

                return bstream;
            }

            auto token_deserialize(const dg::string& bstream) noexcept -> std::expected<Token, exception_t>{

                auto inter_rep  = std::tuple<uint64_t, dg::string>{};
                exception_t err = dg::network_exception::to_cstyle_function(dg::network_compact_serializer::deserialize_into<std::tuple<uint64_t, dg::string>>)(inter_rep, bstream.data(), bstream.size());

                if (dg::network_exception::is_failed(err)){
                    return std::unexpected(err);
                } 

                return Token{std::chrono::nanoseconds(std::get<0>(inter_rep)), std::get<1>(inter_rep)};
            }
    };

    struct Factory{

        static auto spawn_encoder(const dg::string& secret) -> std::unique_ptr<EncoderInterface>{

            const size_t MIN_SECRET_SIZE    = size_t{1} << 5;
            const size_t MAX_SECRET_SIZE    = size_t{1} << 20;

            if (std::clamp(static_cast<size_t>(secret.size()), MIN_SECRET_SIZE, MAX_SECRET_SIZE) != secret.size()){
                dg::network_exception::throw_exception(dg::network_exception::INVALID_ARGUMNET);
            }

            uint64_t numerical_secert                           = dg::network_hash::hash_bytes(secret.data(), secret.size());
            std::unique_ptr<EncoderInterface> base_encoder      = std::make_unique<MurMurEncoder>(numerical_secret);
            mt19937 rand_gen                                    = {};
            std::unique_ptr<EncoderInterface> mt_encoder        = std::make_unique<Mt19937Encoder>(secret, std::move(rand_gen));
            std::unique_ptr<EncoderInterface> combined_encoder  = std::make_unique<DoubleEncoder>(std::move(base_encoder), std::move(mt_encoder));

            return combined_encoder;
        }

        static auto spawn_token_controller(const dg::string& secret) -> std::unique_ptr<TokenControllerInterface>{
            
            std::unique_ptr<EncoderInterface> encoder       = spawn_encoder(secret); 
            std::unique_ptr<TokenControllerInterface> rs    = std::make_unique<TokenController>(std::move(encoder));

            return rs;
        }
    };
}

namespace dg::network_user{

    //precond validations are this component responsibility - give an overview of validations if model is unique_reference by the component

    struct Resource{
        std::unique_ptr<network_auth_utility::TokenControllerInterface> token_controller;
        std::unique_ptr<network_auth_utility::EncoderInterface> pw_encoder;
        std::unique_ptr<network_auth_utility::EncoderInterface> auth_shared_encoder;
        std::chrono::nanoseconds token_expiry;
    };

    inline std::unique_ptr<Resource> resource{};

    void init(const dg::string& private_database_secret, const dg::string& shared_secret, std::chrono::nanoseconds token_expiry){ //these are disk-presistent

        resource                        = std::make_unique<Resource>();
        resource->token_controller      = network_auth_utility::Factory::spawn_token_controller(shared_secret);
        resource->pw_encoder            = network_auth_utility::Factory::spawn_encoder(private_database_secret);
        resource->auth_shared_encoder   = network_auth_utility::Factory::spawn_encoder(shared_secret);
        resource->token_expiry          = token_expiry;
    }

    void deinit() noexcept{

        resource = nullptr;
    }

    auto user_register(const dg::string& user_id, const dg::string& pwd, const dg::string& clearance) noexcept -> exception_t{

        constexpr size_t SALT_FLEX_SZ                   = dg::network_postgres_db::model::LEGACYAUTH_SALT_MAX_LENGTH - dg::network_postgres_db_model::LEGACYAUTH_SALT_MIN_LENGTH;
        size_t salt_length                              = dg::network_postgres_db::model::LEGACYAUTH_SALT_MIN_LENGTH + dg::network_randomizer::randomize_range(SALT_FLEX_SZ);
        dg::string salt                                 = dg::network_randomizer::randomize_string<dg::string>(salt_length);
        dg::string xpwd                                 = pwd + salt;
        std::expected<dg::string, exception_t> encoded  = resource->pw_encoder->encode(xpwd);

        if (!encoded.has_value()){
            return encoded.error();
        }

        auto user = dg::network_postgres_db::model_factory::make_user(user_id, clearance, salt, encoded.value());

        if (!user.has_value()){
            return user.error();
        }

        auto user_commit_wo = dg::network_postgres_db::make_commitable_create_user(user.value()); 

        if (!user_commit_wo.has_value()){
            return user_commit_wo.error();
        }

        return dg::network_postgres_db::commit({std::move(user_commit_wo.value())});
    }

    auto user_deregister(const dg::string& user_id) noexcept -> exception_t{

        auto usrdel_commit_wo = dg::network_postgres_db::make_commitable_delete_user_by_id(user_id);

        if (!usrdel_commit_wo.has_value()){
            return usrdel_commit_wo.error();
        }

        return dg::network_postgres_db::commit({std::move(usrdel_commit_wo.value())});
    }

    auto user_login(const dg::string& user_id, const dg::string& pwd) noexcept -> std::expected<bool, exception_t>{
        
        std::expected<dg::network_postgres_db::model::UserEntry, exception_t> user = dg::network_postgres_db::get_user_by_id(user_id);

        if (!user.has_value()){
            return std::unexpected(user.error());
        }
        
        dg::string xpwd = pwd + user->salt;
        std::expected<dg::string, exception_t> verifiable = resource->pw_encoder->encode(xpwd);

        if (!verifiable.has_value()){
            return std::unexpected(verifiable.error());
        }

        return verifiable.value() == user->verifiable;
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

        return resource->auth_shared_encoder->encode(bstream);
    }

    auto auth_deserialize(const dg::string& payload) noexcept -> std::expected<std::pair<dg::string, dg::string>, exception_t>{

        std::expected<dg::string, exception_t> decoded = resource->auth_shared_encoder->decode(payload);

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

        return resource->token_controller->tokenize(id, resource->token_expiry);
    }

    auto token_extract_userid(const dg::string& token) noexcept -> std::expected<dg::string, exception_t>{

        return resource->token_controller->detokenize(token);
    }
}

#endif