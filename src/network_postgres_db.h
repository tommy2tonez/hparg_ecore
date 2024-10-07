#ifndef __DG_NETWORK_POSTGRES_DB_H__
#define __DG_NETWORK_POSTGRES_DB_H__

#include <mutex>
#include <memory>
#include <stdint.h>
#include <stdlib.h>
#include "network_trivial_serializer.h"
#include "network_std_container.h"
#include <chrono>

namespace dg::network_postgres_db::model{

    static inline constexpr size_t USER_ID_MIN_LENGTH                   = size_t{1};
    static inline constexpr size_t USER_ID_MAX_LENGTH                   = size_t{1} << 5;
    static inline constexpr size_t LEGACYAUTH_SALT_MIN_LENGTH           = size_t{1};
    static inline constexpr size_t LEGACYAUTH_SALT_MAX_LENGTH           = size_t{1} << 5; 
    static inline constexpr size_t LEGACYAUTH_VERIFIABLE_MIN_LENGTH     = size_t{1};
    static inline constexpr size_t LEGACYAUTH_VERIFIABLE_MAX_LENGTH     = size_t{1} << 5;
    static inline constexpr size_t LEGACYAUTH_USER_ID_MIN_LENGTH        = USER_ID_MIN_LENGTH;
    static inline constexpr size_t LEGACYAUTH_USER_ID_MAX_LENGTH        = USER_ID_MAX_LENGTH;
    static inline constexpr size_t HEARTBEAT_PAYLOAD_MIN_LENGTH         = size_t{1};
    static inline constexpr size_t HEARTBEAT_PAYLOAD_MAX_LENGTH         = size_t{1} << 5;
    static inline constexpr size_t SYSTEMLOG_CONTENT_MIN_LENGTH         = size_t{1};
    static inline constexpr size_t SYSTEMLOG_CONTENT_MAX_LENGTH         = size_t{1} << 8;
    static inline constexpr size_t SYSTEMLOG_KIND_MIN_LENGTH            = size_t{1};
    static inline constexpr size_t SYSTEMLOG_KIND_MAX_LENGTH            = size_t{1} << 8;
    static inline constexpr size_t USERLOG_CONTENT_MIN_LENGTH           = size_t{1};
    static inline constexpr size_t USERLOG_CONTENT_MAX_LENGTH           = size_t{1} << 5;
    static inline constexpr size_t USERLOG_KIND_MIN_LENGTH              = size_t{1};
    static inline constexpr size_t USERLOG_KIND_MAX_LENGTH              = size_t{1} << 5;
    static inline constexpr size_t USERLOG_USER_ID_MIN_LENGTH           = USER_ID_MIN_LENGTH;
    static inline constexpr size_t USERLOG_USER_ID_MAX_LENGTH           = USER_ID_MAX_LENGTH; 

    struct LegacyAuth{
        dg::network_std_container::string salt;
        dg::network_std_container::string verifiable;
        dg::network_std_container::string user_id;
    };

    struct LegacyAuthEntry: LegacyAuth{
        dg::network_std_container::string entry_id;
    };

    struct User{
        dg::network_std_container::string id;
        dg::network_std_container::string clearance;
    };

    struct UserEntry: User{
        dg::network_std_container::string entry_id;
    };

    struct HeartBeat{
        dg::network_std_container::string payload;
    };

    struct HeartBeatEntry: HeartBeat{
        dg::network_std_container::string entry_id;
    };

    struct SystemLog{
        dg::network_std_container::string content;
        dg::network_std_container::string kind;
        std::chrono::nanoseconds timestamp;
    };

    struct SystemLogEntry: SystemLog{
        dg::network_std_container::string entry_id;
    };

    struct UserLog{
        dg::network_std_container::string content;
        dg::network_std_container::string kind;
        dg::network_std_container::string user_id;
        std::chrono::nanoseconds timestamp;
    };

    struct UserLogEntry: UserLog{
        dg::network_std_container::string entry_id;
    };

    struct CommitableInterface{
        virtual ~CommitableInterface() noexcept = default;
        virtual auto commit(pqxx::work&) noexcept -> exception_t = 0;
    };

    auto make_legacy_auth(const dg::network_std_container::string& salt, const dg::network_std_container& verifiable, const dg::network_std_container::string& user_id) noexcept -> std::expected<LegacyAuth, exception_t>{

        if (std::clamp(salt.size(), LEGACYAUTH_SALT_MIN_LENGTH, LEGACYAUTH_SALT_MAX_LENGTH) != salt.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(verifiable.size(), LEGACYAUTH_VERIFIABLE_MIN_LENGTH, LEGACYAUTH_VERIFIABLE_MAX_LENGTH) != verifiable.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(user_id.size(), LEGACYAUTH_USER_ID_MIN_LENGTH, LEGACYAUTH_USER_ID_MAX_LENGTH) != user_id.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return LegacyAuth{salt, verifiable, user_id};
    }

    auto make_user(const dg::network_std_container::string& id, const dg::network_std_container::string& clearance) noexcept -> std::expected<User, exception_t>{

        if (std::clamp(id.size(), USER_ID_MIN_LENGTH, USER_ID_MAX_LENGTH) != id.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(clearance.size(), USER_CLEARANCE_MIN_LENGTH, USER_CLEARANCE_MAX_LENGTH) != clearance.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return User{id, clearance};
    }

    auto make_heartbeat(const dg::network_std_container::string& payload) noexcept -> std::expected<HeartBeat, exception_t>{

        if (std::clamp(payload.size(), HEARTBEAT_PAYLOAD_MIN_LENGTH, HEARTBEAT_PAYLOAD_MAX_LENGTH) != payload.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return HeartBeat{payload};
    }

    auto make_systemlog(const dg::network_std_container::string& content, const dg::network_std_container::string& kind, std::chrono::nanoseconds timestamp) noexcept -> std::expected<SystemLog, exception_t>{

        if (std::clamp(content.size(), SYSTEMLOG_CONTENT_MIN_LENGTH, SYSTEMLOG_CONTENT_MAX_LENGTH) != content.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(kind.size(), SYSTEMLOG_KIND_MIN_LENGTH, SYSTEMLOG_KIND_MAX_LENGTH) != kind.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return SystemLog{content, kind, timestamp};        
    }

    auto make_userlog(const dg::network_std_container::string& content, const dg::network_std_container::string& kind, const dg::network_std_container::string& user_id, std::chrono::nanoseconds timestamp) noexcept -> std::expected<UserLog, exception_t>{

        if (std::clamp(content.size(), USERLOG_CONTENT_MIN_LENGTH, USERLOG_CONTENT_MAX_LENGTH) != content.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(kind.size(), USERLOG_KIND_MIN_LENGTH, USERLOG_KIND_MAX_LENGTH) != kind.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }
        
        if (std::clamp(user_id.size(), USERLOG_USER_ID_MIN_LENGTH, USERLOG_USER_ID_MAX_LENGTH) != user_id.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return UserLog{content, kind, user_id, timestamp};
    }
}

namespace dg::network_postgres_db::constants{

    static inline constexpr size_t MAXIMUM_QUERY_LENGTH     = size_t{1} << 12;
    static inline constexpr size_t QUERY_LIMIT              = size_t{1} << 10;
}

namespace dg::network_postgres_db::hex_encoder{

    static inline std::vector<char> hex_fwd_dict{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    static inline std::vector<uint8_t> hex_bwd_dict = []{
        std::vector<uint8_t> rs(256);
        rs[std::bit_cast<uint8_t>(char('0'))] = 0;
        rs[std::bit_cast<uint8_t>(char('1'))] = 1;
        rs[std::bit_cast<uint8_t>(char('2'))] = 2;
        rs[std::bit_cast<uint8_t>(char('3'))] = 3;
        rs[std::bit_cast<uint8_t>(char('4'))] = 4;
        rs[std::bit_cast<uint8_t>(char('5'))] = 5;
        rs[std::bit_cast<uint8_t>(char('6'))] = 6;
        rs[std::bit_cast<uint8_t>(char('7'))] = 7;
        rs[std::bit_cast<uint8_t>(char('8'))] = 8;
        rs[std::bit_cast<uint8_t>(char('9'))] = 9;
        rs[std::bit_cast<uint8_t>(char('a'))] = 10;
        rs[std::bit_cast<uint8_t>(char('b'))] = 11;
        rs[std::bit_cast<uint8_t>(char('c'))] = 12;
        rs[std::bit_cast<uint8_t>(char('d'))] = 13;
        rs[std::bit_cast<uint8_t>(char('e'))] = 14;
        rs[std::bit_cast<uint8_t>(char('f'))] = 15;
        return rs;
    }();

    static auto hex_fr_char(char c) noexcept -> std::pair<char, char>{

        uint8_t uc  = std::bit_cast<uint8_t>(c);
        uint8_t low = uc & static_cast<uint8_t>(0b00001111);
        uint8_t hi  = uc >> 4;

        return {hex_fwd_dict[low], hex_fwd_dict[hi]};
    }

    static auto hex_to_char(char hex_low, char hex_hi) noexcept -> char{

        uint8_t low = hex_bwd_dict[std::bit_cast<uint8_t>(hex_low)];
        uint8_t hi  = hex_bwd_dict[std::bit_cast<uint8_t>(hex_hi)];
        uint8_t uc  = (hi << 4) | low;

        return std::bit_cast<char>(uc);
    }

    static auto encode(const dg::network_std_container::string& s) -> dg::network_std_container::string{

        dg::network_std_container::string rs{}; 
        rs.reserve(s.size() * 2);

        for (char c: s){
            auto [l, h] = hex_fr_char(c);
            rs.push_back(h);
            rs.push_back(l);
        }

        return rs;
    }

    static auto decode(const dg::network_std_container::string& s) -> dg::network_std_container::string{

        if (s.size() % 2 != 0u){
            dg::network_exception::throw_exception(dg::network_exception::BAD_ENCODING_FORMAT);
        }

        size_t rs_sz = s.size() / 2;
        dg::network_std_container::string rs(rs_sz);

        for (size_t i = 0u; i < rs_sz; ++i){
            size_t hidx = i * 2;
            size_t lidx = i * 2 + 1;
            rs[i]       = hex_to_char(s[lidx], s[hidx]); 
        }

        return rs;
    }
}

namespace dg::network_postgres_db::utility{

    template <class ...Args>
    static auto query_format(const char * fmt, Args&& ...args) -> dg::network_std_container::string{

        dg::network_std_container::string rs{};
        std::format_to(std::back_inserter(rs), fmt, std::forward<Args>(args)...);
        
        if (rs.size() > constants::MAXIMUM_QUERY_LENGTH){
            dg::network_exception::throw_exception(dg::network_exception::BAD_POSTGRES_QUERY_LENGTH);
        }

        return rs;
    } 

    static auto encode_sql(const dg::network_std_contaier::string& arg) -> dg::network_std_container::string{

        dg::network_std_container::string integrity_payload(dg::network_compact_serializer::integrity_size(arg));
        dg::network_compact_serializer::integrity_serialize_into(integrity_payload.data(), arg);
        
        return hex_encoder::encode(integrity_payload);
    }

    static auto decode_sql(const dg::network_std_container::string& arg) -> dg::network_std_container::string{

        dg::network_std_container::string integrity_payload = hex_encoder::decode(arg); 
        dg::network_std_container::string org_payload{};
        dg::network_compact_serializer::integrity_deserialize_into(org_payload, integrity_payload.data(), integrity_payload.size());

        return org_payload;
    }

    static auto encode_timestamp(std::chrono::nanoseconds timestamp) -> dg::network_std_container::string{

        static_assert((std::endian::native == std::endian::little || std::endian::native == std::endian::big));
        uint64_t num_rep = timestamp.count();

        if constexpr(std::endian::native == std::endian::little){
            num_rep = std::byteswap(num_rep);
        }

        dg::network_std_container::string rs(8u);
        std::memcpy(rs.data(), &num_rep, sizeof(uint64_t));

        return hex_encoder::encode(rs);
    } 

    static auto decode_timestamp(const dg::network_std_container::string& arg) -> std::chrono::nanoseconds{

        static_assert((std::endian::native == std::endian::little || std::endian::native == std::endian::big));
        dg::network_std_container::string encoded = hex_encoder::decode(arg);

        if (encoded.size() != 8u){
            dg::network_exception::throw_exception(dg::network_exception::BAD_ENCODING_FORMAT);
        }
        
        uint64_t num_rep{};
        std::memcpy(&num_rep, encoded.data(), sizeof(uint64_t));

        if constexpr(std::endian::native == std::endian::little){
            num_rep = std::byteswap(num_rep);
        }

        return std::chrono::nanoseconds(num_rep); 
    }

    static auto quote(const dg::network_std_container::string& arg) -> dg::network_std_container::string{

        const char * fmt = "\"{}\"";
        dg::network_std_container::string rs{};
        std::format_to(std::back_inserter(rs), fmt, arg);

        return rs;
    }
} 

namespace dg::network_postgres_db{

    inline std::unique_ptr<pqxx::connection> pq_conn; //-if performance problem arises - change -> atomic_shared_ptr or multiple instance approach 
    inline std::mutex mtx;

    template <class Lambda>
    class CommitableWrapper: public virtual model::CommitableInterface{

        private:

            Lambda lambda;
        
        public:

            static_assert(std::is_nothrow_destructible_v<Lambda>);
            
            CommitableWrapper(Lambda lambda) noexcept(std::is_nothrow_constructible_v<Lambda>): lambda(std::move(lambda)){}

            auto commit(pqxx::work& transaction_handle) noexcept -> exception_t{

                static_assert(std::is_same_v<exception_t, decltype(lambda(transaction_handle))>);
                static_assert(noexcept(lambda(transaction_handle)));

                return lambda(transaction_handle);
            }
    };

    void init(const dg::network_std_container::string& pq_conn_arg) noexcept -> exception_t{
        
        auto lck_grd = dg::network_genult::lock_guard(mtx);
        auto lambda = [&]{
            pq_conn = std::make_unique<pqxx::connection>(pq_conn_arg.c_str());
        };

        return dg::network_exception::to_cstyle_function(lambda)();
    }

    void deinit() noexcept{

        auto lck_grd = dg::network_genult::lock_guard(mtx);
        pq_conn = nullptr;
    }

    auto get_heartbeat() noexcept -> bool{
        
        auto lck_grd = dg::network_genult::lock_guard(mtx);

        if (!pq_conn){
            return false;
        }

        auto lambda = [&]{
            pqxx::work transaction_handle{*pq_conn};
            dg::network_std_container::string heartbeat_payload = dg::network_randomizer::randomize_string(model::HEARTBEAT_PAYLOAD_MAX_LENGTH);
            dg::network_std_container::string encoded_payload   = utility::quote(utility::encode_sql(heartbeat_payload));
            dg::network_std_container::string inject_query      = utility::query_format("INSERT INTO HeartBeat(payload) VALUES({})", encoded_payload);

            transaction_handle.exec(inject_query.c_str()).no_rows();
            transaction_handle.commit();
            
            dg::network_std_container::string get_query = utility::query_format("SELECT * FROM HeartBeat WHERE HeartBeat.payload = {}", encoded_payload);
            auto rs = transaction_handle.exec(get_query.c_str());
            rs.one_row();
            rs.for_each([&](const dg::network_std_container::string& entry_id, const dg::network_std_container::string& encoded_payload){
                if (heartbeat_payload != utility::decode_sql(encoded_payload)){
                    dg::network_exception::throw_exception(dg::network_exception::INTERNAL_CORRUPTION);
                }
            });

            dg::network_std_container::string del_query = utility::query_format("DELETE FROM HeartBeat WHERE HeartBeat.payload = {}", encoded_payload);
            transaction_handle.exec(del_query.c_str()).no_rows();
            transaction_handle.commit();
            transaction_handle.exec(get_query.c_str()).no_rows();
        };

        exception_t err = dg::network_exception::to_cstyle_function(lambda)();
        return dg::network_exception::is_success(err); //need to be more descriptive + handle internal corruption - internal corruption could leak -
    }

    auto get_user_by_id(const dg::network_std_container::string& id) noexcept -> std::expected<model::UserEntry, exception_t>{

        auto lck_grd = dg::network_genult::lock_guard(mtx);

        if (!pq_conn){
            return std::unexpected(dg::network_exception::POSTGRES_NOT_INITIALIZED);
        }

        if (std::clamp(id.size(), model::USER_ID_MIN_LENGTH, model::USER_ID_MAX_LENGTH) != id.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        auto lambda = [&]{
            pqxx::nontransaction transaction_handle{*pq_conn};
            dg::network_std_container::string query = utility::query_format("SELECT * FROM User WHERE User.id = {}", utility::quote(utility::encode_sql(id)));
            model::UserEntry user{};
            auto rs = transaction_handle.exec(query.c_str());
            rs.one_row();
            rs.for_each([&](const dg::network_std_container::string& entry_id, const dg::network_std_container::string& id, const dg::network_std_container::string& clearance){
                user.entry_id    = entry_id;
                user.id          = utility::decode_sql(id);
                user.clearance   = utility::decode_sql(clearance);
            });

            return user;
        };

        return dg::network_exception::to_cstyle_function(lambda)();
    }

    auto get_legacyauth_by_userid(const dg::network_std_container::string& user_id) noexcept -> std::expected<model::LegacyAuthEntry, exception_t>{

        auto lck_grd = dg::network_genult::lock_guard(mtx);

        if (!pq_conn){
            return std::unexpected(dg::network_exception::POSTGRES_NOT_INITIALIZED);
        }

        if (std::clamp(user_id.size(), model::LEGACYAUTH_USER_ID_MIN_LENGTH, model::LEGACYAUTH_USER_ID_MAX_LENGTH) != user_id.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        auto lambda = [&]{
            pqxx::nontransaction transaction_handle{*pq_conn};
            dg::network_std_container::string query = utility::query_format("SELECT * FROM LegacyAuth WHERE LegacyAuth.user_id = {}", utility::quote(utility::encode_sql(user_id)));
            model::LegacyAuthEntry auth{};
            auto rs = transaction_handle.exec(query.c_str());
            rs.one_row();
            rs.for_each([&](const dg::network_std_container::string& entry_id, const dg::network_std_container::string& salt, const dg::network_std::container::string& verifiable, const dg::network_std_container::string& user_id){
                auth.entry_id   = entry_id;
                auth.salt       = utility::decode_sql(salt);
                auth.verifiable = utility::decode_sql(verifiable);
                auth.user_id    = utility::decode_sql(user_id);
            });

            return auth;
        };

        return dg::network_exception::to_cstyle_function(lambda)();
    }
    
    auto get_systemlog(const dg::network_std_container::string& kind, std::chrono::nanoseconds fr, std::chrono::nanoseconds to, size_t limit) noexcept -> std::expected<dg::network_std_container::vector<model::SystemLogEntry>, exception_t>{

        auto lck_grd = dg::network_genult::lock_guard(mtx);

        if (!pq_conn){
            return std::unexpected(dg::network_exception::POSTGRES_NOT_INITIALIZED);
        }

        if (std::clamp(kind.size(), model::SYSTEMLOG_KIND_MIN_LENGTH, model::SYSTEMLOG_KIND_MAX_LENGTH) != kind.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (limit > constants::QUERY_LIMIT){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        auto lambda = [&]{
            pqxx::nontransaction transaction_handle{*pq_conn};
            auto query      = utility::query_format("SELECT * FROM SystemLog
                                                     WHERE SystemLog.kind = {} 
                                                           AND SystemLog.timestamp > {} AND SystemLog.timestamp < {}
                                                     LIMIT {}", utility::quote(utility::encode_sql(kind)), utility::quote(utility::encode_timestamp(fr)), utility::quote(utility::encode_timestamp(to)), limit);

            auto log_vec    = dg::network_std_container::vector<model::SystemLogEntry>{};
            auto rs         = transaction_handle.exec(query.c_str());
            rs.for_each([&](const dg::network_std_container::string& entry_id, const dg::network_std_container::string& content, const dg::network_std_container::string& kind, const dg::network_std_container::string& timestamp){
                model::SystemLogEntry entry{};
                entry.entry_id  = entry_id;
                entry.content   = utility::decode_sql(content);
                entry.kind      = utility::decode_sql(kind);
                entry.timestamp = utility::decode_timestamp(timestamp);
                log_vec.push_back(std::move(entry));
            });

            return log_vec;
        };

        return dg::network_exception::to_cstyle_function(lambda)();
    }

    auto get_userlog(const dg::network_std_container::string& user_id, const dg::network_std_container::string& kind, std::chrono::nanoseconds fr, std::chrono::nanoseconds to, size_t limit) noexcept -> std::expected<dg::network_std_container::vector<model::UserLogEntry>, exception_t>{

        auto lck_grd = dg::network_genult::lock_guard(mtx);

        if (!pq_conn){
            return std::unexpected(dg::network_exception::POSTGRES_NOT_INITIALIZED);
        }

        if (std::clamp(user_id.size(), model::USERLOG_USER_ID_MIN_LENGTH, model::USERLOG_USER_ID_MAX_LENGTH) != user_id.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(kind.size(), model::USERLOG_KIND_MIN_LENGTH, model::USERLOG_KIND_MAX_LENGTH) != kind.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (limit > constants::QUERY_LIMIT){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        auto lambda = [&]{
            pqxx::nontransaction transaction_handle{*pq_conn};
            auto query      = utility::query_format("SELECT * From UserLog 
                                                     WHERE UserLog.user_id = {} 
                                                           AND UserLog.kind = {} 
                                                           AND UserLog.timestamp > {} AND UserLog.timestamp < {}
                                                     LIMIT {}", utility::quote(utility::encode_sql(user_id)), utility::quote(utility::encode_sql(kind)), 
                                                                utility::quote(utility::encode_timestamp(fr)), utility::quote(utility::encode_timestamp(to)),
                                                                limit);
            auto log_vec    = dg::network_std_container::vector<model::UserLogEntry>{};
            auto rs         = transaction_handle.exec(query.c_str());
            rs.for_each([&](const dg::network_std_container::string& entry_id, const dg::network_std_container::string& content, const dg::network_std_container::string& kind, const dg::network_std_container::string& user_id, const dg::network_std_container::string& timestamp){
                model::UserLogEntry entry{};
                entry.entry_id  = entry_id;
                entry.content   = utility::decode_sql(content);
                entry.kind      = utility::decode_sql(kind);
                entry.user_id   = utility::decode_sql(user_id);
                entry.timestamp = utility::decode_timestamp(timestamp);
                log_vec.push_back(std::move(entry));
            })

            return log_vec;
        };

        return dg::network_exception::to_cstyle_function(lambda)();
    }

    auto make_commitable_create_systemlog(const model::SystemLog& log) noexcept -> std::expected<std::unique_ptr<model::CommitableInterface>, exception_t>{

        auto lambda = [=](pqxx::work& transaction_handle){
            dg::network_std_container::string query = utility::query_format("INSERT INTO SystemLog(content, kind, timestamp) VALUES({}, {}, {})", utility::quote(utility::encode_sql(log.content)), 
                                                                                                                                                  utility::quote(utility::encode_sql(log.kind)), 
                                                                                                                                                  utility::quote(utility::encode_timestamp(log.timestamp)));
            transaction_handle.exec(query.c_str()).no_rows();
        };

        auto func = dg::network_exception::to_cstyle_function(std::move(lambda));
        return {std::in_place_t{}, std::make_unique<CommitableWrapper<decltype(func)>>(std::move(func))};
    };

    auto make_commitable_create_userlog(const model::UserLog& log) noexcept -> std::expected<std::unique_ptr<model::CommitableInterface>, exception_t>{

        auto lambda = [=](pqxx::work& transaction_handle){
            dg::network_std_container::string query = utility::query_format("INSERT INTO UserLog(content, kind, user_id, timestamp) VALUES({}, {}, {}, {})", utility::quote(utility::encode_sql(log.content)),
                                                                                                                                                             utility::quote(utility::encode_sql(log.kind)),
                                                                                                                                                             utility::quote(utility::encode_sql(log.user_id)),
                                                                                                                                                             utility::quote(utility::encode_timestamp(log.timestamp)));
            transaction_handle.exec(query.c_str()).no_rows();
        };

        auto func = dg::network_exception::to_cstyle_function(std::move(lambda));
        return {std::in_place_t{}, std::make_unique<CommitableWrapper<decltype(func)>>(std::move(func))};
    }

    auto make_commitable_create_user(const model::User& user) noexcept -> std::expected<std::unique_ptr<model::CommitableInterface>, exception_t>{

        auto lambda = [=](pqxx::work& transaction_handle){
            dg::network_std_container::string query = utility::query_format("INSERT INTO User(id, clearance) VALUES({}, {})", utility::quote(utility::encode_sql(user.id)), 
                                                                                                                              utility::quote(utility::encode_sql(user.clearance)));
            transaction_handle.exec(query.c_str()).no_rows();
        };

        auto func = dg::network_exception::to_cstyle_function(std::move(lambda));
        return {std::in_place_t{}, std::make_unique<CommitableWrapper<decltype(func)>>(std::move(func))};
    }

    auto make_commitable_create_legacy_auth(const model::LegacyAuth& legacy_auth) noexcept -> std::expected<std::unique_ptr<model::CommitableInterface>, exception_t>{

        auto lambda = [=](pqxx::work& transaction_handle){
            dg::network_std_container::string query = utility::query_format("INSERT INTO LegacyAuth(salt, verifiable, user_id) VALUES({}, {}, {})", utility::quote(utility::encode_sql(legacy_auth.salt)), 
                                                                                                                                                    utility::quote(utility::encode_sql(legacy_auth.verifiable)), 
                                                                                                                                                    utility::quote(utility::encode_sql(legacy_auth.user_id)));
            transaction_handle.exec(query.c_str()).no_rows();
        };

        auto func = dg::network_exception::to_cstyle_function(std::move(lambda));
        return {std::in_place_t{}, std::make_unique<CommitableWrapper<decltype(func)>>(std::move(func))};
    }

    auto make_commitable_delete_user_by_id(const dg::network_std_container::string& id) noexcept -> std::expected<std::unique_ptr<model::CommitableInterface>, exception_t>{

        if (std::clamp(id.size(), model::USER_ID_MIN_LENGTH, model::USER_ID_MAX_LENGTH) != id.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        auto lambda = [=](pqxx::work& transaction_handle){
            dg::network_std_container::string query = utility::query_format("DELETE FROM User WHERE User.id = {}", utility::quote(utility::encode_sql(id)));
            transaction_handle.exec(query.c_str()).no_rows();
        };

        auto func = dg::network_exception::to_cstyle_function(std::move(lambda));
        return {std::in_place_t{}, std::make_unique<CommitableWrapper<decltype(func)>>(std::move(func))};
    }

    void commit(dg::network_std_container::vector<std::unique_ptr<model::CommitableInterface>> commitables) noexcept -> exception_t{

        auto lck_grd = dg::network_genult::lock_guard(mtx);

        if (!pq_conn){
            return dg::network_exception::POSTGRES_NOT_INITIALIZED;
        }

        std::expected<pqxx::work, exception_t> transaction_handle = dg::network_exception::to_cstyle_function([]{return pqxx::work{*pq_conn};})();

        if (!transaction_handle.has_value()){
            return transaction_handle.error();
        }

        for (auto& commitable: commitables){
            exception_t err = commitable->commit(transaction_handle.value());
            
            if (dg::network_exception::is_failed(err)){
                return err;
            }
        }

        return dg::network_exception::to_cstyle_function([&]{transaction_handle.commit();})();
    }
}

#endif