#ifndef __DG_NETWORK_POSTGRES_DB_H__
#define __DG_NETWORK_POSTGRES_DB_H__

#include <mutex>
#include <memory>
#include <stdint.h>
#include <stdlib.h>
#include "network_trivial_serializer.h"
#include "network_compact_serializer.h"
#include "network_exception.h"
#include <chrono>
#include "stdx.h"
#include <functional>
#include <algorithm>
#include <utility>
#include "format"
#include <pqxx/cursor>
#include <pqxx/transaction>
#include <pqxx/nontransaction>
#include <pqxx/pqxx>
#include <format>
#include <random>

namespace dg::network_postgres_db::model{

    static inline constexpr size_t USER_ID_MIN_LENGTH                   = size_t{1};
    static inline constexpr size_t USER_ID_MAX_LENGTH                   = size_t{1} << 5;
    static inline constexpr size_t USER_CLEARANCE_MIN_LENGTH            = size_t{1};
    static inline constexpr size_t USER_CLEARANCE_MAX_LENGTH            = size_t{1} << 5;
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

    struct User{
        stdx::string id;
        stdx::string clearance;
        stdx::string salt;
        stdx::string verifiable;
    };

    struct UserEntry: User{
        stdx::string entry_id;
    };

    struct HeartBeat{
        stdx::string payload;
    };

    struct HeartBeatEntry: HeartBeat{
        stdx::string entry_id;
    };

    struct SystemLog{
        stdx::string content;
        stdx::string kind;
        std::chrono::nanoseconds timestamp;
    };

    struct SystemLogEntry: SystemLog{
        stdx::string entry_id;
    };

    struct UserLog{
        stdx::string content;
        stdx::string kind;
        stdx::string user_id;
        std::chrono::nanoseconds timestamp;
    };

    struct UserLogEntry: UserLog{
        stdx::string entry_id;
    };
}

namespace dg::network_postgres_db::model_factory{

    using namespace dg::network_postgres_db::model;

    auto make_user(const stdx::string& id, const stdx::string& clearance) noexcept -> std::expected<User, exception_t>{

        if (std::clamp(id.size(), USER_ID_MIN_LENGTH, USER_ID_MAX_LENGTH) != id.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(clearance.size(), USER_CLEARANCE_MIN_LENGTH, USER_CLEARANCE_MAX_LENGTH) != clearance.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return User{id, clearance};
    }

    auto make_heartbeat(const stdx::string& payload) noexcept -> std::expected<HeartBeat, exception_t>{

        if (std::clamp(payload.size(), HEARTBEAT_PAYLOAD_MIN_LENGTH, HEARTBEAT_PAYLOAD_MAX_LENGTH) != payload.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return HeartBeat{payload};
    }

    auto make_systemlog(const stdx::string& content, const stdx::string& kind, std::chrono::nanoseconds timestamp) noexcept -> std::expected<SystemLog, exception_t>{

        if (std::clamp(content.size(), SYSTEMLOG_CONTENT_MIN_LENGTH, SYSTEMLOG_CONTENT_MAX_LENGTH) != content.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        if (std::clamp(kind.size(), SYSTEMLOG_KIND_MIN_LENGTH, SYSTEMLOG_KIND_MAX_LENGTH) != kind.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        return SystemLog{content, kind, timestamp};        
    }

    auto make_userlog(const stdx::string& content, const stdx::string& kind, const stdx::string& user_id, std::chrono::nanoseconds timestamp) noexcept -> std::expected<UserLog, exception_t>{

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

    static inline stdx::vector<char> hex_fwd_dict{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    static inline stdx::vector<uint8_t> hex_bwd_dict = []{
        stdx::vector<uint8_t> rs(256);
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

    static auto encode(const stdx::string& s) -> stdx::string{

        stdx::string rs{}; 
        rs.reserve(s.size() * 2);

        for (char c: s){
            auto [l, h] = hex_fr_char(c);
            rs.push_back(h);
            rs.push_back(l);
        }

        return rs;
    }

    static auto decode(const stdx::string& s) -> stdx::string{

        if (s.size() % 2 != 0u){
            dg::network_exception::throw_exception(dg::network_exception::BAD_ENCODING_FORMAT);
        }

        size_t rs_sz = s.size() / 2;
        stdx::string rs(rs_sz, ' ');

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
    static auto query_format(std::format_string<Args...> fmt, Args&& ...args) -> stdx::string{

        stdx::string rs{};
        std::format_to(std::back_inserter(rs), fmt, std::forward<Args>(args)...);
        
        if (rs.size() > constants::MAXIMUM_QUERY_LENGTH){
            dg::network_exception::throw_exception(dg::network_exception::POSTGRES_EXCEED_QUERY_LENGTH_LIMIT);
        }

        return rs;
    } 

    static auto encode_sql(const stdx::string& arg) -> stdx::string{

        stdx::string integrity_payload(dg::network_compact_serializer::integrity_size(arg), ' ');
        dg::network_compact_serializer::integrity_serialize_into(integrity_payload.data(), arg);
        
        return hex_encoder::encode(integrity_payload);
    }

    static auto decode_sql(const stdx::string& arg) -> stdx::string{

        stdx::string integrity_payload = hex_encoder::decode(arg); 
        stdx::string org_payload{};
        dg::network_compact_serializer::integrity_deserialize_into(org_payload, integrity_payload.data(), integrity_payload.size());

        return org_payload;
    }

    static auto encode_timestamp(std::chrono::nanoseconds timestamp) -> stdx::string{

        static_assert((std::endian::native == std::endian::little || std::endian::native == std::endian::big));
        uint64_t num_rep = timestamp.count();

        if constexpr(std::endian::native == std::endian::little){
            num_rep = std::byteswap(num_rep);
        }

        stdx::string rs(8u, ' ');
        std::memcpy(rs.data(), &num_rep, sizeof(uint64_t));

        return hex_encoder::encode(rs);
    } 

    static auto decode_timestamp(const stdx::string& arg) -> std::chrono::nanoseconds{

        static_assert((std::endian::native == std::endian::little || std::endian::native == std::endian::big));
        stdx::string encoded = hex_encoder::decode(arg);

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

    static auto quote(const stdx::string& arg) -> stdx::string{

        stdx::string rs{};
        std::format_to(std::back_inserter(rs), "\"{}\"", arg);

        return rs;
    }

    static auto randomize_string(size_t sz) -> stdx::string{

        auto rs         = stdx::string(sz, ' ');
        auto rand_gen   = std::bind(std::uniform_int_distribution<char>(), std::mt19937{});
        std::generate(rs.begin(), rs.end(), std::ref(rand_gen));

        return rs;
    } 
} 

namespace dg::network_postgres_db{

    class convertible_string{

        private:

            std::string_view view;

        public:

            constexpr convertible_string() = default;

            constexpr convertible_string(std::string_view view) noexcept: view(std::move(view)){}

            template <class ...Args>
            operator std::basic_string<Args...>() const noexcept{
                
                std::basic_string<Args...> rs(view.begin(), view.end());
                return rs;
            }
    };

    struct CommitableInterface{
        virtual ~CommitableInterface() noexcept = default;
        virtual void commit(pqxx::work&) = 0;
    };

    inline std::unique_ptr<pqxx::connection> pq_conn; //-if performance problem arises - change -> atomic_shared_ptr or multiple instance approach 
    inline std::mutex mtx;

    template <class Lambda>
    class CommitableWrapper: public virtual CommitableInterface{

        private:

            Lambda lambda;
        
        public:

            static_assert(std::is_nothrow_destructible_v<Lambda>);
            
            CommitableWrapper(Lambda lambda) noexcept(std::is_nothrow_constructible_v<Lambda>): lambda(std::move(lambda)){}

            void commit(pqxx::work& transaction_handle) noexcept{

                return lambda(transaction_handle);
            }
    };

    void init(const stdx::string& pq_conn_arg){
        
        auto lck_grd = stdx::lock_guard(mtx);
        pq_conn = std::make_unique<pqxx::connection>(pq_conn_arg.c_str());
    }

    void deinit() noexcept{

        auto lck_grd = stdx::lock_guard(mtx);
        pq_conn = nullptr;
    }

    auto get_heartbeat() noexcept -> bool{
        
        auto lck_grd = stdx::lock_guard(mtx);

        if (!pq_conn){
            return false;
        }

        auto lambda = [&]{
            pqxx::work transaction_handle{*pq_conn};
            stdx::string heartbeat_payload = utility::randomize_string(model::HEARTBEAT_PAYLOAD_MAX_LENGTH);
            stdx::string encoded_payload   = utility::quote(utility::encode_sql(heartbeat_payload));
            stdx::string inject_query      = utility::query_format("INSERT INTO HeartBeat(payload) VALUES({})", encoded_payload);

            transaction_handle.exec(inject_query.c_str()).no_rows();
            transaction_handle.commit();
            
            stdx::string get_query = utility::query_format("SELECT * FROM HeartBeat WHERE HeartBeat.payload = {}", encoded_payload);
            auto rs = transaction_handle.exec(get_query.c_str());
            rs.one_row();
            rs.for_each([&](std::string_view entry_id, std::string_view encoded_payload){
                if (heartbeat_payload != utility::decode_sql(convertible_string(encoded_payload))){
                    dg::network_exception::throw_exception(dg::network_exception::POSTGRES_CORRUPTION);
                }
            });
            stdx::string del_query = utility::query_format("DELETE FROM HeartBeat WHERE HeartBeat.payload = {}", encoded_payload);
            transaction_handle.exec(del_query.c_str()).no_rows();
            transaction_handle.commit();
            transaction_handle.exec(get_query.c_str()).no_rows();
        };

        exception_t err = dg::network_exception::to_cstyle_function(lambda)();
        return dg::network_exception::is_success(err); //need to be more descriptive + handle internal corruption - internal corruption could bleed
    }

    auto get_user_by_id(const stdx::string& id) noexcept -> std::expected<model::UserEntry, exception_t>{

        auto lck_grd = stdx::lock_guard(mtx);

        if (!pq_conn){
            return std::unexpected(dg::network_exception::POSTGRES_NOT_INITIALIZED);
        }

        if (std::clamp(id.size(), model::USER_ID_MIN_LENGTH, model::USER_ID_MAX_LENGTH) != id.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        auto lambda = [&]{
            pqxx::nontransaction transaction_handle{*pq_conn};
            stdx::string query = utility::query_format("SELECT * FROM User WHERE User.id = {}", utility::quote(utility::encode_sql(id)));
            model::UserEntry user{};
            auto rs = transaction_handle.exec(query.c_str());
            rs.one_row();
            rs.for_each([&](std::string_view entry_id, std::string_view id, std::string_view clearance, std::string_view salt, std::string_view verifiable){
                user.entry_id       = convertible_string(entry_id);
                user.id             = utility::decode_sql(convertible_string(id));
                user.clearance      = utility::decode_sql(convertible_string(clearance));
                user.salt           = utility::decode_sql(convertible_string(salt));
                user.verifiable     = utility::decode_sql(convertible_string(verifiable));
            });

            return user;
        };

        return dg::network_exception::to_cstyle_function(lambda)();
    }
    
    auto get_systemlog(const stdx::string& kind, std::chrono::nanoseconds fr, std::chrono::nanoseconds to, size_t limit) noexcept -> std::expected<stdx::vector<model::SystemLogEntry>, exception_t>{

        auto lck_grd = stdx::lock_guard(mtx);

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
            auto query      = utility::query_format("SELECT * FROM SystemLog \
                                                     WHERE SystemLog.kind = {} \ 
                                                           AND SystemLog.timestamp > {} AND SystemLog.timestamp < {} \
                                                     LIMIT {}", utility::quote(utility::encode_sql(kind)), utility::quote(utility::encode_timestamp(fr)), utility::quote(utility::encode_timestamp(to)), limit);

            auto log_vec    = stdx::vector<model::SystemLogEntry>{};
            auto rs         = transaction_handle.exec(query.c_str());
            rs.for_each([&](std::string_view entry_id, std::string_view content, std::string_view kind, std::string_view timestamp){
                model::SystemLogEntry entry{};
                entry.entry_id  = convertible_string(entry_id);
                entry.content   = utility::decode_sql(convertible_string(content));
                entry.kind      = utility::decode_sql(convertible_string(kind));
                entry.timestamp = utility::decode_timestamp(convertible_string(timestamp));
                log_vec.push_back(std::move(entry));
            });

            return log_vec;
        };

        return dg::network_exception::to_cstyle_function(lambda)();
    }

    auto get_userlog(const stdx::string& user_id, const stdx::string& kind, std::chrono::nanoseconds fr, std::chrono::nanoseconds to, size_t limit) noexcept -> std::expected<stdx::vector<model::UserLogEntry>, exception_t>{

        auto lck_grd = stdx::lock_guard(mtx);

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
            auto query      = utility::query_format("SELECT * From UserLog \
                                                     WHERE UserLog.user_id = {} \
                                                           AND UserLog.kind = {} \
                                                           AND UserLog.timestamp > {} AND UserLog.timestamp < {} \
                                                     LIMIT {}", utility::quote(utility::encode_sql(user_id)), utility::quote(utility::encode_sql(kind)), 
                                                                utility::quote(utility::encode_timestamp(fr)), utility::quote(utility::encode_timestamp(to)),
                                                                limit);
            auto log_vec    = stdx::vector<model::UserLogEntry>{};
            auto rs         = transaction_handle.exec(query.c_str());
            rs.for_each([&](std::string_view entry_id, std::string_view content, std::string_view kind, std::string_view user_id, std::string_view timestamp){
                model::UserLogEntry entry{};
                entry.entry_id  = convertible_string(entry_id);
                entry.content   = utility::decode_sql(convertible_string(content));
                entry.kind      = utility::decode_sql(convertible_string(kind));
                entry.user_id   = utility::decode_sql(convertible_string(user_id));
                entry.timestamp = utility::decode_timestamp(convertible_string(timestamp));
                log_vec.push_back(std::move(entry));
            });

            return log_vec;
        };

        return dg::network_exception::to_cstyle_function(lambda)();
    }

    auto make_commitable_create_systemlog(const model::SystemLog& log) noexcept -> std::expected<std::unique_ptr<CommitableInterface>, exception_t>{

        auto lambda = [=](pqxx::work& transaction_handle){
            stdx::string query = utility::query_format("INSERT INTO SystemLog(content, kind, timestamp) VALUES({}, {}, {})", utility::quote(utility::encode_sql(log.content)), 
                                                                                                                             utility::quote(utility::encode_sql(log.kind)), 
                                                                                                                             utility::quote(utility::encode_timestamp(log.timestamp)));
            transaction_handle.exec(query.c_str()).no_rows();
        };

        return std::make_unique<CommitableWrapper<decltype(lambda)>>(std::move(lambda));
    };

    auto make_commitable_create_userlog(const model::UserLog& log) noexcept -> std::expected<std::unique_ptr<CommitableInterface>, exception_t>{

        auto lambda = [=](pqxx::work& transaction_handle){
            stdx::string query = utility::query_format("INSERT INTO UserLog(content, kind, user_id, timestamp) VALUES({}, {}, {}, {})", utility::quote(utility::encode_sql(log.content)),
                                                                                                                                        utility::quote(utility::encode_sql(log.kind)),
                                                                                                                                        utility::quote(utility::encode_sql(log.user_id)),
                                                                                                                                        utility::quote(utility::encode_timestamp(log.timestamp)));
            transaction_handle.exec(query.c_str()).no_rows();
        };

        return std::make_unique<CommitableWrapper<decltype(lambda)>>(std::move(lambda));
    }
    
    auto make_commitable_create_user(const model::User& user) noexcept -> std::expected<std::unique_ptr<CommitableInterface>, exception_t>{

        auto lambda = [=](pqxx::work& transaction_handle){
            stdx::string query = utility::query_format("INSERT INTO User(id, clearance, salt, verifiable) VALUES({}, {})", utility::quote(utility::encode_sql(user.id)), 
                                                                                                                           utility::quote(utility::encode_sql(user.clearance)),
                                                                                                                           utility::quote(utility::encode_sql(user.salt)),
                                                                                                                           utility::quote(utility::encode_sql(user.verifiable)));
            transaction_handle.exec(query.c_str()).no_rows();
        };

        return std::make_unique<CommitableWrapper<decltype(lambda)>>(std::move(lambda));
    }

    auto make_commitable_delete_user_by_id(const stdx::string& id) noexcept -> std::expected<std::unique_ptr<CommitableInterface>, exception_t>{

        if (std::clamp(id.size(), model::USER_ID_MIN_LENGTH, model::USER_ID_MAX_LENGTH) != id.size()){
            return std::unexpected(dg::network_exception::INVALID_ARGUMENT);
        }

        auto lambda = [=](pqxx::work& transaction_handle){
            stdx::string query = utility::query_format("DELETE FROM User WHERE User.id = {}", utility::quote(utility::encode_sql(id)));
            transaction_handle.exec(query.c_str()).no_rows();
        };

        return std::make_unique<CommitableWrapper<decltype(lambda)>>(std::move(lambda));
    }

    auto commit(stdx::vector<std::unique_ptr<CommitableInterface>> commitables) noexcept -> exception_t{

        auto lck_grd = stdx::lock_guard(mtx);

        if (!pq_conn){
            return dg::network_exception::POSTGRES_NOT_INITIALIZED;
        }

        auto lambda = [&]{
            pqxx::work work{*pq_conn};

            for (auto& commitable: commitables){
                commitable->commit(work);
             }

             work.commit();
        };

        return dg::network_exception::to_cstyle_function(lambda)();
    }
}

#endif