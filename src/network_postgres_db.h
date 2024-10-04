#ifndef __DG_NETWORK_POSTGRES_DB_H__
#define __DG_NETWORK_POSTGRES_DB_H__
#include <mutex>
#include <memory>

namespace dg::network_postgres_model{

    struct LegacyAuth{
        dg::network_std_container::string salt;
        dg::network_std_container::string verifiable;
        dg::network_std_container::string user_id;
    };

    struct User{
        dg::network_std_container::string user_id;
        dg::network_std_container::string clearance;
    };

    struct LegacyAuthEntry: LegacyAuth{
        dg::network_std_container::string entry_id;
    };

    struct UserEntry: User{
        dg::network_std_container::string user_id;
    };
} 

namespace dg::network_postgres_db{

    inline std::unique_ptr<pqxx::connection> pq_conn;
    inline std::mutex mtx;

    struct CommitableInterface{
        virtual ~CommitableInterface() noexcept = default;
        virtual auto commit(pqxx::work&) noexcept -> exception_t = 0;
    };

    template <class Lambda>
    class CommitableWrapper: public virtual CommitableWrapper{

        private:

            Lambda lambda;
        
        public:

            static_assert(std::is_nothrow_invokable_v<Lambda, std::add_lvalue_reference_t<pqxx::work>>);
            static_assert(std::is_nothrow_destructible_v<Lambda>);
            
            CommitableWrapper(Lambda lambda) noexcept(std::is_nothrow_constructible_v<Lambda>): lambda(std::move(lambda)){}

            auto commit(pqxx::work& transaction_handle) noexcept -> exception_t{

                auto cstyle_function = dg::network_exception::to_cstyle_function(this->lambda);
                static_assert(std::is_same_v<exception_t, decltype(cstyle_function())>);

                return cstyle_function(transaction_handle);
            }
    };

    void init(){
        
        auto lck_grd = dg::network_genult::lock_guard(mtx);

    }

    void deinit() noexcept{

        auto lck_grd = dg::network_genult::lock_guard(mtx);

    }

    auto pqxx_work_open(pqxx::connection& conn) noexcept -> std::expected<pqxx::work, exception_t>{

    } 

    auto pqxx_work_commit(pqxx::work& work) noexcept -> exception_t{

    }

    auto get_heartbeat() noexcept -> bool{

        auto lck_grd = dg::network_genult::lock_guard(mtx);

        if (!pq_conn){
            return false;
        }

        
    }
 
    auto get_entry_user_by_externalid(const dg::network_std_container::string& ext_id) noexcept -> std::expected<UserEntry, exception_t>{

        auto lck_grd = dg::network_genult::lock_guard(mtx);

        if (!pq_conn){
            return std::unexpected(dg::network_exception::POSTGRES_NOT_INITIALIZED);
        }

    }

    auto get_entry_legacyauth_by_username(const dg::network_std_container::string& username) noexcept -> std::expected<LegacyAuthEntry, exception_t>{

        auto lck_grd = dg::network_genult::lock_guard(mtx);

        if (!pq_conn){
            return std::unexpected(dg::network_exception::POSTGRES_NOT_INITIALIZED);
        }
    }

    auto create_entry_user(const User& user, pqxx::work& transaction_handle) noexcept -> exception_t{
        
    }

    auto create_entry_legacy_auth(const LegacyAuth& legacy_auth, pqxx::work& transaction_handle) noexcept -> exception_t{

    }

    template <class Function, class ...Args>
    auto create_commit_task(Function function, Args ...args) noexcept -> std::expected<std::unique_ptr<CommitableInterface>, exception_t>{ //verify Args... nothrow move constructible
        
    }

    void commit(std::vector<std::unique_ptr<CommitableInterface>> executables) noexcept -> exception_t{

        auto lck_grd = dg::network_genult::lock_guard(mtx);

        if (!pq_conn){
            return dg::network_exception::POSTGRES_NOT_INITIALIZED;
        }

        std::expected<pqxx::work, exception_t> transaction_handle = pqxx_work_open(*pq_conn);

        if (!transaction_handle.has_value()){
            return transaction_handle.error();
        }

        for (size_t i = 0u; i < executables.size(); ++i){
            exception_t err = executables[i]->commit(transaction_handle.value());

            if (dg::network_exception::is_failed(err)){
                return err;
            }
        }
        
        exception_t err = pqxx_work_commit(transaction_handle.value());
        return err;
    }
} 

#endif